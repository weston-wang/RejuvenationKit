from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from rejuvenationkit.genomics.annotations import (
    FunctionalAnnotationResult,
    FunctionalAssociation,
    read_functional_annotations,
)
from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    FeatureDomain,
    QueryProvenance,
    ResourceSnapshot,
)
from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicFeatureType


def domain(*, species_taxon_id: int = 9615) -> FeatureDomain:
    return FeatureDomain(
        species_taxon_id=species_taxon_id,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
        genome_assembly="CanFam4" if species_taxon_id == 9615 else "GRCm39",
    )


def snapshot() -> ResourceSnapshot:
    return ResourceSnapshot(
        provider_id="gene-ontology",
        resource_id="go-annotations",
        resource_release="2026-07-01",
        retrieved_at=datetime(2026, 8, 14, 12, tzinfo=UTC),
        response_sha256="a" * 64,
        source_uri="https://example.org/goa.tsv",
        license_id="CC-BY-4.0",
    )


def feature_query(*, selected_domain: FeatureDomain | None = None) -> FeatureCollection:
    return FeatureCollection(
        collection_id="canine-treatment-response-genes",
        domain=selected_domain or domain(),
        feature_ids=("ENSCAFG1", "ENSCAFG2", "ENSCAFG3"),
        source_snapshot_id="b" * 64,
    )


def provenance(
    query: FeatureCollection,
    *,
    selected_domain: FeatureDomain | None = None,
    input_hash: str | None = None,
    complete: bool = True,
    warnings: tuple[str, ...] = (),
    response_checksum: str | None = None,
) -> QueryProvenance:
    return QueryProvenance(
        provider_id="gprofiler",
        provider_version="e115_eg62_p22",
        resources=(snapshot(),),
        domain=selected_domain or query.domain,
        retrieved_at=datetime(2026, 8, 14, 13, tzinfo=UTC),
        input_hash=input_hash or query.content_hash,
        response_checksum=response_checksum,
        query_parameters={"sources": ("GO:BP",), "ordered": False},
        software_versions={"gprofiler-official": "1.0.0"},
        complete=complete,
        warnings=warnings,
    )


def annotations_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_gene": ["ENSCAFG2", "ENSCAFG1", "ENSCAFG1", "ENSCAFG1"],
            "go_id": ["GO:0006914", "GO:0006914", "GO:0006914", "GO:0009987"],
            "go_name": ["autophagy", "autophagy", "autophagy", "cellular process"],
            "source": ["GO:BP", "GO:BP", "GO:BP", "GO:BP"],
            "relation": ["involved_in", "involved_in", "involved_in", "involved_in"],
            "evidence": ["IMP", "IDA", "IDA", None],
            "qualifier": [
                None,
                "contributes_to|acts_upstream_of",
                "acts_upstream_of|contributes_to",
                None,
            ],
            "record_ids": ["GOA:3", "GOA:2", "GOA:1", None],
        }
    )


def read(
    frame: pd.DataFrame,
    *,
    query: FeatureCollection | None = None,
    query_provenance: QueryProvenance | None = None,
    **updates: object,
) -> FunctionalAnnotationResult:
    selected_query = query or feature_query()
    selected_provenance = query_provenance or provenance(selected_query)
    arguments: dict[str, object] = {
        "query": selected_query,
        "provenance": selected_provenance,
        "feature_id_column": "query_gene",
        "term_id_column": "go_id",
        "term_name_column": "go_name",
        "term_namespace_column": "source",
        "relation_column": "relation",
        "evidence_code_column": "evidence",
        "qualifiers_column": "qualifier",
        "source_record_ids_column": "record_ids",
    }
    arguments.update(updates)
    return read_functional_annotations(frame, **arguments)


def test_import_preserves_evidence_deduplicates_and_reports_unmatched_features() -> None:
    result = read(annotations_frame())

    assert result.matched_feature_ids == ("ENSCAFG1", "ENSCAFG2")
    assert result.unmatched_feature_ids == ("ENSCAFG3",)
    assert len(result.associations) == 3
    autophagy = next(
        item
        for item in result.associations
        if item.feature_id == "ENSCAFG1" and item.evidence_code == "IDA"
    )
    assert autophagy.term_name == "autophagy"
    assert autophagy.term_namespace == "GO:BP"
    assert autophagy.relation == "involved_in"
    assert autophagy.qualifiers == ("acts_upstream_of", "contributes_to")
    assert autophagy.source_record_ids == ("GOA:1", "GOA:2")
    assert result.provenance.response_checksum is not None
    assert len(result.provenance.response_checksum) == 64
    assert result.fusion_eligibility == "not_fusible"
    assert not hasattr(result, "to_evidence")
    assert len(result.result_hash) == 64


def test_import_is_deterministic_under_provider_row_reordering() -> None:
    first = read(annotations_frame())
    reordered = read(annotations_frame().iloc[::-1].reset_index(drop=True))

    assert first.associations == reordered.associations
    assert first.result_hash == reordered.result_hash


def test_import_rejects_a_stale_normalized_response_checksum() -> None:
    """Raw snapshot bytes and the normalized typed-result checksum stay distinct."""
    query = feature_query()
    with pytest.raises(ValueError, match="supplied response checksum"):
        read(
            annotations_frame(),
            query=query,
            query_provenance=provenance(query, response_checksum="c" * 64),
        )


def test_exact_duplicates_collapse_but_conflicting_term_names_are_rejected() -> None:
    exact_duplicate = pd.concat(
        [annotations_frame(), annotations_frame().iloc[[0]]], ignore_index=True
    )
    assert read(exact_duplicate) == read(annotations_frame())

    conflicting = annotations_frame().copy()
    conflicting.loc[2, "go_name"] = "not autophagy"
    with pytest.raises(ValueError, match="conflicting term names"):
        read(conflicting)


def test_one_source_record_cannot_describe_conflicting_associations() -> None:
    conflicting = annotations_frame().copy()
    conflicting.loc[3, "record_ids"] = "GOA:1"

    with pytest.raises(ValueError, match="maps to conflicting associations"):
        read(conflicting)


def test_import_rejects_domain_input_hash_resource_and_query_feature_mismatches() -> None:
    selected_query = feature_query()
    mouse = domain(species_taxon_id=10090)
    with pytest.raises(ValueError, match="domain"):
        read(
            annotations_frame(),
            query=selected_query,
            query_provenance=provenance(selected_query, selected_domain=mouse),
        )
    with pytest.raises(ValueError, match="input_hash"):
        read(
            annotations_frame(),
            query=selected_query,
            query_provenance=provenance(selected_query, input_hash="d" * 64),
        )
    missing_resource = provenance(selected_query).model_copy(update={"resources": ()})
    with pytest.raises(ValueError, match="resource snapshot"):
        read(
            annotations_frame(),
            query=selected_query,
            query_provenance=missing_resource,
        )
    unexpected = annotations_frame().copy()
    unexpected.loc[0, "query_gene"] = "NOT_QUERIED"
    with pytest.raises(ValueError, match="outside the exact query"):
        read(unexpected)


def test_truncated_and_incomplete_results_are_explicit_and_cannot_claim_completeness() -> None:
    truncated = read(annotations_frame(), truncated=True)
    assert not truncated.complete
    assert truncated.truncated
    assert "functional_annotation_result_truncated" in truncated.warnings
    assert "functional_annotation_result_incomplete" in truncated.warnings

    selected_query = feature_query()
    incomplete_provenance = provenance(
        selected_query,
        complete=False,
        warnings=("provider_reported_partial_result",),
    )
    incomplete = read(
        annotations_frame(),
        query=selected_query,
        query_provenance=incomplete_provenance,
    )
    assert not incomplete.complete
    assert "provider_reported_partial_result" in incomplete.warnings
    assert "functional_annotation_result_incomplete" in incomplete.warnings

    with pytest.raises(ValueError, match="truncated"):
        read(annotations_frame(), truncated=True, complete=True)
    with pytest.raises(ValueError, match="provenance is incomplete"):
        read(
            annotations_frame(),
            query=selected_query,
            query_provenance=incomplete_provenance,
            complete=True,
        )


def test_empty_complete_result_reports_every_query_feature_as_unmatched() -> None:
    frame = annotations_frame().iloc[0:0]
    result = read(frame)

    assert result.complete
    assert result.associations == ()
    assert result.matched_feature_ids == ()
    assert result.unmatched_feature_ids == feature_query().feature_ids


def test_import_rejects_missing_columns_and_invalid_multivalue_cells() -> None:
    with pytest.raises(ValueError, match="columns are absent"):
        read(annotations_frame().drop(columns="evidence"))

    duplicate_qualifier = annotations_frame().copy()
    duplicate_qualifier.loc[0, "qualifier"] = "part_of|part_of"
    with pytest.raises(ValueError, match="duplicate values"):
        read(duplicate_qualifier)


def test_models_reject_manual_inconsistent_fusion_or_coverage_claims() -> None:
    result = read(annotations_frame())
    payload = result.model_dump()
    payload["fusion_eligibility"] = "fusible"
    with pytest.raises(ValidationError, match="literal_error"):
        FunctionalAnnotationResult.model_validate(payload)

    payload = result.model_dump()
    payload["unmatched_feature_ids"] = ()
    with pytest.raises(ValidationError, match="partition"):
        FunctionalAnnotationResult.model_validate(payload)

    with pytest.raises(ValidationError, match="surrounding whitespace"):
        FunctionalAssociation(
            feature_id=" ENSCAFG1",
            term_id="GO:1",
            term_name="term",
            term_namespace="GO:BP",
            relation="involved_in",
        )
