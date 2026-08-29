from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from rejuvenationkit.genomics.enrichment import (
    FunctionalGeneSet,
    GeneSetCollection,
    MultipleTestingMethod,
    OverrepresentationRequest,
    OverrepresentationResult,
    gene_set_collection_from_frame,
    run_overrepresentation,
)
from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    FeatureDomain,
    ResourceSnapshot,
)
from rejuvenationkit.genomics.schemas import FeatureNamespace, GenomicFeatureType


def _domain(**updates: object) -> FeatureDomain:
    values: dict[str, object] = {
        "species_taxon_id": 10090,
        "feature_type": GenomicFeatureType.GENE,
        "namespace": FeatureNamespace.ENSEMBL,
        "genome_assembly": "GRCm39",
    }
    values.update(updates)
    return FeatureDomain(**values)


def _snapshot(**updates: object) -> ResourceSnapshot:
    values: dict[str, object] = {
        "provider_id": "offline-fixture",
        "resource_id": "toy-pathways",
        "resource_release": "2026-08-01",
        "retrieved_at": datetime(2026, 8, 1, tzinfo=UTC),
        "response_sha256": "a" * 64,
        "source_uri": "https://example.org/toy-pathways.tsv",
        "license_id": "CC0-1.0",
        "citation_ids": ("DOI:10.1/toy",),
    }
    values.update(updates)
    return ResourceSnapshot(**values)


def _features(
    collection_id: str,
    feature_ids: tuple[str, ...],
    *,
    domain: FeatureDomain | None = None,
    source_snapshot_id: str = "study-input-v1",
) -> FeatureCollection:
    return FeatureCollection(
        collection_id=collection_id,
        domain=domain or _domain(),
        feature_ids=feature_ids,
        source_snapshot_id=source_snapshot_id,
    )


def _gene_set(
    gene_set_id: str,
    members: tuple[str, ...],
    *,
    domain: FeatureDomain | None = None,
    resource: ResourceSnapshot | None = None,
) -> FunctionalGeneSet:
    return FunctionalGeneSet(
        gene_set_id=gene_set_id,
        name=f"Pathway {gene_set_id}",
        description=f"Toy pathway {gene_set_id}",
        domain=domain or _domain(),
        member_feature_ids=members,
        resource=resource or _snapshot(),
    )


def _collection(
    *,
    gene_sets: tuple[FunctionalGeneSet, ...] | None = None,
    domain: FeatureDomain | None = None,
    resource: ResourceSnapshot | None = None,
) -> GeneSetCollection:
    resource = resource or _snapshot()
    domain = domain or _domain()
    return GeneSetCollection(
        collection_id="toy-collection",
        name="Toy pathways",
        domain=domain,
        gene_sets=gene_sets
        or (
            _gene_set("A", ("g1", "g2", "g3"), domain=domain, resource=resource),
            _gene_set("B", ("g1", "g2", "g4"), domain=domain, resource=resource),
            _gene_set("C", ("g1", "g4", "g5"), domain=domain, resource=resource),
        ),
        resource=resource,
    )


def _request(
    *,
    selected: tuple[str, ...] = ("g1", "g2", "g3"),
    background: tuple[str, ...] = tuple(f"g{index}" for index in range(1, 11)),
    gene_sets: GeneSetCollection | None = None,
    method: MultipleTestingMethod = MultipleTestingMethod.BENJAMINI_HOCHBERG,
    minimum_term_size: int = 2,
    maximum_term_size: int = 10,
) -> OverrepresentationRequest:
    return OverrepresentationRequest(
        selected_features=_features("selected", selected),
        background_universe=_features("measured-background", background),
        gene_sets=gene_sets or _collection(),
        multiple_testing_method=method,
        minimum_term_size=minimum_term_size,
        maximum_term_size=maximum_term_size,
    )


def _partitioned_result() -> OverrepresentationResult:
    resource = _snapshot()
    domain = _domain()
    collection = _collection(
        domain=domain,
        resource=resource,
        gene_sets=(
            _gene_set("TEST", ("g1", "g2", "g3"), domain=domain, resource=resource),
            _gene_set("SMALL", ("g1",), domain=domain, resource=resource),
            _gene_set("ABSENT", ("g20", "g21"), domain=domain, resource=resource),
        ),
    )
    return run_overrepresentation(
        _request(selected=("g1", "g2", "g9"), gene_sets=collection, minimum_term_size=2)
    )


def _revalidate_result(
    result: OverrepresentationResult,
    **updates: object,
) -> OverrepresentationResult:
    payload = result.model_dump(mode="python")
    payload.update(updates)
    return OverrepresentationResult.model_validate(payload)


def test_known_hypergeometric_p_values_and_bh_adjustment() -> None:
    result = run_overrepresentation(
        _request(),
        executed_at=datetime(2026, 8, 15, tzinfo=UTC),
    )
    by_id = {term.gene_set_id: term for term in result.terms}

    assert by_id["A"].p_value == pytest.approx(1 / 120)
    assert by_id["B"].p_value == pytest.approx(22 / 120)
    assert by_id["C"].p_value == pytest.approx(85 / 120)
    assert by_id["A"].adjusted_p_value == pytest.approx(3 / 120)
    assert by_id["B"].adjusted_p_value == pytest.approx(0.275)
    assert by_id["C"].adjusted_p_value == pytest.approx(85 / 120)
    assert result.multiple_testing.method is MultipleTestingMethod.BENJAMINI_HOCHBERG
    assert result.multiple_testing.family_size == 3
    assert tuple(term.gene_set_id for term in result.terms) == ("A", "B", "C")


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (MultipleTestingMethod.BONFERRONI, (3 / 120, 66 / 120, 1.0)),
        (MultipleTestingMethod.NONE, (1 / 120, 22 / 120, 85 / 120)),
    ],
)
def test_bonferroni_and_no_adjustment_are_local_and_audited(
    method: MultipleTestingMethod,
    expected: tuple[float, float, float],
) -> None:
    result = run_overrepresentation(_request(method=method))
    by_id = {term.gene_set_id: term for term in result.terms}

    assert tuple(by_id[identifier].adjusted_p_value for identifier in ("A", "B", "C")) == (
        pytest.approx(expected[0]),
        pytest.approx(expected[1]),
        pytest.approx(expected[2]),
    )
    assert result.multiple_testing.method is method
    assert "all collection terms" in result.multiple_testing.family_definition


def test_request_rejects_selected_outside_background_and_domain_mismatch() -> None:
    with pytest.raises(ValidationError, match="subset"):
        _request(selected=("g1", "not-measured"))

    with pytest.raises(ValidationError, match="share one domain"):
        OverrepresentationRequest(
            selected_features=_features("selected", ("g1",), domain=_domain()),
            background_universe=_features(
                "background",
                ("g1", "g2"),
                domain=_domain(species_taxon_id=9615),
            ),
            gene_sets=_collection(),
            minimum_term_size=1,
        )

    with pytest.raises(ValidationError, match="gene-set and background domains"):
        _request(gene_sets=_collection(domain=_domain(species_taxon_id=9615)))


def test_collection_rejects_duplicate_terms_mixed_domains_and_resources() -> None:
    resource = _snapshot()
    term = _gene_set("A", ("g1", "g2"), resource=resource)
    with pytest.raises(ValidationError, match="unique"):
        _collection(gene_sets=(term, term), resource=resource)
    with pytest.raises(ValidationError, match="collection domain"):
        _collection(
            gene_sets=(_gene_set("A", ("g1",), domain=_domain(species_taxon_id=9615)),),
        )
    with pytest.raises(ValidationError, match="resource snapshot"):
        _collection(
            gene_sets=(
                _gene_set(
                    "A",
                    ("g1",),
                    resource=_snapshot(resource_release="2026-07-01"),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="gene or protein feature domain"):
        _gene_set(
            "variant-set",
            ("v1",),
            domain=_domain(feature_type=GenomicFeatureType.VARIANT),
        )


def test_functional_sets_accept_uniprot_protein_domains() -> None:
    protein_domain = _domain(
        feature_type=GenomicFeatureType.PROTEIN,
        namespace=FeatureNamespace.UNIPROT,
    )

    term = _gene_set("protein-term", ("P42345",), domain=protein_domain)

    assert term.domain.feature_type is GenomicFeatureType.PROTEIN
    assert term.domain.namespace is FeatureNamespace.UNIPROT


def test_term_size_and_selected_matching_audits_are_explicit() -> None:
    result = _partitioned_result()

    assert result.tested_gene_set_ids == ("TEST",)
    assert result.size_filtered_gene_set_ids == ("SMALL",)
    assert result.unmatched_gene_set_ids == ("ABSENT",)
    assert result.matched_selected_feature_ids == ("g1", "g2")
    assert result.unmatched_selected_feature_ids == ("g9",)
    assert result.warnings == (
        "selected_features_absent_from_tested_gene_sets",
        "gene_sets_without_background_members",
        "gene_sets_filtered_by_background_term_size",
    )


def test_resource_and_query_provenance_are_complete_and_reproducible() -> None:
    request = _request()
    first = run_overrepresentation(
        request,
        executed_at=datetime(2026, 8, 15, 8, tzinfo=UTC),
    )
    second = run_overrepresentation(
        request,
        executed_at=datetime(2026, 8, 15, 9, tzinfo=UTC),
    )

    assert first.provenance.provider_id == "rejuvenationkit.offline_ora"
    assert first.provenance.resources == (request.gene_sets.resource,)
    assert first.provenance.resources[0].resource_release == "2026-08-01"
    assert first.provenance.domain == request.background_universe.domain
    assert first.provenance.input_hash == request.input_hash
    assert first.provenance.query_parameters == request.query_parameters
    assert first.provenance.query_hash == second.provenance.query_hash
    assert first.provenance.response_checksum == second.provenance.response_checksum
    assert first.provenance.response_checksum is not None
    assert len(first.provenance.response_checksum) == 64


def test_gene_set_and_input_order_do_not_change_results_or_query_identity() -> None:
    baseline_request = _request()
    reversed_collection = baseline_request.gene_sets.model_copy(
        update={
            "gene_sets": tuple(
                item.model_copy(
                    update={"member_feature_ids": tuple(reversed(item.member_feature_ids))}
                )
                for item in reversed(baseline_request.gene_sets.gene_sets)
            )
        }
    )
    reordered_request = OverrepresentationRequest(
        selected_features=baseline_request.selected_features.model_copy(
            update={"feature_ids": tuple(reversed(baseline_request.selected_features.feature_ids))}
        ),
        background_universe=baseline_request.background_universe.model_copy(
            update={
                "feature_ids": tuple(reversed(baseline_request.background_universe.feature_ids))
            }
        ),
        gene_sets=reversed_collection,
        multiple_testing_method=baseline_request.multiple_testing_method,
        alpha=baseline_request.alpha,
        minimum_term_size=baseline_request.minimum_term_size,
        maximum_term_size=baseline_request.maximum_term_size,
    )
    first = run_overrepresentation(baseline_request)
    second = run_overrepresentation(reordered_request)

    assert first.terms == second.terms
    assert first.provenance.query_hash == second.provenance.query_hash
    assert first.provenance.response_checksum == second.provenance.response_checksum


def test_frame_loader_is_typed_deterministic_and_rejects_ambiguous_rows() -> None:
    frame = pd.DataFrame(
        {
            "set_id": ("B", "A", "A"),
            "set_name": ("Beta", "Alpha", "Alpha"),
            "member": ("g3", "g2", "g1"),
            "description": ("B description", "A description", "A description"),
        }
    )
    collection = gene_set_collection_from_frame(
        frame,
        collection_id="loaded",
        collection_name="Loaded pathways",
        domain=_domain(),
        resource=_snapshot(),
        gene_set_id_column="set_id",
        member_feature_id_column="member",
        gene_set_name_column="set_name",
        description_column="description",
    )

    assert tuple(item.gene_set_id for item in collection.gene_sets) == ("A", "B")
    assert collection.gene_sets[0].member_feature_ids == ("g1", "g2")
    assert len(collection.normalized_import_sha256) == 64
    reordered = gene_set_collection_from_frame(
        frame.iloc[::-1].reset_index(drop=True),
        collection_id="loaded",
        collection_name="Loaded pathways",
        domain=_domain(),
        resource=_snapshot(),
        gene_set_id_column="set_id",
        member_feature_id_column="member",
        gene_set_name_column="set_name",
        description_column="description",
        expected_normalized_import_sha256=collection.normalized_import_sha256,
    )
    assert reordered.normalized_import_sha256 == collection.normalized_import_sha256

    changed = frame.copy()
    changed.loc[0, "member"] = "g9"
    with pytest.raises(ValueError, match="expected normalized gene-set checksum"):
        gene_set_collection_from_frame(
            changed,
            collection_id="loaded",
            collection_name="Loaded pathways",
            domain=_domain(),
            resource=_snapshot(),
            gene_set_id_column="set_id",
            member_feature_id_column="member",
            gene_set_name_column="set_name",
            description_column="description",
            expected_normalized_import_sha256=collection.normalized_import_sha256,
        )

    forged = collection.model_dump(mode="python")
    forged["normalized_import_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="normalized_import_sha256"):
        GeneSetCollection.model_validate(forged)

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate gene-set membership"):
        gene_set_collection_from_frame(
            duplicate,
            collection_id="loaded",
            collection_name="Loaded pathways",
            domain=_domain(),
            resource=_snapshot(),
            gene_set_id_column="set_id",
            member_feature_id_column="member",
            gene_set_name_column="set_name",
            description_column="description",
        )


def test_overrepresentation_is_directionless_and_not_fusible_evidence() -> None:
    result = run_overrepresentation(_request())

    assert result.analysis_type == "overrepresentation"
    assert result.fusion_eligibility == "not_fusible"
    assert not hasattr(result, "to_evidence")
    assert all(not hasattr(term, "direction") for term in result.terms)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        OverrepresentationRequest(
            selected_features=_features("selected", ("g1",)),
            background_universe=_features("background", ("g1", "g2")),
            gene_sets=_collection(),
            minimum_term_size=1,
            direction="up",
        )


def test_public_result_round_trip_reconstructs_every_term() -> None:
    result = run_overrepresentation(_request())

    reconstructed = OverrepresentationResult.model_validate(result.model_dump(mode="python"))

    assert reconstructed == result


def test_public_result_rejects_forged_overlap_counts_expected_fold_and_raw_p() -> None:
    result = run_overrepresentation(_request())
    first = result.terms[0]
    for update, message in (
        (
            {"overlap_feature_ids": ("g1", "g2"), "selected_overlap_count": 2},
            "overlap",
        ),
        ({"selected_size": first.selected_size + 1}, "counts"),
        ({"expected_overlap": first.expected_overlap + 0.01}, "expected overlap"),
        ({"fold_enrichment": first.fold_enrichment + 0.01}, "fold enrichment"),
        ({"p_value": first.p_value + 0.01}, "raw hypergeometric p-value"),
    ):
        forged_terms = (first.model_copy(update=update), *result.terms[1:])
        with pytest.raises(ValidationError, match=message):
            _revalidate_result(result, terms=forged_terms)


def test_public_result_rejects_forged_adjustment_and_term_order() -> None:
    result = run_overrepresentation(_request())
    forged_adjustment = result.terms[0].model_copy(
        update={"adjusted_p_value": result.terms[0].adjusted_p_value + 0.01}
    )

    with pytest.raises(ValidationError, match="complete multiple-testing family"):
        _revalidate_result(result, terms=(forged_adjustment, *result.terms[1:]))
    with pytest.raises(ValidationError, match="deterministic ordering"):
        _revalidate_result(result, terms=tuple(reversed(result.terms)))


def test_public_result_rejects_forged_term_partitions_and_feature_audit() -> None:
    result = _partitioned_result()
    invalid_updates = (
        ({"tested_gene_set_ids": ("ABSENT",)}, "tested_gene_set_ids"),
        ({"unmatched_gene_set_ids": ()}, "unmatched_gene_set_ids"),
        ({"size_filtered_gene_set_ids": ()}, "size_filtered_gene_set_ids"),
        ({"matched_selected_feature_ids": ("g1",)}, "matched_selected_feature_ids"),
        ({"unmatched_selected_feature_ids": ("g2", "g9")}, "unmatched_selected_feature_ids"),
    )

    for update, message in invalid_updates:
        with pytest.raises(ValidationError, match=message):
            _revalidate_result(result, **update)


def test_public_result_rejects_forged_family_audit_warnings_and_checksum() -> None:
    result = _partitioned_result()
    forged_audit = result.multiple_testing.model_copy(
        update={"family_definition": "only favorable terms"}
    )
    forged_provenance = result.provenance.model_copy(update={"response_checksum": "f" * 64})
    removed_warning = result.warnings[1:]
    provenance_with_removed_warning = result.provenance.model_copy(
        update={"warnings": removed_warning}
    )

    with pytest.raises(ValidationError, match="family definition"):
        _revalidate_result(result, multiple_testing=forged_audit)
    with pytest.raises(ValidationError, match="response checksum"):
        _revalidate_result(result, provenance=forged_provenance)
    with pytest.raises(ValidationError, match="reconstructed ORA warnings"):
        _revalidate_result(
            result,
            warnings=removed_warning,
            provenance=provenance_with_removed_warning,
        )
