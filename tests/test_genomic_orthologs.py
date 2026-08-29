from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from rejuvenationkit import EffectDirection
from rejuvenationkit.genomics import (
    FeatureNamespace,
    GeneSignature,
    GenomicFeatureType,
    OrthologAmbiguityPolicy,
    OrthologMap,
    OrthologRecord,
    SignatureFeature,
    read_ortholog_map,
    translate_signature,
)
from rejuvenationkit.genomics.resources import (
    FeatureCollection,
    FeatureDomain,
    QueryProvenance,
    ResourceSnapshot,
)


def human_domain() -> FeatureDomain:
    return FeatureDomain(
        species_taxon_id=9606,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
    )


def dog_domain() -> FeatureDomain:
    return FeatureDomain(
        species_taxon_id=9615,
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
    )


def resource_snapshot() -> ResourceSnapshot:
    return ResourceSnapshot(
        provider_id="ensembl",
        resource_id="HCOP",
        resource_release="2026-07",
        retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
        response_sha256="a" * 64,
    )


def source_query(
    resource: ResourceSnapshot,
    *,
    domain: FeatureDomain | None = None,
    feature_ids: tuple[str, ...] = ("H1", "H2", "H3"),
) -> FeatureCollection:
    return FeatureCollection(
        collection_id="human-signature-query",
        domain=domain or human_domain(),
        feature_ids=feature_ids,
        source_snapshot_id=resource.snapshot_id,
    )


def query_provenance(
    resource: ResourceSnapshot,
    *,
    domain: FeatureDomain | None = None,
    query: FeatureCollection | None = None,
    provider_id: str | None = None,
) -> QueryProvenance:
    resolved_query = query or source_query(resource)
    return QueryProvenance(
        provider_id=provider_id or resource.provider_id,
        provider_version="ensembl-adapter-1.0.0",
        resources=(resource,),
        domain=domain or human_domain(),
        retrieved_at=resource.retrieved_at,
        input_hash=resolved_query.content_hash,
        query_parameters={"operation": "homology"},
        software_versions={"rejuvenationkit": "0.3.0"},
    )


def source_signature() -> GeneSignature:
    return GeneSignature(
        signature_id="human-signature",
        version="1.0",
        name="Human response signature",
        features=(
            SignatureFeature(feature_id="H1", weight=2),
            SignatureFeature(feature_id="H2", weight=-1),
            SignatureFeature(feature_id="H3", weight=1),
        ),
        namespace=FeatureNamespace.ENSEMBL,
        species_taxon_id=9606,
        tissue="blood",
        target_name="biological_age_delta",
        target_unit="years",
        direction=EffectDirection.LOWER_IS_BETTER,
        resource_id="human-signature-resource",
    )


def mapping() -> OrthologMap:
    resource = resource_snapshot()
    query = source_query(resource)
    return OrthologMap(
        source_species_taxon_id=9606,
        target_species_taxon_id=9615,
        source_namespace=FeatureNamespace.ENSEMBL,
        target_namespace=FeatureNamespace.ENSEMBL,
        source_domain=human_domain(),
        target_domain=dog_domain(),
        resource_id="HCOP",
        resource_version="2026-07",
        records=(
            OrthologRecord(source_feature_id="H1", target_feature_id="D1", confidence=1.0),
            OrthologRecord(source_feature_id="H2", target_feature_id="D2A", confidence=0.9),
            OrthologRecord(source_feature_id="H2", target_feature_id="D2B", confidence=0.8),
        ),
        resource_snapshot=resource,
        source_query=query,
        query_provenance=query_provenance(resource, query=query),
        confidence_definition="Ensembl orthology confidence normalized to [0, 1]",
        confidence_scale=(0.0, 1.0),
    )


def mapping_with_records(
    records: tuple[OrthologRecord, ...],
    *,
    confidence_definition: str | None = "Ensembl orthology confidence normalized to [0, 1]",
    confidence_scale: tuple[float, float] | None = (0.0, 1.0),
) -> OrthologMap:
    resource = resource_snapshot()
    query = source_query(resource)
    return OrthologMap(
        source_species_taxon_id=9606,
        target_species_taxon_id=9615,
        source_namespace=FeatureNamespace.ENSEMBL,
        target_namespace=FeatureNamespace.ENSEMBL,
        source_domain=human_domain(),
        target_domain=dog_domain(),
        resource_id=resource.resource_id,
        resource_version=resource.resource_release,
        records=records,
        resource_snapshot=resource,
        source_query=query,
        query_provenance=query_provenance(resource, query=query),
        confidence_definition=confidence_definition,
        confidence_scale=confidence_scale,
    )


def test_ortholog_translation_reports_missing_and_ambiguous_features() -> None:
    result = translate_signature(
        source_signature(),
        mapping(),
        policy=OrthologAmbiguityPolicy.DISTRIBUTE_WEIGHT,
        minimum_retained_weight_fraction=0.7,
        target_tissue="blood",
    )

    weights = {item.feature_id: item.weight for item in result.signature.features}
    assert weights == {"D1": 2.0, "D2A": -0.5, "D2B": -0.5}
    assert result.mapped_source_feature_ids == ("H1", "H2")
    assert result.missing_source_feature_ids == ("H3",)
    assert result.ambiguous_source_feature_ids == ("H2",)
    assert result.retained_weight_fraction == 0.75
    assert result.post_aggregation_weight_fraction == 0.75
    assert result.signature.species_taxon_id == 9615
    assert "one_to_many_orthologs_present" in result.warnings
    assert "ortholog_features_missing" in result.warnings


def test_translation_carries_exact_provenance_and_preserves_tissue() -> None:
    ortholog_map = mapping()
    result = translate_signature(
        source_signature(),
        ortholog_map,
        policy=OrthologAmbiguityPolicy.DISTRIBUTE_WEIGHT,
        minimum_retained_weight_fraction=0.7,
    )

    assert ortholog_map.resource_snapshot is not None
    assert ortholog_map.query_provenance is not None
    assert result.mapping_snapshot_id == ortholog_map.resource_snapshot.snapshot_id
    assert result.mapping_query_hash == ortholog_map.query_provenance.query_hash
    assert result.mapping_content_hash == ortholog_map.content_hash
    assert f"ortholog_snapshot={result.mapping_snapshot_id}" in result.signature.resource_id
    assert f"ortholog_query={result.mapping_query_hash}" in result.signature.resource_id
    assert f"ortholog_content={result.mapping_content_hash}" in result.signature.resource_id
    assert result.source_domain == human_domain()
    assert result.target_domain == dog_domain()
    assert result.signature.tissue == source_signature().tissue
    assert result.fusion_eligibility == "not_fusible"
    assert not hasattr(result, "to_evidence")

    with pytest.raises(ValueError, match="cannot relabel"):
        translate_signature(
            source_signature(),
            ortholog_map,
            policy=OrthologAmbiguityPolicy.DISTRIBUTE_WEIGHT,
            minimum_retained_weight_fraction=0.7,
            target_tissue="liver",
        )


def test_ortholog_ambiguity_policies_are_explicit() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        translate_signature(source_signature(), mapping())

    highest = translate_signature(
        source_signature(),
        mapping(),
        policy=OrthologAmbiguityPolicy.SELECT_HIGHEST_CONFIDENCE,
        minimum_retained_weight_fraction=0.7,
    )
    assert ("H2", "D2A") in highest.selected_pairs
    assert ("H2", "D2B") not in highest.selected_pairs
    assert highest.signature.tissue == "blood"

    dropped = translate_signature(
        source_signature(),
        mapping(),
        policy=OrthologAmbiguityPolicy.DROP_AMBIGUOUS,
        minimum_retained_weight_fraction=0.4,
    )
    assert dropped.signature.features == (SignatureFeature(feature_id="D1", weight=2),)
    assert set(dropped.dropped_source_feature_ids) == {"H2", "H3"}


def test_confidence_dependent_operations_fail_closed() -> None:
    records_without_confidence = (
        OrthologRecord(source_feature_id="H1", target_feature_id="D1"),
        OrthologRecord(source_feature_id="H2", target_feature_id="D2A"),
        OrthologRecord(source_feature_id="H2", target_feature_id="D2B"),
    )
    unscored = mapping_with_records(
        records_without_confidence,
        confidence_definition=None,
        confidence_scale=None,
    )
    distributed = translate_signature(
        source_signature(),
        unscored,
        policy=OrthologAmbiguityPolicy.DISTRIBUTE_WEIGHT,
        minimum_retained_weight_fraction=0.7,
    )
    assert len(distributed.signature.features) == 3
    assert all(item.confidence is None for item in unscored.records)

    with pytest.raises(ValueError, match="definition and scale"):
        translate_signature(
            source_signature(),
            unscored,
            policy=OrthologAmbiguityPolicy.SELECT_HIGHEST_CONFIDENCE,
            minimum_retained_weight_fraction=0.7,
        )
    with pytest.raises(ValueError, match="definition and scale"):
        translate_signature(
            source_signature(),
            unscored,
            minimum_confidence=0.5,
            policy=OrthologAmbiguityPolicy.DISTRIBUTE_WEIGHT,
            minimum_retained_weight_fraction=0.7,
        )

    declared_but_missing = mapping_with_records(records_without_confidence)
    with pytest.raises(ValueError, match="every candidate"):
        translate_signature(
            source_signature(),
            declared_but_missing,
            minimum_confidence=0.5,
            policy=OrthologAmbiguityPolicy.DISTRIBUTE_WEIGHT,
            minimum_retained_weight_fraction=0.7,
        )
    with pytest.raises(ValueError, match="every candidate"):
        translate_signature(
            source_signature(),
            declared_but_missing,
            policy=OrthologAmbiguityPolicy.SELECT_HIGHEST_CONFIDENCE,
            minimum_retained_weight_fraction=0.7,
        )

    tied = mapping_with_records(
        (
            OrthologRecord(source_feature_id="H1", target_feature_id="D1", confidence=1.0),
            OrthologRecord(source_feature_id="H2", target_feature_id="D2A", confidence=0.9),
            OrthologRecord(source_feature_id="H2", target_feature_id="D2B", confidence=0.9),
        )
    )
    with pytest.raises(ValueError, match="tie"):
        translate_signature(
            source_signature(),
            tied,
            policy=OrthologAmbiguityPolicy.SELECT_HIGHEST_CONFIDENCE,
            minimum_retained_weight_fraction=0.7,
        )


def test_ortholog_translation_enforces_domain_and_retained_coverage() -> None:
    with pytest.raises(ValueError, match="below minimum"):
        translate_signature(
            source_signature(),
            mapping(),
            policy=OrthologAmbiguityPolicy.DROP_AMBIGUOUS,
            minimum_retained_weight_fraction=0.8,
        )
    wrong_species = source_signature().model_copy(update={"species_taxon_id": 10090})
    with pytest.raises(ValueError, match="species"):
        translate_signature(wrong_species, mapping())
    wrong_namespace = source_signature().model_copy(update={"namespace": FeatureNamespace.HGNC})
    with pytest.raises(ValueError, match="namespace"):
        translate_signature(wrong_namespace, mapping())
    with pytest.raises(ValueError, match="minimum_confidence"):
        translate_signature(source_signature(), mapping(), minimum_confidence=1.1)


def test_ortholog_schemas_reject_invalid_resources() -> None:
    with pytest.raises(ValidationError, match="confidence"):
        OrthologRecord(
            source_feature_id="H1",
            target_feature_id="D1",
            confidence=float("nan"),
        )
    with pytest.raises(ValidationError, match="differ"):
        mapping().model_copy(update={"target_species_taxon_id": 9606}).model_validate(
            mapping().model_copy(update={"target_species_taxon_id": 9606}).model_dump()
        )
    duplicated = (*mapping().records, mapping().records[0])
    with pytest.raises(ValidationError, match="unique"):
        mapping().model_copy(update={"records": duplicated}).model_validate(
            mapping().model_copy(update={"records": duplicated}).model_dump()
        )


def test_ortholog_translation_reports_target_cancellation_and_relationship_filtering() -> None:
    collision_map = mapping_with_records(
        (
            OrthologRecord(source_feature_id="H1", target_feature_id="D1"),
            OrthologRecord(source_feature_id="H2", target_feature_id="D1"),
        )
    )
    collision = translate_signature(
        source_signature(),
        collision_map,
        minimum_retained_weight_fraction=0.7,
    )
    assert collision.retained_weight_fraction == 0.75
    assert collision.post_aggregation_weight_fraction == 0.25
    assert collision.translated_absolute_weight == 1.0
    assert collision.translated_weight_fraction == 0.25
    assert collision.target_collisions[0].target_feature_id == "D1"
    assert collision.target_collisions[0].source_feature_ids == ("H1", "H2")
    assert "multiple_source_features_share_target_ortholog" in collision.warnings

    relationship_map = mapping_with_records(
        (
            OrthologRecord(
                source_feature_id="H1",
                target_feature_id="D1",
                relationship="paralog",
            ),
            *mapping().records[1:],
        )
    )
    filtered = translate_signature(
        source_signature(),
        relationship_map,
        policy=OrthologAmbiguityPolicy.DISTRIBUTE_WEIGHT,
        minimum_retained_weight_fraction=0.2,
    )
    assert "ortholog_relationships_excluded" in filtered.warnings
    assert "H1" in filtered.missing_source_feature_ids


def test_read_ortholog_map_freezes_resource_release_and_imports_evidence() -> None:
    resource = ResourceSnapshot(
        provider_id="ensembl",
        resource_id="compara",
        resource_release="release-115",
        retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
        response_sha256="a" * 64,
    )
    query = source_query(resource)
    provenance = query_provenance(resource, query=query)
    frame = pd.DataFrame(
        {
            "human": ["H2", "H1"],
            "dog": ["D2", "D1"],
            "confidence": [0.8, 1.0],
            "relationship": ["ortholog_one2many", "ortholog_one2one"],
            "evidence": ["homology;synteny", "synteny"],
        }
    )

    imported = read_ortholog_map(
        frame,
        source_feature_id_column="human",
        target_feature_id_column="dog",
        confidence_column="confidence",
        confidence_definition="Ensembl orthology confidence normalized to [0, 1]",
        confidence_scale=(0.0, 1.0),
        relationship_column="relationship",
        evidence_ids_column="evidence",
        source_domain=human_domain(),
        target_domain=dog_domain(),
        resource=resource,
        source_query=query,
        provenance=provenance,
    )

    assert imported.resource_version == "release-115"
    assert imported.resource_snapshot == resource
    assert imported.resource_snapshot.response_sha256 == "a" * 64
    assert imported.normalized_import_checksum != imported.resource_snapshot.response_sha256
    assert imported.query_provenance is not None
    assert imported.query_provenance.response_checksum == imported.normalized_import_checksum
    assert [item.source_feature_id for item in imported.records] == ["H1", "H2"]
    assert imported.records[1].evidence_ids == ("homology", "synteny")


def test_normalized_import_checksum_is_order_independent_and_content_sensitive() -> None:
    resource = resource_snapshot()
    query = source_query(resource)
    provenance = query_provenance(resource, query=query)
    frame = pd.DataFrame(
        {
            "human": ["H2", "H1"],
            "dog": ["D2", "D1"],
            "confidence": [0.8, 1.0],
            "evidence": ["synteny;homology", "synteny"],
        }
    )

    def imported(data: pd.DataFrame) -> OrthologMap:
        return read_ortholog_map(
            data,
            source_feature_id_column="human",
            target_feature_id_column="dog",
            confidence_column="confidence",
            confidence_definition="Ensembl confidence",
            confidence_scale=(0.0, 1.0),
            evidence_ids_column="evidence",
            source_domain=human_domain(),
            target_domain=dog_domain(),
            resource=resource,
            source_query=query,
            provenance=provenance,
        )

    baseline = imported(frame)
    reordered = imported(frame.iloc[::-1].reset_index(drop=True))
    changed_frame = frame.copy()
    changed_frame.loc[0, "confidence"] = 0.7
    changed = imported(changed_frame)

    assert reordered.content_hash == baseline.content_hash
    assert changed.content_hash != baseline.content_hash
    assert baseline.resource_snapshot is not None
    assert baseline.resource_snapshot.response_sha256 == "a" * 64
    assert baseline.resource_snapshot.response_sha256 != baseline.content_hash


def test_import_requires_exact_domain_resource_and_confidence_semantics() -> None:
    resource = resource_snapshot()
    query = source_query(resource)
    frame = pd.DataFrame(
        {
            "human": ["H1"],
            "dog": ["D1"],
            "confidence": [1.0],
        }
    )
    wrong_domain_provenance = query_provenance(
        resource,
        domain=dog_domain(),
        query=query,
    )
    with pytest.raises(ValueError, match="exact source domain"):
        read_ortholog_map(
            frame,
            source_feature_id_column="human",
            target_feature_id_column="dog",
            source_domain=human_domain(),
            target_domain=dog_domain(),
            resource=resource,
            source_query=query,
            provenance=wrong_domain_provenance,
            confidence_column="confidence",
            confidence_definition="Ensembl confidence",
            confidence_scale=(0.0, 1.0),
        )

    with pytest.raises(ValueError, match="required with confidence_column"):
        read_ortholog_map(
            frame,
            source_feature_id_column="human",
            target_feature_id_column="dog",
            source_domain=human_domain(),
            target_domain=dog_domain(),
            resource=resource,
            source_query=query,
            provenance=query_provenance(resource, query=query),
            confidence_column="confidence",
        )

    other_resource = resource.model_copy(update={"resource_id": "other"})
    with pytest.raises(ValueError, match="present in query provenance"):
        read_ortholog_map(
            frame.drop(columns="confidence"),
            source_feature_id_column="human",
            target_feature_id_column="dog",
            source_domain=human_domain(),
            target_domain=dog_domain(),
            resource=other_resource,
            source_query=query,
            provenance=query_provenance(resource, query=query),
        )

    wrong_query = source_query(resource, feature_ids=("H1",))
    with pytest.raises(ValueError, match="input_hash"):
        read_ortholog_map(
            frame.drop(columns="confidence"),
            source_feature_id_column="human",
            target_feature_id_column="dog",
            source_domain=human_domain(),
            target_domain=dog_domain(),
            resource=resource,
            source_query=wrong_query,
            provenance=query_provenance(resource, query=query),
        )

    executor_provenance = query_provenance(
        resource,
        query=query,
        provider_id="smarts-bio-executor",
    )
    imported_by_distinct_executor = read_ortholog_map(
        frame.drop(columns="confidence"),
        source_feature_id_column="human",
        target_feature_id_column="dog",
        source_domain=human_domain(),
        target_domain=dog_domain(),
        resource=resource,
        source_query=query,
        provenance=executor_provenance,
    )
    assert imported_by_distinct_executor.query_provenance is not None
    assert imported_by_distinct_executor.query_provenance.provider_id == "smarts-bio-executor"


def test_legacy_map_construction_derives_domains_but_translation_requires_provenance() -> None:
    legacy = OrthologMap(
        source_species_taxon_id=9606,
        target_species_taxon_id=9615,
        source_namespace=FeatureNamespace.ENSEMBL,
        target_namespace=FeatureNamespace.ENSEMBL,
        resource_id="manual-map",
        resource_version="1.0",
        records=(OrthologRecord(source_feature_id="H1", target_feature_id="D1"),),
    )

    assert legacy.source_domain == human_domain()
    assert legacy.target_domain == dog_domain()
    assert len(legacy.content_hash) == 64
    with pytest.raises(ValueError, match="snapshot-bound"):
        translate_signature(
            source_signature(),
            legacy,
            minimum_retained_weight_fraction=0.4,
        )
