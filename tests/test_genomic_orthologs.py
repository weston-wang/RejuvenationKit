import pytest
from pydantic import ValidationError

from rejuvenationkit import EffectDirection
from rejuvenationkit.genomics import (
    FeatureNamespace,
    GeneSignature,
    OrthologAmbiguityPolicy,
    OrthologMap,
    OrthologRecord,
    SignatureFeature,
    translate_signature,
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
    return OrthologMap(
        source_species_taxon_id=9606,
        target_species_taxon_id=9615,
        source_namespace=FeatureNamespace.ENSEMBL,
        target_namespace=FeatureNamespace.ENSEMBL,
        resource_id="HCOP",
        resource_version="2026-07",
        records=(
            OrthologRecord(source_feature_id="H1", target_feature_id="D1", confidence=1.0),
            OrthologRecord(source_feature_id="H2", target_feature_id="D2A", confidence=0.9),
            OrthologRecord(source_feature_id="H2", target_feature_id="D2B", confidence=0.8),
        ),
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
    collision_map = mapping().model_copy(
        update={
            "records": (
                OrthologRecord(source_feature_id="H1", target_feature_id="D1"),
                OrthologRecord(source_feature_id="H2", target_feature_id="D1"),
            )
        }
    )
    collision = translate_signature(
        source_signature(),
        collision_map,
        minimum_retained_weight_fraction=0.7,
    )
    assert collision.retained_weight_fraction == 0.75
    assert collision.post_aggregation_weight_fraction == 0.25
    assert "multiple_source_features_share_target_ortholog" in collision.warnings

    relationship_map = mapping().model_copy(
        update={
            "records": (
                OrthologRecord(
                    source_feature_id="H1",
                    target_feature_id="D1",
                    relationship="paralog",
                ),
                *mapping().records[1:],
            )
        }
    )
    filtered = translate_signature(
        source_signature(),
        relationship_map,
        policy=OrthologAmbiguityPolicy.DISTRIBUTE_WEIGHT,
        minimum_retained_weight_fraction=0.2,
    )
    assert "ortholog_relationships_excluded" in filtered.warnings
    assert "H1" in filtered.missing_source_feature_ids
