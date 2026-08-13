import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from rejuvenationkit import EffectDirection, Modality
from rejuvenationkit.genomics import (
    FeatureEffect,
    FeatureNamespace,
    GeneSignature,
    GenomicFeature,
    GenomicFeatureType,
    GenomicMatrix,
    GenomicMatrixProvenance,
    GenomicSample,
    MatrixScale,
    MissingFeaturePolicy,
    SignatureContrastConfig,
    SignatureFeature,
    aggregate_feature_effects,
    estimate_signature_contrast,
    estimate_signature_contrasts,
    read_feature_effects,
    score_weighted_signature,
)


def signature(*, tissue: str | None = "blood") -> GeneSignature:
    return GeneSignature(
        signature_id="mtor-autophagy",
        version="1.0",
        name="mTOR/autophagy response",
        features=(
            SignatureFeature(feature_id="g1", weight=1.0),
            SignatureFeature(feature_id="g2", weight=-2.0),
            SignatureFeature(feature_id="g3", weight=1.0),
        ),
        namespace=FeatureNamespace.ENSEMBL,
        species_taxon_id=9615,
        tissue=tissue,
        target_name="biological_age_delta",
        target_unit="years",
        direction=EffectDirection.LOWER_IS_BETTER,
        resource_id="prespecified-signature-v1",
    )


def matrix() -> GenomicMatrix:
    cohorts = ("control", "control", "control", "treated", "treated", "treated")
    values = np.asarray(
        [
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 3.0, 1.0],
        ]
    )
    samples = tuple(
        GenomicSample(
            sample_id=f"s{index}",
            subject_id=f"dog-{index}",
            tissue="blood",
            species_taxon_id=9615,
            cohort=cohort,
        )
        for index, cohort in enumerate(cohorts)
    )
    features = tuple(
        GenomicFeature(
            feature_id=identifier,
            feature_type=GenomicFeatureType.GENE,
            namespace=FeatureNamespace.ENSEMBL,
        )
        for identifier in ("g1", "g2", "g3")
    )
    return GenomicMatrix(
        values=values,
        samples=samples,
        features=features,
        scale=MatrixScale.NORMALIZED_EXPRESSION,
        provenance=GenomicMatrixProvenance(source_id="synthetic-canine-rna"),
    )


def test_weighted_signature_score_is_hand_computable() -> None:
    scores = score_weighted_signature(matrix(), signature())

    assert scores.feature_coverage == 1
    assert scores.missing_feature_ids == ()
    assert scores.scores[0].score == pytest.approx(0.0)
    assert scores.scores[1].score == pytest.approx(0.5)
    assert scores.scores[3].score == pytest.approx(-1.0)
    assert scores.to_frame().shape == (6, 5)
    assert scores.matrix_provenance_id == "synthetic-canine-rna"
    assert scores.matrix_scale == "normalized_expression"
    assert len(scores.signature_fingerprint) == 64


def test_signature_missing_feature_policies_are_explicit() -> None:
    reduced = matrix().subset_features(("g1", "g2"))
    with pytest.raises(ValueError, match="absent"):
        score_weighted_signature(
            reduced,
            signature(),
            missing_policy=MissingFeaturePolicy.ERROR,
            minimum_feature_coverage=0.5,
        )
    dropped = score_weighted_signature(
        reduced,
        signature(),
        missing_policy=MissingFeaturePolicy.DROP_AND_RENORMALIZE,
        minimum_feature_coverage=0.5,
    )
    imputed = score_weighted_signature(
        reduced,
        signature(),
        missing_policy=MissingFeaturePolicy.IMPUTE_ZERO,
        minimum_feature_coverage=0.5,
    )
    assert dropped.feature_coverage == 0.75
    assert dropped.scores[0].score == pytest.approx(0.0)
    assert imputed.scores[0].score == pytest.approx(0.0)
    assert "signature_features_missing" in dropped.warnings
    with pytest.raises(ValueError, match="below minimum"):
        score_weighted_signature(reduced, signature(), minimum_feature_coverage=0.9)


def test_sample_level_missing_values_are_reported_or_rejected() -> None:
    source = matrix()
    values = source.dense_values()
    values[0, 0] = np.nan
    missing = GenomicMatrix(
        values=values,
        samples=source.samples,
        features=source.features,
        scale=source.scale,
        provenance=source.provenance,
    )
    with pytest.raises(ValueError, match="sample s0"):
        score_weighted_signature(missing, signature(), missing_policy=MissingFeaturePolicy.ERROR)
    scores = score_weighted_signature(missing, signature())
    assert scores.scores[0].observed_weight_fraction == 0.75
    assert "sample_signature_values_missing" in scores.warnings
    values = source.dense_values()
    values[0, :2] = np.nan
    low_coverage = GenomicMatrix(
        values=values,
        samples=source.samples,
        features=source.features,
        scale=source.scale,
        provenance=source.provenance,
    )
    with pytest.raises(ValueError, match="coverage"):
        score_weighted_signature(low_coverage, signature())


def test_signature_scoring_rejects_raw_counts_and_definition_relabeling() -> None:
    source = matrix()
    raw = GenomicMatrix(
        values=source.dense_values(),
        samples=source.samples,
        features=source.features,
        scale=MatrixScale.RAW_COUNTS,
        provenance=source.provenance,
    )
    with pytest.raises(ValueError, match="not allowed"):
        score_weighted_signature(raw, signature())

    scores = score_weighted_signature(source, signature())
    relabeled = signature().model_copy(update={"target_unit": "months"})
    with pytest.raises(ValueError, match="definitions"):
        estimate_signature_contrast(
            scores,
            relabeled,
            SignatureContrastConfig(treated_cohort="treated", control_cohort="control"),
        )


def test_signature_contrast_rejects_subjects_in_both_arms() -> None:
    source = matrix()
    samples = list(source.samples)
    samples[3] = samples[3].model_copy(update={"subject_id": samples[0].subject_id})
    overlapping = GenomicMatrix(
        values=source.values,
        samples=tuple(samples),
        features=source.features,
        scale=source.scale,
        provenance=source.provenance,
    )
    scores = score_weighted_signature(overlapping, signature())
    with pytest.raises(ValueError, match="both treated and control"):
        estimate_signature_contrast(
            scores,
            signature(),
            SignatureContrastConfig(treated_cohort="treated", control_cohort="control"),
        )


def test_subject_cluster_bootstrap_is_deterministic_and_converts_to_evidence() -> None:
    scores = score_weighted_signature(matrix(), signature())
    config = SignatureContrastConfig(
        treated_cohort="treated",
        control_cohort="control",
        bootstrap_iterations=500,
        random_seed=12,
    )
    first = estimate_signature_contrast(scores, signature(), config)
    second = estimate_signature_contrast(scores, signature(), config)

    assert first == second
    assert first.estimate < 0
    assert first.standard_error > 0
    assert first.treated_subjects == 3
    assert first.control_subjects == 3
    evidence = first.to_evidence(
        evidence_id="rna-signature",
        modality=Modality.TRANSCRIPTOMICS,
        calibration_id="external-canine-validation-v1",
        correlation_group="shared-rna-samples",
    )
    assert evidence.evidence_id == "rna-signature"
    assert evidence.estimand.name == "biological_age_delta"
    assert evidence.correlation_group == "shared-rna-samples"


def test_joint_signature_bootstrap_preserves_subject_covariance() -> None:
    first_signature = signature()
    second_signature = first_signature.model_copy(
        update={
            "signature_id": "mtor-autophagy-related",
            "name": "Related pathway",
            "features": (
                SignatureFeature(feature_id="g1", weight=1),
                SignatureFeature(feature_id="g2", weight=-1),
            ),
        }
    )
    inputs = (
        (score_weighted_signature(matrix(), first_signature), first_signature),
        (score_weighted_signature(matrix(), second_signature), second_signature),
    )
    config = SignatureContrastConfig(
        treated_cohort="treated",
        control_cohort="control",
        bootstrap_iterations=500,
        random_seed=19,
    )
    first = estimate_signature_contrasts(inputs, config)
    second = estimate_signature_contrasts(inputs, config)

    assert first == second
    assert first.covariance.evidence_ids == (
        "mtor-autophagy",
        "mtor-autophagy-related",
    )
    assert first.covariance.effective_sample_size == 6
    assert first.covariance.covariance[0][1] != 0
    assert "diagonal-shrinkage=0.1" in first.covariance.source_id
    assert first.correlation[0][0] == 1
    evidence, covariance = first.to_evidence(
        modality=Modality.TRANSCRIPTOMICS,
        calibration_id="external-v1",
        correlation_group="shared-subjects",
    )
    assert tuple(item.evidence_id for item in evidence) == first.covariance.evidence_ids
    assert covariance == first.covariance


def test_joint_signature_bootstrap_rejects_incompatible_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        estimate_signature_contrasts(
            (),
            SignatureContrastConfig(treated_cohort="treated", control_cohort="control"),
        )
    source = signature()
    scores = score_weighted_signature(matrix(), source)
    duplicate = source.model_copy(update={"name": "duplicate"})
    with pytest.raises(ValueError, match="unique"):
        estimate_signature_contrasts(
            ((scores, source), (scores, duplicate)),
            SignatureContrastConfig(treated_cohort="treated", control_cohort="control"),
        )
    different_target = source.model_copy(
        update={"signature_id": "other", "target_name": "different"}
    )
    different_scores = score_weighted_signature(matrix(), different_target)
    heterogeneous = estimate_signature_contrasts(
        ((scores, source), (different_scores, different_target)),
        SignatureContrastConfig(treated_cohort="treated", control_cohort="control"),
    )
    assert {item.target_name for item in heterogeneous.estimates} == {
        "biological_age_delta",
        "different",
    }


def feature_effects() -> tuple[FeatureEffect, ...]:
    return tuple(
        FeatureEffect(
            feature_id=identifier,
            feature_type=GenomicFeatureType.GENE,
            namespace=FeatureNamespace.ENSEMBL,
            modality=Modality.TRANSCRIPTOMICS,
            contrast="rapamycin-minus-control",
            effect=value,
            standard_error=error,
            effect_unit="log2_fold_change",
            species_taxon_id=9615,
            tissue="blood",
            provenance_id="deseq2-analysis-v1",
        )
        for identifier, value, error in (
            ("g1", -1.0, 0.2),
            ("g2", 0.5, 0.3),
            ("g3", -0.5, 0.4),
        )
    )


def test_feature_effect_aggregation_propagates_covariance() -> None:
    effects = feature_effects()
    independent = aggregate_feature_effects(effects, signature())
    covariance = pd.DataFrame(
        [
            [0.04, -0.03, 0.0],
            [-0.03, 0.09, -0.02],
            [0.0, -0.02, 0.16],
        ],
        index=["g1", "g2", "g3"],
        columns=["g1", "g2", "g3"],
    )
    correlated = aggregate_feature_effects(effects, signature(), covariance=covariance)

    assert independent.estimate == pytest.approx((-1.0 - 1.0 - 0.5) / 4)
    assert "feature_covariance_not_supplied" in independent.warnings
    assert correlated.standard_error > independent.standard_error
    assert correlated.uncertainty_method == "correlation_aware_delta_method"


def test_read_feature_effects_uses_explicit_column_mapping() -> None:
    frame = pd.DataFrame(
        {
            "gene": ["g1", "g2"],
            "lfc": [-1.0, 0.5],
            "lfc_se": [0.2, 0.3],
            "padj": [0.01, 0.2],
        }
    )
    effects = read_feature_effects(
        frame,
        feature_id_column="gene",
        effect_column="lfc",
        standard_error_column="lfc_se",
        adjusted_p_value_column="padj",
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
        modality=Modality.TRANSCRIPTOMICS,
        contrast="rapamycin-minus-control",
        effect_unit="log2_fold_change",
        species_taxon_id=9615,
        tissue="blood",
        provenance_id="deseq2-v1",
    )
    assert effects[0].adjusted_p_value == 0.01
    assert effects[1].feature_id == "g2"
    frame.loc[1, "padj"] = np.nan
    missing_optional = read_feature_effects(
        frame,
        feature_id_column="gene",
        effect_column="lfc",
        standard_error_column="lfc_se",
        adjusted_p_value_column="padj",
        feature_type=GenomicFeatureType.GENE,
        namespace=FeatureNamespace.ENSEMBL,
        modality=Modality.TRANSCRIPTOMICS,
        contrast="rapamycin-minus-control",
        effect_unit="log2_fold_change",
        species_taxon_id=9615,
        tissue="blood",
        provenance_id="deseq2-v1",
    )
    assert missing_optional[1].adjusted_p_value is None
    with pytest.raises(ValueError, match="columns"):
        read_feature_effects(
            frame.drop(columns="lfc_se"),
            feature_id_column="gene",
            effect_column="lfc",
            standard_error_column="lfc_se",
            feature_type=GenomicFeatureType.GENE,
            namespace=FeatureNamespace.ENSEMBL,
            modality=Modality.TRANSCRIPTOMICS,
            contrast="contrast",
            effect_unit="log2_fold_change",
            species_taxon_id=9615,
            tissue="blood",
            provenance_id="analysis",
        )


def test_signature_and_effect_models_reject_invalid_definitions() -> None:
    with pytest.raises(ValidationError, match="nonzero"):
        SignatureFeature(feature_id="g1", weight=0)
    with pytest.raises(ValidationError, match="unique"):
        signature().model_copy(
            update={
                "features": (
                    SignatureFeature(feature_id="g1", weight=1),
                    SignatureFeature(feature_id="g1", weight=2),
                )
            }
        ).model_validate(
            signature()
            .model_copy(
                update={
                    "features": (
                        SignatureFeature(feature_id="g1", weight=1),
                        SignatureFeature(feature_id="g1", weight=2),
                    )
                }
            )
            .model_dump()
        )
    with pytest.raises(ValidationError, match="finite"):
        feature_effects()[0].model_copy(update={"effect": float("nan")}).model_validate(
            feature_effects()[0].model_copy(update={"effect": float("nan")}).model_dump()
        )


def test_feature_effect_aggregation_validates_reporting_policy() -> None:
    with pytest.raises(ValueError, match="minimum_feature_coverage"):
        aggregate_feature_effects(feature_effects(), signature(), minimum_feature_coverage=0)
    with pytest.raises(ValueError, match="confidence_level"):
        aggregate_feature_effects(feature_effects(), signature(), confidence_level=1)
