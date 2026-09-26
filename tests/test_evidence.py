from hashlib import sha256
from math import isclose

import pytest
from pydantic import ValidationError

from rejuvenationkit import (
    CalibrationReference,
    CalibrationValidationStatus,
    CrossModalityCovariancePolicy,
    EffectDirection,
    Estimand,
    EvidenceCovariance,
    EvidenceDispersionPolicy,
    EvidenceEstimate,
    EvidenceFusionConfig,
    EvidenceWeightConstraint,
    FusionModel,
    GeneralizedLeastSquaresFusion,
    HierarchicalEvidenceFusion,
    MissingCovariancePolicy,
    MissingEvidencePolicy,
    Modality,
)


def estimand(name: str = "biological_age_delta") -> Estimand:
    return Estimand(
        name=name,
        unit="years",
        direction=EffectDirection.LOWER_IS_BETTER,
        population="treated-minus-control",
        time_contrast="month-6-minus-baseline",
    )


def evidence(
    evidence_id: str,
    modality: Modality,
    value: float,
    error: float = 1.0,
    *,
    target: Estimand | None = None,
) -> EvidenceEstimate:
    resolved_target = target or estimand()
    calibration_id = f"calibration:{evidence_id}"
    return EvidenceEstimate(
        evidence_id=evidence_id,
        modality=modality,
        estimand=resolved_target,
        estimate=value,
        standard_error=error,
        calibration_id=calibration_id,
        calibration_reference=CalibrationReference(
            calibration_id=calibration_id,
            artifact_hash=sha256(calibration_id.encode()).hexdigest(),
            status=CalibrationValidationStatus.EXTERNALLY_VALIDATED,
            estimand=resolved_target,
            method="held-out external calibration",
            validation_provenance_id=f"validation:{evidence_id}",
        ),
        provenance_id=f"analysis:{evidence_id}",
        tissue="blood",
        species_taxon_id=9615,
        correlation_group="shared-rna" if modality is Modality.TRANSCRIPTOMICS else None,
    )


def test_diagonal_gls_matches_inverse_variance_fixed_effect() -> None:
    estimates = (
        evidence("methylation-clock", Modality.METHYLATION, -4.0, 1.0),
        evidence("rna-clock", Modality.TRANSCRIPTOMICS, -2.0, 2.0),
    )
    result = GeneralizedLeastSquaresFusion().fuse(estimates)

    assert isclose(result.estimate, -3.6)
    assert isclose(result.standard_error, (1 / 1.25) ** 0.5)
    assert result.evidence_weights == pytest.approx({"methylation-clock": 0.8, "rna-clock": 0.2})
    assert result.covariance_source_id == "reported-independent-standard-errors"
    assert result.condition_number == 4.0
    assert result.regularization_applied == 0
    assert result.effective_evidence_count == pytest.approx(1 / (0.8**2 + 0.2**2))
    assert result.weight_concentration_effective_count == result.effective_evidence_count
    assert "shared_correlation_group_assumed_independent" not in result.warnings


def test_positive_correlation_prevents_duplicate_signatures_from_overstating_precision() -> None:
    estimates = (
        evidence("autophagy", Modality.TRANSCRIPTOMICS, -3.0),
        evidence("mtorc1", Modality.TRANSCRIPTOMICS, -3.2),
    )
    with pytest.raises(ValueError, match="joint covariance"):
        GeneralizedLeastSquaresFusion().fuse(estimates)
    independent = GeneralizedLeastSquaresFusion(
        EvidenceFusionConfig(
            missing_covariance_policy=MissingCovariancePolicy.WARN_ASSUME_INDEPENDENT
        )
    ).fuse(estimates)
    assert "shared_correlation_group_assumed_independent" in independent.warnings
    covariance = EvidenceCovariance.from_correlation(
        estimates,
        ((1.0, 0.8), (0.8, 1.0)),
        source_id="held-out-subject-bootstrap-v1",
        effective_sample_size=40,
    )
    correlated = GeneralizedLeastSquaresFusion().fuse(estimates, covariance)

    assert correlated.standard_error > independent.standard_error
    assert correlated.standard_error == pytest.approx((0.9) ** 0.5)
    assert correlated.modality_weights[Modality.TRANSCRIPTOMICS] == pytest.approx(1.0)
    assert "multiple_estimates_share_modality" in correlated.warnings
    assert len(correlated.leave_one_evidence_out) == 2
    assert correlated.leave_one_modality_out == ()


def test_covariance_is_reordered_by_evidence_id() -> None:
    estimates = (
        evidence("a", Modality.CLINICAL, 1.0, 1.0),
        evidence("b", Modality.METHYLATION, 3.0, 2.0),
    )
    covariance = EvidenceCovariance(
        evidence_ids=("b", "a"),
        covariance=((4.0, 0.0), (0.0, 1.0)),
        source_id="reordered",
    )
    result = GeneralizedLeastSquaresFusion().fuse(estimates, covariance)

    assert result.estimate == pytest.approx(1.4)
    assert result.evidence_weights == pytest.approx({"a": 0.8, "b": 0.2})
    assert {item.omitted_modality for item in result.leave_one_modality_out} == {
        Modality.CLINICAL,
        Modality.METHYLATION,
    }


def test_negative_gls_weight_and_quality_flag_are_reported() -> None:
    estimates = (
        evidence("a", Modality.CLINICAL, 0.0, 1.0),
        evidence("b", Modality.METHYLATION, 1.0, 2.0).model_copy(
            update={"quality_flags": ("weak-domain-match",)}
        ),
    )
    covariance = EvidenceCovariance(
        evidence_ids=("a", "b"),
        covariance=((1.0, 1.5), (1.5, 4.0)),
        source_id="correlated",
    )
    result = GeneralizedLeastSquaresFusion().fuse(estimates, covariance)

    assert result.evidence_weights["b"] < 0
    assert "negative_gls_weight" in result.warnings
    assert "input_quality_flags_present" in result.warnings

    constrained = GeneralizedLeastSquaresFusion(
        EvidenceFusionConfig(
            weight_constraint=EvidenceWeightConstraint.NONNEGATIVE,
        )
    ).fuse(estimates, covariance)
    assert all(weight >= 0 for weight in constrained.evidence_weights.values())
    assert constrained.effective_evidence_count >= 1
    assert constrained.standard_error > result.standard_error
    assert "nonnegative_weight_constraint_active" in constrained.warnings


def test_singular_covariance_is_regularized_or_rejected_by_policy() -> None:
    estimates = (
        evidence("a", Modality.CLINICAL, 1.0),
        evidence("b", Modality.METHYLATION, 1.0),
    )
    covariance = EvidenceCovariance(
        evidence_ids=("a", "b"),
        covariance=((1.0, 1.0), (1.0, 1.0)),
        source_id="duplicated-estimator",
    )
    regularized = GeneralizedLeastSquaresFusion().fuse(estimates, covariance)
    assert regularized.regularization_applied > 0
    assert "covariance_regularized" in regularized.warnings

    strict = GeneralizedLeastSquaresFusion(EvidenceFusionConfig(auto_regularize=False))
    with pytest.raises(ValueError, match="singular"):
        strict.fuse(estimates, covariance)


def test_non_psd_and_variance_mismatch_are_rejected() -> None:
    estimates = (
        evidence("a", Modality.CLINICAL, 1.0),
        evidence("b", Modality.METHYLATION, 1.0),
    )
    non_psd = EvidenceCovariance(
        evidence_ids=("a", "b"),
        covariance=((1.0, 2.0), (2.0, 1.0)),
        source_id="invalid",
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        GeneralizedLeastSquaresFusion().fuse(estimates, non_psd)

    mismatched = EvidenceCovariance(
        evidence_ids=("a", "b"),
        covariance=((2.0, 0.0), (0.0, 1.0)),
        source_id="wrong-diagonal",
    )
    with pytest.raises(ValueError, match="standard errors"):
        GeneralizedLeastSquaresFusion().fuse(estimates, mismatched)


def test_covariance_validation_is_invariant_to_marginal_scale() -> None:
    with pytest.raises(ValidationError, match="symmetric relative"):
        EvidenceCovariance(
            evidence_ids=("a", "b"),
            covariance=((1e-12, 9e-13), (0.0, 1e-12)),
            source_id="materially-asymmetric-at-small-scale",
        )

    estimates = (
        evidence("a", Modality.CLINICAL, 0.0, 1.0),
        evidence("b", Modality.METHYLATION, 0.0, 1e-12),
    )
    impossible_correlation = EvidenceCovariance(
        evidence_ids=("a", "b"),
        covariance=((1.0, 2e-12), (2e-12, 1e-24)),
        source_id="standardized-correlation-greater-than-one",
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        GeneralizedLeastSquaresFusion().fuse(estimates, impossible_correlation)

    wrong_small_variance = EvidenceCovariance(
        evidence_ids=("a", "b"),
        covariance=((1.0, 0.0), (0.0, 1e-12)),
        source_id="scale-dependent-diagonal-mismatch",
    )
    with pytest.raises(ValueError, match="standard errors"):
        GeneralizedLeastSquaresFusion().fuse(estimates, wrong_small_variance)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (((1.0, 0.0),), "square"),
        (((1.0, 0.1), (0.2, 1.0)), "symmetric"),
        (((1.0, float("nan")), (float("nan"), 1.0)), "finite"),
        (((0.0, 0.0), (0.0, 1.0)), "positive"),
    ],
)
def test_covariance_schema_rejects_invalid_matrices(
    covariance: tuple[tuple[float, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        EvidenceCovariance(
            evidence_ids=("a", "b"),
            covariance=covariance,
            source_id="invalid",
        )


def test_correlation_constructor_validates_correlation_contract() -> None:
    estimates = (
        evidence("a", Modality.CLINICAL, 1.0),
        evidence("b", Modality.METHYLATION, 1.0),
    )
    with pytest.raises(ValueError, match="square"):
        EvidenceCovariance.from_correlation(estimates, ((1.0,),), source_id="bad")
    with pytest.raises(ValueError, match="symmetric"):
        EvidenceCovariance.from_correlation(
            estimates,
            ((1.0, 0.2), (0.1, 1.0)),
            source_id="bad",
        )
    with pytest.raises(ValueError, match="diagonal"):
        EvidenceCovariance.from_correlation(
            estimates,
            ((0.9, 0.0), (0.0, 1.0)),
            source_id="bad",
        )
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        EvidenceCovariance.from_correlation(
            estimates,
            ((1.0, 1.2), (1.2, 1.0)),
            source_id="bad",
        )


def test_gls_weights_are_invariant_to_measurement_scale() -> None:
    estimates = (
        evidence("a", Modality.CLINICAL, 0.0, 1.0),
        evidence("b", Modality.METHYLATION, 1.0, 2.0),
    )
    covariance = EvidenceCovariance(
        evidence_ids=("a", "b"),
        covariance=((1.0, 0.0), (0.0, 4.0)),
        source_id="base",
    )
    scaled = tuple(
        item.model_copy(
            update={
                "estimate": item.estimate * 1e-10,
                "standard_error": item.standard_error * 1e-10,
            }
        )
        for item in estimates
    )
    scaled_covariance = EvidenceCovariance(
        evidence_ids=("a", "b"),
        covariance=((1e-20, 0.0), (0.0, 4e-20)),
        source_id="scaled",
    )

    base = GeneralizedLeastSquaresFusion().fuse(estimates, covariance)
    tiny = GeneralizedLeastSquaresFusion().fuse(scaled, scaled_covariance)

    assert tiny.evidence_weights == pytest.approx(base.evidence_weights)
    assert tiny.estimate == pytest.approx(base.estimate * 1e-10)


def test_cross_species_and_cross_subject_fusion_require_explicit_policy() -> None:
    canine = evidence("dog", Modality.CLINICAL, 1.0).model_copy(
        update={"subject_id": "dog-1", "species_taxon_id": 9615}
    )
    human = evidence("human", Modality.CLINICAL, 2.0).model_copy(
        update={"subject_id": "human-1", "species_taxon_id": 9606}
    )
    with pytest.raises(ValueError, match="multiple species"):
        GeneralizedLeastSquaresFusion().fuse((canine, human))

    same_species = human.model_copy(update={"species_taxon_id": 9615})
    with pytest.raises(ValueError, match="multiple subjects"):
        GeneralizedLeastSquaresFusion().fuse((canine, same_species))


def test_estimates_are_validated_and_unrelated_targets_are_fused_separately() -> None:
    engine = GeneralizedLeastSquaresFusion(EvidenceFusionConfig(minimum_evidence=2))
    with pytest.raises(ValueError, match="at least 2"):
        engine.fuse((evidence("a", Modality.CLINICAL, 1.0),))

    duplicate = evidence("a", Modality.METHYLATION, 2.0)
    with pytest.raises(ValueError, match="unique"):
        GeneralizedLeastSquaresFusion().fuse((evidence("a", Modality.CLINICAL, 1.0), duplicate))
    with pytest.raises(ValueError, match="complete estimand"):
        GeneralizedLeastSquaresFusion().fuse(
            (
                evidence("a", Modality.CLINICAL, 1.0),
                evidence("b", Modality.METHYLATION, 2.0, target=estimand("inflammation")),
            )
        )

    mixed = (
        evidence("a", Modality.CLINICAL, 1.0),
        evidence("b", Modality.METHYLATION, 2.0, target=estimand("inflammation")),
    )
    grouped = GeneralizedLeastSquaresFusion().fuse_by_estimand(mixed)
    assert len(grouped) == 2
    assert {result.estimate for result in grouped.values()} == {1.0, 2.0}


def test_covariance_ids_must_match_evidence_exactly() -> None:
    estimates = (evidence("a", Modality.CLINICAL, 1.0),)
    covariance = EvidenceCovariance(
        evidence_ids=("different",), covariance=((1.0,),), source_id="wrong"
    )
    with pytest.raises(ValueError, match="exactly match"):
        GeneralizedLeastSquaresFusion().fuse(estimates, covariance)


def test_hierarchical_fusion_combines_signatures_before_modalities() -> None:
    estimates = (
        evidence("rna-autophagy", Modality.TRANSCRIPTOMICS, -4.0),
        evidence("rna-mtor", Modality.TRANSCRIPTOMICS, -3.0),
        evidence("clinical", Modality.CLINICAL, -1.0),
    )
    covariance = EvidenceCovariance(
        evidence_ids=tuple(item.evidence_id for item in estimates),
        covariance=(
            (1.0, 0.7, 0.2),
            (0.7, 1.0, 0.2),
            (0.2, 0.2, 1.0),
        ),
        source_id="subject-bootstrap",
    )
    with pytest.raises(ValueError, match="cross-modality covariance"):
        HierarchicalEvidenceFusion(across_modality_model=FusionModel.FIXED_EFFECT).fuse(
            estimates, covariance
        )
    result = HierarchicalEvidenceFusion(
        EvidenceFusionConfig(
            cross_modality_covariance_policy=CrossModalityCovariancePolicy.WARN_IGNORE
        ),
        across_modality_model=FusionModel.FIXED_EFFECT,
    ).fuse(estimates, covariance)

    assert set(result.within_modality) == {Modality.TRANSCRIPTOMICS, Modality.CLINICAL}
    assert result.within_modality[Modality.TRANSCRIPTOMICS].estimate == pytest.approx(-3.5)
    assert result.across_modality.present_modalities == (
        Modality.CLINICAL,
        Modality.TRANSCRIPTOMICS,
    )
    assert result.warnings == ("cross_modality_covariance_not_propagated_by_hierarchy",)

    custom_confidence = HierarchicalEvidenceFusion(
        EvidenceFusionConfig(
            confidence_level=0.9,
            cross_modality_covariance_policy=CrossModalityCovariancePolicy.WARN_IGNORE,
        ),
        across_modality_model=FusionModel.FIXED_EFFECT,
    ).fuse(estimates, covariance)
    assert custom_confidence.across_modality.confidence_level == 0.9


def test_hierarchy_detects_cross_modality_correlation_at_small_scale() -> None:
    estimates = (
        evidence("genomic", Modality.TRANSCRIPTOMICS, 0.0, 1e-6),
        evidence("clinical", Modality.CLINICAL, 1e-6, 1e-6),
    )
    covariance = EvidenceCovariance(
        evidence_ids=("genomic", "clinical"),
        covariance=((1e-12, 9e-13), (9e-13, 1e-12)),
        source_id="small-unit-correlated-estimates",
    )

    with pytest.raises(ValueError, match="cross-modality covariance"):
        HierarchicalEvidenceFusion().fuse(estimates, covariance)

    warned = HierarchicalEvidenceFusion(
        EvidenceFusionConfig(
            cross_modality_covariance_policy=CrossModalityCovariancePolicy.WARN_IGNORE
        )
    ).fuse(estimates, covariance)
    assert warned.warnings == ("cross_modality_covariance_not_propagated_by_hierarchy",)


def test_evidence_models_reject_nonfinite_values_and_duplicate_flags() -> None:
    with pytest.raises(ValidationError, match="finite"):
        evidence("bad", Modality.CLINICAL, float("inf"))
    with pytest.raises(ValidationError, match="unique"):
        evidence("bad", Modality.CLINICAL, 1.0).model_copy(
            update={"quality_flags": ("flag", "flag")}
        ).model_validate(
            evidence("bad", Modality.CLINICAL, 1.0)
            .model_copy(update={"quality_flags": ("flag", "flag")})
            .model_dump()
        )


def test_fusion_requires_hash_bound_eligible_calibration_by_default() -> None:
    validated = evidence("validated", Modality.CLINICAL, 1.0)
    missing = validated.model_copy(update={"calibration_reference": None})
    with pytest.raises(ValueError, match="CalibrationReference"):
        GeneralizedLeastSquaresFusion().fuse((missing,))

    reference = validated.calibration_reference
    assert reference is not None
    internal = validated.model_copy(
        update={
            "calibration_reference": reference.model_copy(
                update={"status": CalibrationValidationStatus.INTERNAL_CROSS_VALIDATED}
            )
        }
    )
    with pytest.raises(ValueError, match="not fusion eligible"):
        GeneralizedLeastSquaresFusion().fuse((internal,))
    exploratory = GeneralizedLeastSquaresFusion(
        EvidenceFusionConfig(require_fusion_eligible_calibration=False)
    ).fuse((internal,))
    assert exploratory.estimate == 1.0


def test_prespecified_evidence_missingness_is_explicit() -> None:
    available = evidence("available", Modality.CLINICAL, 1.0)
    with pytest.raises(ValueError, match="prespecified evidence is missing"):
        GeneralizedLeastSquaresFusion(
            EvidenceFusionConfig(expected_evidence_ids=("available", "missing"))
        ).fuse((available,))

    warned = GeneralizedLeastSquaresFusion(
        EvidenceFusionConfig(
            expected_evidence_ids=("available", "missing"),
            missing_evidence_policy=MissingEvidencePolicy.WARN,
        )
    ).fuse((available,))
    assert "prespecified_evidence_missing" in warned.warnings


def test_evidence_and_hierarchical_result_mappings_are_immutable() -> None:
    estimates = (
        evidence("rna", Modality.TRANSCRIPTOMICS, -1.0),
        evidence("clinical", Modality.CLINICAL, -0.5),
    )
    result = GeneralizedLeastSquaresFusion().fuse(estimates)
    with pytest.raises(TypeError):
        result.evidence_weights["rna"] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.modality_weights[Modality.CLINICAL] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.standardized_residuals["rna"] = 99.0  # type: ignore[index]
    assert result.model_validate(result.model_dump(mode="python")) == result

    hierarchy = HierarchicalEvidenceFusion().fuse(estimates)
    with pytest.raises(TypeError):
        hierarchy.within_modality[Modality.CLINICAL] = result  # type: ignore[index]
    assert hierarchy.model_validate(hierarchy.model_dump(mode="python")) == hierarchy


def test_disagreeing_evidence_widens_gls_interval() -> None:
    estimates = (
        evidence("clinical", Modality.CLINICAL, -4.0, 0.5),
        evidence("methylation", Modality.METHYLATION, 1.0, 0.5),
        evidence("proteomics", Modality.PROTEOMICS, -1.0, 0.5),
    )
    adjusted = GeneralizedLeastSquaresFusion().fuse(estimates)
    fixed = GeneralizedLeastSquaresFusion(
        EvidenceFusionConfig(dispersion_policy=EvidenceDispersionPolicy.FIXED)
    ).fuse(estimates)

    phi = adjusted.disagreement_score / 2
    assert phi > 1
    assert adjusted.dispersion_factor == pytest.approx(phi)
    assert adjusted.standard_error == pytest.approx(fixed.standard_error * phi**0.5)
    assert adjusted.interval_degrees_of_freedom == 2
    assert "evidence_overdispersed" in adjusted.warnings
    half_width = (adjusted.confidence_interval[1] - adjusted.confidence_interval[0]) / 2
    assert half_width == pytest.approx(4.302652729911275 * fixed.standard_error * phi**0.5)
    assert fixed.interval_method == "wald"
    assert fixed.dispersion_factor == 1.0


def test_agreeing_evidence_never_narrows_below_fixed_effect_interval() -> None:
    estimates = (
        evidence("clinical", Modality.CLINICAL, -1.0, 0.5),
        evidence("methylation", Modality.METHYLATION, -1.0, 0.5),
    )
    adjusted = GeneralizedLeastSquaresFusion().fuse(estimates)
    fixed = GeneralizedLeastSquaresFusion(
        EvidenceFusionConfig(dispersion_policy=EvidenceDispersionPolicy.FIXED)
    ).fuse(estimates)
    assert adjusted.disagreement_score == pytest.approx(0.0)
    assert adjusted.confidence_interval == pytest.approx(fixed.confidence_interval)
    assert adjusted.standard_error == pytest.approx(fixed.standard_error)
