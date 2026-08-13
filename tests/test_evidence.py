from math import isclose

import pytest
from pydantic import ValidationError

from rejuvenationkit import (
    EffectDirection,
    Estimand,
    EvidenceCovariance,
    EvidenceEstimate,
    EvidenceFusionConfig,
    EvidenceWeightConstraint,
    FusionModel,
    GeneralizedLeastSquaresFusion,
    HierarchicalEvidenceFusion,
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
    return EvidenceEstimate(
        evidence_id=evidence_id,
        modality=modality,
        estimand=target or estimand(),
        estimate=value,
        standard_error=error,
        calibration_id=f"calibration:{evidence_id}",
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
    independent = GeneralizedLeastSquaresFusion().fuse(estimates)
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
    result = HierarchicalEvidenceFusion(across_modality_model=FusionModel.FIXED_EFFECT).fuse(
        estimates, covariance
    )

    assert set(result.within_modality) == {Modality.TRANSCRIPTOMICS, Modality.CLINICAL}
    assert result.within_modality[Modality.TRANSCRIPTOMICS].estimate == pytest.approx(-3.5)
    assert result.across_modality.present_modalities == (
        Modality.CLINICAL,
        Modality.TRANSCRIPTOMICS,
    )
    assert result.warnings == ("cross_modality_covariance_not_propagated_by_hierarchy",)

    custom_confidence = HierarchicalEvidenceFusion(
        EvidenceFusionConfig(confidence_level=0.9),
        across_modality_model=FusionModel.FIXED_EFFECT,
    ).fuse(estimates, covariance)
    assert custom_confidence.across_modality.confidence_level == 0.9


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
