from datetime import UTC, datetime
from math import isclose

import pytest
from pydantic import ValidationError

from rejuvenationkit.fusion import (
    FusionConfig,
    FusionModel,
    MissingModalityPolicy,
    ModalityCalibration,
    ModalityEstimate,
    PrecisionWeightedFusion,
)
from rejuvenationkit.schemas import Modality, Observation, Study, Subject


def estimate(modality: Modality, value: float, error: float = 1.0) -> ModalityEstimate:
    return ModalityEstimate(
        modality=modality,
        estimate=value,
        standard_error=error,
        target="biological_age_delta_years",
    )


def test_fixed_effect_fusion_uses_inverse_variance_weights() -> None:
    fusion = PrecisionWeightedFusion(FusionConfig(model=FusionModel.FIXED_EFFECT))
    result = fusion.fuse(
        (
            estimate(Modality.METHYLATION, -4.0, 1.0),
            estimate(Modality.TRANSCRIPTOMICS, -2.0, 2.0),
        )
    )

    assert isclose(result.modality_weights[Modality.METHYLATION], 0.8)
    assert isclose(result.modality_weights[Modality.TRANSCRIPTOMICS], 0.2)
    assert isclose(result.estimate, -3.6)
    assert isclose(result.standard_error, (1 / 1.25) ** 0.5)
    assert result.between_modality_variance == 0
    assert result.confidence_interval[0] < result.estimate < result.confidence_interval[1]


def test_random_effects_widens_uncertainty_when_modalities_disagree() -> None:
    estimates = (
        estimate(Modality.METHYLATION, -8.0, 0.5),
        estimate(Modality.TRANSCRIPTOMICS, 1.0, 0.5),
        estimate(Modality.CLINICAL, -2.0, 0.5),
    )
    fixed = PrecisionWeightedFusion(FusionConfig(model=FusionModel.FIXED_EFFECT)).fuse(estimates)
    random = PrecisionWeightedFusion(FusionConfig(model=FusionModel.RANDOM_EFFECTS)).fuse(estimates)

    assert random.between_modality_variance > 0
    assert random.standard_error > fixed.standard_error
    assert random.heterogeneity_i2 > 0.9
    assert random.disagreement_score == fixed.disagreement_score


def test_calibration_corrects_bias_and_scales_uncertainty() -> None:
    fusion = PrecisionWeightedFusion(
        FusionConfig(
            calibrations=(
                ModalityCalibration(
                    modality=Modality.METHYLATION,
                    bias=2.0,
                    standard_error_scale=2.0,
                    calibration_id="clock-validation-v1",
                ),
            ),
        )
    )
    result = fusion.fuse((estimate(Modality.METHYLATION, -3.0, 0.5),))

    assert result.estimate == -5.0
    assert result.standard_error == 1.0
    assert result.calibration_ids[Modality.METHYLATION] == "clock-validation-v1"


def test_missing_expected_modality_is_reported_without_imputation() -> None:
    fusion = PrecisionWeightedFusion(
        FusionConfig(
            expected_modalities=(Modality.METHYLATION, Modality.CLINICAL),
            missing_modality_policy=MissingModalityPolicy.ALLOW,
        )
    )
    result = fusion.fuse((estimate(Modality.CLINICAL, -1.0),))

    assert result.present_modalities == (Modality.CLINICAL,)
    assert result.missing_modalities == (Modality.METHYLATION,)
    assert result.estimate == -1.0


def test_strict_missing_modality_policy_rejects_incomplete_input() -> None:
    fusion = PrecisionWeightedFusion(
        FusionConfig(
            expected_modalities=(Modality.METHYLATION, Modality.CLINICAL),
            missing_modality_policy=MissingModalityPolicy.ERROR,
        )
    )
    with pytest.raises(ValueError, match="methylation"):
        fusion.fuse((estimate(Modality.CLINICAL, -1.0),))


def test_leave_one_out_identifies_influential_modality() -> None:
    fusion = PrecisionWeightedFusion(FusionConfig(model=FusionModel.FIXED_EFFECT))
    result = fusion.fuse(
        (
            estimate(Modality.METHYLATION, -8.0),
            estimate(Modality.TRANSCRIPTOMICS, -2.0),
            estimate(Modality.CLINICAL, -2.0),
        )
    )
    diagnostics = {item.omitted_modality: item for item in result.leave_one_modality_out}

    assert diagnostics[Modality.METHYLATION].estimate == -2.0
    assert (
        abs(diagnostics[Modality.METHYLATION].estimate_shift) == result.maximum_leave_one_out_shift
    )


def test_fit_records_study_provenance_and_modality_counts() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    study = Study(
        study_id="calibration-cohort",
        subjects=(Subject(subject_id="dog-1", cohort="control"),),
        observations=(
            Observation(
                subject_id="dog-1",
                timestamp=timestamp,
                modality=Modality.CLINICAL,
                feature="frailty",
                value=0.2,
                unit="score",
            ),
            Observation(
                subject_id="dog-1",
                timestamp=timestamp,
                modality=Modality.METHYLATION,
                feature="clock_age",
                value=8.0,
                unit="years",
            ),
        ),
    )
    fusion = PrecisionWeightedFusion().fit(study)

    assert fusion.fitted_study_id == "calibration-cohort"
    assert fusion.fitted_modality_counts == {
        Modality.CLINICAL: 1,
        Modality.METHYLATION: 1,
    }
    result = fusion.fuse((estimate(Modality.CLINICAL, -1.0),))
    assert result.fitted_study_id == "calibration-cohort"


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ((), "at least 1"),
        (
            (
                estimate(Modality.CLINICAL, -1.0),
                estimate(Modality.CLINICAL, -2.0),
            ),
            "unique modalities",
        ),
        (
            (
                estimate(Modality.CLINICAL, -1.0),
                ModalityEstimate(
                    modality=Modality.METHYLATION,
                    estimate=-2.0,
                    standard_error=1.0,
                    target="different_target",
                ),
            ),
            "share one target",
        ),
    ],
)
def test_fusion_rejects_invalid_input(
    inputs: tuple[ModalityEstimate, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PrecisionWeightedFusion().fuse(inputs)


def test_minimum_modality_requirement_is_enforced() -> None:
    fusion = PrecisionWeightedFusion(FusionConfig(minimum_modalities=2))
    with pytest.raises(ValueError, match="at least 2"):
        fusion.fuse((estimate(Modality.CLINICAL, -1.0),))


def test_nonfinite_estimates_and_duplicate_configuration_are_rejected() -> None:
    with pytest.raises(ValidationError, match="finite"):
        estimate(Modality.CLINICAL, float("nan"))
    with pytest.raises(ValidationError, match="expected_modalities must be unique"):
        FusionConfig(expected_modalities=(Modality.CLINICAL, Modality.CLINICAL))
    calibration = ModalityCalibration(
        modality=Modality.CLINICAL,
        calibration_id="clinical-v1",
    )
    with pytest.raises(ValidationError, match="calibrations must have unique modalities"):
        FusionConfig(calibrations=(calibration, calibration))
    with pytest.raises(ValidationError, match="calibration parameters must be finite"):
        ModalityCalibration(
            modality=Modality.CLINICAL,
            bias=float("inf"),
            calibration_id="invalid",
        )


def test_fusion_result_mappings_are_immutable_and_serializable() -> None:
    result = PrecisionWeightedFusion().fuse((estimate(Modality.CLINICAL, -1.0),))

    with pytest.raises(TypeError):
        result.modality_weights[Modality.CLINICAL] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.calibration_ids[Modality.CLINICAL] = "forged"  # type: ignore[index]
    assert result.model_validate(result.model_dump(mode="python")) == result
