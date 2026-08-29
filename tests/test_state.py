from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from rejuvenationkit.schemas import Modality, Observation, Study, Subject
from rejuvenationkit.state import (
    LinearGaussianStateConfig,
    LinearGaussianStateEstimator,
    StateChannel,
    StateEstimate,
    StateEstimationReport,
    StateTrajectory,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def _scalar_config(
    *,
    process_variance: float = 0.0,
    measurement_variance: float = 1.0,
    drift: float = 0.0,
    smooth: bool = False,
) -> LinearGaussianStateConfig:
    return LinearGaussianStateConfig(
        state_names=("latent_health",),
        channels=(
            StateChannel(
                name="clinical-score",
                modality=Modality.CLINICAL,
                feature="score",
                unit="normalized",
                loadings=(1.0,),
                measurement_variance=measurement_variance,
            ),
        ),
        continuous_dynamics=((0.0,),),
        continuous_process_covariance=((process_variance,),),
        continuous_drift=(drift,),
        initial_mean=(0.0,),
        initial_covariance=((1.0,),),
        time_unit_days=1.0,
        smooth=smooth,
    )


def _study(
    values: dict[str, list[tuple[datetime, float, float | None]]],
    *,
    extra_observations: tuple[Observation, ...] = (),
) -> Study:
    subjects = tuple(Subject(subject_id=subject_id, cohort="cohort") for subject_id in values)
    observations = [
        Observation(
            subject_id=subject_id,
            timestamp=timestamp,
            modality=Modality.CLINICAL,
            feature="score",
            value=value,
            unit="normalized",
            standard_error=standard_error,
        )
        for subject_id, rows in values.items()
        for timestamp, value, standard_error in rows
    ]
    return Study(
        study_id="state-study",
        subjects=subjects,
        observations=(*observations, *extra_observations),
    )


def test_scalar_filter_matches_hand_calculation_and_legacy_forecast_flag() -> None:
    study = _study({"s1": [(START, 1.0, None), (START + timedelta(days=1), 1.0, None)]})
    estimator = LinearGaussianStateEstimator(_scalar_config()).fit(study)

    trajectory = estimator.filter(study)[0]

    assert trajectory.estimates[0].mean == pytest.approx((0.5,))
    assert trajectory.estimates[0].covariance[0] == pytest.approx((0.5,))
    assert trajectory.estimates[1].mean == pytest.approx((2 / 3,))
    assert trajectory.estimates[1].covariance[0] == pytest.approx((1 / 3,))
    assert trajectory.coverage is not None
    assert trajectory.coverage.observed_fraction == 1.0
    legacy = StateEstimate(
        subject_id="s1",
        timestamp=START,
        state_names=("state",),
        mean=(0.0,),
        covariance=((1.0,),),
        is_forecast=True,
    )
    assert legacy.estimate_kind == "forecast"


def test_partial_channel_updates_report_coverage_and_standard_error_variance() -> None:
    channels = (
        StateChannel(
            name="immune",
            modality=Modality.CLINICAL,
            feature="immune",
            unit="z",
            loadings=(1.0, 0.0),
            measurement_variance=1.0,
        ),
        StateChannel(
            name="metabolic",
            modality=Modality.METABOLOMICS,
            feature="metabolic",
            unit="z",
            loadings=(0.0, 1.0),
            measurement_variance=1.0,
        ),
    )
    config = LinearGaussianStateConfig(
        state_names=("immune_state", "metabolic_state"),
        channels=channels,
        continuous_dynamics=((0.0, 0.0), (0.0, 0.0)),
        continuous_process_covariance=((0.0, 0.0), (0.0, 0.0)),
        continuous_drift=(0.0, 0.0),
        initial_mean=(0.0, 0.0),
        initial_covariance=((1.0, 0.0), (0.0, 1.0)),
        time_unit_days=1,
        smooth=False,
    )
    study = Study(
        study_id="partial",
        subjects=(Subject(subject_id="s1", cohort="treated"),),
        observations=(
            Observation(
                subject_id="s1",
                timestamp=START,
                modality=Modality.CLINICAL,
                feature="immune",
                value=2.0,
                unit="z",
            ),
            Observation(
                subject_id="s1",
                timestamp=START,
                modality=Modality.METABOLOMICS,
                feature="metabolic",
                value=10.0,
                unit="z",
            ),
            Observation(
                subject_id="s1",
                timestamp=START + timedelta(days=1),
                modality=Modality.CLINICAL,
                feature="immune",
                value=2.0,
                unit="z",
                standard_error=2.0,
            ),
        ),
    )
    estimator = LinearGaussianStateEstimator(config).fit(study)

    trajectory = estimator.filter(study)[0]
    diagnostic = estimator.one_step_diagnostics(study)[0]

    assert trajectory.estimates[-1].mean[1] == pytest.approx(5.0)
    assert trajectory.coverage is not None
    assert trajectory.coverage.observed_channel_values == 3
    assert trajectory.coverage.missing_channel_values == 1
    assert trajectory.coverage.channels[1].missing_timepoints == 1
    assert diagnostic.channel_names == ("immune",)
    assert diagnostic.reported_standard_error_variances == (4.0,)
    assert diagnostic.innovation_covariance[0][0] == pytest.approx(5.5)


def test_irregular_time_process_variance_and_forecast_growth_are_exact() -> None:
    study = _study({"s1": [(START, 0.0, None)]})
    estimator = LinearGaussianStateEstimator(
        _scalar_config(process_variance=2.0, measurement_variance=1.0)
    ).fit(study)
    terminal = estimator.filter(study)[0]

    forecast = estimator.forecast(
        terminal,
        timestamps=(START + timedelta(days=1), START + timedelta(days=3)),
    )

    terminal_variance = terminal.estimates[-1].covariance[0][0]
    assert forecast.estimates[0].covariance[0][0] - terminal_variance == pytest.approx(2.0)
    assert (
        forecast.estimates[1].covariance[0][0] - forecast.estimates[0].covariance[0][0]
    ) == pytest.approx(4.0)
    assert all(item.is_forecast for item in forecast.estimates)


def test_continuous_drift_uses_elapsed_time() -> None:
    study = _study({"s1": [(START, 0.0, None)]})
    estimator = LinearGaussianStateEstimator(_scalar_config(process_variance=0.0, drift=3.0)).fit(
        study
    )
    terminal = estimator.filter(study)[0]
    forecast = estimator.forecast(
        terminal,
        timestamps=(START + timedelta(days=2),),
    )

    assert forecast.estimates[0].mean[0] - terminal.estimates[0].mean[0] == pytest.approx(6.0)


def test_smoothing_reduces_or_preserves_covariance() -> None:
    study = _study(
        {
            "s1": [
                (START, 0.2, None),
                (START + timedelta(days=1), 1.1, None),
                (START + timedelta(days=2), 1.9, None),
            ]
        }
    )
    estimator = LinearGaussianStateEstimator(_scalar_config(process_variance=0.2, smooth=True)).fit(
        study
    )

    filtered = estimator.filter(study)[0]
    smoothed = estimator.smooth(study)[0]

    assert all(item.estimate_kind == "smoothed" for item in smoothed.estimates)
    assert smoothed.estimates[-1].covariance == filtered.estimates[-1].covariance
    assert all(
        smooth.covariance[0][0] <= forward.covariance[0][0] + 1e-12
        for smooth, forward in zip(smoothed.estimates, filtered.estimates, strict=True)
    )


def test_exact_unit_matching_duplicates_and_unconfigured_exclusions_fail_closed() -> None:
    wrong_unit = Study(
        study_id="wrong-unit",
        subjects=(Subject(subject_id="s1", cohort="c"),),
        observations=(
            Observation(
                subject_id="s1",
                timestamp=START,
                modality=Modality.CLINICAL,
                feature="score",
                value=1.0,
                unit="points",
            ),
        ),
    )
    with pytest.raises(ValueError, match="unit does not match"):
        LinearGaussianStateEstimator(_scalar_config()).fit(wrong_unit)

    duplicate = _study(
        {"s1": [(START, 0.0, None), (START, 1.0, None)]},
    )
    estimator = LinearGaussianStateEstimator(_scalar_config()).fit(duplicate)
    with pytest.raises(ValueError, match="duplicate configured channel"):
        estimator.estimate(duplicate)

    unrelated = Observation(
        subject_id="s1",
        timestamp=START,
        modality=Modality.WEARABLE,
        feature="activity",
        value=10.0,
        unit="steps",
    )
    with_extra = _study({"s1": [(START, 0.0, None)]}, extra_observations=(unrelated,))
    trajectory = (
        LinearGaussianStateEstimator(_scalar_config()).fit(with_extra).estimate(with_extra)[0]
    )
    assert len(trajectory.excluded_observations) == 1
    assert trajectory.excluded_observations[0].reason == "channel_not_configured"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"state_names": ("x", "x")}, "state names must be unique"),
        ({"continuous_dynamics": ((0.0, 1.0),)}, "shape"),
        ({"continuous_process_covariance": ((-1.0,),)}, "positive semidefinite"),
        ({"initial_covariance": ((float("nan"),),)}, "finite"),
        ({"continuous_drift": (float("inf"),)}, "finite"),
    ],
)
def test_configuration_rejects_malformed_dimensions_and_covariances(
    overrides: dict[str, object],
    match: str,
) -> None:
    values = _scalar_config().model_dump()
    values.update(overrides)
    with pytest.raises(ValidationError, match=match):
        LinearGaussianStateConfig.model_validate(values)


def test_configuration_rejects_duplicate_channels_and_wrong_loading_dimension() -> None:
    channel = _scalar_config().channels[0]
    values = _scalar_config().model_dump()
    values["channels"] = (channel, channel)
    with pytest.raises(ValidationError, match=r"identities|names"):
        LinearGaussianStateConfig.model_validate(values)

    values = _scalar_config().model_dump()
    values["channels"] = (
        StateChannel(
            name="bad",
            modality=Modality.CLINICAL,
            feature="score",
            unit="normalized",
            loadings=(1.0, 0.0),
            measurement_variance=1.0,
        ),
    )
    with pytest.raises(ValidationError, match="loadings must have length"):
        LinearGaussianStateConfig.model_validate(values)


def test_estimator_requires_explicit_config_and_fit() -> None:
    study = _study({"s1": [(START, 0.0, None)]})
    with pytest.raises(NotImplementedError, match="explicit"):
        LinearGaussianStateEstimator().fit(study)
    with pytest.raises(NotImplementedError, match="explicit"):
        LinearGaussianStateEstimator().estimate(study)
    with pytest.raises(RuntimeError, match="fit"):
        LinearGaussianStateEstimator(_scalar_config()).estimate(study)


def test_forecast_rejects_malformed_times_and_wrong_state_definition() -> None:
    study = _study({"s1": [(START, 0.0, None)]})
    estimator = LinearGaussianStateEstimator(_scalar_config()).fit(study)
    trajectory = estimator.filter(study)[0]
    with pytest.raises(ValueError, match="timezone-aware"):
        estimator.forecast(
            trajectory,
            timestamps=(datetime(2026, 1, 2),),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        estimator.forecast(
            trajectory,
            timestamps=(START + timedelta(days=2), START + timedelta(days=2)),
        )
    with pytest.raises(ValueError, match="follow"):
        estimator.forecast(trajectory, timestamps=(START,))
    wrong = StateTrajectory(
        subject_id="s1",
        estimates=(
            StateEstimate(
                subject_id="s1",
                timestamp=START,
                state_names=("other",),
                mean=(0.0,),
                covariance=((1.0,),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="state names"):
        estimator.forecast(wrong, timestamps=(START + timedelta(days=1),))


def test_deterministic_serialization_is_independent_of_observation_order() -> None:
    unrelated = Observation(
        subject_id="s1",
        timestamp=START + timedelta(hours=2),
        modality=Modality.WEARABLE,
        feature="activity",
        value=50,
        unit="steps",
    )
    first = _study(
        {"s1": [(START + timedelta(days=1), 1.0, None), (START, 0.0, None)]},
        extra_observations=(unrelated,),
    )
    second = first.model_copy(update={"observations": tuple(reversed(first.observations))})

    first_report = LinearGaussianStateEstimator(_scalar_config()).fit(first).estimate_report(first)
    second_report = (
        LinearGaussianStateEstimator(_scalar_config()).fit(second).estimate_report(second)
    )

    assert first_report.model_dump(mode="json") == second_report.model_dump(mode="json")
    assert json.loads(first_report.model_dump_json())["study_id"] == "state-study"
    assert json.loads(_scalar_config().model_dump_json())["channels"][0]["unit"] == "normalized"


def test_state_reports_bind_exact_study_model_and_forecast_parent() -> None:
    study = _study({"s1": [(START, 0.0, 0.1), (START + timedelta(days=1), 1.0, 0.1)]})
    estimator = LinearGaussianStateEstimator(_scalar_config()).fit(study)

    report = estimator.estimate_report(study)
    trajectory = report.trajectories[0]
    forecast = estimator.forecast(
        trajectory,
        timestamps=(START + timedelta(days=2),),
    )

    assert report.study_artifact_hash == trajectory.source_study_artifact_hash
    assert report.model_config_artifact_hash == trajectory.model_config_artifact_hash
    assert forecast.source_trajectory_artifact_hash == trajectory.artifact_hash
    assert len(report.artifact_hash) == 64
    changed = study.model_copy(
        update={
            "metadata": {"changed": True},
        }
    )
    with pytest.raises(ValueError, match="study artifact differs"):
        estimator.estimate_report(changed)

    duplicate_payload = report.model_dump(mode="python")
    duplicate_payload["trajectories"] = (
        duplicate_payload["trajectories"][0],
        duplicate_payload["trajectories"][0],
    )
    with pytest.raises(ValidationError, match="trajectory subjects must be unique"):
        StateEstimationReport.model_validate(duplicate_payload)

    overlap_payload = report.model_dump(mode="python")
    overlap_payload["excluded_subjects"] = (
        {"subject_id": trajectory.subject_id, "reason": "no_configured_observations"},
    )
    with pytest.raises(ValidationError, match="both trajectories and excluded"):
        StateEstimationReport.model_validate(overlap_payload)


def test_subject_without_configured_measurements_is_explicitly_excluded() -> None:
    study = Study(
        study_id="excluded",
        subjects=(Subject(subject_id="s1", cohort="c"),),
        observations=(
            Observation(
                subject_id="s1",
                timestamp=START,
                modality=Modality.WEARABLE,
                feature="activity",
                value=1,
                unit="steps",
            ),
        ),
    )
    report = LinearGaussianStateEstimator(_scalar_config()).fit(study).estimate_report(study)
    assert report.trajectories == ()
    assert report.excluded_subjects[0].subject_id == "s1"


def _calibration_study() -> tuple[Study, tuple[str, ...], tuple[str, ...], dict[str, np.ndarray]]:
    random = np.random.default_rng(714)
    reference = tuple(f"reference-{index:02d}" for index in range(16))
    evaluation = tuple(f"evaluation-{index:02d}" for index in range(12))
    subjects = tuple(
        Subject(
            subject_id=subject_id, cohort="reference" if subject_id in reference else "held-out"
        )
        for subject_id in (*reference, *evaluation)
    )
    observations: list[Observation] = []
    truths: dict[str, np.ndarray] = {}
    for subject_id in (*reference, *evaluation):
        latent = 0.0
        subject_truth = [latent]
        for _ in range(11):
            latent += float(random.normal(0, np.sqrt(0.2)))
            subject_truth.append(latent)
        truths[subject_id] = np.asarray(subject_truth)
        for day, value in enumerate(subject_truth):
            observations.append(
                Observation(
                    subject_id=subject_id,
                    timestamp=START + timedelta(days=day),
                    modality=Modality.CLINICAL,
                    feature="score",
                    value=float(value + random.normal(0, np.sqrt(0.5))),
                    unit="normalized",
                )
            )
    return (
        Study(study_id="calibration", subjects=subjects, observations=tuple(observations)),
        reference,
        evaluation,
        truths,
    )


def test_synthetic_state_rmse_and_held_out_forecast_coverage() -> None:
    study, reference, evaluation, truths = _calibration_study()
    estimator = LinearGaussianStateEstimator(
        _scalar_config(process_variance=0.2, measurement_variance=0.5, smooth=True)
    ).fit(study, reference_subject_ids=reference)
    trajectories = estimator.smooth(study, subject_ids=evaluation)
    estimated = np.asarray(
        [item.mean[0] for trajectory in trajectories for item in trajectory.estimates]
    )
    true = np.concatenate([truths[trajectory.subject_id] for trajectory in trajectories])
    observed = np.asarray(
        [
            row.value
            for subject_id in sorted(evaluation)
            for row in sorted(
                (item for item in study.observations if item.subject_id == subject_id),
                key=lambda item: item.timestamp,
            )
        ]
    )
    calibration = estimator.calibrate_forecasts(
        study,
        reference_subject_ids=reference,
        evaluation_subject_ids=evaluation,
        interval_level=0.90,
    )

    assert np.sqrt(np.mean((estimated - true) ** 2)) < np.sqrt(np.mean((observed - true) ** 2))
    assert 0.75 <= calibration.empirical_coverage <= 1.0
    assert calibration.reference_standardized_innovations == 16 * 11
    assert calibration.evaluation_standardized_innovations == 12 * 11
    assert len(calibration.study_artifact_hash) == 64
    assert len(calibration.model_config_artifact_hash) == 64
    assert len(calibration.reference_diagnostics_artifact_hash) == 64
    assert len(calibration.evaluation_diagnostics_artifact_hash) == 64
    assert len(calibration.artifact_hash) == 64
    calibration_payload = calibration.model_dump(mode="python")
    calibration_payload["evaluation_subject_ids"] = (calibration.reference_subject_ids[0],)
    with pytest.raises(ValidationError, match="must be disjoint"):
        type(calibration).model_validate(calibration_payload)


def test_change_points_use_disjoint_reference_and_detect_shift() -> None:
    reference = tuple(f"r{index}" for index in range(8))
    evaluation = ("shift",)
    values = {
        subject_id: [(START + timedelta(days=day), 0.0, None) for day in range(4)]
        for subject_id in reference
    }
    values["shift"] = [
        (START, 0.0, None),
        (START + timedelta(days=1), 0.0, None),
        (START + timedelta(days=2), 20.0, None),
        (START + timedelta(days=3), 20.0, None),
    ]
    study = _study(values)
    estimator = LinearGaussianStateEstimator(_scalar_config()).fit(study)

    report = estimator.detect_innovation_change_points(
        study,
        reference_subject_ids=reference,
        evaluation_subject_ids=evaluation,
    )

    assert report.reference_innovations == 24
    assert report.false_alarm_scope == "per_innovation_no_trajectory_multiplicity_control"
    assert any(item.detected for item in report.results)
    assert len(report.study_artifact_hash) == 64
    assert len(report.model_config_artifact_hash) == 64
    assert len(report.reference_diagnostics_artifact_hash) == 64
    assert len(report.evaluation_diagnostics_artifact_hash) == 64
    assert len(report.artifact_hash) == 64
    inconsistent_payload = report.model_dump(mode="python")
    inconsistent_payload["results"][0]["detected"] = not report.results[0].detected
    with pytest.raises(ValidationError, match="detections must agree"):
        type(report).model_validate(inconsistent_payload)
    overlap_payload = report.model_dump(mode="python")
    overlap_payload["evaluation_subject_ids"] = (reference[0],)
    with pytest.raises(ValidationError, match="must be disjoint"):
        type(report).model_validate(overlap_payload)
    with pytest.raises(ValueError, match="disjoint"):
        estimator.detect_innovation_change_points(
            study,
            reference_subject_ids=reference,
            evaluation_subject_ids=(reference[0],),
        )
    with pytest.raises(ValueError, match="interval_level"):
        estimator.calibrate_forecasts(
            study,
            reference_subject_ids=reference,
            evaluation_subject_ids=evaluation,
            interval_level=0.5,
        )
