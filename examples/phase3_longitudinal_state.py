"""Demonstrate Phase 3 state estimation on a synthetic longitudinal study.

The latent definition and every value are synthetic. This example exercises
irregular timing, partial channels, held-out calibration, innovation change
points, smoothing, and forecasting; it does not estimate a treatment effect.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from rejuvenationkit.schemas import Modality, Observation, Study, Subject
from rejuvenationkit.state import (
    LinearGaussianStateConfig,
    LinearGaussianStateEstimator,
    StateChannel,
)

START = datetime(2026, 1, 5, tzinfo=UTC)
REFERENCE_SUBJECTS = tuple(f"reference-{index:02d}" for index in range(16))
EVALUATION_SUBJECTS = tuple(f"evaluation-{index:02d}" for index in range(6))
SHIFTED_SUBJECT = "evaluation-05"


def state_config() -> LinearGaussianStateConfig:
    """Declare a two-state model; these loadings are illustrative, not validated biology."""
    return LinearGaussianStateConfig(
        state_names=("inflammatory_burden", "functional_reserve"),
        channels=(
            StateChannel(
                name="inflammation-score",
                modality=Modality.TRANSCRIPTOMICS,
                feature="inflammation_score",
                unit="z_score",
                loadings=(1.0, 0.0),
                measurement_variance=0.20,
            ),
            StateChannel(
                name="frailty-score",
                modality=Modality.CLINICAL,
                feature="frailty_score",
                unit="z_score",
                loadings=(0.60, -0.80),
                measurement_variance=0.25,
            ),
            StateChannel(
                name="activity-score",
                modality=Modality.WEARABLE,
                feature="activity_score",
                unit="z_score",
                loadings=(0.0, 1.0),
                measurement_variance=0.15,
            ),
        ),
        continuous_dynamics=((0.0, 0.0), (0.0, 0.0)),
        continuous_process_covariance=((0.03, 0.0), (0.0, 0.025)),
        continuous_drift=(0.05, -0.04),
        initial_mean=(0.0, 0.0),
        initial_covariance=((0.50, 0.0), (0.0, 0.50)),
        time_unit_days=30.0,
        smooth=True,
    )


def build_synthetic_study() -> Study:
    """Create irregular multichannel trajectories with one injected held-out shift."""
    random = np.random.default_rng(8317)
    subject_ids = (*REFERENCE_SUBJECTS, *EVALUATION_SUBJECTS)
    subjects = tuple(
        Subject(
            subject_id=subject_id,
            cohort="reference" if subject_id in REFERENCE_SUBJECTS else "held-out",
            interventions=() if subject_id in REFERENCE_SUBJECTS else ("synthetic-intervention",),
        )
        for subject_id in subject_ids
    )
    observations: list[Observation] = []
    nominal_days = (0, 27, 63, 104, 151)
    for subject_index, subject_id in enumerate(subject_ids):
        baseline = random.normal(0.0, 0.16, size=2)
        for visit_index, nominal_day in enumerate(nominal_days):
            jitter = 0 if visit_index == 0 else int(random.integers(-3, 4))
            elapsed_days = nominal_day + jitter
            elapsed_units = elapsed_days / 30.0
            latent = baseline + np.asarray((0.05, -0.04)) * elapsed_units
            latent += random.normal(0.0, (0.06, 0.055), size=2) * np.sqrt(max(elapsed_units, 0.2))
            if subject_id == SHIFTED_SUBJECT and visit_index >= 3:
                latent += np.asarray((-1.45, 1.05))
            timestamp = START + timedelta(days=elapsed_days)
            channel_values = (
                (
                    Modality.TRANSCRIPTOMICS,
                    "inflammation_score",
                    float(latent[0] + random.normal(0.0, np.sqrt(0.20))),
                    0.10,
                ),
                (
                    Modality.CLINICAL,
                    "frailty_score",
                    float(0.60 * latent[0] - 0.80 * latent[1] + random.normal(0.0, np.sqrt(0.25))),
                    None,
                ),
                (
                    Modality.WEARABLE,
                    "activity_score",
                    float(latent[1] + random.normal(0.0, np.sqrt(0.15))),
                    0.08,
                ),
            )
            for channel_index, (modality, feature, value, standard_error) in enumerate(
                channel_values
            ):
                # Deliberately omit some wearable visits. Other channels still update the state.
                if channel_index == 2 and (subject_index + visit_index) % 5 == 0:
                    continue
                observations.append(
                    Observation(
                        subject_id=subject_id,
                        timestamp=timestamp,
                        modality=modality,
                        feature=feature,
                        value=value,
                        unit="z_score",
                        standard_error=standard_error,
                    )
                )
        # This measured feature is preserved as an explicit unconfigured exclusion.
        observations.append(
            Observation(
                subject_id=subject_id,
                timestamp=START,
                modality=Modality.CLINICAL,
                feature="body_mass",
                value=float(24.0 + random.normal()),
                unit="kg",
            )
        )
    return Study(
        study_id="synthetic-phase3-state-v1",
        subjects=subjects,
        observations=tuple(observations),
        metadata={"data_status": "fully synthetic; not biological validation"},
    )


def main() -> None:
    """Run the end-to-end synthetic Phase 3 workflow and print audit-ready summaries."""
    study = build_synthetic_study()
    estimator = LinearGaussianStateEstimator(state_config()).fit(
        study,
        reference_subject_ids=REFERENCE_SUBJECTS,
    )
    report = estimator.estimate_report(
        study,
        subject_ids=EVALUATION_SUBJECTS,
        smooth=True,
    )
    calibration = estimator.calibrate_forecasts(
        study,
        reference_subject_ids=REFERENCE_SUBJECTS,
        evaluation_subject_ids=EVALUATION_SUBJECTS,
        interval_level=0.90,
    )
    changes = estimator.detect_innovation_change_points(
        study,
        reference_subject_ids=REFERENCE_SUBJECTS,
        evaluation_subject_ids=EVALUATION_SUBJECTS,
        false_alarm_rate=0.05,
    )

    coverage = [
        trajectory.coverage.observed_fraction
        for trajectory in report.trajectories
        if trajectory.coverage is not None
    ]
    shifted = next(
        trajectory for trajectory in report.trajectories if trajectory.subject_id == SHIFTED_SUBJECT
    )
    terminal = shifted.estimates[-1]
    forecasts = estimator.forecast(
        shifted,
        timestamps=(
            terminal.timestamp + timedelta(days=35),
            terminal.timestamp + timedelta(days=92),
        ),
    )
    detected = tuple(item for item in changes.results if item.detected)
    shifted_detections = tuple(item for item in detected if item.subject_id == SHIFTED_SUBJECT)

    print("Phase 3 synthetic longitudinal state example")
    print(f"Subjects: {len(REFERENCE_SUBJECTS)} reference, {len(EVALUATION_SUBJECTS)} held out")
    print("Prespecified states: inflammatory_burden, functional_reserve")
    print(
        f"Held-out trajectories: {len(report.trajectories)}; "
        f"excluded subjects: {len(report.excluded_subjects)}"
    )
    print(f"Mean observed channel coverage: {float(np.mean(coverage)):.1%}")
    print(
        f"Held-out 90% forecast coverage: {calibration.empirical_coverage:.1%}; "
        f"standardized RMSE: {calibration.standardized_rmse:.2f}"
    )
    print(
        f"Innovation detections: {len(detected)} of {len(changes.results)} held-out "
        f"follow-ups; injected-shift detections: {len(shifted_detections)}"
    )
    print(
        f"{SHIFTED_SUBJECT} terminal smoothed mean: "
        f"inflammation={terminal.mean[0]:+.2f}, reserve={terminal.mean[1]:+.2f}"
    )
    print(
        "Forecast variance after 92 days: "
        f"inflammation={forecasts.estimates[-1].covariance[0][0]:.3f}, "
        f"reserve={forecasts.estimates[-1].covariance[1][1]:.3f}"
    )
    print(
        "Guardrail: all values and the injected shift are synthetic; detections indicate model "
        "surprise, not efficacy, causality, or biological-age reversal."
    )


if __name__ == "__main__":
    main()
