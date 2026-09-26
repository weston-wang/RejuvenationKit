from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from rejuvenationkit.surrogates import (
    SurrogateUnitEffect,
    fit_trial_level,
    individual_level_association,
    predict_outcome_effect,
    simulate_trial_level_precision,
    unit_effects_from_subjects,
    validate_surrogate,
)

COLUMNS = {
    "unit_column": "unit",
    "arm_column": "arm",
    "surrogate_column": "surrogate",
    "outcome_column": "outcome",
}


def subject_frame(
    *,
    trial_correlation: float,
    individual_correlation: float,
    units: int = 12,
    per_arm: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    effects = random.multivariate_normal(
        [0.5, 0.5], [[1.0, trial_correlation], [trial_correlation, 1.0]], units
    )
    rows = []
    for unit, effect in enumerate(effects):
        for arm in ("treated", "control"):
            shift = effect if arm == "treated" else np.zeros(2)
            values = random.multivariate_normal(
                shift,
                [[1.0, individual_correlation], [individual_correlation, 1.0]],
                per_arm,
            )
            rows.extend(
                {"unit": f"u{unit:02d}", "arm": arm, "surrogate": s, "outcome": o}
                for s, o in values
            )
    return pd.DataFrame(rows)


def units_from(frame: pd.DataFrame) -> tuple[SurrogateUnitEffect, ...]:
    return unit_effects_from_subjects(
        frame, treated_label="treated", control_label="control", **COLUMNS
    )


def test_unit_effects_match_welch_arithmetic() -> None:
    frame = subject_frame(trial_correlation=0.9, individual_correlation=0.5, units=3)
    units = units_from(frame)
    first = frame[frame["unit"] == "u00"]
    treated = first[first["arm"] == "treated"][["surrogate", "outcome"]].to_numpy()
    control = first[first["arm"] == "control"][["surrogate", "outcome"]].to_numpy()
    assert units[0].surrogate_effect == pytest.approx(treated[:, 0].mean() - control[:, 0].mean())
    covariance = np.cov(treated, rowvar=False) / 30 + np.cov(control, rowvar=False) / 30
    assert units[0].outcome_standard_error == pytest.approx(np.sqrt(covariance[1, 1]))
    assert units[0].effect_covariance == pytest.approx(covariance[0, 1])
    assert units[0].subjects == 60


def test_valid_surrogate_has_strong_trial_level_association_and_calibrated_predictions() -> None:
    frame = subject_frame(trial_correlation=0.95, individual_correlation=0.5)
    report = validate_surrogate(
        units_from(frame),
        individual_level=individual_level_association(frame, **COLUMNS),
        bootstrap_samples=200,
    )
    assert report.fit.r_squared_trial > 0.7
    assert report.r_squared_trial_interval[0] > 0.3
    assert report.fit.slope > 0
    assert report.held_out_coverage >= 0.75
    assert report.held_out_sign_agreement >= 0.75
    assert report.surrogate_threshold_effect_positive is not None
    assert report.individual_level is not None
    assert report.individual_level.r_squared == pytest.approx(0.25, abs=0.1)
    frame_out = report.held_out_frame()
    assert len(frame_out) == 12
    assert set(frame_out.columns) >= {"predicted_outcome_effect", "covered"}


def test_individual_association_does_not_validate_a_paradoxical_surrogate() -> None:
    frame = subject_frame(trial_correlation=0.0, individual_correlation=0.9, seed=3)
    report = validate_surrogate(
        units_from(frame),
        individual_level=individual_level_association(frame, **COLUMNS),
        bootstrap_samples=200,
    )
    assert report.individual_level is not None
    assert report.individual_level.r_squared > 0.7
    # The trial-level interval must admit "no predictive value".
    assert report.r_squared_trial_interval[0] == 0.0
    assert report.trial_correlation_interval[0] < 0 < report.trial_correlation_interval[1]


def test_trial_level_fit_removes_sampling_error() -> None:
    random = np.random.default_rng(4)
    true = random.multivariate_normal([0, 0], [[1.0, 0.8], [0.8, 1.0]], 400)
    noise_sd = 0.7
    units = tuple(
        SurrogateUnitEffect(
            unit_id=f"u{index}",
            surrogate_effect=float(value[0] + random.normal(0, noise_sd)),
            surrogate_standard_error=noise_sd,
            outcome_effect=float(value[1] + random.normal(0, noise_sd)),
            outcome_standard_error=noise_sd,
        )
        for index, value in enumerate(true)
    )
    fit = fit_trial_level(units)
    naive = np.corrcoef(
        [item.surrogate_effect for item in units], [item.outcome_effect for item in units]
    )[0, 1]
    assert naive < 0.6
    assert fit.trial_correlation == pytest.approx(0.8, abs=0.08)
    assert fit.slope == pytest.approx(0.8, abs=0.12)


def test_prediction_and_precision_helpers() -> None:
    frame = subject_frame(trial_correlation=0.9, individual_correlation=0.5)
    units = units_from(frame)
    report = validate_surrogate(units, bootstrap_samples=100)
    prediction = predict_outcome_effect(
        report, units, surrogate_effect=1.5, surrogate_standard_error=0.1
    )
    low, high = prediction.prediction_interval
    assert low < prediction.predicted_outcome_effect < high
    with pytest.raises(ValueError, match="reproduce"):
        predict_outcome_effect(
            report, units[:-1], surrogate_effect=1.0, surrogate_standard_error=0.1
        )

    table = simulate_trial_level_precision(
        unit_count=8, trial_correlation=0.8, simulations=5, bootstrap_samples=20
    )
    assert len(table) == 5
    assert {"r_squared_trial", "interval_width", "covers_truth"} <= set(table.columns)


def test_inputs_are_validated() -> None:
    with pytest.raises(ValidationError, match="covariance"):
        SurrogateUnitEffect(
            unit_id="u",
            surrogate_effect=0,
            surrogate_standard_error=1,
            outcome_effect=0,
            outcome_standard_error=1,
            effect_covariance=2,
        )
    unit = SurrogateUnitEffect(
        unit_id="u",
        surrogate_effect=0,
        surrogate_standard_error=1,
        outcome_effect=0,
        outcome_standard_error=1,
    )
    with pytest.raises(ValueError, match="three"):
        fit_trial_level((unit, unit))
    with pytest.raises(ValueError, match="unique"):
        fit_trial_level((unit, unit, unit))
    flat = tuple(unit.model_copy(update={"unit_id": f"u{index}"}) for index in range(5))
    with pytest.raises(ValueError, match="do not vary"):
        fit_trial_level(flat)
    frame = subject_frame(trial_correlation=0.5, individual_correlation=0.5, units=3)
    with pytest.raises(ValueError, match="four"):
        validate_surrogate(units_from(frame))
    with pytest.raises(ValueError, match="two complete"):
        unit_effects_from_subjects(
            frame.iloc[:31], treated_label="treated", control_label="control", **COLUMNS
        )
    with pytest.raises(ValueError, match="four complete"):
        individual_level_association(frame.iloc[:3], **COLUMNS)
    with pytest.raises(ValueError):
        simulate_trial_level_precision(unit_count=3, trial_correlation=0.5)
