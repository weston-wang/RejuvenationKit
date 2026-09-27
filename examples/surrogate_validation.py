"""Evaluate an aging biomarker as a surrogate endpoint across independent cohorts.

Two synthetic scenarios with twelve independent cohorts (trials, sites, or
intervention arms each with its own controls) and 40 dogs per arm:

* ``valid``: a cohort's effect on the biomarker (say, epigenetic-age change)
  strongly predicts its effect on the outcome (say, frailty-index change).
* ``paradox``: the biomarker and outcome are tightly correlated across
  individual dogs, but an intervention's effect on the biomarker says nothing
  about its effect on the outcome. Individual-level association alone would
  wrongly endorse this biomarker.

The data are simulated; the point is to show what each diagnostic can and
cannot reveal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rejuvenationkit.surrogates import (
    individual_level_association,
    unit_effects_from_subjects,
    validate_surrogate,
)


def simulate(
    *,
    trial_correlation: float,
    individual_correlation: float,
    cohorts: int = 12,
    dogs_per_arm: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate subject-level surrogate and outcome changes in several cohorts."""
    random = np.random.default_rng(seed)
    effects = random.multivariate_normal(
        [-0.5, -0.4],
        [[1.0, trial_correlation], [trial_correlation, 1.0]],
        cohorts,
    )
    rows = []
    for cohort, effect in enumerate(effects):
        for arm in ("treated", "control"):
            shift = effect if arm == "treated" else np.zeros(2)
            values = random.multivariate_normal(
                shift,
                [[1.0, individual_correlation], [individual_correlation, 1.0]],
                dogs_per_arm,
            )
            rows.extend(
                {
                    "cohort": f"cohort-{cohort:02d}",
                    "arm": arm,
                    "epigenetic_age_change": surrogate,
                    "frailty_change": outcome,
                }
                for surrogate, outcome in values
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run both scenarios and print the diagnostics side by side."""
    for name, trial_correlation, individual_correlation in (
        ("valid", 0.9, 0.5),
        ("paradox", 0.0, 0.9),
    ):
        frame = simulate(
            trial_correlation=trial_correlation,
            individual_correlation=individual_correlation,
        )
        columns = {
            "unit_column": "cohort",
            "arm_column": "arm",
            "surrogate_column": "epigenetic_age_change",
            "outcome_column": "frailty_change",
        }
        units = unit_effects_from_subjects(
            frame, treated_label="treated", control_label="control", **columns
        )
        report = validate_surrogate(
            units,
            individual_level=individual_level_association(frame, **columns),
            bootstrap_samples=500,
        )
        individual = report.individual_level
        assert individual is not None
        low, high = report.r_squared_trial_interval
        print(f"Scenario: {name}")
        print(
            f"  individual-level R2 {individual.r_squared:.2f} "
            f"({individual.r_squared_interval[0]:.2f}-{individual.r_squared_interval[1]:.2f})"
        )
        print(f"  trial-level R2      {report.fit.r_squared_trial:.2f} ({low:.2f}-{high:.2f})")
        print(
            f"  held-out prediction coverage {report.held_out_coverage:.0%}, "
            f"sign agreement {report.held_out_sign_agreement:.0%}, "
            f"mean absolute error {report.held_out_mean_absolute_error:.2f}"
        )
        threshold = report.surrogate_threshold_effect_negative
        described = "none within range" if threshold is None else f"{threshold:.2f}"
        print(
            "  surrogate threshold effect (biomarker reduction needed to predict a "
            f"nonzero outcome effect): {described}"
        )
        if report.warnings:
            print(f"  warnings: {', '.join(report.warnings)}")
        print()


if __name__ == "__main__":
    main()
