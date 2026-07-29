"""Synthetic canine rapamycin study with leakage-safe longitudinal inference."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from rejuvenationkit import (
    ExpectedVisit,
    Modality,
    Observation,
    RandomizedTreatmentEffectEvaluator,
    Study,
    Subject,
    TreatmentEffectConfig,
    VisitFeature,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
FEATURES = (
    VisitFeature(feature="il6", modality=Modality.CLINICAL),
    VisitFeature(feature="crp", modality=Modality.CLINICAL),
    VisitFeature(feature="frailty_index", modality=Modality.CLINICAL),
)


def build_study() -> tuple[Study, tuple[str, ...], tuple[str, ...]]:
    """Create a randomized 60-dog study with a delayed, heterogeneous response."""
    random = np.random.default_rng(17)
    treated_ids = tuple(f"rapa-{index:02d}" for index in range(30))
    control_ids = tuple(f"placebo-{index:02d}" for index in range(30))
    subjects = tuple(
        Subject(
            subject_id=subject_id,
            cohort="rapamycin" if subject_id in treated_ids else "placebo",
            interventions=("rapamycin",) if subject_id in treated_ids else (),
        )
        for subject_id in (*treated_ids, *control_ids)
    )
    observations: list[Observation] = []
    covariance = np.asarray([[0.16, 0.10, 0.04], [0.10, 0.16, 0.04], [0.04, 0.04, 0.09]])
    for subject_id in (*treated_ids, *control_ids):
        starting = random.normal([3.0, 4.0, 0.35], [0.5, 0.6, 0.05])
        response_scale = max(random.normal(1.0, 0.25), 0.2)
        for day, fraction in ((0, 0.0), (30, 0.25), (90, 0.70), (180, 1.0)):
            treatment = (
                np.asarray([-1.0, -1.2, -0.08]) * fraction * response_scale
                if subject_id in treated_ids
                else np.zeros(3)
            )
            values = starting + treatment + random.multivariate_normal(np.zeros(3), covariance)
            for feature, value in zip(FEATURES, values, strict=True):
                observations.append(
                    Observation(
                        subject_id=subject_id,
                        timestamp=START + timedelta(days=day),
                        modality=Modality.CLINICAL,
                        feature=feature.feature,
                        value=float(value),
                        unit="normalized",
                    )
                )
    return (
        Study(study_id="canine-rapamycin-rct", subjects=subjects, observations=tuple(observations)),
        treated_ids,
        control_ids,
    )


def main() -> None:
    """Run cross-validated scoring and randomized group inference."""
    study, treated_ids, control_ids = build_study()
    visits = tuple(
        ExpectedVisit(
            visit_id=name,
            scheduled_at=START + timedelta(days=day),
            required_features=FEATURES,
        )
        for name, day in (("baseline", 0), ("month-1", 30), ("month-3", 90), ("month-6", 180))
    )
    report = RandomizedTreatmentEffectEvaluator(
        TreatmentEffectConfig(
            features=FEATURES,
            cross_validation_folds=5,
            permutations=999,
            bootstrap_samples=999,
            random_seed=19,
        )
    ).evaluate(
        study,
        baseline=visits[0],
        follow_ups=visits[1:],
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
        treated_label="rapamycin",
        control_label="placebo",
    )
    print("Leakage-safe randomized canine rapamycin analysis")
    for visit in report.visit_effects:
        statistic = visit.omnibus_squared_mahalanobis_distance
        print(
            f"{visit.follow_up_visit_id}: omnibus={statistic:.2f}, "
            f"permutation p={visit.permutation_p_value:.3f}"
        )
        for effect in visit.effects:
            print(
                f"  {effect.feature}: effect={effect.difference_in_differences:+.3f} "
                f"95% CI [{effect.confidence_interval_low:+.3f}, "
                f"{effect.confidence_interval_high:+.3f}]"
            )
    scores = report.scores_frame()
    rates = scores.groupby(["follow_up_visit_id", "group"])["detected"].mean()
    print("\nCross-validated unusual-trajectory rates")
    print(rates.to_string())


if __name__ == "__main__":
    main()
