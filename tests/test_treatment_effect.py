from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject
from rejuvenationkit.treatment_effect import (
    RandomizedTreatmentEffectEvaluator,
    TreatmentEffectConfig,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
FEATURES = (
    VisitFeature(feature="inflammation", modality=Modality.CLINICAL),
    VisitFeature(feature="frailty", modality=Modality.CLINICAL),
)
BASELINE = ExpectedVisit(visit_id="baseline", scheduled_at=START, required_features=FEATURES)
MONTH_1 = ExpectedVisit(
    visit_id="month-1",
    scheduled_at=START + timedelta(days=30),
    required_features=FEATURES,
)
MONTH_3 = ExpectedVisit(
    visit_id="month-3",
    scheduled_at=START + timedelta(days=90),
    required_features=FEATURES,
)


def randomized_study(
    *, omit_subject: str | None = None
) -> tuple[Study, tuple[str, ...], tuple[str, ...]]:
    random = np.random.default_rng(42)
    treated_ids = tuple(f"treated-{index:02d}" for index in range(20))
    control_ids = tuple(f"control-{index:02d}" for index in range(20))
    subjects = tuple(
        Subject(
            subject_id=subject_id,
            cohort="treated" if subject_id in treated_ids else "control",
            interventions=("rapamycin",) if subject_id in treated_ids else (),
        )
        for subject_id in (*treated_ids, *control_ids)
    )
    observations: list[Observation] = []
    for subject_id in (*treated_ids, *control_ids):
        baseline = random.normal(0, 0.5, 2)
        treatment = np.array([-1.5, -1.0]) if subject_id in treated_ids else np.zeros(2)
        for visit_index, (timestamp, fraction) in enumerate(
            ((START, 0.0), (START + timedelta(days=30), 0.5), (START + timedelta(days=90), 1.0))
        ):
            values = baseline + treatment * fraction + random.normal(0, 0.20, 2)
            for feature, value in zip(FEATURES, values, strict=True):
                if subject_id == omit_subject and visit_index == 2 and feature == FEATURES[1]:
                    continue
                observations.append(
                    Observation(
                        subject_id=subject_id,
                        timestamp=timestamp,
                        modality=Modality.CLINICAL,
                        feature=feature.feature,
                        value=float(value),
                        unit="score",
                    )
                )
    return (
        Study(study_id="randomized-rapamycin", subjects=subjects, observations=tuple(observations)),
        treated_ids,
        control_ids,
    )


def config() -> TreatmentEffectConfig:
    return TreatmentEffectConfig(
        features=FEATURES,
        cross_validation_folds=4,
        permutations=199,
        bootstrap_samples=199,
        minimum_group_size=8,
        random_seed=7,
    )


def test_randomized_evaluator_detects_longitudinal_group_effect() -> None:
    study, treated_ids, control_ids = randomized_study()
    report = RandomizedTreatmentEffectEvaluator(config()).evaluate(
        study,
        baseline=BASELINE,
        follow_ups=(MONTH_1, MONTH_3),
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
        treated_label="rapamycin",
    )

    assert len(report.subject_scores) == 80
    assert len({score.subject_id for score in report.subject_scores}) == 40
    final = report.visit_effects[-1]
    assert final.permutation_p_value <= 0.01
    assert all(effect.difference_in_differences < -0.7 for effect in final.effects)
    assert all(effect.confidence_interval_high < 0 for effect in final.effects)
    assert set(report.scores_frame()["group"]) == {"rapamycin", "control"}
    assert len(report.effects_frame()) == 4


def test_randomized_evaluator_reports_incomplete_subjects_deterministically() -> None:
    study, treated_ids, control_ids = randomized_study(omit_subject="treated-00")
    evaluator = RandomizedTreatmentEffectEvaluator(config())
    first = evaluator.evaluate(
        study,
        baseline=BASELINE,
        follow_ups=(MONTH_3,),
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
    )
    second = evaluator.evaluate(
        study,
        baseline=BASELINE,
        follow_ups=(MONTH_3,),
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
    )

    assert first == second
    assert first.excluded_subject_ids == ("treated-00",)
    assert first.visit_effects[0].treated_subjects == 19


def test_randomized_evaluator_validates_configuration_and_groups() -> None:
    with pytest.raises(ValidationError, match="unique"):
        TreatmentEffectConfig(features=(FEATURES[0], FEATURES[0]))
    study, treated_ids, control_ids = randomized_study()
    evaluator = RandomizedTreatmentEffectEvaluator(config())
    with pytest.raises(ValueError, match="overlap"):
        evaluator.evaluate(
            study,
            baseline=BASELINE,
            follow_ups=(MONTH_3,),
            treated_subject_ids=treated_ids,
            control_subject_ids=(*control_ids, treated_ids[0]),
        )
    with pytest.raises(ValueError, match="follow-up"):
        evaluator.evaluate(
            study,
            baseline=BASELINE,
            follow_ups=(),
            treated_subject_ids=treated_ids,
            control_subject_ids=control_ids,
        )
    with pytest.raises(ValueError, match="unique"):
        evaluator.evaluate(
            study,
            baseline=BASELINE,
            follow_ups=(MONTH_3, MONTH_3),
            treated_subject_ids=treated_ids,
            control_subject_ids=control_ids,
        )
    with pytest.raises(ValueError, match="after baseline"):
        evaluator.evaluate(
            study,
            baseline=MONTH_3,
            follow_ups=(MONTH_1,),
            treated_subject_ids=treated_ids,
            control_subject_ids=control_ids,
        )
