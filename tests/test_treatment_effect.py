from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import t as student_t

from rejuvenationkit.combinations import AssignmentMechanism
from rejuvenationkit.longitudinal import (
    LongitudinalAlignmentError,
    LongitudinalExclusionReason,
)
from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject
from rejuvenationkit.treatment_effect import (
    CrossValidatedSubjectScore,
    FeatureTreatmentEffect,
    RandomizedTreatmentEffectEvaluator,
    TreatmentEffectConfig,
    TreatmentEffectReport,
    _holm_adjust,
    _omnibus_statistic,
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
        assignment_mechanism=AssignmentMechanism.RANDOMIZED,
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
    assert any(
        item.subject_id == "treated-00"
        and item.visit_id == "month-3"
        and item.reason is LongitudinalExclusionReason.INCOMPLETE_VISIT_VECTOR
        for item in first.exclusions
    )


def test_randomized_evaluator_validates_configuration_and_groups() -> None:
    with pytest.raises(ValidationError, match="assignment_mechanism"):
        TreatmentEffectConfig(features=FEATURES)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="requires randomized assignment"):
        TreatmentEffectConfig(
            features=FEATURES,
            assignment_mechanism=AssignmentMechanism.OBSERVATIONAL,
        )
    with pytest.raises(ValidationError, match="cannot overlap"):
        TreatmentEffectConfig(
            features=(FEATURES[0], FEATURES[0]),
            assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        )
    with pytest.raises(ValidationError, match="cannot overlap"):
        TreatmentEffectConfig(
            features=(
                VisitFeature(feature="inflammation"),
                VisitFeature(feature="inflammation", modality=Modality.CLINICAL),
            ),
            assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        )
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


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_treatment_results_reject_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValidationError, match="finite number"):
        FeatureTreatmentEffect(
            feature="inflammation",
            modality=Modality.CLINICAL,
            treated_mean_change=value,
            control_mean_change=0,
            difference_in_differences=value,
            confidence_interval_low=0,
            confidence_interval_high=1,
        )
    with pytest.raises(ValidationError, match="finite number"):
        CrossValidatedSubjectScore(
            subject_id="subject",
            group="treated",
            follow_up_visit_id="month-1",
            fold=0,
            squared_mahalanobis_distance=value,
            empirical_tail_probability=0.5,
            detected=False,
        )


def test_feature_treatment_effect_rejects_inconsistent_contrast() -> None:
    with pytest.raises(ValidationError, match="does not match group means"):
        FeatureTreatmentEffect(
            feature="inflammation",
            modality=Modality.CLINICAL,
            treated_mean_change=2,
            control_mean_change=1,
            difference_in_differences=2,
            confidence_interval_low=0,
            confidence_interval_high=3,
        )


def test_randomized_evaluator_rejects_mixed_units_across_visits() -> None:
    study, treated_ids, control_ids = randomized_study()
    changed = study.model_copy(
        update={
            "observations": tuple(
                row.model_copy(update={"unit": "different-score"})
                if row.subject_id == treated_ids[0]
                and row.timestamp == START + timedelta(days=90)
                and row.feature == "frailty"
                else row
                for row in study.observations
            )
        }
    )

    with pytest.raises(LongitudinalAlignmentError) as caught:
        RandomizedTreatmentEffectEvaluator(config()).evaluate(
            changed,
            baseline=BASELINE,
            follow_ups=(MONTH_3,),
            treated_subject_ids=treated_ids,
            control_subject_ids=control_ids,
        )

    assert caught.value.exclusions[0].reason is LongitudinalExclusionReason.MIXED_UNITS


def test_report_serializes_compact_fold_calibration_and_inference_provenance() -> None:
    study, treated_ids, control_ids = randomized_study()
    report = RandomizedTreatmentEffectEvaluator(config()).evaluate(
        study,
        baseline=BASELINE,
        follow_ups=(MONTH_1, MONTH_3),
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
        treated_label="rapamycin",
    )

    provenance = report.inference_provenance
    assert provenance is not None
    assert provenance.config == config()
    assert provenance.config.assignment_mechanism is AssignmentMechanism.RANDOMIZED
    assert provenance.baseline_visit == BASELINE
    assert provenance.follow_up_visits == (MONTH_1, MONTH_3)
    assert len(provenance.fold_calibrations) == 8
    assert len(provenance.analysis_input_artifact_hash) == 64
    assert "diagonal shrinkage" in provenance.covariance_method
    calibrations = {
        (item.follow_up_visit_id, item.fold): item for item in provenance.fold_calibrations
    }
    for score in report.subject_scores:
        calibration = calibrations[(score.follow_up_visit_id, score.fold)]
        assert score.calibration_id == calibration.calibration_id
        assert score.calibration_reference_count == calibration.training_control_count
        assert (
            score.calibration_reference_artifact_hash
            == calibration.training_reference_artifact_hash
        )
        assert score.empirical_threshold == calibration.empirical_threshold
        assert score.detected == (
            score.squared_mahalanobis_distance > calibration.empirical_threshold
        )

    serialized = report.model_dump_json()
    assert "reference_score_distribution" not in serialized
    assert "bootstrap_differences" not in serialized
    assert TreatmentEffectReport.model_validate_json(serialized) == report


def test_report_provenance_is_deterministic_and_rejects_relabeling_or_forgery() -> None:
    study, treated_ids, control_ids = randomized_study()
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

    relabeled = first.model_dump(mode="json")
    original_group = relabeled["subject_scores"][0]["group"]
    relabeled["subject_scores"][0]["subject_id"] = (
        control_ids[0] if original_group == "treated" else treated_ids[0]
    )
    with pytest.raises(ValidationError, match="subject score"):
        TreatmentEffectReport.model_validate(relabeled)

    forged = first.model_dump(mode="json")
    calibration = forged["inference_provenance"]["fold_calibrations"][0]
    calibration["empirical_threshold"] += 1
    with pytest.raises(ValidationError, match="calibration_id"):
        TreatmentEffectReport.model_validate(forged)

    wrong_channel = first.model_dump(mode="json")
    wrong_channel["inference_provenance"]["resolved_channels"][0]["feature"] = "forged"
    with pytest.raises(ValidationError, match="resolved inference channel"):
        TreatmentEffectReport.model_validate(wrong_channel)

    unknown_excluded = first.model_dump(mode="json")
    unknown_excluded["excluded_subject_ids"] = ["unknown-subject"]
    with pytest.raises(ValidationError, match="prespecified group"):
        TreatmentEffectReport.model_validate(unknown_excluded)


def test_treatment_effect_fails_closed_when_visits_reuse_source_rows() -> None:
    study, treated_ids, control_ids = randomized_study()
    overlapping_baseline = BASELINE.model_copy(update={"window_after": timedelta(days=30)})
    overlapping_follow_up = MONTH_1.model_copy(update={"window_before": timedelta(days=30)})

    with pytest.raises(ValueError, match="insufficient complete treated"):
        RandomizedTreatmentEffectEvaluator(config()).evaluate(
            study,
            baseline=overlapping_baseline,
            follow_ups=(overlapping_follow_up,),
            treated_subject_ids=treated_ids,
            control_subject_ids=control_ids,
        )


def test_visit_p_values_are_holm_adjusted_across_follow_ups() -> None:
    assert _holm_adjust((0.01, 0.04, 0.03)) == pytest.approx((0.03, 0.06, 0.06))
    study, treated_ids, control_ids = randomized_study()
    report = RandomizedTreatmentEffectEvaluator(config()).evaluate(
        study,
        baseline=BASELINE,
        follow_ups=(MONTH_1, MONTH_3),
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
    )
    raw = tuple(visit.permutation_p_value for visit in report.visit_effects)
    assert tuple(
        visit.holm_adjusted_permutation_p_value for visit in report.visit_effects
    ) == pytest.approx(_holm_adjust(raw))

    forged = report.model_dump(mode="python")
    forged["visit_effects"][0]["holm_adjusted_permutation_p_value"] = raw[0]
    forged["visit_effects"][1]["holm_adjusted_permutation_p_value"] = raw[1]
    if _holm_adjust(raw) != raw:
        with pytest.raises(ValidationError, match="Holm"):
            TreatmentEffectReport.model_validate(forged)


def test_feature_interval_is_welch_t_times_bootstrap_standard_error() -> None:
    study, treated_ids, control_ids = randomized_study()
    report = RandomizedTreatmentEffectEvaluator(
        config().model_copy(update={"bootstrap_samples": 20_000})
    ).evaluate(
        study,
        baseline=BASELINE,
        follow_ups=(MONTH_3,),
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
    )
    for effect in report.visit_effects[0].effects:
        assert effect.standard_error is not None
        half_width = (effect.confidence_interval_high - effect.confidence_interval_low) / 2
        # 20 per arm: Welch df is near 38, so the critical value lies in t(19)..t(38).
        ratio = half_width / effect.standard_error
        assert student_t.ppf(0.975, 38) - 1e-9 <= ratio <= student_t.ppf(0.975, 19) + 1e-9


def test_omnibus_covariance_is_not_inflated_by_the_treatment_effect() -> None:
    random = np.random.default_rng(3)
    base = random.standard_normal((20, 2))
    labels = np.zeros(20, dtype=bool)
    labels[:10] = True
    shifted = base.copy()
    shifted[labels] += 10.0
    settings = config()

    # Arm-centered pooled covariance ignores the location shift, so the statistic
    # uses the same covariance as the unshifted data.
    centered = np.vstack(
        (
            base[labels] - base[labels].mean(axis=0),
            base[~labels] - base[~labels].mean(axis=0),
        )
    )
    covariance = centered.T @ centered / 18
    shrinkage = settings.covariance_shrinkage
    covariance = (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
    covariance += np.eye(2) * settings.covariance_ridge * max(np.trace(covariance) / 2, 1.0)
    difference = shifted[labels].mean(axis=0) - shifted[~labels].mean(axis=0)
    expected = float(difference @ np.linalg.solve(covariance, difference))

    assert _omnibus_statistic(shifted, labels, settings) == pytest.approx(expected)
    # Pooling without centering would absorb the shift into the covariance.
    total = np.cov(shifted, rowvar=False)
    assert expected > 10 * float(difference @ np.linalg.solve(total, difference))


def test_stratified_permutation_is_recorded_and_requires_complete_strata() -> None:
    study, treated_ids, control_ids = randomized_study()
    strata = {
        subject_id: f"block-{int(subject_id[-2:]) % 4}"
        for subject_id in (*treated_ids, *control_ids)
    }
    report = RandomizedTreatmentEffectEvaluator(config()).evaluate(
        study,
        baseline=BASELINE,
        follow_ups=(MONTH_3,),
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
        randomization_strata=strata,
    )
    assert report.inference_provenance is not None
    assert report.inference_provenance.randomization_strata == tuple(sorted(strata.items()))
    assert report.visit_effects[0].permutation_p_value <= 0.01
    TreatmentEffectReport.model_validate(report.model_dump(mode="python"))

    incomplete = dict(strata)
    incomplete.pop(treated_ids[0])
    with pytest.raises(ValueError, match="randomization strata"):
        RandomizedTreatmentEffectEvaluator(config()).evaluate(
            study,
            baseline=BASELINE,
            follow_ups=(MONTH_3,),
            treated_subject_ids=treated_ids,
            control_subject_ids=control_ids,
            randomization_strata=incomplete,
        )
