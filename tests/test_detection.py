from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from rejuvenationkit.detection import (
    ChangeDetectionConfig,
    ChangeDetectionModel,
    ChangeDetectionReport,
    MultivariateChangeDetector,
    SubjectChangeDetection,
    _leave_one_out_scores,
    _regularized_covariance,
)
from rejuvenationkit.longitudinal import (
    AggregationPolicy,
    LongitudinalAlignmentError,
    LongitudinalChannel,
    LongitudinalExclusionReason,
)
from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject

START = datetime(2026, 1, 1, tzinfo=UTC)
FEATURES = (
    VisitFeature(feature="albumin", modality=Modality.CLINICAL),
    VisitFeature(feature="creatinine", modality=Modality.CLINICAL),
)
BASELINE = ExpectedVisit(
    visit_id="baseline",
    scheduled_at=START,
    required_features=FEATURES,
)
FOLLOW_UP = ExpectedVisit(
    visit_id="follow-up",
    scheduled_at=START + timedelta(days=30),
    required_features=FEATURES,
)


def _study() -> tuple[Study, tuple[str, ...], tuple[str, ...]]:
    random = np.random.default_rng(7)
    reference_ids = tuple(f"reference-{index:02d}" for index in range(40))
    shifted_ids = tuple(f"shifted-{index:02d}" for index in range(6))
    subjects = tuple(
        Subject(
            subject_id=subject_id,
            cohort="reference" if subject_id in reference_ids else "shifted",
        )
        for subject_id in (*reference_ids, *shifted_ids)
    )
    covariance = np.array([[1.0, 0.8], [0.8, 1.0]])
    reference_changes = random.multivariate_normal([0, 0], covariance, len(reference_ids))
    shifted_changes = random.multivariate_normal([5, 5], covariance, len(shifted_ids))
    observations: list[Observation] = []
    for subject_id, change in zip(
        (*reference_ids, *shifted_ids),
        np.vstack((reference_changes, shifted_changes)),
        strict=True,
    ):
        for feature, value in zip(FEATURES, change, strict=True):
            observations.extend(
                (
                    Observation(
                        subject_id=subject_id,
                        timestamp=START,
                        modality=Modality.CLINICAL,
                        feature=feature.feature,
                        value=0,
                        unit="value",
                    ),
                    Observation(
                        subject_id=subject_id,
                        timestamp=START + timedelta(days=30),
                        modality=Modality.CLINICAL,
                        feature=feature.feature,
                        value=float(value),
                        unit="value",
                    ),
                )
            )
    return (
        Study(study_id="detection", subjects=subjects, observations=tuple(observations)),
        reference_ids,
        shifted_ids,
    )


def test_detector_finds_correlated_multivariate_shift() -> None:
    study, reference_ids, shifted_ids = _study()
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(
            features=FEATURES,
            covariance_shrinkage=0.1,
            false_alarm_rate=0.05,
            minimum_reference_subjects=20,
        )
    ).fit(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        reference_subject_ids=reference_ids,
    )

    report = detector.score(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        subject_ids=shifted_ids,
    )

    assert sum(result.detected for result in report.results) >= 5
    assert report.model.reference_subjects == 40
    assert report.results_frame()["squared_mahalanobis_distance"].min() > 0

    forged = report.model_dump(mode="python")
    forged["results"][0]["squared_mahalanobis_distance"] += 1
    with pytest.raises(ValidationError, match="score does not match"):
        ChangeDetectionReport.model_validate(forged)


def test_detector_rejects_wildcard_and_exact_alias_channels() -> None:
    with pytest.raises(ValidationError, match="cannot overlap"):
        ChangeDetectionConfig(
            features=(
                VisitFeature(feature="albumin"),
                VisitFeature(feature="albumin", modality=Modality.CLINICAL),
            ),
            minimum_reference_subjects=20,
        )


def test_detector_reports_incomplete_subjects_and_requires_fit() -> None:
    study, reference_ids, _ = _study()
    incomplete = study.model_copy(
        update={
            "subjects": (
                *study.subjects,
                Subject(subject_id="missing", cohort="test"),
            )
        }
    )
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=20)
    )
    with pytest.raises(RuntimeError, match="fit"):
        detector.score(incomplete, baseline=BASELINE, follow_up=FOLLOW_UP)

    detector.fit(
        incomplete,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        reference_subject_ids=reference_ids,
    )
    report = detector.score(
        incomplete,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        subject_ids=("missing",),
    )
    assert report.results == ()
    assert report.excluded_subject_ids == ("missing",)


def test_detector_excludes_mixed_finite_nonfinite_visit_channel() -> None:
    study, reference_ids, shifted_ids = _study()
    affected_subject = shifted_ids[0]
    nonfinite_index = len(study.observations)
    contaminated = study.model_copy(
        update={
            "observations": (
                *study.observations,
                Observation(
                    subject_id=affected_subject,
                    timestamp=FOLLOW_UP.scheduled_at,
                    modality=Modality.CLINICAL,
                    feature="albumin",
                    value=float("nan"),
                    unit="value",
                    replicate_id="nonfinite-replicate",
                ),
            )
        }
    )
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=20)
    ).fit(
        contaminated,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        reference_subject_ids=reference_ids,
    )

    report = detector.score(
        contaminated,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        subject_ids=shifted_ids,
    )

    assert affected_subject in report.excluded_subject_ids
    assert affected_subject not in {item.subject_id for item in report.results}
    nonfinite = tuple(
        item
        for item in report.exclusions
        if item.reason is LongitudinalExclusionReason.NONFINITE_OBSERVATION
    )
    assert len(nonfinite) == 1
    assert nonfinite[0].subject_id == affected_subject
    assert nonfinite[0].visit_id == FOLLOW_UP.visit_id
    assert nonfinite[0].observation_indices == (nonfinite_index,)


def test_detector_validates_configuration_and_reference_size() -> None:
    with pytest.raises(ValidationError, match="at least 2"):
        ChangeDetectionConfig(features=(FEATURES[0],))
    with pytest.raises(ValidationError, match="cannot overlap"):
        ChangeDetectionConfig(features=(FEATURES[0], FEATURES[0]))

    study, reference_ids, _ = _study()
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=20)
    )
    with pytest.raises(ValueError, match="insufficient"):
        detector.fit(
            study,
            baseline=BASELINE,
            follow_up=FOLLOW_UP,
            reference_subject_ids=reference_ids[:10],
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_change_detection_results_reject_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValidationError, match="finite number"):
        SubjectChangeDetection(
            subject_id="candidate",
            change=(value, 0),
            innovation=(0, 0),
            whitened_innovation=(0, 0),
            squared_mahalanobis_distance=1,
            empirical_tail_probability=0.5,
            detected=False,
        )


def test_detector_fails_closed_when_evaluation_reuses_reference_subjects() -> None:
    study, reference_ids, _ = _study()
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=20)
    ).fit(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        reference_subject_ids=reference_ids,
    )

    with pytest.raises(ValueError, match="overlap fitted reference"):
        detector.score(
            study,
            baseline=BASELINE,
            follow_up=FOLLOW_UP,
            subject_ids=(reference_ids[0],),
        )
    with pytest.raises(ValueError, match="overlap fitted reference"):
        detector.score(study, baseline=BASELINE, follow_up=FOLLOW_UP)

    assert detector.model_ is not None
    assert detector.model_.feature_units == ("value", "value")


def test_detector_applies_fitted_reference_units_to_evaluation_subjects() -> None:
    study, reference_ids, shifted_ids = _study()
    exact_channels = tuple(
        LongitudinalChannel(
            feature=feature.feature,
            modality=Modality.CLINICAL,
            unit="value",
            aggregation_policy=AggregationPolicy.CLOSEST_TO_SCHEDULE,
        )
        for feature in FEATURES
    )
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(features=exact_channels, minimum_reference_subjects=20)
    ).fit(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        reference_subject_ids=reference_ids,
    )
    changed = study.model_copy(
        update={
            "observations": tuple(
                row.model_copy(update={"unit": "other"}) if row.subject_id in shifted_ids else row
                for row in study.observations
            )
        }
    )

    with pytest.raises(LongitudinalAlignmentError, match="unit_mismatch"):
        detector.score(
            changed,
            baseline=BASELINE,
            follow_up=FOLLOW_UP,
            subject_ids=shifted_ids,
        )
    assert detector.model_ is not None
    assert detector.model_.aggregation_policies == (
        AggregationPolicy.CLOSEST_TO_SCHEDULE,
        AggregationPolicy.CLOSEST_TO_SCHEDULE,
    )


def test_fitted_detector_is_deterministic_serializable_and_reconstructable() -> None:
    study, reference_ids, shifted_ids = _study()
    config = ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=20)
    first = MultivariateChangeDetector(config).fit(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        reference_subject_ids=reference_ids,
    )
    second = MultivariateChangeDetector(config).fit(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        reference_subject_ids=tuple(reversed(reference_ids)),
    )

    assert first.model_ is not None
    assert second.model_ is not None
    assert first.model_ == second.model_
    assert first.model_.baseline_visit == BASELINE
    assert first.model_.follow_up_visit == FOLLOW_UP
    assert first.model_.resolved_channels
    assert first.model_.reference_subject_ids == tuple(sorted(reference_ids))
    assert len(first.model_.reference_input_artifact_hash or "") == 64
    assert len(first.model_.model_artifact_hash or "") == 64
    assert first.model_.threshold_quantile_method == "higher"

    restored_model = ChangeDetectionModel.model_validate_json(first.model_.model_dump_json())
    restored = MultivariateChangeDetector.from_model(restored_model)
    assert restored.score(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        subject_ids=shifted_ids,
    ) == first.score(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        subject_ids=shifted_ids,
    )


def test_serialized_detector_rejects_tampered_fitted_model_fields() -> None:
    study, reference_ids, _ = _study()
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=20)
    ).fit(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        reference_subject_ids=reference_ids,
    )
    assert detector.model_ is not None
    original = detector.model_.model_dump(mode="json")

    threshold = copy.deepcopy(original)
    threshold["threshold"] = float(threshold["threshold"]) + 1.0

    config = copy.deepcopy(original)
    config["config"]["false_alarm_rate"] = 0.10

    channel = copy.deepcopy(original)
    channel["resolved_channels"][0]["unit"] = "tampered-unit"

    reference_identity = copy.deepcopy(original)
    reference_identity["requested_reference_subject_ids"] = sorted(
        [*reference_identity["requested_reference_subject_ids"], "unexpected-reference"]
    )

    reference_hash = copy.deepcopy(original)
    reference_hash["reference_input_artifact_hash"] = "0" * 64

    for payload, error in (
        (threshold, "threshold"),
        (config, "false-alarm"),
        (channel, "channel"),
        (reference_identity, "artifact hash"),
        (reference_hash, "artifact hash"),
    ):
        with pytest.raises(ValidationError, match=error):
            ChangeDetectionModel.model_validate(payload)
        with pytest.raises(ValidationError, match=error):
            ChangeDetectionModel.model_validate_json(json.dumps(payload))


def test_detector_from_model_revalidates_model_copy_updates() -> None:
    study, reference_ids, _ = _study()
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=20)
    ).fit(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        reference_subject_ids=reference_ids,
    )
    assert detector.model_ is not None
    tampered = detector.model_.model_copy(update={"threshold": detector.model_.threshold + 1.0})

    with pytest.raises(ValidationError, match="threshold"):
        MultivariateChangeDetector.from_model(tampered)


def test_detector_rejects_schedule_drift_and_reference_relabeling() -> None:
    study, reference_ids, shifted_ids = _study()
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=20)
    ).fit(
        study,
        baseline=BASELINE,
        follow_up=FOLLOW_UP,
        reference_subject_ids=reference_ids,
    )
    altered_follow_up = FOLLOW_UP.model_copy(update={"window_after": timedelta(days=1)})
    with pytest.raises(ValueError, match="visit definitions"):
        detector.score(
            study,
            baseline=BASELINE,
            follow_up=altered_follow_up,
            subject_ids=shifted_ids,
        )

    reference_id = reference_ids[0]
    candidate_id = shifted_ids[0]
    relabeled = study.model_copy(
        update={
            "observations": tuple(
                row.model_copy(
                    update={
                        "subject_id": (
                            candidate_id
                            if row.subject_id == reference_id
                            else reference_id
                            if row.subject_id == candidate_id
                            else row.subject_id
                        )
                    }
                )
                for row in study.observations
            )
        }
    )
    with pytest.raises(ValueError, match="reference input"):
        detector.score(
            relabeled,
            baseline=BASELINE,
            follow_up=FOLLOW_UP,
            subject_ids=(shifted_ids[1],),
        )


def test_detector_fails_closed_when_visit_windows_reuse_source_rows() -> None:
    study, reference_ids, _ = _study()
    overlapping_baseline = BASELINE.model_copy(update={"window_after": timedelta(days=30)})
    overlapping_follow_up = FOLLOW_UP.model_copy(update={"window_before": timedelta(days=30)})

    with pytest.raises(ValueError, match="insufficient complete reference"):
        MultivariateChangeDetector(
            ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=20)
        ).fit(
            study,
            baseline=overlapping_baseline,
            follow_up=overlapping_follow_up,
            reference_subject_ids=reference_ids,
        )


def test_reference_scores_are_leave_one_out() -> None:
    study, reference_ids, _ = _study()
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=20)
    ).fit(study, baseline=BASELINE, follow_up=FOLLOW_UP, reference_subject_ids=reference_ids)
    assert detector.model_ is not None
    model = detector.model_
    assert model.reference_score_method == "leave_one_out"

    matrix = np.asarray(
        [
            [
                next(
                    row.value
                    for row in study.observations
                    if row.subject_id == subject_id
                    and row.feature == feature.feature
                    and row.timestamp == FOLLOW_UP.scheduled_at
                )
                for feature in FEATURES
            ]
            for subject_id in model.reference_subject_ids
        ]
    )
    expected = []
    for index in range(len(matrix)):
        training = np.delete(matrix, index, axis=0)
        empirical = np.cov(training, rowvar=False, ddof=1)
        covariance = 0.8 * empirical + 0.2 * np.diag(np.diag(empirical))
        covariance += np.eye(2) * 1e-9 * max(np.trace(covariance) / 2, 1.0)
        innovation = matrix[index] - training.mean(axis=0)
        expected.append(float(innovation @ np.linalg.solve(covariance, innovation)))
    np.testing.assert_allclose(model.reference_score_distribution, sorted(expected), rtol=1e-10)


def test_leave_one_out_threshold_holds_nominal_false_alarm_rate() -> None:
    random = np.random.default_rng(11)
    realized = []
    for _ in range(300):
        reference = random.standard_normal((20, 4))
        covariance = _regularized_covariance(reference, shrinkage=0.2, ridge=1e-9)
        scores = _leave_one_out_scores(reference, shrinkage=0.2, ridge=1e-9)
        threshold = np.quantile(scores, 0.95, method="higher")
        held_out = random.standard_normal((200, 4)) - reference.mean(axis=0)
        held_out_scores = np.einsum("ij,jk,ik->i", held_out, np.linalg.inv(covariance), held_out)
        realized.append(np.mean(held_out_scores > threshold))
    # In-sample calibration realizes about 17% here.
    assert np.mean(realized) < 0.065


def test_detector_requires_more_reference_subjects_than_features() -> None:
    study, reference_ids, _ = _study()
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(features=FEATURES, minimum_reference_subjects=3)
    )
    with pytest.raises(ValueError, match="feature count \\+ 2"):
        detector.fit(
            study,
            baseline=BASELINE,
            follow_up=FOLLOW_UP,
            reference_subject_ids=reference_ids[:3],
        )
