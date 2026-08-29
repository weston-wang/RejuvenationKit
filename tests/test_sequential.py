from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject
from rejuvenationkit.sequential import (
    SequentialDetectionConfig,
    SequentialDetectionModel,
    SequentialDetectionPoint,
    SequentialDetectionReport,
    SequentialTreatmentResponseDetector,
    SubjectSequentialDetection,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
FEATURES = (
    VisitFeature(feature="activity", modality=Modality.WEARABLE),
    VisitFeature(feature="ethanolamine", modality=Modality.METABOLOMICS),
)
VISITS = tuple(
    ExpectedVisit(
        visit_id=f"year-{year}",
        scheduled_at=START + timedelta(days=365 * year),
        required_features=FEATURES,
    )
    for year in range(4)
)


def _synthetic_study() -> tuple[Study, tuple[str, ...], tuple[str, ...]]:
    random = np.random.default_rng(17)
    reference_ids = tuple(f"reference-{index:03d}" for index in range(80))
    held_out_null_ids = tuple(f"null-{index:03d}" for index in range(40))
    sustained_ids = tuple(f"sustained-{index}" for index in range(5))
    transient_id = "transient"
    irregular_id = "irregular"
    missing_id = "missing"
    all_ids = (
        *reference_ids,
        *held_out_null_ids,
        *sustained_ids,
        transient_id,
        irregular_id,
        missing_id,
    )
    subjects = tuple(
        Subject(
            subject_id=subject_id,
            cohort=("reference" if subject_id in reference_ids else "candidate"),
        )
        for subject_id in all_ids
    )
    covariance = np.array([[1.0, 0.65], [0.65, 1.0]])
    observations: list[Observation] = []
    for subject_id in all_ids:
        value: NDArray[np.float64] = np.zeros(2)
        values = [value.copy()]
        for transition in range(3):
            increment = random.multivariate_normal([0, 0], covariance)
            if subject_id in sustained_ids:
                increment += np.array([3.0, 3.0])
            elif subject_id == transient_id:
                if transition == 0:
                    increment += np.array([8.0, 8.0])
                elif transition == 1:
                    increment += np.array([-8.0, -8.0])
            value = value + increment
            values.append(value.copy())
        for visit_index, visit_values in enumerate(values):
            if subject_id == missing_id and visit_index > 0:
                continue
            if subject_id == irregular_id and visit_index == 1:
                continue
            for feature, measurement in zip(FEATURES, visit_values, strict=True):
                observations.append(
                    Observation(
                        subject_id=subject_id,
                        timestamp=START + timedelta(days=365 * visit_index),
                        modality=feature.modality or Modality.CLINICAL,
                        feature=feature.feature,
                        value=float(measurement),
                        unit="normalized",
                    )
                )
    return (
        Study(study_id="sequential", subjects=subjects, observations=tuple(observations)),
        reference_ids,
        (*held_out_null_ids, *sustained_ids, transient_id, irregular_id, missing_id),
    )


def test_sequential_detector_finds_sustained_and_transient_responses() -> None:
    study, reference_ids, candidate_ids = _synthetic_study()
    detector = SequentialTreatmentResponseDetector(
        SequentialDetectionConfig(
            features=FEATURES,
            covariance_shrinkage=0.1,
            false_alarm_rate=0.05,
            minimum_reference_subjects=50,
            persistence_crossings=2,
        )
    ).fit(study, visits=VISITS, reference_subject_ids=reference_ids)

    report = detector.score(study, visits=VISITS, subject_ids=candidate_ids)

    sustained = [item for item in report.results if item.subject_id.startswith("sustained")]
    held_out_nulls = [item for item in report.results if item.subject_id.startswith("null")]
    transient = next(item for item in report.results if item.subject_id == "transient")
    irregular = next(item for item in report.results if item.subject_id == "irregular")
    assert all(item.detected and item.persistent for item in sustained)
    assert sum(item.detected for item in held_out_nulls) <= 8
    assert transient.detected
    assert transient.transient
    assert not transient.persistent
    assert all(not (item.transient and item.persistent) for item in report.results)
    assert irregular.points[0].elapsed_years == pytest.approx(2.0, rel=0.01)
    assert irregular.missing_visit_ids == ("year-1",)
    assert report.excluded_subject_ids == ("missing",)
    assert any(
        item.subject_id == "missing" and item.missing_visit_ids == ("year-1", "year-2", "year-3")
        for item in report.exclusions
    )
    assert report.model.reference_transitions == 240
    assert report.model.feature_units == ("normalized", "normalized")
    assert not report.results_frame().empty
    assert not report.trajectory_frame().empty

    forged = report.model_dump(mode="python")
    forged["results"][0]["peak_cumulative_score"] += 1
    with pytest.raises(ValidationError, match="peak does not match"):
        SequentialDetectionReport.model_validate(forged)


def test_sequential_detector_requires_fit_and_enough_visits() -> None:
    study, reference_ids, candidate_ids = _synthetic_study()
    detector = SequentialTreatmentResponseDetector(
        SequentialDetectionConfig(features=FEATURES, minimum_reference_subjects=50)
    )
    with pytest.raises(RuntimeError, match="fit"):
        detector.score(study, visits=VISITS, subject_ids=candidate_ids)
    with pytest.raises(ValueError, match="at least three"):
        detector.fit(study, visits=VISITS[:2], reference_subject_ids=reference_ids)


def test_sequential_configuration_rejects_duplicate_features() -> None:
    with pytest.raises(ValidationError, match="cannot overlap"):
        SequentialDetectionConfig(features=(FEATURES[0], FEATURES[0]))
    with pytest.raises(ValidationError, match="cannot overlap"):
        SequentialDetectionConfig(
            features=(
                VisitFeature(feature="activity"),
                VisitFeature(feature="activity", modality=Modality.WEARABLE),
            )
        )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        SubjectSequentialDetection(
            subject_id="invalid",
            points=(),
            detected=True,
            persistent=True,
            transient=True,
            peak_cumulative_score=1,
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_sequential_results_reject_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValidationError, match="finite number"):
        SequentialDetectionPoint(
            from_visit_id="baseline",
            to_visit_id="follow-up",
            elapsed_years=value,
            interval_score=1,
            cumulative_score=1,
            empirical_tail_probability=0.5,
            threshold_crossed=False,
        )


def test_sequential_detector_uses_observed_timestamps_and_rejects_reference_overlap() -> None:
    study, reference_ids, _ = _synthetic_study()
    wide_visits = tuple(
        visit.model_copy(
            update={
                "window_before": timedelta(days=45),
                "window_after": timedelta(days=45),
            }
        )
        for visit in VISITS
    )
    shifted = study.model_copy(
        update={
            "observations": tuple(
                row.model_copy(update={"timestamp": row.timestamp + timedelta(days=20)})
                if row.subject_id == "irregular" and row.timestamp == START + timedelta(days=730)
                else row
                for row in study.observations
            )
        }
    )
    detector = SequentialTreatmentResponseDetector(
        SequentialDetectionConfig(features=FEATURES, minimum_reference_subjects=50)
    ).fit(shifted, visits=wide_visits, reference_subject_ids=reference_ids)

    report = detector.score(shifted, visits=wide_visits, subject_ids=("irregular",))
    irregular = report.results[0]
    assert irregular.points[0].to_observed_at == START + timedelta(days=750)
    assert irregular.points[0].elapsed_years == pytest.approx(750 / 365.2425)
    assert irregular.points[0].selected_observation_indices

    with pytest.raises(ValueError, match="overlap fitted reference"):
        detector.score(shifted, visits=wide_visits, subject_ids=(reference_ids[0],))


def test_reference_drift_is_weighted_by_observed_duration() -> None:
    subjects = tuple(Subject(subject_id=f"r{index}", cohort="reference") for index in range(5))
    observations = tuple(
        Observation(
            subject_id=subject.subject_id,
            timestamp=START + timedelta(days=day),
            modality=feature.modality or Modality.CLINICAL,
            feature=feature.feature,
            value=value,
            unit="normalized",
        )
        for subject in subjects
        for day, values in ((0, (0.0, 0.0)), (365, (10.0, 4.0)), (1095, (10.0, 6.0)))
        for feature, value in zip(FEATURES, values, strict=True)
    )
    visits = (
        VISITS[0],
        VISITS[1],
        ExpectedVisit(
            visit_id="year-3",
            scheduled_at=START + timedelta(days=1095),
            required_features=FEATURES,
        ),
    )
    detector = SequentialTreatmentResponseDetector(
        SequentialDetectionConfig(features=FEATURES, minimum_reference_subjects=5)
    ).fit(
        Study(study_id="duration-weighted", subjects=subjects, observations=observations),
        visits=visits,
        reference_subject_ids=tuple(subject.subject_id for subject in subjects),
    )

    assert detector.model_ is not None
    total_years = 1095 / 365.2425
    assert detector.model_.mean_change_per_year == pytest.approx(
        (10 / total_years, 6 / total_years)
    )


def test_sequential_model_is_deterministic_serializable_and_reconstructable() -> None:
    study, reference_ids, candidate_ids = _synthetic_study()
    config = SequentialDetectionConfig(features=FEATURES, minimum_reference_subjects=50)
    first = SequentialTreatmentResponseDetector(config).fit(
        study,
        visits=VISITS,
        reference_subject_ids=reference_ids,
    )
    second = SequentialTreatmentResponseDetector(config).fit(
        study,
        visits=VISITS,
        reference_subject_ids=tuple(reversed(reference_ids)),
    )

    assert first.model_ is not None
    assert second.model_ is not None
    assert first.model_ == second.model_
    assert first.model_.ordered_visits == VISITS
    assert first.model_.resolved_channels
    assert first.model_.reference_subject_ids == tuple(sorted(reference_ids))
    assert len(first.model_.reference_input_artifact_hash or "") == 64
    assert len(first.model_.model_artifact_hash or "") == 64
    assert first.model_.threshold_quantile_method == "higher"

    restored_model = SequentialDetectionModel.model_validate_json(first.model_.model_dump_json())
    restored = SequentialTreatmentResponseDetector.from_model(restored_model)
    selected = candidate_ids[:4]
    assert restored.score(study, visits=VISITS, subject_ids=selected) == first.score(
        study,
        visits=VISITS,
        subject_ids=selected,
    )


def test_serialized_sequential_model_rejects_tampered_fields() -> None:
    study, reference_ids, _ = _synthetic_study()
    detector = SequentialTreatmentResponseDetector(
        SequentialDetectionConfig(features=FEATURES, minimum_reference_subjects=50)
    ).fit(study, visits=VISITS, reference_subject_ids=reference_ids)
    assert detector.model_ is not None
    original = detector.model_.model_dump(mode="json")

    threshold = copy.deepcopy(original)
    threshold["maximum_cumulative_score_threshold"] = (
        float(threshold["maximum_cumulative_score_threshold"]) + 1.0
    )

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
            SequentialDetectionModel.model_validate(payload)
        with pytest.raises(ValidationError, match=error):
            SequentialDetectionModel.model_validate_json(json.dumps(payload))


def test_sequential_from_model_revalidates_model_copy_updates() -> None:
    study, reference_ids, _ = _synthetic_study()
    detector = SequentialTreatmentResponseDetector(
        SequentialDetectionConfig(features=FEATURES, minimum_reference_subjects=50)
    ).fit(study, visits=VISITS, reference_subject_ids=reference_ids)
    assert detector.model_ is not None
    tampered = detector.model_.model_copy(
        update={
            "maximum_cumulative_score_threshold": (
                detector.model_.maximum_cumulative_score_threshold + 1.0
            )
        }
    )

    with pytest.raises(ValidationError, match="threshold"):
        SequentialTreatmentResponseDetector.from_model(tampered)


def test_sequential_detector_rejects_schedule_drift_and_reference_relabeling() -> None:
    study, reference_ids, candidate_ids = _synthetic_study()
    detector = SequentialTreatmentResponseDetector(
        SequentialDetectionConfig(features=FEATURES, minimum_reference_subjects=50)
    ).fit(study, visits=VISITS, reference_subject_ids=reference_ids)

    altered_visits = (
        VISITS[0],
        VISITS[1].model_copy(update={"window_after": timedelta(days=1)}),
        *VISITS[2:],
    )
    with pytest.raises(ValueError, match="visit definitions"):
        detector.score(study, visits=altered_visits, subject_ids=(candidate_ids[0],))

    reference_id = reference_ids[0]
    candidate_id = candidate_ids[0]
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
        detector.score(relabeled, visits=VISITS, subject_ids=(candidate_ids[1],))
