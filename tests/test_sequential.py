from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject
from rejuvenationkit.sequential import (
    SequentialDetectionConfig,
    SequentialTreatmentResponseDetector,
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
        value = np.zeros(2)
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
    assert irregular.points[0].elapsed_years == pytest.approx(2.0, rel=0.01)
    assert report.excluded_subject_ids == ("missing",)
    assert report.model.reference_transitions == 240
    assert not report.results_frame().empty
    assert not report.trajectory_frame().empty


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
    with pytest.raises(ValidationError, match="unique"):
        SequentialDetectionConfig(features=(FEATURES[0], FEATURES[0]))
