from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from rejuvenationkit.detection import ChangeDetectionConfig, MultivariateChangeDetector
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


def test_detector_validates_configuration_and_reference_size() -> None:
    with pytest.raises(ValidationError, match="at least 2"):
        ChangeDetectionConfig(features=(FEATURES[0],))
    with pytest.raises(ValidationError, match="unique"):
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
