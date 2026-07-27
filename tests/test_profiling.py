from datetime import UTC, datetime, timedelta

import pytest

from rejuvenationkit.profiling import StudyProfiler
from rejuvenationkit.qc import ExpectedVisit, QCConfig, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject

START = datetime(2026, 1, 1, tzinfo=UTC)


def _observation(subject_id: str, day: int, feature: str) -> Observation:
    return Observation(
        subject_id=subject_id,
        timestamp=START + timedelta(days=day),
        modality=Modality.CLINICAL,
        feature=feature,
        value=1,
        unit="value",
    )


def test_profile_quantifies_coverage_retention_and_paired_readiness() -> None:
    study = Study(
        study_id="profile",
        subjects=(
            Subject(subject_id="t1", cohort="treated"),
            Subject(subject_id="t2", cohort="treated"),
            Subject(subject_id="c1", cohort="control"),
        ),
        observations=(
            _observation("t1", 0, "albumin"),
            _observation("t2", 0, "albumin"),
            _observation("c1", 0, "albumin"),
            _observation("t1", 30, "albumin"),
            _observation("c1", 30, "albumin"),
        ),
    )
    requirement = (VisitFeature(feature="albumin", modality=Modality.CLINICAL),)
    config = QCConfig(
        expected_visits=(
            ExpectedVisit(visit_id="baseline", scheduled_at=START, required_features=requirement),
            ExpectedVisit(
                visit_id="month-1",
                scheduled_at=START + timedelta(days=30),
                required_features=requirement,
            ),
        )
    )

    profile = StudyProfiler(config).profile(study)

    all_follow_up = next(
        row for row in profile.visit_coverage if row.visit_id == "month-1" and row.cohort == "all"
    )
    assert all_follow_up.observed_subjects == 2
    assert all_follow_up.coverage_fraction == 2 / 3

    treated_retention = next(row for row in profile.visit_retention if row.cohort == "treated")
    assert treated_retention.from_complete_subjects == 2
    assert treated_retention.retained_subjects == 1
    assert treated_retention.retention_fraction == 0.5

    all_paired = next(row for row in profile.paired_readiness if row.cohort == "all")
    assert all_paired.paired_subjects == 2
    assert all_paired.paired_fraction == 2 / 3


def test_profile_handles_missing_relative_anchor_and_empty_tables() -> None:
    study = Study(
        study_id="empty",
        subjects=(Subject(subject_id="s1", cohort="test"),),
        observations=(),
    )
    config = QCConfig(
        expected_visits=(
            ExpectedVisit(
                visit_id="relative",
                anchor_id="first_dose",
                required_features=(VisitFeature(feature="albumin"),),
            ),
        )
    )

    profile = StudyProfiler(config).profile(study)

    assert profile.visit_coverage[0].coverage_fraction == 0
    assert profile.visit_retention == ()
    assert profile.paired_readiness == ()
    assert not profile.coverage_frame().empty
    assert profile.retention_frame().empty
    assert profile.paired_readiness_frame().empty
    assert profile.distributions_frame().iloc[0]["subjects"] == 0
    assert profile.attrition_bias_frame().empty


def test_profile_summarizes_distributions_and_robust_outliers() -> None:
    subjects = tuple(Subject(subject_id=f"s{index}", cohort="test") for index in range(1, 5))
    study = Study(
        study_id="outliers",
        subjects=subjects,
        observations=tuple(
            _observation(subject.subject_id, 0, "albumin").model_copy(update={"value": value})
            for subject, value in zip(subjects, (1.0, 1.0, 1.0, 10.0), strict=True)
        ),
    )
    config = QCConfig(
        expected_visits=(
            ExpectedVisit(
                visit_id="baseline",
                scheduled_at=START,
                required_features=(VisitFeature(feature="albumin", modality=Modality.CLINICAL),),
            ),
        )
    )

    profile = StudyProfiler(config).profile(study)
    distribution = next(row for row in profile.feature_distributions if row.cohort == "all")

    assert distribution.subjects == 4
    assert distribution.median == 1
    assert distribution.outlier_subject_ids == ("s4",)
    frame = profile.distributions_frame()
    assert frame.loc[frame["cohort"] == "all", "outlier_count"].iloc[0] == 1


def test_profile_quantifies_baseline_attrition_bias() -> None:
    subjects = tuple(Subject(subject_id=f"s{index}", cohort="test") for index in range(1, 5))
    baseline = tuple(
        _observation(subject.subject_id, 0, "albumin").model_copy(update={"value": value})
        for subject, value in zip(subjects, (1.0, 2.0, 9.0, 10.0), strict=True)
    )
    follow_up = (
        _observation("s1", 30, "albumin"),
        _observation("s2", 30, "albumin"),
    )
    study = Study(study_id="attrition", subjects=subjects, observations=(*baseline, *follow_up))
    requirement = (VisitFeature(feature="albumin", modality=Modality.CLINICAL),)
    config = QCConfig(
        expected_visits=(
            ExpectedVisit(visit_id="baseline", scheduled_at=START, required_features=requirement),
            ExpectedVisit(
                visit_id="follow-up",
                scheduled_at=START + timedelta(days=30),
                required_features=requirement,
            ),
        )
    )

    profile = StudyProfiler(config).profile(study)
    bias = next(row for row in profile.attrition_bias if row.cohort == "all")

    assert bias.retained_subjects == 2
    assert bias.attrited_subjects == 2
    assert bias.retained_baseline_mean == 1.5
    assert bias.attrited_baseline_mean == 9.5
    assert bias.standardized_mean_difference == pytest.approx(-11.313708)


def test_profiler_rejects_negative_outlier_multiplier() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        StudyProfiler(QCConfig(), outlier_iqr_multiplier=-1)
