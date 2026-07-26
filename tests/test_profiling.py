from datetime import UTC, datetime, timedelta

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
