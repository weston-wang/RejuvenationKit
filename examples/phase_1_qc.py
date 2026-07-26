"""Run Phase 1 feature and expected-visit QC on a small synthetic study."""

from datetime import UTC, datetime, timedelta

from rejuvenationkit import (
    BaselineLongitudinalQC,
    ExpectedVisit,
    FeatureRule,
    Modality,
    Observation,
    QCConfig,
    Study,
    Subject,
    VisitFeature,
)

study = Study(
    study_id="phase-1-demo",
    subjects=(
        Subject(subject_id="mouse-001", cohort="control"),
        Subject(subject_id="mouse-002", cohort="treated", interventions=("therapy-a",)),
    ),
    observations=(
        Observation(
            subject_id="mouse-001",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            modality=Modality.CLINICAL,
            feature="body_mass",
            value=30.1,
            unit="g",
        ),
        Observation(
            subject_id="mouse-001",
            timestamp=datetime(2026, 1, 29, tzinfo=UTC),
            modality=Modality.CLINICAL,
            feature="body_mass",
            value=30.3,
            unit="g",
        ),
        Observation(
            subject_id="mouse-002",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            modality=Modality.CLINICAL,
            feature="body_mass",
            value=31.2,
            unit="g",
        ),
        Observation(
            subject_id="mouse-002",
            timestamp=datetime(2026, 1, 31, tzinfo=UTC),
            modality=Modality.CLINICAL,
            feature="body_mass",
            value=30.9,
            unit="g",
        ),
    ),
)

config = QCConfig(
    feature_rules=(
        FeatureRule(
            feature="body_mass",
            modality=Modality.CLINICAL,
            expected_unit="g",
            minimum=10,
            maximum=60,
            required=True,
        ),
    ),
    expected_visits=(
        ExpectedVisit(
            visit_id="baseline",
            scheduled_at=datetime(2026, 1, 1, tzinfo=UTC),
            required_features=(VisitFeature(feature="body_mass"),),
        ),
        ExpectedVisit(
            visit_id="week-4",
            scheduled_at=datetime(2026, 1, 29, tzinfo=UTC),
            window_before=timedelta(days=1),
            window_after=timedelta(days=2),
            required_features=(VisitFeature(feature="body_mass"),),
        ),
    ),
)

report = BaselineLongitudinalQC(config).run(study)
print(report.summary())
for finding in report.findings:
    print(f"[{finding.severity}] {finding.code}: {finding.message}")
