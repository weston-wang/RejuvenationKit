"""Run the Phase 1 QC pipeline on a small synthetic study."""

from datetime import UTC, datetime

from rejuvenationkit import (
    BaselineLongitudinalQC,
    FeatureRule,
    Modality,
    Observation,
    QCConfig,
    Study,
    Subject,
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
            subject_id="mouse-002",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            modality=Modality.CLINICAL,
            feature="body_mass",
            value=31.2,
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
    )
)

report = BaselineLongitudinalQC(config).run(study)
print(report.summary())
for finding in report.findings:
    print(f"[{finding.severity}] {finding.code}: {finding.message}")
