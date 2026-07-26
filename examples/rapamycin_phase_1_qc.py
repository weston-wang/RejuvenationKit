"""Check staggered visits and treatment/batch balance in a rapamycin study."""

from datetime import UTC, datetime, timedelta

from rejuvenationkit import (
    BaselineLongitudinalQC,
    ExpectedVisit,
    Modality,
    Observation,
    QCConfig,
    Study,
    Subject,
    VisitFeature,
)

start = datetime(2026, 1, 1, tzinfo=UTC)
subjects = (
    Subject(
        subject_id="dog-r1",
        cohort="treated",
        interventions=("rapamycin",),
        anchors={"first_dose": start},
    ),
    Subject(
        subject_id="dog-r2",
        cohort="treated",
        interventions=("rapamycin",),
        anchors={"first_dose": start + timedelta(days=7)},
    ),
    Subject(subject_id="dog-c1", cohort="control", anchors={"first_dose": start}),
    Subject(
        subject_id="dog-c2",
        cohort="control",
        anchors={"first_dose": start + timedelta(days=7)},
    ),
)

# Each treatment group is represented in both lab batches.
batches = {"dog-r1": "A", "dog-r2": "B", "dog-c1": "A", "dog-c2": "B"}
observations = tuple(
    Observation(
        subject_id=subject.subject_id,
        timestamp=subject.anchors["first_dose"] + timedelta(days=28),
        modality=Modality.CLINICAL,
        feature="body_mass",
        value=30.0,
        unit="kg",
        batch_id=batches[subject.subject_id],
    )
    for subject in subjects
)

study = Study(
    study_id="rapamycin-demo",
    subjects=subjects,
    observations=observations,
)
config = QCConfig(
    expected_visits=(
        ExpectedVisit(
            visit_id="month-1-after-first-dose",
            anchor_id="first_dose",
            offset=timedelta(days=28),
            window_before=timedelta(days=3),
            window_after=timedelta(days=3),
            required_features=(VisitFeature(feature="body_mass", modality=Modality.CLINICAL),),
        ),
    )
)

report = BaselineLongitudinalQC(config).run(study)
print(report.summary())
