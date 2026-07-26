"""Create and inspect a minimal validated study."""

from datetime import UTC, datetime

from rejuvenationkit.schemas import Modality, Observation, Study, Subject

study = Study(
    study_id="demo-study",
    subjects=(Subject(subject_id="mouse-001", cohort="treated", interventions=("therapy-a",)),),
    observations=(
        Observation(
            subject_id="mouse-001",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            modality=Modality.CLINICAL,
            feature="body_mass",
            value=31.2,
            unit="g",
            batch_id="batch-01",
        ),
    ),
)

print(study.model_dump_json(indent=2))
