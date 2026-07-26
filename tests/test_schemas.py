from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rejuvenationkit.schemas import Modality, Observation, Study, Subject


def observation(subject_id: str = "s1") -> Observation:
    return Observation(
        subject_id=subject_id,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        modality=Modality.CLINICAL,
        feature="body_mass",
        value=30.0,
        unit="g",
    )


def test_study_accepts_declared_subject() -> None:
    study = Study(
        study_id="study",
        subjects=(Subject(subject_id="s1", cohort="control"),),
        observations=(observation(),),
    )
    assert study.observations[0].subject_id == "s1"


def test_study_rejects_unknown_subject() -> None:
    with pytest.raises(ValidationError, match="unknown subjects"):
        Study(
            study_id="study",
            subjects=(Subject(subject_id="s1", cohort="control"),),
            observations=(observation("missing"),),
        )


def test_observation_requires_timezone() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Observation(
            subject_id="s1",
            timestamp=datetime(2026, 1, 1),
            modality=Modality.CLINICAL,
            feature="body_mass",
            value=30.0,
            unit="g",
        )
