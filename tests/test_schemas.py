from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rejuvenationkit.schemas import Modality, Observation, Study, Subject, study_artifact_hash


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


def test_observation_requires_finite_standard_error() -> None:
    with pytest.raises(ValidationError, match="standard_error must be finite"):
        Observation(
            **{
                **observation().model_dump(),
                "standard_error": float("inf"),
            }
        )


def test_subject_anchors_require_timezone() -> None:
    subject = Subject(
        subject_id="s1",
        cohort="treated",
        anchors={"first_dose": datetime(2026, 1, 1, tzinfo=UTC)},
    )
    assert subject.anchors["first_dose"].tzinfo is UTC

    with pytest.raises(ValidationError, match="anchors must be timezone-aware"):
        Subject(
            subject_id="s1",
            cohort="treated",
            anchors={"first_dose": datetime(2026, 1, 1)},
        )


def test_nested_study_metadata_is_defensively_copied_and_immutable() -> None:
    anchors = {"first_dose": datetime(2026, 1, 1, tzinfo=UTC)}
    subject_attributes = {"site": "north"}
    observation_attributes = {"plate": "p1"}
    metadata = {"protocol": "v1"}
    subject = Subject(
        subject_id="s1",
        cohort="treated",
        anchors=anchors,
        attributes=subject_attributes,
    )
    row = observation().model_copy(update={"attributes": observation_attributes})
    row = Observation.model_validate(row.model_dump())
    study = Study(study_id="study", subjects=(subject,), observations=(row,), metadata=metadata)

    anchors["first_dose"] = datetime(2027, 1, 1, tzinfo=UTC)
    subject_attributes["site"] = "south"
    observation_attributes["plate"] = "p2"
    metadata["protocol"] = "v2"

    assert study.subjects[0].anchors["first_dose"].year == 2026
    assert study.subjects[0].attributes["site"] == "north"
    assert study.observations[0].attributes["plate"] == "p1"
    assert study.metadata["protocol"] == "v1"
    with pytest.raises(TypeError):
        study.metadata["protocol"] = "changed"  # type: ignore[index]
    assert Study.model_validate_json(study.model_dump_json()) == study


def test_subject_rejects_duplicate_interventions_and_nonfinite_metadata() -> None:
    with pytest.raises(ValidationError, match="interventions must be unique"):
        Subject(
            subject_id="s1",
            cohort="treated",
            interventions=("rapamycin", "rapamycin"),
        )
    with pytest.raises(ValidationError, match="must be finite"):
        Study(
            study_id="study",
            subjects=(Subject(subject_id="s1", cohort="control"),),
            observations=(observation(),),
            metadata={"bad": float("nan")},
        )


def test_study_artifact_hash_is_order_invariant_and_content_sensitive() -> None:
    """Logical study identity ignores tuple order but binds scientific content."""
    first = Subject(
        subject_id="s1",
        cohort="treated",
        interventions=("rapamycin", "senolytic"),
    )
    second = Subject(subject_id="s2", cohort="control")
    earlier = observation("s1")
    later = Observation(
        subject_id="s2",
        timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        modality=Modality.CLINICAL,
        feature="body_mass",
        value=28.0,
        unit="g",
    )
    study = Study(
        study_id="hash-study",
        subjects=(first, second),
        observations=(earlier, later),
        metadata={"protocol": "v1"},
    )
    reordered = Study(
        study_id=study.study_id,
        subjects=(
            second,
            first.model_copy(update={"interventions": tuple(reversed(first.interventions))}),
        ),
        observations=(later, earlier),
        metadata={"protocol": "v1"},
    )

    assert study_artifact_hash(study) == study_artifact_hash(reordered)
    changed = study.model_copy(update={"metadata": {"protocol": "v2"}})
    assert study_artifact_hash(study) != study_artifact_hash(Study.model_validate(changed))
