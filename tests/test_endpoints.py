from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from rejuvenationkit.endpoints import (
    EndpointExclusion,
    SubjectEndpoint,
    SubjectEndpointBatch,
)
from rejuvenationkit.evidence import EffectDirection, Estimand

START = datetime(2026, 1, 1, tzinfo=UTC)
ESTIMAND = Estimand(
    name="frailty_change",
    unit="score",
    direction=EffectDirection.LOWER_IS_BETTER,
    population="factorial-study",
    time_contrast="month-6-minus-baseline",
)


def endpoint(subject_id: str, value: float = -1.0) -> SubjectEndpoint:
    return SubjectEndpoint(
        subject_id=subject_id,
        estimand=ESTIMAND,
        estimate=value,
        standard_error=0.2,
        baseline_timestamp=START,
        endpoint_timestamp=START + timedelta(days=180),
        baseline_estimate=5.0,
        provenance_id=f"endpoint:{subject_id}",
    )


def test_endpoint_batch_is_subject_unique_sorted_and_content_bound() -> None:
    batch = SubjectEndpointBatch(
        study_id="factorial",
        batch_id="frailty-v1",
        estimand=ESTIMAND,
        endpoints=(endpoint("dog-2"), endpoint("dog-1")),
        excluded=(EndpointExclusion(subject_id="dog-3", reason="missing_follow_up"),),
        source_artifact_hash="a" * 64,
    )

    assert tuple(item.subject_id for item in batch.endpoints) == ("dog-1", "dog-2")
    assert batch.by_subject()["dog-1"].estimate == -1.0
    assert len(batch.artifact_hash) == 64
    assert batch == SubjectEndpointBatch.model_validate_json(batch.model_dump_json())

    changed = batch.model_copy(update={"source_artifact_hash": "b" * 64})
    assert changed.artifact_hash != batch.artifact_hash


def test_endpoint_contract_rejects_ambiguous_time_and_nonfinite_values() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        endpoint("dog").model_copy(
            update={"endpoint_timestamp": datetime(2026, 1, 2)}
        ).model_validate(
            endpoint("dog")
            .model_copy(update={"endpoint_timestamp": datetime(2026, 1, 2)})
            .model_dump()
        )
    with pytest.raises(ValidationError, match="precede"):
        SubjectEndpoint(
            subject_id="dog",
            estimand=ESTIMAND,
            estimate=1.0,
            baseline_timestamp=START,
            endpoint_timestamp=START,
            provenance_id="endpoint:dog",
        )
    with pytest.raises(ValidationError, match="finite"):
        SubjectEndpoint(
            subject_id="dog",
            estimand=ESTIMAND,
            estimate=float("nan"),
            endpoint_timestamp=START,
            provenance_id="endpoint:dog",
        )


def test_endpoint_batch_rejects_duplicates_mixed_estimands_and_overlap() -> None:
    common = {
        "study_id": "factorial",
        "batch_id": "frailty-v1",
        "estimand": ESTIMAND,
        "source_artifact_hash": "a" * 64,
    }
    with pytest.raises(ValidationError, match="unique"):
        SubjectEndpointBatch(endpoints=(endpoint("dog"), endpoint("dog")), **common)

    other = endpoint("dog").model_copy(
        update={"estimand": ESTIMAND.model_copy(update={"name": "other"})}
    )
    with pytest.raises(ValidationError, match="share"):
        SubjectEndpointBatch(endpoints=(other,), **common)

    with pytest.raises(ValidationError, match="both"):
        SubjectEndpointBatch(
            endpoints=(endpoint("dog"),),
            excluded=(EndpointExclusion(subject_id="dog", reason="missing"),),
            **common,
        )
