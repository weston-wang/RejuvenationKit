"""Subject-level endpoint contracts shared by state and combination analyses."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from math import isfinite
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from rejuvenationkit.evidence import Estimand
from rejuvenationkit.schemas import study_artifact_hash as study_artifact_hash


class EndpointExclusion(BaseModel):
    """A subject excluded while constructing one endpoint batch."""

    model_config = ConfigDict(frozen=True)

    subject_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    detail: str | None = None


class SubjectEndpoint(BaseModel):
    """Exactly one analysis endpoint for one independent biological subject.

    ``estimate`` is already on the declared estimand scale. For an ANCOVA endpoint it
    is normally the follow-up value and ``baseline_estimate`` supplies the prespecified
    baseline covariate. For a change-score endpoint it is normally follow-up minus
    baseline; ``baseline_estimate`` may still be retained for auditability.
    """

    model_config = ConfigDict(frozen=True)

    subject_id: str = Field(min_length=1)
    estimand: Estimand
    estimate: float
    standard_error: float | None = Field(default=None, gt=0)
    endpoint_timestamp: datetime
    baseline_timestamp: datetime | None = None
    baseline_estimate: float | None = None
    provenance_id: str = Field(min_length=1)
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        """Reject ambiguous time order, non-finite values, and duplicate flags."""
        if self.endpoint_timestamp.tzinfo is None or self.endpoint_timestamp.utcoffset() is None:
            raise ValueError("endpoint_timestamp must be timezone-aware")
        if self.baseline_timestamp is not None:
            if (
                self.baseline_timestamp.tzinfo is None
                or self.baseline_timestamp.utcoffset() is None
            ):
                raise ValueError("baseline_timestamp must be timezone-aware")
            if self.baseline_timestamp >= self.endpoint_timestamp:
                raise ValueError("baseline_timestamp must precede endpoint_timestamp")
        numeric = (self.estimate, self.standard_error, self.baseline_estimate)
        if any(value is not None and not isfinite(value) for value in numeric):
            raise ValueError("endpoint numeric values must be finite")
        if len(set(self.quality_flags)) != len(self.quality_flags):
            raise ValueError("quality_flags must be unique")
        return self


class SubjectEndpointBatch(BaseModel):
    """One estimand aligned to at most one endpoint per independent subject."""

    model_config = ConfigDict(frozen=True)

    study_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    estimand: Estimand
    endpoints: tuple[SubjectEndpoint, ...] = Field(min_length=1)
    excluded: tuple[EndpointExclusion, ...] = ()
    source_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        """Bind every endpoint to one estimand and enforce unique subject units."""
        endpoint_ids = [item.subject_id for item in self.endpoints]
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("endpoint subjects must be unique")
        if any(item.estimand != self.estimand for item in self.endpoints):
            raise ValueError("every endpoint must share the batch estimand")
        excluded_ids = [item.subject_id for item in self.excluded]
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError("excluded subjects must be unique")
        overlap = set(endpoint_ids).intersection(excluded_ids)
        if overlap:
            raise ValueError(f"subjects cannot be both endpoints and excluded: {sorted(overlap)}")
        object.__setattr__(
            self,
            "endpoints",
            tuple(sorted(self.endpoints, key=lambda item: item.subject_id)),
        )
        object.__setattr__(
            self,
            "excluded",
            tuple(sorted(self.excluded, key=lambda item: (item.subject_id, item.reason))),
        )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_hash(self) -> str:
        """Return a canonical identity for endpoints, exclusions, and source binding."""
        payload = {
            "schema": "subject-endpoint-batch/v1",
            "study_id": self.study_id,
            "batch_id": self.batch_id,
            "estimand": self.estimand.model_dump(mode="json"),
            "endpoints": [item.model_dump(mode="json") for item in self.endpoints],
            "excluded": [item.model_dump(mode="json") for item in self.excluded],
            "source_artifact_hash": self.source_artifact_hash,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def by_subject(self) -> dict[str, SubjectEndpoint]:
        """Return endpoints keyed by the independent subject identifier."""
        return {item.subject_id: item for item in self.endpoints}
