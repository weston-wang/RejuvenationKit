"""Canonical data contracts shared by all RejuvenationKit phases."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

MetadataValue: TypeAlias = str | int | float | bool


def _freeze_metadata(
    values: Mapping[str, MetadataValue],
    *,
    field_name: str,
) -> Mapping[str, MetadataValue]:
    copied: dict[str, MetadataValue] = {}
    for key, value in values.items():
        if not key or key != key.strip():
            raise ValueError(f"{field_name} keys must be nonblank without surrounding whitespace")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"{field_name} float values must be finite")
        copied[key] = value
    return MappingProxyType(copied)


def _freeze_anchors(values: Mapping[str, datetime]) -> Mapping[str, datetime]:
    copied: dict[str, datetime] = {}
    for key, value in values.items():
        if not key or key != key.strip():
            raise ValueError("anchor keys must be nonblank without surrounding whitespace")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"subject anchors must be timezone-aware: {[key]}")
        copied[key] = value
    return MappingProxyType(copied)


class Modality(StrEnum):
    """Supported high-level measurement modalities."""

    CLINICAL = "clinical"
    GENOMICS = "genomics"
    HISTOLOGY = "histology"
    IMAGING = "imaging"
    METABOLOMICS = "metabolomics"
    METHYLATION = "methylation"
    PROTEOMICS = "proteomics"
    TRANSCRIPTOMICS = "transcriptomics"
    WEARABLE = "wearable"


class Subject(BaseModel):
    """A biological subject and its study-level assignment."""

    model_config = ConfigDict(frozen=True)

    subject_id: str = Field(min_length=1)
    cohort: str = Field(min_length=1)
    interventions: tuple[str, ...] = ()
    anchors: Mapping[str, datetime] = Field(default_factory=dict)
    attributes: Mapping[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_timezone_aware_anchors(self) -> Subject:
        """Reject ambiguous subject event anchors."""
        if len(set(self.interventions)) != len(self.interventions):
            raise ValueError("subject interventions must be unique")
        if any(not item or item != item.strip() for item in self.interventions):
            raise ValueError(
                "subject interventions must be nonblank without surrounding whitespace"
            )
        object.__setattr__(self, "anchors", _freeze_anchors(self.anchors))
        object.__setattr__(
            self,
            "attributes",
            _freeze_metadata(self.attributes, field_name="subject attributes"),
        )
        return self

    @field_serializer("anchors")
    def serialize_anchors(self, value: Mapping[str, datetime]) -> dict[str, datetime]:
        """Serialize immutable anchors through Pydantic's datetime encoder."""
        return dict(value)

    @field_serializer("attributes")
    def serialize_attributes(self, value: Mapping[str, MetadataValue]) -> dict[str, MetadataValue]:
        """Serialize immutable subject attributes as ordinary JSON data."""
        return dict(value)


class Observation(BaseModel):
    """One numeric measurement in a longitudinal study."""

    model_config = ConfigDict(frozen=True)

    subject_id: str = Field(min_length=1)
    timestamp: datetime
    modality: Modality
    feature: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    standard_error: float | None = Field(default=None, gt=0)
    batch_id: str | None = None
    replicate_id: str | None = None
    source_uri: str | None = None
    attributes: Mapping[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_timezone(self) -> Observation:
        """Reject ambiguous timestamps and unusable uncertainty values."""
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.standard_error is not None and not isfinite(self.standard_error):
            raise ValueError("standard_error must be finite")
        object.__setattr__(
            self,
            "attributes",
            _freeze_metadata(self.attributes, field_name="observation attributes"),
        )
        return self

    @field_serializer("attributes")
    def serialize_attributes(self, value: Mapping[str, MetadataValue]) -> dict[str, MetadataValue]:
        """Serialize immutable observation attributes as ordinary JSON data."""
        return dict(value)


class Study(BaseModel):
    """Validated collection of subjects and observations."""

    model_config = ConfigDict(frozen=True)

    study_id: str = Field(min_length=1)
    subjects: tuple[Subject, ...]
    observations: tuple[Observation, ...]
    metadata: Mapping[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> Study:
        """Ensure identifiers are unique and every observation has a subject."""
        ids = [subject.subject_id for subject in self.subjects]
        if len(ids) != len(set(ids)):
            raise ValueError("subject_id values must be unique")
        unknown = {row.subject_id for row in self.observations}.difference(ids)
        if unknown:
            raise ValueError(f"observations reference unknown subjects: {sorted(unknown)}")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, field_name="study metadata"),
        )
        return self

    @field_serializer("metadata")
    def serialize_metadata(self, value: Mapping[str, MetadataValue]) -> dict[str, MetadataValue]:
        """Serialize immutable study metadata as ordinary JSON data."""
        return dict(value)


def study_artifact_hash(study: Study) -> str:
    """Return an order-invariant identity for one validated logical study."""
    subjects: list[dict[str, object]] = []
    for subject in sorted(study.subjects, key=lambda item: item.subject_id):
        payload = subject.model_dump(mode="json")
        payload["interventions"] = sorted(payload["interventions"])
        subjects.append(payload)
    observations = [item.model_dump(mode="json") for item in study.observations]
    observations.sort(
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    encoded = json.dumps(
        {
            "schema": "rejuvenationkit.study-artifact/v1",
            "study_id": study.study_id,
            "subjects": subjects,
            "observations": observations,
            "metadata": dict(study.metadata),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
