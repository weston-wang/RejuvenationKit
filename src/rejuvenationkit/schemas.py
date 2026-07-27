"""Canonical data contracts shared by all RejuvenationKit phases."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Modality(StrEnum):
    """Supported high-level measurement modalities."""

    CLINICAL = "clinical"
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
    anchors: dict[str, datetime] = Field(default_factory=dict)
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_timezone_aware_anchors(self) -> Subject:
        """Reject ambiguous subject event anchors."""
        ambiguous = [
            name
            for name, timestamp in self.anchors.items()
            if timestamp.tzinfo is None or timestamp.utcoffset() is None
        ]
        if ambiguous:
            raise ValueError(f"subject anchors must be timezone-aware: {sorted(ambiguous)}")
        return self


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

    @model_validator(mode="after")
    def require_timezone(self) -> Observation:
        """Reject ambiguous timestamps."""
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return self


class Study(BaseModel):
    """Validated collection of subjects and observations."""

    model_config = ConfigDict(frozen=True)

    study_id: str = Field(min_length=1)
    subjects: tuple[Subject, ...]
    observations: tuple[Observation, ...]
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> Study:
        """Ensure identifiers are unique and every observation has a subject."""
        ids = [subject.subject_id for subject in self.subjects]
        if len(ids) != len(set(ids)):
            raise ValueError("subject_id values must be unique")
        unknown = {row.subject_id for row in self.observations}.difference(ids)
        if unknown:
            raise ValueError(f"observations reference unknown subjects: {sorted(unknown)}")
        return self
