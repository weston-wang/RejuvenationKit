"""Auditable visit alignment for longitudinal measurement channels.

This module is the single boundary between raw :class:`Observation` rows and
visit-level numerical analyses.  It resolves every requested feature to one
modality, one exact unit, and one declared aggregation policy before values are
allowed into a vector.  The selected source-row indices and the effective
observed timestamp remain attached to every aggregate.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from statistics import fmean, median

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject


class AggregationPolicy(StrEnum):
    """Deterministic policy for repeated observations inside one visit window."""

    MEAN = "mean"
    MEDIAN = "median"
    CLOSEST_TO_SCHEDULE = "closest_to_schedule"


class LongitudinalChannel(BaseModel):
    """One exact analysis channel.

    Unlike :class:`VisitFeature`, a channel never uses a wildcard modality and
    always declares its unit and within-window aggregation semantics.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    feature: str = Field(min_length=1)
    modality: Modality
    unit: str = Field(min_length=1)
    aggregation_policy: AggregationPolicy = AggregationPolicy.MEAN


class LongitudinalExclusionReason(StrEnum):
    """Machine-readable reason that a requested visit-level value was unavailable."""

    VISIT_NOT_APPLICABLE = "visit_not_applicable"
    MISSING_ANCHOR = "missing_anchor"
    MISSING_OBSERVATION = "missing_observation"
    NONFINITE_OBSERVATION = "nonfinite_observation"
    AMBIGUOUS_MODALITY = "ambiguous_modality"
    MIXED_UNITS = "mixed_units"
    UNIT_MISMATCH = "unit_mismatch"
    INCOMPLETE_VISIT_VECTOR = "incomplete_visit_vector"
    INSUFFICIENT_COMPLETE_VISITS = "insufficient_complete_visits"
    NONCHRONOLOGICAL_OBSERVED_TIME = "nonchronological_observed_time"
    REUSED_OBSERVATION_ACROSS_VISITS = "reused_observation_across_visits"
    DUPLICATE_RESOLVED_CHANNEL = "duplicate_resolved_channel"


class LongitudinalExclusion(BaseModel):
    """Structured provenance for a value or vector excluded from analysis."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    reason: LongitudinalExclusionReason
    subject_id: str | None = None
    visit_id: str | None = None
    channel_index: int | None = Field(default=None, ge=0)
    feature: str | None = None
    requested_modality: Modality | None = None
    observation_indices: tuple[int, ...] = ()
    observed_modalities: tuple[Modality, ...] = ()
    observed_units: tuple[str, ...] = ()
    missing_channel_indices: tuple[int, ...] = ()
    missing_visit_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_deterministic_collections(self) -> LongitudinalExclusion:
        """Reject duplicate or non-deterministically ordered provenance fields."""
        for name, values in (
            ("observation_indices", self.observation_indices),
            ("missing_channel_indices", self.missing_channel_indices),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be unique and sorted")
        if len(self.missing_visit_ids) != len(set(self.missing_visit_ids)):
            raise ValueError("missing visit identifiers must be unique")
        return self


class VisitAlignedValue(BaseModel):
    """One visit-level aggregate with exact channel and source-row provenance."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    subject_id: str = Field(min_length=1)
    visit_id: str = Field(min_length=1)
    channel_index: int = Field(ge=0)
    channel: LongitudinalChannel
    value: float
    effective_timestamp: datetime
    scheduled_timestamp: datetime
    selected_observation_indices: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_finite_value_and_aware_times(self) -> VisitAlignedValue:
        """Reject unusable aggregates and ambiguous timestamps."""
        if not math.isfinite(self.value):
            raise ValueError("visit-aligned values must be finite")
        for name, timestamp in (
            ("effective_timestamp", self.effective_timestamp),
            ("scheduled_timestamp", self.scheduled_timestamp),
        ):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if tuple(sorted(set(self.selected_observation_indices))) != (
            self.selected_observation_indices
        ):
            raise ValueError("selected observation indices must be unique and sorted")
        return self


class LongitudinalExtraction(BaseModel):
    """All visit-aligned values and exclusions for one extraction request."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    study_id: str
    visit_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    channels: tuple[LongitudinalChannel | None, ...]
    values: tuple[VisitAlignedValue, ...] = ()
    exclusions: tuple[LongitudinalExclusion, ...] = ()

    @model_validator(mode="after")
    def require_unique_resolved_channels(self) -> LongitudinalExtraction:
        """Bind every serialized value to unique declared subject/visit/channel axes."""
        if not self.study_id:
            raise ValueError("longitudinal extraction study_id must be nonblank")
        if not self.visit_ids or len(self.visit_ids) != len(set(self.visit_ids)):
            raise ValueError(
                "longitudinal extraction visit identifiers must be nonempty and unique"
            )
        if any(not item for item in self.visit_ids):
            raise ValueError("longitudinal extraction visit identifiers must be nonblank")
        if tuple(sorted(set(self.subject_ids))) != self.subject_ids:
            raise ValueError(
                "longitudinal extraction subject identifiers must be unique and sorted"
            )
        if any(not item for item in self.subject_ids):
            raise ValueError("longitudinal extraction subject identifiers must be nonblank")
        if not self.channels:
            raise ValueError("longitudinal extraction requires at least one channel")
        exact = tuple(channel for channel in self.channels if channel is not None)
        if len(exact) != len(set(exact)):
            raise ValueError("resolved longitudinal channels must be unique")
        keys: set[tuple[str, str, int]] = set()
        uses: dict[tuple[str, int], set[tuple[str, int]]] = {}
        for value in self.values:
            if value.subject_id not in self.subject_ids or value.visit_id not in self.visit_ids:
                raise ValueError("visit-aligned value lies outside the declared extraction axes")
            if value.channel_index >= len(self.channels):
                raise ValueError("visit-aligned value channel index lies outside the declared axis")
            declared_channel = self.channels[value.channel_index]
            if declared_channel is None or value.channel != declared_channel:
                raise ValueError("visit-aligned value channel does not match the declared channel")
            key = (value.subject_id, value.visit_id, value.channel_index)
            if key in keys:
                raise ValueError("visit-aligned value keys must be unique")
            keys.add(key)
            for observation_index in value.selected_observation_indices:
                uses.setdefault(
                    (value.subject_id, observation_index),
                    set(),
                ).add((value.visit_id, value.channel_index))
        if any(len(value_keys) > 1 for value_keys in uses.values()):
            raise ValueError("serialized extraction reuses an observation across values")
        visit_positions = {visit_id: index for index, visit_id in enumerate(self.visit_ids)}
        by_subject_channel: dict[tuple[str, int], list[VisitAlignedValue]] = {}
        for value in self.values:
            by_subject_channel.setdefault((value.subject_id, value.channel_index), []).append(value)
        for channel_values in by_subject_channel.values():
            ordered = sorted(channel_values, key=lambda item: visit_positions[item.visit_id])
            if any(
                second.effective_timestamp <= first.effective_timestamp
                for first, second in pairwise(ordered)
            ):
                raise ValueError("serialized extraction has nonchronological observed visits")
        for exclusion in self.exclusions:
            if exclusion.subject_id is not None and exclusion.subject_id not in self.subject_ids:
                raise ValueError("longitudinal exclusion subject lies outside the extraction axis")
            if exclusion.visit_id is not None and exclusion.visit_id not in self.visit_ids:
                raise ValueError("longitudinal exclusion visit lies outside the extraction axis")
            if exclusion.channel_index is not None and exclusion.channel_index >= len(
                self.channels
            ):
                raise ValueError("longitudinal exclusion channel lies outside the extraction axis")
            if any(index >= len(self.channels) for index in exclusion.missing_channel_indices):
                raise ValueError("longitudinal exclusion references an unknown missing channel")
            if any(visit_id not in self.visit_ids for visit_id in exclusion.missing_visit_ids):
                raise ValueError("longitudinal exclusion references an unknown missing visit")
        return self

    def value_map(self) -> dict[tuple[str, str, int], VisitAlignedValue]:
        """Index aggregates by subject, visit, and request-channel position."""
        return {(item.subject_id, item.visit_id, item.channel_index): item for item in self.values}


class VisitAlignedVector(BaseModel):
    """A complete ordered channel vector at one observed visit."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        allow_inf_nan=False,
    )

    subject_id: str
    visit_id: str
    channels: tuple[LongitudinalChannel, ...]
    values: tuple[float, ...]
    effective_timestamp: datetime
    selected_observation_indices: tuple[int, ...]

    @model_validator(mode="after")
    def validate_vector_provenance(self) -> VisitAlignedVector:
        """Ensure the vector and its source provenance are internally consistent."""
        if len(self.channels) != len(self.values):
            raise ValueError("visit vector channels and values must have equal length")
        if not self.channels:
            raise ValueError("visit vectors require at least one channel")
        if len(self.channels) != len(set(self.channels)):
            raise ValueError("visit vector channels must be unique")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("visit vector values must be finite")
        if self.effective_timestamp.tzinfo is None or self.effective_timestamp.utcoffset() is None:
            raise ValueError("visit vector effective timestamp must be timezone-aware")
        if tuple(sorted(set(self.selected_observation_indices))) != (
            self.selected_observation_indices
        ):
            raise ValueError("selected observation indices must be unique and sorted")
        return self

    def as_array(self) -> NDArray[np.float64]:
        """Return an independent NumPy representation in channel order."""
        return np.asarray(self.values, dtype=np.float64)


class VisitVectorExtraction(BaseModel):
    """Complete visit vectors plus structured incomplete-vector exclusions."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    vectors: tuple[VisitAlignedVector, ...]
    exclusions: tuple[LongitudinalExclusion, ...]

    @model_validator(mode="after")
    def validate_vector_axes(self) -> VisitVectorExtraction:
        """Reject duplicate vector keys, channel drift, and cross-visit row reuse."""
        keys = tuple((item.subject_id, item.visit_id) for item in self.vectors)
        if len(keys) != len(set(keys)):
            raise ValueError("visit vector subject/visit keys must be unique")
        if self.vectors:
            expected_channels = self.vectors[0].channels
            if any(item.channels != expected_channels for item in self.vectors[1:]):
                raise ValueError("visit vector channels must match one exact shared axis")
        uses: dict[tuple[str, int], set[str]] = {}
        for vector in self.vectors:
            for observation_index in vector.selected_observation_indices:
                uses.setdefault((vector.subject_id, observation_index), set()).add(vector.visit_id)
        if any(len(visit_ids) > 1 for visit_ids in uses.values()):
            raise ValueError("visit vectors reuse a source observation across visits")
        return self

    def vector_map(self) -> dict[tuple[str, str], VisitAlignedVector]:
        """Index complete vectors by subject and visit."""
        return {(item.subject_id, item.visit_id): item for item in self.vectors}


class LongitudinalAlignmentError(ValueError):
    """Raised when a request cannot be resolved to exact comparable channels."""

    def __init__(self, exclusions: tuple[LongitudinalExclusion, ...]) -> None:
        """Retain structured resolution errors while exposing a concise message."""
        self.exclusions = exclusions
        reasons = ", ".join(sorted({item.reason.value for item in exclusions}))
        super().__init__(f"longitudinal channel alignment failed: {reasons}")


def extract_visit_aligned_values(
    study: Study,
    *,
    visits: tuple[ExpectedVisit, ...],
    channels: tuple[VisitFeature | LongitudinalChannel, ...],
    subject_ids: tuple[str, ...] | None = None,
    aggregation_policy: AggregationPolicy = AggregationPolicy.MEAN,
) -> LongitudinalExtraction:
    """Resolve and aggregate exact channels inside expected-visit windows.

    ``VisitFeature`` requests remain supported for compatibility.  Their
    modality and unit are resolved from the selected visit windows, and the
    request fails closed if either is ambiguous.  Pass ``LongitudinalChannel``
    to prespecify an exact modality, unit, and aggregation policy.
    """
    _validate_visits_and_channels(visits, channels)
    known_subjects = {subject.subject_id: subject for subject in study.subjects}
    if subject_ids is None:
        selected_ids = tuple(sorted(known_subjects))
    else:
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("longitudinal subject identifiers must be unique")
        unknown = set(subject_ids).difference(known_subjects)
        if unknown:
            raise ValueError(f"unknown longitudinal subjects: {sorted(unknown)}")
        selected_ids = tuple(sorted(subject_ids))

    selected_set = set(selected_ids)
    candidates = _candidate_indices(study, visits, channels, known_subjects, selected_set)
    resolved: list[LongitudinalChannel | None] = []
    resolution_errors: list[LongitudinalExclusion] = []
    for channel_index, request in enumerate(channels):
        channel, errors = _resolve_channel(
            study,
            channel_index=channel_index,
            request=request,
            candidate_indices=candidates[channel_index],
            default_policy=aggregation_policy,
        )
        resolved.append(channel)
        resolution_errors.extend(errors)
    resolution_errors.extend(
        _duplicate_resolved_channel_exclusions(
            requests=channels,
            resolved=tuple(resolved),
            candidate_indices=candidates,
        )
    )
    if resolution_errors:
        raise LongitudinalAlignmentError(tuple(resolution_errors))

    indices_by_subject_feature: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(study.observations):
        if row.subject_id in selected_set:
            indices_by_subject_feature.setdefault((row.subject_id, row.feature), []).append(index)
    values: list[VisitAlignedValue] = []
    exclusions: list[LongitudinalExclusion] = []
    for subject_id in selected_ids:
        subject = known_subjects[subject_id]
        for visit in visits:
            applicable = _visit_applies(visit, subject)
            target = visit.scheduled_for(subject) if applicable else None
            for channel_index, (request, channel) in enumerate(
                zip(channels, resolved, strict=True)
            ):
                requested_modality = request.modality
                if not applicable:
                    exclusions.append(
                        _exclusion(
                            reason=LongitudinalExclusionReason.VISIT_NOT_APPLICABLE,
                            subject=subject,
                            visit=visit,
                            channel_index=channel_index,
                            request=request,
                        )
                    )
                    continue
                if target is None:
                    exclusions.append(
                        _exclusion(
                            reason=LongitudinalExclusionReason.MISSING_ANCHOR,
                            subject=subject,
                            visit=visit,
                            channel_index=channel_index,
                            request=request,
                        )
                    )
                    continue
                window = visit.window_for(subject)
                assert window is not None
                local_indices = tuple(
                    index
                    for index in indices_by_subject_feature.get((subject_id, request.feature), [])
                    if window[0] <= study.observations[index].timestamp <= window[1]
                    and (
                        requested_modality is None
                        or study.observations[index].modality is requested_modality
                    )
                )
                if channel is None:
                    exclusions.append(
                        LongitudinalExclusion(
                            reason=(
                                LongitudinalExclusionReason.NONFINITE_OBSERVATION
                                if local_indices
                                and not any(
                                    math.isfinite(study.observations[index].value)
                                    for index in local_indices
                                )
                                else LongitudinalExclusionReason.MISSING_OBSERVATION
                            ),
                            subject_id=subject_id,
                            visit_id=visit.visit_id,
                            channel_index=channel_index,
                            feature=request.feature,
                            requested_modality=requested_modality,
                            observation_indices=local_indices,
                        )
                    )
                    continue
                exact_indices = tuple(
                    index
                    for index in local_indices
                    if study.observations[index].modality is channel.modality
                    and study.observations[index].unit == channel.unit
                )
                nonfinite_indices = tuple(
                    index
                    for index in exact_indices
                    if not math.isfinite(study.observations[index].value)
                )
                if nonfinite_indices:
                    exclusions.append(
                        LongitudinalExclusion(
                            reason=LongitudinalExclusionReason.NONFINITE_OBSERVATION,
                            subject_id=subject_id,
                            visit_id=visit.visit_id,
                            channel_index=channel_index,
                            feature=request.feature,
                            requested_modality=requested_modality,
                            observation_indices=nonfinite_indices,
                            observed_modalities=(channel.modality,),
                            observed_units=(channel.unit,),
                        )
                    )
                    continue
                if not exact_indices:
                    exclusions.append(
                        LongitudinalExclusion(
                            reason=LongitudinalExclusionReason.MISSING_OBSERVATION,
                            subject_id=subject_id,
                            visit_id=visit.visit_id,
                            channel_index=channel_index,
                            feature=request.feature,
                            requested_modality=requested_modality,
                            observation_indices=local_indices,
                        )
                    )
                    continue
                selected = _select_indices(
                    study.observations,
                    exact_indices,
                    target=target,
                    policy=channel.aggregation_policy,
                )
                value, effective_timestamp = _aggregate(
                    study.observations,
                    selected,
                    channel.aggregation_policy,
                )
                values.append(
                    VisitAlignedValue(
                        subject_id=subject_id,
                        visit_id=visit.visit_id,
                        channel_index=channel_index,
                        channel=channel,
                        value=value,
                        effective_timestamp=effective_timestamp,
                        scheduled_timestamp=target,
                        selected_observation_indices=selected,
                    )
                )
    nonoverlapping_values, reuse_exclusions = _exclude_cross_visit_reuse(values)
    exclusions.extend(reuse_exclusions)
    chronological_values, chronology_exclusions = _exclude_nonchronological_values(
        nonoverlapping_values,
        visit_ids=tuple(visit.visit_id for visit in visits),
    )
    exclusions.extend(chronology_exclusions)
    return LongitudinalExtraction(
        study_id=study.study_id,
        visit_ids=tuple(visit.visit_id for visit in visits),
        subject_ids=selected_ids,
        channels=tuple(resolved),
        values=chronological_values,
        exclusions=tuple(exclusions),
    )


def complete_visit_vectors(extraction: LongitudinalExtraction) -> VisitVectorExtraction:
    """Build ordered vectors and explicitly mark every incomplete subject visit."""
    lookup = extraction.value_map()
    vectors: list[VisitAlignedVector] = []
    exclusions = list(extraction.exclusions)
    for subject_id in extraction.subject_ids:
        for visit_id in extraction.visit_ids:
            missing = tuple(
                index
                for index in range(len(extraction.channels))
                if (subject_id, visit_id, index) not in lookup
            )
            if missing:
                exclusions.append(
                    LongitudinalExclusion(
                        reason=LongitudinalExclusionReason.INCOMPLETE_VISIT_VECTOR,
                        subject_id=subject_id,
                        visit_id=visit_id,
                        missing_channel_indices=missing,
                    )
                )
                continue
            rows = tuple(
                lookup[(subject_id, visit_id, index)] for index in range(len(extraction.channels))
            )
            exact_channels = tuple(item.channel for item in rows)
            timestamps = tuple(item.effective_timestamp for item in rows)
            indices = tuple(
                sorted({index for item in rows for index in item.selected_observation_indices})
            )
            vectors.append(
                VisitAlignedVector(
                    subject_id=subject_id,
                    visit_id=visit_id,
                    channels=exact_channels,
                    values=tuple(item.value for item in rows),
                    effective_timestamp=_mean_timestamp(timestamps),
                    selected_observation_indices=indices,
                )
            )
    return VisitVectorExtraction(vectors=tuple(vectors), exclusions=tuple(exclusions))


def _validate_visits_and_channels(
    visits: tuple[ExpectedVisit, ...],
    channels: tuple[VisitFeature | LongitudinalChannel, ...],
) -> None:
    if not visits:
        raise ValueError("at least one expected visit is required")
    visit_ids = [visit.visit_id for visit in visits]
    if len(visit_ids) != len(set(visit_ids)):
        raise ValueError("visit identifiers must be unique")
    if not channels:
        raise ValueError("at least one longitudinal channel is required")
    for index, first in enumerate(channels):
        for second in channels[index + 1 :]:
            same_feature = first.feature == second.feature
            overlapping_modality = (
                first.modality is None
                or second.modality is None
                or first.modality is second.modality
            )
            if same_feature and overlapping_modality:
                raise ValueError(
                    "longitudinal feature requests cannot overlap by exact or wildcard modality"
                )


def _duplicate_resolved_channel_exclusions(
    *,
    requests: tuple[VisitFeature | LongitudinalChannel, ...],
    resolved: tuple[LongitudinalChannel | None, ...],
    candidate_indices: dict[int, tuple[int, ...]],
) -> tuple[LongitudinalExclusion, ...]:
    """Return structured errors if distinct requests resolve to one exact channel."""
    positions: dict[LongitudinalChannel, list[int]] = {}
    for channel_index, channel in enumerate(resolved):
        if channel is not None:
            positions.setdefault(channel, []).append(channel_index)
    duplicated = {channel: indices for channel, indices in positions.items() if len(indices) > 1}
    return tuple(
        LongitudinalExclusion(
            reason=LongitudinalExclusionReason.DUPLICATE_RESOLVED_CHANNEL,
            channel_index=channel_index,
            feature=channel.feature,
            requested_modality=requests[channel_index].modality,
            observation_indices=candidate_indices[channel_index],
            observed_modalities=(channel.modality,),
            observed_units=(channel.unit,),
        )
        for channel, indices in duplicated.items()
        for channel_index in indices
    )


def _exclude_cross_visit_reuse(
    values: list[VisitAlignedValue],
) -> tuple[tuple[VisitAlignedValue, ...], tuple[LongitudinalExclusion, ...]]:
    """Fail closed when one source row would satisfy multiple expected visits."""
    uses: dict[tuple[str, int, int], set[str]] = {}
    for value in values:
        for observation_index in value.selected_observation_indices:
            uses.setdefault(
                (value.subject_id, value.channel_index, observation_index),
                set(),
            ).add(value.visit_id)
    reused = {key for key, visit_ids in uses.items() if len(visit_ids) > 1}
    if not reused:
        return tuple(values), ()

    retained: list[VisitAlignedValue] = []
    exclusions: list[LongitudinalExclusion] = []
    for value in values:
        duplicated_indices = tuple(
            index
            for index in value.selected_observation_indices
            if (value.subject_id, value.channel_index, index) in reused
        )
        if not duplicated_indices:
            retained.append(value)
            continue
        exclusions.append(
            LongitudinalExclusion(
                reason=LongitudinalExclusionReason.REUSED_OBSERVATION_ACROSS_VISITS,
                subject_id=value.subject_id,
                visit_id=value.visit_id,
                channel_index=value.channel_index,
                feature=value.channel.feature,
                requested_modality=value.channel.modality,
                observation_indices=duplicated_indices,
                observed_modalities=(value.channel.modality,),
                observed_units=(value.channel.unit,),
            )
        )
    return tuple(retained), tuple(exclusions)


def _exclude_nonchronological_values(
    values: tuple[VisitAlignedValue, ...],
    *,
    visit_ids: tuple[str, ...],
) -> tuple[tuple[VisitAlignedValue, ...], tuple[LongitudinalExclusion, ...]]:
    """Greedily retain only strictly chronological values on each requested channel."""
    visit_positions = {visit_id: index for index, visit_id in enumerate(visit_ids)}
    grouped: dict[tuple[str, int], list[VisitAlignedValue]] = {}
    for value in values:
        grouped.setdefault((value.subject_id, value.channel_index), []).append(value)
    excluded_keys: set[tuple[str, str, int]] = set()
    exclusions: list[LongitudinalExclusion] = []
    for channel_values in grouped.values():
        ordered = sorted(channel_values, key=lambda item: visit_positions[item.visit_id])
        retained = ordered[0]
        for current in ordered[1:]:
            if current.effective_timestamp > retained.effective_timestamp:
                retained = current
                continue
            excluded_keys.add((current.subject_id, current.visit_id, current.channel_index))
            exclusions.append(
                LongitudinalExclusion(
                    reason=LongitudinalExclusionReason.NONCHRONOLOGICAL_OBSERVED_TIME,
                    subject_id=current.subject_id,
                    visit_id=current.visit_id,
                    channel_index=current.channel_index,
                    feature=current.channel.feature,
                    requested_modality=current.channel.modality,
                    observation_indices=tuple(
                        sorted(
                            set(retained.selected_observation_indices).union(
                                current.selected_observation_indices
                            )
                        )
                    ),
                    observed_modalities=(current.channel.modality,),
                    observed_units=(current.channel.unit,),
                )
            )
    return (
        tuple(
            value
            for value in values
            if (value.subject_id, value.visit_id, value.channel_index) not in excluded_keys
        ),
        tuple(exclusions),
    )


def _candidate_indices(
    study: Study,
    visits: tuple[ExpectedVisit, ...],
    channels: tuple[VisitFeature | LongitudinalChannel, ...],
    subjects: dict[str, Subject],
    selected: set[str],
) -> dict[int, tuple[int, ...]]:
    output: dict[int, list[int]] = {index: [] for index in range(len(channels))}
    for observation_index, row in enumerate(study.observations):
        if row.subject_id not in selected:
            continue
        subject = subjects[row.subject_id]
        if not any(
            _visit_applies(visit, subject)
            and (window := visit.window_for(subject)) is not None
            and window[0] <= row.timestamp <= window[1]
            for visit in visits
        ):
            continue
        for channel_index, request in enumerate(channels):
            if row.feature == request.feature and (
                request.modality is None or row.modality is request.modality
            ):
                output[channel_index].append(observation_index)
    return {key: tuple(sorted(set(value))) for key, value in output.items()}


def _resolve_channel(
    study: Study,
    *,
    channel_index: int,
    request: VisitFeature | LongitudinalChannel,
    candidate_indices: tuple[int, ...],
    default_policy: AggregationPolicy,
) -> tuple[LongitudinalChannel | None, tuple[LongitudinalExclusion, ...]]:
    candidates = tuple(study.observations[index] for index in candidate_indices)
    if request.modality is None:
        modalities = tuple(
            sorted({row.modality for row in candidates}, key=lambda item: item.value)
        )
        if len(modalities) > 1:
            return None, (
                LongitudinalExclusion(
                    reason=LongitudinalExclusionReason.AMBIGUOUS_MODALITY,
                    channel_index=channel_index,
                    feature=request.feature,
                    observation_indices=candidate_indices,
                    observed_modalities=modalities,
                    observed_units=tuple(sorted({row.unit for row in candidates})),
                ),
            )
        if not modalities:
            return None, ()
        modality = modalities[0]
    else:
        modality = request.modality
    exact_modality = tuple(row for row in candidates if row.modality is modality)
    observed_units = tuple(sorted({row.unit for row in exact_modality}))
    expected_unit = request.unit if isinstance(request, LongitudinalChannel) else None
    if len(observed_units) > 1:
        return None, (
            LongitudinalExclusion(
                reason=LongitudinalExclusionReason.MIXED_UNITS,
                channel_index=channel_index,
                feature=request.feature,
                requested_modality=modality,
                observation_indices=candidate_indices,
                observed_modalities=(modality,),
                observed_units=observed_units,
            ),
        )
    if expected_unit is not None and observed_units and observed_units[0] != expected_unit:
        return None, (
            LongitudinalExclusion(
                reason=LongitudinalExclusionReason.UNIT_MISMATCH,
                channel_index=channel_index,
                feature=request.feature,
                requested_modality=modality,
                observation_indices=candidate_indices,
                observed_modalities=(modality,),
                observed_units=observed_units,
            ),
        )
    unit = expected_unit or (observed_units[0] if observed_units else None)
    if unit is None:
        return None, ()
    policy = (
        request.aggregation_policy if isinstance(request, LongitudinalChannel) else default_policy
    )
    return (
        LongitudinalChannel(
            feature=request.feature,
            modality=modality,
            unit=unit,
            aggregation_policy=policy,
        ),
        (),
    )


def _exclusion(
    *,
    reason: LongitudinalExclusionReason,
    subject: Subject,
    visit: ExpectedVisit,
    channel_index: int,
    request: VisitFeature | LongitudinalChannel,
) -> LongitudinalExclusion:
    return LongitudinalExclusion(
        reason=reason,
        subject_id=subject.subject_id,
        visit_id=visit.visit_id,
        channel_index=channel_index,
        feature=request.feature,
        requested_modality=request.modality,
    )


def _select_indices(
    observations: tuple[Observation, ...],
    indices: tuple[int, ...],
    *,
    target: datetime,
    policy: AggregationPolicy,
) -> tuple[int, ...]:
    if policy is not AggregationPolicy.CLOSEST_TO_SCHEDULE:
        return tuple(sorted(indices))
    closest = min(
        indices,
        key=lambda index: (
            abs((observations[index].timestamp - target).total_seconds()),
            observations[index].timestamp.astimezone(UTC),
            index,
        ),
    )
    timestamp = observations[closest].timestamp
    return tuple(sorted(index for index in indices if observations[index].timestamp == timestamp))


def _aggregate(
    observations: tuple[Observation, ...],
    indices: tuple[int, ...],
    policy: AggregationPolicy,
) -> tuple[float, datetime]:
    rows = tuple(observations[index] for index in indices)
    values = tuple(row.value for row in rows)
    timestamps = tuple(row.timestamp for row in rows)
    if policy is AggregationPolicy.MEDIAN:
        return float(median(values)), _median_timestamp(timestamps)
    return float(fmean(values)), _mean_timestamp(timestamps)


def _mean_timestamp(timestamps: tuple[datetime, ...]) -> datetime:
    epoch_seconds = tuple(item.astimezone(UTC).timestamp() for item in timestamps)
    return datetime.fromtimestamp(fmean(epoch_seconds), tz=UTC)


def _median_timestamp(timestamps: tuple[datetime, ...]) -> datetime:
    epoch_seconds = tuple(item.astimezone(UTC).timestamp() for item in timestamps)
    return datetime.fromtimestamp(float(median(epoch_seconds)), tz=UTC)


def _visit_applies(visit: ExpectedVisit, subject: Subject) -> bool:
    if not visit.subject_ids and not visit.cohorts:
        return True
    return subject.subject_id in visit.subject_ids or subject.cohort in visit.cohorts
