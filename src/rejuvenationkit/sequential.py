"""Sequential multimodal treatment-response detection."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from hashlib import sha256
from itertools import pairwise
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.longitudinal import (
    AggregationPolicy,
    LongitudinalChannel,
    LongitudinalExclusion,
    LongitudinalExclusionReason,
    VisitAlignedVector,
    complete_visit_vectors,
    extract_visit_aligned_values,
)
from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Study

_DAYS_PER_YEAR = 365.2425
_EMPIRICAL_QUANTILE_METHOD: Literal["higher"] = "higher"


def _serialized_model_artifact_hash(model: BaseModel, *, schema: str) -> str:
    """Hash every serialized model field except the hash itself."""
    payload = {
        "schema": schema,
        "model": model.model_dump(mode="json", exclude={"model_artifact_hash"}),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class SequentialDetectionConfig(BaseModel):
    """Configuration for reference-conditioned sequential detection."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    features: tuple[LongitudinalChannel | VisitFeature, ...] = Field(min_length=2)
    covariance_shrinkage: float = Field(default=0.20, ge=0, le=1)
    covariance_ridge: float = Field(default=1e-9, gt=0)
    false_alarm_rate: float = Field(default=0.05, gt=0, lt=0.5)
    minimum_reference_subjects: int = Field(default=30, ge=5)
    persistence_crossings: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def require_unique_features(self) -> SequentialDetectionConfig:
        """Reject exact or wildcard requests that could resolve to one channel."""
        for index, first in enumerate(self.features):
            for second in self.features[index + 1 :]:
                if first.feature == second.feature and (
                    first.modality is None
                    or second.modality is None
                    or first.modality is second.modality
                ):
                    raise ValueError(
                        "sequential detection features cannot overlap by exact or wildcard modality"
                    )
        return self


class SequentialDetectionModel(BaseModel):
    """Fitted normal-trajectory dynamics and subject-level threshold."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    feature_names: tuple[str, ...]
    feature_modalities: tuple[Modality | None, ...]
    resolved_feature_modalities: tuple[Modality, ...] = ()
    feature_units: tuple[str, ...] = ()
    aggregation_policies: tuple[AggregationPolicy, ...] = ()
    reference_subjects: int = Field(ge=1)
    reference_transitions: int = Field(ge=1)
    mean_change_per_year: tuple[float, ...]
    innovation_covariance_per_year: tuple[tuple[float, ...], ...]
    maximum_cumulative_score_threshold: float = Field(ge=0)
    false_alarm_rate: float = Field(gt=0, lt=0.5)
    threshold_quantile_method: Literal["higher"] = _EMPIRICAL_QUANTILE_METHOD
    reference_score_method: Literal["leave_one_out"] = "leave_one_out"
    study_id: str | None = None
    ordered_visits: tuple[ExpectedVisit, ...] = ()
    resolved_channels: tuple[LongitudinalChannel, ...] = ()
    config: SequentialDetectionConfig | None = None
    requested_reference_subject_ids: tuple[str, ...] = ()
    reference_subject_ids: tuple[str, ...] = ()
    reference_input_artifact_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reference_maximum_score_distribution: tuple[float, ...] = ()
    model_artifact_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_fitted_provenance(self) -> SequentialDetectionModel:
        """Reject incomplete or internally inconsistent serialized fits."""
        dimension = len(self.feature_names)
        if dimension < 2:
            raise ValueError("sequential model requires at least two features")
        if len(self.feature_modalities) != dimension:
            raise ValueError("feature modalities must match feature order")
        for label, values in (
            ("resolved feature modalities", self.resolved_feature_modalities),
            ("feature units", self.feature_units),
            ("aggregation policies", self.aggregation_policies),
        ):
            if values and len(values) != dimension:
                raise ValueError(f"{label} must match feature order")
        if len(self.mean_change_per_year) != dimension:
            raise ValueError("mean change rate must match feature order")
        if len(self.innovation_covariance_per_year) != dimension or any(
            len(row) != dimension for row in self.innovation_covariance_per_year
        ):
            raise ValueError("sequential covariance must be square in feature order")
        mean = np.asarray(self.mean_change_per_year, dtype=np.float64)
        covariance = np.asarray(self.innovation_covariance_per_year, dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("sequential model parameters must be finite")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0.0):
            raise ValueError("sequential covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("sequential covariance must be positive definite") from error

        fitted_markers = (
            self.study_id is not None,
            bool(self.ordered_visits),
            bool(self.resolved_channels),
            self.config is not None,
            bool(self.requested_reference_subject_ids),
            bool(self.reference_subject_ids),
            self.reference_input_artifact_hash is not None,
            bool(self.reference_maximum_score_distribution),
            self.model_artifact_hash is not None,
        )
        if not any(fitted_markers):
            return self
        if not all(fitted_markers):
            raise ValueError("complete fitted sequential provenance is required")

        assert self.config is not None
        assert self.model_artifact_hash is not None
        if len(self.ordered_visits) < 3 or len(
            {visit.visit_id for visit in self.ordered_visits}
        ) != len(self.ordered_visits):
            raise ValueError("fitted sequential visits must contain at least three unique IDs")
        if self.false_alarm_rate != self.config.false_alarm_rate:
            raise ValueError("model false-alarm rate does not match fitted config")
        if self.feature_names != tuple(item.feature for item in self.config.features):
            raise ValueError("model feature names do not match fitted config")
        if self.feature_modalities != tuple(item.modality for item in self.config.features):
            raise ValueError("model feature modalities do not match fitted config")
        if self.resolved_channels and len(self.resolved_channels) != len(self.feature_names):
            raise ValueError("resolved sequential channels must match feature order")
        if self.feature_names != tuple(item.feature for item in self.resolved_channels):
            raise ValueError("resolved channel features do not match model feature order")
        if self.resolved_feature_modalities != tuple(
            item.modality for item in self.resolved_channels
        ):
            raise ValueError("resolved channel modalities do not match model metadata")
        if self.feature_units != tuple(item.unit for item in self.resolved_channels):
            raise ValueError("resolved channel units do not match model metadata")
        if self.aggregation_policies != tuple(
            item.aggregation_policy for item in self.resolved_channels
        ):
            raise ValueError("resolved channel aggregation does not match model metadata")
        for requested, resolved in zip(
            self.config.features,
            self.resolved_channels,
            strict=True,
        ):
            if isinstance(requested, LongitudinalChannel) and requested != resolved:
                raise ValueError("resolved channel does not match exact configured channel")
        if self.reference_subject_ids and (
            tuple(sorted(set(self.reference_subject_ids))) != self.reference_subject_ids
        ):
            raise ValueError("reference subject identifiers must be unique and sorted")
        if self.requested_reference_subject_ids and (
            tuple(sorted(set(self.requested_reference_subject_ids)))
            != self.requested_reference_subject_ids
        ):
            raise ValueError("requested reference identifiers must be unique and sorted")
        if self.reference_subject_ids and self.reference_subjects != len(
            self.reference_subject_ids
        ):
            raise ValueError("reference subject count does not match identifiers")
        if not set(self.reference_subject_ids).issubset(self.requested_reference_subject_ids):
            raise ValueError("complete reference subjects must be requested references")
        if self.reference_subjects < self.config.minimum_reference_subjects:
            raise ValueError("reference subject count violates fitted config")
        expected_transitions = self.reference_subjects * (len(self.ordered_visits) - 1)
        if self.reference_transitions != expected_transitions:
            raise ValueError("reference transition count does not match fitted trajectories")
        if len(self.reference_maximum_score_distribution) != self.reference_subjects:
            raise ValueError("maximum-score distribution must contain one score per subject")
        if tuple(
            sorted(self.reference_maximum_score_distribution)
        ) != self.reference_maximum_score_distribution or any(
            not math.isfinite(score) or score < 0
            for score in self.reference_maximum_score_distribution
        ):
            raise ValueError("maximum-score distribution must be finite, nonnegative, and sorted")
        expected_threshold = float(
            np.quantile(
                np.asarray(
                    self.reference_maximum_score_distribution,
                    dtype=np.float64,
                ),
                1 - self.false_alarm_rate,
                method=self.threshold_quantile_method,
            )
        )
        if self.maximum_cumulative_score_threshold != expected_threshold:
            raise ValueError("sequential threshold does not match reference score quantile")
        expected_hash = _serialized_model_artifact_hash(
            self,
            schema="rejuvenationkit.sequential-detection-model.v1",
        )
        if self.model_artifact_hash != expected_hash:
            raise ValueError("sequential model artifact hash does not match serialized contents")
        return self


class SequentialDetectionPoint(BaseModel):
    """Evidence accumulated through one observed transition."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    from_visit_id: str
    to_visit_id: str
    from_observed_at: datetime | None = None
    to_observed_at: datetime | None = None
    selected_observation_indices: tuple[int, ...] = ()
    elapsed_years: float = Field(gt=0)
    interval_score: float = Field(ge=0)
    cumulative_score: float = Field(ge=0)
    empirical_tail_probability: float = Field(gt=0, le=1)
    threshold_crossed: bool

    @model_validator(mode="after")
    def validate_observed_interval(self) -> SequentialDetectionPoint:
        """Keep optional observed timestamps complete, ordered, and auditable."""
        if (self.from_observed_at is None) != (self.to_observed_at is None):
            raise ValueError("both observed timestamps must be provided together")
        if self.from_observed_at is not None and self.to_observed_at is not None:
            if any(
                item.tzinfo is None or item.utcoffset() is None
                for item in (self.from_observed_at, self.to_observed_at)
            ):
                raise ValueError("observed timestamps must be timezone-aware")
            if self.to_observed_at <= self.from_observed_at:
                raise ValueError("observed transition timestamps must be chronological")
        if tuple(sorted(set(self.selected_observation_indices))) != (
            self.selected_observation_indices
        ):
            raise ValueError("selected observation indices must be unique and sorted")
        return self


class ModalityEvidence(BaseModel):
    """Peak cumulative evidence calculated within one modality."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    modality: Modality | None
    channels: int = Field(ge=1)
    score: float = Field(ge=0)


class SubjectSequentialDetection(BaseModel):
    """Onset and persistence assessment for one subject trajectory."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    subject_id: str
    points: tuple[SequentialDetectionPoint, ...]
    detected: bool
    onset_visit_id: str | None = None
    persistent: bool = False
    transient: bool = False
    peak_cumulative_score: float = Field(ge=0)
    peak_modality_evidence: tuple[ModalityEvidence, ...] = ()
    missing_visit_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_exclusive_trajectory_classification(self) -> SubjectSequentialDetection:
        """Reject simultaneous transient and persistent classifications."""
        if self.transient and self.persistent:
            raise ValueError("transient and persistent classifications are mutually exclusive")
        if len(self.missing_visit_ids) != len(set(self.missing_visit_ids)):
            raise ValueError("missing visit identifiers must be unique")
        return self


class SequentialDetectionReport(BaseModel):
    """Sequential results for scored subjects."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    study_id: str
    visit_ids: tuple[str, ...]
    model: SequentialDetectionModel
    results: tuple[SubjectSequentialDetection, ...]
    excluded_subject_ids: tuple[str, ...] = ()
    exclusions: tuple[LongitudinalExclusion, ...] = ()

    @model_validator(mode="after")
    def validate_scored_trajectories(self) -> SequentialDetectionReport:
        """Bind fitted-report trajectories to their visit, threshold, and classification rules."""
        result_ids = tuple(item.subject_id for item in self.results)
        if tuple(sorted(set(result_ids))) != result_ids:
            raise ValueError("sequential result subjects must be unique and sorted")
        if tuple(sorted(set(self.excluded_subject_ids))) != self.excluded_subject_ids:
            raise ValueError("sequential excluded subjects must be unique and sorted")
        if set(result_ids).intersection(self.excluded_subject_ids):
            raise ValueError("sequential subjects cannot be both scored and excluded")
        if self.model.study_id is None:
            return self
        if self.study_id != self.model.study_id:
            raise ValueError("sequential report study does not match its model")
        if self.visit_ids != tuple(visit.visit_id for visit in self.model.ordered_visits):
            raise ValueError("sequential report visits do not match its model")
        if self.model.config is None:
            raise ValueError("fitted sequential report requires its model configuration")
        positions = {visit_id: index for index, visit_id in enumerate(self.visit_ids)}
        threshold = self.model.maximum_cumulative_score_threshold
        reference_scores = np.asarray(
            self.model.reference_maximum_score_distribution,
            dtype=np.float64,
        )
        expected_modalities = {
            modality: sum(feature.modality is modality for feature in self.model.config.features)
            for modality in {feature.modality for feature in self.model.config.features}
        }
        for result in self.results:
            if not result.points:
                raise ValueError("scored sequential trajectories require at least one transition")
            trajectory_visits = (
                result.points[0].from_visit_id,
                *(point.to_visit_id for point in result.points),
            )
            if (
                any(visit_id not in positions for visit_id in trajectory_visits)
                or any(
                    first.to_visit_id != second.from_visit_id
                    for first, second in pairwise(result.points)
                )
                or any(
                    positions[first] >= positions[second]
                    for first, second in pairwise(trajectory_visits)
                )
            ):
                raise ValueError("sequential transitions must form one ordered trajectory")
            expected_missing = tuple(
                visit_id for visit_id in self.visit_ids if visit_id not in trajectory_visits
            )
            if result.missing_visit_ids != expected_missing:
                raise ValueError("sequential missing visits do not match the trajectory")
            for point_index, point in enumerate(result.points):
                if (
                    point.from_observed_at is None
                    or point.to_observed_at is None
                    or not point.selected_observation_indices
                ):
                    raise ValueError(
                        "fitted sequential transitions require timestamps and source rows"
                    )
                expected_elapsed = _elapsed_years(
                    point.from_observed_at,
                    point.to_observed_at,
                )
                if not math.isclose(
                    point.elapsed_years,
                    expected_elapsed,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError("sequential elapsed time does not match observed timestamps")
                if point_index and (
                    result.points[point_index - 1].to_observed_at != point.from_observed_at
                ):
                    raise ValueError("sequential observed timestamps do not form one trajectory")
                if point.threshold_crossed != (point.cumulative_score > threshold):
                    raise ValueError("sequential crossing flag does not match model threshold")
                expected_tail = float(
                    (1 + np.count_nonzero(reference_scores >= point.cumulative_score))
                    / (len(reference_scores) + 1)
                )
                if not math.isclose(
                    point.empirical_tail_probability,
                    expected_tail,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError("sequential tail probability does not match its model")
            onset = next(
                (point.to_visit_id for point in result.points if point.threshold_crossed),
                None,
            )
            crossings = tuple(point.threshold_crossed for point in result.points)
            final_crossings = 0
            for crossed in reversed(crossings):
                if not crossed:
                    break
                final_crossings += 1
            persistent = final_crossings >= self.model.config.persistence_crossings
            transient = (
                onset is not None
                and not persistent
                and any(
                    crossed and not later
                    for index, crossed in enumerate(crossings)
                    for later in crossings[index + 1 :]
                )
            )
            if result.detected != (onset is not None) or result.onset_visit_id != onset:
                raise ValueError("sequential detection and onset do not match trajectory")
            if result.persistent != persistent or result.transient != transient:
                raise ValueError("sequential classification does not match trajectory")
            peak = max(point.cumulative_score for point in result.points)
            if not math.isclose(
                result.peak_cumulative_score,
                peak,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("sequential peak does not match trajectory")
            modalities = {
                evidence.modality: evidence.channels for evidence in result.peak_modality_evidence
            }
            if (
                len(modalities) != len(result.peak_modality_evidence)
                or modalities != expected_modalities
            ):
                raise ValueError("sequential modality evidence does not match configured channels")
        return self

    def results_frame(self) -> pd.DataFrame:
        """Return one summary row per scored subject."""
        return pd.DataFrame(
            {
                "subject_id": item.subject_id,
                "transitions": len(item.points),
                "detected": item.detected,
                "onset_visit_id": item.onset_visit_id,
                "persistent": item.persistent,
                "transient": item.transient,
                "peak_cumulative_score": item.peak_cumulative_score,
                "missing_visit_ids": ",".join(item.missing_visit_ids),
            }
            for item in self.results
        )

    def trajectory_frame(self) -> pd.DataFrame:
        """Return one row per subject transition."""
        return pd.DataFrame(
            {
                "subject_id": result.subject_id,
                **point.model_dump(mode="json"),
            }
            for result in self.results
            for point in result.points
        )


class SequentialTreatmentResponseDetector:
    """Detect sustained departures from a fitted reference trajectory.

    The detector models each channel as a drift-plus-random-walk process.
    Irregular-time increments are centered by the reference drift and divided
    by the square root of elapsed time. Their covariance is shrunk and whitened.
    Coherent cumulative evidence is the energy of the running mean whitened
    innovation, with a threshold calibrated from each reference subject's
    maximum score across the full trajectory.
    """

    def __init__(self, config: SequentialDetectionConfig) -> None:
        """Create an unfitted sequential detector."""
        self.config = config
        self.model_: SequentialDetectionModel | None = None
        self._mean_rate: NDArray[np.float64] | None = None
        self._covariance: NDArray[np.float64] | None = None
        self._cholesky: NDArray[np.float64] | None = None
        self._reference_maximum_scores: NDArray[np.float64] | None = None
        self._reference_subject_ids: frozenset[str] | None = None
        self._channels: tuple[LongitudinalChannel, ...] | None = None

    @classmethod
    def from_model(
        cls,
        model: SequentialDetectionModel,
    ) -> SequentialTreatmentResponseDetector:
        """Reconstruct a scorer from a complete serialized sequential model."""
        model = SequentialDetectionModel.model_validate(model.model_dump(mode="python"))
        if (
            model.config is None
            or model.study_id is None
            or not model.ordered_visits
            or not model.resolved_channels
            or not model.requested_reference_subject_ids
            or model.reference_input_artifact_hash is None
            or not model.reference_maximum_score_distribution
            or model.model_artifact_hash is None
        ):
            raise ValueError("model does not contain complete fitted sequential provenance")
        detector = cls(model.config)
        covariance = np.asarray(model.innovation_covariance_per_year, dtype=np.float64)
        detector.model_ = model
        detector._mean_rate = np.asarray(model.mean_change_per_year, dtype=np.float64)
        detector._covariance = covariance
        detector._cholesky = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)
        detector._reference_maximum_scores = np.asarray(
            model.reference_maximum_score_distribution,
            dtype=np.float64,
        )
        detector._reference_subject_ids = frozenset(model.requested_reference_subject_ids)
        detector._channels = model.resolved_channels
        return detector

    def fit(
        self,
        study: Study,
        *,
        visits: tuple[ExpectedVisit, ...],
        reference_subject_ids: tuple[str, ...],
    ) -> SequentialTreatmentResponseDetector:
        """Fit reference drift, innovation covariance, and sequential threshold."""
        _validate_visits(visits)
        normalized_reference_ids = tuple(sorted(reference_subject_ids))
        trajectories, excluded, _, _, channels = _complete_trajectories(
            study,
            visits=visits,
            features=self.config.features,
            subject_ids=normalized_reference_ids,
            require_all_visits=True,
        )
        if len(trajectories) < self.config.minimum_reference_subjects:
            raise ValueError(
                "insufficient complete reference trajectories: "
                f"{len(trajectories)} < {self.config.minimum_reference_subjects}; "
                f"excluded={len(excluded)}"
            )
        increments = {
            subject_id: [
                (
                    second.as_array() - first.as_array(),
                    _elapsed_years(first.effective_timestamp, second.effective_timestamp),
                )
                for first, second in pairwise(trajectory)
            ]
            for subject_id, trajectory in trajectories.items()
        }
        mean_rate, covariance = _fit_reference_dynamics(
            list(increments.values()),
            shrinkage=self.config.covariance_shrinkage,
            ridge=self.config.covariance_ridge,
        )
        cholesky = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)
        # Each reference subject's maximum score is computed against dynamics fitted
        # without that subject. In-sample scores are biased low and would inflate the
        # false-alarm rate for new subjects.
        maximum_scores = np.empty(len(increments), dtype=np.float64)
        ordered_increments = list(increments.values())
        for index, sequence in enumerate(ordered_increments):
            held_out_rate, held_out_covariance = _fit_reference_dynamics(
                ordered_increments[:index] + ordered_increments[index + 1 :],
                shrinkage=self.config.covariance_shrinkage,
                ridge=self.config.covariance_ridge,
            )
            maximum_scores[index] = max(
                _cumulative_scores(
                    _innovations(sequence, held_out_rate),
                    np.asarray(np.linalg.cholesky(held_out_covariance), dtype=np.float64),
                )
            )
        threshold = float(
            np.quantile(
                maximum_scores,
                1 - self.config.false_alarm_rate,
                method="higher",
            )
        )
        self._mean_rate = mean_rate
        self._covariance = covariance
        self._cholesky = cholesky
        self._reference_maximum_scores = maximum_scores
        self._reference_subject_ids = frozenset(normalized_reference_ids)
        self._channels = channels
        complete_reference_ids = tuple(sorted(trajectories))
        reference_input_artifact_hash = _reference_input_artifact_hash(
            study_id=study.study_id,
            visits=visits,
            config=self.config,
            channels=channels,
            requested_reference_subject_ids=normalized_reference_ids,
            trajectories=trajectories,
        )
        unsigned_model = SequentialDetectionModel.model_construct(
            feature_names=tuple(item.feature for item in self.config.features),
            feature_modalities=tuple(item.modality for item in self.config.features),
            resolved_feature_modalities=tuple(item.modality for item in channels),
            feature_units=tuple(item.unit for item in channels),
            aggregation_policies=tuple(item.aggregation_policy for item in channels),
            reference_subjects=len(trajectories),
            reference_transitions=sum(len(sequence) for sequence in ordered_increments),
            mean_change_per_year=tuple(float(value) for value in mean_rate),
            innovation_covariance_per_year=tuple(
                tuple(float(value) for value in row) for row in covariance
            ),
            maximum_cumulative_score_threshold=threshold,
            false_alarm_rate=self.config.false_alarm_rate,
            study_id=study.study_id,
            ordered_visits=visits,
            resolved_channels=channels,
            config=self.config,
            requested_reference_subject_ids=normalized_reference_ids,
            reference_subject_ids=complete_reference_ids,
            reference_input_artifact_hash=reference_input_artifact_hash,
            reference_maximum_score_distribution=tuple(
                float(value) for value in sorted(maximum_scores)
            ),
        )
        self.model_ = SequentialDetectionModel.model_validate(
            {
                **unsigned_model.model_dump(mode="python"),
                "model_artifact_hash": _serialized_model_artifact_hash(
                    unsigned_model,
                    schema="rejuvenationkit.sequential-detection-model.v1",
                ),
            }
        )
        return self

    def score(
        self,
        study: Study,
        *,
        visits: tuple[ExpectedVisit, ...],
        subject_ids: tuple[str, ...],
    ) -> SequentialDetectionReport:
        """Score partially observed trajectories with at least two complete visits."""
        _validate_visits(visits)
        if (
            self.model_ is None
            or self._mean_rate is None
            or self._covariance is None
            or self._cholesky is None
            or self._reference_maximum_scores is None
            or self._channels is None
        ):
            raise RuntimeError("fit must be called before score")
        self._validate_scoring_domain(study, visits=visits)
        assert self._reference_subject_ids is not None
        overlap = set(subject_ids).intersection(self._reference_subject_ids)
        if overlap:
            raise ValueError(
                "sequential evaluation subjects overlap fitted reference subjects: "
                f"{sorted(overlap)}"
            )
        trajectories, excluded, exclusions, missing_visits, _ = _complete_trajectories(
            study,
            visits=visits,
            features=self._channels,
            subject_ids=subject_ids,
            require_all_visits=False,
        )
        results = tuple(
            self._score_subject(
                subject_id,
                trajectory,
                missing_visit_ids=missing_visits[subject_id],
            )
            for subject_id, trajectory in sorted(trajectories.items())
        )
        return SequentialDetectionReport(
            study_id=study.study_id,
            visit_ids=tuple(visit.visit_id for visit in visits),
            model=self.model_,
            results=results,
            excluded_subject_ids=excluded,
            exclusions=exclusions,
        )

    def _validate_scoring_domain(
        self,
        study: Study,
        *,
        visits: tuple[ExpectedVisit, ...],
    ) -> None:
        """Reject visit, channel, config, and fitted-reference drift."""
        assert self.model_ is not None
        assert self._channels is not None
        if self.model_.study_id != study.study_id:
            raise ValueError("scoring study does not match fitted sequential domain")
        if self.model_.ordered_visits != visits:
            raise ValueError("scoring visit definitions do not match fitted sequential domain")
        if self.model_.config != self.config:
            raise ValueError("detector configuration does not match fitted sequential domain")
        requested_ids = self.model_.requested_reference_subject_ids
        if not requested_ids or self.model_.reference_input_artifact_hash is None:
            raise ValueError("fitted sequential model is missing reference-input provenance")
        try:
            trajectories, _, _, _, channels = _complete_trajectories(
                study,
                visits=visits,
                features=self._channels,
                subject_ids=requested_ids,
                require_all_visits=True,
            )
        except ValueError as error:
            raise ValueError(
                "fitted sequential reference input is unavailable or altered"
            ) from error
        if channels != self.model_.resolved_channels:
            raise ValueError("scoring channels do not match fitted sequential domain")
        observed_hash = _reference_input_artifact_hash(
            study_id=study.study_id,
            visits=visits,
            config=self.config,
            channels=channels,
            requested_reference_subject_ids=requested_ids,
            trajectories=trajectories,
        )
        if observed_hash != self.model_.reference_input_artifact_hash:
            raise ValueError("fitted sequential reference input is unavailable or altered")

    def _score_subject(
        self,
        subject_id: str,
        trajectory: list[VisitAlignedVector],
        *,
        missing_visit_ids: tuple[str, ...],
    ) -> SubjectSequentialDetection:
        assert self.model_ is not None
        assert self._mean_rate is not None
        assert self._covariance is not None
        assert self._cholesky is not None
        assert self._reference_maximum_scores is not None
        innovations: list[NDArray[np.float64]] = []
        points: list[SequentialDetectionPoint] = []
        cumulative_raw: NDArray[np.float64] = np.zeros(
            len(self.config.features),
            dtype=np.float64,
        )
        peak_raw: NDArray[np.float64] = cumulative_raw.copy()
        peak_score = 0.0
        for first, second in pairwise(trajectory):
            elapsed = _elapsed_years(first.effective_timestamp, second.effective_timestamp)
            raw = (second.as_array() - first.as_array() - self._mean_rate * elapsed) / math.sqrt(
                elapsed
            )
            innovations.append(raw)
            cumulative_raw += raw
            whitened = np.linalg.solve(self._cholesky, raw)
            interval_score = float(whitened @ whitened)
            cumulative_score = _cumulative_score(innovations, self._cholesky)
            if cumulative_score > peak_score:
                peak_score = cumulative_score
                peak_raw = np.asarray(
                    cumulative_raw / math.sqrt(len(innovations)),
                    dtype=np.float64,
                )
            tail = float(
                (1 + np.count_nonzero(self._reference_maximum_scores >= cumulative_score))
                / (len(self._reference_maximum_scores) + 1)
            )
            points.append(
                SequentialDetectionPoint(
                    from_visit_id=first.visit_id,
                    to_visit_id=second.visit_id,
                    from_observed_at=first.effective_timestamp,
                    to_observed_at=second.effective_timestamp,
                    selected_observation_indices=tuple(
                        sorted(
                            set(first.selected_observation_indices).union(
                                second.selected_observation_indices
                            )
                        )
                    ),
                    elapsed_years=elapsed,
                    interval_score=max(interval_score, 0.0),
                    cumulative_score=max(cumulative_score, 0.0),
                    empirical_tail_probability=tail,
                    threshold_crossed=(
                        cumulative_score > self.model_.maximum_cumulative_score_threshold
                    ),
                )
            )
        onset = next((point.to_visit_id for point in points if point.threshold_crossed), None)
        persistent = _persistent_at_end(points, self.config.persistence_crossings)
        detected = onset is not None
        transient = (
            detected
            and not persistent
            and any(
                point.threshold_crossed and not later.threshold_crossed
                for index, point in enumerate(points)
                for later in points[index + 1 :]
            )
        )
        return SubjectSequentialDetection(
            subject_id=subject_id,
            points=tuple(points),
            detected=detected,
            onset_visit_id=onset,
            persistent=persistent,
            transient=transient,
            peak_cumulative_score=max(peak_score, 0.0),
            peak_modality_evidence=_modality_evidence(
                peak_raw,
                self._covariance,
                self.config.features,
            ),
            missing_visit_ids=missing_visit_ids,
        )


def _validate_visits(visits: tuple[ExpectedVisit, ...]) -> None:
    if len(visits) < 3:
        raise ValueError("sequential detection requires at least three expected visits")
    if len({visit.visit_id for visit in visits}) != len(visits):
        raise ValueError("sequential visit identifiers must be unique")


def _complete_trajectories(
    study: Study,
    *,
    visits: tuple[ExpectedVisit, ...],
    features: tuple[VisitFeature | LongitudinalChannel, ...],
    subject_ids: tuple[str, ...],
    require_all_visits: bool,
) -> tuple[
    dict[str, list[VisitAlignedVector]],
    tuple[str, ...],
    tuple[LongitudinalExclusion, ...],
    dict[str, tuple[str, ...]],
    tuple[LongitudinalChannel, ...],
]:
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("sequential detection subject identifiers must be unique")
    known = {subject.subject_id for subject in study.subjects}
    unknown = set(subject_ids).difference(known)
    if unknown:
        raise ValueError(f"unknown sequential detection subjects: {sorted(unknown)}")
    extraction = extract_visit_aligned_values(
        study,
        visits=visits,
        channels=features,
        subject_ids=subject_ids,
    )
    vector_extraction = complete_visit_vectors(extraction)
    vector_map = vector_extraction.vector_map()
    trajectories: dict[str, list[VisitAlignedVector]] = {}
    excluded: list[str] = []
    exclusions = list(vector_extraction.exclusions)
    missing_visits: dict[str, tuple[str, ...]] = {}
    for subject_id in sorted(subject_ids):
        trajectory = [
            vector_map[(subject_id, visit.visit_id)]
            for visit in visits
            if (subject_id, visit.visit_id) in vector_map
        ]
        missing = tuple(
            visit.visit_id for visit in visits if (subject_id, visit.visit_id) not in vector_map
        )
        missing_visits[subject_id] = missing
        nonchronological = next(
            (
                (first, second)
                for first, second in pairwise(trajectory)
                if second.effective_timestamp <= first.effective_timestamp
            ),
            None,
        )
        if nonchronological is not None:
            exclusions.append(
                LongitudinalExclusion(
                    reason=LongitudinalExclusionReason.NONCHRONOLOGICAL_OBSERVED_TIME,
                    subject_id=subject_id,
                    visit_id=nonchronological[1].visit_id,
                    observation_indices=tuple(
                        sorted(
                            set(nonchronological[0].selected_observation_indices).union(
                                nonchronological[1].selected_observation_indices
                            )
                        )
                    ),
                )
            )
            excluded.append(subject_id)
            continue
        enough = len(trajectory) == len(visits) if require_all_visits else len(trajectory) >= 2
        if enough:
            trajectories[subject_id] = trajectory
        else:
            excluded.append(subject_id)
            exclusions.append(
                LongitudinalExclusion(
                    reason=LongitudinalExclusionReason.INSUFFICIENT_COMPLETE_VISITS,
                    subject_id=subject_id,
                    missing_visit_ids=missing,
                )
            )
    exact_channels = tuple(item for item in extraction.channels if item is not None)
    return (
        trajectories,
        tuple(excluded),
        tuple(exclusions),
        missing_visits,
        exact_channels,
    )


def _elapsed_years(first: datetime, second: datetime) -> float:
    difference = second - first
    years = difference.total_seconds() / timedelta(days=_DAYS_PER_YEAR).total_seconds()
    if years <= 0:
        raise ValueError("sequential visits must be chronological for every subject")
    return years


def _innovations(
    sequence: list[tuple[NDArray[np.float64], float]],
    mean_rate: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    return [
        np.asarray((change - mean_rate * elapsed) / math.sqrt(elapsed), dtype=np.float64)
        for change, elapsed in sequence
    ]


def _fit_reference_dynamics(
    sequences: list[list[tuple[NDArray[np.float64], float]]],
    *,
    shrinkage: float,
    ridge: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate pooled drift and regularized innovation covariance."""
    total_change = np.sum([change for sequence in sequences for change, _ in sequence], axis=0)
    total_elapsed = sum(elapsed for sequence in sequences for _, elapsed in sequence)
    mean_rate = np.asarray(total_change / total_elapsed, dtype=np.float64)
    matrix = np.asarray(
        [item for sequence in sequences for item in _innovations(sequence, mean_rate)],
        dtype=np.float64,
    )
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    covariance = (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
    scale = max(float(np.trace(covariance)) / covariance.shape[0], 1.0)
    covariance = covariance + np.eye(covariance.shape[0]) * ridge * scale
    return mean_rate, np.asarray(covariance, dtype=np.float64)


def _cumulative_scores(
    innovations: list[NDArray[np.float64]],
    cholesky: NDArray[np.float64],
) -> list[float]:
    return [
        _cumulative_score(innovations[:index], cholesky) for index in range(1, len(innovations) + 1)
    ]


def _cumulative_score(
    innovations: list[NDArray[np.float64]],
    cholesky: NDArray[np.float64],
) -> float:
    cumulative = np.sum(innovations, axis=0) / math.sqrt(len(innovations))
    whitened = np.linalg.solve(cholesky, cumulative)
    return float(whitened @ whitened)


def _persistent_at_end(
    points: list[SequentialDetectionPoint],
    required_crossings: int,
) -> bool:
    run = 0
    for point in reversed(points):
        if not point.threshold_crossed:
            break
        run += 1
    return run >= required_crossings


def _modality_evidence(
    cumulative_innovation: NDArray[np.float64],
    covariance: NDArray[np.float64],
    features: tuple[LongitudinalChannel | VisitFeature, ...],
) -> tuple[ModalityEvidence, ...]:
    modalities = sorted(
        {feature.modality for feature in features},
        key=lambda item: "" if item is None else item.value,
    )
    evidence: list[ModalityEvidence] = []
    for modality in modalities:
        indices = [index for index, feature in enumerate(features) if feature.modality is modality]
        vector = cumulative_innovation[indices]
        subcovariance = covariance[np.ix_(indices, indices)]
        score = float(vector @ np.linalg.inv(subcovariance) @ vector)
        evidence.append(
            ModalityEvidence(
                modality=modality,
                channels=len(indices),
                score=max(score, 0.0),
            )
        )
    return tuple(evidence)


def _reference_input_artifact_hash(
    *,
    study_id: str,
    visits: tuple[ExpectedVisit, ...],
    config: SequentialDetectionConfig,
    channels: tuple[LongitudinalChannel, ...],
    requested_reference_subject_ids: tuple[str, ...],
    trajectories: dict[str, list[VisitAlignedVector]],
) -> str:
    """Hash labeled irregular-time reference trajectories and their exact domain."""
    payload = {
        "schema": "rejuvenationkit.sequential-reference.v1",
        "study_id": study_id,
        "ordered_visits": [visit.model_dump(mode="json") for visit in visits],
        "config": config.model_dump(mode="json"),
        "resolved_channels": [item.model_dump(mode="json") for item in channels],
        "requested_reference_subject_ids": list(requested_reference_subject_ids),
        "complete_reference_trajectories": [
            {
                "subject_id": subject_id,
                "visits": [
                    {
                        "visit_id": vector.visit_id,
                        "effective_timestamp": vector.effective_timestamp.isoformat(),
                        "values": [float(value) for value in vector.values],
                    }
                    for vector in trajectories[subject_id]
                ],
            }
            for subject_id in sorted(trajectories)
        ],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
