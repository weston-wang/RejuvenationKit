"""DSP-style multivariate detection of longitudinal biological change."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.longitudinal import (
    AggregationPolicy,
    LongitudinalChannel,
    LongitudinalExclusion,
    complete_visit_vectors,
    extract_visit_aligned_values,
)
from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Study

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


class ChangeDetectionConfig(BaseModel):
    """Configuration for covariance-aware longitudinal change detection."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    features: tuple[LongitudinalChannel | VisitFeature, ...] = Field(min_length=2)
    covariance_shrinkage: float = Field(default=0.20, ge=0, le=1)
    covariance_ridge: float = Field(default=1e-9, gt=0)
    false_alarm_rate: float = Field(default=0.05, gt=0, lt=0.5)
    minimum_reference_subjects: int = Field(default=20, ge=3)

    @model_validator(mode="after")
    def require_unique_features(self) -> ChangeDetectionConfig:
        """Reject exact or wildcard requests that could resolve to one channel."""
        for index, first in enumerate(self.features):
            for second in self.features[index + 1 :]:
                if first.feature == second.feature and (
                    first.modality is None
                    or second.modality is None
                    or first.modality is second.modality
                ):
                    raise ValueError(
                        "detection features cannot overlap by exact or wildcard modality"
                    )
        return self


class ChangeDetectionModel(BaseModel):
    """Fitted reference-change distribution and calibrated threshold."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    feature_names: tuple[str, ...]
    feature_modalities: tuple[Modality | None, ...]
    resolved_feature_modalities: tuple[Modality, ...] = ()
    feature_units: tuple[str, ...] = ()
    aggregation_policies: tuple[AggregationPolicy, ...] = ()
    reference_subjects: int = Field(ge=1)
    mean_change: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    threshold: float = Field(ge=0)
    false_alarm_rate: float = Field(gt=0, lt=0.5)
    threshold_quantile_method: Literal["higher"] = _EMPIRICAL_QUANTILE_METHOD
    reference_score_method: Literal["leave_one_out"] = "leave_one_out"
    study_id: str | None = None
    baseline_visit: ExpectedVisit | None = None
    follow_up_visit: ExpectedVisit | None = None
    resolved_channels: tuple[LongitudinalChannel, ...] = ()
    config: ChangeDetectionConfig | None = None
    requested_reference_subject_ids: tuple[str, ...] = ()
    reference_subject_ids: tuple[str, ...] = ()
    reference_input_artifact_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reference_score_distribution: tuple[float, ...] = ()
    model_artifact_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_fitted_provenance(self) -> ChangeDetectionModel:
        """Reject incomplete or internally inconsistent serialized fits."""
        dimension = len(self.feature_names)
        if dimension < 2:
            raise ValueError("detection model requires at least two features")
        if len(self.feature_modalities) != dimension:
            raise ValueError("feature modalities must match feature order")
        for label, values in (
            ("resolved feature modalities", self.resolved_feature_modalities),
            ("feature units", self.feature_units),
            ("aggregation policies", self.aggregation_policies),
        ):
            if values and len(values) != dimension:
                raise ValueError(f"{label} must match feature order")
        if len(self.mean_change) != dimension:
            raise ValueError("mean change must match feature order")
        if len(self.covariance) != dimension or any(
            len(row) != dimension for row in self.covariance
        ):
            raise ValueError("detection covariance must be square in feature order")
        mean = np.asarray(self.mean_change, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("detection model parameters must be finite")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0.0):
            raise ValueError("detection covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("detection covariance must be positive definite") from error

        fitted_markers = (
            self.study_id is not None,
            self.baseline_visit is not None,
            self.follow_up_visit is not None,
            bool(self.resolved_channels),
            self.config is not None,
            bool(self.requested_reference_subject_ids),
            bool(self.reference_subject_ids),
            self.reference_input_artifact_hash is not None,
            bool(self.reference_score_distribution),
            self.model_artifact_hash is not None,
        )
        if not any(fitted_markers):
            return self
        if not all(fitted_markers):
            raise ValueError("complete fitted detection provenance is required")

        assert self.config is not None
        assert self.baseline_visit is not None
        assert self.follow_up_visit is not None
        assert self.model_artifact_hash is not None
        if self.baseline_visit.visit_id == self.follow_up_visit.visit_id:
            raise ValueError("fitted detection visits must be distinct")
        if self.false_alarm_rate != self.config.false_alarm_rate:
            raise ValueError("model false-alarm rate does not match fitted config")
        if self.feature_names != tuple(item.feature for item in self.config.features):
            raise ValueError("model feature names do not match fitted config")
        if self.feature_modalities != tuple(item.modality for item in self.config.features):
            raise ValueError("model feature modalities do not match fitted config")
        if self.resolved_channels and len(self.resolved_channels) != len(self.feature_names):
            raise ValueError("resolved detection channels must match feature order")
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
        if len(self.reference_score_distribution) != self.reference_subjects:
            raise ValueError("reference score distribution must contain one score per subject")
        if tuple(
            sorted(self.reference_score_distribution)
        ) != self.reference_score_distribution or any(
            not math.isfinite(score) or score < 0 for score in self.reference_score_distribution
        ):
            raise ValueError("reference score distribution must be finite, nonnegative, and sorted")
        expected_threshold = float(
            np.quantile(
                np.asarray(self.reference_score_distribution, dtype=np.float64),
                1 - self.false_alarm_rate,
                method=self.threshold_quantile_method,
            )
        )
        if self.threshold != expected_threshold:
            raise ValueError("detection threshold does not match reference score quantile")
        expected_hash = _serialized_model_artifact_hash(
            self,
            schema="rejuvenationkit.change-detection-model.v1",
        )
        if self.model_artifact_hash != expected_hash:
            raise ValueError("detection model artifact hash does not match serialized contents")
        return self


class SubjectChangeDetection(BaseModel):
    """One subject's covariance-normalized longitudinal change score."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    subject_id: str
    change: tuple[float, ...]
    innovation: tuple[float, ...]
    whitened_innovation: tuple[float, ...]
    squared_mahalanobis_distance: float = Field(ge=0)
    empirical_tail_probability: float = Field(gt=0, le=1)
    detected: bool


class ChangeDetectionReport(BaseModel):
    """Scored subjects under one fitted multivariate detector."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    study_id: str
    baseline_visit_id: str
    follow_up_visit_id: str
    model: ChangeDetectionModel
    results: tuple[SubjectChangeDetection, ...]
    excluded_subject_ids: tuple[str, ...] = ()
    exclusions: tuple[LongitudinalExclusion, ...] = ()

    @model_validator(mode="after")
    def validate_scored_results(self) -> ChangeDetectionReport:
        """Bind every subject score to the serialized fitted covariance model."""
        if self.model.study_id is not None and self.study_id != self.model.study_id:
            raise ValueError("change-detection report study does not match its model")
        if self.model.baseline_visit is not None and (
            self.baseline_visit_id != self.model.baseline_visit.visit_id
        ):
            raise ValueError("change-detection report baseline does not match its model")
        if self.model.follow_up_visit is not None and (
            self.follow_up_visit_id != self.model.follow_up_visit.visit_id
        ):
            raise ValueError("change-detection report follow-up does not match its model")
        result_ids = tuple(item.subject_id for item in self.results)
        if tuple(sorted(set(result_ids))) != result_ids:
            raise ValueError("change-detection result subjects must be unique and sorted")
        if tuple(sorted(set(self.excluded_subject_ids))) != self.excluded_subject_ids:
            raise ValueError("change-detection excluded subjects must be unique and sorted")
        if set(result_ids).intersection(self.excluded_subject_ids):
            raise ValueError("change-detection subjects cannot be both scored and excluded")
        mean = np.asarray(self.model.mean_change, dtype=np.float64)
        cholesky = np.linalg.cholesky(np.asarray(self.model.covariance, dtype=np.float64))
        reference_scores = np.asarray(
            self.model.reference_score_distribution,
            dtype=np.float64,
        )
        for item in self.results:
            change = np.asarray(item.change, dtype=np.float64)
            innovation = np.asarray(item.innovation, dtype=np.float64)
            whitened = np.asarray(item.whitened_innovation, dtype=np.float64)
            if any(vector.shape != mean.shape for vector in (change, innovation, whitened)):
                raise ValueError("change-detection result vectors must match model dimension")
            expected_innovation = change - mean
            expected_whitened = np.linalg.solve(cholesky, expected_innovation)
            expected_score = max(float(expected_whitened @ expected_whitened), 0.0)
            if not np.allclose(innovation, expected_innovation, rtol=1e-12, atol=1e-12):
                raise ValueError("change-detection innovation does not match reported change")
            if not np.allclose(whitened, expected_whitened, rtol=1e-12, atol=1e-12):
                raise ValueError("change-detection whitening does not match model covariance")
            if not math.isclose(
                item.squared_mahalanobis_distance,
                expected_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("change-detection score does not match model covariance")
            if item.detected != (expected_score > self.model.threshold):
                raise ValueError("change-detection flag does not match model threshold")
            if reference_scores.size:
                expected_tail = float(
                    (1 + np.count_nonzero(reference_scores >= expected_score))
                    / (len(reference_scores) + 1)
                )
                if not math.isclose(
                    item.empirical_tail_probability,
                    expected_tail,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError("change-detection tail probability does not match model")
        return self

    def results_frame(self) -> pd.DataFrame:
        """Return subject scores as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.results)


class MultivariateChangeDetector:
    """Detect unusual paired changes relative to a reference population.

    The detector centers paired feature changes, estimates their covariance,
    shrinks cross-channel covariance toward a diagonal model, whitens each
    innovation, and thresholds squared Mahalanobis distance at an empirical
    reference quantile.
    """

    def __init__(self, config: ChangeDetectionConfig) -> None:
        """Create an unfitted detector."""
        self.config = config
        self.model_: ChangeDetectionModel | None = None
        self._inverse_covariance: NDArray[np.float64] | None = None
        self._cholesky: NDArray[np.float64] | None = None
        self._reference_scores: NDArray[np.float64] | None = None
        self._reference_subject_ids: frozenset[str] | None = None
        self._channels: tuple[LongitudinalChannel, ...] | None = None

    @classmethod
    def from_model(cls, model: ChangeDetectionModel) -> MultivariateChangeDetector:
        """Reconstruct a scorer from a complete serialized fitted model."""
        model = ChangeDetectionModel.model_validate(model.model_dump(mode="python"))
        if (
            model.config is None
            or model.baseline_visit is None
            or model.follow_up_visit is None
            or model.study_id is None
            or model.reference_input_artifact_hash is None
            or not model.resolved_channels
            or not model.requested_reference_subject_ids
            or not model.reference_score_distribution
            or model.model_artifact_hash is None
        ):
            raise ValueError("model does not contain complete fitted detection provenance")
        detector = cls(model.config)
        covariance = np.asarray(model.covariance, dtype=np.float64)
        detector.model_ = model
        detector._inverse_covariance = np.asarray(
            np.linalg.inv(covariance),
            dtype=np.float64,
        )
        detector._cholesky = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)
        detector._reference_scores = np.asarray(
            model.reference_score_distribution,
            dtype=np.float64,
        )
        detector._reference_subject_ids = frozenset(model.requested_reference_subject_ids)
        detector._channels = model.resolved_channels
        return detector

    def fit(
        self,
        study: Study,
        *,
        baseline: ExpectedVisit,
        follow_up: ExpectedVisit,
        reference_subject_ids: tuple[str, ...] | None = None,
    ) -> MultivariateChangeDetector:
        """Fit the normal-change model using complete paired reference subjects."""
        requested_reference_ids = (
            tuple(sorted(subject.subject_id for subject in study.subjects))
            if reference_subject_ids is None
            else tuple(sorted(reference_subject_ids))
        )
        changes, _, _, channels = _paired_changes(
            study,
            baseline=baseline,
            follow_up=follow_up,
            features=self.config.features,
            subject_ids=reference_subject_ids,
        )
        if len(changes) < self.config.minimum_reference_subjects:
            raise ValueError(
                "insufficient complete reference subjects: "
                f"{len(changes)} < {self.config.minimum_reference_subjects}"
            )
        matrix = np.asarray(list(changes.values()), dtype=float)
        dimension = matrix.shape[1]
        if len(matrix) < dimension + 2:
            raise ValueError(
                "leave-one-out calibration requires at least feature count + 2 complete "
                f"reference subjects: {len(matrix)} < {dimension + 2}"
            )
        mean = matrix.mean(axis=0)
        covariance = _regularized_covariance(
            matrix,
            shrinkage=self.config.covariance_shrinkage,
            ridge=self.config.covariance_ridge,
        )
        cholesky = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)
        inverse = np.asarray(np.linalg.inv(covariance), dtype=np.float64)
        # In-sample distances are biased low because each subject helped fit the mean
        # and covariance, so the threshold is calibrated on leave-one-out scores that
        # are exchangeable with scores of new, held-out subjects.
        scores = _leave_one_out_scores(
            matrix,
            shrinkage=self.config.covariance_shrinkage,
            ridge=self.config.covariance_ridge,
        )
        threshold = float(np.quantile(scores, 1 - self.config.false_alarm_rate, method="higher"))
        self._inverse_covariance = inverse
        self._cholesky = cholesky
        self._reference_scores = scores
        self._reference_subject_ids = frozenset(requested_reference_ids)
        self._channels = channels
        complete_reference_ids = tuple(sorted(changes))
        reference_input_artifact_hash = _reference_input_artifact_hash(
            study_id=study.study_id,
            baseline=baseline,
            follow_up=follow_up,
            config=self.config,
            channels=channels,
            requested_reference_subject_ids=requested_reference_ids,
            changes=changes,
        )
        unsigned_model = ChangeDetectionModel.model_construct(
            feature_names=tuple(item.feature for item in self.config.features),
            feature_modalities=tuple(item.modality for item in self.config.features),
            resolved_feature_modalities=tuple(item.modality for item in channels),
            feature_units=tuple(item.unit for item in channels),
            aggregation_policies=tuple(item.aggregation_policy for item in channels),
            reference_subjects=len(changes),
            mean_change=tuple(float(value) for value in mean),
            covariance=tuple(tuple(float(value) for value in row) for row in covariance),
            threshold=threshold,
            false_alarm_rate=self.config.false_alarm_rate,
            study_id=study.study_id,
            baseline_visit=baseline,
            follow_up_visit=follow_up,
            resolved_channels=channels,
            config=self.config,
            requested_reference_subject_ids=requested_reference_ids,
            reference_subject_ids=complete_reference_ids,
            reference_input_artifact_hash=reference_input_artifact_hash,
            reference_score_distribution=tuple(float(value) for value in sorted(scores)),
        )
        self.model_ = ChangeDetectionModel.model_validate(
            {
                **unsigned_model.model_dump(mode="python"),
                "model_artifact_hash": _serialized_model_artifact_hash(
                    unsigned_model,
                    schema="rejuvenationkit.change-detection-model.v1",
                ),
            }
        )
        return self

    def score(
        self,
        study: Study,
        *,
        baseline: ExpectedVisit,
        follow_up: ExpectedVisit,
        subject_ids: tuple[str, ...] | None = None,
    ) -> ChangeDetectionReport:
        """Score complete paired subjects and report subjects lacking a full vector."""
        if (
            self.model_ is None
            or self._inverse_covariance is None
            or self._cholesky is None
            or self._reference_scores is None
            or self._channels is None
        ):
            raise RuntimeError("fit must be called before score")
        self._validate_scoring_domain(
            study,
            baseline=baseline,
            follow_up=follow_up,
        )
        selected = (
            {subject.subject_id for subject in study.subjects}
            if subject_ids is None
            else set(subject_ids)
        )
        assert self._reference_subject_ids is not None
        overlap = selected.intersection(self._reference_subject_ids)
        if overlap:
            raise ValueError(
                "change-detection evaluation subjects overlap fitted reference subjects: "
                f"{sorted(overlap)}"
            )
        changes, excluded, exclusions, _ = _paired_changes(
            study,
            baseline=baseline,
            follow_up=follow_up,
            features=self._channels,
            subject_ids=subject_ids,
        )
        mean = np.asarray(self.model_.mean_change)
        results: list[SubjectChangeDetection] = []
        for subject_id, values in sorted(changes.items()):
            change = np.asarray(values)
            innovation = change - mean
            whitened = np.linalg.solve(self._cholesky, innovation)
            score = float(innovation @ self._inverse_covariance @ innovation)
            tail = float(
                (1 + np.count_nonzero(self._reference_scores >= score))
                / (len(self._reference_scores) + 1)
            )
            results.append(
                SubjectChangeDetection(
                    subject_id=subject_id,
                    change=tuple(float(value) for value in change),
                    innovation=tuple(float(value) for value in innovation),
                    whitened_innovation=tuple(float(value) for value in whitened),
                    squared_mahalanobis_distance=max(score, 0.0),
                    empirical_tail_probability=tail,
                    detected=score > self.model_.threshold,
                )
            )
        return ChangeDetectionReport(
            study_id=study.study_id,
            baseline_visit_id=baseline.visit_id,
            follow_up_visit_id=follow_up.visit_id,
            model=self.model_,
            results=tuple(results),
            excluded_subject_ids=excluded,
            exclusions=exclusions,
        )

    def _validate_scoring_domain(
        self,
        study: Study,
        *,
        baseline: ExpectedVisit,
        follow_up: ExpectedVisit,
    ) -> None:
        """Reject schedule, channel, config, or reference-input drift after fitting."""
        assert self.model_ is not None
        assert self._channels is not None
        if self.model_.study_id != study.study_id:
            raise ValueError("scoring study does not match fitted detection domain")
        if self.model_.baseline_visit != baseline or self.model_.follow_up_visit != follow_up:
            raise ValueError("scoring visit definitions do not match fitted detection domain")
        if self.model_.config != self.config:
            raise ValueError("detector configuration does not match fitted detection domain")
        requested_ids = self.model_.requested_reference_subject_ids
        if not requested_ids or self.model_.reference_input_artifact_hash is None:
            raise ValueError("fitted detector is missing reference-input provenance")
        try:
            changes, _, _, channels = _paired_changes(
                study,
                baseline=baseline,
                follow_up=follow_up,
                features=self._channels,
                subject_ids=requested_ids,
            )
        except ValueError as error:
            raise ValueError(
                "fitted detection reference input is unavailable or altered"
            ) from error
        if channels != self.model_.resolved_channels:
            raise ValueError("scoring channels do not match fitted detection domain")
        observed_hash = _reference_input_artifact_hash(
            study_id=study.study_id,
            baseline=baseline,
            follow_up=follow_up,
            config=self.config,
            channels=channels,
            requested_reference_subject_ids=requested_ids,
            changes=changes,
        )
        if observed_hash != self.model_.reference_input_artifact_hash:
            raise ValueError("fitted detection reference input is unavailable or altered")


def _regularized_covariance(
    matrix: NDArray[np.float64],
    *,
    shrinkage: float,
    ridge: float,
) -> NDArray[np.float64]:
    """Return the diagonal-shrunk, ridge-stabilized sample covariance of rows."""
    empirical = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    covariance = (1 - shrinkage) * empirical + shrinkage * np.diag(np.diag(empirical))
    scale = max(float(np.trace(covariance)) / covariance.shape[0], 1.0)
    covariance = covariance + np.eye(covariance.shape[0]) * (ridge * scale)
    if not np.isfinite(covariance).all():
        raise ValueError("reference covariance contains non-finite values")
    return np.asarray(covariance, dtype=np.float64)


def _leave_one_out_scores(
    matrix: NDArray[np.float64],
    *,
    shrinkage: float,
    ridge: float,
) -> NDArray[np.float64]:
    """Score each row against a mean and covariance fitted without that row."""
    scores = np.empty(len(matrix), dtype=np.float64)
    for index in range(len(matrix)):
        training = np.delete(matrix, index, axis=0)
        covariance = _regularized_covariance(training, shrinkage=shrinkage, ridge=ridge)
        innovation = matrix[index] - training.mean(axis=0)
        scores[index] = max(float(innovation @ np.linalg.solve(covariance, innovation)), 0.0)
    return scores


def _paired_changes(
    study: Study,
    *,
    baseline: ExpectedVisit,
    follow_up: ExpectedVisit,
    features: tuple[VisitFeature | LongitudinalChannel, ...],
    subject_ids: tuple[str, ...] | None,
) -> tuple[
    dict[str, NDArray[np.float64]],
    tuple[str, ...],
    tuple[LongitudinalExclusion, ...],
    tuple[LongitudinalChannel, ...],
]:
    if subject_ids is not None and len(subject_ids) != len(set(subject_ids)):
        raise ValueError("detection subject identifiers must be unique")
    selected = (
        {subject.subject_id for subject in study.subjects}
        if subject_ids is None
        else set(subject_ids)
    )
    known = {subject.subject_id for subject in study.subjects}
    unknown = selected.difference(known)
    if unknown:
        raise ValueError(f"unknown detection subjects: {sorted(unknown)}")
    extraction = extract_visit_aligned_values(
        study,
        visits=(baseline, follow_up),
        channels=features,
        subject_ids=tuple(sorted(selected)),
    )
    vector_extraction = complete_visit_vectors(extraction)
    vectors = vector_extraction.vector_map()
    changes: dict[str, NDArray[np.float64]] = {}
    excluded: list[str] = []
    for subject_id in sorted(selected):
        baseline_vector = vectors.get((subject_id, baseline.visit_id))
        follow_up_vector = vectors.get((subject_id, follow_up.visit_id))
        if baseline_vector is None or follow_up_vector is None:
            excluded.append(subject_id)
            continue
        changes[subject_id] = np.asarray(
            follow_up_vector.as_array() - baseline_vector.as_array(),
            dtype=np.float64,
        )
    exact_channels = tuple(item for item in extraction.channels if item is not None)
    return changes, tuple(excluded), vector_extraction.exclusions, exact_channels


def _reference_input_artifact_hash(
    *,
    study_id: str,
    baseline: ExpectedVisit,
    follow_up: ExpectedVisit,
    config: ChangeDetectionConfig,
    channels: tuple[LongitudinalChannel, ...],
    requested_reference_subject_ids: tuple[str, ...],
    changes: dict[str, NDArray[np.float64]],
) -> str:
    """Hash the exact labeled vectors and analysis domain used for fitting."""
    payload = {
        "schema": "rejuvenationkit.change-detection-reference.v1",
        "study_id": study_id,
        "baseline_visit": baseline.model_dump(mode="json"),
        "follow_up_visit": follow_up.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "resolved_channels": [item.model_dump(mode="json") for item in channels],
        "requested_reference_subject_ids": list(requested_reference_subject_ids),
        "complete_reference_changes": [
            {
                "subject_id": subject_id,
                "change": [float(value) for value in changes[subject_id]],
            }
            for subject_id in sorted(changes)
        ],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
