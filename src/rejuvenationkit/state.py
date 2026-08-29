"""Prespecified continuous-time linear-Gaussian latent-state estimation.

The estimator in this module deliberately does not discover a biological state
from the supplied study. Callers must specify the state names, observation
loadings, units, dynamics, process noise, and prior. This makes the resulting
trajectory an auditable consequence of a declared model rather than an
implicitly learned biological-age construct.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from scipy.linalg import expm

from rejuvenationkit.schemas import Modality, Observation, Study, study_artifact_hash

_SECONDS_PER_DAY = 86_400.0
_PSD_TOLERANCE = 1e-10


def _canonical_hash(payload: object) -> str:
    """Hash one JSON-compatible state artifact deterministically."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _state_model_artifact_hash(config: LinearGaussianStateConfig) -> str:
    """Bind every declared state, channel, dynamic, covariance, and prior."""
    return _canonical_hash(config.model_dump(mode="json"))


def _finite_vector(values: tuple[float, ...], *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector")
    return vector


def _finite_matrix(
    values: tuple[tuple[float, ...], ...],
    *,
    name: str,
) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite rectangular matrix")
    return matrix


def _validate_covariance(
    values: tuple[tuple[float, ...], ...],
    *,
    dimension: int,
    name: str,
) -> NDArray[np.float64]:
    matrix = _finite_matrix(values, name=name)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=_PSD_TOLERANCE):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    if float(np.min(eigenvalues)) < -_PSD_TOLERANCE:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _matrix_tuple(matrix: NDArray[np.float64]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _vector_tuple(vector: NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(float(value) for value in vector)


def _stable_covariance(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrize a covariance and remove only numerical negative eigenvalues."""
    symmetric = np.asarray((matrix + matrix.T) / 2, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-8 * scale:
        raise ArithmeticError("state covariance became materially indefinite")
    clipped = np.maximum(eigenvalues, 0.0)
    return np.asarray((eigenvectors * clipped) @ eigenvectors.T, dtype=np.float64)


def _require_aware(timestamp: datetime, *, name: str) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


class StateChannel(BaseModel):
    """One prespecified measurement channel and its observation equation.

    The model is ``measurement = offset + loadings @ state + noise``. A
    measurement matches only when modality, feature, and unit all match.
    ``measurement_variance`` is assay/model variance; a row-level reported
    standard error is squared and added to it during filtering.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    modality: Modality
    feature: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    loadings: tuple[float, ...] = Field(min_length=1)
    measurement_variance: float = Field(gt=0)
    offset: float = 0.0

    @model_validator(mode="after")
    def validate_channel(self) -> StateChannel:
        """Reject nonfinite, ambiguous, or uninformative channel definitions."""
        if not self.name.strip() or not self.feature.strip() or not self.unit.strip():
            raise ValueError("channel name, feature, and unit cannot be blank")
        loadings = _finite_vector(self.loadings, name="channel loadings")
        if not np.any(loadings != 0):
            raise ValueError("channel loadings must contain a nonzero value")
        if not math.isfinite(self.measurement_variance) or not math.isfinite(self.offset):
            raise ValueError("channel variance and offset must be finite")
        return self

    @property
    def exact_key(self) -> tuple[Modality, str, str]:
        """Return the exact modality/feature/unit identity."""
        return self.modality, self.feature, self.unit


class LinearGaussianStateConfig(BaseModel):
    """Fully specified continuous-time linear-Gaussian state model.

    Dynamics use the continuous-time stochastic differential equation
    ``dx = (A x + drift) dt + L dW`` with ``L L.T = Qc``. One model time unit
    spans ``time_unit_days`` wall-clock days. Transitions and integrated
    process covariance are discretized exactly with matrix exponentials.
    """

    model_config = ConfigDict(frozen=True)

    state_names: tuple[str, ...] = Field(min_length=1)
    channels: tuple[StateChannel, ...] = Field(min_length=1)
    continuous_dynamics: tuple[tuple[float, ...], ...]
    continuous_process_covariance: tuple[tuple[float, ...], ...]
    continuous_drift: tuple[float, ...]
    initial_mean: tuple[float, ...]
    initial_covariance: tuple[tuple[float, ...], ...]
    time_unit_days: float = Field(default=365.2425, gt=0)
    smooth: bool = True

    @model_validator(mode="after")
    def validate_model(self) -> LinearGaussianStateConfig:
        """Validate dimensions, finite values, uniqueness, and PSD matrices."""
        dimension = len(self.state_names)
        if any(not name.strip() for name in self.state_names):
            raise ValueError("state names cannot be blank")
        if len(set(self.state_names)) != dimension:
            raise ValueError("state names must be unique")
        channel_names = [channel.name for channel in self.channels]
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("channel names must be unique")
        exact_keys = [channel.exact_key for channel in self.channels]
        if len(set(exact_keys)) != len(exact_keys):
            raise ValueError("channel modality/feature/unit identities must be unique")
        for channel in self.channels:
            if len(channel.loadings) != dimension:
                raise ValueError(f"channel {channel.name!r} loadings must have length {dimension}")
        dynamics = _finite_matrix(self.continuous_dynamics, name="continuous_dynamics")
        if dynamics.shape != (dimension, dimension):
            raise ValueError(f"continuous_dynamics must have shape ({dimension}, {dimension})")
        _validate_covariance(
            self.continuous_process_covariance,
            dimension=dimension,
            name="continuous_process_covariance",
        )
        if len(_finite_vector(self.continuous_drift, name="continuous_drift")) != dimension:
            raise ValueError(f"continuous_drift must have length {dimension}")
        if len(_finite_vector(self.initial_mean, name="initial_mean")) != dimension:
            raise ValueError(f"initial_mean must have length {dimension}")
        _validate_covariance(
            self.initial_covariance,
            dimension=dimension,
            name="initial_covariance",
        )
        if not math.isfinite(self.time_unit_days):
            raise ValueError("time_unit_days must be finite")
        return self


class StateEstimate(BaseModel):
    """One uncertain latent-state estimate."""

    model_config = ConfigDict(frozen=True)

    subject_id: str = Field(min_length=1)
    timestamp: datetime
    state_names: tuple[str, ...] = Field(min_length=1)
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    is_forecast: bool = False
    estimate_kind: Literal["filtered", "smoothed", "forecast"] = "filtered"

    @model_validator(mode="before")
    @classmethod
    def preserve_forecast_compatibility(cls, data: Any) -> Any:
        """Infer the new estimate kind when legacy callers set is_forecast only."""
        if (
            isinstance(data, dict)
            and data.get("is_forecast") is True
            and "estimate_kind" not in data
        ):
            return {**data, "estimate_kind": "forecast"}
        return data

    @model_validator(mode="after")
    def validate_estimate(self) -> StateEstimate:
        """Ensure every estimate is finite, dimensioned, aware, and PSD."""
        _require_aware(self.timestamp, name="state timestamp")
        dimension = len(self.state_names)
        if len(set(self.state_names)) != dimension or any(
            not name.strip() for name in self.state_names
        ):
            raise ValueError("state names must be nonblank and unique")
        if len(_finite_vector(self.mean, name="state mean")) != dimension:
            raise ValueError("state mean dimension must match state names")
        _validate_covariance(
            self.covariance,
            dimension=dimension,
            name="state covariance",
        )
        if self.is_forecast != (self.estimate_kind == "forecast"):
            raise ValueError("is_forecast must agree with estimate_kind")
        return self

    @property
    def dimension(self) -> int:
        """Return latent-state dimensionality."""
        return len(self.state_names)


class StateChannelCoverage(BaseModel):
    """Observed and missing time points for one configured channel."""

    model_config = ConfigDict(frozen=True)

    channel_name: str
    modality: Modality
    feature: str
    unit: str
    observed_timepoints: int = Field(ge=0)
    missing_timepoints: int = Field(ge=0)
    reported_standard_errors: int = Field(ge=0)


class StateCoverage(BaseModel):
    """Observation coverage for one subject over inferred measurement times."""

    model_config = ConfigDict(frozen=True)

    timepoints: int = Field(ge=1)
    expected_channel_values: int = Field(ge=1)
    observed_channel_values: int = Field(ge=1)
    missing_channel_values: int = Field(ge=0)
    observed_fraction: float = Field(ge=0, le=1)
    channels: tuple[StateChannelCoverage, ...]


class ObservationExclusion(BaseModel):
    """An observation deliberately excluded because no channel matched it."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    modality: Modality
    feature: str
    unit: str
    reason: Literal["channel_not_configured"] = "channel_not_configured"


class SubjectStateExclusion(BaseModel):
    """A subject for whom no state trajectory could be estimated."""

    model_config = ConfigDict(frozen=True)

    subject_id: str
    reason: Literal["no_configured_observations"] = "no_configured_observations"


class StateTrajectory(BaseModel):
    """Time-ordered state estimates for one subject."""

    model_config = ConfigDict(frozen=True)

    subject_id: str = Field(min_length=1)
    estimates: tuple[StateEstimate, ...] = Field(min_length=1)
    coverage: StateCoverage | None = None
    excluded_observations: tuple[ObservationExclusion, ...] = ()
    source_study_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_config_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_trajectory_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_trajectory(self) -> StateTrajectory:
        """Require coherent subject, state dimension, and strict time ordering."""
        if any(item.subject_id != self.subject_id for item in self.estimates):
            raise ValueError("trajectory estimate subject identifiers must agree")
        state_names = self.estimates[0].state_names
        if any(item.state_names != state_names for item in self.estimates):
            raise ValueError("trajectory state names must remain constant")
        timestamps = [item.timestamp for item in self.estimates]
        if any(second <= first for first, second in pairwise(timestamps)):
            raise ValueError("trajectory timestamps must be strictly increasing")
        if (self.source_study_artifact_hash is None) != (self.model_config_artifact_hash is None):
            raise ValueError("trajectory study and model artifact identities must be paired")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_hash(self) -> str:
        """Bind estimates, coverage, exclusions, model, study, and forecast parent."""
        return _canonical_hash(
            {
                "schema": "rejuvenationkit.state-trajectory/v1",
                "subject_id": self.subject_id,
                "estimates": [item.model_dump(mode="json") for item in self.estimates],
                "coverage": (
                    None if self.coverage is None else self.coverage.model_dump(mode="json")
                ),
                "excluded_observations": [
                    item.model_dump(mode="json") for item in self.excluded_observations
                ],
                "source_study_artifact_hash": self.source_study_artifact_hash,
                "model_config_artifact_hash": self.model_config_artifact_hash,
                "source_trajectory_artifact_hash": self.source_trajectory_artifact_hash,
            }
        )


class OneStepForecastDiagnostic(BaseModel):
    """Pre-update prediction error at one observed follow-up time."""

    model_config = ConfigDict(frozen=True)

    subject_id: str
    timestamp: datetime
    channel_names: tuple[str, ...] = Field(min_length=1)
    observed_values: tuple[float, ...]
    predicted_values: tuple[float, ...]
    innovation_covariance: tuple[tuple[float, ...], ...]
    standardized_innovations: tuple[float, ...]
    normalized_innovation_squared: float = Field(ge=0)
    degrees_of_freedom: int = Field(ge=1)
    reported_standard_error_variances: tuple[float, ...]


class ForecastCalibrationResult(BaseModel):
    """Held-out empirical calibration of marginal one-step forecast errors."""

    model_config = ConfigDict(frozen=True)

    study_id: str = Field(min_length=1)
    study_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_config_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_diagnostics_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_diagnostics_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_subject_ids: tuple[str, ...] = Field(min_length=1)
    evaluation_subject_ids: tuple[str, ...] = Field(min_length=1)
    interval_level: float = Field(gt=0.5, lt=1)
    reference_standardized_innovations: int = Field(ge=1)
    evaluation_standardized_innovations: int = Field(ge=1)
    absolute_standardized_threshold: float = Field(ge=0)
    empirical_coverage: float = Field(ge=0, le=1)
    standardized_bias: float
    standardized_rmse: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_calibration_partition(self) -> ForecastCalibrationResult:
        """Keep held-out partitions disjoint and calibration summaries finite."""
        if len(set(self.reference_subject_ids)) != len(self.reference_subject_ids):
            raise ValueError("forecast reference subjects must be unique")
        if len(set(self.evaluation_subject_ids)) != len(self.evaluation_subject_ids):
            raise ValueError("forecast evaluation subjects must be unique")
        overlap = set(self.reference_subject_ids).intersection(self.evaluation_subject_ids)
        if overlap:
            raise ValueError(
                f"forecast reference and evaluation subjects must be disjoint: {sorted(overlap)}"
            )
        if not all(
            math.isfinite(value)
            for value in (
                self.absolute_standardized_threshold,
                self.empirical_coverage,
                self.standardized_bias,
                self.standardized_rmse,
            )
        ):
            raise ValueError("forecast calibration summaries must be finite")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_hash(self) -> str:
        """Bind the model, study, partitions, diagnostics, and calibration result."""
        return _canonical_hash(
            {
                "schema": "rejuvenationkit.forecast-calibration/v1",
                "result": self.model_dump(mode="json", exclude={"artifact_hash"}),
            }
        )


class InnovationChangePoint(BaseModel):
    """Empirically calibrated innovation evidence at one evaluation time."""

    model_config = ConfigDict(frozen=True)

    subject_id: str = Field(min_length=1)
    timestamp: datetime
    channel_names: tuple[str, ...] = Field(min_length=1)
    innovation_score: float = Field(ge=0)
    empirical_tail_probability: float = Field(gt=0, le=1)
    detected: bool

    @model_validator(mode="after")
    def validate_point(self) -> InnovationChangePoint:
        """Require a usable, finite, time-aware innovation identity."""
        _require_aware(self.timestamp, name="innovation timestamp")
        if len(set(self.channel_names)) != len(self.channel_names) or any(
            not item.strip() for item in self.channel_names
        ):
            raise ValueError("innovation channel names must be nonblank and unique")
        if not math.isfinite(self.innovation_score) or not math.isfinite(
            self.empirical_tail_probability
        ):
            raise ValueError("innovation score and tail probability must be finite")
        return self


class InnovationChangePointReport(BaseModel):
    """Per-innovation change points calibrated only on declared reference subjects."""

    model_config = ConfigDict(frozen=True)

    study_id: str = Field(min_length=1)
    study_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_config_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_diagnostics_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_diagnostics_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_subject_ids: tuple[str, ...] = Field(min_length=1)
    evaluation_subject_ids: tuple[str, ...] = Field(min_length=1)
    false_alarm_rate: float = Field(gt=0, lt=0.5)
    false_alarm_scope: Literal["per_innovation_no_trajectory_multiplicity_control"] = (
        "per_innovation_no_trajectory_multiplicity_control"
    )
    reference_innovations: int = Field(ge=1)
    score_threshold: float = Field(ge=0)
    results: tuple[InnovationChangePoint, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_change_point_partition(self) -> InnovationChangePointReport:
        """Validate held-out identities and per-innovation detection decisions."""
        if len(set(self.reference_subject_ids)) != len(self.reference_subject_ids):
            raise ValueError("change-point reference subjects must be unique")
        if len(set(self.evaluation_subject_ids)) != len(self.evaluation_subject_ids):
            raise ValueError("change-point evaluation subjects must be unique")
        overlap = set(self.reference_subject_ids).intersection(self.evaluation_subject_ids)
        if overlap:
            raise ValueError(
                "change-point reference and evaluation subjects must be disjoint: "
                f"{sorted(overlap)}"
            )
        unknown = {item.subject_id for item in self.results}.difference(self.evaluation_subject_ids)
        if unknown:
            raise ValueError(
                f"change-point results contain unknown evaluation subjects: {sorted(unknown)}"
            )
        keys = [(item.subject_id, item.timestamp, item.channel_names) for item in self.results]
        if len(set(keys)) != len(keys):
            raise ValueError("change-point result identities must be unique")
        if not math.isfinite(self.score_threshold):
            raise ValueError("change-point score threshold must be finite")
        if any(
            item.detected != (item.innovation_score > self.score_threshold) for item in self.results
        ):
            raise ValueError("change-point detections must agree with the per-innovation threshold")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_hash(self) -> str:
        """Bind the exact model, data, calibration diagnostics, and detections."""
        return _canonical_hash(
            {
                "schema": "rejuvenationkit.innovation-change-points/v1",
                "result": self.model_dump(mode="json", exclude={"artifact_hash"}),
            }
        )


class StateEstimationReport(BaseModel):
    """State trajectories plus explicit subject-level exclusions."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    study_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_config_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_state_names: tuple[str, ...]
    selected_subject_ids: tuple[str, ...]
    fit_reference_subject_ids: tuple[str, ...] = ()
    smoothing_applied: bool
    trajectories: tuple[StateTrajectory, ...]
    excluded_subjects: tuple[SubjectStateExclusion, ...] = ()

    @model_validator(mode="after")
    def validate_report_provenance(self) -> StateEstimationReport:
        """Require exact subject partitions and trajectory artifact binding."""
        if len(set(self.selected_subject_ids)) != len(self.selected_subject_ids):
            raise ValueError("selected state-report subjects must be unique")
        if len(set(self.fit_reference_subject_ids)) != len(self.fit_reference_subject_ids):
            raise ValueError("fit reference subjects must be unique")
        if not self.model_state_names or any(not item.strip() for item in self.model_state_names):
            raise ValueError("state-report model state names must be nonblank")
        if len(set(self.model_state_names)) != len(self.model_state_names):
            raise ValueError("state-report model state names must be unique")
        trajectory_ids = tuple(item.subject_id for item in self.trajectories)
        exclusion_ids = tuple(item.subject_id for item in self.excluded_subjects)
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("state-report trajectory subjects must be unique")
        if len(set(exclusion_ids)) != len(exclusion_ids):
            raise ValueError("state-report excluded subjects must be unique")
        overlap = set(trajectory_ids).intersection(exclusion_ids)
        if overlap:
            raise ValueError(
                f"state-report subjects cannot be both trajectories and excluded: {sorted(overlap)}"
            )
        represented = set(trajectory_ids).union(exclusion_ids)
        if represented != set(self.selected_subject_ids):
            raise ValueError("state trajectories and exclusions must partition selected subjects")
        for trajectory in self.trajectories:
            if trajectory.source_study_artifact_hash != self.study_artifact_hash:
                raise ValueError("trajectory study artifact does not match state report")
            if trajectory.model_config_artifact_hash != self.model_config_artifact_hash:
                raise ValueError("trajectory model artifact does not match state report")
            if trajectory.estimates[0].state_names != self.model_state_names:
                raise ValueError("trajectory state names do not match state report")
            expected_kind = (
                "smoothed"
                if self.smoothing_applied and len(trajectory.estimates) > 1
                else "filtered"
            )
            if any(item.estimate_kind != expected_kind for item in trajectory.estimates):
                raise ValueError("trajectory estimate kind does not match state report")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_hash(self) -> str:
        """Bind complete trajectory content to the exact study and state model."""
        return _canonical_hash(
            {
                "schema": "rejuvenationkit.state-estimation-report/v1",
                "report": self.model_dump(mode="json", exclude={"artifact_hash"}),
            }
        )


class LatentStateEstimator(Protocol):
    """Contract for Phase 3 state estimators."""

    def fit(self, study: Study) -> LatentStateEstimator:
        """Validate or learn a prespecified state-space model."""
        ...

    def estimate(self, study: Study) -> tuple[StateTrajectory, ...]:
        """Filter or smooth latent states."""
        ...


class _SubjectRun:
    """Internal results from one deterministic Kalman pass."""

    def __init__(
        self,
        *,
        timestamps: list[datetime],
        filtered_means: list[NDArray[np.float64]],
        filtered_covariances: list[NDArray[np.float64]],
        predicted_means: list[NDArray[np.float64]],
        predicted_covariances: list[NDArray[np.float64]],
        transitions: list[NDArray[np.float64]],
        diagnostics: list[OneStepForecastDiagnostic],
        coverage: StateCoverage,
        exclusions: tuple[ObservationExclusion, ...],
    ) -> None:
        self.timestamps = timestamps
        self.filtered_means = filtered_means
        self.filtered_covariances = filtered_covariances
        self.predicted_means = predicted_means
        self.predicted_covariances = predicted_covariances
        self.transitions = transitions
        self.diagnostics = diagnostics
        self.coverage = coverage
        self.exclusions = exclusions


class LinearGaussianStateEstimator:
    """Continuous-time Kalman filter and Rauch--Tung--Striebel smoother.

    Parameters are never inferred from the evaluation subjects. Configure an
    explicit :class:`LinearGaussianStateConfig`, call :meth:`fit` to validate
    its use with a study, and then call :meth:`estimate`, :meth:`filter`, or
    :meth:`smooth`. Forecast calibration and innovation change points require
    disjoint reference and evaluation subject identifiers.
    """

    def __init__(self, config: LinearGaussianStateConfig | None = None) -> None:
        """Create an estimator; an explicit config is required for operation."""
        self.config = config
        self.fitted_study_id_: str | None = None
        self.fitted_study_artifact_hash_: str | None = None
        self.model_config_artifact_hash_: str | None = None
        self.fit_reference_subject_ids_: tuple[str, ...] = ()
        self.report_: StateEstimationReport | None = None

    def fit(
        self,
        study: Study,
        *,
        reference_subject_ids: tuple[str, ...] | None = None,
    ) -> LinearGaussianStateEstimator:
        """Validate the prespecified model and optionally record reference subjects.

        No dynamics, loadings, noise, or latent definition is learned here.
        Reference identifiers are metadata only; methods that perform empirical
        calibration still require explicit disjoint reference/evaluation sets.
        """
        if self.config is None:
            raise NotImplementedError(
                "state model requires an explicit LinearGaussianStateConfig; "
                "implicit biological-state discovery is not implemented"
            )
        if reference_subject_ids is not None:
            _validate_subject_partition(
                study,
                reference_subject_ids=reference_subject_ids,
                evaluation_subject_ids=(),
                require_evaluation=False,
            )
            if not reference_subject_ids:
                raise ValueError("reference_subject_ids cannot be empty when provided")
            self._run_study(study, subject_ids=reference_subject_ids)
            self.fit_reference_subject_ids_ = tuple(sorted(reference_subject_ids))
        else:
            self._validate_units(study)
            self.fit_reference_subject_ids_ = ()
        self.fitted_study_id_ = study.study_id
        self.fitted_study_artifact_hash_ = study_artifact_hash(study)
        self.model_config_artifact_hash_ = _state_model_artifact_hash(self.config)
        return self

    def estimate(self, study: Study) -> tuple[StateTrajectory, ...]:
        """Return filtered or smoothed trajectories according to the config."""
        if self.config is None:
            raise NotImplementedError(
                "state estimation requires an explicit LinearGaussianStateConfig"
            )
        return self.estimate_report(study).trajectories

    def estimate_report(
        self,
        study: Study,
        *,
        subject_ids: tuple[str, ...] | None = None,
        smooth: bool | None = None,
    ) -> StateEstimationReport:
        """Return trajectories, per-channel coverage, and explicit exclusions."""
        config = self._require_fitted()
        study_hash, config_hash = self._require_fitted_study(study)
        selected = _resolve_subject_ids(study, subject_ids)
        runs, excluded = self._run_study(study, subject_ids=selected)
        use_smoothing = config.smooth if smooth is None else smooth
        trajectories = tuple(
            self._trajectory(subject_id, run, smooth=use_smoothing)
            for subject_id, run in sorted(runs.items())
        )
        report = StateEstimationReport(
            study_id=study.study_id,
            study_artifact_hash=study_hash,
            model_config_artifact_hash=config_hash,
            model_state_names=config.state_names,
            selected_subject_ids=selected,
            fit_reference_subject_ids=self.fit_reference_subject_ids_,
            smoothing_applied=use_smoothing,
            trajectories=trajectories,
            excluded_subjects=excluded,
        )
        self.report_ = report
        return report

    def filter(
        self,
        study: Study,
        *,
        subject_ids: tuple[str, ...] | None = None,
    ) -> tuple[StateTrajectory, ...]:
        """Return forward-filtered trajectories regardless of config default."""
        return self.estimate_report(study, subject_ids=subject_ids, smooth=False).trajectories

    def smooth(
        self,
        study: Study,
        *,
        subject_ids: tuple[str, ...] | None = None,
    ) -> tuple[StateTrajectory, ...]:
        """Return Rauch--Tung--Striebel smoothed trajectories."""
        return self.estimate_report(study, subject_ids=subject_ids, smooth=True).trajectories

    def one_step_diagnostics(
        self,
        study: Study,
        *,
        subject_ids: tuple[str, ...] | None = None,
    ) -> tuple[OneStepForecastDiagnostic, ...]:
        """Return pre-update diagnostics after each subject's first time point."""
        self._require_fitted_study(study)
        selected = _resolve_subject_ids(study, subject_ids)
        runs, _ = self._run_study(study, subject_ids=selected)
        return tuple(
            diagnostic for subject_id in sorted(runs) for diagnostic in runs[subject_id].diagnostics
        )

    def calibrate_forecasts(
        self,
        study: Study,
        *,
        reference_subject_ids: tuple[str, ...],
        evaluation_subject_ids: tuple[str, ...],
        interval_level: float = 0.95,
    ) -> ForecastCalibrationResult:
        """Calibrate on reference residuals and assess only held-out subjects."""
        study_hash, config_hash = self._require_fitted_study(study)
        if not 0.5 < interval_level < 1:
            raise ValueError("interval_level must be between 0.5 and 1")
        reference, evaluation = _validate_subject_partition(
            study,
            reference_subject_ids=reference_subject_ids,
            evaluation_subject_ids=evaluation_subject_ids,
        )
        reference_diagnostics = self.one_step_diagnostics(study, subject_ids=reference)
        evaluation_diagnostics = self.one_step_diagnostics(study, subject_ids=evaluation)
        reference_z = _flatten_standardized(reference_diagnostics)
        evaluation_z = _flatten_standardized(evaluation_diagnostics)
        if not len(reference_z):
            raise ValueError("reference subjects have no one-step forecast innovations")
        if not len(evaluation_z):
            raise ValueError("evaluation subjects have no one-step forecast innovations")
        threshold = _higher_quantile(np.abs(reference_z), interval_level)
        return ForecastCalibrationResult(
            study_id=study.study_id,
            study_artifact_hash=study_hash,
            model_config_artifact_hash=config_hash,
            reference_diagnostics_artifact_hash=_diagnostics_artifact_hash(reference_diagnostics),
            evaluation_diagnostics_artifact_hash=_diagnostics_artifact_hash(evaluation_diagnostics),
            reference_subject_ids=reference,
            evaluation_subject_ids=evaluation,
            interval_level=interval_level,
            reference_standardized_innovations=len(reference_z),
            evaluation_standardized_innovations=len(evaluation_z),
            absolute_standardized_threshold=threshold,
            empirical_coverage=float(np.mean(np.abs(evaluation_z) <= threshold)),
            standardized_bias=float(np.mean(evaluation_z)),
            standardized_rmse=float(np.sqrt(np.mean(evaluation_z**2))),
        )

    def detect_innovation_change_points(
        self,
        study: Study,
        *,
        reference_subject_ids: tuple[str, ...],
        evaluation_subject_ids: tuple[str, ...],
        false_alarm_rate: float = 0.05,
    ) -> InnovationChangePointReport:
        """Detect large individual innovations using a disjoint empirical reference set."""
        study_hash, config_hash = self._require_fitted_study(study)
        if not 0 < false_alarm_rate < 0.5:
            raise ValueError("false_alarm_rate must be between 0 and 0.5")
        reference, evaluation = _validate_subject_partition(
            study,
            reference_subject_ids=reference_subject_ids,
            evaluation_subject_ids=evaluation_subject_ids,
        )
        reference_diagnostics = self.one_step_diagnostics(study, subject_ids=reference)
        evaluation_diagnostics = self.one_step_diagnostics(study, subject_ids=evaluation)
        reference_scores = np.asarray(
            [
                item.normalized_innovation_squared / item.degrees_of_freedom
                for item in reference_diagnostics
            ],
            dtype=np.float64,
        )
        if not len(reference_scores):
            raise ValueError("reference subjects have no one-step forecast innovations")
        if not evaluation_diagnostics:
            raise ValueError("evaluation subjects have no one-step forecast innovations")
        threshold = _higher_quantile(reference_scores, 1 - false_alarm_rate)
        results = tuple(
            InnovationChangePoint(
                subject_id=item.subject_id,
                timestamp=item.timestamp,
                channel_names=item.channel_names,
                innovation_score=(item.normalized_innovation_squared / item.degrees_of_freedom),
                empirical_tail_probability=float(
                    (
                        1
                        + np.count_nonzero(
                            reference_scores
                            >= item.normalized_innovation_squared / item.degrees_of_freedom
                        )
                    )
                    / (len(reference_scores) + 1)
                ),
                detected=(item.normalized_innovation_squared / item.degrees_of_freedom > threshold),
            )
            for item in evaluation_diagnostics
        )
        return InnovationChangePointReport(
            study_id=study.study_id,
            study_artifact_hash=study_hash,
            model_config_artifact_hash=config_hash,
            reference_diagnostics_artifact_hash=_diagnostics_artifact_hash(reference_diagnostics),
            evaluation_diagnostics_artifact_hash=_diagnostics_artifact_hash(evaluation_diagnostics),
            reference_subject_ids=reference,
            evaluation_subject_ids=evaluation,
            false_alarm_rate=false_alarm_rate,
            reference_innovations=len(reference_scores),
            score_threshold=threshold,
            results=results,
        )

    def forecast(
        self,
        trajectory: StateTrajectory,
        *,
        timestamps: tuple[datetime, ...],
    ) -> StateTrajectory:
        """Propagate a trajectory's terminal posterior to future timestamps."""
        config = self._require_fitted()
        if not timestamps:
            raise ValueError("at least one forecast timestamp is required")
        for timestamp in timestamps:
            _require_aware(timestamp, name="forecast timestamp")
        normalized = tuple(timestamp.astimezone(UTC) for timestamp in timestamps)
        previous_time = trajectory.estimates[-1].timestamp.astimezone(UTC)
        if any(second <= first for first, second in pairwise(normalized)):
            raise ValueError("forecast timestamps must be strictly increasing")
        if normalized[0] <= previous_time:
            raise ValueError("forecast timestamps must follow the terminal estimate")
        terminal = trajectory.estimates[-1]
        if terminal.state_names != config.state_names:
            raise ValueError("trajectory state names do not match estimator config")
        if trajectory.model_config_artifact_hash != self.model_config_artifact_hash_:
            raise ValueError("trajectory model artifact does not match estimator config")
        if trajectory.source_study_artifact_hash is None:
            raise ValueError("trajectory lacks a source study artifact")
        mean = np.asarray(terminal.mean, dtype=np.float64)
        covariance = np.asarray(terminal.covariance, dtype=np.float64)
        forecasts: list[StateEstimate] = []
        for timestamp in normalized:
            elapsed = self._elapsed_units(previous_time, timestamp)
            transition, process_covariance, drift = self._discretize(elapsed)
            mean = transition @ mean + drift
            covariance = _stable_covariance(
                transition @ covariance @ transition.T + process_covariance
            )
            forecasts.append(
                StateEstimate(
                    subject_id=trajectory.subject_id,
                    timestamp=timestamp,
                    state_names=config.state_names,
                    mean=_vector_tuple(mean),
                    covariance=_matrix_tuple(covariance),
                    is_forecast=True,
                    estimate_kind="forecast",
                )
            )
            previous_time = timestamp
        return StateTrajectory(
            subject_id=trajectory.subject_id,
            estimates=tuple(forecasts),
            source_study_artifact_hash=trajectory.source_study_artifact_hash,
            model_config_artifact_hash=trajectory.model_config_artifact_hash,
            source_trajectory_artifact_hash=trajectory.artifact_hash,
        )

    def _require_fitted(self) -> LinearGaussianStateConfig:
        if self.config is None:
            raise NotImplementedError(
                "state estimation requires an explicit LinearGaussianStateConfig"
            )
        if self.fitted_study_id_ is None:
            raise RuntimeError("fit must be called before state estimation")
        return self.config

    def _require_fitted_study(self, study: Study) -> tuple[str, str]:
        """Reject a study or model that differs from the fitted artifact."""
        config = self._require_fitted()
        study_hash = study_artifact_hash(study)
        config_hash = _state_model_artifact_hash(config)
        if study.study_id != self.fitted_study_id_:
            raise ValueError("state study_id differs from the fitted study")
        if study_hash != self.fitted_study_artifact_hash_:
            raise ValueError("state study artifact differs from the fitted study")
        if config_hash != self.model_config_artifact_hash_:
            raise ValueError("state model configuration differs from the fitted model")
        return study_hash, config_hash

    def _validate_units(self, study: Study) -> None:
        assert self.config is not None
        allowed: dict[tuple[Modality, str], set[str]] = defaultdict(set)
        for channel in self.config.channels:
            allowed[(channel.modality, channel.feature)].add(channel.unit)
        for observation in study.observations:
            expected = allowed.get((observation.modality, observation.feature))
            if expected is not None and observation.unit not in expected:
                raise ValueError(
                    "observation unit does not match configured channel: "
                    f"subject={observation.subject_id!r}, feature={observation.feature!r}, "
                    f"observed={observation.unit!r}, expected={sorted(expected)!r}"
                )

    def _run_study(
        self,
        study: Study,
        *,
        subject_ids: tuple[str, ...],
    ) -> tuple[dict[str, _SubjectRun], tuple[SubjectStateExclusion, ...]]:
        self._validate_units(study)
        requested = set(subject_ids)
        by_subject: dict[str, list[Observation]] = defaultdict(list)
        for observation in study.observations:
            if observation.subject_id in requested:
                by_subject[observation.subject_id].append(observation)
        runs: dict[str, _SubjectRun] = {}
        excluded: list[SubjectStateExclusion] = []
        for subject_id in subject_ids:
            prepared = self._prepare_subject(subject_id, by_subject.get(subject_id, []))
            if prepared is None:
                excluded.append(SubjectStateExclusion(subject_id=subject_id))
                continue
            timeline, coverage, exclusions = prepared
            runs[subject_id] = self._kalman_pass(
                subject_id,
                timeline=timeline,
                coverage=coverage,
                exclusions=exclusions,
            )
        return runs, tuple(sorted(excluded, key=lambda item: item.subject_id))

    def _prepare_subject(
        self,
        subject_id: str,
        observations: list[Observation],
    ) -> (
        tuple[
            list[tuple[datetime, list[tuple[int, Observation]]]],
            StateCoverage,
            tuple[ObservationExclusion, ...],
        ]
        | None
    ):
        assert self.config is not None
        channel_index = {
            channel.exact_key: index for index, channel in enumerate(self.config.channels)
        }
        matched: dict[datetime, dict[int, Observation]] = defaultdict(dict)
        exclusions: list[ObservationExclusion] = []
        for observation in observations:
            timestamp = observation.timestamp.astimezone(UTC)
            index = channel_index.get((observation.modality, observation.feature, observation.unit))
            if index is None:
                exclusions.append(
                    ObservationExclusion(
                        timestamp=timestamp,
                        modality=observation.modality,
                        feature=observation.feature,
                        unit=observation.unit,
                    )
                )
                continue
            if not math.isfinite(observation.value):
                raise ValueError(
                    f"configured observation values must be finite: subject={subject_id!r}"
                )
            if observation.standard_error is not None and not math.isfinite(
                observation.standard_error
            ):
                raise ValueError(
                    f"configured observation standard errors must be finite: subject={subject_id!r}"
                )
            if index in matched[timestamp]:
                channel = self.config.channels[index]
                raise ValueError(
                    "duplicate configured channel at one subject timestamp: "
                    f"subject={subject_id!r}, timestamp={timestamp.isoformat()}, "
                    f"channel={channel.name!r}"
                )
            matched[timestamp][index] = observation
        if not matched:
            return None
        timeline = [
            (timestamp, sorted(rows.items()))
            for timestamp, rows in sorted(matched.items(), key=lambda item: item[0])
        ]
        timepoints = len(timeline)
        channel_coverage: list[StateChannelCoverage] = []
        observed_total = 0
        for index, channel in enumerate(self.config.channels):
            rows = [items[index] for items in matched.values() if index in items]
            observed = len(rows)
            observed_total += observed
            channel_coverage.append(
                StateChannelCoverage(
                    channel_name=channel.name,
                    modality=channel.modality,
                    feature=channel.feature,
                    unit=channel.unit,
                    observed_timepoints=observed,
                    missing_timepoints=timepoints - observed,
                    reported_standard_errors=sum(row.standard_error is not None for row in rows),
                )
            )
        expected = timepoints * len(self.config.channels)
        coverage = StateCoverage(
            timepoints=timepoints,
            expected_channel_values=expected,
            observed_channel_values=observed_total,
            missing_channel_values=expected - observed_total,
            observed_fraction=observed_total / expected,
            channels=tuple(channel_coverage),
        )
        exclusions.sort(
            key=lambda item: (
                item.timestamp,
                item.modality.value,
                item.feature,
                item.unit,
            )
        )
        return timeline, coverage, tuple(exclusions)

    def _kalman_pass(
        self,
        subject_id: str,
        *,
        timeline: list[tuple[datetime, list[tuple[int, Observation]]]],
        coverage: StateCoverage,
        exclusions: tuple[ObservationExclusion, ...],
    ) -> _SubjectRun:
        assert self.config is not None
        dimension = len(self.config.state_names)
        identity = np.eye(dimension, dtype=np.float64)
        mean = np.asarray(self.config.initial_mean, dtype=np.float64)
        covariance = np.asarray(self.config.initial_covariance, dtype=np.float64)
        filtered_means: list[NDArray[np.float64]] = []
        filtered_covariances: list[NDArray[np.float64]] = []
        predicted_means: list[NDArray[np.float64]] = []
        predicted_covariances: list[NDArray[np.float64]] = []
        transitions: list[NDArray[np.float64]] = []
        diagnostics: list[OneStepForecastDiagnostic] = []
        timestamps: list[datetime] = []
        previous: datetime | None = None
        for timestamp, indexed_observations in timeline:
            if previous is None:
                transition = identity
                predicted_mean = mean
                predicted_covariance = covariance
            else:
                elapsed = self._elapsed_units(previous, timestamp)
                transition, process_covariance, drift = self._discretize(elapsed)
                predicted_mean = transition @ mean + drift
                predicted_covariance = _stable_covariance(
                    transition @ covariance @ transition.T + process_covariance
                )
            rows = [self.config.channels[index] for index, _ in indexed_observations]
            observations = [observation for _, observation in indexed_observations]
            observation_matrix = np.asarray(
                [channel.loadings for channel in rows], dtype=np.float64
            )
            offsets = np.asarray([channel.offset for channel in rows], dtype=np.float64)
            values = np.asarray(
                [observation.value for observation in observations], dtype=np.float64
            )
            reported_variances = np.asarray(
                [
                    0.0 if observation.standard_error is None else observation.standard_error**2
                    for observation in observations
                ],
                dtype=np.float64,
            )
            observation_variances = (
                np.asarray([channel.measurement_variance for channel in rows], dtype=np.float64)
                + reported_variances
            )
            measurement_covariance = np.diag(observation_variances)
            predicted_values = offsets + observation_matrix @ predicted_mean
            innovation = values - predicted_values
            innovation_covariance = _stable_covariance(
                observation_matrix @ predicted_covariance @ observation_matrix.T
                + measurement_covariance
            )
            inverse_innovation = np.linalg.inv(innovation_covariance)
            gain = predicted_covariance @ observation_matrix.T @ inverse_innovation
            mean = predicted_mean + gain @ innovation
            residual_operator = identity - gain @ observation_matrix
            covariance = _stable_covariance(
                residual_operator @ predicted_covariance @ residual_operator.T
                + gain @ measurement_covariance @ gain.T
            )
            if previous is not None:
                diagnostics.append(
                    OneStepForecastDiagnostic(
                        subject_id=subject_id,
                        timestamp=timestamp,
                        channel_names=tuple(channel.name for channel in rows),
                        observed_values=_vector_tuple(values),
                        predicted_values=_vector_tuple(predicted_values),
                        innovation_covariance=_matrix_tuple(innovation_covariance),
                        standardized_innovations=_vector_tuple(
                            innovation / np.sqrt(np.diag(innovation_covariance))
                        ),
                        normalized_innovation_squared=max(
                            float(innovation @ inverse_innovation @ innovation), 0.0
                        ),
                        degrees_of_freedom=len(rows),
                        reported_standard_error_variances=_vector_tuple(reported_variances),
                    )
                )
            timestamps.append(timestamp)
            transitions.append(np.asarray(transition, dtype=np.float64))
            predicted_means.append(np.asarray(predicted_mean, dtype=np.float64))
            predicted_covariances.append(np.asarray(predicted_covariance, dtype=np.float64))
            filtered_means.append(np.asarray(mean, dtype=np.float64))
            filtered_covariances.append(np.asarray(covariance, dtype=np.float64))
            previous = timestamp
        return _SubjectRun(
            timestamps=timestamps,
            filtered_means=filtered_means,
            filtered_covariances=filtered_covariances,
            predicted_means=predicted_means,
            predicted_covariances=predicted_covariances,
            transitions=transitions,
            diagnostics=diagnostics,
            coverage=coverage,
            exclusions=exclusions,
        )

    def _trajectory(
        self,
        subject_id: str,
        run: _SubjectRun,
        *,
        smooth: bool,
    ) -> StateTrajectory:
        assert self.config is not None
        means = [item.copy() for item in run.filtered_means]
        covariances = [item.copy() for item in run.filtered_covariances]
        kind: Literal["filtered", "smoothed"] = "filtered"
        if smooth and len(means) > 1:
            kind = "smoothed"
            for index in range(len(means) - 2, -1, -1):
                transition = run.transitions[index + 1]
                predicted_covariance = run.predicted_covariances[index + 1]
                smoother_gain = (
                    run.filtered_covariances[index]
                    @ transition.T
                    @ np.linalg.pinv(predicted_covariance)
                )
                means[index] = means[index] + smoother_gain @ (
                    means[index + 1] - run.predicted_means[index + 1]
                )
                covariances[index] = _stable_covariance(
                    covariances[index]
                    + smoother_gain
                    @ (covariances[index + 1] - predicted_covariance)
                    @ smoother_gain.T
                )
        estimates = tuple(
            StateEstimate(
                subject_id=subject_id,
                timestamp=timestamp,
                state_names=self.config.state_names,
                mean=_vector_tuple(mean),
                covariance=_matrix_tuple(covariance),
                estimate_kind=kind,
            )
            for timestamp, mean, covariance in zip(run.timestamps, means, covariances, strict=True)
        )
        return StateTrajectory(
            subject_id=subject_id,
            estimates=estimates,
            coverage=run.coverage,
            excluded_observations=run.exclusions,
            source_study_artifact_hash=self.fitted_study_artifact_hash_,
            model_config_artifact_hash=self.model_config_artifact_hash_,
        )

    def _elapsed_units(self, first: datetime, second: datetime) -> float:
        assert self.config is not None
        elapsed = (second - first).total_seconds() / (self.config.time_unit_days * _SECONDS_PER_DAY)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("state observation times must be strictly increasing")
        return elapsed

    def _discretize(
        self,
        elapsed: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Exactly discretize A, Qc, and constant drift for an elapsed interval."""
        assert self.config is not None
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed model time must be finite and nonnegative")
        dimension = len(self.config.state_names)
        if elapsed == 0:
            return (
                np.eye(dimension, dtype=np.float64),
                np.zeros((dimension, dimension), dtype=np.float64),
                np.zeros(dimension, dtype=np.float64),
            )
        dynamics = np.asarray(self.config.continuous_dynamics, dtype=np.float64)
        continuous_covariance = np.asarray(
            self.config.continuous_process_covariance, dtype=np.float64
        )
        drift_rate = np.asarray(self.config.continuous_drift, dtype=np.float64)
        van_loan = np.zeros((2 * dimension, 2 * dimension), dtype=np.float64)
        van_loan[:dimension, :dimension] = dynamics
        van_loan[:dimension, dimension:] = continuous_covariance
        van_loan[dimension:, dimension:] = -dynamics.T
        van_loan_exp = np.asarray(expm(van_loan * elapsed), dtype=np.float64)
        transition = van_loan_exp[:dimension, :dimension]
        process_covariance = _stable_covariance(van_loan_exp[:dimension, dimension:] @ transition.T)
        drift_augmented = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
        drift_augmented[:dimension, :dimension] = dynamics
        drift_augmented[:dimension, dimension] = drift_rate
        drift = np.asarray(
            expm(drift_augmented * elapsed)[:dimension, dimension],
            dtype=np.float64,
        )
        return transition, process_covariance, drift


def _resolve_subject_ids(
    study: Study,
    subject_ids: tuple[str, ...] | None,
) -> tuple[str, ...]:
    known = {subject.subject_id for subject in study.subjects}
    if subject_ids is None:
        return tuple(sorted(known))
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("subject identifiers must be unique")
    unknown = set(subject_ids).difference(known)
    if unknown:
        raise ValueError(f"unknown subject identifiers: {sorted(unknown)}")
    return tuple(sorted(subject_ids))


def _validate_subject_partition(
    study: Study,
    *,
    reference_subject_ids: tuple[str, ...],
    evaluation_subject_ids: tuple[str, ...],
    require_evaluation: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    reference = _resolve_subject_ids(study, reference_subject_ids)
    evaluation = _resolve_subject_ids(study, evaluation_subject_ids)
    if not reference:
        raise ValueError("reference_subject_ids must be nonempty")
    if require_evaluation and not evaluation:
        raise ValueError("evaluation_subject_ids must be nonempty")
    overlap = set(reference).intersection(evaluation)
    if overlap:
        raise ValueError(
            f"reference and evaluation subjects must be disjoint; overlap={sorted(overlap)}"
        )
    return reference, evaluation


def _flatten_standardized(
    diagnostics: tuple[OneStepForecastDiagnostic, ...],
) -> NDArray[np.float64]:
    return np.asarray(
        [value for item in diagnostics for value in item.standardized_innovations],
        dtype=np.float64,
    )


def _diagnostics_artifact_hash(
    diagnostics: tuple[OneStepForecastDiagnostic, ...],
) -> str:
    """Bind an ordered set of one-step diagnostics used for calibration."""
    return _canonical_hash(
        {
            "schema": "rejuvenationkit.one-step-diagnostics/v1",
            "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
        }
    )


def _higher_quantile(values: NDArray[np.float64], probability: float) -> float:
    """Return the deterministic higher empirical quantile without interpolation."""
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("quantile values must be a nonempty finite vector")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("quantile probability must be between zero and one")
    ordered = np.sort(values)
    index = math.ceil(probability * (len(ordered) - 1))
    return float(ordered[index])
