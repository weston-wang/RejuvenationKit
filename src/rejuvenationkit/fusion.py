"""Phase 2: uncertainty-aware multimodal fusion."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from enum import StrEnum
from math import isfinite, sqrt
from statistics import NormalDist
from types import MappingProxyType
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from rejuvenationkit.schemas import Modality, Study


class FusionModel(StrEnum):
    """Statistical model used to combine modality estimates."""

    FIXED_EFFECT = "fixed_effect"
    RANDOM_EFFECTS = "random_effects"


class MissingModalityPolicy(StrEnum):
    """Behavior when an expected modality is absent."""

    ALLOW = "allow"
    ERROR = "error"


class ModalityCalibration(BaseModel):
    """Externally validated calibration applied before fusion.

    ``bias`` is subtracted from the reported estimate. ``standard_error_scale``
    inflates or deflates its reported standard error. A stable ``calibration_id``
    makes the source calibration auditable.
    """

    model_config = ConfigDict(frozen=True)

    modality: Modality
    bias: float = 0.0
    standard_error_scale: float = Field(default=1.0, gt=0)
    calibration_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        """Reject non-finite calibration parameters."""
        if not isfinite(self.bias) or not isfinite(self.standard_error_scale):
            raise ValueError("calibration parameters must be finite")
        return self


class ModalityEstimate(BaseModel):
    """A scalar estimate and uncertainty from one calibrated modality pipeline."""

    model_config = ConfigDict(frozen=True)

    modality: Modality
    estimate: float
    standard_error: float = Field(gt=0)
    target: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        """Reject non-finite values before they enter uncertainty calculations."""
        if not isfinite(self.estimate) or not isfinite(self.standard_error):
            raise ValueError("estimate and standard_error must be finite")
        return self


class LeaveOneModalityOut(BaseModel):
    """Influence diagnostic after omitting one observed modality."""

    model_config = ConfigDict(frozen=True)

    omitted_modality: Modality
    estimate: float
    standard_error: float = Field(gt=0)
    estimate_shift: float


class FusionResult(BaseModel):
    """A fused estimate with uncertainty, missingness, and influence diagnostics."""

    model_config = ConfigDict(frozen=True)

    target: str
    estimate: float
    standard_error: float = Field(gt=0)
    modality_weights: Mapping[Modality, float]
    disagreement_score: float = Field(ge=0)
    confidence_level: float = Field(gt=0, lt=1)
    confidence_interval: tuple[float, float]
    model: FusionModel
    between_modality_variance: float = Field(ge=0)
    heterogeneity_i2: float = Field(ge=0, le=1)
    present_modalities: tuple[Modality, ...]
    missing_modalities: tuple[Modality, ...]
    calibration_ids: Mapping[Modality, str]
    leave_one_modality_out: tuple[LeaveOneModalityOut, ...]
    fitted_study_id: str | None = None

    @model_validator(mode="after")
    def freeze_result_mappings(self) -> Self:
        """Prevent mutation from invalidating a completed scientific result."""
        object.__setattr__(
            self,
            "modality_weights",
            MappingProxyType(dict(self.modality_weights)),
        )
        object.__setattr__(
            self,
            "calibration_ids",
            MappingProxyType(dict(self.calibration_ids)),
        )
        return self

    @field_serializer("modality_weights")
    def serialize_modality_weights(
        self,
        value: Mapping[Modality, float],
    ) -> dict[Modality, float]:
        """Serialize immutable weights through an ordinary mapping."""
        return dict(value)

    @field_serializer("calibration_ids")
    def serialize_calibration_ids(
        self,
        value: Mapping[Modality, str],
    ) -> dict[Modality, str]:
        """Serialize immutable calibration identities through an ordinary mapping."""
        return dict(value)

    @property
    def maximum_leave_one_out_shift(self) -> float:
        """Return the largest absolute estimate change after omitting one modality."""
        return max(
            (abs(item.estimate_shift) for item in self.leave_one_modality_out),
            default=0.0,
        )


class FusionConfig(BaseModel):
    """Configuration for transparent precision-weighted fusion."""

    model_config = ConfigDict(frozen=True)

    model: FusionModel = FusionModel.RANDOM_EFFECTS
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    expected_modalities: tuple[Modality, ...] = ()
    missing_modality_policy: MissingModalityPolicy = MissingModalityPolicy.ALLOW
    minimum_modalities: int = Field(default=1, ge=1)
    calibrations: tuple[ModalityCalibration, ...] = ()

    @model_validator(mode="after")
    def validate_unique_modalities(self) -> Self:
        """Reject ambiguous expected modalities and calibration profiles."""
        if len(set(self.expected_modalities)) != len(self.expected_modalities):
            raise ValueError("expected_modalities must be unique")
        calibration_modalities = [item.modality for item in self.calibrations]
        if len(set(calibration_modalities)) != len(calibration_modalities):
            raise ValueError("calibrations must have unique modalities")
        return self


class MultimodalFusion(Protocol):
    """Contract for Phase 2 fusion implementations."""

    def fit(self, study: Study) -> MultimodalFusion:
        """Record calibration provenance and modality availability from a study."""
        ...

    def fuse(self, estimates: tuple[ModalityEstimate, ...]) -> FusionResult:
        """Fuse calibrated modality estimates."""
        ...


class PrecisionWeightedFusion:
    """Fuse commensurate modality estimates using meta-analytic weighting.

    The estimator never imputes an absent modality. Fixed-effect fusion assumes one
    shared effect; random-effects fusion estimates DerSimonian-Laird between-modality
    variance and widens uncertainty when modalities disagree.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        """Initialize an unfitted fusion estimator."""
        self.config = config or FusionConfig()
        self._fitted_study_id: str | None = None
        self._fitted_modality_counts: dict[Modality, int] = {}

    @property
    def fitted_study_id(self) -> str | None:
        """Identifier of the study most recently inspected by ``fit``."""
        return self._fitted_study_id

    @property
    def fitted_modality_counts(self) -> dict[Modality, int]:
        """Return observation counts used to document modality availability."""
        return dict(self._fitted_modality_counts)

    def fit(self, study: Study) -> PrecisionWeightedFusion:
        """Record study provenance and observed modality availability.

        Numeric bias and uncertainty calibration must be supplied through
        :class:`ModalityCalibration`; this method deliberately does not infer a
        biological truth target from unlabeled study observations.
        """
        counts = Counter(row.modality for row in study.observations)
        self._fitted_study_id = study.study_id
        self._fitted_modality_counts = dict(counts)
        return self

    def fuse(self, estimates: tuple[ModalityEstimate, ...]) -> FusionResult:
        """Fuse estimates while preserving uncertainty and disagreement."""
        target, calibrated, calibration_ids = self._validate_and_calibrate(estimates)
        expected = set(self.config.expected_modalities)
        present = {item.modality for item in calibrated}
        missing = tuple(sorted(expected - present, key=lambda item: item.value))
        if missing and self.config.missing_modality_policy is MissingModalityPolicy.ERROR:
            names = ", ".join(item.value for item in missing)
            raise ValueError(f"missing expected modalities: {names}")

        estimate, standard_error, weights, q, tau_squared, i2 = self._combine(calibrated)
        z_value = NormalDist().inv_cdf(0.5 + self.config.confidence_level / 2)
        interval = (
            estimate - z_value * standard_error,
            estimate + z_value * standard_error,
        )
        sensitivity = tuple(
            self._leave_one_out(item.modality, calibrated, estimate)
            for item in calibrated
            if len(calibrated) > 1
        )
        return FusionResult(
            target=target,
            estimate=estimate,
            standard_error=standard_error,
            modality_weights=weights,
            disagreement_score=q,
            confidence_level=self.config.confidence_level,
            confidence_interval=interval,
            model=self.config.model,
            between_modality_variance=tau_squared,
            heterogeneity_i2=i2,
            present_modalities=tuple(sorted(present, key=lambda item: item.value)),
            missing_modalities=missing,
            calibration_ids=calibration_ids,
            leave_one_modality_out=sensitivity,
            fitted_study_id=self._fitted_study_id,
        )

    def _validate_and_calibrate(
        self,
        estimates: tuple[ModalityEstimate, ...],
    ) -> tuple[str, tuple[ModalityEstimate, ...], dict[Modality, str]]:
        if len(estimates) < self.config.minimum_modalities:
            raise ValueError(
                f"at least {self.config.minimum_modalities} modality estimates are required"
            )
        targets = {item.target for item in estimates}
        if len(targets) != 1:
            raise ValueError("all modality estimates must share one target")
        modalities = [item.modality for item in estimates]
        if len(set(modalities)) != len(modalities):
            raise ValueError("modality estimates must have unique modalities")

        profiles = {item.modality: item for item in self.config.calibrations}
        calibrated: list[ModalityEstimate] = []
        calibration_ids: dict[Modality, str] = {}
        for item in estimates:
            profile = profiles.get(item.modality)
            if profile is None:
                calibrated.append(item)
                calibration_ids[item.modality] = "identity"
            else:
                calibrated.append(
                    item.model_copy(
                        update={
                            "estimate": item.estimate - profile.bias,
                            "standard_error": item.standard_error * profile.standard_error_scale,
                        }
                    )
                )
                calibration_ids[item.modality] = profile.calibration_id
        return targets.pop(), tuple(calibrated), calibration_ids

    def _combine(
        self,
        estimates: tuple[ModalityEstimate, ...],
    ) -> tuple[float, float, dict[Modality, float], float, float, float]:
        fixed_precisions = [1.0 / item.standard_error**2 for item in estimates]
        fixed_total = sum(fixed_precisions)
        fixed_mean = (
            sum(
                precision * item.estimate
                for precision, item in zip(fixed_precisions, estimates, strict=True)
            )
            / fixed_total
        )
        q = sum(
            precision * (item.estimate - fixed_mean) ** 2
            for precision, item in zip(fixed_precisions, estimates, strict=True)
        )
        degrees_of_freedom = len(estimates) - 1
        denominator = fixed_total - sum(value**2 for value in fixed_precisions) / fixed_total
        tau_squared = 0.0
        if (
            self.config.model is FusionModel.RANDOM_EFFECTS
            and degrees_of_freedom > 0
            and denominator > 0
        ):
            tau_squared = max(0.0, (q - degrees_of_freedom) / denominator)
        precisions = [1.0 / (item.standard_error**2 + tau_squared) for item in estimates]
        total = sum(precisions)
        normalized = [value / total for value in precisions]
        estimate = sum(
            weight * item.estimate for weight, item in zip(normalized, estimates, strict=True)
        )
        standard_error = sqrt(1.0 / total)
        weights = {
            item.modality: weight for item, weight in zip(estimates, normalized, strict=True)
        }
        i2 = max(0.0, (q - degrees_of_freedom) / q) if q > 0 else 0.0
        return estimate, standard_error, weights, q, tau_squared, i2

    def _leave_one_out(
        self,
        omitted: Modality,
        estimates: tuple[ModalityEstimate, ...],
        full_estimate: float,
    ) -> LeaveOneModalityOut:
        reduced = tuple(item for item in estimates if item.modality is not omitted)
        estimate, standard_error, _, _, _, _ = self._combine(reduced)
        return LeaveOneModalityOut(
            omitted_modality=omitted,
            estimate=estimate,
            standard_error=standard_error,
            estimate_shift=estimate - full_estimate,
        )
