"""Phase 2: uncertainty-aware multimodal fusion interfaces."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rejuvenationkit.schemas import Modality, Study


class ModalityEstimate(BaseModel):
    """A calibrated scalar estimate from one modality."""

    model_config = ConfigDict(frozen=True)

    modality: Modality
    estimate: float
    standard_error: float = Field(gt=0)
    target: str


class FusionResult(BaseModel):
    """A fused estimate with modality-level diagnostics."""

    model_config = ConfigDict(frozen=True)

    target: str
    estimate: float
    standard_error: float = Field(gt=0)
    modality_weights: dict[Modality, float]
    disagreement_score: float = Field(ge=0)


class MultimodalFusion(Protocol):
    """Contract for Phase 2 fusion implementations."""

    def fit(self, study: Study) -> MultimodalFusion:
        """Calibrate the estimator."""
        ...

    def fuse(self, estimates: tuple[ModalityEstimate, ...]) -> FusionResult:
        """Fuse calibrated modality estimates."""
        ...


class PrecisionWeightedFusion:
    """Placeholder reference fusion estimator."""

    def fit(self, study: Study) -> PrecisionWeightedFusion:
        """Calibrate modality bias and covariance.

        TODO(phase-2): learn calibration without subject/time leakage and record the
        fitted provenance.
        """
        raise NotImplementedError("Phase 2 calibration is not implemented")

    def fuse(self, estimates: tuple[ModalityEstimate, ...]) -> FusionResult:
        """Fuse modality estimates.

        TODO(phase-2): implement correlated precision weighting, missing-modality
        behavior, disagreement diagnostics, and sensitivity analysis.
        """
        raise NotImplementedError("Phase 2 fusion is not implemented")
