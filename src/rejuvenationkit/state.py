"""Phase 3: latent biological-state estimation interfaces."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from rejuvenationkit.schemas import Study


class StateEstimate(BaseModel):
    """One uncertain latent-state estimate."""

    model_config = ConfigDict(frozen=True)

    subject_id: str
    timestamp: datetime
    state_names: tuple[str, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    is_forecast: bool = False

    @property
    def dimension(self) -> int:
        """Return latent-state dimensionality."""
        return len(self.state_names)


class StateTrajectory(BaseModel):
    """Time-ordered state estimates for one subject."""

    model_config = ConfigDict(frozen=True)

    subject_id: str
    estimates: tuple[StateEstimate, ...]


class LatentStateEstimator(Protocol):
    """Contract for Phase 3 state estimators."""

    def fit(self, study: Study) -> LatentStateEstimator:
        """Learn or configure model parameters."""
        ...

    def estimate(self, study: Study) -> tuple[StateTrajectory, ...]:
        """Filter or smooth latent states."""
        ...


class LinearGaussianStateEstimator:
    """Placeholder linear-Gaussian reference estimator."""

    def fit(self, study: Study) -> LinearGaussianStateEstimator:
        """Fit the state-space model.

        TODO(phase-3): define observation adapters, irregular time transitions,
        initialization, and leakage-safe parameter estimation.
        """
        raise NotImplementedError("Phase 3 state model is not implemented")

    def estimate(self, study: Study) -> tuple[StateTrajectory, ...]:
        """Estimate subject trajectories.

        TODO(phase-3): implement filtering, smoothing, missing observations, coverage
        diagnostics, and deterministic forecasts.
        """
        raise NotImplementedError("Phase 3 state estimation is not implemented")
