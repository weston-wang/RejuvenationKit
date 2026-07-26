"""Phase 4: combination-therapy analysis interfaces."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rejuvenationkit.schemas import Study


class InteractionEstimate(BaseModel):
    """Estimated interaction among two or more interventions."""

    model_config = ConfigDict(frozen=True)

    interventions: tuple[str, ...] = Field(min_length=2)
    outcome: str
    interaction: float
    standard_error: float = Field(gt=0)
    reference_model: str


class CombinationAnalysis(Protocol):
    """Contract for combination-therapy estimators."""

    def estimate(self, study: Study, *, outcome: str) -> tuple[InteractionEstimate, ...]:
        """Estimate synergy, additivity, or antagonism."""
        ...


class FactorialCombinationAnalysis:
    """Placeholder factorial interaction estimator."""

    def estimate(self, study: Study, *, outcome: str) -> tuple[InteractionEstimate, ...]:
        """Estimate interaction effects.

        TODO(phase-4): define causal estimands, factorial contrasts, covariate handling,
        multiplicity control, uncertainty propagation, and design-power helpers.
        """
        raise NotImplementedError("Phase 4 combination analysis is not implemented")
