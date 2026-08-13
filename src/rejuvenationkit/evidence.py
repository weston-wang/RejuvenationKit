"""Evidence-level, covariance-aware fusion for Phase 2 analyses."""

from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from math import isfinite, sqrt
from statistics import NormalDist
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import Bounds, LinearConstraint, minimize

from rejuvenationkit.fusion import (
    FusionConfig,
    FusionModel,
    FusionResult,
    ModalityEstimate,
    PrecisionWeightedFusion,
)
from rejuvenationkit.schemas import Modality


class EffectDirection(StrEnum):
    """Scientific interpretation of increasing values for an estimand."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    UNSPECIFIED = "unspecified"


class EvidenceWeightConstraint(StrEnum):
    """Policy for extrapolative generalized least-squares weights."""

    UNCONSTRAINED = "unconstrained"
    NONNEGATIVE = "nonnegative"


class Estimand(BaseModel):
    """A precisely named target that evidence estimates are intended to measure."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    direction: EffectDirection = EffectDirection.UNSPECIFIED
    population: str | None = None
    time_contrast: str | None = None
    transform: str | None = None

    @property
    def key(self) -> str:
        """Return a stable grouping key including every commensurability field."""
        values = (
            self.name,
            self.unit,
            self.direction.value,
            self.population or "",
            self.time_contrast or "",
            self.transform or "",
        )
        return "|".join(values)


class EvidenceEstimate(BaseModel):
    """One calibrated scalar estimate with evidence-level provenance.

    Unlike :class:`~rejuvenationkit.fusion.ModalityEstimate`, several estimates
    may share a modality. Their dependence must be supplied in an
    :class:`EvidenceCovariance` when it is material.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(min_length=1)
    modality: Modality
    estimand: Estimand
    estimate: float
    standard_error: float = Field(gt=0)
    calibration_id: str = Field(min_length=1)
    provenance_id: str = Field(min_length=1)
    subject_id: str | None = None
    sample_id: str | None = None
    tissue: str | None = None
    species_taxon_id: int | None = Field(default=None, gt=0)
    assay_id: str | None = None
    correlation_group: str | None = None
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_numeric_fields(self) -> Self:
        """Reject non-finite estimates before matrix calculations."""
        if not isfinite(self.estimate) or not isfinite(self.standard_error):
            raise ValueError("estimate and standard_error must be finite")
        if len(set(self.quality_flags)) != len(self.quality_flags):
            raise ValueError("quality_flags must be unique")
        return self


class EvidenceCovariance(BaseModel):
    """Externally estimated covariance aligned to named evidence estimates."""

    model_config = ConfigDict(frozen=True)

    evidence_ids: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...]
    source_id: str = Field(min_length=1)
    effective_sample_size: int | None = Field(default=None, ge=2)

    @model_validator(mode="after")
    def validate_shape_and_values(self) -> Self:
        """Validate identifiers, dimensions, finiteness, and symmetry."""
        size = len(self.evidence_ids)
        if size == 0:
            raise ValueError("evidence covariance cannot be empty")
        if len(set(self.evidence_ids)) != size:
            raise ValueError("covariance evidence_ids must be unique")
        if len(self.covariance) != size or any(len(row) != size for row in self.covariance):
            raise ValueError("covariance must be square and aligned to evidence_ids")
        matrix = np.asarray(self.covariance, dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("covariance values must be finite")
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        if np.any(np.diag(matrix) <= 0):
            raise ValueError("covariance diagonal must be positive")
        return self

    @classmethod
    def from_correlation(
        cls,
        estimates: tuple[EvidenceEstimate, ...],
        correlation: tuple[tuple[float, ...], ...],
        *,
        source_id: str,
        effective_sample_size: int | None = None,
    ) -> EvidenceCovariance:
        """Construct covariance from a correlation matrix and reported errors."""
        size = len(estimates)
        if len(correlation) != size or any(len(row) != size for row in correlation):
            raise ValueError("correlation must be square and aligned to estimates")
        matrix = np.asarray(correlation, dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("correlation values must be finite")
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(matrix), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("correlation diagonal must equal one")
        if np.any((matrix < -1) | (matrix > 1)):
            raise ValueError("correlation values must lie in [-1, 1]")
        errors = np.asarray([item.standard_error for item in estimates])
        covariance = matrix * np.outer(errors, errors)
        return cls(
            evidence_ids=tuple(item.evidence_id for item in estimates),
            covariance=tuple(tuple(float(value) for value in row) for row in covariance),
            source_id=source_id,
            effective_sample_size=effective_sample_size,
        )


class LeaveOneEvidenceOut(BaseModel):
    """Influence diagnostic after removing one evidence estimate."""

    model_config = ConfigDict(frozen=True)

    omitted_evidence_id: str
    estimate: float
    standard_error: float = Field(gt=0)
    estimate_shift: float


class LeaveOneEvidenceModalityOut(BaseModel):
    """Influence diagnostic after removing every estimate from one modality."""

    model_config = ConfigDict(frozen=True)

    omitted_modality: Modality
    omitted_evidence_ids: tuple[str, ...]
    estimate: float
    standard_error: float = Field(gt=0)
    estimate_shift: float


class EvidenceFusionConfig(BaseModel):
    """Numerical and reporting policy for generalized least-squares fusion."""

    model_config = ConfigDict(frozen=True)

    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    minimum_evidence: int = Field(default=1, ge=1)
    maximum_condition_number: float = Field(default=1e8, gt=1)
    covariance_ridge: float = Field(default=0.0, ge=0)
    auto_regularize: bool = True
    psd_tolerance: float = Field(default=1e-10, gt=0)
    variance_relative_tolerance: float = Field(default=1e-6, ge=0)
    report_negative_weights: bool = True
    weight_constraint: EvidenceWeightConstraint = EvidenceWeightConstraint.UNCONSTRAINED
    allow_mixed_species: bool = False
    allow_mixed_subjects: bool = False


class EvidenceFusionResult(BaseModel):
    """Covariance-aware estimate with evidence- and modality-level diagnostics."""

    model_config = ConfigDict(frozen=True)

    estimand: Estimand
    estimate: float
    standard_error: float = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)
    confidence_interval: tuple[float, float]
    evidence_weights: dict[str, float]
    modality_weights: dict[Modality, float]
    standardized_residuals: dict[str, float]
    disagreement_score: float = Field(ge=0)
    effective_evidence_count: float = Field(gt=0)
    condition_number: float = Field(ge=1)
    regularization_applied: float = Field(ge=0)
    covariance_source_id: str
    leave_one_evidence_out: tuple[LeaveOneEvidenceOut, ...]
    leave_one_modality_out: tuple[LeaveOneEvidenceModalityOut, ...]
    warnings: tuple[str, ...]

    @property
    def maximum_leave_one_out_shift(self) -> float:
        """Return the largest absolute leave-one-evidence-out shift."""
        return max(
            (abs(item.estimate_shift) for item in self.leave_one_evidence_out),
            default=0.0,
        )

    @property
    def weight_concentration_effective_count(self) -> float:
        """Name the legacy Kish-style field without implying independent information."""
        return self.effective_evidence_count


class HierarchicalFusionResult(BaseModel):
    """Two-level fusion result with within-modality and across-modality evidence."""

    model_config = ConfigDict(frozen=True)

    estimand: Estimand
    within_modality: dict[Modality, EvidenceFusionResult]
    across_modality: FusionResult
    warnings: tuple[str, ...]


class _GLSValues(BaseModel):
    """Private immutable carrier for one numerical GLS solve."""

    model_config = ConfigDict(frozen=True)

    estimate: float
    standard_error: float
    weights: tuple[float, ...]
    residuals: tuple[float, ...]
    residual_standard_errors: tuple[float, ...]
    disagreement: float
    condition_number: float
    ridge: float


class GeneralizedLeastSquaresFusion:
    """Fuse correlated evidence with auditable generalized least squares.

    The class supports several estimates from the same assay modality. A
    covariance matrix estimated from held-out subjects or bootstrap replicates
    should be supplied whenever those estimates share samples, features, or a
    training pipeline.
    """

    def __init__(self, config: EvidenceFusionConfig | None = None) -> None:
        """Initialize the covariance-aware fusion engine."""
        self.config = config or EvidenceFusionConfig()

    def fuse(
        self,
        estimates: tuple[EvidenceEstimate, ...],
        covariance: EvidenceCovariance | None = None,
    ) -> EvidenceFusionResult:
        """Fuse one commensurate evidence set and compute influence diagnostics."""
        self._validate_estimates(estimates)
        matrix, source_id = self._aligned_covariance(estimates, covariance)
        values = self._solve(estimates, matrix)
        z_value = NormalDist().inv_cdf(0.5 + self.config.confidence_level / 2)
        interval = (
            values.estimate - z_value * values.standard_error,
            values.estimate + z_value * values.standard_error,
        )
        evidence_weights = {
            item.evidence_id: weight for item, weight in zip(estimates, values.weights, strict=True)
        }
        modality_weights: defaultdict[Modality, float] = defaultdict(float)
        for item, weight in zip(estimates, values.weights, strict=True):
            modality_weights[item.modality] += weight
        standardized = {
            item.evidence_id: (residual / residual_error if residual_error > 0 else 0.0)
            for item, residual, residual_error in zip(
                estimates,
                values.residuals,
                values.residual_standard_errors,
                strict=True,
            )
        }
        warnings = self._warnings(estimates, values, covariance_supplied=covariance is not None)
        return EvidenceFusionResult(
            estimand=estimates[0].estimand,
            estimate=values.estimate,
            standard_error=values.standard_error,
            confidence_level=self.config.confidence_level,
            confidence_interval=interval,
            evidence_weights=evidence_weights,
            modality_weights=dict(modality_weights),
            standardized_residuals=standardized,
            disagreement_score=values.disagreement,
            effective_evidence_count=1.0 / sum(weight**2 for weight in values.weights),
            condition_number=values.condition_number,
            regularization_applied=values.ridge,
            covariance_source_id=source_id,
            leave_one_evidence_out=self._leave_one_evidence_out(
                estimates,
                matrix,
                values.estimate,
            ),
            leave_one_modality_out=self._leave_one_modality_out(
                estimates,
                matrix,
                values.estimate,
            ),
            warnings=warnings,
        )

    def fuse_by_estimand(
        self,
        estimates: tuple[EvidenceEstimate, ...],
        covariances: dict[str, EvidenceCovariance] | None = None,
    ) -> dict[str, EvidenceFusionResult]:
        """Fuse unrelated targets separately instead of forcing commensurability."""
        grouped: defaultdict[str, list[EvidenceEstimate]] = defaultdict(list)
        for estimate in estimates:
            grouped[estimate.estimand.key].append(estimate)
        supplied = covariances or {}
        return {
            key: self.fuse(tuple(items), supplied.get(key))
            for key, items in sorted(grouped.items())
        }

    def _validate_estimates(self, estimates: tuple[EvidenceEstimate, ...]) -> None:
        if len(estimates) < self.config.minimum_evidence:
            raise ValueError(
                f"at least {self.config.minimum_evidence} evidence estimates are required"
            )
        identifiers = [item.evidence_id for item in estimates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("evidence_id values must be unique")
        estimands = {item.estimand for item in estimates}
        if len(estimands) != 1:
            raise ValueError("all evidence estimates must share one complete estimand")
        species = {item.species_taxon_id for item in estimates if item.species_taxon_id is not None}
        if len(species) > 1 and not self.config.allow_mixed_species:
            raise ValueError("evidence from multiple species requires an explicit fusion policy")
        subjects = {item.subject_id for item in estimates if item.subject_id is not None}
        if len(subjects) > 1 and not self.config.allow_mixed_subjects:
            raise ValueError("evidence from multiple subjects requires an explicit fusion policy")

    def _aligned_covariance(
        self,
        estimates: tuple[EvidenceEstimate, ...],
        covariance: EvidenceCovariance | None,
    ) -> tuple[npt.NDArray[np.float64], str]:
        if covariance is None:
            diagonal = np.asarray([item.standard_error**2 for item in estimates])
            return np.diag(diagonal), "reported-independent-standard-errors"
        expected_ids = {item.evidence_id for item in estimates}
        if set(covariance.evidence_ids) != expected_ids:
            raise ValueError("covariance evidence_ids must exactly match the estimates")
        order = {identifier: index for index, identifier in enumerate(covariance.evidence_ids)}
        indices = [order[item.evidence_id] for item in estimates]
        source = np.asarray(covariance.covariance, dtype=float)
        matrix = source[np.ix_(indices, indices)]
        reported = np.asarray([item.standard_error**2 for item in estimates])
        if not np.allclose(
            np.diag(matrix),
            reported,
            rtol=self.config.variance_relative_tolerance,
            atol=self.config.psd_tolerance * float(np.max(reported)),
        ):
            raise ValueError("covariance diagonal must agree with reported standard errors")
        return matrix, covariance.source_id

    def _solve(
        self,
        estimates: tuple[EvidenceEstimate, ...],
        covariance: npt.NDArray[np.float64],
    ) -> _GLSValues:
        covariance_scale = float(np.max(np.diag(covariance)))
        if not isfinite(covariance_scale) or covariance_scale <= 0:
            raise ValueError("covariance must have positive finite marginal variances")
        normalized_covariance = covariance / covariance_scale
        eigenvalues = np.linalg.eigvalsh(normalized_covariance)
        largest = float(eigenvalues[-1])
        smallest = float(eigenvalues[0])
        if smallest < -self.config.psd_tolerance * max(1.0, largest):
            raise ValueError("covariance must be positive semidefinite")
        normalized_ridge = self.config.covariance_ridge / covariance_scale
        ill_conditioned = (
            smallest + normalized_ridge <= 0
            or (largest + normalized_ridge) / (smallest + normalized_ridge)
            > self.config.maximum_condition_number
        )
        if ill_conditioned:
            if not self.config.auto_regularize:
                raise ValueError("covariance is singular or exceeds maximum_condition_number")
            target = self.config.maximum_condition_number
            required = max(
                self.config.psd_tolerance,
                (largest - target * smallest) / (target - 1),
            )
            normalized_ridge = max(normalized_ridge, required)
        ridge = normalized_ridge * covariance_scale
        regularized_normalized = normalized_covariance + np.eye(covariance.shape[0]) * (
            normalized_ridge
        )
        regularized = regularized_normalized * covariance_scale
        condition_number = float(np.linalg.cond(regularized_normalized))
        ones = np.ones(len(estimates))
        precision_direction = np.linalg.solve(regularized_normalized, ones)
        denominator = float(ones @ precision_direction)
        if not isfinite(denominator) or denominator <= 0:
            raise ValueError("covariance does not yield positive fused precision")
        if self.config.weight_constraint is EvidenceWeightConstraint.NONNEGATIVE:
            initial = 1 / np.diag(regularized_normalized)
            initial = initial / initial.sum()

            def variance_objective(weight: npt.NDArray[np.float64]) -> float:
                return float(weight @ regularized_normalized @ weight)

            optimized = minimize(
                variance_objective,
                initial,
                method="SLSQP",
                bounds=Bounds(np.zeros(len(estimates)), np.ones(len(estimates))),
                constraints=LinearConstraint(ones.reshape(1, -1), 1.0, 1.0),
                options={"ftol": 1e-12, "maxiter": 1_000},
            )
            if not optimized.success:
                raise ValueError(f"nonnegative weight optimization failed: {optimized.message}")
            weights = np.clip(np.asarray(optimized.x, dtype=float), 0.0, 1.0)
            weights = weights / weights.sum()
            fused_variance = float(weights @ regularized @ weights)
        else:
            weights = precision_direction / denominator
            fused_variance = covariance_scale / denominator
        observed = np.asarray([item.estimate for item in estimates])
        estimate = float(weights @ observed)
        standard_error = sqrt(fused_variance)
        residuals = observed - estimate
        residual_variances = (
            np.diag(regularized)
            + fused_variance
            - 2 * np.asarray(regularized @ weights, dtype=float)
        )
        if np.any(
            residual_variances < -self.config.psd_tolerance * float(np.max(np.diag(regularized)))
        ):
            raise ValueError("covariance produced invalid residual uncertainty")
        residual_standard_errors = np.sqrt(np.maximum(0.0, residual_variances))
        disagreement = float(residuals @ np.linalg.solve(regularized, residuals))
        return _GLSValues(
            estimate=estimate,
            standard_error=standard_error,
            weights=tuple(float(value) for value in weights),
            residuals=tuple(float(value) for value in residuals),
            residual_standard_errors=tuple(float(value) for value in residual_standard_errors),
            disagreement=max(0.0, disagreement),
            condition_number=max(1.0, condition_number),
            ridge=ridge,
        )

    def _warnings(
        self,
        estimates: tuple[EvidenceEstimate, ...],
        values: _GLSValues,
        *,
        covariance_supplied: bool,
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        if values.ridge > 0:
            warnings.append("covariance_regularized")
        if self.config.weight_constraint is EvidenceWeightConstraint.NONNEGATIVE:
            warnings.append("nonnegative_weight_constraint_active")
        if self.config.report_negative_weights and any(weight < 0 for weight in values.weights):
            warnings.append("negative_gls_weight")
        if any(item.quality_flags for item in estimates):
            warnings.append("input_quality_flags_present")
        if len({item.modality for item in estimates}) < len(estimates):
            warnings.append("multiple_estimates_share_modality")
        grouped: defaultdict[str, int] = defaultdict(int)
        for item in estimates:
            if item.correlation_group is not None:
                grouped[item.correlation_group] += 1
        if not covariance_supplied and any(count > 1 for count in grouped.values()):
            warnings.append("shared_correlation_group_assumed_independent")
        tissues = {item.tissue for item in estimates if item.tissue is not None}
        if len(tissues) > 1:
            warnings.append("mixed_tissue_evidence")
        if any(item.species_taxon_id is None for item in estimates) and any(
            item.species_taxon_id is not None for item in estimates
        ):
            warnings.append("incomplete_species_metadata")
        return tuple(warnings)

    def _leave_one_evidence_out(
        self,
        estimates: tuple[EvidenceEstimate, ...],
        covariance: npt.NDArray[np.float64],
        full_estimate: float,
    ) -> tuple[LeaveOneEvidenceOut, ...]:
        if len(estimates) == 1:
            return ()
        diagnostics: list[LeaveOneEvidenceOut] = []
        for omitted_index, omitted in enumerate(estimates):
            keep = [index for index in range(len(estimates)) if index != omitted_index]
            reduced = tuple(estimates[index] for index in keep)
            values = self._solve(reduced, covariance[np.ix_(keep, keep)])
            diagnostics.append(
                LeaveOneEvidenceOut(
                    omitted_evidence_id=omitted.evidence_id,
                    estimate=values.estimate,
                    standard_error=values.standard_error,
                    estimate_shift=values.estimate - full_estimate,
                )
            )
        return tuple(diagnostics)

    def _leave_one_modality_out(
        self,
        estimates: tuple[EvidenceEstimate, ...],
        covariance: npt.NDArray[np.float64],
        full_estimate: float,
    ) -> tuple[LeaveOneEvidenceModalityOut, ...]:
        diagnostics: list[LeaveOneEvidenceModalityOut] = []
        for modality in sorted({item.modality for item in estimates}, key=lambda item: item.value):
            keep = [index for index, item in enumerate(estimates) if item.modality is not modality]
            if not keep:
                continue
            reduced = tuple(estimates[index] for index in keep)
            values = self._solve(reduced, covariance[np.ix_(keep, keep)])
            omitted_ids = tuple(item.evidence_id for item in estimates if item.modality is modality)
            diagnostics.append(
                LeaveOneEvidenceModalityOut(
                    omitted_modality=modality,
                    omitted_evidence_ids=omitted_ids,
                    estimate=values.estimate,
                    standard_error=values.standard_error,
                    estimate_shift=values.estimate - full_estimate,
                )
            )
        return tuple(diagnostics)


class HierarchicalEvidenceFusion:
    """Fuse correlated estimates within modalities, then modalities across levels."""

    def __init__(
        self,
        evidence_config: EvidenceFusionConfig | None = None,
        *,
        across_modality_model: FusionModel = FusionModel.RANDOM_EFFECTS,
    ) -> None:
        """Initialize both within- and across-modality estimators."""
        self._within = GeneralizedLeastSquaresFusion(evidence_config)
        self._across_model = across_modality_model

    def fuse(
        self,
        estimates: tuple[EvidenceEstimate, ...],
        covariance: EvidenceCovariance | None = None,
    ) -> HierarchicalFusionResult:
        """Run two-level fusion while keeping duplicated assay evidence visible."""
        self._within._validate_estimates(estimates)
        full_covariance, _ = self._within._aligned_covariance(estimates, covariance)
        grouped: defaultdict[Modality, list[int]] = defaultdict(list)
        for index, estimate in enumerate(estimates):
            grouped[estimate.modality].append(index)
        within: dict[Modality, EvidenceFusionResult] = {}
        modality_estimates: list[ModalityEstimate] = []
        for modality, indices in sorted(grouped.items(), key=lambda item: item[0].value):
            subset = tuple(estimates[index] for index in indices)
            block = full_covariance[np.ix_(indices, indices)]
            block_covariance = EvidenceCovariance(
                evidence_ids=tuple(item.evidence_id for item in subset),
                covariance=tuple(tuple(float(value) for value in row) for row in block),
                source_id=(
                    covariance.source_id if covariance else "reported-independent-standard-errors"
                ),
                effective_sample_size=(covariance.effective_sample_size if covariance else None),
            )
            result = self._within.fuse(subset, block_covariance)
            within[modality] = result
            modality_estimates.append(
                ModalityEstimate(
                    modality=modality,
                    estimate=result.estimate,
                    standard_error=result.standard_error,
                    target=result.estimand.key,
                )
            )
        across = PrecisionWeightedFusion(
            FusionConfig(
                model=self._across_model,
                confidence_level=self._within.config.confidence_level,
            )
        ).fuse(tuple(modality_estimates))
        warnings: list[str] = []
        if covariance is not None and len(grouped) > 1:
            cross_blocks = full_covariance.copy()
            for indices in grouped.values():
                cross_blocks[np.ix_(indices, indices)] = 0
            if np.any(np.abs(cross_blocks) > self._within.config.psd_tolerance):
                warnings.append("cross_modality_covariance_not_propagated_by_hierarchy")
        return HierarchicalFusionResult(
            estimand=estimates[0].estimand,
            within_modality=within,
            across_modality=across,
            warnings=tuple(warnings),
        )
