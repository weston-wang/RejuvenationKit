"""Phase 4 factorial combination-therapy inference and design planning."""

from __future__ import annotations

import json
from collections import defaultdict
from enum import StrEnum
from hashlib import sha256
from itertools import combinations, product
from math import ceil, isclose, isfinite, sqrt
from statistics import NormalDist
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from scipy.stats import t as student_t

from rejuvenationkit.endpoints import SubjectEndpointBatch
from rejuvenationkit.evidence import Estimand
from rejuvenationkit.schemas import Study, study_artifact_hash


class AssignmentMechanism(StrEnum):
    """How intervention exposure was assigned."""

    RANDOMIZED = "randomized"
    OBSERVATIONAL = "observational"


class CovarianceEstimator(StrEnum):
    """Coefficient covariance used for factorial regression."""

    CLASSICAL = "classical"
    HC3 = "hc3"


class MultiplicityMethod(StrEnum):
    """Local adjustment applied across the declared interaction family."""

    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    BONFERRONI = "bonferroni"
    NONE = "none"


class ExtraInterventionPolicy(StrEnum):
    """Handling for subjects exposed to undeclared interventions."""

    ERROR = "error"
    EXCLUDE = "exclude"


class MissingEndpointPolicy(StrEnum):
    """Handling for assigned subjects without an analyzable endpoint."""

    ERROR = "error"
    EXCLUDE = "exclude"


class EndpointWeighting(StrEnum):
    """How subject-level endpoint uncertainty enters regression."""

    UNWEIGHTED = "unweighted"
    INVERSE_VARIANCE_REQUIRED = "inverse_variance_required"
    INVERSE_VARIANCE_IF_COMPLETE = "inverse_variance_if_complete"


def _factorial_terms(
    interventions: tuple[str, ...],
    maximum_order: int,
) -> tuple[tuple[str, ...], ...]:
    """Return the canonical main-effect and interaction term order."""
    return tuple(
        members
        for order in range(1, maximum_order + 1)
        for members in combinations(interventions, order)
    )


def _float_matches(first: float, second: float) -> bool:
    """Compare reconstructed report values without hiding meaningful drift."""
    return isclose(first, second, rel_tol=1e-10, abs_tol=1e-12)


def _probability_matches(first: float, second: float) -> bool:
    """Compare probabilities without treating tiny nonzero tails as zero."""
    return isclose(first, second, rel_tol=1e-10, abs_tol=0.0)


class FactorialCombinationConfig(BaseModel):
    """Prespecified factorial estimand and regression policy."""

    model_config = ConfigDict(frozen=True)

    interventions: tuple[str, ...] = Field(min_length=2)
    assignment_mechanism: AssignmentMechanism
    covariance_estimator: CovarianceEstimator = CovarianceEstimator.HC3
    multiplicity_method: MultiplicityMethod = MultiplicityMethod.BENJAMINI_HOCHBERG
    minimum_cell_size: int = Field(default=3, ge=2)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    maximum_interaction_order: int | None = Field(default=None, ge=2)
    covariates: tuple[str, ...] = ()
    include_baseline_covariate: bool = False
    endpoint_weighting: EndpointWeighting = EndpointWeighting.UNWEIGHTED
    extra_intervention_policy: ExtraInterventionPolicy = ExtraInterventionPolicy.ERROR
    missing_endpoint_policy: MissingEndpointPolicy = MissingEndpointPolicy.EXCLUDE
    maximum_condition_number: float = Field(default=1e10, gt=1)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Reject duplicate or non-identifiable model declarations."""
        if len(set(self.interventions)) != len(self.interventions):
            raise ValueError("interventions must be unique")
        if any(not item.strip() for item in self.interventions):
            raise ValueError("intervention names cannot be blank")
        if len(set(self.covariates)) != len(self.covariates):
            raise ValueError("covariates must be unique")
        if any(not item.strip() for item in self.covariates):
            raise ValueError("covariate names cannot be blank")
        if self.maximum_interaction_order is not None and self.maximum_interaction_order > len(
            self.interventions
        ):
            raise ValueError("maximum_interaction_order cannot exceed intervention count")
        terms = _factorial_terms(
            self.interventions,
            self.maximum_interaction_order or len(self.interventions),
        )
        design_names = ["intercept", *(":".join(term) for term in terms)]
        if self.include_baseline_covariate:
            design_names.append("baseline_endpoint")
        design_names.extend(f"covariate:{item}" for item in self.covariates)
        if len(set(design_names)) != len(design_names):
            raise ValueError(
                "intervention and covariate names produce ambiguous factorial design terms"
            )
        return self

    @property
    def interaction_order(self) -> int:
        """Return the largest fitted interaction order."""
        return self.maximum_interaction_order or len(self.interventions)


class FactorialContrastWeight(BaseModel):
    """One cell coefficient in a factorial departure-from-additivity contrast."""

    model_config = ConfigDict(frozen=True)

    active_interventions: tuple[str, ...]
    coefficient: float


class InteractionEstimate(BaseModel):
    """Estimated departure from additivity on one declared outcome scale.

    A positive or negative coefficient is not automatically biological synergy or
    antagonism. Its interpretation depends on the estimand direction and outcome scale.
    """

    model_config = ConfigDict(frozen=True)

    interventions: tuple[str, ...] = Field(min_length=2)
    outcome: str
    interaction: float
    standard_error: float = Field(gt=0)
    reference_model: str
    estimand: Estimand | None = None
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    confidence_interval: tuple[float, float] | None = None
    statistic: float | None = None
    degrees_of_freedom: int | None = Field(default=None, ge=1)
    p_value: float | None = Field(default=None, ge=0, le=1)
    adjusted_p_value: float | None = Field(default=None, ge=0, le=1)
    analyzable_subjects: int | None = Field(default=None, ge=1)
    assignment_mechanism: AssignmentMechanism | None = None
    covariance_estimator: CovarianceEstimator | None = None
    contrast_weights: tuple[FactorialContrastWeight, ...] = ()
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_estimate(self) -> Self:
        """Validate numeric output and interaction identity."""
        if len(set(self.interventions)) != len(self.interventions):
            raise ValueError("interaction interventions must be unique")
        numeric = (self.interaction, self.standard_error, self.statistic)
        if any(value is not None and not isfinite(value) for value in numeric):
            raise ValueError("interaction numeric values must be finite")
        if self.confidence_interval is not None:
            low, high = self.confidence_interval
            if (
                not isfinite(low)
                or not isfinite(high)
                or low > self.interaction
                or high < self.interaction
            ):
                raise ValueError("confidence_interval must be finite and contain interaction")
        if len(set(self.quality_flags)) != len(self.quality_flags):
            raise ValueError("quality_flags must be unique")
        return self


class FactorialCellSummary(BaseModel):
    """Assignment and endpoint coverage for one factorial cell."""

    model_config = ConfigDict(frozen=True)

    active_interventions: tuple[str, ...]
    assigned_subjects: int = Field(ge=0)
    analyzable_subjects: int = Field(ge=0)
    excluded_subject_ids: tuple[str, ...] = ()
    mean_endpoint: float | None = None
    standard_error: float | None = Field(default=None, ge=0)


class ModelCoefficient(BaseModel):
    """One fitted factorial or covariate regression coefficient."""

    model_config = ConfigDict(frozen=True)

    term: str
    estimate: float
    standard_error: float = Field(gt=0)
    statistic: float
    p_value: float = Field(ge=0, le=1)


class CombinationDesignDiagnostic(BaseModel):
    """Identifiability, balance, and exclusion diagnostics for the fitted model."""

    model_config = ConfigDict(frozen=True)

    design_columns: tuple[str, ...]
    design_rank: int = Field(ge=1)
    residual_degrees_of_freedom: int = Field(ge=1)
    condition_number: float = Field(ge=1)
    required_cells: int = Field(ge=4)
    observed_cells: int = Field(ge=0)
    minimum_analyzable_cell_size: int = Field(ge=0)
    maximum_analyzable_cell_size: int = Field(ge=0)
    excluded_subject_ids: tuple[str, ...] = ()
    endpoint_weighting_used: EndpointWeighting


class CombinationAnalysisReport(BaseModel):
    """Complete factorial analysis with estimates, cells, and audit diagnostics."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    study_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome: str
    estimand: Estimand
    config: FactorialCombinationConfig
    endpoint_batch_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoint_source_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    interactions: tuple[InteractionEstimate, ...]
    coefficients: tuple[ModelCoefficient, ...]
    cells: tuple[FactorialCellSummary, ...]
    diagnostics: CombinationDesignDiagnostic
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_report_reconstruction(self) -> Self:
        """Reconstruct every cross-field identity available without raw study rows."""
        if not self.study_id.strip() or not self.outcome.strip():
            raise ValueError("combination report study_id and outcome must be nonblank")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("combination report warnings must be unique")

        terms = _factorial_terms(self.config.interventions, self.config.interaction_order)
        expected_columns = ["intercept", *(":".join(term) for term in terms)]
        if self.config.include_baseline_covariate:
            expected_columns.append("baseline_endpoint")
        expected_columns.extend(f"covariate:{item}" for item in self.config.covariates)
        if self.diagnostics.design_columns != tuple(expected_columns):
            raise ValueError("combination report design columns do not match its configuration")
        if self.diagnostics.design_rank != len(expected_columns):
            raise ValueError("combination report design rank must equal its declared columns")
        if not isfinite(self.diagnostics.condition_number):
            raise ValueError("combination report condition number must be finite")

        expected_cells = tuple(
            tuple(
                name
                for name, enabled in zip(self.config.interventions, bits, strict=True)
                if enabled
            )
            for bits in product((0, 1), repeat=len(self.config.interventions))
        )
        observed_cells = tuple(item.active_interventions for item in self.cells)
        if observed_cells != expected_cells:
            raise ValueError("combination report must contain every factorial cell exactly once")
        for cell in self.cells:
            if cell.analyzable_subjects > cell.assigned_subjects:
                raise ValueError("factorial cell analyzable count exceeds assigned count")
            if cell.analyzable_subjects < self.config.minimum_cell_size:
                raise ValueError("factorial cell violates configured minimum cell size")
            if len(set(cell.excluded_subject_ids)) != len(cell.excluded_subject_ids):
                raise ValueError("factorial cell excluded subjects must be unique")
            if len(cell.excluded_subject_ids) != (
                cell.assigned_subjects - cell.analyzable_subjects
            ):
                raise ValueError("factorial cell counts do not match its exclusions")
            if cell.mean_endpoint is None or not isfinite(cell.mean_endpoint):
                raise ValueError("analyzable factorial cells require a finite mean endpoint")
            if cell.standard_error is None or not isfinite(cell.standard_error):
                raise ValueError("analyzable factorial cells require finite uncertainty")

        analyzable = sum(item.analyzable_subjects for item in self.cells)
        cell_sizes = tuple(item.analyzable_subjects for item in self.cells)
        expected_required_cells = 2 ** len(self.config.interventions)
        if self.diagnostics.required_cells != expected_required_cells:
            raise ValueError("combination report required-cell count is inconsistent")
        if self.diagnostics.observed_cells != sum(size > 0 for size in cell_sizes):
            raise ValueError("combination report observed-cell count is inconsistent")
        if self.diagnostics.minimum_analyzable_cell_size != min(cell_sizes):
            raise ValueError("combination report minimum cell size is inconsistent")
        if self.diagnostics.maximum_analyzable_cell_size != max(cell_sizes):
            raise ValueError("combination report maximum cell size is inconsistent")
        if self.diagnostics.residual_degrees_of_freedom != analyzable - len(expected_columns):
            raise ValueError("combination report residual degrees of freedom are inconsistent")
        diagnostic_exclusions = self.diagnostics.excluded_subject_ids
        if len(set(diagnostic_exclusions)) != len(diagnostic_exclusions):
            raise ValueError("combination report diagnostic exclusions must be unique")
        flattened_cell_exclusions = tuple(
            subject_id for cell in self.cells for subject_id in cell.excluded_subject_ids
        )
        if len(set(flattened_cell_exclusions)) != len(flattened_cell_exclusions):
            raise ValueError("factorial cell exclusion partitions must be disjoint")
        if tuple(sorted(flattened_cell_exclusions)) != diagnostic_exclusions:
            raise ValueError("factorial cell exclusions must exactly match design diagnostics")

        if len(self.coefficients) != len(expected_columns):
            raise ValueError("combination report coefficient count does not match its design")
        if tuple(item.term for item in self.coefficients) != tuple(expected_columns):
            raise ValueError("combination report coefficient terms do not match its design")
        coefficient_by_term = {item.term: item for item in self.coefficients}
        for coefficient in self.coefficients:
            if not all(
                isfinite(value)
                for value in (
                    coefficient.estimate,
                    coefficient.standard_error,
                    coefficient.statistic,
                )
            ):
                raise ValueError("combination report coefficients must be finite")
            if not _float_matches(
                coefficient.statistic,
                coefficient.estimate / coefficient.standard_error,
            ):
                raise ValueError("combination report coefficient statistic is inconsistent")
            expected_p = float(
                2
                * student_t.sf(
                    abs(coefficient.statistic),
                    df=self.diagnostics.residual_degrees_of_freedom,
                )
            )
            if not _probability_matches(coefficient.p_value, expected_p):
                raise ValueError("combination report coefficient p-value is inconsistent")

        expected_interactions = tuple(term for term in terms if len(term) >= 2)
        if tuple(item.interventions for item in self.interactions) != expected_interactions:
            raise ValueError("combination report interaction family does not match configuration")
        raw_p_values = tuple(item.p_value for item in self.interactions)
        if any(value is None for value in raw_p_values):
            raise ValueError("combination report interactions require raw p-values")
        adjusted = _adjust_p_values(
            tuple(float(value) for value in raw_p_values if value is not None),
            self.config.multiplicity_method,
        )
        quantile = float(
            student_t.ppf(
                0.5 + self.config.confidence_level / 2,
                df=self.diagnostics.residual_degrees_of_freedom,
            )
        )
        expected_flags = (
            ("observational_assignment_noncausal",)
            if self.config.assignment_mechanism is AssignmentMechanism.OBSERVATIONAL
            else ()
        )
        for index, interaction in enumerate(self.interactions):
            term = ":".join(interaction.interventions)
            coefficient = coefficient_by_term[term]
            if interaction.outcome != self.outcome or interaction.estimand != self.estimand:
                raise ValueError("combination report interaction estimand identity is inconsistent")
            if interaction.assignment_mechanism is not self.config.assignment_mechanism:
                raise ValueError("combination report interaction assignment is inconsistent")
            if interaction.covariance_estimator is not self.config.covariance_estimator:
                raise ValueError("combination report interaction covariance policy is inconsistent")
            if interaction.confidence_level != self.config.confidence_level:
                raise ValueError("combination report interaction confidence level is inconsistent")
            if interaction.reference_model != "additive_on_declared_outcome_scale":
                raise ValueError("combination report interaction reference model is inconsistent")
            if interaction.analyzable_subjects != analyzable:
                raise ValueError("combination report interaction subject count is inconsistent")
            if interaction.degrees_of_freedom != self.diagnostics.residual_degrees_of_freedom:
                raise ValueError(
                    "combination report interaction degrees of freedom are inconsistent"
                )
            if interaction.quality_flags != expected_flags:
                raise ValueError("combination report interaction quality flags are inconsistent")
            if interaction.contrast_weights != _contrast_weights(interaction.interventions):
                raise ValueError("combination report interaction contrast weights are inconsistent")
            if not _float_matches(
                interaction.interaction, coefficient.estimate
            ) or not _float_matches(
                interaction.standard_error,
                coefficient.standard_error,
            ):
                raise ValueError("combination report interaction coefficient is inconsistent")
            if interaction.statistic is None or not _float_matches(
                interaction.statistic,
                coefficient.statistic,
            ):
                raise ValueError("combination report interaction statistic is inconsistent")
            if interaction.p_value is None or not _probability_matches(
                interaction.p_value,
                coefficient.p_value,
            ):
                raise ValueError("combination report interaction p-value is inconsistent")
            if interaction.adjusted_p_value is None or not _probability_matches(
                interaction.adjusted_p_value,
                adjusted[index],
            ):
                raise ValueError("combination report adjusted p-value is inconsistent")
            expected_interval = (
                interaction.interaction - quantile * interaction.standard_error,
                interaction.interaction + quantile * interaction.standard_error,
            )
            if interaction.confidence_interval is None or any(
                not _float_matches(observed, expected)
                for observed, expected in zip(
                    interaction.confidence_interval,
                    expected_interval,
                    strict=True,
                )
            ):
                raise ValueError("combination report confidence interval is inconsistent")

        observational_warning = "observational_assignment_noncausal" in self.warnings
        if observational_warning != (
            self.config.assignment_mechanism is AssignmentMechanism.OBSERVATIONAL
        ):
            raise ValueError("combination report observational warning is inconsistent")
        exclusion_warning = "subjects_excluded_from_factorial_analysis" in self.warnings
        if exclusion_warning != bool(self.diagnostics.excluded_subject_ids):
            raise ValueError("combination report exclusion warning is inconsistent")
        weak_warning = "factorial_design_weakly_conditioned" in self.warnings
        if weak_warning != (
            self.diagnostics.condition_number > sqrt(self.config.maximum_condition_number)
        ):
            raise ValueError("combination report conditioning warning is inconsistent")
        if self.config.endpoint_weighting is EndpointWeighting.UNWEIGHTED:
            if self.diagnostics.endpoint_weighting_used is not EndpointWeighting.UNWEIGHTED:
                raise ValueError("combination report endpoint weighting is inconsistent")
        elif self.config.endpoint_weighting is EndpointWeighting.INVERSE_VARIANCE_REQUIRED:
            if (
                self.diagnostics.endpoint_weighting_used
                is not EndpointWeighting.INVERSE_VARIANCE_REQUIRED
            ):
                raise ValueError("combination report endpoint weighting is inconsistent")
        elif self.diagnostics.endpoint_weighting_used not in {
            EndpointWeighting.UNWEIGHTED,
            EndpointWeighting.INVERSE_VARIANCE_IF_COMPLETE,
        }:
            raise ValueError("combination report endpoint weighting is inconsistent")
        incomplete_warning = "incomplete_endpoint_uncertainty_used_unweighted_fit" in self.warnings
        if self.config.endpoint_weighting is EndpointWeighting.INVERSE_VARIANCE_IF_COMPLETE:
            if incomplete_warning != (
                self.diagnostics.endpoint_weighting_used is EndpointWeighting.UNWEIGHTED
            ):
                raise ValueError("combination report endpoint-weighting warning is inconsistent")
        elif incomplete_warning:
            raise ValueError("combination report endpoint-weighting warning is inconsistent")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_hash(self) -> str:
        """Return a deterministic identity for the complete analysis result."""
        payload = {
            "schema": "factorial-combination-report/v1",
            "study_id": self.study_id,
            "study_artifact_hash": self.study_artifact_hash,
            "outcome": self.outcome,
            "estimand": self.estimand.model_dump(mode="json"),
            "config": self.config.model_dump(mode="json"),
            "endpoint_batch_hash": self.endpoint_batch_hash,
            "endpoint_source_artifact_hash": self.endpoint_source_artifact_hash,
            "interactions": [item.model_dump(mode="json") for item in self.interactions],
            "coefficients": [item.model_dump(mode="json") for item in self.coefficients],
            "cells": [item.model_dump(mode="json") for item in self.cells],
            "diagnostics": self.diagnostics.model_dump(mode="json"),
            "warnings": self.warnings,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    def interactions_frame(self) -> pd.DataFrame:
        """Return one tidy row per prespecified interaction term."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.interactions)

    def cells_frame(self) -> pd.DataFrame:
        """Return one tidy row per factorial assignment cell."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.cells)


class CombinationAnalysis(Protocol):
    """Contract for combination-therapy estimators."""

    def estimate(
        self,
        study: Study,
        *,
        outcome: str,
        endpoints: SubjectEndpointBatch | None = None,
    ) -> tuple[InteractionEstimate, ...]:
        """Estimate prespecified departures from additivity."""
        ...


class FactorialCombinationAnalysis:
    """Fit subject-level factorial OLS with classical or HC3 uncertainty."""

    def __init__(self, config: FactorialCombinationConfig) -> None:
        """Create a prespecified factorial analysis."""
        self.config = config
        self.report_: CombinationAnalysisReport | None = None

    def estimate(
        self,
        study: Study,
        *,
        outcome: str,
        endpoints: SubjectEndpointBatch | None = None,
    ) -> tuple[InteractionEstimate, ...]:
        """Return interaction estimates while retaining the full report on ``report_``."""
        if endpoints is None:
            raise ValueError(
                "a SubjectEndpointBatch is required; construct one endpoint per independent subject"
            )
        report = self.analyze(study, endpoints=endpoints, outcome=outcome)
        self.report_ = report
        return report.interactions

    def analyze(
        self,
        study: Study,
        *,
        endpoints: SubjectEndpointBatch,
        outcome: str | None = None,
    ) -> CombinationAnalysisReport:
        """Fit the declared factorial model to exactly one endpoint per subject."""
        if endpoints.study_id != study.study_id:
            raise ValueError("endpoint batch study_id does not match the study")
        outcome_name = outcome or endpoints.estimand.name
        if outcome is not None and outcome not in {endpoints.estimand.name, endpoints.estimand.key}:
            raise ValueError("outcome does not match the endpoint estimand")

        rows, cells, excluded, weighting_used, warnings = self._analysis_rows(study, endpoints)
        design, response, column_names, interaction_columns, weights = self._design_matrix(rows)
        transformed_design, transformed_response = _apply_weights(design, response, weights)
        rank = int(np.linalg.matrix_rank(transformed_design))
        if rank != transformed_design.shape[1]:
            raise ValueError(
                f"factorial design is rank deficient: rank={rank}, "
                f"columns={transformed_design.shape[1]}"
            )
        residual_df = transformed_design.shape[0] - transformed_design.shape[1]
        if residual_df < 1:
            raise ValueError("factorial design requires at least one residual degree of freedom")
        condition = float(np.linalg.cond(transformed_design))
        if not isfinite(condition) or condition > self.config.maximum_condition_number:
            raise ValueError(
                "factorial design condition number exceeds configured maximum: "
                f"{condition:.6g} > {self.config.maximum_condition_number:.6g}"
            )

        beta, covariance = _fit_regression(
            transformed_design,
            transformed_response,
            self.config.covariance_estimator,
            residual_df,
        )
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        if np.any(~np.isfinite(standard_errors)) or np.any(standard_errors <= 0):
            raise ValueError("coefficient uncertainty is zero or non-finite")
        statistics = beta / standard_errors
        raw_p = np.asarray(
            [2 * student_t.sf(abs(value), df=residual_df) for value in statistics],
            dtype=np.float64,
        )
        coefficients = tuple(
            ModelCoefficient(
                term=name,
                estimate=float(beta[index]),
                standard_error=float(standard_errors[index]),
                statistic=float(statistics[index]),
                p_value=float(raw_p[index]),
            )
            for index, name in enumerate(column_names)
        )

        interaction_indices = tuple(index for index, _ in interaction_columns)
        interaction_p = tuple(float(raw_p[index]) for index in interaction_indices)
        adjusted = _adjust_p_values(interaction_p, self.config.multiplicity_method)
        quantile = float(student_t.ppf(0.5 + self.config.confidence_level / 2, df=residual_df))
        interaction_results: list[InteractionEstimate] = []
        for adjusted_index, (column_index, members) in enumerate(interaction_columns):
            value = float(beta[column_index])
            error = float(standard_errors[column_index])
            flags: list[str] = []
            if self.config.assignment_mechanism is AssignmentMechanism.OBSERVATIONAL:
                flags.append("observational_assignment_noncausal")
            interaction_results.append(
                InteractionEstimate(
                    interventions=members,
                    outcome=outcome_name,
                    interaction=value,
                    standard_error=error,
                    reference_model="additive_on_declared_outcome_scale",
                    estimand=endpoints.estimand,
                    confidence_level=self.config.confidence_level,
                    confidence_interval=(value - quantile * error, value + quantile * error),
                    statistic=float(statistics[column_index]),
                    degrees_of_freedom=residual_df,
                    p_value=float(raw_p[column_index]),
                    adjusted_p_value=adjusted[adjusted_index],
                    analyzable_subjects=len(rows),
                    assignment_mechanism=self.config.assignment_mechanism,
                    covariance_estimator=self.config.covariance_estimator,
                    contrast_weights=_contrast_weights(members),
                    quality_flags=tuple(flags),
                )
            )

        cell_sizes = [item.analyzable_subjects for item in cells]
        diagnostics = CombinationDesignDiagnostic(
            design_columns=column_names,
            design_rank=rank,
            residual_degrees_of_freedom=residual_df,
            condition_number=max(condition, 1.0),
            required_cells=2 ** len(self.config.interventions),
            observed_cells=sum(size > 0 for size in cell_sizes),
            minimum_analyzable_cell_size=min(cell_sizes),
            maximum_analyzable_cell_size=max(cell_sizes),
            excluded_subject_ids=tuple(sorted(excluded)),
            endpoint_weighting_used=weighting_used,
        )
        if condition > sqrt(self.config.maximum_condition_number):
            warnings.append("factorial_design_weakly_conditioned")
        report = CombinationAnalysisReport(
            study_id=study.study_id,
            study_artifact_hash=study_artifact_hash(study),
            outcome=outcome_name,
            estimand=endpoints.estimand,
            config=self.config,
            endpoint_batch_hash=endpoints.artifact_hash,
            endpoint_source_artifact_hash=endpoints.source_artifact_hash,
            interactions=tuple(interaction_results),
            coefficients=coefficients,
            cells=cells,
            diagnostics=diagnostics,
            warnings=tuple(sorted(set(warnings))),
        )
        self.report_ = report
        return report

    def _analysis_rows(
        self,
        study: Study,
        endpoints: SubjectEndpointBatch,
    ) -> tuple[
        list[tuple[str, tuple[int, ...], float, float | None, tuple[float, ...]]],
        tuple[FactorialCellSummary, ...],
        set[str],
        EndpointWeighting,
        list[str],
    ]:
        subject_by_id = {item.subject_id: item for item in study.subjects}
        endpoint_by_id = endpoints.by_subject()
        endpoint_subject_ids = set(endpoint_by_id)
        excluded_endpoint_ids = {item.subject_id for item in endpoints.excluded}
        unknown = (endpoint_subject_ids | excluded_endpoint_ids).difference(subject_by_id)
        if unknown:
            raise ValueError(f"endpoint batch references unknown subjects: {sorted(unknown)}")

        declared = set(self.config.interventions)
        assignments: dict[tuple[int, ...], list[str]] = defaultdict(list)
        excluded: set[str] = set(excluded_endpoint_ids)
        candidate_subjects: list[str] = []
        for subject in sorted(study.subjects, key=lambda item: item.subject_id):
            if len(subject.interventions) != len(set(subject.interventions)):
                raise ValueError(f"subject {subject.subject_id} has duplicate interventions")
            bits = tuple(int(name in subject.interventions) for name in self.config.interventions)
            assignments[bits].append(subject.subject_id)
            extras = set(subject.interventions).difference(declared)
            if extras:
                if self.config.extra_intervention_policy is ExtraInterventionPolicy.ERROR:
                    message = (
                        f"subject {subject.subject_id} has undeclared interventions: "
                        f"{sorted(extras)}"
                    )
                    raise ValueError(message)
                excluded.add(subject.subject_id)
                continue
            candidate_subjects.append(subject.subject_id)

        missing = set(candidate_subjects).difference(endpoint_by_id)
        if missing and self.config.missing_endpoint_policy is MissingEndpointPolicy.ERROR:
            raise ValueError(f"subjects are missing endpoints: {sorted(missing)}")
        excluded.update(missing)

        rows: list[tuple[str, tuple[int, ...], float, float | None, tuple[float, ...]]] = []
        endpoint_errors: list[str] = []
        for subject_id in sorted(endpoint_by_id):
            if subject_id in excluded:
                continue
            subject = subject_by_id[subject_id]
            bits = tuple(int(name in subject.interventions) for name in self.config.interventions)
            endpoint = endpoint_by_id[subject_id]
            covariates: list[float] = []
            invalid = False
            if self.config.include_baseline_covariate:
                if endpoint.baseline_estimate is None:
                    endpoint_errors.append(f"{subject_id}:baseline_estimate")
                    invalid = True
                else:
                    covariates.append(endpoint.baseline_estimate)
            for name in self.config.covariates:
                value = subject.attributes.get(name)
                if isinstance(value, bool):
                    covariates.append(float(value))
                elif isinstance(value, (int, float)) and isfinite(float(value)):
                    covariates.append(float(value))
                else:
                    endpoint_errors.append(f"{subject_id}:{name}")
                    invalid = True
            if invalid:
                excluded.add(subject_id)
                continue
            rows.append(
                (
                    subject_id,
                    bits,
                    endpoint.estimate,
                    endpoint.standard_error,
                    tuple(covariates),
                )
            )
        if endpoint_errors and self.config.missing_endpoint_policy is MissingEndpointPolicy.ERROR:
            raise ValueError(f"missing or invalid endpoint covariates: {sorted(endpoint_errors)}")

        analyzable_by_cell: defaultdict[tuple[int, ...], list[float]] = defaultdict(list)
        for _, bits, value, _, _ in rows:
            analyzable_by_cell[bits].append(value)
        cell_summaries: list[FactorialCellSummary] = []
        for bits in product((0, 1), repeat=len(self.config.interventions)):
            key = tuple(bits)
            assigned_ids = assignments.get(key, [])
            values = analyzable_by_cell.get(key, [])
            if len(values) < self.config.minimum_cell_size:
                active = tuple(
                    name
                    for name, enabled in zip(self.config.interventions, key, strict=True)
                    if enabled
                )
                raise ValueError(
                    f"factorial cell {active or ('control',)} has {len(values)} analyzable "
                    f"subjects; minimum is {self.config.minimum_cell_size}"
                )
            standard_error = (
                float(np.std(values, ddof=1) / sqrt(len(values))) if len(values) > 1 else None
            )
            cell_summaries.append(
                FactorialCellSummary(
                    active_interventions=tuple(
                        name
                        for name, enabled in zip(self.config.interventions, key, strict=True)
                        if enabled
                    ),
                    assigned_subjects=len(assigned_ids),
                    analyzable_subjects=len(values),
                    excluded_subject_ids=tuple(
                        sorted(subject_id for subject_id in assigned_ids if subject_id in excluded)
                    ),
                    mean_endpoint=float(np.mean(values)),
                    standard_error=standard_error,
                )
            )

        standard_errors = [row[3] for row in rows]
        warnings: list[str] = []
        if self.config.assignment_mechanism is AssignmentMechanism.OBSERVATIONAL:
            warnings.append("observational_assignment_noncausal")
        weighting_used = EndpointWeighting.UNWEIGHTED
        if self.config.endpoint_weighting is EndpointWeighting.INVERSE_VARIANCE_REQUIRED:
            if any(value is None for value in standard_errors):
                raise ValueError(
                    "inverse-variance weighting requires every endpoint standard error"
                )
            weighting_used = EndpointWeighting.INVERSE_VARIANCE_REQUIRED
        elif self.config.endpoint_weighting is EndpointWeighting.INVERSE_VARIANCE_IF_COMPLETE:
            if all(value is not None for value in standard_errors):
                weighting_used = EndpointWeighting.INVERSE_VARIANCE_IF_COMPLETE
            else:
                warnings.append("incomplete_endpoint_uncertainty_used_unweighted_fit")
        elif any(value is not None for value in standard_errors):
            warnings.append("endpoint_standard_errors_not_propagated")
        if excluded:
            warnings.append("subjects_excluded_from_factorial_analysis")
        return rows, tuple(cell_summaries), excluded, weighting_used, warnings

    def _design_matrix(
        self,
        rows: list[tuple[str, tuple[int, ...], float, float | None, tuple[float, ...]]],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        tuple[str, ...],
        tuple[tuple[int, tuple[str, ...]], ...],
        npt.NDArray[np.float64] | None,
    ]:
        treatment_terms = _factorial_terms(
            self.config.interventions,
            self.config.interaction_order,
        )
        names = ["intercept", *(":".join(term) for term in treatment_terms)]
        if self.config.include_baseline_covariate:
            names.append("baseline_endpoint")
        names.extend(f"covariate:{item}" for item in self.config.covariates)
        positions = {name: index for index, name in enumerate(self.config.interventions)}
        matrix: list[list[float]] = []
        response: list[float] = []
        errors: list[float | None] = []
        for _, bits, value, error, covariates in rows:
            treatment_values = [
                float(np.prod([bits[positions[name]] for name in term])) for term in treatment_terms
            ]
            matrix.append([1.0, *treatment_values, *covariates])
            response.append(value)
            errors.append(error)
        interaction_columns = tuple(
            (index + 1, term) for index, term in enumerate(treatment_terms) if len(term) >= 2
        )
        weights: npt.NDArray[np.float64] | None = None
        if self.config.endpoint_weighting is EndpointWeighting.INVERSE_VARIANCE_REQUIRED or (
            self.config.endpoint_weighting is EndpointWeighting.INVERSE_VARIANCE_IF_COMPLETE
            and all(value is not None for value in errors)
        ):
            if any(value is None for value in errors):
                raise RuntimeError("internal endpoint weighting contract was not satisfied")
            raw = np.asarray(
                [1.0 / value**2 for value in errors if value is not None],
                dtype=np.float64,
            )
            weights = raw / float(np.mean(raw))
        return (
            np.asarray(matrix, dtype=np.float64),
            np.asarray(response, dtype=np.float64),
            tuple(names),
            interaction_columns,
            weights,
        )


class TwoByTwoDesignConfig(BaseModel):
    """Assumptions for an equal-allocation Gaussian two-by-two interaction design."""

    model_config = ConfigDict(frozen=True)

    interventions: tuple[str, str]
    target_interaction: float
    residual_standard_deviation: float = Field(gt=0)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    power: float = Field(default=0.80, gt=0.5, lt=1)
    multiplicity_tests: int = Field(default=1, ge=1)
    expected_dropout_fraction: float = Field(default=0.0, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        """Require two named interventions and a nonzero target interaction."""
        if len(set(self.interventions)) != 2 or any(
            not item.strip() for item in self.interventions
        ):
            raise ValueError("two distinct nonblank interventions are required")
        if not isfinite(self.target_interaction) or self.target_interaction == 0:
            raise ValueError("target_interaction must be finite and nonzero")
        return self


class FactorialDesignRecommendation(BaseModel):
    """Approximate equal-cell enrollment for a continuous two-by-two interaction."""

    model_config = ConfigDict(frozen=True)

    interventions: tuple[str, str]
    analyzable_subjects_per_cell: int = Field(ge=2)
    enrolled_subjects_per_cell: int = Field(ge=2)
    total_analyzable_subjects: int = Field(ge=8)
    total_enrollment: int = Field(ge=8)
    expected_interaction_standard_error: float = Field(gt=0)
    minimum_detectable_interaction: float = Field(gt=0)
    nominal_alpha_per_test: float = Field(gt=0, lt=0.5)
    target_power: float = Field(gt=0.5, lt=1)
    assumptions: tuple[str, ...]


class DesignCellAllocation(BaseModel):
    """A deterministic planned assignment cell for one subject."""

    model_config = ConfigDict(frozen=True)

    subject_id: str
    active_interventions: tuple[str, ...]


def recommend_two_by_two_design(config: TwoByTwoDesignConfig) -> FactorialDesignRecommendation:
    """Plan equal cell sizes using the normal approximation to a four-cell contrast."""
    alpha_per_test = config.alpha / config.multiplicity_tests
    normal = NormalDist()
    critical = normal.inv_cdf(1 - alpha_per_test / 2)
    power_quantile = normal.inv_cdf(config.power)
    numerator = 2 * config.residual_standard_deviation * (critical + power_quantile)
    analyzable = max(2, ceil((numerator / abs(config.target_interaction)) ** 2))
    enrolled = ceil(analyzable / (1 - config.expected_dropout_fraction))
    expected_se = 2 * config.residual_standard_deviation / sqrt(analyzable)
    minimum_detectable = expected_se * (critical + power_quantile)
    return FactorialDesignRecommendation(
        interventions=config.interventions,
        analyzable_subjects_per_cell=analyzable,
        enrolled_subjects_per_cell=enrolled,
        total_analyzable_subjects=4 * analyzable,
        total_enrollment=4 * enrolled,
        expected_interaction_standard_error=expected_se,
        minimum_detectable_interaction=minimum_detectable,
        nominal_alpha_per_test=alpha_per_test,
        target_power=config.power,
        assumptions=(
            "independent_subjects",
            "continuous_gaussian_endpoint",
            "equal_cell_allocation",
            "common_residual_standard_deviation",
            "two_sided_normal_approximation",
            "no_clinic_or_litter_clustering",
            "not_for_survival_or_repeated_measure_endpoints",
        ),
    )


def balanced_factorial_allocation(
    subject_ids: tuple[str, ...],
    *,
    interventions: tuple[str, str],
    random_seed: int = 0,
) -> tuple[DesignCellAllocation, ...]:
    """Allocate declared subjects evenly across four two-by-two cells."""
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("subject_ids must be unique")
    if len(set(interventions)) != 2 or any(not item.strip() for item in interventions):
        raise ValueError("two distinct nonblank interventions are required")
    ordered = np.asarray(sorted(subject_ids), dtype=object)
    random = np.random.default_rng(random_seed)
    random.shuffle(ordered)
    cells = ((), (interventions[0],), (interventions[1],), interventions)
    assignments = [
        DesignCellAllocation(subject_id=str(subject_id), active_interventions=cells[index % 4])
        for index, subject_id in enumerate(ordered)
    ]
    return tuple(sorted(assignments, key=lambda item: item.subject_id))


def _apply_weights(
    design: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if weights is None:
        return design, response
    root = np.sqrt(weights)
    return design * root[:, np.newaxis], response * root


def _fit_regression(
    design: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    covariance_estimator: CovarianceEstimator,
    residual_df: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    beta = np.asarray(np.linalg.lstsq(design, response, rcond=None)[0], dtype=np.float64)
    residuals = response - design @ beta
    bread = np.asarray(np.linalg.inv(design.T @ design), dtype=np.float64)
    if covariance_estimator is CovarianceEstimator.CLASSICAL:
        residual_variance = float(residuals @ residuals / residual_df)
        covariance = bread * residual_variance
    else:
        leverage = np.einsum("ij,jk,ik->i", design, bread, design)
        if np.any(leverage >= 1 - 1e-12):
            raise ValueError("HC3 covariance is undefined for unit-leverage observations")
        adjusted = residuals / (1 - leverage)
        meat = design.T @ (design * adjusted[:, np.newaxis] ** 2)
        covariance = bread @ meat @ bread
    return beta, np.asarray((covariance + covariance.T) / 2, dtype=np.float64)


def _adjust_p_values(
    p_values: tuple[float, ...],
    method: MultiplicityMethod,
) -> tuple[float, ...]:
    if not p_values:
        return ()
    if method is MultiplicityMethod.NONE:
        return p_values
    if method is MultiplicityMethod.BONFERRONI:
        return tuple(min(value * len(p_values), 1.0) for value in p_values)
    order = np.argsort(np.asarray(p_values), kind="stable")
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = len(p_values) - reverse_rank + 1
        running = min(running, p_values[int(index)] * len(p_values) / rank)
        adjusted[int(index)] = min(running, 1.0)
    return tuple(float(value) for value in adjusted)


def _contrast_weights(members: tuple[str, ...]) -> tuple[FactorialContrastWeight, ...]:
    weights: list[FactorialContrastWeight] = []
    for enabled in product((0, 1), repeat=len(members)):
        active = tuple(name for name, flag in zip(members, enabled, strict=True) if flag)
        coefficient = float((-1) ** (len(members) - len(active)))
        weights.append(
            FactorialContrastWeight(active_interventions=active, coefficient=coefficient)
        )
    return tuple(weights)
