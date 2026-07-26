"""Phase 1: configurable quality control for longitudinal studies."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from enum import StrEnum
from statistics import fmean, variance
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.schemas import Modality, Observation, Study


class Severity(StrEnum):
    """Severity of a quality-control finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FeatureRule(BaseModel):
    """Validation policy for one measured feature."""

    model_config = ConfigDict(frozen=True)

    feature: str = Field(min_length=1)
    modality: Modality | None = None
    expected_unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    required: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> FeatureRule:
        """Reject inverted feature ranges."""
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        return self


class QCConfig(BaseModel):
    """Configuration for the baseline longitudinal QC pipeline."""

    model_config = ConfigDict(frozen=True)

    feature_rules: tuple[FeatureRule, ...] = ()
    missingness_warning_fraction: float = Field(default=0.10, ge=0, le=1)
    missingness_error_fraction: float = Field(default=0.25, ge=0, le=1)
    replicate_relative_tolerance: float = Field(default=0.20, ge=0)
    replicate_absolute_floor: float = Field(default=1e-12, gt=0)
    batch_z_threshold: float = Field(default=3.0, gt=0)
    minimum_batch_size: int = Field(default=3, ge=2)
    check_input_order: bool = True

    @model_validator(mode="after")
    def validate_missingness_thresholds(self) -> QCConfig:
        """Ensure warning and error missingness thresholds are ordered."""
        if self.missingness_warning_fraction > self.missingness_error_fraction:
            raise ValueError("missingness warning threshold cannot exceed error threshold")
        keys = [(rule.feature, rule.modality) for rule in self.feature_rules]
        if len(keys) != len(set(keys)):
            raise ValueError("feature rules must have unique feature/modality pairs")
        return self


class QCFinding(BaseModel):
    """A machine-readable quality-control finding."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1)
    severity: Severity
    message: str = Field(min_length=1)
    subject_ids: tuple[str, ...] = ()
    observation_indices: tuple[int, ...] = ()
    context: dict[str, str | int | float | bool] = Field(default_factory=dict)


class QCReport(BaseModel):
    """Output of a longitudinal quality-control run."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    findings: tuple[QCFinding, ...]
    metrics: dict[str, int | float] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return whether the report contains no errors."""
        return all(item.severity is not Severity.ERROR for item in self.findings)

    @property
    def counts(self) -> dict[Severity, int]:
        """Count findings by severity, including zero-count categories."""
        observed = Counter(item.severity for item in self.findings)
        return {severity: observed[severity] for severity in Severity}

    def summary(self) -> str:
        """Return a compact human-readable report summary."""
        counts = self.counts
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.study_id}: {status} "
            f"({counts[Severity.ERROR]} errors, "
            f"{counts[Severity.WARNING]} warnings, "
            f"{counts[Severity.INFO]} info)"
        )


class LongitudinalQC(Protocol):
    """Contract for Phase 1 quality-control implementations."""

    def run(self, study: Study) -> QCReport:
        """Evaluate a study without mutating it."""
        ...


class BaselineLongitudinalQC:
    """Reference Phase 1 quality-control pipeline.

    The implementation is deterministic and dependency-light. It detects malformed
    measurements, missing required features, out-of-order rows, replicate disagreement,
    and simple mean shifts between sufficiently large batches.
    """

    def __init__(self, config: QCConfig | None = None) -> None:
        """Create a pipeline with an optional immutable configuration."""
        self.config = config or QCConfig()

    def run(self, study: Study) -> QCReport:
        """Run all configured checks and return structured findings."""
        findings = [
            *self._check_values_and_ranges(study),
            *self._check_missingness(study),
            *self._check_temporal_order(study),
            *self._check_replicates(study),
            *self._check_batch_drift(study),
        ]
        findings.sort(
            key=lambda item: (
                _severity_rank(item.severity),
                item.code,
                item.subject_ids,
                item.observation_indices,
            )
        )
        metrics: dict[str, int | float] = {
            "subjects": len(study.subjects),
            "observations": len(study.observations),
            "features": len({row.feature for row in study.observations}),
            "batches": len(
                {row.batch_id for row in study.observations if row.batch_id is not None}
            ),
        }
        return QCReport(study_id=study.study_id, findings=tuple(findings), metrics=metrics)

    def _matching_rule(self, row: Observation) -> FeatureRule | None:
        exact: FeatureRule | None = None
        generic: FeatureRule | None = None
        for rule in self.config.feature_rules:
            if rule.feature != row.feature:
                continue
            if rule.modality is row.modality:
                exact = rule
            elif rule.modality is None:
                generic = rule
        return exact or generic

    def _check_values_and_ranges(self, study: Study) -> list[QCFinding]:
        findings: list[QCFinding] = []
        for index, row in enumerate(study.observations):
            if not math.isfinite(row.value):
                findings.append(
                    QCFinding(
                        code="nonfinite_value",
                        severity=Severity.ERROR,
                        message=f"{row.feature} has a non-finite value",
                        subject_ids=(row.subject_id,),
                        observation_indices=(index,),
                    )
                )
                continue
            rule = self._matching_rule(row)
            if rule is None:
                continue
            if rule.expected_unit is not None and row.unit != rule.expected_unit:
                findings.append(
                    QCFinding(
                        code="unexpected_unit",
                        severity=Severity.ERROR,
                        message=(
                            f"{row.feature} uses {row.unit!r}; expected {rule.expected_unit!r}"
                        ),
                        subject_ids=(row.subject_id,),
                        observation_indices=(index,),
                        context={"observed_unit": row.unit, "expected_unit": rule.expected_unit},
                    )
                )
            outside_minimum = rule.minimum is not None and row.value < rule.minimum
            outside_maximum = rule.maximum is not None and row.value > rule.maximum
            if outside_minimum or outside_maximum:
                context: dict[str, str | int | float | bool] = {"value": row.value}
                if rule.minimum is not None:
                    context["minimum"] = rule.minimum
                if rule.maximum is not None:
                    context["maximum"] = rule.maximum
                findings.append(
                    QCFinding(
                        code="out_of_range",
                        severity=Severity.ERROR,
                        message=f"{row.feature} value {row.value:g} is outside its allowed range",
                        subject_ids=(row.subject_id,),
                        observation_indices=(index,),
                        context=context,
                    )
                )
        return findings

    def _check_missingness(self, study: Study) -> list[QCFinding]:
        findings: list[QCFinding] = []
        subject_ids = {subject.subject_id for subject in study.subjects}
        required_rules = [rule for rule in self.config.feature_rules if rule.required]
        for rule in required_rules:
            observed = {
                row.subject_id
                for row in study.observations
                if row.feature == rule.feature
                and (rule.modality is None or row.modality is rule.modality)
                and math.isfinite(row.value)
            }
            missing = tuple(sorted(subject_ids - observed))
            if not missing:
                continue
            fraction = len(missing) / len(subject_ids) if subject_ids else 0.0
            if fraction < self.config.missingness_warning_fraction:
                continue
            severity = (
                Severity.ERROR
                if fraction >= self.config.missingness_error_fraction
                else Severity.WARNING
            )
            findings.append(
                QCFinding(
                    code="required_feature_missing",
                    severity=severity,
                    message=(
                        f"{rule.feature} is missing for {len(missing)}/{len(subject_ids)} subjects"
                    ),
                    subject_ids=missing,
                    context={
                        "feature": rule.feature,
                        "missing_fraction": fraction,
                        "missing_subjects": len(missing),
                    },
                )
            )
        return findings

    def _check_temporal_order(self, study: Study) -> list[QCFinding]:
        if not self.config.check_input_order:
            return []
        findings: list[QCFinding] = []
        previous: dict[tuple[str, Modality, str], tuple[int, Observation]] = {}
        for index, row in enumerate(study.observations):
            key = (row.subject_id, row.modality, row.feature)
            prior = previous.get(key)
            if prior is not None and row.timestamp < prior[1].timestamp:
                findings.append(
                    QCFinding(
                        code="timestamp_out_of_order",
                        severity=Severity.WARNING,
                        message=f"{row.feature} observations are not in chronological input order",
                        subject_ids=(row.subject_id,),
                        observation_indices=(prior[0], index),
                    )
                )
            previous[key] = (index, row)
        return findings

    def _check_replicates(self, study: Study) -> list[QCFinding]:
        groups: defaultdict[
            tuple[str, datetime, Modality, str, str | None], list[tuple[int, Observation]]
        ] = defaultdict(list)
        for index, row in enumerate(study.observations):
            if row.replicate_id is not None:
                key = (row.subject_id, row.timestamp, row.modality, row.feature, row.batch_id)
                groups[key].append((index, row))

        findings: list[QCFinding] = []
        for rows in groups.values():
            replicate_ids = {row.replicate_id for _, row in rows}
            if len(rows) < 2 or len(replicate_ids) < 2:
                continue
            values = [row.value for _, row in rows]
            if not all(math.isfinite(value) for value in values):
                continue
            spread = max(values) - min(values)
            scale = max(abs(fmean(values)), self.config.replicate_absolute_floor)
            relative_spread = spread / scale
            if relative_spread > self.config.replicate_relative_tolerance:
                findings.append(
                    QCFinding(
                        code="replicate_disagreement",
                        severity=Severity.WARNING,
                        message=(
                            f"{rows[0][1].feature} replicates differ by "
                            f"{relative_spread:.1%} relative to their mean"
                        ),
                        subject_ids=(rows[0][1].subject_id,),
                        observation_indices=tuple(index for index, _ in rows),
                        context={
                            "relative_spread": relative_spread,
                            "absolute_spread": spread,
                        },
                    )
                )
        return findings

    def _check_batch_drift(self, study: Study) -> list[QCFinding]:
        groups: defaultdict[tuple[Modality, str, str], list[tuple[int, Observation]]] = defaultdict(
            list
        )
        feature_rows: defaultdict[tuple[Modality, str], list[tuple[int, Observation]]] = (
            defaultdict(list)
        )
        for index, row in enumerate(study.observations):
            if row.batch_id is None or not math.isfinite(row.value):
                continue
            groups[(row.modality, row.feature, row.batch_id)].append((index, row))
            feature_rows[(row.modality, row.feature)].append((index, row))

        findings: list[QCFinding] = []
        for (modality, feature, batch_id), rows in groups.items():
            if len(rows) < self.config.minimum_batch_size:
                continue
            all_rows = feature_rows[(modality, feature)]
            other_values = [row.value for _, row in all_rows if row.batch_id != batch_id]
            batch_values = [row.value for _, row in rows]
            if len(other_values) < self.config.minimum_batch_size:
                continue
            degrees_of_freedom = len(batch_values) + len(other_values) - 2
            pooled_variance = (
                (len(batch_values) - 1) * variance(batch_values)
                + (len(other_values) - 1) * variance(other_values)
            ) / degrees_of_freedom
            pooled_std = math.sqrt(pooled_variance)
            if pooled_std <= self.config.replicate_absolute_floor:
                continue
            z_score = abs(fmean(batch_values) - fmean(other_values)) / pooled_std
            if z_score > self.config.batch_z_threshold:
                findings.append(
                    QCFinding(
                        code="batch_mean_shift",
                        severity=Severity.WARNING,
                        message=(
                            f"{feature} batch {batch_id!r} differs from other batches "
                            f"by {z_score:.2f} pooled standard deviations"
                        ),
                        subject_ids=tuple(sorted({row.subject_id for _, row in rows})),
                        observation_indices=tuple(index for index, _ in rows),
                        context={
                            "batch_id": batch_id,
                            "batch_size": len(batch_values),
                            "standardized_shift": z_score,
                        },
                    )
                )
        return findings


def _severity_rank(severity: Severity) -> int:
    return {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}[severity]
