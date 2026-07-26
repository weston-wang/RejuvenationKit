"""Phase 1: configurable quality control for longitudinal studies."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from enum import StrEnum
from statistics import fmean, variance
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.schemas import Modality, Observation, Study, Subject


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


class VisitFeature(BaseModel):
    """A feature expected within a scheduled visit window."""

    model_config = ConfigDict(frozen=True)

    feature: str = Field(min_length=1)
    modality: Modality | None = None


class ExpectedVisit(BaseModel):
    """An expected visit and its required measurements.

    Empty ``subject_ids`` and ``cohorts`` selectors mean the visit applies to all
    study subjects. When either selector is populated, a subject matching either
    selector is included.
    """

    model_config = ConfigDict(frozen=True)

    visit_id: str = Field(min_length=1)
    scheduled_at: datetime | None = None
    anchor_id: str | None = Field(default=None, min_length=1)
    offset: timedelta = timedelta(0)
    window_before: timedelta = Field(default=timedelta(0), ge=timedelta(0))
    window_after: timedelta = Field(default=timedelta(0), ge=timedelta(0))
    required_features: tuple[VisitFeature, ...] = Field(min_length=1)
    subject_ids: tuple[str, ...] = ()
    cohorts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_visit(self) -> ExpectedVisit:
        """Reject ambiguous schedules and duplicate feature requirements."""
        if (self.scheduled_at is None) == (self.anchor_id is None):
            raise ValueError("provide exactly one of scheduled_at or anchor_id")
        if self.scheduled_at is not None and (
            self.scheduled_at.tzinfo is None or self.scheduled_at.utcoffset() is None
        ):
            raise ValueError("scheduled_at must be timezone-aware")
        keys = [(item.feature, item.modality) for item in self.required_features]
        if len(keys) != len(set(keys)):
            raise ValueError("visit required_features must be unique")
        return self

    def scheduled_for(self, subject: Subject) -> datetime | None:
        """Resolve this visit's target time for a subject.

        Returns ``None`` when a relative visit's required anchor is absent.
        """
        if self.scheduled_at is not None:
            return self.scheduled_at
        if self.anchor_id is None:
            return None
        anchor = subject.anchors.get(self.anchor_id)
        return None if anchor is None else anchor + self.offset

    def window_for(self, subject: Subject) -> tuple[datetime, datetime] | None:
        """Resolve the inclusive visit window for a subject."""
        target = self.scheduled_for(subject)
        if target is None:
            return None
        return target - self.window_before, target + self.window_after

    @property
    def window_start(self) -> datetime:
        """Return the absolute window start.

        Relative schedules require :meth:`window_for` and a subject.
        """
        if self.scheduled_at is None:
            raise ValueError("relative visits require window_for(subject)")
        return self.scheduled_at - self.window_before

    @property
    def window_end(self) -> datetime:
        """Return the absolute window end.

        Relative schedules require :meth:`window_for` and a subject.
        """
        if self.scheduled_at is None:
            raise ValueError("relative visits require window_for(subject)")
        return self.scheduled_at + self.window_after


class QCConfig(BaseModel):
    """Configuration for the baseline longitudinal QC pipeline."""

    model_config = ConfigDict(frozen=True)

    feature_rules: tuple[FeatureRule, ...] = ()
    expected_visits: tuple[ExpectedVisit, ...] = ()
    missingness_warning_fraction: float = Field(default=0.10, ge=0, le=1)
    missingness_error_fraction: float = Field(default=0.25, ge=0, le=1)
    replicate_relative_tolerance: float = Field(default=0.20, ge=0)
    replicate_absolute_floor: float = Field(default=1e-12, gt=0)
    batch_z_threshold: float = Field(default=3.0, gt=0)
    minimum_batch_size: int = Field(default=3, ge=2)
    check_input_order: bool = True
    check_observations_outside_visit_windows: bool = True
    check_batch_confounding: bool = True
    batch_confounding_threshold: float = Field(default=0.80, ge=0, le=1)
    minimum_confounding_group_size: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_missingness_thresholds(self) -> QCConfig:
        """Ensure warning and error missingness thresholds are ordered."""
        if self.missingness_warning_fraction > self.missingness_error_fraction:
            raise ValueError("missingness warning threshold cannot exceed error threshold")
        keys = [(rule.feature, rule.modality) for rule in self.feature_rules]
        if len(keys) != len(set(keys)):
            raise ValueError("feature rules must have unique feature/modality pairs")
        visit_ids = [visit.visit_id for visit in self.expected_visits]
        if len(visit_ids) != len(set(visit_ids)):
            raise ValueError("expected visits must have unique visit_id values")
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
            *self._check_expected_visits(study),
            *self._check_temporal_order(study),
            *self._check_replicates(study),
            *self._check_batch_drift(study),
            *self._check_batch_confounding(study),
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
            "expected_visits": len(self.config.expected_visits),
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

    def _check_expected_visits(self, study: Study) -> list[QCFinding]:
        findings: list[QCFinding] = []
        subject_cohorts = {subject.subject_id: subject.cohort for subject in study.subjects}

        for visit in self.config.expected_visits:
            eligible_subjects = tuple(
                subject
                for subject in study.subjects
                if _visit_applies(visit, subject.subject_id, subject.cohort)
            )
            if not eligible_subjects:
                findings.append(
                    QCFinding(
                        code="expected_visit_has_no_subjects",
                        severity=Severity.WARNING,
                        message=f"Expected visit {visit.visit_id!r} applies to no study subjects",
                        context={"visit_id": visit.visit_id},
                    )
                )
                continue

            targets: dict[str, datetime] = {}
            for subject in eligible_subjects:
                target = _visit_target(visit, subject)
                if target is None:
                    findings.append(
                        QCFinding(
                            code="visit_anchor_missing",
                            severity=Severity.ERROR,
                            message=(
                                f"Subject {subject.subject_id!r} lacks anchor "
                                f"{visit.anchor_id!r} for visit {visit.visit_id!r}"
                            ),
                            subject_ids=(subject.subject_id,),
                            context={
                                "visit_id": visit.visit_id,
                                "anchor_id": visit.anchor_id or "",
                            },
                        )
                    )
                else:
                    targets[subject.subject_id] = target
            matched: dict[str, set[tuple[str, Modality | None]]] = {
                subject_id: set() for subject_id in targets
            }
            for row in study.observations:
                target = targets.get(row.subject_id)
                if target is None or not _inside_visit_window(row, visit, target):
                    continue
                for requirement in visit.required_features:
                    if _matches_visit_feature(row, requirement) and math.isfinite(row.value):
                        matched[row.subject_id].add((requirement.feature, requirement.modality))

            fully_missing = tuple(
                subject_id for subject_id, observed in matched.items() if not observed
            )
            if fully_missing:
                finding = self._visit_missing_finding(
                    code="expected_visit_missing",
                    visit=visit,
                    subject_ids=fully_missing,
                    eligible_subject_count=len(matched),
                    message=(
                        f"Visit {visit.visit_id!r} has no required measurements for "
                        f"{len(fully_missing)}/{len(eligible_subjects)} subjects"
                    ),
                )
                if finding is not None:
                    findings.append(finding)

            for requirement in visit.required_features:
                key = (requirement.feature, requirement.modality)
                partially_missing = tuple(
                    subject_id
                    for subject_id, observed in matched.items()
                    if observed and key not in observed
                )
                if not partially_missing:
                    continue
                finding = self._visit_missing_finding(
                    code="visit_feature_missing",
                    visit=visit,
                    subject_ids=partially_missing,
                    eligible_subject_count=len(matched),
                    message=(
                        f"{requirement.feature} is missing at visit {visit.visit_id!r} for "
                        f"{len(partially_missing)}/{len(eligible_subjects)} subjects"
                    ),
                    feature=requirement.feature,
                    modality=requirement.modality,
                )
                if finding is not None:
                    findings.append(finding)

        if self.config.check_observations_outside_visit_windows:
            findings.extend(self._check_observations_outside_visit_windows(study, subject_cohorts))
        return findings

    def _visit_missing_finding(
        self,
        *,
        code: str,
        visit: ExpectedVisit,
        subject_ids: tuple[str, ...],
        eligible_subject_count: int,
        message: str,
        feature: str | None = None,
        modality: Modality | None = None,
    ) -> QCFinding | None:
        fraction = len(subject_ids) / eligible_subject_count
        if fraction < self.config.missingness_warning_fraction:
            return None
        severity = (
            Severity.ERROR
            if fraction >= self.config.missingness_error_fraction
            else Severity.WARNING
        )
        context: dict[str, str | int | float | bool] = {
            "visit_id": visit.visit_id,
            "missing_fraction": fraction,
            "missing_subjects": len(subject_ids),
            "schedule": _visit_schedule_description(visit),
        }
        if feature is not None:
            context["feature"] = feature
        if modality is not None:
            context["modality"] = modality.value
        return QCFinding(
            code=code,
            severity=severity,
            message=message,
            subject_ids=subject_ids,
            context=context,
        )

    def _check_observations_outside_visit_windows(
        self,
        study: Study,
        subject_cohorts: dict[str, str],
    ) -> list[QCFinding]:
        findings: list[QCFinding] = []
        for index, row in enumerate(study.observations):
            subject = next(item for item in study.subjects if item.subject_id == row.subject_id)
            relevant_visits = [
                (visit, target)
                for visit in self.config.expected_visits
                if _visit_applies(visit, row.subject_id, subject_cohorts[row.subject_id])
                and any(
                    _matches_visit_feature(row, requirement)
                    for requirement in visit.required_features
                )
                and (target := _visit_target(visit, subject)) is not None
            ]
            if not relevant_visits or any(
                _inside_visit_window(row, visit, target) for visit, target in relevant_visits
            ):
                continue
            findings.append(
                QCFinding(
                    code="observation_outside_visit_window",
                    severity=Severity.WARNING,
                    message=(
                        f"{row.feature} at {row.timestamp.isoformat()} is outside all "
                        "applicable expected-visit windows"
                    ),
                    subject_ids=(row.subject_id,),
                    observation_indices=(index,),
                    context={
                        "feature": row.feature,
                        "timestamp": row.timestamp.isoformat(),
                        "candidate_visits": ",".join(
                            visit.visit_id for visit, _ in relevant_visits
                        ),
                    },
                )
            )
        return findings

    def _check_batch_confounding(self, study: Study) -> list[QCFinding]:
        if not self.config.check_batch_confounding:
            return []
        subject_lookup = {subject.subject_id: subject for subject in study.subjects}
        strata: defaultdict[tuple[Modality, str], set[tuple[str, str]]] = defaultdict(set)
        for row in study.observations:
            if row.batch_id is not None:
                strata[(row.modality, row.feature)].add((row.subject_id, row.batch_id))

        interventions = sorted(
            {name for subject in study.subjects for name in subject.interventions}
        )
        findings: list[QCFinding] = []
        for (modality, feature), assignments in strata.items():
            factors: list[tuple[str, dict[str, str]]] = [
                (
                    "cohort",
                    {
                        subject_id: subject_lookup[subject_id].cohort
                        for subject_id, _ in assignments
                    },
                )
            ]
            factors.extend(
                (
                    f"intervention:{intervention}",
                    {
                        subject_id: (
                            "exposed"
                            if intervention in subject_lookup[subject_id].interventions
                            else "unexposed"
                        )
                        for subject_id, _ in assignments
                    },
                )
                for intervention in interventions
            )
            for factor, labels in factors:
                group_sizes = Counter(labels.values())
                batches = {batch_id for _, batch_id in assignments}
                if (
                    len(group_sizes) < 2
                    or len(batches) < 2
                    or min(group_sizes.values()) < self.config.minimum_confounding_group_size
                ):
                    continue
                association = _cramers_v(assignments, labels)
                if association < self.config.batch_confounding_threshold:
                    continue
                findings.append(
                    QCFinding(
                        code="batch_assignment_confounding",
                        severity=Severity.ERROR,
                        message=(
                            f"{factor} is strongly associated with batch for "
                            f"{modality.value}/{feature} (Cramér's V={association:.2f})"
                        ),
                        subject_ids=tuple(sorted(labels)),
                        context={
                            "factor": factor,
                            "feature": feature,
                            "modality": modality.value,
                            "association": association,
                            "batches": len(batches),
                        },
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


def _visit_applies(visit: ExpectedVisit, subject_id: str, cohort: str) -> bool:
    if not visit.subject_ids and not visit.cohorts:
        return True
    return subject_id in visit.subject_ids or cohort in visit.cohorts


def _visit_target(visit: ExpectedVisit, subject: Subject) -> datetime | None:
    return visit.scheduled_for(subject)


def _inside_visit_window(row: Observation, visit: ExpectedVisit, target: datetime) -> bool:
    return target - visit.window_before <= row.timestamp <= target + visit.window_after


def _matches_visit_feature(row: Observation, requirement: VisitFeature) -> bool:
    return row.feature == requirement.feature and (
        requirement.modality is None or row.modality is requirement.modality
    )


def _visit_schedule_description(visit: ExpectedVisit) -> str:
    if visit.scheduled_at is not None:
        return visit.scheduled_at.isoformat()
    return f"{visit.anchor_id}{visit.offset.total_seconds():+g}s"


def _cramers_v(assignments: set[tuple[str, str]], labels: dict[str, str]) -> float:
    groups = sorted(set(labels.values()))
    batches = sorted({batch_id for _, batch_id in assignments})
    counts = {
        (group, batch): sum(
            1
            for subject_id, batch_id in assignments
            if labels[subject_id] == group and batch_id == batch
        )
        for group in groups
        for batch in batches
    }
    total = sum(counts.values())
    row_totals = {group: sum(counts[group, batch] for batch in batches) for group in groups}
    column_totals = {batch: sum(counts[group, batch] for group in groups) for batch in batches}
    chi_square = 0.0
    for group in groups:
        for batch in batches:
            expected = row_totals[group] * column_totals[batch] / total
            if expected > 0:
                chi_square += (counts[group, batch] - expected) ** 2 / expected
    denominator = total * min(len(groups) - 1, len(batches) - 1)
    return math.sqrt(chi_square / denominator) if denominator > 0 else 0.0
