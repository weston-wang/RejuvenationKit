"""One-command Phase 1 study audits and reproducible report bundles."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from rejuvenationkit.detection import (
    ChangeDetectionConfig,
    ChangeDetectionReport,
    MultivariateChangeDetector,
)
from rejuvenationkit.longitudinal import LongitudinalExclusion
from rejuvenationkit.profiling import StudyProfile, StudyProfiler
from rejuvenationkit.qc import BaselineLongitudinalQC, QCConfig, QCReport, Severity
from rejuvenationkit.schemas import Study, study_artifact_hash
from rejuvenationkit.sequential import (
    SequentialDetectionConfig,
    SequentialDetectionReport,
    SequentialTreatmentResponseDetector,
)
from rejuvenationkit.treatment_effect import (
    RandomizedTreatmentEffectEvaluator,
    TreatmentEffectConfig,
    TreatmentEffectReport,
)

_PHASE1_MANAGED_ARTIFACT_NAMES = frozenset(
    {
        "audit.json",
        "audit_overview.png",
        "attrition_bias.csv",
        "change-detection-covariance.png",
        "change-detection-decomposition.png",
        "change-detection-scores.png",
        "change-detection-whitened.png",
        "change_detection_scores.csv",
        "differential_attrition.csv",
        "feature_distributions.csv",
        "findings.csv",
        "longitudinal_exclusions.csv",
        "manifest.json",
        "paired_readiness.csv",
        "sequential-detection-classification.png",
        "sequential-detection-modalities.png",
        "sequential-detection-trajectories.png",
        "sequential_detection_results.csv",
        "sequential_detection_trajectories.csv",
        "summary.md",
        "treatment_effects.csv",
        "treatment_subject_scores.csv",
        "visit_coverage.csv",
        "visit_retention.csv",
    }
)


class ChangeDetectionAuditPlan(BaseModel):
    """Prespecified held-out multivariate analysis included in an audit."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    config: ChangeDetectionConfig
    baseline_visit_id: str = Field(min_length=1)
    follow_up_visit_id: str = Field(min_length=1)
    reference_subject_ids: tuple[str, ...] = Field(min_length=1)
    evaluation_subject_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_distinct_roles(self) -> ChangeDetectionAuditPlan:
        """Reject overlapping visits or calibration/evaluation subjects."""
        if self.baseline_visit_id == self.follow_up_visit_id:
            raise ValueError("change-detection baseline and follow-up must be distinct")
        for name, values in (
            ("reference", self.reference_subject_ids),
            ("evaluation", self.evaluation_subject_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"change-detection {name} subjects must be unique")
        overlap = set(self.reference_subject_ids).intersection(self.evaluation_subject_ids)
        if overlap:
            raise ValueError(
                f"change-detection reference and evaluation subjects overlap: {sorted(overlap)}"
            )
        return self


class TreatmentAuditPlan(BaseModel):
    """Prespecified randomized comparison included in a Phase 1 audit."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    config: TreatmentEffectConfig
    baseline_visit_id: str = Field(min_length=1)
    follow_up_visit_ids: tuple[str, ...] = Field(min_length=1)
    treated_subject_ids: tuple[str, ...] = Field(min_length=1)
    control_subject_ids: tuple[str, ...] = Field(min_length=1)
    treated_label: str = Field(default="treated", min_length=1)
    control_label: str = Field(default="control", min_length=1)

    @model_validator(mode="after")
    def require_unique_visits(self) -> TreatmentAuditPlan:
        """Reject duplicate visits, ambiguous labels, or overlapping subject roles."""
        if len(self.follow_up_visit_ids) != len(set(self.follow_up_visit_ids)):
            raise ValueError("treatment audit follow-up visits must be unique")
        if self.baseline_visit_id in self.follow_up_visit_ids:
            raise ValueError("treatment audit baseline cannot be a follow-up")
        if self.treated_label == self.control_label:
            raise ValueError("treatment audit group labels must be distinct")
        for name, values in (
            ("treated", self.treated_subject_ids),
            ("control", self.control_subject_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"treatment audit {name} subjects must be unique")
        overlap = set(self.treated_subject_ids).intersection(self.control_subject_ids)
        if overlap:
            raise ValueError(f"treatment audit subject groups overlap: {sorted(overlap)}")
        return self


class SequentialDetectionAuditPlan(BaseModel):
    """Prespecified held-out sequential analysis included in an audit."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    config: SequentialDetectionConfig
    visit_ids: tuple[str, ...] = Field(min_length=3)
    reference_subject_ids: tuple[str, ...] = Field(min_length=1)
    evaluation_subject_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_distinct_roles(self) -> SequentialDetectionAuditPlan:
        """Reject duplicate visits, subjects, and calibration/evaluation overlap."""
        if len(self.visit_ids) != len(set(self.visit_ids)):
            raise ValueError("sequential audit visits must be unique")
        for name, values in (
            ("reference", self.reference_subject_ids),
            ("evaluation", self.evaluation_subject_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"sequential audit {name} subjects must be unique")
        overlap = set(self.reference_subject_ids).intersection(self.evaluation_subject_ids)
        if overlap:
            raise ValueError(
                f"sequential reference and evaluation subjects overlap: {sorted(overlap)}"
            )
        return self


class Phase1AuditConfig(BaseModel):
    """Configuration for the complete Phase 1 audit workflow."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    qc: QCConfig
    outlier_iqr_multiplier: float = Field(default=1.5, ge=0)
    include_visualizations: bool = True
    allow_analysis_with_qc_errors: bool = False


class LongitudinalExclusionCount(BaseModel):
    """Count of structured exclusions from one longitudinal analysis."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    analysis: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    events: int = Field(ge=1)


class Phase1AuditReport(BaseModel):
    """Machine-readable Phase 1 audit and its generated artifact manifest."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    schema_version: str = "3"
    software_version: str
    study_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    study_metadata: Mapping[str, str | int | float | bool]
    config: Phase1AuditConfig
    change_detection_plan: ChangeDetectionAuditPlan | None = None
    sequential_detection_plan: SequentialDetectionAuditPlan | None = None
    treatment_plan: TreatmentAuditPlan | None = None
    qc: QCReport
    profile: StudyProfile
    change_detection: ChangeDetectionReport | None = None
    sequential_detection: SequentialDetectionReport | None = None
    treatment_effect: TreatmentEffectReport | None = None
    analysis_blocked_by_qc: bool = False
    analysis_override_applied: bool = False
    longitudinal_exclusion_counts: tuple[LongitudinalExclusionCount, ...] = ()
    artifacts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_analysis_disposition(self) -> Phase1AuditReport:
        """Reconstruct the exact QC gate, identities, results, and artifact disposition."""
        metadata = dict(self.study_metadata)
        if any(
            not key or key != key.strip() or (isinstance(value, float) and not math.isfinite(value))
            for key, value in metadata.items()
        ):
            raise ValueError("Phase 1 study metadata must have valid keys and finite values")
        object.__setattr__(self, "study_metadata", MappingProxyType(metadata))
        if self.schema_version != "3":
            raise ValueError("unsupported Phase 1 audit schema version")
        if self.qc.study_id != self.study_id or self.profile.study_id != self.study_id:
            raise ValueError("Phase 1 QC and profile study identities must match the report")

        plans = (
            self.change_detection_plan,
            self.sequential_detection_plan,
            self.treatment_plan,
        )
        results = (
            self.change_detection,
            self.sequential_detection,
            self.treatment_effect,
        )
        analysis_requested = any(item is not None for item in plans)
        expected_blocked = (
            analysis_requested
            and not self.qc.passed
            and not self.config.allow_analysis_with_qc_errors
        )
        expected_override = (
            analysis_requested and not self.qc.passed and self.config.allow_analysis_with_qc_errors
        )
        if self.analysis_blocked_by_qc != expected_blocked:
            raise ValueError("analysis_blocked_by_qc does not match the serialized QC gate")
        if self.analysis_override_applied != expected_override:
            raise ValueError("analysis_override_applied does not match the serialized QC gate")
        for plan, result in zip(plans, results, strict=True):
            expected_result = plan is not None and not expected_blocked
            if (result is not None) != expected_result:
                raise ValueError("Phase 1 plan/result disposition does not match the QC gate")

        if self.change_detection is not None and self.change_detection_plan is not None:
            _validate_change_detection_result(
                report_study_id=self.study_id,
                plan=self.change_detection_plan,
                result=self.change_detection,
            )
        if self.sequential_detection is not None and self.sequential_detection_plan is not None:
            _validate_sequential_detection_result(
                report_study_id=self.study_id,
                plan=self.sequential_detection_plan,
                result=self.sequential_detection,
            )
        if self.treatment_effect is not None and self.treatment_plan is not None:
            _validate_treatment_result(
                report_study_id=self.study_id,
                plan=self.treatment_plan,
                result=self.treatment_effect,
            )

        expected_exclusions = _longitudinal_exclusion_counts(
            profile=self.profile,
            change_detection=self.change_detection,
            sequential_detection=self.sequential_detection,
            treatment_effect=self.treatment_effect,
        )
        if self.longitudinal_exclusion_counts != expected_exclusions:
            raise ValueError("longitudinal exclusion counts do not match analysis results")
        expected_artifacts = _expected_phase1_artifacts(
            config=self.config,
            change_detection=self.change_detection,
            sequential_detection=self.sequential_detection,
            treatment_effect=self.treatment_effect,
        )
        if self.artifacts != expected_artifacts:
            raise ValueError("Phase 1 artifact names do not match the serialized results")
        return self

    @field_serializer("study_metadata")
    def serialize_study_metadata(
        self,
        value: Mapping[str, str | int | float | bool],
    ) -> dict[str, str | int | float | bool]:
        """Serialize immutable metadata as ordinary JSON data."""
        return dict(value)

    @property
    def passed(self) -> bool:
        """Return whether the underlying quality-control report passed."""
        return self.qc.passed

    def summary_markdown(self) -> str:
        """Render a compact decision-oriented audit summary."""
        counts = self.qc.counts
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"# Phase 1 audit: {self.study_id}",
            "",
            f"**QC status:** {status}",
            "",
            "## Study inventory",
            "",
            f"- Subjects: {int(self.qc.metrics.get('subjects', 0)):,}",
            f"- Observations: {int(self.qc.metrics.get('observations', 0)):,}",
            f"- Features: {int(self.qc.metrics.get('features', 0)):,}",
            f"- Expected visits: {int(self.qc.metrics.get('expected_visits', 0)):,}",
            f"- RejuvenationKit version: {self.software_version}",
            "",
            "## Findings",
            "",
            f"- Errors: {counts[Severity.ERROR]}",
            f"- Warnings: {counts[Severity.WARNING]}",
            f"- Informational: {counts[Severity.INFO]}",
        ]
        if self.analysis_blocked_by_qc:
            lines.extend(
                (
                    "",
                    "## Analysis gate",
                    "",
                    "- Prespecified detection or inference was not run because QC contains errors.",
                    "- Resolve the errors, or explicitly enable the auditable QC override.",
                )
            )
        elif self.analysis_override_applied:
            lines.extend(
                (
                    "",
                    "## Analysis gate",
                    "",
                    "- **QC override applied:** analyses ran despite one or more QC errors.",
                )
            )
        coverage = [item for item in self.profile.visit_coverage if item.cohort == "all"]
        if coverage:
            lowest = min(coverage, key=lambda item: item.coverage_fraction)
            lines.extend(
                (
                    "",
                    "## Analysis readiness",
                    "",
                    (
                        f"- Lowest visit-feature coverage: {lowest.coverage_fraction:.1%} "
                        f"at {lowest.visit_id} / {lowest.feature}"
                    ),
                )
            )
            retention = [item for item in self.profile.visit_retention if item.cohort == "all"]
            if retention:
                weakest = min(retention, key=lambda item: item.retention_fraction)
                lines.append(
                    f"- Lowest complete-case retention: {weakest.retention_fraction:.1%} "
                    f"from {weakest.from_visit_id} to {weakest.to_visit_id}"
                )
            outliers = sum(
                item.outlier_count
                for item in self.profile.feature_distributions
                if item.cohort == "all"
            )
            lines.append(f"- Robust outlier flags: {outliers}")
        if self.treatment_effect is not None:
            lines.extend(("", "## Randomized treatment effects", ""))
            for visit in self.treatment_effect.visit_effects:
                lines.append(
                    f"- {visit.follow_up_visit_id}: permutation "
                    f"p={visit.permutation_p_value:.4f}; "
                    f"{visit.treated_subjects} treated and "
                    f"{visit.control_subjects} control subjects"
                )
        else:
            lines.extend(
                (
                    "",
                    "## Treatment inference",
                    "",
                    "- Not run. This audit makes no treatment-effect claim.",
                )
            )
        if self.change_detection is not None:
            lines.extend(
                (
                    "",
                    "## Held-out multivariate change detection",
                    "",
                    (f"- Calibration subjects: {self.change_detection.model.reference_subjects}"),
                    f"- Evaluation subjects scored: {len(self.change_detection.results)}",
                    (
                        f"- Detected trajectories: "
                        f"{sum(item.detected for item in self.change_detection.results)}"
                    ),
                    (
                        f"- Incomplete evaluation subjects: "
                        f"{len(self.change_detection.excluded_subject_ids)}"
                    ),
                )
            )
        if self.sequential_detection is not None:
            lines.extend(
                (
                    "",
                    "## Held-out sequential change detection",
                    "",
                    f"- Calibration subjects: {self.sequential_detection.model.reference_subjects}",
                    f"- Evaluation subjects scored: {len(self.sequential_detection.results)}",
                    (
                        "- Detected trajectories: "
                        f"{sum(item.detected for item in self.sequential_detection.results)}"
                    ),
                    (
                        "- Persistent trajectories: "
                        f"{sum(item.persistent for item in self.sequential_detection.results)}"
                    ),
                    (
                        "- Incomplete evaluation subjects: "
                        f"{len(self.sequential_detection.excluded_subject_ids)}"
                    ),
                )
            )
        if self.longitudinal_exclusion_counts:
            lines.extend(("", "## Longitudinal exclusions", ""))
            lines.extend(
                f"- {item.analysis} / {item.reason}: {item.events} event(s)"
                for item in self.longitudinal_exclusion_counts
            )
        lines.extend(
            (
                "",
                "## Interpretation",
                "",
                "A passing audit means the configured checks found no errors. "
                "It does not establish efficacy, causal validity, or regulatory suitability.",
                "",
            )
        )
        return "\n".join(lines)


def _validate_partition(
    *,
    requested: tuple[str, ...],
    scored: tuple[str, ...],
    excluded: tuple[str, ...],
    analysis: str,
) -> None:
    """Require every requested subject to be scored or explicitly excluded once."""
    if tuple(sorted(set(scored))) != scored:
        raise ValueError(f"{analysis} scored subject identifiers must be unique and sorted")
    if tuple(sorted(set(excluded))) != excluded:
        raise ValueError(f"{analysis} excluded subject identifiers must be unique and sorted")
    if set(scored).intersection(excluded):
        raise ValueError(f"{analysis} subjects cannot be both scored and excluded")
    if set(scored).union(excluded) != set(requested):
        raise ValueError(f"{analysis} scored/excluded subjects do not match the audit plan")


def _validate_change_detection_result(
    *,
    report_study_id: str,
    plan: ChangeDetectionAuditPlan,
    result: ChangeDetectionReport,
) -> None:
    """Bind a serialized pairwise result to its plan, model, and subject partition."""
    model = result.model
    if result.study_id != report_study_id or model.study_id != report_study_id:
        raise ValueError("change-detection study identity does not match the audit")
    if (
        result.baseline_visit_id != plan.baseline_visit_id
        or result.follow_up_visit_id != plan.follow_up_visit_id
        or model.baseline_visit is None
        or model.follow_up_visit is None
        or model.baseline_visit.visit_id != plan.baseline_visit_id
        or model.follow_up_visit.visit_id != plan.follow_up_visit_id
    ):
        raise ValueError("change-detection visits do not match the audit plan")
    if model.config != plan.config:
        raise ValueError("change-detection configuration does not match the audit plan")
    if model.requested_reference_subject_ids != tuple(sorted(plan.reference_subject_ids)):
        raise ValueError("change-detection reference subjects do not match the audit plan")
    scored_ids = tuple(item.subject_id for item in result.results)
    _validate_partition(
        requested=plan.evaluation_subject_ids,
        scored=scored_ids,
        excluded=result.excluded_subject_ids,
        analysis="change-detection",
    )
    mean = np.asarray(model.mean_change, dtype=np.float64)
    cholesky = np.linalg.cholesky(np.asarray(model.covariance, dtype=np.float64))
    reference_scores = np.asarray(model.reference_score_distribution, dtype=np.float64)
    for item in result.results:
        change = np.asarray(item.change, dtype=np.float64)
        innovation = np.asarray(item.innovation, dtype=np.float64)
        whitened = np.asarray(item.whitened_innovation, dtype=np.float64)
        if any(vector.shape != mean.shape for vector in (change, innovation, whitened)):
            raise ValueError("change-detection result vectors do not match model dimension")
        if not all(np.isfinite(vector).all() for vector in (change, innovation, whitened)):
            raise ValueError("change-detection result vectors must be finite")
        expected_innovation = change - mean
        expected_whitened = np.linalg.solve(cholesky, expected_innovation)
        expected_score = max(float(expected_whitened @ expected_whitened), 0.0)
        expected_tail = float(
            (1 + np.count_nonzero(reference_scores >= expected_score)) / (len(reference_scores) + 1)
        )
        if not np.allclose(innovation, expected_innovation, rtol=1e-12, atol=1e-12):
            raise ValueError("change-detection innovation does not match the reported change")
        if not np.allclose(whitened, expected_whitened, rtol=1e-12, atol=1e-12):
            raise ValueError("change-detection whitened innovation does not match the model")
        if not math.isclose(
            item.squared_mahalanobis_distance,
            expected_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("change-detection score does not match the model")
        if not math.isclose(
            item.empirical_tail_probability,
            expected_tail,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("change-detection tail probability does not match the model")
        if item.detected != (expected_score > model.threshold):
            raise ValueError("change-detection flag does not match the model threshold")


def _validate_sequential_detection_result(
    *,
    report_study_id: str,
    plan: SequentialDetectionAuditPlan,
    result: SequentialDetectionReport,
) -> None:
    """Bind a serialized sequential result to its plan and scored population."""
    model = result.model
    if result.study_id != report_study_id or model.study_id != report_study_id:
        raise ValueError("sequential-detection study identity does not match the audit")
    if (
        result.visit_ids != plan.visit_ids
        or tuple(visit.visit_id for visit in model.ordered_visits) != plan.visit_ids
    ):
        raise ValueError("sequential-detection visits do not match the audit plan")
    if model.config != plan.config:
        raise ValueError("sequential-detection configuration does not match the audit plan")
    if model.requested_reference_subject_ids != tuple(sorted(plan.reference_subject_ids)):
        raise ValueError("sequential-detection reference subjects do not match the audit plan")
    _validate_partition(
        requested=plan.evaluation_subject_ids,
        scored=tuple(item.subject_id for item in result.results),
        excluded=result.excluded_subject_ids,
        analysis="sequential-detection",
    )
    visit_ids = set(plan.visit_ids)
    visit_positions = {visit_id: index for index, visit_id in enumerate(plan.visit_ids)}
    threshold = model.maximum_cumulative_score_threshold
    reference_scores = np.asarray(model.reference_maximum_score_distribution, dtype=np.float64)
    for item in result.results:
        if not item.points:
            raise ValueError("scored sequential trajectories require at least one transition")
        if any(
            point.from_visit_id not in visit_ids or point.to_visit_id not in visit_ids
            for point in item.points
        ):
            raise ValueError("sequential result contains a visit outside the audit plan")
        trajectory_visit_ids = (
            item.points[0].from_visit_id,
            *(point.to_visit_id for point in item.points),
        )
        if any(
            first.to_visit_id != second.from_visit_id for first, second in pairwise(item.points)
        ) or any(
            visit_positions[first] >= visit_positions[second]
            for first, second in pairwise(trajectory_visit_ids)
        ):
            raise ValueError("sequential transitions must form one ordered trajectory")
        expected_missing = tuple(
            visit_id for visit_id in plan.visit_ids if visit_id not in trajectory_visit_ids
        )
        if item.missing_visit_ids != expected_missing:
            raise ValueError("sequential missing visits do not match the reported trajectory")
        for point_index, point in enumerate(item.points):
            if not point.selected_observation_indices:
                raise ValueError("sequential transitions require source-row provenance")
            if point.from_observed_at is None or point.to_observed_at is None:
                raise ValueError("sequential transitions require observed timestamps")
            expected_elapsed = (point.to_observed_at - point.from_observed_at).total_seconds() / (
                365.2425 * 24 * 60 * 60
            )
            if not math.isclose(
                point.elapsed_years,
                expected_elapsed,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("sequential elapsed time does not match observed timestamps")
            if point_index and (
                item.points[point_index - 1].to_observed_at != point.from_observed_at
            ):
                raise ValueError("sequential observed timestamps do not form one trajectory")
            if point.threshold_crossed != (point.cumulative_score > threshold):
                raise ValueError("sequential crossing flag does not match the model threshold")
            expected_tail = float(
                (1 + np.count_nonzero(reference_scores >= point.cumulative_score))
                / (len(reference_scores) + 1)
            )
            if not math.isclose(
                point.empirical_tail_probability,
                expected_tail,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("sequential tail probability does not match the model")
        expected_onset = next(
            (point.to_visit_id for point in item.points if point.threshold_crossed),
            None,
        )
        if item.detected != (expected_onset is not None) or item.onset_visit_id != expected_onset:
            raise ValueError("sequential detection/onset does not match its trajectory")
        crossing_flags = tuple(point.threshold_crossed for point in item.points)
        final_crossings = 0
        for crossed in reversed(crossing_flags):
            if not crossed:
                break
            final_crossings += 1
        expected_persistent = final_crossings >= model.config.persistence_crossings
        expected_transient = (
            expected_onset is not None
            and not expected_persistent
            and any(
                crossed and not later
                for index, crossed in enumerate(crossing_flags)
                for later in crossing_flags[index + 1 :]
            )
        )
        if item.persistent != expected_persistent or item.transient != expected_transient:
            raise ValueError("sequential trajectory classification is internally inconsistent")
        expected_peak = max((point.cumulative_score for point in item.points), default=0.0)
        if not math.isclose(
            item.peak_cumulative_score,
            expected_peak,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("sequential peak score does not match its trajectory")
        expected_modality_channels = Counter(feature.modality for feature in model.config.features)
        observed_modality_channels = {
            evidence.modality: evidence.channels for evidence in item.peak_modality_evidence
        }
        if len(observed_modality_channels) != len(
            item.peak_modality_evidence
        ) or observed_modality_channels != dict(expected_modality_channels):
            raise ValueError("sequential modality evidence does not match configured channels")


def _validate_treatment_result(
    *,
    report_study_id: str,
    plan: TreatmentAuditPlan,
    result: TreatmentEffectReport,
) -> None:
    """Bind randomized inference results to the exact prespecified audit plan."""
    provenance = result.inference_provenance
    if result.study_id != report_study_id or provenance is None:
        raise ValueError("treatment result requires matching randomized-inference provenance")
    if provenance.study_id != report_study_id or result.baseline_visit_id != plan.baseline_visit_id:
        raise ValueError("treatment study or baseline identity does not match the audit plan")
    if provenance.config != plan.config:
        raise ValueError("treatment configuration does not match the audit plan")
    if (
        provenance.baseline_visit.visit_id != plan.baseline_visit_id
        or tuple(visit.visit_id for visit in provenance.follow_up_visits)
        != plan.follow_up_visit_ids
    ):
        raise ValueError("treatment visits do not match the audit plan")
    if (
        provenance.treated_subject_ids != tuple(sorted(plan.treated_subject_ids))
        or provenance.control_subject_ids != tuple(sorted(plan.control_subject_ids))
        or result.treated_label != plan.treated_label
        or result.control_label != plan.control_label
    ):
        raise ValueError("treatment groups or labels do not match the audit plan")
    if tuple(item.follow_up_visit_id for item in result.visit_effects) != plan.follow_up_visit_ids:
        raise ValueError("treatment-effect visits do not match the audit plan")
    expected_channels = tuple((item.feature, item.modality) for item in plan.config.features)
    for visit in result.visit_effects:
        observed_channels = tuple((item.feature, item.modality) for item in visit.effects)
        if observed_channels != expected_channels:
            raise ValueError("treatment feature effects do not match the configured channels")
        scores = tuple(
            score
            for score in result.subject_scores
            if score.follow_up_visit_id == visit.follow_up_visit_id
        )
        treated_count = sum(score.group == plan.treated_label for score in scores)
        control_count = sum(score.group == plan.control_label for score in scores)
        if (treated_count, control_count) != (visit.treated_subjects, visit.control_subjects):
            raise ValueError("treatment-effect subject counts do not match subject scores")
        for effect in visit.effects:
            values = (
                effect.treated_mean_change,
                effect.control_mean_change,
                effect.difference_in_differences,
                effect.confidence_interval_low,
                effect.confidence_interval_high,
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("treatment feature effects must be finite")
            if not math.isclose(
                effect.difference_in_differences,
                effect.treated_mean_change - effect.control_mean_change,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("treatment difference-in-differences is internally inconsistent")
            if effect.confidence_interval_low > effect.confidence_interval_high:
                raise ValueError("treatment confidence interval bounds are inverted")


def _expected_phase1_artifacts(
    *,
    config: Phase1AuditConfig,
    change_detection: ChangeDetectionReport | None,
    sequential_detection: SequentialDetectionReport | None,
    treatment_effect: TreatmentEffectReport | None,
) -> tuple[str, ...]:
    """Return the exact deterministic artifact inventory for one audit disposition."""
    names = [
        "audit.json",
        "summary.md",
        "findings.csv",
        "visit_coverage.csv",
        "visit_retention.csv",
        "paired_readiness.csv",
        "feature_distributions.csv",
        "attrition_bias.csv",
        "differential_attrition.csv",
        "longitudinal_exclusions.csv",
        "manifest.json",
    ]
    if change_detection is not None:
        names.append("change_detection_scores.csv")
        if config.include_visualizations:
            names.extend(
                (
                    "change-detection-covariance.png",
                    "change-detection-scores.png",
                    "change-detection-whitened.png",
                    "change-detection-decomposition.png",
                )
            )
    if sequential_detection is not None:
        names.extend(
            (
                "sequential_detection_results.csv",
                "sequential_detection_trajectories.csv",
            )
        )
        if config.include_visualizations:
            names.extend(
                (
                    "sequential-detection-trajectories.png",
                    "sequential-detection-classification.png",
                    "sequential-detection-modalities.png",
                )
            )
    if treatment_effect is not None:
        names.extend(("treatment_effects.csv", "treatment_subject_scores.csv"))
    if config.include_visualizations:
        names.append("audit_overview.png")
    return tuple(names)


class _PyplotModule(Protocol):
    def subplots(
        self,
        *args: object,
        **kwargs: object,
    ) -> tuple[Any, Any]:
        """Create a figure and axes."""

    def close(self, figure: Any) -> None:
        """Close a figure."""


class Phase1AuditRunner:
    """Run QC, readiness profiling, optional inference, and artifact export."""

    def __init__(self, config: Phase1AuditConfig) -> None:
        """Create an audit runner."""
        self.config = config

    def run(
        self,
        study: Study,
        *,
        output_dir: Path,
        change_detection_plan: ChangeDetectionAuditPlan | None = None,
        sequential_detection_plan: SequentialDetectionAuditPlan | None = None,
        treatment_plan: TreatmentAuditPlan | None = None,
    ) -> Phase1AuditReport:
        """Run the configured audit and write a reproducible report bundle."""
        qc_report = BaselineLongitudinalQC(self.config.qc).run(study)
        profile = StudyProfiler(
            self.config.qc,
            outlier_iqr_multiplier=self.config.outlier_iqr_multiplier,
        ).profile(study)
        analysis_requested = any(
            item is not None
            for item in (
                change_detection_plan,
                sequential_detection_plan,
                treatment_plan,
            )
        )
        analysis_blocked = (
            analysis_requested
            and not qc_report.passed
            and not (self.config.allow_analysis_with_qc_errors)
        )
        override_applied = (
            analysis_requested
            and not qc_report.passed
            and (self.config.allow_analysis_with_qc_errors)
        )
        change_detection = (
            self._change_detection(study, change_detection_plan)
            if change_detection_plan is not None and not analysis_blocked
            else None
        )
        sequential_detection = (
            self._sequential_detection(study, sequential_detection_plan)
            if sequential_detection_plan is not None and not analysis_blocked
            else None
        )
        treatment = (
            self._treatment_effect(study, treatment_plan)
            if treatment_plan is not None and not analysis_blocked
            else None
        )
        exclusion_counts = _longitudinal_exclusion_counts(
            profile=profile,
            change_detection=change_detection,
            sequential_detection=sequential_detection,
            treatment_effect=treatment,
        )
        artifact_names = _expected_phase1_artifacts(
            config=self.config,
            change_detection=change_detection,
            sequential_detection=sequential_detection,
            treatment_effect=treatment,
        )
        report = Phase1AuditReport(
            software_version=_software_version(),
            study_id=study.study_id,
            input_sha256=_study_fingerprint(study),
            study_metadata=dict(study.metadata),
            config=self.config,
            change_detection_plan=change_detection_plan,
            sequential_detection_plan=sequential_detection_plan,
            treatment_plan=treatment_plan,
            qc=qc_report,
            profile=profile,
            change_detection=change_detection,
            sequential_detection=sequential_detection,
            treatment_effect=treatment,
            analysis_blocked_by_qc=analysis_blocked,
            analysis_override_applied=override_applied,
            longitudinal_exclusion_counts=exclusion_counts,
            artifacts=artifact_names,
        )
        self._publish_bundle(report, output_dir)
        return report

    def _change_detection(
        self,
        study: Study,
        plan: ChangeDetectionAuditPlan,
    ) -> ChangeDetectionReport:
        visits = {visit.visit_id: visit for visit in self.config.qc.expected_visits}
        requested = {plan.baseline_visit_id, plan.follow_up_visit_id}
        missing = requested.difference(visits)
        if missing:
            raise ValueError(f"change-detection audit visits are not configured: {sorted(missing)}")
        detector = MultivariateChangeDetector(plan.config).fit(
            study,
            baseline=visits[plan.baseline_visit_id],
            follow_up=visits[plan.follow_up_visit_id],
            reference_subject_ids=plan.reference_subject_ids,
        )
        return detector.score(
            study,
            baseline=visits[plan.baseline_visit_id],
            follow_up=visits[plan.follow_up_visit_id],
            subject_ids=plan.evaluation_subject_ids,
        )

    def _treatment_effect(
        self,
        study: Study,
        plan: TreatmentAuditPlan,
    ) -> TreatmentEffectReport:
        visits = {visit.visit_id: visit for visit in self.config.qc.expected_visits}
        requested = {plan.baseline_visit_id, *plan.follow_up_visit_ids}
        missing = requested.difference(visits)
        if missing:
            raise ValueError(f"treatment audit visits are not configured: {sorted(missing)}")
        return RandomizedTreatmentEffectEvaluator(plan.config).evaluate(
            study,
            baseline=visits[plan.baseline_visit_id],
            follow_ups=tuple(visits[item] for item in plan.follow_up_visit_ids),
            treated_subject_ids=plan.treated_subject_ids,
            control_subject_ids=plan.control_subject_ids,
            treated_label=plan.treated_label,
            control_label=plan.control_label,
        )

    def _sequential_detection(
        self,
        study: Study,
        plan: SequentialDetectionAuditPlan,
    ) -> SequentialDetectionReport:
        visits = {visit.visit_id: visit for visit in self.config.qc.expected_visits}
        missing = set(plan.visit_ids).difference(visits)
        if missing:
            raise ValueError(f"sequential audit visits are not configured: {sorted(missing)}")
        selected_visits = tuple(visits[visit_id] for visit_id in plan.visit_ids)
        detector = SequentialTreatmentResponseDetector(plan.config).fit(
            study,
            visits=selected_visits,
            reference_subject_ids=plan.reference_subject_ids,
        )
        return detector.score(
            study,
            visits=selected_visits,
            subject_ids=plan.evaluation_subject_ids,
        )

    def _publish_bundle(self, report: Phase1AuditReport, output_dir: Path) -> None:
        """Stage a complete bundle before publishing manifest-managed files."""
        if output_dir.is_symlink():
            raise ValueError(f"audit output directory cannot be a symlink: {output_dir}")
        output_parent = output_dir.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"audit output path is not a directory: {output_dir}")
        if output_dir.is_dir():
            unsafe_links = tuple(
                sorted(
                    path.name
                    for path in output_dir.iterdir()
                    if path.name in _PHASE1_MANAGED_ARTIFACT_NAMES and path.is_symlink()
                )
            )
            if unsafe_links:
                raise ValueError(
                    f"audit output contains symlinked managed artifacts: {list(unsafe_links)}"
                )
        with TemporaryDirectory(prefix=f".{output_dir.name}-staging-", dir=output_parent) as raw:
            staging = Path(raw)
            self._write_bundle_files(report, staging)
            expected = set(report.artifacts)
            actual = {path.name for path in staging.iterdir() if path.is_file()}
            if actual != expected:
                raise RuntimeError(
                    "staged audit artifact mismatch: "
                    f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
                )
            previous_managed = _managed_artifacts(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in sorted(expected.difference({"manifest.json"})):
                (staging / name).replace(output_dir / name)
            for name in sorted(previous_managed.difference(expected)):
                stale = output_dir / name
                if stale.is_file():
                    stale.unlink()
            (staging / "manifest.json").replace(output_dir / "manifest.json")

    def _write_bundle_files(self, report: Phase1AuditReport, output_dir: Path) -> None:
        """Write a complete report bundle into an empty staging directory."""
        (output_dir / "summary.md").write_text(
            report.summary_markdown(),
            encoding="utf-8",
        )
        _findings_frame(report.qc).to_csv(output_dir / "findings.csv", index=False)
        frames = {
            "visit_coverage.csv": report.profile.coverage_frame(),
            "visit_retention.csv": report.profile.retention_frame(),
            "paired_readiness.csv": report.profile.paired_readiness_frame(),
            "feature_distributions.csv": report.profile.distributions_frame(),
            "attrition_bias.csv": report.profile.attrition_bias_frame(),
            "differential_attrition.csv": report.profile.differential_attrition_frame(),
        }
        for name, frame in frames.items():
            frame.to_csv(output_dir / name, index=False)
        if report.change_detection is not None:
            _change_detection_results_frame(report.change_detection).to_csv(
                output_dir / "change_detection_scores.csv",
                index=False,
            )
        if report.sequential_detection is not None:
            _sequential_results_frame(report.sequential_detection).to_csv(
                output_dir / "sequential_detection_results.csv",
                index=False,
            )
            _sequential_trajectory_frame(report.sequential_detection).to_csv(
                output_dir / "sequential_detection_trajectories.csv",
                index=False,
            )
        if report.treatment_effect is not None:
            report.treatment_effect.effects_frame().to_csv(
                output_dir / "treatment_effects.csv",
                index=False,
            )
            report.treatment_effect.scores_frame().to_csv(
                output_dir / "treatment_subject_scores.csv",
                index=False,
            )
        _longitudinal_exclusions_frame(report).to_csv(
            output_dir / "longitudinal_exclusions.csv",
            index=False,
        )
        if self.config.include_visualizations:
            _save_audit_overview(report, output_dir / "audit_overview.png")
            if report.change_detection is not None:
                from rejuvenationkit.visualization import save_detection_figures

                save_detection_figures(
                    report.change_detection,
                    output_dir,
                    prefix="change-detection",
                )
            if report.sequential_detection is not None:
                from rejuvenationkit.visualization import save_sequential_figures

                save_sequential_figures(
                    report.sequential_detection,
                    output_dir,
                    prefix="sequential-detection",
                )
        (output_dir / "audit.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        _write_manifest(report, output_dir)


def run_phase1_audit(
    study: Study,
    *,
    config: Phase1AuditConfig,
    output_dir: Path,
    change_detection_plan: ChangeDetectionAuditPlan | None = None,
    sequential_detection_plan: SequentialDetectionAuditPlan | None = None,
    treatment_plan: TreatmentAuditPlan | None = None,
) -> Phase1AuditReport:
    """Run a complete Phase 1 audit in one function call."""
    return Phase1AuditRunner(config).run(
        study,
        output_dir=output_dir,
        change_detection_plan=change_detection_plan,
        sequential_detection_plan=sequential_detection_plan,
        treatment_plan=treatment_plan,
    )


def _findings_frame(report: QCReport) -> pd.DataFrame:
    columns = (
        "code",
        "severity",
        "message",
        "subject_ids",
        "observation_indices",
        "context",
    )
    rows = [
        {
            "code": item.code,
            "severity": item.severity.value,
            "message": item.message,
            "subject_ids": ";".join(item.subject_ids),
            "observation_indices": ";".join(str(value) for value in item.observation_indices),
            "context": json.dumps(dict(item.context), sort_keys=True),
        }
        for item in report.findings
    ]
    return pd.DataFrame(rows, columns=columns)


def _longitudinal_exclusion_sources(
    *,
    profile: StudyProfile,
    change_detection: ChangeDetectionReport | None,
    sequential_detection: SequentialDetectionReport | None,
    treatment_effect: TreatmentEffectReport | None,
) -> tuple[tuple[str, tuple[LongitudinalExclusion, ...]], ...]:
    sources: list[tuple[str, tuple[LongitudinalExclusion, ...]]] = [
        ("profiling", tuple(dict.fromkeys(profile.longitudinal_exclusions)))
    ]
    for name, result in (
        ("change_detection", change_detection),
        ("sequential_detection", sequential_detection),
        ("treatment_effect", treatment_effect),
    ):
        if result is not None:
            sources.append((name, tuple(dict.fromkeys(result.exclusions))))
    return tuple(sources)


def _longitudinal_exclusion_counts(
    *,
    profile: StudyProfile,
    change_detection: ChangeDetectionReport | None,
    sequential_detection: SequentialDetectionReport | None,
    treatment_effect: TreatmentEffectReport | None,
) -> tuple[LongitudinalExclusionCount, ...]:
    counts = Counter(
        (analysis, exclusion.reason.value)
        for analysis, exclusions in _longitudinal_exclusion_sources(
            profile=profile,
            change_detection=change_detection,
            sequential_detection=sequential_detection,
            treatment_effect=treatment_effect,
        )
        for exclusion in exclusions
    )
    return tuple(
        LongitudinalExclusionCount(analysis=analysis, reason=reason, events=events)
        for (analysis, reason), events in sorted(counts.items())
    )


def _longitudinal_exclusions_frame(report: Phase1AuditReport) -> pd.DataFrame:
    columns = (
        "analysis",
        "reason",
        "subject_id",
        "visit_id",
        "channel_index",
        "feature",
        "requested_modality",
        "observation_indices",
        "observed_modalities",
        "observed_units",
        "missing_channel_indices",
        "missing_visit_ids",
    )
    rows = []
    for analysis, exclusions in _longitudinal_exclusion_sources(
        profile=report.profile,
        change_detection=report.change_detection,
        sequential_detection=report.sequential_detection,
        treatment_effect=report.treatment_effect,
    ):
        for item in exclusions:
            rows.append(
                {
                    "analysis": analysis,
                    "reason": item.reason.value,
                    "subject_id": item.subject_id,
                    "visit_id": item.visit_id,
                    "channel_index": item.channel_index,
                    "feature": item.feature,
                    "requested_modality": (
                        item.requested_modality.value
                        if item.requested_modality is not None
                        else None
                    ),
                    "observation_indices": ";".join(
                        str(index) for index in item.observation_indices
                    ),
                    "observed_modalities": ";".join(
                        modality.value for modality in item.observed_modalities
                    ),
                    "observed_units": ";".join(item.observed_units),
                    "missing_channel_indices": ";".join(
                        str(index) for index in item.missing_channel_indices
                    ),
                    "missing_visit_ids": ";".join(item.missing_visit_ids),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _change_detection_results_frame(report: ChangeDetectionReport) -> pd.DataFrame:
    frame = report.results_frame()
    if not frame.empty:
        return frame
    return pd.DataFrame(
        columns=(
            "subject_id",
            "change",
            "innovation",
            "whitened_innovation",
            "squared_mahalanobis_distance",
            "empirical_tail_probability",
            "detected",
        )
    )


def _sequential_results_frame(report: SequentialDetectionReport) -> pd.DataFrame:
    frame = report.results_frame()
    if not frame.empty:
        return frame
    return pd.DataFrame(
        columns=(
            "subject_id",
            "transitions",
            "detected",
            "onset_visit_id",
            "persistent",
            "transient",
            "peak_cumulative_score",
            "missing_visit_ids",
        )
    )


def _sequential_trajectory_frame(report: SequentialDetectionReport) -> pd.DataFrame:
    frame = report.trajectory_frame()
    if not frame.empty:
        return frame
    return pd.DataFrame(
        columns=(
            "subject_id",
            "from_visit_id",
            "to_visit_id",
            "from_observed_at",
            "to_observed_at",
            "selected_observation_indices",
            "elapsed_years",
            "interval_score",
            "cumulative_score",
            "empirical_tail_probability",
            "threshold_crossed",
        )
    )


def _study_fingerprint(study: Study) -> str:
    return study_artifact_hash(study)


def _software_version() -> str:
    try:
        return version("rejuvenationkit")
    except PackageNotFoundError:
        return "0+uninstalled"


def _write_manifest(report: Phase1AuditReport, output_dir: Path) -> None:
    files = sorted(
        name
        for name in report.artifacts
        if name != "manifest.json" and (output_dir / name).is_file()
    )
    manifest = {
        "schema_version": report.schema_version,
        "software_version": report.software_version,
        "study_id": report.study_id,
        "input_sha256": report.input_sha256,
        "artifacts": [
            {
                "path": name,
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256((output_dir / name).read_bytes()).hexdigest(),
            }
            for name in files
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _managed_artifacts(output_dir: Path) -> set[str]:
    """Validate and return managed leaf names from a previous bundle manifest."""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("existing audit manifest cannot be a symlink")
    if not manifest_path.exists():
        return set()
    if not manifest_path.is_file():
        raise ValueError("existing audit manifest is not a regular file")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("existing audit manifest is unreadable or invalid JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), list):
        raise ValueError("existing audit manifest has an invalid artifact inventory")

    names: set[str] = set()
    for item in payload["artifacts"]:
        if not isinstance(item, dict):
            raise ValueError("existing audit manifest artifact entries must be objects")
        name = item.get("path")
        byte_count = item.get("bytes")
        checksum = item.get("sha256")
        if (
            not isinstance(name, str)
            or name == "manifest.json"
            or name not in _PHASE1_MANAGED_ARTIFACT_NAMES
            or Path(name).name != name
            or Path(name).is_absolute()
        ):
            raise ValueError("existing audit manifest contains an unsafe artifact path")
        if name in names:
            raise ValueError("existing audit manifest contains duplicate artifact paths")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError("existing audit manifest contains an invalid artifact size")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError("existing audit manifest contains an invalid artifact checksum")
        artifact = output_dir / name
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError("existing audit manifest references a missing or unsafe artifact")
        contents = artifact.read_bytes()
        if len(contents) != byte_count or sha256(contents).hexdigest() != checksum:
            raise ValueError("existing audit artifact does not match its manifest")
        names.add(name)
    names.add("manifest.json")
    return names


def _save_audit_overview(report: Phase1AuditReport, path: Path) -> None:
    try:
        matplotlib = import_module("matplotlib")
        matplotlib.use("Agg")
        pyplot = cast(_PyplotModule, import_module("matplotlib.pyplot"))
    except ModuleNotFoundError as error:
        raise ImportError(
            'audit visualizations require "rejuvenationkit[visualization]"'
        ) from error
    figure, axes = pyplot.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes_array = list(axes.flat)
    _plot_findings(report, axes_array[0])
    _plot_coverage(report, axes_array[1])
    _plot_retention(report, axes_array[2])
    _plot_outliers(report, axes_array[3])
    figure.suptitle(f"Phase 1 study audit: {report.study_id}")
    figure.savefig(path, dpi=160, bbox_inches="tight")
    pyplot.close(figure)


def _plot_findings(report: Phase1AuditReport, axis: Any) -> None:
    labels = ("error", "warning", "info")
    counts = [report.qc.counts[Severity(label)] for label in labels]
    axis.bar(labels, counts, color=("tab:red", "tab:orange", "tab:blue"))
    axis.set_ylabel("Findings")
    axis.set_title("QC findings by severity")
    axis.grid(axis="y", alpha=0.2)


def _plot_coverage(report: Phase1AuditReport, axis: Any) -> None:
    rows = [item for item in report.profile.visit_coverage if item.cohort == "all"]
    if not rows:
        _empty_axis(axis, "Visit-feature coverage", "No expected-visit coverage")
        return
    frame = pd.DataFrame(
        {
            "visit": item.visit_id,
            "feature": item.feature,
            "coverage": item.coverage_fraction,
        }
        for item in rows
    )
    pivot = frame.pivot(index="feature", columns="visit", values="coverage")
    image = axis.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Visit-feature coverage")
    axis.figure.colorbar(image, ax=axis, label="Coverage")


def _plot_retention(report: Phase1AuditReport, axis: Any) -> None:
    rows = [item for item in report.profile.visit_retention if item.cohort == "all"]
    if not rows:
        _empty_axis(axis, "Complete-case retention", "Fewer than two expected visits")
        return
    labels = [f"{item.from_visit_id} → {item.to_visit_id}" for item in rows]
    values = [item.retention_fraction for item in rows]
    axis.bar(range(len(rows)), values, color="tab:green")
    axis.set_xticks(range(len(rows)), labels, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Retention")
    axis.set_title("Complete-case retention")
    axis.grid(axis="y", alpha=0.2)


def _plot_outliers(report: Phase1AuditReport, axis: Any) -> None:
    rows = [
        item
        for item in report.profile.feature_distributions
        if item.cohort == "all" and item.outlier_count
    ]
    if not rows:
        _empty_axis(axis, "Robust outlier flags", "No outliers flagged")
        return
    labels = [f"{item.visit_id}\n{item.feature}" for item in rows]
    axis.bar(range(len(rows)), [item.outlier_count for item in rows], color="tab:purple")
    axis.set_xticks(range(len(rows)), labels, rotation=30, ha="right")
    axis.set_ylabel("Subjects")
    axis.set_title("Robust outlier flags")
    axis.grid(axis="y", alpha=0.2)


def _empty_axis(axis: Any, title: str, message: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
