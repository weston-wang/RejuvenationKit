"""One-command Phase 1 study audits and reproducible report bundles."""

from __future__ import annotations

import json
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.detection import (
    ChangeDetectionConfig,
    ChangeDetectionReport,
    MultivariateChangeDetector,
)
from rejuvenationkit.profiling import StudyProfile, StudyProfiler
from rejuvenationkit.qc import BaselineLongitudinalQC, QCConfig, QCReport, Severity
from rejuvenationkit.schemas import Study
from rejuvenationkit.treatment_effect import (
    RandomizedTreatmentEffectEvaluator,
    TreatmentEffectConfig,
    TreatmentEffectReport,
)


class ChangeDetectionAuditPlan(BaseModel):
    """Prespecified held-out multivariate analysis included in an audit."""

    model_config = ConfigDict(frozen=True)

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
        overlap = set(self.reference_subject_ids).intersection(self.evaluation_subject_ids)
        if overlap:
            raise ValueError(
                f"change-detection reference and evaluation subjects overlap: {sorted(overlap)}"
            )
        return self


class TreatmentAuditPlan(BaseModel):
    """Prespecified randomized comparison included in a Phase 1 audit."""

    model_config = ConfigDict(frozen=True)

    config: TreatmentEffectConfig
    baseline_visit_id: str = Field(min_length=1)
    follow_up_visit_ids: tuple[str, ...] = Field(min_length=1)
    treated_subject_ids: tuple[str, ...] = Field(min_length=1)
    control_subject_ids: tuple[str, ...] = Field(min_length=1)
    treated_label: str = Field(default="treated", min_length=1)
    control_label: str = Field(default="control", min_length=1)

    @model_validator(mode="after")
    def require_unique_visits(self) -> TreatmentAuditPlan:
        """Reject duplicate or overlapping visit roles."""
        if len(self.follow_up_visit_ids) != len(set(self.follow_up_visit_ids)):
            raise ValueError("treatment audit follow-up visits must be unique")
        if self.baseline_visit_id in self.follow_up_visit_ids:
            raise ValueError("treatment audit baseline cannot be a follow-up")
        return self


class Phase1AuditConfig(BaseModel):
    """Configuration for the complete Phase 1 audit workflow."""

    model_config = ConfigDict(frozen=True)

    qc: QCConfig
    outlier_iqr_multiplier: float = Field(default=1.5, ge=0)
    include_visualizations: bool = True


class Phase1AuditReport(BaseModel):
    """Machine-readable Phase 1 audit and its generated artifact manifest."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1"
    software_version: str
    study_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    study_metadata: dict[str, str | int | float | bool]
    config: Phase1AuditConfig
    change_detection_plan: ChangeDetectionAuditPlan | None = None
    treatment_plan: TreatmentAuditPlan | None = None
    qc: QCReport
    profile: StudyProfile
    change_detection: ChangeDetectionReport | None = None
    treatment_effect: TreatmentEffectReport | None = None
    artifacts: tuple[str, ...] = ()

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
        treatment_plan: TreatmentAuditPlan | None = None,
    ) -> Phase1AuditReport:
        """Run the configured audit and write a reproducible report bundle."""
        output_dir.mkdir(parents=True, exist_ok=True)
        qc_report = BaselineLongitudinalQC(self.config.qc).run(study)
        profile = StudyProfiler(
            self.config.qc,
            outlier_iqr_multiplier=self.config.outlier_iqr_multiplier,
        ).profile(study)
        change_detection = (
            self._change_detection(study, change_detection_plan)
            if change_detection_plan is not None
            else None
        )
        treatment = (
            self._treatment_effect(study, treatment_plan) if treatment_plan is not None else None
        )
        artifact_names = [
            "audit.json",
            "summary.md",
            "findings.csv",
            "visit_coverage.csv",
            "visit_retention.csv",
            "paired_readiness.csv",
            "feature_distributions.csv",
            "attrition_bias.csv",
            "manifest.json",
        ]
        if change_detection is not None:
            artifact_names.append("change_detection_scores.csv")
            if self.config.include_visualizations:
                artifact_names.extend(
                    (
                        "change-detection-covariance.png",
                        "change-detection-scores.png",
                        "change-detection-whitened.png",
                        "change-detection-decomposition.png",
                    )
                )
        if treatment is not None:
            artifact_names.extend(("treatment_effects.csv", "treatment_subject_scores.csv"))
        if self.config.include_visualizations:
            artifact_names.append("audit_overview.png")
        report = Phase1AuditReport(
            software_version=_software_version(),
            study_id=study.study_id,
            input_sha256=_study_fingerprint(study),
            study_metadata=study.metadata,
            config=self.config,
            change_detection_plan=change_detection_plan,
            treatment_plan=treatment_plan,
            qc=qc_report,
            profile=profile,
            change_detection=change_detection,
            treatment_effect=treatment,
            artifacts=tuple(artifact_names),
        )
        self._write_bundle(report, output_dir)
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

    def _write_bundle(self, report: Phase1AuditReport, output_dir: Path) -> None:
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
        }
        for name, frame in frames.items():
            frame.to_csv(output_dir / name, index=False)
        if report.change_detection is not None:
            report.change_detection.results_frame().to_csv(
                output_dir / "change_detection_scores.csv",
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
        if self.config.include_visualizations:
            _save_audit_overview(report, output_dir / "audit_overview.png")
            if report.change_detection is not None:
                from rejuvenationkit.visualization import save_detection_figures

                save_detection_figures(
                    report.change_detection,
                    output_dir,
                    prefix="change-detection",
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
    treatment_plan: TreatmentAuditPlan | None = None,
) -> Phase1AuditReport:
    """Run a complete Phase 1 audit in one function call."""
    return Phase1AuditRunner(config).run(
        study,
        output_dir=output_dir,
        change_detection_plan=change_detection_plan,
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
            "context": json.dumps(item.context, sort_keys=True),
        }
        for item in report.findings
    ]
    return pd.DataFrame(rows, columns=columns)


def _study_fingerprint(study: Study) -> str:
    payload = json.dumps(
        study.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(payload).hexdigest()


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
