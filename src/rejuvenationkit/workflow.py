"""Fail-closed orchestration across the four RejuvenationKit phases.

This module coordinates existing phase implementations without manufacturing
scientific inputs at phase boundaries. Phase 3 therefore requires an explicit
``TimedEvidenceBatch`` and Phase 4 requires an explicit
``SubjectEndpointBatch``. In particular, timestamps are never parsed from an
estimand label and state trajectories are never converted to endpoints by the
workflow.
"""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.audit import (
    ChangeDetectionAuditPlan,
    Phase1AuditConfig,
    Phase1AuditReport,
    SequentialDetectionAuditPlan,
    TreatmentAuditPlan,
    run_phase1_audit,
)
from rejuvenationkit.bridges import TimedEvidenceBatch
from rejuvenationkit.combinations import (
    CombinationAnalysisReport,
    FactorialCombinationAnalysis,
    FactorialCombinationConfig,
)
from rejuvenationkit.endpoints import SubjectEndpointBatch, study_artifact_hash
from rejuvenationkit.evidence import (
    EvidenceCovariance,
    EvidenceEstimate,
    EvidenceFusionConfig,
    EvidenceFusionResult,
    GeneralizedLeastSquaresFusion,
)
from rejuvenationkit.qc import Severity
from rejuvenationkit.schemas import Study
from rejuvenationkit.state import (
    LinearGaussianStateConfig,
    LinearGaussianStateEstimator,
    StateEstimationReport,
)

_REPORT_NAME = "workflow.json"
_MANIFEST_NAME = "workflow-manifest.json"
_PHASE1_DIRECTORY = "phase1"


def _canonical_hash(payload: object) -> str:
    """Hash one already JSON-compatible payload deterministically."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    """Return the SHA-256 identity of one file."""
    return sha256(path.read_bytes()).hexdigest()


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and path != Path(".")


def _reject_symlink_components(root: Path, relative: str, *, context: str) -> None:
    """Reject a symlink at the bundle root, target, or any managed ancestor."""
    if not _safe_relative_path(relative):
        raise ValueError(f"{context} path must be safe and relative: {relative}")
    if root.is_symlink():
        raise ValueError(f"{context} output directory cannot be a symbolic link: {root}")
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{context} path cannot traverse a symbolic link: {relative}")


class WorkflowPhase(StrEnum):
    """Named SDK phases in execution order."""

    PHASE1 = "phase1"
    PHASE2 = "phase2"
    PHASE3 = "phase3"
    PHASE4 = "phase4"


class PhaseExecutionStatus(StrEnum):
    """Disposition of one phase in a completed workflow report."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ArtifactIdentity(BaseModel):
    """A named immutable SHA-256 identity used at a phase boundary."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PhaseDisposition(BaseModel):
    """Execution status plus the exact artifacts consumed and produced."""

    model_config = ConfigDict(frozen=True)

    phase: WorkflowPhase
    status: PhaseExecutionStatus
    reason: str = Field(min_length=1)
    input_artifacts: tuple[ArtifactIdentity, ...] = ()
    output_artifacts: tuple[ArtifactIdentity, ...] = ()


class Phase1WorkflowConfig(BaseModel):
    """Phase 1 audit policy and optional prespecified analyses."""

    model_config = ConfigDict(frozen=True)

    audit: Phase1AuditConfig
    change_detection_plan: ChangeDetectionAuditPlan | None = None
    sequential_detection_plan: SequentialDetectionAuditPlan | None = None
    treatment_plan: TreatmentAuditPlan | None = None


class Phase2WorkflowConfig(BaseModel):
    """Configuration for one covariance-aware GLS evidence fusion."""

    model_config = ConfigDict(frozen=True)

    fusion: EvidenceFusionConfig


class Phase3WorkflowConfig(BaseModel):
    """Fully specified Phase 3 state-space model."""

    model_config = ConfigDict(frozen=True)

    state_model: LinearGaussianStateConfig


class Phase4WorkflowConfig(BaseModel):
    """Prespecified factorial model and exact outcome identity."""

    model_config = ConfigDict(frozen=True)

    factorial: FactorialCombinationConfig
    outcome: str = Field(min_length=1)


class RejuvenationWorkflowConfig(BaseModel):
    """Serialized configuration for a complete or partial four-phase run."""

    model_config = ConfigDict(frozen=True)

    phase1: Phase1WorkflowConfig
    phase2: Phase2WorkflowConfig | None = None
    phase3: Phase3WorkflowConfig | None = None
    phase4: Phase4WorkflowConfig | None = None


class Phase2FusionInput(BaseModel):
    """Exact evidence and covariance supplied to Phase 2 GLS fusion."""

    model_config = ConfigDict(frozen=True)

    input_id: str = Field(min_length=1)
    estimates: tuple[EvidenceEstimate, ...] = Field(min_length=1)
    covariance: EvidenceCovariance

    @model_validator(mode="after")
    def validate_covariance_alignment(self) -> Self:
        """Require covariance to identify exactly the supplied evidence rows."""
        evidence_ids = [item.evidence_id for item in self.estimates]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Phase 2 evidence_id values must be unique")
        if set(evidence_ids) != set(self.covariance.evidence_ids):
            raise ValueError("Phase 2 covariance must identify exactly the supplied evidence")
        return self

    @property
    def artifact_hash(self) -> str:
        """Bind evidence ordering, values, calibration, and covariance exactly."""
        return _canonical_hash(
            {
                "schema": "phase2-gls-input/v1",
                "input_id": self.input_id,
                "estimates": [item.model_dump(mode="json") for item in self.estimates],
                "covariance": self.covariance.model_dump(mode="json"),
            }
        )

    @property
    def covariance_artifact_hash(self) -> str:
        """Return the exact identity of the supplied covariance contract."""
        return _canonical_hash(self.covariance.model_dump(mode="json"))


class RejuvenationWorkflowInputs(BaseModel):
    """Validated data inputs; downstream boundaries stay explicit."""

    model_config = ConfigDict(frozen=True)

    study: Study
    phase2: Phase2FusionInput | None = None
    phase3: TimedEvidenceBatch | None = None
    phase4: SubjectEndpointBatch | None = None


class RejuvenationWorkflowRequest(BaseModel):
    """A paired, serializable workflow configuration and input contract."""

    model_config = ConfigDict(frozen=True)

    config: RejuvenationWorkflowConfig
    inputs: RejuvenationWorkflowInputs

    @model_validator(mode="after")
    def validate_phase_pairing(self) -> Self:
        """Reject configured phases without input and unconfigured phase inputs."""
        pairs = (
            (WorkflowPhase.PHASE2, self.config.phase2, self.inputs.phase2),
            (WorkflowPhase.PHASE3, self.config.phase3, self.inputs.phase3),
            (WorkflowPhase.PHASE4, self.config.phase4, self.inputs.phase4),
        )
        for phase, config, supplied_input in pairs:
            if (config is None) != (supplied_input is None):
                raise ValueError(
                    f"{phase.value} configuration and input must either both be supplied "
                    "or both be omitted"
                )
        if self.inputs.phase3 is not None:
            self._validate_phase3_subjects(self.inputs.phase3)
        if self.inputs.phase4 is not None:
            self._validate_phase4_subjects(self.inputs.phase4)
        return self

    def _validate_phase3_subjects(self, batch: TimedEvidenceBatch) -> None:
        if batch.study_id != self.inputs.study.study_id:
            raise ValueError("Phase 3 timed evidence study_id must match the Phase 1 study")
        source = {item.subject_id: item for item in self.inputs.study.subjects}
        unknown = {item.subject_id for item in batch.subjects}.difference(source)
        if unknown:
            raise ValueError(f"Phase 3 timed evidence contains unknown subjects: {sorted(unknown)}")
        changed = sorted(
            item.subject_id for item in batch.subjects if source[item.subject_id] != item
        )
        if changed:
            raise ValueError(
                f"Phase 3 subject assignments differ from the Phase 1 study: {changed}"
            )

    def _validate_phase4_subjects(self, batch: SubjectEndpointBatch) -> None:
        if batch.study_id != self.inputs.study.study_id:
            raise ValueError("Phase 4 endpoint study_id must match the Phase 1 study")
        known = {item.subject_id for item in self.inputs.study.subjects}
        supplied = {item.subject_id for item in batch.endpoints}.union(
            item.subject_id for item in batch.excluded
        )
        unknown = supplied.difference(known)
        if unknown:
            raise ValueError(f"Phase 4 endpoint batch contains unknown subjects: {sorted(unknown)}")


class WorkflowQCGate(BaseModel):
    """Auditable decision controlling all configured downstream phases."""

    model_config = ConfigDict(frozen=True)

    qc_passed: bool
    error_findings: int = Field(ge=0)
    downstream_configured: bool
    serialized_override_enabled: bool
    override_applied: bool
    downstream_allowed: bool

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        """Keep the gate result consistent with QC and override policy."""
        expected_allowed = self.qc_passed or self.serialized_override_enabled
        if self.downstream_allowed != expected_allowed:
            raise ValueError("downstream_allowed is inconsistent with QC override policy")
        expected_override = (
            not self.qc_passed and self.downstream_configured and self.serialized_override_enabled
        )
        if self.override_applied != expected_override:
            raise ValueError("override_applied is inconsistent with configured downstream phases")
        if self.qc_passed != (self.error_findings == 0):
            raise ValueError("qc_passed must agree with the number of error findings")
        return self


class Phase2WorkflowResult(BaseModel):
    """Phase 2 fusion output with immutable input and output identities."""

    model_config = ConfigDict(frozen=True)

    input_id: str
    input_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    covariance_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fusion: EvidenceFusionResult

    @model_validator(mode="after")
    def validate_result_hash(self) -> Self:
        """Reject a fusion result whose serialized identity has been altered."""
        if self.result_artifact_hash != _canonical_hash(self.fusion.model_dump(mode="json")):
            raise ValueError("Phase 2 result_artifact_hash does not match fusion result")
        return self


class Phase3WorkflowResult(BaseModel):
    """Phase 3 output bound to timed evidence and an exact state model."""

    model_config = ConfigDict(frozen=True)

    timed_evidence_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    derived_study_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_estimation: StateEstimationReport

    @model_validator(mode="after")
    def validate_result_hash(self) -> Self:
        """Reject a state report whose serialized identity has been altered."""
        if self.result_artifact_hash != self.state_estimation.artifact_hash:
            raise ValueError("Phase 3 result_artifact_hash does not match state report")
        if self.derived_study_artifact_hash != self.state_estimation.study_artifact_hash:
            raise ValueError("Phase 3 derived-study identity does not match state report")
        if self.config_artifact_hash != self.state_estimation.model_config_artifact_hash:
            raise ValueError("Phase 3 model identity does not match state report")
        return self


class Phase4WorkflowResult(BaseModel):
    """Phase 4 output bound to the caller-supplied subject endpoint artifact."""

    model_config = ConfigDict(frozen=True)

    endpoint_batch_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoint_source_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    combination_analysis: CombinationAnalysisReport

    @model_validator(mode="after")
    def validate_result_hash(self) -> Self:
        """Reject a combination report whose serialized identity has been altered."""
        if self.result_artifact_hash != self.combination_analysis.artifact_hash:
            raise ValueError("Phase 4 result_artifact_hash does not match combination report")
        if self.endpoint_batch_artifact_hash != self.combination_analysis.endpoint_batch_hash:
            raise ValueError("Phase 4 endpoint identity does not match combination report")
        if (
            self.endpoint_source_artifact_hash
            != self.combination_analysis.endpoint_source_artifact_hash
        ):
            raise ValueError("Phase 4 endpoint source identity does not match combination report")
        return self


class RejuvenationWorkflowReport(BaseModel):
    """Top-level machine-readable report for an executed workflow."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1"] = "1"
    software_version: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    study_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: RejuvenationWorkflowConfig
    qc_gate: WorkflowQCGate
    phase1: Phase1AuditReport
    phase2: Phase2WorkflowResult | None = None
    phase3: Phase3WorkflowResult | None = None
    phase4: Phase4WorkflowResult | None = None
    dispositions: tuple[PhaseDisposition, ...] = Field(min_length=4, max_length=4)
    phase1_subdirectory: Literal["phase1"] = "phase1"
    artifacts: tuple[str, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        """Require one coherent disposition and identity for every phase."""
        if self.software_version != self.phase1.software_version:
            raise ValueError("workflow and Phase 1 software versions must match")
        if self.phase1.config != self.config.phase1.audit:
            raise ValueError("workflow and Phase 1 audit configurations must match")
        if self.phase1.change_detection_plan != self.config.phase1.change_detection_plan:
            raise ValueError("workflow and Phase 1 change-detection plans must match")
        if self.phase1.sequential_detection_plan != self.config.phase1.sequential_detection_plan:
            raise ValueError("workflow and Phase 1 sequential-detection plans must match")
        if self.phase1.treatment_plan != self.config.phase1.treatment_plan:
            raise ValueError("workflow and Phase 1 treatment plans must match")
        expected_gate = WorkflowQCGate(
            qc_passed=self.phase1.qc.passed,
            error_findings=self.phase1.qc.counts[Severity.ERROR],
            downstream_configured=any(
                item is not None
                for item in (self.config.phase2, self.config.phase3, self.config.phase4)
            ),
            serialized_override_enabled=self.config.phase1.audit.allow_analysis_with_qc_errors,
            override_applied=(
                not self.phase1.qc.passed
                and any(
                    item is not None
                    for item in (self.config.phase2, self.config.phase3, self.config.phase4)
                )
                and self.config.phase1.audit.allow_analysis_with_qc_errors
            ),
            downstream_allowed=(
                self.phase1.qc.passed or self.config.phase1.audit.allow_analysis_with_qc_errors
            ),
        )
        if self.qc_gate != expected_gate:
            raise ValueError("workflow QC gate does not match Phase 1 findings and configuration")
        expected_phases = tuple(WorkflowPhase)
        if tuple(item.phase for item in self.dispositions) != expected_phases:
            raise ValueError("workflow dispositions must contain Phase 1 through Phase 4 in order")
        results: tuple[object | None, ...] = (
            self.phase1,
            self.phase2,
            self.phase3,
            self.phase4,
        )
        for disposition, result in zip(self.dispositions, results, strict=True):
            if disposition.phase is WorkflowPhase.PHASE1:
                if disposition.status is not PhaseExecutionStatus.COMPLETED:
                    raise ValueError("Phase 1 must always complete in a published workflow")
                continue
            if (result is not None) != (disposition.status is PhaseExecutionStatus.COMPLETED):
                raise ValueError("phase result presence does not match phase disposition")
        if self.study_id != self.phase1.study_id:
            raise ValueError("workflow and Phase 1 study identifiers must match")
        if self.study_artifact_hash != self.phase1.input_sha256:
            raise ValueError("workflow and Phase 1 study artifact identities must match")
        if self.phase3 is not None and self.phase3.state_estimation.study_id != self.study_id:
            raise ValueError("workflow and Phase 3 study identifiers must match")
        if self.phase4 is not None:
            if self.phase4.combination_analysis.study_id != self.study_id:
                raise ValueError("workflow and Phase 4 study identifiers must match")
            if self.phase4.combination_analysis.study_artifact_hash != self.study_artifact_hash:
                raise ValueError("workflow and Phase 4 study artifact identities must match")
        if len(set(self.artifacts)) != len(self.artifacts):
            raise ValueError("workflow artifact paths must be unique")
        if any(not _safe_relative_path(path) for path in self.artifacts):
            raise ValueError("workflow artifact paths must be safe and relative")
        expected_artifacts = {
            *(f"{_PHASE1_DIRECTORY}/{name}" for name in self.phase1.artifacts),
            _REPORT_NAME,
            _MANIFEST_NAME,
        }
        if set(self.artifacts) != expected_artifacts:
            raise ValueError("workflow artifact inventory must contain the exact Phase 1 bundle")
        self._validate_disposition_identities()
        return self

    def _validate_disposition_identities(self) -> None:
        """Reconstruct every identity available in the nested report."""
        phase1_expected = PhaseDisposition(
            phase=WorkflowPhase.PHASE1,
            status=PhaseExecutionStatus.COMPLETED,
            reason=("qc_passed" if self.phase1.qc.passed else "qc_completed_with_errors"),
            input_artifacts=(
                ArtifactIdentity(name="study", sha256=self.study_artifact_hash),
                ArtifactIdentity(
                    name="phase1_config",
                    sha256=_canonical_hash(self.config.phase1.model_dump(mode="json")),
                ),
            ),
            output_artifacts=(
                ArtifactIdentity(
                    name="phase1_audit_report",
                    sha256=_canonical_hash(self.phase1.model_dump(mode="json")),
                ),
            ),
        )
        if self.dispositions[0] != phase1_expected:
            raise ValueError("Phase 1 disposition identities do not match the report")

        phase_results: tuple[tuple[WorkflowPhase, BaseModel | None, BaseModel | None], ...] = (
            (WorkflowPhase.PHASE2, self.config.phase2, self.phase2),
            (WorkflowPhase.PHASE3, self.config.phase3, self.phase3),
            (WorkflowPhase.PHASE4, self.config.phase4, self.phase4),
        )
        for index, (phase, configured, result) in enumerate(phase_results, start=1):
            disposition = self.dispositions[index]
            if configured is None:
                expected = PhaseDisposition(
                    phase=phase,
                    status=PhaseExecutionStatus.SKIPPED,
                    reason="not_configured",
                )
                if disposition != expected:
                    raise ValueError(f"{phase.value} skipped disposition is inconsistent")
                continue
            if not self.qc_gate.downstream_allowed:
                self._validate_blocked_disposition(disposition, phase, configured)
                continue
            if result is None:
                raise ValueError(f"{phase.value} completed result is missing")
            expected_config_hash = self._phase_config_hash(phase, configured)
            if not isinstance(
                result,
                (Phase2WorkflowResult, Phase3WorkflowResult, Phase4WorkflowResult),
            ):
                raise ValueError(f"{phase.value} result has an unexpected type")
            if result.config_artifact_hash != expected_config_hash:
                raise ValueError(f"{phase.value} result config identity is inconsistent")
            expected = self._completed_disposition(phase, result)
            if disposition != expected:
                raise ValueError(f"{phase.value} disposition identities do not match its result")

    @staticmethod
    def _phase_config_hash(phase: WorkflowPhase, configured: BaseModel) -> str:
        if phase is WorkflowPhase.PHASE3:
            assert isinstance(configured, Phase3WorkflowConfig)
            return _canonical_hash(configured.state_model.model_dump(mode="json"))
        return _canonical_hash(configured.model_dump(mode="json"))

    def _validate_blocked_disposition(
        self,
        disposition: PhaseDisposition,
        phase: WorkflowPhase,
        configured: BaseModel,
    ) -> None:
        if disposition.status is not PhaseExecutionStatus.BLOCKED:
            raise ValueError(f"{phase.value} must be blocked by the failed QC gate")
        if disposition.reason != "phase1_qc_error_without_serialized_override":
            raise ValueError(f"{phase.value} blocked reason is inconsistent")
        if disposition.output_artifacts:
            raise ValueError(f"{phase.value} blocked disposition cannot have outputs")
        expected_names = {
            WorkflowPhase.PHASE2: {"phase2_config", "phase2_gls_input", "phase2_covariance"},
            WorkflowPhase.PHASE3: {"phase3_config", "timed_evidence_batch"},
            WorkflowPhase.PHASE4: {
                "phase4_config",
                "subject_endpoint_batch",
                "endpoint_source_artifact",
            },
        }[phase]
        identities = {item.name: item.sha256 for item in disposition.input_artifacts}
        if set(identities) != expected_names:
            raise ValueError(f"{phase.value} blocked input identity names are incomplete")
        expected_config_hash = self._phase_config_hash(phase, configured)
        if identities[f"{phase.value}_config"] != expected_config_hash:
            raise ValueError(f"{phase.value} blocked config identity is inconsistent")

    @staticmethod
    def _completed_disposition(
        phase: WorkflowPhase,
        result: BaseModel,
    ) -> PhaseDisposition:
        if phase is WorkflowPhase.PHASE2:
            assert isinstance(result, Phase2WorkflowResult)
            return PhaseDisposition(
                phase=phase,
                status=PhaseExecutionStatus.COMPLETED,
                reason="covariance_aware_gls_completed",
                input_artifacts=(
                    ArtifactIdentity(
                        name="phase2_gls_input",
                        sha256=result.input_artifact_hash,
                    ),
                    ArtifactIdentity(
                        name="phase2_covariance",
                        sha256=result.covariance_artifact_hash,
                    ),
                    ArtifactIdentity(name="phase2_config", sha256=result.config_artifact_hash),
                ),
                output_artifacts=(
                    ArtifactIdentity(
                        name="phase2_fusion_result",
                        sha256=result.result_artifact_hash,
                    ),
                ),
            )
        if phase is WorkflowPhase.PHASE3:
            assert isinstance(result, Phase3WorkflowResult)
            return PhaseDisposition(
                phase=phase,
                status=PhaseExecutionStatus.COMPLETED,
                reason="prespecified_state_estimation_completed",
                input_artifacts=(
                    ArtifactIdentity(
                        name="timed_evidence_batch",
                        sha256=result.timed_evidence_artifact_hash,
                    ),
                    ArtifactIdentity(
                        name="phase3_derived_study",
                        sha256=result.derived_study_artifact_hash,
                    ),
                    ArtifactIdentity(name="phase3_config", sha256=result.config_artifact_hash),
                ),
                output_artifacts=(
                    ArtifactIdentity(
                        name="phase3_state_report",
                        sha256=result.result_artifact_hash,
                    ),
                ),
            )
        assert phase is WorkflowPhase.PHASE4
        assert isinstance(result, Phase4WorkflowResult)
        return PhaseDisposition(
            phase=phase,
            status=PhaseExecutionStatus.COMPLETED,
            reason="explicit_subject_endpoint_factorial_analysis_completed",
            input_artifacts=(
                ArtifactIdentity(
                    name="subject_endpoint_batch",
                    sha256=result.endpoint_batch_artifact_hash,
                ),
                ArtifactIdentity(
                    name="endpoint_source_artifact",
                    sha256=result.endpoint_source_artifact_hash,
                ),
                ArtifactIdentity(name="phase4_config", sha256=result.config_artifact_hash),
            ),
            output_artifacts=(
                ArtifactIdentity(
                    name="phase4_combination_report",
                    sha256=result.result_artifact_hash,
                ),
            ),
        )


class WorkflowManifestArtifact(BaseModel):
    """One checksummed workflow-bundle file."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Prevent absolute paths and traversal in a bundle manifest."""
        if not _safe_relative_path(self.path):
            raise ValueError("manifest artifact path must be safe and relative")
        return self


class WorkflowManifest(BaseModel):
    """Checksum manifest written last as the workflow transaction marker."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1"] = "1"
    bundle_type: Literal["rejuvenationkit-workflow"] = "rejuvenationkit-workflow"
    software_version: str
    study_id: str
    study_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    workflow_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[WorkflowManifestArtifact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Require unique paths and bind the named workflow report checksum."""
        paths = [item.path for item in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("workflow manifest paths must be unique")
        reports = [item for item in self.artifacts if item.path == _REPORT_NAME]
        if len(reports) != 1 or reports[0].sha256 != self.workflow_report_sha256:
            raise ValueError("workflow_report_sha256 must identify workflow.json")
        return self


class RejuvenationWorkflowRunner:
    """Execute configured phases behind a serialized Phase 1 QC gate."""

    def run(
        self,
        request: RejuvenationWorkflowRequest,
        *,
        output_dir: Path,
    ) -> RejuvenationWorkflowReport:
        """Execute a request and publish one reproducible workflow bundle."""
        output_parent = output_dir.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        if output_dir.is_symlink():
            raise ValueError(f"workflow output path cannot be a symbolic link: {output_dir}")
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"workflow output path is not a regular directory: {output_dir}")

        with TemporaryDirectory(
            prefix=f".{output_dir.name}-workflow-staging-",
            dir=output_parent,
        ) as raw_staging:
            staging = Path(raw_staging)
            report = self._execute(request, phase1_output_dir=staging / _PHASE1_DIRECTORY)
            report_path = staging / _REPORT_NAME
            report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
            manifest = _build_manifest(report, staging)
            (staging / _MANIFEST_NAME).write_text(
                manifest.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            _publish_staged_bundle(staging, output_dir)
        return report

    def _execute(
        self,
        request: RejuvenationWorkflowRequest,
        *,
        phase1_output_dir: Path,
    ) -> RejuvenationWorkflowReport:
        study = request.inputs.study
        config = request.config
        phase1 = run_phase1_audit(
            study,
            config=config.phase1.audit,
            output_dir=phase1_output_dir,
            change_detection_plan=config.phase1.change_detection_plan,
            sequential_detection_plan=config.phase1.sequential_detection_plan,
            treatment_plan=config.phase1.treatment_plan,
        )
        downstream_configured = any(
            item is not None for item in (config.phase2, config.phase3, config.phase4)
        )
        error_count = phase1.qc.counts[Severity.ERROR]
        gate = WorkflowQCGate(
            qc_passed=phase1.qc.passed,
            error_findings=error_count,
            downstream_configured=downstream_configured,
            serialized_override_enabled=config.phase1.audit.allow_analysis_with_qc_errors,
            override_applied=(
                not phase1.qc.passed
                and downstream_configured
                and config.phase1.audit.allow_analysis_with_qc_errors
            ),
            downstream_allowed=(
                phase1.qc.passed or config.phase1.audit.allow_analysis_with_qc_errors
            ),
        )

        phase2: Phase2WorkflowResult | None = None
        phase3: Phase3WorkflowResult | None = None
        phase4: Phase4WorkflowResult | None = None
        dispositions: list[PhaseDisposition] = [
            PhaseDisposition(
                phase=WorkflowPhase.PHASE1,
                status=PhaseExecutionStatus.COMPLETED,
                reason=("qc_passed" if phase1.qc.passed else "qc_completed_with_errors"),
                input_artifacts=(
                    ArtifactIdentity(name="study", sha256=phase1.input_sha256),
                    ArtifactIdentity(
                        name="phase1_config",
                        sha256=_canonical_hash(config.phase1.model_dump(mode="json")),
                    ),
                ),
                output_artifacts=(
                    ArtifactIdentity(
                        name="phase1_audit_report",
                        sha256=_canonical_hash(phase1.model_dump(mode="json")),
                    ),
                ),
            )
        ]

        if gate.downstream_allowed:
            phase2, disposition = self._run_phase2(config, request.inputs)
            dispositions.append(disposition)
            phase3, disposition = self._run_phase3(config, request.inputs)
            dispositions.append(disposition)
            phase4, disposition = self._run_phase4(study, config, request.inputs)
            dispositions.append(disposition)
        else:
            dispositions.extend(
                self._blocked_or_skipped_disposition(phase, configured, supplied_input)
                for phase, configured, supplied_input in (
                    (WorkflowPhase.PHASE2, config.phase2, request.inputs.phase2),
                    (WorkflowPhase.PHASE3, config.phase3, request.inputs.phase3),
                    (WorkflowPhase.PHASE4, config.phase4, request.inputs.phase4),
                )
            )

        artifacts = (
            *(f"{_PHASE1_DIRECTORY}/{name}" for name in phase1.artifacts),
            _REPORT_NAME,
            _MANIFEST_NAME,
        )
        return RejuvenationWorkflowReport(
            software_version=phase1.software_version,
            study_id=study.study_id,
            study_artifact_hash=phase1.input_sha256,
            config=config,
            qc_gate=gate,
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            phase4=phase4,
            dispositions=tuple(dispositions),
            artifacts=artifacts,
        )

    def _run_phase2(
        self,
        config: RejuvenationWorkflowConfig,
        inputs: RejuvenationWorkflowInputs,
    ) -> tuple[Phase2WorkflowResult | None, PhaseDisposition]:
        if config.phase2 is None:
            return None, self._skipped_disposition(WorkflowPhase.PHASE2)
        phase_input = inputs.phase2
        if phase_input is None:
            raise RuntimeError("validated Phase 2 input is unexpectedly absent")
        fusion = GeneralizedLeastSquaresFusion(config.phase2.fusion).fuse(
            phase_input.estimates,
            phase_input.covariance,
        )
        config_hash = _canonical_hash(config.phase2.model_dump(mode="json"))
        result_hash = _canonical_hash(fusion.model_dump(mode="json"))
        result = Phase2WorkflowResult(
            input_id=phase_input.input_id,
            input_artifact_hash=phase_input.artifact_hash,
            covariance_artifact_hash=phase_input.covariance_artifact_hash,
            config_artifact_hash=config_hash,
            result_artifact_hash=result_hash,
            fusion=fusion,
        )
        return result, PhaseDisposition(
            phase=WorkflowPhase.PHASE2,
            status=PhaseExecutionStatus.COMPLETED,
            reason="covariance_aware_gls_completed",
            input_artifacts=(
                ArtifactIdentity(name="phase2_gls_input", sha256=phase_input.artifact_hash),
                ArtifactIdentity(
                    name="phase2_covariance",
                    sha256=phase_input.covariance_artifact_hash,
                ),
                ArtifactIdentity(name="phase2_config", sha256=config_hash),
            ),
            output_artifacts=(ArtifactIdentity(name="phase2_fusion_result", sha256=result_hash),),
        )

    def _run_phase3(
        self,
        config: RejuvenationWorkflowConfig,
        inputs: RejuvenationWorkflowInputs,
    ) -> tuple[Phase3WorkflowResult | None, PhaseDisposition]:
        if config.phase3 is None:
            return None, self._skipped_disposition(WorkflowPhase.PHASE3)
        phase_input = inputs.phase3
        if phase_input is None:
            raise RuntimeError("validated Phase 3 input is unexpectedly absent")
        state_study = phase_input.to_study()
        estimator = LinearGaussianStateEstimator(config.phase3.state_model).fit(state_study)
        state_report = estimator.estimate_report(state_study)
        input_hash = phase_input.artifact_hash
        derived_hash = study_artifact_hash(state_study)
        config_hash = _canonical_hash(config.phase3.state_model.model_dump(mode="json"))
        result_hash = state_report.artifact_hash
        result = Phase3WorkflowResult(
            timed_evidence_artifact_hash=input_hash,
            derived_study_artifact_hash=derived_hash,
            config_artifact_hash=config_hash,
            result_artifact_hash=result_hash,
            state_estimation=state_report,
        )
        return result, PhaseDisposition(
            phase=WorkflowPhase.PHASE3,
            status=PhaseExecutionStatus.COMPLETED,
            reason="prespecified_state_estimation_completed",
            input_artifacts=(
                ArtifactIdentity(name="timed_evidence_batch", sha256=input_hash),
                ArtifactIdentity(name="phase3_derived_study", sha256=derived_hash),
                ArtifactIdentity(name="phase3_config", sha256=config_hash),
            ),
            output_artifacts=(ArtifactIdentity(name="phase3_state_report", sha256=result_hash),),
        )

    def _run_phase4(
        self,
        study: Study,
        config: RejuvenationWorkflowConfig,
        inputs: RejuvenationWorkflowInputs,
    ) -> tuple[Phase4WorkflowResult | None, PhaseDisposition]:
        if config.phase4 is None:
            return None, self._skipped_disposition(WorkflowPhase.PHASE4)
        phase_input = inputs.phase4
        if phase_input is None:
            raise RuntimeError("validated Phase 4 input is unexpectedly absent")
        analysis = FactorialCombinationAnalysis(config.phase4.factorial).analyze(
            study,
            endpoints=phase_input,
            outcome=config.phase4.outcome,
        )
        endpoint_hash = phase_input.artifact_hash
        config_hash = _canonical_hash(config.phase4.model_dump(mode="json"))
        result = Phase4WorkflowResult(
            endpoint_batch_artifact_hash=endpoint_hash,
            endpoint_source_artifact_hash=phase_input.source_artifact_hash,
            config_artifact_hash=config_hash,
            result_artifact_hash=analysis.artifact_hash,
            combination_analysis=analysis,
        )
        return result, PhaseDisposition(
            phase=WorkflowPhase.PHASE4,
            status=PhaseExecutionStatus.COMPLETED,
            reason="explicit_subject_endpoint_factorial_analysis_completed",
            input_artifacts=(
                ArtifactIdentity(name="subject_endpoint_batch", sha256=endpoint_hash),
                ArtifactIdentity(
                    name="endpoint_source_artifact",
                    sha256=phase_input.source_artifact_hash,
                ),
                ArtifactIdentity(name="phase4_config", sha256=config_hash),
            ),
            output_artifacts=(
                ArtifactIdentity(
                    name="phase4_combination_report",
                    sha256=analysis.artifact_hash,
                ),
            ),
        )

    def _blocked_or_skipped_disposition(
        self,
        phase: WorkflowPhase,
        configured: object | None,
        supplied_input: object | None,
    ) -> PhaseDisposition:
        if configured is None:
            return self._skipped_disposition(phase)
        if supplied_input is None:
            raise RuntimeError(f"validated {phase.value} input is unexpectedly absent")
        return PhaseDisposition(
            phase=phase,
            status=PhaseExecutionStatus.BLOCKED,
            reason="phase1_qc_error_without_serialized_override",
            input_artifacts=self._blocked_input_artifacts(phase, configured, supplied_input),
        )

    @staticmethod
    def _blocked_input_artifacts(
        phase: WorkflowPhase,
        configured: object,
        supplied_input: object,
    ) -> tuple[ArtifactIdentity, ...]:
        if isinstance(configured, Phase3WorkflowConfig):
            config_payload: object = configured.state_model.model_dump(mode="json")
        else:
            config_payload = (
                configured.model_dump(mode="json")
                if isinstance(configured, BaseModel)
                else configured
            )
        identities = [
            ArtifactIdentity(
                name=f"{phase.value}_config",
                sha256=_canonical_hash(config_payload),
            )
        ]
        if isinstance(supplied_input, Phase2FusionInput):
            identities.extend(
                (
                    ArtifactIdentity(
                        name="phase2_gls_input",
                        sha256=supplied_input.artifact_hash,
                    ),
                    ArtifactIdentity(
                        name="phase2_covariance",
                        sha256=supplied_input.covariance_artifact_hash,
                    ),
                )
            )
        elif isinstance(supplied_input, TimedEvidenceBatch):
            identities.append(
                ArtifactIdentity(
                    name="timed_evidence_batch",
                    sha256=supplied_input.artifact_hash,
                )
            )
        elif isinstance(supplied_input, SubjectEndpointBatch):
            identities.extend(
                (
                    ArtifactIdentity(
                        name="subject_endpoint_batch",
                        sha256=supplied_input.artifact_hash,
                    ),
                    ArtifactIdentity(
                        name="endpoint_source_artifact",
                        sha256=supplied_input.source_artifact_hash,
                    ),
                )
            )
        else:
            input_payload = (
                supplied_input.model_dump(mode="json")
                if isinstance(supplied_input, BaseModel)
                else supplied_input
            )
            identities.append(
                ArtifactIdentity(
                    name=f"{phase.value}_input",
                    sha256=_canonical_hash(input_payload),
                )
            )
        return tuple(identities)

    @staticmethod
    def _skipped_disposition(phase: WorkflowPhase) -> PhaseDisposition:
        return PhaseDisposition(
            phase=phase,
            status=PhaseExecutionStatus.SKIPPED,
            reason="not_configured",
        )


def run_rejuvenation_workflow(
    request: RejuvenationWorkflowRequest,
    *,
    output_dir: Path,
) -> RejuvenationWorkflowReport:
    """Execute and publish one validated RejuvenationKit workflow request."""
    return RejuvenationWorkflowRunner().run(request, output_dir=output_dir)


def load_rejuvenation_workflow_report(output_dir: Path) -> RejuvenationWorkflowReport:
    """Verify checksums and load a previously published workflow report."""
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("workflow output path must be a regular, non-symlink directory")
    _reject_symlink_components(output_dir, _MANIFEST_NAME, context="workflow loader")
    _reject_symlink_components(output_dir, _REPORT_NAME, context="workflow loader")
    manifest_path = output_dir / _MANIFEST_NAME
    report_path = output_dir / _REPORT_NAME
    if not manifest_path.is_file() or not report_path.is_file():
        raise ValueError("workflow bundle is missing its report or manifest")
    manifest = WorkflowManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest.artifacts:
        _reject_symlink_components(output_dir, artifact.path, context="workflow loader")
        path = output_dir / artifact.path
        if not path.is_file():
            raise ValueError(f"workflow artifact is missing: {artifact.path}")
        if path.stat().st_size != artifact.bytes or _file_hash(path) != artifact.sha256:
            raise ValueError(f"workflow artifact checksum mismatch: {artifact.path}")
    report = RejuvenationWorkflowReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    expected_manifest_paths = set(report.artifacts).difference({_MANIFEST_NAME})
    observed_manifest_paths = {item.path for item in manifest.artifacts}
    if observed_manifest_paths != expected_manifest_paths:
        raise ValueError("workflow report and manifest artifact inventories disagree")
    if manifest.study_id != report.study_id:
        raise ValueError("workflow manifest and report study identifiers disagree")
    if manifest.study_artifact_hash != report.study_artifact_hash:
        raise ValueError("workflow manifest and report study artifacts disagree")
    if manifest.software_version != report.software_version:
        raise ValueError("workflow manifest and report software versions disagree")
    return report


def _build_manifest(
    report: RejuvenationWorkflowReport,
    staging: Path,
) -> WorkflowManifest:
    expected = set(report.artifacts).difference({_MANIFEST_NAME})
    actual = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and path.name != _MANIFEST_NAME
    }
    if actual != expected:
        raise RuntimeError(
            "staged workflow artifact mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    artifacts = tuple(
        WorkflowManifestArtifact(
            path=name,
            bytes=(staging / name).stat().st_size,
            sha256=_file_hash(staging / name),
        )
        for name in sorted(actual)
    )
    report_sha = next(item.sha256 for item in artifacts if item.path == _REPORT_NAME)
    return WorkflowManifest(
        software_version=report.software_version,
        study_id=report.study_id,
        study_artifact_hash=report.study_artifact_hash,
        workflow_report_sha256=report_sha,
        artifacts=artifacts,
    )


def _read_previous_manifest(output_dir: Path) -> WorkflowManifest | None:
    if output_dir.is_symlink():
        raise ValueError("workflow output directory cannot be a symbolic link")
    _reject_symlink_components(output_dir, _MANIFEST_NAME, context="workflow publisher")
    _reject_symlink_components(output_dir, _REPORT_NAME, context="workflow publisher")
    manifest_path = output_dir / _MANIFEST_NAME
    report_path = output_dir / _REPORT_NAME
    if not manifest_path.is_file() or not report_path.is_file():
        return None
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = WorkflowManifest.model_validate_json(manifest_text)
        report = RejuvenationWorkflowReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if manifest_text != manifest.model_dump_json(indent=2) + "\n":
        raise ValueError("refusing to overwrite modified workflow manifest")
    declared = set(report.artifacts).difference({_MANIFEST_NAME})
    managed = {item.path for item in manifest.artifacts}
    if (
        declared != managed
        or manifest.study_id != report.study_id
        or manifest.study_artifact_hash != report.study_artifact_hash
        or manifest.software_version != report.software_version
    ):
        return None
    return manifest


def _publish_staged_bundle(staging: Path, output_dir: Path) -> None:
    """Publish staged files, protecting unrelated and locally modified files."""
    if output_dir.is_symlink():
        raise ValueError("workflow output directory cannot be a symbolic link")
    staged_files = {
        path.relative_to(staging).as_posix(): path for path in staging.rglob("*") if path.is_file()
    }
    for relative in sorted(staged_files):
        _reject_symlink_components(output_dir, relative, context="workflow publisher")
    previous_manifest = _read_previous_manifest(output_dir)
    previous = (
        {item.path: item for item in previous_manifest.artifacts}
        if previous_manifest is not None
        else {}
    )
    if previous_manifest is not None:
        previous[_MANIFEST_NAME] = WorkflowManifestArtifact(
            path=_MANIFEST_NAME,
            bytes=(output_dir / _MANIFEST_NAME).stat().st_size,
            sha256=_file_hash(output_dir / _MANIFEST_NAME),
        )
    for relative in sorted(previous):
        _reject_symlink_components(output_dir, relative, context="workflow publisher")

    for relative in sorted(staged_files):
        target = output_dir / relative
        if target.exists():
            managed = previous.get(relative)
            if managed is None:
                raise ValueError(f"refusing to overwrite unrelated workflow path: {relative}")
            if not target.is_file():
                raise ValueError(f"workflow artifact target is not a file: {relative}")
            if target.stat().st_size != managed.bytes or _file_hash(target) != managed.sha256:
                raise ValueError(f"refusing to overwrite modified workflow artifact: {relative}")
        parent = target.parent
        while parent != output_dir.parent and parent != output_dir:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"workflow artifact parent is not a directory: {parent}")
            parent = parent.parent

    stale = set(previous).difference(staged_files)
    for relative in sorted(stale):
        target = output_dir / relative
        if not target.exists():
            continue
        managed = previous[relative]
        if not target.is_file():
            raise ValueError(f"managed workflow artifact is not a file: {relative}")
        if target.stat().st_size != managed.bytes or _file_hash(target) != managed.sha256:
            raise ValueError(f"refusing to delete modified workflow artifact: {relative}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for relative, source in sorted(staged_files.items()):
        if relative == _MANIFEST_NAME:
            continue
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
    for relative in sorted(stale):
        target = output_dir / relative
        if target.is_file():
            target.unlink()
    manifest_source = staged_files[_MANIFEST_NAME]
    manifest_source.replace(output_dir / _MANIFEST_NAME)
