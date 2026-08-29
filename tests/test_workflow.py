from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from itertools import product
from pathlib import Path

import pytest
from pydantic import ValidationError

from rejuvenationkit.audit import Phase1AuditConfig
from rejuvenationkit.bridges import TimedEvidenceBatch, TimedEvidenceMeasurement
from rejuvenationkit.combinations import AssignmentMechanism, FactorialCombinationConfig
from rejuvenationkit.endpoints import SubjectEndpoint, SubjectEndpointBatch, study_artifact_hash
from rejuvenationkit.evidence import (
    CalibrationReference,
    CalibrationValidationStatus,
    EffectDirection,
    Estimand,
    EvidenceCovariance,
    EvidenceEstimate,
    EvidenceFusionConfig,
)
from rejuvenationkit.qc import ExpectedVisit, FeatureRule, QCConfig, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject
from rejuvenationkit.state import LinearGaussianStateConfig, StateChannel
from rejuvenationkit.workflow import (
    Phase1WorkflowConfig,
    Phase2FusionInput,
    Phase2WorkflowConfig,
    Phase3WorkflowConfig,
    Phase4WorkflowConfig,
    PhaseExecutionStatus,
    RejuvenationWorkflowConfig,
    RejuvenationWorkflowInputs,
    RejuvenationWorkflowReport,
    RejuvenationWorkflowRequest,
    load_rejuvenation_workflow_report,
    run_rejuvenation_workflow,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
FOLLOW_UP = START + timedelta(days=180)
STATE_ESTIMAND = Estimand(
    name="calibrated_health_signal",
    unit="z",
    direction=EffectDirection.HIGHER_IS_BETTER,
    population="synthetic-factorial-dogs",
)
ENDPOINT_ESTIMAND = Estimand(
    name="frailty_followup",
    unit="score",
    direction=EffectDirection.LOWER_IS_BETTER,
    population="synthetic-factorial-dogs",
    time_contrast="month-6 adjusted for baseline",
)


def _subjects() -> tuple[Subject, ...]:
    result: list[Subject] = []
    for bits in product((0, 1), repeat=2):
        interventions = tuple(
            name for name, enabled in zip(("rapamycin", "senolytic"), bits, strict=True) if enabled
        )
        for index in range(4):
            result.append(
                Subject(
                    subject_id=f"dog-{bits[0]}{bits[1]}-{index}",
                    cohort="randomized",
                    interventions=interventions,
                )
            )
    return tuple(result)


def _study() -> Study:
    subjects = _subjects()
    observations: list[Observation] = []
    for subject_index, subject in enumerate(subjects):
        baseline = 0.05 * (subject_index % 4)
        treated = float("rapamycin" in subject.interventions)
        combined = float({"rapamycin", "senolytic"}.issubset(subject.interventions))
        follow_up = baseline + 0.4 * treated + 0.7 * combined
        for timestamp, value in ((START, baseline), (FOLLOW_UP, follow_up)):
            observations.append(
                Observation(
                    subject_id=subject.subject_id,
                    timestamp=timestamp,
                    modality=Modality.CLINICAL,
                    feature="score",
                    value=value,
                    unit="z",
                )
            )
    return Study(
        study_id="workflow-study",
        subjects=subjects,
        observations=tuple(observations),
    )


def _phase1_config(
    *,
    include_visualizations: bool = False,
    allow_analysis_with_qc_errors: bool = False,
    missing_required_feature: bool = False,
) -> Phase1WorkflowConfig:
    feature = "missing" if missing_required_feature else "score"
    modality = Modality.PROTEOMICS if missing_required_feature else Modality.CLINICAL
    required = (VisitFeature(feature=feature, modality=modality),)
    visits = (
        ExpectedVisit(
            visit_id="baseline",
            scheduled_at=START,
            required_features=required,
        ),
        ExpectedVisit(
            visit_id="month-6",
            scheduled_at=FOLLOW_UP,
            required_features=required,
        ),
    )
    return Phase1WorkflowConfig(
        audit=Phase1AuditConfig(
            qc=QCConfig(
                feature_rules=(
                    FeatureRule(
                        feature=feature,
                        modality=modality,
                        expected_unit="z",
                        required=True,
                    ),
                ),
                expected_visits=visits,
            ),
            include_visualizations=include_visualizations,
            allow_analysis_with_qc_errors=allow_analysis_with_qc_errors,
        )
    )


def _calibration(
    calibration_id: str,
    estimand: Estimand,
) -> CalibrationReference:
    return CalibrationReference(
        calibration_id=calibration_id,
        artifact_hash=sha256(calibration_id.encode()).hexdigest(),
        status=CalibrationValidationStatus.HELD_OUT_VALIDATED,
        estimand=estimand,
        method="prespecified held-out calibration",
        validation_provenance_id=f"validation:{calibration_id}",
    )


def _phase2_input() -> Phase2FusionInput:
    estimand = Estimand(
        name="cohort_rejuvenation",
        unit="z",
        direction=EffectDirection.HIGHER_IS_BETTER,
        population="synthetic-factorial-dogs",
        time_contrast="month-6-minus-baseline",
    )
    estimates = (
        EvidenceEstimate(
            evidence_id="methylation-clock",
            modality=Modality.METHYLATION,
            estimand=estimand,
            estimate=0.8,
            standard_error=0.2,
            calibration_id="meth-cal-v1",
            calibration_reference=_calibration("meth-cal-v1", estimand),
            provenance_id="analysis:methylation:v1",
            species_taxon_id=9615,
            tissue="blood",
            correlation_group="shared-cohort",
        ),
        EvidenceEstimate(
            evidence_id="proteomic-clock",
            modality=Modality.PROTEOMICS,
            estimand=estimand,
            estimate=1.1,
            standard_error=0.3,
            calibration_id="protein-cal-v1",
            calibration_reference=_calibration("protein-cal-v1", estimand),
            provenance_id="analysis:proteomics:v1",
            species_taxon_id=9615,
            tissue="blood",
            correlation_group="shared-cohort",
        ),
    )
    covariance = EvidenceCovariance(
        evidence_ids=tuple(item.evidence_id for item in estimates),
        covariance=((0.04, 0.01), (0.01, 0.09)),
        source_id="held-out-bootstrap:v1",
        effective_sample_size=80,
    )
    return Phase2FusionInput(
        input_id="cohort-fusion-v1",
        estimates=estimates,
        covariance=covariance,
    )


def _phase3_input(study: Study) -> TimedEvidenceBatch:
    calibration = _calibration("state-channel-v1", STATE_ESTIMAND)
    measurements = tuple(
        TimedEvidenceMeasurement(
            evidence=EvidenceEstimate(
                evidence_id=f"state:{row.subject_id}:{row.timestamp.isoformat()}",
                modality=Modality.CLINICAL,
                estimand=STATE_ESTIMAND,
                estimate=row.value,
                standard_error=0.1,
                calibration_id=calibration.calibration_id,
                calibration_reference=calibration,
                provenance_id=f"clinical-assay:{row.subject_id}",
                subject_id=row.subject_id,
                species_taxon_id=9615,
                tissue="blood",
            ),
            timestamp=row.timestamp,
            feature="state_score",
        )
        for row in study.observations
    )
    return TimedEvidenceBatch(
        study_id=study.study_id,
        subjects=study.subjects,
        measurements=measurements,
    )


def _phase3_config() -> Phase3WorkflowConfig:
    return Phase3WorkflowConfig(
        state_model=LinearGaussianStateConfig(
            state_names=("latent_health",),
            channels=(
                StateChannel(
                    name="calibrated-clinical-score",
                    modality=Modality.CLINICAL,
                    feature="state_score",
                    unit="z",
                    loadings=(1.0,),
                    measurement_variance=0.2,
                ),
            ),
            continuous_dynamics=((0.0,),),
            continuous_process_covariance=((0.01,),),
            continuous_drift=(0.0,),
            initial_mean=(0.0,),
            initial_covariance=((1.0,),),
            time_unit_days=180.0,
            smooth=True,
        )
    )


def _phase4_input(study: Study) -> SubjectEndpointBatch:
    endpoints: list[SubjectEndpoint] = []
    for subject_index, subject in enumerate(study.subjects):
        rapamycin = float("rapamycin" in subject.interventions)
        senolytic = float("senolytic" in subject.interventions)
        noise = 0.08 * ((subject_index % 4) - 1.5)
        endpoint = 3.0 - 0.5 * rapamycin - 0.3 * senolytic - rapamycin * senolytic + noise
        endpoints.append(
            SubjectEndpoint(
                subject_id=subject.subject_id,
                estimand=ENDPOINT_ESTIMAND,
                estimate=endpoint,
                standard_error=0.15,
                endpoint_timestamp=FOLLOW_UP,
                baseline_timestamp=START,
                baseline_estimate=4.0 + noise,
                provenance_id=f"prespecified-endpoint:{subject.subject_id}",
            )
        )
    return SubjectEndpointBatch(
        study_id=study.study_id,
        batch_id="factorial-endpoints-v1",
        estimand=ENDPOINT_ESTIMAND,
        endpoints=tuple(endpoints),
        source_artifact_hash=study_artifact_hash(study),
    )


def _full_request() -> RejuvenationWorkflowRequest:
    study = _study()
    phase2 = _phase2_input()
    phase3 = _phase3_input(study)
    phase4 = _phase4_input(study)
    return RejuvenationWorkflowRequest(
        config=RejuvenationWorkflowConfig(
            phase1=_phase1_config(),
            phase2=Phase2WorkflowConfig(
                fusion=EvidenceFusionConfig(
                    minimum_evidence=2,
                    expected_evidence_ids=tuple(item.evidence_id for item in phase2.estimates),
                )
            ),
            phase3=_phase3_config(),
            phase4=Phase4WorkflowConfig(
                factorial=FactorialCombinationConfig(
                    interventions=("rapamycin", "senolytic"),
                    assignment_mechanism=AssignmentMechanism.RANDOMIZED,
                    minimum_cell_size=3,
                ),
                outcome=ENDPOINT_ESTIMAND.name,
            ),
        ),
        inputs=RejuvenationWorkflowInputs(
            study=study,
            phase2=phase2,
            phase3=phase3,
            phase4=phase4,
        ),
    )


def _phase1_only_request(*, include_visualizations: bool = False) -> RejuvenationWorkflowRequest:
    return RejuvenationWorkflowRequest(
        config=RejuvenationWorkflowConfig(
            phase1=_phase1_config(include_visualizations=include_visualizations)
        ),
        inputs=RejuvenationWorkflowInputs(study=_study()),
    )


def test_end_to_end_workflow_preserves_explicit_phase_artifacts(tmp_path: Path) -> None:
    request = _full_request()

    report = run_rejuvenation_workflow(request, output_dir=tmp_path)

    assert report.qc_gate.qc_passed
    assert tuple(item.status for item in report.dispositions) == (
        PhaseExecutionStatus.COMPLETED,
        PhaseExecutionStatus.COMPLETED,
        PhaseExecutionStatus.COMPLETED,
        PhaseExecutionStatus.COMPLETED,
    )
    assert report.phase2 is not None
    assert report.phase2.input_artifact_hash == request.inputs.phase2.artifact_hash  # type: ignore[union-attr]
    assert report.phase3 is not None
    assert (
        report.phase3.timed_evidence_artifact_hash == request.inputs.phase3.artifact_hash  # type: ignore[union-attr]
    )
    assert len(report.phase3.state_estimation.trajectories) == len(request.inputs.study.subjects)
    assert report.phase3.result_artifact_hash == report.phase3.state_estimation.artifact_hash
    assert report.dispositions[2].output_artifacts[0].sha256 == (
        report.phase3.state_estimation.artifact_hash
    )
    assert report.phase4 is not None
    assert report.phase4.combination_analysis.study_artifact_hash == report.study_artifact_hash
    assert (
        report.phase4.combination_analysis.endpoint_source_artifact_hash
        == report.phase4.endpoint_source_artifact_hash
    )
    assert (
        report.phase4.endpoint_batch_artifact_hash == request.inputs.phase4.artifact_hash  # type: ignore[union-attr]
    )
    assert report.phase4.combination_analysis.interactions[0].interaction == pytest.approx(-1.0)
    assert (tmp_path / "phase1" / "audit.json").is_file()
    assert not (tmp_path / "audit.json").exists()
    loaded = load_rejuvenation_workflow_report(tmp_path)
    assert loaded == report


def test_workflow_report_rejects_forged_phase_identities(tmp_path: Path) -> None:
    report = run_rejuvenation_workflow(_full_request(), output_dir=tmp_path)
    payload = report.model_dump(mode="python")
    dispositions = list(payload["dispositions"])
    phase2 = dict(dispositions[1])
    outputs = list(phase2["output_artifacts"])
    outputs[0] = {**outputs[0], "sha256": "0" * 64}
    phase2["output_artifacts"] = outputs
    dispositions[1] = phase2
    payload["dispositions"] = dispositions

    with pytest.raises(ValidationError, match="phase2 disposition identities"):
        RejuvenationWorkflowReport.model_validate(payload)

    assert report.phase3 is not None
    phase3_payload = report.phase3.model_dump(mode="python")
    phase3_payload["config_artifact_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="model identity"):
        type(report.phase3).model_validate(phase3_payload)

    changed_phase2_config = report.config.phase2
    assert changed_phase2_config is not None
    changed_phase2_config = changed_phase2_config.model_copy(
        update={"fusion": changed_phase2_config.fusion.model_copy(update={"confidence_level": 0.9})}
    )
    changed_config_payload = report.model_dump(mode="python")
    changed_config_payload["config"] = report.config.model_copy(
        update={"phase2": changed_phase2_config}
    ).model_dump(mode="python")
    with pytest.raises(ValidationError, match="phase2 result config identity"):
        RejuvenationWorkflowReport.model_validate(changed_config_payload)

    changed_gate_payload = report.model_dump(mode="python")
    changed_gate_payload["qc_gate"] = report.qc_gate.model_copy(
        update={"downstream_configured": False}
    ).model_dump(mode="python")
    with pytest.raises(ValidationError, match="QC gate"):
        RejuvenationWorkflowReport.model_validate(changed_gate_payload)

    changed_version_payload = report.model_dump(mode="python")
    changed_version_payload["software_version"] = "forged-version"
    with pytest.raises(ValidationError, match="software versions"):
        RejuvenationWorkflowReport.model_validate(changed_version_payload)

    changed_phase4_study_payload = report.model_dump(mode="python")
    changed_phase4_study_payload["phase4"]["combination_analysis"]["study_artifact_hash"] = "0" * 64
    changed_phase4_study_payload["phase4"]["result_artifact_hash"] = (
        type(report.phase4.combination_analysis)
        .model_validate(changed_phase4_study_payload["phase4"]["combination_analysis"])
        .artifact_hash
    )
    with pytest.raises(ValidationError, match="Phase 4 study artifact identities"):
        RejuvenationWorkflowReport.model_validate(changed_phase4_study_payload)

    endpoint_source_payload = report.model_dump(mode="python")
    endpoint_source_payload["phase4"]["endpoint_source_artifact_hash"] = "e" * 64
    with pytest.raises(ValidationError, match="endpoint source identity"):
        RejuvenationWorkflowReport.model_validate(endpoint_source_payload)


def test_request_rejects_missing_config_or_input_and_implicit_phase_boundaries() -> None:
    study = _study()
    with pytest.raises(ValidationError, match="phase2 configuration and input"):
        RejuvenationWorkflowRequest(
            config=RejuvenationWorkflowConfig(
                phase1=_phase1_config(),
                phase2=Phase2WorkflowConfig(fusion=EvidenceFusionConfig()),
            ),
            inputs=RejuvenationWorkflowInputs(study=study),
        )
    with pytest.raises(ValidationError, match="phase3 configuration and input"):
        RejuvenationWorkflowRequest(
            config=RejuvenationWorkflowConfig(phase1=_phase1_config()),
            inputs=RejuvenationWorkflowInputs(study=study, phase3=_phase3_input(study)),
        )
    with pytest.raises(ValidationError, match="phase4 configuration and input"):
        RejuvenationWorkflowRequest(
            config=RejuvenationWorkflowConfig(
                phase1=_phase1_config(),
                phase4=Phase4WorkflowConfig(
                    factorial=FactorialCombinationConfig(
                        interventions=("rapamycin", "senolytic"),
                        assignment_mechanism=AssignmentMechanism.RANDOMIZED,
                    ),
                    outcome=ENDPOINT_ESTIMAND.name,
                ),
            ),
            inputs=RejuvenationWorkflowInputs(study=study),
        )


def test_qc_errors_block_downstream_unless_serialized_override_is_enabled(
    tmp_path: Path,
) -> None:
    study = _study()
    phase2_input = _phase2_input()
    phase2_config = Phase2WorkflowConfig(fusion=EvidenceFusionConfig(minimum_evidence=2))
    blocked_request = RejuvenationWorkflowRequest(
        config=RejuvenationWorkflowConfig(
            phase1=_phase1_config(missing_required_feature=True),
            phase2=phase2_config,
        ),
        inputs=RejuvenationWorkflowInputs(study=study, phase2=phase2_input),
    )

    blocked = run_rejuvenation_workflow(blocked_request, output_dir=tmp_path / "blocked")

    assert not blocked.qc_gate.qc_passed
    assert not blocked.qc_gate.downstream_allowed
    assert blocked.phase2 is None
    assert blocked.dispositions[1].status is PhaseExecutionStatus.BLOCKED
    assert blocked.dispositions[1].output_artifacts == ()

    override_phase1 = blocked_request.config.phase1.model_copy(
        update={
            "audit": blocked_request.config.phase1.audit.model_copy(
                update={"allow_analysis_with_qc_errors": True}
            )
        }
    )
    override_request = blocked_request.model_copy(
        update={"config": blocked_request.config.model_copy(update={"phase1": override_phase1})}
    )
    serialized = RejuvenationWorkflowRequest.model_validate_json(override_request.model_dump_json())
    allowed = run_rejuvenation_workflow(serialized, output_dir=tmp_path / "allowed")

    assert allowed.qc_gate.serialized_override_enabled
    assert allowed.qc_gate.override_applied
    assert allowed.phase2 is not None
    assert allowed.dispositions[1].status is PhaseExecutionStatus.COMPLETED


def test_rerun_removes_only_stale_managed_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    run_rejuvenation_workflow(
        _phase1_only_request(include_visualizations=True),
        output_dir=output,
    )
    assert (output / "phase1" / "audit_overview.png").is_file()
    (output / "notes.txt").write_text("keep me", encoding="utf-8")
    (output / "phase1" / "research-notes.txt").write_text(
        "also keep me",
        encoding="utf-8",
    )

    report = run_rejuvenation_workflow(
        _phase1_only_request(include_visualizations=False),
        output_dir=output,
    )

    assert not (output / "phase1" / "audit_overview.png").exists()
    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert (output / "phase1" / "research-notes.txt").read_text(encoding="utf-8") == (
        "also keep me"
    )
    assert load_rejuvenation_workflow_report(output) == report


def test_report_manifest_roundtrip_and_unrelated_collision_are_fail_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "roundtrip"
    report = run_rejuvenation_workflow(_phase1_only_request(), output_dir=output)
    payload = json.loads((output / "workflow-manifest.json").read_text(encoding="utf-8"))

    assert (
        RejuvenationWorkflowReport.model_validate_json(
            (output / "workflow.json").read_text(encoding="utf-8")
        )
        == report
    )
    assert {item["path"] for item in payload["artifacts"]} == set(report.artifacts).difference(
        {"workflow-manifest.json"}
    )
    assert load_rejuvenation_workflow_report(output) == report

    modified_manifest_output = tmp_path / "modified-manifest"
    run_rejuvenation_workflow(_phase1_only_request(), output_dir=modified_manifest_output)
    modified_manifest = modified_manifest_output / "workflow-manifest.json"
    manifest_payload = json.loads(modified_manifest.read_text(encoding="utf-8"))
    modified_manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="modified workflow manifest"):
        run_rejuvenation_workflow(
            _phase1_only_request(),
            output_dir=modified_manifest_output,
        )

    (output / "phase1" / "findings.csv").write_text("locally modified", encoding="utf-8")
    with pytest.raises(ValueError, match="modified workflow artifact"):
        run_rejuvenation_workflow(_phase1_only_request(), output_dir=output)
    assert (output / "phase1" / "findings.csv").read_text(encoding="utf-8") == ("locally modified")

    collision = tmp_path / "collision"
    collision.mkdir()
    (collision / "workflow.json").write_text("unrelated", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated workflow path"):
        run_rejuvenation_workflow(_phase1_only_request(), output_dir=collision)
    assert (collision / "workflow.json").read_text(encoding="utf-8") == "unrelated"


def test_workflow_rejects_symlink_roots_managed_ancestors_and_artifacts(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    escaped_bundle = tmp_path / "escaped-bundle"
    escaped_bundle.mkdir()
    (escaped_bundle / "phase1").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        run_rejuvenation_workflow(_phase1_only_request(), output_dir=escaped_bundle)
    assert tuple(external.iterdir()) == ()

    real_bundle = tmp_path / "real-bundle"
    run_rejuvenation_workflow(_phase1_only_request(), output_dir=real_bundle)
    linked_bundle = tmp_path / "linked-bundle"
    linked_bundle.symlink_to(real_bundle, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink directory"):
        load_rejuvenation_workflow_report(linked_bundle)
    with pytest.raises(ValueError, match="symbolic link"):
        run_rejuvenation_workflow(_phase1_only_request(), output_dir=linked_bundle)

    findings = real_bundle / "phase1" / "findings.csv"
    archived_findings = external / "findings.csv"
    archived_findings.write_bytes(findings.read_bytes())
    findings.unlink()
    findings.symlink_to(archived_findings)
    with pytest.raises(ValueError, match="symbolic link"):
        load_rejuvenation_workflow_report(real_bundle)

    manifest_bundle = tmp_path / "manifest-bundle"
    run_rejuvenation_workflow(_phase1_only_request(), output_dir=manifest_bundle)
    manifest = manifest_bundle / "workflow-manifest.json"
    archived_manifest = external / "workflow-manifest.json"
    archived_manifest.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(archived_manifest)
    with pytest.raises(ValueError, match="symbolic link"):
        load_rejuvenation_workflow_report(manifest_bundle)
    with pytest.raises(ValueError, match="symbolic link"):
        run_rejuvenation_workflow(_phase1_only_request(), output_dir=manifest_bundle)
