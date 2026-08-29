from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from rejuvenationkit.audit import (
    ChangeDetectionAuditPlan,
    Phase1AuditConfig,
    Phase1AuditReport,
    SequentialDetectionAuditPlan,
    TreatmentAuditPlan,
    run_phase1_audit,
)
from rejuvenationkit.combinations import AssignmentMechanism
from rejuvenationkit.detection import ChangeDetectionConfig
from rejuvenationkit.qc import ExpectedVisit, FeatureRule, QCConfig, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject
from rejuvenationkit.sequential import SequentialDetectionConfig
from rejuvenationkit.treatment_effect import TreatmentEffectConfig

START = datetime(2026, 1, 1, tzinfo=UTC)
FEATURES = (
    VisitFeature(feature="albumin", modality=Modality.CLINICAL),
    VisitFeature(feature="crp", modality=Modality.CLINICAL),
)
VISITS = (
    ExpectedVisit(
        visit_id="baseline",
        scheduled_at=START,
        required_features=FEATURES,
    ),
    ExpectedVisit(
        visit_id="month-1",
        scheduled_at=START + timedelta(days=30),
        required_features=FEATURES,
    ),
)


def study() -> tuple[Study, tuple[str, ...], tuple[str, ...]]:
    treated_ids = tuple(f"treated-{index}" for index in range(6))
    control_ids = tuple(f"control-{index}" for index in range(6))
    subjects = tuple(
        Subject(
            subject_id=subject_id,
            cohort="treated" if subject_id in treated_ids else "control",
            interventions=("rapamycin",) if subject_id in treated_ids else (),
        )
        for subject_id in (*treated_ids, *control_ids)
    )
    observations: list[Observation] = []
    for index, subject_id in enumerate((*treated_ids, *control_ids)):
        treated = subject_id in treated_ids
        for day, fraction in ((0, 0), (30, 1)):
            values = (
                3.0 + index * 0.01 + (0.3 if treated else 0.0) * fraction,
                2.0 + index * 0.02 + (-0.8 if treated else 0.0) * fraction,
            )
            for feature, value in zip(FEATURES, values, strict=True):
                observations.append(
                    Observation(
                        subject_id=subject_id,
                        timestamp=START + timedelta(days=day),
                        modality=Modality.CLINICAL,
                        feature=feature.feature,
                        value=value,
                        unit="normalized",
                    )
                )
    return (
        Study(study_id="audit-study", subjects=subjects, observations=tuple(observations)),
        treated_ids,
        control_ids,
    )


def audit_config(
    *,
    visualizations: bool = True,
    allow_analysis_with_qc_errors: bool = False,
) -> Phase1AuditConfig:
    return Phase1AuditConfig(
        qc=QCConfig(
            feature_rules=tuple(
                FeatureRule(
                    feature=item.feature,
                    modality=item.modality,
                    expected_unit="normalized",
                    minimum=0,
                    required=True,
                )
                for item in FEATURES
            ),
            expected_visits=VISITS,
        ),
        include_visualizations=visualizations,
        allow_analysis_with_qc_errors=allow_analysis_with_qc_errors,
    )


def change_detection_plan(
    reference_ids: tuple[str, ...],
    evaluation_ids: tuple[str, ...],
) -> ChangeDetectionAuditPlan:
    return ChangeDetectionAuditPlan(
        config=ChangeDetectionConfig(
            features=FEATURES,
            minimum_reference_subjects=3,
        ),
        baseline_visit_id="baseline",
        follow_up_visit_id="month-1",
        reference_subject_ids=reference_ids,
        evaluation_subject_ids=evaluation_ids,
    )


def test_one_command_audit_writes_complete_observational_bundle(tmp_path: Path) -> None:
    source, treated_ids, control_ids = study()
    report = run_phase1_audit(
        source,
        config=audit_config(),
        output_dir=tmp_path,
        change_detection_plan=change_detection_plan(control_ids, treated_ids),
    )

    assert report.passed
    assert report.treatment_effect is None
    assert report.change_detection is not None
    assert all(item.detected for item in report.change_detection.results)
    assert set(report.artifacts) == {path.name for path in tmp_path.iterdir()}
    assert (tmp_path / "audit_overview.png").stat().st_size > 0
    assert (tmp_path / "change-detection-covariance.png").stat().st_size > 0
    assert "makes no treatment-effect claim" in report.summary_markdown()
    payload = json.loads((tmp_path / "audit.json").read_text())
    assert payload["study_id"] == source.study_id
    assert payload["schema_version"] == "3"
    assert payload["software_version"]
    assert Phase1AuditReport.model_validate_json((tmp_path / "audit.json").read_text()) == report
    with pytest.raises(TypeError):
        report.study_metadata["forged"] = True  # type: ignore[index]
    assert (tmp_path / "findings.csv").read_text().startswith("code,severity,message")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == "3"
    assert manifest["software_version"] == report.software_version
    assert len(manifest["artifacts"]) == len(report.artifacts) - 1
    for artifact in manifest["artifacts"]:
        path = tmp_path / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert len(artifact["sha256"]) == 64


def test_audit_includes_prespecified_randomized_inference(tmp_path: Path) -> None:
    source, treated_ids, control_ids = study()
    plan = TreatmentAuditPlan(
        config=TreatmentEffectConfig(
            features=FEATURES,
            assignment_mechanism=AssignmentMechanism.RANDOMIZED,
            cross_validation_folds=2,
            permutations=99,
            bootstrap_samples=99,
            minimum_group_size=3,
            random_seed=2,
        ),
        baseline_visit_id="baseline",
        follow_up_visit_ids=("month-1",),
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
        treated_label="rapamycin",
        control_label="placebo",
    )
    report = run_phase1_audit(
        source,
        config=audit_config(visualizations=False),
        output_dir=tmp_path,
        treatment_plan=plan,
    )

    assert report.treatment_effect is not None
    assert report.treatment_effect.visit_effects[0].permutation_p_value <= 0.05
    assert (tmp_path / "treatment_effects.csv").stat().st_size > 0
    assert (tmp_path / "treatment_subject_scores.csv").stat().st_size > 0
    assert "audit_overview.png" not in report.artifacts


def test_audit_rejects_unknown_treatment_visit(tmp_path: Path) -> None:
    source, treated_ids, control_ids = study()
    plan = TreatmentAuditPlan(
        config=TreatmentEffectConfig(
            features=FEATURES,
            assignment_mechanism=AssignmentMechanism.RANDOMIZED,
            cross_validation_folds=2,
            permutations=99,
            bootstrap_samples=99,
            minimum_group_size=3,
        ),
        baseline_visit_id="baseline",
        follow_up_visit_ids=("missing",),
        treated_subject_ids=treated_ids,
        control_subject_ids=control_ids,
    )

    with pytest.raises(ValueError, match="not configured"):
        run_phase1_audit(
            source,
            config=audit_config(visualizations=False),
            output_dir=tmp_path,
            treatment_plan=plan,
        )


def test_audit_fingerprint_is_canonical(tmp_path: Path) -> None:
    source, _, _ = study()
    first = source.model_copy(update={"metadata": {"b": "two", "a": "one"}})
    second = source.model_copy(update={"metadata": {"a": "one", "b": "two"}})

    first_report = run_phase1_audit(
        first,
        config=audit_config(visualizations=False),
        output_dir=tmp_path / "first",
    )
    second_report = run_phase1_audit(
        second,
        config=audit_config(visualizations=False),
        output_dir=tmp_path / "second",
    )

    assert first_report.input_sha256 == second_report.input_sha256


def sequential_study() -> tuple[
    Study,
    tuple[ExpectedVisit, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    reference_ids = tuple(f"reference-{index}" for index in range(6))
    evaluation_ids = tuple(f"evaluation-{index}" for index in range(3))
    subjects = tuple(
        Subject(subject_id=subject_id, cohort="reference") for subject_id in reference_ids
    ) + tuple(
        Subject(subject_id=subject_id, cohort="treated", interventions=("rapamycin",))
        for subject_id in evaluation_ids
    )
    visits = tuple(
        ExpectedVisit(
            visit_id=visit_id,
            scheduled_at=START + timedelta(days=day),
            required_features=FEATURES,
        )
        for visit_id, day in (("baseline", 0), ("month-1", 30), ("month-2", 60))
    )
    observations: list[Observation] = []
    for subject_index, subject_id in enumerate((*reference_ids, *evaluation_ids)):
        treated = subject_id in evaluation_ids
        for visit_index, day in enumerate((0, 30, 60)):
            treatment_scale = float(visit_index) if treated else 0.0
            values = (
                3.0
                + subject_index * 0.03
                + visit_index * (0.04 + subject_index * 0.002)
                + 0.7 * treatment_scale,
                2.0
                + subject_index * 0.02
                + visit_index * (-0.03 + subject_index * 0.003)
                - 0.8 * treatment_scale,
            )
            observations.extend(
                Observation(
                    subject_id=subject_id,
                    timestamp=START + timedelta(days=day),
                    modality=Modality.CLINICAL,
                    feature=feature.feature,
                    value=value,
                    unit="normalized",
                )
                for feature, value in zip(FEATURES, values, strict=True)
            )
    return (
        Study(study_id="sequential-audit", subjects=subjects, observations=tuple(observations)),
        visits,
        reference_ids,
        evaluation_ids,
    )


def test_audit_runs_prespecified_sequential_detection(tmp_path: Path) -> None:
    source, visits, reference_ids, evaluation_ids = sequential_study()
    plan = SequentialDetectionAuditPlan(
        config=SequentialDetectionConfig(
            features=FEATURES,
            minimum_reference_subjects=5,
        ),
        visit_ids=tuple(visit.visit_id for visit in visits),
        reference_subject_ids=reference_ids,
        evaluation_subject_ids=evaluation_ids,
    )
    config = Phase1AuditConfig(
        qc=QCConfig(
            feature_rules=tuple(
                FeatureRule(
                    feature=item.feature,
                    modality=item.modality,
                    expected_unit="normalized",
                    minimum=0,
                    required=True,
                )
                for item in FEATURES
            ),
            expected_visits=visits,
        )
    )

    report = run_phase1_audit(
        source,
        config=config,
        output_dir=tmp_path,
        sequential_detection_plan=plan,
    )

    assert report.sequential_detection is not None
    assert report.sequential_detection.results
    assert report.sequential_detection_plan == plan
    assert (tmp_path / "sequential_detection_results.csv").stat().st_size > 0
    assert (tmp_path / "sequential_detection_trajectories.csv").stat().st_size > 0
    assert (tmp_path / "sequential-detection-trajectories.png").stat().st_size > 0
    assert "Held-out sequential change detection" in report.summary_markdown()

    forged_elapsed = report.model_dump(mode="python")
    forged_elapsed["sequential_detection"]["results"][0]["points"][0]["elapsed_years"] += 1
    with pytest.raises(ValidationError, match="elapsed time"):
        Phase1AuditReport.model_validate(forged_elapsed)

    forged_missing = report.model_dump(mode="python")
    forged_missing["sequential_detection"]["results"][0]["missing_visit_ids"] = (
        visits[0].visit_id,
    )
    with pytest.raises(ValidationError, match="missing visits"):
        Phase1AuditReport.model_validate(forged_missing)

    forged_classification = report.model_dump(mode="python")
    result = forged_classification["sequential_detection"]["results"][0]
    result["persistent"] = not result["persistent"]
    result["transient"] = False
    with pytest.raises(ValidationError, match="classification"):
        Phase1AuditReport.model_validate(forged_classification)


def test_audit_blocks_analysis_on_qc_error_without_override(tmp_path: Path) -> None:
    source, treated_ids, control_ids = study()
    invalid = source.model_copy(
        update={
            "observations": (
                source.observations[0].model_copy(update={"value": -1.0}),
                *source.observations[1:],
            )
        }
    )
    plan = change_detection_plan(control_ids, treated_ids)

    blocked = run_phase1_audit(
        invalid,
        config=audit_config(visualizations=False),
        output_dir=tmp_path / "blocked",
        change_detection_plan=plan,
    )
    overridden = run_phase1_audit(
        invalid,
        config=audit_config(
            visualizations=False,
            allow_analysis_with_qc_errors=True,
        ),
        output_dir=tmp_path / "overridden",
        change_detection_plan=plan,
    )

    assert blocked.analysis_blocked_by_qc
    assert blocked.change_detection is None
    assert "change_detection_scores.csv" not in blocked.artifacts
    assert "not run because QC contains errors" in blocked.summary_markdown()
    assert overridden.analysis_override_applied
    assert overridden.change_detection is not None
    assert "QC override applied" in overridden.summary_markdown()


def test_serialized_audit_reconstructs_gate_identity_results_and_artifacts(
    tmp_path: Path,
) -> None:
    source, treated_ids, control_ids = study()
    report = run_phase1_audit(
        source,
        config=audit_config(visualizations=False),
        output_dir=tmp_path / "valid",
        change_detection_plan=change_detection_plan(control_ids, treated_ids),
    )

    mismatched_identity = report.model_dump(mode="python")
    mismatched_identity["study_id"] = "forged-study"
    with pytest.raises(ValidationError, match="study identities"):
        Phase1AuditReport.model_validate(mismatched_identity)

    legacy_schema = report.model_dump(mode="python")
    legacy_schema["schema_version"] = "2"
    with pytest.raises(ValidationError, match="unsupported Phase 1 audit schema"):
        Phase1AuditReport.model_validate(legacy_schema)

    missing_result = report.model_dump(mode="python")
    missing_result["change_detection"] = None
    with pytest.raises(ValidationError, match="plan/result disposition"):
        Phase1AuditReport.model_validate(missing_result)

    forged_score = report.model_dump(mode="python")
    forged_score["change_detection"]["results"][0]["squared_mahalanobis_distance"] += 1
    with pytest.raises(ValidationError, match="score does not match"):
        Phase1AuditReport.model_validate(forged_score)

    forged_exclusions = report.model_dump(mode="python")
    forged_exclusions["longitudinal_exclusion_counts"] = (
        {"analysis": "change_detection", "reason": "invented", "events": 1},
    )
    with pytest.raises(ValidationError, match="exclusion counts"):
        Phase1AuditReport.model_validate(forged_exclusions)

    forged_artifacts = report.model_dump(mode="python")
    forged_artifacts["artifacts"] = tuple(
        name for name in report.artifacts if name != "change_detection_scores.csv"
    )
    with pytest.raises(ValidationError, match="artifact names"):
        Phase1AuditReport.model_validate(forged_artifacts)


def test_serialized_audit_cannot_suppress_a_required_qc_block(tmp_path: Path) -> None:
    source, treated_ids, control_ids = study()
    invalid = source.model_copy(
        update={
            "observations": (
                source.observations[0].model_copy(update={"value": -1.0}),
                *source.observations[1:],
            )
        }
    )
    report = run_phase1_audit(
        invalid,
        config=audit_config(visualizations=False),
        output_dir=tmp_path,
        change_detection_plan=change_detection_plan(control_ids, treated_ids),
    )
    forged = report.model_dump(mode="python")
    forged["analysis_blocked_by_qc"] = False

    with pytest.raises(ValidationError, match="serialized QC gate"):
        Phase1AuditReport.model_validate(forged)


def test_audit_exports_structured_longitudinal_exclusions(tmp_path: Path) -> None:
    source, treated_ids, control_ids = study()
    incomplete = source.model_copy(
        update={
            "observations": tuple(
                row
                for row in source.observations
                if not (
                    row.subject_id == treated_ids[0] and row.timestamp == START + timedelta(days=30)
                )
            )
        }
    )

    report = run_phase1_audit(
        incomplete,
        config=audit_config(visualizations=False),
        output_dir=tmp_path,
        change_detection_plan=change_detection_plan(control_ids, treated_ids),
    )

    assert report.change_detection is not None
    assert treated_ids[0] in report.change_detection.excluded_subject_ids
    assert report.longitudinal_exclusion_counts
    contents = (tmp_path / "longitudinal_exclusions.csv").read_text()
    assert "analysis,reason,subject_id" in contents
    assert "profiling" in contents
    assert "change_detection" in contents
    assert treated_ids[0] in contents
    profiling_counts = tuple(
        item for item in report.longitudinal_exclusion_counts if item.analysis == "profiling"
    )
    assert profiling_counts
    assert sum(item.events for item in profiling_counts) == len(
        set(report.profile.longitudinal_exclusions)
    )


def test_audit_removes_only_stale_manifest_managed_artifacts(tmp_path: Path) -> None:
    source, treated_ids, control_ids = study()
    run_phase1_audit(
        source,
        config=audit_config(),
        output_dir=tmp_path,
        change_detection_plan=change_detection_plan(control_ids, treated_ids),
    )
    unrelated = tmp_path / "researcher-notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    report = run_phase1_audit(
        source,
        config=audit_config(visualizations=False),
        output_dir=tmp_path,
    )

    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert not (tmp_path / "change-detection-scores.png").exists()
    assert not (tmp_path / "audit_overview.png").exists()
    assert set(report.artifacts).issubset({path.name for path in tmp_path.iterdir()})


def test_audit_rejects_symlink_output_directory(tmp_path: Path) -> None:
    source, _, _ = study()
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        run_phase1_audit(
            source,
            config=audit_config(visualizations=False),
            output_dir=linked,
        )
    assert tuple(actual.iterdir()) == ()


def test_audit_rejects_unsafe_or_tampered_existing_manifest(tmp_path: Path) -> None:
    source, _, _ = study()
    output = tmp_path / "bundle"
    run_phase1_audit(
        source,
        config=audit_config(visualizations=False),
        output_dir=output,
    )
    notes = output / "researcher-notes.txt"
    notes.write_text("preserve", encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contents = notes.read_bytes()
    manifest["artifacts"].append(
        {
            "path": notes.name,
            "bytes": len(contents),
            "sha256": sha256(contents).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe artifact path"):
        run_phase1_audit(
            source,
            config=audit_config(visualizations=False),
            output_dir=output,
        )
    assert notes.read_text(encoding="utf-8") == "preserve"


def test_audit_rejects_symlinked_or_checksum_mismatched_managed_artifact(
    tmp_path: Path,
) -> None:
    source, _, _ = study()
    symlink_output = tmp_path / "symlinked"
    run_phase1_audit(
        source,
        config=audit_config(visualizations=False),
        output_dir=symlink_output,
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    summary = symlink_output / "summary.md"
    summary.unlink()
    summary.symlink_to(outside)
    with pytest.raises(ValueError, match="symlinked managed artifacts"):
        run_phase1_audit(
            source,
            config=audit_config(visualizations=False),
            output_dir=symlink_output,
        )
    assert outside.read_text(encoding="utf-8") == "outside"

    tampered_output = tmp_path / "tampered"
    run_phase1_audit(
        source,
        config=audit_config(visualizations=False),
        output_dir=tampered_output,
    )
    (tampered_output / "summary.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its manifest"):
        run_phase1_audit(
            source,
            config=audit_config(visualizations=False),
            output_dir=tampered_output,
        )
