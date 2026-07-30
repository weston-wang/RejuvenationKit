from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rejuvenationkit.audit import (
    ChangeDetectionAuditPlan,
    Phase1AuditConfig,
    TreatmentAuditPlan,
    run_phase1_audit,
)
from rejuvenationkit.detection import ChangeDetectionConfig
from rejuvenationkit.qc import ExpectedVisit, FeatureRule, QCConfig, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject
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


def audit_config(*, visualizations: bool = True) -> Phase1AuditConfig:
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
    assert (tmp_path / "findings.csv").read_text().startswith("code,severity,message")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
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
