"""Run all four phases with explicit, synthetic boundary artifacts.

Every value in this example is deterministic and invented. The example shows
workflow orchestration, provenance, QC gating, and bundle verification; it does
not estimate treatment efficacy or validate a biological-age construct.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

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
    RejuvenationWorkflowConfig,
    RejuvenationWorkflowInputs,
    RejuvenationWorkflowReport,
    RejuvenationWorkflowRequest,
    load_rejuvenation_workflow_report,
    run_rejuvenation_workflow,
)

START = datetime(2026, 1, 5, tzinfo=UTC)
FOLLOW_UP = START + timedelta(days=180)
INTERVENTIONS = ("illustrative-treatment-a", "illustrative-treatment-b")

FUSION_ESTIMAND = Estimand(
    name="synthetic_cohort_summary",
    unit="z",
    direction=EffectDirection.HIGHER_IS_BETTER,
    population="synthetic-factorial-dogs",
    time_contrast="month-6-minus-baseline",
)
STATE_ESTIMAND = Estimand(
    name="synthetic_subject_marker",
    unit="z",
    direction=EffectDirection.HIGHER_IS_BETTER,
    population="synthetic-factorial-dogs",
)
ENDPOINT_ESTIMAND = Estimand(
    name="synthetic_followup_score",
    unit="score",
    direction=EffectDirection.LOWER_IS_BETTER,
    population="synthetic-factorial-dogs",
    time_contrast="month-6 adjusted for baseline",
)


def _artifact_hash(label: str) -> str:
    """Create a stable example-only SHA-256 identity."""
    return sha256(label.encode("utf-8")).hexdigest()


def _calibration(label: str, estimand: Estimand) -> CalibrationReference:
    """Declare a synthetic held-out calibration contract."""
    return CalibrationReference(
        calibration_id=label,
        artifact_hash=_artifact_hash(label),
        status=CalibrationValidationStatus.HELD_OUT_VALIDATED,
        estimand=estimand,
        method="synthetic prespecified held-out calibration",
        validation_provenance_id=f"synthetic-validation:{label}",
    )


def build_study() -> Study:
    """Create four balanced factorial cells with two complete visits per subject."""
    subjects: list[Subject] = []
    observations: list[Observation] = []
    for treatment_a, treatment_b in product((0, 1), repeat=2):
        active = tuple(
            name
            for name, enabled in zip(
                INTERVENTIONS,
                (treatment_a, treatment_b),
                strict=True,
            )
            if enabled
        )
        for replicate in range(4):
            subject_id = f"dog-{treatment_a}{treatment_b}-{replicate}"
            subject = Subject(
                subject_id=subject_id,
                cohort="synthetic-randomized",
                interventions=active,
            )
            subjects.append(subject)
            baseline = 0.05 * (replicate - 1.5)
            follow_up = (
                baseline
                + 0.20 * treatment_a
                - 0.10 * treatment_b
                + 0.15 * treatment_a * treatment_b
            )
            for timestamp, value in ((START, baseline), (FOLLOW_UP, follow_up)):
                observations.append(
                    Observation(
                        subject_id=subject_id,
                        timestamp=timestamp,
                        modality=Modality.CLINICAL,
                        feature="synthetic_score",
                        value=value,
                        unit="z",
                    )
                )
    return Study(
        study_id="synthetic-four-phase-workflow",
        subjects=tuple(subjects),
        observations=tuple(observations),
        metadata={"data_status": "fully synthetic; workflow demonstration only"},
    )


def build_phase2_input() -> Phase2FusionInput:
    """Supply two calibrated cohort summaries and their covariance explicitly."""
    methylation_calibration = _calibration("synthetic-methylation-cal-v1", FUSION_ESTIMAND)
    proteomic_calibration = _calibration("synthetic-proteomic-cal-v1", FUSION_ESTIMAND)
    estimates = (
        EvidenceEstimate(
            evidence_id="synthetic-methylation-summary",
            modality=Modality.METHYLATION,
            estimand=FUSION_ESTIMAND,
            estimate=0.25,
            standard_error=0.20,
            calibration_id=methylation_calibration.calibration_id,
            calibration_reference=methylation_calibration,
            provenance_id="synthetic-analysis:methylation:v1",
            species_taxon_id=9615,
            tissue="blood",
            correlation_group="shared-synthetic-cohort",
        ),
        EvidenceEstimate(
            evidence_id="synthetic-proteomic-summary",
            modality=Modality.PROTEOMICS,
            estimand=FUSION_ESTIMAND,
            estimate=0.40,
            standard_error=0.25,
            calibration_id=proteomic_calibration.calibration_id,
            calibration_reference=proteomic_calibration,
            provenance_id="synthetic-analysis:proteomics:v1",
            species_taxon_id=9615,
            tissue="blood",
            correlation_group="shared-synthetic-cohort",
        ),
    )
    return Phase2FusionInput(
        input_id="synthetic-cohort-fusion-v1",
        estimates=estimates,
        covariance=EvidenceCovariance(
            evidence_ids=tuple(item.evidence_id for item in estimates),
            covariance=((0.0400, 0.0100), (0.0100, 0.0625)),
            source_id="synthetic-held-out-bootstrap:v1",
            effective_sample_size=80,
        ),
    )


def build_phase3_input(study: Study) -> TimedEvidenceBatch:
    """Map subject-level calibrated values to explicit timestamps and a state channel."""
    calibration = _calibration("synthetic-state-channel-cal-v1", STATE_ESTIMAND)
    measurements = tuple(
        TimedEvidenceMeasurement(
            evidence=EvidenceEstimate(
                evidence_id=f"synthetic-state:{row.subject_id}:{row.timestamp.isoformat()}",
                modality=Modality.CLINICAL,
                estimand=STATE_ESTIMAND,
                estimate=row.value,
                standard_error=0.10,
                calibration_id=calibration.calibration_id,
                calibration_reference=calibration,
                provenance_id=f"synthetic-state-assay:{row.subject_id}",
                subject_id=row.subject_id,
                species_taxon_id=9615,
                tissue="blood",
            ),
            timestamp=row.timestamp,
            feature="calibrated_state_marker",
        )
        for row in study.observations
    )
    return TimedEvidenceBatch(
        study_id=study.study_id,
        subjects=study.subjects,
        measurements=measurements,
    )


def build_phase4_input(study: Study) -> SubjectEndpointBatch:
    """Supply one prespecified synthetic endpoint per independent subject."""
    endpoints: list[SubjectEndpoint] = []
    for subject in study.subjects:
        replicate = int(subject.subject_id.rsplit("-", maxsplit=1)[1])
        treatment_a = float(INTERVENTIONS[0] in subject.interventions)
        treatment_b = float(INTERVENTIONS[1] in subject.interventions)
        baseline = 4.0 + 0.04 * (replicate - 1.5)
        follow_up = (
            3.8
            + 0.04 * (replicate - 1.5)
            - 0.20 * treatment_a
            - 0.10 * treatment_b
            - 0.25 * treatment_a * treatment_b
        )
        endpoints.append(
            SubjectEndpoint(
                subject_id=subject.subject_id,
                estimand=ENDPOINT_ESTIMAND,
                estimate=follow_up,
                standard_error=0.15,
                endpoint_timestamp=FOLLOW_UP,
                baseline_timestamp=START,
                baseline_estimate=baseline,
                provenance_id=f"synthetic-prespecified-endpoint:{subject.subject_id}",
                quality_flags=("synthetic_example_only",),
            )
        )
    return SubjectEndpointBatch(
        study_id=study.study_id,
        batch_id="synthetic-factorial-endpoints-v1",
        estimand=ENDPOINT_ESTIMAND,
        endpoints=tuple(endpoints),
        source_artifact_hash=study_artifact_hash(study),
    )


def build_request() -> RejuvenationWorkflowRequest:
    """Construct one fully explicit, serializable four-phase request."""
    study = build_study()
    phase2 = build_phase2_input()
    phase3 = build_phase3_input(study)
    phase4 = build_phase4_input(study)
    required_feature = VisitFeature(
        feature="synthetic_score",
        modality=Modality.CLINICAL,
    )
    return RejuvenationWorkflowRequest(
        config=RejuvenationWorkflowConfig(
            phase1=Phase1WorkflowConfig(
                audit=Phase1AuditConfig(
                    qc=QCConfig(
                        feature_rules=(
                            FeatureRule(
                                feature="synthetic_score",
                                modality=Modality.CLINICAL,
                                expected_unit="z",
                                required=True,
                            ),
                        ),
                        expected_visits=(
                            ExpectedVisit(
                                visit_id="baseline",
                                scheduled_at=START,
                                required_features=(required_feature,),
                            ),
                            ExpectedVisit(
                                visit_id="month-6",
                                scheduled_at=FOLLOW_UP,
                                required_features=(required_feature,),
                            ),
                        ),
                    ),
                    include_visualizations=False,
                )
            ),
            phase2=Phase2WorkflowConfig(
                fusion=EvidenceFusionConfig(
                    minimum_evidence=2,
                    expected_evidence_ids=tuple(item.evidence_id for item in phase2.estimates),
                )
            ),
            phase3=Phase3WorkflowConfig(
                state_model=LinearGaussianStateConfig(
                    state_names=("synthetic_latent_state",),
                    channels=(
                        StateChannel(
                            name="synthetic-calibrated-marker",
                            modality=Modality.CLINICAL,
                            feature="calibrated_state_marker",
                            unit="z",
                            loadings=(1.0,),
                            measurement_variance=0.10,
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
            ),
            phase4=Phase4WorkflowConfig(
                factorial=FactorialCombinationConfig(
                    interventions=INTERVENTIONS,
                    assignment_mechanism=AssignmentMechanism.RANDOMIZED,
                    minimum_cell_size=3,
                    include_baseline_covariate=True,
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


def run_example(output_dir: Path) -> RejuvenationWorkflowReport:
    """Run the example into a caller-selected, no-clobber workflow directory."""
    report = run_rejuvenation_workflow(build_request(), output_dir=output_dir)
    verified = load_rejuvenation_workflow_report(output_dir)
    if verified != report:
        raise RuntimeError("verified workflow report differs from the in-memory report")
    return report


def _print_summary(
    report: RejuvenationWorkflowReport,
    *,
    output_dir: Path,
    temporary: bool,
) -> None:
    """Print phase dispositions and the scientific safety boundaries exercised."""
    print("Synthetic RejuvenationKit four-phase workflow")
    for disposition in report.dispositions:
        print(f"{disposition.phase.value}: {disposition.status.value} ({disposition.reason})")
    print(
        "QC gate: "
        f"passed={report.qc_gate.qc_passed}; "
        f"serialized_override={report.qc_gate.serialized_override_enabled}; "
        f"override_applied={report.qc_gate.override_applied}"
    )
    print(
        "Explicit boundaries: Phase2FusionInput, TimedEvidenceBatch, and "
        "SubjectEndpointBatch were supplied by the caller; no downstream input was inferred."
    )
    print(
        f"Verified bundle: {len(report.artifacts)} declared paths; "
        "workflow-manifest.json checksums passed."
    )
    location_note = "temporary and removed after this example exits" if temporary else "retained"
    print(f"Bundle path: {output_dir} ({location_note})")
    print(
        "Guardrail: all assignments, calibrations, measurements, and endpoints are synthetic. "
        "Completion demonstrates software execution and provenance only, not efficacy, synergy, "
        "causality, or biological-age reversal."
    )


def main(output_dir: Path | None = None) -> None:
    """Run in a temporary directory unless the caller supplies a safe destination."""
    if output_dir is not None:
        report = run_example(output_dir)
        _print_summary(report, output_dir=output_dir, temporary=False)
        return
    with TemporaryDirectory(prefix="rejuvenationkit-four-phase-") as raw_directory:
        temporary_output = Path(raw_directory) / "workflow-bundle"
        report = run_example(temporary_output)
        _print_summary(report, output_dir=temporary_output, temporary=True)


if __name__ == "__main__":
    main()
