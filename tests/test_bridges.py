from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest
from pydantic import ValidationError

from rejuvenationkit.bridges import (
    StateEndpointConfig,
    TimedEvidenceBatch,
    TimedEvidenceMeasurement,
    state_report_to_endpoints,
    study_feature_endpoints,
)
from rejuvenationkit.evidence import (
    CalibrationReference,
    CalibrationValidationStatus,
    EffectDirection,
    Estimand,
    EvidenceEstimate,
)
from rejuvenationkit.longitudinal import LongitudinalChannel
from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject
from rejuvenationkit.state import (
    LinearGaussianStateConfig,
    LinearGaussianStateEstimator,
    StateChannel,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
ESTIMAND = Estimand(
    name="latent_health_followup",
    unit="z",
    direction=EffectDirection.HIGHER_IS_BETTER,
    population="factorial-dogs",
    time_contrast="month-6 adjusted for baseline",
)


def _evidence(subject_id: str, value: float) -> EvidenceEstimate:
    calibration_id = "clock-v1"
    target = Estimand(
        name="biological_age",
        unit="years",
        direction=EffectDirection.LOWER_IS_BETTER,
        population="canine-state-cohort",
    )
    return EvidenceEstimate(
        evidence_id=f"clock:{subject_id}",
        modality=Modality.METHYLATION,
        estimand=target,
        estimate=value,
        standard_error=0.5,
        calibration_id=calibration_id,
        calibration_reference=CalibrationReference(
            calibration_id=calibration_id,
            artifact_hash=sha256(calibration_id.encode()).hexdigest(),
            status=CalibrationValidationStatus.HELD_OUT_VALIDATED,
            estimand=target,
            method="held-out clock calibration",
            validation_provenance_id="clock-validation-v1",
        ),
        provenance_id="assay-analysis-v1",
        subject_id=subject_id,
        species_taxon_id=9615,
        tissue="blood",
    )


def test_timed_evidence_bridge_requires_explicit_timestamp_and_retains_uncertainty() -> None:
    subjects = (Subject(subject_id="dog-1", cohort="treated"),)
    measurement = TimedEvidenceMeasurement(
        evidence=_evidence("dog-1", 8.0),
        timestamp=START,
        feature="methylation_age",
    )
    batch = TimedEvidenceBatch(
        study_id="state-input",
        subjects=subjects,
        measurements=(measurement,),
    )

    study = batch.to_study()

    assert study.observations[0].standard_error == 0.5
    assert (
        study.observations[0].attributes["calibration_artifact_hash"]
        == sha256(b"clock-v1").hexdigest()
    )
    assert (
        study.observations[0].attributes["evidence_estimand_key"]
        == measurement.evidence.estimand.key
    )
    assert study.observations[0].attributes["evidence_species_taxon_id"] == 9615
    assert study.observations[0].attributes["evidence_tissue"] == "blood"
    assert study.metadata["timed_evidence_artifact_hash"] == batch.artifact_hash
    with pytest.raises(ValidationError, match="timezone-aware"):
        TimedEvidenceMeasurement(
            evidence=_evidence("dog-1", 8.0),
            timestamp=datetime(2026, 1, 1),
            feature="methylation_age",
        )
    unverified = _evidence("dog-1", 8.0)
    assert unverified.calibration_reference is not None
    unverified = unverified.model_copy(
        update={
            "calibration_reference": unverified.calibration_reference.model_copy(
                update={"status": CalibrationValidationStatus.UNVERIFIED}
            )
        }
    )
    with pytest.raises(ValidationError, match="held-out or externally validated"):
        TimedEvidenceMeasurement(
            evidence=unverified,
            timestamp=START,
            feature="methylation_age",
        )


def test_timed_evidence_batch_is_order_invariant_and_rejects_mixed_semantics() -> None:
    subjects = (
        Subject(subject_id="dog-2", cohort="treated"),
        Subject(subject_id="dog-1", cohort="control"),
    )
    measurements = (
        TimedEvidenceMeasurement(
            evidence=_evidence("dog-2", 7.5),
            timestamp=START + timedelta(days=180),
            feature="methylation_age",
        ),
        TimedEvidenceMeasurement(
            evidence=_evidence("dog-1", 8.0),
            timestamp=START,
            feature="methylation_age",
        ),
    )
    first = TimedEvidenceBatch(
        study_id="state-input",
        subjects=subjects,
        measurements=measurements,
    )
    reordered = TimedEvidenceBatch(
        study_id="state-input",
        subjects=tuple(reversed(subjects)),
        measurements=tuple(reversed(measurements)),
    )

    assert first == reordered
    assert first.artifact_hash == reordered.artifact_hash
    assert first.to_study() == reordered.to_study()

    baseline = _evidence("dog-1", 8.0)
    reference = baseline.calibration_reference
    assert reference is not None
    changed_estimand = baseline.estimand.model_copy(update={"name": "phenotypic_age"})
    incompatible = (
        baseline.model_copy(
            update={
                "evidence_id": "clock:dog-1:estimand",
                "estimand": changed_estimand,
                "calibration_reference": reference.model_copy(
                    update={"estimand": changed_estimand}
                ),
            }
        ),
        baseline.model_copy(
            update={"evidence_id": "clock:dog-1:species", "species_taxon_id": 10090}
        ),
        baseline.model_copy(update={"evidence_id": "clock:dog-1:tissue", "tissue": "liver"}),
        baseline.model_copy(
            update={"evidence_id": "clock:dog-1:assay", "assay_id": "clock-assay-v2"}
        ),
        baseline.model_copy(
            update={
                "evidence_id": "clock:dog-1:calibration",
                "calibration_id": "clock-v2",
                "calibration_reference": reference.model_copy(
                    update={
                        "calibration_id": "clock-v2",
                        "artifact_hash": sha256(b"clock-v2").hexdigest(),
                    }
                ),
            }
        ),
    )
    for changed in incompatible:
        with pytest.raises(
            ValidationError,
            match="one exact estimand, species, tissue, assay, and calibration reference",
        ):
            TimedEvidenceBatch(
                study_id="state-input",
                subjects=(subjects[1],),
                measurements=(
                    TimedEvidenceMeasurement(
                        evidence=baseline,
                        timestamp=START,
                        feature="methylation_age",
                    ),
                    TimedEvidenceMeasurement(
                        evidence=changed,
                        timestamp=START + timedelta(days=180),
                        feature="methylation_age",
                    ),
                ),
            )

    with pytest.raises(ValidationError, match="evidence_id values must be unique"):
        TimedEvidenceBatch(
            study_id="state-input",
            subjects=(subjects[1],),
            measurements=(
                TimedEvidenceMeasurement(
                    evidence=baseline,
                    timestamp=START,
                    feature="methylation_age",
                ),
                TimedEvidenceMeasurement(
                    evidence=baseline,
                    timestamp=START + timedelta(days=180),
                    feature="methylation_age",
                ),
            ),
        )


def test_state_report_bridge_creates_followup_endpoint_and_baseline_covariate() -> None:
    subject = Subject(subject_id="dog-1", cohort="treated")
    study = Study(
        study_id="state-study",
        subjects=(subject,),
        observations=tuple(
            Observation(
                subject_id="dog-1",
                timestamp=timestamp,
                modality=Modality.CLINICAL,
                feature="score",
                value=value,
                unit="z",
            )
            for timestamp, value in ((START, 0.0), (START + timedelta(days=180), 1.0))
        ),
    )
    config = LinearGaussianStateConfig(
        state_names=("latent_health",),
        channels=(
            StateChannel(
                name="score",
                modality=Modality.CLINICAL,
                feature="score",
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
        time_unit_days=365.2425,
        smooth=True,
    )
    report = LinearGaussianStateEstimator(config).fit(study).estimate_report(study)

    endpoints = state_report_to_endpoints(
        report,
        StateEndpointConfig(batch_id="latent-v1", state_name="latent_health", estimand=ESTIMAND),
    )

    endpoint = endpoints.endpoints[0]
    assert endpoints.source_artifact_hash == report.artifact_hash
    assert endpoint.baseline_estimate == pytest.approx(report.trajectories[0].estimates[0].mean[0])
    assert endpoint.estimate == pytest.approx(report.trajectories[0].estimates[-1].mean[0])
    assert endpoint.standard_error is not None
    assert "baseline_adjustment" in endpoint.quality_flags[0]


def test_raw_study_endpoint_bridge_uses_exact_visit_alignment_and_audits_missingness() -> None:
    subjects = (
        Subject(subject_id="dog-1", cohort="treated"),
        Subject(subject_id="dog-2", cohort="control"),
    )
    observations = (
        Observation(
            subject_id="dog-1",
            timestamp=START,
            modality=Modality.CLINICAL,
            feature="frailty",
            value=5.0,
            unit="score",
        ),
        Observation(
            subject_id="dog-1",
            timestamp=START + timedelta(days=180),
            modality=Modality.CLINICAL,
            feature="frailty",
            value=3.0,
            unit="score",
        ),
        Observation(
            subject_id="dog-2",
            timestamp=START,
            modality=Modality.CLINICAL,
            feature="frailty",
            value=6.0,
            unit="score",
        ),
    )
    study = Study(study_id="raw", subjects=subjects, observations=observations)
    visits = tuple(
        ExpectedVisit(
            visit_id=visit_id,
            scheduled_at=timestamp,
            window_before=timedelta(days=2),
            window_after=timedelta(days=2),
            required_features=(VisitFeature(feature="frailty", modality=Modality.CLINICAL),),
        )
        for visit_id, timestamp in (
            ("baseline", START),
            ("month-6", START + timedelta(days=180)),
        )
    )

    endpoints = study_feature_endpoints(
        study,
        visits=visits,
        channel=LongitudinalChannel(
            feature="frailty",
            modality=Modality.CLINICAL,
            unit="score",
        ),
        baseline_visit_id="baseline",
        endpoint_visit_id="month-6",
        batch_id="frailty-v1",
        estimand=ESTIMAND.model_copy(update={"unit": "score"}),
    )

    assert endpoints.endpoints[0].subject_id == "dog-1"
    assert endpoints.endpoints[0].baseline_estimate == 5.0
    assert endpoints.endpoints[0].estimate == 3.0
    assert endpoints.excluded[0].subject_id == "dog-2"
    assert endpoints.excluded[0].reason == "incomplete_endpoint_pair"
