"""Explicit, provenance-preserving boundaries between SDK phases."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from math import sqrt
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from rejuvenationkit.endpoints import EndpointExclusion, SubjectEndpoint, SubjectEndpointBatch
from rejuvenationkit.evidence import Estimand, EvidenceEstimate
from rejuvenationkit.longitudinal import LongitudinalChannel, extract_visit_aligned_values
from rejuvenationkit.qc import ExpectedVisit
from rejuvenationkit.schemas import Observation, Study, Subject, study_artifact_hash
from rejuvenationkit.state import StateEstimationReport


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class TimedEvidenceMeasurement(BaseModel):
    """A subject-level Phase 2 estimate assigned an explicit Phase 3 time/channel.

    ``EvidenceEstimate.estimand.time_contrast`` is descriptive and is never parsed
    into a timestamp. The caller must supply the actual measurement time and exact
    state-channel feature name here.
    """

    model_config = ConfigDict(frozen=True)

    evidence: EvidenceEstimate
    timestamp: datetime
    feature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        """Require subject identity, aware time, and typed calibration provenance."""
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timed evidence timestamp must be timezone-aware")
        if self.evidence.subject_id is None:
            raise ValueError("timed evidence must identify one biological subject")
        if self.evidence.calibration_reference is None:
            raise ValueError("timed evidence requires a typed CalibrationReference")
        if not self.evidence.calibration_reference.fusion_eligible:
            raise ValueError(
                "timed evidence requires a held-out or externally validated calibration"
            )
        if not self.feature.strip():
            raise ValueError("timed evidence feature cannot be blank")
        return self

    def to_observation(self) -> Observation:
        """Convert without discarding uncertainty or calibration identity."""
        reference = self.evidence.calibration_reference
        assert reference is not None
        subject_id = self.evidence.subject_id
        assert subject_id is not None
        attributes: dict[str, str | int | float | bool] = {
            "evidence_id": self.evidence.evidence_id,
            "evidence_estimand_key": self.evidence.estimand.key,
            "evidence_estimand_name": self.evidence.estimand.name,
            "evidence_effect_direction": self.evidence.estimand.direction.value,
            "evidence_provenance_id": self.evidence.provenance_id,
            "calibration_id": reference.calibration_id,
            "calibration_artifact_hash": reference.artifact_hash,
            "calibration_status": reference.status.value,
            "calibration_method": reference.method,
            "calibration_validation_provenance_id": reference.validation_provenance_id,
            "evidence_quality_flags_json": json.dumps(self.evidence.quality_flags),
        }
        optional_attributes: tuple[tuple[str, str | int | None], ...] = (
            ("evidence_population", self.evidence.estimand.population),
            ("evidence_time_contrast", self.evidence.estimand.time_contrast),
            ("evidence_transform", self.evidence.estimand.transform),
            ("evidence_species_taxon_id", self.evidence.species_taxon_id),
            ("evidence_tissue", self.evidence.tissue),
            ("evidence_sample_id", self.evidence.sample_id),
            ("evidence_assay_id", self.evidence.assay_id),
            ("evidence_correlation_group", self.evidence.correlation_group),
        )
        attributes.update({name: value for name, value in optional_attributes if value is not None})
        return Observation(
            subject_id=subject_id,
            timestamp=self.timestamp,
            modality=self.evidence.modality,
            feature=self.feature,
            value=self.evidence.estimate,
            unit=self.evidence.estimand.unit,
            standard_error=self.evidence.standard_error,
            source_uri=self.evidence.provenance_id,
            attributes=attributes,
        )


class TimedEvidenceBatch(BaseModel):
    """A complete, unique set of timed evidence rows ready for state estimation."""

    model_config = ConfigDict(frozen=True)

    study_id: str = Field(min_length=1)
    subjects: tuple[Subject, ...] = Field(min_length=1)
    measurements: tuple[TimedEvidenceMeasurement, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        """Reject unknown subjects, duplicate rows, and mixed channel semantics."""
        subject_ids = [item.subject_id for item in self.subjects]
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError("timed-evidence subjects must be unique")
        known = set(subject_ids)
        measurement_subject_ids = tuple(
            item.evidence.subject_id
            for item in self.measurements
            if item.evidence.subject_id is not None
        )
        unknown = set(measurement_subject_ids).difference(known)
        if unknown:
            raise ValueError(f"timed evidence references unknown subjects: {sorted(unknown)}")
        keys = [
            (
                measurement_subject_ids[index],
                item.timestamp,
                item.evidence.modality,
                item.feature,
                item.evidence.estimand.unit,
            )
            for index, item in enumerate(self.measurements)
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("timed evidence contains duplicate subject/time/channel rows")
        evidence_ids = [item.evidence.evidence_id for item in self.measurements]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("timed evidence evidence_id values must be unique")
        channel_semantics: dict[tuple[object, ...], tuple[object, ...]] = {}
        for item in self.measurements:
            reference = item.evidence.calibration_reference
            assert reference is not None
            channel = (
                item.evidence.modality,
                item.feature,
                item.evidence.estimand.unit,
            )
            semantics = (
                item.evidence.estimand,
                item.evidence.species_taxon_id,
                item.evidence.tissue,
                item.evidence.assay_id,
                reference,
            )
            previous = channel_semantics.setdefault(channel, semantics)
            if previous != semantics:
                raise ValueError(
                    "timed evidence Phase 3 channels require one exact estimand, species, "
                    "tissue, assay, and calibration reference"
                )
        object.__setattr__(
            self,
            "subjects",
            tuple(sorted(self.subjects, key=lambda item: item.subject_id)),
        )
        object.__setattr__(
            self,
            "measurements",
            tuple(
                sorted(
                    self.measurements,
                    key=lambda item: (
                        item.evidence.subject_id or "",
                        item.timestamp,
                        item.evidence.modality.value,
                        item.feature,
                        item.evidence.estimand.unit,
                        item.evidence.evidence_id,
                    ),
                )
            ),
        )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_hash(self) -> str:
        """Bind every evidence record, timestamp, feature, and subject assignment."""
        return _canonical_hash(
            {
                "schema": "timed-evidence-batch/v1",
                "study_id": self.study_id,
                "subjects": [item.model_dump(mode="json") for item in self.subjects],
                "measurements": [item.model_dump(mode="json") for item in self.measurements],
            }
        )

    def to_study(self) -> Study:
        """Create the canonical Study consumed by Phase 3 estimators."""
        return Study(
            study_id=self.study_id,
            subjects=self.subjects,
            observations=tuple(item.to_observation() for item in self.measurements),
            metadata={
                "source": "phase2_timed_evidence",
                "timed_evidence_artifact_hash": self.artifact_hash,
            },
        )


class StateEndpointConfig(BaseModel):
    """Policy for a Phase 3 follow-up endpoint with an explicit baseline covariate."""

    model_config = ConfigDict(frozen=True)

    batch_id: str = Field(min_length=1)
    state_name: str = Field(min_length=1)
    estimand: Estimand
    minimum_timepoints: int = Field(default=2, ge=2)


def state_report_to_endpoints(
    report: StateEstimationReport,
    config: StateEndpointConfig,
) -> SubjectEndpointBatch:
    """Use terminal latent state as outcome and first state as baseline covariate.

    The adapter intentionally does not manufacture a change-score standard error:
    marginal state covariances do not contain the required cross-time covariance.
    A downstream factorial model can instead prespecify baseline-adjusted analysis.
    """
    if config.state_name not in report.model_state_names:
        raise ValueError(f"state_name is absent from report: {config.state_name!r}")
    state_index = report.model_state_names.index(config.state_name)
    report_hash = report.artifact_hash
    endpoints: list[SubjectEndpoint] = []
    excluded = [
        EndpointExclusion(
            subject_id=item.subject_id,
            reason=item.reason,
            detail="Phase 3 produced no trajectory",
        )
        for item in report.excluded_subjects
    ]
    for trajectory in report.trajectories:
        if len(trajectory.estimates) < config.minimum_timepoints:
            excluded.append(
                EndpointExclusion(
                    subject_id=trajectory.subject_id,
                    reason="insufficient_state_timepoints",
                    detail=(
                        f"observed={len(trajectory.estimates)};required={config.minimum_timepoints}"
                    ),
                )
            )
            continue
        baseline = trajectory.estimates[0]
        endpoint = trajectory.estimates[-1]
        standard_error = sqrt(endpoint.covariance[state_index][state_index])
        endpoints.append(
            SubjectEndpoint(
                subject_id=trajectory.subject_id,
                estimand=config.estimand,
                estimate=endpoint.mean[state_index],
                standard_error=standard_error,
                endpoint_timestamp=endpoint.timestamp,
                baseline_timestamp=baseline.timestamp,
                baseline_estimate=baseline.mean[state_index],
                provenance_id=(
                    f"state-report:{report_hash}:state={config.state_name}:"
                    f"kind={endpoint.estimate_kind}"
                ),
                quality_flags=("state_endpoint_requires_prespecified_baseline_adjustment",),
            )
        )
    if not endpoints:
        raise ValueError("state report contains no analyzable endpoint trajectories")
    return SubjectEndpointBatch(
        study_id=report.study_id,
        batch_id=config.batch_id,
        estimand=config.estimand,
        endpoints=tuple(endpoints),
        excluded=tuple(excluded),
        source_artifact_hash=report_hash,
    )


def study_feature_endpoints(
    study: Study,
    *,
    visits: tuple[ExpectedVisit, ...],
    channel: LongitudinalChannel,
    baseline_visit_id: str,
    endpoint_visit_id: str,
    batch_id: str,
    estimand: Estimand,
) -> SubjectEndpointBatch:
    """Build auditable follow-up/baseline endpoints from one exact raw-data channel."""
    visit_ids = {item.visit_id for item in visits}
    if baseline_visit_id == endpoint_visit_id:
        raise ValueError("baseline and endpoint visit identifiers must differ")
    missing_visits = {baseline_visit_id, endpoint_visit_id}.difference(visit_ids)
    if missing_visits:
        raise ValueError(f"endpoint visits are absent from schedule: {sorted(missing_visits)}")
    extraction = extract_visit_aligned_values(
        study,
        visits=visits,
        channels=(channel,),
    )
    lookup = extraction.value_map()
    endpoints: list[SubjectEndpoint] = []
    excluded: list[EndpointExclusion] = []
    for subject in sorted(study.subjects, key=lambda item: item.subject_id):
        baseline = lookup.get((subject.subject_id, baseline_visit_id, 0))
        endpoint = lookup.get((subject.subject_id, endpoint_visit_id, 0))
        if baseline is None or endpoint is None:
            reasons = sorted(
                {
                    item.reason.value
                    for item in extraction.exclusions
                    if item.subject_id == subject.subject_id
                    and item.visit_id in {baseline_visit_id, endpoint_visit_id}
                }
            )
            excluded.append(
                EndpointExclusion(
                    subject_id=subject.subject_id,
                    reason="incomplete_endpoint_pair",
                    detail=",".join(reasons) or "required visit value unavailable",
                )
            )
            continue
        endpoints.append(
            SubjectEndpoint(
                subject_id=subject.subject_id,
                estimand=estimand,
                estimate=endpoint.value,
                endpoint_timestamp=endpoint.effective_timestamp,
                baseline_timestamp=baseline.effective_timestamp,
                baseline_estimate=baseline.value,
                provenance_id=(
                    f"visit-alignment:{study.study_id}:{channel.modality.value}:"
                    f"{channel.feature}:{channel.unit}:"
                    f"baseline_rows={baseline.selected_observation_indices}:"
                    f"endpoint_rows={endpoint.selected_observation_indices}"
                ),
                quality_flags=("raw_endpoint_standard_error_not_available",),
            )
        )
    if not endpoints:
        raise ValueError("study contains no complete subject endpoint pairs")
    study_hash = study_artifact_hash(study)
    return SubjectEndpointBatch(
        study_id=study.study_id,
        batch_id=batch_id,
        estimand=estimand,
        endpoints=tuple(endpoints),
        excluded=tuple(excluded),
        source_artifact_hash=study_hash,
    )
