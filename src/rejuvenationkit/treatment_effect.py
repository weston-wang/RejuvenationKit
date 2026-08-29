"""Leakage-safe inference for randomized longitudinal treatment studies."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.combinations import AssignmentMechanism
from rejuvenationkit.longitudinal import (
    LongitudinalChannel,
    LongitudinalExclusion,
    LongitudinalExclusionReason,
    complete_visit_vectors,
    extract_visit_aligned_values,
)
from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Study


class TreatmentEffectConfig(BaseModel):
    """Configuration for cross-validated treatment-effect inference."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    features: tuple[LongitudinalChannel | VisitFeature, ...] = Field(min_length=1)
    assignment_mechanism: AssignmentMechanism
    cross_validation_folds: int = Field(default=5, ge=2)
    permutations: int = Field(default=999, ge=99)
    bootstrap_samples: int = Field(default=999, ge=99)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    false_alarm_rate: float = Field(default=0.05, gt=0, lt=0.5)
    covariance_shrinkage: float = Field(default=0.20, ge=0, le=1)
    covariance_ridge: float = Field(default=1e-9, gt=0)
    minimum_group_size: int = Field(default=8, ge=3)
    random_seed: int = 0

    @model_validator(mode="after")
    def require_unique_features(self) -> TreatmentEffectConfig:
        """Require randomization and unambiguous exact or wildcard channels."""
        if self.assignment_mechanism is not AssignmentMechanism.RANDOMIZED:
            raise ValueError(
                "RandomizedTreatmentEffectEvaluator requires randomized assignment; "
                "observational groups require a separately justified observational analysis"
            )
        for index, first in enumerate(self.features):
            for second in self.features[index + 1 :]:
                if first.feature == second.feature and (
                    first.modality is None
                    or second.modality is None
                    or first.modality is second.modality
                ):
                    raise ValueError(
                        "treatment-effect features cannot overlap by exact or wildcard modality"
                    )
        return self


class CrossValidatedSubjectScore(BaseModel):
    """A subject score produced without using that subject for calibration."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    subject_id: str
    group: str
    follow_up_visit_id: str
    fold: int = Field(ge=0)
    squared_mahalanobis_distance: float = Field(ge=0)
    empirical_tail_probability: float = Field(gt=0, le=1)
    detected: bool
    calibration_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    calibration_reference_count: int | None = Field(default=None, ge=2)
    calibration_reference_artifact_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    empirical_threshold: float | None = Field(default=None, ge=0)
    out_of_fold_null_count: int | None = Field(default=None, ge=1)
    out_of_fold_null_artifact_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def require_complete_calibration_reference(self) -> CrossValidatedSubjectScore:
        """Require all compact calibration fields together when any are present."""
        values = (
            self.calibration_id,
            self.calibration_reference_count,
            self.calibration_reference_artifact_hash,
            self.empirical_threshold,
            self.out_of_fold_null_count,
            self.out_of_fold_null_artifact_hash,
        )
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError("subject calibration provenance must be provided together")
        if self.empirical_threshold is not None and self.detected != (
            self.squared_mahalanobis_distance > self.empirical_threshold
        ):
            raise ValueError("detected flag does not match the empirical threshold")
        return self


class FoldCalibrationProvenance(BaseModel):
    """Compact, reconstructable provenance for one out-of-fold scoring model."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    calibration_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    follow_up_visit_id: str = Field(min_length=1)
    fold: int = Field(ge=0)
    training_control_subject_ids: tuple[str, ...] = Field(min_length=2)
    training_control_count: int = Field(ge=2)
    held_out_control_subject_ids: tuple[str, ...] = Field(min_length=1)
    held_out_control_count: int = Field(ge=1)
    training_reference_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_model_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    empirical_threshold: float = Field(ge=0)
    out_of_fold_null_count: int = Field(ge=1)
    out_of_fold_null_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    covariance_method: Literal["empirical covariance with diagonal shrinkage and scaled ridge"] = (
        "empirical covariance with diagonal shrinkage and scaled ridge"
    )

    @model_validator(mode="after")
    def validate_identity_and_counts(self) -> FoldCalibrationProvenance:
        """Bind subject identities, counts, and every compact artifact checksum."""
        for name, values in (
            ("training control", self.training_control_subject_ids),
            ("held-out control", self.held_out_control_subject_ids),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} subject identifiers must be unique and sorted")
        if self.training_control_count != len(self.training_control_subject_ids):
            raise ValueError("training control count does not match identifiers")
        if self.held_out_control_count != len(self.held_out_control_subject_ids):
            raise ValueError("held-out control count does not match identifiers")
        if set(self.training_control_subject_ids).intersection(self.held_out_control_subject_ids):
            raise ValueError("training and held-out control identifiers must be disjoint")
        expected_id = _canonical_hash(self.model_dump(mode="json", exclude={"calibration_id"}))
        if self.calibration_id != expected_id:
            raise ValueError("calibration_id does not match calibration provenance")
        return self


class RandomizedInferenceProvenance(BaseModel):
    """Prespecified analysis domain and compact randomized-inference audit trail."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    study_id: str = Field(min_length=1)
    config: TreatmentEffectConfig
    baseline_visit: ExpectedVisit
    follow_up_visits: tuple[ExpectedVisit, ...] = Field(min_length=1)
    resolved_channels: tuple[LongitudinalChannel, ...] = Field(min_length=1)
    treated_subject_ids: tuple[str, ...] = Field(min_length=1)
    control_subject_ids: tuple[str, ...] = Field(min_length=1)
    analysis_input_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_calibrations: tuple[FoldCalibrationProvenance, ...] = Field(min_length=1)
    covariance_method: Literal["empirical covariance with diagonal shrinkage and scaled ridge"] = (
        "empirical covariance with diagonal shrinkage and scaled ridge"
    )
    omnibus_covariance_population: Literal["pooled complete treated and control change vectors"] = (
        "pooled complete treated and control change vectors"
    )
    permutation_method: Literal[
        "unrestricted subject-label permutation with fixed treated sample size"
    ] = "unrestricted subject-label permutation with fixed treated sample size"
    bootstrap_method: Literal["independent within-group nonparametric percentile bootstrap"] = (
        "independent within-group nonparametric percentile bootstrap"
    )
    fold_assignment_method: Literal[
        "seeded shuffle of sorted group identifiers with round-robin folds"
    ] = "seeded shuffle of sorted group identifiers with round-robin folds"

    @model_validator(mode="after")
    def validate_prespecified_domain(self) -> RandomizedInferenceProvenance:
        """Require deterministic groups and one calibration for every visit-fold."""
        for name, values in (
            ("treated", self.treated_subject_ids),
            ("control", self.control_subject_ids),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} subject identifiers must be unique and sorted")
        if set(self.treated_subject_ids).intersection(self.control_subject_ids):
            raise ValueError("treated and control provenance identifiers must be disjoint")
        visit_ids = tuple(item.visit_id for item in self.follow_up_visits)
        if len(visit_ids) != len(set(visit_ids)):
            raise ValueError("randomized-inference follow-up visits must be unique")
        if self.baseline_visit.visit_id in visit_ids:
            raise ValueError("randomized-inference baseline cannot also be a follow-up")
        if len(self.resolved_channels) != len(self.config.features) or len(
            self.resolved_channels
        ) != len(set(self.resolved_channels)):
            raise ValueError("resolved inference channels must form one unique configured axis")
        for requested, resolved in zip(
            self.config.features,
            self.resolved_channels,
            strict=True,
        ):
            if requested.feature != resolved.feature or (
                requested.modality is not None and requested.modality is not resolved.modality
            ):
                raise ValueError("resolved inference channel does not match configured feature")
            if isinstance(requested, LongitudinalChannel) and requested != resolved:
                raise ValueError("resolved inference channel does not match exact configuration")
        expected = {
            (visit_id, fold)
            for visit_id in visit_ids
            for fold in range(self.config.cross_validation_folds)
        }
        observed = {(item.follow_up_visit_id, item.fold) for item in self.fold_calibrations}
        if observed != expected or len(observed) != len(self.fold_calibrations):
            raise ValueError("fold calibrations must cover every follow-up and fold exactly once")
        return self


class FeatureTreatmentEffect(BaseModel):
    """Difference-in-differences estimate for one measured channel."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    feature: str
    modality: Modality | None
    treated_mean_change: float
    control_mean_change: float
    difference_in_differences: float
    confidence_interval_low: float
    confidence_interval_high: float

    @model_validator(mode="after")
    def validate_effect_arithmetic(self) -> FeatureTreatmentEffect:
        """Bind the reported contrast and confidence-interval ordering."""
        if not np.isclose(
            self.difference_in_differences,
            self.treated_mean_change - self.control_mean_change,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("difference in differences does not match group means")
        if self.confidence_interval_low > self.confidence_interval_high:
            raise ValueError("treatment-effect confidence interval is inverted")
        return self


class VisitTreatmentEffect(BaseModel):
    """Multivariate and channel-level effects at one follow-up visit."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    follow_up_visit_id: str
    treated_subjects: int = Field(ge=1)
    control_subjects: int = Field(ge=1)
    omnibus_squared_mahalanobis_distance: float = Field(ge=0)
    permutation_p_value: float = Field(gt=0, le=1)
    effects: tuple[FeatureTreatmentEffect, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_effect_axis(self) -> VisitTreatmentEffect:
        """Reject duplicate channel effects within one follow-up estimand."""
        keys = tuple((item.feature, item.modality) for item in self.effects)
        if len(keys) != len(set(keys)):
            raise ValueError("visit treatment effects must have unique channels")
        return self


class TreatmentEffectReport(BaseModel):
    """Leakage-safe subject scores and randomized group estimates."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    study_id: str = Field(min_length=1)
    baseline_visit_id: str = Field(min_length=1)
    treated_label: str = Field(min_length=1)
    control_label: str = Field(min_length=1)
    visit_effects: tuple[VisitTreatmentEffect, ...]
    subject_scores: tuple[CrossValidatedSubjectScore, ...]
    excluded_subject_ids: tuple[str, ...] = ()
    exclusions: tuple[LongitudinalExclusion, ...] = ()
    inference_provenance: RandomizedInferenceProvenance | None = None

    @model_validator(mode="after")
    def bind_subject_scores_to_calibration(self) -> TreatmentEffectReport:
        """Prevent score relabeling or detachment from its serialized fold model."""
        if self.treated_label == self.control_label:
            raise ValueError("treated and control labels must be distinct")
        visit_ids = tuple(item.follow_up_visit_id for item in self.visit_effects)
        if len(visit_ids) != len(set(visit_ids)):
            raise ValueError("treatment-effect follow-up visits must be unique")
        if tuple(sorted(set(self.excluded_subject_ids))) != self.excluded_subject_ids:
            raise ValueError("excluded treatment subjects must be unique and sorted")
        if self.inference_provenance is None:
            return self
        provenance = self.inference_provenance
        if provenance.study_id != self.study_id:
            raise ValueError("inference provenance study does not match report")
        if provenance.baseline_visit.visit_id != self.baseline_visit_id:
            raise ValueError("inference provenance baseline does not match report")
        calibrations = {
            (item.follow_up_visit_id, item.fold): item for item in provenance.fold_calibrations
        }
        treated = set(provenance.treated_subject_ids)
        controls = set(provenance.control_subject_ids)
        all_subjects = treated.union(controls)
        if not set(self.excluded_subject_ids).issubset(all_subjects):
            raise ValueError("excluded treatment subjects must belong to a prespecified group")
        excluded_with_incomplete_vector = {
            item.subject_id
            for item in self.exclusions
            if item.reason is LongitudinalExclusionReason.INCOMPLETE_VISIT_VECTOR
            and item.subject_id is not None
        }
        if not set(self.excluded_subject_ids).issubset(excluded_with_incomplete_vector):
            raise ValueError("excluded treatment subjects require incomplete-vector provenance")
        if any(
            item.subject_id is not None and item.subject_id not in all_subjects
            for item in self.exclusions
        ):
            raise ValueError("treatment exclusions contain an unknown subject")
        if visit_ids != tuple(item.visit_id for item in provenance.follow_up_visits):
            raise ValueError("treatment-effect visits do not match inference provenance")
        expected_channels = tuple(
            (item.feature, item.modality) for item in provenance.config.features
        )
        if any(
            tuple((item.feature, item.modality) for item in visit.effects) != expected_channels
            for visit in self.visit_effects
        ):
            raise ValueError("treatment effects do not match the configured channel axis")
        treated_folds = _subject_folds(
            provenance.treated_subject_ids,
            provenance.config.cross_validation_folds,
            provenance.config.random_seed + 1,
        )
        control_folds = _subject_folds(
            provenance.control_subject_ids,
            provenance.config.cross_validation_folds,
            provenance.config.random_seed,
        )
        score_keys = [(score.follow_up_visit_id, score.subject_id) for score in self.subject_scores]
        if len(score_keys) != len(set(score_keys)):
            raise ValueError("subject scores must be unique within each follow-up")
        for score in self.subject_scores:
            calibration = calibrations.get((score.follow_up_visit_id, score.fold))
            if calibration is None or score.calibration_id != calibration.calibration_id:
                raise ValueError("subject score is detached from its fold calibration")
            if (
                score.calibration_reference_count != calibration.training_control_count
                or score.calibration_reference_artifact_hash
                != calibration.training_reference_artifact_hash
                or score.empirical_threshold != calibration.empirical_threshold
                or score.out_of_fold_null_count != calibration.out_of_fold_null_count
                or score.out_of_fold_null_artifact_hash
                != calibration.out_of_fold_null_artifact_hash
            ):
                raise ValueError("subject score calibration provenance does not match fold")
            expected_group = (
                self.treated_label
                if score.subject_id in treated
                else self.control_label
                if score.subject_id in controls
                else None
            )
            if expected_group is None or score.group != expected_group:
                raise ValueError("subject score group does not match prespecified assignment")
            expected_fold = (
                treated_folds[score.subject_id]
                if score.subject_id in treated
                else control_folds[score.subject_id]
            )
            if score.fold != expected_fold:
                raise ValueError("subject score fold does not match prespecified assignment")
        for visit in provenance.follow_up_visits:
            complete_controls = {
                score.subject_id
                for score in self.subject_scores
                if score.follow_up_visit_id == visit.visit_id and score.subject_id in controls
            }
            for fold in range(provenance.config.cross_validation_folds):
                calibration = calibrations[(visit.visit_id, fold)]
                expected_held_out = tuple(
                    sorted(
                        subject_id
                        for subject_id in complete_controls
                        if control_folds[subject_id] == fold
                    )
                )
                expected_training = tuple(sorted(complete_controls.difference(expected_held_out)))
                if (
                    calibration.held_out_control_subject_ids != expected_held_out
                    or calibration.training_control_subject_ids != expected_training
                ):
                    raise ValueError(
                        "fold calibration controls do not match scored control population"
                    )
        return self

    def effects_frame(self) -> pd.DataFrame:
        """Return one tidy row per visit and feature."""
        return pd.DataFrame(
            {
                "follow_up_visit_id": visit.follow_up_visit_id,
                "treated_subjects": visit.treated_subjects,
                "control_subjects": visit.control_subjects,
                "omnibus_squared_mahalanobis_distance": (
                    visit.omnibus_squared_mahalanobis_distance
                ),
                "permutation_p_value": visit.permutation_p_value,
                **effect.model_dump(mode="json"),
            }
            for visit in self.visit_effects
            for effect in visit.effects
        )

    def scores_frame(self) -> pd.DataFrame:
        """Return cross-validated subject scores as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.subject_scores)


class RandomizedTreatmentEffectEvaluator:
    """Estimate longitudinal treatment effects without calibration leakage.

    Each subject is assigned to one fold. A fold's nuisance mean, covariance,
    detection threshold, and empirical null distribution are learned only from
    control subjects in the other folds. Randomized group labels are then
    permuted to test the multivariate difference in change from baseline.
    """

    def __init__(self, config: TreatmentEffectConfig) -> None:
        """Create an evaluator."""
        self.config = config

    def evaluate(
        self,
        study: Study,
        *,
        baseline: ExpectedVisit,
        follow_ups: tuple[ExpectedVisit, ...],
        treated_subject_ids: tuple[str, ...],
        control_subject_ids: tuple[str, ...],
        treated_label: str = "treated",
        control_label: str = "control",
    ) -> TreatmentEffectReport:
        """Evaluate prespecified treated and control subjects at every follow-up."""
        if not follow_ups:
            raise ValueError("at least one follow-up visit is required")
        follow_up_ids = [visit.visit_id for visit in follow_ups]
        if len(follow_up_ids) != len(set(follow_up_ids)):
            raise ValueError("follow-up visit identifiers must be unique")
        if baseline.visit_id in follow_up_ids:
            raise ValueError("baseline cannot also be a follow-up visit")
        _validate_subject_groups(study, treated_subject_ids, control_subject_ids)
        if len(treated_subject_ids) < self.config.minimum_group_size:
            raise ValueError("insufficient treated subjects")
        if len(control_subject_ids) < max(
            self.config.minimum_group_size,
            self.config.cross_validation_folds + 1,
        ):
            raise ValueError("insufficient control subjects for cross-validation")

        selected = (*treated_subject_ids, *control_subject_ids)
        _validate_visit_order(study, baseline, follow_ups, selected)
        folds = {
            **_subject_folds(
                control_subject_ids,
                self.config.cross_validation_folds,
                self.config.random_seed,
            ),
            **_subject_folds(
                treated_subject_ids,
                self.config.cross_validation_folds,
                self.config.random_seed + 1,
            ),
        }
        random = np.random.default_rng(self.config.random_seed)
        visit_effects: list[VisitTreatmentEffect] = []
        subject_scores: list[CrossValidatedSubjectScore] = []
        fold_calibrations: list[FoldCalibrationProvenance] = []
        excluded: set[str] = set()
        exclusions: list[LongitudinalExclusion] = []
        resolved_channels: tuple[LongitudinalChannel, ...] | None = None
        visit_changes: dict[str, dict[str, NDArray[np.float64]]] = {}
        for follow_up in follow_ups:
            changes, missing, visit_exclusions, channels = _paired_changes(
                study,
                baseline=baseline,
                follow_up=follow_up,
                features=self.config.features,
                subject_ids=selected,
            )
            if resolved_channels is None:
                resolved_channels = channels
            elif channels != resolved_channels:
                raise ValueError("resolved treatment-effect channels differ across follow-ups")
            visit_changes[follow_up.visit_id] = changes
            excluded.update(missing)
            exclusions.extend(visit_exclusions)
            treated = {item: changes[item] for item in treated_subject_ids if item in changes}
            controls = {item: changes[item] for item in control_subject_ids if item in changes}
            if len(treated) < self.config.minimum_group_size:
                raise ValueError(f"insufficient complete treated subjects at {follow_up.visit_id}")
            if len(controls) < max(
                self.config.minimum_group_size,
                self.config.cross_validation_folds + 1,
            ):
                raise ValueError(f"insufficient complete control subjects at {follow_up.visit_id}")
            visit_scores, visit_calibrations = self._cross_validated_scores(
                changes,
                controls=controls,
                folds=folds,
                follow_up_visit_id=follow_up.visit_id,
                treated_ids=set(treated),
                control_ids=set(controls),
                treated_label=treated_label,
                control_label=control_label,
            )
            subject_scores.extend(visit_scores)
            fold_calibrations.extend(visit_calibrations)
            visit_effects.append(
                self._visit_effect(
                    treated,
                    controls,
                    follow_up_visit_id=follow_up.visit_id,
                    random=random,
                )
            )
        assert resolved_channels is not None
        inference_provenance = RandomizedInferenceProvenance(
            study_id=study.study_id,
            config=self.config,
            baseline_visit=baseline,
            follow_up_visits=follow_ups,
            resolved_channels=resolved_channels,
            treated_subject_ids=tuple(sorted(treated_subject_ids)),
            control_subject_ids=tuple(sorted(control_subject_ids)),
            analysis_input_artifact_hash=_analysis_input_artifact_hash(
                study_id=study.study_id,
                baseline=baseline,
                follow_ups=follow_ups,
                config=self.config,
                channels=resolved_channels,
                treated_subject_ids=tuple(sorted(treated_subject_ids)),
                control_subject_ids=tuple(sorted(control_subject_ids)),
                treated_label=treated_label,
                control_label=control_label,
                visit_changes=visit_changes,
            ),
            fold_calibrations=tuple(fold_calibrations),
        )
        return TreatmentEffectReport(
            study_id=study.study_id,
            baseline_visit_id=baseline.visit_id,
            treated_label=treated_label,
            control_label=control_label,
            visit_effects=tuple(visit_effects),
            subject_scores=tuple(subject_scores),
            excluded_subject_ids=tuple(sorted(excluded)),
            exclusions=tuple(exclusions),
            inference_provenance=inference_provenance,
        )

    def _cross_validated_scores(
        self,
        changes: dict[str, NDArray[np.float64]],
        *,
        controls: dict[str, NDArray[np.float64]],
        folds: dict[str, int],
        follow_up_visit_id: str,
        treated_ids: set[str],
        control_ids: set[str],
        treated_label: str,
        control_label: str,
    ) -> tuple[
        list[CrossValidatedSubjectScore],
        tuple[FoldCalibrationProvenance, ...],
    ]:
        fold_models: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        training_ids_by_fold: dict[int, tuple[str, ...]] = {}
        held_out_ids_by_fold: dict[int, tuple[str, ...]] = {}
        null_scores: dict[str, float] = {}
        for fold in range(self.config.cross_validation_folds):
            training_ids = tuple(sorted(item for item in controls if folds[item] != fold))
            training = np.asarray(
                [controls[item] for item in training_ids],
                dtype=np.float64,
            )
            if len(training) < 2:
                raise ValueError(
                    f"insufficient control subjects outside cross-validation fold {fold}"
                )
            mean, inverse = _mean_and_inverse_covariance(training, self.config)
            fold_models[fold] = (mean, inverse)
            training_ids_by_fold[fold] = training_ids
            held_out_ids = tuple(sorted(item for item in controls if folds[item] == fold))
            held_out_ids_by_fold[fold] = held_out_ids
            held_out = np.asarray(
                [controls[item] for item in held_out_ids],
                dtype=np.float64,
            )
            if len(held_out):
                scores = _scores(held_out, mean, inverse)
                null_scores.update(
                    {
                        subject_id: float(value)
                        for subject_id, value in zip(held_out_ids, scores, strict=True)
                    }
                )
        ordered_null_ids = tuple(sorted(null_scores))
        reference_scores = np.asarray(
            [null_scores[subject_id] for subject_id in ordered_null_ids],
            dtype=np.float64,
        )
        threshold = float(
            np.quantile(
                reference_scores,
                1 - self.config.false_alarm_rate,
                method="higher",
            )
        )
        null_artifact_hash = _canonical_hash(
            {
                "schema": "rejuvenationkit.oof-null-scores.v1",
                "follow_up_visit_id": follow_up_visit_id,
                "subject_scores": [
                    {
                        "subject_id": subject_id,
                        "score": null_scores[subject_id],
                    }
                    for subject_id in ordered_null_ids
                ],
            }
        )
        calibrations = tuple(
            _build_fold_calibration(
                config=self.config,
                follow_up_visit_id=follow_up_visit_id,
                fold=fold,
                training_control_subject_ids=training_ids_by_fold[fold],
                held_out_control_subject_ids=held_out_ids_by_fold[fold],
                controls=controls,
                mean=fold_models[fold][0],
                inverse=fold_models[fold][1],
                empirical_threshold=threshold,
                out_of_fold_null_count=len(reference_scores),
                out_of_fold_null_artifact_hash=null_artifact_hash,
            )
            for fold in range(self.config.cross_validation_folds)
        )
        calibration_by_fold = {item.fold: item for item in calibrations}
        output: list[CrossValidatedSubjectScore] = []
        for fold in range(self.config.cross_validation_folds):
            mean, inverse = fold_models[fold]
            calibration = calibration_by_fold[fold]
            for subject_id, vector in sorted(changes.items()):
                if folds[subject_id] != fold:
                    continue
                innovation = vector - mean
                score = max(float(innovation @ inverse @ innovation), 0.0)
                tail = float(
                    (1 + np.count_nonzero(reference_scores >= score)) / (len(reference_scores) + 1)
                )
                group = (
                    treated_label
                    if subject_id in treated_ids
                    else control_label
                    if subject_id in control_ids
                    else "unknown"
                )
                output.append(
                    CrossValidatedSubjectScore(
                        subject_id=subject_id,
                        group=group,
                        follow_up_visit_id=follow_up_visit_id,
                        fold=fold,
                        squared_mahalanobis_distance=score,
                        empirical_tail_probability=tail,
                        detected=score > threshold,
                        calibration_id=calibration.calibration_id,
                        calibration_reference_count=calibration.training_control_count,
                        calibration_reference_artifact_hash=(
                            calibration.training_reference_artifact_hash
                        ),
                        empirical_threshold=calibration.empirical_threshold,
                        out_of_fold_null_count=calibration.out_of_fold_null_count,
                        out_of_fold_null_artifact_hash=(calibration.out_of_fold_null_artifact_hash),
                    )
                )
        return output, calibrations

    def _visit_effect(
        self,
        treated: dict[str, NDArray[np.float64]],
        controls: dict[str, NDArray[np.float64]],
        *,
        follow_up_visit_id: str,
        random: np.random.Generator,
    ) -> VisitTreatmentEffect:
        treated_matrix = np.asarray(list(treated.values()), dtype=np.float64)
        control_matrix = np.asarray(list(controls.values()), dtype=np.float64)
        combined = np.vstack((treated_matrix, control_matrix))
        _, inverse = _mean_and_inverse_covariance(combined, self.config)
        observed_difference = treated_matrix.mean(axis=0) - control_matrix.mean(axis=0)
        statistic = max(
            float(observed_difference @ inverse @ observed_difference),
            0.0,
        )
        group_size = len(treated_matrix)
        exceedances = 0
        for _ in range(self.config.permutations):
            permutation = random.permutation(len(combined))
            permuted_treated = combined[permutation[:group_size]]
            permuted_control = combined[permutation[group_size:]]
            difference = permuted_treated.mean(axis=0) - permuted_control.mean(axis=0)
            permuted_statistic = float(difference @ inverse @ difference)
            exceedances += permuted_statistic >= statistic
        permutation_p_value = (exceedances + 1) / (self.config.permutations + 1)

        bootstrap_differences = np.empty(
            (self.config.bootstrap_samples, len(self.config.features)),
            dtype=np.float64,
        )
        for index in range(self.config.bootstrap_samples):
            treated_sample = treated_matrix[
                random.integers(0, len(treated_matrix), len(treated_matrix))
            ]
            control_sample = control_matrix[
                random.integers(0, len(control_matrix), len(control_matrix))
            ]
            bootstrap_differences[index] = treated_sample.mean(axis=0) - control_sample.mean(axis=0)
        tail = (1 - self.config.confidence_level) / 2
        lower = np.quantile(bootstrap_differences, tail, axis=0)
        upper = np.quantile(bootstrap_differences, 1 - tail, axis=0)
        effects = tuple(
            FeatureTreatmentEffect(
                feature=feature.feature,
                modality=feature.modality,
                treated_mean_change=float(treated_matrix[:, index].mean()),
                control_mean_change=float(control_matrix[:, index].mean()),
                difference_in_differences=float(observed_difference[index]),
                confidence_interval_low=float(lower[index]),
                confidence_interval_high=float(upper[index]),
            )
            for index, feature in enumerate(self.config.features)
        )
        return VisitTreatmentEffect(
            follow_up_visit_id=follow_up_visit_id,
            treated_subjects=len(treated_matrix),
            control_subjects=len(control_matrix),
            omnibus_squared_mahalanobis_distance=statistic,
            permutation_p_value=permutation_p_value,
            effects=effects,
        )


def _validate_subject_groups(
    study: Study,
    treated_subject_ids: tuple[str, ...],
    control_subject_ids: tuple[str, ...],
) -> None:
    treated = set(treated_subject_ids)
    controls = set(control_subject_ids)
    if len(treated) != len(treated_subject_ids) or len(controls) != len(control_subject_ids):
        raise ValueError("subject groups cannot contain duplicate identifiers")
    overlap = treated.intersection(controls)
    if overlap:
        raise ValueError(f"treated and control groups overlap: {sorted(overlap)}")
    unknown = treated.union(controls).difference(subject.subject_id for subject in study.subjects)
    if unknown:
        raise ValueError(f"unknown treatment-effect subjects: {sorted(unknown)}")


def _validate_visit_order(
    study: Study,
    baseline: ExpectedVisit,
    follow_ups: tuple[ExpectedVisit, ...],
    subject_ids: tuple[str, ...],
) -> None:
    subjects = {subject.subject_id: subject for subject in study.subjects}
    for subject_id in subject_ids:
        subject = subjects[subject_id]
        baseline_time = baseline.scheduled_for(subject)
        if baseline_time is None:
            continue
        for follow_up in follow_ups:
            follow_up_time = follow_up.scheduled_for(subject)
            if follow_up_time is not None and follow_up_time <= baseline_time:
                raise ValueError(
                    f"follow-up {follow_up.visit_id} must occur after baseline "
                    f"for subject {subject_id}"
                )


def _subject_folds(subject_ids: tuple[str, ...], folds: int, seed: int) -> dict[str, int]:
    ordered = np.asarray(sorted(subject_ids), dtype=object)
    random = np.random.default_rng(seed)
    random.shuffle(ordered)
    return {str(subject_id): index % folds for index, subject_id in enumerate(ordered)}


def _mean_and_inverse_covariance(
    matrix: NDArray[np.float64],
    config: TreatmentEffectConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = np.asarray(matrix.mean(axis=0), dtype=np.float64)
    if matrix.shape[1] == 1:
        covariance = np.asarray([[float(np.var(matrix[:, 0], ddof=1))]], dtype=np.float64)
    else:
        covariance = np.asarray(np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1)))
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        1 - config.covariance_shrinkage
    ) * covariance + config.covariance_shrinkage * diagonal
    scale = max(float(np.trace(covariance)) / covariance.shape[0], 1.0)
    covariance += np.eye(covariance.shape[0]) * config.covariance_ridge * scale
    return mean, np.asarray(np.linalg.inv(covariance), dtype=np.float64)


def _scores(
    matrix: NDArray[np.float64],
    mean: NDArray[np.float64],
    inverse: NDArray[np.float64],
) -> NDArray[np.float64]:
    centered = matrix - mean
    return np.asarray(
        np.einsum("ij,jk,ik->i", centered, inverse, centered),
        dtype=np.float64,
    )


def _paired_changes(
    study: Study,
    *,
    baseline: ExpectedVisit,
    follow_up: ExpectedVisit,
    features: tuple[LongitudinalChannel | VisitFeature, ...],
    subject_ids: tuple[str, ...],
) -> tuple[
    dict[str, NDArray[np.float64]],
    tuple[str, ...],
    tuple[LongitudinalExclusion, ...],
    tuple[LongitudinalChannel, ...],
]:
    extraction = extract_visit_aligned_values(
        study,
        visits=(baseline, follow_up),
        channels=features,
        subject_ids=subject_ids,
    )
    vector_extraction = complete_visit_vectors(extraction)
    vectors = vector_extraction.vector_map()
    changes: dict[str, NDArray[np.float64]] = {}
    excluded: list[str] = []
    for subject_id in sorted(subject_ids):
        start = vectors.get((subject_id, baseline.visit_id))
        follow = vectors.get((subject_id, follow_up.visit_id))
        if start is None or follow is None:
            excluded.append(subject_id)
        else:
            changes[subject_id] = np.asarray(
                follow.as_array() - start.as_array(),
                dtype=np.float64,
            )
    exact_channels = tuple(item for item in extraction.channels if item is not None)
    return changes, tuple(excluded), vector_extraction.exclusions, exact_channels


def _canonical_hash(payload: object) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible data."""
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _labeled_change_artifact_hash(
    *,
    config: TreatmentEffectConfig,
    follow_up_visit_id: str,
    subject_ids: tuple[str, ...],
    changes: dict[str, NDArray[np.float64]],
) -> str:
    """Bind each calibration vector to its control identity and configuration."""
    return _canonical_hash(
        {
            "schema": "rejuvenationkit.fold-reference-changes.v1",
            "config": config.model_dump(mode="json"),
            "follow_up_visit_id": follow_up_visit_id,
            "subject_changes": [
                {
                    "subject_id": subject_id,
                    "change": [float(value) for value in changes[subject_id]],
                }
                for subject_id in subject_ids
            ],
        }
    )


def _build_fold_calibration(
    *,
    config: TreatmentEffectConfig,
    follow_up_visit_id: str,
    fold: int,
    training_control_subject_ids: tuple[str, ...],
    held_out_control_subject_ids: tuple[str, ...],
    controls: dict[str, NDArray[np.float64]],
    mean: NDArray[np.float64],
    inverse: NDArray[np.float64],
    empirical_threshold: float,
    out_of_fold_null_count: int,
    out_of_fold_null_artifact_hash: str,
) -> FoldCalibrationProvenance:
    """Create a self-checking compact fold calibration record."""
    training_hash = _labeled_change_artifact_hash(
        config=config,
        follow_up_visit_id=follow_up_visit_id,
        subject_ids=training_control_subject_ids,
        changes=controls,
    )
    model_hash = _canonical_hash(
        {
            "schema": "rejuvenationkit.fold-score-model.v1",
            "config": config.model_dump(mode="json"),
            "follow_up_visit_id": follow_up_visit_id,
            "fold": fold,
            "training_reference_artifact_hash": training_hash,
            "mean": [float(value) for value in mean],
            "inverse_covariance": [[float(value) for value in row] for row in inverse],
        }
    )
    provisional = FoldCalibrationProvenance.model_construct(
        calibration_id="0" * 64,
        follow_up_visit_id=follow_up_visit_id,
        fold=fold,
        training_control_subject_ids=training_control_subject_ids,
        training_control_count=len(training_control_subject_ids),
        held_out_control_subject_ids=held_out_control_subject_ids,
        held_out_control_count=len(held_out_control_subject_ids),
        training_reference_artifact_hash=training_hash,
        fold_model_artifact_hash=model_hash,
        empirical_threshold=empirical_threshold,
        out_of_fold_null_count=out_of_fold_null_count,
        out_of_fold_null_artifact_hash=out_of_fold_null_artifact_hash,
        covariance_method="empirical covariance with diagonal shrinkage and scaled ridge",
    )
    calibration_id = _canonical_hash(
        provisional.model_dump(mode="json", exclude={"calibration_id"})
    )
    return FoldCalibrationProvenance(
        calibration_id=calibration_id,
        follow_up_visit_id=follow_up_visit_id,
        fold=fold,
        training_control_subject_ids=training_control_subject_ids,
        training_control_count=len(training_control_subject_ids),
        held_out_control_subject_ids=held_out_control_subject_ids,
        held_out_control_count=len(held_out_control_subject_ids),
        training_reference_artifact_hash=training_hash,
        fold_model_artifact_hash=model_hash,
        empirical_threshold=empirical_threshold,
        out_of_fold_null_count=out_of_fold_null_count,
        out_of_fold_null_artifact_hash=out_of_fold_null_artifact_hash,
    )


def _analysis_input_artifact_hash(
    *,
    study_id: str,
    baseline: ExpectedVisit,
    follow_ups: tuple[ExpectedVisit, ...],
    config: TreatmentEffectConfig,
    channels: tuple[LongitudinalChannel, ...],
    treated_subject_ids: tuple[str, ...],
    control_subject_ids: tuple[str, ...],
    treated_label: str,
    control_label: str,
    visit_changes: dict[str, dict[str, NDArray[np.float64]]],
) -> str:
    """Hash all labeled analysis vectors without embedding bootstrap/null arrays."""
    return _canonical_hash(
        {
            "schema": "rejuvenationkit.randomized-treatment-analysis.v1",
            "study_id": study_id,
            "baseline_visit": baseline.model_dump(mode="json"),
            "follow_up_visits": [visit.model_dump(mode="json") for visit in follow_ups],
            "config": config.model_dump(mode="json"),
            "resolved_channels": [channel.model_dump(mode="json") for channel in channels],
            "treated_subject_ids": list(treated_subject_ids),
            "control_subject_ids": list(control_subject_ids),
            "treated_label": treated_label,
            "control_label": control_label,
            "visit_changes": [
                {
                    "follow_up_visit_id": visit.visit_id,
                    "subject_changes": [
                        {
                            "subject_id": subject_id,
                            "change": [
                                float(value) for value in visit_changes[visit.visit_id][subject_id]
                            ],
                        }
                        for subject_id in sorted(visit_changes[visit.visit_id])
                    ],
                }
                for visit in follow_ups
            ],
        }
    )
