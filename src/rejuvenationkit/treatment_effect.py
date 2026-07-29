"""Leakage-safe inference for randomized longitudinal treatment studies."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Study, Subject


class TreatmentEffectConfig(BaseModel):
    """Configuration for cross-validated treatment-effect inference."""

    model_config = ConfigDict(frozen=True)

    features: tuple[VisitFeature, ...] = Field(min_length=1)
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
        """Reject duplicate feature/modality channels."""
        keys = [(item.feature, item.modality) for item in self.features]
        if len(keys) != len(set(keys)):
            raise ValueError("treatment-effect features must be unique")
        return self


class CrossValidatedSubjectScore(BaseModel):
    """A subject score produced without using that subject for calibration."""

    model_config = ConfigDict(frozen=True)

    subject_id: str
    group: str
    follow_up_visit_id: str
    fold: int = Field(ge=0)
    squared_mahalanobis_distance: float = Field(ge=0)
    empirical_tail_probability: float = Field(gt=0, le=1)
    detected: bool


class FeatureTreatmentEffect(BaseModel):
    """Difference-in-differences estimate for one measured channel."""

    model_config = ConfigDict(frozen=True)

    feature: str
    modality: Modality | None
    treated_mean_change: float
    control_mean_change: float
    difference_in_differences: float
    confidence_interval_low: float
    confidence_interval_high: float


class VisitTreatmentEffect(BaseModel):
    """Multivariate and channel-level effects at one follow-up visit."""

    model_config = ConfigDict(frozen=True)

    follow_up_visit_id: str
    treated_subjects: int = Field(ge=1)
    control_subjects: int = Field(ge=1)
    omnibus_squared_mahalanobis_distance: float = Field(ge=0)
    permutation_p_value: float = Field(gt=0, le=1)
    effects: tuple[FeatureTreatmentEffect, ...]


class TreatmentEffectReport(BaseModel):
    """Leakage-safe subject scores and randomized group estimates."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    baseline_visit_id: str
    treated_label: str
    control_label: str
    visit_effects: tuple[VisitTreatmentEffect, ...]
    subject_scores: tuple[CrossValidatedSubjectScore, ...]
    excluded_subject_ids: tuple[str, ...] = ()

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
        excluded: set[str] = set()
        for follow_up in follow_ups:
            changes, missing = _paired_changes(
                study,
                baseline=baseline,
                follow_up=follow_up,
                features=self.config.features,
                subject_ids=selected,
            )
            excluded.update(missing)
            treated = {item: changes[item] for item in treated_subject_ids if item in changes}
            controls = {item: changes[item] for item in control_subject_ids if item in changes}
            if len(treated) < self.config.minimum_group_size:
                raise ValueError(f"insufficient complete treated subjects at {follow_up.visit_id}")
            if len(controls) < max(
                self.config.minimum_group_size,
                self.config.cross_validation_folds + 1,
            ):
                raise ValueError(f"insufficient complete control subjects at {follow_up.visit_id}")
            subject_scores.extend(
                self._cross_validated_scores(
                    changes,
                    controls=controls,
                    folds=folds,
                    follow_up_visit_id=follow_up.visit_id,
                    treated_ids=set(treated),
                    control_ids=set(controls),
                    treated_label=treated_label,
                    control_label=control_label,
                )
            )
            visit_effects.append(
                self._visit_effect(
                    treated,
                    controls,
                    follow_up_visit_id=follow_up.visit_id,
                    random=random,
                )
            )
        return TreatmentEffectReport(
            study_id=study.study_id,
            baseline_visit_id=baseline.visit_id,
            treated_label=treated_label,
            control_label=control_label,
            visit_effects=tuple(visit_effects),
            subject_scores=tuple(subject_scores),
            excluded_subject_ids=tuple(sorted(excluded)),
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
    ) -> list[CrossValidatedSubjectScore]:
        fold_models: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        null_scores: list[float] = []
        for fold in range(self.config.cross_validation_folds):
            training = np.asarray(
                [vector for item, vector in controls.items() if folds[item] != fold],
                dtype=np.float64,
            )
            if len(training) < 2:
                raise ValueError(
                    f"insufficient control subjects outside cross-validation fold {fold}"
                )
            mean, inverse = _mean_and_inverse_covariance(training, self.config)
            fold_models[fold] = (mean, inverse)
            held_out = np.asarray(
                [vector for item, vector in controls.items() if folds[item] == fold],
                dtype=np.float64,
            )
            if len(held_out):
                null_scores.extend(float(value) for value in _scores(held_out, mean, inverse))
        reference_scores = np.asarray(null_scores, dtype=np.float64)
        threshold = float(
            np.quantile(
                reference_scores,
                1 - self.config.false_alarm_rate,
                method="higher",
            )
        )
        output: list[CrossValidatedSubjectScore] = []
        for fold in range(self.config.cross_validation_folds):
            mean, inverse = fold_models[fold]
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
                    )
                )
        return output

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
    features: tuple[VisitFeature, ...],
    subject_ids: tuple[str, ...],
) -> tuple[dict[str, NDArray[np.float64]], tuple[str, ...]]:
    selected = set(subject_ids)
    subjects = {subject.subject_id: subject for subject in study.subjects}
    baseline_values = _visit_values(study, baseline, features, subjects, selected)
    follow_up_values = _visit_values(study, follow_up, features, subjects, selected)
    changes: dict[str, NDArray[np.float64]] = {}
    excluded: list[str] = []
    for subject_id in sorted(selected):
        start = _vector(subject_id, features, baseline_values)
        follow = _vector(subject_id, features, follow_up_values)
        if start is None or follow is None:
            excluded.append(subject_id)
        else:
            changes[subject_id] = follow - start
    return changes, tuple(excluded)


def _visit_values(
    study: Study,
    visit: ExpectedVisit,
    features: tuple[VisitFeature, ...],
    subjects: dict[str, Subject],
    selected: set[str],
) -> dict[tuple[str, str, Modality | None], float]:
    accumulated: defaultdict[tuple[str, str, Modality | None], list[float]] = defaultdict(list)
    for row in study.observations:
        if row.subject_id not in selected or not math.isfinite(row.value):
            continue
        subject = subjects[row.subject_id]
        if not _visit_applies(visit, subject):
            continue
        window = visit.window_for(subject)
        if window is None or not window[0] <= row.timestamp <= window[1]:
            continue
        for feature in features:
            if row.feature == feature.feature and (
                feature.modality is None or row.modality is feature.modality
            ):
                accumulated[(row.subject_id, feature.feature, feature.modality)].append(row.value)
    return {key: sum(values) / len(values) for key, values in accumulated.items()}


def _vector(
    subject_id: str,
    features: tuple[VisitFeature, ...],
    values: dict[tuple[str, str, Modality | None], float],
) -> NDArray[np.float64] | None:
    vector: list[float] = []
    for feature in features:
        value = values.get((subject_id, feature.feature, feature.modality))
        if value is None:
            return None
        vector.append(value)
    return np.asarray(vector, dtype=np.float64)


def _visit_applies(visit: ExpectedVisit, subject: Subject) -> bool:
    if not visit.subject_ids and not visit.cohorts:
        return True
    return subject.subject_id in visit.subject_ids or subject.cohort in visit.cohorts
