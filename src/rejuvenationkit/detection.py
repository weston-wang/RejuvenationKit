"""DSP-style multivariate detection of longitudinal biological change."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject


class ChangeDetectionConfig(BaseModel):
    """Configuration for covariance-aware longitudinal change detection."""

    model_config = ConfigDict(frozen=True)

    features: tuple[VisitFeature, ...] = Field(min_length=2)
    covariance_shrinkage: float = Field(default=0.20, ge=0, le=1)
    covariance_ridge: float = Field(default=1e-9, gt=0)
    false_alarm_rate: float = Field(default=0.05, gt=0, lt=0.5)
    minimum_reference_subjects: int = Field(default=20, ge=3)

    @model_validator(mode="after")
    def require_unique_features(self) -> ChangeDetectionConfig:
        """Reject duplicate feature/modality channels."""
        keys = [(item.feature, item.modality) for item in self.features]
        if len(keys) != len(set(keys)):
            raise ValueError("detection features must be unique")
        return self


class ChangeDetectionModel(BaseModel):
    """Fitted reference-change distribution and calibrated threshold."""

    model_config = ConfigDict(frozen=True)

    feature_names: tuple[str, ...]
    feature_modalities: tuple[Modality | None, ...]
    reference_subjects: int = Field(ge=1)
    mean_change: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    threshold: float = Field(ge=0)
    false_alarm_rate: float = Field(gt=0, lt=0.5)


class SubjectChangeDetection(BaseModel):
    """One subject's covariance-normalized longitudinal change score."""

    model_config = ConfigDict(frozen=True)

    subject_id: str
    change: tuple[float, ...]
    innovation: tuple[float, ...]
    whitened_innovation: tuple[float, ...]
    squared_mahalanobis_distance: float = Field(ge=0)
    empirical_tail_probability: float = Field(gt=0, le=1)
    detected: bool


class ChangeDetectionReport(BaseModel):
    """Scored subjects under one fitted multivariate detector."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    baseline_visit_id: str
    follow_up_visit_id: str
    model: ChangeDetectionModel
    results: tuple[SubjectChangeDetection, ...]
    excluded_subject_ids: tuple[str, ...] = ()

    def results_frame(self) -> pd.DataFrame:
        """Return subject scores as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.results)


class MultivariateChangeDetector:
    """Detect unusual paired changes relative to a reference population.

    The detector centers paired feature changes, estimates their covariance,
    shrinks cross-channel covariance toward a diagonal model, whitens each
    innovation, and thresholds squared Mahalanobis distance at an empirical
    reference quantile.
    """

    def __init__(self, config: ChangeDetectionConfig) -> None:
        """Create an unfitted detector."""
        self.config = config
        self.model_: ChangeDetectionModel | None = None
        self._inverse_covariance: NDArray[np.float64] | None = None
        self._cholesky: NDArray[np.float64] | None = None
        self._reference_scores: NDArray[np.float64] | None = None

    def fit(
        self,
        study: Study,
        *,
        baseline: ExpectedVisit,
        follow_up: ExpectedVisit,
        reference_subject_ids: tuple[str, ...] | None = None,
    ) -> MultivariateChangeDetector:
        """Fit the normal-change model using complete paired reference subjects."""
        changes, _ = _paired_changes(
            study,
            baseline=baseline,
            follow_up=follow_up,
            features=self.config.features,
            subject_ids=reference_subject_ids,
        )
        if len(changes) < self.config.minimum_reference_subjects:
            raise ValueError(
                "insufficient complete reference subjects: "
                f"{len(changes)} < {self.config.minimum_reference_subjects}"
            )
        matrix = np.asarray(list(changes.values()), dtype=float)
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        empirical = np.atleast_2d(np.cov(centered, rowvar=False, ddof=1))
        diagonal = np.diag(np.diag(empirical))
        shrinkage = self.config.covariance_shrinkage
        covariance = (1 - shrinkage) * empirical + shrinkage * diagonal
        scale = max(float(np.trace(covariance)) / covariance.shape[0], 1.0)
        covariance = covariance + np.eye(covariance.shape[0]) * (
            self.config.covariance_ridge * scale
        )
        if not np.isfinite(covariance).all():
            raise ValueError("reference covariance contains non-finite values")
        cholesky = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)
        inverse = np.asarray(np.linalg.inv(covariance), dtype=np.float64)
        scores = np.asarray(
            np.einsum("ij,jk,ik->i", centered, inverse, centered),
            dtype=np.float64,
        )
        threshold = float(np.quantile(scores, 1 - self.config.false_alarm_rate, method="higher"))
        self._inverse_covariance = inverse
        self._cholesky = cholesky
        self._reference_scores = scores
        self.model_ = ChangeDetectionModel(
            feature_names=tuple(item.feature for item in self.config.features),
            feature_modalities=tuple(item.modality for item in self.config.features),
            reference_subjects=len(changes),
            mean_change=tuple(float(value) for value in mean),
            covariance=tuple(tuple(float(value) for value in row) for row in covariance),
            threshold=threshold,
            false_alarm_rate=self.config.false_alarm_rate,
        )
        return self

    def score(
        self,
        study: Study,
        *,
        baseline: ExpectedVisit,
        follow_up: ExpectedVisit,
        subject_ids: tuple[str, ...] | None = None,
    ) -> ChangeDetectionReport:
        """Score complete paired subjects and report subjects lacking a full vector."""
        if (
            self.model_ is None
            or self._inverse_covariance is None
            or self._cholesky is None
            or self._reference_scores is None
        ):
            raise RuntimeError("fit must be called before score")
        changes, excluded = _paired_changes(
            study,
            baseline=baseline,
            follow_up=follow_up,
            features=self.config.features,
            subject_ids=subject_ids,
        )
        mean = np.asarray(self.model_.mean_change)
        results: list[SubjectChangeDetection] = []
        for subject_id, values in sorted(changes.items()):
            change = np.asarray(values)
            innovation = change - mean
            whitened = np.linalg.solve(self._cholesky, innovation)
            score = float(innovation @ self._inverse_covariance @ innovation)
            tail = float(
                (1 + np.count_nonzero(self._reference_scores >= score))
                / (len(self._reference_scores) + 1)
            )
            results.append(
                SubjectChangeDetection(
                    subject_id=subject_id,
                    change=tuple(float(value) for value in change),
                    innovation=tuple(float(value) for value in innovation),
                    whitened_innovation=tuple(float(value) for value in whitened),
                    squared_mahalanobis_distance=max(score, 0.0),
                    empirical_tail_probability=tail,
                    detected=score > self.model_.threshold,
                )
            )
        return ChangeDetectionReport(
            study_id=study.study_id,
            baseline_visit_id=baseline.visit_id,
            follow_up_visit_id=follow_up.visit_id,
            model=self.model_,
            results=tuple(results),
            excluded_subject_ids=excluded,
        )


def _paired_changes(
    study: Study,
    *,
    baseline: ExpectedVisit,
    follow_up: ExpectedVisit,
    features: tuple[VisitFeature, ...],
    subject_ids: tuple[str, ...] | None,
) -> tuple[dict[str, tuple[float, ...]], tuple[str, ...]]:
    selected = (
        {subject.subject_id for subject in study.subjects}
        if subject_ids is None
        else set(subject_ids)
    )
    known = {subject.subject_id for subject in study.subjects}
    unknown = selected.difference(known)
    if unknown:
        raise ValueError(f"unknown detection subjects: {sorted(unknown)}")
    subjects = {subject.subject_id: subject for subject in study.subjects}
    baseline_values = _visit_values(study, baseline, features, subjects, selected)
    follow_up_values = _visit_values(study, follow_up, features, subjects, selected)
    changes: dict[str, tuple[float, ...]] = {}
    excluded: list[str] = []
    for subject_id in sorted(selected):
        baseline_vector = _vector(subject_id, features, baseline_values)
        follow_up_vector = _vector(subject_id, features, follow_up_values)
        if baseline_vector is None or follow_up_vector is None:
            excluded.append(subject_id)
            continue
        changes[subject_id] = tuple(
            follow - start for start, follow in zip(baseline_vector, follow_up_vector, strict=True)
        )
    return changes, tuple(excluded)


def _visit_values(
    study: Study,
    visit: ExpectedVisit,
    features: tuple[VisitFeature, ...],
    subjects: dict[str, Subject],
    selected: set[str],
) -> dict[tuple[str, str, Modality | None], float]:
    accumulated: dict[tuple[str, str, Modality | None], list[float]] = defaultdict(list)
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
            if _matches(row, feature):
                accumulated[(row.subject_id, feature.feature, feature.modality)].append(row.value)
    return {key: sum(values) / len(values) for key, values in accumulated.items()}


def _vector(
    subject_id: str,
    features: tuple[VisitFeature, ...],
    values: dict[tuple[str, str, Modality | None], float],
) -> tuple[float, ...] | None:
    vector: list[float] = []
    for feature in features:
        value = values.get((subject_id, feature.feature, feature.modality))
        if value is None:
            return None
        vector.append(value)
    return tuple(vector)


def _matches(row: Observation, feature: VisitFeature) -> bool:
    return row.feature == feature.feature and (
        feature.modality is None or row.modality is feature.modality
    )


def _visit_applies(visit: ExpectedVisit, subject: Subject) -> bool:
    if not visit.subject_ids and not visit.cohorts:
        return True
    return subject.subject_id in visit.subject_ids or subject.cohort in visit.cohorts
