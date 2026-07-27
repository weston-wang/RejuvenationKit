"""Sequential multimodal treatment-response detection."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rejuvenationkit.qc import ExpectedVisit, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject

_DAYS_PER_YEAR = 365.2425


class SequentialDetectionConfig(BaseModel):
    """Configuration for reference-conditioned sequential detection."""

    model_config = ConfigDict(frozen=True)

    features: tuple[VisitFeature, ...] = Field(min_length=2)
    covariance_shrinkage: float = Field(default=0.20, ge=0, le=1)
    covariance_ridge: float = Field(default=1e-9, gt=0)
    false_alarm_rate: float = Field(default=0.05, gt=0, lt=0.5)
    minimum_reference_subjects: int = Field(default=30, ge=5)
    persistence_crossings: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def require_unique_features(self) -> SequentialDetectionConfig:
        """Reject duplicate feature/modality channels."""
        keys = [(item.feature, item.modality) for item in self.features]
        if len(keys) != len(set(keys)):
            raise ValueError("sequential detection features must be unique")
        return self


class SequentialDetectionModel(BaseModel):
    """Fitted normal-trajectory dynamics and subject-level threshold."""

    model_config = ConfigDict(frozen=True)

    feature_names: tuple[str, ...]
    feature_modalities: tuple[Modality | None, ...]
    reference_subjects: int = Field(ge=1)
    reference_transitions: int = Field(ge=1)
    mean_change_per_year: tuple[float, ...]
    innovation_covariance_per_year: tuple[tuple[float, ...], ...]
    maximum_cumulative_score_threshold: float = Field(ge=0)
    false_alarm_rate: float = Field(gt=0, lt=0.5)


class SequentialDetectionPoint(BaseModel):
    """Evidence accumulated through one observed transition."""

    model_config = ConfigDict(frozen=True)

    from_visit_id: str
    to_visit_id: str
    elapsed_years: float = Field(gt=0)
    interval_score: float = Field(ge=0)
    cumulative_score: float = Field(ge=0)
    empirical_tail_probability: float = Field(gt=0, le=1)
    threshold_crossed: bool


class ModalityEvidence(BaseModel):
    """Peak cumulative evidence calculated within one modality."""

    model_config = ConfigDict(frozen=True)

    modality: Modality | None
    channels: int = Field(ge=1)
    score: float = Field(ge=0)


class SubjectSequentialDetection(BaseModel):
    """Onset and persistence assessment for one subject trajectory."""

    model_config = ConfigDict(frozen=True)

    subject_id: str
    points: tuple[SequentialDetectionPoint, ...]
    detected: bool
    onset_visit_id: str | None = None
    persistent: bool = False
    transient: bool = False
    peak_cumulative_score: float = Field(ge=0)
    peak_modality_evidence: tuple[ModalityEvidence, ...] = ()


class SequentialDetectionReport(BaseModel):
    """Sequential results for scored subjects."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    visit_ids: tuple[str, ...]
    model: SequentialDetectionModel
    results: tuple[SubjectSequentialDetection, ...]
    excluded_subject_ids: tuple[str, ...] = ()

    def results_frame(self) -> pd.DataFrame:
        """Return one summary row per scored subject."""
        return pd.DataFrame(
            {
                "subject_id": item.subject_id,
                "transitions": len(item.points),
                "detected": item.detected,
                "onset_visit_id": item.onset_visit_id,
                "persistent": item.persistent,
                "transient": item.transient,
                "peak_cumulative_score": item.peak_cumulative_score,
            }
            for item in self.results
        )

    def trajectory_frame(self) -> pd.DataFrame:
        """Return one row per subject transition."""
        return pd.DataFrame(
            {
                "subject_id": result.subject_id,
                **point.model_dump(mode="json"),
            }
            for result in self.results
            for point in result.points
        )


class SequentialTreatmentResponseDetector:
    """Detect sustained departures from a fitted reference trajectory.

    The detector models each channel as a drift-plus-random-walk process.
    Irregular-time increments are centered by the reference drift and divided
    by the square root of elapsed time. Their covariance is shrunk and whitened.
    Coherent cumulative evidence is the energy of the running mean whitened
    innovation, with a threshold calibrated from each reference subject's
    maximum score across the full trajectory.
    """

    def __init__(self, config: SequentialDetectionConfig) -> None:
        """Create an unfitted sequential detector."""
        self.config = config
        self.model_: SequentialDetectionModel | None = None
        self._mean_rate: NDArray[np.float64] | None = None
        self._covariance: NDArray[np.float64] | None = None
        self._cholesky: NDArray[np.float64] | None = None
        self._reference_maximum_scores: NDArray[np.float64] | None = None

    def fit(
        self,
        study: Study,
        *,
        visits: tuple[ExpectedVisit, ...],
        reference_subject_ids: tuple[str, ...],
    ) -> SequentialTreatmentResponseDetector:
        """Fit reference drift, innovation covariance, and sequential threshold."""
        _validate_visits(visits)
        trajectories, excluded = _complete_trajectories(
            study,
            visits=visits,
            features=self.config.features,
            subject_ids=reference_subject_ids,
            require_all_visits=True,
        )
        if len(trajectories) < self.config.minimum_reference_subjects:
            raise ValueError(
                "insufficient complete reference trajectories: "
                f"{len(trajectories)} < {self.config.minimum_reference_subjects}; "
                f"excluded={len(excluded)}"
            )
        rates: list[NDArray[np.float64]] = []
        for trajectory in trajectories.values():
            for first, second in pairwise(trajectory):
                elapsed = _elapsed_years(first[1], second[1])
                rates.append((second[2] - first[2]) / elapsed)
        mean_rate = np.asarray(np.mean(rates, axis=0), dtype=np.float64)
        innovations: list[NDArray[np.float64]] = []
        subject_innovations: dict[str, list[NDArray[np.float64]]] = {}
        for subject_id, trajectory in trajectories.items():
            sequence: list[NDArray[np.float64]] = []
            for first, second in pairwise(trajectory):
                elapsed = _elapsed_years(first[1], second[1])
                innovation = (second[2] - first[2] - mean_rate * elapsed) / math.sqrt(elapsed)
                innovations.append(innovation)
                sequence.append(innovation)
            subject_innovations[subject_id] = sequence
        matrix = np.asarray(innovations, dtype=np.float64)
        covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
        diagonal = np.diag(np.diag(covariance))
        shrinkage = self.config.covariance_shrinkage
        covariance = (1 - shrinkage) * covariance + shrinkage * diagonal
        scale = max(float(np.trace(covariance)) / covariance.shape[0], 1.0)
        covariance = np.asarray(
            covariance + np.eye(covariance.shape[0]) * self.config.covariance_ridge * scale,
            dtype=np.float64,
        )
        cholesky = np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)
        maximum_scores = np.asarray(
            [
                max(_cumulative_scores(sequence, cholesky))
                for sequence in subject_innovations.values()
            ],
            dtype=np.float64,
        )
        threshold = float(
            np.quantile(
                maximum_scores,
                1 - self.config.false_alarm_rate,
                method="higher",
            )
        )
        self._mean_rate = mean_rate
        self._covariance = covariance
        self._cholesky = cholesky
        self._reference_maximum_scores = maximum_scores
        self.model_ = SequentialDetectionModel(
            feature_names=tuple(item.feature for item in self.config.features),
            feature_modalities=tuple(item.modality for item in self.config.features),
            reference_subjects=len(trajectories),
            reference_transitions=len(innovations),
            mean_change_per_year=tuple(float(value) for value in mean_rate),
            innovation_covariance_per_year=tuple(
                tuple(float(value) for value in row) for row in covariance
            ),
            maximum_cumulative_score_threshold=threshold,
            false_alarm_rate=self.config.false_alarm_rate,
        )
        return self

    def score(
        self,
        study: Study,
        *,
        visits: tuple[ExpectedVisit, ...],
        subject_ids: tuple[str, ...],
    ) -> SequentialDetectionReport:
        """Score partially observed trajectories with at least two complete visits."""
        _validate_visits(visits)
        if (
            self.model_ is None
            or self._mean_rate is None
            or self._covariance is None
            or self._cholesky is None
            or self._reference_maximum_scores is None
        ):
            raise RuntimeError("fit must be called before score")
        trajectories, excluded = _complete_trajectories(
            study,
            visits=visits,
            features=self.config.features,
            subject_ids=subject_ids,
            require_all_visits=False,
        )
        results = tuple(
            self._score_subject(subject_id, trajectory)
            for subject_id, trajectory in sorted(trajectories.items())
        )
        return SequentialDetectionReport(
            study_id=study.study_id,
            visit_ids=tuple(visit.visit_id for visit in visits),
            model=self.model_,
            results=results,
            excluded_subject_ids=excluded,
        )

    def _score_subject(
        self,
        subject_id: str,
        trajectory: list[tuple[str, datetime, NDArray[np.float64]]],
    ) -> SubjectSequentialDetection:
        assert self.model_ is not None
        assert self._mean_rate is not None
        assert self._covariance is not None
        assert self._cholesky is not None
        assert self._reference_maximum_scores is not None
        innovations: list[NDArray[np.float64]] = []
        points: list[SequentialDetectionPoint] = []
        cumulative_raw: NDArray[np.float64] = np.zeros(
            len(self.config.features),
            dtype=np.float64,
        )
        peak_raw: NDArray[np.float64] = cumulative_raw.copy()
        peak_score = 0.0
        for first, second in pairwise(trajectory):
            elapsed = _elapsed_years(first[1], second[1])
            raw = (second[2] - first[2] - self._mean_rate * elapsed) / math.sqrt(elapsed)
            innovations.append(raw)
            cumulative_raw += raw
            whitened = np.linalg.solve(self._cholesky, raw)
            interval_score = float(whitened @ whitened)
            cumulative_score = _cumulative_score(innovations, self._cholesky)
            if cumulative_score > peak_score:
                peak_score = cumulative_score
                peak_raw = np.asarray(
                    cumulative_raw / math.sqrt(len(innovations)),
                    dtype=np.float64,
                )
            tail = float(
                (1 + np.count_nonzero(self._reference_maximum_scores >= cumulative_score))
                / (len(self._reference_maximum_scores) + 1)
            )
            points.append(
                SequentialDetectionPoint(
                    from_visit_id=first[0],
                    to_visit_id=second[0],
                    elapsed_years=elapsed,
                    interval_score=max(interval_score, 0.0),
                    cumulative_score=max(cumulative_score, 0.0),
                    empirical_tail_probability=tail,
                    threshold_crossed=(
                        cumulative_score > self.model_.maximum_cumulative_score_threshold
                    ),
                )
            )
        onset = next((point.to_visit_id for point in points if point.threshold_crossed), None)
        persistent = _persistent_at_end(points, self.config.persistence_crossings)
        detected = onset is not None
        transient = detected and any(
            point.threshold_crossed and not later.threshold_crossed
            for index, point in enumerate(points)
            for later in points[index + 1 :]
        )
        return SubjectSequentialDetection(
            subject_id=subject_id,
            points=tuple(points),
            detected=detected,
            onset_visit_id=onset,
            persistent=persistent,
            transient=transient,
            peak_cumulative_score=max(peak_score, 0.0),
            peak_modality_evidence=_modality_evidence(
                peak_raw,
                self._covariance,
                self.config.features,
            ),
        )


def _validate_visits(visits: tuple[ExpectedVisit, ...]) -> None:
    if len(visits) < 3:
        raise ValueError("sequential detection requires at least three expected visits")
    if len({visit.visit_id for visit in visits}) != len(visits):
        raise ValueError("sequential visit identifiers must be unique")


def _complete_trajectories(
    study: Study,
    *,
    visits: tuple[ExpectedVisit, ...],
    features: tuple[VisitFeature, ...],
    subject_ids: tuple[str, ...],
    require_all_visits: bool,
) -> tuple[dict[str, list[tuple[str, datetime, NDArray[np.float64]]]], tuple[str, ...]]:
    known = {subject.subject_id for subject in study.subjects}
    unknown = set(subject_ids).difference(known)
    if unknown:
        raise ValueError(f"unknown sequential detection subjects: {sorted(unknown)}")
    subjects = {subject.subject_id: subject for subject in study.subjects}
    values = {
        visit.visit_id: _visit_values(study, visit, features, subjects, set(subject_ids))
        for visit in visits
    }
    trajectories: dict[str, list[tuple[str, datetime, NDArray[np.float64]]]] = {}
    excluded: list[str] = []
    for subject_id in sorted(subject_ids):
        subject = subjects[subject_id]
        trajectory: list[tuple[str, datetime, NDArray[np.float64]]] = []
        for visit in visits:
            target = visit.scheduled_for(subject)
            vector = _vector(subject_id, features, values[visit.visit_id])
            if target is not None and vector is not None:
                trajectory.append((visit.visit_id, target, vector))
        enough = len(trajectory) == len(visits) if require_all_visits else len(trajectory) >= 2
        if enough:
            trajectories[subject_id] = trajectory
        else:
            excluded.append(subject_id)
    return trajectories, tuple(excluded)


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
    return {key: sum(items) / len(items) for key, items in accumulated.items()}


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


def _matches(row: Observation, feature: VisitFeature) -> bool:
    return row.feature == feature.feature and (
        feature.modality is None or row.modality is feature.modality
    )


def _visit_applies(visit: ExpectedVisit, subject: Subject) -> bool:
    if not visit.subject_ids and not visit.cohorts:
        return True
    return subject.subject_id in visit.subject_ids or subject.cohort in visit.cohorts


def _elapsed_years(first: datetime, second: datetime) -> float:
    difference = second - first
    years = difference.total_seconds() / timedelta(days=_DAYS_PER_YEAR).total_seconds()
    if years <= 0:
        raise ValueError("sequential visits must be chronological for every subject")
    return years


def _cumulative_scores(
    innovations: list[NDArray[np.float64]],
    cholesky: NDArray[np.float64],
) -> list[float]:
    return [
        _cumulative_score(innovations[:index], cholesky) for index in range(1, len(innovations) + 1)
    ]


def _cumulative_score(
    innovations: list[NDArray[np.float64]],
    cholesky: NDArray[np.float64],
) -> float:
    cumulative = np.sum(innovations, axis=0) / math.sqrt(len(innovations))
    whitened = np.linalg.solve(cholesky, cumulative)
    return float(whitened @ whitened)


def _persistent_at_end(
    points: list[SequentialDetectionPoint],
    required_crossings: int,
) -> bool:
    run = 0
    for point in reversed(points):
        if not point.threshold_crossed:
            break
        run += 1
    return run >= required_crossings


def _modality_evidence(
    cumulative_innovation: NDArray[np.float64],
    covariance: NDArray[np.float64],
    features: tuple[VisitFeature, ...],
) -> tuple[ModalityEvidence, ...]:
    modalities = sorted(
        {feature.modality for feature in features},
        key=lambda item: "" if item is None else item.value,
    )
    evidence: list[ModalityEvidence] = []
    for modality in modalities:
        indices = [index for index, feature in enumerate(features) if feature.modality is modality]
        vector = cumulative_innovation[indices]
        subcovariance = covariance[np.ix_(indices, indices)]
        score = float(vector @ np.linalg.inv(subcovariance) @ vector)
        evidence.append(
            ModalityEvidence(
                modality=modality,
                channels=len(indices),
                score=max(score, 0.0),
            )
        )
    return tuple(evidence)
