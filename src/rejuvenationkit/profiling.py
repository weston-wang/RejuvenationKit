"""Phase 1 analysis-readiness profiles for longitudinal studies."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import pairwise

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from rejuvenationkit.qc import ExpectedVisit, QCConfig, VisitFeature
from rejuvenationkit.schemas import Modality, Observation, Study, Subject


class VisitCoverage(BaseModel):
    """Observed coverage for one required feature at one scheduled visit."""

    model_config = ConfigDict(frozen=True)

    visit_id: str
    cohort: str
    feature: str
    modality: Modality | None
    eligible_subjects: int = Field(ge=0)
    observed_subjects: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0, le=1)


class VisitRetention(BaseModel):
    """Complete-case retention between two consecutive expected visits."""

    model_config = ConfigDict(frozen=True)

    from_visit_id: str
    to_visit_id: str
    cohort: str
    from_complete_subjects: int = Field(ge=0)
    retained_subjects: int = Field(ge=0)
    retention_fraction: float = Field(ge=0, le=1)


class PairedReadiness(BaseModel):
    """Subjects with a particular feature observed at both visits."""

    model_config = ConfigDict(frozen=True)

    from_visit_id: str
    to_visit_id: str
    cohort: str
    feature: str
    modality: Modality | None
    eligible_subjects: int = Field(ge=0)
    paired_subjects: int = Field(ge=0)
    paired_fraction: float = Field(ge=0, le=1)


class FeatureDistribution(BaseModel):
    """Distribution and Tukey-IQR outliers for one visit-level feature."""

    model_config = ConfigDict(frozen=True)

    visit_id: str
    cohort: str
    feature: str
    modality: Modality | None
    subjects: int = Field(ge=0)
    mean: float | None = None
    standard_deviation: float | None = Field(default=None, ge=0)
    minimum: float | None = None
    first_quartile: float | None = None
    median: float | None = None
    third_quartile: float | None = None
    maximum: float | None = None
    lower_outlier_fence: float | None = None
    upper_outlier_fence: float | None = None
    outlier_subject_ids: tuple[str, ...] = ()

    @property
    def outlier_count(self) -> int:
        """Return the number of subjects outside the robust fences."""
        return len(self.outlier_subject_ids)


class AttritionBias(BaseModel):
    """Baseline difference between retained and missing-follow-up subjects."""

    model_config = ConfigDict(frozen=True)

    from_visit_id: str
    to_visit_id: str
    cohort: str
    feature: str
    modality: Modality | None
    retained_subjects: int = Field(ge=0)
    attrited_subjects: int = Field(ge=0)
    retained_baseline_mean: float | None = None
    attrited_baseline_mean: float | None = None
    standardized_mean_difference: float | None = None


class StudyProfile(BaseModel):
    """Machine-readable Phase 1 coverage and retention profile."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    visit_coverage: tuple[VisitCoverage, ...] = ()
    visit_retention: tuple[VisitRetention, ...] = ()
    paired_readiness: tuple[PairedReadiness, ...] = ()
    feature_distributions: tuple[FeatureDistribution, ...] = ()
    attrition_bias: tuple[AttritionBias, ...] = ()

    def coverage_frame(self) -> pd.DataFrame:
        """Return visit coverage as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.visit_coverage)

    def retention_frame(self) -> pd.DataFrame:
        """Return consecutive-visit retention as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.visit_retention)

    def paired_readiness_frame(self) -> pd.DataFrame:
        """Return feature-level paired-analysis readiness as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.paired_readiness)

    def distributions_frame(self) -> pd.DataFrame:
        """Return distribution and robust-outlier summaries as a tidy table."""
        rows = []
        for item in self.feature_distributions:
            row = item.model_dump(mode="json")
            row["outlier_count"] = item.outlier_count
            rows.append(row)
        return pd.DataFrame(rows)

    def attrition_bias_frame(self) -> pd.DataFrame:
        """Return baseline retained-versus-attrited comparisons as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.attrition_bias)


class StudyProfiler:
    """Summarize expected-visit coverage and longitudinal analysis readiness."""

    def __init__(self, config: QCConfig, *, outlier_iqr_multiplier: float = 1.5) -> None:
        """Create a profiler using the same expected-visit policy as QC."""
        if outlier_iqr_multiplier < 0:
            raise ValueError("outlier_iqr_multiplier must be non-negative")
        self.config = config
        self.outlier_iqr_multiplier = outlier_iqr_multiplier

    def profile(self, study: Study) -> StudyProfile:
        """Build coverage, complete-case retention, and paired-feature tables."""
        values_by_visit = {
            visit.visit_id: _visit_values(study, visit) for visit in self.config.expected_visits
        }
        observed_by_visit = {
            visit_id: _observed_requirements(values) for visit_id, values in values_by_visit.items()
        }
        coverage = self._coverage(study, observed_by_visit)
        retention = self._retention(study, observed_by_visit)
        readiness = self._paired_readiness(study, observed_by_visit)
        distributions = self._distributions(study, values_by_visit)
        attrition = self._attrition_bias(study, values_by_visit)
        return StudyProfile(
            study_id=study.study_id,
            visit_coverage=tuple(coverage),
            visit_retention=tuple(retention),
            paired_readiness=tuple(readiness),
            feature_distributions=tuple(distributions),
            attrition_bias=tuple(attrition),
        )

    def _coverage(
        self,
        study: Study,
        observed: dict[str, dict[str, set[tuple[str, Modality | None]]]],
    ) -> list[VisitCoverage]:
        rows: list[VisitCoverage] = []
        for visit in self.config.expected_visits:
            for cohort, subjects in _eligible_groups(study, visit).items():
                for requirement in visit.required_features:
                    key = (requirement.feature, requirement.modality)
                    count = sum(
                        key in observed[visit.visit_id].get(item.subject_id, set())
                        for item in subjects
                    )
                    rows.append(
                        VisitCoverage(
                            visit_id=visit.visit_id,
                            cohort=cohort,
                            feature=requirement.feature,
                            modality=requirement.modality,
                            eligible_subjects=len(subjects),
                            observed_subjects=count,
                            coverage_fraction=_fraction(count, len(subjects)),
                        )
                    )
        return rows

    def _distributions(
        self,
        study: Study,
        values: dict[str, dict[tuple[str, str, Modality | None], float]],
    ) -> list[FeatureDistribution]:
        rows: list[FeatureDistribution] = []
        for visit in self.config.expected_visits:
            for cohort, subjects in _eligible_groups(study, visit).items():
                subject_ids = {subject.subject_id for subject in subjects}
                for requirement in visit.required_features:
                    measured = {
                        subject_id: value
                        for (subject_id, feature, modality), value in values[visit.visit_id].items()
                        if subject_id in subject_ids
                        and feature == requirement.feature
                        and modality is requirement.modality
                    }
                    rows.append(
                        _distribution(
                            visit=visit,
                            cohort=cohort,
                            requirement=requirement,
                            measured=measured,
                            iqr_multiplier=self.outlier_iqr_multiplier,
                        )
                    )
        return rows

    def _attrition_bias(
        self,
        study: Study,
        values: dict[str, dict[tuple[str, str, Modality | None], float]],
    ) -> list[AttritionBias]:
        rows: list[AttritionBias] = []
        for first, second in pairwise(self.config.expected_visits):
            second_keys = {(item.feature, item.modality) for item in second.required_features}
            shared = [
                item
                for item in first.required_features
                if (item.feature, item.modality) in second_keys
            ]
            for cohort, subjects in _joint_eligible_groups(study, first, second).items():
                subject_ids = {subject.subject_id for subject in subjects}
                for requirement in shared:
                    retained: list[float] = []
                    attrited: list[float] = []
                    for subject_id in subject_ids:
                        key = (subject_id, requirement.feature, requirement.modality)
                        baseline = values[first.visit_id].get(key)
                        if baseline is None:
                            continue
                        target = retained if key in values[second.visit_id] else attrited
                        target.append(baseline)
                    rows.append(
                        _attrition_row(
                            first=first,
                            second=second,
                            cohort=cohort,
                            requirement=requirement,
                            retained=retained,
                            attrited=attrited,
                        )
                    )
        return rows

    def _retention(
        self,
        study: Study,
        observed: dict[str, dict[str, set[tuple[str, Modality | None]]]],
    ) -> list[VisitRetention]:
        rows: list[VisitRetention] = []
        visits = self.config.expected_visits
        for first, second in pairwise(visits):
            for cohort, subjects in _joint_eligible_groups(study, first, second).items():
                first_complete = _complete_subjects(subjects, first, observed[first.visit_id])
                second_complete = _complete_subjects(subjects, second, observed[second.visit_id])
                retained = len(first_complete.intersection(second_complete))
                rows.append(
                    VisitRetention(
                        from_visit_id=first.visit_id,
                        to_visit_id=second.visit_id,
                        cohort=cohort,
                        from_complete_subjects=len(first_complete),
                        retained_subjects=retained,
                        retention_fraction=_fraction(retained, len(first_complete)),
                    )
                )
        return rows

    def _paired_readiness(
        self,
        study: Study,
        observed: dict[str, dict[str, set[tuple[str, Modality | None]]]],
    ) -> list[PairedReadiness]:
        rows: list[PairedReadiness] = []
        visits = self.config.expected_visits
        for first, second in pairwise(visits):
            first_requirements = {
                (item.feature, item.modality): item for item in first.required_features
            }
            second_keys = {(item.feature, item.modality) for item in second.required_features}
            shared = [item for key, item in first_requirements.items() if key in second_keys]
            for cohort, subjects in _joint_eligible_groups(study, first, second).items():
                for requirement in shared:
                    key = (requirement.feature, requirement.modality)
                    paired = sum(
                        key in observed[first.visit_id].get(item.subject_id, set())
                        and key in observed[second.visit_id].get(item.subject_id, set())
                        for item in subjects
                    )
                    rows.append(
                        PairedReadiness(
                            from_visit_id=first.visit_id,
                            to_visit_id=second.visit_id,
                            cohort=cohort,
                            feature=requirement.feature,
                            modality=requirement.modality,
                            eligible_subjects=len(subjects),
                            paired_subjects=paired,
                            paired_fraction=_fraction(paired, len(subjects)),
                        )
                    )
        return rows


def _visit_values(
    study: Study,
    visit: ExpectedVisit,
) -> dict[tuple[str, str, Modality | None], float]:
    measured: dict[tuple[str, str, Modality | None], list[float]] = defaultdict(list)
    subjects = {subject.subject_id: subject for subject in study.subjects}
    for row in study.observations:
        subject = subjects[row.subject_id]
        if not _visit_applies(visit, subject) or not math.isfinite(row.value):
            continue
        window = visit.window_for(subject)
        if window is None or not window[0] <= row.timestamp <= window[1]:
            continue
        for requirement in visit.required_features:
            if _matches(row, requirement):
                measured[(row.subject_id, requirement.feature, requirement.modality)].append(
                    row.value
                )
    return {key: sum(values) / len(values) for key, values in measured.items()}


def _observed_requirements(
    values: dict[tuple[str, str, Modality | None], float],
) -> dict[str, set[tuple[str, Modality | None]]]:
    observed: dict[str, set[tuple[str, Modality | None]]] = defaultdict(set)
    for subject_id, feature, modality in values:
        observed[subject_id].add((feature, modality))
    return observed


def _matches(row: Observation, requirement: VisitFeature) -> bool:
    return row.feature == requirement.feature and (
        requirement.modality is None or row.modality is requirement.modality
    )


def _visit_applies(visit: ExpectedVisit, subject: Subject) -> bool:
    if not visit.subject_ids and not visit.cohorts:
        return True
    return subject.subject_id in visit.subject_ids or subject.cohort in visit.cohorts


def _eligible_groups(study: Study, visit: ExpectedVisit) -> dict[str, tuple[Subject, ...]]:
    eligible = tuple(subject for subject in study.subjects if _visit_applies(visit, subject))
    cohorts = sorted({subject.cohort for subject in eligible})
    groups = {"all": eligible}
    groups.update(
        {
            cohort: tuple(subject for subject in eligible if subject.cohort == cohort)
            for cohort in cohorts
        }
    )
    return groups


def _joint_eligible_groups(
    study: Study,
    first: ExpectedVisit,
    second: ExpectedVisit,
) -> dict[str, tuple[Subject, ...]]:
    eligible = tuple(
        subject
        for subject in study.subjects
        if _visit_applies(first, subject) and _visit_applies(second, subject)
    )
    cohorts = sorted({subject.cohort for subject in eligible})
    groups = {"all": eligible}
    groups.update(
        {
            cohort: tuple(subject for subject in eligible if subject.cohort == cohort)
            for cohort in cohorts
        }
    )
    return groups


def _complete_subjects(
    subjects: tuple[Subject, ...],
    visit: ExpectedVisit,
    observed: dict[str, set[tuple[str, Modality | None]]],
) -> set[str]:
    required = {(item.feature, item.modality) for item in visit.required_features}
    return {
        subject.subject_id
        for subject in subjects
        if required.issubset(observed.get(subject.subject_id, set()))
    }


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _distribution(
    *,
    visit: ExpectedVisit,
    cohort: str,
    requirement: VisitFeature,
    measured: dict[str, float],
    iqr_multiplier: float,
) -> FeatureDistribution:
    if not measured:
        return FeatureDistribution(
            visit_id=visit.visit_id,
            cohort=cohort,
            feature=requirement.feature,
            modality=requirement.modality,
            subjects=0,
        )
    series = pd.Series(measured, dtype=float)
    first_quartile = float(series.quantile(0.25))
    third_quartile = float(series.quantile(0.75))
    spread = third_quartile - first_quartile
    lower = first_quartile - iqr_multiplier * spread
    upper = third_quartile + iqr_multiplier * spread
    outliers = tuple(
        sorted(
            str(subject_id)
            for subject_id, value in measured.items()
            if value < lower or value > upper
        )
    )
    standard_deviation = float(series.std(ddof=1)) if len(series) > 1 else None
    return FeatureDistribution(
        visit_id=visit.visit_id,
        cohort=cohort,
        feature=requirement.feature,
        modality=requirement.modality,
        subjects=len(series),
        mean=float(series.mean()),
        standard_deviation=standard_deviation,
        minimum=float(series.min()),
        first_quartile=first_quartile,
        median=float(series.median()),
        third_quartile=third_quartile,
        maximum=float(series.max()),
        lower_outlier_fence=lower,
        upper_outlier_fence=upper,
        outlier_subject_ids=outliers,
    )


def _attrition_row(
    *,
    first: ExpectedVisit,
    second: ExpectedVisit,
    cohort: str,
    requirement: VisitFeature,
    retained: list[float],
    attrited: list[float],
) -> AttritionBias:
    retained_mean = sum(retained) / len(retained) if retained else None
    attrited_mean = sum(attrited) / len(attrited) if attrited else None
    standardized = _standardized_mean_difference(retained, attrited)
    return AttritionBias(
        from_visit_id=first.visit_id,
        to_visit_id=second.visit_id,
        cohort=cohort,
        feature=requirement.feature,
        modality=requirement.modality,
        retained_subjects=len(retained),
        attrited_subjects=len(attrited),
        retained_baseline_mean=retained_mean,
        attrited_baseline_mean=attrited_mean,
        standardized_mean_difference=standardized,
    )


def _standardized_mean_difference(first: list[float], second: list[float]) -> float | None:
    if len(first) < 2 or len(second) < 2:
        return None
    first_series = pd.Series(first, dtype=float)
    second_series = pd.Series(second, dtype=float)
    pooled_variance = (
        (len(first) - 1) * float(first_series.var(ddof=1))
        + (len(second) - 1) * float(second_series.var(ddof=1))
    ) / (len(first) + len(second) - 2)
    if pooled_variance <= 0:
        return None
    return float((first_series.mean() - second_series.mean()) / math.sqrt(pooled_variance))
