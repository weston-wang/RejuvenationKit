"""Phase 1 analysis-readiness profiles for longitudinal studies."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations, pairwise

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import fisher_exact

from rejuvenationkit.longitudinal import (
    LongitudinalExclusion,
    LongitudinalExclusionReason,
    extract_visit_aligned_values,
)
from rejuvenationkit.qc import ExpectedVisit, QCConfig, VisitFeature
from rejuvenationkit.schemas import Modality, Study, Subject


class VisitCoverage(BaseModel):
    """Observed coverage for one required feature at one scheduled visit."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    visit_id: str
    cohort: str
    feature: str
    modality: Modality | None
    eligible_subjects: int = Field(ge=0)
    observed_subjects: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_coverage_counts(self) -> VisitCoverage:
        """Bind the reported coverage fraction to its exact subject counts."""
        if self.observed_subjects > self.eligible_subjects:
            raise ValueError("observed subjects cannot exceed eligible subjects")
        if not _fractions_match(
            self.coverage_fraction,
            self.observed_subjects,
            self.eligible_subjects,
        ):
            raise ValueError("coverage fraction does not match subject counts")
        return self


class VisitRetention(BaseModel):
    """Complete-case retention between two consecutive expected visits."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    from_visit_id: str
    to_visit_id: str
    cohort: str
    from_complete_subjects: int = Field(ge=0)
    retained_subjects: int = Field(ge=0)
    retention_fraction: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_retention_counts(self) -> VisitRetention:
        """Bind the reported retention fraction to its exact subject counts."""
        if self.retained_subjects > self.from_complete_subjects:
            raise ValueError("retained subjects cannot exceed complete baseline subjects")
        if not _fractions_match(
            self.retention_fraction,
            self.retained_subjects,
            self.from_complete_subjects,
        ):
            raise ValueError("retention fraction does not match subject counts")
        return self


class PairedReadiness(BaseModel):
    """Subjects with a particular feature observed at both visits."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    from_visit_id: str
    to_visit_id: str
    cohort: str
    feature: str
    modality: Modality | None
    eligible_subjects: int = Field(ge=0)
    paired_subjects: int = Field(ge=0)
    paired_fraction: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_paired_counts(self) -> PairedReadiness:
        """Bind the paired fraction to its exact eligible-subject counts."""
        if self.paired_subjects > self.eligible_subjects:
            raise ValueError("paired subjects cannot exceed eligible subjects")
        if not _fractions_match(
            self.paired_fraction,
            self.paired_subjects,
            self.eligible_subjects,
        ):
            raise ValueError("paired fraction does not match subject counts")
        return self


class FeatureDistribution(BaseModel):
    """Distribution and Tukey-IQR outliers for one visit-level feature."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

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

    @model_validator(mode="after")
    def validate_distribution_summary(self) -> FeatureDistribution:
        """Reject impossible sample counts, quantiles, or outlier identities."""
        summary = (
            self.mean,
            self.minimum,
            self.first_quartile,
            self.median,
            self.third_quartile,
            self.maximum,
            self.lower_outlier_fence,
            self.upper_outlier_fence,
        )
        if tuple(sorted(set(self.outlier_subject_ids))) != self.outlier_subject_ids:
            raise ValueError("outlier subject identifiers must be unique and sorted")
        if self.outlier_count > self.subjects:
            raise ValueError("outlier subjects cannot exceed measured subjects")
        if self.subjects == 0:
            if any(value is not None for value in (*summary, self.standard_deviation)):
                raise ValueError("empty distributions cannot report numerical summaries")
            return self
        if any(value is None for value in summary):
            raise ValueError("nonempty distributions require complete numerical summaries")
        if self.subjects == 1 and self.standard_deviation is not None:
            raise ValueError("single-subject distributions cannot report sample deviation")
        if self.subjects > 1 and self.standard_deviation is None:
            raise ValueError("multi-subject distributions require sample deviation")
        assert all(value is not None for value in summary)
        mean, minimum, first, median_value, third, maximum, lower, upper = summary
        assert mean is not None
        assert minimum is not None
        assert first is not None
        assert median_value is not None
        assert third is not None
        assert maximum is not None
        assert lower is not None
        assert upper is not None
        if not (minimum <= first <= median_value <= third <= maximum):
            raise ValueError("distribution quantiles must be ordered")
        if not minimum <= mean <= maximum:
            raise ValueError("distribution mean must lie within its observed range")
        if lower > first or upper < third or lower > upper:
            raise ValueError("distribution outlier fences are inconsistent with quartiles")
        return self


class AttritionBias(BaseModel):
    """Baseline difference between retained and missing-follow-up subjects."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

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

    @model_validator(mode="after")
    def validate_attrition_summary(self) -> AttritionBias:
        """Require means and standardized differences to match available groups."""
        if (self.retained_subjects == 0) != (self.retained_baseline_mean is None):
            raise ValueError("retained baseline mean must match the retained-subject count")
        if (self.attrited_subjects == 0) != (self.attrited_baseline_mean is None):
            raise ValueError("attrited baseline mean must match the attrited-subject count")
        if self.standardized_mean_difference is not None and (
            self.retained_subjects < 2 or self.attrited_subjects < 2
        ):
            raise ValueError("standardized attrition difference requires two subjects per group")
        return self


class DifferentialAttrition(BaseModel):
    """Between-cohort difference in complete-case attrition for one visit interval."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    from_visit_id: str
    to_visit_id: str
    first_cohort: str
    second_cohort: str
    first_at_risk: int = Field(ge=0)
    second_at_risk: int = Field(ge=0)
    first_attrited: int = Field(ge=0)
    second_attrited: int = Field(ge=0)
    first_attrition_fraction: float = Field(ge=0, le=1)
    second_attrition_fraction: float = Field(ge=0, le=1)
    attrition_risk_difference: float = Field(ge=-1, le=1)
    attrition_risk_ratio: float | None = Field(default=None, ge=0)
    odds_ratio: float | None = Field(default=None, ge=0)
    fisher_exact_p_value: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_differential_counts(self) -> DifferentialAttrition:
        """Bind attrition contrasts to their exact cohort counts and fractions."""
        if self.first_attrited > self.first_at_risk or self.second_attrited > self.second_at_risk:
            raise ValueError("attrited subjects cannot exceed subjects at risk")
        if not _fractions_match(
            self.first_attrition_fraction,
            self.first_attrited,
            self.first_at_risk,
        ) or not _fractions_match(
            self.second_attrition_fraction,
            self.second_attrited,
            self.second_at_risk,
        ):
            raise ValueError("attrition fractions do not match cohort counts")
        expected_difference = self.first_attrition_fraction - self.second_attrition_fraction
        if not math.isclose(
            self.attrition_risk_difference,
            expected_difference,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("attrition risk difference does not match cohort fractions")
        expected_ratio = (
            self.first_attrition_fraction / self.second_attrition_fraction
            if self.second_attrition_fraction > 0
            else None
        )
        if (expected_ratio is None) != (self.attrition_risk_ratio is None) or (
            expected_ratio is not None
            and self.attrition_risk_ratio is not None
            and not math.isclose(
                self.attrition_risk_ratio,
                expected_ratio,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("attrition risk ratio does not match cohort fractions")
        if (self.first_at_risk == 0 or self.second_at_risk == 0) and (
            self.odds_ratio is not None or self.fisher_exact_p_value is not None
        ):
            raise ValueError("Fisher statistics require nonempty cohorts")
        if (
            self.first_at_risk > 0
            and self.second_at_risk > 0
            and (self.fisher_exact_p_value is None)
        ):
            raise ValueError("nonempty cohorts require a Fisher exact p-value")
        return self


class StudyProfile(BaseModel):
    """Machine-readable Phase 1 coverage and retention profile."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    study_id: str
    visit_coverage: tuple[VisitCoverage, ...] = ()
    visit_retention: tuple[VisitRetention, ...] = ()
    paired_readiness: tuple[PairedReadiness, ...] = ()
    feature_distributions: tuple[FeatureDistribution, ...] = ()
    attrition_bias: tuple[AttritionBias, ...] = ()
    differential_attrition: tuple[DifferentialAttrition, ...] = ()
    longitudinal_exclusions: tuple[LongitudinalExclusion, ...] = ()

    @model_validator(mode="after")
    def validate_profile_rows(self) -> StudyProfile:
        """Require one deterministic row for every serialized profile estimand."""
        if not self.study_id:
            raise ValueError("study profile identifier must be nonblank")
        keyed_rows = (
            (
                "coverage",
                tuple(
                    (row.visit_id, row.cohort, row.feature, row.modality)
                    for row in self.visit_coverage
                ),
            ),
            (
                "retention",
                tuple(
                    (row.from_visit_id, row.to_visit_id, row.cohort) for row in self.visit_retention
                ),
            ),
            (
                "paired readiness",
                tuple(
                    (
                        row.from_visit_id,
                        row.to_visit_id,
                        row.cohort,
                        row.feature,
                        row.modality,
                    )
                    for row in self.paired_readiness
                ),
            ),
            (
                "distribution",
                tuple(
                    (row.visit_id, row.cohort, row.feature, row.modality)
                    for row in self.feature_distributions
                ),
            ),
            (
                "attrition bias",
                tuple(
                    (
                        row.from_visit_id,
                        row.to_visit_id,
                        row.cohort,
                        row.feature,
                        row.modality,
                    )
                    for row in self.attrition_bias
                ),
            ),
            (
                "differential attrition",
                tuple(
                    (
                        row.from_visit_id,
                        row.to_visit_id,
                        row.first_cohort,
                        row.second_cohort,
                    )
                    for row in self.differential_attrition
                ),
            ),
        )
        for label, keys in keyed_rows:
            if len(keys) != len(set(keys)):
                raise ValueError(f"study profile {label} rows must be unique")
        if len(self.longitudinal_exclusions) != len(set(self.longitudinal_exclusions)):
            raise ValueError("study profile longitudinal exclusions must be unique")
        return self

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

    def differential_attrition_frame(self) -> pd.DataFrame:
        """Return between-cohort attrition comparisons as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.differential_attrition)


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
        cohorts = {subject.cohort for subject in study.subjects}
        if "all" in cohorts and len(cohorts) > 1:
            raise ValueError(
                "cohort label 'all' is reserved for the aggregate profile "
                "when multiple cohorts exist"
            )
        values_by_visit, longitudinal_exclusions = _visit_values(
            study,
            self.config.expected_visits,
        )
        observed_by_visit = {
            visit_id: _observed_requirements(values) for visit_id, values in values_by_visit.items()
        }
        coverage = self._coverage(study, observed_by_visit)
        retention = self._retention(study, observed_by_visit)
        differential_attrition = _differential_attrition(retention)
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
            differential_attrition=differential_attrition,
            longitudinal_exclusions=longitudinal_exclusions,
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
    visits: tuple[ExpectedVisit, ...],
) -> tuple[
    dict[str, dict[tuple[str, str, Modality | None], float]],
    tuple[LongitudinalExclusion, ...],
]:
    measured: dict[str, dict[tuple[str, str, Modality | None], float]] = {
        visit.visit_id: {} for visit in visits
    }
    requirements: dict[tuple[str, Modality | None], VisitFeature] = {}
    exclusions: list[LongitudinalExclusion] = []
    for visit in visits:
        for requirement in visit.required_features:
            requirements.setdefault((requirement.feature, requirement.modality), requirement)
    for key, requirement in requirements.items():
        relevant_visits = tuple(
            visit
            for visit in visits
            if any((item.feature, item.modality) == key for item in visit.required_features)
        )
        extraction = extract_visit_aligned_values(
            study,
            visits=relevant_visits,
            channels=(requirement,),
        )
        exclusions.extend(
            item
            for item in extraction.exclusions
            if item.reason is not LongitudinalExclusionReason.VISIT_NOT_APPLICABLE
        )
        for item in extraction.values:
            measured[item.visit_id][(item.subject_id, *key)] = item.value
    return measured, tuple(dict.fromkeys(exclusions))


def _observed_requirements(
    values: dict[tuple[str, str, Modality | None], float],
) -> dict[str, set[tuple[str, Modality | None]]]:
    observed: dict[str, set[tuple[str, Modality | None]]] = defaultdict(set)
    for subject_id, feature, modality in values:
        observed[subject_id].add((feature, modality))
    return observed


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


def _fractions_match(value: float, numerator: int, denominator: int) -> bool:
    """Compare a serialized fraction with the deterministic count-derived value."""
    return math.isclose(
        value,
        _fraction(numerator, denominator),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _differential_attrition(
    retention: list[VisitRetention],
) -> tuple[DifferentialAttrition, ...]:
    """Compare complete-case attrition between every declared cohort pair."""
    grouped: dict[tuple[str, str], list[VisitRetention]] = defaultdict(list)
    for row in retention:
        if row.cohort != "all":
            grouped[(row.from_visit_id, row.to_visit_id)].append(row)
    comparisons: list[DifferentialAttrition] = []
    for (from_visit_id, to_visit_id), rows in sorted(grouped.items()):
        for first, second in combinations(sorted(rows, key=lambda item: item.cohort), 2):
            first_attrited = first.from_complete_subjects - first.retained_subjects
            second_attrited = second.from_complete_subjects - second.retained_subjects
            first_fraction = _fraction(first_attrited, first.from_complete_subjects)
            second_fraction = _fraction(second_attrited, second.from_complete_subjects)
            risk_ratio = first_fraction / second_fraction if second_fraction > 0 else None
            odds_ratio: float | None = None
            p_value: float | None = None
            if first.from_complete_subjects > 0 and second.from_complete_subjects > 0:
                statistic, probability = fisher_exact(
                    [
                        [first_attrited, first.retained_subjects],
                        [second_attrited, second.retained_subjects],
                    ],
                    alternative="two-sided",
                )
                odds_ratio = float(statistic) if math.isfinite(float(statistic)) else None
                p_value = float(probability)
            comparisons.append(
                DifferentialAttrition(
                    from_visit_id=from_visit_id,
                    to_visit_id=to_visit_id,
                    first_cohort=first.cohort,
                    second_cohort=second.cohort,
                    first_at_risk=first.from_complete_subjects,
                    second_at_risk=second.from_complete_subjects,
                    first_attrited=first_attrited,
                    second_attrited=second_attrited,
                    first_attrition_fraction=first_fraction,
                    second_attrition_fraction=second_fraction,
                    attrition_risk_difference=first_fraction - second_fraction,
                    attrition_risk_ratio=risk_ratio,
                    odds_ratio=odds_ratio,
                    fisher_exact_p_value=p_value,
                )
            )
    return tuple(comparisons)


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
