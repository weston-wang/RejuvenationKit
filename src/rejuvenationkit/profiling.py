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


class StudyProfile(BaseModel):
    """Machine-readable Phase 1 coverage and retention profile."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    visit_coverage: tuple[VisitCoverage, ...] = ()
    visit_retention: tuple[VisitRetention, ...] = ()
    paired_readiness: tuple[PairedReadiness, ...] = ()

    def coverage_frame(self) -> pd.DataFrame:
        """Return visit coverage as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.visit_coverage)

    def retention_frame(self) -> pd.DataFrame:
        """Return consecutive-visit retention as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.visit_retention)

    def paired_readiness_frame(self) -> pd.DataFrame:
        """Return feature-level paired-analysis readiness as a tidy table."""
        return pd.DataFrame(item.model_dump(mode="json") for item in self.paired_readiness)


class StudyProfiler:
    """Summarize expected-visit coverage and longitudinal analysis readiness."""

    def __init__(self, config: QCConfig) -> None:
        """Create a profiler using the same expected-visit policy as QC."""
        self.config = config

    def profile(self, study: Study) -> StudyProfile:
        """Build coverage, complete-case retention, and paired-feature tables."""
        observed_by_visit = {
            visit.visit_id: _observed_requirements(study, visit)
            for visit in self.config.expected_visits
        }
        coverage = self._coverage(study, observed_by_visit)
        retention = self._retention(study, observed_by_visit)
        readiness = self._paired_readiness(study, observed_by_visit)
        return StudyProfile(
            study_id=study.study_id,
            visit_coverage=tuple(coverage),
            visit_retention=tuple(retention),
            paired_readiness=tuple(readiness),
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


def _observed_requirements(
    study: Study,
    visit: ExpectedVisit,
) -> dict[str, set[tuple[str, Modality | None]]]:
    observed: dict[str, set[tuple[str, Modality | None]]] = defaultdict(set)
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
                observed[row.subject_id].add((requirement.feature, requirement.modality))
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
