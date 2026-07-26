"""Phase 1: longitudinal data-quality interfaces."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from rejuvenationkit.schemas import Study


class Severity(StrEnum):
    """Severity of a quality-control finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QCFinding(BaseModel):
    """A machine-readable quality-control finding."""

    model_config = ConfigDict(frozen=True)

    code: str
    severity: Severity
    message: str
    subject_ids: tuple[str, ...] = ()
    observation_indices: tuple[int, ...] = ()


class QCReport(BaseModel):
    """Output of a longitudinal quality-control run."""

    model_config = ConfigDict(frozen=True)

    study_id: str
    findings: tuple[QCFinding, ...]

    @property
    def passed(self) -> bool:
        """Return whether the report contains no errors."""
        return all(item.severity is not Severity.ERROR for item in self.findings)


class LongitudinalQC(Protocol):
    """Contract for Phase 1 quality-control implementations."""

    def run(self, study: Study) -> QCReport:
        """Evaluate a study without mutating it."""
        ...


class BaselineLongitudinalQC:
    """Placeholder for the first reference QC implementation."""

    def run(self, study: Study) -> QCReport:
        """Run baseline checks.

        TODO(phase-1): implement missingness, temporal ordering, range, batch-drift,
        and replicate-consistency checks with configurable policies.
        """
        raise NotImplementedError("Phase 1 baseline QC is not implemented")
