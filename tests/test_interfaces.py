from datetime import UTC, datetime

from rejuvenationkit.qc import QCFinding, QCReport, Severity
from rejuvenationkit.state import StateEstimate


def test_qc_report_passes_without_errors() -> None:
    report = QCReport(
        study_id="study",
        findings=(QCFinding(code="note", severity=Severity.INFO, message="informational"),),
    )
    assert report.passed


def test_qc_report_fails_with_error() -> None:
    report = QCReport(
        study_id="study",
        findings=(QCFinding(code="bad", severity=Severity.ERROR, message="invalid"),),
    )
    assert not report.passed


def test_state_dimension() -> None:
    estimate = StateEstimate(
        subject_id="s1",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        state_names=("immune_age", "metabolic_age"),
        mean=(0.0, 1.0),
        covariance=((1.0, 0.0), (0.0, 1.0)),
    )
    assert estimate.dimension == 2
