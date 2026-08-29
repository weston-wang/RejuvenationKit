from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from rejuvenationkit.bridges import TimedEvidenceBatch
from rejuvenationkit.endpoints import SubjectEndpointBatch
from rejuvenationkit.workflow import (
    Phase2FusionInput,
    PhaseExecutionStatus,
    RejuvenationWorkflowRequest,
    load_rejuvenation_workflow_report,
)


def test_four_phase_example_runs_with_explicit_boundaries_and_guardrails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    example = Path(__file__).parents[1] / "examples" / "four_phase_workflow.py"
    namespace = runpy.run_path(str(example), run_name="four_phase_workflow_example_test")
    build_request = cast(Callable[[], RejuvenationWorkflowRequest], namespace["build_request"])
    main = cast(Callable[[Path | None], None], namespace["main"])
    request = build_request()

    assert isinstance(request.inputs.phase2, Phase2FusionInput)
    assert isinstance(request.inputs.phase3, TimedEvidenceBatch)
    assert isinstance(request.inputs.phase4, SubjectEndpointBatch)

    output_dir = tmp_path / "verified-workflow-bundle"
    main(output_dir)

    output = capsys.readouterr().out
    assert "phase1: completed (qc_passed)" in output
    assert "phase2: completed (covariance_aware_gls_completed)" in output
    assert "phase3: completed (prespecified_state_estimation_completed)" in output
    assert "phase4: completed (explicit_subject_endpoint_factorial_analysis_completed)" in output
    assert "serialized_override=False; override_applied=False" in output
    assert "no downstream input was inferred" in output
    assert "workflow-manifest.json checksums passed" in output
    assert "not efficacy, synergy, causality, or biological-age reversal" in output

    report = load_rejuvenation_workflow_report(output_dir)
    assert all(item.status is PhaseExecutionStatus.COMPLETED for item in report.dispositions)
    assert report.qc_gate.qc_passed
    assert not report.qc_gate.override_applied
    assert report.phase2 is not None
    assert report.phase3 is not None
    assert report.phase4 is not None
