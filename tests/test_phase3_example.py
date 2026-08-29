from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest


def test_phase3_synthetic_example_runs_and_reports_guardrails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    example = Path(__file__).parents[1] / "examples" / "phase3_longitudinal_state.py"
    namespace = runpy.run_path(str(example), run_name="phase3_example_test")
    main = cast(Callable[[], None], namespace["main"])

    main()

    output = capsys.readouterr().out
    assert "Held-out trajectories: 6; excluded subjects: 0" in output
    assert "Mean observed channel coverage:" in output
    assert "injected-shift detections:" in output
    assert "injected-shift detections: 0" not in output
    assert "not efficacy, causality, or biological-age reversal" in output
