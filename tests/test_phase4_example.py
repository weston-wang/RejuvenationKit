from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest


def test_phase4_synthetic_example_runs_and_reports_guardrails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    example = Path(__file__).parents[1] / "examples" / "phase4_factorial_combinations.py"
    namespace = runpy.run_path(str(example), run_name="phase4_example_test")
    main = cast(Callable[[], None], namespace["main"])

    main()

    output = capsys.readouterr().out
    assert "Phase 4 synthetic randomized canine 2 x 2 example" in output
    assert "Subjects: 72 (18 assigned per cell); dog-level randomization seed=9173" in output
    assert "covariance=hc3" in output
    assert "weighting=inverse_variance_required" in output
    assert "multiplicity=benjamini_hochberg" in output
    assert "Cell control: assigned=18, analyzable=18" in output
    assert "Cell rapamycin+senolytic: assigned=18, analyzable=18" in output
    assert "cells=4/4" in output
    assert "Planning illustration:" in output
    assert "not an efficacy claim or automatic evidence of synergy" in output
