from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest


def test_external_biology_example_keeps_context_non_fusible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    example = Path(__file__).parents[1] / "examples" / "external_biology_context.py"
    namespace = runpy.run_path(str(example), run_name="external_biology_example_test")
    functional_context = cast(Callable[[], None], namespace["functional_context"])
    interaction_context = cast(Callable[[], None], namespace["interaction_context"])

    functional_context()
    interaction_context()

    output = capsys.readouterr().out
    assert "Directionless overrepresentation" in output
    assert "Directional ranked-set context" in output
    assert "ranking values are synthetic and are not treatment-efficacy evidence" in output
    assert output.count("fusion eligibility: not_fusible") == 3
    assert "nodes: 3; edges: 3" in output
