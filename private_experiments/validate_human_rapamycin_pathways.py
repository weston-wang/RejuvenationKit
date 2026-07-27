"""Validate rapamycin pathway responses across public human GEO experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from human_rapamycin_geo import (
    contrast_concordance,
    load_contrasts,
    pathway_scores,
)


def main() -> None:
    """Download, normalize, and compare human rapamycin expression contrasts."""
    cache = Path("private_experiments/data/geo")
    contrasts, unsupported = load_contrasts(cache)
    scores = pd.concat([pathway_scores(contrast) for contrast in contrasts], axis=1).T
    concordance = contrast_concordance(contrasts)

    print("REAL HUMAN EX-VIVO RAPAMYCIN PATHWAY VALIDATION")
    print(f"Gene-level contrasts: {len(contrasts)}")
    print("\nRobust standardized pathway effects:")
    print(scores.round(3).to_string())
    print("\nCross-context gene-effect Spearman agreement:")
    print(concordance.round(3).to_string())
    if unsupported:
        print("\nExplicitly unsupported:")
        for message in unsupported:
            print(f"- {message}")
    print(
        "\nInterpretation: agreement supports reproducible transcriptional direction across cell "
        "contexts. Disagreement can represent cell specificity, dose effects, or noise; these "
        "short ex-vivo experiments do not validate longitudinal clinical off-target effects."
    )


if __name__ == "__main__":
    main()
