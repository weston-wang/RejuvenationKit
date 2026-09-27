"""Reanalyze cross-intervention concordance in public GSE131754 mouse liver RNA-seq.

GSE131754 profiles liver transcriptomes under eight lifespan-extending
interventions. Five of them (acarbose, caloric restriction, 17-alpha-estradiol,
Protandim, rapamycin) are compared with the *same* control mice in each age/sex
stratum. Fold-change vectors computed against a shared control inherit that
control's noise, so they correlate even when the interventions' true effects are
unrelated.

This script contrasts the naive correlation of fold-change vectors with the
shared-control-corrected estimate from ``rejuvenationkit.genomics.concordance``,
a disjoint-control cross-check, and a Holm-adjusted within-stratum permutation
test. It is a methods demonstration on one public dataset. It does not reproduce
the source publication's multi-dataset signature pipeline and makes no claim
about that publication's conclusions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rejuvenationkit.datasets import (
    INTERVENTION_CONTROLS,
    download_counts,
    filtered_log2_cpm,
    intervention_sample_table,
    read_counts,
)
from rejuvenationkit.genomics.concordance import (
    InterventionContrast,
    StrataPolicy,
    concordance_family,
)


def main() -> None:
    """Download the pinned matrix, run the reanalysis, and print a summary table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/GSE131754_Interventions_assigned_reads.txt.gz"),
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--output", type=Path, default=None, help="optional CSV path")
    args = parser.parse_args()

    counts = read_counts(download_counts(args.cache))
    values = filtered_log2_cpm(counts)
    samples = intervention_sample_table(values)
    contrasts = tuple(
        InterventionContrast(treated_group=treated, control_group=control)
        for treated, control in INTERVENTION_CONTROLS.items()
    )
    family = concordance_family(
        values,
        samples,
        contrasts,
        strata_policy=StrataPolicy.OWN,
        permutations=args.permutations,
    )
    table = family.to_frame()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)

    print("Public GSE131754 cross-intervention concordance reanalysis")
    print(f"Genes after expression filter: {values.shape[0]:,}; samples: {values.shape[1]}")
    print(f"Permutations per pair: {args.permutations}; Holm adjustment over {len(table)} pairs")
    print()
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(
            table.assign(
                first=table["first"].str.split("-vs-").str[0],
                second=table["second"].str.split("-vs-").str[0],
            )[
                [
                    "first",
                    "second",
                    "shared_control_strata",
                    "naive_correlation",
                    "corrected_correlation",
                    "disjoint_control_correlation",
                    "permutation_p_value",
                    "holm_adjusted_p_value",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:+.3f}")
        )
    shared = table[table["shared_control_strata"] > 0]
    print()
    print(
        f"Shared-control pairs: {len(shared)}; "
        f"mean naive r {shared['naive_correlation'].mean():+.3f}; "
        f"mean corrected r {shared['corrected_correlation'].mean():+.3f}"
    )
    print(
        f"Pairs significant after Holm adjustment: "
        f"{int((table['holm_adjusted_p_value'] <= 0.05).sum())} of {len(table)}"
    )


if __name__ == "__main__":
    main()
