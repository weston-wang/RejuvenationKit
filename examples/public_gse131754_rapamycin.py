"""Download and quality-check a public rapamycin RNA-seq dataset.

This example performs descriptive treatment-response exploration. It does not
estimate lifespan extension or prove biological rejuvenation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rejuvenationkit import BaselineLongitudinalQC, FeatureRule, Modality, QCConfig
from rejuvenationkit.datasets.gse131754 import (
    GSE131754_URL,
    build_study,
    descriptive_log2_fold_changes,
    download_counts,
    read_counts,
    select_rapamycin_and_controls,
)
from rejuvenationkit.schemas import Subject


def analysis_manifest(study_subjects: tuple[Subject, ...]) -> pd.DataFrame:
    """Create a compact, shareable sample manifest from typed subjects."""
    rows = []
    for subject in study_subjects:
        data = subject.model_dump()
        rows.append(
            {
                "sample_id": data["subject_id"],
                "cohort": data["cohort"],
                "interventions": ",".join(data["interventions"]),
                **data["attributes"],
            }
        )
    return pd.DataFrame(rows).sort_values("sample_id")


def main() -> None:
    """Run the reproducible public-data workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("examples/data/cache/GSE131754_assigned_reads.txt.gz"),
    )
    parser.add_argument("--genes", type=int, default=250, help="Variable genes to place in Study")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    path = download_counts(args.cache)
    all_counts = read_counts(path)
    counts = select_rapamycin_and_controls(all_counts)
    variable_genes = tuple(str(item) for item in counts.var(axis=1).nlargest(args.genes).index)
    study = build_study(counts, gene_ids=variable_genes)

    config = QCConfig(
        feature_rules=tuple(
            FeatureRule(
                feature=gene_id,
                modality=Modality.TRANSCRIPTOMICS,
                expected_unit="assigned_reads",
                minimum=0,
                required=True,
            )
            for gene_id in variable_genes
        )
    )
    report = BaselineLongitudinalQC(config).run(study)
    effects = descriptive_log2_fold_changes(counts)
    manifest = analysis_manifest(study.subjects)

    print(f"Source: {GSE131754_URL}")
    print(f"Selected matrix: {counts.shape[0]:,} genes x {counts.shape[1]} samples")
    print(f"Typed QC subset: {len(variable_genes)} genes x {len(study.subjects)} subjects")
    print(report.summary())
    print(
        "Batch note: the processed GEO matrix does not expose sequencing-batch identifiers, "
        "so batch-confounding QC cannot be evaluated from this file."
    )
    print("\nLargest descriptive RAP-minus-control expression differences:")
    print(effects.head(10).to_string())
    print(
        "\nInterpretation: these are exploratory expression differences after library-size "
        "normalization, not tests of lifespan extension or causal rejuvenation."
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.output_dir / "sample_manifest.csv", index=False)
        effects.to_csv(args.output_dir / "descriptive_expression_differences.csv")
        (args.output_dir / "qc_report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
