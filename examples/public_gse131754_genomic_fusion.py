"""Run joint pathway-response analysis on public GSE131754 rapamycin RNA-seq.

The workflow is an engineering validation of genome-scale ingestion, signature
coverage, joint subject bootstrap covariance, and estimand separation. The
small mechanism panels are not validated biological-age clocks. This example
does not estimate lifespan extension or prove rejuvenation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rejuvenationkit.datasets.gse131754 import (
    GSE131754_SHA256,
    GSE131754_URL,
    build_genomic_matrix,
    download_counts,
    rapamycin_mechanism_signatures,
    read_counts,
    select_rapamycin_and_controls,
)
from rejuvenationkit.genomics import (
    GenomicMatrix,
    SignatureContrastConfig,
    estimate_signature_contrasts,
    normalize_counts_log_cpm,
    score_weighted_signature,
)


def analyze_stratum(
    matrix: GenomicMatrix,
    *,
    age_months: int,
    sex: str,
    bootstrap_iterations: int,
) -> None:
    """Print one matched age/sex contrast without pooling different doses."""
    sample_ids = tuple(
        sample.sample_id
        for sample in matrix.samples
        if sample.attributes["age_months"] == age_months and sample.attributes["sex"] == sex
    )
    stratum = matrix.subset_samples(sample_ids)
    signatures = rapamycin_mechanism_signatures()
    dose = "42 ppm for 2 months" if age_months == 6 else "14 ppm for 8 months"
    scored = tuple(
        (score_weighted_signature(stratum, signature), signature) for signature in signatures
    )
    batch = estimate_signature_contrasts(
        scored,
        SignatureContrastConfig(
            treated_cohort="rapamycin",
            control_cohort="control",
            bootstrap_iterations=bootstrap_iterations,
            random_seed=age_months * 10 + (1 if sex == "F" else 2),
            minimum_subjects_per_group=3,
            covariance_shrinkage=0.35,
            estimand_population=f"GSE131754:{age_months}m:{sex}:{dose}",
            time_contrast=f"endpoint after {dose} versus matched control",
        ),
    )
    print(f"\n{age_months} months, sex={sex}, {dose}")
    for estimate in batch.estimates:
        print(
            f"  {estimate.signature_id:27s} {estimate.estimate:+.3f} ± "
            f"{estimate.standard_error:.3f} mean signed log2 CPM"
        )
    maximum_correlation = max(
        abs(batch.correlation[row][column])
        for row in range(len(batch.correlation))
        for column in range(row)
    )
    print(f"  maximum absolute panel correlation {maximum_correlation:.2f}")
    condition_number = float(np.linalg.cond(np.asarray(batch.covariance.covariance, dtype=float)))
    print(f"  shrunk covariance condition number {condition_number:.1f}")
    print(
        "  interpretation: this joint pathway vector is not collapsed into one efficacy score "
        "because the panels are not calibrated as interchangeable measurements"
    )


def main() -> None:
    """Download the public matrix and run four prespecified matched contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("examples/data/cache/GSE131754_assigned_reads.txt.gz"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    args = parser.parse_args()

    counts = select_rapamycin_and_controls(read_counts(download_counts(args.cache)))
    raw = build_genomic_matrix(counts, source_checksum=GSE131754_SHA256)
    signatures = rapamycin_mechanism_signatures()
    feature_ids = tuple(
        dict.fromkeys(
            feature.feature_id for signature in signatures for feature in signature.features
        )
    )
    matrix = normalize_counts_log_cpm(raw, feature_ids=feature_ids)

    print("Public GSE131754 joint genomic-response benchmark")
    print(f"Source: {GSE131754_URL}")
    print(f"Raw matrix: {raw.shape[0]} samples x {raw.shape[1]:,} genes")
    print(f"Scored matrix: {matrix.shape[0]} samples x {matrix.shape[1]} panel genes")
    print(f"Input fingerprint: {raw.content_hash}")
    for age_months in (6, 12):
        for sex in ("F", "M"):
            analyze_stratum(
                matrix,
                age_months=age_months,
                sex=sex,
                bootstrap_iterations=args.bootstrap_iterations,
            )

    print(
        "\nInterpretation limits: age, dose, and exposure duration are not pooled. Panel genes and "
        "directions are small, auditable mechanism fixtures—not externally validated age clocks. "
        "With three mice per arm per stratum, covariance and confidence intervals are unstable."
    )
    print(
        "The benchmark validates software behavior on real genome-scale data; it does not prove "
        "rapamycin efficacy, pathway causality, or lifespan extension."
    )


if __name__ == "__main__":
    main()
