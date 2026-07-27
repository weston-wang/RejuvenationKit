"""Detect held-out multimodal change in longitudinal pet-dog measurements.

This example fuses clinical chemistry and technical-adjusted metabolomics. It
demonstrates covariance-aware signal detection, not rejuvenation or treatment
efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import timedelta
from pathlib import Path

from rejuvenationkit import (
    ChangeDetectionConfig,
    ExpectedVisit,
    Modality,
    MultivariateChangeDetector,
    VisitFeature,
)
from rejuvenationkit.datasets.dog_aging_project import (
    DAP_ARCHIVE_URL,
    build_multimodal_study,
    download_archive,
    read_chemistry,
    read_metabolomics,
    select_visits,
)

CLINICAL_FEATURES = (
    "krt_cp_globulins_value",
    "krt_cp_potassium_value",
)
METABOLOMIC_FEATURES = (
    "Ethanolamine",
    "1/3-Methylhistidine",
)


def main() -> None:
    """Run the public multimodal canine workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("examples/data/cache/dap_baseline_phenotypes.zip"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write covariance, score, whitening, and decomposition PNGs",
    )
    args = parser.parse_args()
    if args.plots and args.output_dir is None:
        parser.error("--plots requires --output-dir")

    archive = download_archive(args.cache)
    chemistry = select_visits(read_chemistry(archive))
    metabolomics = select_visits(read_metabolomics(archive))
    study = build_multimodal_study(
        chemistry,
        metabolomics,
        clinical_features=CLINICAL_FEATURES,
        metabolomic_features=METABOLOMIC_FEATURES,
    )
    features = (
        *(
            VisitFeature(feature=feature, modality=Modality.CLINICAL)
            for feature in CLINICAL_FEATURES
        ),
        *(
            VisitFeature(feature=feature, modality=Modality.METABOLOMICS)
            for feature in METABOLOMIC_FEATURES
        ),
    )
    baseline = ExpectedVisit(
        visit_id="precision_1",
        anchor_id="normalized_precision_1",
        required_features=features,
    )
    follow_up = ExpectedVisit(
        visit_id="precision_2",
        anchor_id="normalized_precision_1",
        offset=timedelta(days=365),
        required_features=features,
    )
    calibration_ids = tuple(
        subject.subject_id
        for subject in study.subjects
        if hashlib.sha256(subject.subject_id.encode()).digest()[0] < 179
    )
    evaluation_ids = tuple(
        subject.subject_id
        for subject in study.subjects
        if hashlib.sha256(subject.subject_id.encode()).digest()[0] >= 179
    )
    detector = MultivariateChangeDetector(
        ChangeDetectionConfig(
            features=features,
            covariance_shrinkage=0.20,
            false_alarm_rate=0.05,
            minimum_reference_subjects=100,
        )
    ).fit(
        study,
        baseline=baseline,
        follow_up=follow_up,
        reference_subject_ids=calibration_ids,
    )
    report = detector.score(
        study,
        baseline=baseline,
        follow_up=follow_up,
        subject_ids=evaluation_ids,
    )
    scores = report.results_frame().sort_values(
        "squared_mahalanobis_distance",
        ascending=False,
    )

    modalities = sorted({row.modality.value for row in study.observations})
    print(f"Source: {DAP_ARCHIVE_URL}")
    print(f"Modalities: {', '.join(modalities)}")
    print(f"Channels: {', '.join(report.model.feature_names)}")
    print(
        f"Calibration complete pairs: {report.model.reference_subjects}; "
        f"held-out scored: {len(report.results)}; "
        f"held-out incomplete: {len(report.excluded_subject_ids)}"
    )
    print(
        f"Detections: {sum(item.detected for item in report.results)}/{len(report.results)} "
        f"at nominal {report.model.false_alarm_rate:.1%} false-alarm rate"
    )
    print("\nLargest held-out covariance-normalized changes:")
    print(
        scores.head(10).to_string(
            index=False,
            columns=[
                "subject_id",
                "squared_mahalanobis_distance",
                "empirical_tail_probability",
                "detected",
            ],
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(
        "\nInterpretation: detections are unusual joint changes across clinical chemistry "
        "and metabolomics. They do not identify benefit, harm, or a treatment response."
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.output_dir / "multimodal_change_scores.csv", index=False)
        (args.output_dir / "multimodal_detection_report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if args.plots:
            from rejuvenationkit.visualization import save_detection_figures

            paths = save_detection_figures(
                report,
                args.output_dir,
                prefix="multimodal",
            )
            print("\nDSP figures:")
            for path in paths:
                print(path)


if __name__ == "__main__":
    main()
