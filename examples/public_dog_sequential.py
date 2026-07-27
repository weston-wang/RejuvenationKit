"""Track sequential multimodal departures in public longitudinal dog data.

This observational example learns normal cohort dynamics from three Dog Aging
Project Precision waves. It demonstrates monitoring mechanics, not treatment
response, rejuvenation, or causal inference.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import timedelta
from pathlib import Path

from rejuvenationkit import (
    ExpectedVisit,
    Modality,
    SequentialDetectionConfig,
    SequentialTreatmentResponseDetector,
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

VISIT_IDS = ("precision_1", "precision_2", "precision_3")
CLINICAL_FEATURES = ("krt_cp_globulins_value", "krt_cp_potassium_value")
METABOLOMIC_FEATURES = ("Ethanolamine", "1/3-Methylhistidine")


def main() -> None:
    """Run the public three-wave canine sequential workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("examples/data/cache/dap_baseline_phenotypes.zip"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.plots and args.output_dir is None:
        parser.error("--plots requires --output-dir")

    archive = download_archive(args.cache)
    chemistry = select_visits(read_chemistry(archive), visits=VISIT_IDS)
    metabolomics = select_visits(read_metabolomics(archive), visits=VISIT_IDS)
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
    visits = tuple(
        ExpectedVisit(
            visit_id=visit_id,
            anchor_id="normalized_precision_1",
            offset=timedelta(days=365 * index),
            required_features=features,
        )
        for index, visit_id in enumerate(VISIT_IDS)
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
    report = (
        SequentialTreatmentResponseDetector(
            SequentialDetectionConfig(
                features=features,
                covariance_shrinkage=0.20,
                false_alarm_rate=0.05,
                minimum_reference_subjects=75,
                persistence_crossings=2,
            )
        )
        .fit(
            study,
            visits=visits,
            reference_subject_ids=calibration_ids,
        )
        .score(
            study,
            visits=visits,
            subject_ids=evaluation_ids,
        )
    )
    results = report.results_frame().sort_values("peak_cumulative_score", ascending=False)
    awaiting_persistence = sum(
        item.detected and not item.persistent and not item.transient for item in report.results
    )

    print(f"Source: {DAP_ARCHIVE_URL}")
    print(f"Visits: {', '.join(report.visit_ids)}")
    print(f"Channels: {', '.join(report.model.feature_names)}")
    print(
        f"Calibration trajectories: {report.model.reference_subjects}; "
        f"transitions: {report.model.reference_transitions}"
    )
    print(
        f"Held-out scored: {len(report.results)}; "
        f"incomplete (<2 complete visits): {len(report.excluded_subject_ids)}"
    )
    print(
        f"Detected departures: {sum(item.detected for item in report.results)}; "
        f"persistent: {sum(item.persistent for item in report.results)}; "
        f"transient: {sum(item.transient for item in report.results)}; "
        f"awaiting persistence: {awaiting_persistence}"
    )
    print("\nLargest held-out sequential departures:")
    print(results.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "\nInterpretation: these are unusual observational trajectories relative to the "
        "calibration cohort. They are not evidence of benefit, harm, or treatment effect."
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_dir / "sequential_subject_summary.csv", index=False)
        report.trajectory_frame().to_csv(
            args.output_dir / "sequential_trajectories.csv",
            index=False,
        )
        (args.output_dir / "sequential_detection_report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if args.plots:
            from rejuvenationkit.visualization import save_sequential_figures

            print("\nSequential DSP figures:")
            for path in save_sequential_figures(report, args.output_dir):
                print(path)


if __name__ == "__main__":
    main()
