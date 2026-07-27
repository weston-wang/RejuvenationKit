"""Quality-check real longitudinal blood chemistry measurements from pet dogs.

This observational example contains no rapamycin assignment and must not be
used to estimate a rapamycin treatment effect.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import timedelta
from pathlib import Path

from rejuvenationkit import (
    BaselineLongitudinalQC,
    ChangeDetectionConfig,
    ExpectedVisit,
    FeatureRule,
    Modality,
    MultivariateChangeDetector,
    QCConfig,
    StudyProfiler,
    VisitFeature,
)
from rejuvenationkit.datasets.dog_aging_project import (
    DAP_ARCHIVE_URL,
    build_study,
    download_archive,
    paired_changes,
    read_chemistry,
    select_visits,
)

FEATURES = (
    "krt_cp_albumin_value",
    "krt_cp_creatinine_value",
    "krt_cp_globulins_value",
    "krt_cp_potassium_value",
)


def main() -> None:
    """Run the reproducible canine longitudinal-data workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("examples/data/cache/dap_baseline_phenotypes.zip"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    archive = download_archive(args.cache)
    chemistry = read_chemistry(archive)
    selected = select_visits(chemistry)
    study = build_study(selected, features=FEATURES)
    requirements = tuple(
        VisitFeature(feature=feature, modality=Modality.CLINICAL) for feature in FEATURES
    )
    config = QCConfig(
        feature_rules=tuple(
            FeatureRule(
                feature=feature,
                modality=Modality.CLINICAL,
                expected_unit="reported_value",
                minimum=0,
                required=True,
            )
            for feature in FEATURES
        ),
        expected_visits=(
            ExpectedVisit(
                visit_id="precision_1",
                anchor_id="normalized_precision_1",
                required_features=requirements,
            ),
            ExpectedVisit(
                visit_id="precision_2",
                anchor_id="normalized_precision_1",
                offset=timedelta(days=365),
                required_features=requirements,
            ),
        ),
    )
    report = BaselineLongitudinalQC(config).run(study)
    profile = StudyProfiler(config).profile(study)
    changes = paired_changes(selected, features=FEATURES)
    coverage = profile.coverage_frame()
    retention = profile.retention_frame()
    readiness = profile.paired_readiness_frame()
    distributions = profile.distributions_frame()
    attrition = profile.attrition_bias_frame()
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
            features=requirements,
            covariance_shrinkage=0.20,
            false_alarm_rate=0.05,
            minimum_reference_subjects=100,
        )
    ).fit(
        study,
        baseline=config.expected_visits[0],
        follow_up=config.expected_visits[1],
        reference_subject_ids=calibration_ids,
    )
    detection = detector.score(
        study,
        baseline=config.expected_visits[0],
        follow_up=config.expected_visits[1],
        subject_ids=evaluation_ids,
    )
    detection_scores = detection.results_frame().sort_values(
        "squared_mahalanobis_distance",
        ascending=False,
    )

    print(f"Source: {DAP_ARCHIVE_URL}")
    print(f"Canine records: {len(selected):,} visit rows from {len(study.subjects):,} dogs")
    print(report.summary())
    print("\nVisit-by-feature coverage (all dogs):")
    print(
        coverage.loc[coverage["cohort"] == "all"]
        .pivot(index="feature", columns="visit_id", values="coverage_fraction")
        .to_string(float_format=lambda value: f"{value:.1%}")
    )
    print("\nComplete-case retention:")
    print(
        retention.loc[retention["cohort"] == "all"].to_string(
            index=False,
            columns=[
                "from_visit_id",
                "to_visit_id",
                "from_complete_subjects",
                "retained_subjects",
                "retention_fraction",
            ],
            formatters={"retention_fraction": lambda value: f"{value:.1%}"},
        )
    )
    print("\nPaired-analysis readiness (all dogs):")
    print(
        readiness.loc[readiness["cohort"] == "all"].to_string(
            index=False,
            columns=["feature", "eligible_subjects", "paired_subjects", "paired_fraction"],
            formatters={"paired_fraction": lambda value: f"{value:.1%}"},
        )
    )
    print("\nVisit-level distributions and robust outliers (all dogs):")
    print(
        distributions.loc[distributions["cohort"] == "all"].to_string(
            index=False,
            columns=[
                "visit_id",
                "feature",
                "subjects",
                "mean",
                "standard_deviation",
                "median",
                "outlier_count",
            ],
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print("\nBaseline attrition-bias diagnostic (all dogs):")
    print(
        attrition.loc[attrition["cohort"] == "all"].to_string(
            index=False,
            columns=[
                "feature",
                "retained_subjects",
                "attrited_subjects",
                "retained_baseline_mean",
                "attrited_baseline_mean",
                "standardized_mean_difference",
            ],
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print("\nDescriptive precision_2 minus precision_1 changes:")
    print(changes.to_string())
    print("\nHeld-out multivariate change detection:")
    print(
        f"Calibration subjects with complete pairs: {detection.model.reference_subjects}; "
        f"held-out subjects scored: {len(detection.results)}; "
        f"held-out incomplete pairs: {len(detection.excluded_subject_ids)}; "
        f"detections: {sum(item.detected for item in detection.results)} "
        f"at nominal {detection.model.false_alarm_rate:.1%} false-alarm rate"
    )
    print(
        detection_scores.head(10).to_string(
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
        "\nInterpretation: this demonstrates canine longitudinal QC and visit-level "
        "missingness. The cohort is observational and has no rapamycin assignment, "
        "so these changes are not estimates of treatment efficacy."
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(args.output_dir / "visit_feature_coverage.csv", index=False)
        retention.to_csv(args.output_dir / "visit_retention.csv", index=False)
        readiness.to_csv(args.output_dir / "paired_analysis_readiness.csv", index=False)
        distributions.to_csv(args.output_dir / "feature_distributions.csv", index=False)
        attrition.to_csv(args.output_dir / "attrition_bias.csv", index=False)
        changes.to_csv(args.output_dir / "paired_chemistry_changes.csv")
        detection_scores.to_csv(
            args.output_dir / "multivariate_change_scores.csv",
            index=False,
        )
        (args.output_dir / "qc_report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
