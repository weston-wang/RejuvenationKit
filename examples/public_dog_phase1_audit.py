"""Generate a complete Phase 1 audit bundle from public Dog Aging Project data.

This is an observational cohort with no rapamycin assignment. The resulting
bundle evaluates data quality and analysis readiness, not treatment efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import timedelta
from pathlib import Path

from rejuvenationkit import (
    ChangeDetectionAuditPlan,
    ChangeDetectionConfig,
    ExpectedVisit,
    FeatureRule,
    Modality,
    Phase1AuditConfig,
    QCConfig,
    VisitFeature,
    run_phase1_audit,
)
from rejuvenationkit.datasets.dog_aging_project import (
    DAP_ARCHIVE_URL,
    build_study,
    download_archive,
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
    """Download public canine measurements and write the complete audit bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("examples/data/cache/dap_baseline_phenotypes.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/public-dog-phase1-audit"),
    )
    args = parser.parse_args()

    archive = download_archive(args.cache)
    chemistry = read_chemistry(archive)
    selected = select_visits(chemistry)
    study = build_study(selected, features=FEATURES)
    requirements = tuple(
        VisitFeature(feature=feature, modality=Modality.CLINICAL) for feature in FEATURES
    )
    config = Phase1AuditConfig(
        qc=QCConfig(
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
    )
    report = run_phase1_audit(
        study,
        config=config,
        output_dir=args.output_dir,
        change_detection_plan=ChangeDetectionAuditPlan(
            config=ChangeDetectionConfig(
                features=requirements,
                covariance_shrinkage=0.20,
                false_alarm_rate=0.05,
                minimum_reference_subjects=100,
            ),
            baseline_visit_id="precision_1",
            follow_up_visit_id="precision_2",
            reference_subject_ids=tuple(
                subject.subject_id
                for subject in study.subjects
                if hashlib.sha256(subject.subject_id.encode()).digest()[0] < 179
            ),
            evaluation_subject_ids=tuple(
                subject.subject_id
                for subject in study.subjects
                if hashlib.sha256(subject.subject_id.encode()).digest()[0] >= 179
            ),
        ),
    )

    print(f"Source: {DAP_ARCHIVE_URL}")
    print(report.qc.summary())
    print(f"Bundle: {args.output_dir.resolve()}")
    print(f"Artifacts: {len(report.artifacts)}")
    if report.change_detection is not None:
        print(
            f"Held-out trajectories: {len(report.change_detection.results)} scored, "
            f"{sum(item.detected for item in report.change_detection.results)} detected"
        )
    print(
        "Interpretation: observational canine quality and readiness audit; "
        "no treatment-effect inference was run."
    )


if __name__ == "__main__":
    main()
