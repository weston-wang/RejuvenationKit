"""Demonstrate sequential response phenotyping in a synthetic TRIAD-like trial.

The generated data resemble the schedule and measurement domains described in
the public TRIAD protocol. They are not Dog Aging Project observations or trial
results and must not be used to infer that rapamycin has these effects.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rejuvenationkit import (
    ExpectedVisit,
    Modality,
    Observation,
    SequentialDetectionConfig,
    SequentialTreatmentResponseDetector,
    Study,
    Subject,
    VisitFeature,
)

FEATURES = (
    VisitFeature(feature="fractional_shortening", modality=Modality.IMAGING),
    VisitFeature(feature="ea_ratio", modality=Modality.IMAGING),
    VisitFeature(feature="daily_activity_minutes", modality=Modality.WEARABLE),
    VisitFeature(feature="executive_function_score", modality=Modality.CLINICAL),
    VisitFeature(feature="globulins", modality=Modality.CLINICAL),
    VisitFeature(feature="alt", modality=Modality.CLINICAL),
)
VISIT_IDS = tuple(f"month_{month:02d}" for month in range(0, 37, 6))
START = datetime(2026, 1, 1, tzinfo=UTC)


def make_trial(seed: int = 2026) -> tuple[Study, tuple[ExpectedVisit, ...]]:
    """Generate 290 placebo and 290 treated longitudinal dog trajectories."""
    random = np.random.default_rng(seed)
    placebo_ids = tuple(f"placebo-{index:03d}" for index in range(290))
    treated_ids = tuple(f"rapamycin-{index:03d}" for index in range(290))
    phenotypes = _assigned_phenotypes(treated_ids, random)
    subjects = tuple(
        Subject(
            subject_id=subject_id,
            cohort="placebo" if subject_id in placebo_ids else "rapamycin",
            anchors={"randomization": START},
            attributes={
                "species": "Canis lupus familiaris",
                "simulated_response_phenotype": phenotypes.get(subject_id, "placebo"),
            },
        )
        for subject_id in (*placebo_ids, *treated_ids)
    )
    visits = tuple(
        ExpectedVisit(
            visit_id=visit_id,
            anchor_id="randomization",
            offset=timedelta(days=182.62125 * index),
            required_features=FEATURES,
        )
        for index, visit_id in enumerate(VISIT_IDS)
    )
    baseline_mean = np.array([28.0, 1.25, 95.0, 78.0, 2.8, 42.0])
    aging_drift = np.array([-0.45, -0.035, -4.0, -2.2, 0.08, 1.2])
    innovation_covariance = _covariance()
    observations: list[Observation] = []
    for subject in subjects:
        value = baseline_mean + random.normal(0, [2.5, 0.12, 14, 7, 0.25, 8])
        phenotype = str(subject.attributes["simulated_response_phenotype"])
        for visit_index, visit in enumerate(visits):
            if visit_index:
                value = (
                    value
                    + 0.5 * aging_drift
                    + random.multivariate_normal(np.zeros(len(FEATURES)), innovation_covariance)
                    + _treatment_increment(phenotype, visit_index)
                )
            for feature, measurement in zip(FEATURES, value, strict=True):
                if subject.cohort == "rapamycin" and visit_index > 0 and random.random() < 0.025:
                    continue
                observations.append(
                    Observation(
                        subject_id=subject.subject_id,
                        timestamp=visit.scheduled_for(subject) or START,
                        modality=feature.modality or Modality.CLINICAL,
                        feature=feature.feature,
                        value=float(measurement),
                        unit="simulated_reported_value",
                    )
                )
    return (
        Study(
            study_id="synthetic-triad-like-rapamycin",
            subjects=subjects,
            observations=tuple(observations),
            metadata={
                "synthetic": True,
                "design": "TRIAD-like randomized placebo-controlled demonstration",
                "warning": "Not DAP observations or rapamycin results",
            },
        ),
        visits,
    )


def main() -> None:
    """Run the synthetic TRIAD-like sequential response demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.plots and args.output_dir is None:
        parser.error("--plots requires --output-dir")

    study, visits = make_trial()
    placebo_ids = tuple(
        subject.subject_id for subject in study.subjects if subject.cohort == "placebo"
    )
    treated_ids = tuple(
        subject.subject_id for subject in study.subjects if subject.cohort == "rapamycin"
    )
    report = (
        SequentialTreatmentResponseDetector(
            SequentialDetectionConfig(
                features=FEATURES,
                covariance_shrinkage=0.20,
                false_alarm_rate=0.05,
                minimum_reference_subjects=250,
                persistence_crossings=2,
            )
        )
        .fit(
            study,
            visits=visits,
            reference_subject_ids=placebo_ids,
        )
        .score(
            study,
            visits=visits,
            subject_ids=treated_ids,
        )
    )
    truth = {
        subject.subject_id: str(subject.attributes["simulated_response_phenotype"])
        for subject in study.subjects
        if subject.cohort == "rapamycin"
    }
    summary = report.results_frame()
    summary["simulated_truth"] = summary["subject_id"].map(truth)
    summary["monitoring_status"] = summary.apply(_monitoring_status, axis=1)
    dominant_modality = {
        item.subject_id: max(
            item.peak_modality_evidence,
            key=lambda evidence: evidence.score,
        ).modality
        for item in report.results
    }
    summary["dominant_modality"] = summary["subject_id"].map(
        lambda subject_id: (
            dominant_modality[subject_id].value
            if dominant_modality[subject_id] is not None
            else "unspecified"
        )
    )
    table = pd.crosstab(
        summary["simulated_truth"],
        summary["monitoring_status"],
        margins=True,
    )
    onsets = Counter(
        item.onset_visit_id for item in report.results if item.onset_visit_id is not None
    )

    print("SYNTHETIC TRIAD-LIKE DEMONSTRATION — NOT DAP TRIAL RESULTS")
    print(f"Dogs: {len(study.subjects)}; visits: {len(visits)}; channels: {len(FEATURES)}")
    print(
        f"Placebo reference trajectories: {report.model.reference_subjects}; "
        f"rapamycin-arm trajectories scored: {len(report.results)}; "
        f"insufficient trajectories: {len(report.excluded_subject_ids)}"
    )
    print("\nInjected phenotype versus inferred monitoring status:")
    print(table.to_string())
    print("\nFirst detected visit:")
    print(pd.Series(onsets, name="dogs").sort_index().to_string())
    print("\nDominant evidence modality among detected dogs:")
    print(
        pd.crosstab(
            summary.loc[summary["detected"], "simulated_truth"],
            summary.loc[summary["detected"], "dominant_modality"],
            margins=True,
        ).to_string()
    )
    print(
        "\nWhat the toolkit adds: subject-level onset, persistence, transient-response "
        "recognition, and modality-localized evidence under a reference-calibrated repeated-look "
        "threshold. This demonstration does not establish rapamycin efficacy."
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_dir / "triad_like_subject_summary.csv", index=False)
        report.trajectory_frame().to_csv(
            args.output_dir / "triad_like_sequential_trajectories.csv",
            index=False,
        )
        table.to_csv(args.output_dir / "triad_like_confusion_table.csv")
        (args.output_dir / "triad_like_report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if args.plots:
            from rejuvenationkit.visualization import save_sequential_figures

            for path in save_sequential_figures(
                report,
                args.output_dir,
                prefix="triad-like",
            ):
                print(path)


def _assigned_phenotypes(
    treated_ids: tuple[str, ...],
    random: np.random.Generator,
) -> dict[str, str]:
    shuffled = list(treated_ids)
    random.shuffle(shuffled)
    counts = (
        ("multisystem_sustained", 58),
        ("cardiac_sustained", 44),
        ("early_transient", 29),
        ("clinical_safety_shift", 15),
    )
    phenotypes: dict[str, str] = {}
    start = 0
    for phenotype, count in counts:
        for subject_id in shuffled[start : start + count]:
            phenotypes[subject_id] = phenotype
        start += count
    for subject_id in shuffled[start:]:
        phenotypes[subject_id] = "no_material_response"
    return phenotypes


def _treatment_increment(phenotype: str, transition: int) -> np.ndarray:
    if phenotype == "multisystem_sustained":
        return np.array([0.8, 0.055, 5.5, 3.2, -0.08, -0.9])
    if phenotype == "cardiac_sustained":
        return np.array([1.1, 0.075, 0.0, 0.0, 0.0, 0.0])
    if phenotype == "early_transient":
        if transition == 1:
            return np.array([4.0, 0.30, 25.0, 16.0, -0.35, -5.0])
        if transition == 2:
            return np.array([-4.0, -0.30, -25.0, -16.0, 0.35, 5.0])
    if phenotype == "clinical_safety_shift":
        return np.array([0.0, 0.0, 0.0, 0.0, 0.24, 5.0])
    return np.zeros(len(FEATURES))


def _covariance() -> np.ndarray:
    standard_deviation = np.array([0.7, 0.045, 3.8, 2.0, 0.07, 1.8])
    correlation = np.eye(len(FEATURES))
    correlation[0, 1] = correlation[1, 0] = 0.65
    correlation[2, 3] = correlation[3, 2] = 0.45
    correlation[4, 5] = correlation[5, 4] = 0.30
    return np.outer(standard_deviation, standard_deviation) * correlation


def _monitoring_status(row: pd.Series) -> str:
    if bool(row["persistent"]):
        return "persistent"
    if bool(row["transient"]):
        return "transient"
    if bool(row["detected"]):
        return "awaiting_persistence"
    return "not_detected"


if __name__ == "__main__":
    main()
