from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from rejuvenationkit.detection import (  # noqa: E402
    ChangeDetectionModel,
    ChangeDetectionReport,
    SubjectChangeDetection,
)
from rejuvenationkit.schemas import Modality  # noqa: E402
from rejuvenationkit.sequential import (  # noqa: E402
    ModalityEvidence,
    SequentialDetectionModel,
    SequentialDetectionPoint,
    SequentialDetectionReport,
    SubjectSequentialDetection,
)
from rejuvenationkit.visualization import (  # noqa: E402
    plot_covariance_structure,
    plot_detection_scores,
    plot_sequential_classification,
    plot_sequential_modality_evidence,
    plot_sequential_trajectories,
    plot_subject_decomposition,
    plot_whitened_innovations,
    save_detection_figures,
    save_sequential_figures,
)


@pytest.fixture
def report() -> ChangeDetectionReport:
    model = ChangeDetectionModel(
        feature_names=("albumin", "ethanolamine"),
        feature_modalities=(None, None),
        reference_subjects=30,
        mean_change=(0, 0),
        covariance=((1.0, 0.5), (0.5, 1.0)),
        threshold=5.0,
        false_alarm_rate=0.05,
    )
    results = (
        SubjectChangeDetection(
            subject_id="dog-1",
            change=(1, 1),
            innovation=(1, 1),
            whitened_innovation=(1, 0.5),
            squared_mahalanobis_distance=1.25,
            empirical_tail_probability=0.5,
            detected=False,
        ),
        SubjectChangeDetection(
            subject_id="dog-2",
            change=(3, 3),
            innovation=(3, 3),
            whitened_innovation=(3, 1),
            squared_mahalanobis_distance=10,
            empirical_tail_probability=0.03,
            detected=True,
        ),
    )
    return ChangeDetectionReport(
        study_id="visual",
        baseline_visit_id="baseline",
        follow_up_visit_id="follow-up",
        model=model,
        results=results,
    )


def test_plot_functions_return_populated_figures(report: ChangeDetectionReport) -> None:
    figures = (
        plot_covariance_structure(report),
        plot_detection_scores(report),
        plot_whitened_innovations(report),
        plot_subject_decomposition(report, "dog-2"),
    )
    assert all(figure.axes for figure in figures)


def test_save_detection_figures(report: ChangeDetectionReport, tmp_path: Path) -> None:
    paths = save_detection_figures(report, tmp_path, prefix="test", dpi=72)

    assert len(paths) == 4
    assert all(path.stat().st_size > 0 for path in paths)


def test_visualization_rejects_invalid_requests(report: ChangeDetectionReport) -> None:
    with pytest.raises(ValueError, match="distinct"):
        plot_whitened_innovations(report, components=(0, 0))
    with pytest.raises(ValueError, match="not present"):
        plot_subject_decomposition(report, "missing")


@pytest.fixture
def sequential_report() -> SequentialDetectionReport:
    model = SequentialDetectionModel(
        feature_names=("activity", "ethanolamine"),
        feature_modalities=(Modality.WEARABLE, Modality.METABOLOMICS),
        reference_subjects=30,
        reference_transitions=60,
        mean_change_per_year=(0, 0),
        innovation_covariance_per_year=((1.0, 0.2), (0.2, 1.0)),
        maximum_cumulative_score_threshold=5.0,
        false_alarm_rate=0.05,
    )
    return SequentialDetectionReport(
        study_id="sequential-visual",
        visit_ids=("baseline", "year-1", "year-2"),
        model=model,
        results=(
            SubjectSequentialDetection(
                subject_id="dog-1",
                points=(
                    SequentialDetectionPoint(
                        from_visit_id="baseline",
                        to_visit_id="year-1",
                        elapsed_years=1,
                        interval_score=1,
                        cumulative_score=1,
                        empirical_tail_probability=0.5,
                        threshold_crossed=False,
                    ),
                    SequentialDetectionPoint(
                        from_visit_id="year-1",
                        to_visit_id="year-2",
                        elapsed_years=1,
                        interval_score=1,
                        cumulative_score=2,
                        empirical_tail_probability=0.4,
                        threshold_crossed=False,
                    ),
                ),
                detected=False,
                peak_cumulative_score=2,
                peak_modality_evidence=(
                    ModalityEvidence(modality=Modality.WEARABLE, channels=1, score=1),
                    ModalityEvidence(modality=Modality.METABOLOMICS, channels=1, score=1),
                ),
            ),
            SubjectSequentialDetection(
                subject_id="dog-2",
                points=(
                    SequentialDetectionPoint(
                        from_visit_id="baseline",
                        to_visit_id="year-1",
                        elapsed_years=1,
                        interval_score=7,
                        cumulative_score=7,
                        empirical_tail_probability=0.03,
                        threshold_crossed=True,
                    ),
                    SequentialDetectionPoint(
                        from_visit_id="year-1",
                        to_visit_id="year-2",
                        elapsed_years=1,
                        interval_score=6,
                        cumulative_score=8,
                        empirical_tail_probability=0.02,
                        threshold_crossed=True,
                    ),
                ),
                detected=True,
                onset_visit_id="year-1",
                persistent=True,
                peak_cumulative_score=8,
                peak_modality_evidence=(
                    ModalityEvidence(modality=Modality.WEARABLE, channels=1, score=5),
                    ModalityEvidence(modality=Modality.METABOLOMICS, channels=1, score=3),
                ),
            ),
        ),
    )


def test_sequential_plot_functions_and_save(
    sequential_report: SequentialDetectionReport,
    tmp_path: Path,
) -> None:
    figures = (
        plot_sequential_trajectories(sequential_report),
        plot_sequential_classification(sequential_report),
        plot_sequential_modality_evidence(sequential_report),
    )
    assert all(figure.axes for figure in figures)
    paths = save_sequential_figures(sequential_report, tmp_path, dpi=72)
    assert len(paths) == 3
    assert all(path.stat().st_size > 0 for path in paths)


def test_sequential_modality_plot_rejects_nonpositive_limit(
    sequential_report: SequentialDetectionReport,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        plot_sequential_modality_evidence(sequential_report, limit=0)
