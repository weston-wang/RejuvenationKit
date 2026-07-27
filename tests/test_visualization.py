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
from rejuvenationkit.visualization import (  # noqa: E402
    plot_covariance_structure,
    plot_detection_scores,
    plot_subject_decomposition,
    plot_whitened_innovations,
    save_detection_figures,
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
