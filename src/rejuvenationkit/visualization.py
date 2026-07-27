"""DSP diagnostic visualizations for multivariate change detection."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray

from rejuvenationkit.detection import ChangeDetectionReport, SubjectChangeDetection
from rejuvenationkit.sequential import SequentialDetectionReport


def plot_covariance_structure(
    report: ChangeDetectionReport,
    *,
    correlation: bool = True,
) -> Figure:
    """Plot the fitted reference covariance or correlation matrix."""
    covariance = np.asarray(report.model.covariance, dtype=float)
    matrix = _correlation(covariance) if correlation else covariance
    labels = report.model.feature_names
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    limit = 1.0 if correlation else max(float(np.abs(matrix).max()), 1e-12)
    image = axis.imshow(
        matrix,
        cmap="coolwarm",
        aspect="equal",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Reference change correlation" if correlation else "Reference change covariance")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("Correlation" if correlation else "Covariance")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(matrix[row, column]) > 0.55 * limit else "black",
            )
    return figure


def plot_detection_scores(report: ChangeDetectionReport) -> Figure:
    """Plot ordered Mahalanobis scores against the calibrated threshold."""
    ordered = sorted(report.results, key=lambda item: item.squared_mahalanobis_distance)
    scores = np.asarray([item.squared_mahalanobis_distance for item in ordered])
    detected = np.asarray([item.detected for item in ordered])
    ranks = np.arange(1, len(ordered) + 1)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.scatter(ranks[~detected], scores[~detected], s=22, alpha=0.7, label="Not detected")
    axis.scatter(ranks[detected], scores[detected], s=34, marker="x", label="Detected")
    axis.axhline(
        report.model.threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({report.model.false_alarm_rate:.1%} FAR)",
    )
    axis.set_xlabel("Held-out subject rank")
    axis.set_ylabel("Squared Mahalanobis distance")
    axis.set_title("Multivariate detection statistic")
    axis.legend()
    axis.grid(alpha=0.2)
    return figure


def plot_whitened_innovations(
    report: ChangeDetectionReport,
    *,
    components: tuple[int, int] = (0, 1),
) -> Figure:
    """Plot two whitened innovation components and projected threshold circle."""
    if components[0] == components[1] or min(components) < 0:
        raise ValueError("components must be distinct non-negative indices")
    dimension = len(report.model.feature_names)
    if max(components) >= dimension:
        raise ValueError(f"component index exceeds detector dimension {dimension}")
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for detected, marker, label in (
        (False, "o", "Not detected"),
        (True, "x", "Detected"),
    ):
        selected = [item for item in report.results if item.detected is detected]
        axis.scatter(
            [item.whitened_innovation[components[0]] for item in selected],
            [item.whitened_innovation[components[1]] for item in selected],
            marker=marker,
            s=34,
            alpha=0.75,
            label=label,
        )
    radius = math.sqrt(report.model.threshold)
    axis.add_patch(
        Circle(
            (0, 0),
            radius,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            label="Projected threshold radius",
        )
    )
    axis.axhline(0, linewidth=0.8, alpha=0.4)
    axis.axvline(0, linewidth=0.8, alpha=0.4)
    axis.set_xlabel(f"Whitened component {components[0] + 1}")
    axis.set_ylabel(f"Whitened component {components[1] + 1}")
    axis.set_title("Whitened longitudinal innovations")
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend()
    axis.grid(alpha=0.2)
    return figure


def plot_subject_decomposition(
    report: ChangeDetectionReport,
    subject_id: str,
) -> Figure:
    """Plot raw feature innovations and whitened component energy."""
    result = _subject_result(report, subject_id)
    labels = report.model.feature_names
    raw = np.asarray(result.innovation)
    energy = np.square(np.asarray(result.whitened_innovation))
    figure, (raw_axis, energy_axis) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        constrained_layout=True,
    )
    raw_axis.bar(labels, raw)
    raw_axis.axhline(0, linewidth=0.8)
    raw_axis.set_ylabel("Follow-up change minus reference mean")
    raw_axis.set_title(f"Subject {subject_id}: channel innovations")
    raw_axis.tick_params(axis="x", rotation=30)

    components = [f"W{index + 1}" for index in range(len(energy))]
    energy_axis.bar(components, energy)
    energy_axis.axhline(
        report.model.threshold,
        linestyle="--",
        linewidth=1.2,
        label="Total-score threshold",
    )
    energy_axis.set_ylabel("Squared whitened amplitude")
    energy_axis.set_title(f"Whitened energy; total D²={result.squared_mahalanobis_distance:.2f}")
    energy_axis.legend()
    energy_axis.grid(axis="y", alpha=0.2)
    return figure


def save_detection_figures(
    report: ChangeDetectionReport,
    output_dir: Path,
    *,
    subject_id: str | None = None,
    prefix: str = "detection",
    dpi: int = 160,
) -> tuple[Path, ...]:
    """Render the standard DSP diagnostic figure set to PNG files."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_subject = subject_id or _highest_scoring_subject(report).subject_id
    figures = (
        ("covariance", plot_covariance_structure(report)),
        ("scores", plot_detection_scores(report)),
        ("whitened", plot_whitened_innovations(report)),
        ("decomposition", plot_subject_decomposition(report, selected_subject)),
    )
    paths: list[Path] = []
    for name, figure in figures:
        path = output_dir / f"{prefix}-{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


def plot_sequential_trajectories(report: SequentialDetectionReport) -> Figure:
    """Plot cumulative evidence through time against the calibrated threshold."""
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for result in report.results:
        color = "tab:red" if result.detected else "tab:blue"
        alpha = 0.8 if result.detected else 0.18
        axis.plot(
            range(1, len(result.points) + 1),
            [point.cumulative_score for point in result.points],
            color=color,
            alpha=alpha,
            linewidth=1.5 if result.detected else 0.8,
        )
    axis.axhline(
        report.model.maximum_cumulative_score_threshold,
        color="black",
        linestyle="--",
        label=f"Subject-level threshold ({report.model.false_alarm_rate:.1%} FAR)",
    )
    axis.set_xticks(
        range(1, len(report.visit_ids)),
        report.visit_ids[1:],
        rotation=25,
        ha="right",
    )
    axis.set_xlabel("Evidence accumulated through visit")
    axis.set_ylabel("Cumulative whitened energy")
    axis.set_title("Sequential departures from reference aging dynamics")
    axis.grid(alpha=0.2)
    axis.legend()
    return figure


def plot_sequential_classification(report: SequentialDetectionReport) -> Figure:
    """Plot peak evidence grouped by undetected, transient, and persistent status."""
    groups = {
        "Undetected": [item for item in report.results if not item.detected],
        "Unconfirmed": [
            item
            for item in report.results
            if item.detected and not item.transient and not item.persistent
        ],
        "Transient": [item for item in report.results if item.transient],
        "Persistent": [item for item in report.results if item.persistent],
    }
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for index, (label, results) in enumerate(groups.items()):
        axis.scatter(
            np.full(len(results), index),
            [item.peak_cumulative_score for item in results],
            alpha=0.65,
            label=f"{label} (n={len(results)})",
        )
    axis.axhline(
        report.model.maximum_cumulative_score_threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    axis.set_xticks(range(len(groups)), groups)
    axis.set_ylabel("Peak cumulative score")
    axis.set_title("Sequential response classification")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    return figure


def plot_sequential_modality_evidence(
    report: SequentialDetectionReport,
    *,
    limit: int = 12,
) -> Figure:
    """Plot peak within-modality evidence for the strongest subjects."""
    if limit < 1:
        raise ValueError("limit must be positive")
    selected = sorted(
        report.results,
        key=lambda item: item.peak_cumulative_score,
        reverse=True,
    )[:limit]
    modalities = sorted(
        {
            evidence.modality.value if evidence.modality is not None else "unspecified"
            for item in selected
            for evidence in item.peak_modality_evidence
        }
    )
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(selected))
    width = 0.8 / max(len(modalities), 1)
    for modality_index, modality in enumerate(modalities):
        scores = [
            next(
                (
                    evidence.score
                    for evidence in item.peak_modality_evidence
                    if (evidence.modality.value if evidence.modality is not None else "unspecified")
                    == modality
                ),
                0.0,
            )
            for item in selected
        ]
        axis.bar(
            x + (modality_index - (len(modalities) - 1) / 2) * width,
            scores,
            width,
            label=modality,
        )
    axis.set_xticks(x, [item.subject_id for item in selected], rotation=45, ha="right")
    axis.set_ylabel("Within-modality peak evidence")
    axis.set_title("Channels driving the strongest sequential departures")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    return figure


def save_sequential_figures(
    report: SequentialDetectionReport,
    output_dir: Path,
    *,
    prefix: str = "sequential",
    dpi: int = 160,
) -> tuple[Path, ...]:
    """Render the standard sequential diagnostic figure set."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = (
        ("trajectories", plot_sequential_trajectories(report)),
        ("classification", plot_sequential_classification(report)),
        ("modalities", plot_sequential_modality_evidence(report)),
    )
    paths: list[Path] = []
    for name, figure in figures:
        path = output_dir / f"{prefix}-{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


def _correlation(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    standard_deviation = np.sqrt(np.diag(covariance))
    denominator = np.outer(standard_deviation, standard_deviation)
    return np.asarray(
        np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0,
        ),
        dtype=np.float64,
    )


def _subject_result(
    report: ChangeDetectionReport,
    subject_id: str,
) -> SubjectChangeDetection:
    try:
        return next(item for item in report.results if item.subject_id == subject_id)
    except StopIteration as error:
        raise ValueError(
            f"subject {subject_id!r} is not present in the detection report"
        ) from error


def _highest_scoring_subject(report: ChangeDetectionReport) -> SubjectChangeDetection:
    if not report.results:
        raise ValueError("detection report has no scored subjects")
    return max(report.results, key=lambda item: item.squared_mahalanobis_distance)
