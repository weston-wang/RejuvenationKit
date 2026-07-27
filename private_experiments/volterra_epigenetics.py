"""Exploratory second-order Volterra modeling of epigenetic-marker response."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VolterraConfig:
    """Configuration for a regularized causal second-order Volterra model."""

    memory: int = 3
    linear_penalty: float = 1e-3
    quadratic_penalty: float = 1e-2
    fit_intercept: bool = True

    def __post_init__(self) -> None:
        """Validate memory and regularization values."""
        if self.memory < 1:
            raise ValueError("memory must be positive")
        if self.linear_penalty < 0 or self.quadratic_penalty < 0:
            raise ValueError("penalties must be non-negative")


@dataclass(frozen=True)
class VolterraComponents:
    """Predicted baseline, linear-memory, and nonlinear-interaction terms."""

    baseline: FloatArray
    linear: FloatArray
    quadratic: FloatArray

    @property
    def total(self) -> FloatArray:
        """Return the sum of all modeled contributions."""
        return self.baseline + self.linear + self.quadratic


class EpigeneticVolterraModel:
    """Fit marker responses to causal exposure history through second order.

    Inputs have shape ``(time, channels)`` and marker responses have shape
    ``(time, markers)``. Multiple animals are passed as separate arrays, which
    prevents lagged history from crossing animal boundaries.
    """

    def __init__(self, config: VolterraConfig | None = None) -> None:
        """Create an unfitted exploratory model."""
        self.config = config or VolterraConfig()
        self.input_channels_: int | None = None
        self.output_markers_: int | None = None
        self.intercept_: FloatArray | None = None
        self.linear_kernel_: FloatArray | None = None
        self.quadratic_kernel_: FloatArray | None = None
        self._linear_coefficients: FloatArray | None = None
        self._quadratic_coefficients: FloatArray | None = None
        self._quadratic_pairs: tuple[tuple[int, int], ...] | None = None

    def fit(
        self,
        exposures: tuple[FloatArray, ...],
        marker_responses: tuple[FloatArray, ...],
    ) -> EpigeneticVolterraModel:
        """Fit regularized kernels from one or more animal trajectories."""
        input_channels, output_markers = _validate_trajectories(
            exposures,
            marker_responses,
            self.config.memory,
        )
        histories, targets = _stack_training_rows(
            exposures,
            marker_responses,
            self.config.memory,
        )
        pairs = tuple(combinations_with_replacement(range(histories.shape[1]), 2))
        quadratic = _quadratic_design(histories, pairs)
        columns: list[FloatArray] = []
        penalties: list[FloatArray] = []
        if self.config.fit_intercept:
            columns.append(np.ones((len(histories), 1), dtype=np.float64))
            penalties.append(np.zeros(1, dtype=np.float64))
        columns.append(histories)
        columns.append(quadratic)
        penalties.extend(
            (
                np.full(histories.shape[1], self.config.linear_penalty),
                np.full(quadratic.shape[1], self.config.quadratic_penalty),
            )
        )
        design = np.column_stack(columns)
        penalty = np.concatenate(penalties)
        gram = design.T @ design + np.diag(penalty)
        coefficients = np.asarray(
            np.linalg.solve(gram, design.T @ targets),
            dtype=np.float64,
        )
        offset = 1 if self.config.fit_intercept else 0
        linear_end = offset + histories.shape[1]

        self.input_channels_ = input_channels
        self.output_markers_ = output_markers
        self.intercept_ = (
            coefficients[0].copy()
            if self.config.fit_intercept
            else np.zeros(output_markers, dtype=np.float64)
        )
        linear_coefficients: FloatArray = coefficients[offset:linear_end].copy()
        quadratic_coefficients: FloatArray = coefficients[linear_end:].copy()
        self._linear_coefficients = linear_coefficients
        self._quadratic_coefficients = quadratic_coefficients
        self._quadratic_pairs = pairs
        self.linear_kernel_ = linear_coefficients.reshape(
            self.config.memory,
            input_channels,
            output_markers,
        ).transpose(2, 1, 0)
        self.quadratic_kernel_ = _symmetric_kernel(
            quadratic_coefficients,
            pairs,
            output_markers,
            input_channels,
            self.config.memory,
        )
        return self

    def predict(self, exposures: tuple[FloatArray, ...]) -> tuple[FloatArray, ...]:
        """Predict each animal's marker trajectory after the initial memory window."""
        return tuple(component.total for component in self.decompose(exposures))

    def decompose(
        self,
        exposures: tuple[FloatArray, ...],
    ) -> tuple[VolterraComponents, ...]:
        """Separate baseline, linear-memory, and nonlinear contributions."""
        self._require_fitted()
        assert self.input_channels_ is not None
        assert self.output_markers_ is not None
        assert self.intercept_ is not None
        assert self._linear_coefficients is not None
        assert self._quadratic_coefficients is not None
        assert self._quadratic_pairs is not None
        results: list[VolterraComponents] = []
        for exposure in exposures:
            array = _as_matrix(exposure, "exposure")
            if array.shape[1] != self.input_channels_:
                raise ValueError("exposure input-channel count differs from fitted model")
            histories = _lagged_histories(array, self.config.memory)
            baseline = np.broadcast_to(
                self.intercept_,
                (len(histories), self.output_markers_),
            ).copy()
            linear = histories @ self._linear_coefficients
            quadratic = (
                _quadratic_design(histories, self._quadratic_pairs) @ self._quadratic_coefficients
            )
            results.append(
                VolterraComponents(
                    baseline=np.asarray(baseline, dtype=np.float64),
                    linear=np.asarray(linear, dtype=np.float64),
                    quadratic=np.asarray(quadratic, dtype=np.float64),
                )
            )
        return tuple(results)

    def _require_fitted(self) -> None:
        if self.linear_kernel_ is None or self.quadratic_kernel_ is None:
            raise RuntimeError("fit the Volterra model before prediction")


def _validate_trajectories(
    exposures: tuple[FloatArray, ...],
    responses: tuple[FloatArray, ...],
    memory: int,
) -> tuple[int, int]:
    if not exposures or len(exposures) != len(responses):
        raise ValueError("exposures and marker responses require matching non-empty trajectories")
    input_channels: int | None = None
    output_markers: int | None = None
    for exposure, response in zip(exposures, responses, strict=True):
        input_array = _as_matrix(exposure, "exposure")
        output_array = _as_matrix(response, "marker response")
        if len(input_array) != len(output_array):
            raise ValueError("exposure and marker response lengths must match within each animal")
        if len(input_array) < memory:
            raise ValueError("each trajectory must contain at least memory samples")
        input_channels = input_channels or input_array.shape[1]
        output_markers = output_markers or output_array.shape[1]
        if input_array.shape[1] != input_channels or output_array.shape[1] != output_markers:
            raise ValueError("all trajectories must have consistent channel counts")
    assert input_channels is not None
    assert output_markers is not None
    return input_channels, output_markers


def _stack_training_rows(
    exposures: tuple[FloatArray, ...],
    responses: tuple[FloatArray, ...],
    memory: int,
) -> tuple[FloatArray, FloatArray]:
    histories = []
    targets = []
    for exposure, response in zip(exposures, responses, strict=True):
        histories.append(_lagged_histories(exposure, memory))
        targets.append(_as_matrix(response, "marker response")[memory - 1 :])
    return (
        np.asarray(np.vstack(histories), dtype=np.float64),
        np.asarray(np.vstack(targets), dtype=np.float64),
    )


def _lagged_histories(exposure: FloatArray, memory: int) -> FloatArray:
    array = _as_matrix(exposure, "exposure")
    rows = [
        np.concatenate([array[index - lag] for lag in range(memory)])
        for index in range(memory - 1, len(array))
    ]
    return np.asarray(rows, dtype=np.float64)


def _quadratic_design(
    histories: FloatArray,
    pairs: tuple[tuple[int, int], ...],
) -> FloatArray:
    return np.asarray(
        np.column_stack([histories[:, first] * histories[:, second] for first, second in pairs]),
        dtype=np.float64,
    )


def _symmetric_kernel(
    coefficients: FloatArray,
    pairs: tuple[tuple[int, int], ...],
    outputs: int,
    inputs: int,
    memory: int,
) -> FloatArray:
    flattened = np.zeros((outputs, inputs * memory, inputs * memory), dtype=np.float64)
    for pair_index, (first, second) in enumerate(pairs):
        flattened[:, first, second] = coefficients[pair_index]
        flattened[:, second, first] = coefficients[pair_index]
    return flattened.reshape(outputs, memory, inputs, memory, inputs).transpose(0, 2, 1, 4, 3)


def _as_matrix(values: FloatArray, label: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite two-dimensional array")
    return array
