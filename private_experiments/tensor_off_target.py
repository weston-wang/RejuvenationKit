"""Tensor factorization for pathway-resolved treatment response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MarkerAnnotation:
    """Biological interpretation attached to one tensor channel."""

    name: str
    domain: str
    pathway: str
    intended_target: bool = False


@dataclass(frozen=True)
class ResponseFactor:
    """One orthogonal multichannel treatment-response factor."""

    index: int
    explained_fraction: float
    dominant_pathway: str
    target_energy_fraction: float
    pathway_energy: dict[str, float]
    top_markers: tuple[str, ...]

    @property
    def off_target(self) -> bool:
        """Flag factors whose loading energy is mostly outside intended targets."""
        return self.target_energy_fraction < 0.5


class TensorResponseModel:
    """Factor treatment responses relative to a placebo/reference tensor.

    Tensors use ``(subjects, time, markers)`` order. The reference mean and
    scale are estimated separately at each visit, after which SVD discovers
    coupled marker factors across treated subject-time response surfaces.
    """

    def __init__(self, rank: int = 4) -> None:
        """Create an unfitted response tensor model."""
        if rank < 1:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.annotations_: tuple[MarkerAnnotation, ...] | None = None
        self.reference_mean_: FloatArray | None = None
        self.reference_scale_: FloatArray | None = None
        self.marker_loadings_: FloatArray | None = None
        self.singular_values_: FloatArray | None = None

    def fit(
        self,
        reference: FloatArray,
        treated: FloatArray,
        annotations: tuple[MarkerAnnotation, ...],
    ) -> TensorResponseModel:
        """Fit marker factors from reference-standardized treatment responses."""
        reference_array = _tensor(reference, "reference")
        treated_array = _tensor(treated, "treated")
        if reference_array.shape[1:] != treated_array.shape[1:]:
            raise ValueError("reference and treated tensors require matching time and markers")
        if len(annotations) != reference_array.shape[2]:
            raise ValueError("one marker annotation is required per tensor channel")
        mean = reference_array.mean(axis=0)
        scale = reference_array.std(axis=0, ddof=1)
        pooled_scale = reference_array.std(axis=(0, 1), ddof=1)
        scale = np.where(scale > 1e-8, scale, pooled_scale[None, :])
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = (treated_array - mean[None, :, :]) / scale[None, :, :]
        matrix = standardized.reshape(-1, standardized.shape[2])
        _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        retained = min(self.rank, len(singular_values))
        self.annotations_ = annotations
        self.reference_mean_ = np.asarray(mean, dtype=np.float64)
        self.reference_scale_ = np.asarray(scale, dtype=np.float64)
        self.marker_loadings_ = np.asarray(right[:retained].T, dtype=np.float64)
        self.singular_values_ = np.asarray(singular_values[:retained], dtype=np.float64)
        return self

    def transform(self, values: FloatArray) -> tuple[FloatArray, ...]:
        """Project subject trajectories into learned response-factor coordinates."""
        self._require_fitted()
        assert self.reference_mean_ is not None
        assert self.reference_scale_ is not None
        assert self.marker_loadings_ is not None
        array = _tensor(values, "values")
        if array.shape[1:] != self.reference_mean_.shape:
            raise ValueError("values tensor differs from fitted time-marker shape")
        standardized = (array - self.reference_mean_[None, :, :]) / self.reference_scale_[
            None, :, :
        ]
        scores = standardized @ self.marker_loadings_
        return tuple(np.asarray(subject, dtype=np.float64) for subject in scores)

    def factors(self, *, top_markers: int = 5) -> tuple[ResponseFactor, ...]:
        """Summarize pathway energy and intended-target specificity."""
        self._require_fitted()
        if top_markers < 1:
            raise ValueError("top_markers must be positive")
        assert self.annotations_ is not None
        assert self.marker_loadings_ is not None
        assert self.singular_values_ is not None
        total_singular_energy = float(np.square(self.singular_values_).sum())
        reports = []
        for index in range(self.marker_loadings_.shape[1]):
            loading = self.marker_loadings_[:, index]
            energy = np.square(loading)
            pathway_energy: dict[str, float] = {}
            for annotation, value in zip(self.annotations_, energy, strict=True):
                pathway_energy[annotation.pathway] = pathway_energy.get(
                    annotation.pathway, 0.0
                ) + float(value)
            target_energy = sum(
                float(value)
                for annotation, value in zip(self.annotations_, energy, strict=True)
                if annotation.intended_target
            )
            ordered = np.argsort(energy)[::-1][:top_markers]
            reports.append(
                ResponseFactor(
                    index=index,
                    explained_fraction=(
                        float(self.singular_values_[index] ** 2) / total_singular_energy
                    ),
                    dominant_pathway=max(pathway_energy, key=pathway_energy.__getitem__),
                    target_energy_fraction=target_energy / float(energy.sum()),
                    pathway_energy=pathway_energy,
                    top_markers=tuple(self.annotations_[item].name for item in ordered),
                )
            )
        return tuple(reports)

    def _require_fitted(self) -> None:
        if self.marker_loadings_ is None:
            raise RuntimeError("fit the tensor response model before use")


def _tensor(values: FloatArray, label: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite subjects-by-time-by-markers tensor")
    if min(array.shape) < 2:
        raise ValueError(f"{label} tensor dimensions must each contain at least two entries")
    return array
