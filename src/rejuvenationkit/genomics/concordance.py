"""Cross-intervention transcriptomic concordance with shared-control correction.

Studies that compare several interventions often reuse one control group per
stratum. Two fold-change vectors computed against the same control mice then
share that control's sampling noise, so they correlate even when the
interventions have unrelated effects. With equal group sizes and no real effects
the expected correlation is about 0.5, which is easily mistaken for a shared
"longevity signature".

:func:`intervention_concordance` estimates the correlation of the *true* effect
vectors instead. It subtracts the shared-control noise covariance, divides by
noise-corrected signal variances, cross-checks the result with an estimator that
never lets both contrasts use the same control animal, and tests the corrected
covariance against a within-stratum label-permutation null.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from itertools import combinations
from math import isfinite, sqrt
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrataPolicy(StrEnum):
    """Which strata each contrast of a pair averages over."""

    COMMON = "common"
    OWN = "own"


class InterventionContrast(BaseModel):
    """One treated group compared with its declared control group."""

    model_config = ConfigDict(frozen=True)

    treated_group: str = Field(min_length=1)
    control_group: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_distinct_groups(self) -> Self:
        """Reject a contrast of a group with itself."""
        if self.treated_group == self.control_group:
            raise ValueError("treated and control groups must differ")
        return self

    @property
    def label(self) -> str:
        """Return a compact ``treated-vs-control`` label."""
        return f"{self.treated_group}-vs-{self.control_group}"


class InterventionConcordance(BaseModel):
    """Naive and shared-control-corrected concordance of two effect vectors."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    first: InterventionContrast
    second: InterventionContrast
    strata_policy: StrataPolicy
    first_strata: tuple[str, ...] = Field(min_length=1)
    second_strata: tuple[str, ...] = Field(min_length=1)
    shared_control_strata: tuple[str, ...]
    feature_count: int = Field(ge=3)
    naive_correlation: float = Field(ge=-1, le=1)
    corrected_correlation: float | None
    disjoint_control_correlation: float | None
    corrected_covariance: float
    shared_control_noise_covariance: float = Field(ge=0)
    first_signal_fraction: float
    second_signal_fraction: float
    permutation_p_value: float = Field(gt=0, le=1)
    permutations: int = Field(ge=1)
    random_seed: int

    @property
    def artifact_correlation(self) -> float | None:
        """Return the part of the naive correlation explained by correction."""
        if self.corrected_correlation is None:
            return None
        return self.naive_correlation - self.corrected_correlation


class ConcordanceFamily(BaseModel):
    """Pairwise concordances with a family-wise multiplicity adjustment."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    results: tuple[InterventionConcordance, ...] = Field(min_length=1)
    holm_adjusted_p_values: tuple[float, ...]
    multiplicity_method: str = "holm"

    @model_validator(mode="after")
    def validate_adjustment(self) -> Self:
        """Bind the adjusted p-values to the raw permutation p-values."""
        expected = holm_adjust(tuple(item.permutation_p_value for item in self.results))
        if len(self.holm_adjusted_p_values) != len(self.results) or not np.allclose(
            self.holm_adjusted_p_values, expected, rtol=1e-12, atol=0.0
        ):
            raise ValueError("Holm-adjusted p-values do not match the raw p-values")
        return self

    def to_frame(self) -> pd.DataFrame:
        """Return one tidy row per intervention pair."""
        return pd.DataFrame(
            {
                "first": item.first.label,
                "second": item.second.label,
                "shared_control_strata": len(item.shared_control_strata),
                "naive_correlation": item.naive_correlation,
                "corrected_correlation": item.corrected_correlation,
                "disjoint_control_correlation": item.disjoint_control_correlation,
                "first_signal_fraction": item.first_signal_fraction,
                "second_signal_fraction": item.second_signal_fraction,
                "permutation_p_value": item.permutation_p_value,
                "holm_adjusted_p_value": adjusted,
            }
            for item, adjusted in zip(self.results, self.holm_adjusted_p_values, strict=True)
        )


def holm_adjust(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Return Holm step-down adjusted p-values in input order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return tuple(adjusted)


class _Design:
    """Integer sample indices for one contrast pair, grouped by stratum."""

    def __init__(
        self,
        samples: pd.DataFrame,
        first: InterventionContrast,
        second: InterventionContrast,
        policy: StrataPolicy,
    ) -> None:
        missing = {"group", "stratum"}.difference(samples.columns)
        if missing:
            raise ValueError(f"sample table is missing columns: {sorted(missing)}")
        self.positions: dict[tuple[str, str], list[int]] = {}
        for position, (group, stratum) in enumerate(
            zip(samples["group"].astype(str), samples["stratum"].astype(str), strict=True)
        ):
            self.positions.setdefault((group, stratum), []).append(position)
        first_strata = self._available(first)
        second_strata = self._available(second)
        if policy is StrataPolicy.COMMON:
            common = tuple(item for item in first_strata if item in second_strata)
            if not common:
                raise ValueError(
                    f"{first.label} and {second.label} share no stratum; "
                    "use StrataPolicy.OWN to compare them across strata"
                )
            first_strata = second_strata = common
        self.first_strata = first_strata
        self.second_strata = second_strata
        self.shared = tuple(
            stratum
            for stratum in first_strata
            if stratum in second_strata and first.control_group == second.control_group
        )
        self.first = first
        self.second = second
        groups = (
            first.treated_group,
            first.control_group,
            second.treated_group,
            second.control_group,
        )
        self.pools = {
            stratum: [
                position
                for group in dict.fromkeys(groups)
                for position in self.positions.get((group, stratum), [])
            ]
            for stratum in sorted(set(first_strata).union(second_strata))
        }

    def _available(self, contrast: InterventionContrast) -> tuple[str, ...]:
        strata = sorted(
            {stratum for group, stratum in self.positions if group == contrast.treated_group}
        )
        usable = tuple(
            stratum
            for stratum in strata
            if len(self.positions.get((contrast.treated_group, stratum), [])) >= 2
            and len(self.positions.get((contrast.control_group, stratum), [])) >= 2
        )
        if not usable:
            raise ValueError(
                f"{contrast.label} needs a stratum with at least two treated and two "
                "control samples"
            )
        return usable

    def labels(self) -> dict[tuple[str, str], list[int]]:
        return self.positions

    def permuted(self, random: np.random.Generator) -> dict[tuple[str, str], list[int]]:
        """Reassign group labels within each stratum, keeping group sizes."""
        permuted = dict(self.positions)
        for stratum, pool in self.pools.items():
            shuffled = list(random.permutation(pool))
            start = 0
            for group in dict.fromkeys(
                (
                    self.first.treated_group,
                    self.first.control_group,
                    self.second.treated_group,
                    self.second.control_group,
                )
            ):
                size = len(self.positions.get((group, stratum), []))
                if size:
                    permuted[(group, stratum)] = [
                        int(item) for item in shuffled[start : start + size]
                    ]
                    start += size
        return permuted


class _Moments:
    """Effect vector and per-feature noise variance of one contrast."""

    def __init__(
        self,
        values: npt.NDArray[np.float64],
        positions: dict[tuple[str, str], list[int]],
        contrast: InterventionContrast,
        strata: tuple[str, ...],
    ) -> None:
        effects = []
        noise = []
        for stratum in strata:
            treated = values[:, positions[(contrast.treated_group, stratum)]]
            control = values[:, positions[(contrast.control_group, stratum)]]
            effects.append(treated.mean(axis=1) - control.mean(axis=1))
            noise.append(
                treated.var(axis=1, ddof=1) / treated.shape[1]
                + control.var(axis=1, ddof=1) / control.shape[1]
            )
        count = len(strata)
        self.effect = np.asarray(np.mean(effects, axis=0), dtype=np.float64)
        self.noise = np.asarray(np.sum(noise, axis=0) / count**2, dtype=np.float64)
        self.signal_variance = float(self.effect.var() - self.noise.mean())
        self.total_variance = float(self.effect.var())


def _shared_noise(
    values: npt.NDArray[np.float64],
    positions: dict[tuple[str, str], list[int]],
    design: _Design,
) -> float:
    """Return the mean covariance that the shared control adds to the two effects."""
    if not design.shared:
        return 0.0
    total = np.zeros(values.shape[0], dtype=np.float64)
    for stratum in design.shared:
        control = values[:, positions[(design.first.control_group, stratum)]]
        total += control.var(axis=1, ddof=1) / control.shape[1]
    return float(total.mean() / (len(design.first_strata) * len(design.second_strata)))


def _covariance(first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]) -> float:
    return float(np.mean((first - first.mean()) * (second - second.mean())))


def _corrected_covariance(
    values: npt.NDArray[np.float64],
    positions: dict[tuple[str, str], list[int]],
    design: _Design,
) -> tuple[float, _Moments, _Moments, float]:
    first = _Moments(values, positions, design.first, design.first_strata)
    second = _Moments(values, positions, design.second, design.second_strata)
    shared = _shared_noise(values, positions, design)
    return _covariance(first.effect, second.effect) - shared, first, second, shared


def _disjoint_covariance(
    values: npt.NDArray[np.float64],
    design: _Design,
) -> float | None:
    """Average covariance when the two contrasts never share a control animal."""
    if not design.shared:
        return None
    controls = {
        stratum: design.positions[(design.first.control_group, stratum)]
        for stratum in design.shared
    }
    splits = max(len(item) for item in controls.values())
    covariances = []
    for split in range(splits):
        positions = dict(design.positions)
        first_positions = dict(positions)
        second_positions = dict(positions)
        for stratum, members in controls.items():
            held = members[split % len(members)]
            first_positions[(design.first.control_group, stratum)] = [held]
            second_positions[(design.second.control_group, stratum)] = [
                item for item in members if item != held
            ]
        first = _effect_only(values, first_positions, design.first, design.first_strata)
        second = _effect_only(values, second_positions, design.second, design.second_strata)
        covariances.append(_covariance(first, second))
    return float(np.mean(covariances))


def _effect_only(
    values: npt.NDArray[np.float64],
    positions: dict[tuple[str, str], list[int]],
    contrast: InterventionContrast,
    strata: tuple[str, ...],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        np.mean(
            [
                values[:, positions[(contrast.treated_group, stratum)]].mean(axis=1)
                - values[:, positions[(contrast.control_group, stratum)]].mean(axis=1)
                for stratum in strata
            ],
            axis=0,
        ),
        dtype=np.float64,
    )


def intervention_concordance(
    values: pd.DataFrame,
    samples: pd.DataFrame,
    first: InterventionContrast,
    second: InterventionContrast,
    *,
    strata_policy: StrataPolicy = StrataPolicy.COMMON,
    permutations: int = 999,
    random_seed: int = 0,
    minimum_signal_fraction: float = 0.1,
) -> InterventionConcordance:
    """Estimate how similar two interventions' true effect vectors are.

    ``values`` is a features-by-samples matrix on an additive scale such as log2
    CPM. ``samples`` is indexed by the same sample identifiers and has ``group``
    and ``stratum`` columns. Each contrast averages treated-minus-control
    differences over its strata. Correlations are Pearson correlations across
    features of those effect vectors.

    The corrected correlation divides by estimated signal variances. When either
    effect vector is mostly noise (signal fraction below
    ``minimum_signal_fraction``) that ratio is unstable, so it is reported as
    ``None``; the permutation test of the corrected covariance remains valid.
    The test permutes group labels within strata, which is exact under the
    complete null of no effects and conservative when real but uncorrelated
    effects are present.
    """
    if not 0 <= minimum_signal_fraction < 1:
        raise ValueError("minimum_signal_fraction must lie in [0, 1)")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    if list(values.columns) != list(samples.index):
        raise ValueError("value columns and sample-table index must list the same samples")
    matrix = np.asarray(values.to_numpy(dtype=float), dtype=np.float64)
    if matrix.shape[0] < 3 or not np.isfinite(matrix).all():
        raise ValueError("values need at least three features and only finite entries")
    design = _Design(samples, first, second, strata_policy)
    covariance, first_moments, second_moments, shared = _corrected_covariance(
        matrix, design.labels(), design
    )
    naive = float(np.corrcoef(first_moments.effect, second_moments.effect)[0, 1])
    signal_product = first_moments.signal_variance * second_moments.signal_variance
    enough_signal = all(
        item.signal_variance > minimum_signal_fraction * item.total_variance
        and item.signal_variance > 0
        for item in (first_moments, second_moments)
    )
    corrected = (
        float(np.clip(covariance / sqrt(signal_product), -1.0, 1.0)) if enough_signal else None
    )
    disjoint_covariance = _disjoint_covariance(matrix, design)
    disjoint = (
        float(np.clip(disjoint_covariance / sqrt(signal_product), -1.0, 1.0))
        if disjoint_covariance is not None and corrected is not None
        else None
    )
    random = np.random.default_rng(random_seed)
    exceedances = 0
    for _ in range(permutations):
        null_covariance, *_ = _corrected_covariance(matrix, design.permuted(random), design)
        exceedances += abs(null_covariance) >= abs(covariance)
    for value in (naive, covariance, shared):
        if not isfinite(value):
            raise ValueError("concordance produced a non-finite statistic")
    return InterventionConcordance(
        first=first,
        second=second,
        strata_policy=strata_policy,
        first_strata=design.first_strata,
        second_strata=design.second_strata,
        shared_control_strata=design.shared,
        feature_count=matrix.shape[0],
        naive_correlation=float(np.clip(naive, -1.0, 1.0)),
        corrected_correlation=corrected,
        disjoint_control_correlation=disjoint,
        corrected_covariance=covariance,
        shared_control_noise_covariance=shared,
        first_signal_fraction=first_moments.signal_variance / first_moments.total_variance,
        second_signal_fraction=second_moments.signal_variance / second_moments.total_variance,
        permutation_p_value=(exceedances + 1) / (permutations + 1),
        permutations=permutations,
        random_seed=random_seed,
    )


def concordance_family(
    values: pd.DataFrame,
    samples: pd.DataFrame,
    contrasts: Sequence[InterventionContrast],
    *,
    strata_policy: StrataPolicy = StrataPolicy.COMMON,
    permutations: int = 999,
    random_seed: int = 0,
    minimum_signal_fraction: float = 0.1,
) -> ConcordanceFamily:
    """Estimate every pairwise concordance and Holm-adjust across the family."""
    if len(contrasts) < 2:
        raise ValueError("at least two contrasts are required")
    labels = [item.label for item in contrasts]
    if len(set(labels)) != len(labels):
        raise ValueError("contrasts must be unique")
    results = tuple(
        intervention_concordance(
            values,
            samples,
            first,
            second,
            strata_policy=strata_policy,
            permutations=permutations,
            random_seed=random_seed + index,
            minimum_signal_fraction=minimum_signal_fraction,
        )
        for index, (first, second) in enumerate(combinations(contrasts, 2))
    )
    return ConcordanceFamily(
        results=results,
        holm_adjusted_p_values=holm_adjust(tuple(item.permutation_p_value for item in results)),
    )
