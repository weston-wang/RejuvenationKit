"""Bring aging-clock predictions from biolearn or pyaging into RejuvenationKit.

Clock computation is a solved packaging problem: `biolearn`
(Biomarkers of Aging Consortium) and `pyaging` implement dozens of published
clocks. RejuvenationKit does not reimplement them. This module takes their
outputs and adds what they do not provide: typed longitudinal observations for
QC, treatment-effect, state-estimation, and surrogate analyses, and a
leakage-safe definition of age acceleration.

Neither library is a dependency. Adapters duck-type their public output
formats:

* ``biolearn``: ``model.predict(geo_data)`` returns a DataFrame indexed by sample
  with a ``"Predicted"`` column, and ``biolearn.mortality.run_predictions``
  returns samples by model names.
* ``pyaging``: ``pyaging.pred.predict_age(adata, clock_names)`` writes one column
  per clock into ``adata.obs``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from math import isfinite
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from rejuvenationkit.schemas import Modality, Observation


class ClockFit(BaseModel):
    """Linear clock-on-age reference line fitted only on reference samples."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    clock: str = Field(min_length=1)
    intercept: float
    slope: float
    reference_samples: int = Field(ge=3)
    residual_standard_deviation: float = Field(ge=0)


class AgeAcceleration(BaseModel):
    """Out-of-sample age acceleration for every sample and clock."""

    model_config = ConfigDict(frozen=True)

    fits: tuple[ClockFit, ...] = Field(min_length=1)
    reference_sample_ids: tuple[str, ...] = Field(min_length=3)
    method: str = (
        "clock regressed on chronological age using reference samples only; "
        "reference samples scored leave-one-out"
    )
    values: dict[str, dict[str, float]]

    def to_frame(self) -> pd.DataFrame:
        """Return samples by ``<clock>_acceleration`` columns."""
        return pd.DataFrame(self.values).rename(columns=lambda name: f"{name}_acceleration")


def _validated_table(table: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if table.empty:
        raise ValueError(f"{source} predictions are empty")
    if not table.index.is_unique:
        raise ValueError(f"{source} sample identifiers must be unique")
    if not table.columns.is_unique:
        raise ValueError(f"{source} clock names must be unique")
    numeric = cast(pd.DataFrame, table.apply(pd.to_numeric, errors="coerce"))
    introduced = numeric.isna() & table.notna()
    if introduced.any().any():
        raise ValueError(f"{source} predictions contain non-numeric values")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(f"{source} predictions contain infinite values")
    numeric.index = numeric.index.map(str)
    numeric.columns = numeric.columns.map(str)
    return numeric


def clock_table_from_biolearn(
    predictions: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    column: str = "Predicted",
) -> pd.DataFrame:
    """Return a samples-by-clocks table from biolearn output.

    Accepts either the samples-by-models frame from ``run_predictions`` or a
    mapping of clock name to a single model's ``predict`` output, from which
    ``column`` is taken.
    """
    if isinstance(predictions, pd.DataFrame):
        return _validated_table(predictions, source="biolearn")
    columns = {}
    for name, frame in predictions.items():
        if column not in frame.columns:
            raise ValueError(f"biolearn output for {name!r} has no {column!r} column")
        columns[name] = frame[column]
    return _validated_table(pd.DataFrame(columns), source="biolearn")


def clock_table_from_pyaging(
    adata_or_obs: Any,
    clock_names: Sequence[str],
) -> pd.DataFrame:
    """Return a samples-by-clocks table from pyaging output.

    Pass the AnnData object returned by ``pyaging.pred.predict_age`` or its
    ``obs`` DataFrame. Clock columns are matched case-insensitively because
    pyaging stores lowercase clock names.
    """
    obs = getattr(adata_or_obs, "obs", adata_or_obs)
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("expected an AnnData object or its obs DataFrame")
    if not clock_names:
        raise ValueError("at least one clock name is required")
    lookup = {str(name).lower(): name for name in obs.columns}
    selected = {}
    for name in clock_names:
        key = lookup.get(name.lower())
        if key is None:
            raise ValueError(f"pyaging output has no column for clock {name!r}")
        selected[name] = obs[key]
    return _validated_table(pd.DataFrame(selected), source="pyaging")


def clock_observations(
    table: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    modality: Modality = Modality.METHYLATION,
    units: str | Mapping[str, str] = "years",
    source: str,
) -> tuple[tuple[Observation, ...], tuple[tuple[str, str], ...]]:
    """Convert clock predictions into typed observations.

    ``samples`` is indexed by sample identifier with ``subject_id`` and
    timezone-aware ``timestamp`` columns, and optionally ``batch_id``. Returns the
    observations and the ``(sample_id, clock)`` pairs that had no prediction.
    Every observation records ``source`` (for example ``"biolearn 0.9.1"``) and
    the sample identifier, so a result can be traced to the clock run.
    """
    if not source.strip():
        raise ValueError("source must name the clock software and version")
    missing_columns = {"subject_id", "timestamp"}.difference(samples.columns)
    if missing_columns:
        raise ValueError(f"sample table is missing columns: {sorted(missing_columns)}")
    table = _validated_table(table, source="clock")
    sample_index = samples.copy()
    sample_index.index = sample_index.index.map(str)
    unknown = set(table.index).difference(sample_index.index)
    if unknown:
        raise ValueError(f"clock predictions reference unknown samples: {sorted(unknown)[:5]}")
    observations: list[Observation] = []
    missing: list[tuple[str, str]] = []
    for sample_id, row in table.iterrows():
        metadata = sample_index.loc[str(sample_id)]
        timestamp = metadata["timestamp"]
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
        if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
            raise ValueError(f"sample {sample_id!r} needs a timezone-aware timestamp")
        batch = metadata.get("batch_id")
        for clock, value in row.items():
            if pd.isna(value):
                missing.append((str(sample_id), str(clock)))
                continue
            unit = units if isinstance(units, str) else units.get(str(clock))
            if unit is None:
                raise ValueError(f"no unit declared for clock {clock!r}")
            observations.append(
                Observation(
                    subject_id=str(metadata["subject_id"]),
                    timestamp=timestamp,
                    modality=modality,
                    feature=str(clock),
                    value=float(value),
                    unit=unit,
                    batch_id=None if batch is None or pd.isna(batch) else str(batch),
                    replicate_id=str(sample_id),
                    source_uri=source,
                    attributes={"clock": str(clock), "sample_id": str(sample_id)},
                )
            )
    return tuple(observations), tuple(missing)


def age_acceleration(
    table: pd.DataFrame,
    chronological_age: pd.Series,
    *,
    reference_samples: Sequence[str],
    clocks: Sequence[str] | None = None,
) -> AgeAcceleration:
    """Compute age acceleration without letting treated samples define "normal".

    Age acceleration is conventionally the residual from regressing clock age on
    chronological age. If that regression includes treated samples, especially
    later follow-up samples in a longitudinal study, the treatment effect
    partly moves into the fitted slope and the residuals understate it. Here the
    line is fitted on ``reference_samples`` only (for example, controls, or
    baseline samples before any intervention). Non-reference samples are scored
    against that line; each reference sample is scored against a line fitted
    without it, so its residual is out-of-sample too.
    """
    table = _validated_table(table, source="clock")
    ages = pd.to_numeric(chronological_age, errors="coerce")
    ages.index = ages.index.map(str)
    reference = tuple(dict.fromkeys(str(item) for item in reference_samples))
    if len(reference) < 3:
        raise ValueError("at least three reference samples are required")
    missing = set(reference).difference(table.index)
    if missing:
        raise ValueError(f"reference samples lack clock predictions: {sorted(missing)[:5]}")
    selected = list(clocks) if clocks is not None else list(table.columns)
    fits: list[ClockFit] = []
    values: dict[str, dict[str, float]] = {}
    for clock in selected:
        if clock not in table.columns:
            raise ValueError(f"unknown clock {clock!r}")
        frame = pd.DataFrame({"clock": table[clock], "age": ages.reindex(table.index)}).dropna()
        reference_frame = frame.loc[[item for item in reference if item in frame.index]]
        if len(reference_frame) < 3:
            raise ValueError(f"clock {clock!r} has fewer than three complete reference samples")
        if float(reference_frame["age"].var(ddof=1)) <= 0:
            raise ValueError("reference samples must span more than one chronological age")
        slope, intercept = np.polyfit(reference_frame["age"], reference_frame["clock"], 1)
        residuals = reference_frame["clock"] - (intercept + slope * reference_frame["age"])
        dof = max(len(reference_frame) - 2, 1)
        fits.append(
            ClockFit(
                clock=clock,
                intercept=float(intercept),
                slope=float(slope),
                reference_samples=len(reference_frame),
                residual_standard_deviation=float(np.sqrt((residuals**2).sum() / dof)),
            )
        )
        column: dict[str, float] = {}
        for sample_id, row in frame.iterrows():
            if sample_id in reference_frame.index:
                others = reference_frame.drop(index=sample_id)
                if others["age"].var(ddof=1) <= 0 or len(others) < 2:
                    raise ValueError("leave-one-out reference fit needs varied ages")
                loo_slope, loo_intercept = np.polyfit(others["age"], others["clock"], 1)
                predicted = loo_intercept + loo_slope * row["age"]
            else:
                predicted = intercept + slope * row["age"]
            value = float(row["clock"] - predicted)
            if not isfinite(value):
                raise ValueError("age acceleration produced a non-finite value")
            column[str(sample_id)] = value
        values[clock] = column
    return AgeAcceleration(fits=tuple(fits), reference_sample_ids=reference, values=values)
