"""Small-sample corrections shared by bootstrap-based group contrasts."""

from __future__ import annotations

from math import sqrt

import numpy as np
import numpy.typing as npt
from scipy.stats import t as student_t


def variance_corrected(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Inflate within-group deviations by sqrt(n / (n - 1)) along the last axis.

    A nonparametric bootstrap of a mean has variance ``s^2 (n - 1) / n^2``, which
    understates the unbiased ``s^2 / n`` by 18% in standard error at n = 3.
    Resampling the inflated values makes the bootstrap variance match ``s^2 / n``
    without changing the group mean.
    """
    size = values.shape[-1]
    mean = values.mean(axis=-1, keepdims=True)
    return np.asarray(mean + sqrt(size / (size - 1)) * (values - mean), dtype=float)


def welch_critical_values(
    treated: npt.NDArray[np.float64],
    control: npt.NDArray[np.float64],
    confidence_level: float,
) -> npt.NDArray[np.float64]:
    """Return per-row Student t critical values with Welch-Satterthwaite df.

    Inputs are variance-corrected values along the last axis, so their population
    variance equals the original unbiased sample variance.
    """
    treated_term = treated.var(axis=-1) / treated.shape[-1]
    control_term = control.var(axis=-1) / control.shape[-1]
    numerator = (treated_term + control_term) ** 2
    denominator = treated_term**2 / (treated.shape[-1] - 1) + control_term**2 / (
        control.shape[-1] - 1
    )
    minimum_df = min(treated.shape[-1], control.shape[-1]) - 1
    degrees_of_freedom = np.where(
        denominator > 0,
        numerator / np.where(denominator > 0, denominator, 1.0),
        minimum_df,
    )
    degrees_of_freedom = np.maximum(degrees_of_freedom, minimum_df)
    return np.asarray(
        student_t.ppf(0.5 + confidence_level / 2, df=degrees_of_freedom),
        dtype=float,
    )
