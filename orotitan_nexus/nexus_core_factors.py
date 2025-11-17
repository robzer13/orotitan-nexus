"""Factor-level utilities for the Nexus Core (v6.8-lite) engine.

The helpers below intentionally stay light-weight and dependency-free:

* winsorize at the 1st/99th percentiles to reduce outlier impact,
* z-score the clamped series,
* map to a 0–100 score via the standard normal CDF,
* optional inversion when "lower is better" (e.g., volatility, EV/EBIT).

All functions are NaN-safe: missing values remain NaN and are never rewarded.
"""

from __future__ import annotations

from math import erf, sqrt
from typing import Optional

import numpy as np
import pandas as pd


def _normal_cdf(z: pd.Series) -> pd.Series:
    """Return standard normal CDF for a Series, yielding values in [0, 1]."""

    return 0.5 * (1.0 + (z / sqrt(2)).apply(erf))


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clamp a Series to the given quantiles, leaving NaNs untouched."""

    if series.dropna().empty:
        return series
    lower_q, upper_q = series.quantile([lower, upper])
    return series.clip(lower=lower_q, upper=upper_q)


def zscore_series(series: pd.Series) -> pd.Series:
    """Standardize a Series (mean/std) with NaN preservation and std safeguards."""

    if series.dropna().empty:
        return series
    mean = series.mean()
    std = series.std(ddof=0)
    if std <= 0 or np.isnan(std):
        # Avoid division by zero; return zeros where data exists, NaN otherwise.
        return series.apply(lambda v: 0.0 if not np.isnan(v) else np.nan)
    return (series - mean) / std


def score_factor(series: pd.Series, *, lower_is_better: bool = False) -> pd.Series:
    """Convert a raw factor series into a 0–100 score.

    Steps: winsorize -> z-score -> optional inversion -> CDF -> 0–100.
    """

    if series.empty:
        return series

    clamped = winsorize_series(series)
    z = zscore_series(clamped)
    if lower_is_better:
        z = -z
    score = _normal_cdf(z) * 100.0
    return score.clip(lower=0.0, upper=100.0)


def weighted_average(scores: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Compute a row-wise weighted average, ignoring NaNs conservatively."""

    if scores.empty:
        return pd.Series(dtype=float)
    aligned_weights = weights.reindex(scores.columns).fillna(0.0)
    valid_mask = ~scores.isna()
    numerator = (scores * aligned_weights).sum(axis=1, skipna=True)
    denominator = (valid_mask * aligned_weights).sum(axis=1, skipna=True)
    result = numerator / denominator
    result[denominator <= 0] = np.nan
    return result.clip(lower=0.0, upper=100.0)


def clamp_series(series: pd.Series, *, lower: float, upper: float) -> pd.Series:
    """Clamp a numeric Series to ``[lower, upper]`` preserving NaNs."""

    return series.clip(lower=lower, upper=upper)

