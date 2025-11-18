"""Portfolio-level risk metrics (HHI, top5)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_hhi(weights: pd.Series) -> float:
    """Herfindahl–Hirschman Index from decimal weights."""

    if weights is None or weights.empty:
        return np.nan
    clean = pd.to_numeric(weights, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float((clean ** 2).sum())


def compute_top5_weight(weights: pd.Series) -> float:
    """Sum of the top 5 weights (decimal)."""

    if weights is None or weights.empty:
        return np.nan
    clean = pd.to_numeric(weights, errors="coerce").dropna().sort_values(ascending=False)
    return float(clean.head(5).sum())

