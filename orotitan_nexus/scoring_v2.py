"""Nexus v2 scoring aggregation (additive, opt-in)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import ProfileSettingsV2
from .factors_v2 import (
    compute_behavioral_score,
    compute_garp_score_v2,
    compute_macro_score,
    compute_momentum_score,
    compute_quality_score,
    compute_risk_score,
)


NEXUS_V2_SCORE = "nexus_v2_score"
NEXUS_V2_BUCKET = "nexus_v2_bucket"


def _blend_scores(values: list[float], weights: list[float]) -> float:
    arr_vals = np.array(values, dtype=float)
    arr_w = np.array(weights, dtype=float)
    mask = ~np.isnan(arr_vals)
    if not mask.any():
        return 0.0
    arr_vals = arr_vals[mask]
    arr_w = arr_w[mask]
    if arr_w.sum() <= 0:
        arr_w = np.ones_like(arr_vals)
    arr_w = arr_w / arr_w.sum()
    return float(np.dot(arr_vals, arr_w))


def _bucket_from_score(score: float, profile: ProfileSettingsV2) -> str:
    if np.isnan(score):
        return "V2_REJECT"
    if score >= profile.v2.elite_min:
        return "V2_ELITE"
    if score >= profile.v2.strong_min:
        return "V2_STRONG"
    if score >= profile.v2.neutral_min:
        return "V2_NEUTRAL"
    if score >= profile.v2.weak_min:
        return "V2_WEAK"
    return "V2_REJECT"


def apply_v2_scores(
    df: pd.DataFrame,
    profile: ProfileSettingsV2,
    *,
    price_df: Optional[pd.DataFrame] = None,
    portfolio_df: Optional[pd.DataFrame] = None,
    start_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Compute Nexus v2 scores/buckets and append to ``df``.

    The function is no-op unless ``profile.v2.enabled`` is True. It does not
    remove or alter existing v1 columns.
    """

    if df.empty or not getattr(profile.v2, "enabled", False):
        return df

    df = df.copy()

    df["garp_score_v2"] = compute_garp_score_v2(df)
    df["quality_score"] = compute_quality_score(df, profile.quality)
    df["momentum_score"] = compute_momentum_score(df, price_df, profile.momentum, start_date)
    df["risk_score_v2"] = compute_risk_score(df, price_df, profile.risk, start_date)
    df["macro_score"] = compute_macro_score(df, profile.macro)
    df["behavioral_score"] = compute_behavioral_score(df, profile.behavioral, portfolio_df)

    weights = profile.v2
    blended = []
    for _, row in df.iterrows():
        blended.append(
            _blend_scores(
                [
                    row.get("garp_score_v2", np.nan),
                    row.get("quality_score", np.nan),
                    row.get("momentum_score", np.nan),
                    row.get("risk_score_v2", np.nan),
                    row.get("macro_score", np.nan),
                    row.get("behavioral_score", np.nan),
                ],
                [
                    weights.garp_weight,
                    weights.quality_weight,
                    weights.momentum_weight,
                    weights.risk_weight,
                    weights.macro_weight,
                    weights.behavioral_weight,
                ],
            )
        )
    df[NEXUS_V2_SCORE] = pd.Series(blended, index=df.index).clip(lower=0.0, upper=100.0)
    df[NEXUS_V2_BUCKET] = df[NEXUS_V2_SCORE].apply(lambda v: _bucket_from_score(v, profile))
    return df

