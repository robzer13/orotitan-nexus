"""Aggregation layer for the Nexus Core (v6.8-lite) score engine."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import NexusCoreSettings, ProfileSettingsV2
from .nexus_core_pillars import (
    compute_behavior_pillar,
    compute_fit_pillar,
    compute_momentum_pillar,
    compute_quality_pillar,
    compute_risk_pillar,
    compute_value_pillar,
)
from .nexus_core_factors import clamp_series

NEXUS_CORE_PREFIX = "nexus_core_"


def _blend_pillars(df: pd.DataFrame, settings: NexusCoreSettings) -> pd.Series:
    """Blend pillar scores into the 0-100 Nexus Core score."""

    weights = np.array(
        [
            settings.weight_q,
            settings.weight_v,
            settings.weight_m,
            settings.weight_r,
            settings.weight_b,
            settings.weight_f,
        ],
        dtype=float,
    )
    pillars = df[[
        f"{NEXUS_CORE_PREFIX}q",
        f"{NEXUS_CORE_PREFIX}v",
        f"{NEXUS_CORE_PREFIX}m",
        f"{NEXUS_CORE_PREFIX}r",
        f"{NEXUS_CORE_PREFIX}b",
        f"{NEXUS_CORE_PREFIX}f",
    ]]
    valid = ~pillars.isna()
    weighted = (pillars * weights).sum(axis=1, skipna=True)
    denom = (valid * weights).sum(axis=1, skipna=True)
    score = weighted / denom
    score[denom <= 0] = np.nan
    return clamp_series(score, lower=0.0, upper=100.0)


def _exceptionality_score(df: pd.DataFrame, settings: NexusCoreSettings) -> pd.Series:
    """Compute the 0–10 exceptionality score using Q/F/global Nexus plus flags."""

    q = df.get(f"{NEXUS_CORE_PREFIX}q", pd.Series(np.nan, index=df.index)).fillna(0.0)
    fit = df.get(f"{NEXUS_CORE_PREFIX}f", pd.Series(np.nan, index=df.index)).fillna(0.0)
    nexus = df.get("nexus_core_score", pd.Series(np.nan, index=df.index)).fillna(0.0)

    base = 0.5 * (q / 10.0) + 0.3 * (fit / 10.0) + 0.2 * (nexus / 10.0)

    diamond = df.get("diamond_lt", df.get("diamond_flag", pd.Series(False, index=df.index))).fillna(False)
    gold = df.get("gold_lt", df.get("gold_flag", pd.Series(False, index=df.index))).fillna(False)
    red = df.get("red_flag", pd.Series(False, index=df.index)).fillna(False)

    base = base + diamond.astype(float) * settings.diamond_bonus
    base = base + gold.astype(float) * settings.gold_bonus
    base = base - red.astype(float) * settings.red_flag_penalty

    return clamp_series(base, lower=0.0, upper=10.0)


def apply_nexus_core_scores(
    df: pd.DataFrame,
    profile: ProfileSettingsV2,
    *,
    price_df: Optional[pd.DataFrame] = None,
    portfolio_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Append Nexus Core pillars, global score, and exceptionality.

    No existing column is removed or renamed; if ``profile.nexus_core.enabled`` is
    False the input dataframe is returned unchanged.
    """

    if df.empty or not getattr(profile.nexus_core, "enabled", False):
        return df

    df = df.copy()
    settings = profile.nexus_core

    df[f"{NEXUS_CORE_PREFIX}q"] = compute_quality_pillar(df, settings)
    df[f"{NEXUS_CORE_PREFIX}v"] = compute_value_pillar(df, settings)
    df[f"{NEXUS_CORE_PREFIX}m"] = compute_momentum_pillar(df, settings)
    df[f"{NEXUS_CORE_PREFIX}r"] = compute_risk_pillar(df, settings)
    df[f"{NEXUS_CORE_PREFIX}b"] = compute_behavior_pillar(df, settings)
    df[f"{NEXUS_CORE_PREFIX}f"] = compute_fit_pillar(df, settings)

    df["nexus_core_score"] = _blend_pillars(df, settings)
    df["nexus_exceptionality"] = _exceptionality_score(df, settings)

    return df

