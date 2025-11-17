"""Pillar computations for the Nexus Core (v6.8-lite) score engine."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import NexusCoreSettings
from .nexus_core_factors import score_factor, weighted_average


def _select_first_available(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first existing column among ``candidates`` or an empty Series."""

    for col in candidates:
        if col in df:
            return df[col]
    return pd.Series(dtype=float, index=df.index)


def compute_quality_pillar(df: pd.DataFrame, settings: NexusCoreSettings) -> pd.Series:
    """Compute the Quality pillar (0–100)."""

    scores: Dict[str, pd.Series] = {}
    scores["roe"] = score_factor(df.get("roe_pct", pd.Series(dtype=float, index=df.index)))
    margin_series = _select_first_available(df, ["operating_margin_pct", "net_margin_pct", "gross_margin_pct"])
    scores["margins"] = score_factor(margin_series)
    scores["fcf_quality"] = score_factor(df.get("fcf_yield_pct", pd.Series(dtype=float, index=df.index)))

    weights = pd.Series(
        {
            "roe": settings.q_roe_weight,
            "margins": settings.q_margin_weight,
            "fcf_quality": settings.q_fcf_quality_weight,
        }
    )
    return weighted_average(pd.DataFrame(scores), weights)


def compute_value_pillar(df: pd.DataFrame, settings: NexusCoreSettings) -> pd.Series:
    """Compute the Value pillar (0–100)."""

    scores: Dict[str, pd.Series] = {}
    scores["ev_ebit"] = score_factor(
        df.get("ev_ebit_fwd", pd.Series(dtype=float, index=df.index)), lower_is_better=True
    )
    scores["pe_fwd"] = score_factor(df.get("pe_fwd", pd.Series(dtype=float, index=df.index)), lower_is_better=True)
    scores["fcf_yield"] = score_factor(df.get("fcf_yield_pct", pd.Series(dtype=float, index=df.index)))
    scores["div_yield"] = score_factor(df.get("div_yield_pct", pd.Series(dtype=float, index=df.index)))
    scores["buyback_yield"] = score_factor(df.get("buyback_yield_pct", pd.Series(dtype=float, index=df.index)))

    weights = pd.Series(
        {
            "ev_ebit": settings.v_ev_ebit_weight,
            "pe_fwd": settings.v_pe_weight,
            "fcf_yield": settings.v_fcf_yield_weight,
            "div_yield": settings.v_div_yield_weight,
            "buyback_yield": settings.v_buyback_yield_weight,
        }
    )
    return weighted_average(pd.DataFrame(scores), weights)


def compute_momentum_pillar(df: pd.DataFrame, settings: NexusCoreSettings) -> pd.Series:
    """Compute the Momentum pillar (0–100)."""

    scores: Dict[str, pd.Series] = {}
    scores["perf_1m"] = score_factor(df.get("perf_1m_pct", pd.Series(dtype=float, index=df.index)))
    scores["perf_3m"] = score_factor(df.get("perf_3m_pct", pd.Series(dtype=float, index=df.index)))
    scores["perf_6m"] = score_factor(df.get("perf_6m_pct", pd.Series(dtype=float, index=df.index)))
    scores["mm50_slope"] = score_factor(df.get("mm50_slope_pct", pd.Series(dtype=float, index=df.index)))
    scores["dist_ph63"] = score_factor(
        df.get("dist_to_ph63_pct", pd.Series(dtype=float, index=df.index)), lower_is_better=True
    )

    weights = pd.Series(
        {
            "perf_1m": settings.m_1m_weight,
            "perf_3m": settings.m_3m_weight,
            "perf_6m": settings.m_6m_weight,
            "mm50_slope": settings.m_slope_weight,
            "dist_ph63": settings.m_dist_ph_weight,
        }
    )
    return weighted_average(pd.DataFrame(scores), weights)


def compute_risk_pillar(df: pd.DataFrame, settings: NexusCoreSettings) -> pd.Series:
    """Compute the (inverse) Risk pillar (0–100: higher is safer)."""

    scores: Dict[str, pd.Series] = {}
    scores["vol"] = score_factor(df.get("natr14_pct", pd.Series(dtype=float, index=df.index)), lower_is_better=True)
    scores["drawdown"] = score_factor(
        df.get("max_drawdown_6m_pct", pd.Series(dtype=float, index=df.index)), lower_is_better=True
    )
    beta = df.get("beta_6m_cac", pd.Series(dtype=float, index=df.index))
    scores["beta"] = score_factor(beta.sub(1.0).abs(), lower_is_better=True)
    scores["liquidity"] = score_factor(df.get("avg_dvol_20d_eur", pd.Series(dtype=float, index=df.index)))

    weights = pd.Series(
        {
            "vol": settings.r_vol_weight,
            "drawdown": settings.r_dd_weight,
            "beta": settings.r_beta_weight,
            "liquidity": settings.r_liquidity_weight,
        }
    )
    return weighted_average(pd.DataFrame(scores), weights)


def compute_behavior_pillar(df: pd.DataFrame, settings: NexusCoreSettings) -> pd.Series:
    """Compute the Behavior pillar (exploratory, 0–100)."""

    scores: Dict[str, pd.Series] = {}
    scores["gap"] = score_factor(df.get("gap_open_pct", pd.Series(dtype=float, index=df.index)).abs(), lower_is_better=True)
    scores["avwap"] = score_factor(df.get("avwap_dist_pct", pd.Series(dtype=float, index=df.index)).abs(), lower_is_better=True)

    weights = pd.Series({"gap": settings.b_gap_weight, "avwap": settings.b_avwap_weight})
    return weighted_average(pd.DataFrame(scores), weights)


def compute_fit_pillar(df: pd.DataFrame, settings: NexusCoreSettings) -> pd.Series:
    """Compute the Fit pillar using simple heuristics (0–100)."""

    bonus = pd.Series(0.0, index=df.index)

    pea_flag = df.get("pea_eligible_bool", df.get("pea_eligible", pd.Series(False, index=df.index))).fillna(False)
    bonus = bonus + pea_flag.astype(float) * settings.fit_pea_bonus

    if "sector" in df.columns and settings.fit_priority_sectors:
        bonus = bonus + df["sector"].isin(settings.fit_priority_sectors).fillna(False).astype(float) * settings.fit_priority_sector_bonus

    if "avg_dvol_20d_eur" in df.columns:
        liquid = df["avg_dvol_20d_eur"] >= settings.fit_liquidity_threshold
        bonus = bonus + liquid.fillna(False).astype(float) * 20.0

    return bonus.clip(lower=0.0, upper=100.0)

