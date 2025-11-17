"""Valuation helpers (composite fair-price + scenario engine)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ProfileSettingsV2, ValuationSettings


@dataclass
class Scenario:
    """Simple bear/base/bull scenario definition."""

    name: str
    metric: float
    multiple: float
    weight: float


def _quality_adjustment(q: float, settings: ValuationSettings) -> float:
    if np.isnan(q):
        return 1.0
    span = settings.quality_adjust_high - settings.quality_adjust_low
    mapped = settings.quality_adjust_low + (np.clip(q, 0.0, 100.0) / 100.0) * span
    return float(np.clip(mapped, settings.quality_adjust_low, settings.quality_adjust_high))


def _safe_div(num: float, denom: float) -> float:
    if np.isnan(num) or np.isnan(denom) or denom == 0:
        return np.nan
    return num / denom


def _compute_sector_median(series: pd.Series, default: float = np.nan) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return default
    return float(clean.median())


def apply_valuation(df: pd.DataFrame, profile: ProfileSettingsV2) -> pd.DataFrame:
    """Compute composite fair price / upside and attach columns.

    The computation is opt-in via ``profile.valuation.enabled`` and is conservative
    with missing inputs (falls back to consensus or leaves NaN). Existing columns
    are preserved; new columns are appended:

    - ``fair_price_composite``
    - ``upside_pct``
    - ``fair_price_source``
    - ``fair_price_divergence_flag``
    """

    settings: ValuationSettings = profile.valuation
    if not settings or not settings.enabled or df.empty:
        return df

    out = df.copy()
    if "price" not in out.columns:
        out["price"] = np.nan

    # Compute cross-sectional proxy medians when sector medians are absent.
    fcf_yield = pd.to_numeric(out.get("FCF_YIELD_pct", pd.Series(np.nan, index=out.index)), errors="coerce")
    p_fcf_series = pd.Series(np.where(fcf_yield > 0, 100.0 / fcf_yield, np.nan), index=out.index)
    ev_ebit_series = pd.to_numeric(out.get("EV_EBIT_FWD", pd.Series(np.nan, index=out.index)), errors="coerce")
    pe_series = pd.to_numeric(out.get("PE_FWD", pd.Series(np.nan, index=out.index)), errors="coerce")

    median_p_fcf = _compute_sector_median(p_fcf_series)
    median_ev_ebit = _compute_sector_median(ev_ebit_series)
    median_pe = _compute_sector_median(pe_series)

    fair_prices: List[float] = []
    composite_sources: List[str] = []
    upsides: List[float] = []
    divergence_flags: List[bool] = []

    quality_series = pd.to_numeric(out.get("nexus_core_q", pd.Series(np.nan, index=out.index)), errors="coerce")
    if quality_series.isna().all():
        quality_series = pd.to_numeric(out.get("quality_score", pd.Series(np.nan, index=out.index)), errors="coerce")

    consensus_col = None
    for candidate in ("Consensus_PT_12m", "consensus_price_target"):
        if candidate in out.columns:
            consensus_col = candidate
            break

    for idx, row in out.iterrows():
        price = _safe_div(float(row.get("price", np.nan)), 1.0)
        quality = quality_series.loc[idx] if not quality_series.empty else np.nan
        q_adj = _quality_adjustment(float(quality) if not np.isnan(quality) else np.nan, settings)

        p_fcf = p_fcf_series.loc[idx] if idx in p_fcf_series.index else np.nan
        ev_ebit = ev_ebit_series.loc[idx] if idx in ev_ebit_series.index else np.nan
        pe_val = pe_series.loc[idx] if idx in pe_series.index else np.nan

        p_fcf_target = median_p_fcf * q_adj if not np.isnan(median_p_fcf) else np.nan
        ev_ebit_target = median_ev_ebit * q_adj if not np.isnan(median_ev_ebit) else np.nan
        pe_target = median_pe * q_adj if not np.isnan(median_pe) else np.nan

        fcf_per_share = row.get("FCF_per_share", np.nan)
        ebit_fwd = row.get("EBIT_FWD", np.nan)
        eps_fwd = row.get("EPS_FWD", np.nan)
        shares = row.get("Shares_outstanding", np.nan)
        net_debt = row.get("NetDebt", np.nan)
        consensus_pt = row.get(consensus_col, np.nan) if consensus_col else np.nan

        fair_price_fcf = np.nan
        if not np.isnan(p_fcf_target) and not np.isnan(fcf_per_share):
            fair_price_fcf = p_fcf_target * fcf_per_share

        fair_price_ev_ebit = np.nan
        if not np.isnan(ev_ebit_target) and not np.isnan(ebit_fwd) and not np.isnan(shares) and shares > 0:
            fair_ev = ev_ebit_target * ebit_fwd
            equity = fair_ev - (net_debt if not np.isnan(net_debt) else 0.0)
            fair_price_ev_ebit = equity / shares

        fair_price_pe = np.nan
        if not np.isnan(pe_target) and not np.isnan(eps_fwd):
            fair_price_pe = pe_target * eps_fwd

        components = []
        weights = []
        for val, w in ((fair_price_fcf, settings.w_fcf), (fair_price_ev_ebit, settings.w_ev_ebit), (fair_price_pe, settings.w_pe)):
            if not np.isnan(val):
                components.append(val)
                weights.append(w)

        fair_price_composite = np.nan
        source = "missing"
        divergence_flag = False
        if components and sum(weights) > 0:
            fair_price_composite = float(np.average(components, weights=weights))
            source = "composite"
        elif settings.enable_fallback_consensus and not np.isnan(consensus_pt):
            fair_price_composite = float(consensus_pt)
            source = "consensus"

        upside_pct = np.nan
        if not np.isnan(fair_price_composite) and not np.isnan(price) and price > 0:
            upside_pct = 100.0 * (fair_price_composite - price) / price
            upside_pct = float(np.clip(upside_pct, settings.min_upside_pct, settings.max_upside_pct))

        if source == "composite" and settings.enable_fallback_consensus and not np.isnan(consensus_pt):
            diff = 100.0 * abs(fair_price_composite - consensus_pt) / consensus_pt if consensus_pt else np.nan
            divergence_flag = bool(not np.isnan(diff) and diff > settings.divergence_warn_pct)

        fair_prices.append(fair_price_composite)
        upsides.append(upside_pct)
        composite_sources.append(source)
        divergence_flags.append(divergence_flag)

    out["fair_price_composite"] = fair_prices
    out["upside_pct"] = upsides
    out["fair_price_source"] = composite_sources
    out["fair_price_divergence_flag"] = divergence_flags
    return out


def compute_scenario_fair_value(
    scenarios: List[Scenario],
    net_debt: float,
    shares_outstanding: float,
) -> Dict[str, object]:
    """Compute per-scenario prices and weighted fair value.

    Missing/invalid inputs yield NaN prices; weights are normalized to sum to 1
    when possible. The function is intentionally generic for reuse across tickers.
    """

    if shares_outstanding is None or shares_outstanding == 0:
        return {"prices": {}, "fair_value": np.nan}

    prices: Dict[str, float] = {}
    for scenario in scenarios:
        ev = scenario.multiple * scenario.metric
        equity = ev - (net_debt if not np.isnan(net_debt) else 0.0)
        prices[scenario.name] = _safe_div(equity, shares_outstanding)

    weights = [s.weight for s in scenarios if s.weight is not None]
    total_w = sum(weights)
    fair_value = np.nan
    if total_w > 0:
        fair_value = 0.0
        for scenario in scenarios:
            w = scenario.weight / total_w if total_w else 0.0
            price_s = prices.get(scenario.name, np.nan)
            if np.isnan(price_s):
                continue
            fair_value += w * price_s

    return {"prices": prices, "fair_value": fair_value}

