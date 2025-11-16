"""Lightweight factor computations for Nexus v2 (additive to v1)."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import BehavioralSettings, MacroSettings, MomentumSettings, ProfileSettingsV2, QualitySettings, RiskSettings
from .backtest import compute_horizon_returns


def _safe_series(values: Iterable[float], default: float = 50.0) -> pd.Series:
    ser = pd.to_numeric(pd.Series(values), errors="coerce")
    return ser.fillna(default)


def _linear_scale(value: float, low: float, high: float) -> float:
    if np.isnan(value):
        return np.nan
    if high == low:
        return float(value >= high) * 100.0
    clipped = min(max(value, low), high)
    return 100.0 * (clipped - low) / (high - low)


def compute_garp_score_v2(df: pd.DataFrame) -> pd.Series:
    """Map existing strict GARP score (0–100) to a v2 pillar score.

    If ``strict_garp_score`` exists, reuse it; otherwise fall back to the
    legacy ``garp_score`` column. NaNs are penalized conservatively.
    """

    base = None
    if "strict_garp_score" in df:
        base = pd.to_numeric(df["strict_garp_score"], errors="coerce")
    elif "garp_score" in df:
        base = pd.to_numeric(df["garp_score"], errors="coerce")
    else:
        base = pd.Series(np.nan, index=df.index)
    return base.fillna(0.0).clip(lower=0.0, upper=100.0)


def compute_quality_score(df: pd.DataFrame, settings: QualitySettings) -> pd.Series:
    """Very light quality proxy built from ROE/ROA/margin if present."""

    roe = pd.to_numeric(df.get("roe", np.nan), errors="coerce")
    roa = pd.to_numeric(df.get("roa", np.nan), errors="coerce")
    margin = pd.to_numeric(df.get("profit_margin", np.nan), errors="coerce")

    roe_score = _safe_series(roe.apply(lambda v: _linear_scale(v, settings.min_roe, settings.min_roe * 2)))
    roa_score = _safe_series(roa.apply(lambda v: _linear_scale(v, 0.0, max(settings.min_roe, 0.1))))
    margin_score = _safe_series(margin.apply(lambda v: _linear_scale(v, settings.min_margin, max(settings.min_margin * 2, 0.1))))

    weights = np.array([
        max(settings.roe_weight, 0.0),
        max(settings.roa_weight, 0.0),
        max(settings.margin_weight, 0.0),
    ])
    if weights.sum() <= 0:
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
    else:
        weights = weights / weights.sum()

    stacked = np.vstack([roe_score, roa_score, margin_score])
    combined = np.nansum(weights.reshape(-1, 1) * stacked, axis=0)
    # Penalize fully missing rows
    mask_all_nan = np.isnan(stacked).all(axis=0)
    combined = np.where(mask_all_nan, 0.0, combined)
    return pd.Series(combined, index=df.index)


def _momentum_component(price_df: pd.DataFrame, tickers: pd.Index, lookback: int, start_date: pd.Timestamp) -> pd.Series:
    if price_df is None or price_df.empty:
        return pd.Series(np.nan, index=tickers)
    ticker_list = tickers.tolist() if hasattr(tickers, "tolist") else list(tickers)
    return compute_horizon_returns(price_df, ticker_list, start_date, lookback)


def compute_momentum_score(
    df: pd.DataFrame,
    price_df: Optional[pd.DataFrame],
    settings: MomentumSettings,
    start_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Price-based momentum using short/medium/long horizons."""

    if price_df is None or start_date is None:
        return pd.Series(50.0, index=df.index)

    tickers = df["ticker"] if "ticker" in df else pd.Index([])
    short_ret = _momentum_component(price_df, tickers, settings.lookback_short_days, start_date)
    med_ret = _momentum_component(price_df, tickers, settings.lookback_medium_days, start_date)
    long_ret = _momentum_component(price_df, tickers, settings.lookback_long_days, start_date)

    weights = np.array([
        max(settings.short_weight, 0.0),
        max(settings.medium_weight, 0.0),
        max(settings.long_weight, 0.0),
    ])
    if weights.sum() <= 0:
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
    else:
        weights = weights / weights.sum()

    stacked = np.vstack([
        short_ret.fillna(np.nan),
        med_ret.fillna(np.nan),
        long_ret.fillna(np.nan),
    ])
    momentum = np.nanmean(weights.reshape(-1, 1) * stacked, axis=0)
    # map returns to score: -50% -> 0, +50% -> 100 linear
    scores = (momentum + 0.5) * 100.0
    scores = np.clip(scores, 0.0, 100.0)
    scores = np.where(np.isnan(momentum), 50.0, scores)
    return pd.Series(scores, index=df.index)


def compute_risk_score(
    df: pd.DataFrame,
    price_df: Optional[pd.DataFrame],
    settings: RiskSettings,
    start_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Risk score where lower volatility/drawdown -> higher score."""

    if price_df is None or start_date is None:
        return pd.Series(50.0, index=df.index)
    tickers = df["ticker"] if "ticker" in df else pd.Index([])
    ticker_list = tickers.tolist() if hasattr(tickers, "tolist") else list(tickers)
    vol_series = compute_horizon_returns(price_df, ticker_list, start_date, settings.lookback_days)
    # reuse returns as proxy for price path; compute drawdown crudely using std dev if history absent
    vol = price_df[price_df["ticker"].isin(ticker_list)].groupby("ticker")["adj_close"].apply(lambda s: s.pct_change().std())
    vol = vol.reindex(tickers)
    vol_score = 100.0 * (1 - vol.clip(upper=settings.max_vol) / settings.max_vol)

    dd = price_df[price_df["ticker"].isin(ticker_list)].groupby("ticker")["adj_close"].apply(
        lambda s: (s / s.cummax() - 1).min()
    )
    dd = dd.reindex(tickers)
    dd_score = 100.0 * (1 - dd.abs().clip(upper=settings.max_drawdown) / settings.max_drawdown)

    weights = np.array([
        max(settings.vol_weight, 0.0),
        max(settings.dd_weight, 0.0),
    ])
    if weights.sum() <= 0:
        weights = np.array([0.5, 0.5])
    else:
        weights = weights / weights.sum()

    stacked = np.vstack([
        vol_score.fillna(np.nan),
        dd_score.fillna(np.nan),
    ])
    combined = np.nanmean(weights.reshape(-1, 1) * stacked, axis=0)
    combined = np.where(np.isnan(combined), 50.0, combined)
    return pd.Series(np.clip(combined, 0.0, 100.0), index=df.index)


def compute_macro_score(df: pd.DataFrame, settings: MacroSettings) -> pd.Series:
    """Return a constant macro stub score (override via YAML later)."""

    return pd.Series(float(settings.default_score), index=df.index)


def compute_behavioral_score(
    df: pd.DataFrame,
    settings: BehavioralSettings,
    portfolio_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Apply light penalties for concentration/knife-catching when possible."""

    base = pd.Series(50.0, index=df.index)
    if portfolio_df is None or portfolio_df.empty:
        return base
    # Simple concentration penalty: if a sector weight exceeds threshold, subtract penalty
    if "sector" in df.columns and "owned" in df.columns:
        owned_mask = df["owned"] == True
        sector_weights = df.loc[owned_mask].groupby("sector")["owned"].count()
        total_owned = max(int(owned_mask.sum()), 1)
        for sector, count in sector_weights.items():
            weight = count / total_owned
            if weight > settings.max_sector_weight:
                base.loc[df["sector"] == sector] = (base.loc[df["sector"] == sector] - settings.concentration_penalty)
    # Knife-catching: strong negative momentum but high GARP score
    if "momentum_score" in df.columns and "strict_garp_score" in df.columns:
        mask = (df["momentum_score"] < 30) & (df["strict_garp_score"] > 70)
        base.loc[mask] = base.loc[mask] - settings.knife_catch_penalty
    return base.clip(lower=0.0, upper=100.0)

