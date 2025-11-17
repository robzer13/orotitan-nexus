"""Regime classification helpers for Nexus v2."""
from __future__ import annotations

import pandas as pd
from typing import Dict, Any

from .config import ProfileSettingsV2
from .backtest import compute_horizon_returns


def label_regimes(price_df: pd.DataFrame, profile: ProfileSettingsV2) -> pd.Series:
    if not (profile and profile.v2.enabled and profile.regime.enabled):
        return pd.Series(dtype=object)
    bench = price_df[price_df["ticker"] == profile.regime.benchmark_ticker].copy()
    if bench.empty:
        return pd.Series(dtype=object)
    bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
    bench.sort_values("date", inplace=True)
    bench["ret"] = bench["adj_close"].pct_change(profile.regime.lookback_days)
    labels = []
    for _, row in bench.iterrows():
        r = row["ret"]
        if pd.isna(r):
            labels.append(profile.regime.neutral_label)
        elif r >= profile.regime.bull_threshold:
            labels.append(profile.regime.bull_label)
        elif r <= profile.regime.bear_threshold:
            labels.append(profile.regime.bear_label)
        else:
            labels.append(profile.regime.neutral_label)
    bench["regime"] = labels
    return bench.set_index("date")["regime"]


def regime_performance(
    df: pd.DataFrame, price_df: pd.DataFrame, profile: ProfileSettingsV2
) -> Dict[str, Any]:
    if not (profile and profile.v2.enabled and profile.regime.enabled):
        return {}
    regimes = label_regimes(price_df, profile)
    if regimes.empty:
        return {}
    results: Dict[str, Any] = {}
    if "nexus_v2_bucket" not in df.columns or "ticker" not in df.columns:
        return {}
    top_mask = df.get("nexus_v2_bucket") == "V2_ELITE"
    top_tickers = df[top_mask]["ticker"].tolist()
    if not top_tickers:
        return {}
    for regime_label in regimes.unique():
        dates = regimes[regime_label == regimes].index
        if len(dates) < 2:
            continue
        start_date = pd.to_datetime(dates.min())
        end_date = pd.to_datetime(dates.max())
        horizon = max((end_date - start_date).days, 1)
        top_returns = compute_horizon_returns(price_df, top_tickers, start_date, horizon)
        bench_returns = compute_horizon_returns(
            price_df,
            [profile.regime.benchmark_ticker],
            start_date,
            horizon,
        )
        results[regime_label] = {
            "top_return": float(top_returns.mean(skipna=True)) if not top_returns.empty else float("nan"),
            "benchmark_return": float(bench_returns.mean(skipna=True)) if not bench_returns.empty else float("nan"),
            "n_names": int(top_returns.notna().sum()) if not top_returns.empty else 0,
        }
    return results
