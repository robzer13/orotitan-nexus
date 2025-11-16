"""Offline backtesting helpers for strict GARP snapshots."""
from __future__ import annotations

from typing import Sequence

import numpy as np

import pandas as pd

from .garp_rules import GARP_FLAG_COLUMN


def compute_horizon_returns(
    price_df: pd.DataFrame,
    tickers: Sequence[str],
    start_date: pd.Timestamp,
    horizon_days: int,
) -> pd.Series:
    """Compute simple cumulative returns over ``horizon_days`` trading days.

    The calculation uses the first trading day on or after ``start_date`` as the
    entry point for each ticker. If there are fewer than ``horizon_days`` prices
    available after that start row, the ticker's return is ``NaN``.
    """

    if not tickers:
        return pd.Series(dtype=float)

    filtered = price_df[price_df["ticker"].isin(tickers)].copy()
    if filtered.empty:
        return pd.Series(dtype=float)

    results = {}
    for ticker, group in filtered.groupby("ticker"):
        series = group.sort_values("date")
        start_mask = series["date"] >= start_date
        if not start_mask.any():
            results[ticker] = np.nan
            continue
        start_idx = start_mask.idxmax()
        start_pos = series.index.get_loc(start_idx)
        end_pos = start_pos + horizon_days
        if end_pos >= len(series):
            results[ticker] = np.nan
            continue
        start_price = series.iloc[start_pos]["adj_close"]
        end_price = series.iloc[end_pos]["adj_close"]
        if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
            results[ticker] = np.nan
        else:
            results[ticker] = float(end_price / start_price - 1.0)

    return pd.Series(results, dtype=float)


def backtest_garp_vs_benchmark(
    snapshot_df: pd.DataFrame,
    price_df: pd.DataFrame,
    start_date: pd.Timestamp,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Compare strict GARP vs an equal-weight benchmark over multiple horizons.

    ``snapshot_df`` must contain ``ticker``, ``strict_pass_garp``, and
    ``data_complete_v1_1``. The benchmark uses all data-complete names, while the
    GARP portfolio uses the strict 5/5 passers within that set. Returns are
    equal-weighted averages of ticker-level horizon returns.
    """

    garp_mask = snapshot_df.get(GARP_FLAG_COLUMN, pd.Series(False, index=snapshot_df.index)).fillna(False)
    complete_mask = snapshot_df.get("data_complete_v1_1", pd.Series(False, index=snapshot_df.index)).fillna(False)

    garp_tickers = snapshot_df.loc[garp_mask & complete_mask, "ticker"].tolist()
    benchmark_tickers = snapshot_df.loc[complete_mask, "ticker"].tolist()

    rows = []
    for horizon in horizons:
        garp_returns = compute_horizon_returns(price_df, garp_tickers, start_date, horizon)
        bench_returns = compute_horizon_returns(price_df, benchmark_tickers, start_date, horizon)

        garp_mean = garp_returns.mean() if not garp_returns.empty else pd.NA
        bench_mean = bench_returns.mean() if not bench_returns.empty else pd.NA

        row = {
            "horizon_days": int(horizon),
            "garp_return": float(garp_mean) if pd.notna(garp_mean) else pd.NA,
            "benchmark_return": float(bench_mean) if pd.notna(bench_mean) else pd.NA,
            "excess_return": float(garp_mean - bench_mean) if pd.notna(garp_mean) and pd.notna(bench_mean) else pd.NA,
            "n_garp_tickers": int(garp_returns.notna().sum()) if not garp_returns.empty else 0,
            "n_benchmark_tickers": int(bench_returns.notna().sum()) if not bench_returns.empty else 0,
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("horizon_days")


def backtest_quintiles_by_score(
    df: pd.DataFrame,
    price_df: pd.DataFrame,
    start_date: pd.Timestamp,
    horizons: Sequence[int],
    score_column: str = "nexus_v2_score",
) -> pd.DataFrame:
    """Compute quintile performance by ``score_column`` over multiple horizons.

    The function filters to ``data_complete_v1_1`` rows, bins into five buckets
    using ``pd.qcut`` (falling back to fewer buckets if not enough distinct
    scores), and returns a MultiIndex DataFrame with per-horizon returns.
    """

    base = df[df.get("data_complete_v1_1", False) == True].copy()
    if base.empty or score_column not in base:
        return pd.DataFrame()

    scores = pd.to_numeric(base[score_column], errors="coerce")
    try:
        buckets = pd.qcut(scores.rank(method="first"), 5, labels=False, duplicates="drop")
    except Exception:
        buckets = pd.Series(np.nan, index=base.index)
    base["_bucket"] = buckets

    records = []
    for bucket_id, bucket_group in base.groupby("_bucket"):
        if pd.isna(bucket_id):
            continue
        tickers = bucket_group["ticker"].tolist()
        for horizon in horizons:
            returns = compute_horizon_returns(price_df, tickers, start_date, horizon)
            mean_ret = returns.mean() if not returns.empty else pd.NA
            records.append(
                {
                    "bucket": int(bucket_id),
                    "horizon_days": int(horizon),
                    "return": float(mean_ret) if pd.notna(mean_ret) else pd.NA,
                    "n_names": int(returns.notna().sum()) if not returns.empty else 0,
                }
            )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    return result.set_index(["bucket", "horizon_days"])
