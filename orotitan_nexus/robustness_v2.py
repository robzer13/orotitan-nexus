"""Robustness helpers for Nexus v2 (walk-forward, sensitivity)."""
from __future__ import annotations

"""Robustness helpers for Nexus v2 (walk-forward and sensitivity)."""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .backtest import backtest_quintiles_by_score
from .config import ProfileSettingsV2

LOGGER = logging.getLogger(__name__)


def _select_score(df: pd.DataFrame, score_column: str) -> pd.Series:
    return pd.to_numeric(df.get(score_column, pd.Series(index=df.index)), errors="coerce")


def run_walkforward_validation(
    df: pd.DataFrame, price_df: pd.DataFrame, profile: ProfileSettingsV2
) -> Dict[str, Any]:
    """Run a light walk-forward validation on nexus_v2_score.

    This splits the available price history into ``n_splits`` contiguous segments
    and evaluates the top bucket/quintile on the configured horizon. The intent
    is to catch time-instability without changing any v1 behavior. Returns a
    dictionary with per-split results and aggregated stats. If not enabled or
    data is insufficient, returns an empty dict.
    """

    if not (profile and profile.v2.enabled and profile.walkforward.enabled):
        return {}
    try:
        score_column = profile.walkforward.score_column
        horizon = int(profile.walkforward.test_horizon_days)
        n_splits = max(int(profile.walkforward.n_splits), 1)
    except Exception:
        return {}

    # use available dates from price_df to define splits
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
    dates = price_df["date"].dropna().sort_values().unique()
    if len(dates) < n_splits * 2:
        return {}

    chunk_size = len(dates) // n_splits
    records = []
    for idx in range(n_splits):
        start_idx = idx * chunk_size
        if start_idx >= len(dates):
            break
        start_date = pd.to_datetime(dates[start_idx])
        segment_prices = price_df[price_df["date"] >= start_date]
        qt = backtest_quintiles_by_score(df, segment_prices, start_date, [horizon], score_column)
        if qt.empty:
            continue
        try:
            if 0 in qt.index.get_level_values("bucket"):
                top_row = qt.xs(0, level="bucket")
            else:
                top_row = qt.iloc[[0]]
            top_ret = float(top_row.iloc[0]["return"]) if not top_row.empty else np.nan
        except Exception:
            top_ret = np.nan
        records.append({"split": idx, "start_date": start_date, "top_bucket_return": top_ret})

    if not records:
        return {}
    df_rec = pd.DataFrame(records)
    return {
        "splits": records,
        "mean_top_return": float(df_rec["top_bucket_return"].mean(skipna=True)),
        "min_top_return": float(df_rec["top_bucket_return"].min(skipna=True)),
        "max_top_return": float(df_rec["top_bucket_return"].max(skipna=True)),
    }


def run_sensitivity_analysis(
    df: pd.DataFrame, price_df: pd.DataFrame, profile: ProfileSettingsV2
) -> Dict[str, Any]:
    """Perturb v2 weights and thresholds to measure ranking stability.

    The implementation is intentionally lightweight: we perturb the selected
    score column multiplicatively and compute a stability metric against the
    baseline ordering. This keeps the behavior deterministic for tests while
    providing signal on fragility. Returns an empty dict when not enabled.
    """

    if not (profile and profile.v2.enabled and profile.sensitivity.enabled):
        return {}
    rng = np.random.default_rng(0)
    score_column = profile.sensitivity.score_column
    base_scores = _select_score(df, score_column)
    if base_scores.isna().all():
        return {}

    n_draws = max(int(profile.sensitivity.n_random_draws), 1)
    perturb_pct = float(profile.sensitivity.weight_perturbation_pct)
    metric = profile.sensitivity.stability_metric
    top_k = max(int(profile.sensitivity.top_k), 1)

    samples = []
    for _ in range(n_draws):
        noise = rng.normal(loc=0.0, scale=perturb_pct, size=len(base_scores))
        perturbed = base_scores * (1.0 + noise)
        value = np.nan
        if metric == "top_k_overlap":
            base_top = set(base_scores.sort_values(ascending=False).head(top_k).index)
            pert_top = set(perturbed.sort_values(ascending=False).head(top_k).index)
            denom = max(len(base_top | pert_top), 1)
            value = len(base_top & pert_top) / denom
        else:  # default to Spearman correlation
            value = base_scores.corr(perturbed, method="pearson")
        samples.append({"metric_value": float(value) if pd.notna(value) else np.nan})

    metric_vals = pd.Series([s["metric_value"] for s in samples])
    return {
        "stability_metric": metric,
        "samples": samples,
        "summary": {
            "mean": float(metric_vals.mean(skipna=True)),
            "std": float(metric_vals.std(skipna=True)),
            "min": float(metric_vals.min(skipna=True)),
            "max": float(metric_vals.max(skipna=True)),
        },
    }
