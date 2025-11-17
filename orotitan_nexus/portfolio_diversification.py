"""Portfolio-level diversification helpers (opt-in)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GeoSettings, ProfileSettings
from .geo import attach_region_exposures
from .portfolio_metrics import compute_hhi, compute_top5_weight


def compute_weights(portfolio_df: pd.DataFrame) -> pd.Series:
    if portfolio_df is None or portfolio_df.empty:
        return pd.Series(dtype=float)
    if "position_value" in portfolio_df.columns:
        values = pd.to_numeric(portfolio_df["position_value"], errors="coerce").fillna(0.0)
    else:
        return pd.Series(dtype=float)
    total = values.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return values / total


def compute_sector_weights(portfolio_df: pd.DataFrame) -> pd.Series:
    weights = compute_weights(portfolio_df)
    if weights.empty or "primary_sector" not in portfolio_df.columns:
        return pd.Series(dtype=float)
    grouped = weights.groupby(portfolio_df["primary_sector"].fillna("Unknown")).sum()
    return grouped.sort_values(ascending=False)


def compute_region_weights(portfolio_df: pd.DataFrame, geo_settings: GeoSettings) -> pd.Series:
    if portfolio_df is None or portfolio_df.empty:
        return pd.Series(dtype=float)
    enriched = attach_region_exposures(portfolio_df, geo_settings)
    weight = compute_weights(enriched)
    if weight.empty:
        return pd.Series(dtype=float)
    region_cols = [c for c in enriched.columns if c.startswith("region_") and c.endswith("_weight")]
    if not region_cols:
        return pd.Series(dtype=float)
    region_weights = {}
    for col in region_cols:
        region_name = col.replace("region_", "").replace("_weight", "")
        region_weights[region_name] = float((pd.to_numeric(enriched[col], errors="coerce") * weight).sum())
    series = pd.Series(region_weights).sort_values(ascending=False)
    total = series.sum()
    if total > 0:
        series = series / total
    return series


def compute_geo_distance(region_weights: pd.Series, geo_settings: GeoSettings) -> dict:
    distances = {}
    targets = geo_settings.target_regions or {}
    if region_weights is None:
        region_weights = pd.Series(dtype=float)
    for region, target in targets.items():
        obs = float(region_weights.get(region, 0.0))
        distances[region] = obs - float(target)
    return distances


def compute_diversification_score(
    weights: pd.Series,
    sector_weights: pd.Series,
    region_weights: pd.Series,
    profile: ProfileSettings,
) -> float:
    """Map concentration metrics to a 0–100 diversification score.

    Heuristic: combine normalized sub-scores for HHI, Top5 and sector caps using
    profile.diversification weights. Missing inputs yield a neutral score of 50.
    """

    settings = profile.diversification
    if weights is None or weights.empty:
        return 50.0
    hhi = compute_hhi(weights)
    top5 = compute_top5_weight(weights)

    def _score_from_bound(value: float, warn: float, limit: float) -> float:
        if limit - warn < 0.1:
            limit = warn + 0.1
        if np.isnan(value):
            return 50.0
        if value <= warn:
            return 100.0
        if value >= limit:
            return 0.0
        # linear decay between warn and limit
        return float(100.0 * (limit - value) / (limit - warn))

    hhi_score = _score_from_bound(hhi, settings.warn_hhi, settings.max_hhi)
    top5_score = _score_from_bound(top5, settings.max_top5_weight, settings.max_top5_weight * 1.3)

    if sector_weights is None or sector_weights.empty:
        sector_score = 100.0
    else:
        over_penalties = []
        for sector, cap in (settings.sector_caps or {}).items():
            actual = float(sector_weights.get(sector, 0.0))
            if actual > cap:
                over_penalties.append(min(100.0, max(0.0, 100.0 * (1 - (actual - cap) / cap))))
        if over_penalties:
            sector_score = float(np.mean(over_penalties))
        else:
            sector_score = 100.0 if sector_weights.max() <= settings.max_sector_weight else 70.0

    total_weight = settings.hhi_weight + settings.top5_weight + settings.sector_weight
    score = (
        settings.hhi_weight * hhi_score
        + settings.top5_weight * top5_score
        + settings.sector_weight * sector_score
    ) / total_weight
    return float(np.clip(score, 0.0, 100.0))
