"""Portfolio-aware compatibility scoring (opt-in)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CompatibilitySettings, ProfileSettings
from .geo import attach_region_exposures
from .portfolio_diversification import compute_diversification_score, compute_weights, compute_region_weights

COMPATIBILITY_SCORE_COLUMN = "compatibility_score"
GEO_FIT_SCORE_COLUMN = "geo_fit_score"
DIVERSIFICATION_IMPACT_SCORE_COLUMN = "diversification_impact_score"


def compute_geo_fit_score(candidate_row: pd.Series, portfolio_region_weights: pd.Series, geo_settings) -> float:
    exposures_df = attach_region_exposures(pd.DataFrame([candidate_row]), geo_settings)
    region_cols = [c for c in exposures_df.columns if c.startswith("region_") and c.endswith("_weight")]
    if not region_cols:
        return 50.0
    target = geo_settings.target_regions or {}
    baseline = portfolio_region_weights if portfolio_region_weights is not None else pd.Series(dtype=float)
    score_components = []
    for col in region_cols:
        region = col.replace("region_", "").replace("_weight", "")
        cand_weight = float(exposures_df.iloc[0][col])
        target_weight = float(target.get(region, 0.0))
        current_weight = float(baseline.get(region, 0.0))
        # Improvement if candidate moves weight toward target
        current_gap = abs(current_weight - target_weight)
        new_gap = abs(current_weight + cand_weight - target_weight)
        improvement = max(0.0, current_gap - new_gap)
        score_components.append(100.0 * improvement)  # simplistic scaling
    if not score_components:
        return 50.0
    return float(np.clip(np.mean(score_components), 0.0, 100.0))


def compute_diversification_impact_score(
    candidate_row: pd.Series,
    portfolio_df: pd.DataFrame,
    profile: ProfileSettings,
    line_budget_value: float,
) -> float:
    # If no portfolio, neutral
    if portfolio_df is None or portfolio_df.empty:
        return 50.0
    base_weights = compute_weights(portfolio_df)
    if base_weights.empty or "position_value" not in portfolio_df.columns:
        return 50.0
    total_value = portfolio_df["position_value"].sum()
    new_value = total_value + max(line_budget_value, 0.0)
    if new_value <= 0:
        return 50.0

    # simulate adding candidate
    sector = candidate_row.get("primary_sector", "Unknown")
    region_df = attach_region_exposures(pd.DataFrame([candidate_row]), profile.geo)
    candidate_weight = max(line_budget_value, 0.0) / new_value
    augmented = pd.concat(
        [
            portfolio_df.copy(),
            pd.DataFrame(
                [
                    {
                        "ticker": candidate_row.get("ticker", ""),
                        "position_value": line_budget_value,
                        "primary_sector": sector,
                        **{c: region_df.iloc[0][c] for c in region_df.columns if c.startswith("region_")},
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    weights_after = compute_weights(augmented)
    sector_after = augmented.groupby(augmented.get("primary_sector", "primary_sector"))["position_value"].sum()
    region_after = compute_region_weights(augmented, profile.geo)

    before_score = compute_diversification_score(base_weights, pd.Series(dtype=float), pd.Series(dtype=float), profile)
    after_score = compute_diversification_score(weights_after, sector_after / sector_after.sum() if not sector_after.empty else pd.Series(dtype=float), region_after, profile)
    if np.isnan(after_score):
        return 50.0
    delta = after_score - before_score
    return float(np.clip(50.0 + delta, 0.0, 100.0))


def compute_line_compatibility_score(
    candidate_row: pd.Series,
    portfolio_df: pd.DataFrame,
    profile: ProfileSettings,
    line_budget_value: float,
) -> float:
    settings: CompatibilitySettings = profile.compatibility
    region_weights = compute_region_weights(portfolio_df, profile.geo) if portfolio_df is not None else pd.Series(dtype=float)
    geo_score = compute_geo_fit_score(candidate_row, region_weights, profile.geo)
    diversification_score = compute_diversification_impact_score(candidate_row, portfolio_df, profile, line_budget_value)

    sector_score = 50.0
    primary_sector = candidate_row.get("primary_sector")
    if primary_sector:
        cap = (profile.diversification.sector_caps or {}).get(primary_sector)
        if cap is not None:
            sector_score = 100.0 if cap >= 0.0 else 50.0
        elif profile.diversification.max_sector_weight > 0:
            sector_score = 100.0

    corr_score = 50.0  # placeholder: real correlation not available in offline tests
    liquidity_score = 50.0
    if "adv_3m" in candidate_row:
        try:
            liquidity = float(candidate_row.get("adv_3m"))
            if liquidity > 0:
                liquidity_score = float(np.clip(10.0 * np.log10(liquidity + 1), 0.0, 100.0))
        except (TypeError, ValueError):
            pass

    score = (
        settings.geo_weight * geo_score
        + settings.sector_weight * sector_score
        + settings.correlation_weight * corr_score
        + settings.liquidity_weight * liquidity_score
    ) / (
        settings.geo_weight + settings.sector_weight + settings.correlation_weight + settings.liquidity_weight
    )

    bonus = 0.0
    if bool(candidate_row.get("PEA_Eligible_bool")):
        bonus += settings.min_pea_bonus
    priority = candidate_row.get("primary_sector")
    if priority and settings.sector_priority_bonus and priority.lower() in {"health", "infra", "infrastructure"}:
        bonus += settings.sector_priority_bonus

    total = float(np.clip(score + bonus, 0.0, 100.0))
    return total
