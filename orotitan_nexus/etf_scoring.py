"""ETF Nexus score (cost/tracking/liquidity/diversification/fit)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EtfScoringSettings, ProfileSettingsV2


def _scale_inverse(value):
    ser = pd.to_numeric(value, errors="coerce") if hasattr(value, "__len__") else value
    if isinstance(ser, pd.Series):
        return (100.0 - ser).clip(lower=0.0, upper=100.0)
    if np.isnan(ser):
        return np.nan
    return float(np.clip(100.0 - ser, 0.0, 100.0))


def compute_etf_nexus_score(
    df: pd.DataFrame, settings: EtfScoringSettings, profile: ProfileSettingsV2
) -> pd.Series:
    """Return a 0–100 ETF score for ETF rows; NaN/0 elsewhere."""

    if df.empty:
        return pd.Series(dtype=float)
    etf_mask = df.get("ETF_bool", False)
    scores = pd.Series(np.nan, index=df.index)
    if not settings.enabled:
        return scores

    ter = pd.to_numeric(df.get("TER_pct", pd.Series(np.nan, index=df.index)), errors="coerce")
    tracking = pd.to_numeric(df.get("Tracking_diff_3Y_pct", pd.Series(np.nan, index=df.index)), errors="coerce")
    adv = pd.to_numeric(df.get("ADV20_EUR", pd.Series(np.nan, index=df.index)), errors="coerce")
    holdings = pd.to_numeric(
        df.get("holdings_count", df.get("underlying_holdings_count", pd.Series(np.nan, index=df.index))), errors="coerce"
    )

    cost_score = _scale_inverse(ter)
    track_score = _scale_inverse(tracking)

    liq_score = pd.Series(
        np.where(adv >= settings.min_liquidity_eur, 100.0, np.clip(adv / settings.min_liquidity_eur * 100.0, 0.0, 100.0)),
        index=df.index,
    )
    divers_score = pd.Series(
        np.where(holdings > 0, np.clip(holdings / 100.0 * 100.0, 0.0, 100.0), 50.0), index=df.index
    )

    fit_score = pd.Series(50.0, index=df.index)
    priority = getattr(profile.nexus_core, "fit_priority_sectors", []) if getattr(profile, "nexus_core", None) else []
    if "sector" in df.columns:
        fit_score = pd.Series(np.where(df["sector"].isin(priority), 80.0, 50.0), index=df.index)

    def _weighted_row(i: int) -> float:
        components = []
        weights = []
        for val, w in (
            (cost_score.iat[i], settings.w_cost),
            (track_score.iat[i], settings.w_track),
            (liq_score.iat[i], settings.w_liq),
            (divers_score.iat[i], settings.w_divers),
            (fit_score.iat[i], settings.w_fit),
        ):
            if not np.isnan(val):
                components.append(val)
                weights.append(w)
        if not components or sum(weights) == 0:
            return np.nan
        return float(np.average(components, weights=weights))

    for idx in df.index:
        if bool(etf_mask.iloc[idx]):
            scores.iloc[idx] = _weighted_row(df.index.get_loc(idx))
        else:
            scores.iloc[idx] = np.nan
    return scores

