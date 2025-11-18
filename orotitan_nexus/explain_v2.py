"""Explainability helpers for Nexus v2 scores."""
from __future__ import annotations

import pandas as pd
from typing import Dict, Any

from .config import ProfileSettingsV2

PILLARS = ["garp", "quality", "momentum", "risk", "macro", "behavioral"]


def add_explain_columns(df: pd.DataFrame, profile: ProfileSettingsV2) -> pd.DataFrame:
    if not (profile and profile.v2.enabled):
        return df
    weights = {
        "garp": profile.v2.garp_weight,
        "quality": profile.v2.quality_weight,
        "momentum": profile.v2.momentum_weight,
        "risk": profile.v2.risk_weight,
        "macro": profile.v2.macro_weight,
        "behavioral": profile.v2.behavioral_weight,
    }
    total_w = sum(max(float(w), 0.0) for w in weights.values()) or 1.0
    df = df.copy()
    for pillar in PILLARS:
        col = f"{pillar}_score"
        contrib_col = f"{pillar}_contrib_v2"
        if col in df:
            df[contrib_col] = pd.to_numeric(df[col], errors="coerce") * (weights[pillar] / total_w)
        else:
            df[contrib_col] = pd.NA
    return df


def explain_ticker(df: pd.DataFrame, ticker: str, profile: ProfileSettingsV2) -> Dict[str, Any]:
    if "ticker" not in df.columns or df.empty:
        return {}
    matches = df[df["ticker"] == ticker]
    if matches.empty:
        return {}
    row = matches.iloc[0]
    contribs = {pillar: row.get(f"{pillar}_contrib_v2") for pillar in PILLARS}
    scores = {pillar: row.get(f"{pillar}_score") for pillar in PILLARS}
    return {
        "ticker": ticker,
        "nexus_v2_score": row.get("nexus_v2_score"),
        "nexus_v2_bucket": row.get("nexus_v2_bucket"),
        "scores": scores,
        "contributions": contribs,
    }
