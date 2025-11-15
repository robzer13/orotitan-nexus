"""Public Python API for OroTitan Nexus."""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .config import load_settings
from .garp_rules import GARP_FLAG_COLUMN
from .orchestrator import run_universe
from .reporting import summarize_garp, summarize_v1


def run_screen(
    *,
    config_path: Optional[str] = None,
    profile_name: Optional[str] = None,
    apply_garp: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Execute the generic screener and return the dataframe plus summary stats."""

    filters, weights, universe, profile = load_settings(config_path, profile_name)
    df = run_universe(
        filters,
        weights,
        universe,
        apply_garp=apply_garp,
        garp_thresholds=profile.garp if apply_garp else None,
    )
    summary = summarize_v1(df, universe, profile.name)
    return df, summary


def run_cac40_garp(
    *,
    config_path: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the CAC40 GARP radar workflow and return full/radar dataframes."""

    filters, weights, universe, profile = load_settings(config_path, profile_name)
    df = run_universe(
        filters,
        weights,
        universe,
        apply_garp=True,
        garp_thresholds=profile.garp,
    )
    mask = df[GARP_FLAG_COLUMN] if GARP_FLAG_COLUMN in df else pd.Series(False, index=df.index)
    radar_df = df[mask].copy()
    if not radar_df.empty:
        radar_df.sort_values(by=["market_cap"], ascending=False, inplace=True)
    summary = summarize_garp(df, universe.name, profile.name)
    return df, radar_df, summary
