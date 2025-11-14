"""Filtering utilities for OroTitan Nexus."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FilterSettings, UniverseSettings
from .normalization import core_v1_1_ready

CATEGORY_ELITE = "ELITE_V1_1"
CATEGORY_WATCHLIST = "WATCHLIST_V1_1"
CATEGORY_REJECT = "REJECT_V1_1"
CATEGORY_EXCLUDED = "EXCLUDED_UNIVERSE"
CATEGORY_DATA_MISSING = "DATA_MISSING"


def compute_universe_ok(row: pd.Series, universe: UniverseSettings) -> bool:
    """Check ADV and market-cap guards for SBF120-style universes."""

    mcap = row.get("market_cap", np.nan)
    adv = row.get("adv_3m", np.nan)
    if np.isnan(mcap) or np.isnan(adv):
        return False
    if mcap < universe.min_market_cap:
        return False
    if adv < universe.min_adv_3m:
        return False
    return True


def passes_strict_filters(row: pd.Series, filters: FilterSettings) -> bool:
    """Evaluate the hard GARP filters."""

    pe_ttm = row.get("pe_ttm", np.nan)
    pe_fwd = row.get("pe_fwd", np.nan)
    de_ratio = row.get("debt_to_equity", np.nan)
    eps_cagr = row.get("eps_cagr", np.nan)
    peg = row.get("peg", np.nan)
    mcap = row.get("market_cap", np.nan)

    if np.isnan(pe_ttm) or pe_ttm <= 0 or pe_ttm > filters.max_pe_ttm:
        return False
    if np.isnan(pe_fwd) or pe_fwd <= 0 or pe_fwd > filters.max_forward_pe:
        return False
    if np.isnan(de_ratio) or de_ratio > filters.max_debt_to_equity:
        return False
    if np.isnan(eps_cagr) or eps_cagr < filters.min_eps_cagr:
        return False
    if np.isnan(peg) or peg > filters.max_peg:
        return False
    if np.isnan(mcap) or mcap < filters.min_market_cap:
        return False
    return True


def compute_data_complete_v1_1(row: pd.Series) -> bool:
    """Return True when the core V1.1 metrics exist (PEG optional)."""

    return core_v1_1_ready(
        row.get("pe_ttm", np.nan),
        row.get("pe_fwd", np.nan),
        row.get("debt_to_equity", np.nan),
        row.get("eps_cagr", np.nan),
    )


def categorize_v1_1(row: pd.Series) -> str:
    """Map V1.1 score and readiness into textual categories."""

    if not row.get("universe_ok", False):
        return CATEGORY_EXCLUDED
    if not row.get("data_complete_v1_1", False):
        return CATEGORY_DATA_MISSING

    score = int(row.get("score_v1_1", 0))
    if score == 5:
        return CATEGORY_ELITE
    if score >= 3:
        return CATEGORY_WATCHLIST
    return CATEGORY_REJECT
