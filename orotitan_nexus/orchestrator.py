"""Universe orchestration helpers for OroTitan Nexus."""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from . import universe as universe_mod
from .config import FilterSettings, WeightSettings, UniverseSettings, GarpThresholds
from .filters import (
    categorize_v1_1,
    compute_data_complete_v1_1,
    compute_universe_exclusion_reason,
    compute_universe_ok,
    passes_strict_filters,
)
from .garp_rules import (
    GARP_BUCKET_COLUMN,
    GARP_FLAG_COLUMN,
    GARP_SCORE_COLUMN,
    apply_garp_rules,
    assign_garp_buckets,
    compute_garp_score as compute_strict_garp_score,
)
from .normalization import build_record_for_ticker
from .scoring import (
    compute_balance_sheet_score,
    compute_garp_score,
    compute_growth_score,
    compute_nexus_score,
    compute_risk_score,
    compute_safety_score,
    compute_size_score,
    compute_valuation_score,
    compute_v1_1_score,
)

LOGGER = logging.getLogger(__name__)

STRICT_SORT_COLUMNS = ["strict_pass", "score_v1_1", "nexus_score", "garp_score", "risk_score"]
STRICT_SORT_ASCENDING = [False, False, False, False, True]


def _ensure_dataframe_schema(df: pd.DataFrame) -> None:
    """Guarantee all downstream columns exist with sane defaults."""

    numeric_columns = [
        "pe_ttm",
        "pe_fwd",
        "debt_to_equity",
        "eps_cagr",
        "peg",
        "market_cap",
        "vol_1y",
        "mdd_1y",
        "adv_3m",
    ]
    for column in numeric_columns:
        if column not in df:
            df[column] = np.nan
    if "eps_cagr_source" not in df:
        df["eps_cagr_source"] = "none"
    else:
        df["eps_cagr_source"] = df["eps_cagr_source"].fillna("none")
    if "universe_exclusion_reason" not in df:
        df["universe_exclusion_reason"] = ""
    else:
        df["universe_exclusion_reason"] = df["universe_exclusion_reason"].fillna("")

    boolean_columns = [
        "has_pe_ttm",
        "has_pe_fwd",
        "has_debt_to_equity",
        "has_eps_cagr",
        "has_peg",
        "has_market_cap",
        "has_risk_data",
        "data_ready_nexus",
        "data_ready_v1_1",
        "data_complete_v1_1",
    ]
    for column in boolean_columns:
        if column not in df:
            df[column] = False


def _populate_boolean_flags(df: pd.DataFrame, filters: FilterSettings) -> None:
    """Populate per-rule boolean flags used by diagnostics and CLI outputs."""

    df["has_pe_ttm"] = df["has_pe_ttm"].fillna(False)
    df["has_pe_fwd"] = df["has_pe_fwd"].fillna(False)
    df["has_debt_to_equity"] = df["has_debt_to_equity"].fillna(False)
    df["has_eps_cagr"] = df["has_eps_cagr"].fillna(False)
    df["has_peg"] = df["has_peg"].fillna(False)
    df["has_market_cap"] = df["has_market_cap"].fillna(False)
    df["has_risk_data"] = df["has_risk_data"].fillna(False)

    df["data_ready_nexus"] = df["data_ready_nexus"].fillna(False)
    df["data_ready_v1_1"] = df["data_ready_v1_1"].fillna(False)
    df["data_complete_v1_1"] = df["data_complete_v1_1"].fillna(False)

    df["per_ok"] = df["pe_ttm"].apply(lambda v: bool(not np.isnan(v) and v <= filters.max_pe_ttm))
    df["per_forward_ok"] = df["pe_fwd"].apply(lambda v: bool(not np.isnan(v) and v <= filters.max_forward_pe))
    df["de_ok"] = df["debt_to_equity"].apply(lambda v: bool(not np.isnan(v) and v <= filters.max_debt_to_equity))
    df["eps_growth_ok"] = df["eps_cagr"].apply(lambda v: bool(not np.isnan(v) and v >= filters.min_eps_cagr))
    df["peg_ok"] = df["peg"].apply(lambda v: bool(not np.isnan(v) and v <= filters.max_peg))
    df["mktcap_ok"] = df["market_cap"].apply(lambda v: bool(not np.isnan(v) and v >= filters.min_market_cap))


def _apply_garp_columns(
    df: pd.DataFrame,
    *,
    apply_garp: bool,
    thresholds: Optional[GarpThresholds],
) -> None:
    """Populate the strict GARP column if requested."""

    if GARP_FLAG_COLUMN not in df:
        df[GARP_FLAG_COLUMN] = False
    else:
        df[GARP_FLAG_COLUMN] = df[GARP_FLAG_COLUMN].fillna(False)
    if GARP_SCORE_COLUMN not in df:
        df[GARP_SCORE_COLUMN] = 0.0
    else:
        df[GARP_SCORE_COLUMN] = pd.to_numeric(df[GARP_SCORE_COLUMN], errors="coerce").fillna(0.0)
    if GARP_BUCKET_COLUMN not in df:
        df[GARP_BUCKET_COLUMN] = "REJECT_GARP"
    else:
        df[GARP_BUCKET_COLUMN] = df[GARP_BUCKET_COLUMN].fillna("REJECT_GARP")

    if not apply_garp or thresholds is None or df.empty:
        return

    mask = df["data_complete_v1_1"].fillna(False)
    if not mask.any():
        return

    evaluated = apply_garp_rules(df.loc[mask].copy(), thresholds)
    evaluated = compute_strict_garp_score(evaluated, thresholds)
    evaluated = assign_garp_buckets(evaluated)

    df.loc[mask, GARP_FLAG_COLUMN] = evaluated[GARP_FLAG_COLUMN]
    df.loc[mask, GARP_SCORE_COLUMN] = evaluated[GARP_SCORE_COLUMN]
    df.loc[mask, GARP_BUCKET_COLUMN] = evaluated[GARP_BUCKET_COLUMN]


def run_universe(
    filters: FilterSettings,
    weights: WeightSettings,
    universe: UniverseSettings,
    *,
    apply_garp: bool = False,
    garp_thresholds: Optional[GarpThresholds] = None,
) -> pd.DataFrame:
    """Run the end-to-end screener for ``universe`` and return the enriched DataFrame."""

    tickers = universe_mod.load_universe(universe)
    LOGGER.info("Téléchargement des fondamentaux pour %d tickers", len(tickers))
    records: List[dict] = []
    for ticker in tickers:
        try:
            record = build_record_for_ticker(ticker, filters, universe)
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to fetch data for %s", ticker)
            record = {
                "ticker": ticker,
                "pe_ttm": np.nan,
                "pe_fwd": np.nan,
                "debt_to_equity": np.nan,
                "eps_cagr": np.nan,
                "peg": np.nan,
                "market_cap": np.nan,
                "vol_1y": np.nan,
                "mdd_1y": np.nan,
                "adv_3m": np.nan,
                "eps_cagr_source": "none",
                "has_pe_ttm": False,
                "has_pe_fwd": False,
                "has_debt_to_equity": False,
                "has_eps_cagr": False,
                "has_peg": False,
                "has_market_cap": False,
                "has_risk_data": False,
                "data_ready_nexus": False,
                "data_ready_v1_1": False,
                "data_complete_v1_1": False,
                "universe_exclusion_reason": "missing_data",
            }
        records.append(record)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    _ensure_dataframe_schema(df)
    _populate_boolean_flags(df, filters)

    df["universe_ok"] = df.apply(lambda row: compute_universe_ok(row, universe), axis=1)
    df["universe_exclusion_reason"] = df.apply(
        lambda row: compute_universe_exclusion_reason(row, universe), axis=1
    )
    df["strict_pass"] = df.apply(lambda row: passes_strict_filters(row, filters), axis=1)

    df["valuation_score"] = df.apply(lambda row: compute_valuation_score(row, filters), axis=1)
    df["growth_score"] = df.apply(lambda row: compute_growth_score(row, filters), axis=1)
    df["balance_sheet_score"] = df.apply(lambda row: compute_balance_sheet_score(row, filters), axis=1)
    df["size_score"] = df.apply(lambda row: compute_size_score(row, filters), axis=1)
    df["garp_score"] = df.apply(lambda row: compute_garp_score(row, filters, weights), axis=1)

    df["risk_score"] = df.apply(lambda row: compute_risk_score(row, filters, weights), axis=1)
    df["safety_score"] = df.apply(lambda row: compute_safety_score(row, filters, weights), axis=1)
    df["nexus_score"] = df.apply(lambda row: compute_nexus_score(row, filters, weights), axis=1)

    df["data_complete_v1_1"] = df.apply(compute_data_complete_v1_1, axis=1)
    df["score_v1_1"] = df.apply(lambda row: compute_v1_1_score(row, filters), axis=1)
    df["category_v1_1"] = df.apply(categorize_v1_1, axis=1)

    _apply_garp_columns(df, apply_garp=apply_garp, thresholds=garp_thresholds)

    df.sort_values(by=STRICT_SORT_COLUMNS, ascending=STRICT_SORT_ASCENDING, inplace=True)
    return df
