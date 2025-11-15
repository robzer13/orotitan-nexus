"""Command-line interface for the OroTitan Nexus screener."""
from __future__ import annotations

import argparse
import logging
from typing import List

import numpy as np
import pandas as pd

from .config import (
    FilterSettings,
    WeightSettings,
    UniverseSettings,
    ProfileSettings,
    build_settings_from_config,
    load_yaml_config,
)
from . import universe as universe_mod
from .normalization import build_record_for_ticker
from .filters import (
    categorize_v1_1,
    compute_data_complete_v1_1,
    compute_universe_ok,
    compute_universe_exclusion_reason,
    passes_strict_filters,
)
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
from .reporting import (
    print_global_preview,
    print_strict_preview,
    print_summary,
    print_ticker_diagnostics,
    print_v1_overlay,
    write_csv,
)

LOGGER = logging.getLogger(__name__)
STRICT_SORT_COLUMNS = ["strict_pass", "score_v1_1", "nexus_score", "garp_score", "risk_score"]
STRICT_SORT_ASCENDING = [False, False, False, False, True]
DEFAULT_OUTPUT = "cac40_screen_results.csv"
DEFAULT_MAX_ROWS = 40
PROFILE_CHOICES = ("defensive", "balanced", "offensive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAC 40 / SBF 120 GARP screener")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="CSV output path")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Number of rows to print in the global preview",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML configuration file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=PROFILE_CHOICES,
        default=None,
        help="Preset Nexus profile to tweak filters/weights",
    )
    parser.add_argument(
        "--detail",
        nargs="+",
        help="One or more tickers for which to display detailed diagnostics",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact OroTitan V1.1 summary in the console",
    )
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def _ensure_columns(df: pd.DataFrame) -> None:
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


def _compute_boolean_flags(df: pd.DataFrame, filters: FilterSettings) -> None:
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
    df["per_forward_ok"] = df["pe_fwd"].apply(
        lambda v: bool(not np.isnan(v) and v <= filters.max_forward_pe)
    )
    df["de_ok"] = df["debt_to_equity"].apply(
        lambda v: bool(not np.isnan(v) and v <= filters.max_debt_to_equity)
    )
    df["eps_growth_ok"] = df["eps_cagr"].apply(
        lambda v: bool(not np.isnan(v) and v >= filters.min_eps_cagr)
    )
    df["peg_ok"] = df["peg"].apply(lambda v: bool(not np.isnan(v) and v <= filters.max_peg))
    df["mktcap_ok"] = df["market_cap"].apply(
        lambda v: bool(not np.isnan(v) and v >= filters.min_market_cap)
    )


def run_screener(
    filters: FilterSettings,
    weights: WeightSettings,
    universe: UniverseSettings,
) -> pd.DataFrame:
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

    _ensure_columns(df)
    _compute_boolean_flags(df, filters)

    df["universe_ok"] = df.apply(lambda row: compute_universe_ok(row, universe), axis=1)
    df["universe_exclusion_reason"] = df.apply(
        lambda row: compute_universe_exclusion_reason(row, universe), axis=1
    )
    df["strict_pass"] = df.apply(lambda row: passes_strict_filters(row, filters), axis=1)

    df["valuation_score"] = df.apply(lambda row: compute_valuation_score(row, filters), axis=1)
    df["growth_score"] = df.apply(lambda row: compute_growth_score(row, filters), axis=1)
    df["balance_sheet_score"] = df.apply(
        lambda row: compute_balance_sheet_score(row, filters), axis=1
    )
    df["size_score"] = df.apply(lambda row: compute_size_score(row, filters), axis=1)
    df["garp_score"] = df.apply(lambda row: compute_garp_score(row, filters, weights), axis=1)

    df["risk_score"] = df.apply(lambda row: compute_risk_score(row, filters, weights), axis=1)
    df["safety_score"] = df.apply(
        lambda row: compute_safety_score(row, filters, weights), axis=1
    )
    df["nexus_score"] = df.apply(lambda row: compute_nexus_score(row, filters, weights), axis=1)

    df["data_complete_v1_1"] = df.apply(compute_data_complete_v1_1, axis=1)
    df["score_v1_1"] = df.apply(lambda row: compute_v1_1_score(row, filters), axis=1)
    df["category_v1_1"] = df.apply(categorize_v1_1, axis=1)

    df.sort_values(by=STRICT_SORT_COLUMNS, ascending=STRICT_SORT_ASCENDING, inplace=True)
    return df


def print_and_export(
    df: pd.DataFrame,
    filters: FilterSettings,
    weights: WeightSettings,
    universe: UniverseSettings,
    profile: ProfileSettings,
    args: argparse.Namespace,
) -> None:
    if df.empty:
        LOGGER.warning(
            "Aucune donnée récupérée : vérifier la connectivité réseau ou la liste de tickers"
        )
        return

    print_strict_preview(df)
    print_global_preview(df, args.max_rows)
    print_v1_overlay(df)

    write_csv(df, args.output)
    LOGGER.info("Résultats complets sauvegardés dans %s", args.output)

    if args.summary:
        print_summary(df, universe, profile.name)

    if args.detail:
        print_ticker_diagnostics(
            df,
            args.detail,
            filters,
            weights,
            universe,
            profile.name,
        )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    config_data = load_yaml_config(args.config)
    filters, weights, universe, profile = build_settings_from_config(config_data, args.profile)
    if profile.name:
        LOGGER.info("Applying Nexus profile: %s", profile.name)

    df = run_screener(filters, weights, universe)
    print_and_export(df, filters, weights, universe, profile, args)
