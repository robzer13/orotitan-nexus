"""Console reporting helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FilterSettings, UniverseSettings, WeightSettings


def _format_float(value: float, precision: int = 2) -> str:
    if value is None or np.isnan(value):
        return "NaN"
    return f"{value:.{precision}f}"


def print_strict_preview(df: pd.DataFrame) -> None:
    strict_df = df[df["strict_pass"]]
    if strict_df.empty:
        print("Aucune valeur ne passe le filtre strict GARP.")
    else:
        print("Valeurs qui passent le filtre strict GARP:")
        print(strict_df.to_string(index=False))


def print_global_preview(df: pd.DataFrame, max_rows: int) -> None:
    print(f"\nAperçu global trié par score (top {max_rows}):")
    print(df.head(max_rows).to_string(index=False))


def print_v1_overlay(df: pd.DataFrame) -> None:
    print("\n=== OroTitan V1.1 – SBF120 GARP filter (0–5 points) ===")
    eligible = df[(df["universe_ok"]) & (df["data_complete_v1_1"])]
    if eligible.empty:
        print("Aucune valeur éligible (universe_ok=False ou données manquantes).")
        return
    ordered = eligible.sort_values(
        by=["score_v1_1", "peg", "debt_to_equity", "vol_1y"],
        ascending=[False, True, True, True],
    )
    v1_top = ordered[ordered["score_v1_1"] >= 3]
    if v1_top.empty:
        print("Aucune valeur n'atteint la Watchlist (score >= 3).")
        return
    cols = [
        "ticker",
        "score_v1_1",
        "category_v1_1",
        "pe_ttm",
        "pe_fwd",
        "eps_cagr",
        "peg",
        "debt_to_equity",
        "market_cap",
        "adv_3m",
    ]
    cols = [column for column in cols if column in v1_top.columns]
    print(v1_top[cols].head(40).to_string(index=False))


def print_ticker_diagnostics(
    df: pd.DataFrame,
    tickers,
    filters: FilterSettings,
    weights: WeightSettings,
    universe: UniverseSettings,
    profile: str | None,
) -> None:
    for ticker in tickers:
        row = df[df["ticker"] == ticker]
        if row.empty:
            print(f"Ticker {ticker} not found in universe; skipping.")
            continue
        row = row.iloc[0]
        print("\n==============================")
        print(f"Diagnostics for {ticker}")
        print("==============================")
        print(f"strict_pass: {row.get('strict_pass', False)}")
        print(f"nexus_score: {_format_float(row.get('nexus_score', np.nan))}")
        print(f"garp_score:  {_format_float(row.get('garp_score', np.nan))}")
        print(f"risk_score:  {_format_float(row.get('risk_score', np.nan))}")
        print(f"safety_score: {_format_float(row.get('safety_score', np.nan))}")
        print(f"category_v1_1: {row.get('category_v1_1', 'N/A')}")
        print(f"profile: {profile or 'none'}")

        print("\nFundamentals:")
        print(f"  PE (ttm):          {_format_float(row.get('pe_ttm', np.nan))}")
        print(f"  PE (forward):      {_format_float(row.get('pe_fwd', np.nan))}")
        print(f"  Debt/Equity:       {_format_float(row.get('debt_to_equity', np.nan))}")
        print(f"  EPS growth (CAGR): {_format_float(row.get('eps_cagr', np.nan))}")
        print(f"  PEG ratio:         {_format_float(row.get('peg', np.nan))}")
        print(f"  Market cap:        {_format_float(row.get('market_cap', np.nan), precision=1)}")

        print("\nHard filter flags:")
        print(
            f"  per_ok:         {row.get('per_ok', False)} (max_pe_ttm = {_format_float(filters.max_pe_ttm)})"
        )
        print(
            f"  per_forward_ok: {row.get('per_forward_ok', False)} (max_forward_pe = {_format_float(filters.max_forward_pe)})"
        )
        print(
            f"  de_ok:          {row.get('de_ok', False)} (max_debt_to_equity = {_format_float(filters.max_debt_to_equity)})"
        )
        print(
            f"  eps_growth_ok:  {row.get('eps_growth_ok', False)} (min_eps_cagr = {_format_float(filters.min_eps_cagr)})"
        )
        print(f"  peg_ok:         {row.get('peg_ok', False)} (max_peg = {_format_float(filters.max_peg)})")
        print(
            f"  mktcap_ok:      {row.get('mktcap_ok', False)} (min_market_cap = {_format_float(filters.min_market_cap)})"
        )

        print("\nGARP sub-scores:")
        print(f"  Valuation score:     {_format_float(row.get('valuation_score', np.nan))}")
        print(f"  Growth score:        {_format_float(row.get('growth_score', np.nan))}")
        print(f"  Balance sheet score: {_format_float(row.get('balance_sheet_score', np.nan))}")
        print(f"  Size score:          {_format_float(row.get('size_score', np.nan))}")
        print(f"  => garp_score:       {_format_float(row.get('garp_score', np.nan))}")

        print("\nRisk metrics:")
        print(f"  vol_1y (annualized): {_format_float(row.get('vol_1y', np.nan))}")
        print(f"  mdd_1y:              {_format_float(row.get('mdd_1y', np.nan))}")
        print(f"  adv_3m:              {_format_float(row.get('adv_3m', np.nan), precision=1)}")
        print(f"  risk_score:          {_format_float(row.get('risk_score', np.nan))}")
        print(f"  safety_score:        {_format_float(row.get('safety_score', np.nan))}")
        print(f"  nexus_score:         {_format_float(row.get('nexus_score', np.nan))}")
