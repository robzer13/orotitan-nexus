"""Console and export helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import FilterSettings, UniverseSettings, WeightSettings
from .garp_rules import GARP_BUCKET_COLUMN, GARP_FLAG_COLUMN, GARP_SCORE_COLUMN


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
    tickers: Iterable[str],
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

        print("\nUniverse & data readiness:")
        exclusion = row.get("universe_exclusion_reason", "") or "n/a"
        print(f"  universe_ok: {row.get('universe_ok', False)} (reason: {exclusion})")
        print(f"  data_ready_nexus: {row.get('data_ready_nexus', False)}")
        print(f"  data_ready_v1_1: {row.get('data_ready_v1_1', False)}")
        print(f"  data_complete_v1_1: {row.get('data_complete_v1_1', False)}")
        print(
            "  has fields: PE_ttm={pe} PE_fwd={pf} D/E={de} EPS={eps} PEG={peg} MCAP={mc} Risk={risk}".format(
                pe=row.get("has_pe_ttm", False),
                pf=row.get("has_pe_fwd", False),
                de=row.get("has_debt_to_equity", False),
                eps=row.get("has_eps_cagr", False),
                peg=row.get("has_peg", False),
                mc=row.get("has_market_cap", False),
                risk=row.get("has_risk_data", False),
            )
        )

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


def summarize_v1(df: pd.DataFrame, universe: UniverseSettings, profile_name: str | None) -> dict:
    """Return structured V1.1 summary data for logging or printing."""

    profile_label = profile_name or "balanced"
    summary = {
        "header": f"=== OroTitan V1.1 summary ({universe.name}, profile={profile_label}) ===",
        "total": int(len(df)),
        "data_complete": int(df.get("data_complete_v1_1", pd.Series(dtype=int)).sum()),
        "strict_pass": int(df.get("strict_pass", pd.Series(dtype=int)).sum()),
        "categories": {},
        "top": [],
    }

    categories = [
        "ELITE_V1_1",
        "WATCHLIST_V1_1",
        "REJECT_V1_1",
        "DATA_MISSING",
        "EXCLUDED_UNIVERSE",
    ]
    counts = (
        df.get("category_v1_1", pd.Series(dtype=str)).value_counts().reindex(categories, fill_value=0)
        if not df.empty
        else pd.Series([0] * len(categories), index=categories)
    )
    summary["categories"] = {category: int(counts[category]) for category in categories}

    ordering = ["score_v1_1", "nexus_score", "garp_score"]
    asc = [False, False, False]
    top_rows = df.sort_values(by=ordering, ascending=asc).head(5)
    for idx, row in enumerate(top_rows.itertuples(), start=1):
        summary["top"].append(
            {
                "rank": idx,
                "ticker": row.ticker,
                "score": int(getattr(row, "score_v1_1", 0)),
                "category": getattr(row, "category_v1_1", "n/a"),
                "reason": getattr(row, "universe_exclusion_reason", "") or "OK",
            }
        )

    return summary


def print_summary(
    df: pd.DataFrame,
    universe: UniverseSettings,
    profile_name: str | None,
) -> None:
    """Display a compact V1.1 summary in the console."""

    if df.empty:
        print("Aucune donnée pour générer un résumé.")
        return

    summary = summarize_v1(df, universe, profile_name)
    print(f"\n{summary['header']}")
    print(f"Total universe        : {summary['total']}")
    print(f"Data-complete (V1.1)  : {summary['data_complete']}")
    print(f"  STRICT_PASS         : {summary['strict_pass']}")
    for category, count in summary["categories"].items():
        print(f"  {category:<18}: {count}")

    print("\nTop 5 by score_v1_1:")
    if not summary["top"]:
        print("  (no data)")
    else:
        for entry in summary["top"]:
            print(
                f"  {entry['rank']}) {entry['ticker']:<8} score={entry['score']} "
                f"cat={entry['category']} reason={entry['reason']}"
            )


def summarize_garp(
    df: pd.DataFrame,
    universe_name: str,
    profile_name: str | None,
    *,
    header: str = "=== OroTitan CAC40 GARP Radar v1.7 ===",
) -> dict:
    """Build summary data for the CAC40 GARP radar."""

    profile_label = profile_name or "balanced"
    strict_mask = df[GARP_FLAG_COLUMN].fillna(False) if GARP_FLAG_COLUMN in df else pd.Series(False, index=df.index)
    summary = {
        "header": header,
        "universe": universe_name,
        "profile": profile_label,
        "total": int(len(df)),
        "data_complete": int(df.get("data_complete_v1_1", pd.Series(dtype=int)).sum()),
        "strict_count": int(strict_mask.sum()),
        "categories": {},
        "top": [],
        "bucket_counts": {},
    }

    categories = [
        "ELITE_V1_1",
        "WATCHLIST_V1_1",
        "REJECT_V1_1",
        "DATA_MISSING",
        "EXCLUDED_UNIVERSE",
    ]
    counts = (
        df.get("category_v1_1", pd.Series(dtype=str)).value_counts().reindex(categories, fill_value=0)
        if not df.empty
        else pd.Series([0] * len(categories), index=categories)
    )
    summary["categories"] = {category: int(counts[category]) for category in categories}

    bucket_counts = (
        df.get(GARP_BUCKET_COLUMN, pd.Series(dtype=str))
        .value_counts()
        .to_dict()
        if not df.empty
        else {}
    )
    summary["bucket_counts"] = {bucket: int(count) for bucket, count in bucket_counts.items()}

    if "nexus_v2_bucket" in df.columns:
        v2_counts = df["nexus_v2_bucket"].value_counts().to_dict()
        summary["v2_bucket_counts"] = {bucket: int(count) for bucket, count in v2_counts.items()}

    garp_df = df[strict_mask]
    if not garp_df.empty:
        if GARP_SCORE_COLUMN in garp_df.columns:
            sort_by = [GARP_SCORE_COLUMN]
            ascending = [False]
        else:
            sort_by = ["score_v1_1"] if "score_v1_1" in garp_df.columns else [garp_df.columns[0]]
            ascending = [False]
        if "score_v1_1" in garp_df.columns and "score_v1_1" not in sort_by:
            sort_by.append("score_v1_1")
            ascending.append(False)
        if "market_cap" in garp_df.columns:
            sort_by.append("market_cap")
            ascending.append(False)
        garp_df = garp_df.sort_values(by=sort_by, ascending=ascending)
        top_df = garp_df.head(5)
    else:
        top_df = garp_df
    for row in top_df.itertuples():
        summary["top"].append(
            {
                "ticker": row.ticker,
                "score": int(getattr(row, "score_v1_1", 0)),
                "category": getattr(row, "category_v1_1", "NA"),
                "reason": getattr(row, "universe_exclusion_reason", "") or "OK",
            }
        )

    return summary


def write_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist the full results DataFrame to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path