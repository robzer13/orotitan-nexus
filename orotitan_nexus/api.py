"""Public Python API for OroTitan Nexus."""
from __future__ import annotations

import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

from .config import load_settings, make_inline_universe, ProfileSettingsV2
from .garp_rules import GARP_FLAG_COLUMN, GARP_SCORE_COLUMN
from .history import GarpRunRecord, append_history_record
from .orchestrator import run_universe
from .reporting import summarize_garp, summarize_v1
from .prices import load_prices
from .backtest import backtest_garp_vs_benchmark
from .backtest import backtest_quintiles_by_score
from .diagnostics import compute_rule_diagnostics
from .scoring_v2 import apply_v2_scores
from .nexus_core_scoring import apply_nexus_core_scores
from .valuation import apply_valuation
from .etf_scoring import compute_etf_nexus_score
from .explain_v2 import explain_ticker, add_explain_columns
from .playbook import run_nexus_playbook, PlaybookSummary

LOGGER = logging.getLogger(__name__)


def _default_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _load_portfolio_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ticker", "quantity", "cost_basis"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Portfolio CSV missing columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str)
    df["position_quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["position_cost_basis"] = pd.to_numeric(df["cost_basis"], errors="coerce")
    return df[["ticker", "position_quantity", "position_cost_basis"]]


def _overlay_portfolio(df: pd.DataFrame, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or portfolio_df.empty:
        return df

    merged = df.merge(portfolio_df, on="ticker", how="left")
    merged["owned"] = merged["position_quantity"].fillna(0) > 0
    merged["owned_bool"] = merged["owned"]

    if "price" in merged.columns:
        merged["position_value"] = merged["position_quantity"] * merged["price"]
    else:
        merged["position_value"] = np.nan
    return merged


def _log_history(summary: dict, history_path: Optional[str], run_id: str, notes: Optional[str]) -> None:
    if not history_path:
        return
    bucket_counts = summary.get("bucket_counts", {}) or {}
    record = GarpRunRecord(
        timestamp=datetime.utcnow(),
        universe_name=summary.get("universe", ""),
        profile_name=summary.get("profile", ""),
        run_id=run_id,
        total=int(summary.get("total", 0)),
        data_complete_v1_1=int(summary.get("data_complete", 0)),
        strict_garp_count=int(summary.get("strict_count", 0)),
        elite_count=int(bucket_counts.get("ELITE_GARP", 0)),
        strong_count=int(bucket_counts.get("STRONG_GARP", 0)),
        borderline_count=int(bucket_counts.get("BORDERLINE_GARP", 0)),
        reject_count=int(bucket_counts.get("REJECT_GARP", 0)),
        notes=notes,
    )
    try:
        append_history_record(Path(history_path), record)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Unable to append GARP history into %s: %s", history_path, exc)


def run_screen(
    *,
    config_path: Optional[str] = None,
    profile_name: Optional[str] = None,
    apply_garp: bool = False,
    explain: bool = False,
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
    if isinstance(profile, ProfileSettingsV2):
        if profile.v2.enabled:
            df = apply_v2_scores(df, profile)
            if explain:
                df = add_explain_columns(df, profile)
        if getattr(profile, "nexus_core", None) and profile.nexus_core.enabled:
            df = apply_nexus_core_scores(df, profile)
        if getattr(profile, "valuation", None) and profile.valuation.enabled:
            df = apply_valuation(df, profile)
        if getattr(profile, "etf_scoring", None) and profile.etf_scoring.enabled:
            df["etf_nexus_score"] = compute_etf_nexus_score(df, profile.etf_scoring, profile)
    summary = summarize_v1(df, universe, profile.name)
    return df, summary


def run_cac40_garp(
    *,
    config_path: Optional[str] = None,
    profile_name: Optional[str] = None,
    history_path: Optional[str] = None,
    run_id: Optional[str] = None,
    notes: Optional[str] = None,
    portfolio_path: Optional[str] = None,
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
    if portfolio_path:
        portfolio_df = _load_portfolio_csv(portfolio_path)
        df = _overlay_portfolio(df, portfolio_df)
    if isinstance(profile, ProfileSettingsV2):
        if profile.v2.enabled:
            df = apply_v2_scores(df, profile)
        if getattr(profile, "nexus_core", None) and profile.nexus_core.enabled:
            df = apply_nexus_core_scores(df, profile)
        if getattr(profile, "valuation", None) and profile.valuation.enabled:
            df = apply_valuation(df, profile)
        if getattr(profile, "etf_scoring", None) and profile.etf_scoring.enabled:
            df["etf_nexus_score"] = compute_etf_nexus_score(df, profile.etf_scoring, profile)
    mask = df[GARP_FLAG_COLUMN] if GARP_FLAG_COLUMN in df else pd.Series(False, index=df.index)
    radar_df = df[mask].copy()
    if not radar_df.empty:
        sort_by = [GARP_SCORE_COLUMN]
        ascending = [False]
        if "score_v1_1" in radar_df.columns:
            sort_by.append("score_v1_1")
            ascending.append(False)
        if "market_cap" in radar_df.columns:
            sort_by.append("market_cap")
            ascending.append(False)
        radar_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
    resolved_run_id = run_id or _default_run_id()
    summary = summarize_garp(df, universe.name, profile.name)
    summary["run_id"] = resolved_run_id
    _log_history(summary, history_path, resolved_run_id, notes)
    return df, radar_df, summary


def run_custom_garp(
    *,
    config_path: Optional[str] = None,
    profile_name: Optional[str] = None,
    tickers: Iterable[str],
    universe_name: str = "CUSTOM",
    history_path: Optional[str] = None,
    run_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the strict GARP radar on an arbitrary ticker universe."""

    filters, weights, universe, profile = load_settings(config_path, profile_name)
    cleaned = [ticker.strip() for ticker in tickers if ticker and ticker.strip()]
    if not cleaned:
        raise ValueError("Custom GARP radar requires at least one ticker")
    inline_universe = make_inline_universe(universe_name or "CUSTOM", cleaned, template=universe)
    df = run_universe(
        filters,
        weights,
        inline_universe,
        apply_garp=True,
        garp_thresholds=profile.garp,
    )
    if isinstance(profile, ProfileSettingsV2):
        if profile.v2.enabled:
            df = apply_v2_scores(df, profile)
        if getattr(profile, "nexus_core", None) and profile.nexus_core.enabled:
            df = apply_nexus_core_scores(df, profile)
    mask = df[GARP_FLAG_COLUMN] if GARP_FLAG_COLUMN in df else pd.Series(False, index=df.index)
    radar_df = df[mask].copy()
    if not radar_df.empty:
        sort_by = [GARP_SCORE_COLUMN, "score_v1_1"] if "score_v1_1" in radar_df.columns else [GARP_SCORE_COLUMN]
        ascending = [False] * len(sort_by)
        if "market_cap" in radar_df.columns:
            sort_by.append("market_cap")
            ascending.append(False)
        radar_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
    resolved_run_id = run_id or _default_run_id()
    summary = summarize_garp(
        df,
        inline_universe.name,
        profile.name,
        header="=== OroTitan Custom GARP Radar v1.7 ===",
    )
    summary["run_id"] = resolved_run_id
    summary["profile_object"] = profile
    _log_history(summary, history_path, resolved_run_id, notes)
    return df, radar_df, summary


def run_garp_backtest_offline(
    snapshot_path: str,
    prices_path: str,
    start_date: str,
    horizons: Sequence[int] = (21, 63, 252),
) -> pd.DataFrame:
    """Offline GARP backtest comparing strict passers vs a benchmark.

    Parameters
    ----------
    snapshot_path : str
        Path to a CSV snapshot of a GARP run (full dataframe) with at least
        ``ticker``, ``strict_pass_garp``, and ``data_complete_v1_1`` columns.
    prices_path : str
        Path to a long-format prices CSV with ``date``, ``ticker``, and
        ``adj_close`` columns.
    start_date : str
        ISO date string (YYYY-MM-DD) from which to start the backtest.
    horizons : sequence of int
        Trading-day horizons for which to compute performance.

    Returns
    -------
    pd.DataFrame
        Indexed by horizon_days with columns ``garp_return``, ``benchmark_return``,
        ``excess_return``, ``n_garp_tickers``, and ``n_benchmark_tickers``.
    """

    snapshot_df = pd.read_csv(snapshot_path)
    price_df = load_prices(Path(prices_path))
    start_ts = pd.to_datetime(start_date, utc=False).normalize()
    return backtest_garp_vs_benchmark(snapshot_df, price_df, start_ts, horizons)


def run_garp_diagnostics_offline(
    snapshot_path: str,
    prices_path: str,
    start_date: str,
    horizons: Sequence[int] = (21, 63, 252),
) -> pd.DataFrame:
    """Offline GARP rule diagnostics comparing pass vs fail cohorts.

    Parameters
    ----------
    snapshot_path : str
        Path to a CSV snapshot with ticker-level GARP rule flags
        (``garp_*_ok``), ``strict_pass_garp``, and ``data_complete_v1_1``.
    prices_path : str
        Path to a long-format prices CSV with ``date``, ``ticker``, and
        ``adj_close`` columns.
    start_date : str
        ISO date string (YYYY-MM-DD) from which to start the forward horizons.
    horizons : sequence of int
        Trading-day horizons for which to compute performance.

    Returns
    -------
    pd.DataFrame
        MultiIndex (rule, horizon_days) with pass/fail returns and counts.
    """

    snapshot_df = pd.read_csv(snapshot_path)
    price_df = load_prices(Path(prices_path))
    start_ts = pd.to_datetime(start_date, utc=False).normalize()
    return compute_rule_diagnostics(snapshot_df, price_df, start_ts, horizons)


def run_score_backtest_offline(
    snapshot_path: str,
    prices_path: str,
    start_date: str,
    horizons: Sequence[int] = (21, 63, 252),
    score_column: str = "nexus_v2_score",
) -> pd.DataFrame:
    """Offline quintile backtest on an arbitrary score column (v2-friendly)."""

    snapshot_df = pd.read_csv(snapshot_path)
    price_df = load_prices(Path(prices_path))
    start_ts = pd.to_datetime(start_date, utc=False).normalize()
    return backtest_quintiles_by_score(snapshot_df, price_df, start_ts, horizons, score_column)


def explain_single_ticker(
    df: pd.DataFrame, ticker: str, profile: Optional[ProfileSettingsV2]
) -> dict:
    """Return a lightweight explanation for a single ticker (v2 only)."""

    if not isinstance(profile, ProfileSettingsV2) or not profile.v2.enabled:
        return {}
    return explain_ticker(df, ticker, profile)


def run_playbook(
    *,
    config_path: Optional[str] = None,
    profile_name: Optional[str] = None,
    universe_override: Optional[Sequence[str]] = None,
    portfolio_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> tuple[pd.DataFrame, PlaybookSummary]:
    """Execute the full pipeline then derive a Nexus Playbook.

    The playbook layer is opt-in via ``profile.playbook.enabled`` and simply
    adds structured BUY/ADD/HOLD/WATCH/AVOID recommendations; existing v1/v2
    outputs remain unchanged when it is disabled.
    """

    filters, weights, universe, profile = load_settings(config_path, profile_name)
    if universe_override:
        universe = make_inline_universe(universe.name, list(universe_override), template=universe)

    df = run_universe(
        filters,
        weights,
        universe,
        apply_garp=True,
        garp_thresholds=profile.garp,
    )

    if isinstance(profile, ProfileSettingsV2):
        if profile.v2.enabled:
            df = apply_v2_scores(df, profile)
        if getattr(profile, "nexus_core", None) and profile.nexus_core.enabled:
            df = apply_nexus_core_scores(df, profile)
        if getattr(profile, "valuation", None) and profile.valuation.enabled:
            df = apply_valuation(df, profile)
        if getattr(profile, "etf_scoring", None) and profile.etf_scoring.enabled:
            df["etf_nexus_score"] = compute_etf_nexus_score(df, profile.etf_scoring, profile)

    # Portfolio overlay is optional; callers can supply a context dict if needed.
    playbook_summary = PlaybookSummary(
        profile_name=profile.name or "",
        universe_name=universe.name,
        run_id=run_id,
        date=None,
        portfolio_hhi=None,
        portfolio_top5_weight=None,
        portfolio_hhi_zone=None,
        portfolio_top5_zone=None,
        total_universe=int(len(df)),
        total_strict_garp=int(df.get(GARP_FLAG_COLUMN, pd.Series([], dtype=bool)).sum()) if GARP_FLAG_COLUMN in df else 0,
        total_core_enabled=int(df.get("nexus_core_score", pd.Series([], dtype=float)).notna().sum()),
        total_owned=0,
        decisions=[],
        counts_by_action={},
        top_by_core=[],
        top_by_v2=[],
        top_by_upside=[],
        top_by_etf=[],
    )

    if isinstance(profile, ProfileSettingsV2) and profile.playbook.enabled:
        playbook_summary = run_nexus_playbook(
            df,
            profile,
            universe_name=universe.name,
            run_id=run_id,
            date_str=None,
            portfolio_context=None,
        )

    return df, playbook_summary
