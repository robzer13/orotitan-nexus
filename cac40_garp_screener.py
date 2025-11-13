"""Command-line GARP screener for CAC40 equities.

This MVP builds a Growth At a Reasonable Price (GARP) screener focused on
fundamental ratios that are easy to obtain from the Yahoo Finance API through
`yfinance`.  The script is intentionally modular so that the sourcing and
scoring logic can evolve independently.
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAC40_TICKERS: List[str] = [
    "MC.PA",
    "OR.PA",
    "AI.PA",
    "BNP.PA",
    # TODO: compléter la liste complète du CAC 40
]

MAX_PE_TTM = 25.0
MAX_DEBT_TO_EQUITY = 0.35
MIN_EPS_CAGR = 0.08

LOGGER = logging.getLogger("cac40_garp_screener")


def get_cac40_tickers() -> List[str]:
    """Return the list of CAC 40 tickers to screen."""

    return CAC40_TICKERS


# ---------------------------------------------------------------------------
# Data retrieval helpers
# ---------------------------------------------------------------------------
def normalize_debt_to_equity(value: Optional[float]) -> float:
    """Normalise le ratio dette/capitaux propres (exprimé en proportion).

    Yahoo Finance renvoie parfois le ratio en pourcentage (ex: 120). On
    considère qu'une valeur supérieure à 10 représente probablement un
    pourcentage, et on divise par 100 dans ce cas.
    """

    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    if value > 10:
        value /= 100.0
    return value


def compute_eps_cagr(earnings_series: Optional[pd.Series]) -> float:
    """Compute EPS CAGR using the yearly `earnings` series as a proxy.

    Parameters
    ----------
    earnings_series : pandas.Series
        Série temporelle des bénéfices annuels (ou EPS) triée par année.
    """

    if earnings_series is None or earnings_series.empty:
        return np.nan

    earnings_series = earnings_series.sort_index()
    first = earnings_series.iloc[0]
    last = earnings_series.iloc[-1]
    n_periods = len(earnings_series) - 1

    if n_periods <= 0:
        return np.nan
    if first <= 0 or last <= 0:
        return np.nan

    cagr = (last / first) ** (1 / n_periods) - 1
    return float(cagr)


def compute_peg(pe_fwd: Optional[float], eps_cagr: Optional[float]) -> float:
    """Return the PEG ratio based on the forward PE and EPS CAGR."""

    if pe_fwd is None or np.isnan(pe_fwd):
        return np.nan
    if eps_cagr is None or np.isnan(eps_cagr) or eps_cagr <= 0:
        return np.nan

    return pe_fwd / (eps_cagr * 100)


def fetch_fundamental_data(ticker: str) -> Dict[str, float]:
    """Fetch Yahoo Finance fundamentals for ``ticker`` and compute derived metrics."""

    yf_ticker = yf.Ticker(ticker)
    info: Dict[str, Optional[float]] = yf_ticker.info or {}

    pe_ttm = info.get("trailingPE")
    pe_fwd = info.get("forwardPE")
    debt_to_equity = normalize_debt_to_equity(info.get("debtToEquity"))
    roe = info.get("returnOnEquity")
    market_cap = info.get("marketCap")

    earnings = yf_ticker.earnings
    earnings_series = earnings["Earnings"] if earnings is not None and not earnings.empty else None
    eps_cagr = compute_eps_cagr(earnings_series)
    peg = compute_peg(pe_fwd, eps_cagr)

    critical_values = [pe_ttm, debt_to_equity, eps_cagr]
    data_incomplete = any(value is None or np.isnan(value) for value in critical_values)

    return {
        "ticker": ticker,
        "pe_ttm": float(pe_ttm) if pe_ttm is not None else np.nan,
        "pe_fwd": float(pe_fwd) if pe_fwd is not None else np.nan,
        "debt_to_equity": float(debt_to_equity) if debt_to_equity is not None else np.nan,
        "eps_cagr": float(eps_cagr) if eps_cagr is not None else np.nan,
        "peg": float(peg) if peg is not None else np.nan,
        "roe": float(roe) if roe is not None else np.nan,
        "market_cap": float(market_cap) if market_cap is not None else np.nan,
        "data_incomplete": data_incomplete,
    }


# ---------------------------------------------------------------------------
# Screening logic
# ---------------------------------------------------------------------------
def passes_hard_filters(row: pd.Series) -> bool:
    """Return True if ``row`` satisfies the strict GARP constraints."""

    if row.get("data_incomplete", False):
        return False
    if not (0 < row.get("pe_ttm", np.nan) <= MAX_PE_TTM):
        return False
    if not (row.get("debt_to_equity", np.nan) <= MAX_DEBT_TO_EQUITY):
        return False
    if not (row.get("eps_cagr", np.nan) >= MIN_EPS_CAGR):
        return False
    return True


def compute_garp_score(row: pd.Series) -> float:
    """Compute the 0-100 GARP score for a company."""

    if not row.get("passes_hard_filters", False):
        return 0.0

    score = 40.0
    pe_fwd = row.get("pe_fwd", np.nan)
    peg = row.get("peg", np.nan)
    eps_cagr = row.get("eps_cagr", np.nan)
    roe = row.get("roe", np.nan)

    if not np.isnan(pe_fwd) and pe_fwd <= 15:
        score += 15

    if not np.isnan(peg):
        if peg <= 1.0:
            score += 25
        elif peg <= 1.2:
            score += 15

    if not np.isnan(eps_cagr) and eps_cagr >= 0.15:
        score += 10

    if not np.isnan(roe):
        if roe >= 0.10:
            score += 10
        elif roe >= 0.08:
            score += 5

    return float(np.clip(score, 0, 100))


def run_screener(tickers: Iterable[str]) -> pd.DataFrame:
    """Run the full pipeline: download, compute metrics, and score the universe."""

    records = []
    for ticker in tickers:
        try:
            fundamentals = fetch_fundamental_data(ticker)
            records.append(fundamentals)
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to retrieve data for %s", ticker)
            records.append(
                {
                    "ticker": ticker,
                    "pe_ttm": np.nan,
                    "pe_fwd": np.nan,
                    "debt_to_equity": np.nan,
                    "eps_cagr": np.nan,
                    "peg": np.nan,
                    "roe": np.nan,
                    "market_cap": np.nan,
                    "data_incomplete": True,
                }
            )

    df = pd.DataFrame.from_records(records)
    df["passes_hard_filters"] = df.apply(passes_hard_filters, axis=1)
    df["garp_score"] = df.apply(compute_garp_score, axis=1)
    return df


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------
def configure_logging(verbose: bool = False) -> None:
    """Configure the global logger for the CLI execution."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the screener."""

    parser = argparse.ArgumentParser(description="Screener GARP pour le CAC40")
    parser.add_argument("--top_n", type=int, default=10, help="Nombre de lignes à afficher")
    parser.add_argument(
        "--output", type=str, default="cac40_garp_screener.csv", help="Chemin du fichier CSV de sortie"
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="Score minimum pour afficher une valeur (après tri décroissant)",
    )
    parser.add_argument("--verbose", action="store_true", help="Active les logs détaillés")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Entrypoint used by the CLI to orchestrate the screener."""

    configure_logging(args.verbose)
    tickers = get_cac40_tickers()
    LOGGER.info("Récupération des données pour %d tickers", len(tickers))
    df = run_screener(tickers)
    df.sort_values(by="garp_score", ascending=False, inplace=True)

    filtered_df = df
    if args.min_score is not None:
        filtered_df = df[df["garp_score"] >= args.min_score]

    LOGGER.info("Top %d valeurs:", args.top_n)
    print(filtered_df.head(args.top_n).to_string(index=False))

    df.to_csv(args.output, index=False)
    LOGGER.info("Résultats sauvegardés dans %s", args.output)


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
