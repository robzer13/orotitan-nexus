"""CAC 40 GARP-style screener with strict filters and weighted scoring.

The script downloads fundamental metrics for CAC 40 constituents via yfinance,
then applies the user-specified hard filters alongside a soft scoring model
based on valuation, growth, balance-sheet quality, and market capitalization.
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Universe configuration
# ---------------------------------------------------------------------------
CAC40_TICKERS: List[str] = [
    "MC.PA",
    "OR.PA",
    "AI.PA",
    "BNP.PA",
    # TODO: compléter la liste complète du CAC 40
]

TRAILING_PE_MAX = 25.0
FORWARD_PE_MAX = 15.0
DEBT_TO_EQUITY_MAX = 35.0
EPS_GROWTH_MIN = 0.15  # 15 %
PEG_MAX = 1.2
MARKET_CAP_MIN = 5e9

STRICT_SORT_COLUMNS = ["strict_pass", "garp_score"]
DEFAULT_OUTPUT = "cac40_screen_results.csv"
DEFAULT_MAX_ROWS = 40

LOGGER = logging.getLogger("cac40_garp_screener")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_cac40_tickers() -> List[str]:
    """Return the list of CAC 40 tickers to evaluate."""

    return CAC40_TICKERS


def safe_float(value: Optional[float]) -> float:
    """Return ``value`` as float, or NaN when conversion fails."""

    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fetch_fundamental_data(ticker: str) -> Dict[str, float]:
    """Fetch Yahoo Finance info for ``ticker`` and extract required fields."""

    yf_ticker = yf.Ticker(ticker)
    info: Dict[str, Optional[float]] = yf_ticker.info or {}

    # EPS growth proxy. Documented to use `earningsGrowth` with a fallback to
    # `earningsQuarterlyGrowth` when annual data is not available.
    eps_growth = info.get("earningsGrowth")
    if eps_growth is None:
        eps_growth = info.get("earningsQuarterlyGrowth")

    return {
        "ticker": ticker,
        "trailingPE": safe_float(info.get("trailingPE")),
        "forwardPE": safe_float(info.get("forwardPE")),
        "debtToEquity": safe_float(info.get("debtToEquity")),
        "eps_growth": safe_float(eps_growth),
        "pegRatio": safe_float(info.get("pegRatio")),
        "marketCap": safe_float(info.get("marketCap")),
    }


def build_dataframe(records: Iterable[Dict[str, float]]) -> pd.DataFrame:
    """Create a DataFrame with boolean filters and scoring columns."""

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    # Strict filter flags.
    df["per_ok"] = df["trailingPE"].apply(
        lambda v: bool(not np.isnan(v) and v < TRAILING_PE_MAX)
    )
    df["per_forward_ok"] = df["forwardPE"].apply(
        lambda v: bool(not np.isnan(v) and v < FORWARD_PE_MAX)
    )
    df["de_ok"] = df["debtToEquity"].apply(
        lambda v: bool(not np.isnan(v) and v < DEBT_TO_EQUITY_MAX)
    )
    df["eps_growth_ok"] = df["eps_growth"].apply(
        lambda v: bool(not np.isnan(v) and v > EPS_GROWTH_MIN)
    )
    df["peg_ok"] = df["pegRatio"].apply(
        lambda v: bool(not np.isnan(v) and v < PEG_MAX)
    )
    df["mktcap_ok"] = df["marketCap"].apply(
        lambda v: bool(not np.isnan(v) and v > MARKET_CAP_MIN)
    )

    df["strict_pass"] = (
        df["per_ok"]
        & df["per_forward_ok"]
        & df["de_ok"]
        & df["eps_growth_ok"]
        & df["peg_ok"]
        & df["mktcap_ok"]
    )

    # Soft scores.
    df["valuation_score"] = df.apply(
        lambda row: compute_valuation_score(
            row["trailingPE"], row["forwardPE"], row["pegRatio"]
        ),
        axis=1,
    )
    df["growth_score"] = df["eps_growth"].apply(score_growth)
    df["balance_sheet_score"] = df["debtToEquity"].apply(score_balance_sheet)
    df["size_score"] = df["marketCap"].apply(score_size)
    df["garp_score"] = df.apply(
        lambda row: compute_final_garp_score(
            row["valuation_score"],
            row["growth_score"],
            row["balance_sheet_score"],
            row["size_score"],
        ),
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------
def interpolate(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linearly interpolate ``value`` between ``(x0, y0)`` and ``(x1, y1)``."""

    if x0 == x1:
        return y1
    return y0 + (value - x0) * (y1 - y0) / (x1 - x0)


def compute_valuation_score(trailing_pe: float, forward_pe: float, peg: float) -> float:
    """Average the valuation metrics while ignoring missing values."""

    parts: List[float] = []
    if not np.isnan(trailing_pe):
        if trailing_pe <= 15:
            parts.append(100.0)
        elif trailing_pe <= 25:
            parts.append(interpolate(trailing_pe, 15, 25, 100.0, 50.0))
        else:
            parts.append(0.0)
    if not np.isnan(forward_pe):
        if forward_pe <= 15:
            parts.append(100.0)
        elif forward_pe <= 25:
            parts.append(interpolate(forward_pe, 15, 25, 100.0, 50.0))
        else:
            parts.append(0.0)
    if not np.isnan(peg):
        if peg <= 1.0:
            parts.append(100.0)
        elif peg <= 1.5:
            parts.append(interpolate(peg, 1.0, 1.5, 100.0, 50.0))
        else:
            parts.append(0.0)

    if not parts:
        return 0.0
    return float(np.mean(parts))


def score_growth(eps_growth: float) -> float:
    """Score EPS growth according to the piecewise rules."""

    if np.isnan(eps_growth):
        return np.nan
    if eps_growth >= 0.15:
        return 100.0
    if eps_growth >= 0.05:
        return interpolate(eps_growth, 0.05, 0.15, 50.0, 100.0)
    if eps_growth > 0:
        return interpolate(eps_growth, 0.0, 0.05, 0.0, 50.0)
    return 0.0


def score_balance_sheet(de_ratio: float) -> float:
    """Score Debt/Equity according to resilience thresholds."""

    if np.isnan(de_ratio):
        return np.nan
    if de_ratio <= 35:
        return 100.0
    if de_ratio <= 70:
        return interpolate(de_ratio, 35.0, 70.0, 100.0, 50.0)
    if de_ratio <= 150:
        return interpolate(de_ratio, 70.0, 150.0, 50.0, 0.0)
    return 0.0


def score_size(market_cap: float) -> float:
    """Score company size as a proxy for robustness."""

    if np.isnan(market_cap):
        return np.nan
    if market_cap >= 50e9:
        return 100.0
    if market_cap >= 5e9:
        return interpolate(market_cap, 5e9, 50e9, 50.0, 100.0)
    return 0.0


def compute_final_garp_score(
    valuation_score: float,
    growth_score: float,
    balance_sheet_score: float,
    size_score: float,
) -> float:
    """Weighted average of the sub-scores, renormalising missing ones."""

    weights = [
        (valuation_score, 0.30),
        (growth_score, 0.30),
        (balance_sheet_score, 0.25),
        (size_score, 0.15),
    ]
    valid = [(score, weight) for score, weight in weights if not np.isnan(score)]
    if not valid:
        return 0.0

    total_weight = sum(weight for _, weight in valid)
    weighted_sum = sum(score * weight for score, weight in valid)
    return float(np.clip(weighted_sum / total_weight, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Screener orchestration
# ---------------------------------------------------------------------------
def run_screener(tickers: Iterable[str]) -> pd.DataFrame:
    """Fetch data for ``tickers`` and compute filters plus scores."""

    records: List[Dict[str, float]] = []
    for ticker in tickers:
        try:
            records.append(fetch_fundamental_data(ticker))
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to fetch data for %s", ticker)
            records.append(
                {
                    "ticker": ticker,
                    "trailingPE": np.nan,
                    "forwardPE": np.nan,
                    "debtToEquity": np.nan,
                    "eps_growth": np.nan,
                    "pegRatio": np.nan,
                    "marketCap": np.nan,
                }
            )

    return build_dataframe(records)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the screener."""

    parser = argparse.ArgumentParser(description="CAC 40 GARP screener")
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Chemin du fichier CSV à générer (défaut: %(default)s)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Nombre de lignes à afficher pour l'aperçu global (défaut: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Active les logs DEBUG pour suivre les téléchargements",
    )
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    """Configure logging format/level for CLI usage."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.verbose)

    tickers = get_cac40_tickers()
    LOGGER.info("Téléchargement des fondamentaux pour %d tickers", len(tickers))

    df = run_screener(tickers)
    if df.empty:
        LOGGER.warning("Aucune donnée récupérée : vérifier la connectivité réseau ou la liste de tickers")
        return

    df.sort_values(by=STRICT_SORT_COLUMNS, ascending=[False, False], inplace=True)

    strict_df = df[df["strict_pass"]]
    if strict_df.empty:
        print("Aucune valeur ne passe le filtre strict GARP.")
    else:
        print("Valeurs qui passent le filtre strict GARP:")
        print(strict_df.to_string(index=False))

    print("\nAperçu global trié par score (top %d):" % args.max_rows)
    print(df.head(args.max_rows).to_string(index=False))

    df.to_csv(args.output, index=False)
    LOGGER.info("Résultats complets sauvegardés dans %s", args.output)


if __name__ == "__main__":
    main()
