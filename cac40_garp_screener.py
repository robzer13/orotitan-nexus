"""CAC 40 GARP-style screener with strict filters, scoring, and risk metrics.

The script downloads fundamental metrics for CAC 40 constituents via yfinance,
then applies the user-specified hard filters alongside a soft scoring model
based on valuation, growth, balance-sheet quality, and market capitalization.
In addition, a price-based risk module computes volatility, drawdown, liquidity
proxies, and a consolidated risk score so that the CSV exposes both upside and
downside perspectives.
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

MAX_PE_TTM = 25.0
MAX_FORWARD_PE = 15.0
MAX_DEBT_TO_EQUITY = 0.35  # 35 %
MIN_EPS_CAGR = 0.08  # 8 % par an par défaut
MAX_PEG = 1.2
MIN_MARKET_CAP = 5e9

STRICT_SORT_COLUMNS = ["strict_pass", "garp_score"]
DEFAULT_OUTPUT = "cac40_screen_results.csv"
DEFAULT_MAX_ROWS = 40
TRADING_DAYS_PER_YEAR = 252
ADV_WINDOW_DAYS = 63  # ~3 months of trading days

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

    debt_to_equity = safe_float(info.get("debtToEquity"))
    if not np.isnan(debt_to_equity) and debt_to_equity > 10:
        # Yahoo Finance sometimes exposes the ratio as a percentage (e.g. 45),
        # so we convert it to a proportion when the raw value looks too large.
        debt_to_equity /= 100.0

    return {
        "ticker": ticker,
        "pe_ttm": safe_float(info.get("trailingPE")),
        "pe_fwd": safe_float(info.get("forwardPE")),
        "debt_to_equity": debt_to_equity,
        "eps_cagr": safe_float(eps_growth),
        "peg": safe_float(info.get("pegRatio")),
        "market_cap": safe_float(info.get("marketCap")),
    }


def fetch_risk_data(ticker: str) -> Dict[str, float]:
    """Download OHLCV data and compute volatility, drawdown, and liquidity."""

    metrics = {"vol_1y": np.nan, "mdd_1y": np.nan, "adv_3m": np.nan}
    try:
        history = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to download price data for %s", ticker, exc_info=True)
        return metrics

    if history.empty:
        LOGGER.warning("No price history retrieved for %s", ticker)
        return metrics

    price_col = "Adj Close" if "Adj Close" in history.columns else "Close"
    prices = history[price_col].dropna()
    if prices.shape[0] >= 2:
        # Use log returns for volatility so extreme moves aggregate properly.
        log_returns = np.log(prices / prices.shift(1)).dropna()
        if not log_returns.empty:
            vol = float(log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            metrics["vol_1y"] = vol

        rolling_max = prices.cummax()
        drawdowns = prices / rolling_max - 1.0
        if not drawdowns.empty:
            metrics["mdd_1y"] = float(drawdowns.min())

    volume_series = history.get("Volume")
    if volume_series is not None:
        recent_volume = volume_series.dropna().tail(ADV_WINDOW_DAYS)
        if not recent_volume.empty:
            metrics["adv_3m"] = float(recent_volume.mean())

    return metrics


def build_dataframe(records: Iterable[Dict[str, float]]) -> pd.DataFrame:
    """Create a DataFrame with boolean filters and scoring columns."""

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    # Strict filter flags.
    df["per_ok"] = df["pe_ttm"].apply(
        lambda v: bool(not np.isnan(v) and v <= MAX_PE_TTM)
    )
    df["per_forward_ok"] = df["pe_fwd"].apply(
        lambda v: bool(not np.isnan(v) and v <= MAX_FORWARD_PE)
    )
    df["de_ok"] = df["debt_to_equity"].apply(
        lambda v: bool(not np.isnan(v) and v <= MAX_DEBT_TO_EQUITY)
    )
    df["eps_growth_ok"] = df["eps_cagr"].apply(
        lambda v: bool(not np.isnan(v) and v >= MIN_EPS_CAGR)
    )
    df["peg_ok"] = df["peg"].apply(lambda v: bool(not np.isnan(v) and v <= MAX_PEG))
    df["mktcap_ok"] = df["market_cap"].apply(
        lambda v: bool(not np.isnan(v) and v >= MIN_MARKET_CAP)
    )

    df["strict_pass"] = df.apply(passes_hard_filters, axis=1)

    # Soft scores.
    df["valuation_score"] = df.apply(
        lambda row: compute_valuation_score(row["pe_ttm"], row["pe_fwd"], row["peg"]),
        axis=1,
    )
    df["growth_score"] = df["eps_cagr"].apply(score_growth)
    df["balance_sheet_score"] = df["debt_to_equity"].apply(score_balance_sheet)
    df["size_score"] = df["market_cap"].apply(score_size)
    df["garp_score"] = df.apply(compute_garp_score, axis=1)

    # Risk metrics and score are expected to be part of the records already, but
    # ensure the columns exist even if a ticker failed to download.
    for column in ("vol_1y", "mdd_1y", "adv_3m"):
        if column not in df:
            df[column] = np.nan
    df["risk_score"] = df.apply(compute_risk_score, axis=1)

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
    if de_ratio <= 0.35:
        return 100.0
    if de_ratio <= 0.70:
        return interpolate(de_ratio, 0.35, 0.70, 100.0, 50.0)
    if de_ratio <= 1.50:
        return interpolate(de_ratio, 0.70, 1.50, 50.0, 0.0)
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


def passes_hard_filters(row: pd.Series) -> bool:
    """Return True if ``row`` satisfies the strict GARP constraints."""

    pe_ttm = row.get("pe_ttm", np.nan)
    pe_fwd = row.get("pe_fwd", np.nan)
    de = row.get("debt_to_equity", np.nan)
    eps_cagr = row.get("eps_cagr", np.nan)
    peg = row.get("peg", np.nan)
    mcap = row.get("market_cap", np.nan)

    if np.isnan(pe_ttm) or pe_ttm <= 0 or pe_ttm > MAX_PE_TTM:
        return False
    if np.isnan(pe_fwd) or pe_fwd <= 0 or pe_fwd > MAX_FORWARD_PE:
        return False
    if np.isnan(de) or de > MAX_DEBT_TO_EQUITY:
        return False
    if np.isnan(eps_cagr) or eps_cagr < MIN_EPS_CAGR:
        return False
    if np.isnan(peg) or peg > MAX_PEG:
        return False
    if np.isnan(mcap) or mcap < MIN_MARKET_CAP:
        return False
    return True


def compute_garp_score(row: pd.Series) -> float:
    """Compute the 0-100 GARP score, even if hard filters fail."""

    pe_ttm = row.get("pe_ttm", np.nan)
    pe_fwd = row.get("pe_fwd", np.nan)
    peg = row.get("peg", np.nan)
    eps_cagr = row.get("eps_cagr", np.nan)
    de = row.get("debt_to_equity", np.nan)
    mcap = row.get("market_cap", np.nan)

    pe_ttm_score = compute_valuation_score(pe_ttm, np.nan, np.nan)
    pe_fwd_score = compute_valuation_score(np.nan, pe_fwd, np.nan)
    peg_score = compute_valuation_score(np.nan, np.nan, peg)

    val_components = [
        score for score in [pe_ttm_score, pe_fwd_score, peg_score] if not np.isnan(score)
    ]
    valuation_score = float(np.mean(val_components)) if val_components else np.nan

    growth_score = score_growth(eps_cagr)
    balance_score = score_balance_sheet(de)
    size_score = score_size(mcap)

    subscores: List[float] = []
    weights: List[float] = []

    if not np.isnan(valuation_score):
        subscores.append(valuation_score)
        weights.append(0.30)
    if not np.isnan(growth_score):
        subscores.append(growth_score)
        weights.append(0.30)
    if not np.isnan(balance_score):
        subscores.append(balance_score)
        weights.append(0.25)
    if not np.isnan(size_score):
        subscores.append(size_score)
        weights.append(0.15)

    if not subscores:
        return 0.0

    weights_arr = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    subscores_arr = np.array(subscores, dtype=float)
    return float(np.clip(np.dot(weights_arr, subscores_arr), 0.0, 100.0))


def compute_risk_score(row: pd.Series) -> float:
    """Compute a 0-100 downside risk score (higher = riskier)."""

    vol = row.get("vol_1y", np.nan)
    mdd = row.get("mdd_1y", np.nan)
    adv = row.get("adv_3m", np.nan)

    def score_volatility(volatility: float) -> float:
        if np.isnan(volatility):
            return np.nan
        if volatility <= 0.15:
            return 0.0
        if volatility <= 0.30:
            return interpolate(volatility, 0.15, 0.30, 20.0, 70.0)
        return 100.0

    def score_drawdown(drawdown: float) -> float:
        if np.isnan(drawdown):
            return np.nan
        depth = abs(drawdown)
        if depth <= 0.10:
            return 0.0
        if depth <= 0.30:
            return interpolate(depth, 0.10, 0.30, 30.0, 70.0)
        return 100.0

    def score_liquidity(avg_volume: float) -> float:
        if np.isnan(avg_volume):
            return np.nan
        if avg_volume >= 5_000_000:
            return 0.0
        if avg_volume >= 1_000_000:
            return interpolate(avg_volume, 1_000_000, 5_000_000, 60.0, 20.0)
        return 100.0

    vol_score = score_volatility(vol)
    dd_score = score_drawdown(mdd)
    liq_score = score_liquidity(adv)

    subscores: List[float] = []
    weights: List[float] = []

    for subscore, weight in (
        (vol_score, 0.40),
        (dd_score, 0.40),
        (liq_score, 0.20),
    ):
        if not np.isnan(subscore):
            subscores.append(subscore)
            weights.append(weight)

    if not subscores:
        return np.nan

    weights_arr = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    subscores_arr = np.array(subscores, dtype=float)
    return float(np.clip(np.dot(weights_arr, subscores_arr), 0.0, 100.0))


# ---------------------------------------------------------------------------
# Screener orchestration
# ---------------------------------------------------------------------------
def run_screener(tickers: Iterable[str]) -> pd.DataFrame:
    """Fetch data for ``tickers`` and compute filters plus scores."""

    records: List[Dict[str, float]] = []
    for ticker in tickers:
        try:
            record = fetch_fundamental_data(ticker)
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
            }

        record.update(fetch_risk_data(ticker))
        records.append(record)

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
