"""Normalization utilities turning raw Yahoo Finance data into clean rows."""
from __future__ import annotations

import logging
from numbers import Number
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import FilterSettings, UniverseSettings
from . import data_fetch

LOGGER = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
ADV_WINDOW_DAYS = 63


def safe_float(value) -> float:
    """Convert ``value`` into ``float`` or ``np.nan``."""

    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _scalar(value) -> float | None:
    """Return a Python float from scalars/Series without FutureWarning."""

    if value is None:
        return None
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        return safe_float(value.iloc[0])
    return None


def normalize_debt_to_equity(raw_value) -> float:
    """Normalize Yahoo Finance debt/equity values into proportions."""

    value = safe_float(raw_value)
    if np.isnan(value):
        return np.nan
    if value > 10:  # Yahoo often exposes D/E as percentages
        return value / 100.0
    return value


def compute_eps_cagr_from_series(series: pd.Series) -> float:
    """Compute CAGR from a series of yearly earnings."""

    if series is None:
        return np.nan
    cleaned = series.dropna()
    if cleaned.shape[0] < 2:
        return np.nan
    cleaned = cleaned.sort_index()
    first = safe_float(cleaned.iloc[0])
    last = safe_float(cleaned.iloc[-1])
    periods = cleaned.shape[0] - 1
    index = cleaned.index
    try:
        if pd.api.types.is_datetime64_any_dtype(index):
            years = (index[-1] - index[0]).days / 365.0
        else:
            years = float(index[-1]) - float(index[0])
        if years > 0:
            periods = years
    except Exception:  # pragma: no cover - fallback to count-based periods
        pass
    if periods <= 0 or first <= 0 or last <= 0:
        return np.nan
    return (last / first) ** (1.0 / periods) - 1.0


def _net_income_series(income_stmt: pd.DataFrame | None) -> pd.Series:
    """Extract the Net Income series from ``income_stmt`` in chronological order."""

    if income_stmt is None or income_stmt.empty:
        return pd.Series(dtype=float)
    if "Net Income" not in income_stmt.index:
        return pd.Series(dtype=float)
    row = income_stmt.loc["Net Income"]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    series = row.apply(safe_float).dropna()
    if series.empty:
        return pd.Series(dtype=float)
    return series.sort_index()


def compute_eps_cagr(info: Dict, income_stmt: pd.DataFrame | None) -> Tuple[float, str]:
    """Return EPS/Net Income CAGR and its source (net_income/yoy/none).

    Yahoo Finance recently exposes yearly financial statements under ``income_stmt``.
    We rely on the Net Income row to build a CAGR over the available years
    (typically 3-5). If Net Income is missing or unusable we fall back to the
    lighter ``earningsGrowth`` hints from ``info`` so the engine still produces a
    best-effort growth proxy.
    """

    eps_cagr = np.nan
    source = "none"
    net_income_series = _net_income_series(income_stmt)
    if not net_income_series.empty:
        eps_cagr = compute_eps_cagr_from_series(net_income_series)
        if not np.isnan(eps_cagr):
            source = "net_income"

    if np.isnan(eps_cagr):
        eps_growth = safe_float(info.get("earningsGrowth"))
        if np.isnan(eps_growth):
            eps_growth = safe_float(info.get("earningsQuarterlyGrowth"))
        if not np.isnan(eps_growth):
            eps_cagr = eps_growth
            source = "yoy"
    return eps_cagr, source


def compute_peg(pe_fwd: float, eps_cagr: float, peg_info: float) -> float:
    """Compute PEG from Yahoo info or from PE/growth."""

    peg = safe_float(peg_info)
    if not np.isnan(peg) and peg > 0:
        return peg
    if not np.isnan(pe_fwd) and not np.isnan(eps_cagr) and eps_cagr > 0:
        return pe_fwd / (eps_cagr * 100.0)
    return np.nan


def compute_risk_metrics(price_df: pd.DataFrame) -> Dict[str, float]:
    """Return volatility, max drawdown, and ADV metrics."""

    metrics = {"vol_1y": np.nan, "mdd_1y": np.nan, "adv_3m": np.nan}
    if price_df is None or price_df.empty:
        return metrics

    price_col = "Adj Close" if "Adj Close" in price_df.columns else "Close"
    prices = price_df.get(price_col)
    if prices is not None:
        prices = prices.dropna()
        if prices.shape[0] >= 2:
            log_returns = np.log(prices / prices.shift(1)).dropna()
            if not log_returns.empty:
                std_value = _scalar(log_returns.std())
                if std_value is not None:
                    metrics["vol_1y"] = float(std_value * np.sqrt(TRADING_DAYS_PER_YEAR))
            running_max = prices.cummax()
            drawdowns = prices / running_max - 1.0
            if not drawdowns.empty:
                dd_value = _scalar(drawdowns.min())
                if dd_value is not None:
                    metrics["mdd_1y"] = float(dd_value)

    volume_series = price_df.get("Volume")
    if volume_series is not None:
        recent = volume_series.dropna().tail(ADV_WINDOW_DAYS)
        if not recent.empty:
            adv_value = _scalar(recent.mean())
            if adv_value is not None:
                metrics["adv_3m"] = float(adv_value)

    return metrics


def _has_positive(value: float) -> bool:
    return bool(not np.isnan(value) and value > 0)


def _has_value(value: float) -> bool:
    return bool(not np.isnan(value))


def core_v1_1_ready(pe_ttm: float, pe_fwd: float, debt_to_equity: float, eps_cagr: float) -> bool:
    """Return True if the core V1.1 metrics are available."""

    return all(
        (
            _has_positive(pe_ttm),
            _has_positive(pe_fwd),
            _has_value(debt_to_equity),
            _has_positive(eps_cagr),
        )
    )


def compute_data_ready_nexus(record: Dict[str, float]) -> bool:
    """Return True when enough data exists to compute Nexus scores."""

    requirements = (
        ("pe_ttm", True),
        ("pe_fwd", True),
        ("debt_to_equity", False),
        ("eps_cagr", True),
        ("market_cap", True),
        ("vol_1y", True),
        ("mdd_1y", False),
        ("adv_3m", True),
    )
    for column, must_be_positive in requirements:
        value = record.get(column, np.nan)
        if value is None or np.isnan(value):
            return False
        if column == "mdd_1y":
            if value >= 0:
                return False
            continue
        if must_be_positive and value <= 0:
            return False
    return True


def compute_data_ready_v1_1(record: Dict[str, float]) -> bool:
    """Return True when V1.1 inputs are present."""

    return core_v1_1_ready(
        record.get("pe_ttm", np.nan),
        record.get("pe_fwd", np.nan),
        record.get("debt_to_equity", np.nan),
        record.get("eps_cagr", np.nan),
    )


def _compute_de_ratio(info: Dict, balance_sheet: pd.DataFrame) -> float:
    de_ratio = normalize_debt_to_equity(info.get("debtToEquity"))
    if not np.isnan(de_ratio):
        return de_ratio
    if balance_sheet is not None and not balance_sheet.empty:
        total_debt = np.nan
        total_equity = np.nan
        if "Total Debt" in balance_sheet.index:
            debt_row = balance_sheet.loc["Total Debt"].dropna()
            if not debt_row.empty:
                total_debt = safe_float(debt_row.iloc[0])
        if "Total Stockholder Equity" in balance_sheet.index:
            equity_row = balance_sheet.loc["Total Stockholder Equity"].dropna()
            if not equity_row.empty:
                total_equity = safe_float(equity_row.iloc[0])
        if not np.isnan(total_debt) and not np.isnan(total_equity) and total_equity != 0:
            return total_debt / total_equity
    return np.nan


def build_record_for_ticker(
    ticker: str, filters: FilterSettings, universe: UniverseSettings
) -> Dict[str, float]:
    """Fetch and normalize all metrics for ``ticker``."""

    raw = data_fetch.fetch_raw_fundamentals(ticker)
    info = raw.get("info", {})
    income_stmt = raw.get("income_stmt")
    balance_sheet = raw.get("balance_sheet")

    current_price = safe_float(info.get("currentPrice"))
    if np.isnan(current_price):
        current_price = safe_float(info.get("regularMarketPrice"))

    trailing_pe = safe_float(info.get("trailingPE"))
    trailing_eps = safe_float(info.get("trailingEps"))
    if np.isnan(trailing_pe) and not np.isnan(current_price) and not np.isnan(trailing_eps) and trailing_eps != 0:
        trailing_pe = current_price / trailing_eps

    forward_pe = safe_float(info.get("forwardPE"))
    forward_eps = safe_float(info.get("forwardEps"))
    if np.isnan(forward_pe) and not np.isnan(current_price) and not np.isnan(forward_eps) and forward_eps != 0:
        forward_pe = current_price / forward_eps

    debt_to_equity = _compute_de_ratio(info, balance_sheet)
    eps_cagr, eps_cagr_source = compute_eps_cagr(info, income_stmt)
    peg = compute_peg(forward_pe, eps_cagr, info.get("pegRatio"))
    market_cap = safe_float(info.get("marketCap"))

    price_history = data_fetch.fetch_price_history(ticker)
    risk_metrics = compute_risk_metrics(price_history)

    record: Dict[str, float] = {
        "ticker": ticker,
        "pe_ttm": trailing_pe,
        "pe_fwd": forward_pe,
        "debt_to_equity": debt_to_equity,
        "eps_cagr": eps_cagr,
        "peg": peg,
        "market_cap": market_cap,
        "vol_1y": risk_metrics["vol_1y"],
        "mdd_1y": risk_metrics["mdd_1y"],
        "adv_3m": risk_metrics["adv_3m"],
        "eps_cagr_source": eps_cagr_source,
    }

    record["has_pe_ttm"] = _has_positive(record["pe_ttm"])
    record["has_pe_fwd"] = _has_positive(record["pe_fwd"])
    record["has_debt_to_equity"] = _has_value(record["debt_to_equity"])
    record["has_eps_cagr"] = _has_positive(record["eps_cagr"])
    record["has_peg"] = _has_positive(record["peg"])
    record["has_market_cap"] = _has_positive(record["market_cap"])
    record["has_risk_data"] = bool(
        _has_positive(record["vol_1y"]) and record["mdd_1y"] < 0 and _has_positive(record["adv_3m"])
    )

    record["data_ready_nexus"] = compute_data_ready_nexus(record)
    record["data_ready_v1_1"] = compute_data_ready_v1_1(record)
    record["data_complete_v1_1"] = bool(
        record["has_pe_ttm"]
        and record["has_pe_fwd"]
        and record["has_debt_to_equity"]
        and record["has_eps_cagr"]
        and record["has_market_cap"]
        and record["has_risk_data"]
    )

    return record
