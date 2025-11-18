"""Data access helpers built on top of yfinance."""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)


def fetch_raw_fundamentals(ticker: str) -> Dict[str, Any]:
    """Return the raw Yahoo Finance structures for ``ticker``."""

    try:
        yf_ticker = yf.Ticker(ticker)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to create yfinance.Ticker for %s: %s", ticker, exc)
        return {
            "info": {},
            "income_stmt": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
        }

    info = yf_ticker.info or {}

    try:
        income_stmt = yf_ticker.income_stmt
    except Exception:  # pragma: no cover - defensive logging
        income_stmt = pd.DataFrame()

    try:
        balance_sheet = yf_ticker.balance_sheet
    except Exception:  # pragma: no cover - defensive logging
        balance_sheet = pd.DataFrame()

    return {"info": info, "income_stmt": income_stmt, "balance_sheet": balance_sheet}


def fetch_price_history(ticker: str, lookback_days: int = 252) -> pd.DataFrame:
    """Download daily price history for ``ticker`` over ``lookback_days``."""

    period = f"{max(int(lookback_days), 5)}d"
    try:
        history = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to download price history for %s: %s", ticker, exc)
        return pd.DataFrame()

    if not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame()
    return history
