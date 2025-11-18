"""Offline price loading utilities for OroTitan Nexus backtests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PRICE_REQUIRED_COLUMNS = ["date", "ticker", "adj_close"]


@dataclass
class PriceLoaderSettings:
    """Settings for loading local price data."""

    csv_path: Path


def load_prices(path: Path, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Load daily prices from a long-format CSV.

    The CSV must contain columns ``date`` (YYYY-MM-DD), ``ticker``, and ``adj_close``.
    If ``tickers`` is provided, the returned dataframe is filtered to that subset.
    Dates are parsed to naive ``Timestamp`` values normalized to midnight. Rows with
    missing ``adj_close`` values are dropped. A ``ValueError`` is raised when
    required columns are absent.
    """

    df = pd.read_csv(path)
    missing = [col for col in PRICE_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Prices file {path} is missing columns: {', '.join(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["adj_close"])

    if tickers is not None:
        tickers_set = {t for t in tickers if t}
        df = df[df["ticker"].isin(tickers_set)]

    return df
