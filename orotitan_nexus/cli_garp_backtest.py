from __future__ import annotations

import argparse
import logging
import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from .api import run_garp_backtest_offline, run_score_backtest_offline
from .cli import configure_logging

LOGGER = logging.getLogger(__name__)
DEFAULT_HORIZONS = "21,63,252"


def _parse_horizons(raw: str) -> List[int]:
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("At least one horizon must be provided")
    return values


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline strict GARP backtest")
    parser.add_argument("--snapshot", required=True, help="CSV snapshot of a GARP run")
    parser.add_argument("--prices", required=True, help="Long-format prices CSV")
    parser.add_argument("--start-date", required=True, help="ISO start date (YYYY-MM-DD)")
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS, help="Comma-separated trading-day horizons")
    parser.add_argument(
        "--score-column",
        default=None,
        help="Optional score column for quintile analysis (e.g. nexus_v2_score)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def _format_pct(value: float | pd.NA) -> str:
    if value is pd.NA or pd.isna(value):
        return "NaN"
    return f"{value * 100:.1f}%"


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        horizons = _parse_horizons(args.horizons)
    except ValueError as exc:  # pragma: no cover - defensive
        LOGGER.error("Invalid horizons: %s", exc)
        raise SystemExit(1) from exc

    result = run_garp_backtest_offline(
        snapshot_path=args.snapshot,
        prices_path=args.prices,
        start_date=args.start_date,
        horizons=horizons,
    )

    LOGGER.info("=== OroTitan GARP Backtest (offline) ===")
    LOGGER.info("Snapshot: %s", Path(args.snapshot).resolve())
    LOGGER.info("Prices  : %s", Path(args.prices).resolve())
    LOGGER.info("Start   : %s", args.start_date)
    LOGGER.info("")
    header = "Horizon  GARP%   Bench%  Excess%  n_garp  n_bench"
    LOGGER.info(header)
    for horizon, row in result.reset_index().iterrows():
        LOGGER.info(
            "%6d  %6s  %6s  %7s  %6d  %7d",
            int(row["horizon_days"]),
            _format_pct(row["garp_return"]),
            _format_pct(row["benchmark_return"]),
            _format_pct(row["excess_return"]),
            int(row["n_garp_tickers"]),
            int(row["n_benchmark_tickers"]),
        )

    if args.score_column:
        quintiles = run_score_backtest_offline(
            snapshot_path=args.snapshot,
            prices_path=args.prices,
            start_date=args.start_date,
            horizons=horizons,
            score_column=args.score_column,
        )
        if not quintiles.empty:
            LOGGER.info("")
            LOGGER.info("Quintile performance by %s", args.score_column)
            for (bucket, horizon), row in quintiles.reset_index().iterrows():
                LOGGER.info(
                    "Q%d h=%d -> %s (n=%d)",
                    int(row["bucket"]) + 1,
                    int(row["horizon_days"]),
                    _format_pct(row["return"]),
                    int(row["n_names"]),
                )


if __name__ == "__main__":  # pragma: no cover
    main()
