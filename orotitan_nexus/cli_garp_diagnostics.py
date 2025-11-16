"""Offline CLI for rule-level GARP diagnostics."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from .api import run_garp_diagnostics_offline
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
    parser = argparse.ArgumentParser(description="Offline GARP rule diagnostics")
    parser.add_argument("--snapshot", required=True, help="CSV snapshot of a GARP run")
    parser.add_argument("--prices", required=True, help="Long-format prices CSV")
    parser.add_argument("--start-date", required=True, help="ISO start date (YYYY-MM-DD)")
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS, help="Comma-separated trading-day horizons")
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

    result = run_garp_diagnostics_offline(
        snapshot_path=args.snapshot,
        prices_path=args.prices,
        start_date=args.start_date,
        horizons=horizons,
    )

    LOGGER.info("=== OroTitan GARP Rule Diagnostics (offline) ===")
    LOGGER.info("Snapshot : %s", Path(args.snapshot).resolve())
    LOGGER.info("Prices   : %s", Path(args.prices).resolve())
    LOGGER.info("Start    : %s", args.start_date)
    LOGGER.info("")

    header = "Rule            Horizon  Pass%   Fail%   Delta%  n_pass  n_fail"
    LOGGER.info(header)
    for (rule, horizon), row in result.reset_index().set_index(["rule", "horizon_days"]).iterrows():
        LOGGER.info(
            "%-15s %7d  %6s  %6s  %7s  %6d  %6d",
            rule,
            int(horizon),
            _format_pct(row["pass_return"]),
            _format_pct(row["fail_return"]),
            _format_pct(row["delta_return"]),
            int(row["n_pass"]),
            int(row["n_fail"]),
        )


if __name__ == "__main__":  # pragma: no cover
    main()
