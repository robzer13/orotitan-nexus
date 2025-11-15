"""Command-line interface for the OroTitan Nexus screener."""
from __future__ import annotations

import argparse
import logging
from typing import List

from .config import (
    ConfigError,
    FilterSettings,
    WeightSettings,
    UniverseSettings,
    ProfileSettings,
    load_settings,
)
from .orchestrator import run_universe
from .reporting import (
    print_global_preview,
    print_strict_preview,
    print_summary,
    print_ticker_diagnostics,
    print_v1_overlay,
    write_csv,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT = "cac40_screen_results.csv"
DEFAULT_MAX_ROWS = 40
PROFILE_CHOICES = ("defensive", "balanced", "offensive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAC 40 / SBF 120 GARP screener")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="CSV output path")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Number of rows to print in the global preview",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML configuration file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=PROFILE_CHOICES,
        default=None,
        help="Preset Nexus profile to tweak filters/weights",
    )
    parser.add_argument(
        "--detail",
        nargs="+",
        help="One or more tickers for which to display detailed diagnostics",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact OroTitan V1.1 summary in the console",
    )
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def run_screener(
    filters: FilterSettings,
    weights: WeightSettings,
    universe: UniverseSettings,
) -> pd.DataFrame:
    return run_universe(filters, weights, universe)


def print_and_export(
    df: pd.DataFrame,
    filters: FilterSettings,
    weights: WeightSettings,
    universe: UniverseSettings,
    profile: ProfileSettings,
    args: argparse.Namespace,
) -> None:
    if df.empty:
        LOGGER.warning(
            "Aucune donnée récupérée : vérifier la connectivité réseau ou la liste de tickers"
        )
        return

    print_strict_preview(df)
    print_global_preview(df, args.max_rows)
    print_v1_overlay(df)

    write_csv(df, args.output)
    LOGGER.info("Résultats complets sauvegardés dans %s", args.output)

    if args.summary:
        print_summary(df, universe, profile.name)

    if args.detail:
        print_ticker_diagnostics(
            df,
            args.detail,
            filters,
            weights,
            universe,
            profile.name,
        )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    try:
        filters, weights, universe, profile = load_settings(args.config, args.profile)
    except ConfigError as exc:  # pragma: no cover - defensive CLI guard
        LOGGER.error("Configuration invalide: %s", exc)
        raise SystemExit(1) from exc
    if profile.name:
        LOGGER.info("Applying Nexus profile: %s", profile.name)

    df = run_screener(filters, weights, universe)
    print_and_export(df, filters, weights, universe, profile, args)
