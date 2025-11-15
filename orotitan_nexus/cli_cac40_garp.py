"""Dedicated CAC40 GARP radar CLI."""
from __future__ import annotations

import argparse
import logging
from typing import Optional

import pandas as pd

from .cli import configure_logging
from .config import ConfigError, load_settings
from .garp_rules import GARP_FLAG_COLUMN
from .orchestrator import run_universe
from .reporting import summarize_garp, write_csv

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = "configs/cac40.yaml"
DEFAULT_FULL_OUTPUT = "cac40_garp_full.csv"
DEFAULT_RADAR_OUTPUT = "cac40_garp_radar.csv"
PROFILE_CHOICES = ("defensive", "balanced", "offensive")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAC40 GARP Radar")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument(
        "--profile",
        type=str,
        choices=PROFILE_CHOICES,
        default=None,
        help="Optional Nexus profile overrides",
    )
    parser.add_argument("--output-full", type=str, default=DEFAULT_FULL_OUTPUT, help="Full CSV output")
    parser.add_argument("--output-radar", type=str, default=DEFAULT_RADAR_OUTPUT, help="Strict GARP CSV output")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        filters, weights, universe, profile = load_settings(args.config, args.profile)
    except ConfigError as exc:  # pragma: no cover - defensive CLI guard
        LOGGER.error("Configuration invalide: %s", exc)
        raise SystemExit(1) from exc

    df = run_universe(
        filters,
        weights,
        universe,
        apply_garp=True,
        garp_thresholds=profile.garp,
    )
    if df.empty:
        LOGGER.warning("Aucune donnée récupérée pour l'univers CAC40")
        return

    write_csv(df, args.output_full)
    LOGGER.info("CSV complet enregistré dans %s", args.output_full)

    radar_df = df[df[GARP_FLAG_COLUMN]].copy()
    if not radar_df.empty:
        radar_df.sort_values(by=["market_cap"], ascending=False, inplace=True)
    write_csv(radar_df, args.output_radar)
    LOGGER.info("CSV radar GARP enregistré dans %s", args.output_radar)

    summary = summarize_garp(df, universe.name, profile.name)
    LOGGER.info(summary["header"])
    LOGGER.info("Profil             : %s", summary["profile"])
    LOGGER.info("Univers            : %s", summary["universe"])
    LOGGER.info("Total valeurs      : %d", summary["total"])
    LOGGER.info("Data-complete V1.1 : %d", summary["data_complete"])
    LOGGER.info("Strict GARP (5/5)  : %d", summary["strict_count"])
    for category, count in summary["categories"].items():
        LOGGER.info("  %s: %d", category, count)

    if not summary["top"]:
        LOGGER.info("Aucune valeur ne valide les 5 règles GARP.")
    else:
        LOGGER.info("Tickers GARP (top %d):", len(summary["top"]))
        for entry in summary["top"]:
            LOGGER.info(
                " - %s score=%d cat=%s reason=%s",
                entry["ticker"],
                entry["score"],
                entry["category"],
                entry["reason"],
            )
if __name__ == "__main__":  # pragma: no cover
    main()
