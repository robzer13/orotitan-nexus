"""Dedicated CAC40 GARP radar CLI."""
from __future__ import annotations

import argparse
import logging
from typing import Optional

import pandas as pd

from .cli import configure_logging, run_screener
from .config import build_settings_from_config, load_yaml_config
from .garp_rules import GARP_FLAG_COLUMN, apply_garp_rules
from .reporting import write_csv

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


def _apply_garp_flag(df: pd.DataFrame, thresholds) -> pd.DataFrame:
    df = df.copy()
    df[GARP_FLAG_COLUMN] = False
    if df.empty:
        return df
    mask = df["data_complete_v1_1"].fillna(False)
    if not mask.any():
        return df
    evaluated = apply_garp_rules(df.loc[mask].copy(), thresholds)
    df.loc[mask, GARP_FLAG_COLUMN] = evaluated[GARP_FLAG_COLUMN]
    return df


def _print_summary(df: pd.DataFrame, universe_name: str, profile_name: Optional[str]) -> None:
    profile_label = profile_name or "balanced"
    total = len(df)
    complete = int(df["data_complete_v1_1"].fillna(False).sum())
    strict = int(df[GARP_FLAG_COLUMN].fillna(False).sum())
    print("=== OroTitan CAC40 GARP Radar v1.3 ===")
    print(f"Profil            : {profile_label}")
    print(f"Univers           : {universe_name}")
    print(f"Total valeurs     : {total}")
    print(f"Data-complete     : {complete}")
    print(f"Strict GARP (5/5) : {strict}")
    categories = df["category_v1_1"].fillna("DATA_MISSING").value_counts()
    mapping = [
        ("ELITE_V1_1", "  ELITE_V1_1        :"),
        ("WATCHLIST_V1_1", "  WATCHLIST_V1_1    :"),
        ("REJECT_V1_1", "  REJECT_V1_1       :"),
        ("DATA_MISSING", "  DATA_MISSING      :"),
        ("EXCLUDED_UNIVERSE", "  EXCLUDED_UNIVERSE :"),
    ]
    for key, label in mapping:
        print(f"{label} {int(categories.get(key, 0))}")

    top = (
        df[df[GARP_FLAG_COLUMN]]
        .sort_values(by=["score_v1_1", "market_cap"], ascending=[False, False])
        .head(5)
    )
    if top.empty:
        print("\nAucune valeur ne valide les 5 règles GARP.")
    else:
        print("\nTickers GARP (top 5):")
        for _, row in top.iterrows():
            reason = row.get("universe_exclusion_reason", "") or "OK"
            print(
                f" - {row['ticker']}  score={row.get('score_v1_1', 0)}  "
                f"cat={row.get('category_v1_1', 'NA')}  reason={reason}"
            )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    config_data = load_yaml_config(args.config)
    filters, weights, universe, profile = build_settings_from_config(config_data, args.profile)

    df = run_screener(filters, weights, universe)
    if df.empty:
        LOGGER.warning("Aucune donnée récupérée pour l'univers CAC40")
        return

    df = _apply_garp_flag(df, profile.garp)
    write_csv(df, args.output_full)
    LOGGER.info("CSV complet enregistré dans %s", args.output_full)

    radar_df = df[df[GARP_FLAG_COLUMN]].copy()
    if not radar_df.empty:
        radar_df.sort_values(by=["market_cap"], ascending=False, inplace=True)
    write_csv(radar_df, args.output_radar)
    LOGGER.info("CSV radar GARP enregistré dans %s", args.output_radar)

    _print_summary(df, universe.name, profile.name)
if __name__ == "__main__":  # pragma: no cover
    main()
