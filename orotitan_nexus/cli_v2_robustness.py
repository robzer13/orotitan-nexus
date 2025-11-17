"""CLI to run v2.2 robustness checks (walk-forward, sensitivity, regimes)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .cli import configure_logging
from .config import load_settings, ProfileSettingsV2
from .robustness_v2 import run_walkforward_validation, run_sensitivity_analysis
from .regimes_v2 import regime_performance

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nexus v2.2 robustness checks")
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--profile", type=str, default=None, help="Profile name")
    parser.add_argument("--snapshot", type=str, required=True, help="Snapshot CSV")
    parser.add_argument("--prices", type=str, required=True, help="Prices CSV")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    _, _, _, profile = load_settings(args.config, args.profile)
    if not isinstance(profile, ProfileSettingsV2) or not profile.v2.enabled:
        LOGGER.info("Profil v2 non activé ; aucune analyse de robustesse exécutée")
        return

    snapshot_df = pd.read_csv(args.snapshot)
    price_df = pd.read_csv(args.prices)

    results = {}
    wf = run_walkforward_validation(snapshot_df, price_df, profile)
    if wf:
        LOGGER.info("Walk-forward: moyenne top bucket = %s", wf.get("mean_top_return"))
        results["walkforward"] = wf
    sens = run_sensitivity_analysis(snapshot_df, price_df, profile)
    if sens:
        LOGGER.info(
            "Sensibilité (%s): moyenne=%.3f std=%.3f",
            sens.get("stability_metric"),
            sens["summary"].get("mean"),
            sens["summary"].get("std"),
        )
        results["sensitivity"] = sens
    regimes = regime_performance(snapshot_df, price_df, profile)
    if regimes:
        LOGGER.info("Performance par régime : %s", list(regimes.keys()))
        results["regimes"] = regimes

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, default=str, indent=2), encoding="utf-8")
        LOGGER.info("Résultats écrits dans %s", args.output_json)


if __name__ == "__main__":  # pragma: no cover
    main()
