"""CLI wrapper for the Nexus Playbook engine."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import api

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OroTitan Nexus Playbook")
    parser.add_argument("--config", help="Path to YAML config", default=None)
    parser.add_argument("--profile", help="Profile name", default=None)
    parser.add_argument("--universe", help="Universe name override", default=None)
    parser.add_argument("--portfolio", help="Optional portfolio CSV", default=None)
    parser.add_argument("--output-full", help="Path to write full CSV", default=None)
    parser.add_argument("--output-playbook-json", help="Path to write Playbook JSON", default=None)
    parser.add_argument("--run-id", help="Optional run identifier", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    df, summary = api.run_playbook(
        config_path=args.config,
        profile_name=args.profile,
        universe_override=None,
        portfolio_path=args.portfolio,
        run_id=args.run_id,
    )

    if args.output_full:
        Path(args.output_full).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_full, index=False)
        LOGGER.info("Full playbook CSV written to %s", args.output_full)

    if args.output_playbook_json:
        Path(args.output_playbook_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_playbook_json, "w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2)
        LOGGER.info("Playbook JSON written to %s", args.output_playbook_json)

    print("=== Nexus Playbook ===")
    print(f"Profile     : {summary.profile_name}")
    print(f"Universe    : {summary.universe_name}")
    print(f"Run ID      : {summary.run_id or ''}")
    print(f"Universe sz : {summary.total_universe}")
    print("Decisions   :", summary.counts_by_action)


if __name__ == "__main__":  # pragma: no cover
    main()

