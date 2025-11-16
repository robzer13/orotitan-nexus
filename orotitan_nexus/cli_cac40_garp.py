"""CLI for the CAC40 GARP radar."""
from __future__ import annotations
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .api import run_cac40_garp
from .cli import configure_logging
from .config import ConfigError
from .history import compute_garp_diff, load_snapshot, save_snapshot
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
    parser.add_argument("--history-path", type=str, default=None, help="Optional CSV file to append run summaries")
    parser.add_argument("--run-id", type=str, default=None, help="Label for this radar run")
    parser.add_argument("--notes", type=str, default=None, help="Optional notes stored in history")
    parser.add_argument("--snapshots-dir", type=str, default=None, help="Folder for per-run ticker snapshots")
    parser.add_argument("--compare-with-run-id", type=str, default=None, help="Previous run_id to diff against")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    try:
        df, radar_df, summary = run_cac40_garp(
            config_path=args.config,
            profile_name=args.profile,
            history_path=args.history_path,
            run_id=run_id,
            notes=args.notes,
        )
    except ConfigError as exc:  # pragma: no cover - defensive CLI guard
        LOGGER.error("Configuration invalide: %s", exc)
        raise SystemExit(1) from exc

    if df.empty:
        LOGGER.warning("Aucune donnée récupérée pour l'univers CAC40")
        return

    write_csv(df, args.output_full)
    LOGGER.info("CSV complet enregistré dans %s", args.output_full)

    write_csv(radar_df, args.output_radar)
    LOGGER.info("CSV radar GARP enregistré dans %s", args.output_radar)

    LOGGER.info(summary["header"])
    LOGGER.info("Profil             : %s", summary["profile"])
    LOGGER.info("Univers            : %s", summary["universe"])
    LOGGER.info("Run ID             : %s", summary.get("run_id"))
    LOGGER.info("Total valeurs      : %d", summary["total"])
    LOGGER.info("Data-complete V1.1 : %d", summary["data_complete"])
    LOGGER.info("Strict GARP (5/5)  : %d", summary["strict_count"])
    for category, count in summary["categories"].items():
        LOGGER.info("  %s: %d", category, count)

    if summary.get("bucket_counts"):
        LOGGER.info("Buckets GARP:")
        for bucket, count in summary["bucket_counts"].items():
            LOGGER.info("  %s: %d", bucket, count)

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

    if args.snapshots_dir:
        snapshot_dir = Path(args.snapshots_dir)
        try:
            save_snapshot(snapshot_dir, summary["run_id"], df)
            LOGGER.info("Snapshot enregistré dans %s", snapshot_dir)
        except Exception as exc:  # pragma: no cover - defensive log
            LOGGER.warning("Impossible d'enregistrer le snapshot: %s", exc)
        if args.compare_with_run_id:
            try:
                previous = load_snapshot(snapshot_dir, args.compare_with_run_id)
            except FileNotFoundError:
                LOGGER.warning(
                    "Snapshot pour run_id=%s introuvable dans %s",
                    args.compare_with_run_id,
                    snapshot_dir,
                )
            else:
                diff = compute_garp_diff(previous, df)
                _log_drift(diff)


def _log_drift(diff: dict) -> None:
    LOGGER.info("=== GARP Drift Report ===")
    LOGGER.info("Nouveaux strict pass : %d", len(diff["new_strict_pass"]))
    for ticker in diff["new_strict_pass"]:
        LOGGER.info("  + %s", ticker)
    LOGGER.info("Sorties strict pass  : %d", len(diff["lost_strict_pass"]))
    for ticker in diff["lost_strict_pass"]:
        LOGGER.info("  - %s", ticker)
    LOGGER.info("Upgrades de bucket   : %d", len(diff["bucket_upgrades"]))
    for ticker, old, new in diff["bucket_upgrades"]:
        LOGGER.info("  ^ %s: %s -> %s", ticker, old, new)
    LOGGER.info("Downgrades de bucket : %d", len(diff["bucket_downgrades"]))
    for ticker, old, new in diff["bucket_downgrades"]:
        LOGGER.info("  v %s: %s -> %s", ticker, old, new)
if __name__ == "__main__":  # pragma: no cover
    main()