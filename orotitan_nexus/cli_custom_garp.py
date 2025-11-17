"""CLI for the custom-universe GARP radar."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .api import run_custom_garp, explain_single_ticker
from .cli import configure_logging
from .config import ConfigError
from .history import compute_garp_diff, load_snapshot, save_snapshot
from .reporting import write_csv

LOGGER = logging.getLogger(__name__)
DEFAULT_FULL_OUTPUT = "custom_garp_full.csv"
DEFAULT_RADAR_OUTPUT = None
PROFILE_CHOICES = ("defensive", "balanced", "offensive")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom universe GARP radar")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument(
        "--profile",
        type=str,
        choices=PROFILE_CHOICES,
        default=None,
        help="Optional Nexus profile overrides",
    )
    parser.add_argument("--universe-name", type=str, default="CUSTOM", help="Label for the custom universe")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers",
    )
    parser.add_argument(
        "--tickers-csv",
        type=str,
        default=None,
        help="CSV file containing a column with tickers",
    )
    parser.add_argument(
        "--tickers-column",
        type=str,
        default="ticker",
        help="Column name to read from --tickers-csv",
    )
    parser.add_argument(
        "--output-full",
        type=str,
        default=DEFAULT_FULL_OUTPUT,
        help="Full CSV output path",
    )
    parser.add_argument(
        "--output-radar",
        type=str,
        default=DEFAULT_RADAR_OUTPUT,
        help="Optional radar CSV output path",
    )
    parser.add_argument("--history-path", type=str, default=None, help="Optional CSV for run history")
    parser.add_argument("--run-id", type=str, default=None, help="Label for this radar run")
    parser.add_argument("--notes", type=str, default=None, help="Optional notes stored alongside history")
    parser.add_argument("--snapshots-dir", type=str, default=None, help="Folder for per-run snapshots")
    parser.add_argument("--compare-with-run-id", type=str, default=None, help="Previous run to diff against")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--explain", type=str, default=None, help="Ticker to explain after the run")
    return parser.parse_args(argv)


def _load_custom_tickers(args: argparse.Namespace) -> List[str]:
    tickers: List[str] = []
    if args.tickers:
        tickers.extend(token.strip() for token in args.tickers.split(","))
    if args.tickers_csv:
        try:
            csv_df = pd.read_csv(args.tickers_csv)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.error("Impossible de lire %s: %s", args.tickers_csv, exc)
            raise SystemExit(1) from exc
        column = args.tickers_column or "ticker"
        if column not in csv_df.columns:
            LOGGER.error("La colonne %s est absente de %s", column, args.tickers_csv)
            raise SystemExit(1)
        tickers.extend(csv_df[column].astype(str).tolist())
    cleaned = []
    seen = set()
    for ticker in tickers:
        ticker = ticker.strip()
        if not ticker:
            continue
        if ticker not in seen:
            seen.add(ticker)
            cleaned.append(ticker)
    if not cleaned:
        LOGGER.error("Aucun ticker fourni via --tickers ou --tickers-csv")
        raise SystemExit(1)
    return cleaned


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    tickers = _load_custom_tickers(args)

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    try:
        df, radar_df, summary = run_custom_garp(
            config_path=args.config,
            profile_name=args.profile,
            tickers=tickers,
            universe_name=args.universe_name,
            history_path=args.history_path,
            run_id=run_id,
            notes=args.notes,
        )
    except ConfigError as exc:  # pragma: no cover - defensive guard
        LOGGER.error("Configuration invalide: %s", exc)
        raise SystemExit(1) from exc

    if df.empty:
        LOGGER.warning("Aucune donnée récupérée pour l'univers custom")
        return

    write_csv(df, args.output_full)
    LOGGER.info("CSV complet enregistré dans %s", args.output_full)

    if args.output_radar:
        write_csv(radar_df, args.output_radar)
        LOGGER.info("CSV radar GARP enregistré dans %s", args.output_radar)

    if args.explain:
        explanation = explain_single_ticker(df, args.explain, summary.get("profile_object"))
        if explanation:
            LOGGER.info("=== Explications v2 pour %s ===", args.explain)
            LOGGER.info("Score v2 : %s", explanation.get("nexus_v2_score"))
            for pillar, contrib in (explanation.get("contributions") or {}).items():
                LOGGER.info("  %s: %s", pillar, contrib)
            print(f"Explications v2 pour {args.explain}: {explanation}")
        else:
            LOGGER.info("Aucune explication v2 disponible (profil sans v2 ou ticker absent).")

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
        LOGGER.info("Aucune valeur de l'univers custom ne valide les 5 règles GARP.")
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
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Impossible d'enregistrer le snapshot custom: %s", exc)
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
