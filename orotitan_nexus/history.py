"""Simple file-based history helpers for GARP radar runs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .garp_rules import GARP_BUCKET_COLUMN, GARP_FLAG_COLUMN, GARP_SCORE_COLUMN

HISTORY_COLUMNS = [
    "timestamp",
    "universe_name",
    "profile_name",
    "run_id",
    "total",
    "data_complete_v1_1",
    "strict_garp_count",
    "elite_count",
    "strong_count",
    "borderline_count",
    "reject_count",
    "notes",
]

SNAPSHOT_PREFIX = "garp_"
SNAPSHOT_SUFFIX = ".csv"


@dataclass
class GarpRunRecord:
    """Aggregate information for one strict GARP radar run."""

    timestamp: datetime
    universe_name: str
    profile_name: str
    run_id: str
    total: int
    data_complete_v1_1: int
    strict_garp_count: int
    elite_count: int
    strong_count: int
    borderline_count: int
    reject_count: int
    notes: Optional[str] = None


def append_history_record(path: Path, record: GarpRunRecord) -> None:
    """Append ``record`` as one line of CSV history."""

    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": record.timestamp.isoformat(),
        "universe_name": record.universe_name,
        "profile_name": record.profile_name,
        "run_id": record.run_id,
        "total": int(record.total),
        "data_complete_v1_1": int(record.data_complete_v1_1),
        "strict_garp_count": int(record.strict_garp_count),
        "elite_count": int(record.elite_count),
        "strong_count": int(record.strong_count),
        "borderline_count": int(record.borderline_count),
        "reject_count": int(record.reject_count),
        "notes": record.notes or "",
    }
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(path, mode="w", header=True, index=False, encoding="utf-8")


def load_history(path: Path) -> pd.DataFrame:
    """Return the history CSV as a dataframe (empty if file absent)."""

    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.read_csv(path)


def _sanitize_run_id(run_id: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in run_id]
    sanitized = "".join(cleaned).strip("-_")
    return sanitized or "run"


def save_snapshot(directory: Path, run_id: str, df: pd.DataFrame) -> Path:
    """Persist the per-ticker dataframe for ``run_id`` inside ``directory``."""

    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{SNAPSHOT_PREFIX}{_sanitize_run_id(run_id)}{SNAPSHOT_SUFFIX}"
    path = directory / filename
    df.to_csv(path, index=False)
    return path


def load_snapshot(directory: Path, run_id: str) -> pd.DataFrame:
    """Load the previously saved snapshot for ``run_id``."""

    filename = f"{SNAPSHOT_PREFIX}{_sanitize_run_id(run_id)}{SNAPSHOT_SUFFIX}"
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


_BUCKET_ORDER = {
    "REJECT_GARP": 0,
    "BORDERLINE_GARP": 1,
    "STRONG_GARP": 2,
    "ELITE_GARP": 3,
}


def _bucket_rank(bucket: str) -> int:
    return _BUCKET_ORDER.get(bucket, 0)


def compute_garp_diff(previous_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, List]:
    """Compare two snapshots and describe strict-pass & bucket drifts."""

    prev = previous_df.set_index("ticker") if not previous_df.empty else pd.DataFrame()
    curr = current_df.set_index("ticker") if not current_df.empty else pd.DataFrame()

    def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
        if frame.empty or column not in frame.columns:
            return pd.Series(dtype=bool)
        return frame[column].fillna(False)

    def _bucket(frame: pd.DataFrame) -> pd.Series:
        if frame.empty or GARP_BUCKET_COLUMN not in frame.columns:
            return pd.Series(dtype=str)
        return frame[GARP_BUCKET_COLUMN].fillna("REJECT_GARP")

    prev_flags = _flag(prev, GARP_FLAG_COLUMN)
    curr_flags = _flag(curr, GARP_FLAG_COLUMN)
    prev_buckets = _bucket(prev)
    curr_buckets = _bucket(curr)

    new_strict = []
    for ticker, is_strict in curr_flags.items():
        if is_strict and not bool(prev_flags.get(ticker, False)):
            new_strict.append(ticker)

    lost_strict = []
    for ticker, was_strict in prev_flags.items():
        if was_strict and not bool(curr_flags.get(ticker, False)):
            lost_strict.append(ticker)

    bucket_upgrades = []
    bucket_downgrades = []
    shared = set(prev_buckets.index) & set(curr_buckets.index)
    for ticker in shared:
        before = prev_buckets.get(ticker, "REJECT_GARP")
        after = curr_buckets.get(ticker, "REJECT_GARP")
        if before == after:
            continue
        if _bucket_rank(after) > _bucket_rank(before):
            bucket_upgrades.append((ticker, before, after))
        elif _bucket_rank(after) < _bucket_rank(before):
            bucket_downgrades.append((ticker, before, after))

    return {
        "new_strict_pass": sorted(new_strict),
        "lost_strict_pass": sorted(lost_strict),
        "bucket_upgrades": bucket_upgrades,
        "bucket_downgrades": bucket_downgrades,
    }
