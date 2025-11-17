import pandas as pd
from datetime import datetime

from orotitan_nexus.history import (
    GarpRunRecord,
    append_history_record,
    compute_garp_diff,
    load_history,
    load_snapshot,
    save_snapshot,
)
from orotitan_nexus.garp_rules import (
    GARP_BUCKET_COLUMN,
    GARP_FLAG_COLUMN,
    GARP_SCORE_COLUMN,
)


def test_append_and_load_history_roundtrip(tmp_path):
    history_path = tmp_path / "history.csv"
    record = GarpRunRecord(
        timestamp=datetime(2024, 1, 1, 9, 0),
        universe_name="TEST",
        profile_name="balanced",
        run_id="run-1",
        total=10,
        data_complete_v1_1=8,
        strict_garp_count=2,
        elite_count=1,
        strong_count=1,
        borderline_count=3,
        reject_count=5,
        notes="baseline",
    )
    append_history_record(history_path, record)
    loaded = load_history(history_path)
    assert len(loaded) == 1
    row = loaded.iloc[0]
    assert row["run_id"] == "run-1"
    assert row["strict_garp_count"] == 2
    assert row["notes"] == "baseline"


def test_save_and_load_snapshot_roundtrip(tmp_path):
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA.PA",
                GARP_FLAG_COLUMN: True,
                GARP_SCORE_COLUMN: 90.0,
                GARP_BUCKET_COLUMN: "ELITE_GARP",
            },
            {
                "ticker": "BBB.PA",
                GARP_FLAG_COLUMN: False,
                GARP_SCORE_COLUMN: 40.0,
                GARP_BUCKET_COLUMN: "REJECT_GARP",
            },
        ]
    )
    path = save_snapshot(tmp_path, "test-run", df)
    assert path.exists()
    loaded = load_snapshot(tmp_path, "test-run")
    assert list(loaded["ticker"]) == ["AAA.PA", "BBB.PA"]


def test_compute_garp_diff_basic():
    prev = pd.DataFrame(
        [
            {
                "ticker": "AAA.PA",
                GARP_FLAG_COLUMN: True,
                GARP_BUCKET_COLUMN: "STRONG_GARP",
            },
            {
                "ticker": "BBB.PA",
                GARP_FLAG_COLUMN: True,
                GARP_BUCKET_COLUMN: "REJECT_GARP",
            },
        ]
    )
    current = pd.DataFrame(
        [
            {
                "ticker": "AAA.PA",
                GARP_FLAG_COLUMN: True,
                GARP_BUCKET_COLUMN: "ELITE_GARP",
            },
            {
                "ticker": "CCC.PA",
                GARP_FLAG_COLUMN: True,
                GARP_BUCKET_COLUMN: "STRONG_GARP",
            },
        ]
    )
    diff = compute_garp_diff(prev, current)
    assert diff["new_strict_pass"] == ["CCC.PA"]
    assert diff["lost_strict_pass"] == ["BBB.PA"]
    assert ("AAA.PA", "STRONG_GARP", "ELITE_GARP") in diff["bucket_upgrades"]
    assert diff["bucket_downgrades"] == []
