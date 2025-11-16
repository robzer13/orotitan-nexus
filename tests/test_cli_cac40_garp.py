import pandas as pd

from orotitan_nexus import cli_cac40_garp
from orotitan_nexus.garp_rules import GARP_BUCKET_COLUMN, GARP_FLAG_COLUMN, GARP_SCORE_COLUMN
from orotitan_nexus.history import save_snapshot


def _sample_df():
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA.PA",
                "pe_ttm": 20.0,
                "pe_fwd": 12.0,
                "debt_to_equity": 0.30,
                "eps_cagr": 0.25,
                "peg": 0.9,
                "market_cap": 25e9,
                "data_complete_v1_1": True,
                "score_v1_1": 4,
                "category_v1_1": "WATCHLIST_V1_1",
                "universe_exclusion_reason": "",
            },
            {
                "ticker": "BBB.PA",
                "pe_ttm": 30.0,
                "pe_fwd": 18.0,
                "debt_to_equity": 0.40,
                "eps_cagr": 0.10,
                "peg": 1.5,
                "market_cap": 10e9,
                "data_complete_v1_1": True,
                "score_v1_1": 1,
                "category_v1_1": "REJECT_V1_1",
                "universe_exclusion_reason": "",
            },
        ]
    )
    df[GARP_FLAG_COLUMN] = [True, False]
    df[GARP_SCORE_COLUMN] = [90.0, 30.0]
    df[GARP_BUCKET_COLUMN] = ["ELITE_GARP", "REJECT_GARP"]
    return df


def _summary(sample, run_id):
    return {
        "header": "=== Test ===",
        "profile": "balanced",
        "universe": "TEST",
        "run_id": run_id,
        "total": len(sample),
        "data_complete": int(sample["data_complete_v1_1"].sum()),
        "strict_count": int(sample[GARP_FLAG_COLUMN].sum()),
        "categories": {
            "ELITE_V1_1": 0,
            "WATCHLIST_V1_1": 1,
            "REJECT_V1_1": 1,
            "DATA_MISSING": 0,
            "EXCLUDED_UNIVERSE": 0,
        },
        "bucket_counts": {"ELITE_GARP": 1, "REJECT_GARP": 1},
        "top": [
            {
                "ticker": "AAA.PA",
                "score": 4,
                "category": "WATCHLIST_V1_1",
                "reason": "OK",
            }
        ],
    }


def test_cli_cac40_garp_creates_outputs(tmp_path, monkeypatch):
    sample = _sample_df()

    def _fake_run(**kwargs):  # pragma: no cover - deterministic stub
        run_id = kwargs.get("run_id", "run")
        radar = sample[sample[GARP_FLAG_COLUMN]].copy()
        return sample.copy(), radar, _summary(sample, run_id)

    monkeypatch.setattr(cli_cac40_garp, "run_cac40_garp", _fake_run)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("profile:\n  name: balanced\n", encoding="utf-8")

    full_path = tmp_path / "full.csv"
    radar_path = tmp_path / "radar.csv"
    cli_cac40_garp.main(
        [
            "--config",
            str(cfg),
            "--output-full",
            str(full_path),
            "--output-radar",
            str(radar_path),
        ]
    )

    full_df = pd.read_csv(full_path)
    assert GARP_FLAG_COLUMN in full_df.columns
    radar_df = pd.read_csv(radar_path)
    if not radar_df.empty:
        assert radar_df[GARP_FLAG_COLUMN].all()
        assert (radar_df[GARP_BUCKET_COLUMN] == "ELITE_GARP").all()


def test_cli_cac40_garp_snapshots_and_diff(tmp_path, monkeypatch):
    sample = _sample_df()

    def _fake_run(**kwargs):  # pragma: no cover
        run_id = kwargs.get("run_id", "run")
        radar = sample[sample[GARP_FLAG_COLUMN]].copy()
        return sample.copy(), radar, _summary(sample, run_id)

    monkeypatch.setattr(cli_cac40_garp, "run_cac40_garp", _fake_run)

    diff_called = {}

    def _fake_diff(prev, current):  # pragma: no cover
        diff_called["called"] = True
        return {
            "new_strict_pass": ["AAA.PA"],
            "lost_strict_pass": [],
            "bucket_upgrades": [],
            "bucket_downgrades": [],
        }

    monkeypatch.setattr(cli_cac40_garp, "compute_garp_diff", _fake_diff)

    snaps = tmp_path / "snaps"
    save_snapshot(snaps, "prev-run", sample)

    full_path = tmp_path / "full.csv"
    radar_path = tmp_path / "radar.csv"

    cli_cac40_garp.main(
        [
            "--config",
            "configs/cac40.yaml",
            "--output-full",
            str(full_path),
            "--output-radar",
            str(radar_path),
            "--snapshots-dir",
            str(snaps),
            "--run-id",
            "new-run",
            "--compare-with-run-id",
            "prev-run",
        ]
    )

    new_snapshot = snaps / "garp_new-run.csv"
    assert new_snapshot.exists()
    assert diff_called.get("called") is True
