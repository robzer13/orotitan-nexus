import pandas as pd

from orotitan_nexus import cli_custom_garp
from orotitan_nexus.garp_rules import GARP_BUCKET_COLUMN, GARP_FLAG_COLUMN, GARP_SCORE_COLUMN


def _sample_df(strict_flags):
    rows = []
    for idx, flag in enumerate(strict_flags):
        rows.append(
            {
                "ticker": f"T{idx}.PA",
                "pe_ttm": 20.0,
                "pe_fwd": 12.0,
                "debt_to_equity": 0.30,
                "eps_cagr": 0.20,
                "peg": 0.9,
                "market_cap": 10e9 + idx,
                "score_v1_1": 4,
                "category_v1_1": "WATCHLIST_V1_1",
                "data_complete_v1_1": True,
                GARP_FLAG_COLUMN: flag,
                GARP_SCORE_COLUMN: 80.0 if flag else 40.0,
                GARP_BUCKET_COLUMN: "STRONG_GARP" if flag else "REJECT_GARP",
            }
        )
    return pd.DataFrame(rows)


def _summary(df, run_id):
    return {
        "header": "=== Custom ===",
        "profile": "balanced",
        "universe": "CUSTOM",
        "run_id": run_id,
        "total": len(df),
        "data_complete": int(df["data_complete_v1_1"].sum()),
        "strict_count": int(df[GARP_FLAG_COLUMN].sum()),
        "categories": {
            "ELITE_V1_1": 0,
            "WATCHLIST_V1_1": len(df),
            "REJECT_V1_1": 0,
            "DATA_MISSING": 0,
            "EXCLUDED_UNIVERSE": 0,
        },
        "bucket_counts": {"STRONG_GARP": 1, "REJECT_GARP": 1},
        "top": [],
    }


def test_cli_custom_garp_with_inline_tickers(tmp_path, monkeypatch):
    sample = _sample_df([True, False])

    def _fake_run(**kwargs):  # pragma: no cover - deterministic stub
        run_id = kwargs.get("run_id", "run")
        radar = sample[sample[GARP_FLAG_COLUMN]].copy()
        return sample.copy(), radar, _summary(sample, run_id)

    monkeypatch.setattr(cli_custom_garp, "run_custom_garp", _fake_run)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("profile:\n  name: balanced\n", encoding="utf-8")

    full_path = tmp_path / "full.csv"
    radar_path = tmp_path / "radar.csv"

    cli_custom_garp.main(
        [
            "--config",
            str(cfg),
            "--tickers",
            "T0.PA,T1.PA",
            "--output-full",
            str(full_path),
            "--output-radar",
            str(radar_path),
        ]
    )

    full_df = pd.read_csv(full_path)
    assert len(full_df) == 2
    radar_df = pd.read_csv(radar_path)
    assert len(radar_df) == 1
    if not radar_df.empty:
        assert radar_df[GARP_FLAG_COLUMN].all()


def test_cli_custom_garp_with_csv_input(tmp_path, monkeypatch):
    sample = _sample_df([False, False])

    monkeypatch.setattr(
        cli_custom_garp,
        "run_custom_garp",
        lambda **kwargs: (sample.copy(), sample[sample[GARP_FLAG_COLUMN]].copy(), _summary(sample, "run")),
    )

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("profile:\n  name: balanced\n", encoding="utf-8")

    csv_path = tmp_path / "tickers.csv"
    csv_path.write_text("symbol\nT0.PA\nT1.PA\n", encoding="utf-8")

    full_path = tmp_path / "full.csv"
    radar_path = tmp_path / "radar.csv"

    cli_custom_garp.main(
        [
            "--config",
            str(cfg),
            "--tickers-csv",
            str(csv_path),
            "--tickers-column",
            "symbol",
            "--output-full",
            str(full_path),
            "--output-radar",
            str(radar_path),
        ]
    )

    radar_df = pd.read_csv(radar_path)
    assert radar_df.empty
