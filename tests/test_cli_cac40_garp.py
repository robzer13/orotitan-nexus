import pandas as pd

from orotitan_nexus import cli_cac40_garp
from orotitan_nexus.garp_rules import GARP_FLAG_COLUMN


def test_cli_cac40_garp_creates_outputs(tmp_path, monkeypatch):
    sample = pd.DataFrame(
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

    sample[GARP_FLAG_COLUMN] = [True, False]

    def _fake_run_universe(*args, **kwargs):  # pragma: no cover - deterministic stub
        return sample.copy()

    monkeypatch.setattr(cli_cac40_garp, "run_universe", _fake_run_universe)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "universe:\n  tickers:\n    - 'AAA.PA'\n    - 'BBB.PA'\nprofile:\n  name: balanced\n",
        encoding="utf-8",
    )

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