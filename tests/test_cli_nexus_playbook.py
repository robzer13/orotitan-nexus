import json
from pathlib import Path

import pandas as pd

from orotitan_nexus import cli_nexus_playbook
from orotitan_nexus.playbook import PlaybookSummary, TickerDecision


def test_cli_nexus_playbook(tmp_path, monkeypatch, capsys):
    df = pd.DataFrame([{"ticker": "AAA.PA", "nexus_core_score": 80.0}])
    decision = TickerDecision(
        ticker="AAA.PA",
        action="BUY",
        rationale=["HIGH_CORE"],
        core_score=80.0,
        v2_score=None,
        exceptionality=None,
        upside_pct=None,
        momentum_score=None,
        risk_score=None,
        etf_nexus_score=None,
        is_owned=False,
        is_etf=False,
    )
    summary = PlaybookSummary(
        profile_name="test",
        universe_name="TEST",
        run_id="run",
        date=None,
        portfolio_hhi=None,
        portfolio_top5_weight=None,
        portfolio_hhi_zone=None,
        portfolio_top5_zone=None,
        total_universe=1,
        total_strict_garp=0,
        total_core_enabled=1,
        total_owned=0,
        decisions=[decision],
        counts_by_action={"BUY": 1},
        top_by_core=["AAA.PA"],
        top_by_v2=[],
        top_by_upside=[],
        top_by_etf=[],
    )

    def _fake_run_playbook(**kwargs):  # pragma: no cover - deterministic stub
        return df.copy(), summary

    monkeypatch.setattr(cli_nexus_playbook.api, "run_playbook", _fake_run_playbook)

    out_csv = tmp_path / "playbook.csv"
    out_json = tmp_path / "playbook.json"
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setattr(
        "sys.argv",
        [
            "cli_nexus_playbook",
            "--config",
            "configs/cac40.yaml",
            "--output-full",
            str(out_csv),
            "--output-playbook-json",
            str(out_json),
        ],
    )

    cli_nexus_playbook.main()
    captured = capsys.readouterr().out
    assert "Nexus Playbook" in captured
    assert out_csv.exists()
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert data["counts_by_action"]["BUY"] == 1

