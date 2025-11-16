import pandas as pd

from orotitan_nexus import cli_custom_garp
from orotitan_nexus.config import ProfileSettingsV2, NexusV2Settings
from orotitan_nexus.garp_rules import GARP_FLAG_COLUMN, GARP_SCORE_COLUMN, GARP_BUCKET_COLUMN


def _sample(profile):
    df = pd.DataFrame(
        {
            "ticker": ["T0"],
            "nexus_v2_score": [80],
            "nexus_v2_bucket": ["V2_STRONG"],
            "garp_score": [80],
            "quality_score": [70],
            "momentum_score": [60],
            "risk_score": [50],
            "macro_score": [40],
            "behavioral_score": [30],
            "data_complete_v1_1": [True],
            GARP_FLAG_COLUMN: [True],
            GARP_SCORE_COLUMN: [80],
            GARP_BUCKET_COLUMN: ["STRONG_GARP"],
        }
    )
    return df


def test_cli_custom_garp_explain(monkeypatch, tmp_path, capsys):
    profile = ProfileSettingsV2(name="balanced", v2=NexusV2Settings(enabled=True))
    df = _sample(profile)

    def _fake_run(**kwargs):
        radar = df[df[GARP_FLAG_COLUMN]].copy()
        summary = {
            "header": "=== Custom ===",
            "profile": profile.name,
            "profile_object": profile,
            "universe": "CUSTOM",
            "run_id": "run",
            "total": 1,
            "data_complete": 1,
            "strict_count": 1,
            "categories": {},
            "bucket_counts": {},
            "top": [],
        }
        return df.copy(), radar, summary

    monkeypatch.setattr(cli_custom_garp, "run_custom_garp", _fake_run)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("profile:\n  name: balanced\n", encoding="utf-8")
    full_path = tmp_path / "full.csv"

    cli_custom_garp.main(
        [
            "--config",
            str(cfg),
            "--tickers",
            "T0",
            "--output-full",
            str(full_path),
            "--explain",
            "T0",
        ]
    )
    captured = capsys.readouterr()
    # ensure no crash and explanation message present
    assert "Explications" in captured.out or "Explications" in captured.err
