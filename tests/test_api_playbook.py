import pandas as pd

from orotitan_nexus import api
from orotitan_nexus.config import FilterSettings, WeightSettings, UniverseSettings, ProfileSettingsV2
from orotitan_nexus.playbook import PlaybookSummary


def _fake_settings():
    filters = FilterSettings()
    weights = WeightSettings()
    universe = UniverseSettings(name="TEST", tickers=["AAA.PA", "BBB.PA"])
    profile = ProfileSettingsV2(name="balanced")
    profile.playbook.enabled = True
    profile.playbook.thresholds.min_core_score_buy = 50.0
    return filters, weights, universe, profile


def test_run_playbook(monkeypatch):
    sample = pd.DataFrame(
        [
            {
                "ticker": "AAA.PA",
                "score_v1_1": 4,
                "nexus_core_score": 80.0,
                "nexus_exceptionality": 8.0,
                "nexus_v2_score": 78.0,
                "upside_pct": 20.0,
                "momentum_score": 60.0,
                "risk_score": 60.0,
                "data_complete_v1_1": True,
                "strict_pass_garp": True,
            },
            {
                "ticker": "BBB.PA",
                "score_v1_1": 3,
                "nexus_core_score": 40.0,
                "nexus_v2_score": 45.0,
                "upside_pct": -5.0,
                "momentum_score": 30.0,
                "risk_score": 30.0,
                "data_complete_v1_1": True,
                "strict_pass_garp": False,
            },
        ]
    )

    monkeypatch.setattr(api, "load_settings", lambda *args, **kwargs: _fake_settings())
    monkeypatch.setattr(api, "run_universe", lambda *args, **kwargs: sample.copy())

    df, playbook_summary = api.run_playbook(config_path=None, profile_name="balanced")
    assert isinstance(playbook_summary, PlaybookSummary)
    assert not df.empty
    assert playbook_summary.total_universe == 2
    assert playbook_summary.decisions

