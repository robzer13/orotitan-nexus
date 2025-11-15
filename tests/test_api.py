import pandas as pd

from orotitan_nexus import api
from orotitan_nexus.config import FilterSettings, WeightSettings, UniverseSettings, ProfileSettings
from orotitan_nexus.garp_rules import GARP_FLAG_COLUMN


def _fake_settings():
    filters = FilterSettings()
    weights = WeightSettings()
    universe = UniverseSettings(name="TEST", tickers=["AAA.PA", "BBB.PA"])
    profile = ProfileSettings(name="balanced")
    return filters, weights, universe, profile


def test_run_screen_returns_summary(monkeypatch):
    sample = pd.DataFrame(
        [
            {
                "ticker": "AAA.PA",
                "score_v1_1": 4,
                "nexus_score": 70.0,
                "garp_score": 65.0,
                "category_v1_1": "WATCHLIST_V1_1",
                "strict_pass": True,
                "data_complete_v1_1": True,
            }
        ]
    )

    monkeypatch.setattr(api, "load_settings", lambda *args, **kwargs: _fake_settings())
    monkeypatch.setattr(api, "run_universe", lambda *args, **kwargs: sample.copy())

    df, summary = api.run_screen(config_path=None, profile_name="balanced")
    assert not df.empty
    assert summary["total"] == 1
    assert summary["strict_pass"] == 1


def test_run_cac40_garp_returns_radar(monkeypatch):
    sample = pd.DataFrame(
        [
            {
                "ticker": "AAA.PA",
                "score_v1_1": 5,
                "category_v1_1": "ELITE_V1_1",
                "market_cap": 20e9,
                "data_complete_v1_1": True,
                GARP_FLAG_COLUMN: True,
            },
            {
                "ticker": "BBB.PA",
                "score_v1_1": 2,
                "category_v1_1": "REJECT_V1_1",
                "market_cap": 10e9,
                "data_complete_v1_1": True,
                GARP_FLAG_COLUMN: False,
            },
        ]
    )

    monkeypatch.setattr(api, "load_settings", lambda *args, **kwargs: _fake_settings())
    monkeypatch.setattr(api, "run_universe", lambda *args, **kwargs: sample.copy())

    full_df, radar_df, summary = api.run_cac40_garp(config_path=None, profile_name="balanced")
    assert len(full_df) == 2
    assert len(radar_df) == 1
    assert radar_df[GARP_FLAG_COLUMN].all()
    assert summary["strict_count"] == 1
