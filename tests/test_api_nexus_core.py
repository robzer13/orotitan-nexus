import pandas as pd

from orotitan_nexus import api
from orotitan_nexus.config import (
    FilterSettings,
    NexusCoreSettings,
    ProfileSettingsV2,
    UniverseSettings,
    WeightSettings,
)
from orotitan_nexus.garp_rules import GARP_FLAG_COLUMN, GARP_SCORE_COLUMN, GARP_BUCKET_COLUMN


def _fake_settings_core():
    filters = FilterSettings()
    weights = WeightSettings()
    universe = UniverseSettings(name="TEST", tickers=["AAA", "BBB"])
    profile = ProfileSettingsV2(name="balanced", nexus_core=NexusCoreSettings(enabled=True))
    return filters, weights, universe, profile


def test_run_custom_garp_with_nexus_core(monkeypatch):
    sample = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "score_v1_1": 5,
                "category_v1_1": "ELITE_V1_1",
                "market_cap": 30e9,
                "data_complete_v1_1": True,
                GARP_FLAG_COLUMN: True,
                GARP_SCORE_COLUMN: 88.0,
                GARP_BUCKET_COLUMN: "ELITE_GARP",
                "roe_pct": 18.0,
                "operating_margin_pct": 12.0,
                "fcf_yield_pct": 6.0,
            },
            {
                "ticker": "BBB",
                "score_v1_1": 3,
                "category_v1_1": "WATCHLIST_V1_1",
                "market_cap": 12e9,
                "data_complete_v1_1": True,
                GARP_FLAG_COLUMN: False,
                GARP_SCORE_COLUMN: 55.0,
                GARP_BUCKET_COLUMN: "BORDERLINE_GARP",
                "roe_pct": 4.0,
                "operating_margin_pct": 3.0,
                "fcf_yield_pct": 1.0,
            },
        ]
    )

    monkeypatch.setattr(api, "load_settings", lambda *args, **kwargs: _fake_settings_core())
    monkeypatch.setattr(api, "run_universe", lambda *args, **kwargs: sample.copy())

    full_df, radar_df, summary = api.run_custom_garp(
        config_path=None,
        profile_name="balanced",
        tickers=["AAA", "BBB"],
    )

    assert "nexus_core_score" in full_df
    assert "nexus_core_q" in full_df
    # v1 still intact
    assert GARP_FLAG_COLUMN in full_df
    assert summary["strict_count"] == 1
    assert radar_df[GARP_FLAG_COLUMN].all()

