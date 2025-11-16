import pandas as pd

import pandas as pd

import pandas as pd

from orotitan_nexus import api
from orotitan_nexus.config import FilterSettings, WeightSettings, UniverseSettings, ProfileSettings
from orotitan_nexus.garp_rules import GARP_BUCKET_COLUMN, GARP_FLAG_COLUMN, GARP_SCORE_COLUMN


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
                GARP_SCORE_COLUMN: 90.0,
                GARP_BUCKET_COLUMN: "ELITE_GARP",
            },
            {
                "ticker": "BBB.PA",
                "score_v1_1": 2,
                "category_v1_1": "REJECT_V1_1",
                "market_cap": 10e9,
                "data_complete_v1_1": True,
                GARP_FLAG_COLUMN: False,
                GARP_SCORE_COLUMN: 40.0,
                GARP_BUCKET_COLUMN: "REJECT_GARP",
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


def test_run_custom_garp(monkeypatch):
    sample = pd.DataFrame(
        [
            {
                "ticker": "AAA.PA",
                "score_v1_1": 5,
                "category_v1_1": "ELITE_V1_1",
                "market_cap": 30e9,
                "data_complete_v1_1": True,
                GARP_FLAG_COLUMN: True,
                GARP_SCORE_COLUMN: 88.0,
                GARP_BUCKET_COLUMN: "ELITE_GARP",
            },
            {
                "ticker": "BBB.PA",
                "score_v1_1": 3,
                "category_v1_1": "WATCHLIST_V1_1",
                "market_cap": 12e9,
                "data_complete_v1_1": True,
                GARP_FLAG_COLUMN: False,
                GARP_SCORE_COLUMN: 55.0,
                GARP_BUCKET_COLUMN: "BORDERLINE_GARP",
            },
        ]
    )

    monkeypatch.setattr(api, "load_settings", lambda *args, **kwargs: _fake_settings())
    monkeypatch.setattr(api, "run_universe", lambda *args, **kwargs: sample.copy())

    full_df, radar_df, summary = api.run_custom_garp(
        config_path=None,
        profile_name="balanced",
        tickers=["AAA.PA", "BBB.PA"],
    )
    assert len(full_df) == 2
    assert len(radar_df) == 1
    assert radar_df[GARP_FLAG_COLUMN].all()
    assert summary["strict_count"] == 1
    assert summary["bucket_counts"]["ELITE_GARP"] == 1


def test_run_cac40_garp_history_logging(monkeypatch, tmp_path):
    sample = pd.DataFrame(
        [
            {
                "ticker": "AAA.PA",
                "score_v1_1": 5,
                "category_v1_1": "ELITE_V1_1",
                "market_cap": 20e9,
                "data_complete_v1_1": True,
                GARP_FLAG_COLUMN: True,
                GARP_SCORE_COLUMN: 90.0,
                GARP_BUCKET_COLUMN: "ELITE_GARP",
            },
            {
                "ticker": "BBB.PA",
                "score_v1_1": 3,
                "category_v1_1": "WATCHLIST_V1_1",
                "market_cap": 10e9,
                "data_complete_v1_1": True,
                GARP_FLAG_COLUMN: False,
                GARP_SCORE_COLUMN: 40.0,
                GARP_BUCKET_COLUMN: "REJECT_GARP",
            },
        ]
    )

    monkeypatch.setattr(api, "load_settings", lambda *args, **kwargs: _fake_settings())
    monkeypatch.setattr(api, "run_universe", lambda *args, **kwargs: sample.copy())

    history_file = tmp_path / "history.csv"
    full_df, radar_df, summary = api.run_cac40_garp(
        config_path=None,
        profile_name="balanced",
        history_path=str(history_file),
        run_id="test-run",
        notes="check",
    )
    assert not full_df.empty
    assert summary["run_id"] == "test-run"
    history_df = pd.read_csv(history_file)
    assert len(history_df) == 1
    assert history_df.iloc[0]["run_id"] == "test-run"
    assert history_df.iloc[0]["strict_garp_count"] == 1


def test_run_garp_diagnostics_offline(tmp_path):
    snapshot = pd.DataFrame(
        [
            {"ticker": "AAA.PA", "data_complete_v1_1": True, "garp_pe_ttm_ok": True, "garp_pe_fwd_ok": True, "garp_debt_to_equity_ok": True, "garp_eps_cagr_ok": True, "garp_peg_ok": True, GARP_FLAG_COLUMN: True},
            {"ticker": "BBB.PA", "data_complete_v1_1": True, "garp_pe_ttm_ok": False, "garp_pe_fwd_ok": True, "garp_debt_to_equity_ok": True, "garp_eps_cagr_ok": True, "garp_peg_ok": True, GARP_FLAG_COLUMN: False},
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2025-01-01", "ticker": "AAA.PA", "adj_close": 1.0},
            {"date": "2025-01-02", "ticker": "AAA.PA", "adj_close": 2.0},
            {"date": "2025-01-01", "ticker": "BBB.PA", "adj_close": 1.0},
            {"date": "2025-01-02", "ticker": "BBB.PA", "adj_close": 1.1},
        ]
    )
    snapshot_path = tmp_path / "snapshot.csv"
    prices_path = tmp_path / "prices.csv"
    snapshot.to_csv(snapshot_path, index=False)
    prices.to_csv(prices_path, index=False)

    result = api.run_garp_diagnostics_offline(
        snapshot_path=str(snapshot_path),
        prices_path=str(prices_path),
        start_date="2025-01-01",
        horizons=[1],
    )

    assert ("pe_ttm", 1) in result.index
    assert "pass_return" in result.columns
