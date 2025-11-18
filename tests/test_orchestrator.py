from orotitan_nexus import orchestrator
from orotitan_nexus.config import FilterSettings, GarpThresholds, UniverseSettings, WeightSettings
from orotitan_nexus.garp_rules import GARP_FLAG_COLUMN


def _record_factory(ticker):
    return {
        "ticker": ticker,
        "pe_ttm": 20.0,
        "pe_fwd": 14.0,
        "debt_to_equity": 0.30,
        "eps_cagr": 0.20,
        "peg": 1.0,
        "market_cap": 15e9,
        "vol_1y": 0.2,
        "mdd_1y": -0.25,
        "adv_3m": 5_000_000,
        "eps_cagr_source": "net_income",
        "has_pe_ttm": True,
        "has_pe_fwd": True,
        "has_debt_to_equity": True,
        "has_eps_cagr": True,
        "has_peg": True,
        "has_market_cap": True,
        "has_risk_data": True,
        "data_ready_nexus": True,
        "data_ready_v1_1": True,
        "data_complete_v1_1": True,
    }


def test_run_universe_sets_garp_flag(monkeypatch):
    monkeypatch.setattr(orchestrator, "build_record_for_ticker", lambda *args, **kwargs: _record_factory("AAA.PA"))

    filters = FilterSettings()
    weights = WeightSettings()
    universe = UniverseSettings(tickers=["AAA.PA"], min_adv_3m=1.0)

    df = orchestrator.run_universe(
        filters,
        weights,
        universe,
        apply_garp=True,
        garp_thresholds=GarpThresholds(),
    )
    assert not df.empty
    assert df[GARP_FLAG_COLUMN].all()
