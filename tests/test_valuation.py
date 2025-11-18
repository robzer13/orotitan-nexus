import pandas as pd

from orotitan_nexus.config import ProfileSettingsV2, ValuationSettings
from orotitan_nexus.valuation import Scenario, apply_valuation, compute_scenario_fair_value


def test_apply_valuation_composite_and_fallback():
    profile = ProfileSettingsV2(valuation=ValuationSettings(enabled=True))
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "price": 100.0,
                "FCF_YIELD_pct": 10.0,
                "EV_EBIT_FWD": 8.0,
                "PE_FWD": 12.0,
                "FCF_per_share": 5.0,
                "EBIT_FWD": 20.0,
                "EPS_FWD": 4.0,
                "Shares_outstanding": 10.0,
                "NetDebt": 10.0,
                "nexus_core_q": 50.0,
                "Consensus_PT_12m": 90.0,
            },
            {
                "ticker": "BBB",
                "price": 100.0,
                "FCF_YIELD_pct": None,
                "EV_EBIT_FWD": None,
                "PE_FWD": None,
                "Consensus_PT_12m": 120.0,
            },
        ]
    )

    valued = apply_valuation(df, profile)
    assert {"fair_price_composite", "upside_pct", "fair_price_source"} <= set(valued.columns)
    row0 = valued.iloc[0]
    assert row0["fair_price_source"] == "composite"
    assert row0["upside_pct"] < 0  # price above fair

    row1 = valued.iloc[1]
    assert row1["fair_price_source"] == "consensus"
    assert row1["upside_pct"] == 20.0


def test_apply_valuation_divergence_flag():
    profile = ProfileSettingsV2(valuation=ValuationSettings(enabled=True, divergence_warn_pct=10.0))
    df = pd.DataFrame(
        [
            {
                "ticker": "CCC",
                "price": 100.0,
                "FCF_YIELD_pct": 20.0,
                "EV_EBIT_FWD": 5.0,
                "PE_FWD": 10.0,
                "FCF_per_share": 5.0,
                "EBIT_FWD": 10.0,
                "EPS_FWD": 3.0,
                "Shares_outstanding": 5.0,
                "Consensus_PT_12m": 200.0,
            }
        ]
    )
    valued = apply_valuation(df, profile)
    assert bool(valued.loc[0, "fair_price_divergence_flag"]) is True


def test_compute_scenario_fair_value():
    scenarios = [
        Scenario(name="Bear", metric=10.0, multiple=5.0, weight=0.2),
        Scenario(name="Base", metric=10.0, multiple=8.0, weight=0.5),
        Scenario(name="Bull", metric=10.0, multiple=12.0, weight=0.3),
    ]
    result = compute_scenario_fair_value(scenarios, net_debt=10.0, shares_outstanding=10.0)
    assert result["prices"]["Bear"] == 4.0
    assert result["prices"]["Base"] == 7.0
    assert result["prices"]["Bull"] == 11.0
    assert 0 < result["fair_value"] < 10

