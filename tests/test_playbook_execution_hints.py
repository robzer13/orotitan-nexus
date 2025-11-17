import pandas as pd

from orotitan_nexus.config import ProfileSettingsV2
from orotitan_nexus.playbook import run_nexus_playbook


def test_playbook_execution_hints():
    profile = ProfileSettingsV2(name="test")
    profile.playbook.enabled = True
    profile.playbook.thresholds.min_core_score_buy = 50.0
    profile.playbook.thresholds.min_upside_buy_pct = 5.0

    df = pd.DataFrame(
        [
            {
                "ticker": "EXEC.PA",
                "price": 100.0,
                "PH20": 102.0,
                "ATR14": 2.0,
                "Volume": 20000,
                "ADV20_shares": 10000,
                "AVG_DVOL_20D_EUR": 2_000_000.0,
                "nexus_core_score": 80.0,
                "nexus_v2_score": 80.0,
                "nexus_exceptionality": 7.0,
                "upside_pct": 20.0,
                "momentum_score": 60.0,
                "risk_score": 60.0,
                "strict_pass_garp": True,
                "ETF_bool": False,
                "owned": False,
            }
        ]
    )

    summary = run_nexus_playbook(df, profile, universe_name="TEST", portfolio_context={"portfolio_value": 50_000})
    decision = summary.decisions[0]
    assert decision.action == profile.playbook.label_buy
    assert decision.entry_price is not None
    assert decision.stop_price is not None
    assert decision.size_qty is None or decision.size_qty >= 0

