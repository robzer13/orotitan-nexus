import pandas as pd

from orotitan_nexus.config import PlaybookSettings, ProfileSettingsV2
from orotitan_nexus.playbook import run_nexus_playbook


def _profile_with_playbook() -> ProfileSettingsV2:
    profile = ProfileSettingsV2(name="test")
    profile.playbook.enabled = True
    profile.playbook.thresholds.min_core_score_buy = 70.0
    profile.playbook.thresholds.min_v2_score_buy = 60.0
    profile.playbook.thresholds.min_upside_buy_pct = 10.0
    profile.playbook.thresholds.min_momentum_score_buy = 40.0
    return profile


def test_playbook_classifies_actions():
    profile = _profile_with_playbook()
    df = pd.DataFrame(
        [
            {
                "ticker": "BUY.PA",
                "nexus_core_score": 80.0,
                "nexus_v2_score": 75.0,
                "nexus_exceptionality": 8.0,
                "upside_pct": 20.0,
                "momentum_score": 60.0,
                "risk_score": 55.0,
                "strict_pass_garp": True,
                "ETF_bool": False,
                "owned": False,
            },
            {
                "ticker": "OWNED.PA",
                "nexus_core_score": 68.0,
                "nexus_v2_score": 65.0,
                "nexus_exceptionality": 6.0,
                "upside_pct": 12.0,
                "momentum_score": 55.0,
                "risk_score": 60.0,
                "strict_pass_garp": True,
                "ETF_bool": False,
                "owned": True,
            },
            {
                "ticker": "ETF.PA",
                "etf_nexus_score": 80.0,
                "ETF_bool": True,
                "owned": False,
            },
        ]
    )

    summary = run_nexus_playbook(df, profile, universe_name="TEST")
    actions = {d.ticker: d.action for d in summary.decisions}
    assert actions["BUY.PA"] == profile.playbook.label_buy
    assert actions["OWNED.PA"] in {profile.playbook.label_add, profile.playbook.label_hold, profile.playbook.label_watch}
    assert actions["ETF.PA"] != ""

