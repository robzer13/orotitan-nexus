import pandas as pd

from orotitan_nexus.config import NexusV2Settings, ProfileSettingsV2, QualitySettings, MomentumSettings, RiskSettings, MacroSettings, BehavioralSettings
from orotitan_nexus.scoring_v2 import apply_v2_scores, NEXUS_V2_SCORE, NEXUS_V2_BUCKET


def _make_profile(enabled: bool = True) -> ProfileSettingsV2:
    return ProfileSettingsV2(
        name="balanced",
        v2=NexusV2Settings(enabled=enabled),
        quality=QualitySettings(),
        momentum=MomentumSettings(),
        risk=RiskSettings(),
        macro=MacroSettings(),
        behavioral=BehavioralSettings(),
    )


def test_apply_v2_scores_basic():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "strict_garp_score": [90, 30],
            "roe": [0.15, 0.05],
            "roa": [0.08, 0.02],
            "profit_margin": [0.2, 0.05],
            "sector": ["Tech", "Tech"],
            "momentum_score": [70, 20],
        }
    )
    profile = _make_profile(True)
    price_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).repeat(2),
            "ticker": ["AAA", "BBB"] * 3,
            "adj_close": [1.0, 1.0, 1.1, 1.0, 1.2, 0.9],
        }
    )
    enriched = apply_v2_scores(df, profile, price_df=price_df, start_date=pd.to_datetime("2024-01-01"))
    assert NEXUS_V2_SCORE in enriched
    assert NEXUS_V2_BUCKET in enriched
    assert enriched.loc[enriched["ticker"] == "AAA", NEXUS_V2_SCORE].iloc[0] > enriched.loc[enriched["ticker"] == "BBB", NEXUS_V2_SCORE].iloc[0]


def test_apply_v2_scores_noop_when_disabled():
    df = pd.DataFrame({"ticker": ["AAA"], "strict_garp_score": [80]})
    profile = _make_profile(False)
    enriched = apply_v2_scores(df, profile)
    assert NEXUS_V2_SCORE not in enriched
