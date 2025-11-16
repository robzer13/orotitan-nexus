import pandas as pd

from orotitan_nexus.config import ProfileSettingsV2, NexusV2Settings
from orotitan_nexus.explain_v2 import add_explain_columns, explain_ticker


def _profile():
    profile = ProfileSettingsV2(name="balanced")
    profile.v2 = NexusV2Settings(enabled=True)
    return profile


def test_add_explain_columns():
    profile = _profile()
    df = pd.DataFrame({
        "ticker": ["A"],
        "garp_score": [80],
        "quality_score": [70],
        "momentum_score": [60],
        "risk_score": [50],
        "macro_score": [40],
        "behavioral_score": [30],
    })
    enriched = add_explain_columns(df, profile)
    assert "garp_contrib_v2" in enriched


def test_explain_ticker():
    profile = _profile()
    df = add_explain_columns(pd.DataFrame({
        "ticker": ["A"],
        "nexus_v2_score": [75],
        "nexus_v2_bucket": ["V2_STRONG"],
        "garp_score": [80],
        "quality_score": [70],
        "momentum_score": [60],
        "risk_score": [50],
        "macro_score": [40],
        "behavioral_score": [30],
    }), profile)
    info = explain_ticker(df, "A", profile)
    assert info["ticker"] == "A"
    assert "contributions" in info
