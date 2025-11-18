import pandas as pd

from orotitan_nexus.config import ProfileSettingsV2, NexusV2Settings, RegimeSettings
from orotitan_nexus.regimes_v2 import label_regimes, regime_performance


def _profile():
    profile = ProfileSettingsV2(name="balanced")
    profile.v2 = NexusV2Settings(enabled=True)
    profile.regime = RegimeSettings(enabled=True, benchmark_ticker="^IDX", lookback_days=1)
    return profile


def test_label_regimes_basic():
    profile = _profile()
    price_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ticker": ["^IDX", "^IDX", "^IDX"],
            "adj_close": [1.0, 1.1, 1.3],
        }
    )
    regimes = label_regimes(price_df, profile)
    assert not regimes.empty


def test_regime_performance_basic():
    profile = _profile()
    df = pd.DataFrame({"ticker": ["A"], "nexus_v2_bucket": ["V2_ELITE"]})
    price_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["^IDX", "A"],
            "adj_close": [1.0, 1.1],
        }
    )
    perf = regime_performance(df, price_df, profile)
    assert isinstance(perf, dict)
