import pandas as pd

from orotitan_nexus.config import EtfScoringSettings, ProfileSettingsV2
from orotitan_nexus.etf_scoring import compute_etf_nexus_score


def test_etf_nexus_score_orders_by_cost_and_liquidity():
    settings = EtfScoringSettings(enabled=True, min_liquidity_eur=100_000.0)
    profile = ProfileSettingsV2(etf_scoring=settings)
    df = pd.DataFrame(
        [
            {"ticker": "ETF1", "ETF_bool": True, "TER_pct": 0.1, "Tracking_diff_3Y_pct": 1.0, "ADV20_EUR": 200_000},
            {"ticker": "ETF2", "ETF_bool": True, "TER_pct": 0.5, "Tracking_diff_3Y_pct": 2.0, "ADV20_EUR": 50_000},
        ]
    )
    scores = compute_etf_nexus_score(df, settings, profile)
    assert scores.iloc[0] > scores.iloc[1]

