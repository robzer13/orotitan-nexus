import pandas as pd

from orotitan_nexus.config import NexusCoreSettings, ProfileSettingsV2
from orotitan_nexus.nexus_core_scoring import apply_nexus_core_scores


def test_nexus_core_scoring_and_exceptionality():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "roe_pct": [18.0, 4.0],
            "operating_margin_pct": [12.0, 3.0],
            "fcf_yield_pct": [6.0, 1.0],
            "ev_ebit_fwd": [7.0, 16.0],
            "pe_fwd": [14.0, 28.0],
            "div_yield_pct": [2.0, 0.5],
            "buyback_yield_pct": [1.5, 0.0],
            "perf_1m_pct": [4.0, -2.0],
            "perf_3m_pct": [8.0, -6.0],
            "perf_6m_pct": [12.0, -10.0],
            "mm50_slope_pct": [1.0, -1.0],
            "dist_to_ph63_pct": [-6.0, -18.0],
            "natr14_pct": [3.0, 12.0],
            "max_drawdown_6m_pct": [6.0, 25.0],
            "beta_6m_cac": [1.0, 1.7],
            "avg_dvol_20d_eur": [3_000_000.0, 400_000.0],
            "gap_open_pct": [1.0, 7.0],
            "avwap_dist_pct": [2.0, 12.0],
            "pea_eligible_bool": [True, False],
            "sector": ["Health", "Energy"],
            "diamond_lt": [True, False],
            "gold_lt": [False, True],
            "red_flag": [False, True],
        }
    )

    profile = ProfileSettingsV2(name="test", nexus_core=NexusCoreSettings(enabled=True))
    scored = apply_nexus_core_scores(df, profile)

    for col in [
        "nexus_core_q",
        "nexus_core_v",
        "nexus_core_m",
        "nexus_core_r",
        "nexus_core_b",
        "nexus_core_f",
        "nexus_core_score",
        "nexus_exceptionality",
    ]:
        assert col in scored

    assert scored.loc[0, "nexus_core_score"] > scored.loc[1, "nexus_core_score"]
    assert 0.0 <= scored["nexus_exceptionality"].min() <= 10.0
    assert scored["nexus_exceptionality"].max() <= 10.0

