import pandas as pd

from orotitan_nexus.config import NexusCoreSettings
from orotitan_nexus.nexus_core_pillars import (
    compute_behavior_pillar,
    compute_fit_pillar,
    compute_momentum_pillar,
    compute_quality_pillar,
    compute_risk_pillar,
    compute_value_pillar,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "roe_pct": [20.0, 5.0],
            "operating_margin_pct": [15.0, 2.0],
            "fcf_yield_pct": [8.0, 1.0],
            "ev_ebit_fwd": [6.0, 14.0],
            "pe_fwd": [12.0, 30.0],
            "div_yield_pct": [3.0, 1.0],
            "buyback_yield_pct": [2.0, 0.0],
            "perf_1m_pct": [5.0, -2.0],
            "perf_3m_pct": [10.0, -5.0],
            "perf_6m_pct": [15.0, -8.0],
            "mm50_slope_pct": [1.0, -1.0],
            "dist_to_ph63_pct": [-5.0, -20.0],
            "natr14_pct": [2.0, 10.0],
            "max_drawdown_6m_pct": [5.0, 20.0],
            "beta_6m_cac": [1.1, 1.6],
            "avg_dvol_20d_eur": [2_000_000.0, 500_000.0],
            "gap_open_pct": [1.0, 5.0],
            "avwap_dist_pct": [2.0, 10.0],
            "pea_eligible_bool": [True, False],
            "sector": ["Health", "Energy"],
        }
    )


def test_pillars_monotonic():
    df = _sample_df()
    settings = NexusCoreSettings(enabled=True)

    q = compute_quality_pillar(df, settings)
    v = compute_value_pillar(df, settings)
    m = compute_momentum_pillar(df, settings)
    r = compute_risk_pillar(df, settings)
    b = compute_behavior_pillar(df, settings)
    f = compute_fit_pillar(df, settings)

    assert q.iloc[0] > q.iloc[1]
    assert v.iloc[0] > v.iloc[1]
    assert m.iloc[0] > m.iloc[1]
    assert r.iloc[0] > r.iloc[1]
    assert b.iloc[0] > b.iloc[1]
    assert f.iloc[0] > f.iloc[1]

