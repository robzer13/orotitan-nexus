import pandas as pd

from orotitan_nexus.diagnostics import compute_rule_diagnostics
from orotitan_nexus.garp_rules import (
    GARP_FLAG_COLUMN,
    RULE_COL_DEBT,
    RULE_COL_EPS_CAGR,
    RULE_COL_PE_FWD,
    RULE_COL_PE_TTM,
    RULE_COL_PEG,
)


def _snapshot_df():
    data = [
        {
            "ticker": "AAA.PA",
            "data_complete_v1_1": True,
            RULE_COL_PE_TTM: True,
            RULE_COL_PE_FWD: True,
            RULE_COL_DEBT: True,
            RULE_COL_EPS_CAGR: True,
            RULE_COL_PEG: True,
            GARP_FLAG_COLUMN: True,
        },
        {
            "ticker": "BBB.PA",
            "data_complete_v1_1": True,
            RULE_COL_PE_TTM: False,
            RULE_COL_PE_FWD: True,
            RULE_COL_DEBT: True,
            RULE_COL_EPS_CAGR: True,
            RULE_COL_PEG: True,
            GARP_FLAG_COLUMN: False,
        },
        {
            "ticker": "CCC.PA",
            "data_complete_v1_1": True,
            RULE_COL_PE_TTM: True,
            RULE_COL_PE_FWD: False,
            RULE_COL_DEBT: False,
            RULE_COL_EPS_CAGR: False,
            RULE_COL_PEG: False,
            GARP_FLAG_COLUMN: False,
        },
        {
            "ticker": "DDD.PA",
            "data_complete_v1_1": True,
            RULE_COL_PE_TTM: False,
            RULE_COL_PE_FWD: False,
            RULE_COL_DEBT: False,
            RULE_COL_EPS_CAGR: False,
            RULE_COL_PEG: False,
            GARP_FLAG_COLUMN: False,
        },
    ]
    return pd.DataFrame(data)


def _prices_df():
    rows = []
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    price_map = {
        "AAA.PA": [1.0, 1.5, 2.0],
        "BBB.PA": [1.0, 1.1, 1.2],
        "CCC.PA": [1.0, 0.9, 0.8],
        "DDD.PA": [1.0, 1.0, 1.0],
    }
    for ticker, prices in price_map.items():
        for date, price in zip(dates, prices):
            rows.append({"date": date, "ticker": ticker, "adj_close": price})
    return pd.DataFrame(rows)


def test_compute_rule_diagnostics_shapes():
    snapshot_df = _snapshot_df()
    price_df = _prices_df()
    result = compute_rule_diagnostics(snapshot_df, price_df, pd.Timestamp("2025-01-01"), [2])
    assert ("pe_ttm", 2) in result.index
    assert ("strict_5of5", 2) in result.index
    for col in ["pass_return", "fail_return", "delta_return", "n_pass", "n_fail"]:
        assert col in result.columns


def test_compute_rule_diagnostics_signal_direction():
    snapshot_df = _snapshot_df()
    price_df = _prices_df()
    result = compute_rule_diagnostics(snapshot_df, price_df, pd.Timestamp("2025-01-01"), [2])
    row = result.loc[("pe_ttm", 2)]
    assert row["delta_return"] > 0
    assert row["n_pass"] == 2
    assert row["n_fail"] == 2
    strict_row = result.loc[("strict_5of5", 2)]
    assert strict_row["n_pass"] == 1
    assert strict_row["n_fail"] == 3
