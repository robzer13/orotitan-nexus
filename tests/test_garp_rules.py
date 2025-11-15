import pandas as pd

from orotitan_nexus.config import GarpThresholds
from orotitan_nexus.garp_rules import GARP_FLAG_COLUMN, apply_garp_rules


def _base_row():
    return {
        "ticker": "AAA.PA",
        "pe_ttm": 20.0,
        "pe_fwd": 12.0,
        "debt_to_equity": 0.30,
        "eps_cagr": 0.20,
        "peg": 0.8,
    }


def test_apply_garp_rules_perfect_match():
    df = pd.DataFrame([_base_row()])
    result = apply_garp_rules(df, GarpThresholds())
    assert result[GARP_FLAG_COLUMN].all()


def test_apply_garp_rules_break_each_rule():
    rows = []
    base = _base_row()
    rows.append({**base, "pe_ttm": 30.0})
    rows.append({**base, "pe_fwd": 16.0})
    rows.append({**base, "debt_to_equity": 0.6})
    rows.append({**base, "eps_cagr": 0.05})
    rows.append({**base, "peg": 1.5})
    df = pd.DataFrame(rows)
    result = apply_garp_rules(df, GarpThresholds())
    assert not result[GARP_FLAG_COLUMN].any()


def test_apply_garp_rules_missing_values_fail():
    row = _base_row()
    row["pe_ttm"] = float("nan")
    df = pd.DataFrame([row])
    result = apply_garp_rules(df, GarpThresholds())
    assert not result[GARP_FLAG_COLUMN].any()
