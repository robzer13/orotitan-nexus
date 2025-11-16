import pandas as pd

from orotitan_nexus.config import GarpThresholds
from orotitan_nexus.garp_rules import (
    GARP_BUCKET_COLUMN,
    GARP_FLAG_COLUMN,
    GARP_SCORE_COLUMN,
    RULE_COL_DEBT,
    RULE_COL_EPS_CAGR,
    RULE_COL_PE_FWD,
    RULE_COL_PE_TTM,
    RULE_COL_PEG,
    apply_garp_rules,
    assign_garp_buckets,
    compute_garp_score as compute_strict_garp_score,
)


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
    for col in [
        RULE_COL_PE_TTM,
        RULE_COL_PE_FWD,
        RULE_COL_DEBT,
        RULE_COL_EPS_CAGR,
        RULE_COL_PEG,
    ]:
        assert result[col].all()


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
    assert not bool(result.loc[0, RULE_COL_PE_TTM])
    assert not bool(result.loc[1, RULE_COL_PE_FWD])
    assert not bool(result.loc[2, RULE_COL_DEBT])
    assert not bool(result.loc[3, RULE_COL_EPS_CAGR])
    assert not bool(result.loc[4, RULE_COL_PEG])


def test_apply_garp_rules_missing_values_fail():
    row = _base_row()
    row["pe_ttm"] = float("nan")
    df = pd.DataFrame([row])
    result = apply_garp_rules(df, GarpThresholds())
    assert not result[GARP_FLAG_COLUMN].any()


def test_compute_garp_score_monotonic():
    better = _base_row()
    weaker = {**_base_row(), "pe_ttm": 26.0, "peg": 1.4, "eps_cagr": 0.12}
    df = pd.DataFrame([better, weaker])
    scored = compute_strict_garp_score(df, GarpThresholds())
    assert scored.loc[0, GARP_SCORE_COLUMN] > scored.loc[1, GARP_SCORE_COLUMN]


def test_compute_garp_score_handles_nans():
    row = _base_row()
    row["peg"] = float("nan")
    df = pd.DataFrame([row])
    scored = compute_strict_garp_score(df, GarpThresholds())
    assert 0.0 <= scored.loc[0, GARP_SCORE_COLUMN] <= 80.0


def test_assign_garp_buckets_boundaries():
    df = pd.DataFrame(
        [
            {**_base_row(), GARP_SCORE_COLUMN: 85.0},
            {**_base_row(), GARP_SCORE_COLUMN: 70.0},
            {**_base_row(), GARP_SCORE_COLUMN: 55.0},
            {**_base_row(), GARP_SCORE_COLUMN: 40.0},
        ]
    )
    bucketed = assign_garp_buckets(df)
    expected = ["ELITE_GARP", "STRONG_GARP", "BORDERLINE_GARP", "REJECT_GARP"]
    assert bucketed[GARP_BUCKET_COLUMN].tolist() == expected


def test_strict_pass_rows_promoted_to_strong():
    df = pd.DataFrame(
        [
            {**_base_row(), GARP_SCORE_COLUMN: 45.0, GARP_FLAG_COLUMN: True},
            {**_base_row(), GARP_SCORE_COLUMN: 90.0, GARP_FLAG_COLUMN: True},
        ]
    )
    bucketed = assign_garp_buckets(df)
    assert (bucketed[GARP_BUCKET_COLUMN].isin({"STRONG_GARP", "ELITE_GARP"})).all()
