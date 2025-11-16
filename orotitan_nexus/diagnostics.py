"""Rule-level diagnostics for GARP calibration."""
from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from .backtest import compute_horizon_returns
from .garp_rules import (
    GARP_FLAG_COLUMN,
    RULE_COL_DEBT,
    RULE_COL_EPS_CAGR,
    RULE_COL_PE_FWD,
    RULE_COL_PE_TTM,
    RULE_COL_PEG,
)

RULE_NAMES: List[tuple[str, str]] = [
    ("pe_ttm", RULE_COL_PE_TTM),
    ("pe_fwd", RULE_COL_PE_FWD),
    ("debt_to_equity", RULE_COL_DEBT),
    ("eps_cagr", RULE_COL_EPS_CAGR),
    ("peg", RULE_COL_PEG),
    ("strict_5of5", GARP_FLAG_COLUMN),
]


def compute_rule_diagnostics(
    snapshot_df: pd.DataFrame,
    price_df: pd.DataFrame,
    start_date: pd.Timestamp,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Compare forward performance for pass vs fail groups per GARP rule.

    The calculation is limited to tickers where ``data_complete_v1_1`` is True.
    For each rule, the function computes equal-weight average returns of the
    tickers that pass and fail the rule over each requested horizon.

    Missing prices or empty groups produce NaN returns for that side.
    """

    base_df = snapshot_df[snapshot_df.get("data_complete_v1_1", False) == True]
    records: List[Dict[str, object]] = []

    for rule_name, col_name in RULE_NAMES:
        if col_name not in base_df.columns:
            raise ValueError(f"Snapshot missing required rule column: {col_name}")

        pass_tickers = base_df.loc[base_df[col_name] == True, "ticker"].tolist()
        fail_tickers = base_df.loc[base_df[col_name] == False, "ticker"].tolist()

        for horizon in horizons:
            pass_returns = compute_horizon_returns(price_df, pass_tickers, start_date, horizon)
            fail_returns = compute_horizon_returns(price_df, fail_tickers, start_date, horizon)

            pass_mean = pass_returns.dropna().mean() if not pass_returns.empty else pd.NA
            fail_mean = fail_returns.dropna().mean() if not fail_returns.empty else pd.NA

            if pd.notna(pass_mean) and pd.notna(fail_mean):
                delta = float(pass_mean - fail_mean)
            else:
                delta = pd.NA

            records.append(
                {
                    "rule": rule_name,
                    "horizon_days": int(horizon),
                    "pass_return": float(pass_mean) if pd.notna(pass_mean) else pd.NA,
                    "fail_return": float(fail_mean) if pd.notna(fail_mean) else pd.NA,
                    "delta_return": delta,
                    "n_pass": int(pass_returns.dropna().shape[0]) if not pass_returns.empty else 0,
                    "n_fail": int(fail_returns.dropna().shape[0]) if not fail_returns.empty else 0,
                }
            )

    result = pd.DataFrame.from_records(records)
    return result.set_index(["rule", "horizon_days"])
