"""Standalone GARP rule evaluation for CAC 40 radar outputs."""
from __future__ import annotations

import pandas as pd

from .config import GarpThresholds

GARP_FLAG_COLUMN = "strict_pass_garp"


def _safe_mask(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return a boolean mask and coerced numeric values."""

    values = pd.to_numeric(series, errors="coerce")
    return values.notna(), values


def apply_garp_rules(df: pd.DataFrame, thresholds: GarpThresholds) -> pd.DataFrame:
    """Apply the five CAC 40 GARP rules to ``df`` and add ``strict_pass_garp``.

    Each rule requires its respective column to be present and finite; missing
    values automatically fail the rule so that a company cannot pass with
    incomplete fundamentals.
    """

    if df.empty:
        result = df.copy()
        result[GARP_FLAG_COLUMN] = False
        return result

    result = df.copy()

    mask, values = _safe_mask(result["pe_ttm"])
    pe_ttm_ok = mask & (values > 0) & (values < thresholds.pe_ttm_max)

    mask, values = _safe_mask(result["pe_fwd"])
    pe_fwd_ok = mask & (values > 0) & (values < thresholds.pe_fwd_max)

    mask, values = _safe_mask(result["debt_to_equity"])
    debt_ok = mask & (values < thresholds.debt_to_equity_max)

    mask, values = _safe_mask(result["eps_cagr"])
    eps_ok = mask & (values > thresholds.eps_cagr_min)

    mask, values = _safe_mask(result["peg"])
    peg_ok = mask & (values > 0) & (values < thresholds.peg_max)

    result[GARP_FLAG_COLUMN] = pe_ttm_ok & pe_fwd_ok & debt_ok & eps_ok & peg_ok
    return result
