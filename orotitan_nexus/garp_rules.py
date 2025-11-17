"""Standalone GARP rule evaluation for CAC 40 radar outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GarpThresholds

GARP_FLAG_COLUMN = "strict_pass_garp"
GARP_SCORE_COLUMN = "strict_garp_score"
GARP_BUCKET_COLUMN = "strict_garp_bucket"
RULE_COL_PE_TTM = "garp_pe_ttm_ok"
RULE_COL_PE_FWD = "garp_pe_fwd_ok"
RULE_COL_DEBT = "garp_debt_to_equity_ok"
RULE_COL_EPS_CAGR = "garp_eps_cagr_ok"
RULE_COL_PEG = "garp_peg_ok"
_BUCKETS = (
    ("BORDERLINE_GARP", 50.0),
    ("STRONG_GARP", 65.0),
    ("ELITE_GARP", 80.0),
)


def _safe_mask(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return a boolean mask and coerced numeric values."""

    values = pd.to_numeric(series, errors="coerce")
    return values.notna(), values


def apply_garp_rules(df: pd.DataFrame, thresholds: GarpThresholds) -> pd.DataFrame:
    """Apply the five CAC 40 GARP rules and expose per-rule booleans.

    Adds the following columns (in addition to leaving existing columns
    untouched):

    - ``garp_pe_ttm_ok``
    - ``garp_pe_fwd_ok``
    - ``garp_debt_to_equity_ok``
    - ``garp_eps_cagr_ok``
    - ``garp_peg_ok``
    - ``strict_pass_garp`` (AND of the five rules)

    Missing values fail the individual rule and therefore the strict pass.
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

    result[RULE_COL_PE_TTM] = pe_ttm_ok
    result[RULE_COL_PE_FWD] = pe_fwd_ok
    result[RULE_COL_DEBT] = debt_ok
    result[RULE_COL_EPS_CAGR] = eps_ok
    result[RULE_COL_PEG] = peg_ok
    result[GARP_FLAG_COLUMN] = pe_ttm_ok & pe_fwd_ok & debt_ok & eps_ok & peg_ok
    return result


def _score_max_rule(values: pd.Series, threshold: float) -> pd.Series:
    """Return scores (0-1) where lower values are better than ``threshold``."""

    values = pd.to_numeric(values, errors="coerce")
    denom = threshold if threshold > 0 else max(values.max(skipna=True) or 1.0, 1.0)
    denom = max(denom, 1e-6)
    scores = pd.Series(0.0, index=values.index, dtype=float)
    finite_mask = values.notna()
    if not finite_mask.any():
        return scores

    safe = values[finite_mask]
    below = safe <= threshold
    above = ~below
    local = pd.Series(0.0, index=safe.index, dtype=float)

    if below.any():
        local.loc[below] = 0.5 + 0.5 * np.clip((threshold - safe.loc[below]) / denom, 0.0, 1.0)
    if above.any():
        local.loc[above] = np.maximum(
            0.0,
            0.5 - 0.5 * np.clip((safe.loc[above] - threshold) / denom, 0.0, 1.0),
        )
    scores.loc[safe.index] = local
    scores.loc[~finite_mask] = 0.0
    return scores


def _score_min_rule(values: pd.Series, threshold: float) -> pd.Series:
    """Return scores (0-1) where higher values are better than ``threshold``."""

    values = pd.to_numeric(values, errors="coerce")
    denom = threshold if threshold > 0 else max(values.max(skipna=True) or 1.0, 1.0)
    denom = max(denom, 1e-6)
    scores = pd.Series(0.0, index=values.index, dtype=float)
    finite_mask = values.notna()
    if not finite_mask.any():
        return scores

    safe = values[finite_mask]
    above = safe >= threshold
    below = ~above
    local = pd.Series(0.0, index=safe.index, dtype=float)

    if above.any():
        local.loc[above] = 0.5 + 0.5 * np.clip((safe.loc[above] - threshold) / denom, 0.0, 1.0)
    if below.any():
        local.loc[below] = np.maximum(
            0.0,
            0.5 - 0.5 * np.clip((threshold - safe.loc[below]) / denom, 0.0, 1.0),
        )
    scores.loc[safe.index] = local
    scores.loc[~finite_mask] = 0.0
    return scores


def compute_garp_score(df: pd.DataFrame, thresholds: GarpThresholds) -> pd.DataFrame:
    """Compute a monotonic 0-100 GARP score based on distance to thresholds."""

    if df.empty:
        result = df.copy()
        result[GARP_SCORE_COLUMN] = 0.0
        return result

    result = df.copy()
    rule_scores = [
        _score_max_rule(result["pe_ttm"], thresholds.pe_ttm_max),
        _score_max_rule(result["pe_fwd"], thresholds.pe_fwd_max),
        _score_max_rule(result["debt_to_equity"], thresholds.debt_to_equity_max),
        _score_min_rule(result["eps_cagr"], thresholds.eps_cagr_min),
        _score_max_rule(result["peg"], thresholds.peg_max),
    ]
    stacked = pd.concat(rule_scores, axis=1)
    stacked = stacked.fillna(0.0)
    scores = stacked.mean(axis=1).clip(0.0, 1.0) * 100.0
    result[GARP_SCORE_COLUMN] = scores
    return result


def assign_garp_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Assign descriptive buckets (ELITE/STRONG/BORDERLINE/REJECT) per row."""

    if df.empty:
        result = df.copy()
        result[GARP_BUCKET_COLUMN] = "REJECT_GARP"
        return result

    result = df.copy()
    scores = pd.to_numeric(result.get(GARP_SCORE_COLUMN, pd.Series(dtype=float)), errors="coerce")
    buckets = pd.Series("REJECT_GARP", index=result.index, dtype=object)

    for label, cutoff in _BUCKETS:
        buckets.loc[scores >= cutoff] = label

    passed = result.get(GARP_FLAG_COLUMN, pd.Series(False, index=result.index)).fillna(False)
    buckets.loc[passed & ~buckets.isin({"ELITE_GARP", "STRONG_GARP"})] = "STRONG_GARP"

    result[GARP_BUCKET_COLUMN] = buckets
    return result
