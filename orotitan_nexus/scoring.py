"""Scoring logic for valuation, growth, risk, and Nexus metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FilterSettings, WeightSettings


def interpolate(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x0 == x1:
        return y1
    return y0 + (value - x0) * (y1 - y0) / (x1 - x0)


def _score_pe(pe: float) -> float:
    if np.isnan(pe):
        return np.nan
    if pe <= 15.0:
        return 100.0
    if pe <= 25.0:
        return interpolate(pe, 15.0, 25.0, 100.0, 50.0)
    return 0.0


def _score_peg(peg: float) -> float:
    if np.isnan(peg):
        return np.nan
    if peg <= 1.0:
        return 100.0
    if peg <= 1.5:
        return interpolate(peg, 1.0, 1.5, 100.0, 50.0)
    return 0.0


def compute_valuation_score(row: pd.Series, filters: FilterSettings) -> float:
    scores = [
        _score_pe(row.get("pe_ttm", np.nan)),
        _score_pe(row.get("pe_fwd", np.nan)),
        _score_peg(row.get("peg", np.nan)),
    ]
    valid = [score for score in scores if not np.isnan(score)]
    if not valid:
        return np.nan
    return float(np.mean(valid))


def compute_growth_score(row: pd.Series, filters: FilterSettings) -> float:
    eps = row.get("eps_cagr", np.nan)
    if np.isnan(eps):
        return np.nan
    if eps >= 0.15:
        return 100.0
    if eps >= 0.05:
        return interpolate(eps, 0.05, 0.15, 50.0, 100.0)
    if eps > 0:
        return interpolate(eps, 0.0, 0.05, 0.0, 50.0)
    return 0.0


def compute_balance_sheet_score(row: pd.Series, filters: FilterSettings) -> float:
    de_ratio = row.get("debt_to_equity", np.nan)
    if np.isnan(de_ratio):
        return np.nan
    if de_ratio <= 0.35:
        return 100.0
    if de_ratio <= 0.70:
        return interpolate(de_ratio, 0.35, 0.70, 100.0, 50.0)
    if de_ratio <= 1.50:
        return interpolate(de_ratio, 0.70, 1.50, 50.0, 0.0)
    return 0.0


def compute_size_score(row: pd.Series, filters: FilterSettings) -> float:
    mcap = row.get("market_cap", np.nan)
    if np.isnan(mcap):
        return np.nan
    if mcap >= 50e9:
        return 100.0
    if mcap >= 5e9:
        return interpolate(mcap, 5e9, 50e9, 50.0, 100.0)
    return 0.0


def compute_garp_score(row: pd.Series, filters: FilterSettings, weights: WeightSettings) -> float:
    if not row.get("data_ready_nexus", False):
        return 0.0
    valuation = compute_valuation_score(row, filters)
    growth = compute_growth_score(row, filters)
    balance = compute_balance_sheet_score(row, filters)
    size = compute_size_score(row, filters)
    subscores = []
    garp_weights = []
    for score_value, weight_value in (
        (valuation, weights.garp_valuation),
        (growth, weights.garp_growth),
        (balance, weights.garp_balance_sheet),
        (size, weights.garp_size),
    ):
        if np.isnan(score_value):
            continue
        subscores.append(score_value)
        garp_weights.append(float(weight_value))
    if not subscores:
        return 0.0
    weights_arr = np.array(garp_weights, dtype=float)
    if np.all(weights_arr <= 0):
        weights_arr = np.ones_like(weights_arr)
    weights_arr /= weights_arr.sum()
    subscores_arr = np.array(subscores, dtype=float)
    return float(np.clip(np.dot(weights_arr, subscores_arr), 0.0, 100.0))


def compute_risk_score(row: pd.Series, filters: FilterSettings, weights: WeightSettings) -> float:
    if not row.get("data_ready_nexus", False):
        return np.nan
    vol = row.get("vol_1y", np.nan)
    mdd = row.get("mdd_1y", np.nan)
    adv = row.get("adv_3m", np.nan)

    def score_volatility(value: float) -> float:
        if np.isnan(value):
            return np.nan
        if value <= 0.15:
            return 0.0
        if value <= 0.30:
            return interpolate(value, 0.15, 0.30, 20.0, 70.0)
        return 100.0

    def score_drawdown(value: float) -> float:
        if np.isnan(value):
            return np.nan
        depth = abs(value)
        if depth <= 0.10:
            return 0.0
        if depth <= 0.30:
            return interpolate(depth, 0.10, 0.30, 30.0, 70.0)
        return 100.0

    def score_liquidity(value: float) -> float:
        if np.isnan(value):
            return np.nan
        if value >= 5_000_000:
            return 0.0
        if value >= 1_000_000:
            return interpolate(value, 1_000_000, 5_000_000, 60.0, 20.0)
        return 100.0

    subscores = []
    weights_arr = []
    for subscore, weight in (
        (score_volatility(vol), 0.40),
        (score_drawdown(mdd), 0.40),
        (score_liquidity(adv), 0.20),
    ):
        if not np.isnan(subscore):
            subscores.append(subscore)
            weights_arr.append(weight)
    if not subscores:
        return np.nan
    w = np.array(weights_arr, dtype=float)
    w /= w.sum()
    subs = np.array(subscores, dtype=float)
    return float(np.clip(np.dot(w, subs), 0.0, 100.0))


def compute_safety_score(row: pd.Series, filters: FilterSettings, weights: WeightSettings) -> float:
    risk_score = row.get("risk_score", np.nan)
    if np.isnan(risk_score):
        return np.nan
    return float(np.clip(100.0 - risk_score, 0.0, 100.0))


def compute_nexus_score(row: pd.Series, filters: FilterSettings, weights: WeightSettings) -> float:
    if not row.get("data_ready_nexus", False):
        return 0.0
    garp_score = row.get("garp_score", np.nan)
    safety_score = row.get("safety_score", np.nan)
    if np.isnan(garp_score) and np.isnan(safety_score):
        return 0.0
    quality_weight = max(float(weights.nexus_quality), 0.0)
    safety_weight = max(float(weights.nexus_safety), 0.0)
    if np.isnan(garp_score):
        return float(np.clip(safety_score, 0.0, 100.0))
    if np.isnan(safety_score):
        return float(np.clip(garp_score, 0.0, 100.0))
    total = quality_weight + safety_weight
    if total <= 0:
        total = 1.0
        quality_weight = safety_weight = 0.5
    blended = (quality_weight * garp_score + safety_weight * safety_score) / total
    return float(np.clip(blended, 0.0, 100.0))


def compute_v1_1_score(row: pd.Series, filters: FilterSettings) -> int:
    if not row.get("universe_ok", False):
        return 0
    if not row.get("data_complete_v1_1", False):
        return 0

    score = 0
    pe_ttm = row.get("pe_ttm", np.nan)
    pe_fwd = row.get("pe_fwd", np.nan)
    de_ratio = row.get("debt_to_equity", np.nan)
    eps_cagr = row.get("eps_cagr", np.nan)
    peg = row.get("peg", np.nan)

    if not np.isnan(pe_ttm) and pe_ttm < filters.max_pe_ttm:
        score += 1
    if not np.isnan(pe_fwd) and pe_fwd < filters.max_forward_pe:
        score += 1
    if not np.isnan(de_ratio) and de_ratio < filters.max_debt_to_equity:
        score += 1
    min_eps = getattr(filters, "min_eps_cagr_v1_1", filters.min_eps_cagr)
    if not np.isnan(eps_cagr) and eps_cagr > min_eps:
        score += 1
    if not np.isnan(peg) and peg > 0 and peg < filters.max_peg:
        score += 1
    return score
