"""Nexus Playbook engine: actions, rationales, and sizing hints."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PlaybookSettings, ProfileSettingsV2
from .execution import (
    compute_breakout_entry,
    compute_initial_stop,
    compute_position_size,
    compute_pullback_entry,
)


@dataclass
class TickerDecision:
    ticker: str
    action: str
    rationale: List[str]
    core_score: Optional[float]
    v2_score: Optional[float]
    exceptionality: Optional[float]
    upside_pct: Optional[float]
    momentum_score: Optional[float]
    risk_score: Optional[float]
    etf_nexus_score: Optional[float]
    is_owned: bool
    is_etf: bool
    sector: Optional[str] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    size_qty: Optional[int] = None
    size_eur: Optional[float] = None


@dataclass
class PlaybookSummary:
    profile_name: str
    universe_name: str
    run_id: Optional[str]
    date: Optional[str]
    portfolio_hhi: Optional[float]
    portfolio_top5_weight: Optional[float]
    portfolio_hhi_zone: Optional[str]
    portfolio_top5_zone: Optional[str]
    total_universe: int
    total_strict_garp: int
    total_core_enabled: int
    total_owned: int
    decisions: List[TickerDecision]
    counts_by_action: Dict[str, int]
    top_by_core: List[str]
    top_by_v2: List[str]
    top_by_upside: List[str]
    top_by_etf: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""

        return {
            "profile_name": self.profile_name,
            "universe_name": self.universe_name,
            "run_id": self.run_id,
            "date": self.date,
            "portfolio_hhi": self.portfolio_hhi,
            "portfolio_top5_weight": self.portfolio_top5_weight,
            "portfolio_hhi_zone": self.portfolio_hhi_zone,
            "portfolio_top5_zone": self.portfolio_top5_zone,
            "total_universe": self.total_universe,
            "total_strict_garp": self.total_strict_garp,
            "total_core_enabled": self.total_core_enabled,
            "total_owned": self.total_owned,
            "counts_by_action": dict(self.counts_by_action),
            "top_by_core": list(self.top_by_core),
            "top_by_v2": list(self.top_by_v2),
            "top_by_upside": list(self.top_by_upside),
            "top_by_etf": list(self.top_by_etf),
            "decisions": [asdict(decision) for decision in self.decisions],
        }


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _core_score(row: pd.Series) -> float:
    return _as_float(row.get("nexus_core_score", np.nan))


def _momentum_score(row: pd.Series) -> float:
    if "momentum_score" in row:
        return _as_float(row.get("momentum_score"))
    if "nexus_core_m" in row:  # fallback if pillar names are used
        return _as_float(row.get("nexus_core_m"))
    return np.nan


def _risk_score(row: pd.Series) -> float:
    if "risk_score" in row:
        return _as_float(row.get("risk_score"))
    if "nexus_core_r" in row:
        return _as_float(row.get("nexus_core_r"))
    return np.nan


def classify_action(row: pd.Series, profile: ProfileSettingsV2) -> tuple[str, List[str]]:
    """Return (action, rationale tags) based on playbook thresholds.

    The logic is intentionally simple and config-driven to keep behavior
    predictable. It never alters existing scores; it only maps scores to
    human-readable actions.
    """

    thresholds = profile.playbook.thresholds
    action = profile.playbook.label_watch
    rationale: List[str] = []

    core = _core_score(row)
    v2_score = _as_float(row.get("nexus_v2_score", np.nan))
    exceptionality = _as_float(row.get("nexus_exceptionality", np.nan))
    upside = _as_float(row.get("upside_pct", np.nan))
    momentum = _momentum_score(row)
    risk = _risk_score(row)
    etf_score = _as_float(row.get("etf_nexus_score", np.nan))

    is_etf = bool(row.get("ETF_bool", False))
    owned = bool(row.get("owned", row.get("owned_bool", False)))

    if is_etf:
        if not np.isnan(etf_score) and etf_score >= thresholds.min_core_score_buy:
            action = profile.playbook.label_buy if not owned else profile.playbook.label_hold
            rationale.append("ETF_CORE")
        else:
            action = profile.playbook.label_avoid
            rationale.append("ETF_WEAK")
        if owned:
            rationale.append("OWNED")
        else:
            rationale.append("ETF_CANDIDATE")
        return action, rationale

    if row.get("Red_Flag", False):
        return profile.playbook.label_red_flag, ["RED_FLAG"]

    strict_pass = bool(row.get("strict_pass_garp", False))

    if owned:
        rationale.append("OWNED")
        if (
            (not np.isnan(core) and core >= thresholds.min_core_score_add)
            and (np.isnan(v2_score) or v2_score >= thresholds.min_v2_score_add)
            and (np.isnan(upside) or upside >= thresholds.min_upside_add_pct)
        ):
            action = profile.playbook.label_add
            rationale.append("ADD_THRESHOLD")
        elif (not np.isnan(core) and core >= thresholds.min_core_score_hold) and (
            np.isnan(upside) or upside >= thresholds.max_tolerable_downside_hold_pct
        ):
            action = profile.playbook.label_hold
            rationale.append("HOLD_THRESHOLD")
        else:
            action = profile.playbook.label_watch
            rationale.append("DOWNGRADE")
    else:
        criteria_core = not np.isnan(core) and core >= thresholds.min_core_score_buy
        criteria_v2 = np.isnan(v2_score) or v2_score >= thresholds.min_v2_score_buy
        criteria_upside = not np.isnan(upside) and upside >= thresholds.min_upside_buy_pct
        criteria_momentum = np.isnan(momentum) or momentum >= thresholds.min_momentum_score_buy
        criteria_exceptional = np.isnan(exceptionality) or exceptionality >= thresholds.min_exceptionality_buy
        if strict_pass:
            rationale.append("GARP_STRICT_PASS")
        if criteria_core:
            rationale.append("HIGH_CORE")
        if criteria_upside:
            rationale.append("HIGH_UPSIDE")
        if criteria_momentum:
            rationale.append("MOMENTUM_OK")

        if criteria_core and criteria_v2 and criteria_upside and criteria_momentum and criteria_exceptional:
            action = profile.playbook.label_buy
        elif (not np.isnan(core) and core >= thresholds.min_core_score_add) and criteria_upside:
            action = profile.playbook.label_watch
        else:
            action = profile.playbook.label_avoid

    if not np.isnan(risk) and risk < thresholds.min_core_score_hold:
        rationale.append("HIGH_RISK")
    return action, rationale


def _safe_adv_eur(row: pd.Series) -> float:
    for key in ("AVG_DVOL_20D_EUR", "avg_dvol_20d_eur", "adv_3m"):
        if key in row and not pd.isna(row.get(key)):
            return float(row.get(key))
    return np.nan


def _compute_execution_hints(
    row: pd.Series,
    profile: ProfileSettingsV2,
    portfolio_context: Optional[dict],
) -> tuple[Optional[float], Optional[float], Optional[int], Optional[float], List[str]]:
    """Return entry, stop, qty, size€ plus extra rationale tags."""

    price = _as_float(row.get("price", row.get("Price", np.nan)))
    atr = _as_float(row.get("ATR14", np.nan))
    adv_eur = _safe_adv_eur(row)
    sector = row.get("sector")
    is_etf = bool(row.get("ETF_bool", False))
    extra_tags: List[str] = []

    entry = None
    stop = None
    qty: Optional[int] = None
    size_eur: Optional[float] = None

    if np.isnan(price) or price <= 0 or np.isnan(atr):
        return entry, stop, qty, size_eur, extra_tags

    entry_row = row
    entry = compute_pullback_entry(entry_row, profile.entry)
    if entry is None:
        entry = compute_breakout_entry(entry_row, profile.entry)

    stop = compute_initial_stop(entry_row, profile.stops, sector=sector or "", is_etf=is_etf)

    budget = profile.playbook.default_budget_per_line_eur
    portfolio_value = None
    if portfolio_context:
        portfolio_value = portfolio_context.get("portfolio_value")
        budget = float(portfolio_context.get("budget_eur", budget))
    risk_fraction = profile.playbook.default_risk_fraction_per_trade
    risk_eur = (portfolio_value * risk_fraction) if portfolio_value else budget * risk_fraction

    if entry is not None and stop is not None and entry > stop and risk_eur > 0:
        qty = compute_position_size(
            entry=entry,
            stop=stop,
            budget_eur=budget,
            risk_eur=risk_eur,
            adv20_eur=adv_eur,
            sizing=profile.sizing,
        )
        if qty <= 0:
            qty = None
            extra_tags.append("SIZE_TOO_SMALL")
        else:
            size_eur = qty * entry
            if not np.isnan(adv_eur) and size_eur > profile.sizing.max_adv_fraction * adv_eur:
                extra_tags.append("ADV_CAP_REACHED")
    return entry, stop, qty, size_eur, extra_tags


def run_nexus_playbook(
    df: pd.DataFrame,
    profile: PlaybookSettings | ProfileSettingsV2,
    universe_name: str,
    run_id: Optional[str] = None,
    date_str: Optional[str] = None,
    portfolio_context: Optional[dict] = None,
) -> PlaybookSummary:
    """Map scored rows to actions and aggregate a PlaybookSummary."""

    if isinstance(profile, ProfileSettingsV2):
        playbook_cfg = profile.playbook
    else:
        playbook_cfg = profile  # type: ignore[assignment]

    decisions: List[TickerDecision] = []
    for _, row in df.iterrows():
        action, rationale = classify_action(row, profile if isinstance(profile, ProfileSettingsV2) else profile)  # type: ignore[arg-type]
        entry = stop = size_eur = None
        qty = None
        if action in (playbook_cfg.label_buy, playbook_cfg.label_add):
            entry, stop, qty, size_eur, extra_tags = _compute_execution_hints(row, profile if isinstance(profile, ProfileSettingsV2) else profile, portfolio_context)  # type: ignore[arg-type]
            rationale.extend(extra_tags)
        decision = TickerDecision(
            ticker=str(row.get("ticker")),
            action=action,
            rationale=rationale,
            core_score=_core_score(row),
            v2_score=_as_float(row.get("nexus_v2_score", np.nan)),
            exceptionality=_as_float(row.get("nexus_exceptionality", np.nan)),
            upside_pct=_as_float(row.get("upside_pct", np.nan)),
            momentum_score=_momentum_score(row),
            risk_score=_risk_score(row),
            etf_nexus_score=_as_float(row.get("etf_nexus_score", np.nan)),
            is_owned=bool(row.get("owned", row.get("owned_bool", False))),
            is_etf=bool(row.get("ETF_bool", False)),
            sector=row.get("sector"),
            entry_price=entry,
            stop_price=stop,
            size_qty=qty,
            size_eur=size_eur,
        )
        decisions.append(decision)

    counts: Dict[str, int] = {}
    for decision in decisions:
        counts[decision.action] = counts.get(decision.action, 0) + 1

    owned_count = sum(1 for d in decisions if d.is_owned)
    top_by_core = [
        d.ticker
        for d in sorted(decisions, key=lambda d: (-(d.core_score if d.core_score is not None else -np.inf)))
        if d.core_score is not None and not np.isnan(d.core_score)
    ][:5]
    top_by_v2 = [
        d.ticker
        for d in sorted(decisions, key=lambda d: (-(d.v2_score if d.v2_score is not None else -np.inf)))
        if d.v2_score is not None and not np.isnan(d.v2_score)
    ][:5]
    top_by_upside = [
        d.ticker
        for d in sorted(decisions, key=lambda d: (-(d.upside_pct if d.upside_pct is not None else -np.inf)))
        if d.upside_pct is not None and not np.isnan(d.upside_pct)
    ][:5]
    top_by_etf = [
        d.ticker
        for d in sorted(decisions, key=lambda d: (-(d.etf_nexus_score if d.etf_nexus_score is not None else -np.inf)))
        if d.is_etf and d.etf_nexus_score is not None and not np.isnan(d.etf_nexus_score)
    ][:5]

    summary = PlaybookSummary(
        profile_name=getattr(profile, "name", ""),
        universe_name=universe_name,
        run_id=run_id,
        date=date_str,
        portfolio_hhi=portfolio_context.get("portfolio_hhi") if portfolio_context else None,
        portfolio_top5_weight=portfolio_context.get("portfolio_top5_weight") if portfolio_context else None,
        portfolio_hhi_zone=portfolio_context.get("portfolio_hhi_zone") if portfolio_context else None,
        portfolio_top5_zone=portfolio_context.get("portfolio_top5_zone") if portfolio_context else None,
        total_universe=int(len(df)),
        total_strict_garp=int(df.get("strict_pass_garp", pd.Series([], dtype=bool)).sum()) if "strict_pass_garp" in df else 0,
        total_core_enabled=int(df.get("nexus_core_score", pd.Series([], dtype=float)).notna().sum()),
        total_owned=owned_count,
        decisions=decisions,
        counts_by_action=counts,
        top_by_core=top_by_core,
        top_by_v2=top_by_v2,
        top_by_upside=top_by_upside,
        top_by_etf=top_by_etf,
    )
    return summary

