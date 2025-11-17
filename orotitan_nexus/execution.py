"""Entry/stop/sizing primitives for Nexus execution layer."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import EntrySettings, StopSettings, SizingSettings


def compute_breakout_entry(price_row, settings: EntrySettings) -> Optional[float]:
    """Return breakout limit price if volume/ATR/PH20 conditions are met."""

    price = float(price_row.get("price", np.nan))
    ph20 = float(price_row.get("PH20", np.nan))
    atr = float(price_row.get("ATR14", np.nan))
    volume = float(price_row.get("Volume", np.nan))
    adv20 = float(price_row.get("ADV20_shares", np.nan))

    if any(np.isnan(x) for x in (price, ph20, atr, volume, adv20)):
        return None
    if adv20 <= 0:
        return None
    if volume < settings.breakout_min_volume_factor * adv20:
        return None

    ph20_limit = ph20 * (1.0 + settings.breakout_ph20_buffer)
    atr_limit = price + settings.breakout_atr_multiplier * atr
    entry_price = min(ph20_limit, atr_limit)
    return float(entry_price)


def compute_pullback_entry(price_row, settings: EntrySettings) -> Optional[float]:
    """Return AVWAP pullback entry if data is present."""

    avwap = float(price_row.get("AVWAP_event_price", np.nan))
    atr = float(price_row.get("ATR14", np.nan))
    if any(np.isnan(x) for x in (avwap, atr)):
        return None
    return float(avwap - settings.pullback_awap_atr * atr)


def compute_initial_stop(price_row, settings: StopSettings, sector: str, is_etf: bool) -> Optional[float]:
    """Sector/asset-based initial stop price."""

    price = float(price_row.get("price", np.nan))
    atr = float(price_row.get("ATR14", np.nan))
    if np.isnan(price):
        return None

    if is_etf:
        drawdown = (settings.etf_min_drawdown + settings.etf_max_drawdown) / 2.0
        return float(price * (1.0 - drawdown))

    if np.isnan(atr):
        return None

    sector_lower = (sector or "").lower()
    if "tech" in sector_lower:
        mult = settings.tech_atr
    elif "energy" in sector_lower:
        mult = settings.energy_atr
    elif "health" in sector_lower:
        mult = (settings.health_min_atr + settings.health_max_atr) / 2.0
    else:
        mult = settings.quality_atr
    return float(price - mult * atr)


def compute_position_size(
    entry: float,
    stop: float,
    budget_eur: float,
    risk_eur: float,
    adv20_eur: float,
    sizing: SizingSettings,
) -> int:
    """Return integer quantity respecting budget/risk and ADV caps."""

    if any(np.isnan(x) for x in (entry, stop, budget_eur, risk_eur)):
        return 0
    if entry <= 0 or stop >= entry:
        return 0
    qty_budget = budget_eur / entry
    qty_risk = risk_eur / (entry - stop)
    qty = math.floor(min(qty_budget, qty_risk))
    if qty <= 0:
        return 0

    cost = qty * entry
    if cost < sizing.min_line_eur:
        return 0

    if adv20_eur > 0:
        max_qty_adv = sizing.max_adv_fraction * adv20_eur / entry
        qty = min(qty, math.floor(max_qty_adv)) if max_qty_adv > 0 else qty

    return max(int(qty), 0)

