"""Configuration helpers for the OroTitan Nexus screener."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - fallback when PyYAML is missing
    yaml = None

LOGGER = logging.getLogger(__name__)

DEFAULT_CAC40_TICKERS: List[str] = [
    "AI.PA",
    "AIR.PA",
    "ALO.PA",
    "ORA.PA",
    "BNP.PA",
    "ACA.PA",
    "GLE.PA",
    "CS.PA",
    "EN.PA",
    "CAP.PA",
    "DG.PA",
    "HO.PA",
    "ML.PA",
    "MC.PA",
    "OR.PA",
    "RI.PA",
    "PUB.PA",
    "VIE.PA",
    "VIV.PA",
    "SU.PA",
    "DSY.PA",
    "KER.PA",
    "EL.PA",
    "ENGI.PA",
    "TTE.PA",
    "SAF.PA",
    "STLA.PA",
    "BN.PA",
    "UL.PA",
    "SAN.PA",
    "LR.PA",
    "RMS.PA",
    "FR.PA",
    "TEP.PA",
    "WLN.PA",
    "CA.PA",
    "SGO.PA",
    "STM.PA",
    "EDEN.PA",
    "ERF.PA",
]


@dataclass
class FilterSettings:
    """Strict GARP filter thresholds."""

    max_pe_ttm: float = 25.0
    max_forward_pe: float = 15.0
    max_debt_to_equity: float = 0.35
    min_eps_cagr: float = 0.08
    min_eps_cagr_v1_1: float = 0.15
    max_peg: float = 1.2
    min_market_cap: float = 5e9


@dataclass
class WeightSettings:
    """Weights for sub-scores and Nexus blending."""

    garp_valuation: float = 0.30
    garp_growth: float = 0.30
    garp_balance_sheet: float = 0.25
    garp_size: float = 0.15
    nexus_quality: float = 0.65
    nexus_safety: float = 0.35


@dataclass
class UniverseSettings:
    """Universe and liquidity constraints."""

    tickers: List[str] = field(default_factory=lambda: list(DEFAULT_CAC40_TICKERS))
    min_market_cap: float = 1e9
    min_adv_3m: float = 100_000.0


@dataclass
class ProfileSettings:
    """Representation of the active Nexus profile."""

    name: Optional[str] = None


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML config file if available."""

    if not path:
        return {}
    if yaml is None:
        LOGGER.warning("PyYAML is not installed; ignoring --config=%s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                LOGGER.warning("Configuration file %s must define a mapping", path)
                return {}
            return data
    except FileNotFoundError:
        LOGGER.warning("Configuration file %s not found; using defaults", path)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to parse %s: %s", path, exc)
    return {}


def _update_dataclass(instance: Any, data: Dict[str, Any]) -> None:
    """Assign known fields from ``data`` onto ``instance``."""

    if not data:
        return
    valid_fields = {field_.name for field_ in fields(instance)}
    for key, value in data.items():
        if key in valid_fields:
            setattr(instance, key, value)


def normalize_garp_weights(weights: WeightSettings) -> None:
    """Ensure GARP weights sum to one while keeping proportions."""

    parts = [
        max(float(weights.garp_valuation), 0.0),
        max(float(weights.garp_growth), 0.0),
        max(float(weights.garp_balance_sheet), 0.0),
        max(float(weights.garp_size), 0.0),
    ]
    total = sum(parts)
    if total <= 0:
        weights.garp_valuation = weights.garp_growth = weights.garp_balance_sheet = weights.garp_size = 0.25
        return
    weights.garp_valuation = parts[0] / total
    weights.garp_growth = parts[1] / total
    weights.garp_balance_sheet = parts[2] / total
    weights.garp_size = parts[3] / total


def normalize_nexus_weights(weights: WeightSettings) -> None:
    """Ensure Nexus weights sum to one."""

    quality = max(float(weights.nexus_quality), 0.0)
    safety = max(float(weights.nexus_safety), 0.0)
    total = quality + safety
    if total <= 0:
        weights.nexus_quality = weights.nexus_safety = 0.5
        return
    weights.nexus_quality = quality / total
    weights.nexus_safety = safety / total


def apply_profile_overrides(
    filters: FilterSettings, weights: WeightSettings, profile_name: Optional[str]
) -> None:
    """Mutate ``filters``/``weights`` according to the requested profile."""

    if not profile_name or profile_name == "balanced":
        normalize_garp_weights(weights)
        normalize_nexus_weights(weights)
        return

    if profile_name == "defensive":
        filters.max_pe_ttm = min(filters.max_pe_ttm, 20.0)
        filters.max_forward_pe = min(filters.max_forward_pe, 13.0)
        filters.max_debt_to_equity = min(filters.max_debt_to_equity, 0.30)
        filters.min_eps_cagr = max(filters.min_eps_cagr, 0.10)
        filters.min_eps_cagr_v1_1 = max(filters.min_eps_cagr_v1_1, 0.15)
        weights.nexus_quality = 0.55
        weights.nexus_safety = 0.45
        weights.garp_balance_sheet = 0.30
        weights.garp_size = 0.20
        weights.garp_valuation = 0.30
        weights.garp_growth = 0.20
    elif profile_name == "offensive":
        filters.max_pe_ttm = max(filters.max_pe_ttm, 30.0)
        filters.max_forward_pe = max(filters.max_forward_pe, 18.0)
        filters.max_debt_to_equity = max(filters.max_debt_to_equity, 0.50)
        filters.min_eps_cagr = min(filters.min_eps_cagr, 0.05)
        filters.min_eps_cagr_v1_1 = min(filters.min_eps_cagr_v1_1, 0.10)
        weights.nexus_quality = 0.75
        weights.nexus_safety = 0.25
        weights.garp_valuation = 0.25
        weights.garp_growth = 0.35
        weights.garp_balance_sheet = 0.20
        weights.garp_size = 0.20
    else:  # pragma: no cover - unsupported profile safeguard
        LOGGER.warning("Unknown profile '%s'; using defaults", profile_name)

    normalize_garp_weights(weights)
    normalize_nexus_weights(weights)


def build_settings_from_config(
    config_data: Dict[str, Any], profile_name: Optional[str]
) -> Tuple[FilterSettings, WeightSettings, UniverseSettings, ProfileSettings]:
    """Instantiate settings from YAML + profile overrides."""

    filters = FilterSettings()
    weights = WeightSettings()
    universe = UniverseSettings()

    if config_data:
        filters_data = config_data.get("filters", {})
        _update_dataclass(filters, filters_data)

        weights_data = config_data.get("weights", {})
        _update_dataclass(weights, weights_data)
        garp_weights = weights_data.get("garp", {})
        for attr, key in (
            ("garp_valuation", "valuation"),
            ("garp_growth", "growth"),
            ("garp_balance_sheet", "balance_sheet"),
            ("garp_size", "size"),
        ):
            if key in garp_weights:
                setattr(weights, attr, garp_weights[key])
        nexus_weights = weights_data.get("nexus", {})
        for attr, key in (("nexus_quality", "quality"), ("nexus_safety", "safety")):
            if key in nexus_weights:
                setattr(weights, attr, nexus_weights[key])

        universe_data = config_data.get("universe", {})
        tickers = universe_data.get("tickers")
        if tickers:
            universe.tickers = list(tickers)
        if "min_market_cap" in universe_data:
            universe.min_market_cap = float(universe_data["min_market_cap"])
        if "min_adv_3m" in universe_data:
            universe.min_adv_3m = float(universe_data["min_adv_3m"])

    apply_profile_overrides(filters, weights, profile_name)
    normalize_garp_weights(weights)
    normalize_nexus_weights(weights)

    profile = ProfileSettings(name=profile_name)
    return filters, weights, universe, profile
