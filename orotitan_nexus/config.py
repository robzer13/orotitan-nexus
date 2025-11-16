"""Configuration helpers for the OroTitan Nexus screener."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Literal

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

    name: str = "CAC40"
    tickers: List[str] = field(default_factory=lambda: list(DEFAULT_CAC40_TICKERS))
    min_market_cap: float = 1e9
    min_adv_3m: float = 100_000.0


@dataclass
class GarpThresholds:
    """Five-rule CAC 40 GARP thresholds."""

    pe_ttm_max: float = 25.0
    pe_fwd_max: float = 15.0
    debt_to_equity_max: float = 0.35
    eps_cagr_min: float = 0.15
    peg_max: float = 1.2


@dataclass
class ProfileSettings:
    """Representation of the active Nexus profile."""

    name: Optional[str] = None
    garp: GarpThresholds = field(default_factory=GarpThresholds)


# --- Nexus v2 settings ----------------------------------------------------


@dataclass
class QualitySettings:
    """Quality factor knobs for Nexus v2 (profitability/balance sheet)."""

    roe_weight: float = 0.4
    roa_weight: float = 0.3
    margin_weight: float = 0.3
    min_roe: float = 0.0
    min_margin: float = 0.0


@dataclass
class MomentumSettings:
    """Momentum factor lookbacks/weights for Nexus v2."""

    lookback_short_days: int = 63
    lookback_medium_days: int = 126
    lookback_long_days: int = 252
    short_weight: float = 0.3
    medium_weight: float = 0.3
    long_weight: float = 0.4


@dataclass
class RiskSettings:
    """Risk factor thresholds for Nexus v2 (volatility/drawdown)."""

    lookback_days: int = 252
    max_vol: float = 0.4
    max_drawdown: float = 0.5
    vol_weight: float = 0.6
    dd_weight: float = 0.4


@dataclass
class MacroSettings:
    """Minimal macro stub for v2 (placeholder for future logic)."""

    default_score: float = 50.0


@dataclass
class BehavioralSettings:
    """Behavioral heuristics for v2 (light penalties/bonuses)."""

    max_sector_weight: float = 0.4
    concentration_penalty: float = 10.0
    knife_catch_penalty: float = 5.0


@dataclass
class NexusV2Settings:
    """Weights and bucket thresholds for Nexus v2 global score."""

    garp_weight: float = 0.40
    quality_weight: float = 0.25
    momentum_weight: float = 0.20
    risk_weight: float = 0.10
    macro_weight: float = 0.05
    behavioral_weight: float = 0.0
    elite_min: float = 85.0
    strong_min: float = 70.0
    neutral_min: float = 50.0
    weak_min: float = 30.0
    enabled: bool = False


@dataclass
class CalibrationSettings:
    """Grid-search controls for v2 weight tuning (opt-in)."""

    enabled: bool = False
    score_column: str = "nexus_v2_score"
    benchmark_column: str = "benchmark_return"
    horizon_days: int = 252
    objective: str = "top_bucket_excess"
    garp_weight_grid: tuple = (0.3, 0.4, 0.5)
    quality_weight_grid: tuple = (0.2, 0.25, 0.3)
    momentum_weight_grid: tuple = (0.15, 0.2, 0.25)
    risk_weight_grid: tuple = (0.05, 0.1, 0.15)
    macro_weight_grid: tuple = (0.0, 0.05)
    behavioral_weight_grid: tuple = (0.0,)
    normalize_weights: bool = True
    max_turnover_penalty: float = 0.0


@dataclass
class WalkForwardSettings:
    enabled: bool = False
    n_splits: int = 4
    min_period_days: int = 252
    test_horizon_days: int = 252
    score_column: str = "nexus_v2_score"
    top_bucket_only: bool = True


@dataclass
class SensitivitySettings:
    enabled: bool = False
    weight_perturbation_pct: float = 0.1
    garp_threshold_perturbation_pct: float = 0.1
    n_random_draws: int = 16
    score_column: str = "nexus_v2_score"
    stability_metric: str = "spearman_corr"
    top_k: int = 20


@dataclass
class RegimeSettings:
    enabled: bool = False
    benchmark_ticker: str = "^FCHI"
    lookback_days: int = 63
    bull_threshold: float = 0.05
    bear_threshold: float = -0.05
    neutral_label: str = "NEUTRAL"
    bull_label: str = "BULL"
    bear_label: str = "BEAR"


@dataclass
class ProfileSettingsV2(ProfileSettings):
    """Profile including Nexus v2 controls (keeps backward compatibility)."""

    quality: QualitySettings = field(default_factory=QualitySettings)
    momentum: MomentumSettings = field(default_factory=MomentumSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    macro: MacroSettings = field(default_factory=MacroSettings)
    behavioral: BehavioralSettings = field(default_factory=BehavioralSettings)
    v2: NexusV2Settings = field(default_factory=NexusV2Settings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    walkforward: WalkForwardSettings = field(default_factory=WalkForwardSettings)
    sensitivity: SensitivitySettings = field(default_factory=SensitivitySettings)
    regime: RegimeSettings = field(default_factory=RegimeSettings)


def make_inline_universe(
    name: str,
    tickers: List[str],
    template: Optional[UniverseSettings] = None,
) -> UniverseSettings:
    """Return a ``UniverseSettings`` instance for runtime/custom universes."""

    cleaned = [ticker.strip() for ticker in tickers if ticker and ticker.strip()]
    if not cleaned:
        raise ConfigError("Custom universe requires at least one valid ticker")

    base = template or UniverseSettings()
    return UniverseSettings(
        name=name or base.name,
        tickers=cleaned,
        min_market_cap=base.min_market_cap,
        min_adv_3m=base.min_adv_3m,
    )


class ConfigError(ValueError):
    """Raised when the provided YAML configuration is invalid."""


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
) -> Tuple[FilterSettings, WeightSettings, UniverseSettings, ProfileSettingsV2]:
    """Instantiate settings from YAML + profile overrides."""

    filters = FilterSettings()
    weights = WeightSettings()
    universe = UniverseSettings()
    garp_thresholds = GarpThresholds()

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
        if "name" in universe_data:
            universe.name = str(universe_data["name"])
        if "min_market_cap" in universe_data:
            universe.min_market_cap = float(universe_data["min_market_cap"])
        if "min_adv_3m" in universe_data:
            universe.min_adv_3m = float(universe_data["min_adv_3m"])

        profile_section = config_data.get("profile", {})
        profile_config_name = profile_section.get("name")
        garp_data = profile_section.get("garp", {})
        if isinstance(garp_data, dict):
            _update_dataclass(garp_thresholds, garp_data)
    else:
        profile_section = {}
        profile_config_name = None

    effective_profile = profile_name or profile_config_name

    apply_profile_overrides(filters, weights, effective_profile)
    normalize_garp_weights(weights)
    normalize_nexus_weights(weights)

    quality_settings = QualitySettings()
    momentum_settings = MomentumSettings()
    risk_settings = RiskSettings()
    macro_settings = MacroSettings()
    behavioral_settings = BehavioralSettings()
    v2_settings = NexusV2Settings()
    calibration_settings = CalibrationSettings()
    walkforward_settings = WalkForwardSettings()
    sensitivity_settings = SensitivitySettings()
    regime_settings = RegimeSettings()

    if isinstance(profile_section, dict):
        for section_name, instance in (
            ("quality", quality_settings),
            ("momentum", momentum_settings),
            ("risk", risk_settings),
            ("macro", macro_settings),
            ("behavioral", behavioral_settings),
            ("v2", v2_settings),
            ("calibration", calibration_settings),
            ("walkforward", walkforward_settings),
            ("sensitivity", sensitivity_settings),
            ("regime", regime_settings),
        ):
            section_data = profile_section.get(section_name, {})
            if isinstance(section_data, dict):
                _update_dataclass(instance, section_data)

    profile = ProfileSettingsV2(
        name=effective_profile,
        garp=garp_thresholds,
        quality=quality_settings,
        momentum=momentum_settings,
        risk=risk_settings,
        macro=macro_settings,
        behavioral=behavioral_settings,
        v2=v2_settings,
        calibration=calibration_settings,
        walkforward=walkforward_settings,
        sensitivity=sensitivity_settings,
        regime=regime_settings,
    )
    return filters, weights, universe, profile


def _require_positive(value: float, label: str, errors: List[str]) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a number")
        return
    if numeric <= 0:
        errors.append(f"{label} must be > 0 (got {value!r})")


def validate_settings(
    filters: FilterSettings,
    universe: UniverseSettings,
    profile: ProfileSettingsV2,
) -> None:
    """Raise ``ConfigError`` if the instantiated settings are inconsistent."""

    errors: List[str] = []

    if not universe.tickers:
        errors.append("universe.tickers must not be empty")
    if not universe.name:
        errors.append("universe.name must be provided")
    _require_positive(universe.min_market_cap, "universe.min_market_cap", errors)
    _require_positive(universe.min_adv_3m, "universe.min_adv_3m", errors)

    _require_positive(filters.max_pe_ttm, "filters.max_pe_ttm", errors)
    _require_positive(filters.max_forward_pe, "filters.max_forward_pe", errors)
    _require_positive(filters.max_debt_to_equity, "filters.max_debt_to_equity", errors)
    _require_positive(filters.min_eps_cagr_v1_1, "filters.min_eps_cagr_v1_1", errors)
    _require_positive(filters.max_peg, "filters.max_peg", errors)
    _require_positive(filters.min_market_cap, "filters.min_market_cap", errors)

    garp = profile.garp if profile else GarpThresholds()
    _require_positive(garp.pe_ttm_max, "profile.garp.pe_ttm_max", errors)
    _require_positive(garp.pe_fwd_max, "profile.garp.pe_fwd_max", errors)
    _require_positive(garp.debt_to_equity_max, "profile.garp.debt_to_equity_max", errors)
    _require_positive(garp.eps_cagr_min, "profile.garp.eps_cagr_min", errors)
    _require_positive(garp.peg_max, "profile.garp.peg_max", errors)

    if errors:
        raise ConfigError("; ".join(errors))


def load_settings(
    config_path: Optional[str],
    profile_name: Optional[str],
) -> Tuple[FilterSettings, WeightSettings, UniverseSettings, ProfileSettingsV2]:
    """Convenience helper for CLI/API layers."""

    config_data = load_yaml_config(config_path)
    filters, weights, universe, profile = build_settings_from_config(config_data, profile_name)
    validate_settings(filters, universe, profile)
    return filters, weights, universe, profile
