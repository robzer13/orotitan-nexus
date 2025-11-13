"""CAC 40 GARP-style screener with strict filters, scoring, and risk metrics.

The script downloads fundamental metrics for CAC 40 constituents via yfinance,
then applies the user-specified hard filters alongside a soft scoring model
based on valuation, growth, balance-sheet quality, and market capitalization.
In addition, a price-based risk module computes volatility, drawdown, liquidity
proxies, and a consolidated risk score so that the CSV exposes both upside and
downside perspectives.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

# ---------------------------------------------------------------------------
# Universe configuration
# ---------------------------------------------------------------------------
CAC40_TICKERS: List[str] = [
    "AI.PA",  # Air Liquide
    "AIR.PA",  # Airbus
    "ALO.PA",  # Alstom
    "ORA.PA",  # Orange
    "BNP.PA",  # BNP Paribas
    "ACA.PA",  # Crédit Agricole
    "GLE.PA",  # Société Générale
    "CS.PA",  # AXA
    "EN.PA",  # Bouygues
    "CAP.PA",  # Capgemini
    "DG.PA",  # Vinci
    "HO.PA",  # Thales
    "ML.PA",  # Michelin
    "MC.PA",  # LVMH
    "OR.PA",  # L'Oréal
    "RI.PA",  # Pernod Ricard
    "PUB.PA",  # Publicis
    "VIE.PA",  # Veolia
    "VIV.PA",  # Vivendi
    "SU.PA",  # Schneider Electric
    "DSY.PA",  # Dassault Systèmes
    "KER.PA",  # Kering
    "EL.PA",  # EssilorLuxottica
    "ENGI.PA",  # Engie
    "TTE.PA",  # TotalEnergies
    "SAF.PA",  # Safran
    "STLA.PA",  # Stellantis
    "BN.PA",  # Danone
    "UL.PA",  # Unibail-Rodamco-Westfield
    "SAN.PA",  # Sanofi
    "LR.PA",  # Legrand
    "RMS.PA",  # Hermès
    "FR.PA",  # Valeo
    "TEP.PA",  # Teleperformance
    "WLN.PA",  # Worldline
    "CA.PA",  # Carrefour
    "SGO.PA",  # Saint-Gobain
    "STM.PA",  # STMicroelectronics
    "EDEN.PA",  # Edenred
    "ERF.PA",  # Eurofins Scientific
]

MAX_PE_TTM = 25.0
MAX_FORWARD_PE = 15.0
MAX_DEBT_TO_EQUITY = 0.35  # 35 %
MIN_EPS_CAGR = 0.08  # 8 % par an par défaut
MAX_PEG = 1.2
MIN_MARKET_CAP = 5e9

QUALITY_WEIGHT = 0.65
SAFETY_WEIGHT = 0.35

STRICT_SORT_COLUMNS = ["strict_pass", "nexus_score", "garp_score", "risk_score"]
STRICT_SORT_ASCENDING = [False, False, False, True]
DEFAULT_OUTPUT = "cac40_screen_results.csv"
DEFAULT_MAX_ROWS = 40
TRADING_DAYS_PER_YEAR = 252
ADV_WINDOW_DAYS = 63  # ~3 months of trading days
PROFILE_CHOICES = ("defensive", "balanced", "offensive")

LOGGER = logging.getLogger("cac40_garp_screener")


@dataclass
class FilterSettings:
    """Container for strict GARP filter thresholds."""

    max_pe_ttm: float = MAX_PE_TTM
    max_forward_pe: float = MAX_FORWARD_PE
    max_debt_to_equity: float = MAX_DEBT_TO_EQUITY
    min_eps_cagr: float = MIN_EPS_CAGR
    max_peg: float = MAX_PEG
    min_market_cap: float = MIN_MARKET_CAP


@dataclass
class WeightSettings:
    """Weights applied to GARP sub-scores and Nexus blending."""

    garp_valuation: float = 0.30
    garp_growth: float = 0.30
    garp_balance_sheet: float = 0.25
    garp_size: float = 0.15
    nexus_quality: float = QUALITY_WEIGHT
    nexus_safety: float = SAFETY_WEIGHT


@dataclass
class Settings:
    """Aggregate settings for the screener, including universe and weights."""

    tickers: List[str] = field(default_factory=lambda: list(CAC40_TICKERS))
    filters: FilterSettings = field(default_factory=FilterSettings)
    weights: WeightSettings = field(default_factory=WeightSettings)


def normalize_garp_weights(weights: WeightSettings) -> None:
    """Force GARP weights to sum to 1.0 while preserving proportions."""

    components = [
        ("garp_valuation", max(float(weights.garp_valuation), 0.0)),
        ("garp_growth", max(float(weights.garp_growth), 0.0)),
        ("garp_balance_sheet", max(float(weights.garp_balance_sheet), 0.0)),
        ("garp_size", max(float(weights.garp_size), 0.0)),
    ]
    total = sum(value for _, value in components)
    if total <= 0:
        for attr in ("garp_valuation", "garp_growth", "garp_balance_sheet", "garp_size"):
            setattr(weights, attr, 0.25)
        return

    for attr, value in components:
        setattr(weights, attr, value / total)


def normalize_nexus_weights(weights: WeightSettings) -> None:
    """Force Nexus weights (quality/safety) to sum to 1.0."""

    quality = max(float(weights.nexus_quality), 0.0)
    safety = max(float(weights.nexus_safety), 0.0)
    total = quality + safety
    if total <= 0:
        weights.nexus_quality = weights.nexus_safety = 0.5
        return

    weights.nexus_quality = quality / total
    weights.nexus_safety = safety / total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_cac40_tickers(settings: Settings) -> List[str]:
    """Return the list of tickers to evaluate, honoring config overrides."""

    if settings.tickers:
        return settings.tickers
    return list(CAC40_TICKERS)


def safe_float(value: Optional[float]) -> float:
    """Return ``value`` as float, or NaN when conversion fails."""

    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _apply_numeric_overrides(target: Any, overrides: Dict[str, Any]) -> None:
    """Update numeric dataclass fields from ``overrides`` when possible."""

    if not isinstance(overrides, dict):
        return
    valid_fields = {field.name for field in fields(target)}
    for key, value in overrides.items():
        if key not in valid_fields:
            continue
        converted = safe_float(value)
        if np.isnan(converted):
            continue
        setattr(target, key, float(converted))


def load_config(path: Optional[str]) -> Settings:
    """Load YAML config overrides and return a populated :class:`Settings`."""

    settings = Settings()
    if not path:
        normalize_garp_weights(settings.weights)
        normalize_nexus_weights(settings.weights)
        return settings

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        LOGGER.warning("Config file %s not found. Using defaults.", path)
        return settings
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.warning("Unable to load config file %s. Using defaults.", path, exc_info=True)
        return settings

    if not isinstance(raw_config, dict):
        LOGGER.warning("Config file %s has an invalid structure. Using defaults.", path)
        return settings

    universe_cfg = raw_config.get("universe")
    if isinstance(universe_cfg, dict):
        tickers = universe_cfg.get("tickers")
        if isinstance(tickers, list) and tickers:
            clean = [str(t).strip() for t in tickers if str(t).strip()]
            if clean:
                settings.tickers = clean

    filters_cfg = raw_config.get("filters")
    if isinstance(filters_cfg, dict):
        _apply_numeric_overrides(settings.filters, filters_cfg)

    weights_cfg = raw_config.get("weights")
    if isinstance(weights_cfg, dict):
        garp_cfg = weights_cfg.get("garp")
        if isinstance(garp_cfg, dict):
            mapping = {
                "valuation": "garp_valuation",
                "growth": "garp_growth",
                "balance_sheet": "garp_balance_sheet",
                "size": "garp_size",
            }
            for cfg_key, attr in mapping.items():
                converted = safe_float(garp_cfg.get(cfg_key))
                if np.isnan(converted):
                    continue
                setattr(settings.weights, attr, float(converted))

        nexus_cfg = weights_cfg.get("nexus")
        if isinstance(nexus_cfg, dict):
            for cfg_key, attr in {"quality": "nexus_quality", "safety": "nexus_safety"}.items():
                converted = safe_float(nexus_cfg.get(cfg_key))
                if np.isnan(converted):
                    continue
                setattr(settings.weights, attr, float(converted))

    normalize_garp_weights(settings.weights)
    normalize_nexus_weights(settings.weights)
    return settings


def apply_profile_overrides(
    filters: FilterSettings, weights: WeightSettings, profile: Optional[str]
) -> None:
    """Mutate ``filters``/``weights`` according to a Nexus profile preset."""

    if not profile or profile == "balanced":
        return

    profile = profile.lower()
    if profile == "defensive":
        filters.max_pe_ttm = min(filters.max_pe_ttm, 20.0)
        filters.max_forward_pe = min(filters.max_forward_pe, 13.0)
        filters.max_debt_to_equity = min(filters.max_debt_to_equity, 0.30)
        filters.min_eps_cagr = max(filters.min_eps_cagr, 0.10)
        weights.nexus_quality = 0.55
        weights.nexus_safety = 0.45
        weights.garp_valuation = 0.30
        weights.garp_growth = 0.25
        weights.garp_balance_sheet = 0.30
        weights.garp_size = 0.15
    elif profile == "offensive":
        filters.max_pe_ttm = max(filters.max_pe_ttm, 30.0)
        filters.max_forward_pe = max(filters.max_forward_pe, 18.0)
        filters.max_debt_to_equity = max(filters.max_debt_to_equity, 0.50)
        filters.min_eps_cagr = min(filters.min_eps_cagr, 0.05)
        weights.nexus_quality = 0.75
        weights.nexus_safety = 0.25
        weights.garp_valuation = 0.25
        weights.garp_growth = 0.35
        weights.garp_balance_sheet = 0.20
        weights.garp_size = 0.20
    else:  # pragma: no cover - defensive guard for future profiles
        return

    normalize_garp_weights(weights)
    normalize_nexus_weights(weights)


def fetch_fundamental_data(ticker: str) -> Dict[str, float]:
    """Fetch Yahoo Finance info for ``ticker`` and extract required fields."""

    yf_ticker = yf.Ticker(ticker)
    info: Dict[str, Optional[float]] = yf_ticker.info or {}

    # EPS growth proxy. Documented to use `earningsGrowth` with a fallback to
    # `earningsQuarterlyGrowth` when annual data is not available.
    eps_growth = info.get("earningsGrowth")
    if eps_growth is None:
        eps_growth = info.get("earningsQuarterlyGrowth")

    debt_to_equity = safe_float(info.get("debtToEquity"))
    if not np.isnan(debt_to_equity) and debt_to_equity > 10:
        # Yahoo Finance sometimes exposes the ratio as a percentage (e.g. 45),
        # so we convert it to a proportion when the raw value looks too large.
        debt_to_equity /= 100.0

    return {
        "ticker": ticker,
        "pe_ttm": safe_float(info.get("trailingPE")),
        "pe_fwd": safe_float(info.get("forwardPE")),
        "debt_to_equity": debt_to_equity,
        "eps_cagr": safe_float(eps_growth),
        "peg": safe_float(info.get("pegRatio")),
        "market_cap": safe_float(info.get("marketCap")),
    }


def fetch_risk_data(ticker: str) -> Dict[str, float]:
    """Download OHLCV data and compute volatility, drawdown, and liquidity."""

    metrics = {"vol_1y": np.nan, "mdd_1y": np.nan, "adv_3m": np.nan}
    try:
        history = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to download price data for %s", ticker, exc_info=True)
        return metrics

    if history.empty:
        LOGGER.warning("No price history retrieved for %s", ticker)
        return metrics

    price_col = "Adj Close" if "Adj Close" in history.columns else "Close"
    prices = history[price_col].dropna()
    if prices.shape[0] >= 2:
        # Use log returns for volatility so extreme moves aggregate properly.
        log_returns = np.log(prices / prices.shift(1)).dropna()
        if not log_returns.empty:
            vol = float(log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            metrics["vol_1y"] = vol

        rolling_max = prices.cummax()
        drawdowns = prices / rolling_max - 1.0
        if not drawdowns.empty:
            metrics["mdd_1y"] = float(drawdowns.min())

    volume_series = history.get("Volume")
    if volume_series is not None:
        recent_volume = volume_series.dropna().tail(ADV_WINDOW_DAYS)
        if not recent_volume.empty:
            metrics["adv_3m"] = float(recent_volume.mean())

    return metrics


def build_dataframe(records: Iterable[Dict[str, float]], settings: Settings) -> pd.DataFrame:
    """Create a DataFrame with boolean filters and scoring columns."""

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    filters = settings.filters
    weights = settings.weights

    # Strict filter flags.
    df["per_ok"] = df["pe_ttm"].apply(
        lambda v: bool(not np.isnan(v) and v <= filters.max_pe_ttm)
    )
    df["per_forward_ok"] = df["pe_fwd"].apply(
        lambda v: bool(not np.isnan(v) and v <= filters.max_forward_pe)
    )
    df["de_ok"] = df["debt_to_equity"].apply(
        lambda v: bool(not np.isnan(v) and v <= filters.max_debt_to_equity)
    )
    df["eps_growth_ok"] = df["eps_cagr"].apply(
        lambda v: bool(not np.isnan(v) and v >= filters.min_eps_cagr)
    )
    df["peg_ok"] = df["peg"].apply(lambda v: bool(not np.isnan(v) and v <= filters.max_peg))
    df["mktcap_ok"] = df["market_cap"].apply(
        lambda v: bool(not np.isnan(v) and v >= filters.min_market_cap)
    )

    df["strict_pass"] = df.apply(lambda row: passes_hard_filters(row, filters), axis=1)

    # Soft scores.
    df["valuation_score"] = df.apply(
        lambda row: compute_valuation_score(row["pe_ttm"], row["pe_fwd"], row["peg"]),
        axis=1,
    )
    df["growth_score"] = df["eps_cagr"].apply(score_growth)
    df["balance_sheet_score"] = df["debt_to_equity"].apply(score_balance_sheet)
    df["size_score"] = df["market_cap"].apply(score_size)
    df["garp_score"] = df.apply(lambda row: compute_garp_score(row, weights), axis=1)

    # Risk metrics and score are expected to be part of the records already, but
    # ensure the columns exist even if a ticker failed to download.
    for column in ("vol_1y", "mdd_1y", "adv_3m"):
        if column not in df:
            df[column] = np.nan
    df["risk_score"] = df.apply(compute_risk_score, axis=1)
    df["safety_score"] = df["risk_score"].apply(compute_safety_score)
    df["nexus_score"] = df.apply(
        lambda row: compute_nexus_score(row["garp_score"], row["safety_score"], weights),
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------
def interpolate(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linearly interpolate ``value`` between ``(x0, y0)`` and ``(x1, y1)``."""

    if x0 == x1:
        return y1
    return y0 + (value - x0) * (y1 - y0) / (x1 - x0)


def compute_valuation_score(trailing_pe: float, forward_pe: float, peg: float) -> float:
    """Average the valuation metrics while ignoring missing values."""

    parts: List[float] = []
    if not np.isnan(trailing_pe):
        if trailing_pe <= 15:
            parts.append(100.0)
        elif trailing_pe <= 25:
            parts.append(interpolate(trailing_pe, 15, 25, 100.0, 50.0))
        else:
            parts.append(0.0)
    if not np.isnan(forward_pe):
        if forward_pe <= 15:
            parts.append(100.0)
        elif forward_pe <= 25:
            parts.append(interpolate(forward_pe, 15, 25, 100.0, 50.0))
        else:
            parts.append(0.0)
    if not np.isnan(peg):
        if peg <= 1.0:
            parts.append(100.0)
        elif peg <= 1.5:
            parts.append(interpolate(peg, 1.0, 1.5, 100.0, 50.0))
        else:
            parts.append(0.0)

    if not parts:
        return 0.0
    return float(np.mean(parts))


def score_growth(eps_growth: float) -> float:
    """Score EPS growth according to the piecewise rules."""

    if np.isnan(eps_growth):
        return np.nan
    if eps_growth >= 0.15:
        return 100.0
    if eps_growth >= 0.05:
        return interpolate(eps_growth, 0.05, 0.15, 50.0, 100.0)
    if eps_growth > 0:
        return interpolate(eps_growth, 0.0, 0.05, 0.0, 50.0)
    return 0.0


def score_balance_sheet(de_ratio: float) -> float:
    """Score Debt/Equity according to resilience thresholds."""

    if np.isnan(de_ratio):
        return np.nan
    if de_ratio <= 0.35:
        return 100.0
    if de_ratio <= 0.70:
        return interpolate(de_ratio, 0.35, 0.70, 100.0, 50.0)
    if de_ratio <= 1.50:
        return interpolate(de_ratio, 0.70, 1.50, 50.0, 0.0)
    return 0.0


def score_size(market_cap: float) -> float:
    """Score company size as a proxy for robustness."""

    if np.isnan(market_cap):
        return np.nan
    if market_cap >= 50e9:
        return 100.0
    if market_cap >= 5e9:
        return interpolate(market_cap, 5e9, 50e9, 50.0, 100.0)
    return 0.0


def passes_hard_filters(row: pd.Series, filters: FilterSettings) -> bool:
    """Return True if ``row`` satisfies the strict GARP constraints."""

    pe_ttm = row.get("pe_ttm", np.nan)
    pe_fwd = row.get("pe_fwd", np.nan)
    de = row.get("debt_to_equity", np.nan)
    eps_cagr = row.get("eps_cagr", np.nan)
    peg = row.get("peg", np.nan)
    mcap = row.get("market_cap", np.nan)

    if np.isnan(pe_ttm) or pe_ttm <= 0 or pe_ttm > filters.max_pe_ttm:
        return False
    if np.isnan(pe_fwd) or pe_fwd <= 0 or pe_fwd > filters.max_forward_pe:
        return False
    if np.isnan(de) or de > filters.max_debt_to_equity:
        return False
    if np.isnan(eps_cagr) or eps_cagr < filters.min_eps_cagr:
        return False
    if np.isnan(peg) or peg > filters.max_peg:
        return False
    if np.isnan(mcap) or mcap < filters.min_market_cap:
        return False
    return True


def compute_garp_score(row: pd.Series, weights: WeightSettings) -> float:
    """Compute the 0-100 GARP score, even if hard filters fail."""

    pe_ttm = row.get("pe_ttm", np.nan)
    pe_fwd = row.get("pe_fwd", np.nan)
    peg = row.get("peg", np.nan)
    eps_cagr = row.get("eps_cagr", np.nan)
    de = row.get("debt_to_equity", np.nan)
    mcap = row.get("market_cap", np.nan)

    pe_ttm_score = compute_valuation_score(pe_ttm, np.nan, np.nan)
    pe_fwd_score = compute_valuation_score(np.nan, pe_fwd, np.nan)
    peg_score = compute_valuation_score(np.nan, np.nan, peg)

    val_components = [
        score for score in [pe_ttm_score, pe_fwd_score, peg_score] if not np.isnan(score)
    ]
    valuation_score = float(np.mean(val_components)) if val_components else np.nan

    growth_score = score_growth(eps_cagr)
    balance_score = score_balance_sheet(de)
    size_score = score_size(mcap)

    subscores: List[float] = []
    garp_weights: List[float] = []

    for score_value, weight_value in (
        (valuation_score, weights.garp_valuation),
        (growth_score, weights.garp_growth),
        (balance_score, weights.garp_balance_sheet),
        (size_score, weights.garp_size),
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


def compute_risk_score(row: pd.Series) -> float:
    """Compute a 0-100 downside risk score (higher = riskier)."""

    vol = row.get("vol_1y", np.nan)
    mdd = row.get("mdd_1y", np.nan)
    adv = row.get("adv_3m", np.nan)

    def score_volatility(volatility: float) -> float:
        if np.isnan(volatility):
            return np.nan
        if volatility <= 0.15:
            return 0.0
        if volatility <= 0.30:
            return interpolate(volatility, 0.15, 0.30, 20.0, 70.0)
        return 100.0

    def score_drawdown(drawdown: float) -> float:
        if np.isnan(drawdown):
            return np.nan
        depth = abs(drawdown)
        if depth <= 0.10:
            return 0.0
        if depth <= 0.30:
            return interpolate(depth, 0.10, 0.30, 30.0, 70.0)
        return 100.0

    def score_liquidity(avg_volume: float) -> float:
        if np.isnan(avg_volume):
            return np.nan
        if avg_volume >= 5_000_000:
            return 0.0
        if avg_volume >= 1_000_000:
            return interpolate(avg_volume, 1_000_000, 5_000_000, 60.0, 20.0)
        return 100.0

    vol_score = score_volatility(vol)
    dd_score = score_drawdown(mdd)
    liq_score = score_liquidity(adv)

    subscores: List[float] = []
    weights: List[float] = []

    for subscore, weight in (
        (vol_score, 0.40),
        (dd_score, 0.40),
        (liq_score, 0.20),
    ):
        if not np.isnan(subscore):
            subscores.append(subscore)
            weights.append(weight)

    if not subscores:
        return np.nan

    weights_arr = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    subscores_arr = np.array(subscores, dtype=float)
    return float(np.clip(np.dot(weights_arr, subscores_arr), 0.0, 100.0))


def compute_safety_score(risk_score: float) -> float:
    """Return a 0-100 safety score as the inverse of ``risk_score``."""

    if np.isnan(risk_score):
        return np.nan
    return float(np.clip(100.0 - risk_score, 0.0, 100.0))


def compute_nexus_score(
    garp_score: float, safety_score: float, weights: WeightSettings
) -> float:
    """Blend quality (garp) and safety scores into a 0-100 Nexus score."""

    garp_nan = np.isnan(garp_score)
    safety_nan = np.isnan(safety_score)

    if garp_nan and safety_nan:
        return 0.0
    if garp_nan:
        return float(np.clip(safety_score, 0.0, 100.0))
    if safety_nan:
        return float(np.clip(garp_score, 0.0, 100.0))

    quality_weight = max(float(weights.nexus_quality), 0.0)
    safety_weight = max(float(weights.nexus_safety), 0.0)
    total = quality_weight + safety_weight
    if total <= 0:
        quality_weight = safety_weight = 1.0
        total = 2.0

    blended = (quality_weight * garp_score + safety_weight * safety_score) / total
    return float(np.clip(blended, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Screener orchestration
# ---------------------------------------------------------------------------
def run_screener(tickers: Iterable[str], settings: Settings) -> pd.DataFrame:
    """Fetch data for ``tickers`` and compute filters plus scores."""

    records: List[Dict[str, float]] = []
    for ticker in tickers:
        try:
            record = fetch_fundamental_data(ticker)
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to fetch data for %s", ticker)
            record = {
                "ticker": ticker,
                "pe_ttm": np.nan,
                "pe_fwd": np.nan,
                "debt_to_equity": np.nan,
                "eps_cagr": np.nan,
                "peg": np.nan,
                "market_cap": np.nan,
            }

        record.update(fetch_risk_data(ticker))
        records.append(record)

    return build_dataframe(records, settings)


def _format_float(value: float, fmt: str = ".2f") -> str:
    """Return a formatted float or ``NaN`` when the value is missing."""

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "NaN"
    if value is None:
        return "NaN"
    return f"{value:{fmt}}"


def print_ticker_diagnostics(
    df: pd.DataFrame,
    tickers: Iterable[str],
    filters: FilterSettings,
    weights: WeightSettings,
    profile: Optional[str] = None,
) -> None:
    """Pretty-print a diagnostic breakdown for the requested tickers."""

    if df.empty:
        print("Aucune donnée disponible pour afficher les diagnostics.")
        return

    if "ticker" not in df.columns:
        print("Colonne 'ticker' absente du DataFrame : impossible d'afficher les diagnostics.")
        return

    profile_label = profile or "none"
    ticker_series = df["ticker"].astype(str)
    ticker_upper = ticker_series.str.upper()

    for ticker in tickers:
        requested = ticker.upper()
        subset = df[ticker_upper == requested]
        if subset.empty:
            print(f"Ticker {ticker} introuvable dans les résultats; ignoré.")
            continue

        row = subset.iloc[0]
        print("=" * 30)
        print(f"Diagnostics for {ticker}")
        print("=" * 30)
        print(f"Profile: {profile_label}")
        print(f"strict_pass: {bool(row.get('strict_pass', False))}")
        print(
            "nexus_score: " + _format_float(row.get("nexus_score", np.nan))
            + f" | garp_score: {_format_float(row.get('garp_score', np.nan))}"
        )
        print(
            "risk_score:  " + _format_float(row.get("risk_score", np.nan))
            + f" | safety_score: {_format_float(row.get('safety_score', np.nan))}"
        )
        print()

        print("Fundamentals:")
        print(f"  PE (ttm):          {_format_float(row.get('pe_ttm', np.nan))}")
        print(f"  PE (forward):      {_format_float(row.get('pe_fwd', np.nan))}")
        print(f"  Debt/Equity:       {_format_float(row.get('debt_to_equity', np.nan))}")
        print(f"  EPS growth (CAGR): {_format_float(row.get('eps_cagr', np.nan))}")
        print(f"  PEG ratio:         {_format_float(row.get('peg', np.nan))}")
        print(f"  Market cap:        {_format_float(row.get('market_cap', np.nan))}")
        print()

        print("Hard filter flags:")
        print(
            f"  per_ok:         {row.get('per_ok', False)} (max_pe_ttm = {_format_float(filters.max_pe_ttm)})"
        )
        print(
            f"  per_forward_ok: {row.get('per_forward_ok', False)} (max_forward_pe = {_format_float(filters.max_forward_pe)})"
        )
        print(
            f"  de_ok:          {row.get('de_ok', False)} (max_debt_to_equity = {_format_float(filters.max_debt_to_equity)})"
        )
        print(
            f"  eps_growth_ok:  {row.get('eps_growth_ok', False)} (min_eps_cagr = {_format_float(filters.min_eps_cagr)})"
        )
        print(f"  peg_ok:         {row.get('peg_ok', False)} (max_peg = {_format_float(filters.max_peg)})")
        print(
            f"  mktcap_ok:      {row.get('mktcap_ok', False)} (min_market_cap = {_format_float(filters.min_market_cap)})"
        )
        print(f"  strict_pass:    {row.get('strict_pass', False)}")
        print()

        print("GARP sub-scores:")
        print(
            f"  Valuation score:     {_format_float(row.get('valuation_score', np.nan))}"
            f" (weight ≈ {_format_float(weights.garp_valuation)})"
        )
        print(
            f"  Growth score:        {_format_float(row.get('growth_score', np.nan))}"
            f" (weight ≈ {_format_float(weights.garp_growth)})"
        )
        print(
            f"  Balance sheet score: {_format_float(row.get('balance_sheet_score', np.nan))}"
            f" (weight ≈ {_format_float(weights.garp_balance_sheet)})"
        )
        print(
            f"  Size score:          {_format_float(row.get('size_score', np.nan))}"
            f" (weight ≈ {_format_float(weights.garp_size)})"
        )
        print(f"  => GARP composite score: {_format_float(row.get('garp_score', np.nan))}")
        print()

        print("Risk metrics:")
        print(f"  vol_1y (ann.):    {_format_float(row.get('vol_1y', np.nan))}")
        print(f"  mdd_1y:           {_format_float(row.get('mdd_1y', np.nan))}")
        print(f"  adv_3m:           {_format_float(row.get('adv_3m', np.nan))}")
        print(f"  => risk_score:    {_format_float(row.get('risk_score', np.nan))}")
        print(f"  => safety_score:  {_format_float(row.get('safety_score', np.nan))}")
        print()

        print("Nexus combined score:")
        print(f"  quality_weight: {_format_float(weights.nexus_quality)}")
        print(f"  safety_weight:  {_format_float(weights.nexus_safety)}")
        print(f"  => nexus_score: {_format_float(row.get('nexus_score', np.nan))}")
        print()

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the screener."""

    parser = argparse.ArgumentParser(description="CAC 40 GARP screener")
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Chemin du fichier CSV à générer (défaut: %(default)s)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Nombre de lignes à afficher pour l'aperçu global (défaut: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Active les logs DEBUG pour suivre les téléchargements",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Chemin d'un fichier YAML pour surcharger les paramètres (optionnel)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=PROFILE_CHOICES,
        default=None,
        help="Preset Nexus profile to tweak filters/weights (optionnel)",
    )
    parser.add_argument(
        "--detail",
        nargs="+",
        help="One or more tickers for which to display a detailed breakdown",
    )
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    """Configure logging format/level for CLI usage."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.verbose)

    settings = load_config(args.config)
    if args.profile:
        LOGGER.info("Applying Nexus profile: %s", args.profile)
    apply_profile_overrides(settings.filters, settings.weights, args.profile)
    tickers = get_cac40_tickers(settings)
    LOGGER.info("Téléchargement des fondamentaux pour %d tickers", len(tickers))

    df = run_screener(tickers, settings)
    if df.empty:
        LOGGER.warning("Aucune donnée récupérée : vérifier la connectivité réseau ou la liste de tickers")
        return

    df.sort_values(by=STRICT_SORT_COLUMNS, ascending=STRICT_SORT_ASCENDING, inplace=True)

    strict_df = df[df["strict_pass"]]
    if strict_df.empty:
        print("Aucune valeur ne passe le filtre strict GARP.")
    else:
        print("Valeurs qui passent le filtre strict GARP:")
        print(strict_df.to_string(index=False))

    print("\nAperçu global trié par score (top %d):" % args.max_rows)
    print(df.head(args.max_rows).to_string(index=False))

    df.to_csv(args.output, index=False)
    LOGGER.info("Résultats complets sauvegardés dans %s", args.output)

    if args.detail:
        print()
        print_ticker_diagnostics(
            df,
            args.detail,
            settings.filters,
            settings.weights,
            profile=args.profile,
        )


if __name__ == "__main__":
    main()