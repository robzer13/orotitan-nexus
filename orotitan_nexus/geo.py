"""Geographic helpers (opt-in, portfolio aware)."""
from __future__ import annotations

from typing import Dict
import pandas as pd

from .config import GeoSettings


UNKNOWN_REGION = "Unknown"


def infer_region_from_country(country: str, geo_settings: GeoSettings) -> str:
    """Map ISO country code to a coarse region using profile defaults.

    If the country is missing or not found in the map, returns ``UNKNOWN_REGION``.
    """

    if not country:
        return UNKNOWN_REGION
    mapping = geo_settings.default_region_map or {}
    return mapping.get(str(country).upper(), UNKNOWN_REGION)


def compute_region_exposure_row(row: pd.Series, geo_settings: GeoSettings) -> Dict[str, float]:
    """Return a normalized region exposure dict for a ticker.

    Preference order:
    1. explicit region_*_weight columns present on the row
    2. fallback to 100% exposure in the inferred region from ``country_of_listing``
    """

    explicit = {k: v for k, v in row.items() if str(k).startswith("region_") and str(k).endswith("_weight")}
    clean: Dict[str, float] = {}
    for key, value in explicit.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        clean[key] = max(0.0, numeric)

    if clean:
        total = sum(clean.values())
        if total > 0:
            return {k: v / total for k, v in clean.items()}

    region = infer_region_from_country(row.get("country_of_listing"), geo_settings)
    return {f"region_{region}_weight": 1.0}


def attach_region_exposures(df: pd.DataFrame, geo_settings: GeoSettings) -> pd.DataFrame:
    """Ensure consistent region weight columns exist.

    This function is NaN-safe and will not mutate the input DataFrame in place.
    """

    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    exposures = [compute_region_exposure_row(row, geo_settings) for _, row in df.iterrows()]
    all_regions: Dict[str, float] = {}
    for exp in exposures:
        for key in exp:
            all_regions[key] = 0.0

    enriched = df.copy()
    for key in all_regions:
        enriched[key] = 0.0

    for idx, exp in enumerate(exposures):
        for key, value in exp.items():
            enriched.iloc[idx, enriched.columns.get_loc(key)] = value
    return enriched
