import pandas as pd

from orotitan_nexus.config import GeoSettings
from orotitan_nexus.geo import attach_region_exposures, compute_region_exposure_row, infer_region_from_country


def test_infer_region_from_country_defaults():
    settings = GeoSettings()
    assert infer_region_from_country("FR", settings) == "EU"
    assert infer_region_from_country("US", settings) == "US"
    assert infer_region_from_country("ZZ", settings) == "Unknown"


def test_region_exposure_row_fallbacks():
    settings = GeoSettings()
    row = pd.Series({"country_of_listing": "FR"})
    exposure = compute_region_exposure_row(row, settings)
    assert exposure == {"region_EU_weight": 1.0}

    row2 = pd.Series({"region_US_weight": 0.3, "region_EU_weight": 0.7})
    exposure2 = compute_region_exposure_row(row2, settings)
    assert round(sum(exposure2.values()), 6) == 1.0
    assert exposure2["region_EU_weight"] > exposure2["region_US_weight"]


def test_attach_region_exposures_creates_columns():
    settings = GeoSettings()
    df = pd.DataFrame([
        {"ticker": "AAA", "country_of_listing": "FR"},
        {"ticker": "BBB", "region_US_weight": 1.0},
    ])
    enriched = attach_region_exposures(df, settings)
    cols = [c for c in enriched.columns if c.startswith("region_")]
    assert cols
    assert enriched.loc[0, "region_EU_weight"] == 1.0
    assert enriched.loc[1, "region_US_weight"] == 1.0
