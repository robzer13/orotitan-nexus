import pandas as pd

from orotitan_nexus.config import GeoSettings, ProfileSettingsV2
from orotitan_nexus.portfolio_diversification import (
    compute_diversification_score,
    compute_geo_distance,
    compute_region_weights,
    compute_sector_weights,
    compute_weights,
)


def _make_profile():
    filters = ProfileSettingsV2()
    return filters


def test_compute_weights_simple():
    df = pd.DataFrame({"position_value": [100, 100]})
    weights = compute_weights(df)
    assert abs(weights.sum() - 1.0) < 1e-6


def test_diversification_score_penalizes_concentration():
    profile = _make_profile()
    concentrated = pd.Series({"A": 0.9, "B": 0.1})
    even = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    sector_weights = pd.Series({})
    score_concentrated = compute_diversification_score(concentrated, sector_weights, pd.Series(dtype=float), profile)
    score_even = compute_diversification_score(even, sector_weights, pd.Series(dtype=float), profile)
    assert score_even > score_concentrated


def test_region_weights_from_country():
    settings = GeoSettings()
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "country_of_listing": ["FR", "US"],
            "position_value": [50, 50],
        }
    )
    region_weights = compute_region_weights(df, settings)
    assert abs(region_weights.sum() - 1.0) < 1e-6
    assert region_weights.get("EU", 0) > 0
    assert region_weights.get("US", 0) > 0


def test_geo_distance_targets():
    settings = GeoSettings(target_regions={"EU": 0.5, "US": 0.5})
    weights = pd.Series({"EU": 0.6, "US": 0.4})
    distances = compute_geo_distance(weights, settings)
    assert distances["EU"] > 0
    assert distances["US"] < 0
