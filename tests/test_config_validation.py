import pytest

from orotitan_nexus.config import (
    ConfigError,
    FilterSettings,
    GarpThresholds,
    ProfileSettings,
    UniverseSettings,
    validate_settings,
)


def test_validate_settings_accepts_defaults():
    filters = FilterSettings()
    universe = UniverseSettings(tickers=["AAA.PA"])
    profile = ProfileSettings(name="balanced")
    validate_settings(filters, universe, profile)


def test_validate_settings_requires_universe():
    filters = FilterSettings()
    universe = UniverseSettings(tickers=[])
    profile = ProfileSettings(name="balanced")
    with pytest.raises(ConfigError):
        validate_settings(filters, universe, profile)


def test_validate_settings_checks_garp_thresholds():
    filters = FilterSettings()
    universe = UniverseSettings(tickers=["AAA.PA"])
    profile = ProfileSettings(name="balanced", garp=GarpThresholds(peg_max=-1.0))
    with pytest.raises(ConfigError):
        validate_settings(filters, universe, profile)
