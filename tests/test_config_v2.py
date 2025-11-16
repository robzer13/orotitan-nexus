from orotitan_nexus.config import build_settings_from_config, NexusV2Settings


def test_profile_v2_defaults_disabled():
    filters, weights, universe, profile = build_settings_from_config({}, None)
    assert hasattr(profile, "v2")
    assert profile.v2.enabled is False


def test_profile_v2_yaml_override():
    config = {
        "profile": {
            "name": "balanced",
            "v2": {"enabled": True, "garp_weight": 0.5, "elite_min": 90},
            "quality": {"roe_weight": 0.5},
        }
    }
    _, _, _, profile = build_settings_from_config(config, None)
    assert isinstance(profile.v2, NexusV2Settings)
    assert profile.v2.enabled is True
    assert profile.v2.garp_weight == 0.5
    assert profile.v2.elite_min == 90
    assert profile.quality.roe_weight == 0.5
