import pandas as pd

from orotitan_nexus.config import build_settings_from_config, PlaybookSettings


def test_playbook_defaults_disabled():
    _, _, _, profile = build_settings_from_config({}, profile_name=None)
    assert hasattr(profile, "playbook")
    assert profile.playbook.enabled is False
    assert isinstance(profile.playbook, PlaybookSettings)


def test_playbook_yaml_override():
    cfg = {
        "profile": {
            "playbook": {
                "enabled": True,
                "thresholds": {"min_core_score_buy": 60.0},
            }
        }
    }
    _, _, _, profile = build_settings_from_config(cfg, profile_name=None)
    assert profile.playbook.enabled is True
    assert profile.playbook.thresholds.min_core_score_buy == 60.0

