import pandas as pd

from orotitan_nexus.config import ProfileSettingsV2, NexusV2Settings, WalkForwardSettings, SensitivitySettings
from orotitan_nexus.robustness_v2 import run_walkforward_validation, run_sensitivity_analysis


def _profile_enabled():
    profile = ProfileSettingsV2(name="balanced")
    profile.v2 = NexusV2Settings(enabled=True)
    profile.walkforward = WalkForwardSettings(enabled=True, n_splits=2, test_horizon_days=1)
    profile.sensitivity = SensitivitySettings(enabled=True, n_random_draws=3, top_k=1)
    return profile


def test_run_walkforward_validation_basic():
    profile = _profile_enabled()
    snapshot = pd.DataFrame({"ticker": ["A", "B"], "nexus_v2_score": [90, 80], "data_complete_v1_1": [True, True]})
    price_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "ticker": ["A", "A", "B", "B"],
            "adj_close": [1, 2, 1, 1.5],
        }
    )
    result = run_walkforward_validation(snapshot, price_df, profile)
    assert isinstance(result, dict)


def test_run_sensitivity_analysis_basic():
    profile = _profile_enabled()
    snapshot = pd.DataFrame({"ticker": ["A", "B"], "nexus_v2_score": [90, 80]})
    price_df = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["A"], "adj_close": [1.0]})
    result = run_sensitivity_analysis(snapshot, price_df, profile)
    assert result["stability_metric"] == profile.sensitivity.stability_metric
    assert "samples" in result
