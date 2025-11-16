from orotitan_nexus.config import (
    load_settings,
    WalkForwardSettings,
    SensitivitySettings,
    RegimeSettings,
)


def test_walkforward_defaults():
    settings = WalkForwardSettings()
    assert settings.enabled is False
    assert settings.n_splits == 4


def test_load_settings_with_robustness(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
profile:
  name: balanced
  walkforward:
    enabled: true
    n_splits: 3
  sensitivity:
    enabled: true
    weight_perturbation_pct: 0.2
  regime:
    enabled: true
    benchmark_ticker: "^TEST"
""",
        encoding="utf-8",
    )
    _, _, _, profile = load_settings(str(cfg), None)
    assert profile.walkforward.enabled is True
    assert profile.walkforward.n_splits == 3
    assert profile.sensitivity.weight_perturbation_pct == 0.2
    assert profile.regime.benchmark_ticker == "^TEST"
