import pandas as pd
import pytest

from orotitan_nexus.portfolio_metrics import compute_hhi, compute_top5_weight


def test_portfolio_hhi_and_top5():
    weights = pd.Series([0.4, 0.3, 0.2, 0.1])
    assert abs(compute_hhi(weights) - 0.30) < 1e-6
    assert compute_top5_weight(weights) == pytest.approx(1.0)

