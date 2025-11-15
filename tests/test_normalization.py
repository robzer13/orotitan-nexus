import numpy as np
import pandas as pd

from orotitan_nexus.normalization import (
    compute_eps_cagr,
    compute_eps_cagr_from_series,
    compute_peg,
)


def test_compute_eps_cagr_from_series_positive():
    series = pd.Series([1.0, 1.331], index=[2019, 2022])
    result = compute_eps_cagr_from_series(series)
    assert np.isfinite(result)
    # 10% CAGR over 3 periods should be close to 0.10
    assert abs(result - 0.10) < 1e-6


def test_compute_eps_cagr_from_series_invalid():
    series = pd.Series([1.0, -2.0], index=[2019, 2020])
    result = compute_eps_cagr_from_series(series)
    assert np.isnan(result)


def test_compute_peg_prefers_info_value():
    assert compute_peg(20.0, 0.15, 1.5) == 1.5


def test_compute_peg_reconstructed_when_missing():
    peg = compute_peg(20.0, 0.10, np.nan)
    assert abs(peg - 2.0) < 1e-9


def test_compute_peg_nan_when_inputs_missing():
    assert np.isnan(compute_peg(np.nan, np.nan, np.nan))


def test_compute_eps_cagr_prefers_income_stmt():
    info = {}
    income_stmt = pd.DataFrame({2019: [1.0], 2022: [1.331]}, index=["Net Income"])
    eps_cagr, source = compute_eps_cagr(info, income_stmt)
    assert source == "net_income"
    assert abs(eps_cagr - 0.10) < 1e-6


def test_compute_eps_cagr_fallback_to_info():
    info = {"earningsGrowth": 0.07}
    income_stmt = pd.DataFrame()
    eps_cagr, source = compute_eps_cagr(info, income_stmt)
    assert source == "yoy"
    assert eps_cagr == 0.07