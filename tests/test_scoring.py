import numpy as np
import pandas as pd

from orotitan_nexus.config import FilterSettings, WeightSettings
from orotitan_nexus.filters import (
    CATEGORY_DATA_MISSING,
    CATEGORY_ELITE,
    CATEGORY_EXCLUDED,
    CATEGORY_REJECT,
    CATEGORY_WATCHLIST,
    categorize_v1_1,
)
from orotitan_nexus.scoring import compute_garp_score, compute_v1_1_score


def test_compute_garp_score_requires_data_ready():
    filters = FilterSettings()
    weights = WeightSettings()
    row = pd.Series({"data_ready_nexus": False})
    assert compute_garp_score(row, filters, weights) == 0.0

    ready_row = pd.Series(
        {
            "data_ready_nexus": True,
            "pe_ttm": 12.0,
            "pe_fwd": 14.0,
            "peg": 1.1,
            "eps_cagr": 0.12,
            "debt_to_equity": 0.30,
            "market_cap": 20e9,
        }
    )
    assert compute_garp_score(ready_row, filters, weights) > 0


def test_compute_v1_1_score():
    filters = FilterSettings()
    row = pd.Series(
        {
            "universe_ok": True,
            "data_complete_v1_1": True,
            "pe_ttm": 15.0,
            "pe_fwd": 14.0,
            "debt_to_equity": 0.20,
            "eps_cagr": 0.20,
            "peg": 1.0,
        }
    )
    assert compute_v1_1_score(row, filters) == 5

    missing_row = row.copy()
    missing_row["data_complete_v1_1"] = False
    assert compute_v1_1_score(missing_row, filters) == 0


def test_categorize_v1_1():
    row = pd.Series({"universe_ok": False})
    assert categorize_v1_1(row) == CATEGORY_EXCLUDED

    row = pd.Series({"universe_ok": True, "data_complete_v1_1": False})
    assert categorize_v1_1(row) == CATEGORY_DATA_MISSING

    row = pd.Series(
        {
            "universe_ok": True,
            "data_complete_v1_1": True,
            "score_v1_1": 5,
        }
    )
    assert categorize_v1_1(row) == CATEGORY_ELITE

    row["score_v1_1"] = 3
    assert categorize_v1_1(row) == CATEGORY_WATCHLIST

    row["score_v1_1"] = 1
    assert categorize_v1_1(row) == CATEGORY_REJECT
