import pandas as pd

from orotitan_nexus.backtest import backtest_quintiles_by_score


def test_backtest_quintiles_by_score_basic():
    snapshot = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "data_complete_v1_1": [True] * 5,
            "nexus_v2_score": [90, 80, 60, 40, 20],
        }
    )
    price_rows = []
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    prices = {
        "AAA": [1.0, 1.2, 1.3],
        "BBB": [1.0, 1.1, 1.2],
        "CCC": [1.0, 1.0, 1.0],
        "DDD": [1.0, 0.9, 0.8],
        "EEE": [1.0, 0.8, 0.7],
    }
    for ticker, series in prices.items():
        for date, price in zip(dates, series):
            price_rows.append({"date": date, "ticker": ticker, "adj_close": price})
    price_df = pd.DataFrame(price_rows)

    result = backtest_quintiles_by_score(snapshot, price_df, pd.to_datetime("2024-01-01"), [2])
    assert not result.empty
    assert (0, 2) in result.index or (1, 2) in result.index
