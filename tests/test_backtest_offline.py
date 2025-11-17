import pandas as pd
import pytest
from datetime import datetime

from orotitan_nexus.prices import load_prices
from orotitan_nexus.backtest import compute_horizon_returns, backtest_garp_vs_benchmark
from orotitan_nexus.api import run_garp_backtest_offline


def test_load_prices_valid_csv(tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "date,ticker,adj_close\n2024-01-01,AAA.PA,1.0\n2024-01-02,AAA.PA,1.1\n",
        encoding="utf-8",
    )
    df = load_prices(csv_path)
    assert set(df.columns) >= {"date", "ticker", "adj_close"}
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert len(df) == 2


def test_load_prices_missing_columns_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("date,ticker\n2024-01-01,AAA.PA\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prices(csv_path)


def test_compute_horizon_returns_basic():
    price_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 1),
            "ticker": ["AAA.PA", "AAA.PA", "AAA.PA"],
            "adj_close": [1.0, 1.5, 2.0],
        }
    )
    returns = compute_horizon_returns(price_df, ["AAA.PA"], pd.Timestamp("2024-01-01"), 2)
    assert pytest.approx(returns["AAA.PA"], rel=1e-6) == 1.0


def test_compute_horizon_returns_insufficient_data():
    price_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAA.PA", "AAA.PA"],
            "adj_close": [1.0, 1.1],
        }
    )
    returns = compute_horizon_returns(price_df, ["AAA.PA"], pd.Timestamp("2024-01-01"), 3)
    assert pd.isna(returns["AAA.PA"])


def test_backtest_garp_vs_benchmark_equal_weight():
    snapshot_df = pd.DataFrame(
        [
            {"ticker": "AAA.PA", "strict_pass_garp": True, "data_complete_v1_1": True},
            {"ticker": "BBB.PA", "strict_pass_garp": False, "data_complete_v1_1": True},
            {"ticker": "CCC.PA", "strict_pass_garp": True, "data_complete_v1_1": True},
        ]
    )
    price_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "ticker": ["AAA.PA", "AAA.PA", "BBB.PA", "BBB.PA", "CCC.PA", "CCC.PA"],
            "adj_close": [1.0, 2.0, 1.0, 1.0, 1.0, 1.5],
        }
    )
    result = backtest_garp_vs_benchmark(snapshot_df, price_df, pd.Timestamp("2024-01-01"), [1])
    row = result.loc[1]
    assert row["n_garp_tickers"] == 2
    assert row["n_benchmark_tickers"] == 3
    assert row["garp_return"] > row["benchmark_return"]
    assert row["excess_return"] == pytest.approx(row["garp_return"] - row["benchmark_return"], rel=1e-6)


def test_run_garp_backtest_offline(tmp_path):
    snapshot = pd.DataFrame(
        [
            {"ticker": "AAA.PA", "strict_pass_garp": True, "data_complete_v1_1": True},
            {"ticker": "BBB.PA", "strict_pass_garp": False, "data_complete_v1_1": True},
        ]
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "ticker": ["AAA.PA", "AAA.PA", "BBB.PA", "BBB.PA"],
            "adj_close": [1.0, 1.2, 1.0, 1.0],
        }
    )
    snapshot_path = tmp_path / "snapshot.csv"
    prices_path = tmp_path / "prices.csv"
    snapshot.to_csv(snapshot_path, index=False)
    prices.to_csv(prices_path, index=False)

    df = run_garp_backtest_offline(
        snapshot_path=str(snapshot_path),
        prices_path=str(prices_path),
        start_date="2024-01-01",
        horizons=(1,),
    )
    assert not df.empty
    assert "garp_return" in df.columns
    assert df.index.tolist() == [1]
