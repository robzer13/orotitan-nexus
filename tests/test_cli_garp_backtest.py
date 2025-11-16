import pandas as pd

from orotitan_nexus import cli_garp_backtest


def test_cli_garp_backtest_smoke(tmp_path, capsys):
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
            "adj_close": [1.0, 1.3, 1.0, 1.0],
        }
    )

    snapshot_path = tmp_path / "snapshot.csv"
    prices_path = tmp_path / "prices.csv"
    snapshot.to_csv(snapshot_path, index=False)
    prices.to_csv(prices_path, index=False)

    cli_garp_backtest.main(
        [
            "--snapshot",
            str(snapshot_path),
            "--prices",
            str(prices_path),
            "--start-date",
            "2024-01-01",
            "--horizons",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert "OroTitan GARP Backtest" in captured.out or "OroTitan GARP Backtest" in captured.err
    assert "1" in captured.out or "1" in captured.err
