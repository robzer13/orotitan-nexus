import pandas as pd

from orotitan_nexus import cli_garp_diagnostics
from orotitan_nexus.garp_rules import (
    GARP_FLAG_COLUMN,
    RULE_COL_DEBT,
    RULE_COL_EPS_CAGR,
    RULE_COL_PE_FWD,
    RULE_COL_PE_TTM,
    RULE_COL_PEG,
)


def test_cli_garp_diagnostics_smoke(tmp_path, capsys):
    snapshot = pd.DataFrame(
        [
            {
                "ticker": "AAA.PA",
                "data_complete_v1_1": True,
                RULE_COL_PE_TTM: True,
                RULE_COL_PE_FWD: True,
                RULE_COL_DEBT: True,
                RULE_COL_EPS_CAGR: True,
                RULE_COL_PEG: True,
                GARP_FLAG_COLUMN: True,
            },
            {
                "ticker": "BBB.PA",
                "data_complete_v1_1": True,
                RULE_COL_PE_TTM: False,
                RULE_COL_PE_FWD: True,
                RULE_COL_DEBT: True,
                RULE_COL_EPS_CAGR: True,
                RULE_COL_PEG: True,
                GARP_FLAG_COLUMN: False,
            },
        ]
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]
            ),
            "ticker": ["AAA.PA", "AAA.PA", "BBB.PA", "BBB.PA"],
            "adj_close": [1.0, 1.2, 1.0, 1.0],
        }
    )

    snapshot_path = tmp_path / "snapshot.csv"
    prices_path = tmp_path / "prices.csv"
    snapshot.to_csv(snapshot_path, index=False)
    prices.to_csv(prices_path, index=False)

    cli_garp_diagnostics.main(
        [
            "--snapshot",
            str(snapshot_path),
            "--prices",
            str(prices_path),
            "--start-date",
            "2025-01-01",
            "--horizons",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert "GARP Rule Diagnostics" in captured.out or "GARP Rule Diagnostics" in captured.err
    assert "pe_ttm" in captured.out or "pe_ttm" in captured.err
