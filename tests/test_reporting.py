import pandas as pd

from orotitan_nexus.config import UniverseSettings
from orotitan_nexus.garp_rules import GARP_FLAG_COLUMN
from orotitan_nexus.reporting import summarize_garp, summarize_v1, write_csv


def test_write_csv_creates_file(tmp_path):
    df = pd.DataFrame(
        {
            "ticker": ["AAA.PA"],
            "score_v1_1": [4],
            "category_v1_1": ["WATCHLIST_V1_1"],
            "data_complete_v1_1": [True],
            GARP_FLAG_COLUMN: [True],
        }
    )
    output = tmp_path / "out.csv"
    path = write_csv(df, output)
    assert path.exists()
    content = output.read_text()
    assert "ticker" in content
    assert "WATCHLIST_V1_1" in content


def test_summarize_helpers_return_counts():
    df = pd.DataFrame(
        {
            "ticker": ["AAA.PA"],
            "score_v1_1": [4],
            "nexus_score": [70],
            "garp_score": [60],
            "category_v1_1": ["WATCHLIST_V1_1"],
            "strict_pass": [True],
            "data_complete_v1_1": [True],
            GARP_FLAG_COLUMN: [True],
        }
    )
    universe = UniverseSettings(name="TEST", tickers=["AAA.PA"])
    v1_summary = summarize_v1(df, universe, "balanced")
    assert v1_summary["total"] == 1
    assert v1_summary["categories"]["WATCHLIST_V1_1"] == 1

    garp_summary = summarize_garp(df, universe.name, "balanced")
    assert garp_summary["strict_count"] == 1
    assert garp_summary["top"][0]["ticker"] == "AAA.PA"
