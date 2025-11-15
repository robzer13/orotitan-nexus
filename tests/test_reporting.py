import pandas as pd

from orotitan_nexus.reporting import write_csv


def test_write_csv_creates_file(tmp_path):
    df = pd.DataFrame(
        {
            "ticker": ["AAA.PA"],
            "score_v1_1": [4],
            "category_v1_1": ["WATCHLIST_V1_1"],
        }
    )
    output = tmp_path / "out.csv"
    path = write_csv(df, output)
    assert path.exists()
    content = output.read_text()
    assert "ticker" in content
    assert "WATCHLIST_V1_1" in content
