import json
import subprocess
import sys


def test_cli_v2_robustness_runs(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
profile:
  name: balanced
  v2:
    enabled: true
  walkforward:
    enabled: true
  sensitivity:
    enabled: true
  regime:
    enabled: true
    benchmark_ticker: "^IDX"
""",
        encoding="utf-8",
    )
    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_text("ticker,nexus_v2_score,data_complete_v1_1\nA,80,True\n", encoding="utf-8")
    prices = tmp_path / "prices.csv"
    prices.write_text("date,ticker,adj_close\n2024-01-01,^IDX,1\n2024-01-02,^IDX,1.1\n2024-01-01,A,1\n", encoding="utf-8")
    output = tmp_path / "out.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "orotitan_nexus.cli_v2_robustness",
            "--config",
            str(cfg),
            "--snapshot",
            str(snapshot),
            "--prices",
            str(prices),
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert output.exists()
    data = json.loads(output.read_text())
    assert isinstance(data, dict)
