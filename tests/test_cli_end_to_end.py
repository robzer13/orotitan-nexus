 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/tests/test_cli_end_to_end.py b/tests/test_cli_end_to_end.py
new file mode 100644
index 0000000000000000000000000000000000000000..ec5922750e5e13c3d775a84fc9b3b57a181832df
--- /dev/null
+++ b/tests/test_cli_end_to_end.py
@@ -0,0 +1,69 @@
+import numpy as np
+import pandas as pd
+
+from orotitan_nexus import cli
+from orotitan_nexus.config import FilterSettings, WeightSettings, UniverseSettings
+from orotitan_nexus import data_fetch
+
+
+def _fake_raw_fundamentals(ticker: str):
+    info = {
+        "trailingPE": 20.0,
+        "forwardPE": 14.0,
+        "debtToEquity": 30.0,
+        "earningsGrowth": 0.12,
+        "pegRatio": np.nan,
+        "marketCap": 15e9,
+        "currentPrice": 100.0,
+        "trailingEps": 5.0,
+        "forwardEps": 6.5,
+    }
+    income_stmt = pd.DataFrame(
+        {
+            2019: [1.0e9],
+            2020: [1.2e9],
+            2021: [1.331e9],
+        },
+        index=["Net Income"],
+    )
+    balance_sheet = pd.DataFrame({0: [1e10, 5e9]}, index=["Total Debt", "Total Stockholder Equity"])
+    return {"info": info, "income_stmt": income_stmt, "balance_sheet": balance_sheet}
+
+
+def _fake_price_history(ticker: str, lookback_days: int = 252):
+    dates = pd.date_range("2022-01-01", periods=lookback_days, freq="B")
+    adj_close = np.linspace(90.0, 110.0, num=lookback_days)
+    # Inject a small dip to ensure max drawdown is negative and risk data is valid.
+    adj_close[lookback_days // 2] = adj_close[lookback_days // 2] - 5.0
+    data = pd.DataFrame(
+        {
+            "Adj Close": adj_close,
+            "Close": adj_close,
+            "Volume": np.full(lookback_days, 6_000_000),
+        },
+        index=dates,
+    )
+    return data
+
+
+def test_run_screener_with_monkeypatched_fetch(monkeypatch):
+    monkeypatch.setattr(data_fetch, "fetch_raw_fundamentals", _fake_raw_fundamentals)
+    monkeypatch.setattr(data_fetch, "fetch_price_history", _fake_price_history)
+
+    filters = FilterSettings()
+    weights = WeightSettings()
+    universe = UniverseSettings(tickers=["AAA.PA", "BBB.PA"])
+
+    df = cli.run_screener(filters, weights, universe)
+    assert not df.empty
+    required_columns = {
+        "score_v1_1",
+        "category_v1_1",
+        "garp_score",
+        "risk_score",
+        "nexus_score",
+        "data_complete_v1_1",
+    }
+    assert required_columns.issubset(df.columns)
+    assert (df["score_v1_1"] >= 3).any()
+    assert (df["category_v1_1"] != "DATA_MISSING").any()
 
EOF
)