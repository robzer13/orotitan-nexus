 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/cac40_garp_screener.py b/cac40_garp_screener.py
new file mode 100644
index 0000000000000000000000000000000000000000..ba4dd06467f3496f7a1e80a914248d85f2ce745a
--- /dev/null
+++ b/cac40_garp_screener.py
@@ -0,0 +1,6 @@
+"""Entry point for the OroTitan Nexus CLI (backward compatibility wrapper)."""
+from orotitan_nexus.cli import main
+
+
+if __name__ == "__main__":
+    main()
 
EOF
)