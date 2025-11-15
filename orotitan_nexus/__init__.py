 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/orotitan_nexus/__init__.py b/orotitan_nexus/__init__.py
new file mode 100644
index 0000000000000000000000000000000000000000..3c231f1b4c09a4ccf286820c66a8be478ad06022
--- /dev/null
+++ b/orotitan_nexus/__init__.py
@@ -0,0 +1,6 @@
+"""OroTitan Nexus equity screener package."""
+
+from . import cli, normalization, filters, scoring, universe
+
+__all__ = ["cli", "normalization", "filters", "scoring", "universe"]
+__version__ = "0.1.0"
 
EOF
)