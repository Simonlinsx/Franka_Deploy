from pathlib import Path
import os
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Match the full-audit wrapper without importing an entire Conda environment.
# The cache is pure Python and the mapper still verifies the bound XLS SHA-256.
try:
    import xlrd  # noqa: F401
except ImportError:
    XLRD_SITE = Path(
        os.environ.get(
            "DEXGRASP_XLRD_SITE",
            "/home/qiaoguanren/anaconda3/pkgs/"
            "xlrd-2.0.1-pyhd3eb1b0_0/site-packages",
        )
    ).expanduser().resolve()
    if (XLRD_SITE / "xlrd/__init__.py").is_file():
        sys.path.insert(0, str(XLRD_SITE))
