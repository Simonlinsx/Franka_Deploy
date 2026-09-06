#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/examples/inspire_mano_pipeline/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing environment. Run examples/setup_inspire_mano_env.sh first." >&2
  exit 1
fi

unset PYTHONPATH
SITE_PACKAGES="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CMEEL_LIB="$SITE_PACKAGES/cmeel.prefix/lib"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda-12.1/lib64}"
export PYTHONPATH="$ROOT_DIR/examples"
export LD_LIBRARY_PATH="$CMEEL_LIB:$CUDA_LIB_DIR"

"$PYTHON" -m pip check
"$PYTHON" -m compileall -q \
  "$ROOT_DIR/examples/inspire_mano_pipeline" \
  "$ROOT_DIR/examples/analyze_inspire_mano_log.py" \
  "$ROOT_DIR/examples/realsense_mano_inspire.py"
"$PYTHON" -m unittest discover -v \
  -s "$ROOT_DIR/examples/inspire_mano_pipeline/tests" \
  -p 'test_*.py'
python3 "$ROOT_DIR/examples/inspire_rh56_test.py" self-test

echo "All offline protocol, retargeting, and watchdog tests passed."
