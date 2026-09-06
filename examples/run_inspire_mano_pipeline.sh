#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/examples/inspire_mano_pipeline/.venv/bin/python"
CACHE_DIR="$ROOT_DIR/examples/inspire_mano_pipeline/output/cache"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing environment. Run examples/setup_inspire_mano_env.sh first." >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"
unset PYTHONPATH
SITE_PACKAGES="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CMEEL_LIB="$SITE_PACKAGES/cmeel.prefix/lib"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda-12.1/lib64}"
# ROS Humble ships a different Pinocchio/EigenPy ABI.  Keep its libraries out
# of this process and place the venv's cmeel bundle first.
export LD_LIBRARY_PATH="$CMEEL_LIB:$CUDA_LIB_DIR"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export YOLO_CONFIG_DIR="$CACHE_DIR/ultralytics"
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"

exec "$PYTHON" "$ROOT_DIR/examples/realsense_mano_inspire.py" "$@"
