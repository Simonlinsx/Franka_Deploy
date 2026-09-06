#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-/home/qiaoguanren/anaconda3/envs/dynamic/bin/python}"
VENV_DIR="$ROOT_DIR/examples/inspire_mano_pipeline/.venv"
WILOR_ROOT="${WILOR_ROOT:-$ROOT_DIR/third_party/wilor-mini}"
DEX_ROOT="${DEX_ROOT:-$ROOT_DIR/third_party/dex-retargeting}"

required_assets=(
  "$WILOR_ROOT/pretrained_models/mano_mean_params.npz"
  "$WILOR_ROOT/pretrained_models/MANO_RIGHT.pkl"
  "$WILOR_ROOT/pretrained_models/wilor_final.ckpt"
  "$WILOR_ROOT/pretrained_models/detector.pt"
  "$DEX_ROOT/src/dex_retargeting/configs/teleop/inspire_hand_right.yml"
  "$DEX_ROOT/assets/robots/hands/inspire_hand/inspire_hand_right.urdf"
)

for asset in "${required_assets[@]}"; do
  if [[ ! -s "$asset" ]]; then
    echo "Required model/config asset is missing or empty: $asset" >&2
    exit 1
  fi
done

if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "Base Python not found: $BASE_PYTHON" >&2
  exit 1
fi

"$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/examples/requirements-inspire-mano.txt"
PYTHONPATH= "$VENV_DIR/bin/python" -m pip check

unset PYTHONPATH
SITE_PACKAGES="$("$VENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CMEEL_LIB="$SITE_PACKAGES/cmeel.prefix/lib"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda-12.1/lib64}"
LD_LIBRARY_PATH="$CMEEL_LIB:$CUDA_LIB_DIR" "$VENV_DIR/bin/python" -c \
  'import cv2, dex_retargeting, numpy, pinocchio, pyrealsense2, torch; print(f"Validated imports: torch={torch.__version__}, numpy={numpy.__version__}, pinocchio={pinocchio.__version__}")'

echo "Environment ready: $VENV_DIR"
echo "WiLoR assets: $WILOR_ROOT"
echo "dex-retargeting assets: $DEX_ROOT"
echo "Run: examples/run_inspire_mano_pipeline.sh --help"
