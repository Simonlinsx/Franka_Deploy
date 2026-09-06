#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/.." && pwd)"
PYTHON_BIN="${DYNAMIC_PCD_PYTHON:-${WORKSPACE_DIR}/.venv/bin/python}"
OUTPUT_ROOT="${1:-${WORKSPACE_DIR}/data/perception_corpus/session_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
# Four default cases contain 930 RGB-D frames. Their uncompressed BGR8+Z16
# payload is about 1.76 GiB; 4 GiB leaves ample room for PNG/MP4 and filesystem
# overhead while still fitting on a nominal 8 GB removable drive.
MIN_FREE_BYTES=$((4 * 1024 * 1024 * 1024))
AVAILABLE_BYTES="$(df -PB1 "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
if [[ ! "${AVAILABLE_BYTES}" =~ ^[0-9]+$ ]] || (( AVAILABLE_BYTES < MIN_FREE_BYTES )); then
  echo "Need at least 4 GiB free for the complete four-case session." >&2
  echo "Output root: ${OUTPUT_ROOT}; available bytes: ${AVAILABLE_BYTES:-unknown}" >&2
  exit 2
fi
cd "${REPO_DIR}"

record_case() {
  local case_name="$1"
  local instruction="$2"
  echo
  echo "============================================================"
  echo "Prepare ${case_name}: ${instruction}"
  read -r -p "Press Enter when the object is ready for ROI selection..."
  "${PYTHON_BIN}" -m dynamic_pcd.apps.record_rgbd_case \
    --config configs/d435_default.yaml \
    --case "${case_name}" \
    --output "${OUTPUT_ROOT}/${case_name}"
}

record_case static_ball \
  "keep the ball still for 5 seconds"
record_case rolling_ball \
  "select it while still; roll it only after RECORDING appears"
record_case rolling_cylinder \
  "select it while still; roll it only after RECORDING appears"
record_case hand_occlusion \
  "use RH56/dexterous-hand fingers: 2s clear -> partial cover -> ~1s full cover -> uncover; never reselect"

echo
echo "[PASS] Four camera-only RGB-D cases saved under: ${OUTPUT_ROOT}"
