#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/.." && pwd)"
PYTHON_BIN="${DYNAMIC_PCD_PYTHON:-${WORKSPACE_DIR}/.venv/bin/python}"
CAMERA_SERIAL="342222071785"
OUTPUT_ROOT="${1:-${WORKSPACE_DIR}/object_pcd_testdata}"
RUN_ID="${2:-thrown_ball_424x240_${CAMERA_SERIAL}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
DURATION_S="${THROWN_BALL_DURATION_S:-12}"
COUNTDOWN_S="${THROWN_BALL_COUNTDOWN_S:-5}"
OBJECT_TEXT="${THROWN_BALL_OBJECT_TEXT:-ball}"
# The 342222071785 A/B recordings on 2026-08-12 showed materially lower
# motion smear and better mask/depth continuity at 60 Hz.  Keep an explicit
# environment override for reproducible 30 Hz comparisons.
FPS="${THROWN_BALL_FPS:-60}"
COLOR_EXPOSURE="${THROWN_BALL_COLOR_EXPOSURE:-}"
COLOR_GAIN="${THROWN_BALL_COLOR_GAIN:-}"
DEPTH_EXPOSURE="${THROWN_BALL_DEPTH_EXPOSURE:-2000}"
DEPTH_GAIN="${THROWN_BALL_DEPTH_GAIN:-32}"
LASER_POWER="${THROWN_BALL_LASER_POWER:-360}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite an existing recording: ${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

echo "[CAMERA ONLY] Franka/RH56 will not be opened."
echo "[PREPARE] Keep the target outside the image during the ${COUNTDOWN_S} s countdown."
echo "[ACTION] After RECORDING appears: wait ~2 s, throw the target through the view one or more times, then leave ~2 s empty."
echo "[LIVE VIEW] A 424x240 preview window will show countdown and recording progress."
echo "[CAMERA PROFILE] 424x240@${FPS}Hz color_exposure=${COLOR_EXPOSURE:-auto} color_gain=${COLOR_GAIN:-auto}"
echo "[DEPTH PROFILE] exposure=${DEPTH_EXPOSURE} gain=${DEPTH_GAIN} laser=${LASER_POWER}"
echo "[OUTPUT] ${OUTPUT_DIR}"

ARGS=(
  -m dynamic_pcd.apps.record_rgbd_case
  --config configs/d435_default.yaml \
  --case thrown_ball \
  --output "${OUTPUT_DIR}" \
  --duration-s "${DURATION_S}" \
  --countdown-s "${COUNTDOWN_S}" \
  --camera-serial "${CAMERA_SERIAL}" \
  --width 424 \
  --height 240 \
  --fps "${FPS}" \
  --camera-frame-only \
  --object-text "${OBJECT_TEXT}"
  --depth-exposure "${DEPTH_EXPOSURE}"
  --depth-gain "${DEPTH_GAIN}"
  --laser-power "${LASER_POWER}"
)

if [[ "${FPS}" -ge 60 ]]; then
  ARGS+=(--max-color-depth-timestamp-skew-ms 7)
else
  ARGS+=(--max-color-depth-timestamp-skew-ms 14)
fi
if [[ -n "${COLOR_EXPOSURE}" ]]; then
  ARGS+=(--color-exposure "${COLOR_EXPOSURE}")
fi
if [[ -n "${COLOR_GAIN}" ]]; then
  ARGS+=(--color-gain "${COLOR_GAIN}")
fi

exec "${PYTHON_BIN}" "${ARGS[@]}"
