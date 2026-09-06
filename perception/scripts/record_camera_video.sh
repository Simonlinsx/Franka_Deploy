#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${1:-${WORKSPACE_DIR}/test_videos}"
FILE_NAME="${2:-realsense_$(date +%Y%m%d_%H%M%S).mp4}"
PYTHON_BIN="${DYNAMIC_PCD_PYTHON:-${WORKSPACE_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ "${FILE_NAME}" != *.mp4 ]]; then
  FILE_NAME="${FILE_NAME}.mp4"
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/${FILE_NAME}"

if [[ -e "${OUTPUT_FILE}" ]]; then
  echo "Refusing to overwrite an existing recording: ${OUTPUT_FILE}" >&2
  exit 2
fi

# exec makes the recorder receive Ctrl+C directly. It releases the MP4 encoder
# before publishing the final filename. No Franka or RH56 interface is opened.
cd "${WORKSPACE_DIR}/perception"
exec "${PYTHON_BIN}" -m dynamic_pcd.apps.record_camera_video \
  --config configs/d435_default.yaml \
  --output "${OUTPUT_FILE}"
