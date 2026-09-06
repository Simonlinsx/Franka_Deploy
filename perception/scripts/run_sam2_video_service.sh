#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
SAM2_ROOT="${SAM2_ROOT:-${WORKSPACE_ROOT}/third_party/sam2}"
SAM2_VIDEO_SERVICE_PYTHON="${SAM2_VIDEO_SERVICE_PYTHON:-/home/qiaoguanren/anaconda3/envs/dynamic/bin/python}"

if [[ ! -x "${SAM2_VIDEO_SERVICE_PYTHON}" ]]; then
  echo "SAM2 video-service Python is not executable: ${SAM2_VIDEO_SERVICE_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${SAM2_ROOT}/sam2/build_sam.py" ]]; then
  echo "Official SAM2 checkout was not found at: ${SAM2_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}:${SAM2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dynamic_pcd_matplotlib}"

# The OpenCV ROI selector deliberately exports PYTHONNOUSERSITE=1 to keep ROS
# and Isaac packages out of its Python 3.9 GUI process.  This launcher switches
# to the pinned Python 3.10 SAM2 environment, whose Torch installation currently
# resolves typing_extensions from its own user site.  Do not leak the selector's
# isolation flag across that interpreter boundary.
unset PYTHONNOUSERSITE

cd "${ROOT_DIR}"
exec "${SAM2_VIDEO_SERVICE_PYTHON}" -m dynamic_pcd.apps.sam2_video_service "$@"
