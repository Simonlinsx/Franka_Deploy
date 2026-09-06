#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUNDED_SAM_ROOT="${GROUNDED_SAM_ROOT:-/home/qiaoguanren/下载/founpose/FoundationPose-main/Grounded-Segment-Anything}"
PROMPT_SERVICE_PYTHON="${PROMPT_SERVICE_PYTHON:-/home/qiaoguanren/anaconda3/envs/foundationpose/bin/python}"

if [[ ! -x "${PROMPT_SERVICE_PYTHON}" ]]; then
  echo "Prompt-service Python is not executable: ${PROMPT_SERVICE_PYTHON}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}:${GROUNDED_SAM_ROOT}/GroundingDINO:${GROUNDED_SAM_ROOT}/segment_anything${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dynamic_pcd_matplotlib}"

cd "${ROOT_DIR}"
exec "${PROMPT_SERVICE_PYTHON}" -m dynamic_pcd.apps.prompt_segmentation_service "$@"
