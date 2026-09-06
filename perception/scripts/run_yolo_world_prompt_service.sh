#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRANKA_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
YOLO_WORLD_PYTHON="${YOLO_WORLD_PYTHON:-${FRANKA_ROOT}/third_party/long_vos_clean/.venv/bin/python}"

if [[ ! -x "${YOLO_WORLD_PYTHON}" ]]; then
  echo "YOLO-World Python is not executable: ${YOLO_WORLD_PYTHON}" >&2
  exit 2
fi

# The deployment's camera-only selector intentionally sets
# PYTHONNOUSERSITE=1 to keep ROS/Isaac Qt packages away from its Python 3.9
# OpenCV process.  This launcher execs a separate Python 3.10 environment whose
# PyTorch dependency `typing_extensions` is installed in that interpreter's
# user site.  Add that exact site explicitly; do not weaken the selector's
# isolation globally.
YOLO_WORLD_USER_SITE="$("${YOLO_WORLD_PYTHON}" -c 'import site; print(site.getusersitepackages())')"
export PYTHONPATH="${ROOT_DIR}:${YOLO_WORLD_USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/tmp/dynamic_pcd_ultralytics}"

cd "${ROOT_DIR}"
exec "${YOLO_WORLD_PYTHON}" -m dynamic_pcd.apps.yolo_world_prompt_service "$@"
