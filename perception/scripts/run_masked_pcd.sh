#!/usr/bin/env bash
set -e
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
export QT_QPA_FONTDIR=${QT_QPA_FONTDIR:-/usr/share/fonts/truetype/dejavu}
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${DYNAMIC_PCD_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" && -x "${REPO_DIR}/../.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/../.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m dynamic_pcd.apps.realtime_masked_pcd \
  --config configs/d435_default.yaml "$@"
