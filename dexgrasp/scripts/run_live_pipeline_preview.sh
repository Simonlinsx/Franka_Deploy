#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
PYTHON_BIN="${DEXGRASP_SHELL_PYTHON:-$WORKSPACE/.venv/bin/python}"

for argument in "$@"; do
  if [[ "$argument" == "--show-current-hand-mesh" ]]; then
    # The live feedback mesh uses the checksum-pinned official XLS mapper.
    # Keep this narrow pure-Python dependency isolated from the selected
    # Open3D/RealSense environment.
    source "$ROOT/scripts/lib/resolve_xlrd_site.sh"
    resolve_dexgrasp_xlrd_site
    break
  fi
done

export PYTHONPATH="$ROOT/src:$WORKSPACE/perception${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT/apps/live_pipeline_preview.py" "$@"
