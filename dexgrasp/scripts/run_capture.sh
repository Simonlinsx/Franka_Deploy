#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
PYTHON_BIN="${DEXGRASP_SHELL_PYTHON:-$WORKSPACE/.venv/bin/python}"

export PYTHONPATH="$ROOT/src:$WORKSPACE/perception${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT/apps/capture_object_grasps.py" "$@"
