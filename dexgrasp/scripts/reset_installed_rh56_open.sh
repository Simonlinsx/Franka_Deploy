#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
PYTHON_BIN="${DEXGRASP_CONTROL_PYTHON:-$WORKSPACE/.venv/bin/python}"

export PYTHONPATH="$ROOT/src:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT/apps/reset_installed_rh56_open.py" "$@"
