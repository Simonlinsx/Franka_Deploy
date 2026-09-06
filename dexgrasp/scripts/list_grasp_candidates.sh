#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
PYTHON_BIN="${DEXGRASP_SHELL_PYTHON:-$WORKSPACE/.venv/bin/python}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT/apps/list_grasp_candidates.py" "$@"
