#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
PYTHON_BIN="${DEXGRASP_REGISTRATION_PYTHON:-$WORKSPACE/.venv/bin/python}"

export PYTHONPATH="$ROOT/src:$WORKSPACE:$WORKSPACE/perception${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT/apps/register_installed_hand.py" "$@"
