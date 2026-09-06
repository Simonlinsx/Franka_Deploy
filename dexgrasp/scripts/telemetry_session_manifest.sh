#!/usr/bin/env bash
set -euo pipefail

# Hardware-free manifest create/verify wrapper.  It never imports the native
# producer, pylibfranka, a serial backend, RealSense, or a GUI.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(cd "$ROOT/.." && pwd)
PYTHON_BIN=${DEXGRASP_CONTROL_PYTHON:-$WORKSPACE/.venv/bin/python}

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT/apps/telemetry_session_manifest.py" "$@"

