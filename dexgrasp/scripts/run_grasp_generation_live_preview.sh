#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"

# This interpreter owns only orchestration, D435 capture and Open3D preview.
# Official inference is launched in a separate, activated Python 3.8 runtime by
# apps/grasp_generation_workflow.py.
DEFAULT_DYNAMIC="/home/qiaoguanren/anaconda3/envs/dynamic/bin/python"
if [[ ! -x "$DEFAULT_DYNAMIC" ]]; then
  DEFAULT_DYNAMIC="$WORKSPACE/.venv/bin/python"
fi
DYNAMIC_PYTHON="${DEXGRASP_DYNAMIC_PYTHON:-${DEXGRASP_SHELL_PYTHON:-$DEFAULT_DYNAMIC}}"

export DEXGRASP_WORKFLOW_DYNAMIC_PYTHON="$DYNAMIC_PYTHON"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$DYNAMIC_PYTHON" "$ROOT/apps/grasp_generation_workflow.py" "$@"
