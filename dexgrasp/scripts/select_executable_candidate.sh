#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec /home/qiaoguanren/anaconda3/bin/python3 \
  "$ROOT/apps/select_executable_candidate.py" "$@"
