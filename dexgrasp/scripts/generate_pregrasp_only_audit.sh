#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_PYTHON=/opt/ros/humble/lib/python3.10/site-packages
ROS_LOCAL_PYTHON=/usr/local/lib/python3.10/dist-packages
export PYTHONPATH="$ROOT/src:$ROS_PYTHON:$ROS_LOCAL_PYTHON${PYTHONPATH:+:$PYTHONPATH}"

exec /usr/bin/python3 "$ROOT/apps/generate_pregrasp_only_audit.py" "$@"
