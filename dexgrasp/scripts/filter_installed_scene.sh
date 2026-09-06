#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_PYTHON="/opt/ros/humble/lib/python3.10/site-packages"
ROS_LOCAL_PYTHON="/opt/ros/humble/local/lib/python3.10/dist-packages"

# Pinocchio and HPP-FCL are the ROS Humble Python 3.10 builds on this host.
# Keep this offline tool out of the Python 3.9 AnyDexGrasp inference runtime.
export PYTHONPATH="$ROOT/src:$ROS_PYTHON:$ROS_LOCAL_PYTHON${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 "$ROOT/apps/filter_installed_scene.py" "$@"
