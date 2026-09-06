#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_PYTHON="/opt/ros/humble/lib/python3.10/site-packages"
ROS_LOCAL_PYTHON="/opt/ros/humble/local/lib/python3.10/dist-packages"

# Deliberately ignore ambient Conda/user sites.  Only project source, ROS
# geometry packages, and (when needed) the isolated pure-Python xlrd site are
# visible to this process.
export PYTHONPATH="$ROOT/src:$ROS_PYTHON:$ROS_LOCAL_PYTHON"
export PYTHONNOUSERSITE=1
source "$ROOT/scripts/lib/resolve_xlrd_site.sh"
resolve_dexgrasp_xlrd_site

# This exact-mesh audit is intentionally an offline background workload.  It
# can occupy one core for several minutes and may create tens of thousands of
# HPP-FCL objects for the observed scene.  On the PREEMPT_RT workstation used
# for the installed-hand tests, letting native math libraries fan out or
# running this workload at the normal interactive priority has previously
# coincided with RT throttling and USB/storage driver timeouts.  Keep every
# native thread pool single-threaded and run the whole process at the lowest
# CPU and I/O scheduling priorities.  This changes scheduling only; collision
# geometry, sampling and pass/fail evidence are unchanged.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

echo "[collision-runtime] cpu=SCHED_IDLE nice=19 io=idle native_threads=1"
exec /usr/bin/chrt --idle 0 \
  /usr/bin/ionice -c 3 \
  /usr/bin/nice -n 19 \
  /usr/bin/python3 "$ROOT/apps/generate_installed_air_audit.py" "$@"
