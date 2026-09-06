#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The orchestrator itself needs only project source plus the isolated xlrd
# fallback.  Each geometry child wrapper adds ROS explicitly; keeping ROS 3.10
# paths out of the D435 Python 3.9 child avoids cross-version contamination.
export PYTHONPATH="$ROOT/src"
export PYTHONNOUSERSITE=1
source "$ROOT/scripts/lib/resolve_xlrd_site.sh"
resolve_dexgrasp_xlrd_site
exec /usr/bin/python3 "$ROOT/apps/prepare_fresh_installed_air_audit.py" "$@"
