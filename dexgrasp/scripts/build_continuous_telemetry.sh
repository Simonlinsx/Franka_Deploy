#!/usr/bin/env bash
set -euo pipefail

# Build the two intentionally separate CPython extensions used by the
# continuous viewer workflow.  No command in this script opens a robot,
# serial port, camera, controller, or GUI.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE_ROOT=$(cd "${ROOT}/.." && pwd)
VIEWER_PYTHON=${DEXGRASP_DYNAMIC_PYTHON:-/home/qiaoguanren/anaconda3/envs/dynamic/bin/python}
CONTROL_PYTHON=${DEXGRASP_CONTROL_PYTHON:-${WORKSPACE_ROOT}/.venv/bin/python}
VIEWER_BUILD_DIR=${ANYDEX_VIEWER_TELEMETRY_BUILD_DIR:-/tmp/anydex-native-telemetry-viewer-py310}
CONTROL_READER_BUILD_DIR=${ANYDEX_CONTROL_TELEMETRY_BUILD_DIR:-/tmp/anydex-native-telemetry-control-py39}
PRODUCER_BUILD_DIR=${ANYDEX_FRANKA_TAP_BUILD_DIR:-/tmp/anydex-franka-telemetry-producer-py39}

BUILD_DIR="$VIEWER_BUILD_DIR" PYTHON="$VIEWER_PYTHON" \
  "$ROOT/native/telemetry/build_offline.sh"

ANYDEX_TELEMETRY_BUILD_DIR="$CONTROL_READER_BUILD_DIR" \
ANYDEX_FRANKA_TAP_BUILD_DIR="$PRODUCER_BUILD_DIR" \
  "$ROOT/native/franka_tap/build_offline.sh"

CROSS_VERSION_DIR=$(mktemp -d /tmp/anydex-telemetry-cross-version.XXXXXX)
trap 'rm -rf "$CROSS_VERSION_DIR"' EXIT
CROSS_VERSION_MAP="$CROSS_VERSION_DIR/session.map"
PYTHONPATH="$PRODUCER_BUILD_DIR/python" \
  "$CONTROL_PYTHON" \
  "$ROOT/native/franka_tap/tests/publish_cross_version_fixture.py" \
  "$CROSS_VERSION_MAP"
PYTHONPATH="$VIEWER_BUILD_DIR/python" \
  "$VIEWER_PYTHON" \
  "$ROOT/native/telemetry/tests/read_cross_version_fixture.py" \
  "$CROSS_VERSION_MAP"

echo "continuous telemetry offline build complete"
echo "viewer reader (Python 3.10): $VIEWER_BUILD_DIR/python"
echo "executor producer (Python 3.9): $PRODUCER_BUILD_DIR/python"
echo "control-side test reader (Python 3.9 only): $CONTROL_READER_BUILD_DIR/python"
