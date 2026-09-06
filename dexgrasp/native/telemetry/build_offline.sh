#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-/tmp/anydex-native-telemetry-build}
CXX=${CXX:-/usr/bin/g++-9}
PYTHON=${PYTHON:-/home/qiaoguanren/code/franka/.venv/bin/python}
CMAKE=${CMAKE:-/usr/bin/cmake}

if [[ ! -x "$CXX" ]]; then
  echo "error: C++ compiler is not executable: $CXX" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python interpreter is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -x "$CMAKE" ]]; then
  echo "error: CMake is not executable: $CMAKE" >&2
  exit 2
fi
if [[ ! -f /usr/lib/cmake/pybind11/pybind11Config.cmake ]]; then
  echo "error: local pybind11 CMake package is missing" >&2
  exit 2
fi

"$CMAKE" \
  -S "$HERE" \
  -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DPython3_EXECUTABLE="$PYTHON" \
  -Dpybind11_DIR=/usr/lib/cmake/pybind11 \
  -DANYDEX_TELEMETRY_BUILD_PYTHON=ON \
  -DANYDEX_TELEMETRY_BUILD_TESTS=ON \
  -DANYDEX_TELEMETRY_BUILD_TOOLS=ON
"$CMAKE" --build "$BUILD_DIR" --parallel
"$CMAKE" -E env \
  "PYTHONPATH=$BUILD_DIR/python" \
  "$CMAKE" --build "$BUILD_DIR" --target test

echo "offline native telemetry build verified: $BUILD_DIR"
