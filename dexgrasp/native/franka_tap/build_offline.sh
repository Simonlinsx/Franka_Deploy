#!/usr/bin/env bash
set -euo pipefail

# Build and validate the fused libfranka readOnce telemetry tap without
# opening a Robot, serial device, camera, or GUI.  The adapter exchanges
# pylibfranka C++ types and therefore must match the exact installed wheel
# ABI; do not replace these checks with an unversioned system libfranka.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR=${ANYDEX_FRANKA_TAP_BUILD_DIR:-/tmp/anydex-franka-telemetry-build}
CORE_BUILD_DIR=${ANYDEX_TELEMETRY_BUILD_DIR:-/tmp/anydex-native-telemetry-build}
PYTHON=${ANYDEX_CONTROL_PYTHON:-/home/qiaoguanren/code/franka/.venv/bin/python}
CXX=${ANYDEX_CXX:-/usr/bin/g++-9}
PYLIBFRANKA_ROOT=${ANYDEX_PYLIBFRANKA_ROOT:-/home/qiaoguanren/code/franka/.venv/lib/python3.9/site-packages}
PYBIND_ROOT=${ANYDEX_PYBIND11_3_ROOT:-/tmp/dexgrasp-pybind-build/pybind11-3.0.1/pybind11}
LIBFRANKA_INCLUDE_DIR=${ANYDEX_LIBFRANKA_INCLUDE_DIR:-/home/qiaoguanren/code/libfranka/include}
LIBFRANKA_SOURCE_DIR=$(dirname "$LIBFRANKA_INCLUDE_DIR")

PYLIBFRANKA_MODULE="$PYLIBFRANKA_ROOT/pylibfranka/_pylibfranka.cpython-39-x86_64-linux-gnu.so"
PYLIBFRANKA_LIBS_DIR=${ANYDEX_PYLIBFRANKA_LIBS_DIR:-"$PYLIBFRANKA_ROOT/pylibfranka.libs"}
PYLIBFRANKA_HASHED_LIBRARY=${ANYDEX_PYLIBFRANKA_HASHED_LIBRARY:-"$PYLIBFRANKA_LIBS_DIR/libfranka-2be07f70.so.0.21.2"}
PYBIND11_DIR="$PYBIND_ROOT/share/cmake/pybind11"
PYBIND_BUILD_ROOT=$(dirname "$(dirname "$PYBIND_ROOT")")
PYBIND_WHEEL="$PYBIND_BUILD_ROOT/pybind11-3.0.1-py3-none-any.whl"
PYBIND_EXTRACT_ROOT="$PYBIND_BUILD_ROOT/pybind11-3.0.1"

EXPECTED_PYLIBFRANKA_SHA256=53ebc14276df92d0687e1673e43df17f0a4a347dedd9a0c08bb01f152e9f2610
EXPECTED_LIBFRANKA_SHA256=956d2f7e85e3c4e127899734a170dff7c91f17f560a4fa2147739631ad721a3d
EXPECTED_PYBIND_WHEEL_SHA256=aa8f0aa6e0a94d3b64adfc38f560f33f15e589be2175e103c0a33c6bce55ee89
EXPECTED_LIBFRANKA_SOURCE_COMMIT=9f9304ec0ac897eff3219a67f612b959948535e2

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required file is missing: $1" >&2
    exit 2
  fi
}

verify_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 2
  fi
}

require_file "$PYTHON"
require_file "$CXX"
require_file "$PYLIBFRANKA_MODULE"
require_file "$PYLIBFRANKA_HASHED_LIBRARY"
require_file "$LIBFRANKA_INCLUDE_DIR/franka/robot_state.h"
verify_sha256 "$PYLIBFRANKA_MODULE" "$EXPECTED_PYLIBFRANKA_SHA256"
verify_sha256 "$PYLIBFRANKA_HASHED_LIBRARY" "$EXPECTED_LIBFRANKA_SHA256"
actual_libfranka_commit=$(git -C "$LIBFRANKA_SOURCE_DIR" rev-parse HEAD)
if [[ "$actual_libfranka_commit" != "$EXPECTED_LIBFRANKA_SOURCE_COMMIT" ]]; then
  echo "libfranka header source commit mismatch" >&2
  echo "expected=$EXPECTED_LIBFRANKA_SOURCE_COMMIT" >&2
  echo "actual=$actual_libfranka_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$LIBFRANKA_SOURCE_DIR" status --short --untracked-files=no)" ]]; then
  echo "tracked libfranka header source has local modifications" >&2
  exit 2
fi

if [[ ! -f "$PYBIND11_DIR/pybind11Config.cmake" ]]; then
  mkdir -p "$PYBIND_BUILD_ROOT"
  "$PYTHON" -m pip download \
    --disable-pip-version-check \
    --no-deps \
    --require-hashes \
    --dest "$PYBIND_BUILD_ROOT" \
    --requirement "$ROOT_DIR/native/franka_tap/requirements-build.txt"
  require_file "$PYBIND_WHEEL"
  verify_sha256 "$PYBIND_WHEEL" "$EXPECTED_PYBIND_WHEEL_SHA256"
  mkdir -p "$PYBIND_EXTRACT_ROOT"
  "$PYTHON" -m zipfile -e "$PYBIND_WHEEL" "$PYBIND_EXTRACT_ROOT"
fi
require_file "$PYBIND11_DIR/pybind11Config.cmake"

BUILD_DIR="$CORE_BUILD_DIR" PYTHON="$PYTHON" CXX="$CXX" \
  "$ROOT_DIR/native/telemetry/build_offline.sh"

/usr/bin/cmake \
  -S "$ROOT_DIR/native/franka_tap" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DPython3_EXECUTABLE="$PYTHON" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DLIBFRANKA_INCLUDE_DIR="$LIBFRANKA_INCLUDE_DIR" \
  -DPYLIBFRANKA_MODULE="$PYLIBFRANKA_MODULE" \
  -DPYLIBFRANKA_MODULE_SHA256="$EXPECTED_PYLIBFRANKA_SHA256" \
  -DPYLIBFRANKA_HASHED_LIBRARY="$PYLIBFRANKA_HASHED_LIBRARY" \
  -DPYLIBFRANKA_HASHED_LIBRARY_SHA256="$EXPECTED_LIBFRANKA_SHA256" \
  -DPYLIBFRANKA_LIBS_DIR="$PYLIBFRANKA_LIBS_DIR"
/usr/bin/cmake --build "$BUILD_DIR" --parallel 2

PYTHONPATH="$BUILD_DIR/python:$CORE_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -c \
  'import _anydex_franka_telemetry; import pylibfranka; print("adapter-first import passed")'
PYTHONPATH="$BUILD_DIR/python:$CORE_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -c \
  'import pylibfranka; import _anydex_franka_telemetry; print("pylibfranka-first import passed")'

PYTHONPATH="$BUILD_DIR/python:$CORE_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}" \
  ANYDEX_EXPECTED_LIBFRANKA="$PYLIBFRANKA_HASHED_LIBRARY" \
  "$PYTHON" "$ROOT_DIR/native/franka_tap/tests/test_offline_binding.py"

PYTHONPATH="$BUILD_DIR/python:$CORE_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}" \
  ANYDEX_EXPECTED_LIBFRANKA="$PYLIBFRANKA_HASHED_LIBRARY" \
  "$PYTHON" "$ROOT_DIR/native/franka_tap/tests/test_native_cartesian_segment.py"

echo "offline fused Franka telemetry tap verified: $BUILD_DIR/python"
