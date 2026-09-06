#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

# Compile and audit the native supervised servo without constructing a Robot
# or touching any camera/serial device. Only the dependency identity probe and
# fake backend tests are executed.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SAFETY_PROFILE=${ANYDEX_V94_SAFETY_PROFILE:-v94}
case "$SAFETY_PROFILE" in
  v94)
    DEFAULT_TEMP_BUILD_DIR=/tmp/anydex-v94-franka-servo-build
    DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/build"
    ;;
  v94_tabletop)
    DEFAULT_TEMP_BUILD_DIR=/tmp/anydex-v94-tabletop-franka-servo-build
    DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/build_v94_tabletop"
    ;;
  v60_palmcatch)
    DEFAULT_TEMP_BUILD_DIR=/tmp/anydex-v60-palmcatch-franka-servo-build
    DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/build_v60_palmcatch"
    ;;
  v61_sixexpert)
    DEFAULT_TEMP_BUILD_DIR=/tmp/anydex-v61-sixexpert-franka-servo-build
    DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/build_v61_sixexpert"
    ;;
  *)
    echo "ANYDEX_V94_SAFETY_PROFILE must be v94, v94_tabletop, v60_palmcatch or v61_sixexpert" >&2
    exit 2
    ;;
esac
TEMP_BUILD_DIR=${ANYDEX_V94_TEMP_BUILD_DIR:-$DEFAULT_TEMP_BUILD_DIR}
OUTPUT_DIR=${ANYDEX_V94_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
CXX=${ANYDEX_CXX:-/usr/bin/g++-9}
CMAKE=${ANYDEX_CMAKE:-/usr/bin/cmake}
CTEST=${ANYDEX_CTEST:-/usr/bin/ctest}
NINJA=${ANYDEX_NINJA:-/home/qiaoguanren/anaconda3/bin/ninja}
LIBFRANKA_SOURCE_DIR=${ANYDEX_LIBFRANKA_SOURCE_DIR:-/home/qiaoguanren/code/libfranka}
LIBFRANKA_INCLUDE_DIR="$LIBFRANKA_SOURCE_DIR/include"
LIBFRANKA_COMMON_INCLUDE_DIR="$LIBFRANKA_SOURCE_DIR/common/include"
PYLIBFRANKA_ROOT=${ANYDEX_PYLIBFRANKA_ROOT:-/home/qiaoguanren/code/franka/.venv/lib/python3.9/site-packages}
PYLIBFRANKA_LIBS_DIR=${ANYDEX_PYLIBFRANKA_LIBS_DIR:-"$PYLIBFRANKA_ROOT/pylibfranka.libs"}
PYLIBFRANKA_HASHED_LIBRARY=${ANYDEX_PYLIBFRANKA_HASHED_LIBRARY:-"$PYLIBFRANKA_LIBS_DIR/libfranka-2be07f70.so.0.21.2"}

EXPECTED_LIBFRANKA_SHA256=956d2f7e85e3c4e127899734a170dff7c91f17f560a4fa2147739631ad721a3d
EXPECTED_LIBFRANKA_SOURCE_COMMIT=9f9304ec0ac897eff3219a67f612b959948535e2
EXPECTED_LIBFRANKA_SONAME=libfranka-2be07f70.so.0.21.2

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

require_file "$CXX"
require_file "$CMAKE"
require_file "$CTEST"
require_file "$NINJA"
require_file "$LIBFRANKA_INCLUDE_DIR/franka/robot.h"
require_file "$LIBFRANKA_COMMON_INCLUDE_DIR/research_interface/robot/service_types.h"
require_file "$PYLIBFRANKA_HASHED_LIBRARY"
verify_sha256 "$PYLIBFRANKA_HASHED_LIBRARY" "$EXPECTED_LIBFRANKA_SHA256"

actual_source_commit=$(git -C "$LIBFRANKA_SOURCE_DIR" rev-parse HEAD)
if [[ "$actual_source_commit" != "$EXPECTED_LIBFRANKA_SOURCE_COMMIT" ]]; then
  echo "libfranka header source commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "$LIBFRANKA_SOURCE_DIR" status --short --untracked-files=no)" ]]; then
  echo "tracked libfranka source has local modifications" >&2
  exit 2
fi

"$CMAKE" \
  -S "$SCRIPT_DIR" \
  -B "$TEMP_BUILD_DIR" \
  -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DANYDEX_V94_SAFETY_PROFILE="$SAFETY_PROFILE" \
  -DBUILD_TESTING=ON \
  -DLIBFRANKA_INCLUDE_DIR="$LIBFRANKA_INCLUDE_DIR" \
  -DLIBFRANKA_COMMON_INCLUDE_DIR="$LIBFRANKA_COMMON_INCLUDE_DIR" \
  -DLIBFRANKA_SOURCE_COMMIT="$EXPECTED_LIBFRANKA_SOURCE_COMMIT" \
  -DPYLIBFRANKA_HASHED_LIBRARY="$PYLIBFRANKA_HASHED_LIBRARY" \
  -DPYLIBFRANKA_HASHED_LIBRARY_SHA256="$EXPECTED_LIBFRANKA_SHA256" \
  -DPYLIBFRANKA_LIBS_DIR="$PYLIBFRANKA_LIBS_DIR"
"$CMAKE" --build "$TEMP_BUILD_DIR" --parallel 2
if [[ "${ANYDEX_V94_RESTRICTED_SANDBOX:-0}" == "1" ]]; then
  echo "restricted sandbox: compiling fake lifecycle but skipping its blocked socket runtime" >&2
  "$CTEST" --test-dir "$TEMP_BUILD_DIR" --output-on-failure \
    -R '(v225_interpolator|v94_(protocol|dependency_identity|python_cpp_arm_contract))_test'
else
  "$CTEST" --test-dir "$TEMP_BUILD_DIR" --output-on-failure
fi

BINARY="$TEMP_BUILD_DIR/v94_franka_servo"
require_file "$BINARY"
if ! readelf -d "$BINARY" | grep -F "Shared library: [$EXPECTED_LIBFRANKA_SONAME]" >/dev/null; then
  echo "production binary does not DT_NEEDED the exact hashed libfranka SONAME" >&2
  exit 2
fi
if ! readelf -d "$BINARY" | grep -F "Library rpath: [$PYLIBFRANKA_LIBS_DIR]" >/dev/null; then
  echo "production binary lacks the exact private DT_RPATH" >&2
  exit 2
fi
loaded_libfranka=$(ldd "$BINARY" | awk -v soname="$EXPECTED_LIBFRANKA_SONAME" '$1 == soname {print $3}')
if [[ -z "$loaded_libfranka" ||
      "$(readlink -f "$loaded_libfranka")" != "$(readlink -f "$PYLIBFRANKA_HASHED_LIBRARY")" ]]; then
  echo "loader map does not resolve the exact pinned wheel libfranka" >&2
  exit 2
fi
verify_sha256 "$loaded_libfranka" "$EXPECTED_LIBFRANKA_SHA256"

mkdir -p "$OUTPUT_DIR"
install -m 0755 "$BINARY" "$OUTPUT_DIR/v94_franka_servo.tmp"
mv -f "$OUTPUT_DIR/v94_franka_servo.tmp" "$OUTPUT_DIR/v94_franka_servo"
BINARY_SHA256=$(sha256sum "$OUTPUT_DIR/v94_franka_servo" | awk '{print $1}')
PRODUCER_BUILD_SHA256=$(tr -d '\n' < "$TEMP_BUILD_DIR/producer_build_sha256.txt")

"$CMAKE" \
  -DMANIFEST_TEMPLATE="$SCRIPT_DIR/manifest.json.in" \
  -DMANIFEST_OUTPUT="$OUTPUT_DIR/manifest.json.tmp" \
  -DBINARY_SHA256="$BINARY_SHA256" \
  -DPRODUCER_BUILD_SHA256="$PRODUCER_BUILD_SHA256" \
  -DLIBFRANKA_PATH="$PYLIBFRANKA_HASHED_LIBRARY" \
  -DLIBFRANKA_SHA256="$EXPECTED_LIBFRANKA_SHA256" \
  -DLIBFRANKA_SOURCE_COMMIT="$EXPECTED_LIBFRANKA_SOURCE_COMMIT" \
  -P "$SCRIPT_DIR/generate_manifest.cmake"
mv -f "$OUTPUT_DIR/manifest.json.tmp" "$OUTPUT_DIR/manifest.json"

echo "offline native servo verified: profile=$SAFETY_PROFILE $OUTPUT_DIR/v94_franka_servo"
echo "manifest: $OUTPUT_DIR/manifest.json"
echo "producer_build_sha256=$PRODUCER_BUILD_SHA256"
