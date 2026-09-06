#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash tools/generate_fr3_candidate_urdf.sh {fr3|fr3v2} OUTPUT.urdf" >&2
  exit 2
fi

variant="$1"
output="$2"
root="${FRANKA_DESCRIPTION_ROOT:-/opt/ros/humble/share/franka_description}"

case "${variant}" in
  fr3)
    relative="robots/fr3/fr3.urdf.xacro"
    expected="f739aa0791d4db3ec9ac166cd77e8a4a0cc15a8284a881fad8271c6c7878c611"
    ;;
  fr3v2)
    relative="robots/fr3v2/fr3v2.urdf.xacro"
    expected="36dd55bfaf7b0d0583edd8b2dea0ee83df7002e83e6e6b33b1689913da17b463"
    ;;
  *)
    echo "Unknown candidate '${variant}'; expected fr3 or fr3v2." >&2
    exit 2
    ;;
esac

source_xacro="${root}/${relative}"
if [[ ! -f "${source_xacro}" ]]; then
  echo "Missing locked FR3 Xacro: ${source_xacro}" >&2
  exit 2
fi
if [[ -e "${output}" ]]; then
  echo "Refusing to overwrite existing output: ${output}" >&2
  exit 2
fi
if ! command -v xacro >/dev/null 2>&1; then
  echo "xacro is not installed or not on PATH." >&2
  exit 2
fi

actual="$(sha256sum "${source_xacro}" | awk '{print $1}')"
if [[ "${actual}" != "${expected}" ]]; then
  echo "Locked Xacro checksum mismatch for ${source_xacro}" >&2
  echo "expected=${expected}" >&2
  echo "actual=${actual}" >&2
  exit 2
fi

xacro "${source_xacro}" \
  hand:=false \
  ros2_control:=false \
  gazebo:=false \
  with_sc:=false \
  -o "${output}"

echo "Generated bare-arm ${variant} candidate: ${output}"
echo "Meshes remain external package://franka_description references."
echo "This does not identify the physical revision and does not model the real EEF/TCP/payload."
