#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/.." && pwd)"
PYTHON_BIN="${DYNAMIC_PCD_PYTHON:-${WORKSPACE_DIR}/.venv/bin/python}"
CAMERA_SERIAL="342222071785"
CONFIG="${WORKSPACE_DIR}/dexgrasp/runs/task_configs/thrown_object-86cb06d5a30505a8.yaml"
CONFIG_SHA256="86cb06d5a30505a8fc69e2c2348654065449416abc387390e494af4d8144d102"
OUTPUT_ROOT="${1:-${WORKSPACE_DIR}/object_pcd_testdata}"
RUN_ID="${2:-thrown_catch_workspace_424x240_60hz_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
OBJECT_TEXT="${THROWN_BALL_OBJECT_TEXT:-small red ball}"
DURATION_S="${THROWN_BALL_DURATION_S:-15}"
COUNTDOWN_S="${THROWN_BALL_COUNTDOWN_S:-5}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing content-addressed thrown-task config: ${CONFIG}" >&2
  exit 2
fi
if ! echo "${CONFIG_SHA256}  ${CONFIG}" | sha256sum --check --status; then
  echo "Thrown-task config hash mismatch; refusing an unpinned recording." >&2
  exit 2
fi

# Read the selected V57 target-center box from the same sealed contract that
# owns reset and ballistic sampling.  Refuse malformed/partial output rather
# than silently falling back to a copied green box.
cd "${WORKSPACE_DIR}"
mapfile -t V57_VALUES < <(
  "${PYTHON_BIN}" -m sim2real.tasks.thrown_contract \
    --resolved-config "${CONFIG}" \
    --recorder-lines
)
if [[ "${#V57_VALUES[@]}" -ne 4 ]]; then
  echo "V57 task contract did not return four recorder values." >&2
  exit 2
fi
V57_CURRICULUM="${V57_VALUES[0]}"
read -r -a WORKSPACE_MIN <<< "${V57_VALUES[1]}"
read -r -a WORKSPACE_MAX <<< "${V57_VALUES[2]}"
V57_CONTRACT_SHA256="${V57_VALUES[3]}"
if [[ "${#WORKSPACE_MIN[@]}" -ne 3 || "${#WORKSPACE_MAX[@]}" -ne 3 ]]; then
  echo "V57 task contract returned malformed target bounds." >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite an existing recording: ${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

echo "[CAMERA ONLY] Franka/RH56 interfaces will not be opened."
echo "[PROFILE] serial=${CAMERA_SERIAL} native=424x240@60Hz RGB exposure=auto"
echo "[DEPTH PROFILE] exposure=2000 gain=32 laser=360"
echo "[V57 CONTRACT] curriculum=${V57_CURRICULUM} sha256=${V57_CONTRACT_SHA256}"
echo "[CATCH TARGET-CENTER BOX robot_base] min=${WORKSPACE_MIN[*]} max=${WORKSPACE_MAX[*]}"
echo "[PREPARE] Keep the target outside the image during the ${COUNTDOWN_S}s countdown."
echo "[ACTION] During RECORDING, throw through the green box three or more times."
echo "[OUTPUT] ${OUTPUT_DIR}"

exec "${PYTHON_BIN}" -m dynamic_pcd.apps.record_rgbd_case \
  --config "${CONFIG}" \
  --case thrown_ball \
  --output "${OUTPUT_DIR}" \
  --duration-s "${DURATION_S}" \
  --countdown-s "${COUNTDOWN_S}" \
  --camera-serial "${CAMERA_SERIAL}" \
  --width 424 \
  --height 240 \
  --fps 60 \
  --max-color-depth-timestamp-skew-ms 7 \
  --object-text "${OBJECT_TEXT}" \
  --overlay-base-workspace-min-m "${WORKSPACE_MIN[@]}" \
  --overlay-base-workspace-max-m "${WORKSPACE_MAX[@]}"
