#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/workspace.sh"

WORKSPACE_ROOT="${FRANKA_WORKSPACE_ROOT}"
PYTHON_BIN="${FRANKA_PYTHON_BIN}"
CHECKPOINT="${WORKSPACE_ROOT}/data/checkpoints/thrown/test/student_pretrain_best_action.pt"
CHECKPOINT_SHA256="55bec1ca17ee925f797c8c457e7693f51283c7c486bba83dc36722695d2357ab"
COMMISSIONING_PROFILE="${WORKSPACE_ROOT}/dexgrasp/configs/fr3_rh56_v57_thrown_alpha0p5_20hz_commissioned.json"

MODE="${1:-admit}"
if [[ $# -gt 0 ]]; then
  shift
fi

franka_verify_sha256 \
  "${CHECKPOINT}" "${CHECKPOINT_SHA256}" "checkpoint" >/dev/null

case "${MODE}" in
  admit)
    # Purely offline: safe-decode the checkpoint and run the packaged
    # observation/action compatibility smoke test. No device is opened.
    exec "${PYTHON_BIN}" -m sim2real.deployment.verify \
      --checkpoint "${CHECKPOINT}" \
      --policy-rate-hz 20 \
      --json \
      "$@"
    ;;
  plan)
    # Purely offline: resolve and print the exact V57/camera/reset/controller
    # deployment contract that would be used by a real run.
    RUN_ID="${THROWN_RUN_ID:-thrown-plan}"
    OBJECT_TEXT="${THROWN_OBJECT_TEXT:-small red ball}"
    STEPS="${THROWN_STEPS:-20}"
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object deploy \
      --checkpoint "${CHECKPOINT}" \
      --profile "${COMMISSIONING_PROFILE}" \
      --steps "${STEPS}" \
      --run-id "${RUN_ID}" \
      --object-text "${OBJECT_TEXT}" \
      "$@"
    ;;
  perception)
    # Camera/GPU only: records the exact final mask and 128-point cloud used
    # by deployment. Franka and RH56 are never opened by this mode.
    RUN_ID="${THROWN_RUN_ID:-$(franka_timestamped_run_id thrown-perception)}"
    OBJECT_TEXT="${THROWN_OBJECT_TEXT:-small red ball}"
    DURATION_S="${THROWN_DURATION_S:-30}"
    VIDEO="${THROWN_VIDEO:-$(franka_default_video_path "${RUN_ID}")}"
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object perception \
      --run-id "${RUN_ID}" \
      --object-text "${OBJECT_TEXT}" \
      --duration "${DURATION_S}" \
      --record-video "${VIDEO}" \
      "$@"
    ;;
  shadow)
    # Read-only robot state + real camera point cloud + checkpoint inference.
    # Policy starts on the sealed throw trigger; no action mapper, reset, or
    # hardware command owner exists in this mode.
    RUN_ID="${THROWN_RUN_ID:-$(franka_timestamped_run_id thrown-shadow)}"
    OBJECT_TEXT="${THROWN_OBJECT_TEXT:-small red ball}"
    VIDEO="${THROWN_VIDEO:-$(franka_default_video_path "${RUN_ID}")}"
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object shadow \
      --checkpoint "${CHECKPOINT}" \
      --profile "${COMMISSIONING_PROFILE}" \
      --test-rollout-trigger \
      --post-trigger-capture-s 3 \
      --object-text "${OBJECT_TEXT}" \
      --run-id "${RUN_ID}" \
      --record-video "${VIDEO}" \
      "$@"
    ;;
  execute)
    # This path deliberately retains both deployment safety latches:
    # --yes-i-am-supervising and the task profile's accepted/enabled state.
    # Until thrown_object.yaml is commissioned, it refuses before any device
    # is opened rather than bypassing the point-cloud acceptance gate.
    RUN_ID="${THROWN_RUN_ID:-$(franka_timestamped_run_id thrown-ckpt)}"
    OBJECT_TEXT="${THROWN_OBJECT_TEXT:-small red ball}"
    STEPS="${THROWN_STEPS:-20}"
    VIDEO="${THROWN_VIDEO:-$(franka_default_video_path "${RUN_ID}")}"
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object deploy \
      --checkpoint "${CHECKPOINT}" \
      --profile "${COMMISSIONING_PROFILE}" \
      --steps "${STEPS}" \
      --run-id "${RUN_ID}" \
      --object-text "${OBJECT_TEXT}" \
      --record-video "${VIDEO}" \
      --record-policy-io \
      --execute \
      --yes-i-am-supervising \
      "$@"
    ;;
  *)
    echo "usage: $0 {admit|plan|perception|shadow|execute} [extra args]" >&2
    exit 2
    ;;
esac
