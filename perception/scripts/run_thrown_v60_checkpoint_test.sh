#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/workspace.sh"

WORKSPACE_ROOT="${FRANKA_WORKSPACE_ROOT}"
PYTHON_BIN="${FRANKA_PYTHON_BIN}"
CANDIDATE="${THROWN_V60_CANDIDATE:-milddr-e60}"
case "${CANDIDATE}" in
  milddr-e60)
    BUNDLE_DIR="${WORKSPACE_ROOT}/data/checkpoints/thrown/thrown_v60_palmcatch_dagger_milddr_e60_a050_candidate_20260814/thrown_v60_palmcatch_dagger_milddr_e60_a050_candidate_20260814"
    EXPECTED_CHECKPOINT_SHA256="f6ac6400c9a6d280942300705a168d77a5cf70fa6ce32eb26790a805c6e043e7"
    ;;
  r3-reach-e100)
    BUNDLE_DIR="${WORKSPACE_ROOT}/data/checkpoints/thrown/thrown_v60_palmcatch_dagger_r3_reach_e100_a050_drvalidated_candidate_20260814/thrown_v60_palmcatch_dagger_r3_reach_e100_a050_drvalidated_candidate_20260814"
    EXPECTED_CHECKPOINT_SHA256="a3c56e90c81914eebd189e12c2cd4dfee73674ac2b05a4b5ed1b1b8c4447c666"
    ;;
  *)
    echo "[REFUSED] unknown THROWN_V60_CANDIDATE=${CANDIDATE}; expected milddr-e60 or r3-reach-e100" >&2
    exit 2
    ;;
esac
CHECKPOINT="${BUNDLE_DIR}/checkpoint.pt"
PROFILE="${WORKSPACE_ROOT}/dexgrasp/configs/fr3_rh56_v60_palmcatch_shadow.json"
FIRST_MOTION_PROFILE="${WORKSPACE_ROOT}/dexgrasp/configs/fr3_rh56_v60_palmcatch_first_motion.json"
OBJECT_TEXT="${THROWN_V60_OBJECT_TEXT:-small patterned beanbag toy}"
MODE="${1:-}"

usage() {
  echo "usage: $0 {admit|audit|config|reset-plan|perception|shadow|first-motion|rollout} [extra args...]" >&2
  echo "  admit      checkpoint I/O/controller admission; no camera or robot" >&2
  echo "  audit      show V60 reset/camera mismatches; no hardware" >&2
  echo "  config     materialize the content-addressed V60 task config" >&2
  echo "  reset-plan print the slow segmented V60 reset; no hardware" >&2
  echo "  perception camera-only trigger/full-flight test; robot interfaces stay closed" >&2
  echo "  shadow     read-only Franka/RH56 state + policy inference; robot writes=0" >&2
  echo "  first-motion operator-supervised V60 test capped at one commanded tick" >&2
  echo "  rollout    complete operator-supervised V60 episode: 72 ticks / 3.6 s" >&2
  echo "  select checkpoint with THROWN_V60_CANDIDATE=milddr-e60|r3-reach-e100" >&2
}

if [[ -z "${MODE}" ]]; then
  usage
  exit 2
fi
shift

cd "${WORKSPACE_ROOT}"
ACTUAL_CHECKPOINT_SHA256="$(franka_verify_sha256 \
  "${CHECKPOINT}" "${EXPECTED_CHECKPOINT_SHA256}" \
  "V60 ${CANDIDATE} checkpoint")"
echo "[V60 candidate] name=${CANDIDATE} checkpoint_sha256=${ACTUAL_CHECKPOINT_SHA256}"

case "${MODE}" in
  admit)
    exec "${PYTHON_BIN}" -m sim2real.deployment.verify \
      --checkpoint "${CHECKPOINT}" \
      --policy-rate-hz 20 \
      --json \
      "$@"
    ;;
  audit)
    exec "${PYTHON_BIN}" -m sim2real.diagnostics.audit_thrown_v60_candidate \
      --checkpoint "${CHECKPOINT}" \
      "$@"
    ;;
  config)
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object_v60 config "$@"
    ;;
  reset-plan)
    exec "${PYTHON_BIN}" -m dexgrasp.apps.reset_franka_v60_thrown "$@"
    ;;
  perception)
    RUN_ID="${THROWN_V60_RUN_ID:-$(franka_timestamped_run_id thrown-v60-perception)}"
    VIDEO="${THROWN_V60_VIDEO:-$(franka_default_video_path "${RUN_ID}")}"
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object_v60 perception \
      --profile "${PROFILE}" \
      --test-rollout-trigger \
      --post-trigger-capture-s 3 \
      --object-text "${OBJECT_TEXT}" \
      --run-id "${RUN_ID}" \
      --record-video "${VIDEO}" \
      --print-every 10 \
      "$@"
    ;;
  shadow)
    RUN_ID="${THROWN_V60_RUN_ID:-$(franka_timestamped_run_id thrown-v60-shadow)}"
    VIDEO="${THROWN_V60_VIDEO:-$(franka_default_video_path "${RUN_ID}")}"
    echo "[READ ONLY] Franka/RH56 commands are unavailable in this mode."
    echo "[PRECONDITION] Franka must already be within 0.005 rad of the V60 q_home."
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object_v60 shadow \
      --checkpoint "${CHECKPOINT}" \
      --profile "${PROFILE}" \
      --test-rollout-trigger \
      --post-trigger-capture-s 3 \
      --object-text "${OBJECT_TEXT}" \
      --run-id "${RUN_ID}" \
      --record-video "${VIDEO}" \
      --print-every 10 \
      "$@"
    ;;
  first-motion)
    RUN_ID="${THROWN_V60_RUN_ID:-$(franka_timestamped_run_id thrown-v60-first-motion)}"
    VIDEO="${THROWN_V60_VIDEO:-$(franka_default_video_path "${RUN_ID}")}"
    echo "[V60 FIRST MOTION] Exactly one commanded 20 Hz policy tick is permitted."
    echo "[SUPERVISION] Keep the emergency stop ready; throw only after ARMED."
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object_v60 deploy \
      --checkpoint "${CHECKPOINT}" \
      --profile "${FIRST_MOTION_PROFILE}" \
      --steps 1 \
      --run-id "${RUN_ID}" \
      --object-text "${OBJECT_TEXT}" \
      --record-video "${VIDEO}" \
      --record-policy-io \
      --execute \
      --yes-i-am-supervising \
      --live-visualization \
      --live-visualization-rate-hz 10 \
      "$@"
    ;;
  rollout)
    RUN_ID="${THROWN_V60_RUN_ID:-$(franka_timestamped_run_id thrown-v60-rollout)}"
    VIDEO="${THROWN_V60_VIDEO:-$(franka_default_video_path "${RUN_ID}")}"
    echo "[V60 FULL ROLLOUT] 72 commanded ticks at 20 Hz (3.6 s)."
    echo "[SUPERVISION] Keep the emergency stop ready; throw only after ARMED."
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object_v60 deploy \
      --checkpoint "${CHECKPOINT}" \
      --profile "${FIRST_MOTION_PROFILE}" \
      --steps 72 \
      --run-id "${RUN_ID}" \
      --object-text "${OBJECT_TEXT}" \
      --record-video "${VIDEO}" \
      --record-policy-io \
      --execute \
      --yes-i-am-supervising \
      --live-visualization \
      --live-visualization-rate-hz 10 \
      "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
