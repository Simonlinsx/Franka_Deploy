#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/workspace.sh"

WORKSPACE_ROOT="${FRANKA_WORKSPACE_ROOT}"
PYTHON_BIN="${FRANKA_PYTHON_BIN}"
BUNDLE_DIR="${WORKSPACE_ROOT}/data/checkpoints/thrown/thrown_v61_sixexpert_visualflight_perceptiondr025_cmp31p9_demo_candidate_20260815/thrown_v61_sixexpert_visualflight_perceptiondr025_cmp31p9_demo_candidate_20260815"
EXPERT="${THROWN_V61_EXPERT:-base-forward}"
OBJECT_TEXT="${THROWN_V61_OBJECT_TEXT:-small patterned beanbag toy}"
MODE="${1:-}"

case "${EXPERT}" in
  base-forward)
    CHECKPOINT="${BUNDLE_DIR}/checkpoint.pt"
    CHECKPOINT_SHA256="f64c8e7ef9561711f65cde3460aab42934803ff5606133bceca3de05225f9f2c"
    ;;
  base-negative-y)
    CHECKPOINT="${BUNDLE_DIR}/checkpoints/base/side_negative_y.pt"
    CHECKPOINT_SHA256="c285adb97fc3a71827cb792b3a7ed0077d1d23a719211c611fac6b4ca2b24836"
    ;;
  base-positive-y)
    CHECKPOINT="${BUNDLE_DIR}/checkpoints/base/side_positive_y.pt"
    CHECKPOINT_SHA256="480f7bdfea004465dffb93e8f65cf37ec2813ce96203b6b8e7869367f93d0d3b"
    ;;
  high-forward)
    CHECKPOINT="${BUNDLE_DIR}/checkpoints/high/forward.pt"
    CHECKPOINT_SHA256="6ea08f2b612a92b84aaa86234d0c3558a7364c49756d7e3f88a64f9d431a293f"
    ;;
  high-negative-y)
    CHECKPOINT="${BUNDLE_DIR}/checkpoints/high/side_negative_y.pt"
    CHECKPOINT_SHA256="eac66c813398cec1b96d6cdc7288197fa943122d560bd3ae1af7dd2f846d9fea"
    ;;
  high-positive-y)
    CHECKPOINT="${BUNDLE_DIR}/checkpoints/high/side_positive_y.pt"
    CHECKPOINT_SHA256="2964b601ac043be547ba46fc2c7d68b355e08a2c989ec41da93147a3dca579b4"
    ;;
  *)
    echo "[REFUSED] unknown THROWN_V61_EXPERT=${EXPERT}" >&2
    echo "Expected: base-forward, base-negative-y, base-positive-y, high-forward, high-negative-y, high-positive-y" >&2
    exit 2
    ;;
esac

usage() {
  echo "usage: $0 {audit|audit-all|runtime-admit|config|reset-plan|set-default|perception|shadow|test-40|rollout}" >&2
  echo "Select one checkpoint with THROWN_V61_EXPERT=<expert>." >&2
  echo "audit/audit-all verify all six hashes/specs/controllers; hardware writes=0." >&2
  echo "runtime-admit verifies the selected checkpoint and its exact 17-D runtime contract." >&2
  echo "config materializes the V61 camera/task contract; hardware writes=0." >&2
  echo "reset-plan validates the V61 reset dry-run; set-default performs the confirmed low-speed Franka-only reset." >&2
  echo "perception runs camera-only mask/point-cloud testing; hardware writes=0." >&2
  echo "shadow runs one selected expert read-only; test-40 permits exactly 40 supervised ticks; rollout remains evidence-gated." >&2
}

if [[ -z "${MODE}" ]]; then
  usage
  exit 2
fi
shift
cd "${WORKSPACE_ROOT}"

ACTUAL_SHA256="$(franka_verify_sha256 \
  "${CHECKPOINT}" "${CHECKPOINT_SHA256}" \
  "V61 ${EXPERT} checkpoint")"
echo "[V61 expert] name=${EXPERT} checkpoint_sha256=${ACTUAL_SHA256}"

verify_selected_runtime_alignment() {
  "${PYTHON_BIN}" -m sim2real.diagnostics.audit_thrown_v61_bundle \
    --bundle "${BUNDLE_DIR}" \
    --expert "${EXPERT}" \
    --only-selected \
    --require-runtime-ready \
    >/dev/null
  echo "[V61 runtime alignment PASS] expert=${EXPERT}; router=disabled; loaded_checkpoints=1"
}

case "${MODE}" in
  audit)
    exec "${PYTHON_BIN}" -m sim2real.diagnostics.audit_thrown_v61_bundle \
      --bundle "${BUNDLE_DIR}" --expert "${EXPERT}" --only-selected "$@"
    ;;
  audit-all)
    exec "${PYTHON_BIN}" -m sim2real.diagnostics.audit_thrown_v61_bundle \
      --bundle "${BUNDLE_DIR}" --expert "${EXPERT}" "$@"
    ;;
  runtime-admit)
    verify_selected_runtime_alignment
    exec "${PYTHON_BIN}" -m sim2real.deployment.verify \
      --checkpoint "${CHECKPOINT}" --policy-rate-hz 20 --json "$@"
    ;;
  config)
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object_v61 config "$@"
    ;;
  reset-plan)
    exec "${PYTHON_BIN}" -m dexgrasp.apps.reset_franka_v61_thrown "$@"
    ;;
  set-default)
    exec "${PYTHON_BIN}" -m dexgrasp.apps.reset_franka_v61_thrown --execute "$@"
    ;;
  perception)
    RUN_ID="${THROWN_V61_RUN_ID:-$(franka_timestamped_run_id thrown-v61-perception)}"
    VIDEO="${THROWN_V61_VIDEO:-$(franka_default_video_path "${RUN_ID}")}"
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object_v61 perception \
      --test-rollout-trigger \
      --post-trigger-capture-s 3 \
      --object-text "${OBJECT_TEXT}" \
      --run-id "${RUN_ID}" \
      --record-video "${VIDEO}" \
      --print-every 10 \
      "$@"
    ;;
  shadow)
    RUN_ID="${THROWN_V61_RUN_ID:-$(franka_timestamped_run_id "thrown-v61-${EXPERT}-shadow")}"
    VIDEO="${THROWN_V61_VIDEO:-$(franka_default_video_path "${RUN_ID}")}"
    PROFILE="${WORKSPACE_ROOT}/dexgrasp/configs/fr3_rh56_v61_sixexpert_shadow.json"
    verify_selected_runtime_alignment
    echo "[READ ONLY] Exactly one expert is loaded; automatic router is disabled."
    echo "[READ ONLY] Franka/RH56 commands are unavailable in this mode."
    echo "[PRECONDITION] Franka must already be within 0.005 rad of the V61 q_home."
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object_v61 shadow \
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
  test-40)
    RUN_ID="${THROWN_V61_RUN_ID:-$(franka_timestamped_run_id "thrown-v61-${EXPERT}-test40")}"
    VIDEO="${THROWN_V61_VIDEO:-$(franka_default_video_path "${RUN_ID}")}"
    PROFILE="${WORKSPACE_ROOT}/dexgrasp/configs/fr3_rh56_v61_sixexpert_40tick.json"
    verify_selected_runtime_alignment
    echo "[V61 SUPERVISED TEST] Exactly 40 commanded ticks at 20 Hz are permitted."
    echo "[SINGLE EXPERT] expert=${EXPERT}; automatic router disabled."
    echo "[SUPERVISION] Keep the emergency stop ready; throw only after ARMED."
    exec "${PYTHON_BIN}" -m sim2real.tasks.launcher thrown_object_v61 deploy \
      --checkpoint "${CHECKPOINT}" \
      --profile "${PROFILE}" \
      --steps 40 \
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
    echo "[REFUSED] V61 full rollout is not enabled yet." >&2
    echo "Run test-40 first; its completed hardware audit and policy I/O must be reviewed before raising the 40-tick cap." >&2
    exit 3
    ;;
  *)
    usage
    exit 2
    ;;
esac
