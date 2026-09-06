#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=configure_franka_nic_irq.sh
source "$SCRIPT_DIR/configure_franka_nic_irq.sh"

TEST_ROOT=$(mktemp -d /tmp/franka-nic-irq-guard-test.XXXXXX)
cleanup() {
  if [[ "$TEST_ROOT" == /tmp/franka-nic-irq-guard-test.* ]] &&
     [[ -d "$TEST_ROOT" ]]; then
    rm -rf -- "$TEST_ROOT"
  fi
}
trap cleanup EXIT

STATE_DIR="$TEST_ROOT/run/franka-nic-irq-guard"
STATE_FILE="$STATE_DIR/state"
SYS_CLASS_NET_ROOT="$TEST_ROOT/sys/class/net"
SYS_CPU_ROOT="$TEST_ROOT/sys/devices/system/cpu"
PROC_IRQ_ROOT="$TEST_ROOT/proc/irq"
SYSTEMD_RUNTIME_UNIT="$TEST_ROOT/run/systemd/system/irqbalance.service"
MOCK_ACTIVE_FILE="$TEST_ROOT/irqbalance.active"
MOCK_FAIL_MASK_FILE="$TEST_ROOT/fail-mask"

require_root() {
  :
}

install() {
  local destination="${!#}"
  mkdir -p "$destination"
  chmod 0755 "$destination"
}

chown() {
  :
}

state_storage_is_valid() {
  [[ -d "$STATE_DIR" ]] &&
    [[ ! -L "$STATE_DIR" ]] &&
    [[ -f "$STATE_FILE" ]] &&
    [[ ! -L "$STATE_FILE" ]] &&
    [[ "$(stat -c '%a' "$STATE_DIR")" == "755" ]] &&
    [[ "$(stat -c '%a' "$STATE_FILE")" == "644" ]]
}

irqbalance_processes() {
  if [[ "$(cat "$MOCK_ACTIVE_FILE")" == "1" ]]; then
    echo "4242"
  fi
}

systemctl() {
  local operation="$1"
  shift
  case "$operation" in
    is-active)
      if [[ "$(cat "$MOCK_ACTIVE_FILE")" == "1" ]]; then
        echo "active"
        return 0
      fi
      echo "inactive"
      return 3
      ;;
    mask)
      local now="0"
      local argument
      for argument in "$@"; do
        [[ "$argument" == "--now" ]] && now="1"
      done
      mkdir -p "$(dirname "$SYSTEMD_RUNTIME_UNIT")"
      ln -sfn /dev/null "$SYSTEMD_RUNTIME_UNIT"
      [[ "$now" == "1" ]] && echo "0" > "$MOCK_ACTIVE_FILE"
      if [[ -e "$MOCK_FAIL_MASK_FILE" ]]; then
        rm -f "$MOCK_FAIL_MASK_FILE"
        return 1
      fi
      ;;
    unmask)
      rm -f "$SYSTEMD_RUNTIME_UNIT"
      ;;
    start)
      echo "1" > "$MOCK_ACTIVE_FILE"
      ;;
    stop)
      echo "0" > "$MOCK_ACTIVE_FILE"
      ;;
    *)
      echo "unexpected mock systemctl operation: $operation" >&2
      return 2
      ;;
  esac
}

write_irq_affinity() {
  local irq="$1"
  local value="$2"
  echo "$value" > "$PROC_IRQ_ROOT/$irq/smp_affinity_list"
  echo "$value" > "$PROC_IRQ_ROOT/$irq/effective_affinity_list"
}

reset_fake_host() {
  rm -rf -- "$TEST_ROOT/run" "$TEST_ROOT/sys" "$TEST_ROOT/proc"
  mkdir -p "$SYS_CLASS_NET_ROOT/franka0/device/msi_irqs"
  mkdir -p "$SYS_CPU_ROOT/cpu6" "$SYS_CPU_ROOT/cpu7"
  mkdir -p "$PROC_IRQ_ROOT/127"
  touch "$SYS_CLASS_NET_ROOT/franka0/device/msi_irqs/127"
  echo "5" > "$PROC_IRQ_ROOT/127/smp_affinity_list"
  echo "5" > "$PROC_IRQ_ROOT/127/effective_affinity_list"
  echo "1" > "$MOCK_ACTIVE_FILE"
  rm -f "$MOCK_FAIL_MASK_FILE"
  APPLY_COMPLETED="0"
  APPLY_SYSTEM_MUTATED="0"
  APPLY_TEMP_FILE=""
}

reset_fake_host
apply_command franka0 6
trap cleanup EXIT
[[ "$(cat "$PROC_IRQ_ROOT/127/smp_affinity_list")" == "6" ]]
[[ "$(cat "$MOCK_ACTIVE_FILE")" == "0" ]]
[[ "$(read_runtime_masked)" == "1" ]]
grep -Fx "schema 2" "$STATE_FILE" >/dev/null
grep -Fx "interface franka0" "$STATE_FILE" >/dev/null
grep -Fx "pinned_cpu 6" "$STATE_FILE" >/dev/null
grep -Fx "irq 127 5 5" "$STATE_FILE" >/dev/null
status_command franka0 >/dev/null
restore_internal
[[ "$(cat "$PROC_IRQ_ROOT/127/smp_affinity_list")" == "5" ]]
[[ "$(cat "$MOCK_ACTIVE_FILE")" == "1" ]]
[[ "$(read_runtime_masked)" == "0" ]]
[[ ! -e "$STATE_DIR" ]]

reset_fake_host
mkdir -p "$(dirname "$SYSTEMD_RUNTIME_UNIT")"
ln -s /dev/null "$SYSTEMD_RUNTIME_UNIT"
echo "0" > "$MOCK_ACTIVE_FILE"
apply_command franka0 7
trap cleanup EXIT
restore_internal
[[ "$(cat "$PROC_IRQ_ROOT/127/smp_affinity_list")" == "5" ]]
[[ "$(cat "$MOCK_ACTIVE_FILE")" == "0" ]]
[[ "$(read_runtime_masked)" == "1" ]]
[[ ! -e "$STATE_DIR" ]]

reset_fake_host
apply_command franka0 6
trap cleanup EXIT
echo "unknown_record unsafe" >> "$STATE_FILE"
set +e
restore_internal
invalid_state_status=$?
set -e
[[ "$invalid_state_status" != "0" ]]
[[ "$(cat "$PROC_IRQ_ROOT/127/smp_affinity_list")" == "6" ]]
[[ "$(cat "$MOCK_ACTIVE_FILE")" == "0" ]]
[[ "$(read_runtime_masked)" == "1" ]]
[[ -f "$STATE_FILE" ]]

reset_fake_host
touch "$MOCK_FAIL_MASK_FILE"
set +e
(
  apply_command franka0 6
)
failure_status=$?
set -e
[[ "$failure_status" != "0" ]]
[[ "$(cat "$PROC_IRQ_ROOT/127/smp_affinity_list")" == "5" ]]
[[ "$(cat "$MOCK_ACTIVE_FILE")" == "1" ]]
[[ "$(read_runtime_masked)" == "0" ]]
[[ ! -e "$STATE_DIR" ]]

echo "PASS: hardware-free Franka NIC IRQ apply/rollback/restore state machine"
