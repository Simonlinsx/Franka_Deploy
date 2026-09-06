#!/usr/bin/env bash
set -euo pipefail

# These are assigned unconditionally for normal execution.  Keeping them as
# variables also lets the dedicated hardware-free test source this file and
# point the functions at a temporary fake sysfs/procfs tree.
STATE_DIR="/run/franka-nic-irq-guard"
STATE_FILE="$STATE_DIR/state"
SYS_CLASS_NET_ROOT="/sys/class/net"
SYS_CPU_ROOT="/sys/devices/system/cpu"
PROC_IRQ_ROOT="/proc/irq"
SYSTEMD_RUNTIME_UNIT="/run/systemd/system/irqbalance.service"
IRQBALANCE_UNIT="irqbalance.service"
STATE_SCHEMA="2"

PARSED_INTERFACE=""
PARSED_PINNED_CPU=""
PARSED_IRQBALANCE_RUNTIME_MASKED=""
PARSED_IRQBALANCE_WAS_ACTIVE=""
PARSED_IRQS=()
PARSED_REQUESTED_AFFINITIES=()
PARSED_EFFECTIVE_AFFINITIES=()

APPLY_COMPLETED="0"
APPLY_TEMP_FILE=""
APPLY_SYSTEM_MUTATED="0"

usage() {
  echo "Usage:"
  echo "  sudo $0 apply --interface IFACE --cpu LOGICAL_CPU"
  echo "  sudo $0 restore"
  echo "  $0 status --interface IFACE"
}

require_root() {
  if [[ "$(id -u)" != "0" ]]; then
    echo "This action requires sudo." >&2
    exit 2
  fi
}

valid_cpu_list() {
  local value="$1"
  if [[ ! "$value" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]; then
    return 1
  fi
  local item lower upper
  local items=()
  IFS=',' read -r -a items <<< "$value"
  for item in "${items[@]}"; do
    if [[ "$item" == *-* ]]; then
      lower="${item%%-*}"
      upper="${item##*-}"
      if ((10#$upper < 10#$lower)); then
        return 1
      fi
    fi
  done
}

resolve_irqs() {
  local interface="$1"
  local device="$SYS_CLASS_NET_ROOT/$interface/device"
  if [[ ! "$interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || [[ ! -d "$device" ]]; then
    echo "Invalid or non-PCI network interface: $interface" >&2
    return 1
  fi
  local values=()
  if [[ -d "$device/msi_irqs" ]]; then
    local listing=""
    if ! listing="$(
      find "$device/msi_irqs" -mindepth 1 -maxdepth 1 -printf '%f\n' |
        sort -n
    )"; then
      echo "Could not enumerate MSI IRQs for $interface." >&2
      return 1
    fi
    if [[ -n "$listing" ]]; then
      mapfile -t values <<< "$listing"
    fi
    local irq
    for irq in "${values[@]}"; do
      if [[ ! "$irq" =~ ^[0-9]+$ ]]; then
        echo "Invalid MSI IRQ entry for $interface: $irq" >&2
        return 1
      fi
    done
  fi
  if [[ "${#values[@]}" == "0" ]] && [[ -r "$device/irq" ]]; then
    local legacy
    legacy="$(tr -d '[:space:]' < "$device/irq")"
    if [[ "$legacy" =~ ^[0-9]+$ ]]; then
      values+=("$legacy")
    fi
  fi
  if [[ "${#values[@]}" == "0" ]]; then
    echo "No IRQ is exposed for $interface." >&2
    return 1
  fi
  printf '%s\n' "${values[@]}"
}

load_irqs() {
  local interface="$1"
  local output=""
  if ! output="$(resolve_irqs "$interface")"; then
    return 1
  fi
  if [[ -z "$output" ]]; then
    echo "No IRQ is exposed for $interface." >&2
    return 1
  fi
  mapfile -t RESOLVED_IRQS <<< "$output"
  if [[ "${#RESOLVED_IRQS[@]}" == "0" ]]; then
    echo "No IRQ is exposed for $interface." >&2
    return 1
  fi
}

irqbalance_processes() {
  pgrep -x irqbalance 2>/dev/null || true
}

read_irqbalance_active() {
  local state
  state="$(systemctl is-active "$IRQBALANCE_UNIT" 2>/dev/null || true)"
  case "$state" in
    active)
      echo "1"
      ;;
    inactive|failed)
      echo "0"
      ;;
    *)
      echo "Could not establish a stable systemd state for $IRQBALANCE_UNIT: ${state:-unavailable}" >&2
      return 1
      ;;
  esac
}

read_runtime_masked() {
  if [[ -L "$SYSTEMD_RUNTIME_UNIT" ]] &&
     [[ "$(readlink "$SYSTEMD_RUNTIME_UNIT")" == "/dev/null" ]]; then
    echo "1"
  else
    echo "0"
  fi
}

read_irq_affinity() {
  local irq="$1"
  local kind="$2"
  local path="$PROC_IRQ_ROOT/$irq/${kind}_affinity_list"
  if [[ ! -r "$path" ]]; then
    echo "IRQ $irq has no readable ${kind}_affinity_list." >&2
    return 1
  fi
  local value
  value="$(tr -d '[:space:]' < "$path")"
  if ! valid_cpu_list "$value"; then
    echo "IRQ $irq has an invalid ${kind} affinity list: $value" >&2
    return 1
  fi
  echo "$value"
}

write_irq_affinity() {
  local irq="$1"
  local value="$2"
  printf '%s\n' "$value" > "$PROC_IRQ_ROOT/$irq/smp_affinity_list"
}

state_storage_is_valid() {
  if [[ ! -d "$STATE_DIR" ]] || [[ -L "$STATE_DIR" ]] ||
     [[ ! -f "$STATE_FILE" ]] || [[ -L "$STATE_FILE" ]]; then
    echo "Saved IRQ state path is missing, not regular, or uses a symlink." >&2
    return 1
  fi
  local directory_owner directory_group directory_mode
  local file_owner file_group file_mode
  directory_owner="$(stat -c '%u' "$STATE_DIR")"
  directory_group="$(stat -c '%g' "$STATE_DIR")"
  directory_mode="$(stat -c '%a' "$STATE_DIR")"
  file_owner="$(stat -c '%u' "$STATE_FILE")"
  file_group="$(stat -c '%g' "$STATE_FILE")"
  file_mode="$(stat -c '%a' "$STATE_FILE")"
  if [[ "$directory_owner" != "0" ]] || [[ "$directory_group" != "0" ]] ||
     [[ "$directory_mode" != "755" ]] || [[ "$file_owner" != "0" ]] ||
     [[ "$file_group" != "0" ]] || [[ "$file_mode" != "644" ]]; then
    echo "Saved IRQ state must be root-owned with directory mode 0755 and file mode 0644." >&2
    return 1
  fi
}

parse_state_file() {
  PARSED_INTERFACE=""
  PARSED_PINNED_CPU=""
  PARSED_IRQBALANCE_RUNTIME_MASKED=""
  PARSED_IRQBALANCE_WAS_ACTIVE=""
  PARSED_IRQS=()
  PARSED_REQUESTED_AFFINITIES=()
  PARSED_EFFECTIVE_AFFINITIES=()
  local seen_schema="0"
  local seen_interface="0"
  local seen_pinned_cpu="0"
  local seen_runtime_mask="0"
  local seen_active="0"
  local kind first second third extra
  while read -r kind first second third extra; do
    if [[ -z "$kind" ]] || [[ -n "${extra:-}" ]]; then
      echo "Saved IRQ state contains an empty or overlong record." >&2
      return 1
    fi
    case "$kind" in
      schema)
        if [[ "$seen_schema" == "1" ]] || [[ "$first" != "$STATE_SCHEMA" ]] ||
           [[ -n "${second:-}" ]] || [[ -n "${third:-}" ]]; then
          echo "Saved IRQ state schema is invalid." >&2
          return 1
        fi
        seen_schema="1"
        ;;
      interface)
        if [[ "$seen_interface" == "1" ]] ||
           [[ ! "$first" =~ ^[A-Za-z0-9_.:-]+$ ]] ||
           [[ -n "${second:-}" ]] || [[ -n "${third:-}" ]]; then
          echo "Saved IRQ interface record is invalid." >&2
          return 1
        fi
        PARSED_INTERFACE="$first"
        seen_interface="1"
        ;;
      pinned_cpu)
        if [[ "$seen_pinned_cpu" == "1" ]] ||
           [[ ! "$first" =~ ^[0-9]+$ ]] ||
           [[ -n "${second:-}" ]] || [[ -n "${third:-}" ]]; then
          echo "Saved pinned CPU record is invalid." >&2
          return 1
        fi
        PARSED_PINNED_CPU="$first"
        seen_pinned_cpu="1"
        ;;
      irqbalance_runtime_masked)
        if [[ "$seen_runtime_mask" == "1" ]] ||
           [[ ! "$first" =~ ^[01]$ ]] ||
           [[ -n "${second:-}" ]] || [[ -n "${third:-}" ]]; then
          echo "Saved runtime-mask record is invalid." >&2
          return 1
        fi
        PARSED_IRQBALANCE_RUNTIME_MASKED="$first"
        seen_runtime_mask="1"
        ;;
      irqbalance_was_active)
        if [[ "$seen_active" == "1" ]] ||
           [[ ! "$first" =~ ^[01]$ ]] ||
           [[ -n "${second:-}" ]] || [[ -n "${third:-}" ]]; then
          echo "Saved irqbalance active-state record is invalid." >&2
          return 1
        fi
        PARSED_IRQBALANCE_WAS_ACTIVE="$first"
        seen_active="1"
        ;;
      irq)
        if [[ ! "$first" =~ ^[0-9]+$ ]] ||
           ! valid_cpu_list "${second:-}" ||
           ! valid_cpu_list "${third:-}"; then
          echo "Saved IRQ affinity record is invalid." >&2
          return 1
        fi
        local existing
        for existing in "${PARSED_IRQS[@]}"; do
          if [[ "$existing" == "$first" ]]; then
            echo "Saved IRQ state contains duplicate IRQ $first." >&2
            return 1
          fi
        done
        PARSED_IRQS+=("$first")
        PARSED_REQUESTED_AFFINITIES+=("$second")
        PARSED_EFFECTIVE_AFFINITIES+=("$third")
        ;;
      *)
        echo "Saved IRQ state contains an unknown record." >&2
        return 1
        ;;
    esac
  done < "$STATE_FILE"
  if [[ "$seen_schema" != "1" ]] || [[ "$seen_interface" != "1" ]] ||
     [[ "$seen_pinned_cpu" != "1" ]] || [[ "$seen_runtime_mask" != "1" ]] ||
     [[ "$seen_active" != "1" ]] || [[ "${#PARSED_IRQS[@]}" == "0" ]]; then
    echo "Saved IRQ state is incomplete." >&2
    return 1
  fi
}

validate_restore_targets() {
  local index irq
  for index in "${!PARSED_IRQS[@]}"; do
    irq="${PARSED_IRQS[$index]}"
    if [[ ! -w "$PROC_IRQ_ROOT/$irq/smp_affinity_list" ]] ||
       [[ ! -r "$PROC_IRQ_ROOT/$irq/effective_affinity_list" ]]; then
      echo "Saved IRQ $irq is no longer writable/readable." >&2
      return 1
    fi
    if [[ "${PARSED_REQUESTED_AFFINITIES[$index]}" != "${PARSED_EFFECTIVE_AFFINITIES[$index]}" ]]; then
      echo "Saved IRQ $irq did not have an exactly restorable requested/effective affinity." >&2
      return 1
    fi
  done
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl is required to restore irqbalance ownership." >&2
    return 1
  fi
}

status_command() {
  local interface="$1"
  RESOLVED_IRQS=()
  if ! load_irqs "$interface"; then
    return 1
  fi
  echo "interface=$interface irqbalance_pids=[$(irqbalance_processes | paste -sd, -)] runtime_masked=$(read_runtime_masked)"
  local irq requested effective
  for irq in "${RESOLVED_IRQS[@]}"; do
    requested="$(read_irq_affinity "$irq" smp)"
    effective="$(read_irq_affinity "$irq" effective)"
    echo "irq=$irq requested=$requested effective=$effective"
  done
  if [[ -e "$STATE_DIR" ]]; then
    if ! state_storage_is_valid || ! parse_state_file; then
      echo "guard_state=$STATE_FILE invalid=true"
      return 1
    fi
    echo "guard_state=$STATE_FILE schema=$STATE_SCHEMA interface=$PARSED_INTERFACE pinned_cpu=$PARSED_PINNED_CPU irqs=[$(printf '%s,' "${PARSED_IRQS[@]}" | sed 's/,$//')] original_runtime_masked=$PARSED_IRQBALANCE_RUNTIME_MASKED original_active=$PARSED_IRQBALANCE_WAS_ACTIVE"
  else
    echo "guard_state=none"
  fi
}

restore_internal() {
  if [[ ! -e "$STATE_DIR" ]]; then
    echo "No saved Franka NIC IRQ state exists at $STATE_FILE." >&2
    return 1
  fi
  # Parse and validate every record and every target before the first write.
  state_storage_is_valid || return 1
  parse_state_file || return 1
  validate_restore_targets || return 1

  local index irq expected_requested expected_effective
  for index in "${!PARSED_IRQS[@]}"; do
    irq="${PARSED_IRQS[$index]}"
    if ! write_irq_affinity "$irq" "${PARSED_REQUESTED_AFFINITIES[$index]}"; then
      echo "Could not restore requested affinity for IRQ $irq." >&2
      return 1
    fi
  done
  for index in "${!PARSED_IRQS[@]}"; do
    irq="${PARSED_IRQS[$index]}"
    expected_requested="${PARSED_REQUESTED_AFFINITIES[$index]}"
    expected_effective="${PARSED_EFFECTIVE_AFFINITIES[$index]}"
    local actual_requested actual_effective
    if ! actual_requested="$(read_irq_affinity "$irq" smp)" ||
       ! actual_effective="$(read_irq_affinity "$irq" effective)"; then
      return 1
    fi
    if [[ "$actual_requested" != "$expected_requested" ]] ||
       [[ "$actual_effective" != "$expected_effective" ]]; then
      echo "IRQ $irq restore verification failed: requested=$actual_requested effective=$actual_effective expected_requested=$expected_requested expected_effective=$expected_effective" >&2
      return 1
    fi
  done

  # Temporarily remove only the runtime mask, restore the active state, then
  # recreate an original runtime mask without --now so an unusual
  # active+masked original state is reproduced rather than silently changed.
  if ! systemctl unmask --runtime irqbalance.service; then
    echo "Could not remove the temporary irqbalance runtime mask." >&2
    return 1
  fi
  if [[ "$PARSED_IRQBALANCE_WAS_ACTIVE" == "1" ]]; then
    if ! systemctl start irqbalance.service; then
      echo "Could not restore active irqbalance state." >&2
      return 1
    fi
  else
    if ! systemctl stop irqbalance.service; then
      echo "Could not restore inactive irqbalance state." >&2
      return 1
    fi
  fi
  if [[ "$PARSED_IRQBALANCE_RUNTIME_MASKED" == "1" ]]; then
    if ! systemctl mask --runtime irqbalance.service; then
      echo "Could not restore the original irqbalance runtime mask." >&2
      return 1
    fi
  fi

  local restored_mask restored_active pids
  restored_mask="$(read_runtime_masked)"
  if ! restored_active="$(read_irqbalance_active)"; then
    return 1
  fi
  pids="$(irqbalance_processes | paste -sd, -)"
  if [[ "$restored_mask" != "$PARSED_IRQBALANCE_RUNTIME_MASKED" ]] ||
     [[ "$restored_active" != "$PARSED_IRQBALANCE_WAS_ACTIVE" ]] ||
     { [[ "$restored_active" == "1" ]] && [[ -z "$pids" ]]; } ||
     { [[ "$restored_active" == "0" ]] && [[ -n "$pids" ]]; }; then
    echo "irqbalance restore verification failed: runtime_masked=$restored_mask active=$restored_active pids=[$pids]" >&2
    return 1
  fi

  local saved_interface="$PARSED_INTERFACE"
  local saved_runtime_mask="$PARSED_IRQBALANCE_RUNTIME_MASKED"
  local saved_active="$PARSED_IRQBALANCE_WAS_ACTIVE"
  rm -f "$STATE_FILE" || return 1
  rmdir "$STATE_DIR" || return 1
  echo "Restored IRQ affinities for $saved_interface; irqbalance_runtime_masked=$saved_runtime_mask irqbalance_active=$saved_active"
}

apply_failure_trap() {
  local original_status="$?"
  trap - EXIT INT TERM
  if [[ "$APPLY_COMPLETED" != "1" ]]; then
    if [[ "$APPLY_SYSTEM_MUTATED" == "1" ]] && [[ -f "$STATE_FILE" ]]; then
      echo "Apply failed; restoring the saved IRQ/service state." >&2
      if ! restore_internal; then
        echo "CRITICAL: automatic restore failed; retain $STATE_FILE and run sudo $0 restore after inspection." >&2
      fi
    else
      rm -f "$STATE_FILE"
      if [[ -n "$APPLY_TEMP_FILE" ]]; then
        rm -f "$APPLY_TEMP_FILE"
      fi
      rmdir "$STATE_DIR" 2>/dev/null || true
    fi
  fi
  exit "$original_status"
}

apply_command() {
  local interface="$1"
  local cpu="$2"
  require_root
  if [[ ! "$cpu" =~ ^[0-9]+$ ]] ||
     [[ ! -d "$SYS_CPU_ROOT/cpu$cpu" ]] ||
     { [[ -f "$SYS_CPU_ROOT/cpu$cpu/online" ]] &&
       [[ "$(tr -d '[:space:]' < "$SYS_CPU_ROOT/cpu$cpu/online")" != "1" ]]; }; then
    echo "CPU $cpu is not a valid online logical CPU." >&2
    exit 2
  fi
  if [[ -e "$STATE_DIR" ]]; then
    echo "Existing state $STATE_DIR must be restored first." >&2
    exit 2
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl is required to own irqbalance during deployment." >&2
    exit 2
  fi
  RESOLVED_IRQS=()
  if ! load_irqs "$interface"; then
    exit 2
  fi

  local active runtime_masked pids
  if ! active="$(read_irqbalance_active)"; then
    exit 2
  fi
  runtime_masked="$(read_runtime_masked)"
  pids="$(irqbalance_processes | paste -sd, -)"
  if [[ "$active" == "0" ]] && [[ -n "$pids" ]]; then
    echo "irqbalance is running outside its active systemd service; refusing to kill an unowned process." >&2
    exit 2
  fi
  if [[ "$active" == "1" ]] && [[ -z "$pids" ]]; then
    echo "irqbalance.service is active but its process cannot be proven." >&2
    exit 2
  fi

  local requested_affinities=()
  local effective_affinities=()
  local irq requested effective requested_again
  for irq in "${RESOLVED_IRQS[@]}"; do
    requested="$(read_irq_affinity "$irq" smp)"
    effective="$(read_irq_affinity "$irq" effective)"
    requested_again="$(read_irq_affinity "$irq" smp)"
    if [[ "$requested" != "$requested_again" ]] ||
       [[ "$requested" != "$effective" ]]; then
      echo "IRQ $irq affinity changed during capture or requested/effective differ: requested=$requested requested_again=$requested_again effective=$effective" >&2
      exit 2
    fi
    if [[ ! -w "$PROC_IRQ_ROOT/$irq/smp_affinity_list" ]]; then
      echo "IRQ $irq affinity is not writable." >&2
      exit 2
    fi
    requested_affinities+=("$requested")
    effective_affinities+=("$effective")
  done

  install -d -o root -g root -m 0755 "$STATE_DIR"
  APPLY_TEMP_FILE="$STATE_DIR/state.tmp.$$"
  APPLY_COMPLETED="0"
  APPLY_SYSTEM_MUTATED="0"
  trap apply_failure_trap EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  {
    echo "schema $STATE_SCHEMA"
    echo "interface $interface"
    echo "pinned_cpu $cpu"
    echo "irqbalance_runtime_masked $runtime_masked"
    echo "irqbalance_was_active $active"
    local index
    for index in "${!RESOLVED_IRQS[@]}"; do
      echo "irq ${RESOLVED_IRQS[$index]} ${requested_affinities[$index]} ${effective_affinities[$index]}"
    done
  } > "$APPLY_TEMP_FILE"
  chown root:root "$APPLY_TEMP_FILE"
  chmod 0644 "$APPLY_TEMP_FILE"
  mv -f "$APPLY_TEMP_FILE" "$STATE_FILE"
  APPLY_TEMP_FILE=""
  if ! state_storage_is_valid || ! parse_state_file ||
     ! validate_restore_targets; then
    echo "The saved IRQ transaction state did not pass self-validation." >&2
    exit 2
  fi

  APPLY_SYSTEM_MUTATED="1"
  if ! systemctl mask --runtime --now irqbalance.service; then
    echo "Could not runtime-mask and stop $IRQBALANCE_UNIT." >&2
    exit 2
  fi
  local masked_after stopped_after
  masked_after="$(read_runtime_masked)"
  stopped_after="$(read_irqbalance_active)"
  pids="$(irqbalance_processes | paste -sd, -)"
  if [[ "$masked_after" != "1" ]] || [[ "$stopped_after" != "0" ]] ||
     [[ -n "$pids" ]]; then
    echo "Could not prove runtime-masked, stopped irqbalance ownership: runtime_masked=$masked_after active=$stopped_after pids=[$pids]" >&2
    exit 2
  fi

  for irq in "${RESOLVED_IRQS[@]}"; do
    if ! write_irq_affinity "$irq" "$cpu"; then
      echo "Could not pin IRQ $irq to CPU $cpu." >&2
      exit 2
    fi
  done
  for irq in "${RESOLVED_IRQS[@]}"; do
    if ! requested="$(read_irq_affinity "$irq" smp)" ||
       ! effective="$(read_irq_affinity "$irq" effective)"; then
      exit 2
    fi
    if [[ "$requested" != "$cpu" ]] || [[ "$effective" != "$cpu" ]]; then
      echo "IRQ $irq pin verification failed: requested=$requested effective=$effective" >&2
      exit 2
    fi
  done

  APPLY_COMPLETED="1"
  trap - EXIT INT TERM
  echo "PASS: $interface IRQs [$(printf '%s,' "${RESOLVED_IRQS[@]}" | sed 's/,$//')] pinned to CPU $cpu; irqbalance runtime-masked and stopped."
  echo "Auditable state: $STATE_FILE (schema=$STATE_SCHEMA, owner=root, mode=0644)"
  echo "Restore later with: sudo $0 restore"
}

main() {
  local command="${1:-}"
  shift || true
  case "$command" in
    apply)
      local interface=""
      local cpu=""
      while [[ "$#" -gt 0 ]]; do
        case "$1" in
          --interface)
            interface="${2:-}"
            shift 2
            ;;
          --cpu)
            cpu="${2:-}"
            shift 2
            ;;
          *)
            usage
            exit 2
            ;;
        esac
      done
      [[ -n "$interface" && -n "$cpu" ]] || { usage; exit 2; }
      apply_command "$interface" "$cpu"
      ;;
    restore)
      [[ "$#" == "0" ]] || { usage; exit 2; }
      require_root
      restore_internal
      ;;
    status)
      local interface=""
      if [[ "${1:-}" == "--interface" ]]; then
        interface="${2:-}"
        shift 2
      fi
      [[ -n "$interface" && "$#" == "0" ]] || { usage; exit 2; }
      status_command "$interface"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
