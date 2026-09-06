#!/usr/bin/env python3
"""Implementation of the operator-supervised V94 deployment command.

The stable public command is ``python -m sim2real.deploy``.  This historical
module name remains executable for compatibility.  Importing either module and
the default CLI path are hardware inert.  Real device factories are constructed
only after the explicit ``--execute`` and ``--yes-i-am-supervising`` flags.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import time
from typing import Callable, Iterator, Mapping, Optional, Sequence

from sim2real.policy import QD_G015_ACTION_CONTROLLER_CONTRACT_ID
from sim2real.policy.rate_mode import (
    ABSOLUTE_MAXIMUM_SUPERVISED_STEPS,
    DEFAULT_POLICY_RATE_MODE,
    resolve_policy_rate_mode,
)
from sim2real.policy.io_recorder import default_policy_io_path
from sim2real.rh56_profile_contract import (
    load_commissioned_rh56_force_set_g,
    load_v94_rh56_profile_command_bounds,
)
from sim2real.observation.camera_profile import (
    resolve_runtime_task_contract,
    task_profile_allows_robot_execution,
    task_profile_maximum_supervised_execute_steps,
    task_profile_policy_rate_hz,
    task_profile_policy_rgbd_resolution,
)
from sim2real.sim_control_alignment import (
    REAL_FRANKA_MAX_COMMAND_ACCELERATION_RAD_S2,
    REAL_FRANKA_MAX_COMMAND_JERK_RAD_S3,
    RH56_SPEED_SET_REGISTER_ORDER as RH56_SPEED_SET,
    sim_control_alignment_summary,
)
from sim2real.runtime.v94_policy_tick_source import (
    QD_G015_STARTUP_NON_ACTUATED_POLICY_STEPS,
)
from sim2real.observation.visualization import mask_video_path_for_recording

MAX_SUPERVISED_STEPS = ABSOLUTE_MAXIMUM_SUPERVISED_STEPS
POLICY_RATE_HZ = DEFAULT_POLICY_RATE_MODE.policy_rate_hz
FRANKA_MAX_SESSION_DURATION_S = 15.0
FRANKA_MAX_COMMAND_SPEED_RAD_S = 0.50
FRANKA_MAX_MEASURED_VELOCITY_RAD_S = 0.70
FRANKA_CONTACT_TORQUE_THRESHOLDS_NM = (
    20.0,
    20.0,
    18.0,
    18.0,
    16.0,
    14.0,
    12.0,
)
FRANKA_CONTACT_FORCE_THRESHOLDS_N = (
    20.0,
    20.0,
    20.0,
    25.0,
    25.0,
    25.0,
)
FRANKA_COLLISION_TORQUE_THRESHOLDS_NM = (
    40.0,
    40.0,
    36.0,
    36.0,
    32.0,
    28.0,
    24.0,
)
FRANKA_COLLISION_FORCE_THRESHOLDS_N = (
    40.0,
    40.0,
    40.0,
    50.0,
    50.0,
    50.0,
)
# The v205 exact-target replay has a measured maximum adjacent simulator
# setpoint delta of 0.018000365 rad.  This is a high-level target admission
# bound, not the physical FCI command velocity: the accepted V225/V226 native
# path interpolates continuously at 1 kHz and applies libfranka's official
# limiter as the final guard.  The v205
# success trace reaches about 0.405 rad/s, so the previous 0.20 rad/s ceiling
# could not reproduce it in real time.
FRANKA_MAX_TICK_TARGET_DELTA_RAD = 0.020
# Legacy V94 keeps a cumulative 1.21 rad home-centered episode cap even for a
# longer supervised run. q_d-g015 omits that relative radius and uses the
# compiled margin-contracted absolute joint intervals; the common tracking,
# velocity, acceleration, jerk, contact/reflex, and limiter guards remain.
FRANKA_MAX_EPISODE_DELTA_RAD = 1.21
FRANKA_MAX_START_ERROR_RAD = 0.01
FRANKA_MAX_TRACKING_ERROR_RAD = 0.01
FRANKA_FIRST_BOOTSTRAP_READ_TO_WRITE_S = 0.0008
FRANKA_STEADY_READ_TO_WRITE_S = 0.0008
# FCI is nominally 1 kHz but explicitly recovers from missing UDP packets.
# The native child accepts every returned state period inside FCI's documented
# 20-missing-packet recovery horizon.  The returned period also includes the
# recovered current packet, so the corresponding maximum is 21 ms; this does
# not increase uncontrolled continuation beyond 20 packets.
FRANKA_FCI_MAX_RECOVERABLE_CONTROL_PERIOD_S = 0.021
FRANKA_PARENT_HEARTBEAT_RECEIPT_TIMEOUT_S = 0.100
FRANKA_POLICY_TARGET_MAX_AGE_S = 0.050
# A missing observation never authorizes a new action: the native servo keeps
# converging to the last accepted bounded target.  Give camera/USB scheduling
# one bounded 500 ms recovery window while the independent 100 ms parent
# heartbeat still detects a dead controller and every newly produced TARGET
# remains freshness-limited to 50 ms.
FRANKA_POLICY_INTER_TARGET_TIMEOUT_S = 0.500
FRANKA_ARRIVAL_GATE_DEFAULT_TOLERANCE_RAD = 0.005
FRANKA_ARRIVAL_GATE_DEFAULT_TIMEOUT_S = 0.350
# STATE arrives best-effort every 16 nominal 1 kHz control cycles.  The 20 Hz
# checkpoint was trained with 0--50 ms proprioception latency; accept the
# latest sample through 40 ms and reserve 40--50 ms for a side-effect-free
# retry.  This avoids scheduler-phase retries while retaining the 50 ms hard
# liveness bound.
FRANKA_OBSERVATION_ACTION_MAX_AGE_S = 0.040
FRANKA_OBSERVATION_HARD_MAX_AGE_S = 0.050
D435_RUNTIME_FRAME_TIMEOUT_S = 0.100
D435_FORMAL_PUBLICATION_STALL_S = 0.400
# Camera-only ROI preflight must use the same publication-liveness contract as
# the formal owner.  The prior standalone 120 ms value rejected a valid fresh
# mask whenever startup SAM2/tracker work skipped three native 30 Hz frames
# (a repeatable ~133 ms gap), even though formal rollout intentionally holds
# the last action while the provider recovers for up to 400 ms.
OBJECT_ROI_PREFLIGHT_MAX_VALID_MASK_GAP_S = D435_FORMAL_PUBLICATION_STALL_S
OBJECT_POINTCLOUD_MAX_AGE_S = 0.200
MAXIMUM_CONSECUTIVE_NO_STAGE_HOLD_S = 0.400
RH56_FORCE_SET_G = 80
# Host-side feedback trip only; neither value writes the RH56 firmware
# CURRENT_LIMIT register.  Use the commissioned RH56 per-axis register ceiling
# here so a healthy motion/contact transient below the device limit is not
# rejected by a second, lower host-only threshold.  Firmware CURRENT_LIMIT,
# ERROR/STATUS and temperature feedback remain authoritative hard protections.
RH56_MAX_RUNNING_CURRENT_MA = 1400
RH56_STOP_SETTLE_MAX_AXIS_CURRENT_MA = 1000
RH56_NOMINAL_FEEDBACK_RATE_HZ = 20.0
RH56_FEEDBACK_HARD_AGE_S = 0.150
RH56_FEEDBACK_TO_COMMAND_OFFSET_UNITS = (0, 0, 0, 0, 0, 15)
RH56_DISABLED_OPEN_TOLERANCE_UNITS = (25, 25, 25, 25, 25, 26)
RH56_TRACKING_SIGNIFICANT_GAP_UNITS = 50
RH56_TRACKING_MIN_PROGRESS_UNITS = 3
RH56_TRACKING_CONTACT_FORCE_DELTA_G = 150
RH56_TRACKING_CONTACT_FORCE_ABSOLUTE_G = 200
RH56_TRACKING_TIMEOUT_S = 0.750
RH56_WRITE_RESPONSE_GRACE_S = 0.020
RH56_EXACT_READBACK_REQUEST_COUNT = 1
RH56_EXACT_READBACK_TIMEOUT_S = 0.050
RH56_COMMAND_MAX_AGE_S = 0.050
# The latest RH56 target is released at 20 Hz through the V94-transparent
# contract envelope above and is held without numeric rewrites while an
# observation is unavailable.  Match the arm's bounded sample-hold recovery
# window; command production freshness remains independently capped at 50 ms.
RH56_INTER_COMMAND_WATCHDOG_S = 0.500
RH56_STOP_TIMEOUT_S = 5.00
DEPLOYMENT_COMPUTE_THREADS = 1
DEFAULT_LIVE_VISUALIZATION_RATE_HZ = 10.0
OBJECT_ROI_PREFLIGHT_VALID_FRAMES = 3
OBJECT_ROI_PREFLIGHT_MAX_ATTEMPTS = 15
OBJECT_ROI_PREFLIGHT_MAX_INVALID_FRAMES = (
    OBJECT_ROI_PREFLIGHT_MAX_ATTEMPTS - OBJECT_ROI_PREFLIGHT_VALID_FRAMES
)
OBJECT_ROI_SELECTOR_PROTOCOL = "v94_isolated_object_roi_selection_v1"
OBJECT_ROI_SELECTOR_MAX_RESULT_BYTES = 4096
from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "dexgrasp/configs/fr3_rh56_v94_seq286_20hz_commissioned.json"
)
DEFAULT_PCD_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "perception/configs/d435_default.yaml"
)


class DeploymentAdmissionError(RuntimeError):
    """A pre-device admission, runtime, or verified-stop failure."""


class DeploymentExecutionError(RuntimeError):
    """Failure after a unique run acquired the real-hardware lease."""


# Historical exception names remain import-compatible.
SupervisedV94RunError = DeploymentAdmissionError
SupervisedV94ExecutionError = DeploymentExecutionError


def _validate_object_roi_preflight_progress(
    frame_ids: object,
    timestamps_s: object,
) -> "object":
    """Validate fresh forward progress using the formal camera time bound."""

    import numpy as np

    ids = np.asarray(frame_ids, dtype=np.int64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if ids.ndim != 1 or timestamps.shape != ids.shape or ids.size < 2:
        raise SupervisedV94RunError(
            "object ROI preflight returned malformed camera progress evidence"
        )
    frame_id_deltas = np.diff(ids)
    if not np.all(frame_id_deltas > 0):
        raise SupervisedV94RunError(
            "object ROI preflight camera frame_id did not strictly advance: "
            f"ids={ids.tolist()}"
        )
    timestamp_deltas = np.diff(timestamps)
    if (
        not np.all(np.isfinite(timestamps))
        or not np.all(timestamp_deltas > 0.0)
    ):
        raise SupervisedV94RunError(
            "object ROI preflight camera timestamps are not finite and "
            f"strictly increasing: values={timestamps.tolist()}"
        )
    if not np.all(
        timestamp_deltas <= OBJECT_ROI_PREFLIGHT_MAX_VALID_MASK_GAP_S
    ):
        raise SupervisedV94RunError(
            "object ROI preflight valid-mask publication gap exceeded "
            f"{OBJECT_ROI_PREFLIGHT_MAX_VALID_MASK_GAP_S * 1000:g}ms: "
            f"frame_ids={ids.tolist()} gaps_s={timestamp_deltas.tolist()}"
        )
    return timestamp_deltas


@dataclass(frozen=True)
class DeploymentRequest:
    """Validated command-line inputs for one bounded hardware deployment."""

    run_id: str
    steps: int
    bundle: Path
    profile: Path
    pcd_config: Path
    execute: bool
    policy_rate_hz: float = POLICY_RATE_HZ
    checkpoint: Optional[Path] = None
    # Set only by ``python -m sim2real.replay_actions``.  The normal deploy
    # CLI never exposes this override.
    replay_actions: Optional[Path] = None
    replay_actions_sha256: str = ""
    replay_action_count: int = 0
    replay_arrival_gated: bool = False
    replay_arrival_tolerance_rad: float = 0.015
    # Diagnostic closed-loop mode: do not infer the next policy action until
    # measured Franka q reaches the last committed policy target.  The native
    # 1 kHz servo continues holding/shaping that target during the wait.
    arrival_gated: bool = False
    arrival_tolerance_rad: float = FRANKA_ARRIVAL_GATE_DEFAULT_TOLERANCE_RAD
    arrival_timeout_s: float = FRANKA_ARRIVAL_GATE_DEFAULT_TIMEOUT_S
    live_visualization: bool = False
    live_visualization_rate_hz: float = DEFAULT_LIVE_VISUALIZATION_RATE_HZ
    record_video_path: Optional[Path] = None
    record_policy_io: bool = False
    object_roi_xywh: Optional[tuple[int, int, int, int]] = None
    select_object_roi: bool = False
    object_text: Optional[str] = None
    # ``guarded`` is the production alias for guarded_v2's unified recovery
    # evidence. guarded_v1 and legacy remain available for reproducible A/B.
    object_mask_mode: str = "guarded"
    # The task profile owns native D435/SAM2 dimensions. This selects the
    # aligned RGB-D/mask resolution used by the policy point-cloud projector.
    policy_rgbd_resolution: str = "848x480"
    interactive_roi_gui_used: bool = False
    object_roi_source: str = "pinned_fixed_roi"
    object_roi_preflight_sha256: str = ""
    object_roi_preflight_valid_frames: int = 0
    object_roi_preflight_invalid_frames: int = 0
    object_roi_preflight_min_policy_points: int = 0
    object_roi_preflight_depth_p50_m: Optional[float] = None
    object_roi_preflight_mask_bbox_xyxy: tuple[int, ...] = ()
    object_roi_preflight_mask_area_px: int = 0
    object_roi_preflight_mask_source: str = ""
    object_roi_preflight_bundle_sha256: str = ""
    object_roi_preflight_pcd_config_sha256: str = ""
    object_roi_preflight_checkpoint_sha256: str = ""
    # Filled from the pinned preparation before camera-only ROI preflight.
    selected_checkpoint_sha256: str = ""
    # Filled only after explicit execution admission, before hardware access.
    compute_threads: int = DEPLOYMENT_COMPUTE_THREADS
    parent_cpu_affinity: tuple[int, ...] = ()
    franka_servo_cpu: Optional[int] = None
    franka_servo_sibling_cpus: tuple[int, ...] = ()
    franka_servo_idle_sibling_cpus: tuple[int, ...] = ()
    franka_nic_interface: str = ""
    franka_nic_irq_numbers: tuple[int, ...] = ()
    franka_nic_irq_cpus: tuple[int, ...] = ()
    franka_nic_irq_requested_cpus: tuple[int, ...] = ()
    franka_nic_reserved_cpus: tuple[int, ...] = ()
    franka_nic_irq_stability_token: str = ""
    visualization_cpus: tuple[int, ...] = ()
    # Inert until the native launcher invokes it immediately before Popen.
    # It is deliberately excluded from every serialized request/audit.
    franka_native_prelaunch_validator: Optional[Callable[[], None]] = None
    # Internal, single-use live camera ownership transfer.  It is never
    # populated by the CLI or serialized into a request/audit fingerprint.
    prewarmed_camera_handoff: Optional[object] = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class _DeploymentCpuPartition:
    parent_cpus: tuple[int, ...]
    servo_cpu: int
    servo_sibling_cpus: tuple[int, ...]
    servo_idle_sibling_cpus: tuple[int, ...]
    visualization_cpus: tuple[int, ...]
    nic_interface: str
    nic_irq_numbers: tuple[int, ...]
    nic_irq_cpus: tuple[int, ...]
    nic_irq_requested_cpus: tuple[int, ...]
    nic_reserved_cpus: tuple[int, ...]
    nic_irq_stability_token: str


def _parse_linux_cpu_list(value: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for raw_item in str(value).strip().split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            lower_text, upper_text = item.split("-", 1)
            lower = int(lower_text)
            upper = int(upper_text)
            if lower < 0 or upper < lower:
                raise ValueError("invalid Linux CPU range")
            cpus.update(range(lower, upper + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError("invalid Linux CPU index")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("Linux CPU list is empty")
    return tuple(sorted(cpus))


def _resolve_ipv4_route_interface(
    robot_ip: str,
    *,
    route_table: Path = Path("/proc/net/route"),
) -> str:
    try:
        destination_ip = int(ipaddress.IPv4Address(str(robot_ip).strip()))
        lines = route_table.read_text().splitlines()
    except (OSError, ValueError) as exc:
        raise SupervisedV94RunError(
            f"could not inspect the Franka IPv4 route: {exc}"
        ) from exc
    candidates: list[tuple[int, int, str]] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            interface = fields[0]
            network = int.from_bytes(bytes.fromhex(fields[1]), "little")
            flags = int(fields[3], 16)
            metric = int(fields[6])
            mask = int.from_bytes(bytes.fromhex(fields[7]), "little")
        except (ValueError, OverflowError):
            continue
        if flags & 0x1 == 0 or destination_ip & mask != network & mask:
            continue
        candidates.append((bin(mask).count("1"), -metric, interface))
    if not candidates:
        raise SupervisedV94RunError(
            f"no active IPv4 route resolves the Franka address {robot_ip}"
        )
    return max(candidates)[2]


def _resolve_nic_irq_affinity(
    interface: str,
    *,
    sys_class_net: Path = Path("/sys/class/net"),
    proc_irq: Path = Path("/proc/irq"),
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    device = sys_class_net / interface / "device"
    msi_directory = device / "msi_irqs"
    irq_numbers: set[int] = set()
    try:
        if msi_directory.is_dir():
            irq_numbers.update(
                int(entry.name)
                for entry in msi_directory.iterdir()
                if entry.name.isdigit()
            )
        if not irq_numbers:
            legacy = int((device / "irq").read_text().strip())
            if legacy >= 0:
                irq_numbers.add(legacy)
    except (OSError, ValueError) as exc:
        raise SupervisedV94RunError(
            f"could not resolve IRQs for Franka interface {interface}: {exc}"
        ) from exc
    if not irq_numbers:
        raise SupervisedV94RunError(
            f"Franka interface {interface} exposes no auditable IRQ"
        )
    irq_cpus: set[int] = set()
    requested_irq_cpus: set[int] = set()
    try:
        for irq in sorted(irq_numbers):
            root = proc_irq / str(irq)
            affinity_path = root / "effective_affinity_list"
            if not affinity_path.is_file():
                affinity_path = root / "smp_affinity_list"
            effective = _parse_linux_cpu_list(affinity_path.read_text())
            requested = _parse_linux_cpu_list((root / "smp_affinity_list").read_text())
            if len(effective) != 1 or requested != effective:
                raise SupervisedV94RunError(
                    f"Franka NIC IRQ {irq} is not exclusively/stably pinned: "
                    f"requested={list(requested)} effective={list(effective)}; "
                    "run dexgrasp/scripts/configure_franka_nic_irq.sh"
                )
            irq_cpus.update(effective)
            requested_irq_cpus.update(requested)
    except (OSError, ValueError) as exc:
        raise SupervisedV94RunError(
            f"could not resolve IRQ CPU affinity for {interface}: {exc}"
        ) from exc
    return (
        tuple(sorted(irq_numbers)),
        tuple(sorted(irq_cpus)),
        tuple(sorted(requested_irq_cpus)),
    )


def _active_irqbalance_pids(
    *,
    proc_root: Path = Path("/proc"),
    pid_file: Path = Path("/run/irqbalance/irqbalance.pid"),
) -> tuple[int, ...]:
    candidates: set[int] = set()
    try:
        if pid_file.is_file():
            candidates.add(int(pid_file.read_text().strip()))
    except (OSError, ValueError):
        pass
    try:
        candidates.update(
            int(entry.name) for entry in proc_root.iterdir() if entry.name.isdigit()
        )
    except OSError as exc:
        raise SupervisedV94RunError(
            f"could not inspect irqbalance process state: {exc}"
        ) from exc
    active = []
    for pid in sorted(candidates):
        try:
            if (proc_root / str(pid) / "comm").read_text().strip() == "irqbalance":
                active.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            continue
    return tuple(active)


def _verify_franka_nic_irq_guard_marker(
    *,
    interface: str,
    irq_numbers: Sequence[int],
    pinned_cpu: int,
    state_path: Path = Path("/run/franka-nic-irq-guard/state"),
    runtime_mask_path: Path = Path("/run/systemd/system/irqbalance.service"),
    required_owner_uid: int = 0,
    required_owner_gid: int = 0,
) -> None:
    """Verify the root-owned helper proof without trusting process state alone."""

    try:
        directory_metadata = state_path.parent.lstat()
        metadata = state_path.lstat()
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != int(required_owner_uid)
            or directory_metadata.st_gid != int(required_owner_gid)
            or stat.S_IMODE(directory_metadata.st_mode) != 0o755
        ):
            raise SupervisedV94RunError(
                "Franka NIC IRQ guard directory must be root-owned mode 0755"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise SupervisedV94RunError(
                "Franka NIC IRQ guard state is not a regular file"
            )
        if (
            metadata.st_uid != int(required_owner_uid)
            or metadata.st_gid != int(required_owner_gid)
            or stat.S_IMODE(metadata.st_mode) != 0o644
        ):
            raise SupervisedV94RunError(
                "Franka NIC IRQ guard state must be root-owned mode 0644"
            )
        lines = state_path.read_text(encoding="utf-8").splitlines()
    except SupervisedV94RunError:
        raise
    except OSError as exc:
        raise SupervisedV94RunError(
            "Franka NIC IRQ guard state is missing/unreadable; run "
            "`sudo dexgrasp/scripts/configure_franka_nic_irq.sh apply "
            f"--interface {interface} --cpu {int(pinned_cpu)}`: {exc}"
        ) from exc

    scalar: dict[str, str] = {}
    marker_irqs: set[int] = set()
    allowed_scalars = {
        "schema",
        "interface",
        "pinned_cpu",
        "irqbalance_runtime_masked",
        "irqbalance_was_active",
    }
    for line_number, raw_line in enumerate(lines, start=1):
        fields = raw_line.split()
        if not fields:
            continue
        key = fields[0]
        if key == "irq":
            if len(fields) != 4:
                raise SupervisedV94RunError(
                    f"malformed IRQ guard state line {line_number}"
                )
            try:
                irq = int(fields[1])
            except ValueError as exc:
                raise SupervisedV94RunError(
                    f"malformed IRQ number on guard state line {line_number}"
                ) from exc
            if irq < 0 or irq in marker_irqs or not fields[2] or not fields[3]:
                raise SupervisedV94RunError(
                    f"invalid/duplicate IRQ guard state line {line_number}"
                )
            marker_irqs.add(irq)
            continue
        if key not in allowed_scalars or len(fields) != 2 or key in scalar:
            raise SupervisedV94RunError(
                f"unknown/duplicate IRQ guard state line {line_number}"
            )
        scalar[key] = fields[1]
    if set(scalar) != allowed_scalars:
        raise SupervisedV94RunError("Franka NIC IRQ guard state is incomplete")
    try:
        schema = int(scalar["schema"])
        marker_cpu = int(scalar["pinned_cpu"])
        prior_masked = int(scalar["irqbalance_runtime_masked"])
        prior_active = int(scalar["irqbalance_was_active"])
    except ValueError as exc:
        raise SupervisedV94RunError(
            "Franka NIC IRQ guard state has non-integer scalar fields"
        ) from exc
    if (
        schema != 2
        or scalar["interface"] != str(interface)
        or marker_cpu != int(pinned_cpu)
        or marker_irqs != {int(value) for value in irq_numbers}
        or prior_masked not in (0, 1)
        or prior_active not in (0, 1)
    ):
        raise SupervisedV94RunError(
            "Franka NIC IRQ guard state does not match current interface, "
            "IRQ set, and pinned CPU"
        )
    try:
        mask_metadata = runtime_mask_path.lstat()
        mask_target = os.readlink(runtime_mask_path)
    except OSError as exc:
        raise SupervisedV94RunError(
            "irqbalance is not runtime-masked by the Franka NIC IRQ guard"
        ) from exc
    if (
        not stat.S_ISLNK(mask_metadata.st_mode)
        or mask_metadata.st_uid != int(required_owner_uid)
        or mask_metadata.st_gid != int(required_owner_gid)
        or mask_target != "/dev/null"
    ):
        raise SupervisedV94RunError(
            "irqbalance runtime mask is not the root-owned /dev/null guard"
        )


def _verify_frozen_franka_nic_admission(
    request: DeploymentRequest,
    *,
    route_table: Path = Path("/proc/net/route"),
    sys_class_net: Path = Path("/sys/class/net"),
    proc_irq: Path = Path("/proc/irq"),
    proc_root: Path = Path("/proc"),
    irqbalance_pid_file: Path = Path("/run/irqbalance/irqbalance.pid"),
    guard_state_path: Path = Path("/run/franka-nic-irq-guard/state"),
    runtime_mask_path: Path = Path("/run/systemd/system/irqbalance.service"),
    guard_required_owner_uid: int = 0,
    guard_required_owner_gid: int = 0,
) -> None:
    """Re-read only the frozen route/IRQ token; never re-select a servo core."""

    expected_interface = str(request.franka_nic_interface)
    expected_irqs = tuple(int(value) for value in request.franka_nic_irq_numbers)
    expected_effective = tuple(int(value) for value in request.franka_nic_irq_cpus)
    expected_requested = tuple(
        int(value) for value in request.franka_nic_irq_requested_cpus
    )
    if (
        not expected_interface
        or not expected_irqs
        or len(expected_effective) != 1
        or expected_requested != expected_effective
        or not request.franka_nic_irq_stability_token
    ):
        raise SupervisedV94RunError("Franka NIC IRQ admission token is incomplete")
    current_interface = _resolve_ipv4_route_interface(
        _profile_franka_ip(request.profile), route_table=route_table
    )
    if current_interface != expected_interface:
        raise SupervisedV94RunError(
            "Franka route interface changed after host admission"
        )
    active = _active_irqbalance_pids(proc_root=proc_root, pid_file=irqbalance_pid_file)
    if active:
        raise SupervisedV94RunError(
            "irqbalance became active after host admission " f"(pids={list(active)})"
        )
    current_irqs, current_effective, current_requested = _resolve_nic_irq_affinity(
        expected_interface,
        sys_class_net=sys_class_net,
        proc_irq=proc_irq,
    )
    stability_payload = {
        "interface": current_interface,
        "irq_numbers": list(current_irqs),
        "effective_cpus": list(current_effective),
        "requested_cpus": list(current_requested),
    }
    current_token = hashlib.sha256(
        json.dumps(
            stability_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        current_irqs != expected_irqs
        or current_effective != expected_effective
        or current_requested != expected_requested
        or current_token != request.franka_nic_irq_stability_token
    ):
        raise SupervisedV94RunError(
            "Franka NIC route/IRQ affinity changed after host admission"
        )
    _verify_franka_nic_irq_guard_marker(
        interface=expected_interface,
        irq_numbers=expected_irqs,
        pinned_cpu=expected_effective[0],
        state_path=guard_state_path,
        runtime_mask_path=runtime_mask_path,
        required_owner_uid=guard_required_owner_uid,
        required_owner_gid=guard_required_owner_gid,
    )


def _unrelated_irq_load_by_cpu(
    *,
    proc_irq: Path,
    proc_interrupts: Path,
    excluded_irqs: Sequence[int],
) -> tuple[Mapping[int, int], tuple[int, ...]]:
    excluded = set(int(value) for value in excluded_irqs)
    load: dict[int, int] = {}
    try:
        lines = proc_interrupts.read_text().splitlines()
    except OSError as exc:
        raise SupervisedV94RunError(
            f"could not inspect /proc/interrupts for servo-core selection: {exc}"
        ) from exc
    if not lines:
        raise SupervisedV94RunError("/proc/interrupts is empty")
    cpu_columns = []
    for token in lines[0].split():
        if token.startswith("CPU") and token[3:].isdigit():
            cpu_columns.append(int(token[3:]))
    if not cpu_columns:
        raise SupervisedV94RunError("/proc/interrupts contains no CPU columns")
    disruptive_cpus: set[int] = set()
    for line in lines[1:]:
        if ":" not in line:
            continue
        label, remainder = line.split(":", 1)
        irq_text = label.strip()
        if not irq_text.isdigit():
            continue
        irq = int(irq_text)
        fields = remainder.split()
        if len(fields) < len(cpu_columns):
            continue
        try:
            counts = [int(value) for value in fields[: len(cpu_columns)]]
        except ValueError:
            continue
        if irq not in excluded:
            for cpu, count in zip(cpu_columns, counts):
                load[cpu] = load.get(cpu, 0) + count
        description = " ".join(fields[len(cpu_columns) :]).lower()
        if irq not in excluded and ("nvidia" in description or "xhci" in description):
            affinity_path = proc_irq / str(irq) / "effective_affinity_list"
            try:
                disruptive_cpus.update(_parse_linux_cpu_list(affinity_path.read_text()))
            except (OSError, ValueError):
                # An unresolvable GPU/xHCI IRQ cannot be proven separate.
                disruptive_cpus.update(cpu_columns)
    return load, tuple(sorted(disruptive_cpus))


def _profile_franka_ip(profile: Path) -> str:
    try:
        payload = json.loads(profile.read_text())
        address = str(payload["franka"]["ip"]).strip()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SupervisedV94RunError(
            f"could not resolve Franka IP from commissioning profile: {exc}"
        ) from exc
    if not address:
        raise SupervisedV94RunError("commissioning profile contains an empty Franka IP")
    return address


def _deployment_cpu_partition(
    *,
    robot_ip: str,
    reserve_visualization: bool = False,
    route_table: Path = Path("/proc/net/route"),
    sys_class_net: Path = Path("/sys/class/net"),
    proc_irq: Path = Path("/proc/irq"),
    proc_interrupts: Path = Path("/proc/interrupts"),
    proc_root: Path = Path("/proc"),
    irqbalance_pid_file: Path = Path("/run/irqbalance/irqbalance.pid"),
    cpu_topology: Path = Path("/sys/devices/system/cpu"),
) -> _DeploymentCpuPartition:
    """Isolate policy/viewers from both the FCI IRQ core and native servo."""

    allowed = set(os.sched_getaffinity(0))
    if len(allowed) < 3:
        raise SupervisedV94RunError(
            "supervised deployment requires at least three allowed logical CPUs"
        )
    groups: dict[tuple[int, int], set[int]] = {}
    try:
        for cpu in sorted(allowed):
            root = cpu_topology / f"cpu{cpu}" / "topology"
            package = int((root / "physical_package_id").read_text().strip())
            core = int((root / "core_id").read_text().strip())
            groups.setdefault((package, core), set()).add(cpu)
    except (OSError, ValueError) as exc:
        raise SupervisedV94RunError(
            f"could not resolve CPU topology for Franka isolation: {exc}"
        ) from exc
    if not groups:
        raise SupervisedV94RunError("no allowed CPU core is available")
    interface = _resolve_ipv4_route_interface(robot_ip, route_table=route_table)
    active_irqbalance = _active_irqbalance_pids(
        proc_root=proc_root, pid_file=irqbalance_pid_file
    )
    if active_irqbalance:
        raise SupervisedV94RunError(
            "irqbalance is active and can move the Franka NIC IRQ during "
            "control; run `sudo dexgrasp/scripts/configure_franka_nic_irq.sh "
            f"apply --interface {interface} --cpu <DEDICATED_CPU>` first "
            f"(pids={list(active_irqbalance)})"
        )
    irq_numbers, irq_cpus, requested_irq_cpus = _resolve_nic_irq_affinity(
        interface, sys_class_net=sys_class_net, proc_irq=proc_irq
    )
    cpu_to_group = {cpu: group for group in groups.values() for cpu in group}
    nic_reserved: set[int] = set()
    for cpu in irq_cpus:
        nic_reserved.update(cpu_to_group.get(cpu, set()))
    nic_groups = {
        tuple(sorted(cpu_to_group[cpu])) for cpu in irq_cpus if cpu in cpu_to_group
    }
    if len(nic_groups) != 1:
        raise SupervisedV94RunError(
            "Franka NIC IRQs are not pinned to one dedicated physical core; "
            "run dexgrasp/scripts/configure_franka_nic_irq.sh"
        )
    irq_load, disruptive_irq_cpus = _unrelated_irq_load_by_cpu(
        proc_irq=proc_irq,
        proc_interrupts=proc_interrupts,
        excluded_irqs=irq_numbers,
    )
    disruptive_reserved: set[int] = set()
    for cpu in disruptive_irq_cpus:
        disruptive_reserved.update(cpu_to_group.get(cpu, set()))
    candidates = sorted(
        (
            group
            for group in groups.values()
            if group.isdisjoint(nic_reserved) and group.isdisjoint(disruptive_reserved)
            # CPU0 carries scheduler/RCU/local-timer housekeeping that is not
            # represented by numeric device IRQ rows in /proc/interrupts.
            and 0 not in group
        ),
        key=lambda values: (
            sum(irq_load.get(cpu, 0) for cpu in values),
            max(values),
        ),
    )
    if not candidates:
        raise SupervisedV94RunError(
            "Franka NIC/GPU/xHCI IRQ affinity leaves no separate servo core"
        )
    reserved = candidates[0]
    visualization = set()
    if reserve_visualization:
        if len(candidates) < 2:
            raise SupervisedV94RunError(
                "live visualization requires a core separate from the "
                "Franka NIC IRQ and servo"
            )
        visualization = candidates[1]
    parent = allowed - nic_reserved - reserved - visualization
    if len(parent) < 2:
        raise SupervisedV94RunError(
            "reserving a Franka servo core would leave fewer than two parent CPUs"
        )
    servo_cpu = min(
        reserved,
        key=lambda cpu: (irq_load.get(cpu, 0), cpu),
    )
    stability_payload = {
        "interface": interface,
        "irq_numbers": list(irq_numbers),
        "effective_cpus": list(irq_cpus),
        "requested_cpus": list(requested_irq_cpus),
    }
    stability_token = hashlib.sha256(
        json.dumps(
            stability_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return _DeploymentCpuPartition(
        parent_cpus=tuple(sorted(parent)),
        servo_cpu=servo_cpu,
        servo_sibling_cpus=tuple(sorted(reserved)),
        servo_idle_sibling_cpus=tuple(sorted(reserved - {servo_cpu})),
        visualization_cpus=tuple(sorted(visualization)),
        nic_interface=interface,
        nic_irq_numbers=irq_numbers,
        nic_irq_cpus=irq_cpus,
        nic_irq_requested_cpus=requested_irq_cpus,
        nic_reserved_cpus=tuple(sorted(nic_reserved)),
        nic_irq_stability_token=stability_token,
    )


def _admit_deployment_compute_isolation(
    request: DeploymentRequest,
) -> DeploymentRequest:
    """Read-only host admission performed before any robot lease or reset."""

    partition = _deployment_cpu_partition(
        robot_ip=_profile_franka_ip(request.profile),
        reserve_visualization=bool(
            request.live_visualization or request.record_video_path is not None
        ),
    )
    admitted = replace(
        request,
        parent_cpu_affinity=partition.parent_cpus,
        franka_servo_cpu=partition.servo_cpu,
        franka_servo_sibling_cpus=partition.servo_sibling_cpus,
        franka_servo_idle_sibling_cpus=(partition.servo_idle_sibling_cpus),
        franka_nic_interface=partition.nic_interface,
        franka_nic_irq_numbers=partition.nic_irq_numbers,
        franka_nic_irq_cpus=partition.nic_irq_cpus,
        franka_nic_irq_requested_cpus=(partition.nic_irq_requested_cpus),
        franka_nic_reserved_cpus=partition.nic_reserved_cpus,
        franka_nic_irq_stability_token=(partition.nic_irq_stability_token),
        visualization_cpus=partition.visualization_cpus,
    )
    _verify_frozen_franka_nic_admission(admitted)
    return admitted


@contextmanager
def _deployment_compute_isolation(
    request: DeploymentRequest,
) -> Iterator[DeploymentRequest]:
    """Bound compute pools and keep policy work off the Franka servo core."""

    partition = _deployment_cpu_partition(
        robot_ip=_profile_franka_ip(request.profile),
        reserve_visualization=bool(
            request.live_visualization or request.record_video_path is not None
        ),
    )
    if (
        not request.franka_nic_irq_stability_token
        or request.franka_nic_irq_stability_token != partition.nic_irq_stability_token
        or request.franka_servo_cpu != partition.servo_cpu
        or request.franka_servo_sibling_cpus != partition.servo_sibling_cpus
    ):
        raise SupervisedV94RunError(
            "Franka NIC IRQ/CPU isolation changed after preflight and before "
            "native control; no policy motion was started"
        )
    _verify_frozen_franka_nic_admission(request)
    parent_cpus = partition.parent_cpus
    servo_cpu = partition.servo_cpu
    servo_siblings = partition.servo_sibling_cpus
    visualization_cpus = partition.visualization_cpus
    previous_affinity = set(os.sched_getaffinity(0))
    enforced_environment = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OPENCV_FOR_THREADS_NUM": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
        "GOMP_SPINCOUNT": "0",
    }
    previous_environment = {name: os.environ.get(name) for name in enforced_environment}
    previous_thread_affinities: dict[int, set[int]] = {}
    limiter = None
    cv2 = None
    previous_opencv_threads: Optional[int] = None
    try:
        for name, value in enforced_environment.items():
            os.environ[name] = value

        # Importing the sim2real package already loads NumPy, so environment
        # variables alone cannot change its live OpenBLAS pool.
        from threadpoolctl import threadpool_info, threadpool_limits

        limiter = threadpool_limits(limits=DEPLOYMENT_COMPUTE_THREADS)
        limiter.__enter__()
        import cv2 as imported_cv2

        cv2 = imported_cv2
        previous_opencv_threads = int(cv2.getNumThreads())
        cv2.setNumThreads(DEPLOYMENT_COMPUTE_THREADS)
        active_pools = [
            value
            for value in threadpool_info()
            if value.get("user_api") in {"blas", "openmp"}
        ]
        if not active_pools or any(
            int(value.get("num_threads", -1)) != DEPLOYMENT_COMPUTE_THREADS
            for value in active_pools
        ):
            raise SupervisedV94RunError(
                "could not verify single-thread BLAS/OpenMP deployment pools"
            )
        if int(cv2.getNumThreads()) != DEPLOYMENT_COMPUTE_THREADS:
            raise SupervisedV94RunError(
                "could not verify single-thread OpenCV deployment"
            )

        # NumPy may have created OpenBLAS workers while sim2real.__init__ was
        # imported.  Affinity is per Linux TID, not process-wide, so moving
        # only the Python main thread would leave those workers on the servo
        # core.  Snapshot, constrain, and verify every current task.  All
        # later camera/runtime threads inherit this parent mask.
        for _pass in range(2):
            for entry in Path("/proc/self/task").iterdir():
                try:
                    tid = int(entry.name)
                    if tid not in previous_thread_affinities:
                        previous_thread_affinities[tid] = set(os.sched_getaffinity(tid))
                    os.sched_setaffinity(tid, set(parent_cpus))
                except (FileNotFoundError, ProcessLookupError):
                    continue
        for entry in Path("/proc/self/task").iterdir():
            try:
                tid = int(entry.name)
                if set(os.sched_getaffinity(tid)) != set(parent_cpus):
                    raise SupervisedV94RunError(
                        f"CPU isolation verification failed for TID {tid}"
                    )
            except (FileNotFoundError, ProcessLookupError):
                continue
        isolated_request = replace(
            request,
            compute_threads=DEPLOYMENT_COMPUTE_THREADS,
            parent_cpu_affinity=parent_cpus,
            franka_servo_cpu=servo_cpu,
            franka_servo_sibling_cpus=servo_siblings,
            franka_servo_idle_sibling_cpus=(partition.servo_idle_sibling_cpus),
            franka_nic_interface=partition.nic_interface,
            franka_nic_irq_numbers=partition.nic_irq_numbers,
            franka_nic_irq_cpus=partition.nic_irq_cpus,
            franka_nic_irq_requested_cpus=(partition.nic_irq_requested_cpus),
            franka_nic_reserved_cpus=partition.nic_reserved_cpus,
            franka_nic_irq_stability_token=(partition.nic_irq_stability_token),
            visualization_cpus=visualization_cpus,
        )
        frozen_request = isolated_request

        def validate_immediately_before_native_popen() -> None:
            _verify_frozen_franka_nic_admission(frozen_request)

        isolated_request = replace(
            isolated_request,
            franka_native_prelaunch_validator=(
                validate_immediately_before_native_popen
            ),
        )
        print(
            "[Compute isolation] "
            f"policy_cpus={list(parent_cpus)} compute_threads=1 "
            f"franka_servo_cpu={servo_cpu} "
            f"servo_idle_siblings={list(partition.servo_idle_sibling_cpus)} "
            f"nic={partition.nic_interface} "
            f"nic_irqs={list(partition.nic_irq_numbers)} "
            f"nic_irq_cpus={list(partition.nic_irq_cpus)} "
            f"nic_requested_cpus="
            f"{list(partition.nic_irq_requested_cpus)} "
            f"nic_reserved_core={list(partition.nic_reserved_cpus)} "
            f"nic_stability={partition.nic_irq_stability_token[:12]} "
            f"visualization_cpus={list(visualization_cpus)}",
            flush=True,
        )
        yield isolated_request
    finally:
        if cv2 is not None and previous_opencv_threads is not None:
            try:
                cv2.setNumThreads(previous_opencv_threads)
            except BaseException:
                pass
        if limiter is not None:
            try:
                limiter.__exit__(None, None, None)
            except BaseException:
                pass
        for tid, affinity in previous_thread_affinities.items():
            try:
                os.sched_setaffinity(tid, affinity)
            except (FileNotFoundError, ProcessLookupError):
                pass
            except BaseException:
                pass
        try:
            os.sched_setaffinity(0, previous_affinity)
        except BaseException:
            pass
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _parse_steps(value: object, *, maximum: int = MAX_SUPERVISED_STEPS) -> int:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("maximum step count must be a positive integer")
    if isinstance(value, bool):
        raise ValueError(f"steps must be an integer in 1..{maximum}")
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"steps must be an integer in 1..{maximum}") from exc
    if str(result) != str(value).strip() or not 1 <= result <= maximum:
        raise ValueError(f"steps must be an integer in 1..{maximum}")
    return result


def _parse_visualization_rate(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("live visualization rate must be in 1..15 Hz")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("live visualization rate must be in 1..15 Hz") from exc
    if not math.isfinite(result) or not 1.0 <= result <= 15.0:
        raise ValueError("live visualization rate must be in 1..15 Hz")
    return result


def _parse_object_roi(
    value: Optional[Sequence[object]],
) -> Optional[tuple[int, int, int, int]]:
    if value is None:
        return None
    raw = tuple(value)
    if len(raw) != 4:
        raise ValueError("object ROI must contain X Y W H")
    parsed = []
    for index, item in enumerate(raw):
        if isinstance(item, bool):
            raise ValueError(f"object ROI[{index}] must be an integer")
        try:
            number = int(str(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"object ROI[{index}] must be an integer") from exc
        if str(number) != str(item).strip():
            raise ValueError(f"object ROI[{index}] must be an integer")
        parsed.append(number)
    x, y, width, height = parsed
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("object ROI requires non-negative X/Y and positive W/H")
    return x, y, width, height


def _parse_object_text(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    if not text:
        raise ValueError("--object-text must be non-empty")
    if len(text) > 128:
        raise ValueError("--object-text must contain at most 128 characters")
    if any(ord(character) < 32 for character in text):
        raise ValueError("--object-text cannot contain control characters")
    return text


def _isolated_roi_selector_environment(
    source: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Build a GUI-only child environment without ROS/Isaac Qt pollution.

    The deployment process may have inherited ROS 2 Python 3.10 paths and
    IsaacGym's bundled Qt libraries even though V94 runs from a Python 3.9
    virtual environment.  Importing cv2 in that mixed process can abort inside
    Qt before Python gets a chance to report an exception.  The ROI selector
    therefore gets a deliberately narrow Python path and a filtered loader
    path while retaining DISPLAY/XAUTHORITY/WAYLAND_DISPLAY.
    """

    environment = dict(os.environ if source is None else source)
    workspace = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(workspace),
            str(workspace / "perception"),
        )
    )
    environment["PYTHONNOUSERSITE"] = "1"

    blocked_markers = (
        "/opt/ros",
        "isaacgym",
        "isaac_sim",
        "isaac-sim",
        "isaacsim",
        "/omni/",
    )
    retained_loader_paths = []
    for entry in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        candidate = entry.strip()
        lowered = candidate.lower()
        if candidate and not any(marker in lowered for marker in blocked_markers):
            retained_loader_paths.append(candidate)
    if retained_loader_paths:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(retained_loader_paths)
    else:
        environment.pop("LD_LIBRARY_PATH", None)

    # These variables may point Qt at ROS/Isaac plugins even after the loader
    # path is cleaned.  OpenCV's wheel can then resolve its own xcb plugin.
    for name in tuple(environment):
        upper = name.upper()
        if (
            upper.startswith("QT_")
            or upper.startswith("ROS_")
            or upper.startswith("AMENT_")
            or upper.startswith("COLCON_")
            or upper.startswith("ISAAC")
            or upper.startswith("CARB_")
            or upper.startswith("OMNI_")
            or upper
            in {
                "CMAKE_PREFIX_PATH",
                "QML2_IMPORT_PATH",
                "ROS_PACKAGE_PATH",
                "PYTHONHOME",
                "LD_PRELOAD",
            }
        ):
            environment.pop(name, None)
    return environment


def _decode_isolated_roi_selector_result(
    payload: bytes,
    *,
    expected_nonce: str,
    expected_camera_serial: str,
    expected_calibration_id: str,
) -> tuple[int, int, int, int]:
    """Validate the dedicated child pipe; stdout/stderr are never trusted."""

    if not payload:
        raise SupervisedV94RunError("isolated object ROI selector returned no result")
    if len(payload) > OBJECT_ROI_SELECTOR_MAX_RESULT_BYTES:
        raise SupervisedV94RunError(
            "isolated object ROI selector result exceeded its fixed bound"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SupervisedV94RunError(
            "isolated object ROI selector returned non-UTF-8 data"
        ) from exc
    if not text.endswith("\n") or text.count("\n") != 1:
        raise SupervisedV94RunError(
            "isolated object ROI selector returned a malformed record"
        )
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SupervisedV94RunError(
            "isolated object ROI selector returned malformed JSON"
        ) from exc
    if not isinstance(record, Mapping):
        raise SupervisedV94RunError(
            "isolated object ROI selector result is not an object"
        )
    if (
        record.get("protocol") != OBJECT_ROI_SELECTOR_PROTOCOL
        or record.get("nonce") != expected_nonce
    ):
        raise SupervisedV94RunError(
            "isolated object ROI selector protocol/nonce mismatch"
        )
    status = record.get("status")
    if status == "error":
        detail = str(record.get("error", "")).strip()
        raise SupervisedV94RunError(
            "isolated object ROI selector failed without opening robot "
            "interfaces: "
            + (detail or "unspecified child error")
        )
    if status != "ok":
        raise SupervisedV94RunError(
            "isolated object ROI selector returned an unknown status"
        )
    if (
        str(record.get("camera_serial", "")) != expected_camera_serial
        or str(record.get("calibration_id", "")) != expected_calibration_id
    ):
        raise SupervisedV94RunError(
            "isolated object ROI selector camera/calibration differs from V94"
        )
    try:
        roi = _parse_object_roi(record.get("roi_xywh"))
    except ValueError as exc:
        raise SupervisedV94RunError(
            "isolated object ROI selector returned invalid numeric XYWH"
        ) from exc
    if roi is None:
        raise SupervisedV94RunError(
            "isolated object ROI selector returned no numeric XYWH"
        )
    return roi


def _read_bounded_pipe(descriptor: int, *, maximum_bytes: int) -> bytes:
    chunks = []
    total = 0
    while True:
        block = os.read(descriptor, min(1024, maximum_bytes + 1 - total))
        if not block:
            break
        chunks.append(block)
        total += len(block)
        if total > maximum_bytes:
            raise SupervisedV94RunError(
                "isolated object ROI selector result exceeded its fixed bound"
            )
    return b"".join(chunks)


def _terminate_selector_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _select_object_roi_isolated(
    request: DeploymentRequest,
    *,
    expected_camera_serial: str,
    expected_calibration_id: str,
) -> tuple[int, int, int, int]:
    """Run OpenCV selectROI in a clean child and accept only its private pipe."""

    nonce = secrets.token_hex(16)
    read_fd, write_fd = os.pipe()
    process: Optional[subprocess.Popen[bytes]] = None
    workspace = Path(__file__).resolve().parents[2]
    command = (
        sys.executable,
        "-m",
        "sim2real.observation.roi_selector",
        "--bundle",
        str(request.bundle),
        "--pcd-config",
        str(request.pcd_config),
        "--nonce",
        nonce,
        "--result-fd",
        str(write_fd),
    )
    if request.object_text is not None:
        command += (
            "--object-text",
            request.object_text,
            "--compact-console",
        )
    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(workspace),
                env=_isolated_roi_selector_environment(),
                stdin=subprocess.DEVNULL,
                # Qt/OpenCV may emit hundreds of non-fatal plugin/font
                # diagnostics even when the window works.  Structured Python
                # failures travel over the private result pipe, while a native
                # crash is reported from the child's return code.
                # Text mode has no Qt GUI diagnostics to suppress. Preserve
                # its detector-service traceback so a dependency/model startup
                # failure is visible instead of collapsing to only `code 1`.
                stderr=(
                    None
                    if request.object_text is not None
                    else subprocess.DEVNULL
                ),
                close_fds=True,
                pass_fds=(write_fd,),
            )
        except OSError as exc:
            raise SupervisedV94RunError(
                "could not start isolated object ROI selector without "
                f"opening robot interfaces: {exc}"
            ) from exc
        finally:
            os.close(write_fd)
            write_fd = -1

        try:
            payload = _read_bounded_pipe(
                read_fd,
                maximum_bytes=OBJECT_ROI_SELECTOR_MAX_RESULT_BYTES,
            )
            returncode = process.wait()
        except BaseException:
            _terminate_selector_child(process)
            raise
        if returncode != 0:
            if payload:
                # A normal Python error is carried over the dedicated pipe.
                _decode_isolated_roi_selector_result(
                    payload,
                    expected_nonce=nonce,
                    expected_camera_serial=expected_camera_serial,
                    expected_calibration_id=expected_calibration_id,
                )
                raise SupervisedV94RunError(
                    "isolated object ROI selector returned success data with "
                    f"nonzero status {returncode}"
                )
            if returncode < 0:
                try:
                    reason = signal.Signals(-returncode).name
                except ValueError:
                    reason = f"signal {-returncode}"
                detail = f"crashed with {reason}"
            else:
                detail = f"exited with status {returncode}"
            raise SupervisedV94RunError(
                "isolated object ROI selector "
                f"{detail}; the selector opened no robot interfaces"
            )
        return _decode_isolated_roi_selector_result(
            payload,
            expected_nonce=nonce,
            expected_camera_serial=expected_camera_serial,
            expected_calibration_id=expected_calibration_id,
        )
    finally:
        if process is not None:
            _terminate_selector_child(process)
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run at most 720 operator-supervised V94 observation->policy->real "
            "Franka/RH56 transactions (experimental; not commissioned C2)."
        )
    )
    parser.add_argument("--steps", default="1")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--policy-rate-hz",
        default=str(int(POLICY_RATE_HZ)),
        metavar="{20,60}",
        help=(
            "policy/control rate mode; must match the selected checkpoint's "
            "control_dt metadata (default: 60)"
        ),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help=(
            "complete verified V94 deployment ZIP used for the observation, "
            "action, reset, scene and calibration contract; its manifest "
            "primary checkpoint is used unless --checkpoint is supplied"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "optional external student/PPO .pt checkpoint; the verified "
            "--bundle remains the observation/action/reset/calibration "
            "contract and the override is compatibility-checked offline "
            "before any robot access"
        ),
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--pcd-config", type=Path, default=DEFAULT_PCD_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--verbose-console",
        action="store_true",
        help=(
            "show the full legacy diagnostic stream; by default real "
            "execution prints only colored operator milestones and failures"
        ),
    )
    parser.add_argument(
        "--arrival-gated",
        action="store_true",
        help=(
            "diagnostic mode: after each dual-ACK command, hold its target "
            "and call the checkpoint again only after every measured Franka "
            "joint has entered --arrival-tolerance-rad or crossed its target"
        ),
    )
    parser.add_argument(
        "--arrival-tolerance-rad",
        default=str(FRANKA_ARRIVAL_GATE_DEFAULT_TOLERANCE_RAD),
        help=(
            "per-axis Franka target-arrival tolerance in 0.001..0.030 rad "
            "(default: 0.005)"
        ),
    )
    parser.add_argument(
        "--arrival-timeout-s",
        default=str(FRANKA_ARRIVAL_GATE_DEFAULT_TIMEOUT_S),
        help=(
            "maximum wait for one closed-loop Franka target in 0.10..0.40 s; "
            "a timeout stops instead of waiting forever (default: 0.35)"
        ),
    )
    object_roi = parser.add_mutually_exclusive_group()
    object_roi.add_argument(
        "--object-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help=(
            "use this current object box after a camera-only preflight; "
            "coordinates are native task-camera XYWH"
        ),
    )
    parser.add_argument(
        "--object-mask-mode",
        choices=("guarded", "guarded_v2", "guarded_v1", "legacy"),
        default="guarded",
        help=(
            "final mask path: guarded/guarded_v2 use unified exact-RGBD "
            "recovery; guarded_v1 preserves former double confirmation; "
            "legacy publishes semantic SAM2 directly and explicitly enables "
            "retained stale_palm policy-input compatibility for A/B"
        ),
    )
    parser.add_argument(
        "--policy-rgbd-resolution",
        choices=("848x480", "424x240"),
        default="848x480",
        help=(
            "aligned RGB-D/mask resolution used to build the 128 policy "
            "points; a resolved task profile owns native capture dimensions "
            "and may require 424x240"
        ),
    )
    object_roi.add_argument(
        "--select-object-roi",
        action="store_true",
        help=(
            "after automatic reset and the stationary dwell, open an isolated "
            "camera-only window to select the current object; drag a box with "
            "visible background margin"
        ),
    )
    object_roi.add_argument(
        "--object-text",
        metavar="TEXT",
        help=(
            "replace manual ROI selection with camera-only automatic text "
            "grounding (for example: --object-text 'ball'); while absent, "
            "the D435 keeps searching and Ctrl+C cancels"
        ),
    )
    parser.add_argument(
        "--live-visualization",
        action="store_true",
        help=(
            "show the exact-frame final policy object mask and 128-point "
            "cloud in an isolated, lossy GUI process, then save the last "
            "exact policy observation under dexgrasp/runs/<run-id>_observation_visualization"
        ),
    )
    parser.add_argument(
        "--live-visualization-rate-hz",
        default=str(DEFAULT_LIVE_VISUALIZATION_RATE_HZ),
        help="viewer refresh rate in 1..15 Hz (default: 10)",
    )
    parser.add_argument(
        "--record-video",
        type=Path,
        default=None,
        metavar="OUTPUT.mp4",
        help=(
            "save the test camera stream with the exact final policy binary "
            "object mask inset at the top right; no live window is opened "
            "unless --live-visualization is also supplied"
        ),
    )
    parser.add_argument(
        "--record-policy-io",
        action="store_true",
        help=(
            "record every accepted policy tick's exact raw and normalized "
            "model input in RAM, then atomically save "
            "dexgrasp/runs/<run-id>_policy_io.npz after stop"
        ),
    )
    parser.add_argument(
        "--yes-i-am-supervising",
        action="store_true",
        help=(
            "authorize immediate automatic RH56/Franka reset followed by the "
            "supervised real-robot run; keep the full reset sweep clear"
        ),
    )
    return parser


def build_deployment_request(args: argparse.Namespace) -> DeploymentRequest:
    """Validate CLI arguments without opening any hardware interface."""

    policy_mode = resolve_policy_rate_mode(args.policy_rate_hz)
    steps = _parse_steps(args.steps, maximum=policy_mode.maximum_supervised_steps)
    visualization_rate = _parse_visualization_rate(args.live_visualization_rate_hz)
    try:
        arrival_tolerance = float(str(args.arrival_tolerance_rad).strip())
    except (TypeError, ValueError) as exc:
        raise SupervisedV94RunError(
            "--arrival-tolerance-rad must be in 0.001..0.030"
        ) from exc
    if not math.isfinite(arrival_tolerance) or not 0.001 <= arrival_tolerance <= 0.030:
        raise SupervisedV94RunError(
            "--arrival-tolerance-rad must be in 0.001..0.030"
        )
    try:
        arrival_timeout = float(str(args.arrival_timeout_s).strip())
    except (TypeError, ValueError) as exc:
        raise SupervisedV94RunError(
            "--arrival-timeout-s must be in 0.10..0.40"
        ) from exc
    if not math.isfinite(arrival_timeout) or not 0.10 <= arrival_timeout <= 0.40:
        raise SupervisedV94RunError(
            "--arrival-timeout-s must be in 0.10..0.40"
        )
    run_id = str(args.run_id or "").strip()
    if args.execute and not run_id:
        raise SupervisedV94RunError("--execute requires a unique non-empty --run-id")
    if args.execute and not args.yes_i_am_supervising:
        raise SupervisedV94RunError("--execute requires --yes-i-am-supervising")
    pcd_config = args.pcd_config.expanduser().resolve()
    task_policy_resolution = task_profile_policy_rgbd_resolution(pcd_config)
    if (
        task_policy_resolution is not None
        and str(args.policy_rgbd_resolution) != task_policy_resolution
    ):
        raise SupervisedV94RunError(
            "--policy-rgbd-resolution differs from the resolved task profile"
        )
    task_policy_rate = task_profile_policy_rate_hz(pcd_config)
    if task_policy_rate is not None and not math.isclose(
        policy_mode.policy_rate_hz,
        task_policy_rate,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise SupervisedV94RunError(
            "--policy-rate-hz differs from the resolved V57 task contract"
        )
    if args.execute and not task_profile_allows_robot_execution(pcd_config):
        raise SupervisedV94RunError(
            "robot execution is disabled by the selected task profile; "
            "finish and accept its camera commissioning first"
        )
    task_execute_cap = task_profile_maximum_supervised_execute_steps(pcd_config)
    if args.execute and task_execute_cap is not None and steps > task_execute_cap:
        raise SupervisedV94RunError(
            "requested steps exceed the selected task profile's supervised "
            f"execution cap of {task_execute_cap}"
        )
    object_roi = _parse_object_roi(args.object_roi)
    object_text = _parse_object_text(args.object_text)
    checkpoint = (
        None if args.checkpoint is None else args.checkpoint.expanduser().resolve()
    )
    if task_policy_rate is not None and checkpoint is None:
        raise SupervisedV94RunError(
            "the resolved V57 thrown-object task requires an explicit "
            "--checkpoint; the legacy V94 bundle policy must not be used "
            "implicitly"
        )
    record_video_path = (
        None
        if args.record_video is None
        else args.record_video.expanduser().resolve()
    )
    if record_video_path is not None:
        if record_video_path.suffix.lower() != ".mp4":
            raise SupervisedV94RunError("--record-video path must end in .mp4")
        if record_video_path.exists():
            raise SupervisedV94RunError(
                f"--record-video refuses to overwrite: {record_video_path}"
            )
        record_mask_video_path = mask_video_path_for_recording(record_video_path)
        if record_mask_video_path.exists():
            raise SupervisedV94RunError(
                "--record-video refuses to overwrite mask sidecar: "
                f"{record_mask_video_path}"
            )
    if bool(args.record_policy_io):
        policy_io_path = default_policy_io_path(run_id or "DRY_RUN")
        if policy_io_path.exists():
            raise SupervisedV94RunError(
                f"--record-policy-io refuses to overwrite: {policy_io_path}"
            )
    if checkpoint is not None and not checkpoint.is_file():
        raise SupervisedV94RunError(f"checkpoint is missing: {checkpoint}")
    if (
        args.execute
        and checkpoint is not None
        and object_roi is None
        and not args.select_object_roi
        and object_text is None
    ):
        raise SupervisedV94RunError(
            "an external --checkpoint requires --object-text, "
            "--select-object-roi, or --object-roi so object evidence is "
            "bound to the selected weights"
        )
    return DeploymentRequest(
        run_id=run_id or "DRY_RUN",
        steps=steps,
        bundle=args.bundle.expanduser().resolve(),
        profile=args.profile.expanduser().resolve(),
        pcd_config=pcd_config,
        execute=bool(args.execute),
        policy_rate_hz=policy_mode.policy_rate_hz,
        checkpoint=checkpoint,
        arrival_gated=bool(args.arrival_gated),
        arrival_tolerance_rad=arrival_tolerance,
        arrival_timeout_s=arrival_timeout,
        live_visualization=bool(args.live_visualization),
        live_visualization_rate_hz=visualization_rate,
        record_video_path=record_video_path,
        record_policy_io=bool(args.record_policy_io),
        object_roi_xywh=object_roi,
        select_object_roi=bool(args.select_object_roi),
        object_text=object_text,
        object_mask_mode=str(args.object_mask_mode),
        policy_rgbd_resolution=str(args.policy_rgbd_resolution),
        object_roi_source=(
            "text_grounding_camera_preflight"
            if object_text is not None
            else (
                "operator_numeric_camera_preflight"
                if object_roi is not None
                else "pinned_fixed_roi"
            )
        ),
    )


def _selected_point_feature_contract(
    request: DeploymentRequest,
) -> tuple[int, str, object, Optional[float]]:
    """Safely derive the policy point layout from the selected checkpoint."""

    from .bundle import (
        MAX_CHECKPOINT_BYTES,
        DeployBundle,
        load_checkpoint_safely,
    )

    if request.checkpoint is None:
        payload = DeployBundle(request.bundle).checkpoint_bytes()
    else:
        path = Path(request.checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        size = path.stat().st_size
        if size > MAX_CHECKPOINT_BYTES:
            raise ValueError(
                f"checkpoint exceeds {MAX_CHECKPOINT_BYTES} byte safety limit"
            )
        payload = path.read_bytes()
        if len(payload) != size:
            raise ValueError("checkpoint changed while it was being read")
    checkpoint = load_checkpoint_safely(payload)
    from sim2real.policy import RollingStudentPolicy
    from sim2real.observation.model import infer_fixed_sphere_radius_m

    policy = RollingStudentPolicy(checkpoint)
    dim = checkpoint.spec.get("point_feature_dim")
    mode = checkpoint.metadata.get("point_features")
    expected_mode = {3: "xyz", 6: "xyzrgb"}.get(dim)
    if expected_mode is None or mode != expected_mode:
        raise ValueError(
            "checkpoint point feature contract must be point_feature_dim=3/"
            "point_features='xyz' or point_feature_dim=6/"
            f"point_features='xyzrgb'; actual dim={dim!r} mode={mode!r}"
        )
    return (
        int(dim),
        expected_mode,
        policy.action_controller,
        infer_fixed_sphere_radius_m(checkpoint.metadata),
    )


def build_deployment_summary(
    request: DeploymentRequest,
) -> Mapping[str, object]:
    """Return the hardware-inert execution plan printed before a run."""

    from .execution_reset import (
        RESET_DWELL_S,
        RESET_FRANKA_ARRIVAL_TOLERANCE_RAD,
        RESET_FRANKA_MAX_JOINT_SPEED_RAD_S,
        RESET_FRANKA_MIN_SEGMENT_DURATION_S,
        RESET_MAX_AXIS_CURRENT_MA,
        RESET_MAX_FRANKA_START_DELTA_RAD,
    )

    policy_mode = resolve_policy_rate_mode(request.policy_rate_hz)
    from .bundle import DeployBundle
    from sim2real.contracts.v94 import V94Contract
    from sim2real.observation.model import (
        PolicyRGBDResolutionAdapter,
        resolve_policy_rgbd_resolution,
    )

    summary_contract = resolve_runtime_task_contract(
        V94Contract.from_bundle(DeployBundle(request.bundle)).with_runtime_policy_rate_hz(
            policy_mode.policy_rate_hz
        ),
        request.pcd_config,
    )
    summary_policy_rgbd_adapter = PolicyRGBDResolutionAdapter(
        camera_K=summary_contract.camera_K,
        source_image_size=(
            summary_contract.camera_width,
            summary_contract.camera_height,
        ),
        target_image_size=resolve_policy_rgbd_resolution(
            request.policy_rgbd_resolution
        ),
    )
    from sim2real.tasks.thrown_contract import v57_task_summary

    thrown_task_summary = v57_task_summary(
        request.pcd_config, summary_contract
    )
    (
        point_feature_dim,
        point_feature_mode,
        action_controller,
        fixed_sphere_completion_radius_m,
    ) = _selected_point_feature_contract(request)
    qd_relative_arm = (
        str(getattr(action_controller, "contract_id", ""))
        == QD_G015_ACTION_CONTROLLER_CONTRACT_ID
    )
    arm_effective_delta = (
        action_controller.arm_raw_gain_rad
        * action_controller.arm_target_filter_alpha
    )
    if not qd_relative_arm:
        arm_effective_delta = min(
            action_controller.maximum_arm_target_step_rad,
            arm_effective_delta,
        )
    from sim2real.policy.rate_mode import rh56_register_delta_envelope

    rh56_target_contract_envelope = rh56_register_delta_envelope(
        policy_rate_hz=policy_mode.policy_rate_hz,
        hardware_rate_hz=policy_mode.rh56_target_rate_hz,
        maximum_semantic_target_step_rad=(
            action_controller.maximum_hand_target_step_rad
        ),
    )
    replay_path = getattr(request, "replay_actions", None)
    replay = None
    replay_summary: Mapping[str, object]
    if replay_path is None:
        replay_summary = {"enabled": False}
    else:
        from sim2real.action_replay import (
            load_replay_actions,
            summarize_replay_actions,
        )

        replay = load_replay_actions(
            replay_path,
            expected_policy_rate_hz=policy_mode.policy_rate_hz,
            selected_steps=request.steps,
        )
        replay_summary = {
            "enabled": True,
            "path": str(Path(replay_path).expanduser().resolve()),
            "sha256": replay.sha256,
            "source_format": replay.source_format,
            "available_actions": replay.action_count,
            "selected_actions": int(request.steps),
            "advance_semantics": (
                "advance_only_after_exact_dual_ack_with_replay_defined_"
                "arrival_boundaries"
                if bool(getattr(request, "replay_arrival_gated", False))
                else "advance_only_after_exact_dual_device_ack"
            ),
            "arrival_gated": bool(
                getattr(request, "replay_arrival_gated", False)
            ),
            "arrival_tolerance_rad": (
                float(getattr(request, "replay_arrival_tolerance_rad", 0.015))
                if bool(getattr(request, "replay_arrival_gated", False))
                else None
            ),
            "maximum_hardware_commands": (
                policy_mode.maximum_supervised_steps
                if bool(getattr(request, "replay_arrival_gated", False))
                else int(request.steps)
            ),
            "execution_contract": (
                "exact_recorded_actuator_targets_when_present_otherwise_"
                "normalized_action13_through_current_deployment_mapper"
            ),
            "recorded_actuator_targets": (
                "direct_transactional_targets; Franka_V225_1khz_interpolator_"
                "plus_official_limiter_retained; RH56_host_slew_bypassed"
            ),
            "observation_pipeline": "live_and_identical_to_policy_deployment",
            "trajectory_preview": summarize_replay_actions(
                replay, selected_steps=int(request.steps)
            ),
        }
    task_execute_cap = task_profile_maximum_supervised_execute_steps(
        Path(request.pcd_config)
    )
    effective_step_cap = min(
        policy_mode.maximum_supervised_steps,
        task_execute_cap
        if task_execute_cap is not None
        else policy_mode.maximum_supervised_steps,
    )
    if request.steps > policy_mode.maximum_supervised_steps:
        raise SupervisedV94RunError(
            f"{policy_mode.name} mode permits at most "
            f"{policy_mode.maximum_supervised_steps} supervised steps"
        )
    rh56_bounds = load_v94_rh56_profile_command_bounds(
        request.profile,
        feedback_to_command_offset_units=(RH56_FEEDBACK_TO_COMMAND_OFFSET_UNITS),
    )
    rh56_force_set_g = load_commissioned_rh56_force_set_g(request.profile)
    if (
        replay is not None
        and replay.recorded_rh56_angle_set_register_order is not None
    ):
        selected = replay.recorded_rh56_angle_set_register_order[
            : int(request.steps)
        ]
        recorded_min = tuple(int(value) for value in selected.min(axis=0))
        recorded_max = tuple(int(value) for value in selected.max(axis=0))
        commissioned_min = tuple(
            int(value) for value in rh56_bounds.minimum_angle_set_register_order
        )
        commissioned_max = tuple(
            int(value) for value in rh56_bounds.maximum_angle_set_register_order
        )
        violations = [
            {
                "register_axis": index,
                "recorded_min": recorded_min[index],
                "recorded_max": recorded_max[index],
                "commissioned_min": commissioned_min[index],
                "commissioned_max": commissioned_max[index],
            }
            for index in range(6)
            if recorded_min[index] < commissioned_min[index]
            or recorded_max[index] > commissioned_max[index]
        ]
        preview = replay_summary.get("trajectory_preview")
        if isinstance(preview, dict):
            target_audit = preview.get("recorded_target_audit")
            if isinstance(target_audit, dict):
                target_audit["rh56_recorded_targets_fit_commissioned_bounds"] = (
                    not violations
                )
                target_audit["rh56_commissioned_bound_violations"] = violations
    return {
        "mode": "experimental_operator_supervised_non_c2",
        "hardware_access": bool(request.execute),
        "execute_requested": request.execute,
        "run_id": request.run_id,
        "steps": request.steps,
        "hard_step_cap": effective_step_cap,
        "policy_mode": {
            "name": policy_mode.name,
            "policy_rate_hz": policy_mode.policy_rate_hz,
            "control_dt_s": policy_mode.control_dt_s,
            "checkpoint_control_dt_must_match": True,
        },
        "sim_control_alignment": sim_control_alignment_summary(),
        "bundle": str(request.bundle),
        "checkpoint": {
            "source": (
                "bundle_manifest_primary"
                if request.checkpoint is None
                else "external_path_override"
            ),
            "path": (None if request.checkpoint is None else str(request.checkpoint)),
            "validation": (
                "bundle_hash_and_golden_replay"
                if request.checkpoint is None
                else (
                    "safe_decode_structural_contract_and_offline_"
                    "observation_smoke;bundle_golden_action_equality_does_"
                    "not_apply"
                )
            ),
            "point_feature_mode": point_feature_mode,
            "point_feature_dim": point_feature_dim,
            "fixed_sphere_completion_radius_m": (
                fixed_sphere_completion_radius_m
            ),
            "action_controller": {
                "contract_id": getattr(action_controller, "contract_id", None),
                "incremental_reference": (
                    "current_shaper_q_d"
                    if qd_relative_arm
                    else "previous_committed_target"
                ),
                "arm_raw_gain_rad": action_controller.arm_raw_gain_rad,
                "arm_target_filter_alpha": (
                    action_controller.arm_target_filter_alpha
                ),
                "arm_effective_max_delta_rad_per_tick": arm_effective_delta,
                "maximum_arm_target_step_rad": (
                    None
                    if qd_relative_arm
                    else action_controller.maximum_arm_target_step_rad
                ),
                "measured_q_target_envelope": (
                    "disabled" if qd_relative_arm else "0.05rad"
                ),
                "hand_target_filter_alpha": (
                    action_controller.hand_target_filter_alpha
                ),
                "maximum_hand_target_step_rad": (
                    action_controller.maximum_hand_target_step_rad
                ),
                "source": "checkpoint_metadata_or_frozen_v94_default",
            },
        },
        "closed_loop_arrival_gate": {
            "enabled": bool(request.arrival_gated and replay_path is None),
            "scope": "franka_only_before_next_checkpoint_inference",
            "arrival_criterion": "per_axis_tolerance_or_crossing_latched",
            "tolerance_rad": (
                float(request.arrival_tolerance_rad)
                if request.arrival_gated and replay_path is None
                else None
            ),
            "per_target_timeout_s": (
                float(request.arrival_timeout_s)
                if request.arrival_gated and replay_path is None
                else None
            ),
            "policy_timing": (
                "arrival_driven_diagnostic_not_fixed_rate"
                if request.arrival_gated and replay_path is None
                else "fixed_rate"
            ),
            "native_franka_v225_1khz_interpolator_retained": True,
            "libfranka_official_rate_limiter_retained": True,
        },
        "action_source": (
            {"kind": "checkpoint_policy", **replay_summary}
            if replay_path is None
            else {
                "kind": (
                    "online_visual_intercept_planner"
                    if replay is not None
                    and replay.tabletop_online_planner is not None
                    else "validated_simulation_action_replay"
                ),
                **replay_summary,
            }
        ),
        "profile": str(request.profile),
        "pcd_config": str(request.pcd_config),
        "thrown_task_contract": thrown_task_summary,
        "object_mask": {
            "mode": request.object_mask_mode,
            "requested_object_mask_mode": request.object_mask_mode,
            "effective_object_mask_mode": (
                "guarded_v2"
                if request.object_mask_mode == "guarded"
                else request.object_mask_mode
            ),
            "provider_publication_mode": (
                "adaptive_fusion"
                if request.object_mask_mode != "legacy"
                else "semantic_sam2"
            ),
            "effective_provider_mask_publication_mode": (
                "guarded_sam2_primary"
                if request.object_mask_mode in ("guarded", "guarded_v2")
                else (
                    "adaptive_fusion"
                    if request.object_mask_mode == "guarded_v1"
                    else "semantic_sam2"
                )
            ),
            "recovery_publication_mode": (
                "unified_three_evidence"
                if request.object_mask_mode in ("guarded", "guarded_v2")
                else (
                    "legacy_double_confirm"
                    if request.object_mask_mode == "guarded_v1"
                    else "semantic_sam2_direct"
                )
            ),
            "effective_provider_recovery_publication_mode": (
                "unified_three_evidence"
                if request.object_mask_mode in ("guarded", "guarded_v2")
                else (
                    "legacy_double_confirm"
                    if request.object_mask_mode == "guarded_v1"
                    else "semantic_sam2_direct"
                )
            ),
            "legacy_comparison_available": True,
            "stale_palm_policy": (
                "legacy_explicit_compatibility"
                if request.object_mask_mode == "legacy"
                else "recoverable_fail_closed_no_policy_history_mapper_stage"
            ),
            "stale_palm_legacy_compatibility_switch": (
                "explicit_--object-mask-mode=legacy_only"
            ),
        },
        "policy_rgbd": {
            "resolution": [
                int(part)
                for part in request.policy_rgbd_resolution.split("x")
            ],
            "native_d435_and_sam2_resolution": [
                summary_contract.camera_width,
                summary_contract.camera_height,
            ],
            "adapter": (
                "identity"
                if summary_policy_rgbd_adapter.stride == 1
                else "aligned_stride2_no_interpolation"
            ),
            "camera_intrinsics_scaled_with_resolution": True,
        },
        "object_roi": {
            "mode": (
                "camera_only_text_grounding_preflight"
                if request.object_text is not None
                else (
                    "camera_only_interactive_preflight"
                    if request.select_object_roi
                    else (
                        "camera_only_numeric_preflight"
                        if request.object_roi_xywh is not None
                        else "pinned_fixed_roi"
                    )
                )
            ),
            "text": request.object_text,
            "xywh": (
                None
                if request.object_roi_xywh is None
                else list(request.object_roi_xywh)
            ),
            "occurs_after_automatic_reset": True,
            "robot_interfaces_opened_during_preflight": False,
        },
        "automatic_reset": {
            "enabled_on_execute": True,
            "sequence": (
                "rh56_canonical_open_disabled_then_prepared_task_q_home"
                if thrown_task_summary is not None
                else "rh56_canonical_open_disabled_then_franka_v94_q_home"
            ),
            "franka_target_q_rad": summary_contract.q_home_rad.tolist(),
            "rh56_strict_current_cap_ma": RESET_MAX_AXIS_CURRENT_MA,
            "franka_max_start_linf_from_v94_home_rad": (
                RESET_MAX_FRANKA_START_DELTA_RAD
            ),
            "franka_arrival_linf_tolerance_rad": (RESET_FRANKA_ARRIVAL_TOLERANCE_RAD),
            "franka_max_joint_speed_rad_s": RESET_FRANKA_MAX_JOINT_SPEED_RAD_S,
            "franka_min_segment_duration_s": (
                RESET_FRANKA_MIN_SEGMENT_DURATION_S
            ),
            "post_reset_stationary_dwell_s": RESET_DWELL_S,
            "operator_supervised_sweep_clear_assumption": (
                "required_but_not_machine_verified"
            ),
            "installed_collision_model_verified": False,
        },
        "live_visualization": {
            "requested": request.live_visualization,
            "background_output_pipeline_enabled": bool(
                request.live_visualization or request.record_video_path is not None
            ),
            "update_rate_hz": request.live_visualization_rate_hz,
            "content": "exact_frame_final_policy_mask_and_128_point_cloud",
            "control_path": "nonblocking_latest_only_isolated_process",
            "record_video_path": (
                None
                if request.record_video_path is None
                else str(request.record_video_path)
            ),
            "record_mask_video_path": (
                None
                if request.record_video_path is None
                else str(mask_video_path_for_recording(request.record_video_path))
            ),
            "record_video_overlay": (
                None
                if request.record_video_path is None
                else "none"
            ),
            "record_video_rate_hz": (
                None
                if request.record_video_path is None
                else min(float(policy_mode.policy_rate_hz), 30.0)
            ),
            "save_after_stop": bool(
                request.live_visualization or request.record_video_path is not None
            ),
            "save_directory": (
                None
                if not (
                    request.live_visualization
                    or request.record_video_path is not None
                )
                else str(
                    Path(__file__).resolve().parents[2]
                    / "dexgrasp"
                    / "runs"
                    / f"{request.run_id}_observation_visualization"
                )
            ),
        },
        "policy_io_recording": {
            "requested": bool(request.record_policy_io),
            "path": (
                str(default_policy_io_path(request.run_id))
                if request.record_policy_io
                else None
            ),
            "capture": (
                "accepted_non_actuated_startup_and_dual_ack_ticks_only"
            ),
            "control_path": "in_memory_copy_only_then_post_stop_atomic_npz",
            "normalization": "checkpoint_constants_applied_after_stop",
        },
        "hard_guards": {
            "franka_maximum_session_duration_s": (FRANKA_MAX_SESSION_DURATION_S),
            "policy_rate_hz": policy_mode.policy_rate_hz,
            "franka_start_linf_from_v94_home_rad": FRANKA_MAX_START_ERROR_RAD,
            "franka_command_speed_rad_s": FRANKA_MAX_COMMAND_SPEED_RAD_S,
            "franka_command_acceleration_rad_s2": (
                REAL_FRANKA_MAX_COMMAND_ACCELERATION_RAD_S2
            ),
            "franka_command_jerk_rad_s3": (
                REAL_FRANKA_MAX_COMMAND_JERK_RAD_S3
            ),
            "franka_measured_velocity_fault_rad_s": (
                FRANKA_MAX_MEASURED_VELOCITY_RAD_S
            ),
            "franka_collision_behavior_source": (
                "libfranka_two_level_contact_hold_then_installed_rh56_collision_stop"
            ),
            "franka_collision_behavior_applied_before_control": True,
            "franka_contact_torque_thresholds_nm": list(
                FRANKA_CONTACT_TORQUE_THRESHOLDS_NM
            ),
            "franka_contact_force_thresholds_n": list(
                FRANKA_CONTACT_FORCE_THRESHOLDS_N
            ),
            "franka_collision_torque_thresholds_nm": list(
                FRANKA_COLLISION_TORQUE_THRESHOLDS_NM
            ),
            "franka_collision_force_thresholds_n": list(
                FRANKA_COLLISION_FORCE_THRESHOLDS_N
            ),
            "franka_contact_response": (
                "native_1khz_filtered_target_hold; resume_latest_policy_target_when_clear"
            ),
            "franka_tick_target_delta_rad": (
                None if qd_relative_arm else FRANKA_MAX_TICK_TARGET_DELTA_RAD
            ),
            "franka_episode_delta_rad": (
                None if qd_relative_arm else FRANKA_MAX_EPISODE_DELTA_RAD
            ),
            "franka_tracking_error_rad": FRANKA_MAX_TRACKING_ERROR_RAD,
            "franka_first_bootstrap_read_to_write_s": (
                FRANKA_FIRST_BOOTSTRAP_READ_TO_WRITE_S
            ),
            "franka_steady_read_to_write_s": FRANKA_STEADY_READ_TO_WRITE_S,
            "franka_fci_max_recoverable_control_period_s": (
                FRANKA_FCI_MAX_RECOVERABLE_CONTROL_PERIOD_S
            ),
            "franka_parent_heartbeat_receipt_timeout_s": (
                FRANKA_PARENT_HEARTBEAT_RECEIPT_TIMEOUT_S
            ),
            "franka_policy_target_max_age_s": FRANKA_POLICY_TARGET_MAX_AGE_S,
            "franka_policy_inter_target_timeout_s": (
                FRANKA_POLICY_INTER_TARGET_TIMEOUT_S
            ),
            "franka_arrival_gate_enabled": bool(
                request.arrival_gated and replay_path is None
            ),
            "franka_arrival_gate_tolerance_rad": (
                float(request.arrival_tolerance_rad)
                if request.arrival_gated and replay_path is None
                else None
            ),
            "franka_arrival_gate_timeout_s": (
                float(request.arrival_timeout_s)
                if request.arrival_gated and replay_path is None
                else None
            ),
            "franka_observation_action_max_age_s": (
                FRANKA_OBSERVATION_ACTION_MAX_AGE_S
            ),
            "franka_observation_hard_max_age_s": (FRANKA_OBSERVATION_HARD_MAX_AGE_S),
            "deployment_compute_threads": DEPLOYMENT_COMPUTE_THREADS,
            "franka_servo_process_affinity_uses_separate_physical_core": True,
            "franka_servo_host_wide_core_exclusivity_claimed": False,
            "franka_servo_single_logical_cpu": True,
            "franka_native_realtime_scheduler_proof": (
                "SCHED_FIFO_max_priority_single_cpu"
            ),
            "franka_nic_irq_affinity_uses_separate_physical_core": True,
            "franka_nic_irq_host_wide_core_exclusivity_claimed": False,
            "franka_nic_irq_stable_ownership": (
                "root_guard_schema2;irqbalance_runtime_masked_and_absent;"
                "requested_equals_effective;pre_reset_context_and_immediate_"
                "pre_popen_exact_reverification"
            ),
            "visualization_is_lossy_and_out_of_band": True,
            "camera_max_policy_actions_per_frame": (
                policy_mode.camera_max_policy_actions_per_frame
            ),
            "qd_g015_startup_non_actuated_policy_steps": (
                QD_G015_STARTUP_NON_ACTUATED_POLICY_STEPS
                if qd_relative_arm and replay_path is None
                else 0
            ),
            "policy_schedule": (
                "phase_locked_no_catch_up"
                if math.isclose(
                    policy_mode.policy_rate_hz,
                    20.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                else "legacy_minimum_period"
            ),
            "camera_runtime_frame_timeout_s": (D435_RUNTIME_FRAME_TIMEOUT_S),
            "camera_formal_publication_stall_s": (D435_FORMAL_PUBLICATION_STALL_S),
            "object_roi_preflight_max_valid_mask_gap_s": (
                OBJECT_ROI_PREFLIGHT_MAX_VALID_MASK_GAP_S
            ),
            "object_pointcloud_max_age_s": OBJECT_POINTCLOUD_MAX_AGE_S,
            "maximum_consecutive_no_stage_observation_hold_s": (
                MAXIMUM_CONSECUTIVE_NO_STAGE_HOLD_S
            ),
            "stale_palm_contract": (
                (
                    "legacy_explicit_compatibility_previous_native_palm_cloud;"
                    "current_formal_camera_frame_drives_transport_liveness;"
                    "pointcloud_source_frame_drives_content_reuse;"
                    "no_local_cloud_content_age_limit"
                )
                if request.object_mask_mode == "legacy"
                else (
                    "guarded_recoverable_fail_closed_before_policy_history_"
                    "mapper_stage;legacy_only_via_explicit_object_mask_mode_legacy"
                )
            ),
            "formal_online_sam2": (
                "honor_enabled_config;asynchronous_exact_frame_result_only;"
                "adaptive_identity_depth_recovery_gates_required"
            ),
            "rh56_speed_set": RH56_SPEED_SET,
            "rh56_force_set_g": rh56_force_set_g,
            "rh56_running_current_ma": RH56_MAX_RUNNING_CURRENT_MA,
            "rh56_stop_settle_transient_current_ma": (
                RH56_STOP_SETTLE_MAX_AXIS_CURRENT_MA
            ),
            "rh56_post_disable_idle_current_ma": 100,
            "rh56_target_schedule_rate_hz": (policy_mode.rh56_target_rate_hz),
            "rh56_target_schedule_semantics": (
                policy_mode.rh56_target_schedule_semantics
            ),
            "rh56_target_contract_envelope_units_per_update": (
                rh56_target_contract_envelope
            ),
            # Legacy audit key retained for consumers of earlier run records.
            "rh56_target_slew_units_per_update": (
                rh56_target_contract_envelope
            ),
            "rh56_min_angle_set_register_order": (
                rh56_bounds.minimum_angle_set_register_order
            ),
            "rh56_max_angle_set_register_order": (
                rh56_bounds.maximum_angle_set_register_order
            ),
            "rh56_feedback_to_command_valid_min": (
                rh56_bounds.feedback_to_command_valid_min
            ),
            "rh56_feedback_to_command_valid_max": (
                rh56_bounds.feedback_to_command_valid_max
            ),
            "rh56_commissioned_exact_targets": (rh56_bounds.commissioned_exact_targets),
            "rh56_nominal_feedback_rate_hz": RH56_NOMINAL_FEEDBACK_RATE_HZ,
            "rh56_feedback_sample_hold_max_age_s": RH56_FEEDBACK_HARD_AGE_S,
            "rh56_feedback_to_command_offset_units": (
                RH56_FEEDBACK_TO_COMMAND_OFFSET_UNITS
            ),
            "rh56_disabled_open_tolerance_units": (RH56_DISABLED_OPEN_TOLERANCE_UNITS),
            "rh56_tracking_significant_gap_units": (
                RH56_TRACKING_SIGNIFICANT_GAP_UNITS
            ),
            "rh56_tracking_min_progress_units": (RH56_TRACKING_MIN_PROGRESS_UNITS),
            "rh56_tracking_contact_force_delta_g": (
                RH56_TRACKING_CONTACT_FORCE_DELTA_G
            ),
            "rh56_tracking_contact_force_absolute_g": (
                RH56_TRACKING_CONTACT_FORCE_ABSOLUTE_G
            ),
            "rh56_tracking_timeout_s": RH56_TRACKING_TIMEOUT_S,
            "rh56_write_response_grace_s": RH56_WRITE_RESPONSE_GRACE_S,
            "rh56_exact_readback_request_count": (RH56_EXACT_READBACK_REQUEST_COUNT),
            "rh56_exact_readback_timeout_s": RH56_EXACT_READBACK_TIMEOUT_S,
            "rh56_command_max_age_s": RH56_COMMAND_MAX_AGE_S,
            "rh56_inter_command_watchdog_s": RH56_INTER_COMMAND_WATCHDOG_S,
            "rh56_stop_timeout_s": RH56_STOP_TIMEOUT_S,
            "rh56_target_ack": (
                "20hz_latest_only_changed_target_single_write_ack; "
                "held_target_no_numeric_rewrite; missing_write_response_"
                "accepted_only_if_one_exact_readback_matches; "
                "no_numeric_retry"
            ),
            "dual_device_same_sequence_ack": True,
            "ctrl_c_requests_both_verified_stop_paths": True,
            "cross_process_franka_rh56_lease": True,
            "permanent_unique_run_id_claim": True,
        },
        "warning": (
            "This path is deliberately not a formal C2 claim. RH56 target "
            "write ACK (or missing-ACK exact readback) and physical "
            "ANGLE_ACT/POS_ACT first-motion proof per exercised axis are "
            "enforced; later healthy contact holds are allowed for this "
            "operator-supervised run."
        ),
    }


def _preflight_current_object_roi(
    request: DeploymentRequest,
    *,
    retain_live_provider: bool = False,
) -> DeploymentRequest:
    """Resolve and validate a movable object's ROI using only the D435.

    The deployment entry point calls this after the automatic reset and its
    stationary dwell, while retaining the cross-process hardware lease.  This
    phase itself opens only the D435: no Franka/RH56 interface is opened and no
    control command is sent.  Interactive selection is converted immediately
    to immutable numeric XYWH; the policy control lifecycle therefore never
    owns a GUI.
    """

    import numpy as np

    from sim2real.observation.capture import (
        _ComputeThreadGuard,
        _initialized_roi_evidence,
        capture_valid_frames,
    )
    from .bundle import DeployBundle
    from sim2real.observation.live_preview import _initialize_provider
    from sim2real.contracts.v94 import V94Contract
    from sim2real.observation.model import (
        MaskedRGBDProjector,
        PolicyRGBDResolutionAdapter,
        infer_fixed_sphere_radius_m,
        resolve_policy_rgbd_resolution,
    )

    if (
        request.object_roi_xywh is None
        and not request.select_object_roi
        and request.object_text is None
    ):
        return request

    def file_sha256(path: Path) -> str:
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
        return result.hexdigest()

    artifact_paths = (request.bundle, request.pcd_config) + (
        () if request.checkpoint is None else (request.checkpoint,)
    )
    artifact_hashes_before = tuple(file_sha256(path) for path in artifact_paths)
    bundle = DeployBundle(request.bundle)
    bundle.verify()
    contract = resolve_runtime_task_contract(
        V94Contract.from_bundle(bundle).with_runtime_policy_rate_hz(
            request.policy_rate_hz
        ),
        request.pcd_config,
    )
    (
        point_feature_dim,
        point_feature_mode,
        _,
        fixed_sphere_completion_radius_m,
    ) = _selected_point_feature_contract(
        request
    )
    if request.checkpoint is None:
        primary = bundle.manifest.get("primary_checkpoint", {})
        selected_checkpoint_sha256 = (
            str(primary.get("sha256", "")).strip().lower()
            if isinstance(primary, Mapping)
            else ""
        )
    else:
        selected_checkpoint_sha256 = artifact_hashes_before[2]
    if (
        request.selected_checkpoint_sha256
        and request.selected_checkpoint_sha256 != selected_checkpoint_sha256
    ):
        raise SupervisedV94RunError(
            "selected checkpoint changed before object ROI preflight"
        )
    requested_roi = request.object_roi_xywh
    if request.select_object_roi or request.object_text is not None:
        print(
            (
                f"[Object grounding START] text={request.object_text!r}"
                if request.object_text is not None
                else "[Object ROI selector START] waiting for operator box"
            ),
            flush=True,
        )
        requested_roi = _select_object_roi_isolated(
            request,
            expected_camera_serial=contract.camera_serial,
            expected_calibration_id=contract.calibration_id,
        )
        print(
            (
                "[Object grounding PASS] "
                if request.object_text is not None
                else "[Object ROI selector PASS] "
            )
            + (
                f"text={request.object_text!r} "
                if request.object_text is not None
                else ""
            )
            + f"numeric_xywh={list(requested_roi)}; "
            "isolated camera child closed; Franka/RH56 interfaces opened=0",
            flush=True,
        )
        if request.object_text is not None:
            print(
                "[Object grounding VERIFYING] candidate locked; keep it visible "
                "for about 2 s while final SAM2 mask, depth and point cloud are "
                "validated",
                flush=True,
            )
    if requested_roi is None:
        raise SupervisedV94RunError("object ROI selection produced no numeric XYWH")
    provider = None
    handoff = None
    compute_guard = None
    started = time.monotonic()
    try:
        compute_guard = _ComputeThreadGuard(DEPLOYMENT_COMPUTE_THREADS).start()
        provider, _online_selection = _initialize_provider(
            request.pcd_config,
            requested_roi,
            disable_online_sam2=False,
            object_mask_mode=request.object_mask_mode,
        )
        initialization = _initialized_roi_evidence(provider)
        resolved_array = np.asarray(initialization["roi_xywh"], dtype=np.int32)
        if resolved_array.shape != (4,):
            raise SupervisedV94RunError(
                "camera-only object ROI preflight returned malformed XYWH"
            )
        resolved_roi = tuple(int(value) for value in resolved_array.tolist())
        initialization_mask_bbox = tuple(
            int(value)
            for value in np.asarray(
                initialization["initialization_mask_bbox_xyxy"],
                dtype=np.int32,
            ).tolist()
        )
        initialization_mask_area = int(
            np.asarray(initialization["initialization_mask_area_px"]).item()
        )
        initialization_mask_source = str(
            np.asarray(initialization["initialization_mask_source"]).item()
        )
        if requested_roi is not None and resolved_roi != tuple(requested_roi):
            raise SupervisedV94RunError(
                "camera provider clipped or changed --object-roi"
            )
        x, y, width, height = resolved_roi
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > contract.camera_width
            or y + height > contract.camera_height
        ):
            raise SupervisedV94RunError(
                "selected object ROI exceeds the V94 camera frame"
            )
        if provider.extrinsics.calibration_id != contract.calibration_id:
            raise SupervisedV94RunError(
                "object ROI preflight calibration differs from V94"
            )
        if provider.extrinsics.camera_serial != contract.camera_serial:
            raise SupervisedV94RunError(
                "object ROI preflight camera serial differs from V94"
            )
        if not np.allclose(
            np.asarray(provider.extrinsics.T_base_camera, dtype=np.float64),
            contract.T_base_camera_optical,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise SupervisedV94RunError(
                "object ROI preflight extrinsics differ from V94"
            )

        payload, counters = capture_valid_frames(
            provider,
            stored_frames=OBJECT_ROI_PREFLIGHT_VALID_FRAMES,
            sample_every_valid_frame=1,
            maximum_attempts=OBJECT_ROI_PREFLIGHT_MAX_ATTEMPTS,
            require_consecutive_valid_frames=True,
        )
        invalid_publications = int(counters["invalid_publications"])
        valid_publications = int(counters["valid_publications"])
        provider_steps = int(counters["provider_steps"])
        retryable_camera_timeouts = int(
            counters.get("retryable_camera_timeouts", 0)
        )
        if (
            valid_publications < OBJECT_ROI_PREFLIGHT_VALID_FRAMES
            or provider_steps != valid_publications + invalid_publications
            or invalid_publications > OBJECT_ROI_PREFLIGHT_MAX_INVALID_FRAMES
        ):
            raise SupervisedV94RunError(
                "object ROI preflight exceeded its bounded invalid-mask warm-up"
            )
        stored_frame_ids = np.asarray(payload["camera_frame_id"], dtype=np.int64)
        stored_timestamps = np.asarray(payload["camera_timestamp_s"], dtype=np.float64)
        expected_preflight_shape = (OBJECT_ROI_PREFLIGHT_VALID_FRAMES,)
        if (
            stored_frame_ids.shape != expected_preflight_shape
            or stored_timestamps.shape != expected_preflight_shape
        ):
            raise SupervisedV94RunError(
                "object ROI preflight returned malformed frame provenance"
            )
        # ``capture_valid_frames(... require_consecutive_valid_frames=True)``
        # already proves three adjacent provider publications were valid.  A
        # semantic/GPU worker may legitimately skip one or more native D435
        # frame numbers between those publications; freshness needs strict
        # forward progress, not an artificial frame_id + 1 rule.
        timestamp_deltas = _validate_object_roi_preflight_progress(
            stored_frame_ids,
            stored_timestamps,
        )
        if not np.allclose(
            np.asarray(payload["frame_camera_K"], dtype=np.float64),
            contract.camera_K[None],
            atol=1.0e-5,
            rtol=0.0,
        ):
            raise SupervisedV94RunError(
                "object ROI preflight D435 intrinsics differ from V94"
            )
        if not np.allclose(
            np.asarray(payload["frame_camera_distortion"], dtype=np.float64),
            0.0,
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise SupervisedV94RunError(
                "object ROI preflight D435 distortion differs from V94"
            )
        if not np.allclose(
            np.asarray(
                payload["frame_depth_scale_m_per_unit"],
                dtype=np.float64,
            ),
            contract.depth_scale_m_per_unit,
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise SupervisedV94RunError(
                "object ROI preflight D435 depth scale differs from V94"
            )
        if np.asarray(payload["rgb"]).shape[1:] != (
            contract.camera_height,
            contract.camera_width,
            3,
        ):
            raise SupervisedV94RunError(
                "object ROI preflight image dimensions differ from V94"
            )

        policy_rgbd_adapter = PolicyRGBDResolutionAdapter(
            camera_K=contract.camera_K,
            source_image_size=(contract.camera_width, contract.camera_height),
            target_image_size=resolve_policy_rgbd_resolution(
                request.policy_rgbd_resolution
            ),
        )
        projector = MaskedRGBDProjector(
            camera_K=policy_rgbd_adapter.camera_K,
            T_base_camera_optical=contract.T_base_camera_optical,
            image_size=policy_rgbd_adapter.target_image_size,
            depth_range_m=contract.depth_range_m,
            point_feature_dim=point_feature_dim,
            maximum_mask_depth_deviation_m=0.055,
            fixed_sphere_completion_radius_m=(
                fixed_sphere_completion_radius_m
            ),
            support_plane_abcd=getattr(
                getattr(provider, "extractor", None),
                "support_plane_abcd",
                None,
            ),
            support_plane_min_clearance_m=float(
                getattr(
                    getattr(provider, "extractor", None),
                    "support_plane_min_clearance_m",
                    0.0,
                )
            ),
        )
        source_counts = []
        depth_values = []
        digest = hashlib.sha256()
        digest.update(artifact_hashes_before[0].encode("ascii"))
        digest.update(artifact_hashes_before[1].encode("ascii"))
        digest.update(selected_checkpoint_sha256.encode("ascii"))
        digest.update(point_feature_mode.encode("ascii"))
        digest.update((request.object_text or "").encode("utf-8"))
        digest.update(request.object_mask_mode.encode("ascii"))
        digest.update(request.policy_rgbd_resolution.encode("ascii"))
        digest.update(policy_rgbd_adapter.camera_K.tobytes())
        digest.update(np.asarray(point_feature_dim, dtype=np.int32).tobytes())
        digest.update(np.asarray(resolved_roi, dtype=np.int32).tobytes())
        digest.update(np.asarray(initialization_mask_bbox, dtype=np.int32).tobytes())
        digest.update(np.asarray(initialization_mask_area, dtype=np.int32).tobytes())
        digest.update(initialization_mask_source.encode("utf-8"))
        for index in range(OBJECT_ROI_PREFLIGHT_VALID_FRAMES):
            native_mask = np.asarray(payload["object_mask"][index], dtype=bool)
            native_depth_raw = np.asarray(
                payload["depth_raw"][index], dtype=np.uint16
            )
            policy_rgbd = policy_rgbd_adapter.adapt(
                color_bgr=np.asarray(payload["rgb"][index], dtype=np.uint8),
                depth_raw=native_depth_raw,
                object_mask=native_mask,
            )
            mask = policy_rgbd.object_mask
            depth_raw = policy_rgbd.depth_raw
            assert depth_raw is not None
            scale = float(payload["frame_depth_scale_m_per_unit"][index])
            point_frame = projector.project(
                color_bgr=policy_rgbd.color_bgr,
                depth_raw=depth_raw,
                depth_scale_m_per_unit=scale,
                object_mask=mask,
                T_base_palm_at_capture=np.eye(4, dtype=np.float64),
                captured_at_s=float(payload["camera_timestamp_s"][index]),
                frame_id=int(payload["camera_frame_id"][index]),
            )
            if point_frame.status != "fresh":
                raise SupervisedV94RunError(
                    "selected object mask/depth cannot produce a fresh policy "
                    f"point cloud: frame={index} status={point_frame.status}"
                )
            points = np.asarray(point_frame.xyzrgb_palm, dtype=np.float32)
            validity = np.asarray(point_frame.valid, dtype=np.float32)
            if (
                points.shape != (128, point_feature_dim)
                or validity.shape != (128,)
                or not np.all(np.isfinite(points))
                or not np.all(np.isfinite(validity))
            ):
                raise SupervisedV94RunError(
                    "selected object produced a malformed/non-finite policy "
                    "point-cloud tensor"
                )
            source_counts.append(int(point_frame.source_valid_points))
            selected_depth = depth_raw[mask].astype(np.float64) * scale
            selected_depth = selected_depth[
                np.isfinite(selected_depth)
                & (selected_depth > contract.depth_range_m[0])
                & (selected_depth < contract.depth_range_m[1])
            ]
            if selected_depth.size < projector.minimum_valid_points:
                raise SupervisedV94RunError(
                    "selected object mask has too few valid depth pixels"
                )
            depth_values.append(selected_depth)
            digest.update(
                np.asarray(payload["camera_frame_id"][index], dtype=np.int64).tobytes()
            )
            digest.update(
                np.asarray(
                    payload["camera_timestamp_s"][index], dtype=np.float64
                ).tobytes()
            )
            digest.update(np.asarray(scale, dtype=np.float64).tobytes())
            digest.update(points.tobytes())
            digest.update(validity.tobytes())
            digest.update(
                np.asarray(point_frame.source_valid_points, dtype=np.int32).tobytes()
            )
        all_depth = np.concatenate(depth_values)
        depth_p50 = float(np.median(all_depth))
        elapsed = time.monotonic() - started
        artifact_hashes_after = tuple(file_sha256(path) for path in artifact_paths)
        if artifact_hashes_after != artifact_hashes_before:
            raise SupervisedV94RunError(
                "deployment bundle/point-cloud config/checkpoint changed "
                "during object ROI preflight"
            )
        preflight_sha256 = digest.hexdigest()
        if retain_live_provider:
            from sim2real.runtime.v94_live_observation_owner import (
                open_prevalidated_d435_handoff,
            )

            adopted_provider = provider
            provider = None
            handoff = open_prevalidated_d435_handoff(
                provider=adopted_provider,
                contract=contract,
                roi_xywh=resolved_roi,
                object_mask_mode=request.object_mask_mode,
                pcd_config_sha256=artifact_hashes_before[1],
                checkpoint_sha256=selected_checkpoint_sha256,
                preflight_sha256=preflight_sha256,
                last_preflight_frame_id=int(stored_frame_ids[-1]),
                last_preflight_timestamp_s=float(stored_timestamps[-1]),
                maximum_publication_stall_s=(
                    D435_FORMAL_PUBLICATION_STALL_S
                ),
            )
    finally:
        try:
            if provider is not None:
                provider.stop()
        finally:
            if compute_guard is not None:
                compute_guard.close()

    try:
        print(
            "[Object ROI preflight PASS] "
            f"xywh={list(resolved_roi)} "
            f"valid_frames={OBJECT_ROI_PREFLIGHT_VALID_FRAMES} "
            f"warmup_invalid_frames={invalid_publications} "
            f"retryable_camera_timeouts={retryable_camera_timeouts} "
            f"max_valid_mask_gap_ms={float(np.max(timestamp_deltas)) * 1000:.1f} "
            f"policy_rgbd={request.policy_rgbd_resolution} "
            f"min_policy_source_points={min(source_counts)} "
            f"depth_p50={depth_p50:.3f}m elapsed_s={elapsed:.3f}; "
            f"camera_session={'continuous' if handoff is not None else 'closed'}; "
            "Franka/RH56 interfaces opened=0",
            flush=True,
        )
        return replace(
            request,
            object_roi_xywh=resolved_roi,
            interactive_roi_gui_used=bool(request.select_object_roi),
            object_roi_source=(
                "text_grounding_camera_preflight"
                if request.object_text is not None
                else (
                    "operator_interactive_camera_preflight"
                    if request.select_object_roi
                    else "operator_numeric_camera_preflight"
                )
            ),
            object_roi_preflight_sha256=preflight_sha256,
            object_roi_preflight_valid_frames=OBJECT_ROI_PREFLIGHT_VALID_FRAMES,
            object_roi_preflight_invalid_frames=invalid_publications,
            object_roi_preflight_min_policy_points=min(source_counts),
            object_roi_preflight_depth_p50_m=depth_p50,
            object_roi_preflight_mask_bbox_xyxy=initialization_mask_bbox,
            object_roi_preflight_mask_area_px=initialization_mask_area,
            object_roi_preflight_mask_source=initialization_mask_source,
            object_roi_preflight_bundle_sha256=artifact_hashes_before[0],
            object_roi_preflight_pcd_config_sha256=artifact_hashes_before[1],
            object_roi_preflight_checkpoint_sha256=(selected_checkpoint_sha256),
            selected_checkpoint_sha256=selected_checkpoint_sha256,
            prewarmed_camera_handoff=handoff,
        )
    except BaseException:
        if handoff is not None:
            handoff.close()
        raise


def _build_persistent_online_sam2_manager(pcd_config: Path):
    """Build the parent-owned service without opening a camera or robot."""

    from dynamic_pcd.config import load_config
    from dynamic_pcd.segmentation.sam2_video_runtime import (
        SAM2VideoServiceManager,
        sam2_video_service_manager_kwargs_from_config,
    )

    config = load_config(str(pcd_config))
    online_cfg = dict(config.get("online_sam2", {}))
    if not bool(online_cfg.get("enabled", False)):
        return None
    if str(config.get("tracker", {}).get("mode", "roi_depth")) != (
        "adaptive_color_depth"
    ):
        return None
    return SAM2VideoServiceManager(
        **sam2_video_service_manager_kwargs_from_config(config)
    )


@contextmanager
def _persistent_online_sam2_service(pcd_config: Path) -> Iterator[object]:
    """Prewarm once and retain the service through all camera phases."""

    manager = _build_persistent_online_sam2_manager(pcd_config)
    if manager is None:
        yield None
        return
    print(
        "[Perception model PREPARING] warming SAM2; do not hold the target yet",
        flush=True,
    )
    try:
        health = manager.start()
    except BaseException:
        manager.close()
        raise
    print(
        "[Perception model READY] SAM2 will stay warm; show the target when "
        "the grounding preview opens",
        flush=True,
    )
    try:
        yield health
    finally:
        try:
            manager.reset()
        except Exception:
            pass
        manager.close()


def execute_deployment(request: DeploymentRequest) -> Mapping[str, object]:
    """Reset first, then acquire object evidence, then start policy control."""

    from .lease import (
        DeploymentLeaseError,
        acquire_v94_deployment_lease,
    )
    from sim2real.runtime.supervised_v94_runtime import (
        prepare_supervised_v94,
        prepare_supervised_v94_artifacts,
        run_supervised_v94,
    )
    from .execution_reset import run_v94_execution_reset

    # First pin the deployment and native ABI without consulting a historical
    # fixed-ROI proof.  This makes a stale protocol/build fail before D435/SAM
    # startup while still allowing --select-object-roi when the old fixed ROI
    # artifact is absent or intentionally irrelevant.
    artifact_preparation = prepare_supervised_v94_artifacts(request)
    print(
        "[Deployment preflight PASS] "
        f"checkpoint={artifact_preparation.checkpoint_source} "
        f"rate={request.policy_rate_hz:g}Hz steps={request.steps}",
        flush=True,
    )
    request = replace(
        request,
        selected_checkpoint_sha256=(artifact_preparation.checkpoint_sha256),
    )
    # This is intentionally before hardware lease/reset: a mobile irqbalance
    # owner or unpinned Franka NIC must fail without moving either device.
    request = _admit_deployment_compute_isolation(request)
    fingerprint_payload = {
        "run_id": request.run_id,
        "steps": int(request.steps),
        "bundle_sha256": artifact_preparation.bundle_sha256,
        "profile_file_sha256": artifact_preparation.profile_file_sha256,
        "pcd_config_sha256": artifact_preparation.pcd_config_sha256,
        "checkpoint_source": artifact_preparation.checkpoint_source,
        "checkpoint_path": (
            None
            if artifact_preparation.checkpoint_path is None
            else str(artifact_preparation.checkpoint_path)
        ),
        "checkpoint_sha256": artifact_preparation.checkpoint_sha256,
        "replay_action_path": (
            None
            if getattr(artifact_preparation, "replay_action_path", None) is None
            else str(artifact_preparation.replay_action_path)
        ),
        "replay_action_sha256": (
            getattr(artifact_preparation, "replay_action_sha256", "") or None
        ),
        "replay_action_count": (
            getattr(artifact_preparation, "replay_action_count", 0) or None
        ),
        "native_binary_sha256": artifact_preparation.native_build.binary_sha256,
        "franka_nic_irq_stability_token": (request.franka_nic_irq_stability_token),
        "live_visualization": bool(request.live_visualization),
        "live_visualization_rate_hz": float(request.live_visualization_rate_hz),
        "record_video_path": (
            None if request.record_video_path is None else str(request.record_video_path)
        ),
        "record_policy_io": bool(request.record_policy_io),
        "policy_io_path": (
            str(default_policy_io_path(request.run_id))
            if request.record_policy_io
            else None
        ),
        "object_roi_xywh": (
            None if request.object_roi_xywh is None else list(request.object_roi_xywh)
        ),
        "object_text": request.object_text,
        "object_mask_mode": request.object_mask_mode,
        "policy_rgbd_resolution": request.policy_rgbd_resolution,
        "object_roi_source": request.object_roi_source,
        "object_roi_preflight_sha256": request.object_roi_preflight_sha256,
        "object_roi_preflight_bundle_sha256": (
            request.object_roi_preflight_bundle_sha256
        ),
        "object_roi_preflight_pcd_config_sha256": (
            request.object_roi_preflight_pcd_config_sha256
        ),
        "object_roi_preflight_checkpoint_sha256": (
            request.object_roi_preflight_checkpoint_sha256
        ),
    }
    request_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    try:
        lease_context = acquire_v94_deployment_lease(
            run_id=request.run_id,
            runs_dir=artifact_preparation.output.parent,
            audit_path=artifact_preparation.output,
            request_fingerprint=request_fingerprint,
        )
    except (DeploymentLeaseError, OSError) as exc:
        raise SupervisedV94RunError(str(exc)) from exc

    with lease_context as lease:
        try:
            lease.mark_phase("perception_model_prewarm")
            with _persistent_online_sam2_service(request.pcd_config):
                lease.mark_phase("automatic_reset")
                execution_reset = run_v94_execution_reset(
                    profile=artifact_preparation.profile,
                    profile_path=request.profile,
                    bundle_path=request.bundle,
                    contract_q_home_rad=artifact_preparation.contract.q_home_rad,
                    maximum_franka_start_delta_rad=(FRANKA_MAX_EPISODE_DELTA_RAD),
                )
                # The current object is deliberately resolved only after both
                # devices reached reset and the reset helper completed its
                # dwell. This phase is camera-only and sends no Franka/RH56
                # command.
                lease.mark_phase("object_preflight")
                camera_handoff = None
                try:
                    request = _preflight_current_object_roi(
                        request,
                        retain_live_provider=True,
                    )
                    camera_handoff = request.prewarmed_camera_handoff
                    preparation = prepare_supervised_v94(request)
                    if not (
                        artifact_preparation.output == preparation.output
                        and artifact_preparation.bundle_sha256
                        == preparation.bundle_sha256
                        and artifact_preparation.profile_file_sha256
                        == preparation.profile_file_sha256
                        and artifact_preparation.pcd_config_sha256
                        == preparation.pcd_config_sha256
                        and artifact_preparation.checkpoint_source
                        == preparation.checkpoint_source
                        and artifact_preparation.checkpoint_path
                        == preparation.checkpoint_path
                        and artifact_preparation.checkpoint_sha256
                        == preparation.checkpoint_sha256
                        and artifact_preparation.native_build
                        == preparation.native_build
                        and artifact_preparation.pinned_checkpoint_bytes
                        == preparation.pinned_checkpoint_bytes
                        and getattr(
                            artifact_preparation, "replay_action_path", None
                        )
                        == getattr(preparation, "replay_action_path", None)
                        and getattr(
                            artifact_preparation, "replay_action_sha256", ""
                        )
                        == getattr(preparation, "replay_action_sha256", "")
                        and getattr(
                            artifact_preparation, "replay_action_count", 0
                        )
                        == getattr(preparation, "replay_action_count", 0)
                        and getattr(
                            artifact_preparation,
                            "pinned_replay_action_bytes",
                            None,
                        )
                        == getattr(
                            preparation,
                            "pinned_replay_action_bytes",
                            None,
                        )
                    ):
                        raise SupervisedV94RunError(
                            "deployment artifact/native identity changed during "
                            "post-reset camera-only object preflight"
                        )
                    lease.mark_phase("policy_runtime")
                    with _deployment_compute_isolation(
                        request
                    ) as isolated_request:
                        result = run_supervised_v94(
                            isolated_request,
                            preparation=preparation,
                            execution_reset=execution_reset,
                            prewarmed_camera_handoff=camera_handoff,
                        )
                    lease.finalize(result="PASS")
                    return result
                finally:
                    if camera_handoff is not None:
                        camera_handoff.close()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            raise SupervisedV94ExecutionError(
                f"real execution failed after hardware lease acquisition: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    real_runner: Callable[
        [DeploymentRequest], Mapping[str, object]
    ] = execute_deployment,
) -> int:
    try:
        args = build_parser().parse_args(argv)
        request = build_deployment_request(args)
        summary = dict(build_deployment_summary(request))
        if not request.execute:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        from sim2real.console_output import (
            compact_deployment_console,
            emit_operator_line,
        )

        verbose_console = bool(args.verbose_console)
        if verbose_console:
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        else:
            checkpoint_label = (
                str(request.checkpoint)
                if request.checkpoint is not None
                else "bundle primary checkpoint"
            )
            emit_operator_line(
                "[Deployment START] "
                f"run={request.run_id} rate={request.policy_rate_hz:g}Hz "
                f"steps={request.steps} checkpoint={checkpoint_label}"
            )
        with compact_deployment_console(enabled=not verbose_console):
            result = dict(real_runner(request))
        if verbose_console:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            completed = int(result.get("completed_policy_steps", 0))
            if result.get("action_source") == "online_visual_intercept_planner":
                completed_frames = int(
                    result.get("completed_replay_frames", 0)
                )
                progress = (
                    f"templates={completed_frames}/{request.steps} "
                    f"hardware_commands={completed}"
                )
            else:
                progress = f"completed={completed}/{request.steps}"
            emit_operator_line(
                "[Rollout PASS] "
                f"{progress} "
                f"franka_stop={bool(result.get('franka_stop_verified', False))} "
                f"rh56_stop={bool(result.get('rh56_disabled_verified', False))}"
            )
        return 0
    except (ValueError, OSError, SupervisedV94RunError) as exc:
        try:
            from sim2real.console_output import emit_operator_line

            emit_operator_line(f"V94 deployment: REFUSED: {exc}", error=True)
        except ImportError:
            print(f"V94 deployment: REFUSED: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        from sim2real.console_output import emit_operator_line

        emit_operator_line(
            "V94 deployment: interrupted; verified stop was requested",
            error=True,
        )
        return 130
    except SystemExit:
        # Preserve argparse's standard exit contract for --help.  In
        # particular, do not relabel the expected exit(0) as a runtime fault.
        raise
    except BaseException as exc:
        from sim2real.console_output import emit_operator_line

        emit_operator_line(
            f"V94 deployment: FAILED: {type(exc).__name__}: {exc}",
            error=True,
        )
        return 1


# Compatibility aliases for existing scripts and recorded test procedures.
# New code should use the explicit deployment-oriented names above.
SupervisedRequest = DeploymentRequest
_request = build_deployment_request
_summary = build_deployment_summary
_run_real = execute_deployment


if __name__ == "__main__":
    raise SystemExit(main())
