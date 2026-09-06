"""Production wiring for the bounded, operator-supervised V94 experiment.

The public CLI requires the explicit ``--execute`` and
``--yes-i-am-supervising`` flags.  This module still performs all offline
mapping checks and fresh read-only hardware preflights before arming either
actuator.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, is_dataclass
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import threading
import time
from typing import Any, Mapping, Optional, Sequence
import uuid

import numpy as np

from sim2real.deployment.franka_action_audit import audit_v94_franka_action_mapping
from sim2real.deployment.rh56_action_audit import audit_v94_rh56_action_mapping
from .bounded_c2_runtime import (
    SupervisedV94RuntimeFactory,
    _seal_supervised_v94_admission,
)
from sim2real.closed_loop_core import ClosedLoopSafetyGate, MotionAuthorization, SafetyState
from sim2real.deployment.bundle import (
    MAX_CHECKPOINT_BYTES,
    DeployBundle,
    load_checkpoint_safely,
)
from robot_control.franka.session import (
    FrankaPersistentSession,
    _has_active_errors,
    _issue_supervised_franka_preflight_token,
    _require_clear_flags,
    _robot_mode_name,
    _state_vectors,
    load_experimental_supervised_franka_envelope,
)
from robot_control.franka.backend import (
    AUDITED_PYLIBFRANKA_VERSION,
)
from robot_control.franka.native_session import (
    FrankaNativeSupervisedSessionProxy,
    NativeHelloExpectation,
    PopenSeqpacketChildLauncher,
    V94NativePODCodec,
    V94NativeControllerMode,
)
from sim2real.policy import ActionControllerParameters, RollingStudentPolicy
from sim2real.policy.io_recorder import PolicyIORecorder, default_policy_io_path
from sim2real.policy.rate_mode import (
    DEFAULT_POLICY_RATE_MODE,
    resolve_policy_rate_mode,
)
from robot_control.rh56.linux_transport import LinuxRH56TransactionalTransport
from sim2real.rh56_profile_contract import (
    load_commissioned_rh56_force_set_g,
    load_v94_rh56_profile_command_bounds,
)
from robot_control.rh56.actuator import (
    RH56ActuatorState,
    RH56_IDLE_STATUSES,
    RH56TransactionalActuator,
    _issue_supervised_rh56_preflight,
)
from robot_control.rh56.watchdog import RH56OwnedSession, RH56WatchdogOwner
from sim2real.observation.camera_profile import (
    resolve_runtime_task_contract,
    task_profile_maximum_supervised_execute_steps,
    task_profile_rollout_trigger,
)
from sim2real.sim_control_alignment import (
    RH56_SPEED_SET_REGISTER_ORDER as RH56_SPEED_SET,
)
from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13, V94Contract
from sim2real.contracts.actions import (
    LEGACY_FRANKA_ACTION_CONTRACT_ID,
    QD_G015_FRANKA_ACTION_CONTRACT_ID,
)
from sim2real.observation.model import infer_fixed_sphere_radius_m
from .v94_policy_tick_source import QD_G015_STARTUP_NON_ACTUATED_POLICY_STEPS
from .v94_live_observation_owner import (
    D435ObjectCameraOwner,
    LiveD435ProviderFactory,
    PrewarmedD435CameraHandoff,
    ProductionV94PolicyTickSourceFactory,
)
from sim2real.observation.visualization import (
    V94LiveVisualizer,
    mask_video_path_for_recording,
)
from sim2real.deployment.execution_reset import (
    RESET_MAX_FRANKA_START_DELTA_RAD,
    V94ExecutionResetError,
    V94ExecutionResetProof,
    run_v94_execution_reset,
    validate_v94_execution_reset_proof,
)
from sim2real.deployment.verify import verify_v94_bundle, verify_v94_checkpoint_payload

SHADOW21 = (
    Path(__file__).resolve().parents[2]
    / "dexgrasp/runs/v94_live_readonly_clean_dkms_power_on_reuse_guard_20260722_21.npz"
)
VERIFIED_OBJECT_ROI_EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "dexgrasp/runs/v94_provider_soak_current_roi_morph5_20260723.npz"
)
VERIFIED_OBJECT_ROI_EVIDENCE_SHA256 = (
    "9148fb6bd76d0f88be7ce02ea6380a51a11406034a905c20bce00de10e5c253d"
)
VERIFIED_OBJECT_ROI_XYWH = (460, 235, 120, 100)
VERIFIED_OBJECT_ROI_MIN_INIT_MARGIN_PX = 8
VERIFIED_OBJECT_ROI_MIN_CAMERA_SPAN_S = 60.0
VERIFIED_OBJECT_ROI_MIN_PROVIDER_STEPS = 1801
VERIFIED_OBJECT_ROI_SAMPLE_EVERY = 30
CURRENT_OBJECT_ROI_PREFLIGHT_MAX_INVALID_FRAMES = 12
CURRENT_OBJECT_ROI_SAM2_MASK_SOURCES = frozenset(
    {"sam2", "online_sam2_box"}
)
CURRENT_OBJECT_ROI_SAM2_MIN_IMAGE_MARGIN_PX = 2
CURRENT_OBJECT_ROI_SAM2_MAX_MASK_AREA_RATIO = 4.0
CURRENT_OBJECT_ROI_SAM2_MAX_BBOX_AREA_RATIO = 6.0
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,80}$")
POLICY_RATE_HZ = DEFAULT_POLICY_RATE_MODE.policy_rate_hz
FRANKA_MAX_MEASURED_VELOCITY_RAD_S = 0.70
# The native state ring publishes at 62.5 Hz (one sample every 16 ms).  A
# 25 ms action gate needlessly rejects a healthy sample when capture alignment
# and inference straddle the next publication.  The selected 20 Hz checkpoint
# was trained with 0--50 ms proprioception latency, so accepting a latest state
# up to 40 ms old removes release jitter without exceeding its training range.
FRANKA_OBSERVATION_ACTION_MAX_AGE_S = 0.040
FRANKA_OBSERVATION_HARD_MAX_AGE_S = 0.050
D435_RUNTIME_FRAME_TIMEOUT_MS = 100
D435_FORMAL_PUBLICATION_STALL_S = 0.400
# Online SAM2 and point-cloud projection add processing latency after capture.
# Accept a still-live, exact-frame cloud throughout the same bounded window as
# the camera publisher; frame reuse and publication-stall guards independently
# prevent a frozen camera stream from driving new actions indefinitely.
OBJECT_POINTCLOUD_MAX_AGE_S = 0.200
# A side-effect-free observation retry must terminate with its real perception
# reason well before either 500 ms actuator inter-command watchdog.  The same
# 200 ms absolute budget covers both a stalled publisher and progressing camera
# frames whose masks remain unusable, leaving roughly 280 ms after one 60 Hz
# tick for the verified RH56 disable path (whose own I/O deadline is 250 ms).
MAXIMUM_CONSECUTIVE_NO_STAGE_HOLD_S = 0.400
AUTHORIZATION_LIFETIME_S = 60.0
HARD_RUNTIME_DEADLINE_S = 45.0
RH56_WATCHDOG_S = 0.050
# A recoverable observation/USB scheduling gap holds the last already-bounded
# target.  This is deliberately separate from the unchanged 50 ms freshness
# deadline for every newly produced hand command.
RH56_INTER_COMMAND_WATCHDOG_S = 0.500
RH56_IO_DEADLINE_S = 0.25
# Keep the proven 20 Hz latest-only transport schedule.  The selected 20/60 Hz
# policy mode derives the matching one- or three-tick register envelope so an
# RH56 release never adds a second, slower trajectory.
# Installed q6 evidence shows that
# STATUS may trail an already-motionless ANGLE_ACT, and a completed 250-step
# run showed that a hand still physically tracking its final target needed
# more than 2.5 seconds to reach the entirely post-disable STATUS=2 tail.
# Preserve the strict idle/current/drift proof and give that one normal stop
# transaction a bounded five-second budget.
RH56_STOP_TIMEOUT_S = 5.00
RH56_FORCE_SET_G = 80
# Host-side REG_CURRENT feedback trip only.  Match the installed hand's
# independent 1400 mA/axis firmware CURRENT_LIMIT instead of imposing a second,
# lower software trip.  This code never writes that register; ERROR/STATUS,
# temperature, firmware limiting and verified stop remain fail-closed.
RH56_MAX_RUNNING_CURRENT_MA = 1400
# Stopping first replaces the moving target with the measured pose.  The
# installed hand has shown a brief 411--429 mA braking/hold transient here
# even though the completed policy trajectory and final disabled state were
# healthy.  Keep the stop-transition gate independently named even though it
# currently shares the 1000 mA bound with the active-motion monitor; the strict
# 100 mA post-disable idle proof remains separate.
RH56_STOP_SETTLE_MAX_AXIS_CURRENT_MA = 1000
# ANGLE_SET target transactions and complete safety-feedback polls share the
# same 115200-baud owner.  At 20 Hz each, their compact on-wire payload is
# 20 * (58 + 100) * 10 = 31,600 bit/s (27.4% of line rate), so the owner can
# serialize both without overlapping transport access.
RH56_FEEDBACK_RATE_HZ = 20.0
RH56_FEEDBACK_POLL_TRIGGER_S = 1.0 / RH56_FEEDBACK_RATE_HZ
# Keep three nominal feedback periods of jitter tolerance.  This is a hard
# fault age, not the nominal sample period used by the observation pipeline.
RH56_FEEDBACK_HARD_AGE_S = 0.150
RH56_FEEDBACK_TO_COMMAND_OFFSET_UNITS = (0, 0, 0, 0, 0, 15)
# Five bend axes retain the profile's 25-unit open tolerance.  Use the reset
# driver's commissioned 30-unit q6 endpoint tolerance here as well: installed
# q6 repeatedly settles at 973 after ANGLE_SET=1000 is released while status,
# error, current and two-sample drift all prove a healthy disabled idle state.
# Numeric commands and active tracking bounds remain unchanged.
RH56_DISABLED_OPEN_TOLERANCE_UNITS = (25, 25, 25, 25, 25, 30)
# Installed-hand evidence shows a roughly 30--34 unit open-position deadband:
# status/current can indicate an active drive while ANGLE_ACT/POS_ACT remain
# stationary.  Treat only >=5% travel as a physical-motion challenge.  Larger
# ring/middle/thumb commands are still required to show measured progress.
RH56_TRACKING_SIGNIFICANT_GAP_UNITS = 50
RH56_TRACKING_MIN_PROGRESS_UNITS = 3
RH56_TRACKING_CONTACT_FORCE_DELTA_G = 150
RH56_TRACKING_CONTACT_FORCE_ABSOLUTE_G = 200
RH56_TRACKING_TIMEOUT_S = 0.750
# A command failure performs the complete bounded stop transaction before the
# owner ticket receives the underlying exception.  Keep the caller-side wait
# longer than the command watchdog plus that cleanup so a generic ticket
# timeout cannot hide the serial/transaction error that caused the stop.
RH56_OWNER_RESPONSE_TIMEOUT_S = 5.75
# A first normal stop may consume five seconds; if it faults, the owner still
# gets one complete fail-safe disable attempt before its thread is joined.
RH56_OWNER_JOIN_TIMEOUT_S = 11.0
NATIVE_SERVO_BUILD_DIR = (
    Path(__file__).resolve().parents[2] / "dexgrasp/native/v94_franka_servo/build"
)
NATIVE_SERVO_MANIFEST = NATIVE_SERVO_BUILD_DIR / "manifest.json"
NATIVE_SERVO_TABLETOP_BUILD_DIR = (
    NATIVE_SERVO_BUILD_DIR.parent / "build_v94_tabletop"
)
NATIVE_SERVO_TABLETOP_MANIFEST = (
    NATIVE_SERVO_TABLETOP_BUILD_DIR / "manifest.json"
)
TABLETOP_PROFILE_ID = "fr3_rh56_v94_reset_locked_v1"
NATIVE_SERVO_V60_BUILD_DIR = (
    NATIVE_SERVO_BUILD_DIR.parent / "build_v60_palmcatch"
)
NATIVE_SERVO_V60_MANIFEST = NATIVE_SERVO_V60_BUILD_DIR / "manifest.json"
V60_PALMCATCH_PROFILE_ID = "fr3_rh56_v60_palmcatch_single_tick_v1"
NATIVE_SERVO_V61_BUILD_DIR = (
    NATIVE_SERVO_BUILD_DIR.parent / "build_v61_sixexpert"
)
NATIVE_SERVO_V61_MANIFEST = NATIVE_SERVO_V61_BUILD_DIR / "manifest.json"
V61_SIXEXPERT_PROFILE_ID = "fr3_rh56_v61_sixexpert_40tick_v1"
NATIVE_SERVO_SOURCE_DIR = NATIVE_SERVO_BUILD_DIR.parent
NATIVE_SERVO_CONTRACT_HEADER = (
    NATIVE_SERVO_SOURCE_DIR / "include/anydex/v94_franka_servo/safety_limits.hpp"
)
NATIVE_SERVO_TABLETOP_CONTRACT_HEADER = (
    NATIVE_SERVO_SOURCE_DIR
    / "profiles/v94_tabletop/include/anydex/v94_franka_servo/safety_limits.hpp"
)
NATIVE_SERVO_V60_CONTRACT_HEADER = (
    NATIVE_SERVO_SOURCE_DIR
    / "profiles/v60_palmcatch/include/anydex/v94_franka_servo/safety_limits.hpp"
)
NATIVE_SERVO_V61_CONTRACT_HEADER = (
    NATIVE_SERVO_SOURCE_DIR
    / "profiles/v61_sixexpert/include/anydex/v94_franka_servo/safety_limits.hpp"
)
NATIVE_SERVO_PRODUCER_SOURCES = (
    "include/anydex/v94_franka_servo/backend.hpp",
    "include/anydex/v94_franka_servo/channel.hpp",
    "include/anydex/v94_franka_servo/dependency_identity.hpp",
    "include/anydex/v94_franka_servo/franka_v225_interpolator.hpp",
    "include/anydex/v94_franka_servo/libfranka_backend.hpp",
    "include/anydex/v94_franka_servo/protocol.hpp",
    "include/anydex/v94_franka_servo/safety_limits.hpp",
    "include/anydex/v94_franka_servo/servo_core.hpp",
    "src/channel.cpp",
    "src/dependency_identity.cpp",
    "src/libfranka_backend.cpp",
    "src/main.cpp",
    "src/protocol.cpp",
    "src/servo_core.cpp",
)
EXPECTED_NATIVE_LIBFRANKA_SHA256 = (
    "956d2f7e85e3c4e127899734a170dff7c91f17f560a4fa2147739631ad721a3d"
)
EXPECTED_NATIVE_LIBFRANKA_COMMIT = "9f9304ec0ac897eff3219a67f612b959948535e2"


class SupervisedV94RuntimeError(RuntimeError):
    pass


def _prefer_terminal_owner_fault(
    failure: Optional[BaseException], runtime: Any
) -> Optional[BaseException]:
    """Replace a raced parent-side symptom with its actuator root fault.

    After the native Franka reader faults, its last state naturally becomes
    stale and an in-flight dual-device commit can also time out.  Cleanup joins
    that reader, so its frozen terminal telemetry is authoritative by the time
    this helper runs.  A normal parent-requested stop is not a hardware root
    fault and must not replace an independent RH56/perception failure.
    """

    if failure is None or isinstance(failure, (KeyboardInterrupt, SystemExit)):
        return failure
    telemetry = getattr(runtime, "franka_telemetry", None)
    franka_fault = getattr(telemetry, "fault_reason", None)
    if (
        isinstance(franka_fault, str)
        and franka_fault.strip()
        and franka_fault.strip() != "parent requested supervised stop"
    ):
        return SupervisedV94RuntimeError(
            "Franka owner fault preceded the parent-side runtime symptom: "
            + franka_fault.strip()
        )
    return failure


@dataclass(frozen=True)
class NativeServoBuildIdentity:
    manifest_path: Path
    manifest_sha256: str
    binary_path: Path
    binary_sha256: str
    producer_build_sha256: str
    libfranka_path: Path
    libfranka_sha256: str
    libfranka_source_commit: str
    protocol_version: int
    state_decimation: int
    safety_limits_schema: int
    compiled_profile_sha256: str
    compiled_envelope_sha256: str


@dataclass(frozen=True)
class FreshHardwarePreflight:
    franka_q_rad: tuple[float, ...]
    franka_dq_rad_s: tuple[float, ...]
    franka_qhome_linf_rad: float
    franka_static_provenance_verified: bool
    rh56_angle_set: tuple[int, ...]
    rh56_angle_act: tuple[int, ...]
    rh56_max_abs_current_ma: int


@dataclass(frozen=True)
class VerifiedObjectROI:
    """Pinned provider proof plus the numeric prompt used for this run."""

    xywh: tuple[int, int, int, int]
    evidence_path: Optional[str]
    evidence_sha256: str
    camera_serial: str
    calibration_id: str
    checkpoint_sha256: str
    valid_publications: int
    invalid_publications: int
    acquisition_elapsed_s: float
    position_source: str = "pinned_fixed_roi"
    preflight_sha256: str = ""
    preflight_valid_frames: int = 0
    preflight_invalid_frames: int = 0
    preflight_min_policy_points: int = 0
    preflight_depth_p50_m: Optional[float] = None
    preflight_mask_bbox_xyxy: tuple[int, ...] = ()
    preflight_mask_area_px: int = 0
    preflight_mask_source: str = ""
    preflight_bundle_sha256: str = ""
    preflight_pcd_config_sha256: str = ""


@dataclass(frozen=True)
class PreparedSupervisedV94Artifacts:
    """Artifact/native identity that has no object-ROI evidence dependency."""

    output: Path
    profile: Mapping[str, Any]
    contract: V94Contract
    envelope: Any
    native_build: NativeServoBuildIdentity
    bundle_sha256: str
    profile_file_sha256: str
    pcd_config_sha256: str
    checkpoint_source: str
    checkpoint_path: Optional[Path]
    checkpoint_sha256: str
    pinned_checkpoint_bytes: bytes = field(repr=False)
    replay_action_path: Optional[Path] = None
    replay_action_sha256: str = ""
    replay_action_count: int = 0
    pinned_replay_action_bytes: Optional[bytes] = field(default=None, repr=False)


@dataclass(frozen=True)
class PreparedSupervisedV94:
    """Hardware-free inputs pinned before policy control begins."""

    output: Path
    profile: Mapping[str, Any]
    contract: V94Contract
    envelope: Any
    object_roi: VerifiedObjectROI
    native_build: NativeServoBuildIdentity
    bundle_sha256: str
    profile_file_sha256: str
    pcd_config_sha256: str
    checkpoint_source: str
    checkpoint_path: Optional[Path]
    checkpoint_sha256: str
    pinned_checkpoint_bytes: bytes = field(repr=False)
    replay_action_path: Optional[Path] = None
    replay_action_sha256: str = ""
    replay_action_count: int = 0
    pinned_replay_action_bytes: Optional[bytes] = field(default=None, repr=False)


class _OperatorSupervisedBoundary:
    """Manual Ctrl+C supervision, explicitly not a physical interlock claim."""

    physical_interlocks_configured = False
    independent_command_watchdogs_configured = True
    verified_dual_device_stop_supported = True
    operator_supervision_confirmed = True

    def require_motion(self, admission: Any, *, now_monotonic_s: float) -> None:
        if admission.classification != "experimental_operator_supervised_non_c2":
            raise SupervisedV94RuntimeError(
                "supervised admission classification changed"
            )
        admission.require_active(now_monotonic_s=now_monotonic_s)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_servo_producer_build_sha256(
    safety_profile: str = "v94",
) -> str:
    """Recompute the exact CMake producer digest without executing a binary."""

    material = bytearray()
    sources = NATIVE_SERVO_PRODUCER_SOURCES
    if safety_profile in {"v94_tabletop", "v60_palmcatch", "v61_sixexpert"}:
        insertion = sources.index("src/channel.cpp")
        profile_directory = safety_profile
        sources = (
            *sources[:insertion],
            f"profiles/{profile_directory}/include/anydex/"
            "v94_franka_servo/safety_limits.hpp",
            *sources[insertion:],
        )
    elif safety_profile != "v94":
        raise SupervisedV94RuntimeError("unknown native Franka safety profile")
    for relative in sources:
        source = NATIVE_SERVO_SOURCE_DIR / relative
        if not source.is_file():
            raise SupervisedV94RuntimeError(
                f"native Franka producer source is missing: {source}"
            )
        material.extend(f"{relative}:{_sha256(source)}\n".encode("ascii"))
    return hashlib.sha256(bytes(material)).hexdigest()


def _native_servo_compiled_contract_digests(
    safety_profile: str = "v94",
) -> tuple[str, str]:
    """Read source constants proven to match the manifest producer digest."""

    try:
        header = {
            "v94": NATIVE_SERVO_CONTRACT_HEADER,
            "v94_tabletop": NATIVE_SERVO_TABLETOP_CONTRACT_HEADER,
            "v60_palmcatch": NATIVE_SERVO_V60_CONTRACT_HEADER,
            "v61_sixexpert": NATIVE_SERVO_V61_CONTRACT_HEADER,
        }[safety_profile]
        text = header.read_text(encoding="utf-8", errors="strict")
    except BaseException as exc:
        raise SupervisedV94RuntimeError(
            f"native Franka contract header is unreadable: {exc}"
        ) from exc

    def one(name: str) -> str:
        match = re.search(
            rf"\b{name}\s*=\s*\n?\s*\"([0-9a-f]{{64}})\"\s*;",
            text,
        )
        if match is None:
            raise SupervisedV94RuntimeError(
                f"native Franka contract header has no unique {name}"
            )
        return match.group(1)

    if safety_profile not in {
        "v94",
        "v94_tabletop",
        "v60_palmcatch",
        "v61_sixexpert",
    }:
        raise SupervisedV94RuntimeError("unknown native Franka safety profile")
    return one("kProfileSha256"), one("kEnvelopeSha256")


def _load_native_servo_build_identity(
    manifest_path: Path = NATIVE_SERVO_MANIFEST,
) -> NativeServoBuildIdentity:
    """Validate the exact native executable/dependency set without devices."""

    source = manifest_path.expanduser().resolve()
    variants = {
        NATIVE_SERVO_MANIFEST.resolve(): (NATIVE_SERVO_BUILD_DIR, "v94"),
        NATIVE_SERVO_TABLETOP_MANIFEST.resolve(): (
            NATIVE_SERVO_TABLETOP_BUILD_DIR,
            "v94_tabletop",
        ),
        NATIVE_SERVO_V60_MANIFEST.resolve(): (
            NATIVE_SERVO_V60_BUILD_DIR,
            "v60_palmcatch",
        ),
        NATIVE_SERVO_V61_MANIFEST.resolve(): (
            NATIVE_SERVO_V61_BUILD_DIR,
            "v61_sixexpert",
        ),
    }
    if source not in variants or not source.is_file():
        raise SupervisedV94RuntimeError(
            f"native Franka build manifest is missing: {source}"
        )
    selected_build_dir, safety_profile = variants[source]

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate manifest key: {key}")
            result[key] = value
        return result

    try:
        manifest_bytes = source.read_bytes()
        payload = json.loads(
            manifest_bytes.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except BaseException as exc:
        raise SupervisedV94RuntimeError(
            f"native Franka build manifest is invalid: {type(exc).__name__}: {exc}"
        ) from exc
    required = {
        "schema_version",
        "binary_path",
        "binary_sha256",
        "producer_build_sha256",
        "libfranka_path",
        "libfranka_sha256",
        "libfranka_source_commit",
        "protocol_version",
        "state_decimation",
        "safety_limits_schema",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise SupervisedV94RuntimeError(
            "native Franka build manifest fields differ from schema v1"
        )
    for name, expected in (
        ("schema_version", 1),
        ("protocol_version", 4),
        ("state_decimation", 16),
        ("safety_limits_schema", 3),
    ):
        value = payload[name]
        if isinstance(value, bool) or value != expected:
            raise SupervisedV94RuntimeError(
                f"native Franka manifest {name} differs from {expected}"
            )

    def digest(name: str, length: int) -> str:
        value = str(payload[name]).strip().lower()
        if len(value) != length or re.fullmatch(r"[0-9a-f]+", value) is None:
            raise SupervisedV94RuntimeError(
                f"native Franka manifest {name} is not a {length}-digit hex digest"
            )
        return value

    binary_sha256 = digest("binary_sha256", 64)
    producer_build_sha256 = digest("producer_build_sha256", 64)
    libfranka_sha256 = digest("libfranka_sha256", 64)
    source_commit = digest("libfranka_source_commit", 40)
    current_producer_build_sha256 = _native_servo_producer_build_sha256(
        safety_profile
    )
    if producer_build_sha256 != current_producer_build_sha256:
        raise SupervisedV94RuntimeError(
            "native Franka executable is stale relative to its producer "
            "sources; rebuild it before hardware access"
        )
    compiled_profile_sha256, compiled_envelope_sha256 = (
        _native_servo_compiled_contract_digests(safety_profile)
    )
    if libfranka_sha256 != EXPECTED_NATIVE_LIBFRANKA_SHA256:
        raise SupervisedV94RuntimeError(
            "native Franka manifest names a different libfranka binary"
        )
    if source_commit != EXPECTED_NATIVE_LIBFRANKA_COMMIT:
        raise SupervisedV94RuntimeError(
            "native Franka manifest names a different libfranka source commit"
        )

    binary_value = Path(str(payload["binary_path"]))
    binary_path = (
        binary_value if binary_value.is_absolute() else source.parent / binary_value
    ).resolve()
    expected_binary = (selected_build_dir / "v94_franka_servo").resolve()
    if binary_path != expected_binary or not binary_path.is_file():
        raise SupervisedV94RuntimeError(
            "native Franka executable path is missing or outside the pinned build path"
        )
    if not os.access(binary_path, os.X_OK) or _sha256(binary_path) != binary_sha256:
        raise SupervisedV94RuntimeError(
            "native Franka executable permission/SHA-256 verification failed"
        )
    # The manifest binds the complete executable bytes, while these two
    # selected strings prove that those bytes contain the expected compiled
    # ARM contract rather than the other supported safety-profile variant.
    binary_bytes = binary_path.read_bytes()
    for label, compiled_digest in (
        ("profile", compiled_profile_sha256),
        ("envelope", compiled_envelope_sha256),
    ):
        if binary_bytes.count(compiled_digest.encode("ascii")) != 1:
            raise SupervisedV94RuntimeError(
                f"native Franka executable does not uniquely embed its {label} digest"
            )

    lib_value = Path(str(payload["libfranka_path"]))
    lib_path = (
        lib_value if lib_value.is_absolute() else source.parent / lib_value
    ).resolve()
    expected_lib = (
        Path(__file__).resolve().parents[2]
        / ".venv/lib/python3.9/site-packages/pylibfranka.libs/"
        "libfranka-2be07f70.so.0.21.2"
    ).resolve()
    if lib_path != expected_lib or not lib_path.is_file():
        raise SupervisedV94RuntimeError(
            "native Franka manifest libfranka path differs from the installed wheel"
        )
    if _sha256(lib_path) != libfranka_sha256:
        raise SupervisedV94RuntimeError("installed native libfranka SHA-256 changed")

    return NativeServoBuildIdentity(
        manifest_path=source,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        binary_path=binary_path,
        binary_sha256=binary_sha256,
        producer_build_sha256=producer_build_sha256,
        libfranka_path=lib_path,
        libfranka_sha256=libfranka_sha256,
        libfranka_source_commit=source_commit,
        protocol_version=4,
        state_decimation=16,
        safety_limits_schema=3,
        compiled_profile_sha256=compiled_profile_sha256,
        compiled_envelope_sha256=compiled_envelope_sha256,
    )


def _open_verified_native_executable_fd(build: NativeServoBuildIdentity) -> int:
    """Open and hash the exact inode that the launcher will execute."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(build.binary_path, flags)
    except OSError as exc:
        raise SupervisedV94RuntimeError(
            f"cannot pin native Franka executable: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SupervisedV94RuntimeError(
                "native Franka executable is not a regular file"
            )
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != build.binary_sha256:
            raise SupervisedV94RuntimeError(
                "native Franka executable changed after permit validation"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _load_verified_object_roi(
    *,
    bundle_path: Path,
    contract: V94Contract,
    selected_checkpoint_sha256: str,
) -> VerifiedObjectROI:
    """Load the pinned headless ROI evidence before any hardware preflight.

    The source run used this exact camera/calibration/checkpoint for more than
    60 seconds of *camera timestamps* and published 1801 consecutive valid
    masks with no invalid publication.  Pinning the whole NPZ hash prevents a
    mutable run artifact from silently changing the ROI or its provenance.
    """

    source = VERIFIED_OBJECT_ROI_EVIDENCE.resolve()
    if not source.is_file():
        raise SupervisedV94RuntimeError(
            f"verified fixed-ROI evidence is missing: {source}"
        )
    evidence_sha256 = _sha256(source)
    if evidence_sha256 != VERIFIED_OBJECT_ROI_EVIDENCE_SHA256:
        raise SupervisedV94RuntimeError("verified fixed-ROI evidence SHA-256 changed")

    required = {
        "capture_schema_version",
        "hardware_writes",
        "robot_command_writes",
        "camera_serial",
        "calibration_id",
        "bundle_contract",
        "checkpoint_sha256",
        "roi_xywh",
        "valid_publications",
        "invalid_publications",
        "acquisition_elapsed_s",
        "online_sam2_runtime_mode",
        "franka_interface_opened",
        "rh56_interface_opened",
        "disk_io_during_acquisition",
        "configured_compute_threads",
        "sample_every_valid_frame",
        "provider_steps",
        "camera_timestamp_s",
        "initialization_prompt_bbox_xyxy",
        "initialization_mask_bbox_xyxy",
        "initialization_mask_area_px",
        "initialization_mask_source",
    }
    try:
        with np.load(source, allow_pickle=False) as evidence:
            missing = sorted(required.difference(evidence.files))
            if missing:
                raise SupervisedV94RuntimeError(
                    "fixed-ROI evidence fields are missing: " + ", ".join(missing)
                )

            def scalar(name: str) -> Any:
                value = np.asarray(evidence[name])
                if value.shape != ():
                    raise SupervisedV94RuntimeError(
                        f"fixed-ROI evidence {name} must be scalar"
                    )
                return value.item()

            schema = int(scalar("capture_schema_version"))
            hardware_writes = bool(scalar("hardware_writes"))
            robot_writes = bool(scalar("robot_command_writes"))
            camera_serial = str(scalar("camera_serial"))
            calibration_id = str(scalar("calibration_id"))
            bundle_contract = str(scalar("bundle_contract"))
            checkpoint_sha256 = str(scalar("checkpoint_sha256"))
            roi_values = np.asarray(evidence["roi_xywh"])
            valid_publications = int(scalar("valid_publications"))
            invalid_publications = int(scalar("invalid_publications"))
            acquisition_elapsed_s = float(scalar("acquisition_elapsed_s"))
            sam2_mode = str(scalar("online_sam2_runtime_mode"))
            franka_opened = bool(scalar("franka_interface_opened"))
            rh56_opened = bool(scalar("rh56_interface_opened"))
            disk_io_during_acquisition = bool(scalar("disk_io_during_acquisition"))
            configured_compute_threads = int(scalar("configured_compute_threads"))
            sample_every = int(scalar("sample_every_valid_frame"))
            provider_steps = int(scalar("provider_steps"))
            camera_timestamps_s = np.asarray(
                evidence["camera_timestamp_s"], dtype=np.float64
            )
            init_prompt_values = np.asarray(evidence["initialization_prompt_bbox_xyxy"])
            init_mask_values = np.asarray(evidence["initialization_mask_bbox_xyxy"])
            init_mask_area = int(scalar("initialization_mask_area_px"))
            init_mask_source = str(scalar("initialization_mask_source"))
    except SupervisedV94RuntimeError:
        raise
    except BaseException as exc:
        raise SupervisedV94RuntimeError(
            f"could not read fixed-ROI evidence: {type(exc).__name__}: {exc}"
        ) from exc

    bundle = DeployBundle(bundle_path)
    bundle.verify()
    expected_contract = str(bundle.manifest.get("bundle_contract", ""))
    expected_checkpoint = str(selected_checkpoint_sha256).strip().lower()
    if (
        schema != 1
        or hardware_writes
        or robot_writes
        or franka_opened
        or rh56_opened
        or disk_io_during_acquisition
    ):
        raise SupervisedV94RuntimeError(
            "fixed-ROI evidence is not an isolated read-only schema-v1 capture"
        )
    if camera_serial != contract.camera_serial:
        raise SupervisedV94RuntimeError(
            "fixed-ROI evidence camera serial differs from the V94 contract"
        )
    if calibration_id != contract.calibration_id:
        raise SupervisedV94RuntimeError(
            "fixed-ROI evidence calibration differs from the V94 contract"
        )
    if bundle_contract != expected_contract or checkpoint_sha256 != expected_checkpoint:
        raise SupervisedV94RuntimeError(
            "fixed-ROI evidence bundle/checkpoint provenance differs; "
            "an external --checkpoint requires --select-object-roi or "
            "--object-roi"
        )
    if (
        valid_publications < VERIFIED_OBJECT_ROI_MIN_PROVIDER_STEPS
        or invalid_publications != 0
        or provider_steps != valid_publications
        or sample_every != VERIFIED_OBJECT_ROI_SAMPLE_EVERY
        or configured_compute_threads != 1
        or not np.isfinite(acquisition_elapsed_s)
        or acquisition_elapsed_s < 60.0
        or sam2_mode != "disabled_by_live_cli"
    ):
        raise SupervisedV94RuntimeError(
            "fixed-ROI evidence did not retain its validated provider soak"
        )
    if roi_values.shape != (4,) or not np.issubdtype(roi_values.dtype, np.integer):
        raise SupervisedV94RuntimeError("fixed-ROI evidence ROI is not int32[4]")
    roi = tuple(int(value) for value in roi_values.tolist())
    if roi != VERIFIED_OBJECT_ROI_XYWH:
        raise SupervisedV94RuntimeError("fixed-ROI evidence ROI changed")
    x, y, width, height = roi
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > contract.camera_width
        or y + height > contract.camera_height
    ):
        raise SupervisedV94RuntimeError("fixed object ROI exceeds the camera frame")
    if (
        camera_timestamps_s.ndim != 1
        or camera_timestamps_s.size < 2
        or not np.all(np.isfinite(camera_timestamps_s))
        or not np.all(np.diff(camera_timestamps_s) > 0.0)
        or float(camera_timestamps_s[-1] - camera_timestamps_s[0])
        < VERIFIED_OBJECT_ROI_MIN_CAMERA_SPAN_S
    ):
        raise SupervisedV94RuntimeError(
            "fixed-ROI evidence lacks 60 seconds of increasing camera timestamps"
        )
    if (
        init_prompt_values.shape != (4,)
        or init_mask_values.shape != (4,)
        or not np.issubdtype(init_prompt_values.dtype, np.integer)
        or not np.issubdtype(init_mask_values.dtype, np.integer)
    ):
        raise SupervisedV94RuntimeError(
            "fixed-ROI initialization boxes are not integer xyxy vectors"
        )
    init_prompt = tuple(int(value) for value in init_prompt_values.tolist())
    init_mask = tuple(int(value) for value in init_mask_values.tolist())
    expected_prompt = (x, y, x + width, y + height)
    if init_prompt != expected_prompt or init_mask_source != "grabcut":
        raise SupervisedV94RuntimeError(
            "fixed-ROI initialization did not use GrabCut on the pinned ROI"
        )
    mx1, my1, mx2, my2 = init_mask
    px1, py1, px2, py2 = init_prompt
    margins = (mx1 - px1, my1 - py1, px2 - mx2, py2 - my2)
    mask_bbox_area = (mx2 - mx1) * (my2 - my1)
    if (
        mx2 <= mx1
        or my2 <= my1
        or min(margins) < VERIFIED_OBJECT_ROI_MIN_INIT_MARGIN_PX
        or init_mask_area < 20
        or init_mask_area >= mask_bbox_area
    ):
        raise SupervisedV94RuntimeError(
            "fixed-ROI GrabCut initialization lacks safe margin/non-solid mask"
        )
    return VerifiedObjectROI(
        xywh=roi,
        evidence_path=str(source),
        evidence_sha256=evidence_sha256,
        camera_serial=camera_serial,
        calibration_id=calibration_id,
        checkpoint_sha256=checkpoint_sha256,
        valid_publications=valid_publications,
        invalid_publications=invalid_publications,
        acquisition_elapsed_s=acquisition_elapsed_s,
    )


def _resolve_object_roi(
    *,
    request: Any,
    bundle_path: Path,
    contract: V94Contract,
    checkpoint_sha256: str,
    bundle_sha256: Optional[str] = None,
    pcd_config_sha256: Optional[str] = None,
) -> VerifiedObjectROI:
    """Bind either the pinned location or a camera-preflighted current ROI.

    The fixed prompt needs its historical NPZ because that file is its proof.
    A movable target instead carries a run-local, camera-only preflight digest
    that binds the deployment bundle, point-cloud config, prompt, mask, depth,
    and final policy point clouds.  It must not inherit or depend on an
    unrelated historical fixed-ROI artifact.
    """

    requested = getattr(request, "object_roi_xywh", None)
    if requested is None:
        return _load_verified_object_roi(
            bundle_path=bundle_path,
            contract=contract,
            selected_checkpoint_sha256=checkpoint_sha256,
        )
    try:
        raw = tuple(requested)
    except TypeError as exc:
        raise SupervisedV94RuntimeError(
            "current object ROI must contain integer XYWH"
        ) from exc
    if len(raw) != 4:
        raise SupervisedV94RuntimeError("current object ROI must contain integer XYWH")
    values = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            raise SupervisedV94RuntimeError(
                f"current object ROI[{index}] must be an integer"
            )
        try:
            numeric = float(value)
            integral = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SupervisedV94RuntimeError(
                f"current object ROI[{index}] must be an integer"
            ) from exc
        if not np.isfinite(numeric) or numeric != float(integral):
            raise SupervisedV94RuntimeError(
                f"current object ROI[{index}] must be an integer"
            )
        values.append(integral)
    x, y, width, height = values
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > contract.camera_width
        or y + height > contract.camera_height
    ):
        raise SupervisedV94RuntimeError(
            "current object ROI exceeds the V94 camera frame"
        )

    source = str(getattr(request, "object_roi_source", "")).strip()
    if source not in {
        "operator_numeric_camera_preflight",
        "operator_interactive_camera_preflight",
        "text_grounding_camera_preflight",
    }:
        raise SupervisedV94RuntimeError(
            "current object ROI lacks camera-only preflight provenance"
        )
    preflight_sha256 = (
        str(getattr(request, "object_roi_preflight_sha256", "")).strip().lower()
    )
    valid_frames = int(getattr(request, "object_roi_preflight_valid_frames", 0))
    invalid_frames = int(getattr(request, "object_roi_preflight_invalid_frames", -1))
    minimum_points = int(getattr(request, "object_roi_preflight_min_policy_points", 0))
    try:
        mask_bbox = tuple(
            int(value)
            for value in getattr(
                request,
                "object_roi_preflight_mask_bbox_xyxy",
                (),
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SupervisedV94RuntimeError(
            "current object ROI preflight mask bbox is invalid"
        ) from exc
    mask_area = int(getattr(request, "object_roi_preflight_mask_area_px", 0))
    mask_source = (
        str(getattr(request, "object_roi_preflight_mask_source", "")).strip().lower()
    )
    preflight_bundle_sha256 = (
        str(getattr(request, "object_roi_preflight_bundle_sha256", "")).strip().lower()
    )
    preflight_pcd_config_sha256 = (
        str(getattr(request, "object_roi_preflight_pcd_config_sha256", ""))
        .strip()
        .lower()
    )
    preflight_checkpoint_sha256 = (
        str(getattr(request, "object_roi_preflight_checkpoint_sha256", ""))
        .strip()
        .lower()
    )
    try:
        depth_p50 = float(
            getattr(request, "object_roi_preflight_depth_p50_m", float("nan"))
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SupervisedV94RuntimeError(
            "current object ROI preflight depth statistic is invalid"
        ) from exc
    px1, py1, px2, py2 = x, y, x + width, y + height
    if len(mask_bbox) == 4:
        mx1, my1, mx2, my2 = mask_bbox
        mask_bbox_area = (mx2 - mx1) * (my2 - my1)
        mask_margins = (
            mx1 - px1,
            my1 - py1,
            px2 - mx2,
            py2 - my2,
        )
    else:
        mask_bbox_area = 0
        mask_margins = (-1, -1, -1, -1)
    prompt_area = width * height
    mask_shape_valid = bool(
        len(mask_bbox) == 4
        and mx2 > mx1
        and my2 > my1
        and mask_area >= 20
        and mask_area < mask_bbox_area
    )
    mask_source_geometry_valid = False
    if mask_shape_valid and mask_source == "grabcut":
        mask_source_geometry_valid = bool(
            min(mask_margins) >= VERIFIED_OBJECT_ROI_MIN_INIT_MARGIN_PX
        )
    elif mask_shape_valid and mask_source in CURRENT_OBJECT_ROI_SAM2_MASK_SOURCES:
        image_margins = (
            mx1,
            my1,
            contract.camera_width - mx2,
            contract.camera_height - my2,
        )
        mask_source_geometry_valid = bool(
            min(image_margins) >= CURRENT_OBJECT_ROI_SAM2_MIN_IMAGE_MARGIN_PX
            and mask_area
            <= CURRENT_OBJECT_ROI_SAM2_MAX_MASK_AREA_RATIO * prompt_area
            and mask_bbox_area
            <= CURRENT_OBJECT_ROI_SAM2_MAX_BBOX_AREA_RATIO * prompt_area
        )
    if (
        not re.fullmatch(r"[0-9a-f]{64}", preflight_sha256)
        or valid_frames < 3
        or not 0 <= invalid_frames <= CURRENT_OBJECT_ROI_PREFLIGHT_MAX_INVALID_FRAMES
        or minimum_points < 16
        or not np.isfinite(depth_p50)
        or not contract.depth_range_m[0] < depth_p50 < contract.depth_range_m[1]
        or not mask_source_geometry_valid
        or not re.fullmatch(r"[0-9a-f]{64}", preflight_bundle_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", preflight_pcd_config_sha256)
        or bundle_sha256 is None
        or pcd_config_sha256 is None
        or preflight_bundle_sha256 != str(bundle_sha256).lower()
        or preflight_pcd_config_sha256 != str(pcd_config_sha256).lower()
        or not re.fullmatch(r"[0-9a-f]{64}", preflight_checkpoint_sha256)
        or preflight_checkpoint_sha256 != str(checkpoint_sha256).lower()
    ):
        raise SupervisedV94RuntimeError(
            "current object ROI did not retain its camera-only mask/depth/"
            "policy-point preflight"
        )
    bundle = DeployBundle(bundle_path)
    bundle.verify()
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise SupervisedV94RuntimeError(
            "current object ROI lacks a pinned selected checkpoint"
        )
    return VerifiedObjectROI(
        xywh=(x, y, width, height),
        evidence_path=None,
        evidence_sha256="",
        camera_serial=contract.camera_serial,
        calibration_id=contract.calibration_id,
        checkpoint_sha256=checkpoint_sha256,
        valid_publications=valid_frames,
        invalid_publications=invalid_frames,
        acquisition_elapsed_s=0.0,
        position_source=source,
        preflight_sha256=preflight_sha256,
        preflight_valid_frames=valid_frames,
        preflight_invalid_frames=invalid_frames,
        preflight_min_policy_points=minimum_points,
        preflight_depth_p50_m=depth_p50,
        preflight_mask_bbox_xyxy=mask_bbox,
        preflight_mask_area_px=mask_area,
        preflight_mask_source=mask_source,
        preflight_bundle_sha256=preflight_bundle_sha256,
        preflight_pcd_config_sha256=preflight_pcd_config_sha256,
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupervisedV94RuntimeError(f"JSON root is not an object: {path}")
    return value


def _validate_request(request: Any) -> tuple[Path, dict[str, Any], V94Contract, Any]:
    try:
        policy_mode = resolve_policy_rate_mode(
            getattr(request, "policy_rate_hz", POLICY_RATE_HZ)
        )
    except ValueError as exc:
        raise SupervisedV94RuntimeError(str(exc)) from exc
    run_id = str(request.run_id).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise SupervisedV94RuntimeError(
            "run-id must be 3..81 safe filename characters [A-Za-z0-9_.-]"
        )
    raw_steps = request.steps
    if isinstance(raw_steps, (bool, np.bool_)) or not isinstance(
        raw_steps, (int, np.integer)
    ):
        raise SupervisedV94RuntimeError(
            f"{policy_mode.name} supervised steps must remain in "
            f"1..{policy_mode.maximum_supervised_steps}"
        )
    if not 1 <= int(raw_steps) <= policy_mode.maximum_supervised_steps:
        raise SupervisedV94RuntimeError(
            f"{policy_mode.name} supervised steps must remain in "
            f"1..{policy_mode.maximum_supervised_steps}"
        )
    task_execute_cap = task_profile_maximum_supervised_execute_steps(
        Path(request.pcd_config)
    )
    if (
        bool(getattr(request, "execute", False))
        and task_execute_cap is not None
        and int(raw_steps) > task_execute_cap
    ):
        raise SupervisedV94RuntimeError(
            "requested steps exceed the selected task profile's supervised "
            f"execution cap of {task_execute_cap}"
        )
    required_paths = [
        (request.bundle, "bundle"),
        (request.profile, "profile"),
        (request.pcd_config, "point-cloud config"),
    ]
    checkpoint_override = getattr(request, "checkpoint", None)
    if checkpoint_override is not None:
        required_paths.append((checkpoint_override, "checkpoint"))
    replay_override = getattr(request, "replay_actions", None)
    if replay_override is not None:
        required_paths.append((replay_override, "replay action file"))
    for path, label in required_paths:
        if not Path(path).is_file():
            raise SupervisedV94RuntimeError(f"{label} is missing: {path}")
    output = (
        Path(__file__).resolve().parents[2]
        / "dexgrasp/runs"
        / f"v94_supervised_real_{run_id}.json"
    )
    if output.exists():
        raise SupervisedV94RuntimeError(
            f"run-id is not unique; audit artifact already exists: {output}"
        )
    if bool(getattr(request, "record_policy_io", False)):
        policy_io_output = default_policy_io_path(run_id)
        if policy_io_output.exists():
            raise SupervisedV94RuntimeError(
                "run-id is not unique; policy-I/O artifact already exists: "
                f"{policy_io_output}"
            )
    profile = _read_json(Path(request.profile))
    try:
        load_v94_rh56_profile_command_bounds(
            profile,
            feedback_to_command_offset_units=(RH56_FEEDBACK_TO_COMMAND_OFFSET_UNITS),
        )
    except ValueError as exc:
        raise SupervisedV94RuntimeError(
            f"selected V94 RH56 profile is not commissioned: {exc}"
        ) from exc
    bundle = DeployBundle(Path(request.bundle))
    bundle.verify()
    verify_v94_bundle(request.bundle)
    contract = resolve_runtime_task_contract(
        V94Contract.from_bundle(bundle).with_runtime_policy_rate_hz(
            policy_mode.policy_rate_hz
        ),
        Path(request.pcd_config),
    )
    envelope = load_experimental_supervised_franka_envelope(request.profile)
    calibration = profile.get("calibration", {})
    if calibration.get("id") != contract.calibration_id:
        raise SupervisedV94RuntimeError("profile/bundle calibration IDs differ")
    if calibration.get("camera_serial") != contract.camera_serial:
        raise SupervisedV94RuntimeError("profile/bundle camera serials differ")

    franka_mapping = audit_v94_franka_action_mapping(
        request.bundle,
        request.profile,
        expected_reset_q_rad=contract.q_home_rad,
        expected_joint_limits_rad=contract.joint_limits_rad,
    )
    if (
        franka_mapping.packaged_reference_mapping_confirmed is not True
        or franka_mapping.axis_order_identity is not True
        or franka_mapping.sign_transform != "identity (no negation)"
        or abs(franka_mapping.effective_gain_rad_per_tick - 0.003) > 1e-7
    ):
        raise SupervisedV94RuntimeError("Franka action-to-target mapping audit failed")
    rh56_mapping = audit_v94_rh56_action_mapping(
        bundle_path=request.bundle,
        commissioning_profile_path=request.profile,
        shadow_path=SHADOW21 if SHADOW21.is_file() else None,
    )
    if rh56_mapping.get("mathematical_mapping_confirmed") is not True:
        raise SupervisedV94RuntimeError("RH56 action-to-register mapping audit failed")
    return output, profile, contract, envelope


def _fresh_franka_preflight(
    *,
    robot_ip: str,
    q_home_rad: np.ndarray,
    maximum_start_error_rad: float,
    envelope: Any,
) -> tuple[tuple[float, ...], tuple[float, ...], float, bool]:
    module = importlib.import_module("pylibfranka")
    if str(getattr(module, "__version__", "")) != AUDITED_PYLIBFRANKA_VERSION:
        raise SupervisedV94RuntimeError(
            "pylibfranka version changed from audited backend"
        )
    robot = module.Robot(robot_ip, module.RealtimeConfig.kEnforce)
    try:
        state = robot.read_once()
        if state is None:
            raise SupervisedV94RuntimeError("Franka fresh preflight returned no state")
        q, dq = _state_vectors(state)
        # Full rigid-transform/payload/inertia checks are intentionally done
        # on this read-only state, outside the active FCI read->write window.
        validator = object.__new__(FrankaPersistentSession)
        validator.envelope = envelope
        validator._validate_static_provenance(state)
        if _robot_mode_name(getattr(state, "robot_mode", None)) != "idle":
            raise SupervisedV94RuntimeError("Franka is not Idle at fresh preflight")
        if _has_active_errors(getattr(state, "current_errors", None)):
            raise SupervisedV94RuntimeError("Franka current_errors are active")
        _require_clear_flags(state, "joint_contact", 7)
        _require_clear_flags(state, "joint_collision", 7)
        _require_clear_flags(state, "cartesian_contact", 6)
        _require_clear_flags(state, "cartesian_collision", 6)
        if float(np.max(np.abs(dq))) > 0.01:
            raise SupervisedV94RuntimeError("Franka is not stationary enough")
        start_error = float(np.max(np.abs(q - q_home_rad.astype(np.float64))))
        if start_error > maximum_start_error_rad:
            raise SupervisedV94RuntimeError(
                f"Franka q differs from V94 q_home by {start_error:.9f}rad"
            )
        success = float(getattr(state, "control_command_success_rate", np.nan))
        if not np.isfinite(success) or not 0.0 <= success <= 1.0:
            raise SupervisedV94RuntimeError("Franka success-rate field is invalid")
        return tuple(q.tolist()), tuple(dq.tolist()), start_error, True
    finally:
        del robot
        gc.collect()


def _fresh_rh56_preflight(
    *, port: str, baud: int, hand_id: int, open_targets: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    transport = LinuxRH56TransactionalTransport(port, baud_rate=baud, hand_id=hand_id)
    transport.open()
    try:
        targets = []
        feedback = []
        for _ in range(2):
            targets.append(
                tuple(
                    transport.read_angle_set(
                        deadline_monotonic_s=time.monotonic() + RH56_IO_DEADLINE_S
                    )
                )
            )
            feedback.append(
                transport.read_safety_feedback(
                    deadline_monotonic_s=time.monotonic() + RH56_IO_DEADLINE_S
                )
            )
        if any(value != (-1,) * 6 for value in targets):
            raise SupervisedV94RuntimeError("RH56 ANGLE_SET is not disabled (-1 x6)")
        expected_open = np.asarray(open_targets, dtype=np.int64)
        if expected_open.shape != (6,):
            raise SupervisedV94RuntimeError("RH56 open target profile is invalid")
        for sample in feedback:
            if any(sample.errors):
                raise SupervisedV94RuntimeError("RH56 has a fresh device error")
            if any(status not in RH56_IDLE_STATUSES for status in sample.statuses):
                raise SupervisedV94RuntimeError(
                    "RH56 disabled status is not idle (expected only 2 or 255)"
                )
            if max(sample.temperatures_c) >= 60:
                raise SupervisedV94RuntimeError("RH56 temperature is >=60C")
            if max(abs(value) for value in sample.currents_ma) > 100:
                raise SupervisedV94RuntimeError("RH56 disabled current exceeds 100mA")
            angle_error = np.abs(np.asarray(sample.angles) - expected_open)
            if np.any(
                angle_error
                > np.asarray(
                    RH56_DISABLED_OPEN_TOLERANCE_UNITS,
                    dtype=np.int64,
                )
            ):
                raise SupervisedV94RuntimeError(
                    "RH56 is disabled but not physically open"
                )
        if max(abs(a - b) for a, b in zip(feedback[0].angles, feedback[1].angles)) > 8:
            raise SupervisedV94RuntimeError("RH56 open feedback drift exceeds 8 units")
        latest = feedback[-1]
        return targets[-1], latest.angles, max(abs(v) for v in latest.currents_ma)
    finally:
        transport.close()


def _make_rh56_owner_factory(
    *,
    port: str,
    baud: int,
    hand_id: int,
    speed: Sequence[int],
    force: int,
    open_targets: Sequence[int],
    minimum_angle_set_register_order: Sequence[int],
    maximum_angle_set_register_order: Sequence[int],
):
    expected_open = tuple(int(value) for value in open_targets)
    if len(expected_open) != 6:
        raise ValueError("RH56 open_targets must contain six values")
    requested_speed = tuple(int(value) for value in speed)
    if len(requested_speed) != 6 or any(
        value < 0 or value > 1000 for value in requested_speed
    ):
        raise ValueError("RH56 speed must contain six values in [0,1000]")
    requested_force = (int(force),) * 6
    if any(value < 0 or value > 1000 for value in requested_force):
        raise ValueError("RH56 force must lie in [0,1000]")
    command_minimum = tuple(int(value) for value in minimum_angle_set_register_order)
    command_maximum = tuple(int(value) for value in maximum_angle_set_register_order)
    if (
        len(command_minimum) != 6
        or len(command_maximum) != 6
        or any(
            lower < 0 or upper > 1000 or lower >= upper
            for lower, upper in zip(command_minimum, command_maximum)
        )
    ):
        raise ValueError("RH56 command intervals are invalid")

    def build(*, admission: Any, action_ledger: Any, fault_callback: Any):
        def open_arm_session() -> RH56OwnedSession:
            transport = LinuxRH56TransactionalTransport(
                port, baud_rate=baud, hand_id=hand_id
            )
            actuator: Optional[RH56TransactionalActuator] = None
            original_speed: Optional[tuple[int, ...]] = None
            original_force: Optional[tuple[int, ...]] = None
            restore_lock = threading.Lock()
            restored = False

            def deadline() -> float:
                return time.monotonic() + RH56_IO_DEADLINE_S

            def restore_and_close() -> None:
                nonlocal restored
                with restore_lock:
                    if restored:
                        return
                    restored = True
                errors = []
                stop_confirmed = bool(actuator is not None and actuator.stop_confirmed)
                if stop_confirmed:
                    try:
                        if original_speed is not None:
                            transport.write_speed_set(
                                original_speed, deadline_monotonic_s=deadline()
                            )
                            if (
                                transport.read_speed_set(
                                    deadline_monotonic_s=deadline()
                                )
                                != original_speed
                            ):
                                raise SupervisedV94RuntimeError(
                                    "RH56 SPEED_SET restore mismatch"
                                )
                        if original_force is not None:
                            transport.write_force_set(
                                original_force, deadline_monotonic_s=deadline()
                            )
                            if (
                                transport.read_force_set(
                                    deadline_monotonic_s=deadline()
                                )
                                != original_force
                            ):
                                raise SupervisedV94RuntimeError(
                                    "RH56 FORCE_SET restore mismatch"
                                )
                    except BaseException as exc:
                        errors.append(exc)
                else:
                    # Never raise speed/force while a numeric ANGLE_SET may
                    # still be active.  Close the descriptor in the safe
                    # direction and make the entire run fail visibly.
                    errors.append(
                        SupervisedV94RuntimeError(
                            "RH56 stop unconfirmed; SPEED/FORCE restore skipped"
                        )
                    )
                try:
                    transport.close()
                except BaseException as exc:
                    errors.append(exc)
                if errors:
                    raise SupervisedV94RuntimeError(
                        "RH56 restore/close failed: "
                        + "; ".join(f"{type(e).__name__}: {e}" for e in errors)
                    )

            transport.open()
            try:
                # The earlier preflight used a descriptor that has now been
                # closed.  Re-establish the disabled/open invariant on this
                # exact owner descriptor before changing SPEED_SET/FORCE_SET;
                # otherwise a reconnect or an external writer creates a
                # time-of-check/time-of-use gap immediately before arming.
                owner_targets = []
                owner_feedback = []
                for _ in range(2):
                    owner_targets.append(
                        tuple(transport.read_angle_set(deadline_monotonic_s=deadline()))
                    )
                    owner_feedback.append(
                        transport.read_safety_feedback(deadline_monotonic_s=deadline())
                    )
                if any(target != (-1,) * 6 for target in owner_targets):
                    raise SupervisedV94RuntimeError(
                        "RH56 owner reopen found ANGLE_SET active"
                    )
                for sample in owner_feedback:
                    if any(sample.errors):
                        raise SupervisedV94RuntimeError(
                            "RH56 owner reopen found a device error"
                        )
                    if any(
                        status not in RH56_IDLE_STATUSES for status in sample.statuses
                    ):
                        raise SupervisedV94RuntimeError(
                            "RH56 owner reopen status is not idle "
                            "(expected only 2 or 255)"
                        )
                    if max(sample.temperatures_c) >= 60:
                        raise SupervisedV94RuntimeError(
                            "RH56 owner reopen temperature is >=60C"
                        )
                    if max(abs(value) for value in sample.currents_ma) > 100:
                        raise SupervisedV94RuntimeError(
                            "RH56 owner reopen disabled current exceeds 100mA"
                        )
                    if any(
                        abs(actual - expected) > tolerance
                        for actual, expected, tolerance in zip(
                            sample.angles,
                            expected_open,
                            RH56_DISABLED_OPEN_TOLERANCE_UNITS,
                        )
                    ):
                        raise SupervisedV94RuntimeError(
                            "RH56 owner reopen is disabled but not physically open"
                        )
                if (
                    max(
                        abs(first - second)
                        for first, second in zip(
                            owner_feedback[0].angles, owner_feedback[1].angles
                        )
                    )
                    > 8
                ):
                    raise SupervisedV94RuntimeError(
                        "RH56 owner reopen feedback drift exceeds 8 units"
                    )
                original_speed = transport.read_speed_set(
                    deadline_monotonic_s=deadline()
                )
                original_force = transport.read_force_set(
                    deadline_monotonic_s=deadline()
                )
                transport.write_speed_set(
                    requested_speed, deadline_monotonic_s=deadline()
                )
                if (
                    transport.read_speed_set(deadline_monotonic_s=deadline())
                    != requested_speed
                ):
                    raise SupervisedV94RuntimeError("RH56 SPEED_SET readback mismatch")
                transport.write_force_set(
                    requested_force, deadline_monotonic_s=deadline()
                )
                if (
                    transport.read_force_set(deadline_monotonic_s=deadline())
                    != requested_force
                ):
                    raise SupervisedV94RuntimeError("RH56 FORCE_SET readback mismatch")
                actuator = RH56TransactionalActuator(
                    transport,
                    command_watchdog_timeout_s=RH56_WATCHDOG_S,
                    supervised_inter_command_watchdog_timeout_s=(
                        RH56_INTER_COMMAND_WATCHDOG_S
                    ),
                    maximum_feedback_age_s=RH56_FEEDBACK_HARD_AGE_S,
                    maximum_running_axis_current_ma=(RH56_MAX_RUNNING_CURRENT_MA),
                    maximum_temperature_c=60,
                    stop_timeout_s=RH56_STOP_TIMEOUT_S,
                    stop_verify_samples=3,
                    stop_verify_interval_s=0.02,
                    stop_max_axis_current_ma=100,
                    stop_settle_max_axis_current_ma=(
                        RH56_STOP_SETTLE_MAX_AXIS_CURRENT_MA
                    ),
                    stop_max_angle_drift_units=8,
                    stop_max_position_drift_units=8,
                    feedback_to_command_offset_units=(
                        RH56_FEEDBACK_TO_COMMAND_OFFSET_UNITS
                    ),
                    stop_hold_command_minimum=(command_minimum),
                    stop_hold_command_maximum=(command_maximum),
                )
                actuator.claim_for_current_thread()
                actuator.arm_supervised(
                    preflight=admission.rh56_preflight,
                    authorization=admission.authorization,
                    safety_gate=admission.safety_gate,
                    run_id=admission.run_id,
                    now_monotonic_s=time.monotonic(),
                )
                return RH56OwnedSession(actuator=actuator, close=restore_and_close)
            except BaseException:
                try:
                    if (
                        actuator is not None
                        and actuator.state is not RH56ActuatorState.DISARMED
                    ):
                        actuator.disable_and_verify()
                    else:
                        for _ in range(2):
                            transport.write_angle_set(
                                (-1,) * 6, deadline_monotonic_s=deadline()
                            )
                            if (
                                transport.read_angle_set(
                                    deadline_monotonic_s=deadline()
                                )
                                != (-1,) * 6
                            ):
                                raise SupervisedV94RuntimeError(
                                    "RH56 setup cleanup disable mismatch"
                                )
                finally:
                    restore_and_close()
                raise

        return RH56WatchdogOwner(
            open_arm_session,
            action_ledger,
            fault_callback=fault_callback,
            startup_timeout_s=3.0,
            response_timeout_s=RH56_OWNER_RESPONSE_TIMEOUT_S,
            join_timeout_s=RH56_OWNER_JOIN_TIMEOUT_S,
            bootstrap_feedback_samples=2,
            feedback_history_capacity=16,
            supervised_target_only=True,
            # Complete feedback is independent of the selected policy loop. It
            # is serialized by the same owner as the 20 Hz target stream, so
            # no feedback read can overlap an ANGLE_SET transaction.
            feedback_period_s=RH56_FEEDBACK_POLL_TRIGGER_S,
            feedback_hard_age_s=RH56_FEEDBACK_HARD_AGE_S,
            tracking_significant_gap_units=(RH56_TRACKING_SIGNIFICANT_GAP_UNITS),
            tracking_min_progress_units=RH56_TRACKING_MIN_PROGRESS_UNITS,
            tracking_contact_force_delta_g=(
                RH56_TRACKING_CONTACT_FORCE_DELTA_G
            ),
            tracking_contact_force_absolute_g=(
                RH56_TRACKING_CONTACT_FORCE_ABSOLUTE_G
            ),
            tracking_timeout_s=RH56_TRACKING_TIMEOUT_S,
            thread_name=f"rh56-supervised-{admission.run_id}",
        )

    return build


def _make_native_franka_session_factory(
    *,
    build: NativeServoBuildIdentity,
    robot_ip: str,
    servo_cpus: Optional[Sequence[int]] = None,
    prelaunch_validator: Optional[Any] = None,
):
    address = str(robot_ip).strip()
    if not address:
        raise ValueError("robot_ip must be non-empty")
    pinned_cpus = tuple(int(value) for value in (servo_cpus or ()))
    if any(value < 0 for value in pinned_cpus) or len(set(pinned_cpus)) != len(
        pinned_cpus
    ):
        raise ValueError("servo_cpus must be unique nonnegative CPU indices")
    taskset = Path("/usr/bin/taskset")
    if pinned_cpus and not taskset.is_file():
        raise SupervisedV94RuntimeError(
            "native Franka CPU isolation requires /usr/bin/taskset"
        )
    if prelaunch_validator is not None and not callable(prelaunch_validator):
        raise TypeError("prelaunch_validator must be callable")

    def construct(*, admission: Any, action_ledger: Any):
        expected_parent_pid = os.getpid()
        executable_fd = _open_verified_native_executable_fd(build)
        expectation = NativeHelloExpectation(
            state_decimation=build.state_decimation,
            safety_limits_schema=build.safety_limits_schema,
            libfranka_sha256=build.libfranka_sha256,
            libfranka_source_commit=build.libfranka_source_commit,
            producer_build_sha256=build.producer_build_sha256,
        )
        codec = V94NativePODCodec(
            hello_expectation=expectation,
            reference_q_rad=admission.franka_reference_q_rad,
            maximum_target_count=admission.requested_policy_steps,
            expected_servo_cpu=(pinned_cpus[0] if len(pinned_cpus) == 1 else None),
        )
        launcher = None
        try:

            def native_argv(critical_fd: int, telemetry_fd: int) -> tuple[str, ...]:
                binary_argv = (
                    f"/proc/self/fd/{executable_fd}",
                    "--execute-supervised-v94",
                    "--robot-ip",
                    address,
                    "--critical-fd",
                    str(critical_fd),
                    "--telemetry-fd",
                    str(telemetry_fd),
                    "--parent-pid",
                    str(expected_parent_pid),
                )
                if not pinned_cpus:
                    return binary_argv
                return (
                    str(taskset),
                    "--cpu-list",
                    ",".join(str(value) for value in pinned_cpus),
                    *binary_argv,
                )

            launcher = PopenSeqpacketChildLauncher(
                native_argv,
                cwd=str(Path(__file__).resolve().parents[2]),
                pinned_executable_fd=executable_fd,
                prelaunch_validator=prelaunch_validator,
            )
            return FrankaNativeSupervisedSessionProxy(
                run_id=admission.run_id,
                envelope=admission.franka_envelope,
                action_ledger=action_ledger,
                codec=codec,
                launcher=launcher,
                controller_mode=(
                    V94NativeControllerMode.QD_G015
                    if admission.franka_action_contract_id
                    == QD_G015_FRANKA_ACTION_CONTRACT_ID
                    else V94NativeControllerMode.LEGACY
                ),
                heartbeat_interval_s=0.010,
                event_poll_interval_s=0.002,
                startup_timeout_s=5.0,
                shutdown_timeout_s=5.0,
            )
        except BaseException:
            if launcher is None:
                try:
                    os.close(executable_fd)
                except OSError:
                    pass
            else:
                launcher.close()
            raise

    return construct


def _write_audit(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


_RH56_FORCE_AXIS_NAMES = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)
_GRAM_FORCE_TO_NEWTON = 9.80665e-3


def _write_rh56_force_csv(
    path: Path,
    committed_commands: Sequence[Mapping[str, Any]],
) -> int:
    """Write one post-stop row per committed command's measured hand state.

    This is intentionally derived from the already detached audit ledger after
    both actuator owners have stopped.  It performs no device reads and adds no
    work to the policy/control loop.
    """

    fixed_columns = (
        "sequence",
        "observation_realtime_s",
        "feedback_captured_realtime_s",
        "feedback_captured_monotonic_s",
        "feedback_age_s",
        "feedback_fresh",
    )
    per_axis_fields = (
        "target_angle_set",
        "angle_act",
        "position_act",
        "force_gf",
        "force_n",
        "current_ma",
        "status",
    )
    fieldnames = list(fixed_columns)
    for axis in _RH56_FORCE_AXIS_NAMES:
        fieldnames.extend(f"{axis}_{field}" for field in per_axis_fields)

    def six(value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            return (None,) * 6
        return tuple(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows_written = 0
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for command in committed_commands:
            feedback_raw = command.get("measured_rh56_feedback_at_commit")
            feedback = feedback_raw if isinstance(feedback_raw, Mapping) else {}
            targets = six(command.get("rh56_angle_set_register_order"))
            angles = six(feedback.get("angles"))
            positions = six(feedback.get("positions"))
            forces = six(feedback.get("forces_g"))
            currents = six(feedback.get("currents_ma"))
            statuses = six(feedback.get("statuses"))
            row: dict[str, object] = {
                "sequence": command.get("sequence"),
                "observation_realtime_s": command.get("observation_realtime_s"),
                "feedback_captured_realtime_s": feedback.get(
                    "captured_realtime_s"
                ),
                "feedback_captured_monotonic_s": feedback.get(
                    "captured_monotonic_s"
                ),
                "feedback_age_s": feedback.get("latest_age_s"),
                "feedback_fresh": feedback.get("fresh"),
            }
            for index, axis in enumerate(_RH56_FORCE_AXIS_NAMES):
                force_gf = forces[index]
                force_n = (
                    None
                    if force_gf is None
                    else float(force_gf) * _GRAM_FORCE_TO_NEWTON
                )
                row.update(
                    {
                        f"{axis}_target_angle_set": targets[index],
                        f"{axis}_angle_act": angles[index],
                        f"{axis}_position_act": positions[index],
                        f"{axis}_force_gf": force_gf,
                        f"{axis}_force_n": force_n,
                        f"{axis}_current_ma": currents[index],
                        f"{axis}_status": statuses[index],
                    }
                )
            writer.writerow(row)
            rows_written += 1
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return rows_written


def prepare_supervised_v94_artifacts(
    request: Any,
) -> PreparedSupervisedV94Artifacts:
    """Pin artifacts/native ABI without requiring any historical ROI proof."""

    checkpoint_override = getattr(request, "checkpoint", None)
    checkpoint_path = (
        None
        if checkpoint_override is None
        else Path(checkpoint_override).expanduser().resolve()
    )
    replay_override = getattr(request, "replay_actions", None)
    replay_action_path = (
        None
        if replay_override is None
        else Path(replay_override).expanduser().resolve()
    )
    artifact_paths = (
        Path(request.bundle).expanduser().resolve(),
        Path(request.profile).expanduser().resolve(),
        Path(request.pcd_config).expanduser().resolve(),
    )
    if checkpoint_path is not None:
        artifact_paths += (checkpoint_path,)
    if replay_action_path is not None:
        artifact_paths += (replay_action_path,)
    artifact_hashes_before = tuple(_sha256(path) for path in artifact_paths)
    output, profile, contract, envelope = _validate_request(request)
    # Resolve and hash the exact native servo executable before opening either
    # device.  The resulting identity is included in the run-bound permit, so
    # replacing the binary or its pinned libfranka dependency cannot happen
    # between confirmation and motion authorization.
    profile_id = str(profile.get("profile_id", ""))
    if profile_id == TABLETOP_PROFILE_ID:
        native_build = _load_native_servo_build_identity(
            NATIVE_SERVO_TABLETOP_MANIFEST
        )
    elif profile_id == V60_PALMCATCH_PROFILE_ID:
        native_build = _load_native_servo_build_identity(
            NATIVE_SERVO_V60_MANIFEST
        )
    elif profile_id == V61_SIXEXPERT_PROFILE_ID:
        native_build = _load_native_servo_build_identity(
            NATIVE_SERVO_V61_MANIFEST
        )
    else:
        native_build = _load_native_servo_build_identity()
    if (
        native_build.compiled_profile_sha256 != envelope.profile_sha256
        or native_build.compiled_envelope_sha256 != envelope.binding_sha256
    ):
        raise SupervisedV94RuntimeError(
            "native Franka compiled profile/envelope differs from the "
            "selected deployment profile; rebuild it before hardware access"
        )
    bundle = DeployBundle(artifact_paths[0])
    if checkpoint_path is None:
        pinned_checkpoint_bytes = bundle.checkpoint_bytes()
        primary = bundle.manifest.get("primary_checkpoint", {})
        expected_checkpoint_sha256 = (
            str(primary.get("sha256", "")) if isinstance(primary, Mapping) else ""
        )
        checkpoint_source = "bundle_primary"
        if (
            len(expected_checkpoint_sha256) != 64
            or hashlib.sha256(pinned_checkpoint_bytes).hexdigest()
            != expected_checkpoint_sha256
        ):
            raise SupervisedV94RuntimeError(
                "pinned policy checkpoint differs from the verified bundle manifest"
            )
        selected_mode = resolve_policy_rate_mode(
            getattr(request, "policy_rate_hz", POLICY_RATE_HZ)
        )
        if (
            replay_action_path is None
            and selected_mode.name != DEFAULT_POLICY_RATE_MODE.name
        ):
            raise SupervisedV94RuntimeError(
                "the transferred bundle primary checkpoint is a 60 Hz model; "
                "20 Hz mode requires a matching external --checkpoint"
            )
    else:
        checkpoint_size = checkpoint_path.stat().st_size
        if checkpoint_size > MAX_CHECKPOINT_BYTES:
            raise SupervisedV94RuntimeError(
                f"checkpoint exceeds {MAX_CHECKPOINT_BYTES} byte safety limit"
            )
        pinned_checkpoint_bytes = checkpoint_path.read_bytes()
        if len(pinned_checkpoint_bytes) != checkpoint_size:
            raise SupervisedV94RuntimeError(
                "external checkpoint changed while it was being read"
            )
        expected_checkpoint_sha256 = hashlib.sha256(pinned_checkpoint_bytes).hexdigest()
        if expected_checkpoint_sha256 != artifact_hashes_before[3]:
            raise SupervisedV94RuntimeError(
                "external checkpoint changed between hashing and pinning"
            )
        checkpoint_source = "external_path_override"
        # This deliberately does not compare actions to the bundle checkpoint:
        # different weights should produce different actions.  It does prove
        # safe decoding, exact V94 model/I/O compatibility, selected
        # control_dt, and finite inference on both packaged observation sets.
        verify_v94_checkpoint_payload(
            artifact_paths[0],
            pinned_checkpoint_bytes,
            expected_control_dt_s=contract.control_dt_s,
        )
    pinned_replay_action_bytes: Optional[bytes] = None
    replay_action_sha256 = ""
    replay_action_count = 0
    if replay_action_path is not None:
        from sim2real.action_replay import (
            MAX_REPLAY_ACTION_BYTES,
            load_replay_actions_payload,
        )

        replay_size = replay_action_path.stat().st_size
        if replay_size > MAX_REPLAY_ACTION_BYTES:
            raise SupervisedV94RuntimeError(
                f"replay action file exceeds {MAX_REPLAY_ACTION_BYTES} byte limit"
            )
        pinned_replay_action_bytes = replay_action_path.read_bytes()
        if len(pinned_replay_action_bytes) != replay_size:
            raise SupervisedV94RuntimeError(
                "replay action file changed while it was being read"
            )
        replay = load_replay_actions_payload(
            pinned_replay_action_bytes,
            suffix=replay_action_path.suffix,
            expected_policy_rate_hz=contract.policy_rate_hz,
            selected_steps=int(request.steps),
        )
        replay_action_sha256 = replay.sha256
        replay_action_count = replay.action_count
        expected_replay_hash = (
            str(getattr(request, "replay_actions_sha256", "")).strip().lower()
        )
        if expected_replay_hash and expected_replay_hash != replay_action_sha256:
            raise SupervisedV94RuntimeError(
                "replay action file differs from the CLI-validated payload"
            )
        expected_replay_count = int(getattr(request, "replay_action_count", 0))
        if expected_replay_count and expected_replay_count != replay_action_count:
            raise SupervisedV94RuntimeError(
                "replay action count differs from the CLI-validated payload"
            )
    artifact_hashes_after = tuple(_sha256(path) for path in artifact_paths)
    if artifact_hashes_after != artifact_hashes_before:
        raise SupervisedV94RuntimeError(
            "deployment bundle/profile/point-cloud config/checkpoint/replay changed "
            "during validation"
        )
    return PreparedSupervisedV94Artifacts(
        output=output,
        profile=profile,
        contract=contract,
        envelope=envelope,
        native_build=native_build,
        bundle_sha256=artifact_hashes_after[0],
        profile_file_sha256=artifact_hashes_after[1],
        pcd_config_sha256=artifact_hashes_after[2],
        checkpoint_source=checkpoint_source,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=expected_checkpoint_sha256,
        pinned_checkpoint_bytes=pinned_checkpoint_bytes,
        replay_action_path=replay_action_path,
        replay_action_sha256=replay_action_sha256,
        replay_action_count=replay_action_count,
        pinned_replay_action_bytes=pinned_replay_action_bytes,
    )


def prepare_supervised_v94(request: Any) -> PreparedSupervisedV94:
    """Complete every filesystem/offline check before policy hardware access."""

    artifacts = prepare_supervised_v94_artifacts(request)
    object_roi = _resolve_object_roi(
        request=request,
        bundle_path=Path(request.bundle),
        contract=artifacts.contract,
        checkpoint_sha256=artifacts.checkpoint_sha256,
        bundle_sha256=artifacts.bundle_sha256,
        pcd_config_sha256=artifacts.pcd_config_sha256,
    )
    return PreparedSupervisedV94(
        output=artifacts.output,
        profile=artifacts.profile,
        contract=artifacts.contract,
        envelope=artifacts.envelope,
        object_roi=object_roi,
        native_build=artifacts.native_build,
        bundle_sha256=artifacts.bundle_sha256,
        profile_file_sha256=artifacts.profile_file_sha256,
        pcd_config_sha256=artifacts.pcd_config_sha256,
        checkpoint_source=artifacts.checkpoint_source,
        checkpoint_path=artifacts.checkpoint_path,
        checkpoint_sha256=artifacts.checkpoint_sha256,
        pinned_checkpoint_bytes=artifacts.pinned_checkpoint_bytes,
        replay_action_path=artifacts.replay_action_path,
        replay_action_sha256=artifacts.replay_action_sha256,
        replay_action_count=artifacts.replay_action_count,
        pinned_replay_action_bytes=artifacts.pinned_replay_action_bytes,
    )


def _require_unchanged_preparation(
    before_reset: PreparedSupervisedV94,
    after_reset: PreparedSupervisedV94,
) -> None:
    """Refuse inference if any deployment input changed before runtime."""

    unchanged = (
        before_reset.output == after_reset.output
        and before_reset.bundle_sha256 == after_reset.bundle_sha256
        and before_reset.profile_file_sha256 == after_reset.profile_file_sha256
        and before_reset.pcd_config_sha256 == after_reset.pcd_config_sha256
        and before_reset.checkpoint_source == after_reset.checkpoint_source
        and before_reset.checkpoint_path == after_reset.checkpoint_path
        and before_reset.checkpoint_sha256 == after_reset.checkpoint_sha256
        and before_reset.pinned_checkpoint_bytes == after_reset.pinned_checkpoint_bytes
        and before_reset.replay_action_path == after_reset.replay_action_path
        and before_reset.replay_action_sha256 == after_reset.replay_action_sha256
        and before_reset.replay_action_count == after_reset.replay_action_count
        and before_reset.pinned_replay_action_bytes
        == after_reset.pinned_replay_action_bytes
        and before_reset.object_roi == after_reset.object_roi
        and before_reset.native_build == after_reset.native_build
    )
    if not unchanged:
        raise SupervisedV94RuntimeError(
            "deployment inputs changed during automatic reset; inference refused"
        )


def _supervised_runtime_completed_requested_work(
    result: Optional[Mapping[str, Any]],
    *,
    requested_steps: int,
    replay_source_completion_is_authoritative: bool,
) -> bool:
    """Validate command-count or transactional replay completion."""

    if result is None:
        return False
    if replay_source_completion_is_authoritative:
        # Online visual planning deliberately decouples runtime ticks from
        # template indices.  After its audited lift it may jump over identical
        # hold samples to the final template frame.  The committed replay
        # frame count proves template completion; the dual-ACK ledger must
        # still be contiguous through every command that was actually issued.
        return bool(
            result.get("replay_source_complete") is True
            and int(result.get("completed_replay_frames", -1))
            == int(requested_steps)
            and int(result.get("last_dual_ack_sequence", -1))
            == int(result.get("completed_policy_steps", -2))
            and result.get("stopped_early") is False
        )
    return bool(
        int(result.get("completed_policy_steps", -1)) == int(requested_steps)
        and int(result.get("last_dual_ack_sequence", -1))
        == int(requested_steps)
        and result.get("stopped_early") is False
    )


def run_supervised_v94(
    request: Any,
    *,
    preparation: Optional[PreparedSupervisedV94] = None,
    execution_reset: Optional[V94ExecutionResetProof] = None,
    prewarmed_camera_handoff: Optional[PrewarmedD435CameraHandoff] = None,
) -> Mapping[str, object]:
    before_reset = (
        prepare_supervised_v94(request) if preparation is None else preparation
    )
    if execution_reset is None:
        execution_reset = run_v94_execution_reset(
            profile=before_reset.profile,
            profile_path=Path(request.profile),
            bundle_path=Path(request.bundle),
            contract_q_home_rad=before_reset.contract.q_home_rad,
            maximum_franka_start_delta_rad=(RESET_MAX_FRANKA_START_DELTA_RAD),
        )
    # The reset can take several seconds.  Re-validate every mutable deployment
    # input after it, before a fresh hardware preflight or policy/runtime owner
    # is constructed.  Policy weights are loaded from the pinned bytes below,
    # never by reopening deploy.zip after this check.
    prepared = prepare_supervised_v94(request)
    _require_unchanged_preparation(before_reset, prepared)
    try:
        execution_reset = validate_v94_execution_reset_proof(
            execution_reset,
            contract_q_home_rad=prepared.contract.q_home_rad,
            maximum_start_delta_rad=RESET_MAX_FRANKA_START_DELTA_RAD,
        )
    except V94ExecutionResetError as exc:
        raise SupervisedV94RuntimeError(
            "automatic reset proof is missing or differs from the V94 start contract"
        ) from exc

    output = prepared.output
    profile = prepared.profile
    contract = prepared.contract
    policy_mode = resolve_policy_rate_mode(
        getattr(request, "policy_rate_hz", POLICY_RATE_HZ)
    )
    arrival_gated_replay = bool(
        prepared.pinned_replay_action_bytes is not None
        and getattr(request, "replay_arrival_gated", False)
    )
    arrival_gated_policy = bool(
        prepared.pinned_replay_action_bytes is None
        and getattr(request, "arrival_gated", False)
    )
    # ``request.steps`` counts trajectory frames.  Arrival-gated replay may
    # legally repeat the current exact target while Franka catches up, so seal
    # the session for the mode's complete command budget instead of pretending
    # one trajectory frame must equal one hardware transaction.
    runtime_command_budget = (
        policy_mode.maximum_supervised_steps
        if arrival_gated_replay
        else int(request.steps)
    )
    envelope = prepared.envelope
    object_roi = prepared.object_roi
    native_build = prepared.native_build
    live_visualization_requested = bool(
        getattr(request, "live_visualization", False)
    )
    record_video_path = getattr(request, "record_video_path", None)
    visual_output_pipeline_requested = bool(
        live_visualization_requested or record_video_path is not None
    )
    live_visualization_snapshot_directory = (
        Path(__file__).resolve().parents[2]
        / "dexgrasp"
        / "runs"
        / f"{request.run_id}_observation_visualization"
    )
    checkpoint_data = load_checkpoint_safely(prepared.pinned_checkpoint_bytes)
    fixed_sphere_completion_radius_m = infer_fixed_sphere_radius_m(
        getattr(checkpoint_data, "metadata", {})
    )
    if prepared.pinned_replay_action_bytes is None:
        policy = RollingStudentPolicy(checkpoint_data)
        action_source_kind = "checkpoint_policy"
        replay_source_format = None
        replay_trajectory_preview = None
    else:
        from sim2real.action_replay import (
            TransactionalReplayActionPolicy,
            load_replay_actions_payload,
            summarize_replay_actions,
        )

        assert prepared.replay_action_path is not None
        replay_sequence = load_replay_actions_payload(
            prepared.pinned_replay_action_bytes,
            suffix=prepared.replay_action_path.suffix,
            expected_policy_rate_hz=policy_mode.policy_rate_hz,
            selected_steps=int(request.steps),
        )
        point_feature_dim = int(checkpoint_data.spec.get("point_feature_dim"))
        if replay_sequence.tabletop_online_planner is not None:
            from motion_planning.online_tabletop import (
                TransactionalTabletopOnlinePlannerPolicy,
            )

            policy = TransactionalTabletopOnlinePlannerPolicy(
                replay_sequence,
                contract=contract,
                action_controller=ActionControllerParameters.from_metadata(
                    checkpoint_data.metadata
                ),
                point_feature_dim=point_feature_dim,
                history_length=int(checkpoint_data.spec.get("history")),
                proprio_dim=int(checkpoint_data.spec.get("proprio_dim")),
                selected_steps=int(request.steps),
            )
            action_source_kind = "online_visual_intercept_planner"
        else:
            policy = TransactionalReplayActionPolicy(
                replay_sequence,
                point_feature_dim=point_feature_dim,
                selected_steps=int(request.steps),
                arrival_gated=bool(
                    getattr(request, "replay_arrival_gated", False)
                ),
                arrival_tolerance_rad=float(
                    getattr(request, "replay_arrival_tolerance_rad", 0.015)
                ),
                q_home_rad=contract.q_home_rad,
            )
            action_source_kind = "validated_simulation_action_replay"
        replay_source_format = replay_sequence.source_format
        replay_trajectory_preview = summarize_replay_actions(
            replay_sequence,
            selected_steps=int(request.steps),
        )
    # Test doubles and legacy transactional policies default to the original
    # V94 XYZRGB contract; real RollingStudentPolicy instances always expose
    # both attributes explicitly.
    policy_point_feature_dim = int(getattr(policy, "point_feature_dim", 6))
    policy_point_feature_mode = str(
        getattr(
            policy,
            "point_feature_mode",
            "xyz" if policy_point_feature_dim == 3 else "xyzrgb",
        )
    )
    policy_io_recorder: Optional[PolicyIORecorder] = None
    if bool(getattr(request, "record_policy_io", False)):
        if not isinstance(policy, RollingStudentPolicy):
            raise SupervisedV94RuntimeError(
                "--record-policy-io requires checkpoint-policy inference; "
                "exact-action replay has a different history/normalization contract"
            )
        action_controller = getattr(policy, "action_controller", None)
        policy_io_recorder = PolicyIORecorder(
            default_policy_io_path(request.run_id),
            maximum_records=(
                int(request.steps) + QD_G015_STARTUP_NON_ACTUATED_POLICY_STEPS
            ),
            pointcloud_mean=policy.point_mean,
            pointcloud_std=policy.point_std,
            proprio_mean=policy.proprio_mean,
            proprio_std=policy.proprio_std,
            q_hand_close_rad=contract.q_hand_close_rad,
            metadata={
                "schema_version": 1,
                "kind": "real_checkpoint_policy_io",
                "run_id": str(request.run_id),
                "checkpoint_sha256": str(prepared.checkpoint_sha256),
                "checkpoint_source": str(prepared.checkpoint_source),
                "policy_rate_hz": float(policy_mode.policy_rate_hz),
                "history_length": int(policy.history_length),
                "proprio_dim": int(policy.proprio_dim),
                "point_feature_dim": int(policy.point_feature_dim),
                "point_feature_mode": str(policy.point_feature_mode),
                "action_controller_contract_id": str(
                    getattr(action_controller, "contract_id", "unknown")
                ),
                "sample_timing": "pre_action",
                "lstm_state_contract": (
                    "zero_initialized_each_call_external_history_is_complete_state"
                ),
                "accepted_tick_contract": (
                    "startup_non_actuated_then_dual_ack_hardware_commits"
                ),
            },
        )
    franka = profile["franka"]
    inspire = profile["inspire"]
    try:
        rh56_bounds = load_v94_rh56_profile_command_bounds(
            profile,
            feedback_to_command_offset_units=(RH56_FEEDBACK_TO_COMMAND_OFFSET_UNITS),
        )
        rh56_force_set_g = load_commissioned_rh56_force_set_g(profile)
    except ValueError as exc:
        raise SupervisedV94RuntimeError(
            f"selected V94 RH56 profile is not commissioned: {exc}"
        ) from exc
    hardware = FreshHardwarePreflight(
        *(
            _fresh_franka_preflight(
                robot_ip=str(franka["ip"]),
                q_home_rad=contract.q_home_rad,
                maximum_start_error_rad=0.01,
                envelope=envelope,
            )
        ),
        *(
            _fresh_rh56_preflight(
                port=str(inspire["port"]),
                baud=int(inspire["baud"]),
                hand_id=int(inspire["hand_id"]),
                open_targets=inspire["open_targets"],
            )
        ),
    )

    issued = time.monotonic()
    authorization_id = str(uuid.uuid4())
    permit_payload = {
        "classification": "experimental_operator_supervised_non_c2",
        "run_id": request.run_id,
        "steps": int(request.steps),
        "policy_mode": {
            "name": policy_mode.name,
            "policy_rate_hz": policy_mode.policy_rate_hz,
            "control_dt_s": policy_mode.control_dt_s,
        },
        "authorization_id": authorization_id,
        "bundle_sha256": prepared.bundle_sha256,
        "policy_checkpoint": {
            "source": prepared.checkpoint_source,
            "path": (
                None
                if prepared.checkpoint_path is None
                else str(prepared.checkpoint_path)
            ),
            "sha256": prepared.checkpoint_sha256,
            "point_feature_mode": policy_point_feature_mode,
            "point_feature_dim": policy_point_feature_dim,
            "validation": (
                "bundle_hash_and_golden_replay"
                if prepared.checkpoint_source == "bundle_primary"
                else (
                    "safe_decode_structural_contract_and_offline_"
                    "observation_smoke;bundle_golden_action_equality_does_"
                    "not_apply"
                )
            ),
        },
        "action_source": {
            "kind": action_source_kind,
            "replay_path": (
                None
                if prepared.replay_action_path is None
                else str(prepared.replay_action_path)
            ),
            "replay_sha256": prepared.replay_action_sha256 or None,
            "available_replay_actions": prepared.replay_action_count or None,
            "selected_actions": int(request.steps),
            "source_format": replay_source_format,
            "execution_contract": (
                "exact_recorded_actuator_targets; Franka_V225_1khz_"
                "interpolator_plus_official_limiter_retained; "
                "RH56_host_slew_bypassed"
                if prepared.replay_action_path is not None
                else "transactional_checkpoint_policy"
            ),
            "recorded_actuator_targets": (
                "direct_transactional_targets"
                if prepared.replay_action_path is not None
                else None
            ),
            "trajectory_preview": replay_trajectory_preview,
            "advance_semantics": (
                (
                    "advance_only_after_exact_dual_ack_with_replay_defined_"
                    "arrival_boundaries"
                    if arrival_gated_replay
                    else "advance_only_after_exact_dual_device_ack"
                )
                if prepared.replay_action_path is not None
                else (
                    "next_inference_only_after_dual_ack_and_measured_"
                    "franka_target_arrival"
                    if arrival_gated_policy
                    else "transactional_checkpoint_policy"
                )
            ),
            "arrival_gated": bool(
                arrival_gated_replay or arrival_gated_policy
            ),
            "arrival_gate_scope": (
                "franka_only_before_next_checkpoint_inference"
                if arrival_gated_policy
                else (
                    "franka_exact_replay_target_before_frame_advance"
                    if arrival_gated_replay
                    else None
                )
            ),
            "arrival_tolerance_rad": (
                float(getattr(request, "replay_arrival_tolerance_rad", 0.015))
                if arrival_gated_replay
                else (
                    float(getattr(request, "arrival_tolerance_rad", 0.005))
                    if arrival_gated_policy
                    else None
                )
            ),
            "arrival_timeout_s": (
                float(getattr(request, "arrival_timeout_s", 0.350))
                if arrival_gated_policy
                else None
            ),
            "maximum_hardware_commands": runtime_command_budget,
        },
        "profile_sha256": envelope.profile_sha256,
        "profile_file_sha256": prepared.profile_file_sha256,
        "pcd_config_sha256": prepared.pcd_config_sha256,
        "object_mask_mode": str(
            getattr(request, "object_mask_mode", "guarded")
        ),
        "requested_object_mask_mode": str(
            getattr(request, "object_mask_mode", "guarded")
        ),
        "effective_object_mask_mode": (
            "guarded_v2"
            if str(getattr(request, "object_mask_mode", "guarded"))
            == "guarded"
            else str(getattr(request, "object_mask_mode", "guarded"))
        ),
        "native_franka_servo": _jsonable(native_build),
        "object_roi": _jsonable(object_roi),
        "execution_reset": _jsonable(execution_reset),
        "fresh_hardware": _jsonable(hardware),
        "franka_native_dynamic_state": {
            "maximum_command_velocity_rad_s": 0.50,
            "maximum_measured_velocity_rad_s": (FRANKA_MAX_MEASURED_VELOCITY_RAD_S),
        },
        "franka_observation_timing": {
            "state_decimation": int(native_build.state_decimation),
            "nominal_state_rate_hz": (1000.0 / float(native_build.state_decimation)),
            "action_maximum_state_age_s": (FRANKA_OBSERVATION_ACTION_MAX_AGE_S),
            "hard_maximum_state_age_s": (FRANKA_OBSERVATION_HARD_MAX_AGE_S),
            "maximum_camera_pose_skew_s": 0.025,
            "stale_action_behavior": "side_effect_free_no_stage_retry",
        },
        "object_perception_contract": {
            "camera_session_lifecycle": (
                "camera_only_preflight_owner_continues_into_rollout"
                if prewarmed_camera_handoff is not None
                else "runtime_opens_fresh_camera_owner"
            ),
            "stale_palm": (
                "legacy_explicit_compatibility_previous_native_palm_cloud"
                if str(getattr(request, "object_mask_mode", "guarded"))
                == "legacy"
                else "recoverable_fail_closed_before_policy_history_mapper_stage"
            ),
            "stale_palm_legacy_compatibility_switch": (
                "explicit_--object-mask-mode=legacy_only"
            ),
            "liveness_frame": "current_formal_camera_frame",
            "local_stale_cloud_content_age_limit_s": (
                None
                if str(getattr(request, "object_mask_mode", "guarded"))
                == "legacy"
                else "not_applicable_stale_cloud_is_not_policy_input"
            ),
            "online_sam2": (
                "enabled_exact_frame_with_adaptive_identity_depth_recovery_gates"
                if str(getattr(request, "object_mask_mode", "guarded"))
                != "legacy"
                else "enabled_exact_frame_legacy_semantic_sam2_only"
            ),
            "mask_publication_mode": (
                "semantic_sam2"
                if str(getattr(request, "object_mask_mode", "guarded"))
                == "legacy"
                else "adaptive_fusion"
            ),
            "requested_object_mask_mode": str(
                getattr(request, "object_mask_mode", "guarded")
            ),
            "effective_object_mask_mode": (
                "guarded_v2"
                if str(getattr(request, "object_mask_mode", "guarded"))
                == "guarded"
                else str(getattr(request, "object_mask_mode", "guarded"))
            ),
            "effective_provider_mask_publication_mode": (
                "semantic_sam2"
                if str(getattr(request, "object_mask_mode", "guarded"))
                == "legacy"
                else "adaptive_fusion"
            ),
            "effective_provider_recovery_publication_mode": (
                "unified_three_evidence"
                if str(getattr(request, "object_mask_mode", "guarded"))
                in ("guarded", "guarded_v2")
                else (
                    "legacy_double_confirm"
                    if str(getattr(request, "object_mask_mode", "guarded"))
                    == "guarded_v1"
                    else "semantic_sam2_direct"
                )
            ),
        },
        "rh56_timing_contract": {
            "command_max_age_s": RH56_WATCHDOG_S,
            "inter_command_watchdog_s": RH56_INTER_COMMAND_WATCHDOG_S,
            "stop_timeout_s": RH56_STOP_TIMEOUT_S,
            "policy_rate_hz": policy_mode.policy_rate_hz,
            "target_rate_hz": policy_mode.rh56_target_rate_hz,
            "feedback_rate_hz": RH56_FEEDBACK_RATE_HZ,
            "feedback_hard_age_s": RH56_FEEDBACK_HARD_AGE_S,
            "speed_set": RH56_SPEED_SET,
            "force_set_g": rh56_force_set_g,
            "maximum_running_axis_current_ma": (RH56_MAX_RUNNING_CURRENT_MA),
            "stop_settle_max_axis_current_ma": (RH56_STOP_SETTLE_MAX_AXIS_CURRENT_MA),
            "post_disable_max_axis_current_ma": 100,
            "maximum_register_delta_per_update": (
                policy_mode.rh56_max_register_delta_per_update
            ),
            "minimum_angle_set_register_order": (
                rh56_bounds.minimum_angle_set_register_order
            ),
            "maximum_angle_set_register_order": (
                rh56_bounds.maximum_angle_set_register_order
            ),
            "feedback_to_command_offset_units": (RH56_FEEDBACK_TO_COMMAND_OFFSET_UNITS),
            "disabled_open_tolerance_units": (RH56_DISABLED_OPEN_TOLERANCE_UNITS),
            "tracking_significant_gap_units": (RH56_TRACKING_SIGNIFICANT_GAP_UNITS),
            "tracking_min_progress_units": (RH56_TRACKING_MIN_PROGRESS_UNITS),
            "tracking_contact_force_delta_g": (
                RH56_TRACKING_CONTACT_FORCE_DELTA_G
            ),
            "tracking_contact_force_absolute_g": (
                RH56_TRACKING_CONTACT_FORCE_ABSOLUTE_G
            ),
            "tracking_timeout_s": RH56_TRACKING_TIMEOUT_S,
        },
        "compute_isolation": {
            "compute_threads": int(getattr(request, "compute_threads", 1)),
            "parent_cpu_affinity": list(getattr(request, "parent_cpu_affinity", ())),
            "franka_servo_cpu": getattr(request, "franka_servo_cpu", None),
            "franka_servo_sibling_cpus": list(
                getattr(request, "franka_servo_sibling_cpus", ())
            ),
            "franka_servo_idle_sibling_cpus": list(
                getattr(request, "franka_servo_idle_sibling_cpus", ())
            ),
            "franka_nic_interface": str(getattr(request, "franka_nic_interface", "")),
            "franka_nic_irq_numbers": list(
                getattr(request, "franka_nic_irq_numbers", ())
            ),
            "franka_nic_irq_cpus": list(getattr(request, "franka_nic_irq_cpus", ())),
            "franka_nic_irq_requested_cpus": list(
                getattr(request, "franka_nic_irq_requested_cpus", ())
            ),
            "franka_nic_reserved_cpus": list(
                getattr(request, "franka_nic_reserved_cpus", ())
            ),
            "franka_nic_irq_stability_token": str(
                getattr(request, "franka_nic_irq_stability_token", "")
            ),
            "visualization_cpus": list(getattr(request, "visualization_cpus", ())),
        },
        "live_visualization": {
            "requested": live_visualization_requested,
            "background_output_pipeline_enabled": (
                visual_output_pipeline_requested
            ),
            "update_rate_hz": float(
                getattr(request, "live_visualization_rate_hz", 10.0)
            ),
            "record_video_rate_hz": (
                None
                if record_video_path is None
                else min(float(policy_mode.policy_rate_hz), 30.0)
            ),
            "content": "exact_frame_final_policy_mask_and_128_point_cloud",
            "save_after_stop": visual_output_pipeline_requested,
            "save_directory": (
                str(live_visualization_snapshot_directory)
                if visual_output_pipeline_requested
                else None
            ),
            "record_video_path": (
                None if record_video_path is None else str(record_video_path)
            ),
            "record_mask_video_path": (
                None
                if record_video_path is None
                else str(mask_video_path_for_recording(record_video_path))
            ),
            "record_video_overlay": (
                None
                if record_video_path is None
                else "none"
            ),
        },
    }
    permit_sha256 = hashlib.sha256(
        json.dumps(permit_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    authorization = MotionAuthorization(
        run_id=request.run_id,
        authorization_id=authorization_id,
        issued_monotonic_s=issued,
        expires_monotonic_s=issued + AUTHORIZATION_LIFETIME_S,
    )
    gate = ClosedLoopSafetyGate()
    # These are software gate inputs bound to the explicit supervised-execute
    # admission and fresh preflights; no physical-interlock claim is made.
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        authorization,
        run_id=request.run_id,
        now_monotonic_s=time.monotonic(),
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    franka_token = _issue_supervised_franka_preflight_token(
        envelope=envelope,
        run_id=request.run_id,
        confirmed_permit_sha256=permit_sha256,
        issued_monotonic_s=issued,
        expires_monotonic_s=authorization.expires_monotonic_s,
    )
    rh56_token = _issue_supervised_rh56_preflight(
        run_id=request.run_id,
        confirmed_permit_sha256=permit_sha256,
        commissioning_profile_sha256=envelope.profile_sha256,
        watchdog_timeout_s=RH56_WATCHDOG_S,
        inter_command_watchdog_timeout_s=RH56_INTER_COMMAND_WATCHDOG_S,
    )
    franka_action_contract_id = getattr(
        getattr(policy, "action_controller", None),
        "contract_id",
        LEGACY_FRANKA_ACTION_CONTRACT_ID,
    )
    admission = _seal_supervised_v94_admission(
        run_id=request.run_id,
        requested_policy_steps=runtime_command_budget,
        policy_rate_hz=policy_mode.policy_rate_hz,
        hard_deadline_monotonic_s=issued + HARD_RUNTIME_DEADLINE_S,
        authorization=authorization,
        franka_preflight_token=franka_token,
        rh56_preflight=rh56_token,
        franka_envelope=envelope,
        safety_gate=gate,
        franka_reference_q_rad=contract.q_home_rad,
        maximum_start_error_rad=0.01,
        maximum_tick_target_delta_rad=0.020,
        maximum_episode_delta_rad=(
            None
            if franka_action_contract_id
            == QD_G015_FRANKA_ACTION_CONTRACT_ID
            else RESET_MAX_FRANKA_START_DELTA_RAD
        ),
        franka_static_provenance_prevalidated=(
            hardware.franka_static_provenance_verified
        ),
        franka_action_contract_id=franka_action_contract_id,
        initial_previous_action13=getattr(
            policy,
            "initial_previous_action13",
            INITIAL_PREVIOUS_ACTION13,
        ),
    )

    if prewarmed_camera_handoff is None:
        camera_owner = D435ObjectCameraOwner(
            provider_factory=LiveD435ProviderFactory(
                config_path=Path(request.pcd_config),
                roi_xywh=object_roi.xywh,
                # The reviewed provider keeps online SAM2 asynchronous and
                # admits a result only for its exact RGB-D frame after the
                # adaptive identity/depth/recovery gates.
                disable_online_sam2=False,
                object_mask_mode=str(
                    getattr(request, "object_mask_mode", "guarded")
                ),
                required_runtime_frame_timeout_ms=(
                    D435_RUNTIME_FRAME_TIMEOUT_MS
                ),
            ),
            maximum_publication_stall_s=D435_FORMAL_PUBLICATION_STALL_S,
        )
        camera_owner_preopened = False
    else:
        if not isinstance(
            prewarmed_camera_handoff,
            PrewarmedD435CameraHandoff,
        ):
            raise SupervisedV94RuntimeError(
                "prewarmed camera handoff has the wrong runtime type"
            )
        camera_owner = prewarmed_camera_handoff.consume(
            contract=contract,
            roi_xywh=object_roi.xywh,
            object_mask_mode=str(
                getattr(request, "object_mask_mode", "guarded")
            ),
            pcd_config_sha256=prepared.pcd_config_sha256,
            checkpoint_sha256=prepared.checkpoint_sha256,
            preflight_sha256=object_roi.preflight_sha256,
        )
        camera_owner_preopened = True
        print(
            "[Object session handoff PASS] validated D435/SAM2 tracker "
            "continues into rollout without camera restart",
            flush=True,
        )
    live_visualizer = (
        V94LiveVisualizer(
            update_rate_hz=float(getattr(request, "live_visualization_rate_hz", 10.0)),
            cpu_affinity=getattr(request, "visualization_cpus", ()),
            show_live_windows=live_visualization_requested,
            save_directory=live_visualization_snapshot_directory,
            record_video_path=record_video_path,
            record_video_rate_hz=min(float(policy_mode.policy_rate_hz), 30.0),
            selected_bbox_xyxy=(
                int(object_roi.xywh[0]),
                int(object_roi.xywh[1]),
                int(object_roi.xywh[0] + object_roi.xywh[2]),
                int(object_roi.xywh[1] + object_roi.xywh[3]),
            ),
        )
        if visual_output_pipeline_requested
        else None
    )
    source_factory = ProductionV94PolicyTickSourceFactory(
        contract=contract,
        policy=policy,
        camera_owner=camera_owner,
        camera_owner_preopened=camera_owner_preopened,
        maximum_pose_skew_s=0.025,
        maximum_franka_action_age_s=(FRANKA_OBSERVATION_ACTION_MAX_AGE_S),
        maximum_franka_age_s=FRANKA_OBSERVATION_HARD_MAX_AGE_S,
        # Policy semantics follow the selected 20/60 Hz checkpoint mode.  The
        # physical RH56 target remains a transactional 20 Hz stream, and
        # complete feedback is sampled at 20 Hz by the same serial owner.
        maximum_rh56_age_s=RH56_FEEDBACK_HARD_AGE_S,
        maximum_future_skew_s=0.010,
        # The finite-difference window must include normal scheduler/serial
        # jitter beyond the nominal 50 ms feedback period; freshness remains
        # independently capped at 150 ms.
        maximum_velocity_dt_s=RH56_FEEDBACK_HARD_AGE_S,
        camera_poll_interval_s=0.001,
        maximum_object_pointcloud_age_s=OBJECT_POINTCLOUD_MAX_AGE_S,
        maximum_actions_per_camera_frame=(
            policy_mode.camera_max_policy_actions_per_frame
        ),
        policy_rgbd_resolution=str(
            getattr(request, "policy_rgbd_resolution", "848x480")
        ),
        requested_object_mask_mode=str(
            getattr(request, "object_mask_mode", "guarded")
        ),
        effective_object_mask_mode=(
            "guarded_v2"
            if str(getattr(request, "object_mask_mode", "guarded"))
            == "guarded"
            else str(getattr(request, "object_mask_mode", "guarded"))
        ),
        franka_arrival_gate_enabled=arrival_gated_policy,
        franka_arrival_gate_tolerance_rad=float(
            getattr(request, "arrival_tolerance_rad", 0.005)
        ),
        franka_arrival_gate_timeout_s=float(
            getattr(request, "arrival_timeout_s", 0.350)
        ),
        live_visualizer=live_visualizer,
        policy_io_recorder=policy_io_recorder,
        fixed_sphere_completion_radius_m=fixed_sphere_completion_radius_m,
        rollout_trigger_config=task_profile_rollout_trigger(
            Path(request.pcd_config)
        ),
        rh56_hardware_rate_hz=policy_mode.rh56_target_rate_hz,
        rh56_maximum_register_delta_per_update=(
            policy_mode.rh56_max_register_delta_per_update
        ),
        rh56_minimum_angle_set_register_order=(
            rh56_bounds.minimum_angle_set_register_order
        ),
        rh56_maximum_angle_set_register_order=(
            rh56_bounds.maximum_angle_set_register_order
        ),
        rh56_feedback_to_command_offset_units=(RH56_FEEDBACK_TO_COMMAND_OFFSET_UNITS),
        rh56_feedback_to_command_valid_min=(rh56_bounds.feedback_to_command_valid_min),
        rh56_feedback_to_command_valid_max=(rh56_bounds.feedback_to_command_valid_max),
    )
    runtime_factory = SupervisedV94RuntimeFactory(
        supervised_franka_session_factory=_make_native_franka_session_factory(
            build=native_build,
            robot_ip=str(franka["ip"]),
            servo_cpus=(
                (int(request.franka_servo_cpu),)
                if getattr(request, "franka_servo_cpu", None) is not None
                else ()
            ),
            prelaunch_validator=getattr(
                request, "franka_native_prelaunch_validator", None
            ),
        ),
        rh56_watchdog_owner_factory=_make_rh56_owner_factory(
            port=str(inspire["port"]),
            baud=int(inspire["baud"]),
            hand_id=int(inspire["hand_id"]),
            speed=RH56_SPEED_SET,
            force=rh56_force_set_g,
            open_targets=inspire["open_targets"],
            minimum_angle_set_register_order=(
                rh56_bounds.minimum_angle_set_register_order
            ),
            maximum_angle_set_register_order=(
                rh56_bounds.maximum_angle_set_register_order
            ),
        ),
        policy_tick_source_factory=source_factory,
        live_safety_supervisor=_OperatorSupervisedBoundary(),
        # The proxy may spend up to three seconds each obtaining STOP_PROOF,
        # waiting for critical EOF, joining, and joining once more after a
        # forced terminate.  Leave one second of outer scheduling margin.
        franka_stop_join_timeout_s=13.0,
        commit_poll_interval_s=0.0005,
        maximum_consecutive_no_stage_hold_s=(MAXIMUM_CONSECUTIVE_NO_STAGE_HOLD_S),
    )
    runtime = runtime_factory(admission)
    result: Optional[Mapping[str, Any]] = None
    failure: Optional[BaseException] = None
    stop_proof = None
    gc_was_enabled = gc.isenabled()
    gc.collect()
    if gc_was_enabled:
        # The bounded run lasts at most eight seconds.  Reference counting
        # still releases arrays immediately; deferring cyclic collection
        # prevents a visualization-heavy generation sweep from creating a
        # >100 ms policy-target gap.
        gc.disable()
    try:
        result = runtime.run(
            maximum_policy_steps=runtime_command_budget,
            hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
            stop_requested=threading.Event(),
        )
    except BaseException as exc:
        failure = exc
    finally:
        try:
            runtime.request_stop("supervised terminal cleanup")
            try:
                stop_proof = runtime.stop_and_verify()
            except BaseException as exc:
                if failure is None:
                    failure = exc
        finally:
            if gc_was_enabled:
                gc.enable()
    failure = _prefer_terminal_owner_fault(failure, runtime)
    stopped = bool(
        stop_proof is not None
        and stop_proof.franka_stop_verified
        and stop_proof.rh56_disabled_verified
        and runtime.stop_errors == ()
    )
    policy_io_log: dict[str, object] = {
        "requested": policy_io_recorder is not None,
        "path": (
            None
            if policy_io_recorder is None
            else str(default_policy_io_path(request.run_id))
        ),
        "records": 0,
        "written_after_stop_attempt": True,
        "dual_device_stop_verified_before_write": bool(stopped),
        "error": None,
    }
    if policy_io_recorder is not None:
        try:
            policy_io_log.update(dict(policy_io_recorder.save()))
            print(
                "[Policy I/O] "
                f"saved={policy_io_log.get('path')} "
                f"ticks={policy_io_log.get('records', 0)}",
                flush=True,
            )
        except Exception as exc:
            # Diagnostics are explicitly fail-soft.  This runs only after both
            # owners have received their stop request; a disk/logging problem
            # must not rewrite the authoritative hardware outcome.
            policy_io_log["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[Policy I/O][WARN] {policy_io_log['error']}",
                file=sys.stderr,
                flush=True,
            )
    online_visual_replay = action_source_kind == "online_visual_intercept_planner"
    complete = _supervised_runtime_completed_requested_work(
        result,
        requested_steps=int(request.steps),
        replay_source_completion_is_authoritative=bool(
            arrival_gated_replay or online_visual_replay
        ),
    )
    tracking = None if result is None else result.get("rh56_physical_tracking")
    tracking_verdict = (
        str(tracking.get("verdict", "")).strip()
        if isinstance(tracking, Mapping)
        else ""
    )
    tracking_acceptable = tracking_verdict in {
        "not_exercised",
        "verified",
    }
    if failure is None and complete and not tracking_acceptable:
        failure = SupervisedV94RuntimeError(
            "RH56 physical tracking was challenged but not verified: "
            f"verdict={tracking_verdict or 'missing'}"
        )
    elif failure is None and not complete:
        failure = SupervisedV94RuntimeError(
            "arrival-gated replay exhausted its hardware-command budget before "
            "the final Franka target arrived"
            if arrival_gated_replay
            else "runtime returned without every requested dual-ACK policy step"
        )
    committed_commands = tuple(getattr(runtime, "committed_commands_snapshot", ()))
    runtime_ledger = getattr(runtime, "action_ledger", None)
    last_dual_ack_sequence = getattr(
        runtime_ledger,
        "last_committed_sequence",
        None,
    )
    if last_dual_ack_sequence is None and committed_commands:
        last_dual_ack_sequence = max(
            int(item.get("sequence", 0)) for item in committed_commands
        )
    if last_dual_ack_sequence is None:
        last_dual_ack_sequence = (
            int(result.get("last_dual_ack_sequence", 0)) if result is not None else 0
        )
    runtime_diagnostics = getattr(
        runtime,
        "runtime_diagnostics_snapshot",
        {},
    )
    rh56_force_log_path = output.with_name(
        f"{output.stem}_rh56_force.csv"
    )
    rh56_force_log_rows = 0
    rh56_force_log_error: Optional[str] = None
    try:
        rh56_force_log_rows = _write_rh56_force_csv(
            rh56_force_log_path,
            committed_commands,
        )
        print(
            "[RH56 force log] "
            f"{rh56_force_log_path} rows={rh56_force_log_rows}",
            flush=True,
        )
    except BaseException as exc:
        # This happens only after both owners have been asked to stop.  Keep
        # the hardware result while exposing the post-run logging failure in
        # the authoritative JSON audit.
        rh56_force_log_error = f"{type(exc).__name__}: {exc}"
        print(
            f"[RH56 force log][WARN] {rh56_force_log_error}",
            file=sys.stderr,
            flush=True,
        )
    audit = {
        "schema_version": 1,
        "kind": (
            "v94_action_replay_real_non_c2"
            if prepared.replay_action_path is not None
            else "v94_supervised_real_non_c2"
        ),
        "classification": admission.classification,
        "authorizes_c2": False,
        "run_id": request.run_id,
        "requested_steps": int(request.steps),
        "result": "PASS" if failure is None and stopped and complete else "FAIL",
        "permit": permit_payload,
        "permit_sha256": permit_sha256,
        "runtime": result,
        "native_franka_servo": native_build,
        "object_roi": object_roi,
        "execution_reset": execution_reset,
        "live_visualization": (
            {"requested": False}
            if live_visualizer is None
            else dict(live_visualizer.stats())
        ),
        "fresh_hardware_preflight": hardware,
        "stop_proof": stop_proof,
        "stop_errors": runtime.stop_errors,
        "franka_telemetry": runtime.franka_telemetry,
        # Retained independently of runtime.run()'s normal return mapping so
        # terminal hardware faults still disclose which observation holds and
        # RH56 command-window waits preceded the first fault.
        "runtime_diagnostics": runtime_diagnostics,
        "rh56_force_log": {
            "path": str(rh56_force_log_path),
            "rows": int(rh56_force_log_rows),
            "axis_order": list(_RH56_FORCE_AXIS_NAMES),
            "force_raw_unit": "gf",
            "newton_conversion": _GRAM_FORCE_TO_NEWTON,
            "sampling_semantics": (
                "one_cached_20hz_feedback_snapshot_at_each_successful_"
                "dual_ack_command_commit"
            ),
            "written_after_stop_attempt": True,
            "dual_device_stop_verified_before_write": bool(stopped),
            "error": rh56_force_log_error,
        },
        "policy_io": policy_io_log,
        # These fields remain populated on a FAIL after dual ACK, even when
        # runtime.run() could not return its normal result mapping.
        "last_dual_ack_sequence": int(last_dual_ack_sequence),
        "committed_commands": committed_commands,
        "failure_pending_command": (
            getattr(runtime, "failure_pending_command", None)
            if failure is not None
            else None
        ),
        "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
        "warning": (
            "operator-supervised non-C2 run; RH56 target readback and physical "
            "ANGLE_ACT/POS_ACT first-motion proof are enforced; later healthy "
            "contact holds are allowed"
        ),
    }
    _write_audit(output, audit)
    if failure is not None:
        if isinstance(failure, KeyboardInterrupt):
            raise failure
        raise SupervisedV94RuntimeError(
            f"supervised runtime failed; audit={output}: {type(failure).__name__}: {failure}"
        ) from failure
    if not stopped:
        raise SupervisedV94RuntimeError(
            f"dual-device verified stop/restore failed; audit={output}"
        )
    if gate.state is SafetyState.ARMED:
        gate.disarm_after_verified_stop(
            run_id=request.run_id,
            franka_stop_verified=True,
            rh56_disabled_verified=True,
        )
    return {
        "result": "PASS",
        "classification": admission.classification,
        "authorizes_c2": False,
        "completed_policy_steps": int(result["completed_policy_steps"]),
        "audit": str(output),
        "rh56_force_log": (
            None
            if rh56_force_log_error is not None
            else str(rh56_force_log_path)
        ),
        "policy_io": policy_io_log,
        "franka_stop_verified": True,
        "rh56_disabled_verified": True,
        "rh56_speed_force_restored": True,
        "object_roi_xywh": list(object_roi.xywh),
        "interactive_roi_gui_used": bool(
            getattr(request, "interactive_roi_gui_used", False)
        ),
        "live_visualization": (
            {"requested": False}
            if live_visualizer is None
            else dict(live_visualizer.stats())
        ),
        "native_franka_servo_binary_sha256": native_build.binary_sha256,
        "checkpoint_source": prepared.checkpoint_source,
        "checkpoint_sha256": prepared.checkpoint_sha256,
        "action_source": action_source_kind,
        "replay_action_sha256": prepared.replay_action_sha256 or None,
        "replay_action_count": prepared.replay_action_count or None,
        "replay_arrival_gated": arrival_gated_replay,
        "closed_loop_arrival_gated": arrival_gated_policy,
        "franka_arrival_gate": _jsonable(
            result.get("observation_source", {}).get(
                "franka_arrival_gate", {}
            )
            if isinstance(result.get("observation_source"), Mapping)
            else {}
        ),
        "final_franka_arrival_gate": _jsonable(
            result.get("final_franka_arrival_gate")
        ),
        "completed_replay_frames": int(
            result.get("completed_replay_frames", 0)
        ),
        "repeated_replay_target_commands": int(
            result.get("repeated_replay_target_commands", 0)
        ),
        "point_feature_mode": policy_point_feature_mode,
        "point_feature_dim": policy_point_feature_dim,
        "execution_reset": _jsonable(execution_reset),
        "rh56_physical_tracking": _jsonable(tracking),
        "camera_session_lifecycle": (
            "camera_only_preflight_owner_continues_into_rollout"
            if prewarmed_camera_handoff is not None
            else "runtime_opens_fresh_camera_owner"
        ),
    }


__all__ = [
    "FreshHardwarePreflight",
    "PreparedSupervisedV94Artifacts",
    "SupervisedV94RuntimeError",
    "VerifiedObjectROI",
    "prepare_supervised_v94",
    "prepare_supervised_v94_artifacts",
    "run_supervised_v94",
]
