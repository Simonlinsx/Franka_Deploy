#!/usr/bin/env python3
"""Supervised C1 probe of the V94 J1 action-to-real-control mapping.

The default invocation is hardware-free.  Real execution requires
``--execute`` plus exact, run-scoped operator confirmations.  The command
keeps the installed RH56 exclusively open, verifies all six axes are already
open/idle/disabled, and never writes an RH56 register.

For Franka it reuses the reviewed :class:`FrankaSequenceDriver` used by the
installed reset path.  A fresh Idle state supplies ``q0``.  The production
``V94ActionMapper`` then maps ``action[0]=+1`` to the exact float32-compatible
target near ``q0[0] + 0.003 rad``.  A dedicated C1 motion envelope uses an
arrival tolerance of 0.0002 rad (1/15 of the step), so the generic 0.02 rad
reset tolerance cannot
silently swallow the probe.  Fresh Idle/q/dq samples prove the outbound motion
and the return to exact bundle ``q_home``.  The control-loop telemetry must also
prove that a joint control handle actually emitted samples for both legs.

This validates only one stop-to-stop target mapping and sign.  It does not
commission or validate the checkpoint's 60 Hz closed-loop dynamics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydex_pipeline.control_config import (  # noqa: E402
    load_control_config,
    verify_adapter_assets,
)
from anydex_pipeline.host_network_preflight import (  # noqa: E402
    require_uncontended_franka_https_link,
)
from sim2real.deployment.franka_action_audit import (  # noqa: E402
    DEFAULT_PROFILE as V94_MAPPING_PROFILE,
    audit_v94_franka_action_mapping,
)
from sim2real.deployment.bundle import DeployBundle  # noqa: E402
from sim2real.contracts.actions import V94ActionMapper  # noqa: E402
from sim2real.contracts.v94 import V94Contract  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_BUNDLE = WORKSPACE / "data/test_fixtures/sim2real/deploy.zip"

INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
HAND_OPEN_TOKEN = "RH56_OPEN_DISABLED"
WORKSPACE_TOKEN = "FR3_J1_PLUS_003_AND_RETURN_SWEEP_CLEAR"
STOP_TOKEN = "FR3_PHYSICAL_STOP_READY"
PLA_TOKEN = "PLA_LOW_SPEED_UNLOADED_ONLY"
MAPPING_TOKEN = "V94_ACTION0_PLUS_ONE_MAPS_POSITIVE_J1"
C1_SCOPE_TOKEN = "C1_SINGLE_STEP_ONLY_NOT_60HZ_C2"
AUTHORIZATION_PREFIX = "V94_J1_PLUS_003_RETURN_C1:"
RECOVERY_AUTHORIZATION_PREFIX = "V94_EXACT_Q_HOME_RECOVERY_C1:"
RECOVERY_WORKSPACE_TOKEN = "FR3_CURRENT_TO_V94_HOME_SWEEP_CLEAR"
RECOVERY_TARGET_TOKEN = "V94_EXACT_Q_HOME_RECOVERY"

MICROPROBE_JOINT_INDEX = 0
MICROPROBE_JOINT_NAME = "panda_joint1"
MICROPROBE_ACTION_VALUE = 1.0
MICROPROBE_NOMINAL_DELTA_RAD = 0.003
MICROPROBE_MAPPING_ATOL_RAD = 2.0e-6
MICROPROBE_RESET_ENVELOPE_RAD = 0.05
# The first physical endpoint residual was 0.000101378 rad.  A 0.0002 rad
# endpoint band gives just under 2x margin over that measured residual while
# remaining exactly 1/15 (6.67%) of the 0.003 rad probe step.  Mapping math
# remains guarded separately at 2e-6 rad.
MICROPROBE_ARRIVAL_TOLERANCE_RAD = 2.0e-4
MICROPROBE_START_ALIGNMENT_TOLERANCE_RAD = 2.0e-4
MICROPROBE_MAX_SPEED_RAD_S = 0.05
MICROPROBE_MAX_SEGMENT_RAD = 0.004
MICROPROBE_MAX_SETTLED_DQ_RAD_S = 0.005
MICROPROBE_MAX_CONTINUOUS_TRACKING_ERROR_RAD = 0.005
MICROPROBE_PROOF_SAMPLES = 3
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{5,79}$")


class MicroprobeStopUnconfirmed(RuntimeError):
    """The command could not prove Franka stopped after requesting stop."""


@dataclass(frozen=True)
class V94J1MappedTarget:
    action13: np.ndarray
    measured_q_rad: np.ndarray
    origin_q_rad: np.ndarray
    peak_q_rad: np.ndarray
    target_delta_rad: float
    measured_to_origin_quantization_rad: float
    reset_linf_error_rad: float


@dataclass(frozen=True)
class V94FrankaMicroprobeResult:
    run_id: str
    target_delta_rad: float
    outbound_measured_delta_rad: float
    outbound_target_error_rad: float
    return_target_error_rad: float
    outbound_control_samples: int
    return_control_samples: int
    maximum_read_to_write_us: float
    franka_stop_verified: bool
    rh56_final_disabled_verified: bool
    rh56_command_writes: int
    evidence: Mapping[str, Any]
    scope: str = "C1_single_stop_to_stop_mapping_only_not_60Hz_C2"


@dataclass(frozen=True)
class V94FrankaRecoveryResult:
    run_id: str
    initial_home_error_rad: float
    final_home_error_rad: float
    control_samples: int
    maximum_read_to_write_us: float
    franka_stop_verified: bool
    rh56_final_disabled_verified: bool
    rh56_command_writes: int
    evidence: Mapping[str, Any]
    scope: str = "C1_exact_v94_q_home_recovery_only"


def _load_default_reset_module() -> Any:
    source = Path(__file__).with_name("reset_franka_default.py")
    spec = importlib.util.spec_from_file_location(
        "dexgrasp_v94_microprobe_reset_helpers", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reviewed installed reset helpers: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DEFAULT_RESET = _load_default_reset_module()


def _load_hardware_types() -> Mapping[str, Any]:
    """Lazy hardware imports reached only after every exact token passes."""

    return DEFAULT_RESET._load_hardware_types()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Supervised C1 V94 action[0]=+1 -> J1 +~0.003rad -> return probe; "
            "default is hardware-free"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--port", help="override RH56 serial port from config")
    parser.add_argument(
        "--output",
        type=Path,
        help="exclusive JSON evidence path; default is dexgrasp/runs/<run-id>.json",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="run the supervised +0.003 rad J1 probe and exact-q_home return",
    )
    mode.add_argument(
        "--recover-to-v94-home",
        action="store_true",
        help="only recover Franka from the current safe envelope to exact V94 q_home",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--authorization-token")
    parser.add_argument("--confirm-installed", metavar=INSTALLED_TOKEN)
    parser.add_argument("--confirm-hand-open", metavar=HAND_OPEN_TOKEN)
    parser.add_argument("--confirm-workspace-clear", metavar=WORKSPACE_TOKEN)
    parser.add_argument("--confirm-stop-ready", metavar=STOP_TOKEN)
    parser.add_argument("--confirm-pla-low-speed", metavar=PLA_TOKEN)
    parser.add_argument("--confirm-mapping", metavar=MAPPING_TOKEN)
    parser.add_argument("--confirm-c1-scope", metavar=C1_SCOPE_TOKEN)
    parser.add_argument(
        "--confirm-recovery-target", metavar=RECOVERY_TARGET_TOKEN
    )
    return parser


def _require_execute_authorization(args: argparse.Namespace) -> str:
    run_id = str(args.run_id or "").strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "--run-id must contain 6..80 letters/digits/._- and start alphanumeric"
        )
    expected_authorization = AUTHORIZATION_PREFIX + run_id
    required = (
        ("--authorization-token", args.authorization_token, expected_authorization),
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-hand-open", args.confirm_hand_open, HAND_OPEN_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, WORKSPACE_TOKEN),
        ("--confirm-stop-ready", args.confirm_stop_ready, STOP_TOKEN),
        ("--confirm-pla-low-speed", args.confirm_pla_low_speed, PLA_TOKEN),
        ("--confirm-mapping", args.confirm_mapping, MAPPING_TOKEN),
        ("--confirm-c1-scope", args.confirm_c1_scope, C1_SCOPE_TOKEN),
    )
    missing = [
        f"{flag} {token}"
        for flag, actual, token in required
        if actual != token
    ]
    if missing:
        raise ValueError(
            "exact run-scoped confirmations required before hardware import: "
            + "; ".join(missing)
        )
    return run_id


def _require_recovery_authorization(args: argparse.Namespace) -> str:
    run_id = str(args.run_id or "").strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "--run-id must contain 6..80 letters/digits/._- and start alphanumeric"
        )
    expected_authorization = RECOVERY_AUTHORIZATION_PREFIX + run_id
    required = (
        ("--authorization-token", args.authorization_token, expected_authorization),
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-hand-open", args.confirm_hand_open, HAND_OPEN_TOKEN),
        (
            "--confirm-workspace-clear",
            args.confirm_workspace_clear,
            RECOVERY_WORKSPACE_TOKEN,
        ),
        ("--confirm-stop-ready", args.confirm_stop_ready, STOP_TOKEN),
        ("--confirm-pla-low-speed", args.confirm_pla_low_speed, PLA_TOKEN),
        (
            "--confirm-recovery-target",
            args.confirm_recovery_target,
            RECOVERY_TARGET_TOKEN,
        ),
    )
    missing = [
        f"{flag} {token}"
        for flag, actual, token in required
        if actual != token
    ]
    if missing:
        raise ValueError(
            "exact run-scoped recovery confirmations required before hardware import: "
            + "; ".join(missing)
        )
    return run_id


def _validate_static_request(
    config: Mapping[str, Any], bundle_path: Path
) -> Mapping[str, Any]:
    DEFAULT_RESET._validate_static_request(config)
    franka = config["franka"]
    if float(franka["default_max_joint_velocity_rad_s"]) > MICROPROBE_MAX_SPEED_RAD_S:
        raise ValueError("profile speed exceeds the fixed 0.05rad/s C1 probe ceiling")
    tolerance_fraction = (
        MICROPROBE_ARRIVAL_TOLERANCE_RAD / MICROPROBE_NOMINAL_DELTA_RAD
    )
    if tolerance_fraction > (1.0 / 15.0) + 1.0e-12:
        raise ValueError("C1 endpoint tolerance exceeds 1/15 of the probe step")
    if config["tool"].get("installed_collision_model_verified") is not False:
        raise ValueError(
            "this supervised C1 path expects the explicit per-run clear-workspace "
            "authority used while the installed collision model remains uncommissioned"
        )
    bundle = DeployBundle(bundle_path)
    verification = bundle.verify()
    contract = V94Contract.from_bundle(bundle)
    profile_limits = np.asarray(franka["joint_limits_rad"], dtype=np.float64)
    if not np.array_equal(
        profile_limits, contract.joint_limits_rad.astype(np.float64)
    ):
        raise ValueError("V94 and reviewed C1 Franka joint limits differ")
    # Mapping/reset semantics are audited against the V94-specific profile.
    # The separate C1 control config contributes the already-reviewed physical
    # driver/tool limits below; its cable-friendly default q is intentionally
    # not the V94 training reset and must not be substituted into this audit.
    audit = audit_v94_franka_action_mapping(bundle_path, V94_MAPPING_PROFILE)
    if not audit.packaged_reference_mapping_confirmed:
        raise ValueError("V94 packaged Franka target mapping audit failed")
    if not audit.axis_order_identity or audit.sign_transform != "identity (no negation)":
        raise ValueError("V94 Franka action axis/sign mapping is not identity")
    if not np.isclose(
        audit.effective_gain_rad_per_tick,
        MICROPROBE_NOMINAL_DELTA_RAD,
        atol=MICROPROBE_MAPPING_ATOL_RAD,
        rtol=0.0,
    ):
        raise ValueError("V94 full-scale action no longer maps to ~0.003rad")
    return {
        "bundle_contract": verification.bundle_contract,
        "checkpoint_sha256": verification.primary_checkpoint_sha256,
        "effective_gain_rad_per_tick": audit.effective_gain_rad_per_tick,
        "axis_order_identity": audit.axis_order_identity,
        "sign_transform": audit.sign_transform,
    }


def _build_microprobe_limits(config: Mapping[str, Any], limits_type: Any) -> Any:
    franka = config["franka"]
    dynamics = franka["expected_end_effector"]
    return limits_type(
        expected_F_T_EE=np.asarray(franka["expected_F_T_EE"], dtype=np.float64),
        expected_m_ee_kg=float(dynamics["mass_kg"]),
        expected_F_x_Cee_m=np.asarray(dynamics["F_x_Cee_m"], dtype=np.float64),
        expected_I_ee_kg_m2=np.asarray(dynamics["inertia_kg_m2"], dtype=np.float64),
        joint_limits_rad=np.asarray(franka["joint_limits_rad"], dtype=np.float64),
        joint_limit_margin_rad=float(franka["joint_limit_margin_rad"]),
        max_joint_speed_rad_s=min(
            MICROPROBE_MAX_SPEED_RAD_S,
            float(franka["default_max_joint_velocity_rad_s"]),
        ),
        max_joint_segment_rad=MICROPROBE_MAX_SEGMENT_RAD,
        # Preserve the reviewed reset duration.  For 0.003 rad this is much
        # slower than the 0.05 rad/s ceiling and covers the full link-quality
        # observation window in FrankaMotionLimits.
        min_joint_duration_s=float(franka["default_min_duration_s"]),
        joint_arrival_tolerance_rad=MICROPROBE_ARRIVAL_TOLERANCE_RAD,
        max_continuous_joint_tracking_error_rad=(
            MICROPROBE_MAX_CONTINUOUS_TRACKING_ERROR_RAD
        ),
        settle_time_s=float(franka["settle_time_s"]),
        settle_timeout_s=float(franka["settle_timeout_s"]),
        settle_poll_s=float(franka["settle_poll_s"]),
        settle_max_dq_rad_s=MICROPROBE_MAX_SETTLED_DQ_RAD_S,
        stop_verify_max_dq_rad_s=MICROPROBE_MAX_SETTLED_DQ_RAD_S,
    )


def _map_live_j1_target(
    initial_q_rad: Sequence[float], *, contract: V94Contract
) -> V94J1MappedTarget:
    measured = np.asarray(initial_q_rad, dtype=np.float64)
    if measured.shape != (7,) or not np.all(np.isfinite(measured)):
        raise ValueError("fresh Franka q must contain seven finite radians")
    reset_error = float(
        np.max(
            np.abs(measured - contract.q_home_rad.astype(np.float64)), initial=0.0
        )
    )
    if reset_error > MICROPROBE_RESET_ENVELOPE_RAD:
        raise RuntimeError(
            "fresh Franka q is outside the conservative V94 reset envelope: "
            f"Linf={reset_error:.9f}rad > {MICROPROBE_RESET_ENVELOPE_RAD:.9f}rad"
        )
    if reset_error > MICROPROBE_START_ALIGNMENT_TOLERANCE_RAD:
        raise RuntimeError(
            "fresh Franka q is safe but not aligned tightly enough to V94 q_home: "
            f"Linf={reset_error:.9f}rad > "
            f"{MICROPROBE_START_ALIGNMENT_TOLERANCE_RAD:.9f}rad; "
            "run --recover-to-v94-home before another probe"
        )
    origin = measured.astype(np.float32)
    action = np.zeros(13, dtype=np.float32)
    action[MICROPROBE_JOINT_INDEX] = np.float32(MICROPROBE_ACTION_VALUE)
    action[7:] = np.float32(-1.0)
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=origin,
        initial_hand_target_q_policy_order_rad=np.zeros(6, dtype=np.float32),
        joint_limits_rad=contract.joint_limits_rad,
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    mapped = mapper.map(action, measured_q_rad=origin)
    peak = np.asarray(mapped.franka_target_q_rad, dtype=np.float64)
    origin64 = origin.astype(np.float64)
    delta_vector = peak - origin64
    target_delta = float(delta_vector[MICROPROBE_JOINT_INDEX])
    if target_delta <= 0.0 or not np.isclose(
        target_delta,
        MICROPROBE_NOMINAL_DELTA_RAD,
        atol=MICROPROBE_MAPPING_ATOL_RAD,
        rtol=0.0,
    ):
        raise RuntimeError(
            "V94 action[0]=+1 did not produce positive J1 ~0.003rad: "
            f"actual={target_delta:.9f}"
        )
    # The production mapper deliberately executes float32 EMA arithmetic.
    # ``0.2*q + 0.8*q`` can therefore differ from q by a few ulps even when an
    # action axis is zero.  Preserve the exact mapped target, but reject any
    # non-J1 change larger than the same 2e-6rad replay/mapping tolerance.
    if (
        np.max(np.abs(np.delete(delta_vector, MICROPROBE_JOINT_INDEX)))
        > MICROPROBE_MAPPING_ATOL_RAD
    ):
        raise RuntimeError("V94 J1 microprobe materially changed another arm axis")
    return V94J1MappedTarget(
        action13=action,
        measured_q_rad=measured.copy(),
        origin_q_rad=origin64,
        peak_q_rad=peak,
        target_delta_rad=target_delta,
        measured_to_origin_quantization_rad=float(
            np.max(np.abs(measured - origin64))
        ),
        reset_linf_error_rad=reset_error,
    )


def _state_q_dq(state: Any) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(getattr(state, "q", None), dtype=np.float64)
    dq = np.asarray(getattr(state, "dq", None), dtype=np.float64)
    if q.shape != (7,) or dq.shape != (7,) or not (
        np.all(np.isfinite(q)) and np.all(np.isfinite(dq))
    ):
        raise RuntimeError("Franka proof state q/dq is malformed")
    return q, dq


def _require_control_telemetry(arm: Any, *, label: str, previous: Any = None) -> Any:
    telemetry = getattr(arm, "last_control_loop_telemetry", None)
    if telemetry is None:
        raise RuntimeError(
            f"{label} produced no Franka control-loop telemetry; microstep may have been swallowed"
        )
    if telemetry is previous:
        raise RuntimeError(f"{label} did not create a distinct Franka control handle")
    if str(getattr(telemetry, "kind", "")) != "joint":
        raise RuntimeError(f"{label} telemetry is not a joint control segment")
    samples = int(getattr(telemetry, "samples", 0))
    if samples < 2:
        raise RuntimeError(f"{label} emitted fewer than two joint control samples")
    overruns = int(getattr(telemetry, "read_to_write_overruns", -1))
    if overruns != 0:
        raise RuntimeError(f"{label} had {overruns} FCI read-to-write overruns")
    return telemetry


def _read_motion_proof(
    arm: Any,
    *,
    expected_q_rad: np.ndarray,
    measured_start_q_rad: np.ndarray,
    expected_j1_delta_rad: float,
    label: str,
    sleep: Any,
    evidence: Optional[dict[str, Any]] = None,
) -> Mapping[str, Any]:
    worst_target_error = 0.0
    worst_other_axis_drift = 0.0
    maximum_dq = 0.0
    measured_j1_delta = 0.0
    proof = {} if evidence is None else evidence
    q_samples: list[list[float]] = []
    dq_samples: list[list[float]] = []
    proof.update(
        {
            "label": label,
            "expected_q_rad": expected_q_rad.tolist(),
            "measured_start_q_rad": measured_start_q_rad.tolist(),
            "expected_j1_delta_rad": float(expected_j1_delta_rad),
            "arrival_tolerance_rad": MICROPROBE_ARRIVAL_TOLERANCE_RAD,
            "completed": False,
            "q_samples_rad": q_samples,
            "dq_samples_rad_s": dq_samples,
        }
    )
    for index in range(MICROPROBE_PROOF_SAMPLES):
        state = arm.robot.read_once()
        arm._validate_state(state, require_idle=True, enforce_success=False)
        q, dq = _state_q_dq(state)
        q_samples.append(q.tolist())
        dq_samples.append(dq.tolist())
        target_error = float(np.max(np.abs(q - expected_q_rad)))
        measured_delta_vector = q - measured_start_q_rad
        measured_j1_delta = float(measured_delta_vector[MICROPROBE_JOINT_INDEX])
        other_drift = float(
            np.max(
                np.abs(np.delete(measured_delta_vector, MICROPROBE_JOINT_INDEX)),
                initial=0.0,
            )
        )
        max_dq = float(np.max(np.abs(dq), initial=0.0))
        worst_target_error = max(worst_target_error, target_error)
        worst_other_axis_drift = max(worst_other_axis_drift, other_drift)
        maximum_dq = max(maximum_dq, max_dq)
        proof.update(
            {
                "measured_j1_delta_rad": measured_j1_delta,
                "worst_target_error_rad": worst_target_error,
                "worst_other_axis_drift_rad": worst_other_axis_drift,
                "maximum_abs_dq_rad_s": maximum_dq,
                "sample_count": len(q_samples),
            }
        )
        if target_error > MICROPROBE_ARRIVAL_TOLERANCE_RAD:
            raise RuntimeError(
                f"{label} target error {target_error:.9f}rad exceeds "
                f"{MICROPROBE_ARRIVAL_TOLERANCE_RAD:.9f}rad"
            )
        if max_dq > MICROPROBE_MAX_SETTLED_DQ_RAD_S:
            raise RuntimeError(
                f"{label} max|dq| {max_dq:.9f}rad/s exceeds settled bound"
            )
        if abs(measured_j1_delta - expected_j1_delta_rad) > (
            MICROPROBE_ARRIVAL_TOLERANCE_RAD
        ):
            raise RuntimeError(
                f"{label} measured J1 delta {measured_j1_delta:.9f}rad differs "
                f"from expected {expected_j1_delta_rad:.9f}rad"
            )
        if worst_other_axis_drift > MICROPROBE_ARRIVAL_TOLERANCE_RAD:
            raise RuntimeError(f"{label} changed a non-J1 axis beyond tolerance")
        if index + 1 < MICROPROBE_PROOF_SAMPLES:
            sleep(0.02)
    proof["completed"] = True
    return proof


def _telemetry_evidence(value: Any) -> Mapping[str, Any]:
    return {
        "kind": str(value.kind),
        "samples": int(value.samples),
        "max_read_to_write_us": float(value.max_read_to_write_us),
        "read_to_write_overruns": int(value.read_to_write_overruns),
    }


def _hand_evidence(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    result = {}
    for name in (
        "hand_id",
        "angles",
        "angle_targets",
        "currents",
        "errors",
        "statuses",
        "speeds",
        "force_limits",
    ):
        value = snapshot[name]
        if isinstance(value, (tuple, list, np.ndarray)):
            result[name] = [int(item) for item in value]
        else:
            result[name] = int(value)
    return result


def _attach_microprobe_evidence(
    exc: BaseException, evidence: Mapping[str, Any]
) -> BaseException:
    setattr(exc, "microprobe_evidence", dict(evidence))
    return exc


def _session_evidence(run_id: str, mode: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "phase": "initializing",
        "failure_phase": None,
        "outbound_command_started": False,
        "outbound_command_completed": False,
        "outbound_proof_completed": False,
        "return_command_started": False,
        "return_command_completed": False,
        "return_proof_completed": False,
        "return_to_v94_home_proven": False,
        "franka_stop": {
            "attempted": False,
            "verified": False,
            "method": (
                "FrankaSequenceDriver.stop: Robot.stop followed by consecutive "
                "fresh Idle/stationary q/dq validation"
            ),
        },
        "rh56_command_writes": 0,
        "rh56_final_disabled_verified": False,
    }


def run_microprobe(
    config: Mapping[str, Any],
    *,
    bundle_path: Path,
    run_id: str,
    port_override: Optional[str] = None,
    hardware_types: Optional[Mapping[str, Any]] = None,
    network_preflight: Any = None,
    sleep: Any = time.sleep,
) -> V94FrankaMicroprobeResult:
    """Run the supervised two-leg probe and always verify final stop/hand state."""

    types = _load_hardware_types() if hardware_types is None else hardware_types
    active_network_preflight = (
        require_uncontended_franka_https_link
        if network_preflight is None
        else network_preflight
    )
    limits = _build_microprobe_limits(config, types["FrankaMotionLimits"])
    contract = V94Contract.from_bundle(DeployBundle(bundle_path))
    exact_q_home = contract.q_home_rad.astype(np.float64)
    api = types["rh56_api"]
    inspire = config["inspire"]
    open_min_angles = DEFAULT_RESET._profile_open_min_angles(config)
    port = str(port_override or inspire["port"])
    serial_context = None
    serial_entered = False
    hand = None
    arm = None
    hand_was_verified = False
    failure: Optional[BaseException] = None
    stop_failure: Optional[BaseException] = None
    hand_final_failure: Optional[BaseException] = None
    result: Optional[V94FrankaMicroprobeResult] = None
    evidence = _session_evidence(run_id, "probe")

    try:
        evidence["phase"] = "rh56_preflight"
        serial_context = api.LinuxSerial(port, int(inspire["baud"]), 0.5, False)
        entered = serial_context.__enter__()
        serial_entered = True
        serial_port = serial_context if entered is None else entered
        hand = api.RH56Hand(serial_port, int(inspire["hand_id"]))
        _, hand_preflight = DEFAULT_RESET._read_stable_hand_pair(
            hand,
            expected_hand_id=int(inspire["hand_id"]),
            label="microprobe preflight",
            open_min_angles=open_min_angles,
            sleep=sleep,
        )
        hand_was_verified = True
        evidence["rh56_preflight"] = _hand_evidence(hand_preflight)
        DEFAULT_RESET._print_hand("microprobe preflight PASS", hand_preflight)

        evidence["phase"] = "franka_connect"
        active_network_preflight(str(config["franka"]["ip"]))
        arm = types["FrankaSequenceDriver"].connect(
            str(config["franka"]["ip"]), limits, enforce_realtime=True
        )
        evidence["phase"] = "fresh_state_and_mapping"
        initial = arm.robot.read_once()
        initial_success = arm._validate_state(
            initial, require_idle=True, enforce_success=False
        )
        initial_q, initial_dq = _state_q_dq(initial)
        evidence["fresh_measured_q0_rad"] = initial_q.tolist()
        evidence["fresh_measured_dq0_rad_s"] = initial_dq.tolist()
        evidence["fresh_control_success_rate"] = float(initial_success)
        if float(np.max(np.abs(initial_dq), initial=0.0)) > (
            MICROPROBE_MAX_SETTLED_DQ_RAD_S
        ):
            raise RuntimeError("Franka fresh Idle state is not stationary enough")
        mapped = _map_live_j1_target(initial_q, contract=contract)
        evidence.update(
            {
                "action13": mapped.action13.tolist(),
                "v94_float32_origin_q_rad": mapped.origin_q_rad.tolist(),
                "exact_peak_target_q_rad": mapped.peak_q_rad.tolist(),
                "return_target_q_rad": exact_q_home.tolist(),
                "target_delta_rad": mapped.target_delta_rad,
                "reset_linf_error_rad": mapped.reset_linf_error_rad,
                "reset_linf_limit_rad": MICROPROBE_RESET_ENVELOPE_RAD,
                "start_alignment_limit_rad": (
                    MICROPROBE_START_ALIGNMENT_TOLERANCE_RAD
                ),
                "arrival_tolerance_rad": MICROPROBE_ARRIVAL_TOLERANCE_RAD,
                "arrival_tolerance_fraction_of_step": (
                    MICROPROBE_ARRIVAL_TOLERANCE_RAD
                    / MICROPROBE_NOMINAL_DELTA_RAD
                ),
                "maximum_speed_rad_s": float(limits.max_joint_speed_rad_s),
            }
        )
        print(
            "[Franka C1 preflight] run_id={} q0={} success_rate={:.6f} "
            "float32_q_error={:.9g}rad".format(
                run_id,
                np.round(initial_q, 9).tolist(),
                float(initial_success),
                mapped.measured_to_origin_quantization_rad,
            ),
            flush=True,
        )
        print(
            "[V94 mapping] action[0]=+1.0 joint={} target_delta={:.9f}rad "
            "arrival_tol={:.7f}rad ({:.2f}% step) max_speed={:.3f}rad/s".format(
                MICROPROBE_JOINT_NAME,
                mapped.target_delta_rad,
                MICROPROBE_ARRIVAL_TOLERANCE_RAD,
                100.0
                * MICROPROBE_ARRIVAL_TOLERANCE_RAD
                / MICROPROBE_NOMINAL_DELTA_RAD,
                float(limits.max_joint_speed_rad_s),
            ),
            flush=True,
        )

        evidence["phase"] = "outbound_command"
        evidence["outbound_command_started"] = True
        arm.move_joints(mapped.peak_q_rad)
        evidence["outbound_command_completed"] = True
        outbound_telemetry = _require_control_telemetry(arm, label="outbound")
        evidence["outbound_telemetry"] = _telemetry_evidence(outbound_telemetry)
        expected_outbound_delta = float(
            mapped.peak_q_rad[MICROPROBE_JOINT_INDEX]
            - mapped.measured_q_rad[MICROPROBE_JOINT_INDEX]
        )
        evidence["phase"] = "outbound_proof"
        outbound_proof: dict[str, Any] = {}
        evidence["outbound_proof"] = outbound_proof
        outbound = _read_motion_proof(
            arm,
            expected_q_rad=mapped.peak_q_rad,
            measured_start_q_rad=mapped.measured_q_rad,
            expected_j1_delta_rad=expected_outbound_delta,
            label="outbound",
            sleep=sleep,
            evidence=outbound_proof,
        )
        evidence["outbound_proof_completed"] = True
        if outbound["measured_j1_delta_rad"] <= 0.0:
            raise RuntimeError("outbound J1 direction was not positive")

        evidence["phase"] = "return_to_exact_v94_home_command"
        evidence["return_command_started"] = True
        arm.move_joints(exact_q_home)
        evidence["return_command_completed"] = True
        return_telemetry = _require_control_telemetry(
            arm, label="return", previous=outbound_telemetry
        )
        evidence["return_telemetry"] = _telemetry_evidence(return_telemetry)
        evidence["phase"] = "return_to_exact_v94_home_proof"
        return_proof: dict[str, Any] = {}
        evidence["return_proof"] = return_proof
        returned = _read_motion_proof(
            arm,
            expected_q_rad=exact_q_home,
            measured_start_q_rad=exact_q_home,
            expected_j1_delta_rad=0.0,
            label="return_to_exact_v94_home",
            sleep=sleep,
            evidence=return_proof,
        )
        evidence["return_proof_completed"] = True
        evidence["return_to_v94_home_proven"] = True
        evidence["phase"] = "motion_complete_pending_terminal_safety_proof"
        max_read_to_write = max(
            float(outbound_telemetry.max_read_to_write_us),
            float(return_telemetry.max_read_to_write_us),
        )
        result = V94FrankaMicroprobeResult(
            run_id=run_id,
            target_delta_rad=mapped.target_delta_rad,
            outbound_measured_delta_rad=outbound["measured_j1_delta_rad"],
            outbound_target_error_rad=outbound["worst_target_error_rad"],
            return_target_error_rad=returned["worst_target_error_rad"],
            outbound_control_samples=int(outbound_telemetry.samples),
            return_control_samples=int(return_telemetry.samples),
            maximum_read_to_write_us=max_read_to_write,
            franka_stop_verified=False,
            rh56_final_disabled_verified=False,
            rh56_command_writes=0,
            evidence=evidence,
        )
    except BaseException as exc:
        failure = exc
        evidence["failure_phase"] = evidence["phase"]
        evidence["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_record = evidence["franka_stop"]
        if arm is not None:
            stop_record["attempted"] = True
            try:
                arm.stop()
                stop_record["verified"] = True
            except BaseException as exc:
                stop_record["error"] = f"{type(exc).__name__}: {exc}"
                stop_failure = exc
        if hand is not None and hand_was_verified:
            try:
                _, hand_final = DEFAULT_RESET._read_stable_hand_pair(
                    hand,
                    expected_hand_id=int(inspire["hand_id"]),
                    label="microprobe final",
                    open_min_angles=open_min_angles,
                    sleep=sleep,
                )
                DEFAULT_RESET._print_hand("microprobe final PASS", hand_final)
                evidence["rh56_final_disabled_verified"] = True
                evidence["rh56_final"] = _hand_evidence(hand_final)
            except BaseException as exc:
                evidence["rh56_final_error"] = f"{type(exc).__name__}: {exc}"
                hand_final_failure = exc
        if serial_context is not None and serial_entered:
            try:
                serial_context.__exit__(None, None, None)
            except BaseException as exc:
                evidence["rh56_serial_close_error"] = f"{type(exc).__name__}: {exc}"
                if hand_final_failure is None:
                    hand_final_failure = exc
        if failure is None and stop_failure is None and hand_final_failure is None:
            evidence["phase"] = "complete"
        if result is not None:
            result = V94FrankaMicroprobeResult(
                **{
                    **result.__dict__,
                    "franka_stop_verified": bool(stop_record["verified"]),
                    "rh56_final_disabled_verified": bool(
                        evidence["rh56_final_disabled_verified"]
                    ),
                    "evidence": evidence,
                }
            )

    if stop_failure is not None:
        detail = f"{type(stop_failure).__name__}: {stop_failure}"
        if failure is not None:
            detail += f"; original error: {type(failure).__name__}: {failure}"
        terminal = MicroprobeStopUnconfirmed(
            "STOP UNCONFIRMED: use Franka's physical stop immediately; " + detail
        )
        raise _attach_microprobe_evidence(terminal, evidence) from stop_failure
    if hand_final_failure is not None:
        detail = f"{type(hand_final_failure).__name__}: {hand_final_failure}"
        if failure is not None:
            detail += f"; original error: {type(failure).__name__}: {failure}"
        terminal = RuntimeError(
            "Franka stop was verified, but final RH56 all-six disabled readback "
            "failed: " + detail
        )
        raise _attach_microprobe_evidence(terminal, evidence) from hand_final_failure
    if failure is not None:
        raise _attach_microprobe_evidence(failure, evidence)
    if result is None or not result.franka_stop_verified or not (
        result.rh56_final_disabled_verified
    ):
        terminal = RuntimeError("microprobe terminal safety proof is incomplete")
        raise _attach_microprobe_evidence(terminal, evidence)
    print(
        "[PASS] C1 V94 J1 mapping: target_delta={:.9f}rad "
        "measured_delta={:.9f}rad outbound_error={:.9f}rad "
        "home_error={:.9f}rad; Franka stop verified; RH56 writes=0".format(
            result.target_delta_rad,
            result.outbound_measured_delta_rad,
            result.outbound_target_error_rad,
            result.return_target_error_rad,
        ),
        flush=True,
    )
    print(
        "[scope] single supervised stop-to-stop target mapping only; "
        "exact V94 q_home return proven; 60Hz C2 dynamics remain unvalidated",
        flush=True,
    )
    return result


def run_recovery_to_v94_home(
    config: Mapping[str, Any],
    *,
    bundle_path: Path,
    run_id: str,
    port_override: Optional[str] = None,
    hardware_types: Optional[Mapping[str, Any]] = None,
    network_preflight: Any = None,
    sleep: Any = time.sleep,
) -> V94FrankaRecoveryResult:
    """Recover only from the safe envelope to exact bundle q_home."""

    types = _load_hardware_types() if hardware_types is None else hardware_types
    active_network_preflight = (
        require_uncontended_franka_https_link
        if network_preflight is None
        else network_preflight
    )
    limits = _build_microprobe_limits(config, types["FrankaMotionLimits"])
    contract = V94Contract.from_bundle(DeployBundle(bundle_path))
    exact_q_home = contract.q_home_rad.astype(np.float64)
    api = types["rh56_api"]
    inspire = config["inspire"]
    open_min_angles = DEFAULT_RESET._profile_open_min_angles(config)
    port = str(port_override or inspire["port"])
    serial_context = None
    serial_entered = False
    hand = None
    arm = None
    hand_was_verified = False
    failure: Optional[BaseException] = None
    stop_failure: Optional[BaseException] = None
    hand_final_failure: Optional[BaseException] = None
    result: Optional[V94FrankaRecoveryResult] = None
    evidence: dict[str, Any] = {
        "run_id": run_id,
        "mode": "recover_to_exact_v94_q_home",
        "phase": "initializing",
        "failure_phase": None,
        "exact_q_home_target_rad": exact_q_home.tolist(),
        "safe_start_envelope_rad": MICROPROBE_RESET_ENVELOPE_RAD,
        "arrival_tolerance_rad": MICROPROBE_ARRIVAL_TOLERANCE_RAD,
        "maximum_speed_rad_s": float(limits.max_joint_speed_rad_s),
        "recovery_command_started": False,
        "recovery_command_completed": False,
        "recovery_proof_completed": False,
        "return_to_v94_home_proven": False,
        "franka_stop": {
            "attempted": False,
            "verified": False,
            "method": (
                "FrankaSequenceDriver.stop: Robot.stop followed by consecutive "
                "fresh Idle/stationary q/dq validation"
            ),
        },
        "rh56_command_writes": 0,
        "rh56_final_disabled_verified": False,
    }

    try:
        evidence["phase"] = "rh56_preflight"
        serial_context = api.LinuxSerial(port, int(inspire["baud"]), 0.5, False)
        entered = serial_context.__enter__()
        serial_entered = True
        serial_port = serial_context if entered is None else entered
        hand = api.RH56Hand(serial_port, int(inspire["hand_id"]))
        _, hand_preflight = DEFAULT_RESET._read_stable_hand_pair(
            hand,
            expected_hand_id=int(inspire["hand_id"]),
            label="V94 q_home recovery preflight",
            open_min_angles=open_min_angles,
            sleep=sleep,
        )
        hand_was_verified = True
        evidence["rh56_preflight"] = _hand_evidence(hand_preflight)
        DEFAULT_RESET._print_hand("V94 q_home recovery preflight PASS", hand_preflight)

        evidence["phase"] = "franka_fresh_state"
        active_network_preflight(str(config["franka"]["ip"]))
        arm = types["FrankaSequenceDriver"].connect(
            str(config["franka"]["ip"]), limits, enforce_realtime=True
        )
        initial = arm.robot.read_once()
        initial_success = arm._validate_state(
            initial, require_idle=True, enforce_success=False
        )
        initial_q, initial_dq = _state_q_dq(initial)
        initial_error = float(
            np.max(np.abs(initial_q - exact_q_home), initial=0.0)
        )
        initial_max_dq = float(np.max(np.abs(initial_dq), initial=0.0))
        evidence.update(
            {
                "fresh_measured_q0_rad": initial_q.tolist(),
                "fresh_measured_dq0_rad_s": initial_dq.tolist(),
                "fresh_control_success_rate": float(initial_success),
                "initial_home_error_rad": initial_error,
                "initial_max_abs_dq_rad_s": initial_max_dq,
            }
        )
        if initial_max_dq > MICROPROBE_MAX_SETTLED_DQ_RAD_S:
            raise RuntimeError("Franka fresh Idle state is not stationary enough")
        if initial_error > MICROPROBE_RESET_ENVELOPE_RAD:
            raise RuntimeError(
                "recovery refused before motion: current q is outside the V94 "
                f"safe envelope ({initial_error:.9f}rad > "
                f"{MICROPROBE_RESET_ENVELOPE_RAD:.9f}rad)"
            )

        motion_required = initial_error > MICROPROBE_ARRIVAL_TOLERANCE_RAD
        evidence["motion_required"] = motion_required
        telemetry = None
        if motion_required:
            evidence["phase"] = "recovery_command"
            evidence["recovery_command_started"] = True
            arm.move_joints(exact_q_home)
            evidence["recovery_command_completed"] = True
            telemetry = _require_control_telemetry(arm, label="q_home recovery")
            evidence["recovery_telemetry"] = _telemetry_evidence(telemetry)

        evidence["phase"] = "recovery_proof"
        recovery_proof: dict[str, Any] = {}
        evidence["recovery_proof"] = recovery_proof
        proof = _read_motion_proof(
            arm,
            expected_q_rad=exact_q_home,
            measured_start_q_rad=exact_q_home,
            expected_j1_delta_rad=0.0,
            label="recovery_to_exact_v94_q_home",
            sleep=sleep,
            evidence=recovery_proof,
        )
        evidence["recovery_proof_completed"] = True
        evidence["return_to_v94_home_proven"] = True
        evidence["phase"] = "recovery_complete_pending_terminal_safety_proof"
        result = V94FrankaRecoveryResult(
            run_id=run_id,
            initial_home_error_rad=initial_error,
            final_home_error_rad=float(proof["worst_target_error_rad"]),
            control_samples=0 if telemetry is None else int(telemetry.samples),
            maximum_read_to_write_us=(
                0.0 if telemetry is None else float(telemetry.max_read_to_write_us)
            ),
            franka_stop_verified=False,
            rh56_final_disabled_verified=False,
            rh56_command_writes=0,
            evidence=evidence,
        )
    except BaseException as exc:
        failure = exc
        evidence["failure_phase"] = evidence["phase"]
        evidence["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_record = evidence["franka_stop"]
        if arm is not None:
            stop_record["attempted"] = True
            try:
                arm.stop()
                stop_record["verified"] = True
            except BaseException as exc:
                stop_record["error"] = f"{type(exc).__name__}: {exc}"
                stop_failure = exc
        if hand is not None and hand_was_verified:
            try:
                _, hand_final = DEFAULT_RESET._read_stable_hand_pair(
                    hand,
                    expected_hand_id=int(inspire["hand_id"]),
                    label="V94 q_home recovery final",
                    open_min_angles=open_min_angles,
                    sleep=sleep,
                )
                DEFAULT_RESET._print_hand("V94 q_home recovery final PASS", hand_final)
                evidence["rh56_final_disabled_verified"] = True
                evidence["rh56_final"] = _hand_evidence(hand_final)
            except BaseException as exc:
                evidence["rh56_final_error"] = f"{type(exc).__name__}: {exc}"
                hand_final_failure = exc
        if serial_context is not None and serial_entered:
            try:
                serial_context.__exit__(None, None, None)
            except BaseException as exc:
                evidence["rh56_serial_close_error"] = f"{type(exc).__name__}: {exc}"
                if hand_final_failure is None:
                    hand_final_failure = exc
        if failure is None and stop_failure is None and hand_final_failure is None:
            evidence["phase"] = "complete"
        if result is not None:
            result = V94FrankaRecoveryResult(
                **{
                    **result.__dict__,
                    "franka_stop_verified": bool(stop_record["verified"]),
                    "rh56_final_disabled_verified": bool(
                        evidence["rh56_final_disabled_verified"]
                    ),
                    "evidence": evidence,
                }
            )

    if stop_failure is not None:
        detail = f"{type(stop_failure).__name__}: {stop_failure}"
        if failure is not None:
            detail += f"; original error: {type(failure).__name__}: {failure}"
        terminal = MicroprobeStopUnconfirmed(
            "STOP UNCONFIRMED during V94 q_home recovery: " + detail
        )
        raise _attach_microprobe_evidence(terminal, evidence) from stop_failure
    if hand_final_failure is not None:
        detail = f"{type(hand_final_failure).__name__}: {hand_final_failure}"
        if failure is not None:
            detail += f"; original error: {type(failure).__name__}: {failure}"
        terminal = RuntimeError(
            "Franka recovery stop was verified, but final RH56 disabled readback "
            "failed: " + detail
        )
        raise _attach_microprobe_evidence(terminal, evidence) from hand_final_failure
    if failure is not None:
        raise _attach_microprobe_evidence(failure, evidence)
    if result is None or not result.franka_stop_verified or not (
        result.rh56_final_disabled_verified
    ):
        terminal = RuntimeError("V94 q_home recovery terminal proof is incomplete")
        raise _attach_microprobe_evidence(terminal, evidence)
    print(
        "[PASS] exact V94 q_home recovery: initial_error={:.9f}rad "
        "final_error={:.9f}rad; Franka stop verified; RH56 writes=0".format(
            result.initial_home_error_rad,
            result.final_home_error_rad,
        ),
        flush=True,
    )
    return result


def _q_home_digest(bundle_path: Path) -> str:
    q_home = V94Contract.from_bundle(DeployBundle(bundle_path)).q_home_rad
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(q_home, dtype="<f4")).tobytes()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class _ExclusiveEvidence:
    """Reserve one run artifact before hardware and write one terminal object."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        if not self.path.parent.is_dir():
            raise ValueError(f"evidence parent directory does not exist: {self.path.parent}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(str(self.path), flags, 0o600)
        self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
        self._written = False

    def write(self, value: Mapping[str, Any]) -> None:
        if self._written:
            raise RuntimeError("terminal microprobe evidence was already written")
        json.dump(value, self._stream, sort_keys=True, allow_nan=False)
        self._stream.write("\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._written = True

    def close(self) -> None:
        self._stream.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, config_path = load_control_config(args.config)
        assets = verify_adapter_assets(config, config_path)
        mapping = _validate_static_request(config, args.bundle)
        print(
            "[config] {}\n[adapter] verified={} sha256={}".format(
                config_path, assets.mesh_path, assets.mesh_sha256
            )
        )
        print(
            "[V94 mapping audit] contract={} checkpoint_sha256={} "
            "axis_order_identity={} sign={} effective_gain={:.9f}rad/tick".format(
                mapping["bundle_contract"],
                mapping["checkpoint_sha256"],
                mapping["axis_order_identity"],
                mapping["sign_transform"],
                mapping["effective_gain_rad_per_tick"],
            )
        )
        print(
            "[fixed C1 plan] fresh q0 -> action[0]=+1 exact V94 target "
            "(+~0.003rad J1) -> fresh q/dq proof -> exact V94 q_home -> fresh proof; "
            "safe_reset_Linf<=0.05rad, start_alignment<=0.0002rad, "
            "arrival_tol=0.0002rad (6.67% step), "
            "speed<=0.05rad/s, RH56 writes=0; "
            "q_home_contract_sha256={}".format(_q_home_digest(args.bundle))
        )
        if not args.execute and not args.recover_to_v94_home:
            print(
                "[DRY RUN] No hardware driver was imported. Use either --execute "
                "for the probe or --recover-to-v94-home for recovery, with the "
                "corresponding exact run-scoped confirmations."
            )
            return 0
        recovery_mode = bool(args.recover_to_v94_home)
        run_id = (
            _require_recovery_authorization(args)
            if recovery_mode
            else _require_execute_authorization(args)
        )
        output = (
            args.output
            if args.output is not None
            else ROOT
            / (
                f"runs/v94_franka_qhome_recovery_{run_id}.json"
                if recovery_mode
                else f"runs/v94_franka_action_microprobe_{run_id}.json"
            )
        )
        evidence = _ExclusiveEvidence(output)
        evidence_base = {
            "schema_version": 2,
            "kind": (
                "v94_franka_c1_exact_qhome_recovery"
                if recovery_mode
                else "v94_franka_c1_action_mapping_microprobe"
            ),
            "run_id": run_id,
            "scope": (
                "C1_exact_v94_q_home_recovery_only"
                if recovery_mode
                else "C1_single_stop_to_stop_mapping_only_not_60Hz_C2"
            ),
            "inputs": {
                "bundle": str(args.bundle.expanduser().resolve()),
                "bundle_sha256": _sha256_file(args.bundle),
                "profile": str(args.config.expanduser().resolve()),
                "profile_sha256": _sha256_file(args.config),
                "adapter_sha256": assets.mesh_sha256,
                "checkpoint_sha256": mapping["checkpoint_sha256"],
            },
            "authorization": {
                "run_scoped_token_matched": True,
                "token_value_logged": False,
                "workspace_clear_confirmed": True,
                "physical_stop_ready_confirmed": True,
            },
        }
        print(
            "[authorization] run-scoped {} token accepted for run_id={}; "
            "workspace sweep and physical stop confirmations accepted".format(
                "recovery" if recovery_mode else "probe", run_id
            ),
            flush=True,
        )
        try:
            if recovery_mode:
                result = run_recovery_to_v94_home(
                    config,
                    bundle_path=args.bundle,
                    run_id=run_id,
                    port_override=args.port,
                )
            else:
                result = run_microprobe(
                    config,
                    bundle_path=args.bundle,
                    run_id=run_id,
                    port_override=args.port,
                )
        except BaseException as exc:
            partial = getattr(exc, "microprobe_evidence", {})
            stop_verified = bool(
                partial.get("franka_stop", {}).get("verified", False)
            )
            hand_verified = bool(
                partial.get("rh56_final_disabled_verified", False)
            )
            home_proven = bool(partial.get("return_to_v94_home_proven", False))
            evidence.write(
                {
                    **evidence_base,
                    "result": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "terminal_proof_available": stop_verified and hand_verified,
                    "return_to_v94_home_proven": home_proven,
                    "position_statement": (
                        "exact_v94_q_home_proven"
                        if home_proven
                        else "unknown_after_verified_stop_do_not_assume_returned"
                    ),
                    "partial_session": partial,
                }
            )
            print(f"[evidence] {evidence.path}", flush=True)
            if not home_proven:
                print(
                    "[recovery required] return to exact V94 q_home was not proven; "
                    "do not start another probe before read-only inspection and the "
                    "separate --recover-to-v94-home path",
                    file=sys.stderr,
                    flush=True,
                )
            raise
        else:
            evidence.write(
                {
                    **evidence_base,
                    "result": "PASS",
                    "terminal_proof_available": True,
                    "return_to_v94_home_proven": True,
                    "session": asdict(result),
                }
            )
            print(f"[evidence] {evidence.path}", flush=True)
        finally:
            evidence.close()
        return 0
    except KeyboardInterrupt:
        print(
            "[interrupted] Ctrl+C received; inspect the exclusive evidence for "
            "the actual Franka stop/RH56 final proof and position status",
            file=sys.stderr,
        )
        return 130
    except MicroprobeStopUnconfirmed as exc:
        print(f"[{exc}]", file=sys.stderr)
        return 5
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[failed] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
