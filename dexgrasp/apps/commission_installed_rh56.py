#!/usr/bin/env python3
"""Supervised installed RH56 q6/coupled-air commissioning.

The ``run`` command is the only hardware-capable path.  Exact safety tokens,
profile/assets, output exclusivity, and request bounds are checked before this
file imports pylibfranka or the RH56 serial driver.  Franka is read-only: no
controller, stop, EE, or load command is created.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydex_pipeline.control_config import load_control_config, verify_adapter_assets
from anydex_pipeline.rh56_commissioning import (
    EVIDENCE_KIND,
    SCHEMA_VERSION,
    STAGE1_Q6_TARGET,
    atomic_write_evidence,
    build_snapshot_candidate_binding,
    build_stage1_prerequisite_binding,
    derived_config_proposal,
    group_telemetry,
    json_sha256,
    load_evidence,
    seal_evidence,
    sha256_file,
    verify_applied_config,
    verify_evidence,
)
from anydex_pipeline.rh56_hand_path import (
    Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS,
    build_rh56_no_contact_execution_path,
    rh56_feedback_envelope_policy,
)


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
POWER_TOKEN = "RH56_24V_CUTOFF_READY"
STOP_TOKEN = "FR3_STOP_READY"
CLEAR_TOKEN = "INSTALLED_AIR_WORKSPACE_CLEAR"
NO_CONTACT_TOKEN = "PLA_LOW_SPEED_NO_CONTACT"
WIDE_Q6_TOKEN = "RH56_Q6_WIDE_RANGE_COMMISSIONING"
COUPLED_TOKEN = "RH56_COUPLED_AIR_CLOSURE"
EXACT_AIR_TARGET_TOKEN = "RH56_EXACT_CANDIDATE_AIR_TARGET"
COMMISSIONING_SPEED = 40
COMMISSIONING_FORCE_G = 80
STOP_MAX_AXIS_CURRENT_MA = 100
COMMISSION_BEND_OPEN_MIN_ANGLE = 980
COMMISSION_Q6_OPEN_MAX_TOLERANCE_UNITS = 30


class _StoreManualTarget(argparse.Action):
    """Record whether a manual target flag was explicitly present."""

    def __call__(self, parser, namespace, values, option_string=None):
        del parser
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_supplied", True)


def _snapshot_file_state(path: Path) -> tuple[int, int, int, int, int]:
    stat = Path(path).expanduser().resolve().stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _resolve_snapshot_targets(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Optional[dict[str, Any]]:
    """Resolve manual defaults or one immutable official snapshot candidate."""

    if args.snapshot is None:
        if args.candidate_index is not None:
            raise ValueError("--candidate-index requires --snapshot")
        return None
    if getattr(args, "_target_q6_supplied", False) or getattr(
        args, "_bend_targets_supplied", False
    ):
        raise ValueError(
            "--snapshot cannot be combined with --target-q6 or --bend-targets"
        )
    if not args.coupled_air_close:
        raise ValueError("--snapshot requires --coupled-air-close")
    binding = build_snapshot_candidate_binding(
        args.snapshot,
        config=config,
        candidate_index=args.candidate_index,
    )
    targets = binding["candidate"]["hand_targets"]
    args.target_q6 = int(targets[5])
    args.bend_targets = [int(value) for value in targets[:5]]
    args.snapshot = Path(binding["path"])
    print(
        "[snapshot target] index={} score={:.6f} type={} targets={} selection={}".format(
            binding["candidate"]["index"],
            binding["candidate"]["official_score"],
            binding["candidate"]["type_id"],
            binding["candidate"]["hand_targets"],
            binding["selection_method"],
        )
    )
    return binding


def _record_runtime_failure(runtime: Mapping[str, Any], message: str) -> None:
    result = runtime["result"]
    result["status"] = "fail"
    previous = result.get("operation_error")
    result["operation_error"] = message if previous is None else f"{previous}; {message}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _requested_q6_waypoints(target_q6: int, step_units: int) -> tuple[int, ...]:
    points = list(range(1000 - int(step_units), int(target_q6), -int(step_units)))
    if not points or points[-1] != int(target_q6):
        points.append(int(target_q6))
    return tuple(points)


def _requested_q6_return_waypoints(
    target_q6: int, step_units: int
) -> tuple[int, ...]:
    target = int(target_q6)
    step = int(step_units)
    # The one-command return is an empirical Stage-1 exception proven only
    # for q6=900.  Applying it to a future wide-range endpoint such as 646
    # would create an unaudited 354-unit jump.  Every lower endpoint follows
    # the canonical reversal bootstrap (+50 first, then <= step) instead.
    if target == 900:
        return (1000,)
    path = build_rh56_no_contact_execution_path(
        (1000, 1000, 1000, 1000, 1000, target),
        step_units=step,
    )
    return tuple(
        item.command_targets[5]
        for item in path.waypoints
        if item.phase.startswith("q6_reverse_")
    )


def _requested_coupled_waypoints(
    bend_targets: Sequence[int], target_q6: int, step_units: int
) -> tuple[tuple[int, ...], ...]:
    """Build the exact one-bend-axis-at-a-time command path."""

    current = [1000] * 5
    waypoints: list[tuple[int, ...]] = []
    for axis, requested in enumerate(int(value) for value in bend_targets):
        while current[axis] != requested:
            current[axis] = max(requested, current[axis] - int(step_units))
            waypoints.append(tuple(current) + (int(target_q6),))
    return tuple(waypoints)


def _requested_coupled_return_waypoints(
    coupled_waypoints: Sequence[Sequence[int]], target_q6: int, step_units: int
) -> tuple[tuple[int, ...], ...]:
    forward = tuple(tuple(int(value) for value in item) for item in coupled_waypoints)
    if not forward:
        return ()
    path = build_rh56_no_contact_execution_path(
        forward[-1], step_units=int(step_units)
    )
    return tuple(
        item.command_targets
        for item in path.waypoints
        if item.phase.startswith("bend_reverse_")
    )


def _matrix_from_franka(values: Any, name: str) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (16,):
        array = array.reshape((4, 4), order="F")
    if array.shape != (4, 4) or not np.all(np.isfinite(array)):
        raise RuntimeError(f"Franka {name} feedback is malformed")
    return [[float(item) for item in row] for row in array]


def _franka_state_json(state: Any) -> dict[str, Any]:
    return {
        "robot_mode": str(getattr(state, "robot_mode")),
        "q_rad": [float(value) for value in np.asarray(state.q, dtype=np.float64)],
        "dq_rad_s": [float(value) for value in np.asarray(state.dq, dtype=np.float64)],
        "O_T_EE": _matrix_from_franka(state.O_T_EE, "O_T_EE"),
        "F_T_EE": _matrix_from_franka(state.F_T_EE, "F_T_EE"),
        "m_ee_kg": float(state.m_ee),
        "F_x_Cee_m": [
            float(value) for value in np.asarray(state.F_x_Cee, dtype=np.float64)
        ],
        "I_ee_kg_m2": np.asarray(state.I_ee, dtype=np.float64).reshape(-1).tolist(),
        "m_load_kg": float(state.m_load),
        "m_total_kg": float(state.m_total),
        "read_at_unix_s": float(time.time()),
    }


def _rh56_snapshot_json(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in snapshot.items():
        if isinstance(value, (tuple, list)):
            output[str(key)] = [int(item) for item in value]
        elif isinstance(value, (int, float, str, bool)) or value is None:
            output[str(key)] = value
        else:
            output[str(key)] = str(value)
    return output


def _require_fresh_commissioning_q6_open(
    snapshot: Mapping[str, Any],
    *,
    q6_open_min_angle: int,
) -> int:
    """Require a motionless q6 start; lower states belong to recovery."""

    initial_angles = snapshot.get("angles")
    if (
        not isinstance(initial_angles, list)
        or len(initial_angles) != 6
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in initial_angles
        )
    ):
        raise RuntimeError(
            "RH56 initial ANGLE_ACT snapshot must contain six integers"
        )
    actual_q6 = int(initial_angles[5])
    if not int(q6_open_min_angle) <= actual_q6 <= 1000:
        raise RuntimeError(
            "fresh commissioning requires q6 already inside the "
            f"profile-derived open band {q6_open_min_angle}..1000; "
            f"actual={actual_q6}. Run the evidence-bound recovery workflow "
            "before commissioning a lower q6."
        )
    return actual_q6


def _profile_q6_open_min_angle(config: Mapping[str, Any]) -> int:
    """Derive the physical q6 open floor without weakening bend-axis gates."""

    inspire = config.get("inspire")
    if not isinstance(inspire, Mapping):
        raise ValueError("control profile has no inspire object")
    open_targets = inspire.get("open_targets")
    q6_range = inspire.get("thumb_rotate_validated_realtime_range")
    tolerance = inspire.get("arrival_tolerance_units")
    if (
        not isinstance(open_targets, list)
        or len(open_targets) != 6
        or any(isinstance(value, bool) or not isinstance(value, int) for value in open_targets)
    ):
        raise ValueError("inspire.open_targets must contain six integers")
    if (
        not isinstance(q6_range, list)
        or len(q6_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in q6_range)
    ):
        raise ValueError(
            "inspire.thumb_rotate_validated_realtime_range must contain two integers"
        )
    if isinstance(tolerance, bool) or not isinstance(tolerance, int):
        raise ValueError("inspire.arrival_tolerance_units must be an integer")
    lower, upper = (int(value) for value in q6_range)
    target = int(open_targets[5])
    if not 0 <= lower <= upper <= 1000 or target != upper:
        raise ValueError(
            "inspire q6 open target must equal a valid commissioned range endpoint"
        )
    if not 0 <= int(tolerance) <= COMMISSION_Q6_OPEN_MAX_TOLERANCE_UNITS:
        raise ValueError(
            "inspire.arrival_tolerance_units exceeds the reviewed q6 open "
            f"limit {COMMISSION_Q6_OPEN_MAX_TOLERANCE_UNITS}"
        )
    minimum = target - int(tolerance)
    if not lower <= minimum <= upper:
        raise ValueError(
            "profile-derived q6 open feedback threshold is outside its "
            "validated realtime range"
        )
    return minimum


def _build_installed_commissioning_driver_type(
    reset_driver_type: Any,
    driver_error_type: Any,
) -> Any:
    """Build the current-session driver without importing hardware at module load.

    The inherited commissioning implementation has one scalar ``open_min_angle``
    used for all axes.  The installed q6 has a separately observed physical
    endpoint, while the five bend axes must retain their original 980-unit
    floor.  This subclass uses the profile-derived q6 floor in the inherited q6
    checks and independently enforces the 980-unit bend floor on every reset,
    q6-sweep, and q6-return feedback sample.
    """

    class InstalledRH56CommissioningDriver(reset_driver_type):
        def __init__(
            self,
            hand: Any,
            api: Any,
            *,
            q6_open_min_angle: int,
            **driver_kwargs: Any,
        ) -> None:
            if "open_min_angle" in driver_kwargs:
                raise ValueError(
                    "installed commissioning derives open thresholds from the profile"
                )
            driver_kwargs["open_min_angle"] = int(q6_open_min_angle)
            super().__init__(
                hand,
                api,
                q6_open_min_angle=int(q6_open_min_angle),
                **driver_kwargs,
            )

        @staticmethod
        def _requires_open_bends(phase: str) -> bool:
            return (
                phase.startswith("reset_open_")
                or phase == "q6_sweep_preflight"
                or phase.startswith("q6_step_")
                or phase == "coupled_air_close_preflight"
                or phase == "q6_return_preflight"
                or (
                    phase.startswith("q6_return_")
                    and phase != "q6_return_adopt_verify"
                )
            )

        def _read_feedback(
            self,
            phase: str,
            started: float,
            *,
            allow_external_gate_bypass_after_disabled_readback: bool = False,
        ) -> Any:
            feedback = super()._read_feedback(
                phase,
                started,
                allow_external_gate_bypass_after_disabled_readback=(
                    allow_external_gate_bypass_after_disabled_readback
                ),
            )
            if self._requires_open_bends(str(phase)) and any(
                int(angle) < COMMISSION_BEND_OPEN_MIN_ANGLE
                for angle in feedback.angles[:5]
            ):
                raise driver_error_type(
                    f"{phase}: installed commissioning requires all five bend "
                    f"axes >= {COMMISSION_BEND_OPEN_MIN_ANGLE}; "
                    f"actual={feedback.angles[:5]}"
                )
            return feedback

    return InstalledRH56CommissioningDriver


def _source_bindings(assets: Any) -> list[dict[str, str]]:
    paths = {
        "commission_cli": Path(__file__).resolve(),
        "commission_evidence_module": (
            ROOT / "src/anydex_pipeline/rh56_commissioning.py"
        ).resolve(),
        "rh56_hand_path": (
            ROOT / "src/anydex_pipeline/rh56_hand_path.py"
        ).resolve(),
        "rh56_reset_open": (
            ROOT / "src/anydex_pipeline/rh56_reset_open.py"
        ).resolve(),
        "rh56_sequence_driver": (
            ROOT / "src/anydex_pipeline/inspire_sequence_driver.py"
        ).resolve(),
        "rh56_register_api": (WORKSPACE / "examples/inspire_rh56_test.py").resolve(),
        "franka_sequence_driver": (
            ROOT / "src/anydex_pipeline/franka_sequence_driver.py"
        ).resolve(),
        "control_config_module": (
            ROOT / "src/anydex_pipeline/control_config.py"
        ).resolve(),
        "adapter_mesh": Path(assets.mesh_path).resolve(),
        "adapter_provenance": Path(assets.provenance_path).resolve(),
        "actuator_to_joint_xlsx": (
            ROOT
            / "third_party/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf"
            / "inspire_hand_routine_to_angle-use.xlsx"
        ).resolve(),
        "driver_to_angle_xls": (
            ROOT
            / "third_party/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf"
            / "driver_routine_to_angle.xls"
        ).resolve(),
        "actuator_to_urdf_generator": (
            ROOT
            / "third_party/AnyDexGrasp/generate_mesh_and_pointcloud"
            / "recover_inspire_hand_to_stl.py"
        ).resolve(),
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("required commissioning provenance is missing: " + "; ".join(missing))
    return [
        {"name": name, "path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    ]


def _build_limits(config: Mapping[str, Any], limits_type: Any) -> Any:
    franka = config["franka"]
    dynamics = franka["expected_end_effector"]
    return limits_type(
        expected_F_T_EE=np.asarray(franka["expected_F_T_EE"], dtype=np.float64),
        expected_m_ee_kg=float(dynamics["mass_kg"]),
        expected_F_x_Cee_m=np.asarray(dynamics["F_x_Cee_m"], dtype=np.float64),
        expected_I_ee_kg_m2=np.asarray(
            dynamics["inertia_kg_m2"], dtype=np.float64
        ),
        joint_limits_rad=np.asarray(franka["joint_limits_rad"], dtype=np.float64),
        joint_limit_margin_rad=float(franka["joint_limit_margin_rad"]),
        max_joint_speed_rad_s=float(franka["default_max_joint_velocity_rad_s"]),
        max_joint_segment_rad=float(franka["default_max_joint_segment_rad"]),
        min_joint_duration_s=float(franka["default_min_duration_s"]),
        joint_arrival_tolerance_rad=float(franka["default_arrival_tolerance_rad"]),
    )


def _validate_run_request(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    exact = (
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-24v-cutoff", args.confirm_24v_cutoff, POWER_TOKEN),
        ("--confirm-franka-stop", args.confirm_franka_stop, STOP_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, CLEAR_TOKEN),
        ("--confirm-no-contact", args.confirm_no_contact, NO_CONTACT_TOKEN),
    )
    missing = [f"{flag} {token}" for flag, actual, token in exact if actual != token]
    inspire = config["inspire"]
    current_lower = int(inspire["thumb_rotate_validated_realtime_range"][0])
    if int(args.target_q6) < current_lower and args.confirm_wide_q6 != WIDE_Q6_TOKEN:
        missing.append(f"--confirm-wide-q6 {WIDE_Q6_TOKEN}")
    if int(args.target_q6) < STAGE1_Q6_TARGET and args.stage1_evidence is None:
        missing.append("--stage1-evidence PATH_TO_FORMAL_Q6_900_PASS")
    if args.coupled_air_close and args.confirm_coupled_closure != COUPLED_TOKEN:
        missing.append(f"--confirm-coupled-closure {COUPLED_TOKEN}")
    bends = tuple(int(value) for value in args.bend_targets)
    exact_target_scope = args.coupled_air_close and any(
        value < 800 or value > 950 for value in bends
    )
    if exact_target_scope and args.confirm_exact_air_target != EXACT_AIR_TARGET_TOKEN:
        missing.append(f"--confirm-exact-air-target {EXACT_AIR_TARGET_TOKEN}")
    if missing:
        raise ValueError("exact confirmations required before hardware import: " + "; ".join(missing))

    if isinstance(args.target_q6, bool) or not 0 <= int(args.target_q6) <= 1000:
        raise ValueError("--target-q6 must be an integer in 0..1000")
    if not 10 <= int(args.q6_step) <= 50:
        raise ValueError("--q6-step must be in 10..50")
    if not 50 <= int(args.max_axis_current_ma) <= 1400:
        raise ValueError("--max-axis-current-ma must be in 50..1400")
    if not 1 <= int(args.max_inactive_drift_units) <= 20:
        raise ValueError("--max-inactive-drift-units must be in 1..20")
    if len(bends) != 5 or any(value < 0 or value > 1000 for value in bends):
        raise ValueError("--bend-targets must contain five values in 0..1000")
    if not args.coupled_air_close and tuple(bends) != (900,) * 5:
        raise ValueError("custom --bend-targets require --coupled-air-close")

    tool = config["tool"]
    if tool.get("material") != "PLA" or tool.get("low_speed_unloaded_commissioning_only") is not True:
        raise ValueError("this entry requires the reviewed PLA low-speed/unloaded profile")
    for key in (
        "installed_on_franka_verified",
        "mount_transform_commissioned",
        "assembled_yaw_verified",
        "source_origin_datum_verified",
    ):
        if tool.get(key) is not True:
            raise ValueError(f"tool.{key} must be true before installed commissioning")
    if int(inspire["open_speed"]) > COMMISSIONING_SPEED or int(inspire["close_speed"]) > COMMISSIONING_SPEED:
        raise ValueError("profile hand speeds exceed the fixed commissioning speed 40")
    if int(inspire["force_limit_g"]) > COMMISSIONING_FORCE_G:
        raise ValueError("profile force limit exceeds the fixed commissioning limit 80g")
    if Path(args.output).expanduser().resolve().exists():
        raise FileExistsError(f"evidence output already exists: {Path(args.output).expanduser().resolve()}")


def _run_hardware_session(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run only after the caller has completed every offline gate."""

    # These are intentionally lazy.  Unit tests assert that the missing-token
    # path cannot reach any of these imports or device constructors.
    import pylibfranka
    from anydex_pipeline.franka_sequence_driver import (
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )
    from anydex_pipeline.inspire_sequence_driver import RH56SequenceDriverError
    from anydex_pipeline.rh56_reset_open import RH56ResetOpenDriver

    RH56InstalledCommissioningDriver = _build_installed_commissioning_driver_type(
        RH56ResetOpenDriver,
        RH56SequenceDriverError,
    )

    robot = None
    arm = None
    franka_initial = None
    franka_gate_check_count = 0
    franka_gate_last = franka_initial
    franka_gate_failure: Optional[str] = None

    def continuous_franka_read_only_gate() -> None:
        nonlocal franka_gate_check_count, franka_gate_last, franka_gate_failure
        franka_gate_check_count += 1
        try:
            state = robot.read_once()
            arm._validate_state(
                state,
                require_idle=True,
                enforce_success=False,
                enforce_joint_limit_margin=False,
            )
            franka_gate_last = _franka_state_json(state)
        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            if franka_gate_failure is None:
                franka_gate_failure = message
            raise RuntimeError(
                "continuous Franka read-only safety gate failed: " + message
            ) from exc

    driver = None
    initial_hand = None
    final_hand = None
    q6_waypoints = _requested_q6_waypoints(args.target_q6, args.q6_step)
    q6_return_waypoints = _requested_q6_return_waypoints(
        args.target_q6, args.q6_step
    )
    adopted_disabled = False
    q6_sweep_pass = False
    q6_return_pass = False
    coupled_pass = False
    reopened = False
    disabled = False
    feedback_envelope_policy_enforced = False
    feedback_envelope_policy_sha256: Optional[str] = None
    primary_error: Optional[str] = None
    stop_error: Optional[str] = None
    final_arm = None
    preopen_reset_q6_waypoints: tuple[int, ...] = ()
    q6_open_min_angle = _profile_q6_open_min_angle(config)
    try:
        driver = RH56InstalledCommissioningDriver.connect(
            port=str(config["inspire"]["port"]),
            baud=int(config["inspire"]["baud"]),
            hand_id=int(config["inspire"]["hand_id"]),
            thumb_rotate_range=(0, 1000),
            motion_timeout_s=float(config["inspire"]["motion_timeout_s"]),
            angle_tolerance=int(config["inspire"]["arrival_tolerance_units"]),
            thumb_preshape_step_units=min(int(args.q6_step), 50),
            stop_max_axis_current_ma=STOP_MAX_AXIS_CURRENT_MA,
            q6_open_min_angle=q6_open_min_angle,
        )
        expected_feedback_policy = rh56_feedback_envelope_policy(
            int(config["inspire"]["arrival_tolerance_units"])
        )
        if driver.feedback_envelope_policy != expected_feedback_policy:
            raise RuntimeError(
                "RH56 runtime feedback-envelope policy differs from commissioning request"
            )
        feedback_envelope_policy_enforced = True
        feedback_envelope_policy_sha256 = str(
            expected_feedback_policy["sha256"]
        )
        # The first post-connect RH56 operation must be an all-six disable.
        # Do not leave a stale numeric target active while pylibfranka is
        # constructed or the installed-load gate is evaluated.
        driver.adopt_disabled_state_and_verify()
        adopted_disabled = True
        initial_hand = _rh56_snapshot_json(driver.hand.snapshot())
        if int(initial_hand.get("hand_id", -1)) != int(config["inspire"]["hand_id"]):
            raise RuntimeError("RH56 HAND_ID register differs from the control profile")
        _require_fresh_commissioning_q6_open(
            initial_hand,
            q6_open_min_angle=q6_open_min_angle,
        )

        robot = pylibfranka.Robot(
            str(config["franka"]["ip"]), pylibfranka.RealtimeConfig.kIgnore
        )
        limits = _build_limits(config, FrankaMotionLimits)
        arm = FrankaSequenceDriver(robot, pylibfranka, limits)
        initial_arm_state = robot.read_once()
        arm._validate_state(
            initial_arm_state,
            require_idle=True,
            enforce_success=False,
            enforce_joint_limit_margin=False,
        )
        franka_initial = _franka_state_json(initial_arm_state)
        # Installation is one-shot and only permitted while the hand is still
        # verified disabled.  Every subsequent driver feedback read is bound
        # to a fresh Franka read-only safety gate.
        driver.install_external_safety_check(continuous_franka_read_only_gate)
        preopen_reset_q6_waypoints = driver.reset_to_open(
            max_axis_current_ma=int(args.max_axis_current_ma),
            endpoint_stable_samples=int(args.endpoint_stable_samples),
            max_inactive_drift_units=int(args.max_inactive_drift_units),
        )
        if tuple(preopen_reset_q6_waypoints):
            raise RuntimeError(
                "fresh commissioning must not perform pre-open q6 recovery; "
                "use the evidence-bound recovery workflow first"
            )
        reached_waypoints = driver.commission_thumb_sweep(
            int(args.target_q6),
            step_units=int(args.q6_step),
            max_axis_current_ma=int(args.max_axis_current_ma),
            endpoint_stable_samples=int(args.endpoint_stable_samples),
            max_inactive_drift_units=int(args.max_inactive_drift_units),
        )
        if tuple(reached_waypoints) != q6_waypoints:
            raise RuntimeError("RH56 driver returned an unexpected q6 waypoint path")
        q6_sweep_pass = True
        if args.coupled_air_close:
            driver.commission_coupled_air_close(
                tuple(int(value) for value in args.bend_targets),
                max_axis_current_ma=int(args.max_axis_current_ma),
                endpoint_stable_samples=int(args.endpoint_stable_samples),
                minimum_bend_target=0,
                maximum_bend_target=1000,
                step_units=int(args.q6_step),
            )
            coupled_pass = True
        returned_waypoints = driver.return_commissioned_thumb_to_open(
            max_axis_current_ma=int(args.max_axis_current_ma),
            endpoint_stable_samples=int(args.endpoint_stable_samples),
            max_inactive_drift_units=int(args.max_inactive_drift_units),
            direct_q6_return_to_open=(int(args.target_q6) == 900),
        )
        if tuple(returned_waypoints) != q6_return_waypoints:
            raise RuntimeError("RH56 driver returned an unexpected q6 return path")
        q6_return_pass = True
        reopened = True
    except BaseException as exc:
        primary_error = f"{type(exc).__name__}: {exc}"
    finally:
        if driver is not None:
            try:
                driver.disable_and_verify()
                disabled = True
                final_hand = _rh56_snapshot_json(driver.hand.snapshot())
            except BaseException as exc:
                stop_error = f"{type(exc).__name__}: {exc}"
                print(
                    "[hardware][STOP UNCONFIRMED] cut RH56 24 V immediately if "
                    "the hand is moving: " + stop_error,
                    file=sys.stderr,
                    flush=True,
                )
            try:
                driver.close()
            except BaseException as exc:
                message = f"{type(exc).__name__}: {exc}"
                stop_error = message if stop_error is None else f"{stop_error}; {message}"
                disabled = False
                print(
                    "[hardware][STOP UNCONFIRMED] cut RH56 24 V immediately if "
                    "the hand is moving: " + message,
                    file=sys.stderr,
                    flush=True,
                )
        if robot is not None and arm is not None:
            try:
                final_arm_state = robot.read_once()
                arm._validate_state(
                    final_arm_state,
                    require_idle=True,
                    enforce_success=False,
                    enforce_joint_limit_margin=False,
                )
                final_arm = _franka_state_json(final_arm_state)
            except BaseException as exc:
                message = (
                    "final Franka read-only gate failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                primary_error = (
                    message
                    if primary_error is None
                    else f"{primary_error}; {message}"
                )

    telemetry = [] if driver is None else list(driver.telemetry)
    coupled_waypoints = (
        _requested_coupled_waypoints(
            args.bend_targets, int(args.target_q6), int(args.q6_step)
        )
        if args.coupled_air_close
        else ()
    )
    coupled_return_waypoints = _requested_coupled_return_waypoints(
        coupled_waypoints, int(args.target_q6), int(args.q6_step)
    )
    observations = group_telemetry(
        telemetry,
        q6_waypoints,
        q6_return_waypoints=q6_return_waypoints,
        coupled_waypoints=coupled_waypoints,
        coupled_return_waypoints=coupled_return_waypoints,
    )
    observations["preopen_reset_q6_waypoints"] = list(
        preopen_reset_q6_waypoints
    )
    last_disable = (
        observations["disable_feedback"][-1]
        if observations["disable_feedback"]
        else None
    )
    final = {
        "angle_targets": None if last_disable is None else last_disable["angle_targets"],
        "angles": None if last_disable is None else last_disable["angles"],
        "currents": None if last_disable is None else last_disable["currents"],
        "errors": None if last_disable is None else last_disable["errors"],
        "statuses": None if last_disable is None else last_disable["statuses"],
        "temperatures": None if last_disable is None else last_disable["temperatures"],
        "reopened_and_verified": bool(reopened),
        "disabled_verified": bool(disabled),
        "snapshot_after_disable": final_hand,
    }
    passed = (
        primary_error is None
        and stop_error is None
        and adopted_disabled
        and q6_sweep_pass
        and q6_return_pass
        and reopened
        and disabled
        and feedback_envelope_policy_enforced
        and feedback_envelope_policy_sha256 is not None
        and (coupled_pass if args.coupled_air_close else True)
        and final_arm is not None
        and franka_gate_check_count > 0
        and franka_gate_failure is None
    )
    return {
        "franka_read_only": {
            "connection": "read_once_only_no_controller_no_robot_write",
            "initial": franka_initial,
            "final": final_arm,
            "continuous_gate": {
                "check_count": int(franka_gate_check_count),
                "last_success": franka_gate_last,
                "failure": franka_gate_failure,
            },
            "verified": (
                final_arm is not None
                and franka_gate_check_count > 0
                and franka_gate_failure is None
            ),
        },
        "rh56_device": {
            "port": str(config["inspire"]["port"]),
            "resolved_port": str(Path(str(config["inspire"]["port"])).resolve()),
            "hand_id": int(config["inspire"]["hand_id"]),
            "baud": int(config["inspire"]["baud"]),
            "initial_snapshot": initial_hand,
        },
        "result": {
            "status": "pass" if passed else "fail",
            "adopted_disabled_verified": bool(adopted_disabled),
            "q6_sweep_pass": bool(q6_sweep_pass),
            "q6_return_pass": bool(q6_return_pass),
            "coupled_closure_pass": bool(coupled_pass),
            "reopened_and_verified": bool(reopened),
            "disabled_verified": bool(disabled),
            "feedback_envelope_enforced": bool(
                feedback_envelope_policy_enforced
            ),
            "feedback_envelope_policy_sha256": (
                feedback_envelope_policy_sha256
            ),
            "operation_error": primary_error,
            "stop_error": stop_error,
        },
        "observations": observations,
        "final": final,
        "q6_waypoints": list(q6_waypoints),
        "preopen_reset_q6_waypoints": list(preopen_reset_q6_waypoints),
    }


def _run_command(args: argparse.Namespace) -> int:
    config, config_path = load_control_config(args.config)
    assets = verify_adapter_assets(config, config_path)
    snapshot_candidate = _resolve_snapshot_targets(args, config)
    _validate_run_request(args, config)
    snapshot_state_before = (
        _snapshot_file_state(Path(snapshot_candidate["path"]))
        if snapshot_candidate is not None
        else None
    )
    stage1_prerequisite = None
    if int(args.target_q6) < STAGE1_Q6_TARGET:
        stage1_prerequisite = build_stage1_prerequisite_binding(
            args.stage1_evidence,
            expected_config=config,
            expected_config_path=config_path,
        )
    sources = _source_bindings(assets)
    config_sha256_before = sha256_file(config_path)
    started = _utc_now()
    run_id = str(uuid.uuid4())
    requested_waypoints = _requested_q6_waypoints(args.target_q6, args.q6_step)
    requested_return_waypoints = _requested_q6_return_waypoints(
        args.target_q6, args.q6_step
    )
    requested_coupled_waypoints = (
        _requested_coupled_waypoints(
            args.bend_targets, args.target_q6, args.q6_step
        )
        if args.coupled_air_close
        else ()
    )
    requested_coupled_return_waypoints = _requested_coupled_return_waypoints(
        requested_coupled_waypoints, args.target_q6, args.q6_step
    )
    try:
        runtime = _run_hardware_session(args, config)
    except BaseException as exc:
        # A constructor/import failure happened after confirmations but before a
        # driver could provide structured recovery evidence.
        print(
            "[hardware][STOP UNCONFIRMED] RH56 connection/stop state is unavailable; "
            "cut 24 V immediately if the hand is moving",
            file=sys.stderr,
            flush=True,
        )
        runtime = {
            "franka_read_only": None,
            "rh56_device": None,
            "result": {
                "status": "fail",
                "adopted_disabled_verified": False,
                "q6_sweep_pass": False,
                "q6_return_pass": False,
                "coupled_closure_pass": False,
                "reopened_and_verified": False,
                "disabled_verified": False,
                "feedback_envelope_enforced": False,
                "feedback_envelope_policy_sha256": None,
                "operation_error": f"{type(exc).__name__}: {exc}",
                "stop_error": "RH56 connection/stop state unavailable",
            },
            "observations": {
                "all_feedback": [],
                "adopt_disable_feedback": [],
                "q6_steps": [],
                "q6_return_steps": [],
                "coupled_air_close_feedback": [],
                "coupled_air_close_steps": [],
                "coupled_air_return_steps": [],
                "final_open_feedback": None,
                "disable_feedback": [],
                "actual_q6_range": None,
                "actual_q6_return_range": None,
                "preopen_reset_q6_waypoints": [],
            },
            "final": {
                "angle_targets": None,
                "angles": None,
                "currents": None,
                "errors": None,
                "statuses": None,
                "temperatures": None,
                "reopened_and_verified": False,
                "disabled_verified": False,
                "snapshot_after_disable": None,
            },
            "q6_waypoints": list(requested_waypoints),
            "preopen_reset_q6_waypoints": [],
        }
        runtime["observations"]["q6_steps"] = [
            {"target_q6": int(target), "feedback": []}
            for target in requested_waypoints
        ]
        runtime["observations"]["q6_return_steps"] = [
            {"target_q6": int(target), "feedback": []}
            for target in requested_return_waypoints
        ]

    config_sha256_after = sha256_file(config_path)
    if config_sha256_after != config_sha256_before:
        _record_runtime_failure(
            runtime, "control profile changed during the commissioning run"
        )
    if snapshot_candidate is not None:
        try:
            snapshot_path = Path(snapshot_candidate["path"])
            snapshot_state_after = _snapshot_file_state(snapshot_path)
            snapshot_sha256_after = sha256_file(snapshot_path)
        except OSError as exc:
            _record_runtime_failure(
                runtime,
                f"official snapshot became unavailable during the commissioning run: {exc}",
            )
        else:
            if (
                snapshot_state_after != snapshot_state_before
                or snapshot_sha256_after != snapshot_candidate["file_sha256"]
            ):
                _record_runtime_failure(
                    runtime, "official snapshot changed during the commissioning run"
                )

    request = {
        "safety_scope": "installed_on_FR3_PLA_low_speed_unloaded_free_air_only",
        "target_q6": int(args.target_q6),
        "requested_q6_range": [int(args.target_q6), 1000],
        "step_units": int(args.q6_step),
        "q6_waypoints": list(requested_waypoints),
        "q6_return_waypoints": list(requested_return_waypoints),
        "q6_return_strategy": (
            "direct_stage1_v1"
            if int(args.target_q6) == 900
            else "canonical_reverse_bootstrap_v3"
        ),
        "speed": COMMISSIONING_SPEED,
        "force_limit_g": COMMISSIONING_FORCE_G,
        "motion_timeout_s_per_step": float(config["inspire"]["motion_timeout_s"]),
        "angle_tolerance_units": int(config["inspire"]["arrival_tolerance_units"]),
        "q6_open_min_angle": _profile_q6_open_min_angle(config),
        "feedback_envelope_policy": rh56_feedback_envelope_policy(
            int(config["inspire"]["arrival_tolerance_units"])
        ),
        "q6_endpoint_tolerance_units": min(
            int(config["inspire"]["arrival_tolerance_units"]), 20
        ),
        "q6_reverse_hysteresis_tolerance_units": (
            Q6_REVERSE_HYSTERESIS_TOLERANCE_UNITS
        ),
        "endpoint_stable_samples": int(args.endpoint_stable_samples),
        "max_axis_current_ma": int(args.max_axis_current_ma),
        "stop_max_axis_current_ma": STOP_MAX_AXIS_CURRENT_MA,
        "max_inactive_drift_units": int(args.max_inactive_drift_units),
        "aggregate_current_policy": "telemetry_only_device_limits_remain_active",
        "target_source": (
            "official_snapshot_candidate"
            if snapshot_candidate is not None
            else "manual_cli"
        ),
        "coupled_closure_requested": bool(args.coupled_air_close),
        "coupled_targets": (
            [int(value) for value in args.bend_targets] + [int(args.target_q6)]
            if args.coupled_air_close
            else None
        ),
        "coupled_step_units": int(args.q6_step) if args.coupled_air_close else None,
        "coupled_command_waypoints": (
            [list(item) for item in requested_coupled_waypoints]
            if args.coupled_air_close
            else None
        ),
        "coupled_return_command_waypoints": (
            [list(item) for item in requested_coupled_return_waypoints]
            if args.coupled_air_close
            else None
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "run_id": run_id,
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "motion_authorized": False,
        "control_profile": {
            "path": str(config_path),
            "file_sha256": config_sha256_before,
            "parsed_sha256": json_sha256(config),
            "snapshot": copy.deepcopy(config),
        },
        "snapshot_candidate": snapshot_candidate,
        "stage1_prerequisite": stage1_prerequisite,
        "source_bindings": sources,
        "operator_confirmations": {
            "installed_on_fr3": True,
            "24v_cutoff_ready": True,
            "franka_stop_ready": True,
            "workspace_clear": True,
            "no_contact_PLA_scope": True,
            "wide_q6_confirmed": int(args.target_q6)
            < int(config["inspire"]["thumb_rotate_validated_realtime_range"][0]),
            "coupled_air_close_confirmed": bool(args.coupled_air_close),
            "exact_air_target_confirmed": (
                args.confirm_exact_air_target == EXACT_AIR_TARGET_TOKEN
            ),
        },
        "request": request,
        "franka_read_only": runtime["franka_read_only"],
        "rh56_device": runtime["rh56_device"],
        "result": runtime["result"],
        "observations": runtime["observations"],
        "final": runtime["final"],
    }
    evidence = seal_evidence(payload)
    file_digest = atomic_write_evidence(args.output, evidence)
    print(f"[evidence] {Path(args.output).expanduser().resolve()}")
    print(f"[evidence] file_sha256={file_digest}")
    print(
        "[result] status={} q6_sweep={} coupled={} reopened={} disabled={}".format(
            evidence["result"]["status"],
            evidence["result"]["q6_sweep_pass"],
            evidence["result"]["coupled_closure_pass"],
            evidence["result"]["reopened_and_verified"],
            evidence["result"]["disabled_verified"],
        )
    )
    if evidence["result"]["status"] != "pass":
        if (
            evidence["result"].get("disabled_verified") is True
            and evidence["result"].get("stop_error") is None
        ):
            print(
                "[safe stop verified] motion acceptance failed, but final "
                "all-six ANGLE_SET=-1 was verified; inspect the evidence "
                "before any retry",
                file=sys.stderr,
            )
        else:
            print(
                "[STOP UNCONFIRMED] inspect evidence and cut 24 V immediately "
                "if the hand is moving",
                file=sys.stderr,
            )
        return 1
    verification = verify_evidence(
        Path(args.output),
        config_path=config_path,
        require_coupled=bool(args.coupled_air_close),
    )
    if not verification.passed:
        print("[evidence self-check] LOCKED", file=sys.stderr)
        for blocker in verification.blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 1
    print("[evidence self-check] PASS")
    print(
        "[config] unchanged; run verify/propose-config-update, then "
        "materialize-config-update after manual evidence review"
    )
    return 0


def _verify_command(args: argparse.Namespace, *, applied: bool) -> int:
    result = (
        verify_applied_config(
            args.evidence,
            args.config,
            require_coupled=bool(args.require_coupled),
        )
        if applied
        else verify_evidence(
            args.evidence,
            config_path=args.config,
            require_coupled=bool(args.require_coupled),
        )
    )
    if not result.passed:
        print("[verify] LOCKED", file=sys.stderr)
        for blocker in result.blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 2
    print("[verify] PASS")
    print(json.dumps(result.proposal, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _propose_command(args: argparse.Namespace) -> int:
    result = verify_evidence(
        args.evidence,
        config_path=args.config,
        require_coupled=bool(args.require_coupled),
    )
    if not result.passed:
        for blocker in result.blockers:
            print(f"[proposal blocker] {blocker}", file=sys.stderr)
        return 2
    evidence, _ = load_evidence(args.evidence)
    proposal = derived_config_proposal(evidence)
    print(json.dumps(proposal, ensure_ascii=False, sort_keys=True, indent=2))
    print(
        "[proposal only] no config file was modified; audit this evidence and "
        "use materialize-config-update to create the exact reviewed profile",
        file=sys.stderr,
    )
    return 0


def _materialize_command(args: argparse.Namespace) -> int:
    """Create, but never overwrite, the exact evidence-derived profile.

    Keeping the original commissioning profile immutable makes the Stage-1
    and Stage-2 evidence lineage replayable.  The derived profile must live
    beside it so relative adapter asset paths keep the same meaning.
    """

    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output_config).expanduser().resolve()
    if output_path == config_path:
        raise ValueError("materialize-config-update refuses to overwrite the source profile")
    if output_path.parent != config_path.parent:
        raise ValueError(
            "--output-config must be beside --config so relative asset paths remain bound"
        )
    if output_path.exists():
        raise FileExistsError(f"derived config already exists: {output_path}")

    result = verify_evidence(
        args.evidence,
        config_path=config_path,
        require_coupled=True,
    )
    if not result.passed:
        print("[materialize] LOCKED", file=sys.stderr)
        for blocker in result.blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 2

    evidence, _ = load_evidence(args.evidence)
    profile = evidence.get("control_profile", {}).get("snapshot")
    if not isinstance(profile, Mapping):
        raise ValueError("verified evidence has no bound control-profile snapshot")
    proposal = derived_config_proposal(evidence)
    updated = copy.deepcopy(dict(profile))
    inspire = updated.get("inspire")
    if not isinstance(inspire, dict):
        raise ValueError("bound control profile has no mutable inspire object")
    inspire["thumb_rotate_validated_realtime_range"] = proposal[
        "inspire.thumb_rotate_validated_realtime_range"
    ]
    inspire["six_axis_coupled_closure_commissioned"] = proposal[
        "inspire.six_axis_coupled_closure_commissioned"
    ]
    inspire["commissioned_air_closure_targets"] = proposal[
        "inspire.commissioned_air_closure_targets"
    ]

    digest = atomic_write_evidence(output_path, updated)
    applied = verify_applied_config(
        args.evidence,
        output_path,
        require_coupled=True,
    )
    if not applied.passed:
        print(
            "[materialize] LOCKED after write; the read-only derived file is not "
            "accepted and must not be used",
            file=sys.stderr,
        )
        for blocker in applied.blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 2
    print("[materialize] PASS")
    print(f"[materialize] source_config={config_path}")
    print(f"[materialize] derived_config={output_path}")
    print(f"[materialize] derived_file_sha256={digest}")
    print(
        "[materialize] original profile unchanged; pass --config "
        f"{output_path} to subsequent plan/audit/control commands"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Commission installed RH56 q6/coupled free-air motion with "
            "tamper-evident local records"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="explicitly confirmed hardware run")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(_target_q6_supplied=False, _bend_targets_supplied=False)
    run.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help=(
            "strict official snapshot supplying all six coupled targets; when "
            "--candidate-index is omitted, the greatest official score wins"
        ),
    )
    run.add_argument(
        "--candidate-index",
        type=int,
        default=None,
        help="immutable snapshot index; explicit 0 selects the first candidate",
    )
    run.add_argument(
        "--target-q6",
        type=int,
        default=900,
        action=_StoreManualTarget,
    )
    run.add_argument(
        "--stage1-evidence",
        type=Path,
        default=None,
        help=(
            "required for every target below q6=900; must be a current-source, "
            "same-profile, q6-only formal Stage1 PASS"
        ),
    )
    run.add_argument("--q6-step", type=int, default=25)
    run.add_argument("--max-axis-current-ma", type=int, default=400)
    run.add_argument(
        "--max-inactive-drift-units", type=int, default=8, choices=range(1, 21)
    )
    run.add_argument("--endpoint-stable-samples", type=int, default=3, choices=range(2, 11))
    run.add_argument("--coupled-air-close", action="store_true")
    run.add_argument(
        "--bend-targets",
        type=int,
        nargs=5,
        default=[900] * 5,
        action=_StoreManualTarget,
    )
    run.add_argument("--confirm-installed", metavar=INSTALLED_TOKEN)
    run.add_argument("--confirm-24v-cutoff", metavar=POWER_TOKEN)
    run.add_argument("--confirm-franka-stop", metavar=STOP_TOKEN)
    run.add_argument("--confirm-workspace-clear", metavar=CLEAR_TOKEN)
    run.add_argument("--confirm-no-contact", metavar=NO_CONTACT_TOKEN)
    run.add_argument("--confirm-wide-q6", metavar=WIDE_Q6_TOKEN)
    run.add_argument("--confirm-coupled-closure", metavar=COUPLED_TOKEN)
    run.add_argument("--confirm-exact-air-target", metavar=EXACT_AIR_TARGET_TOKEN)

    for command, help_text in (
        ("verify", "verify evidence checksums against its unmodified input profile"),
        ("verify-applied", "verify an exact manually applied three-field profile update"),
        ("propose-config-update", "print, but never apply, the evidence-derived patch"),
    ):
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument("--evidence", type=Path, required=True)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        sub.add_argument("--require-coupled", action="store_true")
    materialize = subparsers.add_parser(
        "materialize-config-update",
        help=(
            "create a new read-only profile containing exactly the three "
            "fields derived from a coupled PASS"
        ),
    )
    materialize.add_argument("--evidence", type=Path, required=True)
    materialize.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    materialize.add_argument("--output-config", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "verify":
            return _verify_command(args, applied=False)
        if args.command == "verify-applied":
            return _verify_command(args, applied=True)
        if args.command == "propose-config-update":
            return _propose_command(args)
        if args.command == "materialize-config-update":
            return _materialize_command(args)
        raise RuntimeError(f"unsupported command {args.command}")
    except KeyboardInterrupt:
        print(
            "[interrupted] cut RH56 24 V immediately if any motion remains",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[failed] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
