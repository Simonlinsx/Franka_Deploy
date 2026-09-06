#!/usr/bin/env python3
"""Recover an evidence-bound interrupted RH56 coupled-air run to open.

The command never moves Franka.  It keeps Franka under a continuous read-only
Idle/load gate while it reopens the already-commanded RH56 bend prefix, returns
q6 over the canonical small-step path, and finally verifies all-six
``ANGLE_SET=-1`` plus stable idle feedback.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
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
    atomic_write_evidence,
    json_sha256,
    seal_evidence,
    sha256_file,
)
from anydex_pipeline.rh56_interrupted_recovery import (
    InterruptedRecoveryDriver,
    InterruptedRecoveryPlan,
    RECOVERY_EVIDENCE_KIND,
    RECOVERY_MODE_Q6_RETURN,
    RECOVERY_ROUTE_NEAR_OPEN_RESET,
    RECOVERY_ROUTE_SEALED_Q6_RETURN,
    build_interrupted_recovery_plan,
)
from anydex_pipeline.inspire_sequence_driver import RH56StopUnconfirmed
from anydex_pipeline.rh56_reset_open import (
    DEFAULT_FORCES,
    DEFAULT_SPEEDS,
    RH56ResetSettingsRestoreError,
)


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
POWER_TOKEN = "RH56_24V_CUTOFF_READY"
STOP_TOKEN = "FR3_STOP_READY"
CLEAR_TOKEN = "INSTALLED_AIR_WORKSPACE_CLEAR"
NO_CONTACT_TOKEN = "PLA_LOW_SPEED_NO_CONTACT"
RECOVERY_TOKEN = "RH56_INTERRUPTED_OPEN_RECOVERY"
STOP_MAX_AXIS_CURRENT_MA = 100


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        "F_x_Cee_m": [float(value) for value in np.asarray(state.F_x_Cee)],
        "I_ee_kg_m2": np.asarray(state.I_ee, dtype=np.float64).reshape(-1).tolist(),
        "m_load_kg": float(state.m_load),
        "m_total_kg": float(state.m_total),
        "read_at_unix_s": float(time.time()),
    }


def _rh56_snapshot_json(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in snapshot.items():
        if isinstance(value, (tuple, list)):
            output[str(key)] = [int(item) for item in value]
        elif isinstance(value, (int, float, str, bool)) or value is None:
            output[str(key)] = value
        else:
            output[str(key)] = str(value)
    return output


def _build_limits(config: Mapping[str, Any], limits_type: Any) -> Any:
    franka = config["franka"]
    dynamics = franka["expected_end_effector"]
    return limits_type(
        expected_F_T_EE=np.asarray(franka["expected_F_T_EE"], dtype=np.float64),
        expected_m_ee_kg=float(dynamics["mass_kg"]),
        expected_F_x_Cee_m=np.asarray(dynamics["F_x_Cee_m"], dtype=np.float64),
        expected_I_ee_kg_m2=np.asarray(dynamics["inertia_kg_m2"], dtype=np.float64),
        joint_limits_rad=np.asarray(franka["joint_limits_rad"], dtype=np.float64),
        joint_limit_margin_rad=float(franka["joint_limit_margin_rad"]),
        max_joint_speed_rad_s=float(franka["default_max_joint_velocity_rad_s"]),
        max_joint_segment_rad=float(franka["default_max_joint_segment_rad"]),
        min_joint_duration_s=float(franka["default_min_duration_s"]),
        joint_arrival_tolerance_rad=float(franka["default_arrival_tolerance_rad"]),
    )


def _source_bindings(assets: Any) -> list[dict[str, str]]:
    paths = {
        "recovery_cli": Path(__file__).resolve(),
        "recovery_module": (
            ROOT / "src/anydex_pipeline/rh56_interrupted_recovery.py"
        ).resolve(),
        "rh56_sequence_driver": (
            ROOT / "src/anydex_pipeline/inspire_sequence_driver.py"
        ).resolve(),
        "rh56_hand_path": (ROOT / "src/anydex_pipeline/rh56_hand_path.py").resolve(),
        "rh56_reset_open": (
            ROOT / "src/anydex_pipeline/rh56_reset_open.py"
        ).resolve(),
        "rh56_register_api": (WORKSPACE / "examples/inspire_rh56_test.py").resolve(),
        "franka_sequence_driver": (
            ROOT / "src/anydex_pipeline/franka_sequence_driver.py"
        ).resolve(),
        "adapter_mesh": Path(assets.mesh_path).resolve(),
        "adapter_provenance": Path(assets.provenance_path).resolve(),
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("required recovery provenance is missing: " + "; ".join(missing))
    return [
        {"name": name, "path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    ]


def _validate_run_request(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    required = (
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-24v-cutoff", args.confirm_24v_cutoff, POWER_TOKEN),
        ("--confirm-franka-stop", args.confirm_franka_stop, STOP_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, CLEAR_TOKEN),
        ("--confirm-no-contact", args.confirm_no_contact, NO_CONTACT_TOKEN),
        ("--confirm-recovery", args.confirm_recovery, RECOVERY_TOKEN),
    )
    missing = [f"{flag} {token}" for flag, actual, token in required if actual != token]
    if missing:
        raise ValueError("exact confirmations required before hardware import: " + "; ".join(missing))
    if not 50 <= int(args.max_axis_current_ma) <= 400:
        raise ValueError("--max-axis-current-ma must be in 50..400")
    if not 1 <= int(args.max_inactive_drift_units) <= 20:
        raise ValueError("--max-inactive-drift-units must be in 1..20")
    if Path(args.output).expanduser().resolve().exists():
        raise FileExistsError(f"recovery evidence output already exists: {Path(args.output).resolve()}")
    inspire = config["inspire"]
    if int(inspire["open_speed"]) > 40 or int(inspire["close_speed"]) > 40:
        raise ValueError("profile hand speeds exceed the fixed recovery speed 40")
    if int(inspire["force_limit_g"]) > 80:
        raise ValueError("profile force limit exceeds the fixed recovery limit 80g")


def _run_hardware_session(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    plan: InterruptedRecoveryPlan,
) -> dict[str, Any]:
    # Hardware imports remain below every token/evidence/profile check.
    import pylibfranka
    from anydex_pipeline.franka_sequence_driver import (
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )

    driver = None
    robot = None
    arm = None
    initial_hand = None
    final_hand = None
    franka_initial = None
    franka_final = None
    gate_count = 0
    gate_last = None
    gate_failure = None
    adopted = False
    recovered = False
    disabled = False
    executed_q6_return_waypoints: tuple[int, ...] = ()
    executed_reset_q6_waypoints: tuple[int, ...] = ()
    recovery_route = None
    recovery_route_initial_q6 = None
    operation_error = None
    stop_error = None
    cleanup_errors: list[str] = []

    def franka_read_only_gate() -> None:
        nonlocal gate_count, gate_last, gate_failure
        gate_count += 1
        try:
            state = robot.read_once()
            arm._validate_state(
                state,
                require_idle=True,
                enforce_success=False,
                enforce_joint_limit_margin=False,
            )
            gate_last = _franka_state_json(state)
        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            if gate_failure is None:
                gate_failure = message
            raise RuntimeError("continuous Franka read-only safety gate failed: " + message) from exc

    try:
        driver = InterruptedRecoveryDriver.connect(
            port=str(config["inspire"]["port"]),
            baud=int(config["inspire"]["baud"]),
            hand_id=int(config["inspire"]["hand_id"]),
            thumb_rotate_range=(0, 1000),
            motion_timeout_s=float(config["inspire"]["motion_timeout_s"]),
            angle_tolerance=int(config["inspire"]["arrival_tolerance_units"]),
            thumb_preshape_step_units=plan.step_units,
            stop_max_axis_current_ma=STOP_MAX_AXIS_CURRENT_MA,
        )
        driver.adopt_disabled_state_and_verify()
        adopted = True
        initial_hand = _rh56_snapshot_json(driver.hand.snapshot())
        if int(initial_hand.get("hand_id", -1)) != int(config["inspire"]["hand_id"]):
            raise RuntimeError("RH56 HAND_ID differs from the control profile")

        robot = pylibfranka.Robot(
            str(config["franka"]["ip"]), pylibfranka.RealtimeConfig.kIgnore
        )
        arm = FrankaSequenceDriver(robot, pylibfranka, _build_limits(config, FrankaMotionLimits))
        state = robot.read_once()
        arm._validate_state(
            state,
            require_idle=True,
            enforce_success=False,
            enforce_joint_limit_margin=False,
        )
        franka_initial = _franka_state_json(state)
        driver.install_external_safety_check(franka_read_only_gate)
        returned = driver.recover_interrupted_coupled_air_close_to_open(
            plan,
            max_axis_current_ma=int(args.max_axis_current_ma),
            endpoint_stable_samples=int(args.endpoint_stable_samples),
            max_inactive_drift_units=int(args.max_inactive_drift_units),
        )
        recovery_route = driver.last_recovery_route
        recovery_route_initial_q6 = driver.last_recovery_initial_q6
        if recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET:
            executed_reset_q6_waypoints = tuple(int(value) for value in returned)
        else:
            executed_q6_return_waypoints = tuple(int(value) for value in returned)
        if (
            plan.recovery_mode != RECOVERY_MODE_Q6_RETURN
            and executed_q6_return_waypoints != plan.q6_return_waypoints
        ):
            raise RuntimeError("driver returned an unexpected q6 recovery path")
        if (
            recovery_route == RECOVERY_ROUTE_SEALED_Q6_RETURN
            and (
                not executed_q6_return_waypoints
                or executed_q6_return_waypoints[-1] != 1000
            )
        ):
            raise RuntimeError("driver returned an incomplete live-anchored q6 recovery path")
        if (
            recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET
            and (
                any(
                    not plan.profile_near_open_q6_range[0]
                    <= value
                    <= plan.profile_near_open_q6_range[1]
                    for value in executed_reset_q6_waypoints
                )
                or (
                    executed_reset_q6_waypoints
                    and executed_reset_q6_waypoints[-1]
                    != plan.profile_near_open_q6_range[1]
                )
            )
        ):
            raise RuntimeError("driver returned an invalid profile near-open reset path")
        recovered = True
    except BaseException as exc:
        if driver is not None:
            recovery_route = driver.last_recovery_route
            recovery_route_initial_q6 = driver.last_recovery_initial_q6
        operation_error = f"{type(exc).__name__}: {exc}"
    finally:
        if driver is not None:
            try:
                driver.disable_and_verify()
                disabled = True
                final_hand = _rh56_snapshot_json(driver.hand.snapshot())
            except RH56ResetSettingsRestoreError as exc:
                # The near-open reset cleanup has already proven the physical
                # all-six stop.  Preserve that fact and report only the later
                # default-setting restore failure.
                disabled = True
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
                try:
                    final_hand = _rh56_snapshot_json(driver.hand.snapshot())
                except BaseException as snapshot_exc:
                    cleanup_errors.append(
                        "final snapshot after verified stop failed: "
                        f"{type(snapshot_exc).__name__}: {snapshot_exc}"
                    )
            except RH56StopUnconfirmed as exc:
                stop_error = f"{type(exc).__name__}: {exc}"
                print(
                    "[hardware][STOP UNCONFIRMED] cut RH56 24 V immediately if the hand is moving: "
                    + stop_error,
                    file=sys.stderr,
                    flush=True,
                )
            try:
                driver.close()
            except RH56StopUnconfirmed as exc:
                message = f"{type(exc).__name__}: {exc}"
                stop_error = message if stop_error is None else f"{stop_error}; {message}"
                disabled = False
            except BaseException as exc:
                message = f"{type(exc).__name__}: {exc}"
                if recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET and disabled:
                    cleanup_errors.append(message)
                else:
                    stop_error = (
                        message if stop_error is None else f"{stop_error}; {message}"
                    )
                    disabled = False
        if robot is not None and arm is not None:
            try:
                state = robot.read_once()
                arm._validate_state(
                    state,
                    require_idle=True,
                    enforce_success=False,
                    enforce_joint_limit_margin=False,
                )
                franka_final = _franka_state_json(state)
            except BaseException as exc:
                message = f"final Franka read-only gate failed: {type(exc).__name__}: {exc}"
                operation_error = message if operation_error is None else f"{operation_error}; {message}"

    telemetry = [] if driver is None else [asdict(item) for item in driver.telemetry]
    for item in telemetry:
        for key in (
            "angle_targets", "angles", "positions", "forces", "currents",
            "errors", "statuses", "temperatures",
        ):
            if item.get(key) is not None:
                item[key] = list(item[key])
    last = telemetry[-1] if telemetry else None
    route_dispatch_samples = [
        item
        for item in telemetry
        if item.get("phase") == "boundary_state_snapshot"
    ]
    route_dispatch = (
        route_dispatch_samples[0]
        if len(route_dispatch_samples) == 1
        else None
    )
    near_open_route = recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET
    expected_speeds = DEFAULT_SPEEDS if near_open_route else plan.original_speeds
    expected_forces = DEFAULT_FORCES if near_open_route else plan.original_forces
    settings_restore_target = (
        "reset_defaults" if near_open_route else "failed_run_snapshot"
    )
    final_angles_ok = (
        last is not None
        and all(int(value) >= 980 for value in last["angles"][:5])
        and int(last["angles"][5])
        >= (
            int(plan.q6_open_min_angle)
            if near_open_route
            else 980
        )
    )
    final_currents_ok = (
        last is not None
        and (
            all(
                abs(int(value)) <= STOP_MAX_AXIS_CURRENT_MA
                for value in last["currents"]
            )
            if near_open_route
            else all(
                abs(int(value)) <= STOP_MAX_AXIS_CURRENT_MA
                for value in last["currents"]
            )
        )
    )
    passed = (
        operation_error is None
        and stop_error is None
        and not cleanup_errors
        and adopted
        and recovered
        and disabled
        and franka_final is not None
        and gate_count > 0
        and gate_failure is None
        and recovery_route_initial_q6 is not None
        and route_dispatch is not None
        and int(route_dispatch["angles"][5])
        == int(recovery_route_initial_q6)
        and last is not None
        and last["angle_targets"] == [-1] * 6
        and final_angles_ok
        and final_currents_ok
        and last["errors"] == [0] * 6
        and all(int(value) == 2 for value in last["statuses"])
        and all(int(value) < 50 for value in last["temperatures"])
        and final_hand is not None
        and tuple(final_hand.get("speeds", ())) == expected_speeds
        and tuple(final_hand.get("force_limits", ())) == expected_forces
    )
    return {
        "rh56_device": {
            "port": str(config["inspire"]["port"]),
            "resolved_port": str(Path(str(config["inspire"]["port"])).resolve()),
            "hand_id": int(config["inspire"]["hand_id"]),
            "baud": int(config["inspire"]["baud"]),
            "initial_snapshot": initial_hand,
        },
        "franka_read_only": {
            "connection": "read_once_only_no_controller_no_robot_write",
            "initial": franka_initial,
            "final": franka_final,
            "continuous_gate": {
                "check_count": gate_count,
                "last_success": gate_last,
                "failure": gate_failure,
            },
            "verified": franka_final is not None and gate_count > 0 and gate_failure is None,
        },
        "result": {
            "status": "pass" if passed else "fail",
            "adopted_disabled_verified": adopted,
            "recovered_open_verified": recovered,
            "disabled_verified": disabled,
            "recovery_route": recovery_route,
            "recovery_route_initial_q6": recovery_route_initial_q6,
            "evidence_bound_path_executed": (
                recovery_route is not None and not near_open_route
            ),
            "operation_error": operation_error,
            "stop_error": stop_error,
            "cleanup_error": (
                None if not cleanup_errors else "; ".join(cleanup_errors)
            ),
        },
        "route_dispatch": route_dispatch,
        "executed_q6_return_waypoints": list(executed_q6_return_waypoints),
        "executed_reset_q6_waypoints": list(executed_reset_q6_waypoints),
        "telemetry": telemetry,
        "final": {
            "angle_targets": None if last is None else last["angle_targets"],
            "angles": None if last is None else last["angles"],
            "currents": None if last is None else last["currents"],
            "errors": None if last is None else last["errors"],
            "statuses": None if last is None else last["statuses"],
            "temperatures": None if last is None else last["temperatures"],
            "snapshot_after_disable": final_hand,
            "original_settings_restored": (
                final_hand is not None
                and tuple(final_hand.get("speeds", ())) == expected_speeds
                and tuple(final_hand.get("force_limits", ())) == expected_forces
            ),
            "settings_restore_target": settings_restore_target,
        },
    }


def _run_command(args: argparse.Namespace) -> int:
    config, config_path = load_control_config(args.config)
    assets = verify_adapter_assets(config, config_path)
    _validate_run_request(args, config)
    plan = build_interrupted_recovery_plan(
        args.failed_evidence,
        expected_config=config,
        expected_config_path=config_path,
    )
    sources = _source_bindings(assets)
    config_sha_before = sha256_file(config_path)
    started = _utc_now()
    try:
        runtime = _run_hardware_session(args, config, plan)
    except BaseException as exc:
        runtime = {
            "rh56_device": None,
            "franka_read_only": None,
            "result": {
                "status": "fail",
                "adopted_disabled_verified": False,
                "recovered_open_verified": False,
                "disabled_verified": False,
                "recovery_route": None,
                "recovery_route_initial_q6": None,
                "evidence_bound_path_executed": False,
                "operation_error": f"{type(exc).__name__}: {exc}",
                "stop_error": "RH56 connection/stop state unavailable",
                "cleanup_error": None,
            },
            "route_dispatch": None,
            "executed_q6_return_waypoints": [],
            "executed_reset_q6_waypoints": [],
            "telemetry": [],
            "final": {
                "angle_targets": None,
                "angles": None,
                "currents": None,
                "errors": None,
                "statuses": None,
                "temperatures": None,
                "snapshot_after_disable": None,
                "original_settings_restored": False,
                "settings_restore_target": None,
            },
        }
    if sha256_file(config_path) != config_sha_before:
        runtime["result"]["status"] = "fail"
        runtime["result"]["operation_error"] = "control profile changed during recovery"

    payload = {
        "schema_version": 1,
        "kind": RECOVERY_EVIDENCE_KIND,
        "run_id": str(uuid.uuid4()),
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "motion_authorized": False,
        "commissioning_unlock_claimed": False,
        "control_profile": {
            "path": str(config_path),
            "file_sha256": config_sha_before,
            "parsed_sha256": json_sha256(config),
            "snapshot": copy.deepcopy(config),
        },
        "failed_commissioning_binding": plan.as_binding(),
        "source_bindings": sources,
        "operator_confirmations": {
            "installed_on_fr3": True,
            "24v_cutoff_ready": True,
            "franka_stop_ready": True,
            "workspace_clear": True,
            "no_contact_PLA_scope": True,
            "interrupted_open_recovery_confirmed": True,
        },
        "request": {
            "safety_scope": "installed_on_FR3_PLA_low_speed_unloaded_recovery_only",
            "interrupted_targets": list(plan.interrupted_targets),
            "bend_return_waypoints": [list(item) for item in plan.bend_return_waypoints],
            "q6_return_waypoints": list(
                runtime.get(
                    "executed_q6_return_waypoints", plan.q6_return_waypoints
                )
            ),
            "evidence_bound_q6_return_waypoints": list(plan.q6_return_waypoints),
            "recovery_mode": plan.recovery_mode,
            "allowed_recovery_routes": list(
                plan.as_binding()["allowed_recovery_routes"]
            ),
            "selected_recovery_route": runtime["result"].get(
                "recovery_route"
            ),
            "route_dispatch_initial_q6": runtime["result"].get(
                "recovery_route_initial_q6"
            ),
            "permitted_live_q6_range": list(plan.permitted_live_q6_range),
            "profile_near_open_q6_range": list(
                plan.profile_near_open_q6_range
            ),
            "q6_open_min_angle": int(plan.q6_open_min_angle),
            "step_units": plan.step_units,
            "speed": 40,
            "force_limit_g": 80,
            "max_axis_current_ma": int(args.max_axis_current_ma),
            "stop_max_axis_current_ma": STOP_MAX_AXIS_CURRENT_MA,
            "endpoint_stable_samples": int(args.endpoint_stable_samples),
            "max_inactive_drift_units": int(args.max_inactive_drift_units),
        },
        **runtime,
    }
    evidence = seal_evidence(payload)
    digest = atomic_write_evidence(args.output, evidence)
    print(f"[recovery evidence] {Path(args.output).expanduser().resolve()}")
    print(f"[recovery evidence] file_sha256={digest}")
    print(
        "[recovery result] status={} open={} disabled={}".format(
            evidence["result"]["status"],
            evidence["result"]["recovered_open_verified"],
            evidence["result"]["disabled_verified"],
        )
    )
    if evidence["result"]["status"] != "pass":
        if evidence["result"]["disabled_verified"] and evidence["result"]["stop_error"] is None:
            print("[safe stop verified] recovery failed but all-six output is disabled", file=sys.stderr)
        else:
            print("[STOP UNCONFIRMED] cut RH56 24 V immediately if the hand is moving", file=sys.stderr)
        return 1
    print("[recovery self-check] PASS; hand is open and all-six ANGLE_SET=-1")
    print("[commissioning] unchanged and still LOCKED; this recovery record cannot unlock q6/coupled control")
    return 0


def _inspect_command(args: argparse.Namespace) -> int:
    config, config_path = load_control_config(args.config)
    verify_adapter_assets(config, config_path)
    plan = build_interrupted_recovery_plan(
        args.failed_evidence,
        expected_config=config,
        expected_config_path=config_path,
    )
    print(json.dumps(plan.as_binding(), ensure_ascii=False, indent=2, sort_keys=True))
    print(
        f"[{args.command} only] no hardware module was imported and no register was written"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover an interrupted installed RH56 air-close path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "dry-run"):
        inspect = subparsers.add_parser(
            command, help="derive the evidence-bound recovery without hardware"
        )
        inspect.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        inspect.add_argument("--failed-evidence", type=Path, required=True)

    run = subparsers.add_parser("run", help="execute the exact evidence-bound recovery")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--failed-evidence", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-axis-current-ma", type=int, default=400)
    run.add_argument("--max-inactive-drift-units", type=int, default=8, choices=range(1, 21))
    run.add_argument("--endpoint-stable-samples", type=int, default=3, choices=range(2, 11))
    run.add_argument("--confirm-installed", metavar=INSTALLED_TOKEN)
    run.add_argument("--confirm-24v-cutoff", metavar=POWER_TOKEN)
    run.add_argument("--confirm-franka-stop", metavar=STOP_TOKEN)
    run.add_argument("--confirm-workspace-clear", metavar=CLEAR_TOKEN)
    run.add_argument("--confirm-no-contact", metavar=NO_CONTACT_TOKEN)
    run.add_argument("--confirm-recovery", metavar=RECOVERY_TOKEN)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in ("inspect", "dry-run"):
            return _inspect_command(args)
        if args.command == "run":
            return _run_command(args)
        raise RuntimeError(f"unsupported command {args.command}")
    except KeyboardInterrupt:
        print("[interrupted] cut RH56 24 V immediately if any motion remains", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[failed] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
