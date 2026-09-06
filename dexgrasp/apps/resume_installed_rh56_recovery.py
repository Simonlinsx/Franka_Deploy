#!/usr/bin/env python3
"""Resume the sealed bend-envelope failure of installed RH56 open recovery."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", ROOT / "apps", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recover_installed_rh56 as recovery_app
from anydex_pipeline.control_config import load_control_config, verify_adapter_assets
from anydex_pipeline.rh56_commissioning import (
    atomic_write_evidence,
    json_sha256,
    seal_evidence,
    sha256_file,
)
from anydex_pipeline.rh56_interrupted_resume import (
    BEND_REVERSE_BOOTSTRAP_UNITS,
    BEND_REVERSE_OVERSHOOT_TOLERANCE_UNITS,
    InterruptedResumeDriver,
    InterruptedResumePlan,
    RESUME_EVIDENCE_KIND,
    build_interrupted_resume_plan,
)


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
POWER_TOKEN = "RH56_24V_CUTOFF_READY"
STOP_TOKEN = "FR3_STOP_READY"
CLEAR_TOKEN = "INSTALLED_AIR_WORKSPACE_CLEAR"
NO_CONTACT_TOKEN = "PLA_LOW_SPEED_NO_CONTACT"
RESUME_TOKEN = "RH56_INTERRUPTED_RESUME_OPEN"
STOP_MAX_AXIS_CURRENT_MA = 100


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_bindings(assets: Any) -> list[dict[str, str]]:
    paths = {
        "resume_cli": Path(__file__).resolve(),
        "resume_module": (
            ROOT / "src/anydex_pipeline/rh56_interrupted_resume.py"
        ).resolve(),
        "recovery_cli": (ROOT / "apps/recover_installed_rh56.py").resolve(),
        "recovery_module": (
            ROOT / "src/anydex_pipeline/rh56_interrupted_recovery.py"
        ).resolve(),
        "rh56_sequence_driver": (
            ROOT / "src/anydex_pipeline/inspire_sequence_driver.py"
        ).resolve(),
        "rh56_hand_path": (ROOT / "src/anydex_pipeline/rh56_hand_path.py").resolve(),
        "rh56_register_api": (WORKSPACE / "examples/inspire_rh56_test.py").resolve(),
        "franka_sequence_driver": (
            ROOT / "src/anydex_pipeline/franka_sequence_driver.py"
        ).resolve(),
        "adapter_mesh": Path(assets.mesh_path).resolve(),
        "adapter_provenance": Path(assets.provenance_path).resolve(),
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("required resume provenance is missing: " + "; ".join(missing))
    return [
        {"name": name, "path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    ]


def _validate_request(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    required = (
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-24v-cutoff", args.confirm_24v_cutoff, POWER_TOKEN),
        ("--confirm-franka-stop", args.confirm_franka_stop, STOP_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, CLEAR_TOKEN),
        ("--confirm-no-contact", args.confirm_no_contact, NO_CONTACT_TOKEN),
        ("--confirm-resume", args.confirm_resume, RESUME_TOKEN),
    )
    missing = [f"{flag} {token}" for flag, actual, token in required if actual != token]
    if missing:
        raise ValueError("exact confirmations required before hardware import: " + "; ".join(missing))
    if not 50 <= int(args.max_axis_current_ma) <= 400:
        raise ValueError("--max-axis-current-ma must be in 50..400")
    if not 1 <= int(args.max_inactive_drift_units) <= 20:
        raise ValueError("--max-inactive-drift-units must be in 1..20")
    if Path(args.output).expanduser().resolve().exists():
        raise FileExistsError(f"resume evidence output already exists: {Path(args.output).resolve()}")
    inspire = config["inspire"]
    if int(inspire["open_speed"]) > 40 or int(inspire["close_speed"]) > 40:
        raise ValueError("profile hand speeds exceed fixed resume speed 40")
    if int(inspire["force_limit_g"]) > 80:
        raise ValueError("profile force limit exceeds fixed resume force 80g")


def _run_hardware_session(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    plan: InterruptedResumePlan,
) -> dict[str, Any]:
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
    operation_error = None
    stop_error = None

    def franka_gate() -> None:
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
            gate_last = recovery_app._franka_state_json(state)
        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            if gate_failure is None:
                gate_failure = message
            raise RuntimeError("continuous Franka read-only gate failed: " + message) from exc

    try:
        driver = InterruptedResumeDriver.connect(
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
        initial_hand = recovery_app._rh56_snapshot_json(driver.hand.snapshot())
        if int(initial_hand.get("hand_id", -1)) != int(config["inspire"]["hand_id"]):
            raise RuntimeError("RH56 HAND_ID differs from the control profile")
        robot = pylibfranka.Robot(
            str(config["franka"]["ip"]), pylibfranka.RealtimeConfig.kIgnore
        )
        arm = FrankaSequenceDriver(
            robot,
            pylibfranka,
            recovery_app._build_limits(config, FrankaMotionLimits),
        )
        state = robot.read_once()
        arm._validate_state(
            state,
            require_idle=True,
            enforce_success=False,
            enforce_joint_limit_margin=False,
        )
        franka_initial = recovery_app._franka_state_json(state)
        driver.install_external_safety_check(franka_gate)
        returned = driver.resume_interrupted_open_to_completion(
            plan,
            max_axis_current_ma=int(args.max_axis_current_ma),
            endpoint_stable_samples=int(args.endpoint_stable_samples),
            max_inactive_drift_units=int(args.max_inactive_drift_units),
        )
        if tuple(returned) != plan.q6_return_waypoints:
            raise RuntimeError("driver returned an unexpected q6 resume path")
        recovered = True
    except BaseException as exc:
        operation_error = f"{type(exc).__name__}: {exc}"
    finally:
        if driver is not None:
            try:
                driver.disable_and_verify()
                disabled = True
                final_hand = recovery_app._rh56_snapshot_json(driver.hand.snapshot())
            except BaseException as exc:
                stop_error = f"{type(exc).__name__}: {exc}"
                print(
                    "[hardware][STOP UNCONFIRMED] cut RH56 24 V immediately if the hand is moving: "
                    + stop_error,
                    file=sys.stderr,
                    flush=True,
                )
            try:
                driver.close()
            except BaseException as exc:
                message = f"{type(exc).__name__}: {exc}"
                stop_error = message if stop_error is None else f"{stop_error}; {message}"
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
                franka_final = recovery_app._franka_state_json(state)
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
    settings_restored = (
        final_hand is not None
        and tuple(final_hand.get("speeds", ())) == plan.original_speeds
        and tuple(final_hand.get("force_limits", ())) == plan.original_forces
    )
    passed = (
        operation_error is None
        and stop_error is None
        and adopted
        and recovered
        and disabled
        and settings_restored
        and franka_final is not None
        and gate_count > 0
        and gate_failure is None
        and last is not None
        and last["angle_targets"] == [-1] * 6
        and all(int(value) >= 980 for value in last["angles"])
        and all(abs(int(value)) <= STOP_MAX_AXIS_CURRENT_MA for value in last["currents"])
        and last["errors"] == [0] * 6
        and all(int(value) in (2, 0xFF) for value in last["statuses"])
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
            "resumed_open_verified": recovered,
            "disabled_verified": disabled,
            "original_settings_restored": settings_restored,
            "operation_error": operation_error,
            "stop_error": stop_error,
        },
        "runtime_bend_waypoints": (
            [] if driver is None else [list(item) for item in driver.last_resume_bend_waypoints]
        ),
        "runtime_bend_actual_endpoints": (
            [] if driver is None else list(driver.last_resume_bend_actual_endpoints)
        ),
        "telemetry": telemetry,
        "final": {
            "angle_targets": None if last is None else last["angle_targets"],
            "angles": None if last is None else last["angles"],
            "currents": None if last is None else last["currents"],
            "errors": None if last is None else last["errors"],
            "statuses": None if last is None else last["statuses"],
            "temperatures": None if last is None else last["temperatures"],
            "snapshot_after_disable": final_hand,
        },
    }


def _run_command(args: argparse.Namespace) -> int:
    config, config_path = load_control_config(args.config)
    assets = verify_adapter_assets(config, config_path)
    _validate_request(args, config)
    plan = build_interrupted_resume_plan(
        args.failed_recovery,
        expected_config=config,
        expected_config_path=config_path,
    )
    config_sha = sha256_file(config_path)
    sources = _source_bindings(assets)
    started = _utc_now()
    print(
        "[resume] evidence-bound live pinky range={}..{}, q6={}; "
        "next bend target is derived from fresh stable ANGLE_ACT".format(
            plan.permitted_live_pinky_range[0],
            plan.permitted_live_pinky_range[1],
            plan.target_q6,
        ),
        flush=True,
    )
    try:
        runtime = _run_hardware_session(args, config, plan)
    except BaseException as exc:
        runtime = {
            "rh56_device": None,
            "franka_read_only": None,
            "result": {
                "status": "fail",
                "adopted_disabled_verified": False,
                "resumed_open_verified": False,
                "disabled_verified": False,
                "original_settings_restored": False,
                "operation_error": f"{type(exc).__name__}: {exc}",
                "stop_error": "RH56 connection/stop state unavailable",
            },
            "runtime_bend_waypoints": [],
            "runtime_bend_actual_endpoints": [],
            "telemetry": [],
            "final": {
                "angle_targets": None, "angles": None, "currents": None,
                "errors": None, "statuses": None, "temperatures": None,
                "snapshot_after_disable": None,
            },
        }
    if sha256_file(config_path) != config_sha:
        runtime["result"]["status"] = "fail"
        runtime["result"]["operation_error"] = "control profile changed during resume"
    payload = {
        "schema_version": 1,
        "kind": RESUME_EVIDENCE_KIND,
        "run_id": str(uuid.uuid4()),
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "motion_authorized": False,
        "commissioning_unlock_claimed": False,
        "control_profile": {
            "path": str(config_path),
            "file_sha256": config_sha,
            "parsed_sha256": json_sha256(config),
            "snapshot": copy.deepcopy(config),
        },
        "failed_recovery_binding": plan.as_binding(),
        "source_bindings": sources,
        "operator_confirmations": {
            "installed_on_fr3": True,
            "24v_cutoff_ready": True,
            "franka_stop_ready": True,
            "workspace_clear": True,
            "no_contact_PLA_scope": True,
            "interrupted_resume_confirmed": True,
        },
        "request": {
            "safety_scope": "installed_on_FR3_PLA_low_speed_unloaded_resume_only",
            "live_pinky_range": list(plan.permitted_live_pinky_range),
            "target_generation": "fresh_actual_plus_50_then_fresh_actual_plus_25_v1",
            "bend_reverse_bootstrap_units": BEND_REVERSE_BOOTSTRAP_UNITS,
            "bend_reverse_overshoot_tolerance_units": BEND_REVERSE_OVERSHOOT_TOLERANCE_UNITS,
            "q6_return_waypoints": list(plan.q6_return_waypoints),
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
    print(f"[resume evidence] {Path(args.output).expanduser().resolve()}")
    print(f"[resume evidence] file_sha256={digest}")
    print(
        "[resume result] status={} open={} disabled={}".format(
            evidence["result"]["status"],
            evidence["result"]["resumed_open_verified"],
            evidence["result"]["disabled_verified"],
        )
    )
    if evidence["result"]["status"] != "pass":
        if evidence["result"]["disabled_verified"] and evidence["result"]["stop_error"] is None:
            print("[safe stop verified] resume failed but all-six output is disabled", file=sys.stderr)
        else:
            print("[STOP UNCONFIRMED] cut RH56 24 V immediately if the hand is moving", file=sys.stderr)
        return 1
    print("[resume self-check] PASS; hand is open and all-six ANGLE_SET=-1")
    print("[commissioning] unchanged and still LOCKED")
    return 0


def _inspect_command(args: argparse.Namespace) -> int:
    config, config_path = load_control_config(args.config)
    verify_adapter_assets(config, config_path)
    plan = build_interrupted_resume_plan(
        args.failed_recovery,
        expected_config=config,
        expected_config_path=config_path,
    )
    print(json.dumps(plan.as_binding(), ensure_ascii=False, indent=2, sort_keys=True))
    print("[inspect only] no hardware module was imported and no register was written")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume interrupted installed RH56 open recovery")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="derive resume bounds without hardware")
    inspect.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    inspect.add_argument("--failed-recovery", type=Path, required=True)
    run = subparsers.add_parser("run", help="execute the evidence-bound resume")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--failed-recovery", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-axis-current-ma", type=int, default=400)
    run.add_argument("--max-inactive-drift-units", type=int, default=8, choices=range(1, 21))
    run.add_argument("--endpoint-stable-samples", type=int, default=3, choices=range(2, 11))
    run.add_argument("--confirm-installed", metavar=INSTALLED_TOKEN)
    run.add_argument("--confirm-24v-cutoff", metavar=POWER_TOKEN)
    run.add_argument("--confirm-franka-stop", metavar=STOP_TOKEN)
    run.add_argument("--confirm-workspace-clear", metavar=CLEAR_TOKEN)
    run.add_argument("--confirm-no-contact", metavar=NO_CONTACT_TOKEN)
    run.add_argument("--confirm-resume", metavar=RESUME_TOKEN)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
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
