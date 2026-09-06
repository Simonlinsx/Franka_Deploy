#!/usr/bin/env python3
"""Reset an installed RH56 to its reusable all-open, all-disabled state."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydex_pipeline.control_config import load_control_config, verify_adapter_assets
from anydex_pipeline.rh56_reset_open import (
    DEFAULT_FORCES,
    DEFAULT_SPEEDS,
    RESET_BEND_OPEN_MIN_ANGLE,
    RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
    RESET_Q6_FEEDBACK_RECOVERY_MARGIN_UNITS,
    RH56ResetOpenDriver,
)
from anydex_pipeline.inspire_sequence_driver import RH56StopUnconfirmed


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
MAX_RESET_HOST_CURRENT_MA = 1000
INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
POWER_TOKEN = "RH56_24V_CUTOFF_READY"
STOP_TOKEN = "FR3_STOP_READY"
CLEAR_TOKEN = "INSTALLED_AIR_WORKSPACE_CLEAR"
NO_CONTACT_TOKEN = "PLA_LOW_SPEED_NO_CONTACT"
RESET_TOKEN = "RH56_RESET_OPEN"


@dataclass(frozen=True)
class RH56ResetOpenProof:
    """Successful installed-hand reset proof returned to other entry points."""

    q6_waypoints: tuple[int, ...]
    final_angles: tuple[int, ...]
    final_targets: tuple[int, ...]
    final_currents_ma: tuple[int, ...]
    final_statuses: tuple[int, ...]
    final_speeds: tuple[int, ...]
    final_forces_g: tuple[int, ...]
    franka_read_only_verified: bool = True
    rh56_disabled_verified: bool = True


def _profile_q6_open_min_angle(config: Mapping[str, Any]) -> int:
    """Return the installed q6 physical-feedback open threshold.

    The command/reset contract remains ANGLE_SET=1000.  The live feedback gate
    uses the profile's commissioned arrival tolerance so normal q6 endpoint
    offset is not mistaken for an incomplete reset.
    """

    inspire = config["inspire"]
    target_value = inspire["open_targets"][5]
    tolerance_value = inspire["arrival_tolerance_units"]
    for name, value in (
        ("inspire.open_targets[5]", target_value),
        ("inspire.arrival_tolerance_units", tolerance_value),
    ):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not np.isfinite(numeric) or numeric != round(numeric):
            raise ValueError(f"{name} must be an integer")
    target = int(target_value)
    tolerance = int(tolerance_value)
    q6_lower, q6_upper = (
        int(value)
        for value in inspire["thumb_rotate_validated_realtime_range"]
    )
    if target != q6_upper:
        raise ValueError(
            "inspire q6 open target must equal the commissioned range endpoint"
        )
    if not 0 <= tolerance <= RESET_Q6_ARRIVAL_TOLERANCE_UNITS:
        raise ValueError(
            "inspire.arrival_tolerance_units exceeds the reset q6 endpoint "
            f"limit {RESET_Q6_ARRIVAL_TOLERANCE_UNITS}"
        )
    minimum = target - tolerance
    if not q6_lower <= minimum <= q6_upper:
        raise ValueError(
            "derived q6 physical-open threshold is outside the commissioned range"
        )
    return minimum


def _validate_final_reset_open_snapshot(
    snapshot: Mapping[str, Any],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Validate the disabled cleanup tail without weakening active reset.

    ``reset_to_open`` has already proved q6 reached its active endpoint (or
    was already open) and bounded its post-disable movement.  This outer
    snapshot therefore verifies stop/default restoration and the five bend
    axes only.  Repeating an absolute q6 floor here would turn encoder release
    backlash into a false failure.
    """

    final_angles = tuple(int(value) for value in snapshot["angles"])
    final_targets = tuple(int(value) for value in snapshot["angle_targets"])
    final_currents = tuple(int(value) for value in snapshot["currents"])
    final_statuses = tuple(int(value) for value in snapshot["statuses"])
    final_speeds = tuple(int(value) for value in snapshot["speeds"])
    final_forces = tuple(int(value) for value in snapshot["force_limits"])
    if (
        final_targets != (-1,) * 6
        or any(value < RESET_BEND_OPEN_MIN_ANGLE for value in final_angles[:5])
        or any(abs(value) > 100 for value in final_currents)
        or not all(value in (2, 0xFF) for value in final_statuses)
        or final_speeds != DEFAULT_SPEEDS
        or final_forces != DEFAULT_FORCES
    ):
        raise RuntimeError(
            "final reset-open verification failed: angles={} targets={} "
            "currents={} statuses={} speeds={} forces={}".format(
                final_angles,
                final_targets,
                final_currents,
                final_statuses,
                final_speeds,
                final_forces,
            )
        )
    return (
        final_angles,
        final_targets,
        final_currents,
        final_statuses,
        final_speeds,
        final_forces,
    )


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


def _validate_reset_parameters(
    config: Mapping[str, Any],
    *,
    max_axis_current_ma: int,
    max_inactive_drift_units: int,
    endpoint_stable_samples: int,
) -> None:
    for name, value, lower, upper in (
        (
            "--max-axis-current-ma",
            max_axis_current_ma,
            50,
            MAX_RESET_HOST_CURRENT_MA,
        ),
        ("--max-inactive-drift-units", max_inactive_drift_units, 1, 20),
        ("--endpoint-stable-samples", endpoint_stable_samples, 2, 10),
    ):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be in {lower}..{upper}") from exc
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(numeric)
            or numeric != round(numeric)
            or not lower <= int(numeric) <= upper
        ):
            raise ValueError(f"{name} must be in {lower}..{upper}")
    inspire = config["inspire"]
    if int(inspire["open_speed"]) > 40 or int(inspire["force_limit_g"]) > 80:
        raise ValueError("profile exceeds fixed reset motion limits 40/80g")
    _profile_q6_open_min_angle(config)


def _validate_run_request(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    required = (
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-24v-cutoff", args.confirm_24v_cutoff, POWER_TOKEN),
        ("--confirm-franka-stop", args.confirm_franka_stop, STOP_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, CLEAR_TOKEN),
        ("--confirm-no-contact", args.confirm_no_contact, NO_CONTACT_TOKEN),
        ("--confirm-reset-open", args.confirm_reset_open, RESET_TOKEN),
    )
    missing = [
        f"{flag} {token}"
        for flag, actual, token in required
        if actual != token
    ]
    if missing:
        raise ValueError(
            "exact confirmations required before hardware import: "
            + "; ".join(missing)
        )
    _validate_reset_parameters(
        config,
        max_axis_current_ma=args.max_axis_current_ma,
        max_inactive_drift_units=args.max_inactive_drift_units,
        endpoint_stable_samples=args.endpoint_stable_samples,
    )


def run_installed_rh56_reset_open(
    config: Mapping[str, Any],
    *,
    port_override: Optional[str] = None,
    max_axis_current_ma: int = 400,
    max_inactive_drift_units: int = 8,
    endpoint_stable_samples: int = 3,
    hardware_types: Optional[Mapping[str, Any]] = None,
) -> RH56ResetOpenProof:
    """Reset the installed RH56 and return proof, without CLI confirmations.

    Callers must establish their own execution-admission boundary.  The same
    profile, motion, current, external Franka-idle gate, and verified-stop
    contract used by the standalone CLI remains in force.
    """

    _validate_reset_parameters(
        config,
        max_axis_current_ma=max_axis_current_ma,
        max_inactive_drift_units=max_inactive_drift_units,
        endpoint_stable_samples=endpoint_stable_samples,
    )
    if hardware_types is None:
        # Hardware imports remain lazy: importing this module or validating a
        # dry-run cannot enumerate or connect either device.
        import pylibfranka
        from anydex_pipeline.franka_sequence_driver import (
            FrankaMotionLimits,
            FrankaSequenceDriver,
        )

        reset_driver_type = RH56ResetOpenDriver
    else:
        pylibfranka = hardware_types["pylibfranka"]
        FrankaMotionLimits = hardware_types["FrankaMotionLimits"]
        FrankaSequenceDriver = hardware_types["FrankaSequenceDriver"]
        reset_driver_type = hardware_types.get(
            "RH56ResetOpenDriver", RH56ResetOpenDriver
        )

    driver = None
    robot = None
    arm = None
    operation_error: Optional[BaseException] = None
    stop_error: Optional[BaseException] = None
    cleanup_errors: list[BaseException] = []
    q6_waypoints: tuple[int, ...] = ()
    final_snapshot = None
    q6_open_min_angle = _profile_q6_open_min_angle(config)

    def franka_read_only_gate() -> None:
        state = robot.read_once()
        arm._validate_state(
            state,
            require_idle=True,
            enforce_success=False,
            enforce_joint_limit_margin=False,
        )

    try:
        driver = reset_driver_type.connect(
            port=str(port_override or config["inspire"]["port"]),
            baud=int(config["inspire"]["baud"]),
            hand_id=int(config["inspire"]["hand_id"]),
            thumb_rotate_range=tuple(
                int(value)
                for value in config["inspire"][
                    "thumb_rotate_validated_realtime_range"
                ]
            ),
            motion_timeout_s=float(config["inspire"]["motion_timeout_s"]),
            angle_tolerance=int(config["inspire"]["arrival_tolerance_units"]),
            open_min_angle=RESET_BEND_OPEN_MIN_ANGLE,
            q6_open_min_angle=q6_open_min_angle,
            q6_feedback_recovery_min_angle=max(
                0,
                int(
                    config["inspire"][
                        "thumb_rotate_validated_realtime_range"
                    ][0]
                )
                - RESET_Q6_FEEDBACK_RECOVERY_MARGIN_UNITS,
            ),
            stop_max_axis_current_ma=min(int(max_axis_current_ma), 400),
            # A q6 motion released at its physical endpoint can keep reporting
            # STATUS=0 briefly after ANGLE_SET=-1.  Wait through that bounded
            # low-speed coast instead of declaring stop unconfirmed at 1.5 s.
            stop_verify_timeout_s=5.0,
        )
        # This is deliberately the first RH56 action: stale numeric targets are
        # disabled and physical idle is proven before settings or state are used.
        driver.adopt_disabled_state_and_verify()
        initial = driver.hand.snapshot()
        print(
            "[RH56 initial] angles={} targets={} currents={} speeds={} forces={}".format(
                list(initial["angles"]),
                list(initial["angle_targets"]),
                list(initial["currents"]),
                list(initial["speeds"]),
                list(initial["force_limits"]),
            ),
            flush=True,
        )

        # Franka remains read-only and Idle for the whole installed-hand reset.
        robot = pylibfranka.Robot(
            str(config["franka"]["ip"]), pylibfranka.RealtimeConfig.kIgnore
        )
        arm = FrankaSequenceDriver(
            robot,
            pylibfranka,
            _build_limits(config, FrankaMotionLimits),
        )
        arm._validate_state(
            robot.read_once(),
            require_idle=True,
            enforce_success=False,
            enforce_joint_limit_margin=False,
        )
        driver.install_external_safety_check(franka_read_only_gate)
        print("[Franka] read-only Idle/load gate active; no controller is created")

        q6_waypoints = driver.reset_to_open(
            max_axis_current_ma=int(max_axis_current_ma),
            endpoint_stable_samples=int(endpoint_stable_samples),
            max_inactive_drift_units=int(max_inactive_drift_units),
            # Deployment reset needs the endpoint, not five independent axis
            # commissioning proofs.  Start all bend axes in one batch so the
            # reset duration is one hand-opening motion instead of five.
            simultaneous_bend_open=True,
            # After the bends are clear, issue the q6 open endpoint once.  The
            # segmented q6 sweep remains available only to evidence-oriented
            # commissioning/recovery callers.
            direct_q6_endpoint_open=True,
        )
    except BaseException as exc:
        operation_error = exc
    finally:
        if driver is not None:
            try:
                driver.disable_and_verify()
                final_snapshot = driver.hand.snapshot()
            except RH56StopUnconfirmed as exc:
                stop_error = exc
            except BaseException as exc:
                cleanup_errors.append(exc)
                try:
                    final_snapshot = driver.hand.snapshot()
                except BaseException as snapshot_exc:
                    cleanup_errors.append(snapshot_exc)
            try:
                driver.close()
            except RH56StopUnconfirmed as exc:
                stop_error = exc
            except BaseException as exc:
                cleanup_errors.append(exc)
        # pylibfranka releases Robot ownership in its destructor.  Make that
        # release explicit before a following reset phase or native child is
        # allowed to construct the next sole owner.
        arm = None
        robot = None
        gc.collect()

    if stop_error is not None:
        raise RuntimeError(
            "STOP UNCONFIRMED; cut RH56 24 V immediately if the hand is moving: "
            f"{type(stop_error).__name__}: {stop_error}"
        ) from operation_error
    if isinstance(operation_error, KeyboardInterrupt):
        if cleanup_errors:
            details = "; ".join(
                f"{type(exc).__name__}: {exc}" for exc in cleanup_errors
            )
            raise RuntimeError(
                "reset-open was interrupted and motion is stopped, but "
                "cleanup/default-setting restore failed: "
                f"{details}"
            ) from cleanup_errors[0]
        # The finally block above has already disabled all six targets,
        # verified physical idle, closed the hand transport, and released the
        # read-only Franka owner.  Preserve Ctrl+C for the formal caller so it
        # can report the interrupted (130) outcome instead of a generic fault.
        raise operation_error
    if operation_error is not None:
        raise RuntimeError(
            "reset-open failed, but all-six disable/idle was verified: "
            f"{type(operation_error).__name__}: {operation_error}"
        ) from operation_error
    if cleanup_errors:
        details = "; ".join(
            f"{type(exc).__name__}: {exc}" for exc in cleanup_errors
        )
        raise RuntimeError(
            "reset-open motion is stopped, but cleanup/default-setting restore "
            f"failed: {details}"
        ) from cleanup_errors[0]
    if final_snapshot is None:
        raise RuntimeError("reset-open ended without a final RH56 snapshot")

    (
        final_angles,
        final_targets,
        final_currents,
        final_statuses,
        final_speeds,
        final_forces,
    ) = _validate_final_reset_open_snapshot(
        final_snapshot,
    )
    print(f"[q6 actual-anchored targets] {list(q6_waypoints)}")
    print(
        "[PASS] RH56 angles={} targets={} speeds={} forces={}".format(
            list(final_angles),
            list(final_targets),
            list(final_speeds),
            list(final_forces),
        )
    )
    print("[PASS] Franka was read-only; this command created no arm controller")
    return RH56ResetOpenProof(
        q6_waypoints=q6_waypoints,
        final_angles=final_angles,
        final_targets=final_targets,
        final_currents_ma=final_currents,
        final_statuses=final_statuses,
        final_speeds=final_speeds,
        final_forces_g=final_forces,
    )


def _run_hardware_session(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[int, ...]:
    """Compatibility wrapper retained for the standalone CLI tests."""

    proof = run_installed_rh56_reset_open(
        config,
        port_override=args.port,
        max_axis_current_ma=args.max_axis_current_ma,
        max_inactive_drift_units=args.max_inactive_drift_units,
        endpoint_stable_samples=args.endpoint_stable_samples,
    )
    return proof.q6_waypoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reset an installed RH56 to all-open/all-disabled"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute the reset-open sequence")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--port")
    run.add_argument("--max-axis-current-ma", type=int, default=400)
    run.add_argument(
        "--max-inactive-drift-units", type=int, default=8, choices=range(1, 21)
    )
    run.add_argument(
        "--endpoint-stable-samples", type=int, default=3, choices=range(2, 11)
    )
    run.add_argument("--confirm-installed", metavar=INSTALLED_TOKEN)
    run.add_argument("--confirm-24v-cutoff", metavar=POWER_TOKEN)
    run.add_argument("--confirm-franka-stop", metavar=STOP_TOKEN)
    run.add_argument("--confirm-workspace-clear", metavar=CLEAR_TOKEN)
    run.add_argument("--confirm-no-contact", metavar=NO_CONTACT_TOKEN)
    run.add_argument("--confirm-reset-open", metavar=RESET_TOKEN)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, config_path = load_control_config(args.config)
        verify_adapter_assets(config, config_path)
        _validate_run_request(args, config)
        _run_hardware_session(args, config)
        return 0
    except KeyboardInterrupt:
        print(
            "[interrupted] shutdown verification is running; cut 24 V only if "
            "motion remains or STOP UNCONFIRMED is printed",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[failed] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
