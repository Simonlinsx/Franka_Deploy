#!/usr/bin/env python3
"""Return the installed FR3 + RH56 assembly to the configured default q.

This command never moves the RH56.  Before constructing a Franka control
handle it opens the RH56 serial port exclusively and proves, twice, that all
six axes are open, idle, and disabled.  The serial descriptor remains open
throughout the Franka motion so another process cannot acquire the hand and
change its pose while the arm is moving.

The arm motion is delegated unchanged to :class:`FrankaSequenceDriver`; this
keeps the reviewed real-time state gates, command-success watchdog, deadline
checks, bounded joint segmentation, Ctrl+C stop path, and final physical-stop
verification in one implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Hardware-free imports only.  pylibfranka and the serial implementation are
# imported by _load_hardware_types(), after every exact confirmation.
from anydex_pipeline.control_config import (  # noqa: E402
    load_control_config,
    verify_adapter_assets,
)
from anydex_pipeline.host_network_preflight import (  # noqa: E402
    require_uncontended_franka_https_link,
)


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"

INSTALLED_TOKEN = "RH56_INSTALLED_ON_FR3"
HAND_OPEN_TOKEN = "RH56_OPEN_DISABLED"
SWEEP_CLEAR_TOKEN = "FR3_RH56_CURRENT_TO_DEFAULT_SWEEP_CLEAR"
STOP_TOKEN = "FR3_STOP_READY"
PLA_TOKEN = "PLA_LOW_SPEED_UNLOADED_ONLY"

OPEN_MIN_ANGLE = 980
MAX_IDLE_CURRENT_MA = 100
MAX_STABLE_DRIFT_UNITS = 8
EXPECTED_DEFAULT_SPEEDS = (1000,) * 6
EXPECTED_DEFAULT_FORCES = (500,) * 6
IDLE_STATUSES = (2, 0xFF)


def _profile_open_min_angles(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Return bend-open minima plus the commissioned q6 lower endpoint.

    This is a read-only Franka reset gate, not another RH56 reset proof.  The
    five bend axes must be open; disabled q6 only has to remain inside its
    commissioned range.
    """

    inspire = config["inspire"]
    try:
        targets = tuple(int(value) for value in inspire["open_targets"])
        tolerance = int(inspire["arrival_tolerance_units"])
        q6_range = tuple(
            int(value)
            for value in inspire["thumb_rotate_validated_realtime_range"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("RH56 open feedback contract is malformed") from exc
    if len(targets) != 6 or targets != (1000,) * 6:
        raise ValueError("profile does not define the canonical all-six RH56 open state")
    if len(q6_range) != 2 or not 0 <= tolerance <= 30:
        raise ValueError("RH56 q6 open feedback contract is outside reviewed bounds")
    # Franka reset never writes the hand.  It requires the five bend axes open
    # and every RH56 axis disabled/idle, but q6's absolute encoder value is not
    # a second reset-to-open proof.  Accept the profile's commissioned q6
    # range instead of inventing another endpoint threshold.
    return (OPEN_MIN_ANGLE,) * 5 + (q6_range[0],)


class DefaultStopUnconfirmed(RuntimeError):
    """The command could not prove Franka stopped after attempting stop."""


@dataclass(frozen=True)
class FrankaDefaultResetProof:
    initial_q_rad: tuple[float, ...]
    target_q_rad: tuple[float, ...]
    final_q_rad: tuple[float, ...]
    initial_linf_delta_rad: float
    final_linf_error_rad: float
    maximum_start_delta_rad: Optional[float]
    franka_stop_verified: bool = True
    rh56_open_disabled_verified: bool = True


@dataclass(frozen=True)
class FrankaOnlyResetProof:
    """Numeric proof for an arm-only reset that never accesses RH56."""

    initial_q_rad: tuple[float, ...]
    target_q_rad: tuple[float, ...]
    final_q_rad: tuple[float, ...]
    initial_linf_delta_rad: float
    final_linf_error_rad: float
    maximum_start_delta_rad: Optional[float]
    franka_stop_verified: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the installed RH56 is open/disabled, then return only "
            "Franka to the configured low-speed default_q"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port", help="override the RH56 serial port from config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-installed", metavar=INSTALLED_TOKEN)
    parser.add_argument("--confirm-hand-open", metavar=HAND_OPEN_TOKEN)
    parser.add_argument("--confirm-workspace-clear", metavar=SWEEP_CLEAR_TOKEN)
    parser.add_argument("--confirm-stop-ready", metavar=STOP_TOKEN)
    parser.add_argument("--confirm-pla-low-speed", metavar=PLA_TOKEN)
    return parser


def _require_confirmations(args: argparse.Namespace) -> None:
    required = (
        ("--confirm-installed", args.confirm_installed, INSTALLED_TOKEN),
        ("--confirm-hand-open", args.confirm_hand_open, HAND_OPEN_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, SWEEP_CLEAR_TOKEN),
        ("--confirm-stop-ready", args.confirm_stop_ready, STOP_TOKEN),
        ("--confirm-pla-low-speed", args.confirm_pla_low_speed, PLA_TOKEN),
    )
    missing = [
        "{} {}".format(flag, token)
        for flag, actual, token in required
        if actual != token
    ]
    if missing:
        raise ValueError(
            "exact confirmations required before hardware import: "
            + "; ".join(missing)
        )


def _validate_static_request(config: Mapping[str, Any]) -> None:
    franka = config["franka"]
    tool = config["tool"]
    if tool.get("installed_on_franka_verified") is not True:
        raise ValueError("profile does not verify the RH56 is installed on Franka")
    if tool.get("low_speed_unloaded_commissioning_only") is not True:
        raise ValueError(
            "this reset entry is restricted to the low-speed unloaded profile"
        )
    velocity = float(franka["default_max_joint_velocity_rad_s"])
    if not 0.0 < velocity <= 0.05:
        raise ValueError(
            "Franka default speed must remain in (0, 0.05] rad/s; got {}".format(
                velocity
            )
        )
    segment = float(franka["default_max_joint_segment_rad"])
    if not 0.0 < segment <= 0.20:
        raise ValueError(
            "Franka default segment must remain in (0, 0.20] rad; got {}".format(
                segment
            )
        )
    default_q = np.asarray(franka["default_q_rad"], dtype=np.float64)
    if default_q.shape != (7,) or not np.all(np.isfinite(default_q)):
        raise ValueError("franka.default_q_rad is malformed")
    if tuple(int(value) for value in config["inspire"]["open_targets"]) != (1000,) * 6:
        raise ValueError("profile does not define the canonical all-six RH56 open state")


def _load_hardware_types() -> Mapping[str, Any]:
    """Lazy hardware imports reached only after the operator gate."""

    from anydex_pipeline.franka_sequence_driver import (
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )

    return {
        "rh56_api": importlib.import_module("examples.inspire_rh56_test"),
        "FrankaMotionLimits": FrankaMotionLimits,
        "FrankaSequenceDriver": FrankaSequenceDriver,
    }


def _load_franka_hardware_types() -> Mapping[str, Any]:
    """Load only Franka types; importing this path never imports RH56."""

    from anydex_pipeline.franka_sequence_driver import (
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )

    return {
        "FrankaMotionLimits": FrankaMotionLimits,
        "FrankaSequenceDriver": FrankaSequenceDriver,
    }


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
        settle_time_s=float(franka["settle_time_s"]),
        settle_timeout_s=float(franka["settle_timeout_s"]),
        settle_poll_s=float(franka["settle_poll_s"]),
    )


def _six(snapshot: Mapping[str, Any], name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in snapshot[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("RH56 snapshot field {} is malformed".format(name)) from exc
    if len(values) != 6:
        raise RuntimeError("RH56 snapshot field {} must contain six values".format(name))
    return values


def _verify_open_disabled_snapshot(
    snapshot: Mapping[str, Any],
    *,
    expected_hand_id: int,
    label: str,
    open_min_angles: Sequence[int] = (OPEN_MIN_ANGLE,) * 6,
) -> tuple[int, ...]:
    try:
        actual_hand_id = int(snapshot["hand_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("{} RH56 hand_id is malformed".format(label)) from exc
    if actual_hand_id != int(expected_hand_id):
        raise RuntimeError(
            "{} RH56 hand_id={} differs from profile {}".format(
                label, actual_hand_id, int(expected_hand_id)
            )
        )

    angles = _six(snapshot, "angles")
    targets = _six(snapshot, "angle_targets")
    currents = _six(snapshot, "currents")
    errors = _six(snapshot, "errors")
    statuses = _six(snapshot, "statuses")
    speeds = _six(snapshot, "speeds")
    forces = _six(snapshot, "force_limits")
    problems = []
    if targets != (-1,) * 6:
        problems.append("ANGLE_SET={} (required [-1]*6)".format(list(targets)))
    minimums = tuple(int(value) for value in open_min_angles)
    if len(minimums) != 6:
        raise ValueError("open_min_angles must contain six values")
    if any(value < minimum for value, minimum in zip(angles, minimums)):
        problems.append(
            "angles={} (required minimums {})".format(
                list(angles), list(minimums)
            )
        )
    if any(abs(value) > MAX_IDLE_CURRENT_MA for value in currents):
        problems.append(
            "currents={} (idle bound {}mA)".format(
                list(currents), MAX_IDLE_CURRENT_MA
            )
        )
    if errors != (0,) * 6:
        problems.append("errors={}".format(list(errors)))
    if not all(value in IDLE_STATUSES for value in statuses):
        problems.append("statuses={} (not all idle)".format(list(statuses)))
    if speeds != EXPECTED_DEFAULT_SPEEDS:
        problems.append(
            "speeds={} (run RH56 default to restore [1000]*6)".format(list(speeds))
        )
    if forces != EXPECTED_DEFAULT_FORCES:
        problems.append(
            "forces={} (run RH56 default to restore [500]*6)".format(list(forces))
        )
    if problems:
        raise RuntimeError(
            "{} RH56 is not in canonical open/disabled default: {}. Run "
            "./scripts/reset_installed_rh56_open.sh first.".format(
                label, "; ".join(problems)
            )
        )
    return angles


def _verify_stable_hand_pair(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    expected_hand_id: int,
    label: str,
    open_min_angles: Sequence[int] = (OPEN_MIN_ANGLE,) * 6,
) -> None:
    first_angles = _verify_open_disabled_snapshot(
        first,
        expected_hand_id=expected_hand_id,
        label=label + " sample 1",
        open_min_angles=open_min_angles,
    )
    second_angles = _verify_open_disabled_snapshot(
        second,
        expected_hand_id=expected_hand_id,
        label=label + " sample 2",
        open_min_angles=open_min_angles,
    )
    drift = max(abs(right - left) for left, right in zip(first_angles, second_angles))
    if drift > MAX_STABLE_DRIFT_UNITS:
        raise RuntimeError(
            "{} RH56 angle feedback drifted {} units while disabled; refusing "
            "Franka motion".format(label, drift)
        )


def _read_stable_hand_pair(
    hand: Any,
    *,
    expected_hand_id: int,
    label: str,
    open_min_angles: Sequence[int] = (OPEN_MIN_ANGLE,) * 6,
    sleep: Any = time.sleep,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    first = hand.snapshot()
    sleep(0.10)
    second = hand.snapshot()
    _verify_stable_hand_pair(
        first,
        second,
        expected_hand_id=expected_hand_id,
        label=label,
        open_min_angles=open_min_angles,
    )
    return first, second


def _print_hand(label: str, snapshot: Mapping[str, Any]) -> None:
    print(
        "[RH56 {}] angles={} ANGLE_SET={} currents={} speeds={} forces={}".format(
            label,
            list(_six(snapshot, "angles")),
            list(_six(snapshot, "angle_targets")),
            list(_six(snapshot, "currents")),
            list(_six(snapshot, "speeds")),
            list(_six(snapshot, "force_limits")),
        ),
        flush=True,
    )


def run_franka_only_reset(
    config: Mapping[str, Any],
    *,
    hardware_types: Optional[Mapping[str, Any]] = None,
    network_preflight: Any = None,
    maximum_start_delta_rad: Optional[float] = None,
) -> FrankaOnlyResetProof:
    """Move and stop only Franka without importing or accessing RH56."""

    if maximum_start_delta_rad is not None:
        if isinstance(maximum_start_delta_rad, (bool, np.bool_)):
            raise ValueError("maximum_start_delta_rad must be finite and positive")
        try:
            maximum_start_delta_rad = float(maximum_start_delta_rad)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "maximum_start_delta_rad must be finite and positive"
            ) from exc
        if (
            not np.isfinite(maximum_start_delta_rad)
            or maximum_start_delta_rad <= 0.0
        ):
            raise ValueError("maximum_start_delta_rad must be finite and positive")

    types = (
        _load_franka_hardware_types()
        if hardware_types is None
        else hardware_types
    )
    active_network_preflight = (
        require_uncontended_franka_https_link
        if network_preflight is None
        else network_preflight
    )
    limits = _build_limits(config, types["FrankaMotionLimits"])
    arm_type = types["FrankaSequenceDriver"]
    arm = None
    initial_q = None
    target_q = None
    final_q = None
    initial_delta = None
    final_error = None
    failure: Optional[BaseException] = None
    stop_failure: Optional[BaseException] = None

    try:
        active_network_preflight(str(config["franka"]["ip"]))
        arm = arm_type.connect(
            str(config["franka"]["ip"]), limits, enforce_realtime=True
        )
        initial = arm.robot.read_once()
        initial_success = arm._validate_state(
            initial, require_idle=True, enforce_success=False
        )
        initial_q = np.asarray(initial.q, dtype=np.float64)
        target_q = np.asarray(config["franka"]["default_q_rad"], dtype=np.float64)
        initial_delta = float(np.max(np.abs(initial_q - target_q)))
        print(
            "[Franka preflight] q={} success_rate={:.6f}".format(
                np.round(initial_q, 7).tolist(), float(initial_success)
            ),
            flush=True,
        )
        print(
            "[Franka default] target_q={} max_joint_speed={:.3f}rad/s "
            "max_segment={:.3f}rad".format(
                np.round(target_q, 7).tolist(),
                float(config["franka"]["default_max_joint_velocity_rad_s"]),
                float(config["franka"]["default_max_joint_segment_rad"]),
            ),
            flush=True,
        )
        if (
            maximum_start_delta_rad is not None
            and initial_delta > maximum_start_delta_rad
        ):
            raise RuntimeError(
                "Franka reset start differs from target by "
                f"{initial_delta:.9f}rad, exceeding formal envelope "
                f"{maximum_start_delta_rad:.9f}rad"
            )
        arm.move_joints(target_q)
        final = arm.robot.read_once()
        arm._validate_state(final, require_idle=True, enforce_success=False)
        final_q = np.asarray(final.q, dtype=np.float64)
        final_error = float(np.max(np.abs(final_q - target_q)))
        tolerance = float(config["franka"]["default_arrival_tolerance_rad"])
        if final_error > tolerance:
            raise RuntimeError(
                "Franka default final error {:.7f}rad exceeds {:.7f}rad".format(
                    final_error, tolerance
                )
            )
        print(
            "[Franka default PASS] q={} max_error={:.7f}rad".format(
                np.round(final_q, 7).tolist(), final_error
            ),
            flush=True,
        )
    except BaseException as exc:
        failure = exc
    finally:
        if arm is not None:
            try:
                arm.stop()
            except BaseException as exc:
                stop_failure = exc
            telemetry = getattr(arm, "last_control_loop_telemetry", None)
            if telemetry is not None:
                print(
                    "[Franka/FCI timing] kind={} samples={} "
                    "max_read_to_write={:.3f}us over_500us={}".format(
                        telemetry.kind,
                        int(telemetry.samples),
                        float(telemetry.max_read_to_write_us),
                        int(telemetry.read_to_write_overruns),
                    ),
                    flush=True,
                )

    if stop_failure is not None:
        detail = "{}: {}".format(type(stop_failure).__name__, stop_failure)
        if failure is not None:
            detail += "; original error: {}: {}".format(
                type(failure).__name__, failure
            )
        raise DefaultStopUnconfirmed(
            "STOP UNCONFIRMED: use Franka's physical stop immediately; " + detail
        ) from stop_failure
    if failure is not None:
        raise failure
    if (
        initial_q is None
        or target_q is None
        or final_q is None
        or initial_delta is None
        or final_error is None
    ):
        raise RuntimeError("Franka-only reset completed without numeric proof")
    print(
        "[PASS] Franka is at configured default; RH56 was not accessed; "
        "Franka physical stop verified",
        flush=True,
    )
    return FrankaOnlyResetProof(
        initial_q_rad=tuple(float(value) for value in initial_q),
        target_q_rad=tuple(float(value) for value in target_q),
        final_q_rad=tuple(float(value) for value in final_q),
        initial_linf_delta_rad=initial_delta,
        final_linf_error_rad=final_error,
        maximum_start_delta_rad=maximum_start_delta_rad,
    )


def run_reset(
    config: Mapping[str, Any],
    *,
    port_override: Optional[str] = None,
    hardware_types: Optional[Mapping[str, Any]] = None,
    network_preflight: Any = None,
    sleep: Any = time.sleep,
    maximum_start_delta_rad: Optional[float] = None,
) -> FrankaDefaultResetProof:
    """Verify RH56 default, move only Franka, then prove both are stationary."""

    if maximum_start_delta_rad is not None:
        if isinstance(maximum_start_delta_rad, (bool, np.bool_)):
            raise ValueError("maximum_start_delta_rad must be finite and positive")
        try:
            maximum_start_delta_rad = float(maximum_start_delta_rad)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "maximum_start_delta_rad must be finite and positive"
            ) from exc
        if (
            not np.isfinite(maximum_start_delta_rad)
            or maximum_start_delta_rad <= 0.0
        ):
            raise ValueError("maximum_start_delta_rad must be finite and positive")

    types = _load_hardware_types() if hardware_types is None else hardware_types
    active_network_preflight = (
        require_uncontended_franka_https_link
        if network_preflight is None
        else network_preflight
    )
    api = types["rh56_api"]
    limits = _build_limits(config, types["FrankaMotionLimits"])
    arm_type = types["FrankaSequenceDriver"]
    inspire = config["inspire"]
    open_min_angles = _profile_open_min_angles(config)
    port = str(port_override or inspire["port"])
    serial_context = None
    serial_entered = False
    hand = None
    arm = None
    hand_was_verified = False
    initial_q = None
    target_q = None
    final_q = None
    initial_delta = None
    final_error = None
    failure: Optional[BaseException] = None
    stop_failure: Optional[BaseException] = None
    hand_final_failure: Optional[BaseException] = None

    try:
        serial_context = api.LinuxSerial(
            port, int(inspire["baud"]), 0.5, False
        )
        entered = serial_context.__enter__()
        serial_entered = True
        serial_port = serial_context if entered is None else entered
        hand = api.RH56Hand(serial_port, int(inspire["hand_id"]))
        _, hand_preflight = _read_stable_hand_pair(
            hand,
            expected_hand_id=int(inspire["hand_id"]),
            label="preflight",
            open_min_angles=open_min_angles,
            sleep=sleep,
        )
        hand_was_verified = True
        _print_hand("preflight PASS", hand_preflight)

        # Close the Desk/preflight race immediately before creating the sole
        # Franka owner.  No Franka read thread runs beside the 1 kHz loop.
        active_network_preflight(str(config["franka"]["ip"]))
        arm = arm_type.connect(
            str(config["franka"]["ip"]), limits, enforce_realtime=True
        )
        initial = arm.robot.read_once()
        initial_success = arm._validate_state(
            initial, require_idle=True, enforce_success=False
        )
        initial_q = np.asarray(initial.q, dtype=np.float64)
        target_q = np.asarray(config["franka"]["default_q_rad"], dtype=np.float64)
        initial_delta = float(np.max(np.abs(initial_q - target_q)))
        print(
            "[Franka preflight] q={} success_rate={:.6f}".format(
                np.round(initial_q, 7).tolist(), float(initial_success)
            ),
            flush=True,
        )
        print(
            "[Franka default] target_q={} max_joint_speed={:.3f}rad/s "
            "max_segment={:.3f}rad".format(
                np.round(target_q, 7).tolist(),
                float(config["franka"]["default_max_joint_velocity_rad_s"]),
                float(config["franka"]["default_max_joint_segment_rad"]),
            ),
            flush=True,
        )
        if (
            maximum_start_delta_rad is not None
            and initial_delta > maximum_start_delta_rad
        ):
            raise RuntimeError(
                "Franka reset start differs from target by "
                f"{initial_delta:.9f}rad, exceeding formal envelope "
                f"{maximum_start_delta_rad:.9f}rad"
            )
        arm.move_joints(target_q)
        final = arm.robot.read_once()
        arm._validate_state(final, require_idle=True, enforce_success=False)
        final_q = np.asarray(final.q, dtype=np.float64)
        error = float(np.max(np.abs(final_q - target_q)))
        final_error = error
        tolerance = float(config["franka"]["default_arrival_tolerance_rad"])
        if error > tolerance:
            raise RuntimeError(
                "Franka default final error {:.7f}rad exceeds {:.7f}rad".format(
                    error, tolerance
                )
            )
        print(
            "[Franka default PASS] q={} max_error={:.7f}rad".format(
                np.round(final_q, 7).tolist(), error
            ),
            flush=True,
        )
    except BaseException as exc:
        failure = exc
    finally:
        if arm is not None:
            try:
                arm.stop()
            except BaseException as exc:
                stop_failure = exc
            telemetry = getattr(arm, "last_control_loop_telemetry", None)
            if telemetry is not None:
                print(
                    "[Franka/FCI timing] kind={} samples={} "
                    "max_read_to_write={:.3f}us over_500us={}".format(
                        telemetry.kind,
                        int(telemetry.samples),
                        float(telemetry.max_read_to_write_us),
                        int(telemetry.read_to_write_overruns),
                    ),
                    flush=True,
                )
        if hand is not None and hand_was_verified:
            try:
                _, hand_final = _read_stable_hand_pair(
                    hand,
                    expected_hand_id=int(inspire["hand_id"]),
                    label="final",
                    open_min_angles=open_min_angles,
                    sleep=sleep,
                )
                _print_hand("final PASS", hand_final)
            except BaseException as exc:
                hand_final_failure = exc
        if serial_context is not None and serial_entered:
            try:
                serial_context.__exit__(None, None, None)
            except BaseException as exc:
                if hand_final_failure is None:
                    hand_final_failure = exc

    if stop_failure is not None:
        detail = "{}: {}".format(type(stop_failure).__name__, stop_failure)
        if failure is not None:
            detail += "; original error: {}: {}".format(
                type(failure).__name__, failure
            )
        raise DefaultStopUnconfirmed(
            "STOP UNCONFIRMED: use Franka's physical stop immediately; " + detail
        ) from stop_failure
    if hand_final_failure is not None:
        detail = "{}: {}".format(type(hand_final_failure).__name__, hand_final_failure)
        if failure is not None:
            detail += "; original error: {}: {}".format(
                type(failure).__name__, failure
            )
        raise RuntimeError(
            "Franka stop was verified, but final RH56 open/disabled readback "
            "failed: " + detail
        ) from hand_final_failure
    if failure is not None:
        raise failure
    print(
        "[PASS] Franka is at configured default; RH56 remained open/disabled; "
        "Franka physical stop verified",
        flush=True,
    )
    if (
        initial_q is None
        or target_q is None
        or final_q is None
        or initial_delta is None
        or final_error is None
    ):
        raise RuntimeError("Franka reset completed without numeric proof")
    return FrankaDefaultResetProof(
        initial_q_rad=tuple(float(value) for value in initial_q),
        target_q_rad=tuple(float(value) for value in target_q),
        final_q_rad=tuple(float(value) for value in final_q),
        initial_linf_delta_rad=initial_delta,
        final_linf_error_rad=final_error,
        maximum_start_delta_rad=maximum_start_delta_rad,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, config_path = load_control_config(args.config)
        assets = verify_adapter_assets(config, config_path)
        _validate_static_request(config)
        print(
            "[config] {}\n[adapter] verified={} sha256={}".format(
                config_path, assets.mesh_path, assets.mesh_sha256
            )
        )
        if args.dry_run:
            print(
                "[DRY RUN] RH56 must already be canonical open/disabled; Franka "
                "would move only to default_q={} at <=0.05rad/s. No hardware "
                "driver was imported.".format(config["franka"]["default_q_rad"])
            )
            return 0
        _require_confirmations(args)
        print(
            "[path authority] exact per-run current-to-default swept-workspace "
            "confirmation accepted; this does not modify or unlock the profile"
        )
        run_reset(config, port_override=args.port)
        return 0
    except KeyboardInterrupt:
        print(
            "[interrupted] Ctrl+C received; Franka stop and RH56 final readback "
            "completed",
            file=sys.stderr,
        )
        return 130
    except DefaultStopUnconfirmed as exc:
        print("[{}]".format(exc), file=sys.stderr)
        return 5
    except (OSError, RuntimeError, ValueError) as exc:
        print("[failed] {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
