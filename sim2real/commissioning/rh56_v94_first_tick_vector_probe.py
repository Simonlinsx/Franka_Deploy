#!/usr/bin/env python3
"""Supervised RH56 amplitude/concurrency probe for the shadow21 first tick.

Dry-run is the default.  The armed path commissions the two axes whose first
tick lies beyond the 960 mapping microprobe, returns to open after each, then
executes the exact six-register shadow21 proposal once.  Franka is read only.
This can establish a supervised k=1 hand envelope; it never authorizes C2.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
import uuid

import numpy as np

from .rh56_v94_all_axis_microprobe import (
    V94MaskedAxisProbeDriver,
    _verify_axis_recovery,
    derive_all_axis_mapper_contract,
)
from .rh56_v94_microprobe import (
    ANGLE_TOLERANCE_UNITS,
    COMMISSIONING_FORCE_G,
    COMMISSIONING_SPEED,
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_PROFILE,
    DISABLED_TARGETS,
    ENDPOINT_STABLE_SAMPLES,
    FrankaReadOnlyStationaryGate,
    MAX_AXIS_CURRENT_MA,
    MAX_INACTIVE_DRIFT_UNITS,
    MOTION_TIMEOUT_S,
    MicroprobeError,
    ProfileBinding,
    STOP_MAX_AXIS_CURRENT_MA,
    _default_franka_connector,
    _sha256_bytes,
    _telemetry_payload,
    _validate_disabled_open_snapshot,
    _write_exclusive_json,
    load_v94_profile,
)


DEFAULT_SHADOW21 = (
    Path(__file__).resolve().parents[2]
    / "dexgrasp/runs/v94_live_readonly_clean_dkms_power_on_reuse_guard_20260722_21.npz"
)
EXACT_FIRST_TICK_VECTOR = (1000, 990, 938, 1000, 905, 987)
MIDDLE_MANUFACTURER_AXIS = 2
MIDDLE_COMMISSION_PATH = (970, 940, 930)
THUMB_BEND_MANUFACTURER_AXIS = 4
THUMB_BEND_COMMISSION_PATH = (970, 940, 910, 900)
MASKED_PHASE_PREFIX = "first_tick_masked"
VECTOR_PHASE = "first_tick_exact_vector"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def derive_first_tick_contract(
    shadow_path: str | Path = DEFAULT_SHADOW21,
) -> dict[str, Any]:
    source = Path(shadow_path).expanduser().resolve()
    mapper = derive_all_axis_mapper_contract()
    with np.load(source, allow_pickle=False) as data:
        required = {
            "raw_policy_action13",
            "executed_policy_action13",
            "rh56_angle_act_register_order",
            "rh56_angle_set_register_order",
            "rh56_proposed_angle_set_register_order",
            "rh56_target_q_policy_order_rad",
            "rh56_virtual_q_policy_order_rad",
            "action_clipped",
            "action_reasons_json",
            "write",
            "hardware_writes",
            "robot_command_writes",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise MicroprobeError(f"shadow21 is missing fields: {missing}")
        raw = np.asarray(data["raw_policy_action13"][0], dtype=np.float64)
        executed = np.asarray(
            data["executed_policy_action13"][0], dtype=np.float64
        )
        actual = tuple(
            int(value) for value in data["rh56_angle_act_register_order"][0]
        )
        live_targets = tuple(
            int(value) for value in data["rh56_angle_set_register_order"][0]
        )
        proposed = tuple(
            int(value)
            for value in data["rh56_proposed_angle_set_register_order"][0]
        )
        target_q = [
            float(value) for value in data["rh56_target_q_policy_order_rad"][0]
        ]
        virtual_q = [
            float(value) for value in data["rh56_virtual_q_policy_order_rad"][0]
        ]
        clipped = bool(data["action_clipped"][0])
        reasons = str(data["action_reasons_json"][0])
        write = bool(data["write"][0])
        hardware_writes = bool(np.asarray(data["hardware_writes"]).item())
        robot_writes = bool(np.asarray(data["robot_command_writes"]).item())
    if raw.shape != (13,) or executed.shape != (13,):
        raise MicroprobeError("shadow21 first action is not 13D")
    if not np.array_equal(raw, executed):
        raise MicroprobeError("shadow21 raw/executed first actions differ")
    if proposed != EXACT_FIRST_TICK_VECTOR:
        raise MicroprobeError(
            f"shadow21 first vector changed: {proposed}, expected "
            f"{EXACT_FIRST_TICK_VECTOR}"
        )
    if live_targets != DISABLED_TARGETS or write or hardware_writes or robot_writes:
        raise MicroprobeError("shadow21 source is not a read-only disabled capture")
    if not clipped or "RH56 virtual target step clipped" not in reasons:
        raise MicroprobeError("shadow21 first-tick clip provenance changed")
    return {
        "shadow21_path": str(source),
        "shadow21_sha256": _sha256_bytes(source.read_bytes()),
        "bundle_path": mapper["bundle_path"],
        "bundle_sha256": mapper["bundle_sha256"],
        "bundle_contract": mapper["bundle_contract"],
        "raw_policy_action13": [float(value) for value in raw],
        "executed_policy_action13": [float(value) for value in executed],
        "raw_executed_linf": 0.0,
        "mapper_clip_reason": "RH56 virtual target step clipped",
        "first_angle_act_manufacturer_order": list(actual),
        "source_live_angle_set": list(live_targets),
        "exact_first_tick_vector_manufacturer_order": list(proposed),
        "target_q_policy_order_rad": target_q,
        "virtual_q_policy_order_rad": virtual_q,
        "delta_from_first_angle_act_units": [
            int(target) - int(start) for target, start in zip(proposed, actual)
        ],
        "maximum_abs_delta_from_first_angle_act_units": max(
            abs(int(target) - int(start))
            for target, start in zip(proposed, actual)
        ),
        "source_hardware_writes": False,
    }


class V94FirstTickVectorProbeDriver(V94MaskedAxisProbeDriver):
    """Conservative staged and exact-vector commissioning motions."""

    def _require_probe_preflight(self, phase: str, started: float):
        self._ensure_motion_allowed()
        if not self._disabled_verified:
            raise RuntimeError(f"{phase} requires verified all-six disable")
        feedback = self._read_feedback(f"{phase}_preflight", started)
        self._check_fault_feedback(feedback, f"{phase} preflight")
        if feedback.angle_targets != DISABLED_TARGETS:
            raise RuntimeError(f"{phase} requires ANGLE_SET all -1")
        if any(value < self.open_min_angle for value in feedback.angles):
            raise RuntimeError(f"{phase} requires all axes open")
        if any(value not in (2, 0xFF) for value in feedback.statuses):
            raise RuntimeError(f"{phase} requires all axes idle")
        caps = self._commissioning_current_caps(MAX_AXIS_CURRENT_MA)
        self._check_commissioning_currents(feedback, caps, f"{phase} preflight")
        self._observe_validated_feedback(feedback, f"{phase}_preflight")
        self._configure_commissioning_settings()
        return feedback, tuple(caps)

    def commission_masked_axis_path(
        self,
        manufacturer_axis: int,
        waypoints: Sequence[int],
    ) -> Tuple[int, ...]:
        axis = int(manufacturer_axis)
        path = tuple(int(value) for value in waypoints)
        if axis not in (MIDDLE_MANUFACTURER_AXIS, THUMB_BEND_MANUFACTURER_AXIS):
            raise ValueError("only middle and thumb-bend amplitude paths are allowed")
        expected_path = (
            MIDDLE_COMMISSION_PATH
            if axis == MIDDLE_MANUFACTURER_AXIS
            else THUMB_BEND_COMMISSION_PATH
        )
        if path != expected_path:
            raise ValueError(f"axis {axis} path must be {expected_path}")
        phase_root = f"{MASKED_PHASE_PREFIX}_m{axis}"
        with self._operation_scope():
            try:
                started = self._monotonic()
                preflight, caps = self._require_probe_preflight(phase_root, started)
                inactive_reference = tuple(int(value) for value in preflight.angles)
                previous_configuration = inactive_reference
                previous_actual = int(preflight.angles[axis])
                for waypoint in path:
                    phase = f"{phase_root}_{waypoint:04d}"
                    expected_targets = tuple(
                        waypoint if index == axis else -1 for index in range(6)
                    )
                    current_configuration = list(previous_configuration)
                    current_configuration[axis] = waypoint
                    current_configuration = tuple(current_configuration)
                    segment_start = previous_actual
                    commanded_delta = abs(waypoint - segment_start)
                    required_progress = (
                        0 if commanded_delta < 10 else max(3, commanded_delta // 2)
                    )
                    self._run_external_safety_check(phase, "before numeric target write")
                    self._write_six_verified(
                        self._constant("REG_ANGLE_SET"),
                        expected_targets,
                        numeric_motion=True,
                    )
                    deadline = self._monotonic() + self.motion_timeout_s
                    stable = 0
                    while True:
                        if self._stop_requested.is_set():
                            raise KeyboardInterrupt
                        feedback = self._read_feedback(phase, started)
                        self._check_fault_feedback(feedback, phase)
                        self._check_commissioning_currents(feedback, caps, phase)
                        if feedback.angle_targets != expected_targets:
                            raise RuntimeError(f"{phase}: ANGLE_SET readback changed")
                        self._check_feedback_envelope(
                            feedback,
                            previous_configuration,
                            current_configuration,
                            phase,
                        )
                        for index in range(6):
                            if index == axis:
                                continue
                            if (
                                feedback.statuses[index] not in (2, 0xFF)
                                or abs(
                                    int(feedback.angles[index])
                                    - inactive_reference[index]
                                )
                                > MAX_INACTIVE_DRIFT_UNITS
                            ):
                                raise RuntimeError(f"{phase}: inactive axis moved")
                        status = int(feedback.statuses[axis])
                        if status not in (0, 1, 2):
                            raise RuntimeError(f"{phase}: unsafe selected status {status}")
                        actual = int(feedback.angles[axis])
                        if actual > segment_start + max(4, ANGLE_TOLERANCE_UNITS // 2):
                            raise RuntimeError(f"{phase}: selected axis moved opposite")
                        self._observe_validated_feedback(feedback, phase)
                        reached = (
                            status == 2
                            and abs(actual - waypoint) <= ANGLE_TOLERANCE_UNITS
                            and segment_start - actual >= required_progress
                        )
                        stable = stable + 1 if reached else 0
                        if stable >= ENDPOINT_STABLE_SAMPLES:
                            previous_actual = actual
                            previous_configuration = current_configuration
                            break
                        if self._monotonic() >= deadline:
                            raise RuntimeError(f"{phase}: endpoint timeout at {actual}")
                        self._sleep(self.poll_interval_s)
                self._numeric_hold_targets = tuple(
                    path[-1] if index == axis else -1 for index in range(6)
                )
                self._numeric_hold_current_caps = caps
                self._disabled_verified = False
                return path
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

    def commission_exact_first_tick_vector(self) -> Tuple[int, ...]:
        phase = VECTOR_PHASE
        expected = EXACT_FIRST_TICK_VECTOR
        with self._operation_scope():
            try:
                started = self._monotonic()
                preflight, caps = self._require_probe_preflight(phase, started)
                start = tuple(int(value) for value in preflight.angles)
                required_progress = tuple(
                    0
                    if abs(target - actual) < 10
                    else max(3, abs(target - actual) // 2)
                    for target, actual in zip(expected, start)
                )
                self._run_external_safety_check(phase, "before numeric target write")
                self._write_six_verified(
                    self._constant("REG_ANGLE_SET"), expected, numeric_motion=True
                )
                deadline = self._monotonic() + self.motion_timeout_s
                stable = 0
                while True:
                    if self._stop_requested.is_set():
                        raise KeyboardInterrupt
                    feedback = self._read_feedback(phase, started)
                    self._check_fault_feedback(feedback, phase)
                    self._check_commissioning_currents(feedback, caps, phase)
                    if feedback.angle_targets != expected:
                        raise RuntimeError(f"{phase}: ANGLE_SET readback changed")
                    self._check_feedback_envelope(
                        feedback, start, expected, phase
                    )
                    reached_all = True
                    for axis, (target, initial, progress) in enumerate(
                        zip(expected, start, required_progress)
                    ):
                        status = int(feedback.statuses[axis])
                        if status not in (0, 1, 2):
                            raise RuntimeError(f"{phase}: unsafe status on axis {axis}")
                        actual = int(feedback.angles[axis])
                        slack = max(4, ANGLE_TOLERANCE_UNITS // 2)
                        if target < initial and actual > initial + slack:
                            raise RuntimeError(f"{phase}: axis {axis} moved opposite")
                        if target > initial and actual < initial - slack:
                            raise RuntimeError(f"{phase}: axis {axis} moved opposite")
                        moved = abs(actual - initial) >= progress
                        reached_all = reached_all and (
                            status == 2
                            and abs(actual - target) <= ANGLE_TOLERANCE_UNITS
                            and moved
                        )
                    self._observe_validated_feedback(feedback, phase)
                    stable = stable + 1 if reached_all else 0
                    if stable >= ENDPOINT_STABLE_SAMPLES:
                        break
                    if self._monotonic() >= deadline:
                        raise RuntimeError(
                            f"{phase}: endpoint timeout at {feedback.angles}"
                        )
                    self._sleep(self.poll_interval_s)
                self._numeric_hold_targets = expected
                self._numeric_hold_current_caps = caps
                self._disabled_verified = False
                return expected
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise


def _default_hand_connector(profile: Mapping[str, Any]):
    inspire = profile["inspire"]
    return V94FirstTickVectorProbeDriver.connect(
        port=str(inspire["port"]),
        baud=int(inspire["baud"]),
        hand_id=int(inspire["hand_id"]),
        serial_timeout_s=0.25,
        thumb_rotate_range=tuple(
            int(value)
            for value in inspire["thumb_rotate_validated_realtime_range"]
        ),
        motion_timeout_s=MOTION_TIMEOUT_S,
        poll_interval_s=0.08,
        angle_tolerance=ANGLE_TOLERANCE_UNITS,
        open_min_angle=980,
        thumb_preshape_step_units=40,
        stop_max_axis_current_ma=STOP_MAX_AXIS_CURRENT_MA,
        stop_verify_samples=3,
        stop_verify_interval_s=0.10,
    )


def _verify_masked_path(
    rows: Sequence[Mapping[str, Any]],
    *,
    axis: int,
    path: Sequence[int],
    initial_angles: Sequence[int],
) -> dict[str, Any]:
    start = tuple(int(value) for value in initial_angles)
    maximum_current = 0
    maximum_inactive_drift = 0
    for waypoint in path:
        phase = f"{MASKED_PHASE_PREFIX}_m{axis}_{int(waypoint):04d}"
        samples = [item for item in rows if item.get("phase") == phase]
        if len(samples) < ENDPOINT_STABLE_SAMPLES:
            raise MicroprobeError(f"{phase} lacks stable feedback")
        expected = tuple(
            int(waypoint) if index == axis else -1 for index in range(6)
        )
        for item in samples:
            targets = tuple(int(value) for value in item["angle_targets"])
            angles = tuple(int(value) for value in item["angles"])
            currents = tuple(int(value) for value in item["currents"])
            errors = tuple(int(value) for value in item["errors"])
            statuses = tuple(int(value) for value in item["statuses"])
            temperatures = tuple(int(value) for value in item["temperatures"])
            if targets != expected:
                raise MicroprobeError(f"{phase} exact masked readback failed")
            if (
                any(errors)
                or max(temperatures) >= 60
                or any(abs(value) > MAX_AXIS_CURRENT_MA for value in currents)
                or statuses[axis] not in (0, 1, 2)
                or any(
                    statuses[index] not in (2, 0xFF)
                    for index in range(6)
                    if index != axis
                )
            ):
                raise MicroprobeError(f"{phase} safety feedback failed")
            if angles[axis] > start[axis] + max(
                4, ANGLE_TOLERANCE_UNITS // 2
            ):
                raise MicroprobeError(f"{phase} direction proof failed")
            maximum_current = max(
                maximum_current, *(abs(value) for value in currents)
            )
            maximum_inactive_drift = max(
                maximum_inactive_drift,
                *(
                    abs(angles[index] - start[index])
                    for index in range(6)
                    if index != axis
                ),
            )
        tail = samples[-ENDPOINT_STABLE_SAMPLES:]
        if any(
            int(item["statuses"][axis]) != 2
            or abs(int(item["angles"][axis]) - int(waypoint))
            > ANGLE_TOLERANCE_UNITS
            for item in tail
        ):
            raise MicroprobeError(f"{phase} endpoint proof failed")
    if maximum_inactive_drift > MAX_INACTIVE_DRIFT_UNITS:
        raise MicroprobeError("masked path inactive-axis drift exceeded limit")
    return {
        "manufacturer_axis": int(axis),
        "commanded_waypoints": [int(value) for value in path],
        "final_target": int(path[-1]),
        "exact_masked_readback_confirmed": True,
        "maximum_abs_axis_current_ma": int(maximum_current),
        "maximum_inactive_axis_drift_units": int(maximum_inactive_drift),
        "direction_and_endpoint_confirmed": True,
    }


def _verify_exact_vector(
    rows: Sequence[Mapping[str, Any]], initial_angles: Sequence[int]
) -> dict[str, Any]:
    samples = [item for item in rows if item.get("phase") == VECTOR_PHASE]
    if len(samples) < ENDPOINT_STABLE_SAMPLES:
        raise MicroprobeError("exact first-tick vector lacks stable feedback")
    start = tuple(int(value) for value in initial_angles)
    maximum_current = 0
    for item in samples:
        targets = tuple(int(value) for value in item["angle_targets"])
        angles = tuple(int(value) for value in item["angles"])
        currents = tuple(int(value) for value in item["currents"])
        errors = tuple(int(value) for value in item["errors"])
        statuses = tuple(int(value) for value in item["statuses"])
        temperatures = tuple(int(value) for value in item["temperatures"])
        if targets != EXACT_FIRST_TICK_VECTOR:
            raise MicroprobeError("exact first-tick ANGLE_SET readback failed")
        if (
            any(errors)
            or max(temperatures) >= 60
            or any(abs(value) > MAX_AXIS_CURRENT_MA for value in currents)
            or any(value not in (0, 1, 2) for value in statuses)
        ):
            raise MicroprobeError("exact first-tick safety feedback failed")
        for axis, (target, initial, actual) in enumerate(
            zip(EXACT_FIRST_TICK_VECTOR, start, angles)
        ):
            slack = max(4, ANGLE_TOLERANCE_UNITS // 2)
            if target < initial and actual > initial + slack:
                raise MicroprobeError(
                    f"exact first-tick axis {axis} moved opposite"
                )
            if target > initial and actual < initial - slack:
                raise MicroprobeError(
                    f"exact first-tick axis {axis} moved opposite"
                )
        maximum_current = max(maximum_current, *(abs(value) for value in currents))
    final = tuple(int(value) for value in samples[-1]["angles"])
    if any(
        abs(actual - target) > ANGLE_TOLERANCE_UNITS
        for actual, target in zip(final, EXACT_FIRST_TICK_VECTOR)
    ):
        raise MicroprobeError("exact first-tick endpoint proof failed")
    for axis, (target, initial, actual) in enumerate(
        zip(EXACT_FIRST_TICK_VECTOR, start, final)
    ):
        commanded_delta = abs(target - initial)
        required_progress = (
            0 if commanded_delta < 10 else max(3, commanded_delta // 2)
        )
        if abs(actual - initial) < required_progress:
            raise MicroprobeError(
                f"exact first-tick axis {axis} lacks required progress"
            )
    return {
        "exact_target_manufacturer_order": list(EXACT_FIRST_TICK_VECTOR),
        "initial_angle_act_manufacturer_order": list(start),
        "final_angle_act_manufacturer_order": list(final),
        "delta_from_initial_units": [
            actual - initial for actual, initial in zip(final, start)
        ],
        "maximum_abs_axis_current_ma": int(maximum_current),
        "concurrent_six_register_readback_confirmed": True,
        "direction_and_endpoint_confirmed": True,
    }


def build_plan(
    binding: ProfileBinding,
    run_id: str,
    *,
    contract: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    active_contract = derive_first_tick_contract() if contract is None else dict(contract)
    return {
        "kind": "v94_rh56_first_tick_vector_commissioning",
        "run_id": str(run_id),
        "mode": "DRY_RUN_DISARMED",
        "device_access": False,
        "hardware_writes": False,
        "profile": {"path": str(binding.path), "sha256": binding.sha256},
        "contract": active_contract,
        "fixed_limits": {
            "speed": COMMISSIONING_SPEED,
            "force_limit_g": COMMISSIONING_FORCE_G,
            "maximum_axis_current_ma": MAX_AXIS_CURRENT_MA,
            "maximum_inactive_drift_units": MAX_INACTIVE_DRIFT_UNITS,
            "endpoint_tolerance_units": ANGLE_TOLERANCE_UNITS,
        },
        "stages": [
            {"name": "middle_amplitude", "axis": 2, "path": list(MIDDLE_COMMISSION_PATH)},
            {"name": "reset_open_after_middle", "policy_tick": False},
            {"name": "thumb_bend_amplitude", "axis": 4, "path": list(THUMB_BEND_COMMISSION_PATH)},
            {"name": "reset_open_after_thumb_bend", "policy_tick": False},
            {"name": "exact_first_tick_vector", "target": list(EXACT_FIRST_TICK_VECTOR)},
            {"name": "final_reset_open", "policy_tick": False},
        ],
        "operator_requirements": [
            "RH56 is unloaded and in free air",
            "the complete hand workspace is clear",
            "RH56 24 V cutoff and Franka user stop are immediately reachable",
            "Franka is stationary in Idle at V94 q_home",
            "the operator watches every stage and can press Ctrl+C",
        ],
        "eligible_for_supervised_k1_hand_envelope": False,
        "authorizes_c2": False,
    }


def run_hardware_session(
    binding: ProfileBinding,
    run_id: str,
    *,
    hand_connector: Callable[[Mapping[str, Any]], Any] = _default_hand_connector,
    franka_connector: Callable[[Mapping[str, Any]], tuple[Any, Callable[..., Any]]] = _default_franka_connector,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    contract = derive_first_tick_contract()
    driver = None
    gate = None
    old_sigint = None
    initial_hand = None
    final_hand = None
    stage_records: list[dict[str, Any]] = []
    operation_error: Optional[str] = None
    cleanup_errors: list[str] = []
    stop_verified = False
    settings_restored = False
    interrupted = False
    try:
        driver = hand_connector(binding.payload)

        def handle_sigint(_signum: int, _frame: Any) -> None:
            driver.request_stop()
            raise KeyboardInterrupt

        old_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handle_sigint)
        driver.adopt_disabled_state_and_verify()
        initial_hand = _validate_disabled_open_snapshot(
            driver.read_state_snapshot(), name="initial"
        )
        robot, validate_state = franka_connector(binding.payload)
        gate = FrankaReadOnlyStationaryGate(
            robot, validate_state, binding.payload["franka"]["default_q_rad"]
        )
        gate.prove_initial(sleep=sleep)
        driver.install_external_safety_check(gate.check)

        for name, axis, path in (
            ("middle_amplitude", MIDDLE_MANUFACTURER_AXIS, MIDDLE_COMMISSION_PATH),
            ("thumb_bend_amplitude", THUMB_BEND_MANUFACTURER_AXIS, THUMB_BEND_COMMISSION_PATH),
        ):
            start = _validate_disabled_open_snapshot(
                driver.read_state_snapshot(), name=f"{name}_start"
            )
            telemetry_start = len(driver.telemetry)
            returned = tuple(driver.commission_masked_axis_path(axis, path))
            if returned != tuple(path):
                raise MicroprobeError(f"{name} returned the wrong path")
            rows = _telemetry_payload(driver)[telemetry_start:]
            proof = _verify_masked_path(
                rows, axis=axis, path=path, initial_angles=start["angles"]
            )
            recovery_start = len(driver.telemetry)
            q6_path = tuple(
                driver.reset_to_open(
                    max_axis_current_ma=MAX_AXIS_CURRENT_MA,
                    endpoint_stable_samples=ENDPOINT_STABLE_SAMPLES,
                    max_inactive_drift_units=MAX_INACTIVE_DRIFT_UNITS,
                )
            )
            recovery_rows = _telemetry_payload(driver)[recovery_start:]
            recovery = _verify_axis_recovery(
                recovery_rows, selected_axis=axis, q6_waypoints=q6_path
            )
            stage_records.append(
                {"name": name, "proof": proof, "recovery_proof": recovery}
            )

        vector_start = _validate_disabled_open_snapshot(
            driver.read_state_snapshot(), name="exact_vector_start"
        )
        telemetry_start = len(driver.telemetry)
        returned_vector = tuple(driver.commission_exact_first_tick_vector())
        if returned_vector != EXACT_FIRST_TICK_VECTOR:
            raise MicroprobeError("driver returned the wrong exact first-tick vector")
        vector_rows = _telemetry_payload(driver)[telemetry_start:]
        vector_proof = _verify_exact_vector(
            vector_rows, vector_start["angles"]
        )
        recovery_start = len(driver.telemetry)
        q6_path = tuple(
            driver.reset_to_open(
                max_axis_current_ma=MAX_AXIS_CURRENT_MA,
                endpoint_stable_samples=ENDPOINT_STABLE_SAMPLES,
                max_inactive_drift_units=MAX_INACTIVE_DRIFT_UNITS,
            )
        )
        recovery_rows = _telemetry_payload(driver)[recovery_start:]
        final_recovery = _verify_axis_recovery(
            recovery_rows, selected_axis=5, q6_waypoints=q6_path
        )
        final_recovery["recovery_scope"] = "exact_first_tick_all_axes"
        final_recovery["selected_probe_axis"] = None
        stage_records.append(
            {
                "name": "exact_first_tick_vector",
                "proof": vector_proof,
                "recovery_proof": final_recovery,
            }
        )
    except KeyboardInterrupt:
        interrupted = True
        operation_error = "KeyboardInterrupt: operator requested stop"
    except BaseException as exc:
        operation_error = f"{type(exc).__name__}: {exc}"
    finally:
        if driver is not None:
            try:
                driver.disable_and_verify()
                stop_verified = True
                settings_restored = True
                final_hand = _validate_disabled_open_snapshot(
                    driver.read_state_snapshot(), name="final"
                )
            except BaseException as exc:
                cleanup_errors.append(
                    f"disable/stop/settings verification: {type(exc).__name__}: {exc}"
                )
            try:
                driver.close()
            except BaseException as exc:
                cleanup_errors.append(f"driver close: {type(exc).__name__}: {exc}")
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)

    profile_unchanged = False
    try:
        profile_unchanged = _sha256_bytes(binding.path.read_bytes()) == binding.sha256
    except OSError as exc:
        cleanup_errors.append(f"profile recheck: {type(exc).__name__}: {exc}")
    complete = bool(
        len(stage_records) == 3
        and all(item.get("proof") and item.get("recovery_proof") for item in stage_records)
    )
    passed = bool(
        operation_error is None
        and not cleanup_errors
        and complete
        and stop_verified
        and settings_restored
        and final_hand is not None
        and gate is not None
        and profile_unchanged
    )
    return {
        "kind": "v94_rh56_first_tick_vector_commissioning_evidence",
        "schema_version": 1,
        "run_id": str(run_id),
        "completed_at_utc": _utc_now(),
        "result": "PASS" if passed else "FAIL",
        "interrupted": bool(interrupted),
        "contract": contract,
        "profile": {
            "path": str(binding.path),
            "sha256": binding.sha256,
            "unchanged_after_run": bool(profile_unchanged),
        },
        "stage_records": stage_records,
        "exact_first_tick_vector_manufacturer_order": list(EXACT_FIRST_TICK_VECTOR),
        "franka_read_only": None if gate is None else gate.report(),
        "initial_hand": initial_hand,
        "telemetry": [] if driver is None else _telemetry_payload(driver),
        "cleanup": {
            "double_disable_and_stationary_stop_verified": bool(stop_verified),
            "default_speed_force_settings_restored": bool(settings_restored),
            "final_hand": final_hand,
            "errors": cleanup_errors,
        },
        "eligible_for_supervised_k1_hand_envelope": bool(passed),
        "authorizes_c2": False,
        "operation_error": operation_error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run by default; commission the exact shadow21 RH56 first tick"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    input_fn: Callable[[str], str] = input,
    stdin_isatty: Optional[Callable[[], bool]] = None,
    run_id_factory: Callable[[], Any] = uuid.uuid4,
    hardware_session: Callable[..., dict[str, Any]] = run_hardware_session,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        binding = load_v94_profile(args.config)
        run_id = str(run_id_factory())
        contract = derive_first_tick_contract()
        plan = build_plan(binding, run_id, contract=contract)
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            print("[dry-run] no device was opened and no register was written")
            return 0
        tty_check = sys.stdin.isatty if stdin_isatty is None else stdin_isatty
        if not tty_check():
            raise MicroprobeError("--execute requires an interactive TTY")
        output = (
            DEFAULT_EVIDENCE_DIR
            / f"v94_rh56_first_tick_vector_probe_{run_id}.json"
            if args.output is None
            else args.output
        ).expanduser().resolve()
        if output.exists():
            raise MicroprobeError(f"evidence output already exists: {output}")
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("\nBefore confirming, verify ALL conditions:")
        for requirement in plan["operator_requirements"]:
            print(f"  - {requirement}")
        phrase = f"EXECUTE-RH56-FIRST-TICK-VECTOR {run_id}"
        answer = input_fn(f"\nType exactly:\n{phrase}\n> ")
        if answer.strip() != phrase:
            raise MicroprobeError("run-scoped interactive confirmation did not match")
        evidence = hardware_session(binding, run_id)
        _write_exclusive_json(output, evidence)
        print(f"[first-tick-vector] result={evidence['result']} evidence={output}")
        if evidence["result"] != "PASS":
            if not evidence.get("cleanup", {}).get(
                "double_disable_and_stationary_stop_verified", False
            ):
                print(
                    "[STOP UNCONFIRMED] cut RH56 24 V immediately if motion remains",
                    file=sys.stderr,
                )
            return 130 if evidence.get("interrupted") else 2
        print(
            "[first-tick-vector] PASS: eligible for supervised k=1 hand envelope; "
            "C2 is not authorized"
        )
        return 0
    except KeyboardInterrupt:
        print(
            "[interrupted][STOP STATE UNKNOWN] cut RH56 24 V if motion remains",
            file=sys.stderr,
        )
        return 130
    except (OSError, MicroprobeError, ValueError) as exc:
        print(f"[first-tick-vector][refused] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
