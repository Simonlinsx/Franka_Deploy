#!/usr/bin/env python3
"""Supervised physical mapping probe for all six V94 RH56 policy axes.

The default command is an offline dry-run.  The hardware path is deliberately
interactive and performs six independent first-policy-tick probes.  For each
policy hand axis a fresh production :class:`V94ActionMapper` maps action -0.6
(all other hand actions -1) from the open semantic state.  It must produce one
and only one manufacturer-register change, 1000 -> 960, at the exact expected
policy-to-register permutation.

The physical write is stricter than the mapper vector: only that selected
manufacturer register is numeric; the other five ``ANGLE_SET`` registers stay
-1.  Every axis is returned to all-open/all-disabled by the separately
labelled ``RH56ResetOpenDriver.reset_to_open`` recovery before the next axis.
Recovery is not a policy tick.  The released 6D-to-12D workbook remains a
kinematic reconstruction; this probe never claims twelve independently
measured or independently driven URDF joints.
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

from .rh56_v94_microprobe import (
    ANGLE_TOLERANCE_UNITS,
    COMMISSIONING_FORCE_G,
    COMMISSIONING_SPEED,
    DEFAULT_BUNDLE,
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_PROFILE,
    DISABLED_TARGETS,
    ENDPOINT_STABLE_SAMPLES,
    FrankaReadOnlyStationaryGate,
    MAX_AXIS_CURRENT_MA,
    MAX_INACTIVE_DRIFT_UNITS,
    MOTION_TIMEOUT_S,
    MicroprobeError,
    OPEN_ENDPOINT,
    OPEN_TARGETS,
    PROBE_ENDPOINT,
    ProfileBinding,
    STOP_MAX_AXIS_CURRENT_MA,
    _default_franka_connector,
    _json_safe,
    _sha256_bytes,
    _telemetry_payload,
    _validate_disabled_open_snapshot,
    _write_exclusive_json,
    load_v94_profile,
)


DEXGRASP_ROOT = Path(__file__).resolve().parents[2] / "dexgrasp"
DEXGRASP_SOURCE = DEXGRASP_ROOT / "src"
if str(DEXGRASP_SOURCE) not in sys.path:
    sys.path.insert(0, str(DEXGRASP_SOURCE))

from anydex_pipeline.inspire_sequence_driver import (  # noqa: E402
    AXIS_NAMES,
    RH56MotionStopped,
    RH56SequenceDriverError,
)
from anydex_pipeline.rh56_reset_open import RH56ResetOpenDriver  # noqa: E402


MAPPER_SELECTED_ACTION = -0.6
POLICY_HAND_ACTION_OFFSET = 7
PROBE_PHASE_PREFIX = "v94_masked_axis_probe"
EXPECTED_POLICY_ORDER = (
    "thumb_rotation",
    "thumb_bending",
    "index",
    "middle",
    "ring",
    "little",
)
EXPECTED_REGISTER_ORDER = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bending",
    "thumb_rotation",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def derive_all_axis_mapper_contract(
    bundle_path: str | Path = DEFAULT_BUNDLE,
) -> dict[str, Any]:
    """Derive all six one-tick register permutations using production code."""

    import numpy as np

    from sim2real.deployment.bundle import DeployBundle
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import (
        POLICY_HAND_ORDER,
        REGISTER_HAND_ORDER,
        V94Contract,
    )

    if tuple(POLICY_HAND_ORDER) != EXPECTED_POLICY_ORDER:
        raise MicroprobeError("V94 policy hand-axis order changed")
    if tuple(REGISTER_HAND_ORDER) != EXPECTED_REGISTER_ORDER:
        raise MicroprobeError("V94 manufacturer register order changed")
    source = Path(bundle_path).expanduser().resolve()
    bundle = DeployBundle(source)
    verification = bundle.verify()
    contract = V94Contract.from_bundle(bundle)
    axes = []
    for policy_axis, name in enumerate(POLICY_HAND_ORDER):
        mapper = V94ActionMapper(
            initial_arm_target_q_rad=contract.q_home_rad,
            initial_hand_target_q_policy_order_rad=np.zeros(6, dtype=np.float32),
            joint_limits_rad=contract.joint_limits_rad,
            q_hand_close_rad=contract.q_hand_close_rad,
        )
        hand_action = [-1.0] * 6
        hand_action[policy_axis] = MAPPER_SELECTED_ACTION
        action = np.asarray([0.0] * 7 + hand_action, dtype=np.float32)
        mapped = mapper.map(action, measured_q_rad=contract.q_home_rad)
        registers = tuple(int(value) for value in mapped.rh56_angle_set_register_order)
        changed = tuple(index for index, value in enumerate(registers) if value != 1000)
        register_axis = REGISTER_HAND_ORDER.index(name)
        expected = list(OPEN_TARGETS)
        expected[register_axis] = PROBE_ENDPOINT
        if registers != tuple(expected) or changed != (register_axis,):
            raise MicroprobeError(
                f"production mapper cross-coupled policy axis {name}: {registers}"
            )
        if mapped.reasons:
            raise MicroprobeError(
                f"production mapper unexpectedly clipped policy axis {name}: "
                f"{mapped.reasons}"
            )
        axes.append(
            {
                "policy_action_index": POLICY_HAND_ACTION_OFFSET + policy_axis,
                "policy_axis_index": policy_axis,
                "policy_axis": name,
                "manufacturer_register_index_zero_based": register_axis,
                "manufacturer_register_axis_human": register_axis + 1,
                "manufacturer_register_name": REGISTER_HAND_ORDER[register_axis],
                "policy_action13": [float(value) for value in action],
                "semantic_target_q_policy_order_rad": [
                    float(value) for value in mapped.rh56_target_q_policy_order_rad
                ],
                "mapper_register_vector": list(registers),
                "physical_masked_angle_set": [
                    PROBE_ENDPOINT if index == register_axis else -1
                    for index in range(6)
                ],
            }
        )
    return {
        "implementation": "sim2real.contracts.actions.V94ActionMapper",
        "bundle_path": str(source),
        "bundle_sha256": _sha256_bytes(source.read_bytes()),
        "bundle_contract": verification.bundle_contract,
        "initial_hand_semantic_q_policy_order_rad": [0.0] * 6,
        "selected_hand_action": MAPPER_SELECTED_ACTION,
        "unselected_hand_action": -1.0,
        "axes": axes,
        "all_six_exact_single_register_changes": True,
    }


class V94MaskedAxisProbeDriver(RH56ResetOpenDriver):
    """One numeric manufacturer axis at a time; all others remain disabled."""

    def commission_masked_axis_960(
        self,
        manufacturer_axis: int,
        *,
        max_axis_current_ma: int = MAX_AXIS_CURRENT_MA,
        endpoint_stable_samples: int = ENDPOINT_STABLE_SAMPLES,
        max_inactive_drift_units: int = MAX_INACTIVE_DRIFT_UNITS,
    ) -> Tuple[int, ...]:
        if isinstance(manufacturer_axis, bool) or not isinstance(manufacturer_axis, int):
            raise ValueError("manufacturer_axis must be an integer in 0..5")
        axis = int(manufacturer_axis)
        if not 0 <= axis < 6:
            raise ValueError("manufacturer_axis must be an integer in 0..5")
        if not 2 <= int(endpoint_stable_samples) <= 10:
            raise ValueError("endpoint_stable_samples must be in 2..10")
        if not 1 <= int(max_inactive_drift_units) <= 20:
            raise ValueError("max_inactive_drift_units must be in 1..20")
        if axis == 5:
            lower, upper = self.thumb_rotate_range
            if not lower <= PROBE_ENDPOINT < OPEN_ENDPOINT <= upper:
                raise RH56SequenceDriverError(
                    "axis-5 probe is outside the commissioned thumb range"
                )

        expected_targets = tuple(
            PROBE_ENDPOINT if index == axis else -1 for index in range(6)
        )
        current_configuration = tuple(
            PROBE_ENDPOINT if index == axis else OPEN_ENDPOINT for index in range(6)
        )
        phase = f"{PROBE_PHASE_PREFIX}_m{axis}_{PROBE_ENDPOINT:04d}"
        with self._operation_scope():
            try:
                self._ensure_motion_allowed()
                if not self._disabled_verified:
                    raise RH56SequenceDriverError(
                        "masked single-axis probe requires verified all-six disable"
                    )
                started = self._monotonic()
                preflight = self._read_feedback(f"{phase}_preflight", started)
                self._check_fault_feedback(preflight, f"{phase} preflight")
                if preflight.angle_targets != DISABLED_TARGETS:
                    raise RH56SequenceDriverError(
                        "masked single-axis probe requires ANGLE_SET all -1"
                    )
                if any(value < self.open_min_angle for value in preflight.angles):
                    raise RH56SequenceDriverError(
                        f"masked single-axis probe requires all axes open: {preflight.angles}"
                    )
                if not all(value in (2, 0xFF) for value in preflight.statuses):
                    raise RH56SequenceDriverError(
                        "masked single-axis probe requires all axes idle"
                    )
                caps = self._commissioning_current_caps(max_axis_current_ma)
                self._check_commissioning_currents(preflight, caps, f"{phase} preflight")
                self._observe_validated_feedback(preflight, f"{phase}_preflight")
                self._configure_commissioning_settings()
                self._run_external_safety_check(phase, "before numeric target write")
                self._write_six_verified(
                    self._constant("REG_ANGLE_SET"),
                    expected_targets,
                    numeric_motion=True,
                )

                inactive_reference = tuple(int(value) for value in preflight.angles)
                active_start = int(preflight.angles[axis])
                commanded_delta = max(0, active_start - PROBE_ENDPOINT)
                required_progress = (
                    0 if commanded_delta < 10 else max(3, commanded_delta // 2)
                )
                opposite_slack = max(4, self.angle_tolerance // 2)
                deadline = self._monotonic() + self.motion_timeout_s
                stable = 0
                while True:
                    if self._stop_requested.is_set():
                        raise RH56MotionStopped(
                            f"masked manufacturer axis {axis} probe was interrupted"
                        )
                    feedback = self._read_feedback(phase, started)
                    self._check_fault_feedback(feedback, phase)
                    self._check_commissioning_currents(feedback, caps, phase)
                    if feedback.angle_targets != expected_targets:
                        raise RH56SequenceDriverError(
                            f"{phase}: ANGLE_SET changed: {feedback.angle_targets}"
                        )
                    self._check_feedback_envelope(
                        feedback,
                        OPEN_TARGETS,
                        current_configuration,
                        phase,
                        previous_phase="start_open",
                        current_phase=f"manufacturer_axis_{axis}_first_tick",
                    )
                    for index in range(6):
                        if index == axis:
                            continue
                        if (
                            feedback.angles[index] < self.open_min_angle
                            or abs(
                                int(feedback.angles[index])
                                - inactive_reference[index]
                            )
                            > int(max_inactive_drift_units)
                            or feedback.statuses[index] not in (2, 0xFF)
                        ):
                            raise RH56SequenceDriverError(
                                f"{phase}: inactive {AXIS_NAMES[index]} moved or left idle"
                            )
                    status = int(feedback.statuses[axis])
                    if status == 3:
                        raise RH56SequenceDriverError(
                            f"{phase}: selected axis reported contact in free air"
                        )
                    if status not in (0, 1, 2):
                        raise RH56SequenceDriverError(
                            f"{phase}: selected axis returned unsupported status {status}"
                        )
                    actual = int(feedback.angles[axis])
                    if actual > active_start + opposite_slack:
                        raise RH56SequenceDriverError(
                            f"{phase}: selected axis moved opposite the command"
                        )
                    self._observe_validated_feedback(feedback, phase)
                    reached = (
                        status == 2
                        and abs(actual - PROBE_ENDPOINT) <= ANGLE_TOLERANCE_UNITS
                        and active_start - actual >= required_progress
                    )
                    stable = stable + 1 if reached else 0
                    if stable >= int(endpoint_stable_samples):
                        break
                    if self._monotonic() >= deadline:
                        raise RH56SequenceDriverError(
                            f"{phase}: timed out; ANGLE_ACT={actual}, target=960, "
                            f"status={status}"
                        )
                    self._sleep(self.poll_interval_s)

                self._preshaped_q6 = None
                self._commissioned_q6_waypoints = None
                self._commissioned_bend_waypoints = None
                self._numeric_hold_targets = expected_targets
                self._numeric_hold_current_caps = tuple(caps)
                self._disabled_verified = False
                return expected_targets
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise


def _default_hand_connector(profile: Mapping[str, Any]) -> V94MaskedAxisProbeDriver:
    inspire = profile["inspire"]
    return V94MaskedAxisProbeDriver.connect(
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


def _verify_axis_probe(
    telemetry: Sequence[Mapping[str, Any]],
    *,
    derivation: Mapping[str, Any],
    initial_angles: Sequence[int],
) -> dict[str, Any]:
    axis = int(derivation["manufacturer_register_index_zero_based"])
    phase = f"{PROBE_PHASE_PREFIX}_m{axis}_{PROBE_ENDPOINT:04d}"
    rows = [item for item in telemetry if item.get("phase") == phase]
    if len(rows) < ENDPOINT_STABLE_SAMPLES:
        raise MicroprobeError(f"axis {axis} lacks stable probe feedback")
    start = tuple(int(value) for value in initial_angles)
    if len(start) != 6:
        raise MicroprobeError(f"axis {axis} initial ANGLE_ACT vector is malformed")
    expected = tuple(int(value) for value in derivation["physical_masked_angle_set"])
    max_inactive_drift = 0
    max_current = 0
    max_temperature = 0
    for row in rows:
        targets = tuple(int(value) for value in row["angle_targets"])
        angles = tuple(int(value) for value in row["angles"])
        currents = tuple(int(value) for value in row["currents"])
        errors = tuple(int(value) for value in row["errors"])
        statuses = tuple(int(value) for value in row["statuses"])
        temperatures = tuple(int(value) for value in row["temperatures"])
        if not all(
            len(values) == 6
            for values in (
                targets,
                angles,
                currents,
                errors,
                statuses,
                temperatures,
            )
        ):
            raise MicroprobeError(f"axis {axis} feedback vector is malformed")
        if targets != expected:
            raise MicroprobeError(f"axis {axis} exact masked readback failed")
        if any(errors) or max(temperatures) >= 60:
            raise MicroprobeError(f"axis {axis} fault/temperature check failed")
        if statuses[axis] not in (0, 1, 2):
            raise MicroprobeError(f"axis {axis} selected status is unsafe")
        if any(
            statuses[index] not in (2, 0xFF)
            for index in range(6)
            if index != axis
        ):
            raise MicroprobeError(f"axis {axis} inactive status left idle")
        max_current = max(max_current, *(abs(value) for value in currents))
        max_temperature = max(max_temperature, *temperatures)
        max_inactive_drift = max(
            max_inactive_drift,
            *(
                abs(angles[index] - start[index])
                for index in range(6)
                if index != axis
            ),
        )
        if angles[axis] > start[axis] + max(4, ANGLE_TOLERANCE_UNITS // 2):
            raise MicroprobeError(f"axis {axis} moved opposite the command")
    final = tuple(int(value) for value in rows[-1]["angles"])
    if max_current > MAX_AXIS_CURRENT_MA:
        raise MicroprobeError(f"axis {axis} exceeded 100mA")
    if max_inactive_drift > MAX_INACTIVE_DRIFT_UNITS:
        raise MicroprobeError(f"axis {axis} caused inactive-axis drift")
    if (
        abs(final[axis] - PROBE_ENDPOINT) > ANGLE_TOLERANCE_UNITS
        or final[axis] >= start[axis]
    ):
        raise MicroprobeError(f"axis {axis} direction/endpoint check failed")
    stable_tail = rows[-ENDPOINT_STABLE_SAMPLES:]
    if any(
        int(row["statuses"][axis]) != 2
        or abs(int(row["angles"][axis]) - PROBE_ENDPOINT)
        > ANGLE_TOLERANCE_UNITS
        or int(row["angles"][axis]) >= start[axis]
        for row in stable_tail
    ):
        raise MicroprobeError(f"axis {axis} lacks stable idle endpoint feedback")
    return {
        "policy_action_index": int(derivation["policy_action_index"]),
        "policy_axis": str(derivation["policy_axis"]),
        "manufacturer_register_index_zero_based": axis,
        "manufacturer_register_name": str(derivation["manufacturer_register_name"]),
        "mapper_register_vector": list(derivation["mapper_register_vector"]),
        "physical_masked_angle_set": list(expected),
        "initial_selected_angle_act": int(start[axis]),
        "final_selected_angle_act": int(final[axis]),
        "feedback_samples": len(rows),
        "maximum_inactive_axis_drift_units": int(max_inactive_drift),
        "maximum_abs_axis_current_ma": int(max_current),
        "maximum_temperature_c": int(max_temperature),
        "physical_direction_confirmed": True,
        "urdf_12d_independent_measurement_claimed": False,
    }


def _verify_axis_recovery(
    telemetry: Sequence[Mapping[str, Any]],
    *,
    selected_axis: int,
    q6_waypoints: Sequence[int],
) -> dict[str, Any]:
    rows = [
        item for item in telemetry if str(item.get("phase", "")).startswith("reset_open")
    ]
    bend_rows = [
        item
        for item in rows
        if str(item.get("phase", "")).startswith("reset_open_bend_m")
    ]
    if len(bend_rows) < 5 * ENDPOINT_STABLE_SAMPLES:
        raise MicroprobeError(
            f"axis {selected_axis} recovery lacks bend-helper feedback transcript"
        )
    helper_max_current = 0
    for bend_axis in range(5):
        phase = f"reset_open_bend_m{bend_axis}_1000"
        axis_rows = [item for item in bend_rows if item.get("phase") == phase]
        if len(axis_rows) < ENDPOINT_STABLE_SAMPLES:
            raise MicroprobeError(
                f"axis {selected_axis} recovery lacks {phase} stable feedback"
            )
        expected_targets = tuple(
            OPEN_ENDPOINT if index == bend_axis else -1 for index in range(6)
        )
        inactive_reference = tuple(int(value) for value in axis_rows[0]["angles"])
        for item in axis_rows:
            targets = tuple(int(value) for value in item["angle_targets"])
            angles = tuple(int(value) for value in item["angles"])
            currents = tuple(int(value) for value in item["currents"])
            errors = tuple(int(value) for value in item["errors"])
            statuses = tuple(int(value) for value in item["statuses"])
            temperatures = tuple(int(value) for value in item["temperatures"])
            if targets != expected_targets:
                raise MicroprobeError(
                    f"axis {selected_axis} recovery helper lost masked target"
                )
            if (
                any(errors)
                or max(temperatures) >= 60
                or any(abs(value) > MAX_AXIS_CURRENT_MA for value in currents)
            ):
                raise MicroprobeError(
                    f"axis {selected_axis} recovery helper safety gate failed"
                )
            helper_max_current = max(
                helper_max_current, *(abs(value) for value in currents)
            )
            for inactive_axis in range(6):
                if inactive_axis == bend_axis:
                    continue
                if (
                    statuses[inactive_axis] not in (2, 0xFF)
                    or abs(
                        angles[inactive_axis]
                        - inactive_reference[inactive_axis]
                    )
                    > MAX_INACTIVE_DRIFT_UNITS
                ):
                    raise MicroprobeError(
                        f"axis {selected_axis} recovery helper moved inactive axis"
                    )
        stable_tail = axis_rows[-ENDPOINT_STABLE_SAMPLES:]
        if any(
            int(item["statuses"][bend_axis]) != 2
            or int(item["angles"][bend_axis]) < 980
            for item in stable_tail
        ):
            raise MicroprobeError(
                f"axis {selected_axis} recovery helper lacks stable open endpoint"
            )
    final_rows = [item for item in rows if item.get("phase") == "reset_open_final"]
    if not final_rows:
        raise MicroprobeError(f"axis {selected_axis} recovery lacks final feedback")
    final = final_rows[-1]
    targets = tuple(int(value) for value in final["angle_targets"])
    angles = tuple(int(value) for value in final["angles"])
    currents = tuple(int(value) for value in final["currents"])
    statuses = tuple(int(value) for value in final["statuses"])
    errors = tuple(int(value) for value in final["errors"])
    temperatures = tuple(int(value) for value in final["temperatures"])
    if (
        targets != DISABLED_TARGETS
        or any(value < 980 for value in angles)
        or any(abs(value) > STOP_MAX_AXIS_CURRENT_MA for value in currents)
        or any(value not in (2, 0xFF) for value in statuses)
        or any(errors)
        or max(temperatures) >= 60
    ):
        raise MicroprobeError(
            f"axis {selected_axis} recovery did not finish open/disabled/idle"
        )
    q6_path = tuple(int(value) for value in q6_waypoints)
    if q6_path and q6_path[-1] != OPEN_ENDPOINT:
        raise MicroprobeError("reset recovery q6 path did not finish at 1000")
    return {
        "policy_tick": False,
        "selected_probe_axis": int(selected_axis),
        "bend_open_helper_used": True,
        "bend_open_helper_host_current_cap_ma": MAX_AXIS_CURRENT_MA,
        "bend_open_helper_feedback_samples": len(bend_rows),
        "bend_open_helper_maximum_abs_axis_current_ma": int(helper_max_current),
        "bend_open_helper_exact_masked_transcript_confirmed": True,
        "q6_commanded_waypoints": list(q6_path),
        "q6_deadband_escape_used": any(
            "backoff" in str(item.get("phase", "")) for item in rows
        ),
        "final_angles": list(angles),
        "final_targets": list(targets),
        "verified_open_disabled_idle": True,
    }


def build_plan(
    binding: ProfileBinding,
    run_id: str,
    *,
    mapper_contract: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    contract = (
        derive_all_axis_mapper_contract()
        if mapper_contract is None
        else dict(mapper_contract)
    )
    return {
        "kind": "v94_rh56_all_six_policy_axis_mapping_microprobe",
        "run_id": str(run_id),
        "mode": "DRY_RUN_DISARMED",
        "device_access": False,
        "hardware_writes": False,
        "profile": {"path": str(binding.path), "sha256": binding.sha256},
        "mapper_contract": contract,
        "driver_audit": {
            "existing_coupled_bend_helper_accepted_for_policy_probe": False,
            "reason": (
                "it numerically commands other axes to 1000 instead of keeping "
                "their ANGLE_SET at -1"
            ),
            "replacement": (
                "V94MaskedAxisProbeDriver.commission_masked_axis_960 with exact "
                "batch readback and five disabled non-target registers"
            ),
        },
        "fixed_limits": {
            "speed": COMMISSIONING_SPEED,
            "force_limit_g": COMMISSIONING_FORCE_G,
            "maximum_axis_current_ma": MAX_AXIS_CURRENT_MA,
            "maximum_inactive_drift_units": MAX_INACTIVE_DRIFT_UNITS,
            "endpoint_tolerance_units": ANGLE_TOLERANCE_UNITS,
            "motion_timeout_s_per_probe_or_recovery_segment": MOTION_TIMEOUT_S,
        },
        "sequence": [
            "prove all-open/all-disabled and Franka read-only Idle/stationary",
            "for each V94 policy hand axis: execute one masked mapper-derived 960 probe",
            "verify selected ANGLE_ACT direction and all inactive/fault/current gates",
            "run audited reset_to_open recovery (not a policy tick)",
            "finish all-open/all-disabled and restore speed/force settings",
        ],
        "urdf_12d_statement": (
            "no claim of twelve independent actuators or measurements; the official "
            "6D-to-12D workbook remains only a kinematic reconstruction"
        ),
        "operator_requirements": [
            "RH56 is unloaded and in free air",
            "the entire hand/adapter workspace is clear for all six small motions",
            "RH56 24 V cutoff and Franka user stop are immediately reachable",
            "Franka is stationary in Idle at the V94 reset pose",
            "the operator continuously watches each single-axis motion",
        ],
        "authorizes_closed_loop": False,
    }


def run_hardware_session(
    binding: ProfileBinding,
    run_id: str,
    *,
    hand_connector: Callable[[Mapping[str, Any]], Any] = _default_hand_connector,
    franka_connector: Callable[
        [Mapping[str, Any]], tuple[Any, Callable[..., Any]]
    ] = _default_franka_connector,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    mapper_contract = derive_all_axis_mapper_contract()
    driver = None
    gate = None
    old_sigint = None
    initial_hand = None
    final_hand = None
    axis_records: list[dict[str, Any]] = []
    operation_error: Optional[str] = None
    stop_errors: list[str] = []
    stop_verified = False
    settings_restored = False
    interrupted = False
    try:
        driver = hand_connector(binding.payload)

        def _handle_sigint(_signum: int, _frame: Any) -> None:
            if driver is not None:
                driver.request_stop()
            raise KeyboardInterrupt

        old_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handle_sigint)
        driver.adopt_disabled_state_and_verify()
        initial_hand = _validate_disabled_open_snapshot(
            driver.read_state_snapshot(), name="initial"
        )
        robot, validate_state = franka_connector(binding.payload)
        gate = FrankaReadOnlyStationaryGate(
            robot,
            validate_state,
            binding.payload["franka"]["default_q_rad"],
        )
        gate.prove_initial(sleep=sleep)
        driver.install_external_safety_check(gate.check)

        for derivation in mapper_contract["axes"]:
            axis = int(derivation["manufacturer_register_index_zero_based"])
            record: dict[str, Any] = {
                "derivation": dict(derivation),
                "probe_proof": None,
                "recovery_proof": None,
            }
            axis_records.append(record)
            start = _validate_disabled_open_snapshot(
                driver.read_state_snapshot(), name=f"axis_{axis}_start"
            )
            telemetry_start = len(driver.telemetry)
            returned = tuple(
                driver.commission_masked_axis_960(
                    axis,
                    max_axis_current_ma=MAX_AXIS_CURRENT_MA,
                    endpoint_stable_samples=ENDPOINT_STABLE_SAMPLES,
                    max_inactive_drift_units=MAX_INACTIVE_DRIFT_UNITS,
                )
            )
            expected = tuple(int(value) for value in derivation["physical_masked_angle_set"])
            if returned != expected:
                raise MicroprobeError(f"axis {axis} driver returned wrong target")
            probe_rows = _telemetry_payload(driver)[telemetry_start:]
            record["probe_proof"] = _verify_axis_probe(
                probe_rows,
                derivation=derivation,
                initial_angles=start["angles"],
            )

            recovery_start = len(driver.telemetry)
            q6_waypoints = tuple(
                driver.reset_to_open(
                    max_axis_current_ma=MAX_AXIS_CURRENT_MA,
                    endpoint_stable_samples=ENDPOINT_STABLE_SAMPLES,
                    max_inactive_drift_units=MAX_INACTIVE_DRIFT_UNITS,
                )
            )
            recovery_rows = _telemetry_payload(driver)[recovery_start:]
            record["recovery_proof"] = _verify_axis_recovery(
                recovery_rows,
                selected_axis=axis,
                q6_waypoints=q6_waypoints,
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
                stop_errors.append(
                    f"disable/stop/settings verification: {type(exc).__name__}: {exc}"
                )
            try:
                driver.close()
            except BaseException as exc:
                stop_errors.append(f"driver close: {type(exc).__name__}: {exc}")
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)

    profile_unchanged = False
    try:
        profile_unchanged = _sha256_bytes(binding.path.read_bytes()) == binding.sha256
    except OSError as exc:
        stop_errors.append(f"profile recheck: {type(exc).__name__}: {exc}")
    if not profile_unchanged and operation_error is None:
        operation_error = "MicroprobeError: V94 profile changed during execution"
    all_axes_passed = bool(
        len(axis_records) == 6
        and all(
            item.get("probe_proof") is not None
            and item.get("recovery_proof") is not None
            for item in axis_records
        )
    )
    passed = bool(
        operation_error is None
        and not stop_errors
        and all_axes_passed
        and stop_verified
        and settings_restored
        and final_hand is not None
        and gate is not None
        and profile_unchanged
    )
    return {
        "kind": "v94_rh56_all_six_policy_axis_mapping_microprobe_evidence",
        "schema_version": 1,
        "run_id": str(run_id),
        "completed_at_utc": _utc_now(),
        "result": "PASS" if passed else "FAIL",
        "interrupted": bool(interrupted),
        "profile": {
            "path": str(binding.path),
            "sha256": binding.sha256,
            "unchanged_after_run": bool(profile_unchanged),
        },
        "mapper_contract": mapper_contract,
        "axis_records": axis_records,
        "all_six_policy_to_register_to_angle_act_directions_confirmed": all_axes_passed,
        "urdf_12d_independent_measurement_claimed": False,
        "franka_read_only": None if gate is None else gate.report(),
        "initial_hand": initial_hand,
        "telemetry": [] if driver is None else _telemetry_payload(driver),
        "cleanup": {
            "double_disable_and_stationary_stop_verified": bool(stop_verified),
            "default_speed_force_settings_restored": bool(settings_restored),
            "final_hand": final_hand,
            "errors": stop_errors,
        },
        "eligible_as_full_six_axis_physical_mapping_evidence": bool(passed),
        "authorizes_closed_loop": False,
        "operation_error": operation_error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run by default; optionally probe all six V94 policy-to-RH56 "
            "physical axis mappings one axis at a time"
        )
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
        mapper_contract = derive_all_axis_mapper_contract()
        plan = build_plan(binding, run_id, mapper_contract=mapper_contract)
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            print("[dry-run] no device was opened and no register was written")
            return 0
        tty_check = sys.stdin.isatty if stdin_isatty is None else stdin_isatty
        if not tty_check():
            raise MicroprobeError("--execute requires an interactive TTY")
        output = (
            DEFAULT_EVIDENCE_DIR / f"v94_rh56_all6_mapping_microprobe_{run_id}.json"
            if args.output is None
            else args.output
        ).expanduser().resolve()
        if output.exists():
            raise MicroprobeError(f"evidence output already exists: {output}")
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("\nBefore confirming, verify ALL conditions:")
        for requirement in plan["operator_requirements"]:
            print(f"  - {requirement}")
        phrase = f"EXECUTE-RH56-ALL6-MAPPING {run_id}"
        answer = input_fn(f"\nType exactly:\n{phrase}\n> ")
        if answer.strip() != phrase:
            raise MicroprobeError("run-scoped interactive confirmation did not match")
        evidence = hardware_session(binding, run_id)
        _write_exclusive_json(output, evidence)
        print(f"[all6-microprobe] result={evidence['result']} evidence={output}")
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
            "[all6-microprobe] PASS: six policy->register->ANGLE_ACT directions "
            "confirmed; this still does not authorize closed-loop motion"
        )
        return 0
    except KeyboardInterrupt:
        print(
            "[interrupted][STOP STATE UNKNOWN] cut RH56 24 V if motion remains",
            file=sys.stderr,
        )
        return 130
    except (OSError, MicroprobeError, ValueError) as exc:
        print(f"[all6-microprobe][refused] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
