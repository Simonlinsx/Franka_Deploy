#!/usr/bin/env python3
"""Supervised V94 RH56 thumb-rotation physical-mapping microprobe.

This is a deliberately narrow C1 commissioning entry point.  The default
invocation is an offline dry-run.  ``--execute`` is interactive and permits
only manufacturer register axis 5 (zero based, the sixth RH56 register) for
one production-mapper tick from 1000 -> 960.  The hand then returns open over
the separately labelled, audited adaptive reset path; endpoint deadband
recovery is never presented as another policy tick.

Franka is opened read-only.  No Franka controller, control handle, stop,
end-effector, load, or command write is created.  A fresh Franka Idle and
stationary sample gates every RH56 feedback sample.  RH56 I/O is delegated to
the reviewed ``RH56SequenceDriver`` so exact target readback, current/fault/
temperature/status checks, two-pass disable, multi-sample physical-stop
verification, and temporary speed/force restoration are not reimplemented
here.

This probe is physical mapping evidence for one RH56 axis.  It is not a V94
policy run and cannot authorize closed-loop execution.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
import uuid


WORKSPACE = Path(__file__).resolve().parents[2]
DEXGRASP_ROOT = WORKSPACE / "dexgrasp"
DEFAULT_PROFILE = DEXGRASP_ROOT / "configs/fr3_rh56_v94_commissioning.json"
DEFAULT_EVIDENCE_DIR = DEXGRASP_ROOT / "runs"
from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE

PROFILE_ID = "fr3_rh56_v94_reset_locked_v1"
POLICY_CONTRACT = "v94_inspire_semantic_13d"
REGISTER_AXIS = 5
REGISTER_AXIS_NAME = "thumb_rotation"
POLICY_ACTION_INDEX = 7
OPEN_ENDPOINT = 1000
PROBE_ENDPOINT = 960
PROBE_STEP_UNITS = 40
COMMISSIONING_SPEED = 40
COMMISSIONING_FORCE_G = 80
MAX_AXIS_CURRENT_MA = 100
STOP_MAX_AXIS_CURRENT_MA = 100
MAX_INACTIVE_DRIFT_UNITS = 8
ENDPOINT_STABLE_SAMPLES = 3
ANGLE_TOLERANCE_UNITS = 20
MOTION_TIMEOUT_S = 5.0
FRANKA_MAX_DQ_RAD_S = 0.020
FRANKA_MAX_RESET_ERROR_RAD = 0.050
FRANKA_MAX_SESSION_DRIFT_RAD = 0.003
FRANKA_INITIAL_STABLE_SAMPLES = 3
FRANKA_INITIAL_SAMPLE_INTERVAL_S = 0.05
PYLIBFRANKA_VERSION = "0.21.2"
MAPPER_PROBE_ACTION = -0.6
MAPPER_MAX_ROLLOUT_STEPS = 1
DISABLED_TARGETS: Tuple[int, ...] = (-1,) * 6
OPEN_TARGETS: Tuple[int, ...] = (OPEN_ENDPOINT,) * 6


class MicroprobeError(RuntimeError):
    """A fail-closed profile, admission, feedback, or cleanup failure."""


@dataclass(frozen=True)
class ProfileBinding:
    path: Path
    sha256: str
    payload: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_v94_profile(path: str | Path) -> ProfileBinding:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MicroprobeError(f"V94 commissioning profile not found: {source}")
    raw = source.read_bytes()
    if len(raw) > 1024 * 1024:
        raise MicroprobeError("V94 commissioning profile is unexpectedly large")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MicroprobeError(f"invalid V94 commissioning profile: {exc}") from exc
    if not isinstance(payload, dict):
        raise MicroprobeError("V94 commissioning profile must contain an object")
    _validate_v94_profile(payload)
    return ProfileBinding(source, _sha256_bytes(raw), payload)


def _validate_v94_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("profile_id") != PROFILE_ID:
        raise MicroprobeError(f"profile_id must be exactly {PROFILE_ID}")
    if profile.get("mode") != "commissioning_locked":
        raise MicroprobeError("microprobe requires the locked commissioning profile")
    if profile.get("policy_contract") != POLICY_CONTRACT:
        raise MicroprobeError(f"policy_contract must be exactly {POLICY_CONTRACT}")
    inspire = profile.get("inspire")
    franka = profile.get("franka")
    tool = profile.get("tool")
    if not all(isinstance(item, Mapping) for item in (inspire, franka, tool)):
        raise MicroprobeError("profile franka/inspire/tool sections are required")
    assert isinstance(inspire, Mapping)
    assert isinstance(franka, Mapping)
    assert isinstance(tool, Mapping)

    if tuple(inspire.get("open_targets", ())) != OPEN_TARGETS:
        raise MicroprobeError("V94 RH56 open_targets must be all 1000")
    validated = tuple(inspire.get("thumb_rotate_validated_realtime_range", ()))
    if len(validated) != 2:
        raise MicroprobeError("thumb rotation validated range is missing")
    try:
        lower, upper = (int(validated[0]), int(validated[1]))
    except (TypeError, ValueError) as exc:
        raise MicroprobeError("thumb rotation validated range is malformed") from exc
    if not lower <= PROBE_ENDPOINT < OPEN_ENDPOINT <= upper:
        raise MicroprobeError(
            "the fixed 960..1000 microprobe is outside the validated thumb range"
        )
    if int(inspire.get("open_speed", -1)) > COMMISSIONING_SPEED or int(
        inspire.get("close_speed", -1)
    ) > COMMISSIONING_SPEED:
        raise MicroprobeError("profile RH56 speed exceeds the fixed speed-40 probe")
    if int(inspire.get("force_limit_g", -1)) > COMMISSIONING_FORCE_G:
        raise MicroprobeError("profile RH56 force exceeds the fixed 80g probe")
    if int(inspire.get("baud", -1)) != 115200 or int(
        inspire.get("hand_id", -1)
    ) != 1:
        raise MicroprobeError("unexpected RH56 baud or hand id")
    if not isinstance(inspire.get("port"), str) or not inspire["port"]:
        raise MicroprobeError("RH56 serial port is missing")

    default_q = tuple(franka.get("default_q_rad", ()))
    if len(default_q) != 7 or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in default_q
    ):
        raise MicroprobeError("Franka V94 reset q is malformed")
    if not isinstance(franka.get("ip"), str) or not franka["ip"]:
        raise MicroprobeError("Franka IP is missing")
    for key in (
        "installed_on_franka_verified",
        "mount_transform_commissioned",
        "assembled_yaw_verified",
        "source_origin_datum_verified",
    ):
        if tool.get(key) is not True:
            raise MicroprobeError(f"tool.{key} must be true")
    if tool.get("material") != "PLA" or tool.get(
        "low_speed_unloaded_commissioning_only"
    ) is not True:
        raise MicroprobeError("probe is limited to the installed unloaded PLA setup")


def derive_v94_mapper_probe_endpoint(
    bundle_path: str | Path = DEFAULT_BUNDLE,
) -> dict[str, Any]:
    """Use the production V94 mapper to derive the exact axis-5 endpoint.

    With the production 0.20 target filter and 0.05-rad virtual-step cap,
    ``action[7] == -0.6`` from the open semantic state produces q=0.05 rad and
    manufacturer register axis 5 equal to 960 on the first policy tick.  The
    physical probe executes that exact production-mapper endpoint.
    """

    import numpy as np

    from sim2real.deployment.bundle import DeployBundle
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import V94Contract

    bundle_source = Path(bundle_path).expanduser().resolve()
    bundle = DeployBundle(bundle_source)
    verification = bundle.verify()
    contract = V94Contract.from_bundle(bundle)
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=contract.q_home_rad,
        initial_hand_target_q_policy_order_rad=np.zeros(6, dtype=np.float32),
        joint_limits_rad=contract.joint_limits_rad,
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    action = np.asarray(
        [0.0] * 7 + [MAPPER_PROBE_ACTION] + [-1.0] * 5,
        dtype=np.float32,
    )
    selected = None
    rollout = []
    for tick in range(1, MAPPER_MAX_ROLLOUT_STEPS + 1):
        mapped = mapper.map(action, measured_q_rad=contract.q_home_rad)
        registers = tuple(int(value) for value in mapped.rh56_angle_set_register_order)
        rollout.append(
            {
                "tick": tick,
                "register_axis5": registers[REGISTER_AXIS],
                "semantic_thumb_rotation_q_rad": float(
                    mapped.rh56_target_q_policy_order_rad[0]
                ),
            }
        )
        if registers[:5] != OPEN_TARGETS[:5]:
            raise MicroprobeError(
                "production mapper coupled a non-target RH56 register during derivation"
            )
        if registers[REGISTER_AXIS] == PROBE_ENDPOINT:
            selected = (tick, mapped, registers)
            break
        if registers[REGISTER_AXIS] < PROBE_ENDPOINT:
            raise MicroprobeError("production mapper skipped below the 960 endpoint")
    if selected is None:
        raise MicroprobeError("production mapper did not produce the exact 960 endpoint")
    tick, mapped, registers = selected
    if tick != 1 or registers != OPEN_TARGETS[:5] + (PROBE_ENDPOINT,):
        raise MicroprobeError(
            f"unexpected V94 mapper endpoint derivation: tick={tick}, registers={registers}"
        )
    return {
        "implementation": "sim2real.contracts.actions.V94ActionMapper",
        "bundle_path": str(bundle_source),
        "bundle_sha256": _sha256_bytes(bundle_source.read_bytes()),
        "bundle_contract": verification.bundle_contract,
        "initial_hand_semantic_q_policy_order_rad": [0.0] * 6,
        "policy_action13": [float(value) for value in action],
        "selected_tick": int(tick),
        "selected_semantic_q_policy_order_rad": [
            float(value) for value in mapped.rh56_target_q_policy_order_rad
        ],
        "selected_register_vector_manufacturer_order": list(registers),
        "rollout": rollout,
        "important_scope": (
            "offline mapper rollout only; physical probe masks manufacturer axes "
            "0..4 to -1 and executes only the derived axis-5 endpoint"
        ),
    }


def build_plan(
    binding: ProfileBinding,
    run_id: str,
    *,
    mapper_derivation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    inspire = binding.payload["inspire"]
    derivation = (
        derive_v94_mapper_probe_endpoint()
        if mapper_derivation is None
        else dict(mapper_derivation)
    )
    register_vector = tuple(
        int(value)
        for value in derivation["selected_register_vector_manufacturer_order"]
    )
    if register_vector != OPEN_TARGETS[:5] + (PROBE_ENDPOINT,):
        raise MicroprobeError("mapper derivation does not bind the fixed probe endpoint")
    masked_probe_target = DISABLED_TARGETS[:5] + (register_vector[REGISTER_AXIS],)
    return {
        "kind": "v94_rh56_thumb_rotation_physical_mapping_microprobe",
        "run_id": str(run_id),
        "mode": "DRY_RUN_DISARMED",
        "device_access": False,
        "hardware_writes": False,
        "profile": {"path": str(binding.path), "sha256": binding.sha256},
        "scope": {
            "policy_action_index": POLICY_ACTION_INDEX,
            "policy_axis": REGISTER_AXIS_NAME,
            "manufacturer_register_index_zero_based": REGISTER_AXIS,
            "manufacturer_register_axis_human": 6,
            "policy_tick_command_path": [OPEN_ENDPOINT, PROBE_ENDPOINT],
            "policy_tick_numeric_write": list(masked_probe_target),
            "mapping_semantics": (
                "positive V94 thumb_rotation semantic action increases virtual q "
                "and therefore decreases manufacturer ANGLE_SET[5]"
            ),
            "mapper_derivation": derivation,
            "physical_probe_scope": (
                "only action[7] -> manufacturer register[5] -> ANGLE_ACT[5]; "
                "the mapper's six-register vector is not written unchanged"
            ),
            "recovery": {
                "implementation": (
                    "anydex_pipeline.rh56_reset_open."
                    "RH56ResetOpenDriver.reset_to_open"
                ),
                "policy_tick": False,
                "target": OPEN_ENDPOINT,
                "path": (
                    "live-state anchored; open-end deadband may require a "
                    "bounded backoff, double-disable/fresh anchor, then 1000"
                ),
            },
        },
        "fixed_limits": {
            "speed": COMMISSIONING_SPEED,
            "force_limit_g": COMMISSIONING_FORCE_G,
            "host_max_axis_current_ma": MAX_AXIS_CURRENT_MA,
            "stop_max_axis_current_ma": STOP_MAX_AXIS_CURRENT_MA,
            "motion_timeout_s_per_segment": MOTION_TIMEOUT_S,
            "inactive_axis_drift_units": MAX_INACTIVE_DRIFT_UNITS,
            "endpoint_tolerance_units": ANGLE_TOLERANCE_UNITS,
        },
        "franka": {
            "ip": str(binding.payload["franka"]["ip"]),
            "access": "read_once_only_no_controller_no_robot_write",
            "required_mode": "Idle",
            "max_abs_dq_rad_s": FRANKA_MAX_DQ_RAD_S,
            "max_reset_error_rad": FRANKA_MAX_RESET_ERROR_RAD,
            "max_session_drift_rad": FRANKA_MAX_SESSION_DRIFT_RAD,
        },
        "rh56": {
            "port": str(inspire["port"]),
            "start_and_finish": "all-six ANGLE_SET=-1; idle/stationary verified",
            "temporary_settings_restored_after_verified_disable": True,
        },
        "operator_requirements": [
            "RH56 carries no object and the hand is in free air",
            "the full hand/adapter workspace is clear",
            "RH56 24 V cutoff and Franka user stop are immediately reachable",
            "Franka is stationary in Idle at the V94 reset pose",
        ],
        "authorizes_closed_loop": False,
    }


class FrankaReadOnlyStationaryGate:
    """Fresh ``read_once`` Idle/rest gate; it has no command methods."""

    def __init__(
        self,
        robot: Any,
        validate_state: Callable[..., Any],
        expected_reset_q_rad: Sequence[float],
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._robot = robot
        self._validate_state = validate_state
        self._expected = tuple(float(value) for value in expected_reset_q_rad)
        if len(self._expected) != 7:
            raise ValueError("expected_reset_q_rad must contain seven values")
        self._monotonic = monotonic
        self._anchor: Optional[Tuple[float, ...]] = None
        self.check_count = 0
        self.max_abs_dq_rad_s = 0.0
        self.max_reset_error_rad = 0.0
        self.max_session_drift_rad = 0.0
        self.last_sample: Optional[dict[str, Any]] = None

    @staticmethod
    def _seven_finite(value: Any, name: str) -> Tuple[float, ...]:
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise MicroprobeError(f"Franka {name} is malformed") from exc
        if len(result) != 7 or not all(math.isfinite(item) for item in result):
            raise MicroprobeError(f"Franka {name} is malformed")
        return result

    def check(self) -> None:
        state = self._robot.read_once()
        self._validate_state(state, require_idle=True, enforce_success=False)
        q = self._seven_finite(getattr(state, "q", None), "q")
        dq = self._seven_finite(getattr(state, "dq", None), "dq")
        max_dq = max(abs(value) for value in dq)
        reset_error = max(abs(value - ref) for value, ref in zip(q, self._expected))
        if max_dq > FRANKA_MAX_DQ_RAD_S:
            raise MicroprobeError(
                f"Franka is not stationary: max|dq|={max_dq:.6f}rad/s"
            )
        if reset_error > FRANKA_MAX_RESET_ERROR_RAD:
            raise MicroprobeError(
                "Franka is outside the V94 reset envelope: "
                f"max|q-q_reset|={reset_error:.6f}rad"
            )
        if self._anchor is None:
            self._anchor = q
        drift = max(abs(value - anchor) for value, anchor in zip(q, self._anchor))
        if drift > FRANKA_MAX_SESSION_DRIFT_RAD:
            raise MicroprobeError(
                f"Franka moved during the hand probe: max drift={drift:.6f}rad"
            )
        self.check_count += 1
        self.max_abs_dq_rad_s = max(self.max_abs_dq_rad_s, max_dq)
        self.max_reset_error_rad = max(self.max_reset_error_rad, reset_error)
        self.max_session_drift_rad = max(self.max_session_drift_rad, drift)
        self.last_sample = {
            "captured_monotonic_s": float(self._monotonic()),
            "robot_mode": str(getattr(state, "robot_mode", "")),
            "q_rad": list(q),
            "dq_rad_s": list(dq),
        }

    def prove_initial(
        self,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        for index in range(FRANKA_INITIAL_STABLE_SAMPLES):
            self.check()
            if index + 1 < FRANKA_INITIAL_STABLE_SAMPLES:
                sleep(FRANKA_INITIAL_SAMPLE_INTERVAL_S)

    def report(self) -> dict[str, Any]:
        return {
            "connection": "read_once_only_no_controller_no_robot_write",
            "check_count": int(self.check_count),
            "maximum_abs_dq_rad_s": float(self.max_abs_dq_rad_s),
            "maximum_reset_error_rad": float(self.max_reset_error_rad),
            "maximum_session_drift_rad": float(self.max_session_drift_rad),
            "last_sample": self.last_sample,
            "verified": self.check_count >= FRANKA_INITIAL_STABLE_SAMPLES,
        }


def _validate_disabled_open_snapshot(snapshot: Any, *, name: str) -> dict[str, Any]:
    payload = _json_safe(snapshot)
    if not isinstance(payload, Mapping):
        raise MicroprobeError(f"{name} RH56 snapshot is malformed")
    targets = tuple(int(value) for value in payload.get("angle_targets", ()))
    angles = tuple(int(value) for value in payload.get("angles", ()))
    currents = tuple(int(value) for value in payload.get("currents", ()))
    errors = tuple(int(value) for value in payload.get("errors", ()))
    statuses = tuple(int(value) for value in payload.get("statuses", ()))
    temperatures = tuple(int(value) for value in payload.get("temperatures", ()))
    if not all(len(item) == 6 for item in (targets, angles, currents, errors, statuses, temperatures)):
        raise MicroprobeError(f"{name} RH56 snapshot lacks six-axis feedback")
    if targets != DISABLED_TARGETS:
        raise MicroprobeError(f"{name} RH56 ANGLE_SET is not all -1: {targets}")
    if any(value < 980 for value in angles):
        raise MicroprobeError(f"{name} RH56 is not approximately open: {angles}")
    if any(abs(value) > STOP_MAX_AXIS_CURRENT_MA for value in currents):
        raise MicroprobeError(f"{name} RH56 current is not idle: {currents}")
    if any(errors):
        raise MicroprobeError(f"{name} RH56 ERROR is nonzero: {errors}")
    if any(value not in (2, 0xFF) for value in statuses):
        raise MicroprobeError(f"{name} RH56 STATUS is not idle: {statuses}")
    if max(temperatures) >= 60:
        raise MicroprobeError(f"{name} RH56 temperature is unsafe: {temperatures}")
    return dict(payload)


def _telemetry_payload(driver: Any) -> list[dict[str, Any]]:
    return [dict(_json_safe(item)) for item in tuple(getattr(driver, "telemetry", ()))]


def _verify_probe_telemetry(
    telemetry: Sequence[Mapping[str, Any]], initial_angles: Sequence[int]
) -> dict[str, Any]:
    initial = tuple(int(value) for value in initial_angles)
    if len(initial) != 6:
        raise MicroprobeError("initial RH56 angle vector is malformed")
    forward = [item for item in telemetry if item.get("phase") == "q6_step_0960"]
    if len(forward) < ENDPOINT_STABLE_SAMPLES:
        raise MicroprobeError("probe lacks stable outbound feedback")

    inactive_max_drift = 0
    maximum_current = 0
    maximum_temperature = 0
    for phase, samples, target in (("forward", forward, PROBE_ENDPOINT),):
        expected = DISABLED_TARGETS[:5] + (target,)
        for sample in samples:
            targets = tuple(int(value) for value in sample["angle_targets"])
            angles = tuple(int(value) for value in sample["angles"])
            currents = tuple(int(value) for value in sample["currents"])
            errors = tuple(int(value) for value in sample["errors"])
            statuses = tuple(int(value) for value in sample["statuses"])
            temperatures = tuple(int(value) for value in sample["temperatures"])
            if targets != expected:
                raise MicroprobeError(f"{phase} ANGLE_SET readback mismatch: {targets}")
            if any(errors) or max(temperatures) >= 60:
                raise MicroprobeError(f"{phase} RH56 fault/temperature feedback failed")
            if any(status not in (2, 0xFF) for status in statuses[:5]):
                raise MicroprobeError(f"{phase} inactive axis left idle")
            if statuses[5] not in (0, 1, 2):
                raise MicroprobeError(f"{phase} thumb rotation status is unsupported")
            maximum_current = max(maximum_current, *(abs(value) for value in currents))
            maximum_temperature = max(maximum_temperature, *temperatures)
            inactive_max_drift = max(
                inactive_max_drift,
                *(abs(value - reference) for value, reference in zip(angles[:5], initial[:5])),
            )
    if maximum_current > MAX_AXIS_CURRENT_MA:
        raise MicroprobeError("probe exceeded the fixed 100mA current cap")
    if inactive_max_drift > MAX_INACTIVE_DRIFT_UNITS:
        raise MicroprobeError("an inactive RH56 axis drifted beyond the fixed bound")
    forward_last = tuple(int(value) for value in forward[-1]["angles"])
    if abs(forward_last[5] - PROBE_ENDPOINT) > ANGLE_TOLERANCE_UNITS:
        raise MicroprobeError("thumb rotation did not reach the 960 endpoint")
    if forward_last[5] >= initial[5]:
        raise MicroprobeError("thumb rotation physical direction is inverted or absent")
    return {
        "forward_samples": len(forward),
        "initial_angle_act_axis5": int(initial[5]),
        "forward_endpoint_angle_act_axis5": int(forward_last[5]),
        "maximum_inactive_axis_drift_units": int(inactive_max_drift),
        "maximum_abs_axis_current_ma": int(maximum_current),
        "maximum_temperature_c": int(maximum_temperature),
        "physical_direction_confirmed": True,
        "policy_mapping_conclusion": (
            "V94 policy action[7] positive -> semantic thumb rotation closes -> "
            "manufacturer ANGLE_SET[5] decreases; observed ANGLE_ACT[5] decreased"
        ),
    }


def _verify_reset_recovery_telemetry(
    telemetry: Sequence[Mapping[str, Any]], waypoints: Sequence[int]
) -> dict[str, Any]:
    reset_rows = [
        item for item in telemetry if str(item.get("phase", "")).startswith("reset_open")
    ]
    final_rows = [item for item in reset_rows if item.get("phase") == "reset_open_final"]
    if not final_rows:
        raise MicroprobeError("adaptive reset recovery lacks final feedback")
    final = final_rows[-1]
    targets = tuple(int(value) for value in final["angle_targets"])
    angles = tuple(int(value) for value in final["angles"])
    currents = tuple(int(value) for value in final["currents"])
    errors = tuple(int(value) for value in final["errors"])
    statuses = tuple(int(value) for value in final["statuses"])
    temperatures = tuple(int(value) for value in final["temperatures"])
    if targets != DISABLED_TARGETS:
        raise MicroprobeError("adaptive reset did not finish all-six disabled")
    if any(value < 980 for value in angles):
        raise MicroprobeError(f"adaptive reset did not reach open feedback: {angles}")
    if any(abs(value) > STOP_MAX_AXIS_CURRENT_MA for value in currents):
        raise MicroprobeError("adaptive reset final current is not idle")
    if any(errors) or any(value not in (2, 0xFF) for value in statuses):
        raise MicroprobeError("adaptive reset final fault/status check failed")
    if max(temperatures) >= 60:
        raise MicroprobeError("adaptive reset final temperature is unsafe")
    requested = tuple(int(value) for value in waypoints)
    if not requested or requested[-1] != OPEN_ENDPOINT:
        raise MicroprobeError(f"adaptive reset did not command final 1000: {requested}")
    return {
        "implementation": (
            "anydex_pipeline.rh56_reset_open.RH56ResetOpenDriver.reset_to_open"
        ),
        "policy_tick": False,
        "commanded_axis5_waypoints": list(requested),
        "deadband_escape_backoff_used": any(
            "backoff" in str(item.get("phase", "")) for item in reset_rows
        ),
        "final_angles": list(angles),
        "final_targets": list(targets),
        "final_currents_ma": list(currents),
        "final_statuses": list(statuses),
        "verified_open_disabled_idle": True,
    }


def _default_hand_connector(profile: Mapping[str, Any]) -> Any:
    source_root = DEXGRASP_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from anydex_pipeline.rh56_reset_open import RH56ResetOpenDriver

    inspire = profile["inspire"]
    return RH56ResetOpenDriver.connect(
        port=str(inspire["port"]),
        baud=int(inspire["baud"]),
        hand_id=int(inspire["hand_id"]),
        serial_timeout_s=0.25,
        thumb_rotate_range=(900, 1000),
        motion_timeout_s=MOTION_TIMEOUT_S,
        poll_interval_s=0.08,
        angle_tolerance=ANGLE_TOLERANCE_UNITS,
        open_min_angle=980,
        thumb_preshape_step_units=PROBE_STEP_UNITS,
        stop_max_axis_current_ma=STOP_MAX_AXIS_CURRENT_MA,
        stop_verify_samples=3,
        stop_verify_interval_s=0.10,
    )


def _default_franka_connector(
    profile: Mapping[str, Any],
) -> tuple[Any, Callable[..., Any]]:
    try:
        installed_version = importlib.metadata.version("pylibfranka")
    except importlib.metadata.PackageNotFoundError as exc:
        raise MicroprobeError("pylibfranka is not installed in this Python") from exc
    if installed_version != PYLIBFRANKA_VERSION:
        raise MicroprobeError(
            f"pylibfranka must be pinned to {PYLIBFRANKA_VERSION}, got {installed_version}"
        )
    import numpy as np
    import pylibfranka

    source_root = DEXGRASP_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from anydex_pipeline.franka_sequence_driver import (
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )

    franka = profile["franka"]
    dynamics = franka["expected_end_effector"]
    limits = FrankaMotionLimits(
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
    robot = pylibfranka.Robot(
        str(franka["ip"]), pylibfranka.RealtimeConfig.kIgnore
    )
    validator = FrankaSequenceDriver(robot, pylibfranka, limits)._validate_state
    return robot, validator


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
    """Execute the fixed microprobe after the CLI's run-scoped confirmation."""

    driver = None
    gate = None
    old_sigint = None
    initial_hand: Optional[dict[str, Any]] = None
    final_hand: Optional[dict[str, Any]] = None
    mapping_proof: Optional[dict[str, Any]] = None
    reached_forward: Tuple[int, ...] = ()
    recovery_waypoints: Tuple[int, ...] = ()
    recovery_proof: Optional[dict[str, Any]] = None
    operation_error: Optional[str] = None
    stop_errors: list[str] = []
    stop_verified = False
    settings_restored = False
    interrupted = False
    mapper_derivation = derive_v94_mapper_probe_endpoint()
    mapper_registers = tuple(
        int(value)
        for value in mapper_derivation[
            "selected_register_vector_manufacturer_order"
        ]
    )
    if mapper_registers != OPEN_TARGETS[:5] + (PROBE_ENDPOINT,):
        raise MicroprobeError("production mapper did not derive the fixed endpoint")
    probe_endpoint = mapper_registers[REGISTER_AXIS]

    try:
        driver = hand_connector(binding.payload)

        def _handle_sigint(_signum: int, _frame: Any) -> None:
            if driver is not None:
                driver.request_stop()
            raise KeyboardInterrupt

        old_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handle_sigint)

        # First post-connect action is always all-six disable.  No device
        # setting or Franka connection is touched before this succeeds.
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

        reached_forward = tuple(
            driver.commission_thumb_sweep(
                probe_endpoint,
                step_units=PROBE_STEP_UNITS,
                max_axis_current_ma=MAX_AXIS_CURRENT_MA,
                endpoint_stable_samples=ENDPOINT_STABLE_SAMPLES,
                max_inactive_drift_units=MAX_INACTIVE_DRIFT_UNITS,
            )
        )
        if reached_forward != (probe_endpoint,):
            raise MicroprobeError(f"unexpected forward command path: {reached_forward}")
        # Seal the action->register->physical-direction claim before recovery.
        # The adaptive reset below is explicitly not a policy tick.
        mapping_proof = _verify_probe_telemetry(
            _telemetry_payload(driver), initial_hand["angles"]
        )
        recovery_waypoints = tuple(
            driver.reset_to_open(
                max_axis_current_ma=MAX_AXIS_CURRENT_MA,
                endpoint_stable_samples=ENDPOINT_STABLE_SAMPLES,
                max_inactive_drift_units=MAX_INACTIVE_DRIFT_UNITS,
            )
        )
        telemetry = _telemetry_payload(driver)
        recovery_proof = _verify_reset_recovery_telemetry(
            telemetry, recovery_waypoints
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
                stop_errors.append(f"disable/stop/settings verification: {type(exc).__name__}: {exc}")
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

    telemetry = [] if driver is None else _telemetry_payload(driver)
    passed = bool(
        operation_error is None
        and not stop_errors
        and mapping_proof is not None
        and recovery_proof is not None
        and reached_forward == (PROBE_ENDPOINT,)
        and bool(recovery_waypoints)
        and recovery_waypoints[-1] == OPEN_ENDPOINT
        and stop_verified
        and settings_restored
        and final_hand is not None
        and gate is not None
        and gate.check_count >= FRANKA_INITIAL_STABLE_SAMPLES
        and profile_unchanged
    )
    return {
        "kind": "v94_rh56_thumb_rotation_physical_mapping_microprobe_evidence",
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
        "scope": {
            "policy_action_index": POLICY_ACTION_INDEX,
            "policy_axis": REGISTER_AXIS_NAME,
            "manufacturer_register_index_zero_based": REGISTER_AXIS,
            "manufacturer_register_axis_human": 6,
            "policy_tick_command_path": [OPEN_ENDPOINT, PROBE_ENDPOINT],
            "recovery_target": OPEN_ENDPOINT,
            "recovery_is_policy_tick": False,
            "authorizes_closed_loop": False,
        },
        "limits": build_plan(
            binding, run_id, mapper_derivation=mapper_derivation
        )["fixed_limits"],
        "mapper_derivation": mapper_derivation,
        "franka_read_only": None if gate is None else gate.report(),
        "initial_hand": initial_hand,
        "forward_waypoints": list(reached_forward),
        "recovery_waypoints": list(recovery_waypoints),
        "mapping_proof": mapping_proof,
        "recovery_proof": recovery_proof,
        "telemetry": telemetry,
        "cleanup": {
            "double_disable_and_stationary_stop_verified": bool(stop_verified),
            "temporary_speed_force_settings_restored": bool(settings_restored),
            "final_hand": final_hand,
            "errors": stop_errors,
        },
        "operation_error": operation_error,
    }


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if not destination.parent.is_dir():
        raise MicroprobeError(f"evidence directory does not exist: {destination.parent}")
    encoded = (
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run by default; optionally execute one supervised V94 RH56 "
            "thumb-rotation 1000->960 policy-tick microprobe followed by an "
            "audited adaptive reset-to-open recovery"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="enter the interactive hardware path; omitted means no device access",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="exclusive evidence path; default name includes the generated run-id",
    )
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
        mapper_derivation = derive_v94_mapper_probe_endpoint()
        plan = build_plan(
            binding, run_id, mapper_derivation=mapper_derivation
        )
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            print("[dry-run] no device was opened and no register was written")
            return 0

        tty_check = sys.stdin.isatty if stdin_isatty is None else stdin_isatty
        if not tty_check():
            raise MicroprobeError("--execute requires an interactive TTY; piping confirmation is refused")
        output = (
            DEFAULT_EVIDENCE_DIR / f"v94_rh56_thumb_rotation_microprobe_{run_id}.json"
            if args.output is None
            else args.output
        ).expanduser().resolve()
        if output.exists():
            raise MicroprobeError(f"evidence output already exists: {output}")

        print(json.dumps(plan, indent=2, sort_keys=True))
        print("\nBefore confirming, verify ALL conditions:")
        for requirement in plan["operator_requirements"]:
            print(f"  - {requirement}")
        phrase = f"EXECUTE-RH56-Q6-MICROPROBE {run_id}"
        answer = input_fn(f"\nType exactly:\n{phrase}\n> ")
        if answer.strip() != phrase:
            raise MicroprobeError("run-scoped interactive confirmation did not match")

        evidence = hardware_session(binding, run_id)
        _write_exclusive_json(output, evidence)
        print(f"[microprobe] result={evidence['result']} evidence={output}")
        cleanup = evidence.get("cleanup", {})
        if evidence.get("result") != "PASS":
            if not cleanup.get("double_disable_and_stationary_stop_verified", False):
                print(
                    "[STOP UNCONFIRMED] cut RH56 24 V immediately if motion remains",
                    file=sys.stderr,
                )
            return 130 if evidence.get("interrupted") else 2
        print(
            "[microprobe] PASS: action[7] -> manufacturer axis5 physical direction "
            "confirmed; this does not authorize closed-loop motion"
        )
        return 0
    except KeyboardInterrupt:
        print(
            "[interrupted][STOP STATE UNKNOWN] cut RH56 24 V if motion remains",
            file=sys.stderr,
        )
        return 130
    except (OSError, MicroprobeError, ValueError) as exc:
        print(f"[microprobe][refused] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
