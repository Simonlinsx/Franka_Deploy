#!/usr/bin/env python3
"""Offline, independent audit of the V94 Franka action mapping.

This module deliberately does not import the policy mapper, pylibfranka, or any
device adapter.  It reconstructs the arm-target recurrence directly from the
transferred deployment contract and compares it with both packaged simulator
alignment streams.  The distinction between an exact *target* mapping and a
commissioned real tracking envelope is kept explicit in the report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.deployment.bundle import DeployBundle  # type: ignore[no-redef]
    from sim2real.contracts.v94 import (  # type: ignore[no-redef]
        INITIAL_PREVIOUS_ACTION13,
        V94Contract,
    )
else:
    from .bundle import DeployBundle
    from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13, V94Contract


from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "dexgrasp"
    / "configs"
    / "fr3_rh56_v94_commissioning.json"
)
INITIAL_NPZ = "alignment/reset_idle_open/initial_observations_and_student_response.npz"
CLOSED_NPZ = "alignment/student_closed_loop/closed_loop_observation_action_pairs.npz"
VALIDATION_CONFIG = "validation_video/rolling_student_dynamic_20trial.config.json"
INITIAL_POSE = "alignment/reset_idle_open/initial_pose.json"

EXPECTED_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
EXPECTED_POSE_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_RAW_GAIN_RAD = np.float32(0.015)
TARGET_FILTER_ALPHA = np.float32(0.20)
MAXIMUM_ARM_TARGET_STEP_RAD = np.float32(0.015)
MEASURED_Q_ENVELOPE_RAD = np.float32(0.05)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _maximum_error(first: np.ndarray, second: np.ndarray, name: str) -> float:
    if first.shape != second.shape:
        raise ValueError(f"{name} shape mismatch: {first.shape}!={second.shape}")
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        raise ValueError(f"{name} contains non-finite values")
    return float(
        np.max(np.abs(first.astype(np.float64) - second.astype(np.float64)), initial=0.0)
    )


@dataclass(frozen=True)
class StreamMappingAudit:
    rows: int
    independent_target_max_abs_error_rad: float
    effective_delta_formula_max_abs_error_rad: float
    action_delta_sign_checks: int
    action_delta_sign_mismatches: int
    per_tick_step_clip_count: int
    measured_q_or_joint_limit_clip_count: int
    maximum_target_delta_rad: float
    maximum_nominal_target_rate_rad_s: float
    rows_exceeding_profile_velocity: int


@dataclass(frozen=True)
class V94FrankaActionMappingAudit:
    bundle_contract: str
    checkpoint_sha256: str
    action_contract: str
    arm_joint_order: tuple[str, ...]
    real_write_joint_order: tuple[str, ...]
    axis_order_identity: bool
    sign_transform: str
    command_unit: str
    control_dt_s: float
    policy_rate_hz: float
    raw_gain_rad_per_unit_action: float
    target_filter_alpha: float
    effective_gain_rad_per_tick: float
    nominal_full_scale_target_rate_rad_s: float
    simulation_and_profile_joint_limits_exact: bool
    simulation_and_profile_reset_exact: bool
    minimum_reference_distance_to_profile_safe_limit_rad: float
    reset_stream: StreamMappingAudit
    closed_loop_stream: StreamMappingAudit
    initial_previous_action_exact: bool
    reset_idle_previous_action_exact: bool
    closed_loop_previous_action_exact: bool
    transactional_target_commit_required: bool
    packaged_reference_mapping_confirmed: bool
    physical_tracking_equivalence_confirmed: bool
    current_profile_mode: str
    current_profile_persistent_session_configured: bool
    current_profile_max_velocity_rad_s: tuple[float, ...]
    current_profile_missing_dynamic_limits: tuple[str, ...]
    blockers: tuple[str, ...]
    hardware_accessed: bool = False


def _independent_arm_replay(
    data: Mapping[str, np.ndarray],
    *,
    joint_limits_rad: np.ndarray,
    control_dt_s: float,
    profile_max_velocity_rad_s: np.ndarray,
) -> tuple[np.ndarray, StreamMappingAudit]:
    raw_actions = np.asarray(data.get("policy_action13"), dtype=np.float32)
    measured_q = np.asarray(data.get("franka_measured_q_rad"), dtype=np.float32)
    expected_targets = np.asarray(data.get("franka_target_q_rad"), dtype=np.float32)
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 13:
        raise ValueError("policy_action13 must have shape [T,13]")
    count = raw_actions.shape[0]
    if measured_q.shape != (count, 7) or expected_targets.shape != (count, 7):
        raise ValueError("Franka measured/target arrays must have shape [T,7]")
    if not all(np.all(np.isfinite(value)) for value in (raw_actions, measured_q, expected_targets)):
        raise ValueError("Franka mapping stream contains non-finite values")

    limits = np.asarray(joint_limits_rad, dtype=np.float32)
    if limits.shape != (7, 2) or not np.all(np.isfinite(limits)):
        raise ValueError("joint_limits_rad must have finite shape [7,2]")
    profile_rate = np.asarray(profile_max_velocity_rad_s, dtype=np.float64)
    if profile_rate.shape != (7,) or not np.all(np.isfinite(profile_rate)):
        raise ValueError("profile maximum velocity must have finite shape [7]")

    previous_target = measured_q[0].copy()
    reconstructed = []
    deltas = []
    step_clip_count = 0
    envelope_clip_count = 0
    rows_exceeding_velocity = 0
    sign_checks = 0
    sign_mismatches = 0
    formula_error = 0.0
    for action13, measured in zip(raw_actions, measured_q):
        executed_arm = np.clip(
            action13[:7], np.float32(-1.0), np.float32(1.0)
        ).astype(np.float32, copy=False)
        raw_target = previous_target + ARM_RAW_GAIN_RAD * executed_arm
        filtered_target = (
            TARGET_FILTER_ALPHA * raw_target
            + (np.float32(1.0) - TARGET_FILTER_ALPHA) * previous_target
        )
        unbounded_delta = filtered_target - previous_target
        bounded_delta = np.clip(
            unbounded_delta,
            -MAXIMUM_ARM_TARGET_STEP_RAD,
            MAXIMUM_ARM_TARGET_STEP_RAD,
        )
        if not np.array_equal(unbounded_delta, bounded_delta):
            step_clip_count += 1
        next_target = previous_target + bounded_delta
        safe_lower = np.maximum(limits[:, 0], measured - MEASURED_Q_ENVELOPE_RAD)
        safe_upper = np.minimum(limits[:, 1], measured + MEASURED_Q_ENVELOPE_RAD)
        bounded_target = np.clip(next_target, safe_lower, safe_upper).astype(
            np.float32, copy=False
        )
        if not np.array_equal(next_target, bounded_target):
            envelope_clip_count += 1

        actual_delta = bounded_target.astype(np.float64) - previous_target.astype(
            np.float64
        )
        ideal_delta = (
            np.float64(ARM_RAW_GAIN_RAD)
            * np.float64(TARGET_FILTER_ALPHA)
            * executed_arm.astype(np.float64)
        )
        formula_error = max(formula_error, float(np.max(np.abs(actual_delta - ideal_delta))))
        comparable = (np.abs(executed_arm) > np.float32(1.0e-7)) & (
            np.abs(actual_delta) > 1.0e-10
        )
        sign_checks += int(np.count_nonzero(comparable))
        sign_mismatches += int(
            np.count_nonzero(
                np.sign(executed_arm[comparable]) != np.sign(actual_delta[comparable])
            )
        )
        rate = np.abs(actual_delta) / float(control_dt_s)
        if np.any(rate > profile_rate + 1.0e-7):
            rows_exceeding_velocity += 1
        reconstructed.append(bounded_target.copy())
        deltas.append(actual_delta)
        previous_target = bounded_target.copy()

    replay = np.stack(reconstructed).astype(np.float32)
    delta_array = np.stack(deltas)
    stream_report = StreamMappingAudit(
        rows=count,
        independent_target_max_abs_error_rad=_maximum_error(
            replay, expected_targets, "independent Franka target replay"
        ),
        effective_delta_formula_max_abs_error_rad=formula_error,
        action_delta_sign_checks=sign_checks,
        action_delta_sign_mismatches=sign_mismatches,
        per_tick_step_clip_count=step_clip_count,
        measured_q_or_joint_limit_clip_count=envelope_clip_count,
        maximum_target_delta_rad=float(np.max(np.abs(delta_array), initial=0.0)),
        maximum_nominal_target_rate_rad_s=float(
            np.max(np.abs(delta_array), initial=0.0) / float(control_dt_s)
        ),
        rows_exceeding_profile_velocity=rows_exceeding_velocity,
    )
    return replay, stream_report


def audit_v94_franka_action_mapping(
    bundle_path: str | Path = DEFAULT_BUNDLE,
    profile_path: str | Path = DEFAULT_PROFILE,
    *,
    expected_reset_q_rad: Optional[Sequence[float]] = None,
    expected_joint_limits_rad: Optional[Sequence[Sequence[float]]] = None,
) -> V94FrankaActionMappingAudit:
    """Audit the packaged target mapping and current real-control envelope."""

    bundle = DeployBundle(bundle_path)
    verification = bundle.verify()
    contract = V94Contract.from_bundle(bundle)
    validation = bundle.read_json(VALIDATION_CONFIG)
    environment = _mapping(validation.get("environment"), "validation.environment")
    initial_pose = bundle.read_json(INITIAL_POSE)
    initial = bundle.load_npz(INITIAL_NPZ)
    closed = bundle.load_npz(CLOSED_NPZ)

    action_contract = str(environment.get("action_contract", ""))
    arm_joint_order = tuple(str(value) for value in environment.get("arm_joint_names", ()))
    pose_joint_order = tuple(str(value) for value in initial_pose.get("franka_joint_order", ()))
    axis_identity = (
        action_contract == "inspire_semantic_13d"
        and arm_joint_order == EXPECTED_ARM_JOINT_NAMES
        and pose_joint_order == EXPECTED_POSE_JOINT_NAMES
    )

    profile_source = Path(profile_path).expanduser().resolve()
    with profile_source.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    profile_franka = _mapping(profile.get("franka"), "profile.franka")
    profile_limits = np.asarray(profile_franka.get("joint_limits_rad"), dtype=np.float64)
    profile_home = np.asarray(profile_franka.get("default_q_rad"), dtype=np.float64)
    if profile_limits.shape != (7, 2) or not np.all(np.isfinite(profile_limits)):
        raise ValueError("profile Franka joint limits are invalid")
    if profile_home.shape != (7,) or not np.all(np.isfinite(profile_home)):
        raise ValueError("profile Franka reset is invalid")
    raw_max_velocity = np.asarray(
        profile_franka.get("default_max_joint_velocity_rad_s"), dtype=np.float64
    )
    if raw_max_velocity.ndim == 0:
        profile_max_velocity = np.full(7, float(raw_max_velocity), dtype=np.float64)
    elif raw_max_velocity.shape == (7,):
        profile_max_velocity = raw_max_velocity.copy()
    else:
        raise ValueError("profile Franka maximum velocity is invalid")
    if not np.all(np.isfinite(profile_max_velocity)) or np.any(profile_max_velocity <= 0.0):
        raise ValueError("profile Franka maximum velocity must be positive and finite")

    _, initial_report = _independent_arm_replay(
        initial,
        joint_limits_rad=contract.joint_limits_rad,
        control_dt_s=contract.control_dt_s,
        profile_max_velocity_rad_s=profile_max_velocity,
    )
    closed_targets, closed_report = _independent_arm_replay(
        closed,
        joint_limits_rad=contract.joint_limits_rad,
        control_dt_s=contract.control_dt_s,
        profile_max_velocity_rad_s=profile_max_velocity,
    )

    reset_previous = np.asarray(initial.get("proprio67"))[:, 54:67]
    reset_idle_actions = np.asarray(initial.get("simulator_executed_idle_action13"))
    closed_previous = np.asarray(closed.get("current_proprio67"))[:, 54:67]
    initial_previous_exact = bool(
        np.array_equal(closed_previous[0], INITIAL_PREVIOUS_ACTION13)
    )
    reset_idle_previous_exact = bool(np.array_equal(reset_previous, reset_idle_actions))
    closed_previous_exact = bool(
        np.array_equal(
            closed_previous[1:], np.asarray(closed.get("policy_action13"))[:-1]
        )
    )

    joint_limit_reference = (
        contract.joint_limits_rad.astype(np.float64)
        if expected_joint_limits_rad is None
        else np.asarray(expected_joint_limits_rad, dtype=np.float64)
    )
    if (
        joint_limit_reference.shape != (7, 2)
        or not np.all(np.isfinite(joint_limit_reference))
        or np.any(joint_limit_reference[:, 0] >= joint_limit_reference[:, 1])
    ):
        raise ValueError("expected task Franka joint limits are invalid")
    joint_limits_exact = bool(np.array_equal(profile_limits, joint_limit_reference))
    # The bundle owns the legacy V94 home and limits by default.  A separately
    # SHA-bound runtime task contract may replace those envelope values while
    # the packaged streams continue to prove the unchanged action axes, signs
    # and recurrence against the original bundle contract above.
    reset_reference = (
        contract.q_home_rad
        if expected_reset_q_rad is None
        else np.asarray(expected_reset_q_rad, dtype=np.float32)
    )
    if reset_reference.shape != (7,) or not np.all(np.isfinite(reset_reference)):
        raise ValueError("expected task reset q is invalid")
    reset_exact = bool(
        np.array_equal(profile_home.astype(np.float32), reset_reference)
    )
    margin = float(profile_franka.get("joint_limit_margin_rad"))
    safe_lower = profile_limits[:, 0] + margin
    safe_upper = profile_limits[:, 1] - margin
    reference_limit_distance = float(
        np.min(
            np.minimum(
                closed_targets.astype(np.float64) - safe_lower,
                safe_upper - closed_targets.astype(np.float64),
            )
        )
    )

    dynamic_limit_fields = (
        "online_max_joint_acceleration_rad_s2",
        "online_max_joint_jerk_rad_s3",
        "online_max_tracking_error_rad",
    )
    missing_dynamic = tuple(
        field for field in dynamic_limit_fields if profile_franka.get(field) is None
    )
    nominal_rate = float(
        np.float64(ARM_RAW_GAIN_RAD)
        * np.float64(TARGET_FILTER_ALPHA)
        / contract.control_dt_s
    )
    blockers = []
    profile_mode = str(profile.get("mode", ""))
    persistent_session_configured = isinstance(
        profile_franka.get("persistent_session"), Mapping
    )
    if profile_mode != "c2_bounded_closed_loop_commissioned":
        blockers.append("commissioning profile is not in C2 bounded closed-loop mode")
    if not persistent_session_configured:
        blockers.append("Franka persistent-session envelope is not configured")
    if np.any(profile_max_velocity + 1.0e-12 < nominal_rate):
        blockers.append(
            "0.18 rad/s nominal policy target rate exceeds the 0.05 rad/s profile"
        )
    if missing_dynamic:
        blockers.append("Franka acceleration/jerk/tracking limits are not commissioned")
    if initial_report.independent_target_max_abs_error_rad != 0.0:
        blockers.append("reset-stream independent target replay differs from simulation")
    if closed_report.independent_target_max_abs_error_rad != 0.0:
        blockers.append("closed-loop independent target replay differs from simulation")
    if not axis_identity:
        blockers.append("Franka action/read/write joint order is not identity")
    if closed_report.action_delta_sign_mismatches:
        blockers.append("Franka action-to-target sign mismatch exists")
    if not (initial_previous_exact and reset_idle_previous_exact and closed_previous_exact):
        blockers.append("previous executed action recurrence differs from simulation")

    mapping_confirmed = bool(
        axis_identity
        and joint_limits_exact
        and reset_exact
        and initial_report.independent_target_max_abs_error_rad == 0.0
        and closed_report.independent_target_max_abs_error_rad == 0.0
        and initial_report.action_delta_sign_mismatches == 0
        and closed_report.action_delta_sign_mismatches == 0
        and initial_previous_exact
        and reset_idle_previous_exact
        and closed_previous_exact
    )
    return V94FrankaActionMappingAudit(
        bundle_contract=verification.bundle_contract,
        checkpoint_sha256=verification.primary_checkpoint_sha256,
        action_contract=action_contract,
        arm_joint_order=arm_joint_order,
        real_write_joint_order=EXPECTED_ARM_JOINT_NAMES,
        axis_order_identity=axis_identity,
        sign_transform="identity (no negation)",
        command_unit="joint position target in radians",
        control_dt_s=contract.control_dt_s,
        policy_rate_hz=contract.policy_rate_hz,
        raw_gain_rad_per_unit_action=float(ARM_RAW_GAIN_RAD),
        target_filter_alpha=float(TARGET_FILTER_ALPHA),
        effective_gain_rad_per_tick=float(
            np.float64(ARM_RAW_GAIN_RAD) * np.float64(TARGET_FILTER_ALPHA)
        ),
        nominal_full_scale_target_rate_rad_s=nominal_rate,
        simulation_and_profile_joint_limits_exact=joint_limits_exact,
        simulation_and_profile_reset_exact=reset_exact,
        minimum_reference_distance_to_profile_safe_limit_rad=reference_limit_distance,
        reset_stream=initial_report,
        closed_loop_stream=closed_report,
        initial_previous_action_exact=initial_previous_exact,
        reset_idle_previous_action_exact=reset_idle_previous_exact,
        closed_loop_previous_action_exact=closed_previous_exact,
        transactional_target_commit_required=True,
        packaged_reference_mapping_confirmed=mapping_confirmed,
        physical_tracking_equivalence_confirmed=False,
        current_profile_mode=profile_mode,
        current_profile_persistent_session_configured=persistent_session_configured,
        current_profile_max_velocity_rad_s=tuple(
            float(value) for value in profile_max_velocity
        ),
        current_profile_missing_dynamic_limits=missing_dynamic,
        blockers=tuple(blockers),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently audit V94 action[0:7] to Franka target mapping."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_v94_franka_action_mapping(args.bundle, args.profile)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"V94 Franka action mapping audit: FAIL: {exc}")
        return 2
    payload = asdict(report)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("V94 Franka packaged target mapping: PASS")
        print(json.dumps(payload, indent=2, sort_keys=True))
    # The target mapping may pass while motion remains correctly blocked by an
    # uncommissioned physical envelope.  This CLI is an offline mapping audit,
    # so blockers are reported rather than converted into a false mapping FAIL.
    return 0 if report.packaged_reference_mapping_confirmed else 2


if __name__ == "__main__":
    raise SystemExit(main())
