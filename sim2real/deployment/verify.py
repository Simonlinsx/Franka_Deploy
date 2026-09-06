#!/usr/bin/env python3
"""Offline acceptance test for the transferred V94 deployment bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.deployment.bundle import (
        MAX_CHECKPOINT_BYTES,
        DeployBundle,
        load_checkpoint_safely,
    )
    from sim2real.policy import (
        QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
        RollingStudentPolicy,
    )
    from sim2real.policy.rate_mode import resolve_policy_rate_mode
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import V94Contract
    from sim2real.observation.model import (
        PolicyHistory,
        PolicyPointFrame,
        Proprio67Builder,
        pose_from_position_quaternion_wxyz,
    )
else:
    from .bundle import (
        MAX_CHECKPOINT_BYTES,
        DeployBundle,
        load_checkpoint_safely,
    )
    from sim2real.policy import (
        QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
        RollingStudentPolicy,
    )
    from sim2real.policy.rate_mode import resolve_policy_rate_mode
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import V94Contract
    from sim2real.observation.model import (
        PolicyHistory,
        PolicyPointFrame,
        Proprio67Builder,
        pose_from_position_quaternion_wxyz,
    )


from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE
INITIAL_NPZ = "alignment/reset_idle_open/initial_observations_and_student_response.npz"
CLOSED_NPZ = "alignment/student_closed_loop/closed_loop_observation_action_pairs.npz"


@dataclass(frozen=True)
class V94VerificationReport:
    bundle_contract: str
    bundle_status: str
    checked_hashes: int
    checkpoint_sha256: str
    checkpoint_iteration: int
    point_feature_mode: str
    point_feature_dim: int
    model_parameters: int
    initial_action_max_abs_error: float
    initial_privileged_max_abs_error: float
    initial_hold_logit_max_abs_error: float
    closed_cpu_replay_max_abs_error: float
    closed_gpu_action_max_abs_error: float
    initial_target_max_abs_error: float
    closed_target_max_abs_error: float
    proprio_layout_max_abs_error: float
    policy_rate_hz: float
    nominal_max_arm_velocity_rad_s: float
    hardware_writes: bool = False


@dataclass(frozen=True)
class V94CheckpointCompatibilityReport:
    """Offline proof that an external checkpoint fits the V94 I/O contract.

    The original bundle's golden actions belong to its manifest checkpoint and
    therefore cannot be an equality oracle for different weights.  An override
    instead has to pass the same safe decoder, architecture/metadata checks,
    control period, normalization shapes, and finite forward passes over both
    packaged observation sets.
    """

    bundle_contract: str
    base_bundle_checkpoint_sha256: str
    checkpoint_sha256: str
    checkpoint_iteration_or_epoch: int
    checkpoint_format: str
    checkpoint_task: str
    point_feature_mode: str
    point_feature_dim: int
    model_parameters: int
    control_dt_s: float
    initial_observation_actions: int
    closed_observation_actions: int
    initial_action_max_abs: float
    closed_action_max_abs: float
    initial_action_max_abs_delta_from_bundle_checkpoint: Optional[float]
    closed_action_max_abs_delta_from_bundle_checkpoint: Optional[float]
    normalization_identical_to_bundle_checkpoint: bool
    contract_compatible: bool = True
    bundle_golden_action_equality_applies: bool = False
    hardware_writes: bool = False


def _maximum_error(actual: np.ndarray, expected: np.ndarray, name: str) -> float:
    first = np.asarray(actual)
    second = np.asarray(expected)
    if first.shape != second.shape:
        raise ValueError(f"{name} shape mismatch: {first.shape}!={second.shape}")
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        raise ValueError(f"{name} contains non-finite values")
    return float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))


def _require_at_most(value: float, limit: float, name: str) -> None:
    if not np.isfinite(value) or value > limit:
        raise ValueError(f"{name}={value:.9g} exceeds tolerance {limit:.9g}")


def _run_initial(policy: RollingStudentPolicy, data: dict) -> tuple[np.ndarray, ...]:
    history = PolicyHistory(point_feature_dim=policy.point_feature_dim)
    actions = []
    privileged = []
    holds = []
    count = int(data["logical_step"].shape[0])
    for index in range(count):
        frame = PolicyPointFrame(
            xyzrgb_palm=data["pointcloud_xyzrgb_palm"][
                index, :, : policy.point_feature_dim
            ],
            valid=data["pointcloud_valid"][index],
            captured_at_s=float(data["time_s"][index]),
            frame_id=int(data["logical_step"][index]),
            source_valid_points=int(np.sum(data["pointcloud_valid"][index])),
            status="packaged_alignment",
        )
        point_history, valid_history, proprio_history = history.append(
            frame, data["proprio67"][index]
        )
        output = policy.act(point_history, valid_history, proprio_history)
        actions.append(output.action13)
        privileged.append(output.predicted_privileged32)
        holds.append(output.predicted_hold_logit)
    return np.stack(actions), np.stack(privileged), np.asarray(holds)


def _run_closed(policy: RollingStudentPolicy, data: dict) -> np.ndarray:
    output = []
    for index in range(data["episode_step"].shape[0]):
        if not np.array_equal(
            data["current_pointcloud_xyzrgb_palm"][index],
            data["pointcloud_history_xyzrgb_palm"][index, -1],
        ):
            raise ValueError("closed-loop current point cloud is not history[-1]")
        if not np.array_equal(
            data["current_proprio67"][index], data["proprio_history67"][index, -1]
        ):
            raise ValueError("closed-loop current proprio is not history[-1]")
        result = policy.act(
            data["pointcloud_history_xyzrgb_palm"][
                index, :, :, : policy.point_feature_dim
            ],
            data["pointcloud_valid_history"][index],
            data["proprio_history67"][index],
        )
        output.append(result.action13)
    previous = data["proprio_history67"][1:, -1, 54:67]
    if not np.array_equal(previous, data["policy_action13"][:-1]):
        raise ValueError(
            "closed-loop proprio previous-action slice does not match prior action"
        )
    return np.stack(output)


def _run_extended_smoke(
    policy: RollingStudentPolicy,
    legacy_initial: dict,
    *,
    qd_g015: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Exercise one extended contract without a false golden action.

    The transferred deployment bundle contains authoritative camera/palm and
    67-D robot examples, but its golden actions belong to the original 4x67
    checkpoint.  Reuse two physically grounded frames, append bounded
    controller state when required, and prove deterministic finite inference
    for both the reset bootstrap and one history shift.  q_d-g015 starts the
    previous-action channel at zeros and chains the first accepted action into
    the shifted frame, matching its reset/update contract.
    """

    if policy.history_length not in (8, 16) or policy.proprio_dim not in (67, 96):
        raise ValueError("extended smoke requires an 8/16x67/96 contract")
    if not qd_g015 and policy.proprio_dim != 96:
        raise ValueError("legacy extended smoke requires the V258 8x96 contract")
    points = np.asarray(
        legacy_initial["pointcloud_xyzrgb_palm"][:2, :, : policy.point_feature_dim],
        dtype=np.float32,
    )
    valid = np.asarray(legacy_initial["pointcloud_valid"][:2], dtype=np.float32)
    base67 = np.asarray(legacy_initial["proprio67"][:2], dtype=np.float32).copy()
    if points.shape != (2, 128, policy.point_feature_dim):
        raise ValueError("packaged point examples cannot seed extended smoke")
    if valid.shape != (2, 128) or base67.shape != (2, 67):
        raise ValueError("packaged robot examples cannot seed extended smoke")

    if qd_g015:
        base67[0, 54:67] = policy.initial_previous_action13

    controller0 = None
    controller1 = None
    if policy.proprio_dim == 96:
        controller0 = np.zeros((29,), dtype=np.float32)
        controller0[28] = np.float32(1.0)
        controller1 = controller0.copy()
        controller1[0:7] = np.float32(0.25)
        controller1[7:14] = np.float32(-0.02)
        controller1[14:21] = np.float32(0.05)
        controller1[21:28] = np.float32(-0.10)
        proprio0 = np.concatenate((base67[0], controller0)).astype(np.float32)
    else:
        proprio0 = base67[0]

    point_history0 = np.repeat(
        points[0][None, :, :], policy.history_length, axis=0
    )
    valid_history0 = np.repeat(
        valid[0][None, :], policy.history_length, axis=0
    )
    proprio_history0 = np.repeat(
        proprio0[None, :], policy.history_length, axis=0
    )
    initial = policy.act(point_history0, valid_history0, proprio_history0).action13
    repeated = policy.act(point_history0, valid_history0, proprio_history0).action13
    if not np.array_equal(initial, repeated):
        raise ValueError("extended checkpoint deterministic-mean inference is not exact")

    if qd_g015:
        base67[1, 54:67] = initial
    if controller1 is not None:
        proprio1 = np.concatenate((base67[1], controller1)).astype(np.float32)
    else:
        proprio1 = base67[1]

    shifted_points = np.concatenate((point_history0[1:], points[1][None]), axis=0)
    shifted_valid = np.concatenate((valid_history0[1:], valid[1][None]), axis=0)
    shifted_proprio = np.concatenate(
        (proprio_history0[1:], proprio1[None]), axis=0
    )
    shifted = policy.act(shifted_points, shifted_valid, shifted_proprio).action13
    actions = np.stack((initial, shifted)).astype(np.float32)
    if not np.all(np.isfinite(actions)) or np.max(np.abs(actions)) > 1.0 + 1.0e-7:
        raise ValueError(
            "extended checkpoint smoke action is non-finite or outside [-1,1]"
        )
    return actions[:1], actions[1:]


def _run_v258_extended_smoke(
    policy: RollingStudentPolicy,
    legacy_initial: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for the legacy V258 8x96 smoke."""

    return _run_extended_smoke(policy, legacy_initial, qd_g015=False)


def _require_exact_numeric(
    actual: object,
    expected: float,
    name: str,
    *,
    contract_name: str,
) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{contract_name} checkpoint {name} is invalid") from exc
    if not np.isfinite(value) or not np.isclose(
        value, expected, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError(
            f"{contract_name} checkpoint {name} mismatch: "
            f"expected={expected}, actual={value}"
        )


def _require_extended_checkpoint_common(
    checkpoint,
    policy: RollingStudentPolicy,
    *,
    contract_name: str,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    metadata = checkpoint.metadata
    controller = metadata.get("action_controller")
    arm = controller.get("arm") if isinstance(controller, Mapping) else None
    hand = controller.get("hand") if isinstance(controller, Mapping) else None
    expected_scalars = (
        (
            metadata.get("history"),
            policy.history_length,
            "metadata history",
        ),
        (metadata.get("proprio_dim"), policy.proprio_dim, "metadata proprio_dim"),
        (
            metadata.get("deployable_controller_state"),
            policy.proprio_dim == 96,
            "deployable_controller_state",
        ),
        (
            checkpoint.spec.get("action_distribution"),
            "state_dependent_gaussian",
            "action_distribution",
        ),
        (policy.point_feature_dim, 3, "point_feature_dim"),
        (policy.compact_privileged_dim, 33, "compact_privileged_dim"),
    )
    for actual, expected, name in expected_scalars:
        if actual != expected:
            raise ValueError(
                f"{contract_name} checkpoint {name} mismatch: "
                f"expected={expected!r}, actual={actual!r}"
            )
    if not isinstance(controller, Mapping) or not isinstance(arm, Mapping) or not isinstance(hand, Mapping):
        raise ValueError(
            f"{contract_name} checkpoint is missing the action-controller sections"
        )
    if set(checkpoint.normalization) != {
        "pointcloud_mean",
        "pointcloud_std",
        "proprio_mean",
        "proprio_std",
    }:
        raise ValueError(
            f"{contract_name} checkpoint normalization keys differ from contract"
        )
    if controller.get("target_update_clock") != "policy":
        raise ValueError(
            f"{contract_name} checkpoint target update clock must be policy"
        )
    for actual, expected, name in (
        (checkpoint.spec.get("action_log_std_min"), -5.0, "action log-std min"),
        (checkpoint.spec.get("action_log_std_max"), -0.5, "action log-std max"),
        (controller.get("control_dt_s"), 0.05, "controller dt"),
        (controller.get("policy_frequency_hz"), 20.0, "controller frequency"),
        (controller.get("physics_hold_substeps"), 6.0, "physics hold substeps"),
        (hand.get("moving_average"), 0.737856, "hand moving average"),
        (
            hand.get("max_target_delta_rad_per_policy_step"),
            0.3,
            "hand target delta",
        ),
    ):
        _require_exact_numeric(
            actual,
            expected,
            name,
            contract_name=contract_name,
        )
    return controller, arm, hand


def _validate_v258_checkpoint_contract(
    checkpoint,
    policy: RollingStudentPolicy,
) -> None:
    if (policy.history_length, policy.proprio_dim) not in ((8, 96), (16, 96)):
        raise ValueError(
            "legacy extended checkpoint requires history=8/16 and proprio_dim=96"
        )
    _controller, arm, _hand = _require_extended_checkpoint_common(
        checkpoint,
        policy,
        contract_name="V258",
    )
    for actual, expected, name in (
        (arm.get("delta_scale_rad_per_policy_step"), 0.045, "arm delta scale"),
        (arm.get("moving_average"), 0.4, "arm moving average"),
        (
            arm.get("max_target_delta_rad_per_policy_step"),
            0.045,
            "arm target delta",
        ),
        (arm.get("tracking_error_limit_rad"), 0.05, "arm tracking envelope"),
    ):
        _require_exact_numeric(actual, expected, name, contract_name="V258")


def _validate_qd_g015_checkpoint_contract(
    checkpoint,
    policy: RollingStudentPolicy,
) -> None:
    if policy.history_length != 8 or policy.proprio_dim not in (67, 96):
        raise ValueError(
            "q_d-g015 checkpoint requires history=8 and proprio_dim=67 or 96"
        )
    controller, arm, _hand = _require_extended_checkpoint_common(
        checkpoint,
        policy,
        contract_name="q_d-g015",
    )
    if arm.get("incremental_reference") not in {
        "shaper_q_d",
        "current_shaper_q_d",
    }:
        raise ValueError(
            "q_d-g015 checkpoint arm incremental reference must be shaper_q_d"
        )
    for actual, expected, name in (
        (arm.get("delta_scale_rad_per_policy_step"), 0.15, "arm raw delta scale"),
        (arm.get("moving_average"), 0.4, "arm blend alpha"),
        (
            arm.get("effective_delta_scale_after_ema_rad_per_policy_step"),
            0.06,
            "arm effective delta scale",
        ),
        (
            arm.get("max_target_delta_rad_per_policy_step"),
            0.0,
            "arm explicit target delta limiter",
        ),
        (
            arm.get("tracking_error_limit_rad"),
            0.0,
            "arm measured-q target envelope",
        ),
    ):
        _require_exact_numeric(actual, expected, name, contract_name="q_d-g015")

    shaper = controller.get("franka_command_shaper")
    if not isinstance(shaper, Mapping) or shaper.get("enabled") is not True:
        raise ValueError("q_d-g015 checkpoint requires the 1 kHz Franka shaper")
    if shaper.get("mode") != "libfranka_joint_position_interpolated":
        raise ValueError("q_d-g015 checkpoint Franka shaper mode mismatch")
    if shaper.get("servo_ticks_per_physics_step") != [8, 8, 9]:
        raise ValueError("q_d-g015 checkpoint virtual-servo schedule mismatch")
    for actual, expected, name in (
        (shaper.get("servo_dt_s"), 0.001, "shaper servo dt"),
        (
            shaper.get("lowpass_cutoff_frequency_hz"),
            100.0,
            "shaper low-pass cutoff",
        ),
        (
            shaper.get("interpolator_natural_frequency_hz"),
            6.0,
            "shaper natural frequency",
        ),
        (
            shaper.get("interpolator_damping_ratio"),
            1.0,
            "shaper damping ratio",
        ),
        (shaper.get("velocity_limit_rad_s"), 0.5, "shaper velocity limit"),
        (
            shaper.get("acceleration_limit_rad_s2"),
            5.0,
            "shaper acceleration limit",
        ),
        (shaper.get("jerk_limit_rad_s3"), 250.0, "shaper jerk limit"),
    ):
        _require_exact_numeric(actual, expected, name, contract_name="q_d-g015")

    checkpoint_contract = checkpoint.metadata.get("checkpoint_controller_contract")
    if isinstance(checkpoint_contract, Mapping) and checkpoint_contract.get("enabled") is not True:
        raise ValueError("q_d-g015 checkpoint controller provenance is disabled")


def _verify_mapping(data: dict, contract: V94Contract) -> float:
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=data["franka_measured_q_rad"][0],
        initial_hand_target_q_policy_order_rad=data["rh56_virtual_q_policy_order_rad"][
            0
        ],
        joint_limits_rad=contract.joint_limits_rad,
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    arm_targets = []
    hand_targets = []
    registers = []
    for index, action in enumerate(data["policy_action13"]):
        mapped = mapper.map(action, measured_q_rad=data["franka_measured_q_rad"][index])
        arm_targets.append(mapped.franka_target_q_rad)
        hand_targets.append(mapped.rh56_target_q_policy_order_rad)
        registers.append(mapped.rh56_angle_set_register_order)
    errors = (
        _maximum_error(
            np.stack(arm_targets), data["franka_target_q_rad"], "Franka targets"
        ),
        _maximum_error(
            np.stack(hand_targets),
            data["rh56_target_q_policy_order_rad"],
            "RH56 targets",
        ),
        _maximum_error(
            np.stack(registers),
            data["rh56_angle_set_register_order"],
            "RH56 registers",
        ),
    )
    maximum = max(errors)
    _require_at_most(maximum, 5.0e-7, "action target replay error")
    return maximum


def _verify_proprio(data: dict, contract: V94Contract) -> float:
    index = 0
    expected = data["proprio67"][index]
    palm = pose_from_position_quaternion_wxyz(
        contract.reference_palm_position_base_m,
        contract.reference_palm_quaternion_base_wxyz,
    )
    builder = Proprio67Builder(
        q_home_rad=contract.q_home_rad,
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    actual = builder.build(
        franka_q_rad=data["franka_measured_q_rad"][index],
        franka_dq_rad_s=data["franka_measured_dq_rad_s"][index],
        rh56_virtual_q_policy_order_rad=data["rh56_virtual_q_policy_order_rad"][index],
        rh56_virtual_dq_policy_order_rad_s=data["rh56_virtual_dq_policy_order_rad_s"][
            index
        ],
        T_base_palm=palm,
        palm_linear_velocity_base_m_s=expected[33:36] / np.float32(0.1),
        palm_angular_velocity_base_rad_s=expected[36:39] / np.float32(0.1),
        fingertip_positions_base_m=contract.reference_fingertip_positions_base_m,
        previous_executed_action13=expected[54:67],
    )
    error = _maximum_error(actual, expected, "proprio67 layout")
    _require_at_most(error, 5.0e-7, "proprio67 layout error")
    return error


def verify_v94_bundle(path: str | Path = DEFAULT_BUNDLE) -> V94VerificationReport:
    bundle = DeployBundle(path)
    bundle_report = bundle.verify()
    contract = V94Contract.from_bundle(bundle)
    checkpoint = load_checkpoint_safely(bundle.checkpoint_bytes())
    policy = RollingStudentPolicy(checkpoint)
    parameter_count = int(
        sum(value.size for value in checkpoint.model_state_dict.values())
    )
    if parameter_count != 2_522_192:
        raise ValueError(f"unexpected V94 parameter count {parameter_count}")
    if not np.isclose(
        float(checkpoint.metadata.get("control_dt", np.nan)),
        contract.control_dt_s,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError("checkpoint and deployment control periods disagree")

    initial = bundle.load_npz(INITIAL_NPZ)
    closed = bundle.load_npz(CLOSED_NPZ)
    actions, privileged, holds = _run_initial(policy, initial)
    initial_action_error = _maximum_error(
        actions, initial["policy_action13"], "initial action replay"
    )
    initial_privileged_error = _maximum_error(
        privileged,
        initial["predicted_privileged32"],
        "initial privileged replay",
    )
    initial_hold_error = _maximum_error(
        holds, initial["predicted_hold_logit"], "initial hold replay"
    )
    _require_at_most(initial_action_error, 2.0e-6, "initial action replay error")
    # NumPy/BLAS reduction order differs slightly from the packaged CUDA/Torch
    # reference on the unbounded privileged auxiliary head.
    _require_at_most(
        initial_privileged_error, 1.0e-5, "initial privileged replay error"
    )
    _require_at_most(initial_hold_error, 1.0e-6, "initial hold replay error")

    closed_actions = _run_closed(policy, closed)
    closed_replay_error = _maximum_error(
        closed_actions,
        closed["checkpoint_replayed_action13"],
        "closed CPU replay",
    )
    closed_gpu_error = _maximum_error(
        closed_actions, closed["policy_action13"], "closed GPU executed action"
    )
    _require_at_most(closed_replay_error, 2.0e-6, "closed CPU replay error")
    _require_at_most(closed_gpu_error, 3.5e-4, "closed GPU action error")

    initial_target_error = _verify_mapping(initial, contract)
    closed_target_error = _verify_mapping(closed, contract)
    proprio_error = _verify_proprio(initial, contract)
    return V94VerificationReport(
        bundle_contract=bundle_report.bundle_contract,
        bundle_status=contract.bundle_status,
        checked_hashes=bundle_report.checked_files,
        checkpoint_sha256=bundle_report.primary_checkpoint_sha256,
        checkpoint_iteration=checkpoint.iteration,
        point_feature_mode=policy.point_feature_mode,
        point_feature_dim=policy.point_feature_dim,
        model_parameters=parameter_count,
        initial_action_max_abs_error=initial_action_error,
        initial_privileged_max_abs_error=initial_privileged_error,
        initial_hold_logit_max_abs_error=initial_hold_error,
        closed_cpu_replay_max_abs_error=closed_replay_error,
        closed_gpu_action_max_abs_error=closed_gpu_error,
        initial_target_max_abs_error=initial_target_error,
        closed_target_max_abs_error=closed_target_error,
        proprio_layout_max_abs_error=proprio_error,
        policy_rate_hz=contract.policy_rate_hz,
        nominal_max_arm_velocity_rad_s=contract.maximum_nominal_arm_velocity_rad_s,
    )


def verify_v94_checkpoint_payload(
    bundle_path: str | Path,
    checkpoint_payload: bytes,
    *,
    expected_control_dt_s: Optional[float] = None,
) -> V94CheckpointCompatibilityReport:
    """Validate pinned external weights against one selected runtime rate."""

    # Retain the complete manifest/hash/golden verification as the authority
    # for observations, action mapping, reset, scene and calibration.
    base_report = verify_v94_bundle(bundle_path)
    bundle = DeployBundle(bundle_path)
    contract = V94Contract.from_bundle(bundle)
    expected_control_dt = (
        contract.control_dt_s
        if expected_control_dt_s is None
        else float(expected_control_dt_s)
    )
    if not np.isfinite(expected_control_dt) or expected_control_dt <= 0.0:
        raise ValueError("expected checkpoint control period must be positive")
    checkpoint = load_checkpoint_safely(checkpoint_payload)
    policy = RollingStudentPolicy(checkpoint)
    parameter_count = int(
        sum(value.size for value in checkpoint.model_state_dict.values())
    )
    if parameter_count != policy.expected_model_parameter_count:
        raise ValueError(
            "unexpected V94-compatible checkpoint parameter count "
            f"{parameter_count}; expected {policy.expected_model_parameter_count} "
            f"for {policy.point_feature_mode.upper()} mode"
        )
    for name, value in checkpoint.model_state_dict.items():
        if not np.all(np.isfinite(np.asarray(value))):
            raise ValueError(f"external checkpoint weight {name} contains non-finite values")
    for name, value in checkpoint.normalization.items():
        if not np.all(np.isfinite(np.asarray(value))):
            raise ValueError(f"external checkpoint normalization {name} is non-finite")
    control_dt = float(checkpoint.metadata.get("control_dt", np.nan))
    if not np.isclose(
        control_dt,
        expected_control_dt,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "checkpoint control period disagrees with the selected runtime "
            f"mode: checkpoint={control_dt!r}s expected={expected_control_dt!r}s"
        )

    initial = bundle.load_npz(INITIAL_NPZ)
    if policy.history_length in (8, 16) and policy.proprio_dim in (67, 96):
        qd_g015 = (
            policy.action_controller.contract_id
            == QD_G015_ACTION_CONTROLLER_CONTRACT_ID
        )
        if qd_g015:
            _validate_qd_g015_checkpoint_contract(checkpoint, policy)
        else:
            _validate_v258_checkpoint_contract(checkpoint, policy)
        initial_actions, closed_actions = _run_extended_smoke(
            policy,
            initial,
            qd_g015=qd_g015,
        )
        initial_abs = float(np.max(np.abs(initial_actions.astype(np.float64))))
        closed_abs = float(np.max(np.abs(closed_actions.astype(np.float64))))
        return V94CheckpointCompatibilityReport(
            bundle_contract=base_report.bundle_contract,
            base_bundle_checkpoint_sha256=base_report.checkpoint_sha256,
            checkpoint_sha256=hashlib.sha256(checkpoint_payload).hexdigest(),
            checkpoint_iteration_or_epoch=checkpoint.iteration,
            checkpoint_format=("ppo" if checkpoint.ppo_state_dict else "student_pretrain"),
            checkpoint_task=str(checkpoint.metadata.get("task", "")),
            point_feature_mode=policy.point_feature_mode,
            point_feature_dim=policy.point_feature_dim,
            model_parameters=parameter_count,
            control_dt_s=control_dt,
            initial_observation_actions=int(initial_actions.shape[0]),
            closed_observation_actions=int(closed_actions.shape[0]),
            initial_action_max_abs=initial_abs,
            closed_action_max_abs=closed_abs,
            initial_action_max_abs_delta_from_bundle_checkpoint=None,
            closed_action_max_abs_delta_from_bundle_checkpoint=None,
            normalization_identical_to_bundle_checkpoint=False,
        )

    closed = bundle.load_npz(CLOSED_NPZ)
    initial_actions, privileged, hold_logits = _run_initial(policy, initial)
    closed_actions = _run_closed(policy, closed)
    for name, value in (
        ("initial actions", initial_actions),
        ("initial privileged predictions", privileged),
        ("initial hold logits", hold_logits),
        ("closed actions", closed_actions),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"external checkpoint {name} contain non-finite values")
    initial_abs = float(np.max(np.abs(initial_actions.astype(np.float64))))
    closed_abs = float(np.max(np.abs(closed_actions.astype(np.float64))))
    if initial_abs > 1.0 + 1.0e-7 or closed_abs > 1.0 + 1.0e-7:
        raise ValueError("external checkpoint action exceeds normalized [-1,1]")

    base_checkpoint = load_checkpoint_safely(bundle.checkpoint_bytes())
    if set(checkpoint.model_state_dict) != set(base_checkpoint.model_state_dict):
        raise ValueError(
            "external checkpoint model keys differ from the bundle architecture"
        )
    checkpoint_variant_shapes = {
        "point_encoder.0.weight",
        "privileged_head.2.weight",
        "privileged_head.2.bias",
        "action_head.0.weight",
    }
    for name, value in checkpoint.model_state_dict.items():
        base_value = np.asarray(base_checkpoint.model_state_dict[name])
        candidate = np.asarray(value)
        if name not in checkpoint_variant_shapes and candidate.shape != base_value.shape:
            raise ValueError(
                f"external checkpoint weight shape differs for {name}: "
                f"{candidate.shape}!={base_value.shape}"
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError(
                f"external checkpoint weight {name} contains non-finite values"
            )
    if set(checkpoint.normalization) != set(base_checkpoint.normalization):
        raise ValueError(
            "external checkpoint normalization keys differ from the bundle"
        )
    for name, value in checkpoint.normalization.items():
        base_value = np.asarray(base_checkpoint.normalization[name])
        candidate = np.asarray(value)
        if (
            name not in ("pointcloud_mean", "pointcloud_std")
            and candidate.shape != base_value.shape
        ):
            raise ValueError(
                f"external checkpoint normalization shape differs for {name}: "
                f"{candidate.shape}!={base_value.shape}"
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError(f"external checkpoint normalization {name} is non-finite")
    normalization_identical = policy.point_feature_dim == 6 and all(
        np.array_equal(
            np.asarray(checkpoint.normalization[name]),
            np.asarray(base_checkpoint.normalization[name]),
        )
        for name in checkpoint.normalization
    )
    return V94CheckpointCompatibilityReport(
        bundle_contract=base_report.bundle_contract,
        base_bundle_checkpoint_sha256=base_report.checkpoint_sha256,
        checkpoint_sha256=hashlib.sha256(checkpoint_payload).hexdigest(),
        checkpoint_iteration_or_epoch=checkpoint.iteration,
        checkpoint_format=("ppo" if checkpoint.ppo_state_dict else "student_pretrain"),
        checkpoint_task=str(checkpoint.metadata.get("task", "")),
        point_feature_mode=policy.point_feature_mode,
        point_feature_dim=policy.point_feature_dim,
        model_parameters=parameter_count,
        control_dt_s=control_dt,
        initial_observation_actions=int(initial_actions.shape[0]),
        closed_observation_actions=int(closed_actions.shape[0]),
        initial_action_max_abs=initial_abs,
        closed_action_max_abs=closed_abs,
        initial_action_max_abs_delta_from_bundle_checkpoint=_maximum_error(
            initial_actions,
            initial["policy_action13"],
            "external checkpoint initial action comparison",
        ),
        closed_action_max_abs_delta_from_bundle_checkpoint=_maximum_error(
            closed_actions,
            closed["checkpoint_replayed_action13"],
            "external checkpoint closed action comparison",
        ),
        normalization_identical_to_bundle_checkpoint=normalization_identical,
    )


def verify_v94_checkpoint_override(
    bundle_path: str | Path,
    checkpoint_path: str | Path,
    *,
    expected_control_dt_s: Optional[float] = None,
) -> V94CheckpointCompatibilityReport:
    """Read one external checkpoint once and validate it entirely offline."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    size = path.stat().st_size
    if size > MAX_CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint exceeds {MAX_CHECKPOINT_BYTES} byte safety limit")
    payload = path.read_bytes()
    if len(payload) != size:
        raise ValueError("checkpoint changed while it was being read")
    return verify_v94_checkpoint_payload(
        bundle_path,
        payload,
        expected_control_dt_s=expected_control_dt_s,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify V94 hashes, checkpoint forward pass, and action mapping offline."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "optional external student/PPO checkpoint to validate against the "
            "bundle observation/action contract"
        ),
    )
    parser.add_argument(
        "--policy-rate-hz",
        default="60",
        metavar="{20,60}",
        help="expected checkpoint control rate (default: 60)",
    )
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy_mode = resolve_policy_rate_mode(args.policy_rate_hz)
        if args.checkpoint is None and policy_mode.policy_rate_hz != 60.0:
            raise ValueError(
                "20 Hz verification requires an external --checkpoint; "
                "the transferred bundle primary is 60 Hz"
            )
        report = (
            verify_v94_bundle(args.bundle)
            if args.checkpoint is None
            else verify_v94_checkpoint_override(
                args.bundle,
                args.checkpoint,
                expected_control_dt_s=policy_mode.control_dt_s,
            )
        )
        payload = asdict(report)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(
                "V94 offline checkpoint compatibility: PASS"
                if args.checkpoint is not None
                else "V94 offline deployment alignment: PASS"
            )
            for name, value in payload.items():
                print(f"  {name}: {value}")
            print("  hardware: untouched (offline verifier only)")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"V94 offline deployment alignment: FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
