#!/usr/bin/env python3
"""Identify FR3 end-effector mass and CoM from static multi-pose data.

The operator manually guides the arm between poses.  This program only creates
short-lived read-only Franka connections, calls ``read_once``/``load_model``,
and evaluates the local model.  It never starts a controller and never sends a
motion, parameter, recovery, or stop command.

At rest, ``tau_ext_hat_filtered`` contains the gravity torque left after the
currently configured total load has been subtracted.  libfranka's own gravity
model is used to form a linear regressor in the four gravity parameters
``[mass, mass*com_x, mass*com_y, mass*com_z]``.  A constant bias is fitted for
each joint, and Huber IRLS suppresses isolated samples.  Static data cannot
identify rotational inertia; use the CAD/URDF candidate for inertia after mass
and CoM have been accepted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "dexgrasp/configs/fr3_rh56_v94_commissioning.json"
DEFAULT_RUNS = ROOT / "dexgrasp/runs"
GRAVITY_EARTH_M_S2 = 9.81


class StaticLoadIdentificationError(RuntimeError):
    """Raised when a read-only sample set cannot support identification."""


@dataclass(frozen=True)
class PoseAggregate:
    index: int
    q_rad: np.ndarray
    dq_abs_max_rad_s: float
    tau_ext_median_nm: np.ndarray
    tau_ext_mad_nm: np.ndarray
    O_F_ext_median_n_nm: np.ndarray
    gravity_regressor: np.ndarray
    sample_count: int
    robot_mode: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _errors_active(errors: Any) -> bool:
    try:
        return bool(errors)
    except Exception:
        text = str(errors).strip()
        return text not in ("", "[]", "{}", "None")


def _mode_name(mode: Any) -> str:
    return str(mode).strip().lower().rsplit(".", 1)[-1]


def _set_total_load_state(state: Any, mass: float, center: Sequence[float]) -> None:
    state.m_total = float(mass)
    state.F_x_Ctotal = [float(value) for value in center]


def gravity_parameter_regressor(model: Any, state: Any) -> np.ndarray:
    """Return the exact libfranka 7x4 gravity regressor for total EE load."""

    original_mass = float(state.m_total)
    original_center = np.asarray(state.F_x_Ctotal, dtype=np.float64).copy()

    def gravity(mass: float, center: Sequence[float]) -> np.ndarray:
        _set_total_load_state(state, mass, center)
        result = np.asarray(model.gravity(state), dtype=np.float64)
        if result.shape != (7,) or not np.all(np.isfinite(result)):
            raise StaticLoadIdentificationError("libfranka gravity returned invalid data")
        return result

    try:
        robot_only = gravity(0.0, (0.0, 0.0, 0.0))
        unit_mass_at_flange = gravity(1.0, (0.0, 0.0, 0.0))
        columns = [unit_mass_at_flange - robot_only]
        for axis in range(3):
            center = np.zeros(3, dtype=np.float64)
            center[axis] = 1.0
            columns.append(gravity(1.0, center) - unit_mass_at_flange)
        regressor = np.column_stack(columns)

        # Prove that the basis exactly reconstructs the currently configured
        # total-load gravity before using it for identification.
        _set_total_load_state(state, original_mass, original_center)
        configured = np.asarray(model.gravity(state), dtype=np.float64)
        theta = np.r_[original_mass, original_mass * original_center]
        reconstructed = robot_only + regressor @ theta
        mismatch = float(np.max(np.abs(configured - reconstructed)))
        if mismatch > 1.0e-8:
            raise StaticLoadIdentificationError(
                "libfranka gravity is not linear in the expected mass/first-moment "
                f"basis: max mismatch={mismatch:.3e}Nm"
            )
        return regressor
    finally:
        _set_total_load_state(state, original_mass, original_center)


def _validate_static_state(state: Any, maximum_dq_rad_s: float) -> None:
    mode = _mode_name(state.robot_mode)
    # Releasing the Pilot guiding/enable input can leave an otherwise healthy,
    # stationary FR3 in UserStopped rather than Idle.  Both modes are valid for
    # read-only gravity identification; Move/Guiding/Reflex/Recovery are not.
    if mode not in ("idle", "kidle", "userstopped", "kuserstopped"):
        raise StaticLoadIdentificationError(
            "robot must be Idle or UserStopped after releasing guiding buttons; "
            f"mode={state.robot_mode}"
        )
    if _errors_active(state.current_errors):
        raise StaticLoadIdentificationError(
            f"current Franka errors are active: {state.current_errors}"
        )
    for field in (
        "joint_contact",
        "joint_collision",
        "cartesian_contact",
        "cartesian_collision",
    ):
        values = np.asarray(getattr(state, field), dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise StaticLoadIdentificationError(f"{field} is invalid")
        if np.any(values != 0.0):
            raise StaticLoadIdentificationError(
                f"{field} is active; identification requires an unloaded/no-contact pose"
            )
    dq = np.asarray(state.dq, dtype=np.float64)
    if dq.shape != (7,) or not np.all(np.isfinite(dq)):
        raise StaticLoadIdentificationError("Franka dq is invalid")
    if float(np.max(np.abs(dq))) > maximum_dq_rad_s:
        raise StaticLoadIdentificationError(
            f"Franka is not stationary: max|dq|={np.max(np.abs(dq)):.6f}rad/s"
        )


def capture_pose(
    *,
    robot_ip: str,
    index: int,
    sample_count: int,
    sample_period_s: float,
    maximum_dq_rad_s: float,
) -> tuple[PoseAggregate, dict[str, Any]]:
    try:
        import pylibfranka
    except ImportError as exc:
        raise StaticLoadIdentificationError(
            "pylibfranka is unavailable; use the repository .venv Python"
        ) from exc

    robot = None
    try:
        robot = pylibfranka.Robot(robot_ip, pylibfranka.RealtimeConfig.kIgnore)
        model = robot.load_model()
        states = []
        pose_regressor = None
        for sample_index in range(sample_count):
            state = robot.read_once()
            _validate_static_state(state, maximum_dq_rad_s)
            states.append(state)
            if sample_index == 0:
                # pylibfranka 0.21.2 reliably propagates locally replaced
                # RobotState load fields into Model.gravity for the first
                # read_once object of a connection.  Repeating this operation
                # on later read_once objects can silently keep the configured
                # load instead.  A pose is required to remain stationary, so
                # its gravity regressor is constant and is computed exactly
                # once from the first state.  All subsequent states are used
                # only for stationarity checks and robust sensor aggregation.
                pose_regressor = gravity_parameter_regressor(model, state)
            if sample_index + 1 < sample_count:
                time.sleep(sample_period_s)

        q = np.stack([np.asarray(state.q, dtype=np.float64) for state in states])
        dq = np.stack([np.asarray(state.dq, dtype=np.float64) for state in states])
        tau = np.stack(
            [np.asarray(state.tau_ext_hat_filtered, dtype=np.float64) for state in states]
        )
        wrench = np.stack(
            [np.asarray(state.O_F_ext_hat_K, dtype=np.float64) for state in states]
        )
        if pose_regressor is None:
            raise StaticLoadIdentificationError("pose has no gravity regressor")
        aggregate = PoseAggregate(
            index=index,
            q_rad=np.median(q, axis=0),
            dq_abs_max_rad_s=float(np.max(np.abs(dq))),
            tau_ext_median_nm=np.median(tau, axis=0),
            tau_ext_mad_nm=np.median(
                np.abs(tau - np.median(tau, axis=0)), axis=0
            ),
            O_F_ext_median_n_nm=np.median(wrench, axis=0),
            gravity_regressor=pose_regressor,
            sample_count=sample_count,
            robot_mode=str(states[-1].robot_mode),
        )
        initial = states[0]
        dynamics = {
            "m_ee_kg": float(initial.m_ee),
            "F_x_Cee_m": np.asarray(initial.F_x_Cee, dtype=np.float64),
            "I_ee_kg_m2_column_major": np.asarray(
                initial.I_ee, dtype=np.float64
            ),
            "m_load_kg": float(initial.m_load),
            "F_x_Cload_m": np.asarray(initial.F_x_Cload, dtype=np.float64),
            "I_load_kg_m2_column_major": np.asarray(
                initial.I_load, dtype=np.float64
            ),
            "m_total_kg": float(initial.m_total),
            "F_x_Ctotal_m": np.asarray(initial.F_x_Ctotal, dtype=np.float64),
            "F_T_EE_column_major": np.asarray(initial.F_T_EE, dtype=np.float64),
            "EE_T_K_column_major": np.asarray(initial.EE_T_K, dtype=np.float64),
        }
        return aggregate, dynamics
    finally:
        # Robot has no explicit close method in pylibfranka.  Destroy the
        # short-lived read-only connection before asking the operator to guide
        # to the next pose.
        robot = None
        gc.collect()


def _build_fit_system(
    poses: Sequence[PoseAggregate],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    observations = []
    joints = []
    for pose in poses:
        if pose.gravity_regressor.shape != (7, 4):
            raise StaticLoadIdentificationError("pose regressor must be 7x4")
        for joint in range(7):
            row = np.zeros(11, dtype=np.float64)
            row[:4] = pose.gravity_regressor[joint]
            row[4 + joint] = 1.0
            rows.append(row)
            observations.append(pose.tau_ext_median_nm[joint])
            joints.append(joint)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(observations, dtype=np.float64),
        np.asarray(joints, dtype=np.int64),
    )


def fit_static_correction(poses: Sequence[PoseAggregate]) -> dict[str, Any]:
    if len(poses) < 6:
        raise StaticLoadIdentificationError("at least six diverse poses are required")
    design, observed, joints = _build_fit_system(poses)

    # Remove per-joint means to assess the four physical columns independently
    # of the explicitly fitted constant torque biases.
    physical = design[:, :4].copy()
    for joint in range(7):
        mask = joints == joint
        physical[mask] -= np.mean(physical[mask], axis=0)
    singular = np.linalg.svd(physical, compute_uv=False)
    rank = int(np.linalg.matrix_rank(physical, tol=1.0e-10))
    condition = float(np.inf if singular[-1] <= 0.0 else singular[0] / singular[-1])
    if rank < 4:
        raise StaticLoadIdentificationError(
            "pose orientations do not identify all mass/CoM parameters; "
            f"physical rank={rank}/4"
        )
    if condition > 2.0e4:
        raise StaticLoadIdentificationError(
            "pose orientations are insufficiently diverse; "
            f"physical condition number={condition:.1f}"
        )

    weights = np.ones(observed.shape, dtype=np.float64)
    solution = np.zeros(11, dtype=np.float64)
    for _ in range(12):
        root_weight = np.sqrt(weights)
        weighted_design = design * root_weight[:, None]
        weighted_observed = observed * root_weight
        solution, *_ = np.linalg.lstsq(
            weighted_design, weighted_observed, rcond=None
        )
        residual = observed - design @ solution
        joint_scale = np.empty(7, dtype=np.float64)
        for joint in range(7):
            values = residual[joints == joint]
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            joint_scale[joint] = max(1.4826 * mad, 0.01)
        normalized = np.abs(residual) / joint_scale[joints]
        next_weights = np.ones_like(weights)
        outside = normalized > 1.5
        next_weights[outside] = 1.5 / normalized[outside]
        if float(np.max(np.abs(next_weights - weights))) < 1.0e-5:
            weights = next_weights
            break
        weights = next_weights

    residual = observed - design @ solution
    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    dof = max(1, observed.size - design.shape[1])
    sigma2 = float(np.sum(weights * residual**2) / dof)
    covariance = sigma2 * np.linalg.pinv(weighted_design.T @ weighted_design)
    return {
        "delta_gravity_parameters": solution[:4],
        "joint_torque_bias_nm": solution[4:],
        "parameter_covariance": covariance[:4, :4],
        "parameter_standard_error": np.sqrt(
            np.maximum(np.diag(covariance[:4, :4]), 0.0)
        ),
        "residual_nm": residual,
        "residual_rmse_nm": float(np.sqrt(np.mean(residual**2))),
        "residual_max_abs_nm": float(np.max(np.abs(residual))),
        "huber_weight_min": float(np.min(weights)),
        "physical_rank": rank,
        "physical_condition_number": condition,
        "physical_singular_values": singular,
    }


def identify(
    poses: Sequence[PoseAggregate], current_dynamics: dict[str, Any]
) -> dict[str, Any]:
    fit = fit_static_correction(poses)
    current_mass = float(current_dynamics["m_total_kg"])
    current_center = np.asarray(
        current_dynamics["F_x_Ctotal_m"], dtype=np.float64
    )
    current_parameters = np.r_[current_mass, current_mass * current_center]
    candidate_parameters = current_parameters + fit["delta_gravity_parameters"]
    candidate_mass = float(candidate_parameters[0])
    if not math.isfinite(candidate_mass) or not 0.1 <= candidate_mass <= 3.0:
        raise StaticLoadIdentificationError(
            f"identified mass is physically implausible: {candidate_mass:.6f}kg"
        )
    candidate_center = candidate_parameters[1:] / candidate_mass
    if not np.all(np.isfinite(candidate_center)) or float(
        np.linalg.norm(candidate_center)
    ) > 0.30:
        raise StaticLoadIdentificationError(
            f"identified CoM is physically implausible: {candidate_center.tolist()}m"
        )

    covariance = np.asarray(fit["parameter_covariance"], dtype=np.float64)
    h = candidate_parameters[1:]
    jacobian = np.zeros((3, 4), dtype=np.float64)
    jacobian[:, 0] = -h / (candidate_mass**2)
    jacobian[:, 1:] = np.eye(3) / candidate_mass
    center_covariance = jacobian @ covariance @ jacobian.T
    center_standard_error = np.sqrt(
        np.maximum(np.diag(center_covariance), 0.0)
    )
    mass_standard_error = float(fit["parameter_standard_error"][0])

    accepted = bool(
        mass_standard_error <= 0.08
        and float(np.max(center_standard_error)) <= 0.015
        and fit["residual_rmse_nm"] <= 0.35
        and fit["physical_condition_number"] <= 2.0e4
    )
    return {
        "accepted_for_desk_candidate": accepted,
        "current_total_load": {
            "mass_kg": current_mass,
            "F_x_Ctotal_m": current_center,
            "gravity_parameters": current_parameters,
        },
        "identified_total_load": {
            "mass_kg": candidate_mass,
            "mass_standard_error_kg": mass_standard_error,
            "F_x_Ctotal_m": candidate_center,
            "F_x_Ctotal_standard_error_m": center_standard_error,
            "gravity_parameters": candidate_parameters,
        },
        "fit": fit,
    }


def _same_dynamics(reference: dict[str, Any], actual: dict[str, Any]) -> bool:
    scalar_fields = ("m_ee_kg", "m_load_kg", "m_total_kg")
    vector_fields = (
        "F_x_Cee_m",
        "I_ee_kg_m2_column_major",
        "F_x_Cload_m",
        "I_load_kg_m2_column_major",
        "F_x_Ctotal_m",
        "F_T_EE_column_major",
        "EE_T_K_column_major",
    )
    return all(
        abs(float(reference[field]) - float(actual[field])) <= 1.0e-9
        for field in scalar_fields
    ) and all(
        np.allclose(reference[field], actual[field], atol=1.0e-9, rtol=0.0)
        for field in vector_fields
    )


def _pose_to_json(pose: PoseAggregate) -> dict[str, Any]:
    return {
        "index": pose.index,
        "q_rad": pose.q_rad,
        "dq_abs_max_rad_s": pose.dq_abs_max_rad_s,
        "tau_ext_median_nm": pose.tau_ext_median_nm,
        "tau_ext_mad_nm": pose.tau_ext_mad_nm,
        "O_F_ext_median_n_nm": pose.O_F_ext_median_n_nm,
        "gravity_regressor": pose.gravity_regressor,
        "sample_count": pose.sample_count,
        "robot_mode": pose.robot_mode,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--poses", type=int, default=8)
    parser.add_argument("--samples-per-pose", type=int, default=80)
    parser.add_argument("--sample-period-s", type=float, default=0.01)
    parser.add_argument("--maximum-dq-rad-s", type=float, default=0.008)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 6 <= args.poses <= 16:
        parser.error("--poses must be in 6..16")
    if not 20 <= args.samples_per_pose <= 500:
        parser.error("--samples-per-pose must be in 20..500")
    if not 0.002 <= args.sample_period_s <= 0.05:
        parser.error("--sample-period-s must be in 0.002..0.05")
    if not 0.001 <= args.maximum_dq_rad_s <= 0.02:
        parser.error("--maximum-dq-rad-s must be in 0.001..0.02")

    profile_path = args.profile.expanduser().resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    robot_ip = str(profile["franka"]["ip"])
    print(
        "[READ-ONLY] 本程序不会创建控制器或发送任何机器人/末端写命令。\n"
        "请先在 Desk 清除上一次 reflex；RH56 保持空载，整套末端不要接触桌面、物体或人。\n"
        "每次用 Pilot 引导到一个新的静止姿态，松开全部按钮，等待机械臂 Idle 后按回车。\n"
        "重点改变手腕朝向；位置可小范围变化。任何时候 Ctrl+C 均可退出。",
        flush=True,
    )

    poses = []
    reference_dynamics = None
    for index in range(1, args.poses + 1):
        prompt = (
            f"\n[{index}/{args.poses}] 移到无接触静止姿态（与之前腕部方向不同），"
            "松开引导按钮后按 Enter；输入 q 退出: "
        )
        response = input(prompt).strip().lower()
        if response in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        aggregate, dynamics = capture_pose(
            robot_ip=robot_ip,
            index=index,
            sample_count=args.samples_per_pose,
            sample_period_s=args.sample_period_s,
            maximum_dq_rad_s=args.maximum_dq_rad_s,
        )
        if reference_dynamics is None:
            reference_dynamics = dynamics
        elif not _same_dynamics(reference_dynamics, dynamics):
            raise StaticLoadIdentificationError(
                "Franka configured EE/load parameters changed between poses"
            )
        poses.append(aggregate)
        print(
            f"[pose {index} PASS] max|dq|={aggregate.dq_abs_max_rad_s:.6f}rad/s "
            f"tau_ext={np.round(aggregate.tau_ext_median_nm, 4).tolist()}Nm",
            flush=True,
        )

    assert reference_dynamics is not None
    result = identify(poses, reference_dynamics)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else DEFAULT_RUNS / f"fr3_rh56_static_load_identification_{timestamp}.json"
    )
    payload = {
        "schema_version": 1,
        "classification": "read_only_static_multi_pose_gravity_identification",
        "hardware_writes": False,
        "controller_created": False,
        "limitations": {
            "rotational_inertia_identified": False,
            "inertia_source_required": "RH56_URDF_plus_V7_adapter_CAD",
            "requires_stationary_unloaded_no_contact_poses": True,
            "candidate_must_be_written_in_Franka_Desk_then_read_back": True,
        },
        "profile": str(profile_path),
        "robot_ip": robot_ip,
        "capture": {
            "pose_count": len(poses),
            "samples_per_pose": args.samples_per_pose,
            "sample_period_s": args.sample_period_s,
            "maximum_dq_rad_s": args.maximum_dq_rad_s,
            "configured_dynamics": reference_dynamics,
            "poses": [_pose_to_json(pose) for pose in poses],
        },
        "identification": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    identified = result["identified_total_load"]
    print(f"\n[SAVED] {output}")
    print(
        "[RESULT] accepted={} mass={:.6f}+/-{:.6f}kg CoM={} +/-{}m "
        "fit_rmse={:.4f}Nm condition={:.1f}".format(
            result["accepted_for_desk_candidate"],
            identified["mass_kg"],
            identified["mass_standard_error_kg"],
            np.round(identified["F_x_Ctotal_m"], 7).tolist(),
            np.round(identified["F_x_Ctotal_standard_error_m"], 7).tolist(),
            result["fit"]["residual_rmse_nm"],
            result["fit"]["physical_condition_number"],
        )
    )
    if not result["accepted_for_desk_candidate"]:
        print(
            "[NOT ACCEPTED] 数据质量/姿态多样性不足；不要写入 Desk，换更丰富的腕部方向重采。"
        )
        return 3
    print("[CANDIDATE ONLY] 尚未写入 Desk，也未修改部署 profile/native 合同。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 未发送任何硬件写命令。", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, StaticLoadIdentificationError) as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
