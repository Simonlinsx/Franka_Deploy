"""Read-only RH56 unloaded multi-pose force-baseline characterization.

The operator guides Franka between poses with the Pilot. Hardware interfaces
are opened only after the operator releases the Pilot and presses Enter. The
program calls Franka ``read_once`` and reads RH56 registers; it never creates a
controller and never writes either device.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from sim2real.io import FrankaStateReader, InspireStateReader


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "dexgrasp/configs/fr3_rh56_v94_commissioning.json"
DEFAULT_RUNS = ROOT / "dexgrasp/runs"
CONFIRMATION = "RH56_UNLOADED_NO_CONTACT"
HARDWARE_AXES = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)
POSE_LABELS_9 = (
    "reference_start",
    "palm_up",
    "palm_left",
    "palm_right",
    "palm_forward",
    "palm_backward",
    "diagonal_a",
    "diagonal_b",
    "reference_return",
)
POSE_PROMPTS_9 = (
    "选择一个便于重复的初始参考姿态",
    "将掌面方向翻到与参考明显相反",
    "让掌面大致朝左",
    "让掌面大致朝右",
    "让掌面大致朝前",
    "让掌面大致朝后",
    "选择一个斜向姿态 A",
    "选择另一个不同的斜向姿态 B",
    "尽量准确回到第 1 个参考姿态",
)


class ForceBaselineCalibrationError(RuntimeError):
    """A capture is unsafe, non-static, or insufficient for analysis."""


@dataclass(frozen=True)
class PoseAggregate:
    index: int
    label: str
    robot_mode: str
    force_median_gf: np.ndarray
    force_mean_gf: np.ndarray
    force_std_gf: np.ndarray
    force_mad_gf: np.ndarray
    force_min_gf: np.ndarray
    force_max_gf: np.ndarray
    angles_median: np.ndarray
    positions_median: np.ndarray
    temperatures_median_c: np.ndarray
    q_median_rad: np.ndarray
    gravity_ee_median: np.ndarray
    maximum_abs_dq_rad_s: float
    within_pose_orientation_span_deg: float
    sample_count: int


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(
        second, dtype=np.float64
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _rotation_angle_deg_from_gravity(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(
        np.clip(
            np.dot(np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)),
            -1.0,
            1.0,
        )
    )
    return math.degrees(math.acos(cosine))


def _gravity_in_ee(T_base_ee: np.ndarray) -> np.ndarray:
    rotation = np.asarray(T_base_ee, dtype=np.float64)[:3, :3]
    result = rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1.0e-5:
        raise ForceBaselineCalibrationError("Franka EE gravity vector is invalid")
    return result


def _validate_franka(observation: Any, maximum_dq_rad_s: float) -> None:
    if str(observation.robot_mode) not in ("idle", "userstopped"):
        raise ForceBaselineCalibrationError(
            "Franka must be Idle or UserStopped after releasing Pilot; "
            f"mode={observation.robot_mode}"
        )
    if tuple(observation.current_errors):
        raise ForceBaselineCalibrationError(
            f"Franka reports active errors={observation.current_errors}"
        )
    maximum_dq = float(np.max(np.abs(observation.dq)))
    if maximum_dq > maximum_dq_rad_s:
        raise ForceBaselineCalibrationError(
            f"Franka is not stationary: max|dq|={maximum_dq:.6f}rad/s"
        )
    for name in (
        "joint_contact",
        "joint_collision",
        "cartesian_contact",
        "cartesian_collision",
    ):
        values = np.asarray(getattr(observation, name), dtype=np.float64)
        if np.any(values != 0.0):
            raise ForceBaselineCalibrationError(
                f"Franka {name} is active; unloaded/no-contact capture required"
            )


def _validate_hand(observation: Any, maximum_idle_current_ma: int) -> None:
    targets = tuple(int(value) for value in observation.angle_targets)
    errors = tuple(int(value) for value in observation.errors)
    statuses = tuple(int(value) for value in observation.statuses)
    currents = tuple(int(value) for value in observation.currents)
    if targets != (-1,) * 6:
        raise ForceBaselineCalibrationError(
            f"RH56 must remain disabled with ANGLE_SET=-1; actual={targets}"
        )
    if any(errors):
        raise ForceBaselineCalibrationError(f"RH56 reports errors={errors}")
    if not all(value in (2, 0xFF) for value in statuses):
        raise ForceBaselineCalibrationError(
            f"RH56 must be stationary/idle; statuses={statuses}"
        )
    if any(abs(value) > maximum_idle_current_ma for value in currents):
        raise ForceBaselineCalibrationError(
            f"RH56 idle current exceeds {maximum_idle_current_ma}mA; "
            f"currents={currents}"
        )


def _pose_labels(count: int) -> Tuple[Tuple[str, str], ...]:
    if count == 9:
        return tuple(zip(POSE_LABELS_9, POSE_PROMPTS_9))
    result = []
    for index in range(count):
        if index == 0:
            result.append(("reference_start", "选择一个便于重复的初始参考姿态"))
        elif index + 1 == count:
            result.append(
                ("reference_return", "尽量准确回到第 1 个参考姿态")
            )
        else:
            result.append(
                (
                    f"orientation_{index:02d}",
                    "选择一个与已采姿态明显不同的腕部方向",
                )
            )
    return tuple(result)


def capture_pose(
    *,
    robot_ip: str,
    hand_settings: Mapping[str, Any],
    index: int,
    label: str,
    samples: int,
    sample_period_s: float,
    maximum_dq_rad_s: float,
    maximum_idle_current_ma: int,
) -> Tuple[PoseAggregate, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    rotations: List[np.ndarray] = []
    forces: List[np.ndarray] = []
    angles: List[np.ndarray] = []
    positions: List[np.ndarray] = []
    temperatures: List[np.ndarray] = []
    q_values: List[np.ndarray] = []
    gravity_values: List[np.ndarray] = []
    maximum_dq = 0.0
    robot_mode = ""

    franka = FrankaStateReader(robot_ip, enforce_realtime=False)
    hand = InspireStateReader(
        port=str(hand_settings["port"]),
        baud=int(hand_settings["baud"]),
        hand_id=int(hand_settings["hand_id"]),
        timeout_s=0.5,
        snapshot_mode="compact_policy",
    )
    try:
        franka.start()
        hand.start()
        next_tick = time.monotonic()
        for sample_index in range(samples):
            arm = franka.read()
            inspire = hand.read()
            _validate_franka(arm, maximum_dq_rad_s)
            _validate_hand(inspire, maximum_idle_current_ma)

            rotation = np.asarray(arm.T_base_ee[:3, :3], dtype=np.float64)
            gravity = _gravity_in_ee(arm.T_base_ee)
            force = np.asarray(inspire.forces, dtype=np.int64)
            angle = np.asarray(inspire.angles, dtype=np.int64)
            position = np.asarray(inspire.positions, dtype=np.int64)
            temperature = np.asarray(inspire.temperatures_c, dtype=np.int64)
            current = np.asarray(inspire.currents, dtype=np.int64)
            q = np.asarray(arm.q, dtype=np.float64)
            maximum_dq = max(maximum_dq, float(np.max(np.abs(arm.dq))))
            robot_mode = str(arm.robot_mode)

            rotations.append(rotation)
            gravity_values.append(gravity)
            forces.append(force)
            angles.append(angle)
            positions.append(position)
            temperatures.append(temperature)
            q_values.append(q)
            row: Dict[str, Any] = {
                "pose_index": index,
                "pose_label": label,
                "sample_index": sample_index,
                "franka_captured_at_s": float(arm.captured_at_s),
                "rh56_captured_at_s": float(inspire.captured_at_s),
                "robot_mode": robot_mode,
                "maximum_abs_dq_rad_s": float(np.max(np.abs(arm.dq))),
            }
            for axis in range(7):
                row[f"franka_q{axis + 1}_rad"] = float(q[axis])
            for axis, coordinate in enumerate(("x", "y", "z")):
                row[f"gravity_ee_{coordinate}"] = float(gravity[axis])
            for axis, name in enumerate(HARDWARE_AXES):
                row[f"{name}_angle_act"] = int(angle[axis])
                row[f"{name}_position_act"] = int(position[axis])
                row[f"{name}_force_gf"] = int(force[axis])
                row[f"{name}_current_ma"] = int(current[axis])
                row[f"{name}_temperature_c"] = int(temperature[axis])
            rows.append(row)

            next_tick += sample_period_s
            if sample_index + 1 < samples:
                time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        hand.close()
        franka.close()
        gc.collect()

    force_array = np.stack(forces).astype(np.float64)
    angle_array = np.stack(angles).astype(np.float64)
    position_array = np.stack(positions).astype(np.float64)
    temperature_array = np.stack(temperatures).astype(np.float64)
    q_array = np.stack(q_values).astype(np.float64)
    gravity_array = np.stack(gravity_values).astype(np.float64)
    orientation_span = max(
        _rotation_angle_deg(rotations[0], rotation) for rotation in rotations
    )
    force_median = np.median(force_array, axis=0)
    aggregate = PoseAggregate(
        index=index,
        label=label,
        robot_mode=robot_mode,
        force_median_gf=force_median,
        force_mean_gf=np.mean(force_array, axis=0),
        force_std_gf=np.std(force_array, axis=0, ddof=1),
        force_mad_gf=np.median(np.abs(force_array - force_median), axis=0),
        force_min_gf=np.min(force_array, axis=0),
        force_max_gf=np.max(force_array, axis=0),
        angles_median=np.median(angle_array, axis=0),
        positions_median=np.median(position_array, axis=0),
        temperatures_median_c=np.median(temperature_array, axis=0),
        q_median_rad=np.median(q_array, axis=0),
        gravity_ee_median=np.median(gravity_array, axis=0),
        maximum_abs_dq_rad_s=maximum_dq,
        within_pose_orientation_span_deg=orientation_span,
        sample_count=samples,
    )
    return aggregate, rows


def analyze_poses(poses: Sequence[PoseAggregate]) -> Dict[str, Any]:
    if len(poses) < 6:
        raise ValueError("at least six poses are required")
    force = np.stack([pose.force_median_gf for pose in poses]).astype(np.float64)
    gravity = np.stack([pose.gravity_ee_median for pose in poses]).astype(np.float64)
    design = np.column_stack([np.ones(len(poses)), gravity])
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, force, rcond=None
    )
    predicted = design @ coefficients
    residual = force - predicted

    reference = force[0]
    spans = np.ptp(force, axis=0)
    max_reference_drift = np.max(np.abs(force - reference), axis=0)
    sum_squares_total = np.sum(
        (force - np.mean(force, axis=0, keepdims=True)) ** 2, axis=0
    )
    sum_squares_residual = np.sum(residual**2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = np.where(
            sum_squares_total > 1.0e-12,
            1.0 - sum_squares_residual / sum_squares_total,
            0.0,
        )
    return_orientation_error_deg = _rotation_angle_deg_from_gravity(
        gravity[0], gravity[-1]
    )
    return_force_delta = force[-1] - force[0]
    angles = np.stack([pose.angles_median for pose in poses]).astype(np.float64)
    positions = np.stack([pose.positions_median for pose in poses]).astype(np.float64)
    angle_change = np.abs(angles - angles[0][None, :])
    position_change = np.abs(positions - positions[0][None, :])
    maximum_angle_change = float(np.max(angle_change))
    maximum_position_change = float(np.max(position_change))
    position_pose_index, position_axis = np.unravel_index(
        int(np.argmax(position_change)), position_change.shape
    )
    quality_reasons = []
    if int(rank) < 4:
        quality_reasons.append("insufficient_gravity_orientation_coverage")
    if return_orientation_error_deg > 5.0:
        quality_reasons.append("reference_return_orientation_error_exceeds_5deg")
    if maximum_angle_change > 10.0:
        quality_reasons.append("rh56_angle_configuration_changed_over_10_units")
    if maximum_position_change > 10.0:
        quality_reasons.append("rh56_actuator_position_changed_over_10_units")

    axis_reports = []
    classifications = []
    for axis, name in enumerate(HARDWARE_AXES):
        span = float(spans[axis])
        classification = (
            "low" if span < 50.0 else "moderate" if span <= 150.0 else "high"
        )
        classifications.append(classification)
        axis_reports.append(
            {
                "axis": axis,
                "name": name,
                "reference_median_gf": float(reference[axis]),
                "pose_median_min_gf": float(np.min(force[:, axis])),
                "pose_median_max_gf": float(np.max(force[:, axis])),
                "pose_median_span_gf": span,
                "maximum_abs_drift_from_reference_gf": float(
                    max_reference_drift[axis]
                ),
                "gravity_linear_model": {
                    "intercept_gf": float(coefficients[0, axis]),
                    "gravity_ee_coefficients_gf": coefficients[1:, axis],
                    "residual_rmse_gf": float(
                        np.sqrt(np.mean(residual[:, axis] ** 2))
                    ),
                    "r_squared": float(r_squared[axis]),
                },
                "return_reference_delta_gf": float(return_force_delta[axis]),
                "orientation_sensitivity": classification,
            }
        )

    if "high" in classifications:
        overall = "high_pose_dependence"
    elif "moderate" in classifications:
        overall = "moderate_pose_dependence"
    else:
        overall = "low_pose_dependence"
    return {
        "overall_classification": overall,
        "quality": {
            "valid_for_gravity_compensation": not quality_reasons,
            "reasons": quality_reasons,
            "gravity_orientation_coverage_full_rank": int(rank) == 4,
            "maximum_hand_angle_change_from_reference_units": (
                maximum_angle_change
            ),
            "maximum_hand_position_change_from_reference_units": (
                maximum_position_change
            ),
            "maximum_position_change_pose_index": int(position_pose_index + 1),
            "maximum_position_change_axis": HARDWARE_AXES[int(position_axis)],
            "per_axis_maximum_position_change_from_reference_units": np.max(
                position_change, axis=0
            ),
        },
        "pose_median_force_gf_hardware_order": force,
        "gravity_ee_unit_vectors": gravity,
        "gravity_design_rank": int(rank),
        "gravity_design_singular_values": singular_values,
        "return_reference": {
            "orientation_error_deg": return_orientation_error_deg,
            "comparable_within_5deg": return_orientation_error_deg <= 5.0,
            "force_delta_gf_hardware_order": return_force_delta,
        },
        "axes": axis_reports,
        "candidate_compensation": {
            "status": "candidate_only_not_applied",
            "equation": (
                "baseline_gf = intercept + dot(gravity_ee_unit, coefficients); "
                "corrected_gf = raw_gf - baseline_gf"
            ),
            "requires_fixed_hand_configuration": True,
        },
    }


def _pose_json(pose: PoseAggregate) -> Dict[str, Any]:
    return {field: getattr(pose, field) for field in pose.__dataclass_fields__}


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _exclusive_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--poses", type=int, default=9)
    parser.add_argument("--samples-per-pose", type=int, default=40)
    parser.add_argument("--sample-rate-hz", type=float, default=20.0)
    parser.add_argument("--maximum-dq-rad-s", type=float, default=0.01)
    parser.add_argument("--maximum-idle-current-ma", type=int, default=100)
    parser.add_argument("--hand-configuration", default="open")
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--confirm-unloaded", metavar="TOKEN", required=True)
    args = parser.parse_args(argv)
    if args.confirm_unloaded != CONFIRMATION:
        parser.error(f"--confirm-unloaded must be {CONFIRMATION}")
    if not 6 <= args.poses <= 16:
        parser.error("--poses must be in 6..16")
    if not 20 <= args.samples_per_pose <= 200:
        parser.error("--samples-per-pose must be in 20..200")
    if not 5.0 <= args.sample_rate_hz <= 30.0:
        parser.error("--sample-rate-hz must be in 5..30")
    if not 0.001 <= args.maximum_dq_rad_s <= 0.05:
        parser.error("--maximum-dq-rad-s must be in 0.001..0.05")
    if not 20 <= args.maximum_idle_current_ma <= 200:
        parser.error("--maximum-idle-current-ma must be in 20..200")

    profile_path = args.profile.expanduser().resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    robot_ip = str(profile["franka"]["ip"])
    hand_settings = profile["inspire"]
    sample_period_s = 1.0 / float(args.sample_rate_hz)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = (
        args.output_prefix.expanduser().resolve()
        if args.output_prefix is not None
        else DEFAULT_RUNS / f"rh56_force_unloaded_multipoise_{timestamp}"
    )
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[READ-ONLY] 不创建 Franka 控制器，不写 Franka/RH56。\n"
        "RH56 必须空载、无接触、六轴 ANGLE_SET=-1；保持同一手指构型。\n"
        "每次先用 Pilot 调整腕部方向，松开所有按钮并静止，再按 Enter。\n"
        "Franka Idle 或 UserStopped 均可；采样期间不要触碰机器人。",
        flush=True,
    )

    poses: List[PoseAggregate] = []
    rows: List[Dict[str, Any]] = []
    labels = _pose_labels(int(args.poses))
    for pose_index, (label, instruction) in enumerate(labels, start=1):
        while True:
            response = input(
                f"\n[{pose_index}/{args.poses}] {instruction}；松开 Pilot 后按 Enter，"
                "输入 q 退出: "
            ).strip().lower()
            if response in ("q", "quit", "exit"):
                raise KeyboardInterrupt
            pose, pose_rows = capture_pose(
                robot_ip=robot_ip,
                hand_settings=hand_settings,
                index=pose_index,
                label=label,
                samples=int(args.samples_per_pose),
                sample_period_s=sample_period_s,
                maximum_dq_rad_s=float(args.maximum_dq_rad_s),
                maximum_idle_current_ma=int(args.maximum_idle_current_ma),
            )
            if label == "reference_return" and poses:
                return_error = _rotation_angle_deg_from_gravity(
                    poses[0].gravity_ee_median, pose.gravity_ee_median
                )
                if return_error > 5.0:
                    print(
                        f"[RETRY] 回参考姿态误差={return_error:.2f}deg > 5deg；"
                        "本次 40 帧丢弃，请调整腕部方向后重新采样。",
                        flush=True,
                    )
                    continue
            break
        poses.append(pose)
        rows.extend(pose_rows)
        print(
            f"[pose {pose_index} PASS] gravity_ee="
            f"{np.round(pose.gravity_ee_median, 3).tolist()} "
            f"force_median_gf={np.rint(pose.force_median_gf).astype(int).tolist()} "
            f"max|dq|={pose.maximum_abs_dq_rad_s:.6f}rad/s",
            flush=True,
        )
        position_change = np.abs(
            pose.positions_median - poses[0].positions_median
        )
        if float(np.max(position_change)) > 10.0:
            changed_axis = int(np.argmax(position_change))
            print(
                "[CONFIGURATION WARNING] RH56 POS_ACT 相对参考姿态变化过大："
                f"axis={HARDWARE_AXES[changed_axis]} "
                f"delta={position_change[changed_axis]:.1f} units；"
                "最终重力补偿会被标记为无效。",
                flush=True,
            )

    analysis = analyze_poses(poses)
    reference_angles = poses[0].angles_median
    maximum_hand_angle_change = float(
        np.max(
            np.abs(
                np.stack([pose.angles_median for pose in poses])
                - reference_angles[None, :]
            )
        )
    )
    payload = {
        "schema": "rh56_unloaded_multipoise_force_baseline_v1",
        "created_at_local": datetime.now().astimezone().isoformat(),
        "classification": "read_only_unloaded_force_baseline_characterization",
        "hardware_writes": False,
        "franka_controller_created": False,
        "profile": str(profile_path),
        "robot_ip": robot_ip,
        "hand_configuration": str(args.hand_configuration),
        "hardware_axis_order": list(HARDWARE_AXES),
        "capture": {
            "pose_count": len(poses),
            "samples_per_pose": int(args.samples_per_pose),
            "nominal_sample_rate_hz": float(args.sample_rate_hz),
            "maximum_hand_angle_change_from_reference_units": (
                maximum_hand_angle_change
            ),
            "maximum_hand_position_change_from_reference_units": analysis[
                "quality"
            ]["maximum_hand_position_change_from_reference_units"],
            "poses": [_pose_json(pose) for pose in poses],
        },
        "analysis": analysis,
        "limitations": {
            "external_load_applied": False,
            "force_scale_to_fingertip_newtons_identified": False,
            "gravity_compensation_is_candidate_only": True,
            "valid_only_for_same_hand_configuration": True,
        },
    }
    _exclusive_csv(csv_path, rows)
    try:
        _exclusive_json(json_path, payload)
    except BaseException:
        try:
            csv_path.unlink()
        except OSError:
            pass
        raise

    print(f"\n[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")
    print(
        "[RESULT] "
        + str(analysis["overall_classification"])
        + " force spans(gf)="
        + str(
            [
                round(float(axis["pose_median_span_gf"]), 1)
                for axis in analysis["axes"]
            ]
        )
    )
    if not analysis["quality"]["valid_for_gravity_compensation"]:
        print(
            "[QUALITY WARNING] candidate compensation is NOT valid: "
            + ", ".join(analysis["quality"]["reasons"])
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 未发送任何硬件写命令。")
