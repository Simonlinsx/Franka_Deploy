#!/usr/bin/env python3
"""Run the V94 checkpoint over recorded real RGB-D/robot data, offline only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
    from sim2real.policy import RollingStudentPolicy
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13, V94Contract
    from sim2real.observation.kinematics import RH56FeedbackMapper
    from sim2real.observation.model import (
        MaskedRGBDProjector,
        PolicyHistory,
        Proprio67Builder,
        pose_from_position_quaternion_wxyz,
    )
else:
    from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
    from sim2real.policy import RollingStudentPolicy
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13, V94Contract
    from sim2real.observation.kinematics import RH56FeedbackMapper
    from sim2real.observation.model import (
        MaskedRGBDProjector,
        PolicyHistory,
        Proprio67Builder,
        pose_from_position_quaternion_wxyz,
    )


from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load safe replay NPZ {path}: {exc}") from exc


def _required(data: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"recorded replay is missing {name}")
    return np.asarray(data[name])


def replay_recording(
    *,
    bundle_path: str | Path,
    input_path: str | Path,
    maximum_camera_age_s: float = 0.10,
) -> Dict[str, np.ndarray]:
    """Return policy/target arrays without opening any hardware interface."""

    bundle = DeployBundle(bundle_path)
    bundle.verify()
    contract = V94Contract.from_bundle(bundle)
    checkpoint = load_checkpoint_safely(bundle.checkpoint_bytes())
    policy = RollingStudentPolicy(checkpoint)
    data = _load_npz(Path(input_path).expanduser().resolve())
    rgb = _required(data, "rgb")
    mask = _required(data, "object_mask")
    q = _required(data, "franka_q")
    dq = _required(data, "franka_dq")
    hand_angles = _required(data, "rh56_angle_act")
    palm_position = _required(data, "palm_position_base")
    palm_quaternion = _required(data, "palm_quaternion_base_wxyz")
    palm_linear = _required(data, "palm_linear_velocity_base")
    palm_angular = _required(data, "palm_angular_velocity_base")
    fingertips = _required(data, "fingertip_positions_base")
    count = int(q.shape[0])
    shapes = {
        "rgb": (count, contract.camera_height, contract.camera_width, 3),
        "object_mask": (count, contract.camera_height, contract.camera_width),
        "franka_q": (count, 7),
        "franka_dq": (count, 7),
        "rh56_angle_act": (count, 6),
        "palm_position_base": (count, 3),
        "palm_quaternion_base_wxyz": (count, 4),
        "palm_linear_velocity_base": (count, 3),
        "palm_angular_velocity_base": (count, 3),
        "fingertip_positions_base": (count, 5, 3),
    }
    for name, shape in shapes.items():
        if np.asarray(data[name]).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if rgb.dtype != np.uint8 or mask.dtype != np.bool_:
        raise ValueError("rgb must be uint8 BGR and object_mask must be bool")
    if count <= 0:
        raise ValueError("recorded replay must contain at least one control step")

    if "depth_m" in data:
        depth = np.asarray(data["depth_m"], dtype=np.float32)
    elif "depth_raw" in data:
        raw_depth = np.asarray(data["depth_raw"])
        if raw_depth.dtype != np.uint16:
            raise ValueError("depth_raw must be uint16")
        depth = raw_depth.astype(np.float32) * np.float32(
            contract.depth_scale_m_per_unit
        )
    else:
        raise ValueError("recorded replay needs depth_m or depth_raw")
    if depth.shape != (count, contract.camera_height, contract.camera_width):
        raise ValueError("recorded depth has the wrong shape")

    control_time = np.asarray(
        data.get(
            "timestamp_s",
            np.arange(count, dtype=np.float64) * contract.control_dt_s,
        ),
        dtype=np.float64,
    )
    camera_time = np.asarray(
        data.get("camera_timestamp_s", control_time), dtype=np.float64
    )
    frame_ids = np.asarray(
        data.get("camera_frame_id", np.arange(count)), dtype=np.int64
    )
    if (
        control_time.shape != (count,)
        or camera_time.shape != (count,)
        or frame_ids.shape != (count,)
    ):
        raise ValueError(
            "timestamp_s/camera_timestamp_s/camera_frame_id must have shape [T]"
        )
    if not (np.all(np.isfinite(control_time)) and np.all(np.isfinite(camera_time))):
        raise ValueError("recorded timestamps must be finite")
    age = control_time - camera_time
    if np.any(age < -0.01) or np.any(age > float(maximum_camera_age_s)):
        raise ValueError(
            f"recorded camera age is outside [-0.01,{maximum_camera_age_s}] seconds"
        )
    if np.any(np.diff(control_time) <= 0.0):
        raise ValueError("recorded control timestamps must increase strictly")

    holds = np.asarray(data.get("hold_arm_target", np.zeros(count, dtype=bool)))
    if holds.shape != (count,) or holds.dtype != np.bool_:
        raise ValueError("hold_arm_target must be bool [T]")

    projector = MaskedRGBDProjector(
        camera_K=contract.camera_K,
        T_base_camera_optical=contract.T_base_camera_optical,
        image_size=(contract.camera_width, contract.camera_height),
        depth_range_m=contract.depth_range_m,
    )
    hand_mapper = RH56FeedbackMapper(q_hand_close_rad=contract.q_hand_close_rad)
    proprio_builder = Proprio67Builder(
        q_home_rad=contract.q_home_rad,
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    history = PolicyHistory()
    first_hand = hand_mapper.map(hand_angles[0]).q_policy_order_rad
    target_mapper = V94ActionMapper(
        initial_arm_target_q_rad=q[0],
        initial_hand_target_q_policy_order_rad=first_hand,
        joint_limits_rad=contract.joint_limits_rad,
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    previous_action = INITIAL_PREVIOUS_ACTION13.copy()
    previous_hand_q: Optional[np.ndarray] = None
    previous_hand_time: Optional[float] = None
    last_frame_id: Optional[int] = None
    last_point_frame = None

    raw_actions = []
    executed_actions = []
    arm_targets = []
    hand_targets = []
    hand_registers = []
    predicted_privileged = []
    predicted_hold6 = []
    predicted_hold_logit = []
    valid_counts = []
    point_status = []

    for index in range(count):
        T_base_palm = pose_from_position_quaternion_wxyz(
            palm_position[index], palm_quaternion[index]
        )
        if last_frame_id is not None and int(frame_ids[index]) == last_frame_id:
            point_frame = last_point_frame
        else:
            point_frame = projector.project(
                color_bgr=rgb[index],
                depth_m=depth[index],
                object_mask=mask[index],
                T_base_palm_at_capture=T_base_palm,
                captured_at_s=float(camera_time[index]),
                frame_id=int(frame_ids[index]),
            )
            last_frame_id = int(frame_ids[index])
            last_point_frame = point_frame
        if point_frame is None or float(np.sum(point_frame.valid)) < 1.0:
            raise ValueError(f"no deployable object point cloud at step {index}")

        hand_q = hand_mapper.map(hand_angles[index]).q_policy_order_rad
        if previous_hand_q is None:
            hand_dq = np.zeros(6, dtype=np.float32)
        else:
            dt = float(control_time[index] - previous_hand_time)
            hand_dq = ((hand_q - previous_hand_q) / dt).astype(np.float32)
        previous_hand_q = hand_q.copy()
        previous_hand_time = float(control_time[index])
        proprio = proprio_builder.build(
            franka_q_rad=q[index],
            franka_dq_rad_s=dq[index],
            rh56_virtual_q_policy_order_rad=hand_q,
            rh56_virtual_dq_policy_order_rad_s=hand_dq,
            T_base_palm=T_base_palm,
            palm_linear_velocity_base_m_s=palm_linear[index],
            palm_angular_velocity_base_rad_s=palm_angular[index],
            fingertip_positions_base_m=fingertips[index],
            previous_executed_action13=previous_action,
        )
        point_history, valid_history, proprio_history = history.append(
            point_frame, proprio
        )
        output = policy.act(point_history, valid_history, proprio_history)
        mapped = target_mapper.map(
            output.action13,
            measured_q_rad=q[index],
            hold_arm_target=bool(holds[index]),
        )
        previous_action = mapped.executed_policy_action13.copy()
        raw_actions.append(output.action13)
        executed_actions.append(mapped.executed_policy_action13)
        arm_targets.append(mapped.franka_target_q_rad)
        hand_targets.append(mapped.rh56_target_q_policy_order_rad)
        hand_registers.append(mapped.rh56_angle_set_register_order)
        predicted_privileged.append(output.predicted_privileged32)
        predicted_hold6.append(output.predicted_hold6)
        predicted_hold_logit.append(output.predicted_hold_logit)
        valid_counts.append(int(np.sum(point_frame.valid)))
        point_status.append(point_frame.status)

    return {
        "raw_policy_action13": np.stack(raw_actions).astype(np.float32),
        "executed_policy_action13": np.stack(executed_actions).astype(np.float32),
        "franka_target_q_rad": np.stack(arm_targets).astype(np.float32),
        "rh56_target_q_policy_order_rad": np.stack(hand_targets).astype(np.float32),
        "rh56_angle_set_register_order": np.stack(hand_registers).astype(np.int32),
        "predicted_privileged32": np.stack(predicted_privileged).astype(np.float32),
        "predicted_hold6": np.stack(predicted_hold6).astype(np.float32),
        "predicted_hold_logit": np.asarray(predicted_hold_logit, dtype=np.float32),
        "valid_point_count": np.asarray(valid_counts, dtype=np.int32),
        "point_status": np.asarray(point_status, dtype="<U32"),
        "hardware_writes": np.asarray(False),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay recorded real observations through V94 without hardware writes."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-camera-age-s", type=float, default=0.10)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="reserved hardware mode; always refused by this offline replay entry",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.execute:
            raise RuntimeError(
                "replay_v94 is offline-only and contains no Franka/RH56 write path"
            )
        result = replay_recording(
            bundle_path=args.bundle,
            input_path=args.input,
            maximum_camera_age_s=float(args.max_camera_age_s),
        )
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **result)
        print(
            f"[offline replay complete] steps={result['raw_policy_action13'].shape[0]} "
            f"output={output}; no hardware interface was opened"
        )
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"[offline replay failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
