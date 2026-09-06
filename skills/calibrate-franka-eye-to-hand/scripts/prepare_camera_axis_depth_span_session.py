#!/usr/bin/env python3
"""Prepare offline-only 40 mm camera-axis depth-span motion edges.

The sequence starts and finishes at the previously captured S01 pose.  It
keeps the EEF orientation constant and moves the marker centre along the
frozen camera optical axis in six 40 mm steps (three near, six back to far,
three back to S01).  No hardware module is imported here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path("/home/qiaoguanren/code/franka")
OUT = ROOT / "beta/dynamic_object_pcd/calibration_runs/fr3-d435-342222071785-20260813-424x240-60hz-depth-span-plans"
S01 = [
    [0.6944427490234375, 0.594502329826355, 0.4053594172000885, 0.4260011911392212],
    [0.6263642311096191, -0.7767239212989807, 0.06608971953392029, 0.21128949522972107],
    [0.35414284467697144, 0.2080071121454239, -0.9117652773857117, 0.8238862752914429],
    [0.0, 0.0, 0.0, 1.0],
]
STEP = [0.005023370693166713, 0.03955872530107847, 0.0031421329432168304]
LEVELS = {
    "S01": 0,
    "N01": 1,
    "N02": 2,
    "N03": 3,
    "F01": -1,
    "F02": -2,
    "F03": -3,
}
SEQUENCE = [
    ("S01", "N01"), ("N01", "N02"), ("N02", "N03"),
    ("N03", "N02"), ("N02", "N01"), ("N01", "S01"),
    ("S01", "F01"), ("F01", "F02"), ("F02", "F03"),
    ("F03", "F02"), ("F02", "F01"), ("F01", "S01"),
]


def pose(level: int) -> list[list[float]]:
    result = copy.deepcopy(S01)
    for i in range(3):
        result[i][3] += level * STEP[i]
    return result


def publish(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(path), flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def base(anchor_id: str) -> dict:
    level = LEVELS[anchor_id]
    anchor = pose(level)
    return {
        "schema_version": 1,
        "kind": "franka_eye_to_hand_commissioning_plan",
        "status": "explicit_single_pose_motion_authorized",
        "provenance": {
            "purpose": "native_424x240_60hz_aligned_depth_camera_axis_span",
            "source_text": "我固定好了，你自己调整位置，并且复验",
            "safety_confirmation_text": "不需要管周围情况，现在是安全的，别考虑那么多",
            "camera_serial": "342222071785",
            "stream_profile": "424x240_at_60hz",
            "frozen_camera_axis_base": [-0.12558426732916783, -0.9889681325269616, -0.07855332358042076],
        },
        "topology": {"type": "eye_to_hand", "camera_mount": "fixed_external", "transform_to_solve": "T_base_camera"},
        "hardware": {"robot": {
            "type": "Franka Research 3", "ip": "172.16.0.2", "pose_source": "O_T_EE",
            "expected_F_T_EE": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "expected_m_ee_kg": 0.607,
            "expected_F_x_Cee_m": [0.0, 0.0, 0.07599999755620956],
            "expected_I_ee_kg_m2": [[0.001509999972768128, 0.0, 0.0], [0.0, 0.0016899999463930726, 0.0], [0.0, 0.0, 0.0004419999895617366]],
            "expected_external_load": {"m_load_kg": 0.0, "F_x_Cload_m": [0.0, 0.0, 0.0], "I_load_kg_m2": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
        }},
        "target": {"type": "aruco", "dictionary": "DICT_6X6_50", "marker_id": 42, "marker_length_m": 0.19, "mount_must_not_change_until_finish": True},
        "reference_pose": {"source": f"predicted_exact_{anchor_id}_from_frozen_axis", "xyz_base_m": [anchor[i][3] for i in range(3)], "T_base_ee": anchor},
        "safety": {
            "motion_authorized": True, "emergency_stop_reachable": True,
            "full_large_board_bracket_and_cable_sequence_swept_volume_clear": True,
            "target_rigid_confirmed": True, "camera_fixed_confirmed": True,
            "board_max_extent_from_eef_m": 0.30,
            "table_height_near_workspace_m": 0.038,
            "minimum_eef_z_m": 0.750,
            "conservative_board_to_table_margin_at_minimum_eef_z_m": 0.412,
            "workspace_bounds_base_m": {"x": [0.350, 0.480], "y": [0.070, 0.350], "z": [0.750, 0.880]},
            "maximum_translation_step_m": 0.075, "maximum_rotation_step_deg": 20.0,
            "maximum_translation_norm_from_anchor_m": 0.050, "maximum_rotation_angle_from_anchor_deg": 16.0,
            "maximum_translation_velocity_m_s": 0.005, "maximum_translation_acceleration_m_s2": 0.00385,
            "maximum_angular_velocity_rad_s": 0.020, "maximum_angular_acceleration_rad_s2": 0.0154,
            "maximum_segment_translation_m": 0.020, "maximum_segment_rotation_deg": 5.0,
            "minimum_segment_duration_s": 4.0, "cartesian_endpoint_hold_s": 6.0,
            "cartesian_pose_controller_mode": "joint_impedance", "settle_time_s": 0.5,
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    records = []
    for ordinal, (start, target) in enumerate(SEQUENCE, 1):
        plan = base(start)
        plan["session_slug"] = f"fr3-d435-342222071785-20260813-depth-span-{ordinal:02d}-{start}-to-{target}"
        plan["training_poses"] = [{"id": start, "xyz_offset_base_m": [0.0, 0.0, 0.0], "rotation_vector_eef_deg": [0.0, 0.0, 0.0]}]
        delta = LEVELS[target] - LEVELS[start]
        plan["holdout_poses"] = [{"id": target, "xyz_offset_base_m": [delta*x for x in STEP], "rotation_vector_eef_deg": [0.0, 0.0, 0.0]}]
        plan["planned_sequence"] = {"single_edge_only": True, "start_pose_id": start, "target_pose_id": target, "capture_at_target": target in {"N03", "F03"}}
        plan["motion_history"] = []
        plan["motion_authorization"] = {"explicit_user_authorization_recorded": True, "scope": "single_pose", "start_pose_id": start, "pose_id": target, "source_text": "我固定好了，你自己调整位置，并且复验", "authorization_interpretation": "execute_reviewed_camera_axis_depth_span_without_repeated_prompts", "consumed": False}
        name = f"{ordinal:02d}-{start}-to-{target}.plan.yaml"
        payload = yaml.safe_dump(plan, sort_keys=False, allow_unicode=True).encode("utf-8")
        path = OUT / name
        publish(path, payload)
        records.append({"ordinal": ordinal, "start": start, "target": target, "plan": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "capture": target in {"N03", "F03"}})
    payload = (json.dumps({"schema_version": 1, "kind": "camera_axis_depth_span_plan_manifest", "hardware_opened": False, "robot_motion_commanded": False, "edges": records}, indent=2, sort_keys=True) + "\n").encode()
    publish(OUT / "manifest.json", payload)
    print(json.dumps({"manifest": str(OUT / "manifest.json"), "edge_count": len(records), "sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
