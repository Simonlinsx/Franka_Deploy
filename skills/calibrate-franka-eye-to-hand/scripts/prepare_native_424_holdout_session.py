#!/usr/bin/env python3
"""Create the immutable S01<->H01..H05 native 424x240@60 holdout edges.

This is deliberately session-specific and offline-only.  It opens neither the
camera nor Franka.  The fresh S01 reference below is the read-only endpoint
captured on 2026-08-13 after the operator fixed the marker and authorized the
bounded revalidation sequence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path("/home/qiaoguanren/code/franka")
OUTPUT = ROOT / "beta/dynamic_object_pcd/calibration_runs/fr3-d435-342222071785-20260813-424x240-60hz-native-holdout-S01-plans-v7"
ANCHOR = [
    [0.6944427490234375, 0.594502329826355, 0.4053594172000885, 0.4260011911392212],
    [0.6263642311096191, -0.7767239212989807, 0.06608971953392029, 0.21128949522972107],
    [0.35414284467697144, 0.2080071121454239, -0.9117652773857117, 0.8238862752914429],
    [0.0, 0.0, 0.0, 1.0],
]
POSES = {
    "S01": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    "H01": ([-0.025, 0.0, 0.015], [6.0, 0.0, 0.0]),
    "H02": ([0.025, 0.0, -0.015], [-6.0, 0.0, 0.0]),
    "H03": ([0.0, 0.040, 0.0], [0.0, 0.0, 10.0]),
    "H04": ([0.0, -0.040, 0.0], [0.0, 0.0, -10.0]),
    "H05": ([0.030, 0.020, 0.010], [0.0, 8.0, 0.0]),
    "H06": ([0.0, 0.042, 0.0], [0.0, 0.0, 10.5]),
    "H07": ([0.0, 0.042, 0.0], [0.0, 0.0, 10.1]),
    "H08": ([0.0, -0.042, 0.0], [0.0, 0.0, -10.1]),
    "H09": ([0.0, -0.044, 0.0], [0.0, 0.0, -11.0]),
    "H10": ([0.015, -0.047, 0.0], [0.0, 0.0, -11.2]),
    "H11": ([0.005023370693166713, 0.03955872530107847, 0.0031421329432168304], [0.0, 0.0, 0.0]),
    "H12": ([-0.005023370693166713, -0.03955872530107847, -0.0031421329432168304], [0.0, 0.0, 0.0]),
}


def _publish_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _base() -> dict:
    return {
        "schema_version": 1,
        "kind": "franka_eye_to_hand_commissioning_plan",
        "status": "explicit_single_pose_motion_authorized",
        "provenance": {
            "purpose": "native_424x240_60hz_reused_extrinsic_independent_holdout",
            "source_text": "我固定好了，你自己调整位置，并且复验",
            "safety_confirmation_text": "不需要管周围情况，现在是安全的，别考虑那么多",
            "current_state_read_only_verified_idle": True,
            "camera_serial": "342222071785",
            "stream_profile": "424x240_at_60hz",
            "session_reference_capture": str(ROOT / "beta/dynamic_object_pcd/calibration_runs/fr3-d435-342222071785-424x240-60hz-physical-reuse-20260813.yaml"),
        },
        "topology": {"type": "eye_to_hand", "camera_mount": "fixed_external", "transform_to_solve": "T_base_camera"},
        "hardware": {
            "robot": {
                "type": "Franka Research 3",
                "ip": "172.16.0.2",
                "pose_source": "O_T_EE",
                "expected_F_T_EE": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                "expected_m_ee_kg": 0.607,
                "expected_F_x_Cee_m": [0.0, 0.0, 0.07599999755620956],
                "expected_I_ee_kg_m2": [[0.001509999972768128, 0.0, 0.0], [0.0, 0.0016899999463930726, 0.0], [0.0, 0.0, 0.0004419999895617366]],
                "expected_external_load": {"m_load_kg": 0.0, "F_x_Cload_m": [0.0, 0.0, 0.0], "I_load_kg_m2": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
            },
            "camera": {
                "type": "Intel RealSense D435",
                "serial": "342222071785",
                "color_profile": {"width": 424, "height": 240, "fps": 60, "format": "bgr8"},
                "depth_profile": {"width": 424, "height": 240, "fps": 60, "format": "z16", "aligned_to": "color"},
            },
        },
        "target": {"type": "aruco", "dictionary": "DICT_6X6_50", "marker_id": 42, "marker_length_m": 0.19, "mount_must_not_change_until_holdout_finish": True},
        "reference_pose": {"source": "fresh_escalated_read_only_S01_O_T_EE", "xyz_base_m": [0.4260011911392212, 0.21128949522972107, 0.8238862752914429], "T_base_ee": ANCHOR},
        "safety": {
            "motion_authorized": True,
            "emergency_stop_reachable": True,
            "full_large_board_bracket_and_cable_sequence_swept_volume_clear": True,
            "target_rigid_confirmed": True,
            "camera_fixed_confirmed": True,
            "board_max_extent_from_eef_m": 0.30,
            "table_height_near_workspace_m": 0.038,
            "minimum_eef_z_m": 0.750,
            "conservative_board_to_table_margin_at_minimum_eef_z_m": 0.412,
            "workspace_bounds_base_m": {"x": [0.350, 0.480], "y": [0.150, 0.290], "z": [0.750, 0.880]},
            "maximum_translation_step_m": 0.075,
            "maximum_rotation_step_deg": 20.0,
            "maximum_translation_norm_from_anchor_m": 0.050,
            "maximum_rotation_angle_from_anchor_deg": 16.0,
            "maximum_translation_velocity_m_s": 0.005,
            "maximum_translation_acceleration_m_s2": 0.00385,
            "maximum_angular_velocity_rad_s": 0.020,
            "maximum_angular_acceleration_rad_s2": 0.0154,
            "maximum_segment_translation_m": 0.020,
            "maximum_segment_rotation_deg": 5.0,
            "minimum_segment_duration_s": 4.0,
            "cartesian_endpoint_hold_s": 6.0,
            "cartesian_pose_controller_mode": "joint_impedance",
            "settle_time_s": 0.5,
        },
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    sequence = []
    for holdout in ("H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09", "H10", "H11", "H12"):
        sequence.extend((("S01", holdout, holdout), (holdout, "S01", None)))
    records = []
    for ordinal, (start, target, capture) in enumerate(sequence, 1):
        active_holdout = target if target.startswith("H") else start
        plan = copy.deepcopy(_base())
        plan["session_slug"] = f"fr3-d435-342222071785-20260813-424x240-60hz-native-holdout-{ordinal:02d}-{start}-to-{target}"
        plan["training_poses"] = [{"id": "S01", "xyz_offset_base_m": POSES["S01"][0], "rotation_vector_eef_deg": POSES["S01"][1]}]
        plan["holdout_poses"] = [{"id": active_holdout, "xyz_offset_base_m": POSES[active_holdout][0], "rotation_vector_eef_deg": POSES[active_holdout][1]}]
        plan["planned_sequence"] = {"single_edge_only": True, "start_pose_id": start, "target_pose_id": target, "independent_holdout_capture_id": capture}
        plan["motion_history"] = []
        plan["motion_authorization"] = {
            "explicit_user_authorization_recorded": True,
            "scope": "single_pose",
            "start_pose_id": start,
            "pose_id": target,
            "source_text": "我固定好了，你自己调整位置，并且复验",
            "authorization_interpretation": "execute_reviewed_native_424x240_holdout_sequence_without_repeated_prompts",
            "consumed": False,
        }
        name = f"{ordinal:02d}-{start}-to-{target}.plan.yaml"
        payload = yaml.safe_dump(plan, sort_keys=False, allow_unicode=True).encode("utf-8")
        path = OUTPUT / name
        _publish_exclusive(path, payload)
        records.append({"ordinal": ordinal, "start": start, "target": target, "capture": capture, "plan": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {"schema_version": 1, "kind": "native_424x240_60hz_holdout_plan_manifest", "hardware_opened": False, "robot_motion_commanded": False, "edges": records}
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _publish_exclusive(OUTPUT / "manifest.json", payload)
    print(json.dumps({"manifest": str(OUTPUT / "manifest.json"), "edge_count": len(records), "sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
