from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import yaml


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_reused_extrinsic_holdout.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_reused_extrinsic_holdout", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _pose(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    angle = np.radians(yaw_deg)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = [x, y, z]
    return transform


def _documents(tmp_path: Path):
    T_base_camera = _pose(0.7, 1.1, 0.9, -80.0)
    T_ee_target = _pose(0.02, -0.01, 0.18, 35.0)
    ee_poses = (
        _pose(0.45, 0.10, 0.65, -15.0),
        _pose(0.51, 0.13, 0.71, 5.0),
        _pose(0.41, 0.18, 0.75, 22.0),
        _pose(0.50, 0.22, 0.68, 40.0),
        _pose(0.43, 0.25, 0.72, 62.0),
        _pose(0.54, 0.16, 0.78, 85.0),
    )
    camera = {
        "name": "Intel RealSense D435",
        "serial": "342222071785",
        "intrinsics": {
            "width": 424,
            "height": 240,
            "fx": 302.3836364746094,
            "fy": 302.2502746582031,
            "ppx": 210.95738220214844,
            "ppy": 123.3145980834961,
        },
    }
    target = {
        "type": "aruco",
        "dictionary": "DICT_6X6_50",
        "marker_id": 42,
        "marker_length_m": 0.19,
    }
    calibration = {
        "schema_version": 1,
        "kind": "camera_robot_extrinsic_calibration",
        "calibration_type": "eye_to_hand",
        "calibration_id": "test",
        "camera": camera,
        "target": target,
        "T_base_camera": T_base_camera.tolist(),
    }
    samples = []
    for index, T_base_ee in enumerate(ee_poses):
        T_camera_target = (
            np.linalg.inv(T_base_camera) @ T_base_ee @ T_ee_target
        )
        samples.append(
            {
                "frame_id": index + 1,
                "T_base_ee": T_base_ee.tolist(),
                "T_camera_target": T_camera_target.tolist(),
            }
        )
    dataset = {
        "schema_version": 1,
        "kind": "eye_to_hand_dataset",
        "camera": camera,
        "target": target,
        "samples": samples,
    }
    calibration_path = tmp_path / "calibration.yaml"
    dataset_path = tmp_path / "holdout.yaml"
    calibration_path.write_text(yaml.safe_dump(calibration), encoding="utf-8")
    dataset_path.write_text(yaml.safe_dump(dataset), encoding="utf-8")
    return calibration_path, dataset_path, dataset


def test_session_reference_is_excluded_and_frozen_extrinsic_passes(tmp_path):
    calibration_path, dataset_path, _dataset = _documents(tmp_path)
    result = MODULE.validate(
        calibration_path,
        dataset_path,
        minimum_holdout_samples=5,
        minimum_translation_span_m=0.08,
        minimum_rotation_span_deg=20.0,
        max_translation_p95_mm=10.0,
        max_rotation_p95_deg=1.5,
    )
    assert result["decision"] == "holdout_pass"
    assert result["dataset"]["reference_sample_index"] == 0
    assert result["dataset"]["holdout_sample_count"] == 5
    assert result["scope"]["T_base_camera_fitted"] is False
    assert result["translation_error_mm"]["max"] < 1.0e-9
    assert result["rotation_error_deg"]["max"] < 1.0e-6


def test_bad_independent_pose_fails_without_refitting_extrinsic(tmp_path):
    calibration_path, dataset_path, dataset = _documents(tmp_path)
    bad = np.asarray(dataset["samples"][-1]["T_camera_target"], dtype=np.float64)
    bad[0, 3] += 0.04
    dataset["samples"][-1]["T_camera_target"] = bad.tolist()
    dataset_path.write_text(yaml.safe_dump(dataset), encoding="utf-8")
    result = MODULE.validate(
        calibration_path,
        dataset_path,
        minimum_holdout_samples=5,
        minimum_translation_span_m=0.08,
        minimum_rotation_span_deg=20.0,
        max_translation_p95_mm=10.0,
        max_rotation_p95_deg=1.5,
    )
    assert result["decision"] == "holdout_fail"
    assert result["translation_error_mm"]["p95"] > 10.0
    assert any("translation p95" in failure for failure in result["failures"])
