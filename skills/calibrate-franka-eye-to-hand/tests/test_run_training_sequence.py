from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import cv2
import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_training_sequence.py"
SPEC = importlib.util.spec_from_file_location("training_sequence_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sequence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sequence
SPEC.loader.exec_module(sequence)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _so3_exp(rotation_vector: Sequence[float]) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.eye(3)
    axis = vector / angle
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _target_from_plan(plan: Mapping[str, Any], pose_id: str) -> np.ndarray:
    anchor = np.asarray(plan["reference_pose"]["T_base_ee"], dtype=np.float64)
    entry = next(item for item in plan["training_poses"] if item["id"] == pose_id)
    target = anchor.copy()
    target[:3, 3] += np.asarray(entry["xyz_offset_base_m"], dtype=np.float64)
    target[:3, :3] = anchor[:3, :3] @ _so3_exp(
        np.radians(entry["rotation_vector_eef_deg"])
    )
    return target


def _sample(T_base_ee: np.ndarray, index: int) -> Dict[str, Any]:
    T_camera_target = np.eye(4)
    T_camera_target[2, 3] = 1.1
    return {
        "timestamp": float(1000 + index),
        "frame_id": index,
        "reprojection_error_px": 0.2,
        "T_base_ee": T_base_ee.tolist(),
        "T_camera_target": T_camera_target.tolist(),
    }


def _build_environment(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    calibration_root = workspace_root / "beta" / "dynamic_object_pcd"
    (calibration_root / "dynamic_pcd" / "apps").mkdir(parents=True)
    (calibration_root / "dynamic_pcd" / "calibration").mkdir(parents=True)
    (calibration_root / "configs").mkdir(parents=True)
    (calibration_root / "calibration_runs").mkdir(parents=True)

    capture_cli = calibration_root / "dynamic_pcd" / "apps" / "calibrate_eye_to_hand.py"
    capture_cli.write_text("# reviewed fake capture cli\n", encoding="utf-8")
    detector = calibration_root / "dynamic_pcd" / "calibration" / "aruco.py"
    detector.write_text("# reviewed fake detector\n", encoding="utf-8")
    stationary = calibration_root / "dynamic_pcd" / "calibration" / "stationary.py"
    stationary.write_text("# reviewed fake stationary aggregator\n", encoding="utf-8")
    transforms = calibration_root / "dynamic_pcd" / "calibration" / "transforms.py"
    transforms.write_text("# reviewed fake calibration transforms\n", encoding="utf-8")
    skill_scripts = (
        workspace_root
        / "skills"
        / "calibrate-franka-eye-to-hand"
        / "scripts"
    )
    skill_scripts.mkdir(parents=True)
    motion_script = skill_scripts / "move_calibration_pose.py"
    motion_script.write_text("# reviewed fake motion wrapper\n", encoding="utf-8")
    pipeline_root = workspace_root / "dexgrasp" / "src" / "anydex_pipeline"
    pipeline_root.mkdir(parents=True)
    motion_driver = pipeline_root / "franka_sequence_driver.py"
    motion_driver.write_text("# reviewed fake motion driver\n", encoding="utf-8")
    link_preflight = pipeline_root / "host_network_preflight.py"
    link_preflight.write_text("# reviewed fake link preflight\n", encoding="utf-8")
    inspection_script = skill_scripts / "inspect_aruco_frame.py"
    inspection_script.write_text("# reviewed fake inspection\n", encoding="utf-8")
    frame_telemetry_script = skill_scripts / "capture_aruco_frame_telemetry.py"
    frame_telemetry_script.write_text(
        "# reviewed fake frame telemetry\n", encoding="utf-8"
    )
    capture_config = calibration_root / "configs" / "d435_default.yaml"
    capture_config.write_text(
        yaml.safe_dump(
            {"camera": {"serial": "old", "width": 848, "height": 480, "fps": 30}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    python_executable = workspace_root / ".venv" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_target = workspace_root / "fake-base-python"
    python_target.write_text("#!/bin/sh\n", encoding="utf-8")
    python_executable.symlink_to(python_target)

    anchor = np.eye(4)
    anchor[:3, 3] = [0.49, 0.17, 0.73]
    offsets = {
        1: ([0.000, 0.000, 0.000], [0.0, 0.0, 0.0]),
        2: ([0.035, 0.000, 0.000], [7.5, 0.0, 0.0]),
        3: ([0.035, 0.000, 0.000], [0.0, 7.5, 0.0]),
        4: ([0.050, 0.000, 0.000], [0.0, 16.0, 0.0]),
        5: ([0.035, 0.025, 0.000], [0.0, 7.5, 0.0]),
        6: ([0.000, 0.050, 0.000], [0.0, 0.0, 0.0]),
        7: ([-0.025, 0.035, 0.000], [0.0, 7.5, 0.0]),
        8: ([-0.050, 0.000, 0.000], [0.0, 16.0, 0.0]),
        9: ([-0.035, -0.025, 0.000], [0.0, 7.5, 0.0]),
        10: ([0.000, -0.050, 0.000], [0.0, 0.0, 0.0]),
        11: ([-0.020, -0.020, 0.020], [0.0, 0.0, 10.0]),
        12: ([-0.020, 0.000, 0.040], [0.0, 5.0, 5.0]),
        13: ([0.000, 0.000, 0.050], [0.0, 10.0, 0.0]),
        14: ([-0.030, 0.000, 0.040], [-7.5, 0.0, 0.0]),
        15: ([0.000, 0.000, 0.000], [-16.0, 0.0, 0.0]),
        16: ([0.000, 0.000, -0.025], [-7.5, 0.0, 0.0]),
        17: ([0.030, 0.000, -0.025], [0.0, 0.0, 0.0]),
        18: ([0.000, -0.030, -0.025], [0.0, 0.0, -10.0]),
        19: ([0.000, 0.000, -0.025], [0.0, -7.5, 0.0]),
        20: ([-0.030, 0.000, -0.025], [0.0, -16.0, 0.0]),
    }
    training_poses = [
        {
            "id": "T{:02d}".format(index),
            "xyz_offset_base_m": offset,
            "rotation_vector_eef_deg": rotation,
            "state": "captured" if index == 1 else "planned",
            **({"sample_index": index} if index == 1 else {}),
        }
        for index, (offset, rotation) in offsets.items()
    ]
    master_path = calibration_root / "calibration_runs" / "master.yaml"
    claims_dir = calibration_root / "calibration_runs" / ".training-edge-claims"
    master = {
        "schema_version": 1,
        "kind": sequence.PLAN_KIND,
        "status": sequence.MASTER_RUN_STATUS,
        "session_slug": "fr3-test-training",
        "provenance": {
            "calibration_root_path": str(calibration_root),
            "master_plan_path": str(master_path),
            "canonical_claims_dir": str(claims_dir),
            "training_orchestrator_sha256": _sha(SCRIPT),
            "python_executable_path": str(python_executable),
            "python_executable_resolved_path": str(python_executable.resolve()),
            "python_executable_sha256": _sha(python_executable),
            "motion_wrapper_sha256": _sha(motion_script),
            "motion_driver_sha256": _sha(motion_driver),
            "link_preflight_sha256": _sha(link_preflight),
            "inspection_script_sha256": _sha(inspection_script),
            "aruco_detector_sha256": _sha(detector),
            "stationary_aggregator_sha256": _sha(stationary),
            "calibration_transforms_sha256": _sha(transforms),
            "capture_config_sha256": _sha(capture_config),
            "capture_cli_sha256": _sha(capture_cli),
            "frame_telemetry_sha256": _sha(frame_telemetry_script),
        },
        "topology": {"type": "eye_to_hand", "camera_mount": "fixed_external"},
        "hardware": {
            "robot": {"type": "Franka Research 3", "ip": "172.16.0.2"},
            "camera": {
                "serial": "342222071785",
                "color_profile": {"width": 848, "height": 480, "fps": 30},
            },
        },
        "target": {
            "type": "aruco",
            "dictionary": "DICT_6X6_50",
            "marker_id": 42,
            "marker_length_m": 0.190,
        },
        "reference_pose": {
            "xyz_base_m": anchor[:3, 3].tolist(),
            "T_base_ee": anchor.tolist(),
        },
        "safety": {
            "motion_authorized": True,
            "emergency_stop_reachable": True,
            "full_large_board_bracket_and_cable_sequence_swept_volume_clear": True,
            "target_rigid_confirmed": True,
            "camera_fixed_confirmed": True,
            "workspace_bounds_base_m": {
                "x": [0.43, 0.55],
                "y": [0.11, 0.23],
                "z": [0.70, 0.79],
            },
            "minimum_eef_z_m": 0.70,
            "maximum_translation_step_m": 0.075,
            "maximum_rotation_step_deg": 20.0,
            "maximum_translation_norm_from_anchor_m": 0.050,
            "maximum_rotation_angle_from_anchor_deg": 16.0,
            "maximum_translation_velocity_m_s": 0.005,
            "maximum_translation_acceleration_m_s2": 0.00385,
            "maximum_angular_velocity_rad_s": 0.020,
            "maximum_angular_acceleration_rad_s2": 0.0154,
            "cartesian_endpoint_hold_s": 6.0,
            "cartesian_pose_controller_mode": "joint_impedance",
            "settle_time_s": 0.5,
        },
        "planned_sequence": {
            "training": ["T{:02d}".format(index) for index in range(1, 21)]
        },
        "collection_state": {
            "current_pose_id": "T01",
            "current_training_sample_count": 1,
            "next_pose_id": "T02",
            "next_capture_index": 2,
            "formal_training_dataset": "calibration_runs/training.yaml",
            "training_sequence_artifacts_dir": (
                "calibration_runs/sequence-artifacts"
            ),
        },
        "training_poses": training_poses,
        "holdout_poses": [
            {
                "id": "H01",
                "xyz_offset_base_m": [0.0, 0.0, 0.02],
                "rotation_vector_eef_deg": [0.0, 5.0, 0.0],
            }
        ],
        "motion_authorization": {
            "explicit_user_authorization_recorded": True,
            "scope": sequence.MASTER_AUTHORIZATION_SCOPE,
            "start_pose_id": "T01",
            "pose_id": "T02",
            "source_text": "continue the reviewed sequence",
            "authorization_interpretation": "continuous reviewed training authorization",
            "authorized_training_suffix": [
                "T{:02d}".format(index) for index in range(2, 21)
            ],
            "authorized_edges": [
                "T{:02d}_to_T{:02d}".format(index, index + 1)
                for index in range(1, 20)
            ],
            "consumed": False,
        },
    }
    dataset = calibration_root / "calibration_runs" / "training.yaml"
    T01 = _target_from_plan(master, "T01")
    dataset_document = {
        "schema_version": 1,
        "kind": "eye_to_hand_dataset",
        "created_at": "2026-08-11T00:00:00+00:00",
        "transform_convention": "T_A_B maps coordinates from frame B to frame A",
        "units": {"translation": "m", "angle": "rad"},
        "camera": {
            "type": "Intel RealSense RGB-D",
            "serial": "342222071785",
            "depth_scale": 0.001,
            "intrinsics": {
                "width": 848,
                "height": 480,
                "fx": 604.7,
                "fy": 604.5,
                "ppx": 421.9,
                "ppy": 246.6,
                "model": "distortion.inverse_brown_conrady",
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
        "target": {
            "type": "aruco",
            "dictionary": "DICT_6X6_50",
            "marker_id": 42,
            "marker_length_m": 0.19,
        },
        "robot": {"type": "Franka", "ip": "172.16.0.2", "pose": "O_T_EE"},
        "samples": [_sample(T01, 1)],
    }
    dataset.write_text(
        yaml.safe_dump(dataset_document, sort_keys=False), encoding="utf-8"
    )
    master["collection_state"]["initial_dataset_sha256"] = _sha(dataset)
    master_path.write_text(
        yaml.safe_dump(master, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    config = sequence.OrchestratorConfig(
        master_plan=master_path,
        expected_master_sha256=_sha(master_path),
        dataset=dataset,
        artifacts_dir=calibration_root / "calibration_runs" / "sequence-artifacts",
        calibration_root=calibration_root,
        capture_config=capture_config,
        motion_script=motion_script,
        motion_driver=motion_driver,
        link_preflight=link_preflight,
        inspection_script=inspection_script,
        frame_telemetry_script=frame_telemetry_script,
        python_executable=str(python_executable),
    )
    return config, master, dataset


class FakeSubprocessRunner:
    def __init__(
        self,
        dataset: Path,
        *,
        fail_phase: Optional[str] = None,
        append_count: int = 1,
        mutate_prefix: bool = False,
        coherent_frames: int = 114,
        telemetry_success: bool = True,
        invalid_debug_image: bool = False,
        inspection_margin_px: float = 30.0,
        inspection_edge_px: float = 90.0,
        inspection_fx: float = 604.7,
        inspection_depth_scale: float = 0.001,
        frame_telemetry_detected: int = 118,
        frame_telemetry_reprojection_pass: int = 118,
        frame_telemetry_coherent: int = 114,
        frame_telemetry_translation_p95_m: float = 0.001,
        frame_telemetry_rotation_p95_deg: float = 0.3,
        frame_telemetry_reprojection_p95_px: float = 0.5,
        frame_telemetry_serial: str = "342222071785",
        frame_telemetry_fx: float = 604.7,
        frame_telemetry_depth_scale: float = 0.001,
        bad_frame_telemetry_json: bool = False,
        capture_untrimmed_requested: int = 120,
        capture_untrimmed_pass: int = 118,
        capture_untrimmed_translation_p95_m: float = 0.001,
        capture_untrimmed_rotation_p95_deg: float = 0.3,
        capture_untrimmed_reprojection_p95_px: float = 0.5,
        capture_untrimmed_sentinel_count: int = 1,
        bad_capture_untrimmed_json: bool = False,
        duplicate_capture_untrimmed_key: bool = False,
        mutate_after_preview: Optional[Path] = None,
        mutate_after_telemetry: Optional[Path] = None,
        mutate_plan_after_telemetry: bool = False,
        mutate_claim_after_telemetry: bool = False,
        mutate_after_capture: Optional[Path] = None,
        import_probe_capture_path: Optional[str] = None,
        import_probe_detector_path: Optional[str] = None,
        import_probe_stationary_path: Optional[str] = None,
        import_probe_transforms_path: Optional[str] = None,
    ):
        self.dataset = dataset
        self.fail_phase = fail_phase
        self.append_count = append_count
        self.mutate_prefix = mutate_prefix
        self.coherent_frames = coherent_frames
        self.telemetry_success = telemetry_success
        self.invalid_debug_image = invalid_debug_image
        self.inspection_margin_px = inspection_margin_px
        self.inspection_edge_px = inspection_edge_px
        self.inspection_fx = inspection_fx
        self.inspection_depth_scale = inspection_depth_scale
        self.frame_telemetry_detected = frame_telemetry_detected
        self.frame_telemetry_reprojection_pass = frame_telemetry_reprojection_pass
        self.frame_telemetry_coherent = frame_telemetry_coherent
        self.frame_telemetry_translation_p95_m = frame_telemetry_translation_p95_m
        self.frame_telemetry_rotation_p95_deg = frame_telemetry_rotation_p95_deg
        self.frame_telemetry_reprojection_p95_px = frame_telemetry_reprojection_p95_px
        self.frame_telemetry_serial = frame_telemetry_serial
        self.frame_telemetry_fx = frame_telemetry_fx
        self.frame_telemetry_depth_scale = frame_telemetry_depth_scale
        self.bad_frame_telemetry_json = bad_frame_telemetry_json
        self.capture_untrimmed_requested = capture_untrimmed_requested
        self.capture_untrimmed_pass = capture_untrimmed_pass
        self.capture_untrimmed_translation_p95_m = (
            capture_untrimmed_translation_p95_m
        )
        self.capture_untrimmed_rotation_p95_deg = capture_untrimmed_rotation_p95_deg
        self.capture_untrimmed_reprojection_p95_px = (
            capture_untrimmed_reprojection_p95_px
        )
        self.capture_untrimmed_sentinel_count = capture_untrimmed_sentinel_count
        self.bad_capture_untrimmed_json = bad_capture_untrimmed_json
        self.duplicate_capture_untrimmed_key = duplicate_capture_untrimmed_key
        self.mutate_after_preview = mutate_after_preview
        self.mutate_after_telemetry = mutate_after_telemetry
        self.mutate_plan_after_telemetry = mutate_plan_after_telemetry
        self.mutate_claim_after_telemetry = mutate_claim_after_telemetry
        self.mutate_after_capture = mutate_after_capture
        self.import_probe_capture_path = import_probe_capture_path
        self.import_probe_detector_path = import_probe_detector_path
        self.import_probe_stationary_path = import_probe_stationary_path
        self.import_probe_transforms_path = import_probe_transforms_path
        self.commands = []
        self.phases = []
        self.environments = []
        self.last_target: Optional[np.ndarray] = None
        self.current_plan_path: Optional[Path] = None

    @staticmethod
    def _value(argv: Sequence[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        phase: str,
        env: Optional[Mapping[str, str]] = None,
    ):
        command = tuple(str(value) for value in argv)
        self.commands.append(command)
        self.phases.append(phase)
        self.environments.append(None if env is None else dict(env))
        assert env is not None
        assert env["PYTHONPATH"] == str(cwd.resolve())
        assert env["PYTHONNOUSERSITE"] == "1"
        assert "PYTHONHOME" not in env
        if phase == "import_preflight":
            capture_path = (
                self.import_probe_capture_path
                or str(
                    (
                        cwd
                        / "dynamic_pcd"
                        / "apps"
                        / "calibrate_eye_to_hand.py"
                    ).resolve()
                )
            )
            detector_path = (
                self.import_probe_detector_path
                or str((cwd / "dynamic_pcd" / "calibration" / "aruco.py").resolve())
            )
            stationary_path = (
                self.import_probe_stationary_path
                or str(
                    (cwd / "dynamic_pcd" / "calibration" / "stationary.py").resolve()
                )
            )
            transforms_path = (
                self.import_probe_transforms_path
                or str(
                    (cwd / "dynamic_pcd" / "calibration" / "transforms.py").resolve()
                )
            )
            payload = {
                "dynamic_pcd.apps.calibrate_eye_to_hand": capture_path,
                "dynamic_pcd.calibration.aruco": detector_path,
                "dynamic_pcd.calibration.stationary": stationary_path,
                "dynamic_pcd.calibration.transforms": transforms_path,
            }
            return sequence.CommandResult(
                command,
                0,
                sequence._IMPORT_PROBE_SENTINEL
                + json.dumps(payload, sort_keys=True, separators=(",", ":"))
                + "\n",
            )
        if ":preview" in phase:
            plan_path = Path(self._value(command, "--plan"))
            self.current_plan_path = plan_path
            plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
            pose_id = self._value(command, "--pose-id")
            plan_sha = _sha(plan_path)
            assert self._value(command, "--expect-plan-sha256") == plan_sha
            assert (
                not list(
                    (cwd / "calibration_runs" / ".training-edge-claims").glob(
                        "*.claim.json"
                    )
                )
                or len(self.commands) > 4
            )
            self.last_target = _target_from_plan(plan, pose_id)
            if self.fail_phase == "preview":
                return sequence.CommandResult(command, 2, "preview rejected\n")
            payload = {
                "plan": str(plan_path.resolve()),
                "plan_sha256": plan_sha,
                "pose_id": pose_id,
                "start_pose_id": plan["motion_authorization"]["start_pose_id"],
                "status": sequence.RUN_STATUS,
                "run_authorized": True,
                "dynamics_ready": True,
                "camera_capture_performed": False,
                "target_T_base_ee": self.last_target.tolist(),
            }
            if self.mutate_after_preview is not None:
                self.mutate_after_preview.write_text(
                    "# mutated after preview\n", encoding="utf-8"
                )
            return sequence.CommandResult(command, 0, json.dumps(payload))

        if ":motion" in phase:
            plan_path = Path(self._value(command, "--plan"))
            plan_sha = _sha(plan_path)
            pose_id = self._value(command, "--pose-id")
            assert list(
                (cwd / "calibration_runs" / ".training-edge-claims").glob(
                    "*.claim.json"
                )
            ), "canonical claim must precede motion"
            assert self._value(command, "--confirm-plan-sha256") == plan_sha
            assert self._value(command, "--confirm-pose-id") == pose_id
            assert self._value(command, "--confirm-e-stop") == sequence.E_STOP_TOKEN
            assert (
                self._value(command, "--confirm-swept-volume")
                == sequence.SWEPT_VOLUME_TOKEN
            )
            assert (
                self._value(command, "--confirm-target-rigid")
                == sequence.TARGET_RIGID_TOKEN
            )
            assert (
                self._value(command, "--confirm-camera-fixed")
                == sequence.CAMERA_FIXED_TOKEN
            )
            if self.fail_phase == "motion":
                return sequence.CommandResult(command, 1, "motion rejected\n")
            return sequence.CommandResult(
                command,
                0,
                "[calibration pose] CAPTURE_READY pose={} plan_sha256={} "
                "camera_capture_performed=false control_loop_telemetry={}\n".format(
                    pose_id,
                    plan_sha,
                    json.dumps(
                        {
                            "kind": "cartesian_pose",
                            "samples": 1000,
                            "max_read_to_write_ns": 100000,
                            "read_to_write_overruns": 0,
                            "success_qualified": self.telemetry_success,
                            "success_qualification_positive_writes": 100,
                            "success_qualification_control_time_s": 0.1,
                            "success_qualification_wall_time_s": 0.1,
                            "success_qualification_rate": 1000.0,
                        },
                        separators=(",", ":"),
                    ),
                ),
            )

        if ":inspect" in phase:
            raw_path = Path(self._value(command, "--raw-output"))
            annotated_path = Path(self._value(command, "--annotated-output"))
            image = np.zeros((480, 848, 3), dtype=np.uint8)
            assert cv2.imwrite(str(raw_path), image)
            assert cv2.imwrite(str(annotated_path), image)
            payload = {
                "camera_name": "Intel RealSense D435",
                "camera_serial": "342222071785",
                "frame_id": 10,
                "depth_scale": self.inspection_depth_scale,
                "intrinsics": {
                    "width": 848,
                    "height": 480,
                    "fx": self.inspection_fx,
                    "fy": 604.5,
                    "ppx": 421.9,
                    "ppy": 246.6,
                    "model": "distortion.inverse_brown_conrady",
                    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "image_width": 848,
                "image_height": 480,
                "fx": self.inspection_fx,
                "fy": 604.5,
                "focus_laplacian_variance": 100.0,
                "raw_output": str(raw_path.resolve()),
                "annotated_output": str(annotated_path.resolve()),
                "detections": [
                    {
                        "dictionary": "DICT_6X6_50",
                        "marker_id": 42,
                        "edge_min_px": self.inspection_edge_px,
                        "minimum_image_margin_px": self.inspection_margin_px,
                        "complete_in_image": True,
                    }
                ],
            }
            return sequence.CommandResult(
                command,
                0,
                "[RealSense] started\nARUCO_FRAME_INSPECTION_JSON="
                + json.dumps(payload, separators=(",", ":"))
                + "\n",
            )

        if ":telemetry" in phase:
            assert self._value(command, "--frames") == "120"
            assert self._value(command, "--max-reprojection-error-px") == "0.5"
            assert self._value(command, "--max-translation-deviation-m") == "0.001"
            assert self._value(command, "--max-rotation-deviation-deg") == "0.3"
            assert self._value(command, "--camera-serial") == "342222071785"
            output_path = Path(self._value(command, "--output"))
            if self.fail_phase == "telemetry_process":
                return sequence.CommandResult(command, 2, "telemetry rejected\n")
            if self.bad_frame_telemetry_json:
                output_path.write_text("{not-json\n", encoding="utf-8")
                return sequence.CommandResult(
                    command,
                    0,
                    "ARUCO_FRAME_TELEMETRY_JSON={}\n".format(
                        json.dumps(
                            {
                                "output": str(output_path.resolve()),
                                "camera_serial": self.frame_telemetry_serial,
                            },
                            separators=(",", ":"),
                        )
                    ),
                )

            detected = self.frame_telemetry_detected
            reprojection_pass = self.frame_telemetry_reprojection_pass
            coherent = self.frame_telemetry_coherent
            next_index = 0
            invalid_indices = list(range(next_index, next_index + 120 - detected))
            next_index += len(invalid_indices)
            reprojection_indices = list(
                range(next_index, next_index + detected - reprojection_pass)
            )
            next_index += len(reprojection_indices)
            coherent_indices = list(range(next_index, next_index + coherent))
            next_index += len(coherent_indices)
            coherence_indices = list(
                range(next_index, next_index + reprojection_pass - coherent)
            )
            assert next_index + len(coherence_indices) == 120
            payload = {
                "schema_version": 1,
                "kind": "realsense_aruco_frame_telemetry",
                "scope": {
                    "camera_only": True,
                    "read_only": True,
                    "franka_fci_opened": False,
                    "robot_motion_commanded": False,
                },
                "camera": {
                    "requested_serial": self.frame_telemetry_serial,
                    "opened_serial": self.frame_telemetry_serial,
                    "name": "Fake RealSense",
                    "depth_scale": self.frame_telemetry_depth_scale,
                    "intrinsics": {
                        "width": 848,
                        "height": 480,
                        "fx": self.frame_telemetry_fx,
                        "fy": 604.5,
                        "ppx": 421.9,
                        "ppy": 246.6,
                        "model": "distortion.inverse_brown_conrady",
                        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                    "intrinsics_consistent_across_frames": True,
                    "config_path": str(Path(self._value(command, "--config")).resolve()),
                },
                "target": {
                    "dictionary": "DICT_6X6_50",
                    "marker_id": 42,
                    "marker_length_m": 0.19,
                },
                "capture": {
                    "requested_frame_count": 120,
                    "captured_frame_count": 120,
                },
                "thresholds": {
                    "max_reprojection_error_px": 0.5,
                    "max_translation_deviation_m": 0.001,
                    "max_rotation_deviation_deg": 0.3,
                    "minimum_focus_laplacian_variance": None,
                },
                "coherence": {
                    "detected_valid_count": detected,
                    "reprojection_gate_pass_count": reprojection_pass,
                    "coherent_pose_count": coherent,
                    "coherent_pose_fraction": coherent / 120.0,
                    "medoid_capture_index": coherent_indices[0],
                    "coherent_capture_indices": coherent_indices,
                    "coherence_outlier_capture_indices": coherence_indices,
                    "invalid_detection_capture_indices": invalid_indices,
                    "reprojection_outlier_capture_indices": reprojection_indices,
                    "aggregate_T_camera_target": np.eye(4).tolist(),
                    "translation_jitter_p95_m": 0.0004,
                    "rotation_jitter_p95_deg": 0.2,
                    "inlier_reprojection_error_p95_px": 0.3,
                    "all_reprojection_pass_translation_jitter_p95_m": (
                        self.frame_telemetry_translation_p95_m
                    ),
                    "all_reprojection_pass_rotation_jitter_p95_deg": (
                        self.frame_telemetry_rotation_p95_deg
                    ),
                    "all_reprojection_pass_reprojection_error_p95_px": (
                        self.frame_telemetry_reprojection_p95_px
                    ),
                },
                "abnormal_images": [],
                "frames": [{"capture_index": index} for index in range(120)],
            }
            output_path.write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            summary = {
                "output": str(output_path.resolve()),
                "camera_serial": self.frame_telemetry_serial,
                "captured": 120,
                "valid": detected,
                "reprojection_gate_pass": reprojection_pass,
                "coherent": coherent,
                "abnormal_images": 0,
            }
            telemetry_result = sequence.CommandResult(
                command,
                0,
                "ARUCO_FRAME_TELEMETRY_JSON={}\n".format(
                    json.dumps(summary, sort_keys=True, separators=(",", ":"))
                ),
            )
            mutation_path = self.mutate_after_telemetry
            if self.mutate_plan_after_telemetry:
                assert self.current_plan_path is not None
                mutation_path = self.current_plan_path
            if self.mutate_claim_after_telemetry:
                claims = list(
                    (cwd / "calibration_runs" / ".training-edge-claims").glob(
                        "*.claim.json"
                    )
                )
                assert len(claims) == 1
                mutation_path = claims[0]
            if mutation_path is not None:
                mutation_path.chmod(0o644)
                mutation_path.write_bytes(
                    mutation_path.read_bytes() + b"\n# mutated after telemetry\n"
                )
            return telemetry_result

        assert ":capture" in phase
        if self.fail_phase == "capture_process":
            return sequence.CommandResult(command, 2, "capture rejected\n")
        assert self._value(command, "--frames") == "120"
        assert self._value(command, "--min-valid-frame-fraction") == "0.95"
        assert self._value(command, "--min-valid-frames") == "114"
        assert self._value(command, "--max-reprojection-error") == "0.5"
        assert self._value(command, "--max-target-translation-jitter") == "0.001"
        assert self._value(command, "--max-target-rotation-jitter") == "0.3"
        assert self._value(command, "--min-all-reprojection-pass-frames") == "118"
        assert (
            self._value(
                command, "--max-all-reprojection-pass-translation-jitter"
            )
            == "0.001"
        )
        assert (
            self._value(command, "--max-all-reprojection-pass-rotation-jitter")
            == "0.3"
        )
        assert (
            self._value(command, "--max-all-reprojection-pass-reprojection-error")
            == "0.5"
        )
        assert self._value(command, "--min-pose-translation") == "0.01"
        assert self._value(command, "--min-pose-rotation") == "3.0"
        assert command.count("0.01") == 1
        assert self._value(command, "--camera-serial") == "342222071785"
        assert self.last_target is not None
        document = yaml.safe_load(self.dataset.read_text(encoding="utf-8"))
        if self.mutate_prefix:
            document["samples"][0]["frame_id"] = 999
        initial_count = len(document["samples"])
        formal_untrimmed_pass = (
            self.capture_untrimmed_requested == 120
            and self.capture_untrimmed_pass >= 118
            and self.capture_untrimmed_pass <= 120
            and 0.0 <= self.capture_untrimmed_translation_p95_m <= 0.001
            and 0.0 <= self.capture_untrimmed_rotation_p95_deg <= 0.3
            and 0.0 <= self.capture_untrimmed_reprojection_p95_px <= 0.5
        )
        if self.coherent_frames >= 114 and formal_untrimmed_pass:
            for addition in range(self.append_count):
                document["samples"].append(
                    _sample(self.last_target, initial_count + addition + 1)
                )
            document["created_at"] = "2026-08-11T01:00:00+00:00"
            self.dataset.write_text(
                yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
            )
        debug_path = Path(self._value(command, "--debug-image"))
        if self.invalid_debug_image:
            debug_path.write_bytes(b"not-an-image")
        else:
            assert cv2.imwrite(str(debug_path), np.zeros((480, 848, 3), dtype=np.uint8))
        reported_count = initial_count + 1
        output = (
            "Captured sample {}: coherent={}/120, target jitter p95=1.00 mm/"
            "0.300 deg, reprojection p95=0.500px; EEF xyz=[0 0 0]\n"
        ).format(reported_count, self.coherent_frames)
        if self.bad_capture_untrimmed_json:
            sentinel_payload = "{not-json"
        elif self.duplicate_capture_untrimmed_key:
            sentinel_payload = (
                '{"requested_frame_count":120,"requested_frame_count":120,'
                '"all_reprojection_pass_count":118,'
                '"all_reprojection_pass_translation_jitter_p95_m":0.001,'
                '"all_reprojection_pass_rotation_jitter_p95_deg":0.3,'
                '"all_reprojection_pass_reprojection_error_p95_px":0.5}'
            )
        else:
            sentinel_payload = json.dumps(
                {
                    "requested_frame_count": self.capture_untrimmed_requested,
                    "all_reprojection_pass_count": self.capture_untrimmed_pass,
                    "all_reprojection_pass_translation_jitter_p95_m": (
                        self.capture_untrimmed_translation_p95_m
                    ),
                    "all_reprojection_pass_rotation_jitter_p95_deg": (
                        self.capture_untrimmed_rotation_p95_deg
                    ),
                    "all_reprojection_pass_reprojection_error_p95_px": (
                        self.capture_untrimmed_reprojection_p95_px
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        output += (
            sequence._CAPTURE_UNTRIMMED_SENTINEL + sentinel_payload + "\n"
        ) * self.capture_untrimmed_sentinel_count
        if self.mutate_after_capture is not None:
            self.mutate_after_capture.chmod(0o644)
            self.mutate_after_capture.write_bytes(
                self.mutate_after_capture.read_bytes() + b"\n# mutated after capture\n"
            )
        return sequence.CommandResult(command, 0, output)


def test_happy_path_runs_exact_T02_through_T20_with_five_bound_stages(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    runner = FakeSubprocessRunner(dataset)

    result = sequence.run_sequence(
        config, runner=runner, clock=lambda: "2026-08-11T00:00:00+00:00"
    )

    assert result.completed_pose_ids == tuple("T{:02d}".format(i) for i in range(2, 21))
    assert result.final_sample_count == 20
    assert len(yaml.safe_load(dataset.read_text())["samples"]) == 20
    assert len(runner.commands) == 1 + 19 * 5
    assert runner.phases[0] == "import_preflight"
    assert [phase.rsplit(":", 1)[-1] for phase in runner.phases[1:6]] == [
        "preview",
        "motion",
        "inspect",
        "telemetry",
        "capture",
    ]

    edges = config.artifacts_dir / "edges"
    plans = sorted(edges.glob("*.plan.yaml"))
    assert len(plans) == 19
    first_plan = yaml.safe_load(
        next(path for path in plans if path.name.startswith("T01-to-T02-")).read_text()
    )
    assert [entry["id"] for entry in first_plan["training_poses"]] == ["T01", "T02"]
    assert first_plan["holdout_poses"] == []
    assert first_plan["planned_sequence"]["holdout_poses_included"] is False
    assert first_plan["motion_authorization"]["scope"] == "single_pose"
    assert first_plan["motion_authorization"]["start_pose_id"] == "T01"
    assert first_plan["motion_authorization"]["pose_id"] == "T02"
    assert first_plan["motion_authorization"]["source_authorized_training_suffix"] == [
        "T{:02d}".format(index) for index in range(2, 21)
    ]
    claims_dir = config.calibration_root / "calibration_runs" / ".training-edge-claims"
    assert len(list(claims_dir.glob("*.claim.json"))) == 19
    assert len(list(edges.glob("*.complete.json"))) == 19
    assert len(list(edges.glob("*.frame-telemetry.json"))) == 19
    assert not list(edges.glob("*.failed.json"))
    assert result.completion_receipt.exists()

    capture_commands = [
        command
        for command, phase in zip(runner.commands, runner.phases)
        if phase.endswith(":capture")
    ]
    assert len(capture_commands) == 19
    assert all("H01" not in " ".join(command) for command in runner.commands)
    first_receipt = json.loads(
        next(path for path in edges.glob("T01-to-T02-*.complete.json")).read_text()
    )
    telemetry_receipt = first_receipt["independent_frame_telemetry"]
    assert telemetry_receipt["detected_valid_count"] == 118
    assert telemetry_receipt["coherent_pose_count"] == 114
    assert telemetry_receipt[
        "all_reprojection_pass_translation_jitter_p95_m"
    ] == 0.001
    formal_untrimmed = first_receipt["capture_metrics"][
        "formal_same_batch_untrimmed"
    ]
    assert formal_untrimmed == {
        "requested_frame_count": 120,
        "all_reprojection_pass_count": 118,
        "all_reprojection_pass_translation_jitter_p95_m": 0.001,
        "all_reprojection_pass_rotation_jitter_p95_deg": 0.3,
        "all_reprojection_pass_reprojection_error_p95_px": 0.5,
    }
    assert first_receipt["frame_telemetry_executable"]["sha256"] == _sha(
        config.frame_telemetry_script
    )
    orchestrator_provenance = first_plan["provenance"][
        "headless_training_orchestrator"
    ]
    assert orchestrator_provenance["frame_telemetry_script"] == str(
        config.frame_telemetry_script.resolve()
    )
    assert orchestrator_provenance["frame_telemetry_sha256"] == _sha(
        config.frame_telemetry_script
    )


def test_strict_resume_from_captured_T06_runs_only_T07_through_T20(tmp_path):
    config, master, dataset = _build_environment(tmp_path)
    document = yaml.safe_load(dataset.read_text(encoding="utf-8"))
    for index in range(2, 7):
        pose_id = "T{:02d}".format(index)
        document["samples"].append(_sample(_target_from_plan(master, pose_id), index))
        pose = next(item for item in master["training_poses"] if item["id"] == pose_id)
        pose["state"] = "captured"
        pose["sample_index"] = index
    dataset.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    master["collection_state"]["initial_dataset_sha256"] = _sha(dataset)
    master["collection_state"].update(
        {
            "current_pose_id": "T06",
            "current_training_sample_count": 6,
            "next_pose_id": "T07",
            "next_capture_index": 7,
        }
    )
    master["motion_authorization"].update(
        {
            "start_pose_id": "T06",
            "pose_id": "T07",
            "authorized_training_suffix": [
                "T{:02d}".format(index) for index in range(7, 21)
            ],
            "authorized_edges": [
                "T{:02d}_to_T{:02d}".format(index, index + 1)
                for index in range(6, 20)
            ],
        }
    )
    config.master_plan.write_text(
        yaml.safe_dump(master, sort_keys=False), encoding="utf-8"
    )
    config = replace(config, expected_master_sha256=_sha(config.master_plan))
    runner = FakeSubprocessRunner(dataset)

    result = sequence.run_sequence(config, runner=runner)

    assert result.completed_pose_ids == tuple(
        "T{:02d}".format(index) for index in range(7, 21)
    )
    assert len(runner.commands) == 1 + 14 * 5
    assert len(yaml.safe_load(dataset.read_text())["samples"]) == 20


def test_preview_failure_stops_before_claim_or_motion(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    runner = FakeSubprocessRunner(dataset, fail_phase="preview")

    with pytest.raises(sequence.SequenceFailure, match="subprocess exited") as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "preview"
    assert len(runner.commands) == 2
    edges = config.artifacts_dir / "edges"
    assert not list(
        (config.calibration_root / "calibration_runs" / ".training-edge-claims").glob(
            "*.claim.json"
        )
    )
    assert len(list(edges.glob("*.failed.json"))) == 1
    assert len(yaml.safe_load(dataset.read_text())["samples"]) == 1


def test_motion_failure_keeps_claim_and_never_captures_or_advances(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    runner = FakeSubprocessRunner(dataset, fail_phase="motion")

    with pytest.raises(sequence.SequenceFailure) as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "motion"
    assert len(runner.commands) == 3
    edges = config.artifacts_dir / "edges"
    claims = list(
        (config.calibration_root / "calibration_runs" / ".training-edge-claims").glob(
            "*.claim.json"
        )
    )
    assert len(claims) == 1
    claim = json.loads(claims[0].read_text())
    assert claim["state"] == "claimed_before_motion"
    assert claim["edge"] == "T01_to_T02"
    assert len(list(edges.glob("*.failed.json"))) == 1
    assert len(yaml.safe_load(dataset.read_text())["samples"]) == 1

    second_runner = FakeSubprocessRunner(dataset)
    with pytest.raises(sequence.SequenceFailure):
        sequence.run_sequence(config, runner=second_runner)
    assert second_runner.commands == []


def test_unqualified_motion_telemetry_stops_before_inspection(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    runner = FakeSubprocessRunner(dataset, telemetry_success=False)

    with pytest.raises(sequence.SequenceFailure, match="not qualified") as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "motion"
    assert len(runner.commands) == 3
    assert len(
        list(
            (
                config.calibration_root
                / "calibration_runs"
                / ".training-edge-claims"
            ).glob("*.claim.json")
        )
    ) == 1


@pytest.mark.parametrize(
    "runner_kwargs,error_text",
    [
        ({"inspection_margin_px": 19.9}, "marker margin"),
        ({"inspection_edge_px": 79.9}, "marker edge"),
        ({"inspection_fx": 605.0}, "intrinsic fx changed"),
        ({"inspection_depth_scale": 0.002}, "depth scale changed"),
    ],
)
def test_live_inspection_gate_stops_before_capture_and_preserves_dataset(
    tmp_path, runner_kwargs, error_text
):
    config, _, dataset = _build_environment(tmp_path)
    before = dataset.read_bytes()
    runner = FakeSubprocessRunner(dataset, **runner_kwargs)

    with pytest.raises(sequence.SequenceFailure, match=error_text) as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "inspect"
    assert len(runner.commands) == 4
    assert dataset.read_bytes() == before
    assert len(
        list(
            (
                config.calibration_root
                / "calibration_runs"
                / ".training-edge-claims"
            ).glob("*.claim.json")
        )
    ) == 1


@pytest.mark.parametrize(
    "runner_kwargs,error_text",
    [
        (
            {
                "frame_telemetry_detected": 117,
                "frame_telemetry_reprojection_pass": 117,
            },
            "valid detections 117/120",
        ),
        (
            {
                "frame_telemetry_detected": 118,
                "frame_telemetry_reprojection_pass": 117,
            },
            "reprojection-pass frames 117/120",
        ),
        ({"frame_telemetry_coherent": 113}, "coherent frames 113/120"),
        (
            {"frame_telemetry_translation_p95_m": 0.001001},
            "translation_jitter_p95_m",
        ),
        (
            {"frame_telemetry_rotation_p95_deg": 0.301},
            "rotation_jitter_p95_deg",
        ),
        (
            {"frame_telemetry_reprojection_p95_px": 0.501},
            "reprojection_error_p95_px",
        ),
    ],
)
def test_independent_telemetry_gates_stop_before_capture_and_preserve_dataset(
    tmp_path, runner_kwargs, error_text
):
    config, _, dataset = _build_environment(tmp_path)
    before = dataset.read_bytes()
    runner = FakeSubprocessRunner(dataset, **runner_kwargs)

    with pytest.raises(sequence.SequenceFailure, match=error_text) as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "telemetry"
    assert len(runner.commands) == 5
    assert runner.phases[-1].endswith(":telemetry")
    assert not any(phase.endswith(":capture") for phase in runner.phases)
    assert dataset.read_bytes() == before


@pytest.mark.parametrize(
    "runner_kwargs,error_text",
    [
        ({"frame_telemetry_serial": "wrong-camera"}, "wrong camera serial"),
        ({"frame_telemetry_fx": 605.0}, "intrinsic fx changed"),
        ({"frame_telemetry_depth_scale": 0.002}, "depth scale changed"),
        ({"bad_frame_telemetry_json": True}, "invalid JSON"),
    ],
)
def test_bad_telemetry_identity_or_json_never_reaches_capture(
    tmp_path, runner_kwargs, error_text
):
    config, _, dataset = _build_environment(tmp_path)
    before = dataset.read_bytes()
    runner = FakeSubprocessRunner(dataset, **runner_kwargs)

    with pytest.raises(sequence.SequenceFailure, match=error_text) as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "telemetry"
    assert len(runner.commands) == 5
    assert not any(phase.endswith(":capture") for phase in runner.phases)
    assert dataset.read_bytes() == before


@pytest.mark.parametrize(
    "append_count,mutate_prefix,error_text",
    [
        (2, False, "exactly 2 samples"),
        (1, True, "changed pre-existing dataset sample"),
    ],
)
def test_dataset_transaction_violation_stops_after_first_capture(
    tmp_path, append_count, mutate_prefix, error_text
):
    config, _, dataset = _build_environment(tmp_path)
    runner = FakeSubprocessRunner(
        dataset, append_count=append_count, mutate_prefix=mutate_prefix
    )

    with pytest.raises(sequence.SequenceFailure, match=error_text) as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "capture"
    assert len(runner.commands) == 6
    edges = config.artifacts_dir / "edges"
    assert len(
        list(
            (
                config.calibration_root
                / "calibration_runs"
                / ".training-edge-claims"
            ).glob("*.claim.json")
        )
    ) == 1
    assert len(list(edges.glob("*.failed.json"))) == 1
    assert not list(edges.glob("*.complete.json"))


def test_collector_113_of_120_summary_is_rejected_without_dataset_change(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    before = dataset.read_bytes()
    runner = FakeSubprocessRunner(dataset, coherent_frames=113)

    with pytest.raises(sequence.SequenceFailure, match="114/120") as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "capture"
    assert len(runner.commands) == 6
    assert dataset.read_bytes() == before


@pytest.mark.parametrize(
    "runner_kwargs,error_text",
    [
        (
            {"capture_untrimmed_pass": 117},
            "reprojection-pass frames must be at least 118/120",
        ),
        (
            {"capture_untrimmed_translation_p95_m": 0.001001},
            "translation_jitter_p95_m exceeded",
        ),
        (
            {"capture_untrimmed_rotation_p95_deg": 0.301},
            "rotation_jitter_p95_deg exceeded",
        ),
        (
            {"capture_untrimmed_reprojection_p95_px": 0.501},
            "reprojection_error_p95_px exceeded",
        ),
    ],
)
def test_formal_same_batch_untrimmed_gate_rejects_before_dataset_append(
    tmp_path, runner_kwargs, error_text
):
    config, _, dataset = _build_environment(tmp_path)
    before = dataset.read_bytes()
    runner = FakeSubprocessRunner(dataset, **runner_kwargs)

    with pytest.raises(sequence.SequenceFailure, match=error_text) as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "capture"
    assert len(runner.commands) == 6
    assert runner.phases[-1].endswith(":capture")
    assert dataset.read_bytes() == before


@pytest.mark.parametrize(
    "runner_kwargs,error_text",
    [
        ({"capture_untrimmed_sentinel_count": 0}, "exactly one"),
        ({"capture_untrimmed_sentinel_count": 2}, "exactly one"),
        ({"bad_capture_untrimmed_json": True}, "invalid JSON"),
        ({"duplicate_capture_untrimmed_key": True}, "invalid JSON"),
        ({"capture_untrimmed_requested": 119}, "frame count must be 120"),
    ],
)
def test_formal_capture_requires_one_strict_untrimmed_sentinel(
    tmp_path, runner_kwargs, error_text
):
    config, _, dataset = _build_environment(tmp_path)
    before = dataset.read_bytes()
    runner = FakeSubprocessRunner(dataset, **runner_kwargs)

    with pytest.raises(sequence.SequenceFailure, match=error_text) as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "capture"
    assert not list((config.artifacts_dir / "edges").glob("*.complete.json"))
    if runner_kwargs.get("capture_untrimmed_requested") == 119:
        assert dataset.read_bytes() == before


def test_capture_debug_image_must_be_decodable(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    runner = FakeSubprocessRunner(dataset, invalid_debug_image=True)

    with pytest.raises(sequence.SequenceFailure, match="decodable image") as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "capture"
    assert len(runner.commands) == 6


def test_master_hash_mismatch_has_no_subprocess_or_artifact(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    config = replace(config, expected_master_sha256="0" * 64)
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="master SHA mismatch"):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []
    assert not config.artifacts_dir.exists()


def test_initial_dataset_sha_mismatch_has_zero_subprocess_and_zero_motion(tmp_path):
    config, master, dataset = _build_environment(tmp_path)
    master["collection_state"]["initial_dataset_sha256"] = "f" * 64
    config.master_plan.write_text(
        yaml.safe_dump(master, sort_keys=False), encoding="utf-8"
    )
    config = replace(config, expected_master_sha256=_sha(config.master_plan))
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="initial_dataset_sha256"):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []
    assert not config.artifacts_dir.exists()


@pytest.mark.parametrize(
    "artifact_field",
    [
        "motion_driver",
        "link_preflight",
        "inspection_script",
        "capture_config",
        "frame_telemetry_script",
    ],
)
def test_reviewed_motion_provenance_drift_is_rejected_before_subprocess(
    tmp_path, artifact_field
):
    config, _, dataset = _build_environment(tmp_path)
    artifact = getattr(config, artifact_field)
    artifact.write_text("# changed after review\n", encoding="utf-8")
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="SHA does not match") as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "preflight"
    assert runner.commands == []
    assert not config.artifacts_dir.exists()


def test_reviewed_capture_cli_drift_is_rejected_before_subprocess(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    capture_cli = (
        config.calibration_root
        / "dynamic_pcd"
        / "apps"
        / "calibrate_eye_to_hand.py"
    )
    capture_cli.write_text("# changed capture CLI\n", encoding="utf-8")
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="capture-CLI SHA"):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []
    assert not config.artifacts_dir.exists()


@pytest.mark.parametrize("filename", ["stationary.py", "transforms.py"])
def test_reviewed_capture_math_drift_is_rejected_before_subprocess(
    tmp_path, filename
):
    config, _, dataset = _build_environment(tmp_path)
    reviewed = config.calibration_root / "dynamic_pcd" / "calibration" / filename
    reviewed.write_text("# changed capture math\n", encoding="utf-8")
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="SHA does not match"):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []
    assert not config.artifacts_dir.exists()


@pytest.mark.parametrize("artifact_field", ["capture_config", "frame_telemetry_script"])
def test_telemetry_input_drift_after_preview_still_has_zero_motion(
    tmp_path, artifact_field
):
    config, _, dataset = _build_environment(tmp_path)
    runner = FakeSubprocessRunner(
        dataset, mutate_after_preview=getattr(config, artifact_field)
    )

    with pytest.raises(sequence.SequenceFailure, match="immutable input changed") as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "preview"
    assert len(runner.commands) == 2
    assert runner.phases == ["import_preflight", "T01_to_T02:preview"]
    assert not any(phase.endswith(":motion") for phase in runner.phases)
    assert len(yaml.safe_load(dataset.read_text())["samples"]) == 1


def test_master_binds_the_only_permitted_artifacts_ledger(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    config = replace(config, artifacts_dir=tmp_path / "alternate-ledger")
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="master ledger path"):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []
    assert not config.artifacts_dir.exists()


def test_same_master_bytes_cannot_be_replayed_from_an_alternate_root(tmp_path):
    config, _, _ = _build_environment(tmp_path)
    alternate_root = tmp_path / "alternate-root"
    shutil.copytree(config.calibration_root, alternate_root)
    alternate_config = replace(
        config,
        master_plan=alternate_root / "calibration_runs" / "master.yaml",
        dataset=alternate_root / "calibration_runs" / "training.yaml",
        artifacts_dir=alternate_root / "calibration_runs" / "sequence-artifacts",
        calibration_root=alternate_root,
        capture_config=alternate_root / "configs" / "d435_default.yaml",
    )
    runner = FakeSubprocessRunner(alternate_config.dataset)

    with pytest.raises(sequence.SequenceFailure, match="calibration-root path"):
        sequence.run_sequence(alternate_config, runner=runner)

    assert runner.commands == []
    assert not alternate_config.artifacts_dir.exists()


@pytest.mark.parametrize(
    "mutation,error_text",
    [
        ("scope", "authorization scope"),
        ("suffix", "authorized_training_suffix"),
        ("edges", "authorized_edges"),
        ("status", "master status"),
        ("consumed", "consumed must be exactly false"),
    ],
)
def test_master_requires_exact_remaining_suffix_authorization(
    tmp_path, mutation, error_text
):
    config, master, dataset = _build_environment(tmp_path)
    if mutation == "scope":
        master["motion_authorization"]["scope"] = "single_pose"
    elif mutation == "suffix":
        master["motion_authorization"]["authorized_training_suffix"] = ["T02"]
    elif mutation == "edges":
        master["motion_authorization"]["authorized_edges"] = ["T01_to_T02"]
    elif mutation == "status":
        master["status"] = sequence.RUN_STATUS
    else:
        master["motion_authorization"]["consumed"] = None
    config.master_plan.write_text(
        yaml.safe_dump(master, sort_keys=False), encoding="utf-8"
    )
    config = replace(config, expected_master_sha256=_sha(config.master_plan))
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match=error_text):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []
    assert not config.artifacts_dir.exists()


@pytest.mark.parametrize("field", ["motion_driver", "link_preflight"])
def test_same_hash_dependency_copy_is_not_the_wrapper_import_path(tmp_path, field):
    config, _, dataset = _build_environment(tmp_path)
    reviewed = getattr(config, field)
    copied = tmp_path / reviewed.name
    copied.write_bytes(reviewed.read_bytes())
    config = replace(config, **{field: copied})
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="canonical module"):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []


def test_python_launch_symlink_identity_is_bound_not_just_target_hash(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    resolved_target = Path(config.python_executable).resolve()
    config = replace(config, python_executable=str(resolved_target))
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="Python launch path"):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []


def test_relative_python_launch_path_is_rejected_before_subprocess(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    config = replace(config, python_executable=".venv/bin/python")
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="absolute path"):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []


def test_relative_master_bound_root_is_rejected_before_subprocess(tmp_path):
    config, master, dataset = _build_environment(tmp_path)
    master["provenance"]["calibration_root_path"] = "beta/dynamic_object_pcd"
    config.master_plan.write_text(
        yaml.safe_dump(master, sort_keys=False), encoding="utf-8"
    )
    config = replace(config, expected_master_sha256=_sha(config.master_plan))
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="absolute path"):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []


@pytest.mark.parametrize(
    "override_field",
    [
        "import_probe_capture_path",
        "import_probe_detector_path",
        "import_probe_stationary_path",
        "import_probe_transforms_path",
    ],
)
def test_import_probe_path_hijack_stops_before_artifacts_or_motion(
    tmp_path, override_field
):
    config, _, dataset = _build_environment(tmp_path)
    runner = FakeSubprocessRunner(
        dataset, **{override_field: str(tmp_path / "hijacked_module.py")}
    )

    with pytest.raises(sequence.SequenceFailure, match="import resolution") as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "import_preflight"
    assert runner.phases == ["import_preflight"]
    assert not config.artifacts_dir.exists()


@pytest.mark.parametrize(
    "path_name",
    [
        "master",
        "dataset",
        "python_target",
        "motion_script",
        "motion_driver",
        "link_preflight",
        "inspection_script",
        "frame_telemetry_script",
        "capture_config",
        "capture_cli",
        "detector",
        "stationary",
        "transforms",
    ],
)
def test_capture_adjacent_recheck_rejects_input_change_before_collector(
    tmp_path, path_name
):
    config, _, dataset = _build_environment(tmp_path)
    paths = {
        "master": config.master_plan,
        "dataset": dataset,
        "python_target": Path(config.python_executable).resolve(),
        "motion_script": config.motion_script,
        "motion_driver": config.motion_driver,
        "link_preflight": config.link_preflight,
        "inspection_script": config.inspection_script,
        "frame_telemetry_script": config.frame_telemetry_script,
        "capture_config": config.capture_config,
        "capture_cli": (
            config.calibration_root
            / "dynamic_pcd"
            / "apps"
            / "calibrate_eye_to_hand.py"
        ),
        "detector": (
            config.calibration_root / "dynamic_pcd" / "calibration" / "aruco.py"
        ),
        "stationary": (
            config.calibration_root
            / "dynamic_pcd"
            / "calibration"
            / "stationary.py"
        ),
        "transforms": (
            config.calibration_root
            / "dynamic_pcd"
            / "calibration"
            / "transforms.py"
        ),
    }
    runner = FakeSubprocessRunner(
        dataset, mutate_after_telemetry=paths[path_name]
    )

    with pytest.raises(sequence.SequenceFailure) as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase in ("telemetry", "capture")
    assert len(runner.commands) == 5
    assert not any(phase.endswith(":capture") for phase in runner.phases)


@pytest.mark.parametrize(
    "runner_kwargs",
    [
        {"mutate_plan_after_telemetry": True},
        {"mutate_claim_after_telemetry": True},
    ],
)
def test_capture_adjacent_recheck_binds_plan_and_canonical_claim(
    tmp_path, runner_kwargs
):
    config, _, dataset = _build_environment(tmp_path)
    runner = FakeSubprocessRunner(dataset, **runner_kwargs)

    with pytest.raises(sequence.SequenceFailure) as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "capture"
    assert len(runner.commands) == 5
    assert not any(phase.endswith(":capture") for phase in runner.phases)


def test_post_capture_recheck_rejects_executable_change_before_receipt(tmp_path):
    config, _, dataset = _build_environment(tmp_path)
    stationary = (
        config.calibration_root
        / "dynamic_pcd"
        / "calibration"
        / "stationary.py"
    )
    runner = FakeSubprocessRunner(dataset, mutate_after_capture=stationary)

    with pytest.raises(sequence.SequenceFailure, match="immutable input changed") as caught:
        sequence.run_sequence(config, runner=runner)

    assert caught.value.phase == "receipt"
    assert runner.phases[-1].endswith(":capture")
    assert not list((config.artifacts_dir / "edges").glob("*.complete.json"))


def test_master_current_pose_must_match_captured_prefix(tmp_path):
    config, master, dataset = _build_environment(tmp_path)
    master["collection_state"]["current_pose_id"] = "T02"
    config.master_plan.write_text(
        yaml.safe_dump(master, sort_keys=False), encoding="utf-8"
    )
    config = replace(config, expected_master_sha256=_sha(config.master_plan))
    runner = FakeSubprocessRunner(dataset)

    with pytest.raises(sequence.SequenceFailure, match="current_pose_id"):
        sequence.run_sequence(config, runner=runner)

    assert runner.commands == []
