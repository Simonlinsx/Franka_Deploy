from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_ROOT = REPOSITORY_ROOT / "beta" / "dynamic_object_pcd"
sys.path.insert(0, str(CALIBRATION_ROOT))

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_rgbd_physical.py"
SPEC = importlib.util.spec_from_file_location("rgbd_physical_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
physical = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = physical
SPEC.loader.exec_module(physical)

from dynamic_pcd.calibration.aruco import ArucoDetection
from dynamic_pcd.types import CameraIntrinsics, RGBDFrame


@pytest.mark.parametrize("use_session_target", [False, True])
def test_station_capture_with_fakes_writes_bound_raw_evidence_and_never_moves_robot(
    tmp_path, use_session_target,
):
    config_path = tmp_path / "camera.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "camera": {
                    "serial": "wrong",
                    "width": 100,
                    "height": 80,
                    "fps": 30,
                    "spatial_filter": False,
                    "temporal_filter": False,
                    "hole_filter": False,
                }
            }
        ),
        encoding="utf-8",
    )
    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "camera_robot_extrinsic_calibration",
                "calibration_type": "eye_to_hand",
                "calibration_id": "eye-to-hand-test",
                "base_frame": "robot_base",
                "camera_frame": "camera_color_optical_frame",
                "T_base_camera": np.eye(4).tolist(),
                "T_ee_target": np.eye(4).tolist(),
                "camera": {
                    "name": "Intel RealSense D435",
                    "serial": "342222071785",
                    "depth_scale": 0.001,
                    "intrinsics": {
                        "width": 100,
                        "height": 80,
                        "fx": 100.0,
                        "fy": 100.0,
                        "ppx": 50.0,
                        "ppy": 40.0,
                        "distortion": [0.0] * 5,
                    },
                },
                "target": {
                    "type": "aruco",
                    "dictionary": "DICT_6X6_50",
                    "marker_id": 42,
                    "marker_length_m": 0.19,
                },
                "solver": {"quality": {"status": "pass", "warnings": []}},
            }
        ),
        encoding="utf-8",
    )
    intrinsics = CameraIntrinsics(
        width=100,
        height=80,
        fx=100.0,
        fy=100.0,
        ppx=50.0,
        ppy=40.0,
        model="none",
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    session_target_report = None
    if use_session_target:
        session_target_report = tmp_path / "session-target.json"
        session_target_report.write_text(
            physical.json.dumps(
                {
                    "schema_version": 1,
                    "kind": "reused_eye_to_hand_extrinsic_session_holdout",
                    "decision": "holdout_pass",
                    "calibration": {
                        "sha256": physical._sha256(calibration_path),
                        "calibration_id": "eye-to-hand-test",
                        "T_base_camera_reused_unchanged": True,
                    },
                    "T_ee_target_session": np.eye(4).tolist(),
                }
            ),
            encoding="utf-8",
        )

    class FakeCamera:
        instances = []

        def __init__(self, config):
            self.config = dict(config)
            self.device_serial = "342222071785"
            self.device_name = "Intel RealSense D435"
            self.depth_scale = 0.001
            self.frame_id = 0
            self.stopped = False
            self.__class__.instances.append(self)

        def start(self):
            pass

        def get_frame(self):
            self.frame_id += 1
            return RGBDFrame(
                color_bgr=np.zeros((80, 100, 3), dtype=np.uint8),
                depth_raw=np.full((80, 100), 1000, dtype=np.uint16),
                depth_scale=0.001,
                intrinsics=intrinsics,
                timestamp=float(self.frame_id),
                frame_id=self.frame_id,
            )

        def stop(self):
            self.stopped = True

    class FakeRobot:
        instances = []

        def __init__(self, ip):
            self.ip = ip
            self.closed = False
            self.read_count = 0
            self.__class__.instances.append(self)

        def connect(self):
            pass

        def read_T_base_ee(self):
            self.read_count += 1
            transform = np.eye(4)
            transform[2, 3] = 1.0
            return transform

        def close(self):
            self.closed = True

    def detector(_image, _intrinsics, spec):
        transform = np.eye(4)
        transform[2, 3] = 1.0
        return ArucoDetection(
            valid=True,
            corners=np.asarray(
                [[35.0, 25.0], [65.0, 25.0], [65.0, 55.0], [35.0, 55.0]],
                dtype=np.float32,
            ),
            marker_id=spec.marker_id,
            T_camera_target=transform,
            reprojection_error_px=0.1,
            message="ok",
        )

    output_path = tmp_path / "station.json"
    evidence_dir = tmp_path / "station-evidence"
    report = physical.capture_station(
        config_path=config_path,
        calibration_path=calibration_path,
        output_path=output_path,
        evidence_dir=evidence_dir,
        camera_serial="342222071785",
        station_id="middle-center",
        distance_band="middle",
        image_region="center",
        frames=3,
        max_reprojection_error_px=0.5,
        min_reprojection_pass_frames=3,
        min_coherent_frames=3,
        max_pnp_translation_jitter_m=0.001,
        max_pnp_rotation_jitter_deg=0.3,
        max_untrimmed_translation_jitter_m=0.001,
        max_untrimmed_rotation_jitter_deg=0.3,
        max_untrimmed_reprojection_error_px=0.5,
        min_rgbd_valid_frames=3,
        interior_scale=0.75,
        pixel_stride=1,
        depth_min_m=0.8,
        depth_max_m=1.2,
        max_depth_from_pnp_m=0.05,
        plane_residual_limit_m=0.005,
        minimum_depth_points=100,
        intrinsics_tolerance_px=0.05,
        robot_ip="172.16.0.2",
        max_robot_translation_m=0.0005,
        max_robot_rotation_deg=0.1,
        known_base_center_m=None,
        known_base_source=None,
        known_base_evidence=None,
        session_target_report=session_target_report,
        camera_factory=FakeCamera,
        robot_factory=FakeRobot,
        detector=detector,
    )

    assert report["station_local_pass"] is True
    assert report["hardware_access"] == {
        "camera_read_only": True,
        "robot_state_read_only": True,
        "robot_motion_interface_present": False,
        "robot_commanded": False,
    }
    assert output_path.exists()
    assert len(report["raw_rgbd_evidence"]) == 3
    for evidence in report["raw_rgbd_evidence"]:
        path = Path(evidence["path"])
        assert path.exists()
        assert physical._sha256(path) == evidence["sha256"]
    assert FakeCamera.instances[-1].stopped is True
    assert FakeRobot.instances[-1].closed is True
    assert FakeRobot.instances[-1].read_count == 2
    assert report["known_base_reference"]["source"] == (
        "robot_kinematics_using_session_T_ee_target"
        if use_session_target
        else "robot_kinematics_using_calibration_T_ee_target"
    )
    verified = physical._verify_station_report_provenance(report, output_path)
    assert len(verified["raw_rgbd_evidence"]) == 3


def test_depth_plane_intersection_recovers_marker_center():
    intrinsics = CameraIntrinsics(
        width=100,
        height=80,
        fx=100.0,
        fy=100.0,
        ppx=50.0,
        ppy=40.0,
        model="none",
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    frame = RGBDFrame(
        color_bgr=np.zeros((80, 100, 3), dtype=np.uint8),
        depth_raw=np.full((80, 100), 1000, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=intrinsics,
        timestamp=1.0,
        frame_id=1,
    )
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 1.0]
    detection = ArucoDetection(
        valid=True,
        corners=np.asarray(
            [[35.0, 25.0], [65.0, 25.0], [65.0, 55.0], [35.0, 55.0]],
            dtype=np.float32,
        ),
        marker_id=42,
        T_camera_target=transform,
        reprojection_error_px=0.1,
        message="ok",
    )

    result = physical.measure_target_depth(
        frame,
        detection,
        T_base_camera=np.eye(4),
        known_center_base_m=[0.0, 0.0, 1.0],
        interior_scale=0.75,
        pixel_stride=1,
        depth_min_m=0.8,
        depth_max_m=1.2,
        max_depth_from_pnp_m=0.05,
        plane_residual_limit_m=0.005,
        minimum_depth_points=100,
    )

    assert result["valid_depth_fraction"] == pytest.approx(1.0)
    assert result["marker_center_camera_depth_m"] == pytest.approx([0.0, 0.0, 1.0])
    assert result["depth_minus_pnp_z_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["depth_minus_pnp_3d_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["depth_minus_known_base_3d_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["depth_plane_residual_p95_m"] < 1.0e-12


def _station(
    station_id: str,
    band: str,
    region: str,
    target_z_m: float,
    signed_bias_m: float,
    *,
    independent: bool = False,
) -> dict:
    frames = []
    for index in range(5):
        transform = np.eye(4)
        transform[2, 3] = target_z_m
        frames.append(
            {
                "capture_index": index,
                "detection_valid": True,
                "reprojection_gate_pass": True,
                "rgbd_valid": True,
                "T_camera_target": transform.tolist(),
                "rgbd": {
                    "valid_depth_fraction": 0.98,
                    "depth_plane_residual_p95_m": 0.001,
                    "depth_minus_pnp_z_m": signed_bias_m,
                    "depth_minus_pnp_3d_m": abs(signed_bias_m) + 0.001,
                    "depth_minus_known_base_z_m": signed_bias_m + 0.001,
                    "depth_minus_known_base_3d_m": abs(signed_bias_m) + 0.003,
                },
            }
        )
    return {
        "schema_version": 1,
        "kind": "eye_to_hand_rgbd_physical_station",
        "station_id": station_id,
        "distance_band": band,
        "image_region": region,
        "station_local_pass": True,
        "source_calibration": {
            "calibration_id": "eye-to-hand-test",
            "sha256": "a" * 64,
        },
        "camera": {
            "opened_serial": "342222071785",
            "name": "Intel RealSense D435",
        },
        "known_base_reference": {
            "independent_of_calibration": independent,
            "evidence_path": "/survey/evidence.yaml" if independent else None,
            "evidence_sha256": "b" * 64 if independent else None,
        },
        "raw_rgbd_evidence": [
            {"path": f"/raw/{station_id}-{index}.npz", "sha256": "c" * 64}
            for index in range(3)
        ],
        "frames": frames,
    }


def _evaluate(reports, **overrides):
    arguments = {
        "expected_serial": "342222071785",
        "expected_camera_name": "Intel RealSense D435",
        "minimum_stations": 3,
        "minimum_image_regions": 3,
        "minimum_distance_span_m": 0.12,
        "minimum_rgbd_valid_frames": 5,
        "minimum_raw_evidence_frames": 3,
        "minimum_depth_valid_fraction_p05": 0.90,
        "max_plane_residual_p95_m": 0.005,
        "max_depth_pnp_z_abs_p95_m": 0.010,
        "max_depth_pnp_z_abs_max_m": 0.020,
        "max_depth_pnp_3d_p95_m": 0.012,
        "max_known_base_z_abs_p95_m": 0.010,
        "max_known_base_z_abs_max_m": 0.015,
        "max_known_base_3d_p95_m": 0.015,
        "max_known_base_3d_max_m": 0.025,
        "max_station_signed_bias_range_m": 0.005,
        "require_independent_known_base": True,
    }
    arguments.update(overrides)
    return physical.evaluate_station_reports(reports, **arguments)


def _passing_stations():
    return [
        _station("near-left", "near", "left", 1.07, 0.002, independent=True),
        _station("middle-center", "middle", "center", 1.14, 0.003),
        _station("far-right", "far", "right", 1.22, 0.004),
    ]


def test_three_distance_reports_pass_explicit_physical_gates():
    result = _evaluate(_passing_stations())

    assert result["physical_rgbd_status"] == "pass"
    assert result["target_distance_span_m"] == pytest.approx(0.15)
    assert result["independent_known_base_evidence_present"] is True
    assert result["failures"] == []
    assert result["missing_evidence"] == []


def test_missing_independent_known_base_remains_provisional():
    reports = _passing_stations()
    reports[0]["known_base_reference"] = {
        "independent_of_calibration": False,
        "evidence_path": None,
        "evidence_sha256": None,
    }

    result = _evaluate(reports)

    assert result["physical_rgbd_status"] == "provisional"
    assert result["failures"] == []
    assert "no independently surveyed" in result["missing_evidence"][0]


def test_camera_product_name_mismatch_is_rejected_not_silently_accepted():
    result = _evaluate(
        _passing_stations(),
        expected_camera_name="Intel RealSense D455",
    )

    assert result["physical_rgbd_status"] == "rejected"
    assert any("camera name" in message for message in result["failures"])


def test_one_bad_distance_cannot_hide_behind_cross_station_aggregation():
    reports = _passing_stations()
    bad = copy.deepcopy(reports[-1])
    for record in bad["frames"]:
        record["rgbd"]["depth_minus_known_base_3d_m"] = 0.030
    reports[-1] = bad

    result = _evaluate(reports)

    assert result["physical_rgbd_status"] == "rejected"
    assert any(
        "far-right: known-base 3D p95" in message for message in result["failures"]
    )


def test_capture_cli_refuses_hardware_without_explicit_acknowledgement(
    tmp_path, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("hardware factory must not be reached")

    monkeypatch.setattr(physical, "capture_station", forbidden)
    exit_code = physical.main(
        [
            "capture-station",
            "--config",
            str(tmp_path / "camera.yaml"),
            "--calibration",
            str(tmp_path / "calibration.yaml"),
            "--output",
            str(tmp_path / "station.json"),
            "--evidence-dir",
            str(tmp_path / "station-evidence"),
            "--camera-serial",
            "342222071785",
            "--station-id",
            "near-left",
            "--distance-band",
            "near",
            "--image-region",
            "left",
        ]
    )

    assert exit_code == 2
