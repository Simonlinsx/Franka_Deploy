from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_ROOT = REPOSITORY_ROOT / "beta" / "dynamic_object_pcd"
sys.path.insert(0, str(CALIBRATION_ROOT))

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "capture_aruco_frame_telemetry.py"
)
SPEC = importlib.util.spec_from_file_location("aruco_telemetry_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
telemetry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = telemetry
SPEC.loader.exec_module(telemetry)

from dynamic_pcd.calibration.aruco import ArucoDetection, ArucoMarkerSpec
from dynamic_pcd.types import CameraIntrinsics, RGBDFrame


def _pose(x_m: float = 0.0) -> np.ndarray:
    transform = np.eye(4)
    transform[0, 3] = x_m
    return transform


def _pose_with_z_rotation(x_m: float, rotation_deg: float) -> np.ndarray:
    transform = _pose(x_m)
    angle = np.radians(rotation_deg)
    transform[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    return transform


def _record(index: int, transform: np.ndarray, reprojection: float = 0.2) -> dict:
    return {
        "capture_index": index,
        "detection": {
            "valid": True,
            "T_camera_target": transform.tolist(),
            "reprojection_error_px": reprojection,
        },
        "diagnostics": {
            "reprojection_gate_pass": False,
            "coherence_candidate": False,
            "translation_from_medoid_m": None,
            "rotation_from_medoid_deg": None,
            "coherent_pose_inlier": False,
            "abnormal_reasons": [],
            "raw_image": None,
        },
    }


def test_reports_exact_116_of_120_coherence_and_preserves_outlier_indices():
    records = [_record(index, _pose()) for index in range(116)]
    records.extend(_record(index, _pose(0.020)) for index in range(116, 120))

    result = telemetry.analyze_pose_coherence(
        records,
        max_reprojection_error_px=1.0,
        max_translation_deviation_m=0.003,
        max_rotation_deviation_deg=1.0,
    )

    assert result["detected_valid_count"] == 120
    assert result["reprojection_gate_pass_count"] == 120
    assert result["coherent_pose_count"] == 116
    assert result["coherent_pose_fraction"] == 116 / 120
    assert result["coherence_outlier_capture_indices"] == [116, 117, 118, 119]
    assert result["coherent_capture_indices"] == list(range(116))
    for index in range(116, 120):
        assert records[index]["diagnostics"]["abnormal_reasons"] == [
            "pose_coherence_outlier"
        ]
        assert records[index]["diagnostics"]["translation_from_medoid_m"] == 0.020


def test_reports_untrimmed_p95_over_every_reprojection_pass_frame():
    records = [_record(index, _pose(), reprojection=0.2) for index in range(114)]
    records.extend(
        _record(
            index,
            _pose_with_z_rotation(0.02002, 6.02),
            reprojection=6.22,
        )
        for index in range(114, 120)
    )

    result = telemetry.analyze_pose_coherence(
        records,
        max_reprojection_error_px=10.0,
        max_translation_deviation_m=0.003,
        max_rotation_deviation_deg=1.0,
    )

    assert result["reprojection_gate_pass_count"] == 120
    assert result["coherent_pose_count"] == 114
    assert result[
        "all_reprojection_pass_translation_jitter_p95_m"
    ] == pytest.approx(0.001001)
    assert result[
        "all_reprojection_pass_rotation_jitter_p95_deg"
    ] == pytest.approx(0.301)
    assert result[
        "all_reprojection_pass_reprojection_error_p95_px"
    ] == pytest.approx(0.501)


def test_separates_invalid_detection_and_reprojection_gate_failure():
    records = [_record(0, _pose())]
    invalid = _record(1, _pose())
    invalid["detection"].update(
        {"valid": False, "T_camera_target": None, "reprojection_error_px": None}
    )
    records.append(invalid)
    records.append(_record(2, _pose(), reprojection=1.25))

    result = telemetry.analyze_pose_coherence(
        records,
        max_reprojection_error_px=1.0,
        max_translation_deviation_m=0.003,
        max_rotation_deviation_deg=1.0,
    )

    assert result["detected_valid_count"] == 2
    assert result["reprojection_gate_pass_count"] == 1
    assert result["coherent_pose_count"] == 1
    assert result["invalid_detection_capture_indices"] == [1]
    assert result["reprojection_outlier_capture_indices"] == [2]
    assert records[1]["diagnostics"]["abnormal_reasons"] == ["detection_invalid"]
    assert records[2]["diagnostics"]["abnormal_reasons"] == [
        "reprojection_gate_failed"
    ]

    all_reprojection_failed = [_record(0, _pose(), reprojection=2.0)]
    empty_coherence = telemetry.analyze_pose_coherence(
        all_reprojection_failed,
        max_reprojection_error_px=1.0,
        max_translation_deviation_m=0.003,
        max_rotation_deviation_deg=1.0,
    )
    assert empty_coherence["detected_valid_count"] == 1
    assert empty_coherence["reprojection_gate_pass_count"] == 0
    assert empty_coherence["coherent_pose_count"] == 0
    assert (
        empty_coherence["all_reprojection_pass_translation_jitter_p95_m"] is None
    )
    assert empty_coherence["all_reprojection_pass_rotation_jitter_p95_deg"] is None
    assert (
        empty_coherence["all_reprojection_pass_reprojection_error_p95_px"] is None
    )


class _FakeCamera:
    instances = []

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.device_serial = None
        self.device_name = "Fake RealSense"
        self.depth_scale = 0.001
        self.started = False
        self.stopped = False
        self.next_frame = 0
        self.__class__.instances.append(self)

    def start(self):
        self.started = True
        self.device_serial = str(self.cfg["serial"])

    def stop(self):
        self.stopped = True

    def get_frame(self, timeout_ms=1000):
        assert timeout_ms == 25
        self.next_frame += 1
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, self.next_frame : self.next_frame + 2] = 255
        return RGBDFrame(
            color_bgr=image,
            depth_raw=np.zeros((24, 32), dtype=np.uint16),
            depth_scale=self.depth_scale,
            intrinsics=CameraIntrinsics(
                width=32,
                height=24,
                fx=20.0,
                fy=20.0,
                ppx=16.0,
                ppy=12.0,
                model="none",
                distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            timestamp=1_700_000_000.0 + self.next_frame / 30.0,
            frame_id=self.next_frame,
        )


def _fake_detector(_image, _intrinsics, spec):
    frame_number = _FakeCamera.instances[-1].next_frame
    if frame_number == 3:
        return ArucoDetection(
            valid=False,
            corners=np.zeros((0, 2), dtype=np.float32),
            marker_id=spec.marker_id,
            T_camera_target=None,
            reprojection_error_px=float("inf"),
            message="not detected",
        )
    transform = _pose(0.020 if frame_number == 2 else 0.0)
    return ArucoDetection(
        valid=True,
        corners=np.asarray([[2, 2], [12, 2], [12, 12], [2, 12]], np.float32),
        marker_id=spec.marker_id,
        T_camera_target=transform,
        reprojection_error_px=0.2,
        message="ok",
    )


def test_camera_only_capture_writes_strict_json_and_no_images_by_default(tmp_path):
    _FakeCamera.instances.clear()
    config_path = tmp_path / "camera.yaml"
    config_path.write_text(
        yaml.safe_dump({"camera": {"serial": "wrong", "width": 32, "height": 24}}),
        encoding="utf-8",
    )
    output_path = tmp_path / "telemetry.json"

    payload = telemetry.capture_telemetry(
        config_path=config_path,
        camera_serial="342222071785",
        target_spec=ArucoMarkerSpec("DICT_6X6_50", 42, 0.190),
        frame_count=3,
        output_path=output_path,
        max_reprojection_error_px=1.0,
        max_translation_deviation_m=0.003,
        max_rotation_deviation_deg=1.0,
        timeout_ms=25,
        camera_factory=_FakeCamera,
        detector=_fake_detector,
    )

    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document == payload
    assert payload["scope"] == {
        "camera_only": True,
        "read_only": True,
        "franka_fci_opened": False,
        "robot_motion_commanded": False,
    }
    assert payload["camera"]["opened_serial"] == "342222071785"
    assert payload["capture"]["captured_frame_count"] == 3
    assert payload["coherence"]["coherent_pose_count"] == 1
    assert payload["frames"][2]["detection"]["reprojection_error_px"] is None
    assert payload["abnormal_images"] == []
    assert list(tmp_path.glob("*.png")) == []
    assert _FakeCamera.instances[-1].cfg["serial"] == "342222071785"
    assert _FakeCamera.instances[-1].started is True
    assert _FakeCamera.instances[-1].stopped is True


def test_optional_png_output_contains_only_abnormal_candidates(tmp_path):
    _FakeCamera.instances.clear()
    config_path = tmp_path / "camera.yaml"
    config_path.write_text(yaml.safe_dump({"camera": {}}), encoding="utf-8")
    output_path = tmp_path / "telemetry.json"
    abnormal_dir = tmp_path / "abnormal"

    payload = telemetry.capture_telemetry(
        config_path=config_path,
        camera_serial="342222071785",
        target_spec=ArucoMarkerSpec("DICT_6X6_50", 42, 0.190),
        frame_count=3,
        output_path=output_path,
        max_reprojection_error_px=1.0,
        max_translation_deviation_m=0.003,
        max_rotation_deviation_deg=1.0,
        abnormal_images_dir=abnormal_dir,
        timeout_ms=25,
        camera_factory=_FakeCamera,
        detector=_fake_detector,
    )

    images = sorted(abnormal_dir.glob("*.png"))
    assert len(images) == 2
    assert len(payload["abnormal_images"]) == 2
    assert payload["frames"][0]["diagnostics"]["raw_image"] is None
    assert payload["frames"][1]["diagnostics"]["raw_image"] is not None
    assert payload["frames"][2]["diagnostics"]["raw_image"] is not None


def test_source_has_no_robot_or_fci_dependency():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "FrankaStateReader" not in source
    assert "pylibfranka" not in source
    assert "anydex_pipeline" not in source
