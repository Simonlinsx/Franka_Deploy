import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dynamic_pcd.apps.record_rgbd_case import (
    DEFAULT_CONFIG,
    _draw_status,
    _expanded_provider_roi,
    _frame_record,
    _project_base_workspace_overlay,
    _write_integrity_manifest,
    record_case,
)
import dynamic_pcd.apps.record_rgbd_case as recorder_module
import dynamic_pcd.camera.recorded_rgbd_camera as recorded_camera_module
from dynamic_pcd.camera.recorded_rgbd_camera import (
    EndOfRecording,
    RecordedRGBDCase,
    RecordedRGBDCamera,
)
from dynamic_pcd.types import CameraIntrinsics, RGBDFrame


def _frame(index: int) -> RGBDFrame:
    return RGBDFrame(
        color_bgr=np.full((3, 4, 3), index + 10, dtype=np.uint8),
        depth_raw=np.full((3, 4), index + 100, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=4,
            height=3,
            fx=10.0,
            fy=11.0,
            ppx=2.0,
            ppy=1.5,
            model="none",
            distortion=(0.0,) * 5,
        ),
        timestamp=1000.0 + index / 30.0,
        frame_id=index + 1,
        retrieved_at_s=1000.01 + index / 30.0,
        timestamp_domain="timestamp_domain.global_time",
        depth_timestamp_s=1000.001 + index / 30.0,
        color_depth_timestamp_skew_s=0.001,
        color_depth_epoch_timestamp_skew_s=0.001,
        sensor_frame_number=101 + index,
        depth_sensor_frame_number=201 + index,
        retrieved_monotonic_s=500.0 + index / 30.0,
        host_clock_pair_span_s=1.0e-6,
        capture_diagnostic={"outcome": "accepted"},
    )


def _case(tmp_path: Path) -> Path:
    case = tmp_path / "case"
    (case / "color").mkdir(parents=True)
    (case / "depth").mkdir()
    (case / "initialization").mkdir()
    initialization = _frame(-1)
    assert cv2.imwrite(
        str(case / "initialization" / "color.png"),
        initialization.color_bgr,
    )
    assert cv2.imwrite(
        str(case / "initialization" / "depth.png"),
        initialization.depth_raw,
    )
    initialization_record = _frame_record(-1, initialization)
    initialization_record.update(
        {
            "role": "tracker_initialization",
            "color_path": "initialization/color.png",
            "depth_path": "initialization/depth.png",
        }
    )
    records = []
    for index in range(3):
        frame = _frame(index)
        assert cv2.imwrite(str(case / "color" / f"{index:06d}.png"), frame.color_bgr)
        assert cv2.imwrite(str(case / "depth" / f"{index:06d}.png"), frame.depth_raw)
        records.append(_frame_record(index, frame))
    (case / "frames.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "schema": "dynamic_object_pcd_rgbd_case_v1",
        "complete": True,
        "case": "static_ball",
        "frame_count": 3,
        "initialization_frame": initialization_record,
        "integrity_manifest": "MANIFEST.sha256",
        "image_width": 4,
        "image_height": 3,
        "nominal_fps": 30,
        "depth_scale_m_per_unit": 0.001,
        "camera_serial": "camera-1",
        "camera_intrinsics": _frame(0).intrinsics.to_dict(),
        "camera_K": _frame(0).intrinsics.as_matrix().tolist(),
        "T_base_camera": np.eye(4).tolist(),
        "provider_initialization_roi_xyxy": [0, 0, 4, 3],
    }
    (case / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_integrity_manifest(case)
    return case


def test_recorded_case_round_trip_is_lossless(tmp_path: Path) -> None:
    case = RecordedRGBDCase(_case(tmp_path))
    assert len(case) == 3
    np.testing.assert_array_equal(
        case.initialization_frame().color_bgr, _frame(-1).color_bgr
    )
    for index, replayed in enumerate(case):
        expected = _frame(index)
        np.testing.assert_array_equal(replayed.color_bgr, expected.color_bgr)
        np.testing.assert_array_equal(replayed.depth_raw, expected.depth_raw)
        assert replayed.frame_id == expected.frame_id
        assert replayed.sensor_frame_number == expected.sensor_frame_number
        assert replayed.timestamp == expected.timestamp


def test_recorded_camera_stops_at_exact_end(tmp_path: Path) -> None:
    camera = RecordedRGBDCamera(_case(tmp_path), realtime=False)
    camera.start()
    assert [camera.get_frame().frame_id for _ in range(3)] == [1, 2, 3]
    try:
        camera.get_frame()
    except EndOfRecording:
        pass
    else:
        raise AssertionError("playback did not report its exact end")


def test_provider_roi_padding_is_clipped() -> None:
    actual = _expanded_provider_roi(
        np.asarray([2, 3, 8, 9]),
        width=10,
        height=12,
        padding_px=4,
    )
    np.testing.assert_array_equal(actual, [0, 0, 10, 12])


def test_robot_base_workspace_overlay_projects_and_only_changes_live_view() -> None:
    frame = RGBDFrame(
        color_bgr=np.zeros((80, 100, 3), dtype=np.uint8),
        depth_raw=np.full((80, 100), 1000, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=100,
            height=80,
            fx=50.0,
            fy=50.0,
            ppx=50.0,
            ppy=40.0,
        ),
        timestamp=1.0,
        frame_id=1,
    )
    overlay = _project_base_workspace_overlay(
        frame,
        np.eye(4, dtype=np.float64),
        [-0.2, -0.1, 1.0],
        [0.2, 0.1, 1.2],
    )
    assert overlay["all_vertices_inside_image"] is True
    assert overlay["reference_frame"] == "robot_base"
    assert overlay["lossless_frames_annotated"] is False
    assert len(overlay["projected_vertices_uv"]) == 8
    assert len(overlay["edge_vertex_indices"]) == 12
    rendered = _draw_status(frame, None, (), overlay)
    assert np.count_nonzero(rendered) > 0
    np.testing.assert_array_equal(frame.color_bgr, np.zeros((80, 100, 3), np.uint8))


def test_robot_base_workspace_overlay_rejects_out_of_image_corner() -> None:
    frame = RGBDFrame(
        color_bgr=np.zeros((80, 100, 3), dtype=np.uint8),
        depth_raw=np.ones((80, 100), dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(100, 80, 50.0, 50.0, 50.0, 40.0),
        timestamp=1.0,
        frame_id=1,
    )
    try:
        _project_base_workspace_overlay(
            frame,
            np.eye(4, dtype=np.float64),
            [2.0, -0.1, 1.0],
            [2.2, 0.1, 1.2],
        )
    except ValueError as exc:
        assert "not fully visible" in str(exc)
    else:
        raise AssertionError("out-of-image workspace unexpectedly accepted")


def test_zero_depth_and_complete_visual_occlusion_are_retained(
    tmp_path: Path,
) -> None:
    path = _case(tmp_path)
    black = np.zeros((3, 4, 3), dtype=np.uint8)
    zero_depth = np.zeros((3, 4), dtype=np.uint16)
    assert cv2.imwrite(str(path / "color" / "000001.png"), black)
    assert cv2.imwrite(str(path / "depth" / "000001.png"), zero_depth)
    _write_integrity_manifest(path)

    replayed = RecordedRGBDCase(path).frame(1)
    np.testing.assert_array_equal(replayed.color_bgr, black)
    np.testing.assert_array_equal(replayed.depth_raw, zero_depth)


def test_integrity_check_detects_corrupted_png(tmp_path: Path) -> None:
    path = _case(tmp_path)
    assert cv2.imwrite(
        str(path / "depth" / "000002.png"),
        np.zeros((3, 4), dtype=np.uint16),
    )
    try:
        RecordedRGBDCase(path)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("corrupted recording unexpectedly passed integrity")


def test_rebase_prevents_post_initialization_catch_up(
    tmp_path: Path, monkeypatch
) -> None:
    clock = [10.0]
    sleeps = []

    def fake_sleep(duration: float) -> None:
        sleeps.append(float(duration))
        clock[0] += float(duration)

    monkeypatch.setattr(
        recorded_camera_module.time, "monotonic", lambda: clock[0]
    )
    monkeypatch.setattr(recorded_camera_module.time, "sleep", fake_sleep)
    camera = RecordedRGBDCamera(_case(tmp_path), realtime=True)
    camera.start()
    clock[0] = 12.0  # Simulate expensive tracker/SAM initialization.
    camera.rebase_timing()
    assert camera.get_frame().frame_id == 1
    assert sleeps == []
    assert camera.get_frame().frame_id == 2
    assert len(sleeps) == 1
    assert abs(sleeps[0] - 1.0 / 30.0) < 1.0e-9


def test_thrown_ball_capture_is_unboxed_424x240_camera_frame_case(
    tmp_path: Path, monkeypatch
) -> None:
    serial = "342222071785"

    def capture_frame(index: int) -> RGBDFrame:
        height, width = 240, 424
        return RGBDFrame(
            color_bgr=np.full((height, width, 3), 20 + index, dtype=np.uint8),
            depth_raw=np.full((height, width), 900 + index, dtype=np.uint16),
            depth_scale=0.001,
            intrinsics=CameraIntrinsics(
                width=width,
                height=height,
                fx=310.0,
                fy=311.0,
                ppx=212.0,
                ppy=120.0,
                model="none",
                distortion=(0.0,) * 5,
            ),
            timestamp=1000.0 + index / 30.0,
            frame_id=index + 1,
            retrieved_at_s=1000.001 + index / 30.0,
            timestamp_domain="timestamp_domain.global_time",
            depth_timestamp_s=1000.0 + index / 30.0,
            color_depth_timestamp_skew_s=0.0,
            color_depth_epoch_timestamp_skew_s=0.0,
            sensor_frame_number=100 + index,
            depth_sensor_frame_number=200 + index,
            retrieved_monotonic_s=500.0 + index / 30.0,
            host_clock_pair_span_s=1.0e-6,
            capture_diagnostic={"outcome": "accepted"},
        )

    class FakeCamera:
        def __init__(self, cfg) -> None:
            assert cfg["serial"] == serial
            assert (cfg["width"], cfg["height"], cfg["fps"]) == (424, 240, 30)
            self.device_serial = serial
            self.device_name = "Intel RealSense D435"
            self.device_firmware_version = "test"
            self.device_usb_type_descriptor = "3.2"
            self.sdk_version = "test"
            self.index = 0

        def start(self) -> None:
            return None

        def get_frame(self) -> RGBDFrame:
            frame = capture_frame(self.index)
            self.index += 1
            return frame

        def stop(self) -> None:
            return None

    monkeypatch.setattr(recorder_module, "RealSenseCamera", FakeCamera)
    output = tmp_path / "thrown_ball"
    saved = record_case(
        argparse.Namespace(
            case="thrown_ball",
            config=DEFAULT_CONFIG,
            output=output,
            duration_s=1.0 / 30.0,
            countdown_s=0.0,
            camera_serial=serial,
            width=424,
            height=240,
            fps=30,
            max_color_depth_timestamp_skew_ms=14.0,
            camera_frame_only=True,
            object_text="ball",
            roi=None,
            no_preview=True,
            png_compression=1,
        )
    )

    assert saved == output.resolve()
    manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case"] == "thrown_ball"
    assert manifest["camera_serial"] == serial
    assert (manifest["image_width"], manifest["image_height"]) == (424, 240)
    assert manifest["nominal_fps"] == 30
    assert manifest["capture_config_overrides"][
        "camera.max_color_depth_timestamp_skew_s"
    ] == 0.014
    assert manifest["initialization_mode"] == (
        "automatic_text_grounding_on_recorded_frames"
    )
    assert manifest["initialization_frame"]["role"] == (
        "automatic_grounding_preroll"
    )
    assert manifest["tight_roi_xyxy"] is None
    assert manifest["provider_initialization_roi_xyxy"] is None
    assert manifest["object_initially_required_visible"] is False
    assert manifest["object_text"] == "ball"
    assert manifest["reference_frame"] == "camera_color_optical_frame"
    assert manifest["calibration_id"] == ""
    assert manifest["franka_interface_opened"] is False
    assert manifest["rh56_interface_opened"] is False
    assert manifest["robot_command_writes"] is False
    assert (saved / "preview.mp4").stat().st_size > 0
    assert (saved / "color" / "000000.png").is_file()
    assert (saved / "depth" / "000000.png").is_file()
