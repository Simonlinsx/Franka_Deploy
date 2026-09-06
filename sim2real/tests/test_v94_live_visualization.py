from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sim2real.observation.visualization import (
    V94LiveVisualizationError,
    V94LiveVisualizationSample,
    V94LiveVisualizer,
    _recording_mask_frame_bgr,
    _policy_cloud_geometry_and_colors,
    _recording_frame_bgr,
    mask_video_path_for_recording,
    _save_visualization_sample,
)


def _sample(frame_id: int = 7) -> V94LiveVisualizationSample:
    return V94LiveVisualizationSample(
        sequence=3,
        frame_id=frame_id,
        captured_realtime_s=100.0,
        color_bgr=np.zeros((480, 848, 3), dtype=np.uint8),
        object_mask=np.zeros((480, 848), dtype=bool),
        pointcloud_xyzrgb_palm=np.zeros((128, 6), dtype=np.float32),
        pointcloud_valid=np.ones(128, dtype=np.float32),
        T_base_palm_at_capture=np.eye(4, dtype=np.float64),
        source_valid_points=1000,
    )


def test_policy_side_offer_is_reference_only_and_duplicate_lossy():
    viewer = V94LiveVisualizer(update_rate_hz=10.0)
    viewer._open = True
    sample = _sample()

    assert viewer.try_publish(sample) is True
    assert viewer._latest is sample
    assert viewer.try_publish(_sample(frame_id=7)) is False
    stats = viewer.stats()
    assert stats["offered_unique_frames"] == 1
    assert stats["duplicate_camera_frames_skipped"] == 1


def test_cloud_viewer_supports_xyz_and_xyzrgb_without_changing_geometry():
    xyz = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    valid = np.asarray([1.0, 0.0], dtype=np.float32)
    visible_xyz, xyz_colors = _policy_cloud_geometry_and_colors(xyz, valid)
    visible_xyzrgb, rgb_colors = _policy_cloud_geometry_and_colors(
        np.concatenate([xyz, np.asarray([[1.2, -0.1, 0.5], [0.0, 0.0, 0.0]])], axis=1),
        valid,
    )
    np.testing.assert_array_equal(visible_xyz, visible_xyzrgb)
    np.testing.assert_allclose(xyz_colors, [[0.15, 0.75, 1.0]])
    np.testing.assert_allclose(rgb_colors, [[1.0, 0.0, 0.5]])


def test_recording_keeps_rgb_clean_and_writes_full_resolution_mask_sidecar():
    sample = _sample()
    color = np.zeros_like(sample.color_bgr)
    color[..., 0] = 17
    color[..., 1] = 43
    color[..., 2] = 91
    mask = np.zeros((480, 848), dtype=bool)
    mask[120:360, 200:650] = True
    payload = type("Payload", (), {
        "sequence": 3,
        "frame_id": 7,
        "color_bgr": color,
        "object_mask": mask,
        "source_valid_points": 1000,
        "final_policy_points": 128,
    })()
    frame = _recording_frame_bgr(payload)
    mask_frame = _recording_mask_frame_bgr(payload)
    np.testing.assert_array_equal(frame, color)
    assert not np.shares_memory(frame, color)
    assert mask_frame.shape == color.shape
    np.testing.assert_array_equal(mask_frame[..., 0] > 0, mask)
    np.testing.assert_array_equal(mask_frame[..., 0], mask_frame[..., 1])
    np.testing.assert_array_equal(mask_frame[..., 1], mask_frame[..., 2])
    assert set(np.unique(mask_frame).tolist()) == {0, 255}


def test_mask_video_path_is_deterministic_sidecar(tmp_path):
    assert mask_video_path_for_recording(tmp_path / "trial.mp4") == (
        tmp_path / "trial_mask.mp4"
    ).resolve()


def test_record_video_path_requires_mp4(tmp_path):
    with pytest.raises(ValueError, match="must end in .mp4"):
        V94LiveVisualizer(record_video_path=tmp_path / "run.avi")


def test_record_only_mode_does_not_require_live_display(monkeypatch, tmp_path):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    viewer = V94LiveVisualizer(
        show_live_windows=False,
        record_video_path=tmp_path / "run.mp4",
    )

    assert viewer.show_live_windows is False
    assert viewer.record_video_path == (tmp_path / "run.mp4").resolve()
    assert viewer.record_mask_video_path == (tmp_path / "run_mask.mp4").resolve()


def test_recording_accepts_distinct_policy_sequences_on_same_camera_frame(tmp_path):
    viewer = V94LiveVisualizer(
        update_rate_hz=10.0,
        record_video_path=tmp_path / "run.mp4",
        record_video_rate_hz=20.0,
    )
    viewer._open = True
    first = _sample(frame_id=7)
    second = replace(first, sequence=4)

    assert viewer.try_publish(first) is True
    viewer._wake.clear()
    assert viewer.try_publish(second) is True
    assert viewer.stats()["record_video_rate_hz"] == 20.0
    assert viewer.stats()["duplicate_camera_frames_skipped"] == 1


@pytest.mark.parametrize("rate", (0.0, 30.1, float("nan")))
def test_invalid_record_video_rate_is_rejected(tmp_path, rate):
    with pytest.raises(ValueError, match="record video rate must be in 1..30"):
        V94LiveVisualizer(
            record_video_path=tmp_path / "run.mp4",
            record_video_rate_hz=rate,
        )


@pytest.mark.parametrize("rate", (0.0, 15.1, float("nan")))
def test_invalid_visualization_rate_is_rejected(rate):
    with pytest.raises(ValueError, match="1..15"):
        V94LiveVisualizer(update_rate_hz=rate)


def test_explicit_viewer_without_display_fails_before_process(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    viewer = V94LiveVisualizer(update_rate_hz=10.0)
    with pytest.raises(V94LiveVisualizationError, match="DISPLAY"):
        viewer.open()
    assert viewer._mask_process is None
    assert viewer._cloud_process is None


def test_exact_policy_visualization_snapshot_saves_bbox_mask_cloud_and_raw_arrays(
    tmp_path,
):
    color = np.zeros((48, 64, 3), dtype=np.uint8)
    color[..., 2] = 80
    mask = np.zeros((48, 64), dtype=bool)
    mask[12:31, 20:43] = True
    cloud = np.zeros((128, 3), dtype=np.float32)
    cloud[:, 0] = np.linspace(-0.02, 0.02, 128)
    cloud[:, 1] = np.linspace(0.01, 0.03, 128)
    cloud[:, 2] = 0.04
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (0.4, 0.0, 0.5)
    sample = V94LiveVisualizationSample(
        sequence=19,
        frame_id=23,
        captured_realtime_s=100.0,
        color_bgr=color,
        object_mask=mask,
        pointcloud_xyzrgb_palm=cloud,
        pointcloud_valid=np.ones(128, dtype=np.float32),
        T_base_palm_at_capture=transform,
        source_valid_points=461,
        requested_object_mask_mode="guarded",
        effective_object_mask_mode="guarded_v2",
        effective_provider_mask_publication_mode="guarded_sam2_primary",
        effective_provider_recovery_publication_mode="unified_three_evidence",
        provider_published_mask_valid=True,
        provider_published_mask_area_px=300,
        provider_published_mask_bbox_xyxy=(21, 13, 40, 27),
        mask_source="adaptive_hand_guard_fallback",
        mask_message="online SAM2 foreign appendage rejected",
        online_sam2_status="tracked_disagree: foreign appendage",
        projector_effective_policy_mask_provenance=(
            "current_frame_fixed_sphere_completion_mask"
        ),
        projector_effective_policy_mask_source_frame_id=23,
        projector_effective_policy_mask_source_captured_realtime_s=100.0,
        projector_effective_policy_mask_area_px=int(mask.sum()),
        projector_effective_policy_mask_bbox_xyxy=(20, 12, 42, 30),
    )

    manifest = _save_visualization_sample(
        sample,
        output_directory=tmp_path,
        selected_bbox_xyxy=(15, 8, 48, 36),
    )

    for name in (
        "color_rgb.png",
        "rgb_with_selected_bbox.png",
        "final_policy_mask.png",
        "final_policy_mask_overlay.png",
        "final_policy_pointcloud_robot_base.png",
        "final_policy_mask.npy",
        "final_policy_points_palm.npy",
        "final_policy_points_robot_base.npy",
        "final_policy_valid.npy",
        "T_base_palm_at_capture.npy",
        "metadata.json",
    ):
        assert (tmp_path / name).is_file()
        assert (tmp_path / name).stat().st_size > 0
    assert manifest["selected_bbox_xyxy"] == [15, 8, 48, 36]
    assert manifest["schema_version"] == 2
    assert manifest["mask_provenance_schema"] == "provider_projector_split_v1"
    assert manifest["final_mask_bbox_xyxy"] == [20, 12, 42, 30]
    assert manifest["final_mask_area_px"] == int(mask.sum())
    assert manifest["requested_object_mask_mode"] == "guarded"
    assert manifest["effective_object_mask_mode"] == "guarded_v2"
    assert (
        manifest["effective_provider_mask_publication_mode"]
        == "guarded_sam2_primary"
    )
    assert (
        manifest["provider_published_mask_source"]
        == "adaptive_hand_guard_fallback"
    )
    assert manifest["provider_published_mask_area_px"] == 300
    assert manifest["provider_published_mask_bbox_xyxy"] == [21, 13, 40, 27]
    assert "foreign appendage" in manifest["provider_published_mask_message"]
    assert "foreign appendage" in manifest["provider_online_sam2_status"]
    assert (
        manifest["projector_effective_policy_mask_provenance"]
        == "current_frame_fixed_sphere_completion_mask"
    )
    assert manifest["projector_effective_policy_mask_source_frame_id"] == 23
    assert manifest["projector_effective_policy_mask_area_px"] == int(mask.sum())
    assert manifest["projector_effective_policy_mask_bbox_xyxy"] == [20, 12, 42, 30]
    assert (
        manifest["projector_effective_source_resolution_mask_area_px"]
        == int(mask.sum())
    )
    assert "final_mask_source" not in manifest
    assert manifest["policy_point_feature_mode"] == "xyz"
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["sequence"] == 19
    np.testing.assert_array_equal(np.load(tmp_path / "final_policy_mask.npy"), mask)
    base = np.load(tmp_path / "final_policy_points_robot_base.npy")
    np.testing.assert_allclose(base[:, 0], cloud[:, 0] + 0.4)
    np.testing.assert_allclose(base[:, 2], cloud[:, 2] + 0.5)


def test_robot_base_snapshot_is_never_labelled_or_saved_as_palm(tmp_path):
    sample = replace(
        _sample(),
        point_coordinate_frame="robot_base_via_identity_T_base_palm",
    )

    manifest = _save_visualization_sample(
        sample,
        output_directory=tmp_path,
        selected_bbox_xyxy=None,
    )

    assert manifest["pointcloud_input_frame"] == "robot_base"
    assert manifest["pointcloud_input_frame_provenance"] == (
        "robot_base_via_identity_T_base_palm"
    )
    assert manifest["palm_frame_geometry_validated"] is False
    assert "raw_policy_points_palm" not in manifest["files"]
    assert not (tmp_path / "final_policy_points_palm.npy").exists()
    np.testing.assert_array_equal(
        np.load(tmp_path / "final_policy_points_robot_base.npy"),
        sample.pointcloud_xyzrgb_palm[:, :3],
    )


def test_robot_base_snapshot_requires_identity_transform(tmp_path):
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 0.1
    sample = replace(
        _sample(),
        point_coordinate_frame="robot_base_via_identity_T_base_palm",
        T_base_palm_at_capture=transform,
    )

    with pytest.raises(ValueError, match="requires identity"):
        _save_visualization_sample(
            sample,
            output_directory=tmp_path,
            selected_bbox_xyxy=None,
        )


def test_close_saves_retained_snapshot_after_viewer_cleanup(tmp_path):
    viewer = V94LiveVisualizer(
        update_rate_hz=10.0,
        save_directory=tmp_path,
        selected_bbox_xyxy=(1, 1, 20, 20),
    )
    # Simulate a short run that offered a policy observation but ended before
    # the GUI bridge published it.
    viewer._latest = _sample()

    viewer.close()

    assert viewer.stats()["snapshot_saved"] is True
    assert viewer.stats()["snapshot_save_error"] is None
    assert (tmp_path / "metadata.json").is_file()
