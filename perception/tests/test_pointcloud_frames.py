import numpy as np
import pytest

from dynamic_pcd.pointcloud.extractor import ObjectPointCloudExtractor
from dynamic_pcd.types import CameraIntrinsics, ObjectPCDPacket, RGBDFrame


def _synthetic_frame_and_mask():
    height, width = 8, 10
    frame = RGBDFrame(
        color_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        depth_raw=np.full((height, width), 1000, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=100.0,
            fy=100.0,
            ppx=4.5,
            ppy=3.5,
        ),
        timestamp=1.0,
        frame_id=1,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[1:7, 2:8] = 1
    return frame, mask


def _extract(center_policy_points: bool):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [0.5, 0.05, 0.1]
    extractor = ObjectPointCloudExtractor(
        {
            "num_points": 64,
            "use_rgb": False,
            "center_policy_points": center_policy_points,
            "erode_kernel": 0,
            "stride": 1,
            "voxel_size": 0.0,
            "remove_outliers": False,
        },
        T_base_camera=transform,
    )
    frame, mask = _synthetic_frame_and_mask()
    return extractor.extract(frame, mask)


def test_reference_points_remain_absolute_when_policy_points_are_centered():
    result = _extract(center_policy_points=True)

    assert result.valid
    assert result.points.shape == (36, 3)
    assert result.policy_points.shape == (64, 3)
    assert result.reference_points.shape == (64, 3)
    np.testing.assert_allclose(result.center, [0.5, 0.05, 1.1], atol=1e-6)
    np.testing.assert_allclose(
        result.policy_points + result.center[None, :],
        result.reference_points,
        atol=1e-6,
    )
    # Absolute reference samples must live near the transformed cloud, not origin.
    assert float(np.median(result.reference_points[:, 2])) > 1.0


def test_uncentered_policy_points_equal_absolute_reference_points():
    result = _extract(center_policy_points=False)

    assert result.valid
    np.testing.assert_allclose(
        result.policy_points, result.reference_points, atol=0.0
    )


def test_sanitized_core_extraction_preserves_sparse_verified_depth_pixels():
    """Normal masks keep erosion; an already-clean occlusion core does not."""

    height, width = 12, 50
    frame = RGBDFrame(
        color_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        depth_raw=np.full((height, width), 1000, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=100.0,
            fy=100.0,
            ppx=24.5,
            ppy=5.5,
        ),
        timestamp=1.0,
        frame_id=1,
    )
    # The reviewed hand-occlusion regression leaves a similarly thin
    # appearance-clean core. A 3x3 erosion deletes it completely, although all
    # 70 exact-mask depth pixels are valid.
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[5:7, 5:40] = 1
    extractor = ObjectPointCloudExtractor(
        {
            "num_points": 128,
            "use_rgb": False,
            "center_policy_points": False,
            "erode_kernel": 3,
            "stride": 1,
            "voxel_size": 0.0,
            "remove_outliers": False,
        },
        T_base_camera=np.eye(4),
    )

    ordinary = extractor.extract(frame, mask)
    sanitized = extractor.extract_sanitized_core(
        frame, mask, erode_kernel=0
    )

    assert not ordinary.valid
    assert sanitized.valid
    assert sanitized.points.shape == (70, 3)
    assert sanitized.policy_points.shape == (128, 3)


def test_scene_cloud_uses_same_base_transform_and_excludes_object_mask():
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [0.5, 0.05, 0.1]
    extractor = ObjectPointCloudExtractor(
        {
            "z_min": 0.25,
            "z_max": 1.20,
            "erode_kernel": 0,
            "voxel_size": 0.0,
            "remove_outliers": False,
        },
        T_base_camera=transform,
    )
    frame, mask = _synthetic_frame_and_mask()
    frame.color_bgr[:] = [10, 20, 30]

    scene = extractor.extract_scene(frame, exclude_mask=mask, stride=2)

    # 4x5 subsampled pixels minus 3x3 target pixels.
    assert scene.points.shape == (11, 3)
    assert scene.colors.shape == (11, 3)
    np.testing.assert_allclose(scene.points[0], [0.455, 0.015, 1.1], atol=1e-6)
    np.testing.assert_allclose(
        scene.colors[0], np.asarray([30, 20, 10]) / 255.0, atol=1.0 / 255.0
    )
    # Both object and scene APIs must use exactly the same robot-base transform.
    obj = extractor.extract(frame, mask)
    assert obj.valid
    np.testing.assert_allclose(np.median(scene.points[:, 2]), obj.center[2], atol=1e-6)


def test_scene_cloud_rejects_misaligned_exclusion_mask():
    frame, _ = _synthetic_frame_and_mask()
    extractor = ObjectPointCloudExtractor(
        {"z_min": 0.25, "z_max": 1.20}, T_base_camera=np.eye(4)
    )

    with pytest.raises(ValueError, match="exclude_mask shape"):
        extractor.extract_scene(frame, exclude_mask=np.ones((2, 2)), stride=2)


def test_object_cloud_removes_table_band_without_changing_semantic_mask():
    frame, mask = _synthetic_frame_and_mask()
    # Split the selected object pixels between the calibrated support surface
    # at z=1.0 and an object cap 20 mm above it.
    frame.depth_raw[1:4, 2:8] = 1000
    frame.depth_raw[4:7, 2:8] = 1020
    frame.depth_raw[1, 2] = 0
    extractor = ObjectPointCloudExtractor(
        {
            "num_points": 32,
            "erode_kernel": 0,
            "voxel_size": 0.0,
            "remove_outliers": False,
            "support_plane_abcd": [0.0, 0.0, 1.0, -1.0],
            "support_plane_min_clearance_m": 0.006,
        },
        T_base_camera=np.eye(4),
    )

    result = extractor.extract(frame, mask)

    assert result.valid
    assert result.points.shape == (18, 3)
    assert np.all(result.points[:, 2] > 1.006)
    filtered_mask = extractor.filter_semantic_mask_above_support_plane(
        frame, mask
    )
    # Table pixels with reliable depth are suppressed, while the unknown-depth
    # SAM2 silhouette at [1,2] is retained to avoid clipping object edges.
    assert int(filtered_mask.sum()) == 19
    assert filtered_mask[1, 2] == 1
    assert not np.any(filtered_mask[1:4, 3:8])
    assert np.all(filtered_mask[4:7, 2:8])
    # The caller-owned semantic mask remains untouched for tracker state.
    assert int(mask.sum()) == 36


@pytest.mark.parametrize(
    "config, message",
    [
        ({"support_plane_abcd": [0.0, 0.0, 0.0, 1.0]}, "normal must be non-zero"),
        (
            {
                "support_plane_abcd": [0.0, 0.0, 1.0, 0.0],
                "support_plane_min_clearance_m": -0.001,
            },
            "must be finite and non-negative",
        ),
    ],
)
def test_support_plane_configuration_is_validated(config, message):
    with pytest.raises(ValueError, match=message):
        ObjectPointCloudExtractor(config, T_base_camera=np.eye(4))


def test_policy_observation_preserves_frame_and_calibration_provenance():
    reference = np.asarray([[0.55, 0.06, 0.12]], dtype=np.float32)
    packet = ObjectPCDPacket(
        pcd_current=reference.copy(),
        pcd_history=None,
        center=np.asarray([0.55, 0.06, 0.12], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        bbox_xyxy=np.asarray([1, 2, 3, 4], dtype=np.int32),
        timestamp=10.0,
        frame_id=4,
        valid=True,
        pcd_reference=reference,
        reference_frame="robot_base",
        point_frame="robot_base",
        calibration_id="eye-to-hand-test",
        T_base_camera=np.eye(4, dtype=np.float32),
        camera_serial="camera-test",
    )

    observation = packet.to_policy_obs()

    assert observation["object_pcd_reference"] is reference
    assert observation["reference_frame"] == "robot_base"
    assert observation["point_frame"] == "robot_base"
    assert observation["calibration_id"] == "eye-to-hand-test"
    assert observation["camera_serial"] == "camera-test"
    np.testing.assert_array_equal(observation["T_base_camera"], np.eye(4))
