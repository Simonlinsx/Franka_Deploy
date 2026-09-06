from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.observation.capture import (
    _initialized_roi_evidence,
    capture_valid_frames,
)
from sim2real.observation.pointcloud_filters import (
    FILTER_NAMES,
    build_mask_candidates,
    erode_mask,
    fit_fixed_radius_sphere,
    fit_fixed_radius_sphere_on_support_plane,
    robust_depth_mask,
)


def test_erode_mask_removes_one_pixel_boundary() -> None:
    mask = np.zeros((9, 11), dtype=bool)
    mask[2:7, 3:9] = True
    result = erode_mask(mask, 3)
    expected = np.zeros_like(mask)
    expected[3:6, 4:8] = True
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(mask[2:7, 3:9], True)


def test_robust_depth_mask_rejects_far_mask_pollution() -> None:
    depth = np.full((12, 12), 0.90, dtype=np.float32)
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 2:10] = True
    depth[2, 2:6] = 1.05
    selected, center, width = robust_depth_mask(depth, mask)
    assert center == pytest.approx(0.90, abs=1.0e-6)
    assert width == pytest.approx(0.012)
    assert not np.any(selected[2, 2:6])
    assert np.count_nonzero(selected) == np.count_nonzero(mask) - 4


def test_fixed_radius_sphere_fit_is_robust_to_sparse_outliers() -> None:
    rng = np.random.default_rng(7)
    center = np.asarray([0.568, 0.044, 0.068])
    radius = 0.030
    directions = rng.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    surface = center + radius * directions
    surface += rng.normal(scale=0.0005, size=surface.shape)
    outliers = rng.uniform([0.48, -0.03, 0.00], [0.66, 0.12, 0.15], size=(35, 3))
    fit = fit_fixed_radius_sphere(
        np.concatenate((surface, outliers)),
        camera_origin_base_m=np.asarray([0.10, -0.20, 0.80]),
        radius_m=radius,
    )
    assert fit.valid
    np.testing.assert_allclose(fit.center_base_m, center, atol=0.002)
    assert fit.residual_median_m < 0.0015
    assert fit.shell_inlier_fraction > 0.85


def test_build_mask_candidates_has_explicit_fixed_set() -> None:
    depth = np.full((30, 40), 0.90, dtype=np.float32)
    mask = np.zeros((30, 40), dtype=bool)
    mask[8:23, 12:29] = True
    depth[8, 12:18] = 1.10
    K = np.asarray([[100.0, 0.0, 20.0], [0.0, 100.0, 15.0], [0.0, 0.0, 1.0]])
    candidates = build_mask_candidates(
        depth,
        mask,
        camera_K=K,
        T_base_camera_optical=np.eye(4),
        sphere_radius_m=None,
    )
    assert tuple(candidates.masks) == FILTER_NAMES
    assert np.count_nonzero(candidates.masks["erode3"]) < np.count_nonzero(mask)
    assert np.count_nonzero(candidates.masks["robust_depth"]) == np.count_nonzero(mask) - 6
    assert not np.any(candidates.masks["sphere_shell_support_plane"])
    assert not candidates.sphere_fit.valid


def test_support_plane_constrained_sphere_has_physical_center_height() -> None:
    rng = np.random.default_rng(13)
    plane = np.asarray([0.0, 0.0, 1.0, -0.038])
    center = np.asarray([0.568, 0.044, 0.068])
    radius = 0.030
    directions = rng.normal(size=(400, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    surface = center + radius * directions
    surface += rng.normal(scale=0.0005, size=surface.shape)
    fit = fit_fixed_radius_sphere_on_support_plane(
        surface,
        support_plane_abcd=plane,
        radius_m=radius,
    )
    assert fit.valid
    np.testing.assert_allclose(fit.center_base_m, center, atol=0.001)
    assert fit.center_base_m[2] == pytest.approx(0.068, abs=1.0e-10)


class _FakeIntrinsics:
    distortion = ()

    @staticmethod
    def as_matrix() -> np.ndarray:
        return np.eye(3, dtype=np.float32)


class _FakeProvider:
    def __init__(self) -> None:
        self.index = 0
        self.last_timings_ms = {}

    def step_mask_only(self):
        self.index += 1
        valid = self.index != 1
        frame = SimpleNamespace(
            color_bgr=np.full((4, 5, 3), self.index, dtype=np.uint8),
            depth_raw=np.full((4, 5), 900, dtype=np.uint16),
            depth_scale=0.001,
            intrinsics=_FakeIntrinsics(),
            timestamp=100.0 + self.index / 30.0,
            retrieved_at_s=100.01 + self.index / 30.0,
            frame_id=self.index,
            sensor_frame_number=1000 + self.index,
            depth_sensor_frame_number=2000 + self.index,
            timestamp_domain="timestamp_domain.global_time",
        )
        mask = np.ones((4, 5), dtype=np.uint8) if valid else np.zeros((4, 5), dtype=np.uint8)
        result = SimpleNamespace(
            valid=valid,
            mask=mask,
            bbox_xyxy=np.asarray([0, 0, 5, 4]),
            message="valid" if valid else "invalid",
        )
        self.last_timings_ms = {"camera": 4.0, "total": 6.0}
        return frame, result


def test_capture_valid_frames_is_bounded_thinned_and_disk_free() -> None:
    payload, counters = capture_valid_frames(
        _FakeProvider(), stored_frames=2, sample_every_valid_frame=2
    )
    # Step 1 is invalid; valid publications 1 and 3 (provider steps 2 and 4)
    # are selected while the intervening valid publication is tracked only.
    np.testing.assert_array_equal(payload["camera_frame_id"], [2, 4])
    assert payload["rgb"].shape == (2, 4, 5, 3)
    assert payload["depth_raw"].dtype == np.uint16
    assert payload["object_mask"].dtype == np.bool_
    assert counters == {
        "provider_steps": 4,
        "valid_publications": 3,
        "invalid_publications": 1,
        "retryable_camera_timeouts": 0,
        "maximum_consecutive_retryable_camera_timeouts": 0,
        "stored_frames": 2,
        "invalid_reasons": {"invalid": 1},
    }


def test_capture_can_wait_for_a_consecutive_valid_terminal_streak() -> None:
    provider = _FakeProvider()
    original = provider.step_mask_only

    def valid_valid_invalid_then_valid_valid():
        frame, result = original()
        result.valid = provider.index not in (1, 3)
        result.message = (
            "valid"
            if result.valid
            else ("invalid" if provider.index == 1 else "lost")
        )
        return frame, result

    provider.step_mask_only = valid_valid_invalid_then_valid_valid
    payload, counters = capture_valid_frames(
        provider,
        stored_frames=2,
        maximum_attempts=5,
        require_consecutive_valid_frames=True,
    )

    np.testing.assert_array_equal(payload["camera_frame_id"], [4, 5])
    assert counters["provider_steps"] == 5
    assert counters["valid_publications"] == 3
    assert counters["invalid_publications"] == 2
    assert counters["invalid_reasons"] == {"invalid": 1, "lost": 1}


def test_capture_valid_frames_rejects_non_increasing_frame_id() -> None:
    provider = _FakeProvider()
    original = provider.step_mask_only

    def repeated():
        frame, result = original()
        if provider.index == 2:
            frame.frame_id = 1
        return frame, result

    provider.step_mask_only = repeated
    with pytest.raises(RuntimeError, match="strictly increase"):
        capture_valid_frames(provider, stored_frames=1)


def test_capture_valid_frames_retries_an_isolated_camera_timeout() -> None:
    class RetryableCameraTimeout(RuntimeError):
        retryable_camera_timeout = True

    provider = _FakeProvider()
    original = provider.step_mask_only
    calls = 0

    def timeout_then_frames():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableCameraTimeout("temporary RGB-D skew")
        return original()

    provider.step_mask_only = timeout_then_frames
    payload, counters = capture_valid_frames(provider, stored_frames=1)

    np.testing.assert_array_equal(payload["camera_frame_id"], [2])
    assert calls == 3
    assert counters["provider_steps"] == 2
    assert counters["retryable_camera_timeouts"] == 1
    assert counters["maximum_consecutive_retryable_camera_timeouts"] == 1


def test_capture_valid_frames_rejects_sustained_camera_timeouts() -> None:
    class RetryableCameraTimeout(RuntimeError):
        retryable_camera_timeout = True

    provider = _FakeProvider()
    calls = 0

    def always_timeout():
        nonlocal calls
        calls += 1
        raise RetryableCameraTimeout("camera remains unavailable")

    provider.step_mask_only = always_timeout
    with pytest.raises(RuntimeError, match="exceeded 2 consecutive"):
        capture_valid_frames(
            provider,
            stored_frames=1,
            maximum_consecutive_retryable_camera_timeouts=2,
        )
    assert calls == 3


def test_interactive_capture_records_actual_numeric_prompt_roi() -> None:
    provider = SimpleNamespace(
        last_bbox_initialization_evidence=SimpleNamespace(
            source="grabcut",
            prompt_bbox_xyxy=np.asarray([420, 240, 516, 328], dtype=np.int32),
            mask_bbox_xyxy=np.asarray([441, 259, 489, 307], dtype=np.int32),
            mask_area_px=1234,
        )
    )

    evidence = _initialized_roi_evidence(provider)

    np.testing.assert_array_equal(evidence["roi_xywh"], [420, 240, 96, 88])
    np.testing.assert_array_equal(
        evidence["initialization_prompt_bbox_xyxy"], [420, 240, 516, 328]
    )
    np.testing.assert_array_equal(
        evidence["initialization_mask_bbox_xyxy"], [441, 259, 489, 307]
    )
    assert evidence["initialization_mask_area_px"].item() == 1234
    assert evidence["initialization_mask_source"].item() == "grabcut"


@pytest.mark.parametrize(
    "provider",
    [
        SimpleNamespace(last_bbox_initialization_evidence=None),
        SimpleNamespace(
            last_bbox_initialization_evidence=SimpleNamespace(
                source="grabcut",
                prompt_bbox_xyxy=np.asarray([1, 2, 3]),
                mask_bbox_xyxy=np.asarray([1, 2, 3, 4]),
                mask_area_px=4,
            )
        ),
        SimpleNamespace(
            last_bbox_initialization_evidence=SimpleNamespace(
                source="",
                prompt_bbox_xyxy=np.asarray([1, 2, 3, 4]),
                mask_bbox_xyxy=np.asarray([1, 2, 3, 4]),
                mask_area_px=4,
            )
        ),
    ],
)
def test_capture_refuses_missing_or_malformed_initialized_roi(provider) -> None:
    with pytest.raises(RuntimeError, match="(evidence|mask source)"):
        _initialized_roi_evidence(provider)
