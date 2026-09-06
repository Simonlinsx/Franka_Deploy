import cv2
import numpy as np
import pytest

from dynamic_pcd.segmentation import manual
from dynamic_pcd.segmentation.manual import GrabCutMaskInitializer
from dynamic_pcd.types import CameraIntrinsics, RGBDFrame


def _frame() -> RGBDFrame:
    height, width = 24, 32
    return RGBDFrame(
        color_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        depth_raw=np.full((height, width), 700, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=50.0,
            fy=50.0,
            ppx=width / 2,
            ppy=height / 2,
        ),
        timestamp=1.0,
        frame_id=1,
    )


def _assert_failed_without_box_fallback(result, frame, expected_message):
    assert not result.valid
    assert result.score == 0.0
    assert result.source == "grabcut_failed"
    assert expected_message in result.message
    assert "box" not in result.source.lower()
    assert "box" not in result.message.lower()
    assert result.mask.shape == frame.depth_raw.shape
    assert result.mask.dtype == np.uint8
    assert not np.any(result.mask)


def test_grabcut_too_small_prompt_fails_closed(monkeypatch):
    frame = _frame()

    def unexpected_grabcut(*_args, **_kwargs):
        pytest.fail("cv2.grabCut must not run for an undersized prompt")

    monkeypatch.setattr(manual.cv2, "grabCut", unexpected_grabcut)
    result = GrabCutMaskInitializer().initialize(
        frame,
        np.asarray([4, 6, 8, 10], dtype=np.int32),
    )

    _assert_failed_without_box_fallback(result, frame, "smaller than 5x5")
    np.testing.assert_array_equal(result.bbox_xyxy, [4, 6, 8, 10])


@pytest.mark.parametrize("foreground_pixels", [0, 19])
def test_grabcut_too_few_foreground_pixels_fails_closed(
    monkeypatch,
    foreground_pixels,
):
    frame = _frame()

    def sparse_grabcut(_image, mask, _rect, _bgd, _fgd, _iters, _mode):
        mask.flat[:foreground_pixels] = cv2.GC_FGD

    monkeypatch.setattr(manual.cv2, "grabCut", sparse_grabcut)
    result = GrabCutMaskInitializer(morph_kernel=1).initialize(
        frame,
        np.asarray([2, 3, 28, 21], dtype=np.int32),
    )

    _assert_failed_without_box_fallback(result, frame, "fewer than 20")
    np.testing.assert_array_equal(result.bbox_xyxy, [2, 3, 28, 21])


def test_grabcut_exception_fails_closed(monkeypatch):
    frame = _frame()

    def failing_grabcut(*_args, **_kwargs):
        raise RuntimeError("synthetic OpenCV failure")

    monkeypatch.setattr(manual.cv2, "grabCut", failing_grabcut)
    result = GrabCutMaskInitializer().initialize(
        frame,
        np.asarray([2, 3, 28, 21], dtype=np.int32),
    )

    _assert_failed_without_box_fallback(result, frame, "synthetic OpenCV failure")
    assert "RuntimeError" in result.message
    np.testing.assert_array_equal(result.bbox_xyxy, [2, 3, 28, 21])
