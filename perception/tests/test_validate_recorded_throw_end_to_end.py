from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT = (
    WORKSPACE
    / "perception"
    / "scripts"
    / "validate_recorded_throw_end_to_end.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "validate_recorded_throw_end_to_end", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # ``dataclasses`` resolves postponed annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mask_metrics_and_invalid_run_are_fail_closed() -> None:
    module = _module()
    reference = np.zeros((4, 5), dtype=bool)
    reference[1:3, 1:4] = True
    actual = np.zeros_like(reference)
    actual[1:3, 2:4] = True

    metrics = module._mask_metrics(actual, reference)
    assert metrics == pytest.approx(
        {"iou": 4.0 / 6.0, "recall": 4.0 / 6.0, "precision": 1.0}
    )
    assert module._longest_false_run([True, False, False, True, False]) == 2


def test_reference_indices_require_one_60hz_to_20hz_phase(tmp_path: Path) -> None:
    module = _module()
    mask = np.zeros((4, 5), dtype=np.uint8)
    for index in (3, 6, 9):
        assert cv2.imwrite(str(tmp_path / f"{index:06d}.png"), mask)
    assert module._reference_indices(tmp_path) == (3, 6, 9)

    assert cv2.imwrite(str(tmp_path / "000010.png"), mask)
    with pytest.raises(Exception, match="one 60 Hz -> 20 Hz phase"):
        module._reference_indices(tmp_path)


@pytest.mark.parametrize("use_rgb,expected_dim", [(False, 3), (True, 6)])
def test_projector_matches_live_depth_and_feature_contract(
    use_rgb: bool, expected_dim: int
) -> None:
    module = _module()
    case = SimpleNamespace(
        width=424,
        height=240,
        intrinsics=SimpleNamespace(
            fx=302.3836364746094,
            fy=302.2502746582031,
            ppx=210.95738220214844,
            ppy=123.3145980834961,
        ),
    )
    provider = SimpleNamespace(
        extrinsics=SimpleNamespace(T_base_camera=np.eye(4, dtype=np.float64))
    )
    cfg = {
        "camera": {"z_min": 0.25, "z_max": 1.2},
        "pointcloud": {
            "use_rgb": use_rgb,
            "temporal_fallback": "motion_compensated",
            "temporal_fallback_max_stale_s": 0.25,
            "temporal_fallback_max_stale_steps": 5,
            "temporal_fallback_max_image_speed_px_s": 2400.0,
        },
    }

    projector = module._projector(case, provider, cfg)
    assert projector.num_points == 128
    assert projector.minimum_valid_points == 16
    assert projector.point_feature_dim == expected_dim
    assert projector.maximum_mask_depth_deviation_m == pytest.approx(0.055)
    assert projector.temporal_fallback == "motion_compensated"
    assert projector.temporal_fallback_max_stale_s == pytest.approx(0.25)
    assert projector.temporal_fallback_max_stale_steps == 5
    assert (projector.width, projector.height) == (424, 240)
    assert (projector.z_min, projector.z_max) == pytest.approx((0.25, 1.2))
