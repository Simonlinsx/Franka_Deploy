from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT = (
    WORKSPACE
    / "perception"
    / "scripts"
    / "validate_thrown_catch_pointcloud.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "validate_thrown_catch_pointcloud", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _overlay() -> dict:
    minimum = np.asarray(
        [0.0342689514, -0.0960582733, 0.4168298006], dtype=np.float64
    )
    maximum = np.asarray(
        [0.3342689514, 0.2039417267, 0.7168298006], dtype=np.float64
    )
    vertices = [
        [x, y, z]
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    return {
        "reference_frame": "robot_base",
        "minimum_base_m": minimum.tolist(),
        "maximum_base_m": maximum.tolist(),
        "vertices_base_m": vertices,
        "all_vertices_inside_image": True,
        "lossless_frames_annotated": False,
    }


def test_workspace_requires_exact_robot_base_overlay_contract() -> None:
    module = _module()
    expected_minimum = np.asarray(_overlay()["minimum_base_m"])
    expected_maximum = np.asarray(_overlay()["maximum_base_m"])
    minimum, maximum = module._workspace(
        {"live_workspace_overlay": _overlay()},
        expected_minimum=expected_minimum,
        expected_maximum=expected_maximum,
    )
    assert minimum.tolist() == pytest.approx(expected_minimum.tolist())
    assert maximum.tolist() == pytest.approx(expected_maximum.tolist())

    malformed = _overlay()
    malformed["lossless_frames_annotated"] = True
    with pytest.raises(ValueError, match="must not modify lossless"):
        module._workspace(
            {"live_workspace_overlay": malformed},
            expected_minimum=expected_minimum,
            expected_maximum=expected_maximum,
        )

    malformed = _overlay()
    malformed["vertices_base_m"][0][0] += 0.001
    with pytest.raises(ValueError, match="vertices do not match"):
        module._workspace(
            {"live_workspace_overlay": malformed},
            expected_minimum=expected_minimum,
            expected_maximum=expected_maximum,
        )

    with pytest.raises(ValueError, match="differs from V57"):
        module._workspace(
            {"live_workspace_overlay": _overlay()},
            expected_minimum=expected_minimum + np.asarray([0.001, 0.0, 0.0]),
            expected_maximum=expected_maximum,
        )


def test_exact_current_128_rejects_stale_partial_and_nonfinite() -> None:
    module = _module()
    frame = SimpleNamespace(
        status="fresh",
        frame_id=7,
        captured_at_s=1.25,
        xyzrgb_palm=np.ones((128, 3), dtype=np.float32),
        valid=np.ones(128, dtype=np.float32),
    )
    assert module._exact_current_128(frame, frame_id=7, timestamp=1.25)

    frame.status = "stale_palm"
    assert not module._exact_current_128(frame, frame_id=7, timestamp=1.25)
    frame.status = "fresh"
    frame.valid[-1] = 0.0
    assert not module._exact_current_128(frame, frame_id=7, timestamp=1.25)
    frame.valid[-1] = 1.0
    frame.xyzrgb_palm[0, 0] = np.nan
    assert not module._exact_current_128(frame, frame_id=7, timestamp=1.25)


def test_physical_report_must_bind_independent_reference_and_calibration(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "physical.json"
    report = {
        "schema_version": 1,
        "kind": "eye_to_hand_rgbd_physical_validation",
        "physical_rgbd_status": "pass",
        "independent_known_base_evidence_present": True,
        "failures": [],
        "missing_evidence": [],
        "camera_serial": "342222071785",
        "calibration_id": "eye-to-hand-test",
        "calibration_sha256": "a" * 64,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    assert module._physical_report(
        path,
        serial="342222071785",
        calibration_id="eye-to-hand-test",
        calibration_sha256="a" * 64,
    ) == report

    report["independent_known_base_evidence_present"] = False
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="no independent"):
        module._physical_report(
            path,
            serial="342222071785",
            calibration_id="eye-to-hand-test",
            calibration_sha256="a" * 64,
        )


def test_inside_uses_all_three_robot_base_axes() -> None:
    module = _module()
    minimum = np.asarray([0.034, -0.096, 0.416])
    maximum = np.asarray([0.334, 0.204, 0.717])
    points = np.asarray(
        [
            [0.18, 0.05, 0.56],
            [0.033, 0.05, 0.56],
            [0.18, 0.205, 0.56],
            [0.18, 0.05, 0.718],
        ]
    )
    assert module._inside(points, minimum, maximum).tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_runtime_calibration_binding_is_exact_and_native_60hz() -> None:
    module = _module()
    config_path = WORKSPACE / "perception" / "configs" / "d435_default.yaml"
    calibration_path = (
        WORKSPACE
        / "perception"
        / "configs"
        / "calibrations"
        / "fr3_d435_342222071785_eye_to_hand_424x240_60hz_reuse_v1.yaml"
    )
    cfg = module.load_config(
        str(config_path),
        overrides={
            "camera": {
                "serial": "342222071785",
                "width": 424,
                "height": 240,
                "fps": 60,
            },
            "extrinsics": {
                "calibration_file": str(calibration_path),
                "require_calibration": True,
                "require_quality_pass": True,
                "strict_camera_serial": True,
            },
        },
    )
    resolved = module.resolve_extrinsics(cfg["extrinsics"])
    calibration_path = Path(resolved.source).resolve()
    manifest = {
        "camera_serial": "342222071785",
        "image_width": 424,
        "image_height": 240,
        "nominal_fps": 60,
        "calibration_id": resolved.calibration_id,
        "T_base_camera": np.asarray(
            resolved.T_base_camera, dtype=np.float32
        ).tolist(),
    }
    module._bind_runtime_calibration(
        cfg=cfg,
        calibration_path=calibration_path,
        manifest=manifest,
    )

    wrong_transform = dict(manifest)
    wrong_transform["T_base_camera"] = np.asarray(
        resolved.T_base_camera, dtype=np.float32
    ).copy()
    wrong_transform["T_base_camera"][0, 3] += np.float32(0.001)
    wrong_transform["T_base_camera"] = wrong_transform["T_base_camera"].tolist()
    with pytest.raises(ValueError, match="T_base_camera differs"):
        module._bind_runtime_calibration(
            cfg=cfg,
            calibration_path=calibration_path,
            manifest=wrong_transform,
        )

    wrong_profile = dict(manifest, nominal_fps=30)
    with pytest.raises(ValueError, match="camera profile differs"):
        module._bind_runtime_calibration(
            cfg=cfg,
            calibration_path=calibration_path,
            manifest=wrong_profile,
        )
