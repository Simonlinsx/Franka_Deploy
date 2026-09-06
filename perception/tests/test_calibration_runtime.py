from pathlib import Path

import numpy as np
import pytest
import yaml

from dynamic_pcd.calibration.io import (
    calibration_id_for,
    load_calibration,
    resolve_extrinsics,
)
from dynamic_pcd.calibration.transforms import transform_points
from dynamic_pcd.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "d435_default.yaml"
CALIBRATION_PATH = (
    REPO_ROOT / "configs" / "calibrations" / "fr3_d435_eye_to_hand.yaml"
)
EXPECTED_CALIBRATION_ID = "eye-to-hand-b722bce10485c8a3"
EXPECTED_CAMERA_SERIAL = "337322072188"


def _calibration_document():
    with CALIBRATION_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_deployment_config_resolves_verified_calibration():
    cfg = load_config(str(CONFIG_PATH))
    resolved = resolve_extrinsics(cfg["extrinsics"])

    assert Path(resolved.source) == CALIBRATION_PATH.resolve()
    assert resolved.calibrated
    assert resolved.reference_frame == "robot_base"
    assert resolved.base_frame == "robot_base"
    assert resolved.camera_frame == "camera_color_optical_frame"
    assert resolved.camera_serial == EXPECTED_CAMERA_SERIAL
    assert resolved.calibration_id == EXPECTED_CALIBRATION_ID
    assert resolved.quality_status == "pass"
    np.testing.assert_allclose(
        resolved.T_base_camera[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7
    )


def test_recorded_id_is_derived_from_transform_and_camera_serial():
    record = load_calibration(str(CALIBRATION_PATH))

    assert (
        calibration_id_for(record["T_base_camera"], EXPECTED_CAMERA_SERIAL)
        == EXPECTED_CALIBRATION_ID
    )


def test_camera_to_base_transform_has_expected_direction_and_value():
    record = load_calibration(str(CALIBRATION_PATH))
    point_camera = np.asarray([0.1, -0.2, 0.7], dtype=np.float64)

    point_base = transform_points(record["T_base_camera"], point_camera)

    np.testing.assert_allclose(
        point_base,
        [0.59125686, 0.13708076, 0.40691222],
        atol=1e-8,
    )
    inverse_result = transform_points(
        np.linalg.inv(record["T_base_camera"]), point_camera
    )
    assert not np.allclose(point_base, inverse_result, atol=1e-3)


def test_load_calibration_rejects_stale_id_after_transform_change(tmp_path):
    document = _calibration_document()
    document["T_base_camera"][0][3] += 0.001
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration_id mismatch"):
        load_calibration(str(path))


def test_quality_gate_rejects_nonpassing_calibration(tmp_path):
    document = _calibration_document()
    document["solver"]["quality"]["status"] = "fail"
    path = tmp_path / "failed_quality.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="quality must be pass"):
        resolve_extrinsics(
            {
                "calibration_file": str(path),
                "require_calibration": True,
                "require_quality_pass": True,
            }
        )


def test_required_calibration_rejects_identity_fallback():
    with pytest.raises(ValueError, match="requires an eye-to-hand calibration"):
        resolve_extrinsics(
            {
                "require_calibration": True,
                "T_base_camera": np.eye(4).tolist(),
            }
        )
