from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import yaml

from dynamic_pcd.calibration.transforms import validate_rigid_transform


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResolvedExtrinsics:
    T_base_camera: np.ndarray
    calibrated: bool
    source: str
    calibration_id: Optional[str]
    base_frame: str
    camera_frame: str
    camera_serial: Optional[str]
    quality_status: Optional[str] = None

    @property
    def reference_frame(self) -> str:
        return self.base_frame if self.calibrated else self.camera_frame


def calibration_id_for(
    T_base_camera: np.ndarray, camera_serial: Optional[str]
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(T_base_camera, dtype="<f8").tobytes())
    digest.update((camera_serial or "unknown").encode("utf-8"))
    return "eye-to-hand-" + digest.hexdigest()[:16]


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return values


def load_calibration(path: str) -> Dict[str, Any]:
    source = Path(path)
    document = _load_yaml_mapping(source)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported calibration schema_version="
            f"{document.get('schema_version')!r}"
        )
    if document.get("kind") != "camera_robot_extrinsic_calibration":
        raise ValueError(f"{source} is not a camera/robot extrinsic calibration")
    if document.get("calibration_type") != "eye_to_hand":
        raise ValueError("only fixed-camera eye_to_hand calibrations are supported")
    if not document.get("base_frame") or not document.get("camera_frame"):
        raise ValueError("calibration must name base_frame and camera_frame")

    transform = validate_rigid_transform(
        document.get("T_base_camera"), name="calibration.T_base_camera"
    )
    document["T_base_camera"] = transform
    if document.get("T_ee_target") is not None:
        document["T_ee_target"] = validate_rigid_transform(
            document["T_ee_target"], name="calibration.T_ee_target"
        )

    camera_values = document.get("camera") or {}
    serial = camera_values.get("serial")
    serial_text = None if serial in (None, "") else str(serial)
    recorded_id = str(document.get("calibration_id") or "")
    expected_id = calibration_id_for(transform, serial_text)
    if not recorded_id:
        raise ValueError("calibration has no calibration_id")
    if recorded_id != expected_id:
        raise ValueError(
            f"calibration_id mismatch: file has {recorded_id}, expected {expected_id}"
        )
    return document


def resolve_extrinsics(
    config: Optional[Mapping[str, Any]],
) -> ResolvedExtrinsics:
    values = dict(config or {})
    calibration_file = values.get("calibration_file")
    require_calibration = bool(values.get("require_calibration", False))
    require_quality_pass = bool(values.get("require_quality_pass", False))

    if calibration_file:
        record = load_calibration(str(calibration_file))
        camera_values = record.get("camera") or {}
        quality = (record.get("solver") or {}).get("quality") or {}
        quality_status = (
            None if quality.get("status") is None else str(quality["status"])
        )
        if require_quality_pass and quality_status != "pass":
            raise ValueError(
                f"calibration quality must be pass, got {quality_status or 'missing'}"
            )
        return ResolvedExtrinsics(
            T_base_camera=record["T_base_camera"].astype(np.float32),
            calibrated=True,
            source=str(Path(str(calibration_file)).resolve()),
            calibration_id=str(record["calibration_id"]),
            base_frame=str(record["base_frame"]),
            camera_frame=str(record["camera_frame"]),
            camera_serial=(
                None
                if camera_values.get("serial") in (None, "")
                else str(camera_values["serial"])
            ),
            quality_status=quality_status,
        )

    transform = validate_rigid_transform(
        values.get("T_base_camera", np.eye(4)), name="extrinsics.T_base_camera"
    )
    calibrated_value = values.get("calibrated")
    calibrated = (
        not np.allclose(transform, np.eye(4))
        if calibrated_value is None
        else bool(calibrated_value)
    )
    if require_calibration and not calibrated:
        raise ValueError(
            "robot-frame output requires an eye-to-hand calibration; set "
            "extrinsics.calibration_file"
        )
    return ResolvedExtrinsics(
        T_base_camera=transform.astype(np.float32),
        calibrated=calibrated,
        source="inline" if calibrated else "identity (camera frame)",
        calibration_id=(
            calibration_id_for(transform, None) if calibrated else None
        ),
        base_frame=str(values.get("base_frame", "robot_base")),
        camera_frame=str(
            values.get("camera_frame", "camera_color_optical_frame")
        ),
        camera_serial=None,
        quality_status=None,
    )
