#!/usr/bin/env python3
"""Validate the portable sim-to-real scene package without third-party modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PACKAGE_ROOT / "scene_manifest.json"
EVIDENCE_CLASSES = {
    "measured",
    "derived",
    "configured",
    "reported",
    "assumed",
    "unknown",
}


class ValidationError(RuntimeError):
    pass


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [sum(float(a[i][k]) * float(b[k][j]) for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def _transpose(a: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*a)]


def _det3(r: Sequence[Sequence[float]]) -> float:
    return (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )


def _max_abs_difference(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    return max(abs(float(x) - float(y)) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def _identity(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _validate_transform(value: Any, name: str, atol: float = 1e-9) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValidationError(f"{name} must be a 4x4 array")
    if any(not isinstance(row, list) or len(row) != 4 for row in value):
        raise ValidationError(f"{name} must be a 4x4 array")
    try:
        matrix = [[float(x) for x in row] for row in value]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} contains a non-numeric value") from exc
    if any(not math.isfinite(x) for row in matrix for x in row):
        raise ValidationError(f"{name} contains a non-finite value")
    if max(abs(matrix[3][i] - [0.0, 0.0, 0.0, 1.0][i]) for i in range(4)) > atol:
        raise ValidationError(f"{name} has an invalid homogeneous last row")
    rotation = [row[:3] for row in matrix[:3]]
    orthogonality = _matmul(_transpose(rotation), rotation)
    if _max_abs_difference(orthogonality, _identity(3)) > atol:
        raise ValidationError(f"{name} rotation is not orthonormal")
    determinant = _det3(rotation)
    if abs(determinant - 1.0) > atol:
        raise ValidationError(f"{name} rotation determinant is {determinant}, expected +1")


def _walk(value: Any, path: str = "root") -> Iterable[Tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _require_mapping(root: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ValidationError(f"manifest.{key} must be an object")
    return value


def _validate_structure(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != "1.0.0":
        raise ValidationError("unsupported or missing schema_version")
    for key in (
        "package",
        "sources",
        "coordinate_conventions",
        "camera",
        "calibration_quality",
        "calibration_target",
        "robot",
        "scene",
        "workspace",
        "training_target",
        "timing_and_sensor_model",
        "readiness",
    ):
        _require_mapping(manifest, key)

    package = manifest["package"]
    if package.get("canonical_length_unit") != "m":
        raise ValidationError("canonical_length_unit must be m")
    if package.get("canonical_angle_unit") != "rad":
        raise ValidationError("canonical_angle_unit must be rad")
    if not isinstance(package.get("live_camera_accessed"), bool):
        raise ValidationError("live_camera_accessed must be a boolean")
    if package.get("live_robot_accessed") is not False:
        raise ValidationError("scene packaging must not access or command the robot")
    if package.get("live_camera_accessed"):
        required_measurements = {
            "table_plane_roi_lower_center",
            "table_plane_roi_lower_right",
            "table_plane_roi_upper_left",
            "table_plane_summary",
        }
        missing = required_measurements - set(manifest["sources"])
        if missing:
            raise ValidationError(
                "live camera survey is declared but measurement sources are missing: "
                + ", ".join(sorted(missing))
            )

    for path, value in _walk(manifest):
        if path.endswith(".evidence_class") and value not in EVIDENCE_CLASSES:
            raise ValidationError(f"{path} has unsupported value {value!r}")

    source_ids = set(manifest["sources"])
    for path, value in _walk(manifest):
        if path.endswith(".source_refs"):
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                raise ValidationError(f"{path} must be a string array")
            missing = set(value) - source_ids
            if missing:
                raise ValidationError(f"{path} names unknown sources: {sorted(missing)}")


def _validate_packaged_sources(manifest: Mapping[str, Any], check_originals: bool) -> int:
    checked = 0
    for source_id, source in manifest["sources"].items():
        portable = source.get("portable_path")
        expected = source.get("sha256")
        if portable is not None:
            relative = Path(str(portable))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValidationError(f"source {source_id} has a non-portable path")
            path = PACKAGE_ROOT / relative
            if not path.is_file():
                raise ValidationError(f"source {source_id} snapshot is missing: {path}")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValidationError(f"source {source_id} has no valid sha256")
            actual = _sha256(path)
            if actual != expected:
                raise ValidationError(
                    f"source {source_id} checksum mismatch: expected {expected}, got {actual}"
                )
            checked += 1

        if check_originals and source.get("original_path") and expected:
            original = Path(str(source["original_path"]))
            if not original.is_file():
                raise ValidationError(f"source {source_id} original is missing: {original}")
            actual = _sha256(original)
            if actual != expected:
                raise ValidationError(
                    f"source {source_id} original changed since packaging: expected {expected}, got {actual}"
                )
    return checked


def _validate_camera(manifest: Mapping[str, Any]) -> None:
    camera = manifest["camera"]
    stream = camera["stream"]
    intrinsics = camera["pinhole_intrinsics"]
    width = int(stream["width_px"])
    height = int(stream["height_px"])
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    if width <= 0 or height <= 0 or fx <= 0.0 or fy <= 0.0:
        raise ValidationError("camera dimensions and focal lengths must be positive")
    if not (0.0 <= cx < width and 0.0 <= cy < height):
        raise ValidationError("camera principal point lies outside the image")
    expected_k = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    if _max_abs_difference(intrinsics["K"], expected_k) > 1e-12:
        raise ValidationError("camera K does not match fx/fy/cx/cy")

    frustum = camera["derived_frustum"]
    hfov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    vfov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    if abs(hfov - float(frustum["horizontal_fov_deg_symmetric_approx"])) > 1e-10:
        raise ValidationError("derived horizontal FOV is inconsistent")
    if abs(vfov - float(frustum["vertical_fov_deg_symmetric_approx"])) > 1e-10:
        raise ValidationError("derived vertical FOV is inconsistent")

    extrinsics = camera["extrinsics"]
    t_base_camera = extrinsics["T_base_camera_color_optical"]
    t_camera_base = extrinsics["T_camera_color_optical_base"]
    _validate_transform(t_base_camera, "camera.extrinsics.T_base_camera_color_optical")
    _validate_transform(t_camera_base, "camera.extrinsics.T_camera_color_optical_base")
    if _max_abs_difference(_matmul(t_base_camera, t_camera_base), _identity(4)) > 1e-9:
        raise ValidationError("camera forward and inverse extrinsics are inconsistent")

    translation = [t_base_camera[i][3] for i in range(3)]
    if max(abs(a - b) for a, b in zip(translation, extrinsics["translation_base_m"])) > 1e-12:
        raise ValidationError("camera translation_base_m is inconsistent with extrinsics")
    axes = extrinsics["optical_axes_in_base"]
    expected_axes = {
        "right_plus_x": [t_base_camera[i][0] for i in range(3)],
        "down_plus_y": [t_base_camera[i][1] for i in range(3)],
        "forward_plus_z": [t_base_camera[i][2] for i in range(3)],
    }
    for name, expected in expected_axes.items():
        if max(abs(a - b) for a, b in zip(axes[name], expected)) > 1e-12:
            raise ValidationError(f"camera optical axis {name} is inconsistent")

    quat = extrinsics["rotation_quaternion_xyzw_camera_to_base"]
    if len(quat) != 4 or abs(sum(float(x) ** 2 for x in quat) - 1.0) > 1e-9:
        raise ValidationError("camera quaternion is not unit length")

    simulator = camera["negative_z_forward_simulator_pose"]
    t_base_sim = simulator["T_base_camera_sim"]
    _validate_transform(t_base_sim, "camera.negative_z_forward_simulator_pose.T_base_camera_sim")
    conversion = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    if _max_abs_difference(_matmul(t_base_camera, conversion), t_base_sim) > 1e-12:
        raise ValidationError("negative-z-forward simulator pose is inconsistent")


def _validate_robot_and_scene(manifest: Mapping[str, Any]) -> None:
    robot = manifest["robot"]
    _validate_transform(robot["base_pose"]["T_world_robot_base"], "robot.base_pose.T_world_robot_base")
    _validate_transform(
        manifest["coordinate_conventions"]["simulation_world"]["T_world_robot_base"],
        "coordinate_conventions.simulation_world.T_world_robot_base",
    )
    _validate_transform(
        manifest["calibration_target"]["T_ee_target_during_calibration"],
        "calibration_target.T_ee_target_during_calibration",
    )

    candidate = robot["candidate_description_joint_limits"]
    for key in ("position_lower", "position_upper", "velocity", "effort"):
        if not isinstance(candidate.get(key), list) or len(candidate[key]) != 7:
            raise ValidationError(f"robot candidate joint limit {key} must have seven values")
    if any(lo >= hi for lo, hi in zip(candidate["position_lower"], candidate["position_upper"])):
        raise ValidationError("robot candidate lower joint limits must be below upper limits")

    hover = manifest["workspace"]["commissioned_hover_safety_envelope"]
    for lower_key, upper_key in (
        ("object_center_min", "object_center_max"),
        ("eef_min_default", "eef_max"),
    ):
        lower = hover[lower_key]
        upper = hover[upper_key]
        if len(lower) != 3 or len(upper) != 3 or any(a >= b for a, b in zip(lower, upper)):
            raise ValidationError(f"invalid hover bounds {lower_key}/{upper_key}")

    tabletop = manifest["scene"]["tabletop"]
    if tabletop.get("evidence_class") == "unknown":
        for key in ("plane_in_robot_base", "height_z_at_reference_xy_m", "corners_robot_base_m"):
            if tabletop.get(key) is not None:
                raise ValidationError(f"unknown tabletop field {key} must remain null")
    elif tabletop.get("evidence_class") == "measured":
        plane = tabletop.get("plane_in_robot_base")
        if not isinstance(plane, list) or len(plane) != 4:
            raise ValidationError("measured tabletop plane must contain [a,b,c,d]")
        try:
            a, b, c, d = (float(value) for value in plane)
        except (TypeError, ValueError) as exc:
            raise ValidationError("measured tabletop plane must be numeric") from exc
        norm = math.sqrt(a * a + b * b + c * c)
        if abs(norm - 1.0) > 1e-9 or c <= 0.0:
            raise ValidationError("measured tabletop normal must be unit length with +z orientation")
        reference_xy = tabletop.get("reference_xy_m")
        if not isinstance(reference_xy, list) or len(reference_xy) != 2:
            raise ValidationError("measured tabletop reference_xy_m must have two values")
        x, y = (float(value) for value in reference_xy)
        expected_z = -(a * x + b * y + d) / c
        actual_z = float(tabletop.get("height_z_at_reference_xy_m"))
        if abs(expected_z - actual_z) > 1e-9:
            raise ValidationError("tabletop reference height is inconsistent with its plane")


def validate(manifest_path: Path, check_originals: bool, require_complete: bool) -> Tuple[int, List[str]]:
    manifest = _load_json(manifest_path)
    _validate_structure(manifest)
    checked = _validate_packaged_sources(manifest, check_originals=check_originals)
    _validate_camera(manifest)
    _validate_robot_and_scene(manifest)

    blockers = manifest["readiness"].get("hard_blockers")
    if not isinstance(blockers, list) or not blockers:
        raise ValidationError("readiness.hard_blockers must be a non-empty list")
    blockers = [str(item) for item in blockers]
    if require_complete:
        raise ValidationError(
            "scene spec is structurally valid but incomplete: " + "; ".join(blockers)
        )
    return checked, blockers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--check-originals",
        action="store_true",
        help="also require original absolute source files and their snapshot hashes",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail while any documented hard blocker remains",
    )
    args = parser.parse_args(argv)
    try:
        checked, blockers = validate(
            args.manifest.resolve(),
            check_originals=args.check_originals,
            require_complete=args.require_complete,
        )
    except ValidationError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    print(f"[PASS] scene manifest semantics and {checked} packaged checksums are valid")
    print(f"[INCOMPLETE] {len(blockers)} hard blockers remain before full sim2real scene alignment:")
    for blocker in blockers:
        print(f"  - {blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
