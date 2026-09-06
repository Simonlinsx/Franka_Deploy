"""Fail-closed evidence for returns from a rigidly routed tool cable.

This module deliberately does *not* turn a dark point into a robot point.  A
point may be excluded from the static scene only when all of the following are
bound in one replayable record:

* the operator identifies the exact component and confirms that it is rigidly
  routed to the installed RH56;
* the selected point indices lie inside a measured cable route in the hand
  frame and form a dark, connected component;
* the complete selected component and route are away from the camera border;
* at least two captures provide useful tool-pose excitation and the component
  follows the hand-frame motion substantially better than a static obstacle.

Even a passing result authorizes only removal of the *listed indices* from a
captured static-obstacle cloud.  It never authorizes robot motion.  The route
and its radius must additionally be inserted as moving collision geometry and
audited over the complete robot path.  This distinction prevents a flexible
cable from disappearing from both sides of a collision query.

The implementation is hardware-independent and opens no camera, serial port,
FCI connection, or GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "rh56_attached_cable_return_evidence"
ALGORITHM = "two_pose_rigid_hand_frame_component_v1"

IDENTITY_CONFIRMATION_TOKEN = "RH56_ATTACHED_CABLE_VISUALLY_IDENTIFIED"
RIGID_ROUTE_CONFIRMATION_TOKEN = "RH56_CABLE_RIGID_ROUTE_UNCHANGED"

# Fixed policy limits.  They are intentionally not function arguments: a
# caller cannot make a difficult scene pass by weakening thresholds.
MIN_COMPONENT_POINTS = 8
MAX_COMPONENT_POINTS = 5000
MAX_COMPONENT_EDGE_M = 0.008
MIN_COMPONENT_SPAN_M = 0.010
MAX_ROUTE_RADIUS_M = 0.010
DEPTH_RETURN_ROUTE_GUARD_M = 0.004
MAX_ROUTE_RESIDUAL_M = MAX_ROUTE_RADIUS_M + DEPTH_RETURN_ROUTE_GUARD_M
CAMERA_BORDER_MARGIN_PX = 12.0
MAX_MEDIAN_COLOR_NORM = 0.80
MAX_P90_COLOR_NORM = 1.05
MIN_PREDICTED_BASE_DISPLACEMENT_M = 0.025
MAX_HAND_CENTROID_RESIDUAL_M = 0.006
MAX_HAND_SHAPE_P95_M = 0.008
MIN_STATIC_MODEL_DISADVANTAGE_M = 0.010


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    shape = ",".join(str(item) for item in array.shape)
    digest = hashlib.sha256()
    digest.update(("dtype={};shape={};".format(dtype, shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_text(value: str, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise ValueError("{} must be a lowercase SHA-256 digest".format(name))
    return text


def _rigid_transform(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("{} must be a finite 4x4 transform".format(name))
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("{} has an invalid homogeneous row".format(name))
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError("{} rotation is not orthonormal".format(name))
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7):
        raise ValueError("{} rotation determinant is not +1".format(name))
    return matrix.copy()


def _points(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (3,) or len(result) == 0:
        raise ValueError("{} must be a non-empty (N,3) array".format(name))
    if not np.all(np.isfinite(result)):
        raise ValueError("{} contains NaN or infinity".format(name))
    return result.copy()


def _colors(value: np.ndarray, count: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count, 3) or not np.all(np.isfinite(result)):
        raise ValueError("scene_colors_srgb must be a finite (N,3) array")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("scene_colors_srgb must be normalized to [0,1]")
    return result.copy()


@dataclass(frozen=True)
class AttachedCableCapture:
    """One source capture and the operator-labelled cable-return indices."""

    capture_id: str
    scene_points_base: np.ndarray
    scene_colors_srgb: np.ndarray
    component_indices: np.ndarray
    T_base_hand: np.ndarray
    T_base_camera: np.ndarray
    camera_intrinsics: np.ndarray
    captured_at_unix_s: float

    def __post_init__(self) -> None:
        capture_id = str(self.capture_id).strip()
        if not capture_id:
            raise ValueError("capture_id must be non-empty")
        points = _points(self.scene_points_base, "scene_points_base")
        colors = _colors(self.scene_colors_srgb, len(points))
        indices = np.asarray(self.component_indices, dtype=np.int64)
        if (
            indices.ndim != 1
            or len(indices) == 0
            or np.any(indices < 0)
            or np.any(indices >= len(points))
            or not np.array_equal(indices, np.unique(indices))
        ):
            raise ValueError(
                "component_indices must be sorted, unique, and index the scene"
            )
        intrinsics = np.asarray(self.camera_intrinsics, dtype=np.float64)
        if intrinsics.shape != (6,) or not np.all(np.isfinite(intrinsics)):
            raise ValueError(
                "camera_intrinsics must be [width,height,fx,fy,ppx,ppy]"
            )
        if np.any(intrinsics[:4] <= 0.0):
            raise ValueError("camera dimensions and focal lengths must be positive")
        timestamp = float(self.captured_at_unix_s)
        if not np.isfinite(timestamp) or timestamp <= 0.0:
            raise ValueError("captured_at_unix_s must be finite and positive")
        object.__setattr__(self, "capture_id", capture_id)
        object.__setattr__(self, "scene_points_base", points)
        object.__setattr__(self, "scene_colors_srgb", colors)
        object.__setattr__(self, "component_indices", indices.copy())
        object.__setattr__(
            self, "T_base_hand", _rigid_transform(self.T_base_hand, "T_base_hand")
        )
        object.__setattr__(
            self,
            "T_base_camera",
            _rigid_transform(self.T_base_camera, "T_base_camera"),
        )
        object.__setattr__(self, "camera_intrinsics", intrinsics.copy())
        object.__setattr__(self, "captured_at_unix_s", timestamp)


@dataclass(frozen=True)
class AttachedCableReturnAudit:
    exclusion_authorized: bool
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        payload = dict(self.evidence)
        json.dumps(payload, sort_keys=True, allow_nan=False)
        if bool(payload.get("attached_return_exclusion_authorized")) != bool(
            self.exclusion_authorized
        ):
            raise ValueError("audit boolean disagrees with evidence")
        if payload.get("motion_authorized") is not False:
            raise ValueError("attached-cable evidence must never authorize motion")
        object.__setattr__(self, "evidence", payload)


def _transform_points(T_target_source: np.ndarray, points_source: np.ndarray) -> np.ndarray:
    return (
        np.asarray(T_target_source[:3, :3], dtype=np.float64) @ points_source.T
    ).T + np.asarray(T_target_source[:3, 3], dtype=np.float64)


def _point_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    best = np.full(len(points), np.inf, dtype=np.float64)
    for left, right in zip(polyline[:-1], polyline[1:]):
        delta = right - left
        squared = float(delta @ delta)
        if squared <= 1e-12:
            continue
        fraction = np.clip(((points - left) @ delta) / squared, 0.0, 1.0)
        projection = left + fraction[:, None] * delta
        best = np.minimum(best, np.linalg.norm(points - projection, axis=1))
    if not np.all(np.isfinite(best)):
        raise ValueError("route_centerline_hand_m must contain a nonzero segment")
    return best


def _is_connected(points: np.ndarray) -> bool:
    if len(points) <= 1:
        return True
    # A bounded point count keeps this deterministic and avoids pulling a
    # clustering dependency into the motion-safety evidence path.
    visited = np.zeros(len(points), dtype=np.bool_)
    visited[0] = True
    frontier = [0]
    threshold2 = MAX_COMPONENT_EDGE_M * MAX_COMPONENT_EDGE_M
    while frontier:
        left = frontier.pop()
        delta = points - points[left]
        neighbors = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) <= threshold2)
        new = neighbors[~visited[neighbors]]
        if len(new):
            visited[new] = True
            frontier.extend(int(item) for item in new)
    return bool(np.all(visited))


def _nearest_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), 256):
        block = left[start : start + 256]
        squared = np.sum((block[:, None, :] - right[None, :, :]) ** 2, axis=2)
        result[start : start + len(block)] = np.sqrt(np.min(squared, axis=1))
    return result


def _shape_p95(left: np.ndarray, right: np.ndarray) -> float:
    distances = np.concatenate(
        (_nearest_distances(left, right), _nearest_distances(right, left))
    )
    return float(np.percentile(distances, 95.0))


def _project_border_margin_px(
    points_base: np.ndarray,
    T_base_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> Tuple[float, bool]:
    camera = _transform_points(np.linalg.inv(T_base_camera), points_base)
    positive_depth = bool(np.all(camera[:, 2] > 0.0))
    if not positive_depth:
        return -math.inf, False
    width, height, fx, fy, ppx, ppy = intrinsics
    u = fx * camera[:, 0] / camera[:, 2] + ppx
    v = fy * camera[:, 1] / camera[:, 2] + ppy
    margins = np.concatenate((u, width - 1.0 - u, v, height - 1.0 - v))
    return float(np.min(margins)), True


def _project_route_tube_border_margin_px(
    route_points_base: np.ndarray,
    T_base_camera: np.ndarray,
    intrinsics: np.ndarray,
    tube_radius_m: float,
) -> Tuple[float, bool]:
    """Conservatively shrink centerline pixel margin by the tube projection."""

    centerline_margin, positive = _project_border_margin_px(
        route_points_base, T_base_camera, intrinsics
    )
    camera = _transform_points(np.linalg.inv(T_base_camera), route_points_base)
    minimum_depth = float(np.min(camera[:, 2]))
    radius = float(tube_radius_m)
    if not positive or minimum_depth <= radius:
        return -math.inf, False
    fx = float(intrinsics[2])
    fy = float(intrinsics[3])
    # A sphere of ``radius`` around each centerline point projects no farther
    # than this pinhole bound.  Subtracting it covers the full swept tube, not
    # merely the sampled centerline.
    pixel_radius_bound = max(fx, fy) * radius / (minimum_depth - radius)
    return centerline_margin - pixel_radius_bound, True


def audit_attached_cable_returns(
    captures: Sequence[AttachedCableCapture],
    *,
    route_centerline_hand_m: np.ndarray,
    route_outer_radius_m: float,
    identity_evidence_sha256: str,
    official_parent_link: str,
    official_mesh_relation_evidence_sha256: str,
    operator_identity_confirmation: str,
    operator_rigid_route_confirmation: str,
) -> AttachedCableReturnAudit:
    """Audit exact labelled indices without granting robot-motion authority."""

    values = tuple(captures)
    if len(values) < 2:
        raise ValueError("at least two captures are required")
    if not all(isinstance(item, AttachedCableCapture) for item in values):
        raise TypeError("captures must contain AttachedCableCapture values")
    ids = tuple(item.capture_id for item in values)
    if len(set(ids)) != len(ids):
        raise ValueError("capture_id values must be unique")
    route = _points(route_centerline_hand_m, "route_centerline_hand_m")
    if len(route) < 2:
        raise ValueError("route_centerline_hand_m must contain at least two points")
    radius = float(route_outer_radius_m)
    if not np.isfinite(radius) or not 0.001 <= radius <= MAX_ROUTE_RADIUS_M:
        raise ValueError(
            "route_outer_radius_m must be in [0.001,{:.3f}]".format(
                MAX_ROUTE_RADIUS_M
            )
        )
    identity_sha = _sha256_text(
        identity_evidence_sha256, "identity_evidence_sha256"
    )
    parent_link = str(official_parent_link).strip()
    relation_sha = _sha256_text(
        official_mesh_relation_evidence_sha256,
        "official_mesh_relation_evidence_sha256",
    )

    failures = []
    if parent_link != "Link111":
        failures.append("official parent mesh relation must identify RH56 Link111")
    if operator_identity_confirmation != IDENTITY_CONFIRMATION_TOKEN:
        failures.append("operator cable identity confirmation is missing or incorrect")
    if operator_rigid_route_confirmation != RIGID_ROUTE_CONFIRMATION_TOKEN:
        failures.append("operator rigid-route confirmation is missing or incorrect")

    capture_evidence = []
    components_hand = []
    components_base = []
    for capture in values:
        indices = capture.component_indices
        component_base = capture.scene_points_base[indices]
        component_colors = capture.scene_colors_srgb[indices]
        component_hand = _transform_points(
            np.linalg.inv(capture.T_base_hand), component_base
        )
        components_base.append(component_base)
        components_hand.append(component_hand)
        count_ok = MIN_COMPONENT_POINTS <= len(indices) <= MAX_COMPONENT_POINTS
        connected = count_ok and _is_connected(component_hand)
        span = float(np.max(np.ptp(component_hand, axis=0)))
        route_residuals = _point_polyline_distances(component_hand, route)
        route_max = float(np.max(route_residuals))
        color_norms = np.linalg.norm(component_colors, axis=1)
        median_color_norm = float(np.median(color_norms))
        p90_color_norm = float(np.percentile(color_norms, 90.0))
        component_border, component_positive_depth = _project_border_margin_px(
            component_base, capture.T_base_camera, capture.camera_intrinsics
        )
        route_base = _transform_points(capture.T_base_hand, route)
        route_border, route_positive_depth = _project_route_tube_border_margin_px(
            route_base,
            capture.T_base_camera,
            capture.camera_intrinsics,
            radius + DEPTH_RETURN_ROUTE_GUARD_M,
        )
        complete_in_frame = bool(
            component_positive_depth
            and route_positive_depth
            and component_border >= CAMERA_BORDER_MARGIN_PX
            and route_border >= CAMERA_BORDER_MARGIN_PX
        )
        local_failures = []
        if not count_ok:
            local_failures.append("component point count is outside the fixed bounds")
        if not connected:
            local_failures.append("component is not 3-D connected")
        if span < MIN_COMPONENT_SPAN_M:
            local_failures.append("component span is too small to identify a cable segment")
        if route_max > radius + DEPTH_RETURN_ROUTE_GUARD_M:
            local_failures.append("component leaves the measured hand-frame cable route")
        if (
            median_color_norm > MAX_MEDIAN_COLOR_NORM
            or p90_color_norm > MAX_P90_COLOR_NORM
        ):
            local_failures.append("component is not consistently dark")
        if not complete_in_frame:
            local_failures.append("component or measured route is clipped by camera border")
        failures.extend("{}: {}".format(capture.capture_id, item) for item in local_failures)
        capture_evidence.append(
            {
                "capture_id": capture.capture_id,
                "captured_at_unix_s": capture.captured_at_unix_s,
                "scene_point_count": int(len(capture.scene_points_base)),
                "scene_points_sha256": _array_sha256(
                    capture.scene_points_base, "<f8"
                ),
                "scene_colors_sha256": _array_sha256(
                    capture.scene_colors_srgb, "<f8"
                ),
                "component_indices_sha256": _array_sha256(indices, "<i8"),
                "component_points_sha256": _array_sha256(component_base, "<f8"),
                "component_point_count": int(len(indices)),
                "component_span_m": span,
                "connected_with_max_edge_m": bool(connected),
                "maximum_component_edge_m": MAX_COMPONENT_EDGE_M,
                "route_residual_maximum_m": route_max,
                "median_normalized_srgb_norm": median_color_norm,
                "p90_normalized_srgb_norm": p90_color_norm,
                "component_camera_border_margin_px": component_border,
                "route_camera_border_margin_px": route_border,
                "complete_component_and_route_in_frame": complete_in_frame,
                "T_base_hand_sha256": _array_sha256(
                    capture.T_base_hand, "<f8"
                ),
                "T_base_camera_sha256": _array_sha256(
                    capture.T_base_camera, "<f8"
                ),
                "camera_intrinsics_sha256": _array_sha256(
                    capture.camera_intrinsics, "<f8"
                ),
                "failures": local_failures,
            }
        )

    pair_evidence = []
    excited_pair_count = 0
    for left_index in range(len(values)):
        for right_index in range(left_index + 1, len(values)):
            left = values[left_index]
            right = values[right_index]
            left_hand = components_hand[left_index]
            right_hand = components_hand[right_index]
            left_base = components_base[left_index]
            right_base = components_base[right_index]
            left_centroid_hand = np.mean(left_hand, axis=0)
            right_centroid_hand = np.mean(right_hand, axis=0)
            left_centroid_base = np.mean(left_base, axis=0)
            right_centroid_base = np.mean(right_base, axis=0)
            predicted_right_base = _transform_points(
                right.T_base_hand, left_centroid_hand[None, :]
            )[0]
            predicted_displacement = float(
                np.linalg.norm(predicted_right_base - left_centroid_base)
            )
            actual_displacement = float(
                np.linalg.norm(right_centroid_base - left_centroid_base)
            )
            hand_centroid_residual = float(
                np.linalg.norm(right_centroid_hand - left_centroid_hand)
            )
            comotion_residual = float(
                np.linalg.norm(right_centroid_base - predicted_right_base)
            )
            shape_p95 = _shape_p95(left_hand, right_hand)
            excited = predicted_displacement >= MIN_PREDICTED_BASE_DISPLACEMENT_M
            if excited:
                excited_pair_count += 1
                if hand_centroid_residual > MAX_HAND_CENTROID_RESIDUAL_M:
                    failures.append(
                        "{} / {}: hand-frame centroid is not repeatable".format(
                            left.capture_id, right.capture_id
                        )
                    )
                if shape_p95 > MAX_HAND_SHAPE_P95_M:
                    failures.append(
                        "{} / {}: hand-frame component shape is not repeatable".format(
                            left.capture_id, right.capture_id
                        )
                    )
                if actual_displacement - comotion_residual < MIN_STATIC_MODEL_DISADVANTAGE_M:
                    failures.append(
                        "{} / {}: hand-attached model does not beat static-obstacle model".format(
                            left.capture_id, right.capture_id
                        )
                    )
            pair_evidence.append(
                {
                    "left_capture_id": left.capture_id,
                    "right_capture_id": right.capture_id,
                    "predicted_attached_base_displacement_m": predicted_displacement,
                    "actual_base_displacement_m": actual_displacement,
                    "hand_frame_centroid_residual_m": hand_centroid_residual,
                    "attached_model_base_residual_m": comotion_residual,
                    "static_model_disadvantage_m": actual_displacement
                    - comotion_residual,
                    "bidirectional_hand_shape_p95_m": shape_p95,
                    "pose_excitation_sufficient": bool(excited),
                }
            )
    if excited_pair_count == 0:
        failures.append(
            "no capture pair provides the required attached-component displacement"
        )

    failures = list(dict.fromkeys(failures))
    passed = not failures
    evidence: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "algorithm": ALGORITHM,
        "identity_evidence_sha256": identity_sha,
        "official_parent_link": parent_link,
        "official_mesh_relation_evidence_sha256": relation_sha,
        "operator_identity_confirmation_exact": (
            operator_identity_confirmation == IDENTITY_CONFIRMATION_TOKEN
        ),
        "operator_rigid_route_confirmation_exact": (
            operator_rigid_route_confirmation == RIGID_ROUTE_CONFIRMATION_TOKEN
        ),
        "route_centerline_hand_m": route.tolist(),
        "route_centerline_sha256": _array_sha256(route, "<f8"),
        "route_outer_radius_m": radius,
        "fixed_policy": {
            "minimum_component_points": MIN_COMPONENT_POINTS,
            "maximum_component_points": MAX_COMPONENT_POINTS,
            "maximum_component_edge_m": MAX_COMPONENT_EDGE_M,
            "minimum_component_span_m": MIN_COMPONENT_SPAN_M,
            "maximum_route_radius_m": MAX_ROUTE_RADIUS_M,
            "depth_return_route_guard_m": DEPTH_RETURN_ROUTE_GUARD_M,
            "camera_border_margin_px": CAMERA_BORDER_MARGIN_PX,
            "maximum_median_color_norm": MAX_MEDIAN_COLOR_NORM,
            "maximum_p90_color_norm": MAX_P90_COLOR_NORM,
            "minimum_predicted_base_displacement_m": (
                MIN_PREDICTED_BASE_DISPLACEMENT_M
            ),
            "maximum_hand_centroid_residual_m": MAX_HAND_CENTROID_RESIDUAL_M,
            "maximum_hand_shape_p95_m": MAX_HAND_SHAPE_P95_M,
            "minimum_static_model_disadvantage_m": MIN_STATIC_MODEL_DISADVANTAGE_M,
        },
        "captures": capture_evidence,
        "capture_pairs": pair_evidence,
        "excited_pair_count": excited_pair_count,
        "failures": failures,
        "attached_return_exclusion_authorized": passed,
        "exclusion_scope": (
            "only the exact component_indices bound for each listed capture"
        ),
        "moving_cable_collision_envelope_required": True,
        "motion_authorized": False,
        "meaning": (
            "return-identity evidence only; route tube must be collision-audited "
            "as moving installed geometry before any motion"
        ),
    }
    return AttachedCableReturnAudit(
        exclusion_authorized=passed,
        evidence=evidence,
    )


__all__ = [
    "ALGORITHM",
    "ARTIFACT_TYPE",
    "AttachedCableCapture",
    "AttachedCableReturnAudit",
    "CAMERA_BORDER_MARGIN_PX",
    "IDENTITY_CONFIRMATION_TOKEN",
    "RIGID_ROUTE_CONFIRMATION_TOKEN",
    "audit_attached_cable_returns",
]
