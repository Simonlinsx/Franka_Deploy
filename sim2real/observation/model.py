"""V94 native-palm point-cloud and 67-D proprioception construction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from typing import Any, Deque, Mapping, Optional, Tuple

import numpy as np

from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13, Q_HAND_SEMANTIC_CLOSE_RAD
from .pointcloud_filters import fit_fixed_radius_sphere


POLICY_RGBD_RESOLUTION_SIZES = {
    "848x480": (848, 480),
    "424x240": (424, 240),
}


def resolve_policy_rgbd_resolution(value: object) -> Tuple[int, int]:
    """Resolve the two explicitly supported policy-side RGB-D resolutions."""

    name = str(value).strip().lower()
    try:
        return POLICY_RGBD_RESOLUTION_SIZES[name]
    except KeyError as exc:
        supported = ", ".join(POLICY_RGBD_RESOLUTION_SIZES)
        raise ValueError(
            f"policy RGB-D resolution must be one of: {supported}"
        ) from exc


@dataclass(frozen=True)
class PolicyRGBDInputs:
    """One aligned RGB-D/mask frame after policy-resolution adaptation."""

    color_bgr: np.ndarray
    object_mask: np.ndarray
    depth_m: Optional[np.ndarray]
    depth_raw: Optional[np.ndarray]


class PolicyRGBDResolutionAdapter:
    """Deterministically decimate aligned native RGB-D for policy projection.

    The D435 and semantic tracker remain at native resolution.  For the
    424x240 A/B path, all three aligned arrays select the same top-left sample
    from each native 2x2 cell.  Scaling ``fx/fy/cx/cy`` by the same factor then
    makes deprojection of every retained pixel geometrically identical to its
    native-resolution counterpart; no RGB/depth interpolation or mixed-edge
    averaging is introduced.
    """

    def __init__(
        self,
        *,
        camera_K: np.ndarray,
        source_image_size: Tuple[int, int],
        target_image_size: Tuple[int, int],
    ) -> None:
        self.source_width, self.source_height = (
            int(source_image_size[0]),
            int(source_image_size[1]),
        )
        self.target_width, self.target_height = (
            int(target_image_size[0]),
            int(target_image_size[1]),
        )
        if min(
            self.source_width,
            self.source_height,
            self.target_width,
            self.target_height,
        ) <= 0:
            raise ValueError("policy RGB-D image dimensions must be positive")
        if (
            self.source_width % self.target_width != 0
            or self.source_height % self.target_height != 0
        ):
            raise ValueError(
                "policy RGB-D target must be an integer decimation of the source"
            )
        stride_x = self.source_width // self.target_width
        stride_y = self.source_height // self.target_height
        if stride_x != stride_y or stride_x not in (1, 2):
            raise ValueError(
                "policy RGB-D adaptation supports only native or exact 2x decimation"
            )
        self.stride = int(stride_x)
        source_K = _finite(camera_K, (3, 3), "camera_K")
        self.camera_K = source_K.copy()
        self.camera_K[0, 0] /= self.stride
        self.camera_K[0, 2] /= self.stride
        self.camera_K[1, 1] /= self.stride
        self.camera_K[1, 2] /= self.stride

    @property
    def source_image_size(self) -> Tuple[int, int]:
        return self.source_width, self.source_height

    @property
    def target_image_size(self) -> Tuple[int, int]:
        return self.target_width, self.target_height

    def adapt(
        self,
        *,
        color_bgr: np.ndarray,
        object_mask: np.ndarray,
        depth_m: Optional[np.ndarray] = None,
        depth_raw: Optional[np.ndarray] = None,
    ) -> PolicyRGBDInputs:
        color = np.asarray(color_bgr)
        mask = np.asarray(object_mask, dtype=bool)
        expected_color_shape = (self.source_height, self.source_width, 3)
        expected_depth_shape = (self.source_height, self.source_width)
        if color.shape != expected_color_shape:
            raise ValueError(
                f"native color shape must be {expected_color_shape}, got {color.shape}"
            )
        if mask.shape != expected_depth_shape:
            raise ValueError(
                f"native object mask shape must be {expected_depth_shape}, got {mask.shape}"
            )
        if (depth_m is None) == (depth_raw is None):
            raise ValueError("provide exactly one of depth_m or depth_raw")
        metric = None if depth_m is None else np.asarray(depth_m)
        raw = None if depth_raw is None else np.asarray(depth_raw)
        depth = metric if metric is not None else raw
        assert depth is not None
        if depth.shape != expected_depth_shape:
            raise ValueError(
                f"native depth shape must be {expected_depth_shape}, got {depth.shape}"
            )
        stride = self.stride
        rows = slice(None, None, stride)
        columns = slice(None, None, stride)
        return PolicyRGBDInputs(
            color_bgr=np.ascontiguousarray(color[rows, columns]),
            object_mask=np.ascontiguousarray(mask[rows, columns]),
            depth_m=(
                None
                if metric is None
                else np.ascontiguousarray(metric[rows, columns])
            ),
            depth_raw=(
                None if raw is None else np.ascontiguousarray(raw[rows, columns])
            ),
        )

    def mask_to_source_resolution(self, object_mask: np.ndarray) -> np.ndarray:
        """Expand the exact policy mask for native-resolution visualization."""

        mask = np.asarray(object_mask, dtype=bool)
        expected = (self.target_height, self.target_width)
        if mask.shape != expected:
            raise ValueError(
                f"policy object mask shape must be {expected}, got {mask.shape}"
            )
        if self.stride == 1:
            return mask.copy()
        expanded = np.repeat(
            np.repeat(mask, self.stride, axis=0), self.stride, axis=1
        )
        return np.ascontiguousarray(
            expanded[: self.source_height, : self.source_width]
        )


def infer_fixed_sphere_radius_m(metadata: Mapping[str, Any]) -> Optional[float]:
    """Infer a single sphere radius from serialized checkpoint provenance.

    This intentionally returns ``None`` unless the checkpoint says that only
    one asset ID is active and the dataset provenance names that asset as a
    sphere with a millimetre diameter (for example ``sphere60``).  A generic
    or mixed-asset policy must not silently receive sphere completion.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    weights = None
    for container_name in (
        "ppo_tabletop_domain_randomization",
        "tabletop_domain_randomization",
    ):
        container = metadata.get(container_name)
        if isinstance(container, Mapping):
            candidate = container.get("asset_sampling_weights")
            if candidate is not None:
                weights = candidate
                break
    if weights is None:
        weights = metadata.get("tabletop_asset_sampling_weights")
    if weights is None:
        return None
    numeric = np.asarray(weights, dtype=np.float64)
    if (
        numeric.ndim != 1
        or numeric.size < 1
        or not np.all(np.isfinite(numeric))
        or np.any(numeric < 0.0)
    ):
        return None
    active = np.flatnonzero(numeric > 1.0e-8)
    if active.size != 1:
        return None
    active_id = int(active[0])
    datasets = metadata.get("merged_datasets")
    if not isinstance(datasets, (list, tuple)):
        return None
    matching_names = []
    for entry in datasets:
        if not isinstance(entry, Mapping):
            continue
        try:
            first = int(entry.get("source_id_start"))
            count = int(entry.get("source_id_count"))
        except (TypeError, ValueError):
            continue
        if count > 0 and first <= active_id < first + count:
            matching_names.append(str(entry.get("name", "")))
    if len(matching_names) != 1:
        return None
    match = re.fullmatch(
        r"sphere[_-]?(\d+(?:\.\d+)?)(?:mm)?",
        matching_names[0].strip().lower(),
    )
    if match is None:
        return None
    diameter_mm = float(match.group(1))
    if not np.isfinite(diameter_mm) or not 20.0 <= diameter_mm <= 200.0:
        return None
    return diameter_mm * 0.0005


def _finite(value: np.ndarray, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape {shape}")
    return result


def _rigid(value: np.ndarray, name: str) -> np.ndarray:
    result = _finite(value, (4, 4), name)
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1.0e-7, rtol=0.0):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5, rtol=0.0):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5, rtol=0.0):
        raise ValueError(f"{name} rotation determinant must be +1")
    return result


def rotation_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized, stable-sign quaternion for a rotation matrix."""

    matrix = _rigid(
        np.block(
            [
                [np.asarray(rotation, dtype=np.float64), np.zeros((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        ),
        "rotation",
    )[:3, :3]
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def quaternion_wxyz_to_rotation(value: np.ndarray) -> np.ndarray:
    quaternion = _finite(value, (4,), "quaternion_wxyz").copy()
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("quaternion_wxyz must be non-zero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_from_position_quaternion_wxyz(
    position: np.ndarray, quaternion_wxyz: np.ndarray
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_wxyz_to_rotation(quaternion_wxyz)
    result[:3, 3] = _finite(position, (3,), "position")
    return result


@dataclass(frozen=True)
class PolicyPointFrame:
    # Historical field name retained for bundle/audit compatibility.  Its
    # final dimension is 3 in XYZ mode and 6 in XYZRGB mode.
    xyzrgb_palm: np.ndarray
    valid: np.ndarray
    captured_at_s: float
    frame_id: int
    source_valid_points: int
    status: str
    # The current frame/time identify the policy estimate.  During bounded
    # motion compensation these fields retain the last frame that contributed
    # measured object depth, so the estimate is never mislabeled as fresh.
    measured_at_s: Optional[float] = None
    measured_frame_id: Optional[int] = None
    fallback_age_s: float = 0.0
    fallback_steps: int = 0

    @property
    def point_features_palm(self) -> np.ndarray:
        """Neutral alias for the legacy ``xyzrgb_palm`` field name."""

        return self.xyzrgb_palm

    @property
    def point_feature_dim(self) -> int:
        return int(np.asarray(self.xyzrgb_palm).shape[-1])

    @property
    def point_feature_mode(self) -> str:
        return "xyz" if self.point_feature_dim == 3 else "xyzrgb"


@dataclass(frozen=True)
class ProjectorEffectiveMaskProvenance:
    """Identity and geometry of the mask that owns the returned policy cloud.

    Provider publication provenance and projector provenance are deliberately
    different concepts.  The provider describes the current RGB-D frame it
    published; this record describes the mask whose pixels actually own the
    returned point cloud.  They differ during ``stale_palm`` fallback.
    """

    kind: str
    source_frame_id: Optional[int]
    source_captured_at_s: Optional[float]
    area_px: int
    bbox_xyxy: Optional[Tuple[int, int, int, int]]


def _projector_mask_provenance(
    mask: Optional[np.ndarray],
    *,
    kind: str,
    source_frame_id: Optional[int],
    source_captured_at_s: Optional[float],
) -> ProjectorEffectiveMaskProvenance:
    if mask is None:
        return ProjectorEffectiveMaskProvenance(
            kind=str(kind),
            source_frame_id=None,
            source_captured_at_s=None,
            area_px=0,
            bbox_xyxy=None,
        )
    binary = np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(binary)
    bbox = (
        None
        if columns.size == 0
        else (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
        )
    )
    return ProjectorEffectiveMaskProvenance(
        kind=str(kind),
        source_frame_id=(
            None if source_frame_id is None else int(source_frame_id)
        ),
        source_captured_at_s=(
            None
            if source_captured_at_s is None
            else float(source_captured_at_s)
        ),
        area_px=int(columns.size),
        bbox_xyxy=bbox,
    )


class MaskedRGBDProjector:
    """Deterministic row-major/evenly-spaced RGB-D policy projector.

    ``legacy_stale_palm`` retains the last valid cloud in its capture-time palm
    frame.  ``motion_compensated`` instead translates only the latest measured
    cloud to the exact current RGB-mask centroid and re-expresses it in the
    current palm frame.  The latter is bounded by age, tick count and image
    speed and never recursively compounds a predicted cloud.
    """

    def __init__(
        self,
        *,
        camera_K: np.ndarray,
        T_base_camera_optical: np.ndarray,
        image_size: Tuple[int, int],
        depth_range_m: Tuple[float, float],
        num_points: int = 128,
        minimum_valid_points: int = 16,
        point_feature_dim: int = 6,
        maximum_mask_depth_deviation_m: Optional[float] = None,
        fixed_sphere_completion_radius_m: Optional[float] = None,
        support_plane_abcd: Optional[np.ndarray] = None,
        support_plane_min_clearance_m: float = 0.0,
        temporal_fallback: str = "legacy_stale_palm",
        temporal_fallback_max_stale_s: float = 0.25,
        temporal_fallback_max_stale_steps: int = 5,
        temporal_fallback_max_image_speed_px_s: float = 2400.0,
    ) -> None:
        self.K = _finite(camera_K, (3, 3), "camera_K")
        self.T_base_camera = _rigid(T_base_camera_optical, "T_base_camera_optical")
        self.width, self.height = (int(image_size[0]), int(image_size[1]))
        self.z_min, self.z_max = (float(depth_range_m[0]), float(depth_range_m[1]))
        self.num_points = int(num_points)
        self.minimum_valid_points = int(minimum_valid_points)
        fallback_mode = str(temporal_fallback).strip().lower()
        if fallback_mode not in ("legacy_stale_palm", "motion_compensated"):
            raise ValueError(
                "temporal_fallback must be legacy_stale_palm or motion_compensated"
            )
        self.temporal_fallback = fallback_mode
        self.temporal_fallback_max_stale_s = float(
            temporal_fallback_max_stale_s
        )
        if (
            not np.isfinite(self.temporal_fallback_max_stale_s)
            or not 0.05 <= self.temporal_fallback_max_stale_s <= 1.0
        ):
            raise ValueError("temporal_fallback_max_stale_s must be in 0.05..1.0")
        if (
            isinstance(temporal_fallback_max_stale_steps, bool)
            or not 1 <= int(temporal_fallback_max_stale_steps) <= 20
        ):
            raise ValueError("temporal_fallback_max_stale_steps must be in 1..20")
        self.temporal_fallback_max_stale_steps = int(
            temporal_fallback_max_stale_steps
        )
        self.temporal_fallback_max_image_speed_px_s = float(
            temporal_fallback_max_image_speed_px_s
        )
        if (
            not np.isfinite(self.temporal_fallback_max_image_speed_px_s)
            or self.temporal_fallback_max_image_speed_px_s <= 0.0
        ):
            raise ValueError(
                "temporal_fallback_max_image_speed_px_s must be positive"
            )
        if isinstance(point_feature_dim, bool) or int(point_feature_dim) not in (3, 6):
            raise ValueError("point_feature_dim must be 3 (XYZ) or 6 (XYZRGB)")
        self.point_feature_dim = int(point_feature_dim)
        self.point_feature_mode = "xyz" if self.point_feature_dim == 3 else "xyzrgb"
        if maximum_mask_depth_deviation_m is None:
            self.maximum_mask_depth_deviation_m = None
        else:
            maximum_depth_deviation = float(maximum_mask_depth_deviation_m)
            if (
                not np.isfinite(maximum_depth_deviation)
                or not 0.005 <= maximum_depth_deviation <= 0.20
            ):
                raise ValueError(
                    "maximum_mask_depth_deviation_m must be None or 0.005..0.20"
                )
            self.maximum_mask_depth_deviation_m = maximum_depth_deviation
        if fixed_sphere_completion_radius_m is None:
            self.fixed_sphere_completion_radius_m = None
        else:
            sphere_radius = float(fixed_sphere_completion_radius_m)
            if not np.isfinite(sphere_radius) or not 0.010 <= sphere_radius <= 0.100:
                raise ValueError(
                    "fixed_sphere_completion_radius_m must be None or 0.010..0.100"
                )
            self.fixed_sphere_completion_radius_m = sphere_radius
        self._support_plane_normal_base: Optional[np.ndarray] = None
        self._support_plane_offset_base = 0.0
        clearance = float(support_plane_min_clearance_m)
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                "support_plane_min_clearance_m must be finite and non-negative"
            )
        self.support_plane_min_clearance_m = clearance
        if support_plane_abcd is not None:
            plane = np.asarray(support_plane_abcd, dtype=np.float64)
            if plane.shape != (4,) or not np.all(np.isfinite(plane)):
                raise ValueError("support_plane_abcd must have finite shape (4,)")
            magnitude = float(np.linalg.norm(plane[:3]))
            if magnitude <= 1.0e-9:
                raise ValueError("support_plane_abcd normal must be non-zero")
            normal = plane[:3] / magnitude
            offset = float(plane[3] / magnitude)
            if normal[2] < 0.0:
                normal *= -1.0
                offset *= -1.0
            if normal[2] < 0.9:
                raise ValueError(
                    "support_plane_abcd must describe an upward tabletop"
                )
            self._support_plane_normal_base = normal
            self._support_plane_offset_base = offset
        self.last_sphere_completion: Optional[Mapping[str, object]] = None
        self.last_effective_object_mask: Optional[np.ndarray] = None
        self.last_effective_object_mask_provenance = _projector_mask_provenance(
            None,
            kind="unavailable_before_first_projection",
            source_frame_id=None,
            source_captured_at_s=None,
        )
        self._sphere_center_camera_offset: Optional[np.ndarray] = None
        self._last_completed_sphere_center_camera: Optional[np.ndarray] = None
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image_size must be positive")
        if not (0.0 < self.z_min < self.z_max):
            raise ValueError("depth_range_m must be positive and ordered")
        if self.num_points <= 0 or not (
            1 <= self.minimum_valid_points <= self.num_points
        ):
            raise ValueError("invalid policy point-count thresholds")
        if not np.allclose(
            self.K,
            [
                [self.K[0, 0], 0, self.K[0, 2]],
                [0, self.K[1, 1], self.K[1, 2]],
                [0, 0, 1],
            ],
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise ValueError("camera_K must be a zero-skew pinhole matrix")
        if self.K[0, 0] <= 0.0 or self.K[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        self._latest_good: Optional[PolicyPointFrame] = None
        self._latest_good_effective_object_mask: Optional[np.ndarray] = None
        self._latest_good_input_object_mask: Optional[np.ndarray] = None
        self._latest_good_T_base_palm: Optional[np.ndarray] = None
        self._latest_good_depth_median_m: Optional[float] = None
        self._motion_compensated_steps = 0

    @staticmethod
    def _mask_centroid_uv(mask: np.ndarray) -> Optional[np.ndarray]:
        rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
        if rows.size == 0:
            return None
        return np.asarray(
            [float(np.mean(columns)), float(np.mean(rows))], dtype=np.float64
        )

    def _motion_compensated_point_frame(
        self,
        *,
        current_mask: np.ndarray,
        current_palm: np.ndarray,
        timestamp: float,
        frame_id: int,
        source_count: int,
    ) -> Optional[PolicyPointFrame]:
        """Translate the latest measured cloud with the current RGB centroid.

        The D435 often loses depth on a fast small object while RGB/SAM2 stays
        current.  The checkpoint was trained with at most five steps/0.25 s of
        motion-compensated fallback.  This implementation keeps the last
        measured depth, moves the cloud by the exact current image centroid,
        and re-expresses it in the current palm frame.  It never compounds one
        predicted cloud into the next prediction.
        """

        previous = self._latest_good
        previous_mask = self._latest_good_input_object_mask
        previous_palm = self._latest_good_T_base_palm
        previous_depth = self._latest_good_depth_median_m
        if (
            previous is None
            or previous_mask is None
            or previous_palm is None
            or previous_depth is None
        ):
            return None
        current_centroid = self._mask_centroid_uv(current_mask)
        previous_centroid = self._mask_centroid_uv(previous_mask)
        if current_centroid is None or previous_centroid is None:
            return None
        age_s = float(timestamp) - float(previous.captured_at_s)
        next_step = int(self._motion_compensated_steps) + 1
        if (
            not np.isfinite(age_s)
            or age_s <= 0.0
            or age_s > self.temporal_fallback_max_stale_s
            or next_step > self.temporal_fallback_max_stale_steps
        ):
            return None
        displacement_px = current_centroid - previous_centroid
        image_speed_px_s = float(np.linalg.norm(displacement_px) / age_s)
        if image_speed_px_s > self.temporal_fallback_max_image_speed_px_s:
            return None
        depth_value = float(previous_depth)
        if not np.isfinite(depth_value) or not self.z_min < depth_value < self.z_max:
            return None

        delta_camera = np.asarray(
            [
                displacement_px[0] * depth_value / self.K[0, 0],
                displacement_px[1] * depth_value / self.K[1, 1],
                0.0,
            ],
            dtype=np.float64,
        )
        delta_base = delta_camera @ self.T_base_camera[:3, :3].T
        output = previous.xyzrgb_palm.copy()
        valid = previous.valid.copy()
        selected = valid > 0.5
        if int(np.count_nonzero(selected)) < 1:
            return None
        previous_xyz_palm = output[selected, :3].astype(np.float64)
        previous_xyz_base = (
            previous_xyz_palm @ previous_palm[:3, :3].T
            + previous_palm[:3, 3]
        )
        current_xyz_base = previous_xyz_base + delta_base
        current_xyz_palm = (
            current_xyz_base - current_palm[:3, 3]
        ) @ current_palm[:3, :3]
        if not np.all(np.isfinite(current_xyz_palm)):
            return None
        output[selected, :3] = current_xyz_palm.astype(np.float32)
        self._motion_compensated_steps = next_step
        self.last_effective_object_mask = np.asarray(
            current_mask, dtype=bool
        ).copy()
        self.last_effective_object_mask_provenance = _projector_mask_provenance(
            self.last_effective_object_mask,
            kind="current_rgb_mask_motion_compensated_from_previous_depth",
            source_frame_id=int(frame_id),
            source_captured_at_s=float(timestamp),
        )
        return PolicyPointFrame(
            xyzrgb_palm=output,
            valid=valid,
            captured_at_s=float(timestamp),
            frame_id=int(frame_id),
            source_valid_points=int(source_count),
            status="motion_compensated",
            measured_at_s=float(previous.captured_at_s),
            measured_frame_id=int(previous.frame_id),
            fallback_age_s=float(age_s),
            fallback_steps=int(next_step),
        )

    def _filter_pixels_above_support_plane(
        self,
        rows: np.ndarray,
        columns: np.ndarray,
        depth_m: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove tabletop depth without changing the semantic object mask.

        SAM2 owns the two-dimensional silhouette.  The calibrated support
        plane is a three-dimensional point-cloud constraint, so applying it
        here avoids fragmenting and then eroding a correct cube/object mask.
        """

        normal = self._support_plane_normal_base
        if normal is None or depth_m.size == 0:
            return rows, columns, depth_m
        z = np.asarray(depth_m, dtype=np.float64)
        x = (columns.astype(np.float64) - self.K[0, 2]) / self.K[0, 0] * z
        y = (rows.astype(np.float64) - self.K[1, 2]) / self.K[1, 1] * z
        camera_points = np.stack((x, y, z), axis=1)
        base_points = (
            camera_points @ self.T_base_camera[:3, :3].T
            + self.T_base_camera[:3, 3]
        )
        signed_height = (
            base_points @ normal + self._support_plane_offset_base
        )
        keep = (
            np.all(np.isfinite(base_points), axis=1)
            & np.isfinite(signed_height)
            & (signed_height > self.support_plane_min_clearance_m)
        )
        return rows[keep], columns[keep], depth_m[keep]

    @staticmethod
    def _dilate_binary_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
        radius = int(radius_px)
        source = np.asarray(mask, dtype=bool)
        if radius <= 0:
            return source.copy()
        kernel = 2 * radius + 1
        padded = np.pad(source.astype(np.int32), radius, mode="constant")
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
        integral = integral.cumsum(axis=0).cumsum(axis=1)
        window_sum = (
            integral[kernel:, kernel:]
            - integral[:-kernel, kernel:]
            - integral[kernel:, :-kernel]
            + integral[:-kernel, :-kernel]
        )
        return window_sum > 0

    def _complete_fixed_sphere_depth(
        self,
        *,
        object_mask: np.ndarray,
        reliable_rows: np.ndarray,
        reliable_columns: np.ndarray,
        reliable_depth_m: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Complete missing/mixed D435 pixels on one known-radius sphere.

        The reliable partial depth cap estimates fresh sphere motion.  The
        final depths are near camera-ray/sphere intersections inside a small
        dilation of the semantic mask, matching the simulator's strict
        primitive-surface RGB-D observation without hallucinating through a
        large occlusion.  If the robust fit is not strong enough, the measured
        reliable input is returned unchanged.
        """

        radius = self.fixed_sphere_completion_radius_m
        self.last_sphere_completion = None
        if radius is None or reliable_depth_m.size < 16:
            return reliable_rows, reliable_columns, reliable_depth_m

        reliable_z = reliable_depth_m.astype(np.float64)
        reliable_x = (
            (reliable_columns.astype(np.float64) - self.K[0, 2])
            / self.K[0, 0]
            * reliable_z
        )
        reliable_y = (
            (reliable_rows.astype(np.float64) - self.K[1, 2])
            / self.K[1, 1]
            * reliable_z
        )
        reliable_camera = np.stack(
            (reliable_x, reliable_y, reliable_z), axis=1
        )
        fit = fit_fixed_radius_sphere(
            reliable_camera,
            camera_origin_base_m=np.zeros(3, dtype=np.float64),
            radius_m=float(radius),
            shell_tolerance_m=0.006,
            huber_width_m=0.004,
        )
        if (
            not fit.valid
            or not np.isfinite(fit.residual_p90_m)
            or fit.residual_p90_m > 0.020
            or fit.shell_inlier_fraction < 0.35
        ):
            self.last_sphere_completion = {
                "applied": False,
                "radius_m": float(radius),
                "fit_residual_p90_m": float(fit.residual_p90_m),
                "fit_shell_inlier_fraction": float(fit.shell_inlier_fraction),
            }
            return reliable_rows, reliable_columns, reliable_depth_m

        semantic_mask = np.asarray(object_mask, dtype=bool)
        mask_rows, mask_columns = np.nonzero(semantic_mask)
        if mask_rows.size < self.minimum_valid_points:
            return reliable_rows, reliable_columns, reliable_depth_m
        center = np.asarray(fit.center_base_m, dtype=np.float64).copy()
        if self._sphere_center_camera_offset is None:
            # A partial visible depth cap constrains distance well but is weak
            # tangentially.  Anchor that ambiguity once to the first accepted
            # SAM2 silhouette centre; later motion still comes from fresh
            # depth fits, so this is not a frozen image-space object pose.
            mask_u = 0.5 * (
                float(np.min(mask_columns)) + float(np.max(mask_columns))
            )
            mask_v = 0.5 * (
                float(np.min(mask_rows)) + float(np.max(mask_rows))
            )
            anchored = center.copy()
            anchored[0] = (mask_u - self.K[0, 2]) / self.K[0, 0] * center[2]
            anchored[1] = (mask_v - self.K[1, 2]) / self.K[1, 1] * center[2]
            offset = anchored - center
            offset[2] = 0.0
            if float(np.linalg.norm(offset)) > 0.020:
                self.last_sphere_completion = {
                    "applied": False,
                    "radius_m": float(radius),
                    "reason": "semantic/depth sphere centres disagree",
                    "center_anchor_offset_m": float(np.linalg.norm(offset)),
                    "fit_residual_p90_m": float(fit.residual_p90_m),
                    "fit_shell_inlier_fraction": float(fit.shell_inlier_fraction),
                }
                return reliable_rows, reliable_columns, reliable_depth_m
            self._sphere_center_camera_offset = offset
        center += self._sphere_center_camera_offset
        if (
            self._last_completed_sphere_center_camera is not None
            and float(
                np.linalg.norm(
                    center - self._last_completed_sphere_center_camera
                )
            )
            > 0.050
        ):
            self.last_sphere_completion = {
                "applied": False,
                "radius_m": float(radius),
                "reason": "completed sphere centre jumped over 50 mm",
                "fit_residual_p90_m": float(fit.residual_p90_m),
                "fit_shell_inlier_fraction": float(fit.shell_inlier_fraction),
            }
            return reliable_rows, reliable_columns, reliable_depth_m

        projected_u = self.K[0, 0] * center[0] / center[2] + self.K[0, 2]
        projected_v = self.K[1, 1] * center[1] / center[2] + self.K[1, 2]
        projected_radius = max(self.K[0, 0], self.K[1, 1]) * radius / max(
            center[2] - radius, 1.0e-6
        )
        half_extent = int(np.ceil(projected_radius * 1.15)) + 2
        x1 = max(0, int(np.floor(projected_u)) - half_extent)
        x2 = min(self.width, int(np.ceil(projected_u)) + half_extent + 1)
        y1 = max(0, int(np.floor(projected_v)) - half_extent)
        y2 = min(self.height, int(np.ceil(projected_v)) + half_extent + 1)
        local_rows, local_columns = np.indices((y2 - y1, x2 - x1))
        semantic_neighborhood_local = self._dilate_binary_mask(
            semantic_mask[y1:y2, x1:x2], radius_px=8
        )
        rows = (local_rows.reshape(-1) + y1).astype(np.int64)
        columns = (local_columns.reshape(-1) + x1).astype(np.int64)
        direction = np.stack(
            (
                (columns.astype(np.float64) - self.K[0, 2]) / self.K[0, 0],
                (rows.astype(np.float64) - self.K[1, 2]) / self.K[1, 1],
                np.ones(rows.size, dtype=np.float64),
            ),
            axis=1,
        )
        quadratic_a = np.sum(direction * direction, axis=1)
        quadratic_b = -2.0 * (direction @ center)
        quadratic_c = float(center @ center - radius * radius)
        discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
        intersects = discriminant >= 0.0
        completed_depth = np.full(rows.size, np.nan, dtype=np.float64)
        square_root = np.sqrt(np.maximum(discriminant[intersects], 0.0))
        near = (
            -quadratic_b[intersects] - square_root
        ) / (2.0 * quadratic_a[intersects])
        completed_depth[intersects] = near
        semantic_neighborhood = semantic_neighborhood_local.reshape(-1)
        usable = (
            intersects
            & semantic_neighborhood
            & np.isfinite(completed_depth)
            & (completed_depth > self.z_min)
            & (completed_depth < self.z_max)
        )
        if int(np.count_nonzero(usable)) < self.minimum_valid_points:
            self.last_sphere_completion = {
                "applied": False,
                "radius_m": float(radius),
                "fit_residual_p90_m": float(fit.residual_p90_m),
                "fit_shell_inlier_fraction": float(fit.shell_inlier_fraction),
            }
            return reliable_rows, reliable_columns, reliable_depth_m

        rows = rows[usable]
        columns = columns[usable]
        completed_depth = completed_depth[usable]
        effective_mask = np.zeros_like(semantic_mask, dtype=bool)
        effective_mask[rows, columns] = True
        self.last_effective_object_mask = effective_mask
        self._last_completed_sphere_center_camera = center.copy()

        self.last_sphere_completion = {
            "applied": True,
            "radius_m": float(radius),
            "fit_center_camera_m": center.copy(),
            "fit_residual_median_m": float(fit.residual_median_m),
            "fit_residual_p90_m": float(fit.residual_p90_m),
            "fit_shell_inlier_fraction": float(fit.shell_inlier_fraction),
            "semantic_mask_points": int(np.count_nonzero(object_mask)),
            "completed_surface_points": int(rows.size),
            "reliable_measured_points": int(reliable_depth_m.size),
            "semantic_neighborhood_radius_px": 8,
        }
        return rows, columns, completed_depth.astype(np.float32)

    def project(
        self,
        *,
        color_bgr: np.ndarray,
        depth_m: Optional[np.ndarray] = None,
        depth_raw: Optional[np.ndarray] = None,
        depth_scale_m_per_unit: Optional[float] = None,
        object_mask: np.ndarray,
        T_base_palm_at_capture: np.ndarray,
        captured_at_s: float,
        frame_id: int,
    ) -> PolicyPointFrame:
        color = np.asarray(color_bgr)
        mask = np.asarray(object_mask, dtype=bool)
        if color.shape != (self.height, self.width, 3) or color.dtype != np.uint8:
            raise ValueError(f"color_bgr must be uint8 {(self.height, self.width, 3)}")
        if mask.shape != (self.height, self.width):
            raise ValueError(f"object_mask must have shape {(self.height, self.width)}")
        if (depth_m is None) == (depth_raw is None):
            raise ValueError("provide exactly one of depth_m or depth_raw")
        timestamp = float(captured_at_s)
        if not np.isfinite(timestamp):
            raise ValueError("captured_at_s must be finite")
        palm = _rigid(T_base_palm_at_capture, "T_base_palm_at_capture")
        # This owned copy is the effective projector mask unless optional
        # fixed-sphere completion replaces it below.  It is not yet provenance
        # for a returned cloud: the too-few-points branch may instead retain the
        # previous fresh mask or report that no effective mask exists.
        self.last_effective_object_mask = mask.copy()

        if depth_raw is None:
            if depth_scale_m_per_unit is not None:
                raise ValueError("depth_scale_m_per_unit is valid only with depth_raw")
            depth = np.asarray(depth_m, dtype=np.float32)
            if depth.shape != (self.height, self.width):
                raise ValueError(f"depth_m must have shape {(self.height, self.width)}")
            selected_mask = (
                mask
                & np.isfinite(depth)
                & (depth > np.float32(self.z_min))
                & (depth < np.float32(self.z_max))
            )
            rows, columns = np.nonzero(selected_mask)
            selected_depth_m = depth[rows, columns]
        else:
            raw = np.asarray(depth_raw)
            if raw.shape != (self.height, self.width) or raw.dtype != np.uint16:
                raise ValueError(
                    "depth_raw must be uint16 with shape "
                    f"{(self.height, self.width)}"
                )
            if depth_scale_m_per_unit is None:
                raise ValueError("depth_raw requires depth_scale_m_per_unit")
            scale = float(depth_scale_m_per_unit)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("depth_scale_m_per_unit must be finite and positive")

            # Preserve the contract's row-major selection while converting
            # only object-mask pixels from Z16 to metres.  Materializing an
            # 848x480 float32 depth image here costs several milliseconds and
            # is unnecessary because every unmasked pixel is discarded.
            masked_flat = np.flatnonzero(mask)
            masked_depth_m = raw.reshape(-1)[masked_flat].astype(np.float32) * scale
            depth_valid = (
                np.isfinite(masked_depth_m)
                & (masked_depth_m > np.float32(self.z_min))
                & (masked_depth_m < np.float32(self.z_max))
            )
            selected_flat = masked_flat[depth_valid]
            selected_depth_m = masked_depth_m[depth_valid]
            rows = selected_flat // self.width
            columns = selected_flat % self.width

        # Keep the complete SAM2 silhouette, but remove calibrated tabletop
        # depth in 3-D before estimating the object's depth centre.  Doing the
        # support-plane operation on the 2-D mask used to fragment valid cube
        # masks and made a later erosion erase the remaining object pixels.
        rows, columns, selected_depth_m = (
            self._filter_pixels_above_support_plane(
                rows, columns, selected_depth_m
            )
        )

        # The semantic mask remains complete even where D435 depth at an
        # object boundary is invalid or mixed with the farther background.
        # Filter those depth outliers here, after mask selection, so they do
        # not stretch the policy cloud while the displayed/tracked SAM2 mask
        # still represents the full visible object silhouette.
        if (
            self.maximum_mask_depth_deviation_m is not None
            and selected_depth_m.size > 0
        ):
            median_depth = float(np.median(selected_depth_m))
            foreground_depth = (
                np.abs(selected_depth_m - median_depth)
                <= self.maximum_mask_depth_deviation_m
            )
            rows = rows[foreground_depth]
            columns = columns[foreground_depth]
            selected_depth_m = selected_depth_m[foreground_depth]

        rows, columns, selected_depth_m = self._complete_fixed_sphere_depth(
            object_mask=mask,
            reliable_rows=rows,
            reliable_columns=columns,
            reliable_depth_m=selected_depth_m,
        )
        # A shape-specific completion is optional; the support-plane contract
        # remains authoritative for every object type after that operation.
        rows, columns, selected_depth_m = (
            self._filter_pixels_above_support_plane(
                rows, columns, selected_depth_m
            )
        )

        source_count = int(rows.size)
        if source_count < self.minimum_valid_points:
            if self._latest_good is None:
                self.last_effective_object_mask = None
                self.last_effective_object_mask_provenance = (
                    _projector_mask_provenance(
                        None,
                        kind="unavailable_invalid_too_few_points",
                        source_frame_id=None,
                        source_captured_at_s=None,
                    )
                )
                return PolicyPointFrame(
                    xyzrgb_palm=np.zeros(
                        (self.num_points, self.point_feature_dim), dtype=np.float32
                    ),
                    valid=np.zeros(self.num_points, dtype=np.float32),
                    captured_at_s=timestamp,
                    frame_id=int(frame_id),
                    source_valid_points=source_count,
                    status="invalid_too_few_points",
                )
            if self.temporal_fallback == "motion_compensated":
                compensated = self._motion_compensated_point_frame(
                    current_mask=mask,
                    current_palm=palm,
                    timestamp=timestamp,
                    frame_id=int(frame_id),
                    source_count=source_count,
                )
                if compensated is not None:
                    return compensated
                self.last_effective_object_mask = None
                self.last_effective_object_mask_provenance = (
                    _projector_mask_provenance(
                        None,
                        kind="unavailable_motion_fallback_rejected",
                        source_frame_id=None,
                        source_captured_at_s=None,
                    )
                )
                return PolicyPointFrame(
                    xyzrgb_palm=np.zeros(
                        (self.num_points, self.point_feature_dim), dtype=np.float32
                    ),
                    valid=np.zeros(self.num_points, dtype=np.float32),
                    captured_at_s=timestamp,
                    frame_id=int(frame_id),
                    source_valid_points=source_count,
                    status="invalid_motion_fallback_rejected",
                )
            previous = self._latest_good
            previous_mask = self._latest_good_effective_object_mask
            if previous_mask is None:
                raise RuntimeError(
                    "retained policy cloud has no matching effective mask"
                )
            self.last_effective_object_mask = previous_mask.copy()
            self.last_effective_object_mask_provenance = (
                _projector_mask_provenance(
                    self.last_effective_object_mask,
                    kind="retained_previous_fresh_projector_mask",
                    source_frame_id=int(previous.frame_id),
                    source_captured_at_s=float(previous.captured_at_s),
                )
            )
            # This return happens before applying the current capture's palm
            # transform.  The packaged policy explicitly consumes the prior
            # cloud in its native capture-time palm frame ("stale_palm").
            return PolicyPointFrame(
                xyzrgb_palm=previous.xyzrgb_palm.copy(),
                valid=previous.valid.copy(),
                captured_at_s=previous.captured_at_s,
                frame_id=previous.frame_id,
                source_valid_points=source_count,
                status="stale_palm",
                measured_at_s=float(previous.captured_at_s),
                measured_frame_id=int(previous.frame_id),
            )

        if source_count > self.num_points:
            indices = np.linspace(
                0, source_count - 1, self.num_points, dtype=np.float64
            ).astype(np.int64)
            rows = rows[indices]
            columns = columns[indices]
            selected_depth_m = selected_depth_m[indices]
            output_count = self.num_points
        else:
            output_count = source_count

        z = selected_depth_m.astype(np.float64)
        x = (columns.astype(np.float64) - self.K[0, 2]) / self.K[0, 0] * z
        y = (rows.astype(np.float64) - self.K[1, 2]) / self.K[1, 1] * z
        camera_points = np.stack([x, y, z], axis=1)
        base_points = (
            camera_points @ self.T_base_camera[:3, :3].T + self.T_base_camera[:3, 3]
        )
        palm_points = (base_points - palm[:3, 3]) @ palm[:3, :3]
        output = np.zeros((self.num_points, self.point_feature_dim), dtype=np.float32)
        validity = np.zeros(self.num_points, dtype=np.float32)
        output[:output_count, :3] = palm_points.astype(np.float32)
        if self.point_feature_dim == 6:
            rgb = color[rows, columns, ::-1].astype(np.float64) / 255.0
            output[:output_count, 3:] = rgb.astype(np.float32)
        validity[:output_count] = 1.0
        result = PolicyPointFrame(
            xyzrgb_palm=output,
            valid=validity,
            captured_at_s=timestamp,
            frame_id=int(frame_id),
            source_valid_points=source_count,
            status="fresh",
            measured_at_s=timestamp,
            measured_frame_id=int(frame_id),
        )
        effective_mask = self.last_effective_object_mask
        if effective_mask is None:
            raise RuntimeError("fresh policy cloud has no effective object mask")
        completion_applied = bool(
            self.last_sphere_completion is not None
            and self.last_sphere_completion.get("applied", False)
        )
        self.last_effective_object_mask_provenance = (
            _projector_mask_provenance(
                effective_mask,
                kind=(
                    "current_frame_fixed_sphere_completion_mask"
                    if completion_applied
                    else "current_frame_projector_input_from_provider_mask"
                ),
                source_frame_id=int(frame_id),
                source_captured_at_s=timestamp,
            )
        )
        # Keep a private copy so a caller cannot corrupt the retained sensor
        # fallback by mutating arrays in the fresh frame it received.
        self._latest_good = PolicyPointFrame(
            xyzrgb_palm=result.xyzrgb_palm.copy(),
            valid=result.valid.copy(),
            captured_at_s=result.captured_at_s,
            frame_id=result.frame_id,
            source_valid_points=result.source_valid_points,
            status=result.status,
            measured_at_s=result.measured_at_s,
            measured_frame_id=result.measured_frame_id,
        )
        self._latest_good_effective_object_mask = effective_mask.copy()
        self._latest_good_input_object_mask = mask.copy()
        self._latest_good_T_base_palm = palm.copy()
        self._latest_good_depth_median_m = float(np.median(selected_depth_m))
        self._motion_compensated_steps = 0
        return result


class PolicyHistory:
    """Checkpoint-sized logical policy-rate history, oldest to newest."""

    def __init__(
        self,
        length: int = 4,
        point_feature_dim: Optional[int] = None,
        proprio_dim: int = 67,
    ) -> None:
        if isinstance(length, bool) or int(length) not in (4, 8, 16):
            raise ValueError("policy history length must be 4, 8 or 16")
        if isinstance(proprio_dim, bool) or int(proprio_dim) not in (67, 96):
            raise ValueError("policy proprio_dim must be 67 or 96")
        if (int(length), int(proprio_dim)) not in (
            (4, 67),
            (8, 96),
            (16, 96),
        ):
            raise ValueError(
                "supported history/proprio pairs are (4,67), (8,96) and (16,96)"
            )
        self.length = int(length)
        self.proprio_dim = int(proprio_dim)
        self._points: Deque[np.ndarray] = deque(maxlen=self.length)
        self._valid: Deque[np.ndarray] = deque(maxlen=self.length)
        self._proprio: Deque[np.ndarray] = deque(maxlen=self.length)
        if point_feature_dim is not None and (
            isinstance(point_feature_dim, bool) or int(point_feature_dim) not in (3, 6)
        ):
            raise ValueError("point_feature_dim must be 3 (XYZ) or 6 (XYZRGB)")
        self._point_feature_dim = (
            None if point_feature_dim is None else int(point_feature_dim)
        )

    def clear(self) -> None:
        self._points.clear()
        self._valid.clear()
        self._proprio.clear()

    def append(
        self, point_frame: PolicyPointFrame, proprioception: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(point_frame.xyzrgb_palm, dtype=np.float32)
        valid = np.asarray(point_frame.valid, dtype=np.float32)
        proprio = np.asarray(proprioception, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] != 128 or points.shape[1] not in (3, 6):
            raise ValueError("policy point frame must have shape (128,3) or (128,6)")
        if self._point_feature_dim is None:
            self._point_feature_dim = int(points.shape[1])
        if points.shape != (128, self._point_feature_dim):
            raise ValueError(
                "policy point feature mode changed within one history: "
                f"expected (128,{self._point_feature_dim}), actual={points.shape}"
            )
        if valid.shape != (128,) or proprio.shape != (self.proprio_dim,):
            raise ValueError(
                "policy valid/proprio shapes must be (128,) and "
                f"({self.proprio_dim},)"
            )
        if not (
            np.all(np.isfinite(points))
            and np.all(np.isfinite(valid))
            and np.all(np.isfinite(proprio))
        ):
            raise ValueError("policy history values must be finite")
        if not self._points:
            for _ in range(self.length):
                self._points.append(points.copy())
                self._valid.append(valid.copy())
                self._proprio.append(proprio.copy())
        else:
            self._points.append(points.copy())
            self._valid.append(valid.copy())
            self._proprio.append(proprio.copy())
        return self.arrays()

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self._points) != self.length:
            raise RuntimeError("policy history has not been bootstrapped")
        return (
            np.stack(tuple(self._points)).astype(np.float32),
            np.stack(tuple(self._valid)).astype(np.float32),
            np.stack(tuple(self._proprio)).astype(np.float32),
        )


class Proprio67Builder:
    def __init__(
        self,
        *,
        q_home_rad: np.ndarray,
        q_hand_close_rad: np.ndarray = Q_HAND_SEMANTIC_CLOSE_RAD,
    ) -> None:
        self.q_home = _finite(q_home_rad, (7,), "q_home_rad").astype(np.float32)
        self.q_hand_close = _finite(q_hand_close_rad, (6,), "q_hand_close_rad").astype(
            np.float32
        )
        if np.any(self.q_hand_close <= 0.0):
            raise ValueError("q_hand_close_rad must be positive")

    def build(
        self,
        *,
        franka_q_rad: np.ndarray,
        franka_dq_rad_s: np.ndarray,
        rh56_virtual_q_policy_order_rad: np.ndarray,
        rh56_virtual_dq_policy_order_rad_s: np.ndarray,
        T_base_palm: np.ndarray,
        palm_linear_velocity_base_m_s: np.ndarray,
        palm_angular_velocity_base_rad_s: np.ndarray,
        fingertip_positions_base_m: np.ndarray,
        previous_executed_action13: np.ndarray = INITIAL_PREVIOUS_ACTION13,
    ) -> np.ndarray:
        q = _finite(franka_q_rad, (7,), "franka_q_rad").astype(np.float32)
        dq = _finite(franka_dq_rad_s, (7,), "franka_dq_rad_s").astype(np.float32)
        hand_q = _finite(
            rh56_virtual_q_policy_order_rad,
            (6,),
            "rh56_virtual_q_policy_order_rad",
        ).astype(np.float32)
        hand_dq = _finite(
            rh56_virtual_dq_policy_order_rad_s,
            (6,),
            "rh56_virtual_dq_policy_order_rad_s",
        ).astype(np.float32)
        palm = _rigid(T_base_palm, "T_base_palm")
        linear = _finite(
            palm_linear_velocity_base_m_s,
            (3,),
            "palm_linear_velocity_base_m_s",
        ).astype(np.float32)
        angular = _finite(
            palm_angular_velocity_base_rad_s,
            (3,),
            "palm_angular_velocity_base_rad_s",
        ).astype(np.float32)
        tips = _finite(fingertip_positions_base_m, (5, 3), "fingertip_positions_base_m")
        previous = _finite(
            previous_executed_action13, (13,), "previous_executed_action13"
        ).astype(np.float32)

        relative_tips_palm = (tips - palm[:3, 3]) @ palm[:3, :3]
        hand_scaled = np.float32(2.0) * hand_q / self.q_hand_close - np.float32(1.0)
        result = np.concatenate(
            [
                q - self.q_home,
                np.float32(0.1) * dq,
                hand_scaled,
                np.float32(0.1) * hand_dq,
                palm[:3, 3].astype(np.float32),
                rotation_to_quaternion_wxyz(palm[:3, :3]).astype(np.float32),
                np.float32(0.1) * linear,
                np.float32(0.1) * angular,
                relative_tips_palm.astype(np.float32).reshape(15),
                previous,
            ]
        ).astype(np.float32)
        if result.shape != (67,) or not np.all(np.isfinite(result)):
            raise RuntimeError("constructed proprioception is not finite 67-D")
        return result


__all__ = [
    "MaskedRGBDProjector",
    "POLICY_RGBD_RESOLUTION_SIZES",
    "PolicyHistory",
    "PolicyPointFrame",
    "PolicyRGBDInputs",
    "PolicyRGBDResolutionAdapter",
    "Proprio67Builder",
    "pose_from_position_quaternion_wxyz",
    "quaternion_wxyz_to_rotation",
    "resolve_policy_rgbd_resolution",
    "rotation_to_quaternion_wxyz",
]
