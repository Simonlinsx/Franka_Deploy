"""Pure, offline point-cloud mask variants for V94 perception audits.

Nothing in this module opens a device or changes the live policy path.  The
filters are deliberately explicit candidates: callers can measure their
geometry/action sensitivity before deciding whether any of them belongs in a
future reviewed deployment contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


FILTER_NAMES: Tuple[str, ...] = (
    "raw",
    "erode3",
    "robust_depth",
    "erode3_robust_depth",
    "support_plane_clearance",
    "erode3_support_plane_clearance",
    "sphere_shell_support_plane",
)


@dataclass(frozen=True)
class SphereFit:
    center_base_m: np.ndarray
    radius_m: float
    residual_median_m: float
    residual_p90_m: float
    shell_inlier_fraction: float
    valid: bool


@dataclass(frozen=True)
class MaskCandidates:
    masks: Mapping[str, np.ndarray]
    raw_valid_depth_points: int
    robust_depth_center_m: float
    robust_depth_half_width_m: float
    sphere_fit: SphereFit


def _finite_matrix(value: np.ndarray, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape {shape}")
    return result


def _validated_rgbd_mask(
    depth_m: np.ndarray, object_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(object_mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth_m/object_mask must be equally shaped 2-D arrays")
    return depth, mask


def erode_mask(object_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Return a binary erosion without mutating the supplied mask."""

    mask = np.asarray(object_mask, dtype=bool)
    kernel = int(kernel_size)
    if mask.ndim != 2:
        raise ValueError("object_mask must be 2-D")
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("erosion kernel_size must be a positive odd integer")
    if kernel == 1:
        return mask.copy()
    radius = kernel // 2
    padded = np.pad(
        mask.astype(np.int32), ((radius, radius), (radius, radius)), mode="constant"
    )
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    window_sum = (
        integral[kernel:, kernel:]
        - integral[:-kernel, kernel:]
        - integral[kernel:, :-kernel]
        + integral[:-kernel, :-kernel]
    )
    return window_sum == kernel * kernel


def robust_depth_mask(
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    *,
    minimum_half_width_m: float = 0.012,
    maximum_half_width_m: float = 0.045,
    mad_scale: float = 3.0,
) -> Tuple[np.ndarray, float, float]:
    """Keep an adaptive median/MAD depth band inside the semantic mask.

    The 45 mm default cap mirrors the transferred simulator's configured RGB-D
    depth tolerance.  This remains only an A/B candidate, not an assertion that
    the real tracker mask and simulator semantic mask have identical edges.
    """

    depth, mask = _validated_rgbd_mask(depth_m, object_mask)
    lower = float(minimum_half_width_m)
    upper = float(maximum_half_width_m)
    scale = float(mad_scale)
    if not all(np.isfinite(v) for v in (lower, upper, scale)):
        raise ValueError("robust depth parameters must be finite")
    if lower <= 0.0 or upper < lower or scale <= 0.0:
        raise ValueError("invalid robust depth parameters")
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    values = depth[valid].astype(np.float64)
    if values.size == 0:
        return np.zeros_like(mask), float("nan"), float("nan")
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    robust_sigma = 1.4826 * mad
    half_width = float(np.clip(scale * robust_sigma, lower, upper))
    selected = valid & (np.abs(depth.astype(np.float64) - center) <= half_width)
    return selected, center, half_width


def masked_points_base(
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    *,
    camera_K: np.ndarray,
    T_base_camera_optical: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject masked finite positive depths into robot-base coordinates."""

    depth, mask = _validated_rgbd_mask(depth_m, object_mask)
    K = _finite_matrix(camera_K, (3, 3), "camera_K")
    transform = _finite_matrix(
        T_base_camera_optical, (4, 4), "T_base_camera_optical"
    )
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64), rows, columns
    z = depth[rows, columns].astype(np.float64)
    x = (columns.astype(np.float64) - K[0, 2]) / K[0, 0] * z
    y = (rows.astype(np.float64) - K[1, 2]) / K[1, 1] * z
    camera = np.stack((x, y, z), axis=1)
    base = camera @ transform[:3, :3].T + transform[:3, 3]
    return base, rows, columns


def fit_fixed_radius_sphere(
    points_base_m: np.ndarray,
    *,
    camera_origin_base_m: np.ndarray,
    radius_m: float,
    shell_tolerance_m: float = 0.006,
    huber_width_m: float = 0.004,
    maximum_iterations: int = 30,
) -> SphereFit:
    """Robustly fit a known-radius sphere to a partial visible surface.

    The camera-facing surface centroid is shifted away from the camera by one
    radius for initialization.  Huber-weighted Gauss-Newton then estimates
    only the center; the commissioned radius never changes.  A partial sphere
    is intrinsically less constrained than a full scan, so fit quality is
    always returned and must be audited rather than trusted implicitly.
    """

    points = np.asarray(points_base_m, dtype=np.float64)
    origin = np.asarray(camera_origin_base_m, dtype=np.float64)
    radius = float(radius_m)
    tolerance = float(shell_tolerance_m)
    huber = float(huber_width_m)
    iterations = int(maximum_iterations)
    invalid = SphereFit(
        center_base_m=np.full(3, np.nan, dtype=np.float64),
        radius_m=radius,
        residual_median_m=float("nan"),
        residual_p90_m=float("nan"),
        shell_inlier_fraction=0.0,
        valid=False,
    )
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points_base_m must have shape [N,3]")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("camera_origin_base_m must have finite shape [3]")
    if not all(np.isfinite(v) for v in (radius, tolerance, huber)):
        raise ValueError("sphere parameters must be finite")
    if radius <= 0.0 or tolerance <= 0.0 or huber <= 0.0 or iterations <= 0:
        raise ValueError("sphere parameters must be positive")
    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.shape[0] < 16:
        return invalid

    surface_center = np.median(finite, axis=0)
    viewing_direction = surface_center - origin
    norm = float(np.linalg.norm(viewing_direction))
    if norm <= 1.0e-9:
        return invalid
    center = surface_center + viewing_direction / norm * radius

    # Regularization is tiny relative to the normal matrix but prevents a
    # nearly planar visible cap from producing an unbounded tangential step.
    for _ in range(iterations):
        delta_points = center[None, :] - finite
        distances = np.linalg.norm(delta_points, axis=1)
        usable = distances > 1.0e-9
        if int(np.count_nonzero(usable)) < 16:
            return invalid
        residual = distances[usable] - radius
        jacobian = delta_points[usable] / distances[usable, None]
        absolute = np.abs(residual)
        weights = np.ones_like(absolute)
        beyond = absolute > huber
        weights[beyond] = huber / absolute[beyond]
        weighted_jacobian = jacobian * weights[:, None]
        normal = jacobian.T @ weighted_jacobian + np.eye(3) * 1.0e-6
        gradient = jacobian.T @ (weights * residual)
        try:
            step = -np.linalg.solve(normal, gradient)
        except np.linalg.LinAlgError:
            return invalid
        maximum_step = radius * 0.5
        step_norm = float(np.linalg.norm(step))
        if step_norm > maximum_step:
            step *= maximum_step / step_norm
        center += step
        if step_norm < 1.0e-7:
            break

    absolute_residual = np.abs(np.linalg.norm(finite - center, axis=1) - radius)
    median = float(np.median(absolute_residual))
    p90 = float(np.percentile(absolute_residual, 90.0))
    fraction = float(np.mean(absolute_residual <= tolerance))
    # A fit that explains fewer than a quarter of the raw mask is too weak to
    # create a diagnostic shell candidate.  Its metrics remain visible.
    valid_fit = bool(
        np.all(np.isfinite(center))
        and median <= max(tolerance, radius * 0.30)
        and fraction >= 0.25
    )
    return SphereFit(
        center_base_m=center.astype(np.float64),
        radius_m=radius,
        residual_median_m=median,
        residual_p90_m=p90,
        shell_inlier_fraction=fraction,
        valid=valid_fit,
    )


def _normalized_support_plane(plane_abcd: np.ndarray) -> Tuple[np.ndarray, float]:
    plane = np.asarray(plane_abcd, dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise ValueError("support_plane_abcd must have finite shape [4]")
    magnitude = float(np.linalg.norm(plane[:3]))
    if magnitude <= 1.0e-9:
        raise ValueError("support plane normal must be non-zero")
    normal = plane[:3] / magnitude
    offset = float(plane[3] / magnitude)
    # The commissioned tabletop normal is expected to point roughly upward.
    # Canonicalizing the sign keeps positive distance on the object side.
    if normal[2] < 0.0:
        normal *= -1.0
        offset *= -1.0
    return normal, offset


def support_plane_clearance_mask(
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    *,
    camera_K: np.ndarray,
    T_base_camera_optical: np.ndarray,
    support_plane_abcd: np.ndarray,
    minimum_clearance_m: float = 0.006,
) -> Tuple[np.ndarray, float, float]:
    """Remove mask points inside a conservative band around the support plane."""

    clearance = float(minimum_clearance_m)
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("minimum_clearance_m must be finite and non-negative")
    normal, offset = _normalized_support_plane(support_plane_abcd)
    points, rows, columns = masked_points_base(
        depth_m,
        object_mask,
        camera_K=camera_K,
        T_base_camera_optical=T_base_camera_optical,
    )
    selected = np.zeros(np.asarray(object_mask).shape, dtype=bool)
    if points.shape[0] == 0:
        return selected, float("nan"), float("nan")
    signed_height = points @ normal + offset
    keep = signed_height > clearance
    selected[rows[keep], columns[keep]] = True
    return selected, float(np.median(signed_height)), float(np.percentile(signed_height, 5.0))


def fit_fixed_radius_sphere_on_support_plane(
    points_base_m: np.ndarray,
    *,
    support_plane_abcd: np.ndarray,
    radius_m: float,
    shell_tolerance_m: float = 0.006,
    huber_width_m: float = 0.004,
    maximum_iterations: int = 30,
) -> SphereFit:
    """Fit sphere center tangent coordinates with contact-plane height fixed.

    The center is constrained to the plane parallel to the measured support
    plane at exactly one radius on its positive side.  This removes the weak
    depth-axis degree of freedom of a camera-visible spherical cap and makes
    the physical assumption (sphere resting on this table) fully auditable.
    """

    points = np.asarray(points_base_m, dtype=np.float64)
    radius = float(radius_m)
    tolerance = float(shell_tolerance_m)
    huber = float(huber_width_m)
    iterations = int(maximum_iterations)
    normal, offset = _normalized_support_plane(support_plane_abcd)
    invalid = SphereFit(
        center_base_m=np.full(3, np.nan, dtype=np.float64),
        radius_m=radius,
        residual_median_m=float("nan"),
        residual_p90_m=float("nan"),
        shell_inlier_fraction=0.0,
        valid=False,
    )
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points_base_m must have shape [N,3]")
    if not all(np.isfinite(v) for v in (radius, tolerance, huber)):
        raise ValueError("sphere parameters must be finite")
    if radius <= 0.0 or tolerance <= 0.0 or huber <= 0.0 or iterations <= 0:
        raise ValueError("sphere parameters must be positive")
    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.shape[0] < 16:
        return invalid

    # Build an orthonormal basis spanning the support plane.
    helper = np.asarray([1.0, 0.0, 0.0])
    if abs(float(normal @ helper)) > 0.8:
        helper = np.asarray([0.0, 1.0, 0.0])
    tangent0 = np.cross(normal, helper)
    tangent0 /= np.linalg.norm(tangent0)
    tangent1 = np.cross(normal, tangent0)
    tangent = np.stack((tangent0, tangent1), axis=1)

    initial = np.median(finite, axis=0)
    initial += (radius - (float(normal @ initial) + offset)) * normal
    anchor = initial.copy()
    coordinates = np.zeros(2, dtype=np.float64)
    for _ in range(iterations):
        center = anchor + tangent @ coordinates
        delta_points = center[None, :] - finite
        distances = np.linalg.norm(delta_points, axis=1)
        usable = distances > 1.0e-9
        if int(np.count_nonzero(usable)) < 16:
            return invalid
        residual = distances[usable] - radius
        jacobian_center = delta_points[usable] / distances[usable, None]
        jacobian = jacobian_center @ tangent
        absolute = np.abs(residual)
        weights = np.ones_like(absolute)
        beyond = absolute > huber
        weights[beyond] = huber / absolute[beyond]
        weighted_jacobian = jacobian * weights[:, None]
        normal_matrix = jacobian.T @ weighted_jacobian + np.eye(2) * 1.0e-6
        gradient = jacobian.T @ (weights * residual)
        try:
            step = -np.linalg.solve(normal_matrix, gradient)
        except np.linalg.LinAlgError:
            return invalid
        maximum_step = radius * 0.5
        step_norm = float(np.linalg.norm(step))
        if step_norm > maximum_step:
            step *= maximum_step / step_norm
        coordinates += step
        if step_norm < 1.0e-7:
            break
    center = anchor + tangent @ coordinates
    absolute_residual = np.abs(np.linalg.norm(finite - center, axis=1) - radius)
    median = float(np.median(absolute_residual))
    p90 = float(np.percentile(absolute_residual, 90.0))
    fraction = float(np.mean(absolute_residual <= tolerance))
    valid_fit = bool(median <= max(tolerance, radius * 0.30) and fraction >= 0.25)
    return SphereFit(
        center_base_m=center,
        radius_m=radius,
        residual_median_m=median,
        residual_p90_m=p90,
        shell_inlier_fraction=fraction,
        valid=valid_fit,
    )


def build_mask_candidates(
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    *,
    camera_K: np.ndarray,
    T_base_camera_optical: np.ndarray,
    sphere_radius_m: Optional[float] = 0.030,
    sphere_shell_tolerance_m: float = 0.006,
    support_plane_abcd: Optional[np.ndarray] = None,
    support_plane_clearance_m: float = 0.006,
) -> MaskCandidates:
    """Build the fixed A/B candidate set for one exact RGB-D frame."""

    depth, raw = _validated_rgbd_mask(depth_m, object_mask)
    valid_depth = raw & np.isfinite(depth) & (depth > 0.0)
    eroded = erode_mask(raw, 3)
    depth_selected, depth_center, depth_half_width = robust_depth_mask(depth, raw)

    base_points, rows, columns = masked_points_base(
        depth,
        raw,
        camera_K=camera_K,
        T_base_camera_optical=T_base_camera_optical,
    )
    if support_plane_abcd is None:
        support_mask = np.zeros_like(raw)
    else:
        support_mask, _height_median, _height_p05 = support_plane_clearance_mask(
            depth,
            raw,
            camera_K=camera_K,
            T_base_camera_optical=T_base_camera_optical,
            support_plane_abcd=support_plane_abcd,
            minimum_clearance_m=support_plane_clearance_m,
        )
    if sphere_radius_m is None or support_plane_abcd is None:
        sphere_fit = SphereFit(
            center_base_m=np.full(3, np.nan, dtype=np.float64),
            radius_m=float("nan"),
            residual_median_m=float("nan"),
            residual_p90_m=float("nan"),
            shell_inlier_fraction=0.0,
            valid=False,
        )
    else:
        sphere_fit = fit_fixed_radius_sphere_on_support_plane(
            base_points,
            support_plane_abcd=support_plane_abcd,
            radius_m=float(sphere_radius_m),
            shell_tolerance_m=float(sphere_shell_tolerance_m),
        )
    sphere_mask = np.zeros_like(raw)
    if sphere_fit.valid and base_points.shape[0] > 0:
        residual = np.abs(
            np.linalg.norm(base_points - sphere_fit.center_base_m, axis=1)
            - sphere_fit.radius_m
        )
        keep = residual <= float(sphere_shell_tolerance_m)
        sphere_mask[rows[keep], columns[keep]] = True

    masks: Dict[str, np.ndarray] = {
        "raw": raw.copy(),
        "erode3": eroded,
        "robust_depth": depth_selected,
        "erode3_robust_depth": eroded & depth_selected,
        "support_plane_clearance": support_mask,
        "erode3_support_plane_clearance": eroded & support_mask,
        "sphere_shell_support_plane": sphere_mask,
    }
    return MaskCandidates(
        masks=masks,
        raw_valid_depth_points=int(np.count_nonzero(valid_depth)),
        robust_depth_center_m=float(depth_center),
        robust_depth_half_width_m=float(depth_half_width),
        sphere_fit=sphere_fit,
    )


__all__ = [
    "FILTER_NAMES",
    "MaskCandidates",
    "SphereFit",
    "build_mask_candidates",
    "erode_mask",
    "fit_fixed_radius_sphere",
    "fit_fixed_radius_sphere_on_support_plane",
    "masked_points_base",
    "robust_depth_mask",
    "support_plane_clearance_mask",
]
