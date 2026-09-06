from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from dynamic_pcd.calibration.transforms import validate_rigid_transform
from dynamic_pcd.types import RGBDFrame
from dynamic_pcd.utils.geometry import erode_mask, largest_component


def _try_open3d():
    try:
        import open3d as o3d
        return o3d
    except Exception:
        return None


@dataclass
class ExtractedObjectPCD:
    points: np.ndarray          # [M, 3]
    colors: np.ndarray          # [M, 3], RGB in [0, 1]
    policy_points: np.ndarray   # [N, 3] or [N, 6]
    reference_points: np.ndarray  # [N, 3] or [N, 6], never object-centered
    center: np.ndarray          # [3]
    valid: bool
    message: str = ""


@dataclass
class ExtractedScenePCD:
    """Lightweight color scene cloud expressed in the extractor base frame."""

    points: np.ndarray          # [M, 3]
    colors: np.ndarray          # [M, 3], RGB in [0, 1]


class ObjectPointCloudExtractor:
    def __init__(self, cfg: Dict[str, Any], T_base_camera: Optional[np.ndarray] = None):
        self.cfg = cfg
        transform = np.eye(4) if T_base_camera is None else T_base_camera
        self.T_base_camera = validate_rigid_transform(
            transform, name="T_base_camera"
        ).astype(np.float32)
        self._scene_ray_cache: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]] = {}
        self._support_plane_normal: Optional[np.ndarray] = None
        self._support_plane_offset = 0.0
        plane = self.cfg.get("support_plane_abcd")
        if plane is not None:
            plane_array = np.asarray(plane, dtype=np.float64)
            if plane_array.shape != (4,) or not np.all(np.isfinite(plane_array)):
                raise ValueError("support_plane_abcd must contain four finite values")
            magnitude = float(np.linalg.norm(plane_array[:3]))
            if magnitude <= 1.0e-9:
                raise ValueError("support_plane_abcd normal must be non-zero")
            normal = plane_array[:3] / magnitude
            offset = float(plane_array[3] / magnitude)
            if normal[2] < 0.0:
                normal *= -1.0
                offset *= -1.0
            if normal[2] < 0.9:
                raise ValueError("support_plane_abcd must describe an upward tabletop")
            clearance = float(self.cfg.get("support_plane_min_clearance_m", 0.0))
            if not np.isfinite(clearance) or clearance < 0.0:
                raise ValueError(
                    "support_plane_min_clearance_m must be finite and non-negative"
                )
            self._support_plane_normal = normal.astype(np.float32)
            self._support_plane_offset = offset

    @property
    def support_plane_abcd(self) -> Optional[np.ndarray]:
        if self._support_plane_normal is None:
            return None
        return np.concatenate(
            (
                self._support_plane_normal.astype(np.float64),
                np.asarray([self._support_plane_offset], dtype=np.float64),
            )
        )

    @property
    def support_plane_min_clearance_m(self) -> float:
        if self._support_plane_normal is None:
            return 0.0
        return float(self.cfg.get("support_plane_min_clearance_m", 0.0))

    def extract(
        self,
        frame: RGBDFrame,
        mask: np.ndarray,
        *,
        erode_kernel_override: Optional[int] = None,
    ) -> ExtractedObjectPCD:
        mask_u8 = (mask > 0).astype(np.uint8)

        erode_kernel = int(
            self.cfg.get("erode_kernel", 3)
            if erode_kernel_override is None
            else erode_kernel_override
        )
        if erode_kernel < 0:
            raise ValueError("erode_kernel_override must be >=0")
        if erode_kernel > 0:
            mask_u8 = erode_mask(mask_u8, erode_kernel)

        # Keep the largest object-like component after erosion.
        mask_u8 = largest_component(mask_u8, min_area=20)

        stride = max(1, int(self.cfg.get("stride", 1)))
        depth_raw = frame.depth_raw[::stride, ::stride]
        mask_s = mask_u8[::stride, ::stride].astype(bool)
        color_s = frame.color_bgr[::stride, ::stride, :]

        cam_cfg_zmin = 0.0
        cam_cfg_zmax = 10.0
        z_min = float(self.cfg.get("z_min", cam_cfg_zmin)) if "z_min" in self.cfg else cam_cfg_zmin
        z_max = float(self.cfg.get("z_max", cam_cfg_zmax)) if "z_max" in self.cfg else cam_cfg_zmax

        # Select masked pixels before converting depth to float.  The previous
        # implementation allocated float depth plus full-frame U/V index maps
        # on every frame even though the object usually occupies only a few
        # percent of the image.
        candidate = mask_s & np.isfinite(depth_raw)
        v_idx, u_idx = np.nonzero(candidate)
        if len(v_idx) < 10:
            return self._invalid("too few valid depth pixels after mask")

        z = depth_raw[v_idx, u_idx].astype(np.float32) * float(frame.depth_scale)
        depth_valid = np.isfinite(z) & (z > z_min) & (z < z_max)
        if np.count_nonzero(depth_valid) < 10:
            return self._invalid("too few valid depth pixels after mask")
        z = z[depth_valid]
        u = u_idx[depth_valid].astype(np.float32) * stride
        v = v_idx[depth_valid].astype(np.float32) * stride

        intr = frame.intrinsics
        x = (u - intr.ppx) / intr.fx * z
        y = (v - intr.ppy) / intr.fy * z
        pts_cam = np.stack([x, y, z], axis=1).astype(np.float32)

        pts_base = self._transform_points(pts_cam)
        colors = color_s[
            v_idx[depth_valid], u_idx[depth_valid], ::-1
        ].astype(np.float32) / 255.0

        pts_base, colors = self._support_plane_crop(pts_base, colors)
        if len(pts_base) < 10:
            return self._invalid("too few points above support plane")

        pts_base, colors = self._workspace_crop(pts_base, colors)
        if len(pts_base) < 10:
            return self._invalid("too few points after workspace crop")

        pts_base, colors = self._clean_points(pts_base, colors)
        if len(pts_base) < 10:
            return self._invalid("too few points after cleaning")

        center = np.median(pts_base, axis=0).astype(np.float32)
        policy_points, reference_points = self._sample_policy_points(
            pts_base, colors, center
        )

        return ExtractedObjectPCD(
            points=pts_base.astype(np.float32),
            colors=colors.astype(np.float32),
            policy_points=policy_points.astype(np.float32),
            reference_points=reference_points.astype(np.float32),
            center=center,
            valid=True,
            message=f"points={len(pts_base)}",
        )

    def extract_sanitized_core(
        self,
        frame: RGBDFrame,
        mask: np.ndarray,
        *,
        erode_kernel: int,
    ) -> ExtractedObjectPCD:
        """Extract an already identity-sanitized mask with bounded erosion."""

        return self.extract(
            frame,
            mask,
            erode_kernel_override=int(erode_kernel),
        )

    def extract_scene(
        self,
        frame: RGBDFrame,
        exclude_mask: Optional[np.ndarray] = None,
        stride: int = 4,
    ) -> ExtractedScenePCD:
        """Back-project a lightweight scene cloud into ``robot_base``.

        This path deliberately skips voxel filtering and statistical outlier
        removal so that visualization cannot dominate the real-time perception
        loop.  ``exclude_mask`` is normally the segmented object mask; omitting
        those pixels prevents the dim scene geometry from hiding the separately
        highlighted object geometry.
        """

        depth_raw = np.asarray(frame.depth_raw)
        color_bgr = np.asarray(frame.color_bgr)
        if depth_raw.ndim != 2:
            raise ValueError(f"depth_raw must be HxW, got {depth_raw.shape}")
        if color_bgr.shape[:2] != depth_raw.shape:
            raise ValueError(
                "aligned color/depth shape mismatch: "
                f"color={color_bgr.shape[:2]}, depth={depth_raw.shape}"
            )

        stride = max(1, int(stride))
        depth = depth_raw[::stride, ::stride].astype(np.float32) * float(frame.depth_scale)
        z_min = float(self.cfg.get("z_min", 0.0))
        z_max = float(self.cfg.get("z_max", 10.0))
        valid = np.isfinite(depth) & (depth > z_min) & (depth < z_max)

        if exclude_mask is not None:
            mask = np.asarray(exclude_mask)
            if mask.shape != depth_raw.shape:
                raise ValueError(
                    f"exclude_mask shape {mask.shape} does not match depth {depth_raw.shape}"
                )
            valid &= ~mask[::stride, ::stride].astype(bool)

        if not np.any(valid):
            return ExtractedScenePCD(
                points=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.float32),
            )

        intr = frame.intrinsics
        key = (
            depth_raw.shape[0],
            depth_raw.shape[1],
            stride,
            float(intr.fx),
            float(intr.fy),
            float(intr.ppx),
            float(intr.ppy),
        )
        rays = self._scene_ray_cache.get(key)
        if rays is None:
            vv, uu = np.mgrid[
                0 : depth_raw.shape[0] : stride,
                0 : depth_raw.shape[1] : stride,
            ]
            ray_x = (uu.astype(np.float32) - float(intr.ppx)) / float(intr.fx)
            ray_y = (vv.astype(np.float32) - float(intr.ppy)) / float(intr.fy)
            rays = (ray_x, ray_y)
            self._scene_ray_cache[key] = rays

        ray_x, ray_y = rays
        z = depth[valid]
        pts_cam = np.stack(
            [ray_x[valid] * z, ray_y[valid] * z, z], axis=1
        ).astype(np.float32)
        pts_base = self._transform_points(pts_cam).astype(np.float32)
        colors = (
            color_bgr[::stride, ::stride, ::-1][valid].astype(np.float32)
            / 255.0
        )
        finite = np.all(np.isfinite(pts_base), axis=1)
        return ExtractedScenePCD(points=pts_base[finite], colors=colors[finite])

    def _transform_points(self, pts_cam: np.ndarray) -> np.ndarray:
        # Rigid transforms do not require allocating homogeneous coordinates.
        return (
            pts_cam @ self.T_base_camera[:3, :3].T
            + self.T_base_camera[:3, 3]
        )

    def _workspace_crop(self, pts: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mn = self.cfg.get("workspace_min", None)
        mx = self.cfg.get("workspace_max", None)
        if mn is None or mx is None:
            return pts, colors
        mn = np.asarray(mn, dtype=np.float32)
        mx = np.asarray(mx, dtype=np.float32)
        keep = np.all(pts >= mn[None, :], axis=1) & np.all(pts <= mx[None, :], axis=1)
        return pts[keep], colors[keep]

    def _support_plane_keep_mask(self, pts: np.ndarray) -> np.ndarray:
        """Return object points safely above the calibrated tabletop.

        This is an observation filter, not a robot workspace limit. It leaves
        the tracker-owned semantic state unchanged while preventing
        table/mixed-depth pixels from lowering the policy object cloud.
        """

        finite = np.all(np.isfinite(pts), axis=1)
        if self._support_plane_normal is None:
            return finite
        clearance = float(self.cfg.get("support_plane_min_clearance_m", 0.0))
        signed_height = (
            pts @ self._support_plane_normal + self._support_plane_offset
        )
        return finite & (signed_height > clearance)

    def _support_plane_crop(
        self, pts: np.ndarray, colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        keep = self._support_plane_keep_mask(pts)
        return pts[keep], colors[keep]

    def filter_semantic_mask_above_support_plane(
        self, frame: RGBDFrame, mask: np.ndarray
    ) -> np.ndarray:
        """Suppress proven tabletop pixels while retaining unknown-depth shape.

        SAM2 can correctly label an object edge even when the D435 has no
        reliable depth there. A blanket depth intersection therefore cuts off
        valid silhouettes such as the top of a sphere. This filter only
        changes semantic pixels whose depth proves they lie on/below the
        calibrated support-plane clearance.
        """

        semantic = (np.asarray(mask) > 0).astype(np.uint8)
        if semantic.shape != frame.depth_raw.shape:
            raise ValueError(
                f"mask shape {semantic.shape} differs from depth "
                f"{frame.depth_raw.shape}"
            )
        if self._support_plane_normal is None or not np.any(semantic):
            return semantic

        rows, columns = np.nonzero(semantic)
        depth_m = (
            np.asarray(frame.depth_raw)[rows, columns].astype(np.float32)
            * float(frame.depth_scale)
        )
        z_min = float(self.cfg.get("z_min", 0.0))
        z_max = float(self.cfg.get("z_max", 10.0))
        reliable = np.isfinite(depth_m) & (depth_m > z_min) & (depth_m < z_max)
        if not np.any(reliable):
            return semantic

        z = depth_m[reliable]
        u = columns[reliable].astype(np.float32)
        v = rows[reliable].astype(np.float32)
        intrinsics = frame.intrinsics
        x = (u - float(intrinsics.ppx)) / float(intrinsics.fx) * z
        y = (v - float(intrinsics.ppy)) / float(intrinsics.fy) * z
        points_camera = np.stack((x, y, z), axis=1).astype(np.float32)
        points_base = self._transform_points(points_camera)
        keep = self._support_plane_keep_mask(points_base)

        filtered = semantic.copy()
        reliable_rows = rows[reliable]
        reliable_columns = columns[reliable]
        filtered[reliable_rows[~keep], reliable_columns[~keep]] = 0
        return filtered

    def _clean_points(self, pts: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        o3d = _try_open3d()
        voxel_size = float(self.cfg.get("voxel_size", 0.003) or 0.0)
        remove_outliers = bool(self.cfg.get("remove_outliers", True))
        if o3d is None or (voxel_size <= 0 and not remove_outliers):
            return pts, colors

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)

        if remove_outliers and len(pcd.points) >= 30:
            nb = int(self.cfg.get("outlier_nb_neighbors", 20))
            std = float(self.cfg.get("outlier_std_ratio", 1.5))
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)

        pts2 = np.asarray(pcd.points).astype(np.float32)
        cols2 = np.asarray(pcd.colors).astype(np.float32)
        return pts2, cols2

    def _sample_policy_points(
        self, pts: np.ndarray, colors: np.ndarray, center: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = int(self.cfg.get("num_points", 1024))
        use_rgb = bool(self.cfg.get("use_rgb", False))
        center_policy = bool(self.cfg.get("center_policy_points", True))

        if len(pts) >= n:
            idx = np.random.choice(len(pts), n, replace=False)
        else:
            idx = np.random.choice(len(pts), n, replace=True)
        xyz_reference = pts[idx].astype(np.float32)
        xyz = xyz_reference.copy()
        if center_policy:
            xyz = xyz - center[None, :]
        if use_rgb:
            sampled_colors = colors[idx].astype(np.float32)
            return (
                np.concatenate([xyz, sampled_colors], axis=1),
                np.concatenate([xyz_reference, sampled_colors], axis=1),
            )
        return xyz, xyz_reference

    def _invalid(self, msg: str) -> ExtractedObjectPCD:
        n = int(self.cfg.get("num_points", 1024))
        c = 6 if bool(self.cfg.get("use_rgb", False)) else 3
        return ExtractedObjectPCD(
            points=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            policy_points=np.zeros((n, c), dtype=np.float32),
            reference_points=np.zeros((n, c), dtype=np.float32),
            center=np.zeros(3, dtype=np.float32),
            valid=False,
            message=msg,
        )
