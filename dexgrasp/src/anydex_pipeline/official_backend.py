"""Lazy, robot-free wrapper around AnyDexGrasp's representation network.

The upstream entry point (``robot_inspire.py``) parses robot CLI arguments at
import time and constructs its point cloud from a hard-coded depth camera.  This
module deliberately does neither: it accepts an already segmented point cloud
and imports PyTorch, MinkowskiEngine, and the vendored model only when a backend
is loaded.

Coordinate contract
-------------------
``infer`` accepts XYZ points in metres in ``camera_color_optical_frame``
(RealSense optical convention: +x right, +y down, +z forward).  Every returned
translation and rotation is in that same frame.  In a returned rotation matrix,
column 0 is the grasp approach axis, column 1 is the finger opening/closing axis,
and column 2 is the gripper height axis.  No camera-to-robot transform is applied
here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np


CAMERA_FRAME = "camera_color_optical_frame"
GRASPNET_ARRAY_LEN = 17
OFFICIAL_NO_FLIP_FEATURE_WIDTH = 2261


class RepresentationFeatureLayout(str, Enum):
    """Versioned layout of the representation feature matrix.

    The robot-free bridge emits the official feature vector before the
    upstream augmentation loop appends its final ``if_flip`` scalar.  Keeping
    that distinction explicit prevents the decision heads from silently
    reading the seed id as an angle class.
    """

    OFFICIAL_RIGHT_HAND_NO_FLIP_V1 = "official_right_hand_no_flip_v1"


class OfficialBackendError(RuntimeError):
    """Base error raised by the vendored AnyDexGrasp backend."""


class OfficialBackendUnavailable(OfficialBackendError):
    """Raised when an optional native/model dependency cannot be loaded."""


class OfficialCheckpointError(OfficialBackendError):
    """Raised when a representation checkpoint cannot be loaded safely."""


def _official_right_hand_orientation_mask(predictions: Any) -> Any:
    """Return the released Inspire filter ``R[1, 1] >= 0``.

    ``predictions`` uses the internal 15-value layout
    ``score,width,depth,R(9),translation(3)``.  The function intentionally
    works for both NumPy and torch arrays so the exact upstream convention can
    be unit-tested without loading CUDA.
    """

    if predictions.ndim != 2 or predictions.shape[1] != 15:
        raise ValueError("representation predictions must have shape (N, 15)")
    rotations = predictions[:, 3:12].reshape(-1, 3, 3)
    return rotations[:, 1, 1] >= 0


@dataclass(frozen=True)
class RepresentationGraspBatch:
    """Representation-network results in the camera optical frame.

    ``grasps`` uses the standard GraspNet 17-value layout::

        score, width, height, depth, rotation(9 row-major),
        translation(3), object_id

    ``features`` is the upstream per-candidate feature vector used by the
    optional Inspire decision network.  It remains aligned with ``grasps``.
    """

    grasps: np.ndarray
    features: np.ndarray
    voxel_points_camera: np.ndarray
    feature_layout: RepresentationFeatureLayout
    frame_id: str = CAMERA_FRAME

    def __post_init__(self) -> None:
        grasps = np.asarray(self.grasps)
        features = np.asarray(self.features)
        points = np.asarray(self.voxel_points_camera)
        if grasps.ndim != 2 or grasps.shape[1] != GRASPNET_ARRAY_LEN:
            raise ValueError(f"grasps must have shape (N, {GRASPNET_ARRAY_LEN})")
        if features.ndim != 2 or features.shape[0] != grasps.shape[0]:
            raise ValueError("features must have shape (N, F) and align with grasps")
        if (
            self.feature_layout
            is not RepresentationFeatureLayout.OFFICIAL_RIGHT_HAND_NO_FLIP_V1
        ):
            raise ValueError(
                "unsupported representation feature layout: "
                f"{self.feature_layout!r}"
            )
        if features.shape[1] != OFFICIAL_NO_FLIP_FEATURE_WIDTH:
            raise ValueError(
                "official_right_hand_no_flip_v1 features must have shape "
                f"(N, {OFFICIAL_NO_FLIP_FEATURE_WIDTH})"
            )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("voxel_points_camera must have shape (M, 3)")

    def __len__(self) -> int:
        return int(self.grasps.shape[0])

    @property
    def scores(self) -> np.ndarray:
        return self.grasps[:, 0]

    @property
    def widths(self) -> np.ndarray:
        return self.grasps[:, 1]

    @property
    def depths(self) -> np.ndarray:
        return self.grasps[:, 3]

    @property
    def rotation_matrices(self) -> np.ndarray:
        return self.grasps[:, 4:13].reshape(-1, 3, 3)

    @property
    def translations(self) -> np.ndarray:
        return self.grasps[:, 13:16]


def default_anydex_repo() -> Path:
    """Return the expected location of the vendored official checkout."""

    return Path(__file__).resolve().parents[2] / "third_party" / "AnyDexGrasp"


class OfficialRepresentationBackend:
    """Run the official AnyDexGrasp representation model on an XYZ point cloud.

    Parameters mirror the official Inspire robot script where useful.  The
    upstream camera-specific x/y workspace mask is intentionally not enabled by
    default because the caller has already supplied an object point cloud.  The
    official approach-direction filter (camera-z component > 0.92) is retained
    by default and can be disabled with ``min_approach_camera_z=None``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        anydex_repo: str | Path | None = None,
        device: str | None = None,
        use_graspnet_v2: bool = True,
        half_views: bool = False,
        voxel_size: float = 0.005,
        min_width: float = 0.01,
        max_width: float = 0.10,
        min_approach_camera_z: float | None = 0.92,
        workspace_camera_xyz: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ]
        | None = None,
        allow_unsafe_checkpoint: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.anydex_repo = Path(anydex_repo or default_anydex_repo()).expanduser().resolve()
        self.requested_device = device
        self.use_graspnet_v2 = bool(use_graspnet_v2)
        self.half_views = bool(half_views)
        self.voxel_size = float(voxel_size)
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.min_approach_camera_z = min_approach_camera_z
        self.workspace_camera_xyz = workspace_camera_xyz
        self.allow_unsafe_checkpoint = bool(allow_unsafe_checkpoint)

        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if not 0 <= self.min_width < self.max_width:
            raise ValueError("width limits must satisfy 0 <= min_width < max_width")
        if workspace_camera_xyz is not None:
            if len(workspace_camera_xyz) != 3 or any(lo >= hi for lo, hi in workspace_camera_xyz):
                raise ValueError("workspace_camera_xyz must contain increasing x/y/z bounds")

        self._torch: Any | None = None
        self._me: Any | None = None
        self._viewpoint_to_matrix: Any | None = None
        self._net: Any | None = None
        self._device: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._net is not None

    @property
    def device(self) -> str | None:
        return None if self._device is None else str(self._device)

    def _add_official_import_paths(self) -> None:
        if not self.anydex_repo.is_dir():
            raise OfficialBackendUnavailable(
                f"AnyDexGrasp checkout not found: {self.anydex_repo}"
            )
        # The official sources use bare imports such as ``from resunet import``.
        # Prepending these directories reproduces its CLI import environment
        # without importing robot_inspire.py (which parses argparse at import).
        paths = (
            self.anydex_repo / "pointnet2",
            self.anydex_repo / "utils",
            self.anydex_repo / "models",
            self.anydex_repo,
        )
        for path in paths:
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

    def load(self) -> "OfficialRepresentationBackend":
        """Load optional native dependencies and the trusted model state dict."""

        if self.loaded:
            return self
        if not self.checkpoint_path.is_file():
            raise OfficialCheckpointError(
                f"representation checkpoint not found: {self.checkpoint_path}"
            )

        self._add_official_import_paths()
        try:
            torch = importlib.import_module("torch")
            me = importlib.import_module("MinkowskiEngine")
            model_module = importlib.import_module(
                "models.minkowski_graspnet_single_point"
            )
            pt_utils = importlib.import_module("pt_utils")
        except Exception as exc:  # Native import failures are often not ImportError.
            raise OfficialBackendUnavailable(
                "Could not import the official AnyDexGrasp inference stack. "
                "Install its pinned PyTorch, MinkowskiEngine, pointnet2, and knn dependencies."
            ) from exc

        device_name = self.requested_device
        if device_name is None:
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)

        num_depth = 5 if self.use_graspnet_v2 else 4
        net = model_module.MinkowskiGraspNet(
            num_depth=num_depth,
            num_seed=2048,
            is_training=False,
            half_views=self.half_views,
        )

        load_kwargs = {"map_location": "cpu"}
        try:
            checkpoint = torch.load(
                str(self.checkpoint_path), weights_only=True, **load_kwargs
            )
        except TypeError as exc:
            if not self.allow_unsafe_checkpoint:
                raise OfficialCheckpointError(
                    "This PyTorch version does not support safe weights_only loading. "
                    "Upgrade PyTorch or explicitly set allow_unsafe_checkpoint=True "
                    "only for a checkpoint you trust."
                ) from exc
            checkpoint = torch.load(str(self.checkpoint_path), **load_kwargs)
        except Exception as exc:
            if not self.allow_unsafe_checkpoint:
                raise OfficialCheckpointError(
                    "Safe checkpoint loading failed. Do not enable unsafe loading unless "
                    "the checkpoint is fully trusted."
                ) from exc
            checkpoint = torch.load(str(self.checkpoint_path), **load_kwargs)

        if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, Mapping):
            state_dict = checkpoint
        else:
            raise OfficialCheckpointError(
                "representation checkpoint must be a state dict or contain model_state_dict"
            )

        try:
            net.load_state_dict(state_dict)
            net.to(device)
            net.eval()
        except Exception as exc:
            raise OfficialCheckpointError(
                "checkpoint is incompatible with the configured official representation model"
            ) from exc

        self._torch = torch
        self._me = me
        self._viewpoint_to_matrix = pt_utils.batch_viewpoint_params_to_matrix
        self._net = net
        self._device = device
        return self

    def infer(
        self,
        points_camera: np.ndarray,
        *,
        score_threshold: float | None = None,
        max_grasps: int | None = None,
        sort_by_score: bool = True,
    ) -> RepresentationGraspBatch:
        """Infer two-finger representation grasps from camera-frame XYZ points.

        Args:
            points_camera: ``(N, 3)`` XYZ in metres in
                ``camera_color_optical_frame``.  Passing robot-base points here
                is incorrect because the representation network and approach
                filter are camera-frame specific.
            score_threshold: Optional lower bound on representation score.
            max_grasps: Optional number of candidates retained after filtering.
            sort_by_score: Sort descending before applying ``max_grasps``.

        Returns:
            A NumPy-only batch.  Its poses remain in the camera optical frame.
        """

        if not self.loaded:
            self.load()
        if max_grasps is not None and max_grasps < 1:
            raise ValueError("max_grasps must be positive when supplied")

        points = np.asarray(points_camera, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_camera must have shape (N, 3)")
        points = np.ascontiguousarray(points[np.isfinite(points).all(axis=1)])
        if points.shape[0] == 0:
            raise ValueError("points_camera contains no finite points")

        torch = self._torch
        me = self._me
        assert torch is not None and me is not None and self._net is not None

        # Keep the upstream integer truncation convention for checkpoint parity.
        coords = np.ascontiguousarray(points / self.voxel_size, dtype=np.int32)
        quantized = me.utils.sparse_quantize(coords, return_index=True)
        if isinstance(quantized, tuple):
            unique_indices = quantized[-1]
        else:  # Compatibility with older MinkowskiEngine return conventions.
            unique_indices = quantized
        if hasattr(unique_indices, "detach"):
            unique_indices = unique_indices.detach().cpu().numpy()
        unique_indices = np.asarray(unique_indices, dtype=np.int64).reshape(-1)
        if unique_indices.size == 0:
            raise ValueError("point cloud is empty after sparse voxel quantization")

        coords = coords[unique_indices]
        voxel_points = np.ascontiguousarray(points[unique_indices])
        point_features = torch.from_numpy(voxel_points)
        coords_batch, features_batch = me.utils.sparse_collate(
            [coords], [point_features]
        )
        sinput = me.SparseTensor(
            features_batch, coords_batch, device=self._device
        )
        end_points = {"sinput": sinput, "point_clouds": [sinput.F]}
        with torch.no_grad():
            end_points = self._net(end_points)
            predictions, features = self._parse_predictions(end_points)

        if predictions is None:
            return RepresentationGraspBatch(
                np.empty((0, GRASPNET_ARRAY_LEN), dtype=np.float32),
                np.empty((0, OFFICIAL_NO_FLIP_FEATURE_WIDTH), dtype=np.float32),
                voxel_points,
                RepresentationFeatureLayout.OFFICIAL_RIGHT_HAND_NO_FLIP_V1,
            )

        mask = (predictions[:, 1] > self.min_width) & (
            predictions[:, 1] < self.max_width
        )
        if self.min_approach_camera_z is not None:
            # predictions[:, 3:12] is row-major R; index 9 is R[2, 0].
            mask &= predictions[:, 9] > float(self.min_approach_camera_z)
        if score_threshold is not None:
            mask &= predictions[:, 0] >= float(score_threshold)
        if self.workspace_camera_xyz is not None:
            translations = predictions[:, 12:15]
            for axis, (lower, upper) in enumerate(self.workspace_camera_xyz):
                mask &= (translations[:, axis] > lower) & (
                    translations[:, axis] < upper
                )
        # Reproduce the released Inspire right-hand path exactly.  Upstream
        # ``flip_ggarray`` marks rotations with R[1, 1] < 0, appends an
        # ``if_flip`` feature, and then discards every marked candidate before
        # the decision heads.  Our robot-free path has no augmentation/flip
        # row, so it performs that same rejection directly.
        mask &= _official_right_hand_orientation_mask(predictions)

        predictions = predictions[mask]
        features = features[mask]

        heights = torch.full(
            (predictions.shape[0], 1), 0.03, dtype=predictions.dtype,
            device=predictions.device,
        )
        object_ids = torch.full_like(heights, -1.0)
        # prediction layout: score, width, depth, R(9), t(3)
        graspnet = torch.cat(
            [predictions[:, :2], heights, predictions[:, 2:], object_ids], dim=1
        )
        graspnet_np = graspnet.detach().cpu().numpy().astype(np.float32, copy=False)
        features_np = features.detach().cpu().numpy().astype(np.float32, copy=False)

        if sort_by_score and graspnet_np.shape[0]:
            order = np.argsort(-graspnet_np[:, 0], kind="stable")
            graspnet_np = graspnet_np[order]
            features_np = features_np[order]
        if max_grasps is not None:
            graspnet_np = graspnet_np[:max_grasps]
            features_np = features_np[:max_grasps]

        return RepresentationGraspBatch(
            grasps=graspnet_np,
            features=features_np,
            voxel_points_camera=voxel_points,
            feature_layout=(
                RepresentationFeatureLayout.OFFICIAL_RIGHT_HAND_NO_FLIP_V1
            ),
        )

    def _parse_predictions(self, end_points: Mapping[str, Any]) -> tuple[Any | None, Any | None]:
        """Port of robot_inspire.parse_preds without importing the robot CLI."""

        torch = self._torch
        viewpoint_to_matrix = self._viewpoint_to_matrix
        assert torch is not None and viewpoint_to_matrix is not None

        before_generator = end_points["before_generator"]
        point_features = end_points["point_features"]
        coords = end_points["sinput"].C
        objectness = end_points["stage1_objectness_pred"]
        objectness_mask = torch.argmax(objectness, dim=1).bool()
        seed_xyz = end_points["stage2_seed_xyz"]
        seed_inds = end_points["stage2_seed_inds"]
        view_xyz = end_points["stage2_view_xyz"]
        view_inds = end_points["stage2_view_inds"]
        view_scores = end_points["stage2_view_scores"]
        grasp_scores = end_points["stage3_grasp_scores"]
        generator_features = end_points["stage3_grasp_features"].reshape(
            grasp_scores.shape[0], grasp_scores.shape[1], -1
        )
        grasp_widths = self.max_width * end_points[
            "stage3_normalized_grasp_widths"
        ]
        grasp_widths = torch.clamp(grasp_widths, max=self.max_width)

        prediction_batches = []
        feature_batches = []
        for batch_index in range(seed_xyz.shape[0]):
            cloud_mask = coords[:, 0] == batch_index
            selected_seed_inds = seed_inds[batch_index]
            selected_objectness = objectness_mask[cloud_mask][selected_seed_inds]
            if not bool(selected_objectness.any().item()):
                continue

            # This intentionally matches the released inference: it checks that
            # at least one seed is object-like but does not mask individual seeds.
            xyz = seed_xyz[batch_index]
            point_feature = point_features[batch_index]
            before = before_generator[batch_index]
            directions = view_xyz[batch_index]
            direction_inds = view_inds[batch_index]
            direction_scores = view_scores[batch_index]
            scores = grasp_scores[batch_index]
            widths = grasp_widths[batch_index]
            raw_scores = scores.reshape(scores.shape[0], -1).clone()

            num_seeds = scores.shape[0]
            # Pair antipodal 48-angle predictions into 24 physical angles.
            paired_scores = torch.minimum(scores[:, :24, :], scores[:, 24:, :])
            paired_scores, angle_classes = torch.max(paired_scores, dim=1)
            angles = (angle_classes.float() - 12.0) / 24.0 * np.pi

            angle_index = angle_classes.unsqueeze(1)
            positive_width = torch.gather(widths, 1, angle_index).squeeze(1)
            negative_width = torch.gather(widths, 1, angle_index + 24).squeeze(1)

            best_scores, depth_classes = torch.max(
                paired_scores, dim=1, keepdim=True
            )
            depths = (depth_classes.float() + 1.0) * 0.01
            depths -= 0.01
            depths[depth_classes == 0] = 0.005
            angles = torch.gather(angles, 1, depth_classes)
            positive_width = torch.gather(positive_width, 1, depth_classes)
            negative_width = torch.gather(negative_width, 1, depth_classes)

            rotations = viewpoint_to_matrix(
                -directions, angles.squeeze(1)
            ).reshape(num_seeds, 9)
            combined_width = positive_width + negative_width
            predictions = torch.cat(
                [best_scores, combined_width, depths, rotations, xyz], dim=1
            )

            feature_vector = torch.cat(
                [
                    raw_scores,
                    generator_features[batch_index],
                    before,
                    point_feature,
                    direction_inds.reshape(num_seeds, 1),
                    direction_scores.reshape(num_seeds, 1),
                    selected_seed_inds.reshape(num_seeds, 1),
                    angles * 24.0 / np.pi + 12.0,
                    depths,
                ],
                dim=1,
            )
            prediction_batches.append(predictions)
            feature_batches.append(feature_vector)

        if not prediction_batches:
            return None, None
        return torch.cat(prediction_batches, dim=0), torch.cat(feature_batches, dim=0)


def infer_representation_grasps(
    points_camera: np.ndarray,
    checkpoint_path: str | Path,
    **backend_kwargs: Any,
) -> RepresentationGraspBatch:
    """One-shot convenience wrapper; prefer a persistent backend for live use."""

    infer_keys = {"score_threshold", "max_grasps", "sort_by_score"}
    infer_kwargs = {
        key: backend_kwargs.pop(key) for key in tuple(backend_kwargs) if key in infer_keys
    }
    backend = OfficialRepresentationBackend(checkpoint_path, **backend_kwargs)
    return backend.infer(points_camera, **infer_kwargs)
