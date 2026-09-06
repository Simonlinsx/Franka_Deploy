from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .camera import PalmDepthStabilizer, estimate_palm_depth
from .model import CameraFrame, ManoDetection


DEFAULT_WILOR_ROOT = Path("/home/qiaoguanren/下载/WiLoR_OL/wilor_mini")


def detector_label_for_physical_hand(
    physical_handedness: str, input_mirrored: bool
) -> str:
    """Map a physical hand to the label in the image presented to WiLoR.

    The bundled detector explicitly names class 0 ``left`` and class 1
    ``right``.  Those labels match physical handedness for an unmirrored
    camera image.  Horizontally mirroring an image swaps only the detector
    label; it must not change the physical-hand convention used later by MANO
    canonicalization and robot retargeting.
    """

    handedness = physical_handedness.lower()
    if handedness not in ("right", "left"):
        raise ValueError("physical_handedness must be right or left")
    if not isinstance(input_mirrored, bool):
        raise TypeError("input_mirrored must be a bool")
    physical_is_right = handedness == "right"
    detector_is_right = physical_is_right != input_mirrored
    return "right" if detector_is_right else "left"


def _extract_mano_faces(pipeline, vertex_count: int = 778) -> Optional[np.ndarray]:
    """Return a validated MANO triangle table from the loaded WiLoR model."""

    model = getattr(pipeline, "wilor_model", None)
    mano = getattr(model, "mano", None)
    for attribute in ("faces", "faces_tensor"):
        value = getattr(mano, attribute, None)
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        faces = np.asarray(value)
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
            continue
        if not np.all(np.isfinite(faces)):
            continue
        rounded = np.rint(faces)
        if not np.array_equal(faces, rounded):
            continue
        faces = rounded.astype(np.int32, copy=True)
        if int(faces.min()) < 0 or int(faces.max()) >= vertex_count:
            continue
        faces.setflags(write=False)
        return faces
    return None


def project_mano_vertices(
    vertices: np.ndarray,
    translation,
    focal_length,
    image_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Project MANO vertices with WiLoR's full-frame perspective camera."""

    points = np.asarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        return None
    try:
        camera_translation = np.asarray(translation, dtype=np.float32).reshape(-1)
        focal = np.asarray(focal_length, dtype=np.float32).reshape(-1)
        height, width = (int(image_shape[0]), int(image_shape[1]))
    except (TypeError, ValueError, IndexError):
        return None
    if camera_translation.size != 3 or focal.size not in (1, 2):
        return None
    if height <= 0 or width <= 0:
        return None
    if not np.all(np.isfinite(camera_translation)) or not np.all(np.isfinite(focal)):
        return None
    if np.any(focal <= 0):
        return None
    focal_xy = np.repeat(focal, 2) if focal.size == 1 else focal
    camera_points = points + camera_translation[None, :]
    depth = camera_points[:, 2]
    if not np.all(np.isfinite(camera_points)) or np.any(depth <= 1e-6):
        return None
    center = np.asarray((width / 2.0, height / 2.0), dtype=np.float32)
    projected = camera_points[:, :2] / depth[:, None]
    projected = projected * focal_xy[None, :] + center[None, :]
    if not np.all(np.isfinite(projected)):
        return None
    return projected.astype(np.float32, copy=False)


def _restore_physical_mano_coordinates(
    keypoints_3d: np.ndarray,
    vertices: np.ndarray,
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    *,
    input_mirrored: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Undo WiLoR's handedness conversion for an externally mirrored image.

    WiLoR flips a detector-class-0 crop before MANO inference, then reflects
    the predicted MANO x coordinates and the y/z rotation-vector components
    back to a left-hand result.  ``--mirror-input`` can make the *same
    physical hand* receive that opposite detector label.  In that case the
    final reflection describes the image handedness, not the physical hand,
    and must be undone before wrist canonicalization and retargeting.

    Image-space keypoints, bbox coordinates, and projected vertices are not
    handled here: they must remain in the (mirrored) frame coordinate system
    so the preview and depth lookup stay aligned with the displayed image.
    """

    if not input_mirrored:
        return keypoints_3d, vertices, global_orient, hand_pose

    physical_keypoints = np.array(keypoints_3d, dtype=np.float32, copy=True)
    physical_vertices = np.array(vertices, dtype=np.float32, copy=True)
    physical_global_orient = np.array(
        global_orient, dtype=np.float32, copy=True
    )
    physical_hand_pose = np.array(hand_pose, dtype=np.float32, copy=True)
    physical_keypoints[..., 0] *= -1.0
    physical_vertices[..., 0] *= -1.0
    physical_global_orient[..., 1:3] *= -1.0
    physical_hand_pose[..., 1:3] *= -1.0
    return (
        physical_keypoints,
        physical_vertices,
        physical_global_orient,
        physical_hand_pose,
    )


class WiLoRBackend:
    """Lazy WiLoR-mini wrapper which always consumes uint8 BGR images."""

    REQUIRED_WEIGHTS = (
        "mano_mean_params.npz",
        "MANO_RIGHT.pkl",
        "wilor_final.ckpt",
        "detector.pt",
    )

    def __init__(
        self,
        wilor_root: Path = DEFAULT_WILOR_ROOT,
        device: str = "auto",
        hand_confidence: float = 0.5,
        handedness: str = "right",
        input_mirrored: bool = False,
        strict_single_hand: bool = True,
        focal_length: float = 5000.0,
        operator_roi: Optional[Sequence[float]] = None,
        depth_hold_seconds: float = 0.12,
    ) -> None:
        self.wilor_root = Path(wilor_root).expanduser().resolve()
        self.hand_confidence = hand_confidence
        # ``handedness`` is the physical hand requested by the caller.  It is
        # deliberately independent from the detector label after any input
        # mirroring.
        self.physical_handedness = handedness.lower()
        # Backwards-compatible alias; detector_handedness below is the label
        # that must never be substituted for this physical convention.
        self.handedness = self.physical_handedness
        self.input_mirrored = input_mirrored
        self.strict_single_hand = strict_single_hand
        if operator_roi is None:
            self.operator_roi = None
        else:
            roi = tuple(float(value) for value in operator_roi)
            if (
                len(roi) != 4
                or not np.all(np.isfinite(roi))
                or not (0.0 <= roi[0] < roi[2] <= 1.0)
                or not (0.0 <= roi[1] < roi[3] <= 1.0)
            ):
                raise ValueError(
                    "operator_roi must be normalized x1,y1,x2,y2 within 0..1"
                )
            self.operator_roi = roi
        self._depth_stabilizer = PalmDepthStabilizer(
            max_hold_seconds=depth_hold_seconds
        )
        self.last_diagnostics = {
            "candidate_count": 0,
            "matching_candidate_count": 0,
            "detector_labels": [],
            "expected_detector_label": None,
            "result": "not_run",
        }
        if self.physical_handedness not in ("right", "left"):
            raise ValueError("handedness must be right or left")
        self.detector_handedness = detector_label_for_physical_hand(
            self.physical_handedness, self.input_mirrored
        )
        weights_dir = self.wilor_root / "pretrained_models"
        missing = [
            weights_dir / name
            for name in self.REQUIRED_WEIGHTS
            if not (weights_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "missing local WiLoR weights: " + ", ".join(map(str, missing))
            )

        package_parent = str(self.wilor_root.parent)
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)

        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        if device == "auto":
            selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = device
        self.device = torch.device(selected_device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._pipeline = WiLorHandPose3dEstimationPipeline(
            device=self.device,
            dtype=dtype,
            verbose=False,
            focal_length=float(focal_length),
            wilor_pretrained_dir=str(self.wilor_root),
        )
        self._mesh_faces = _extract_mano_faces(self._pipeline)

    def predict(self, frame: CameraFrame) -> Optional[ManoDetection]:
        # A few lightweight unit tests construct the backend without running
        # ``__init__``.  Keep that supported, but disable temporal holding for
        # those synthetic instances so tests cannot accidentally manufacture
        # depth evidence.
        depth_stabilizer = getattr(self, "_depth_stabilizer", None)
        if depth_stabilizer is None:
            depth_stabilizer = PalmDepthStabilizer(max_hold_seconds=0.0)
            self._depth_stabilizer = depth_stabilizer
        image = np.asarray(frame.color_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("WiLoR input must be uint8 HxWx3 BGR")
        outputs = self._pipeline.predict(
            np.ascontiguousarray(image),
            hand_conf=self.hand_confidence,
            rescale_factor=2.5,
        )
        self.last_diagnostics = self._diagnose_predictions(outputs, image.shape[:2])
        selected = self._select_prediction(outputs, image.shape[:2])
        if selected is None:
            depth_stabilizer.reset()
            return None
        want_right = self.physical_handedness == "right"
        pred = selected["wilor_preds"]
        image_frame_3d = np.asarray(
            pred["pred_keypoints_3d"][0], dtype=np.float32
        )
        keypoints_2d = np.asarray(pred["pred_keypoints_2d"][0], dtype=np.float32)
        global_orient = np.asarray(pred["global_orient"][0, 0], dtype=np.float32)
        hand_pose = np.asarray(pred["hand_pose"][0], dtype=np.float32)
        image_frame_vertices = np.asarray(
            pred["pred_vertices"][0], dtype=np.float32
        )
        if image_frame_3d.shape != (21, 3) or keypoints_2d.shape != (21, 2):
            self.last_diagnostics["result"] = "invalid_keypoint_shape"
            return None
        if not np.all(np.isfinite(image_frame_3d)) or not np.all(
            np.isfinite(keypoints_2d)
        ):
            self.last_diagnostics["result"] = "nonfinite_keypoints"
            return None

        # Projection must use WiLoR's image-frame geometry and camera before
        # restoring the physical MANO convention.  This keeps overlays aligned
        # with a statically mirrored source image.
        vertices_2d = project_mano_vertices(
            image_frame_vertices,
            pred.get("pred_cam_t_full"),
            pred.get("scaled_focal_length"),
            image.shape[:2],
        )
        raw_3d, vertices, global_orient, hand_pose = (
            _restore_physical_mano_coordinates(
                image_frame_3d,
                image_frame_vertices,
                global_orient,
                hand_pose,
                input_mirrored=getattr(self, "input_mirrored", False),
            )
        )

        # WiLoR includes global wrist rotation.  dex-retargeting expects the
        # wrist-local MANO frame, so remove translation and global orientation.
        # dex-retargeting's vector configs are calibrated against the wrist
        # frame produced by its official MediaPipe front-end.  WiLoR is a MANO
        # model, but its native MANO axes are not the operator axes expected by
        # that front-end; rebuild the same wrist frame before optimization.
        try:
            canonical = canonicalize_for_dex(raw_3d, is_right=want_right)
        except ValueError:
            self.last_diagnostics["result"] = "degenerate_canonical_geometry"
            return None
        palm_width = float(np.linalg.norm(canonical[5] - canonical[17]))
        if not 0.02 <= palm_width <= 0.20:
            self.last_diagnostics["result"] = "implausible_palm_width"
            return None
        mesh_faces = getattr(self, "_mesh_faces", None)
        if (
            vertices_2d is None
            or mesh_faces is None
            or mesh_faces.size == 0
            or int(mesh_faces.max()) >= len(vertices)
        ):
            mesh_faces = None
        bbox = np.asarray(selected["hand_bbox"], dtype=np.float32)
        spatial_depth = estimate_palm_depth(frame.depth_m, keypoints_2d)
        depth = depth_stabilizer.apply(
            spatial_depth,
            bbox,
            frame.captured_at_monotonic,
        )
        self.last_diagnostics["palm_depth"] = {
            "source": depth.source,
            "reason": depth.reason,
            "raw_depth_m": depth.raw_depth_m,
            "effective_depth_m": depth.depth_m,
            "age_seconds": depth.age_seconds,
            "roi_sample_count": depth.roi_sample_count,
            "inlier_count": depth.inlier_count,
            "valid_pixel_count": depth.valid_pixel_count,
            "radius_px": depth.radius_px,
        }
        self.last_diagnostics["result"] = "accepted"
        return ManoDetection(
            is_right=want_right,
            bbox_xyxy=bbox,
            keypoints_3d_raw=raw_3d,
            keypoints_3d_canonical=canonical.astype(np.float32),
            keypoints_2d=keypoints_2d,
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=np.asarray(pred["betas"][0], dtype=np.float32),
            vertices=vertices,
            captured_at_monotonic=frame.captured_at_monotonic,
            frame_number=frame.frame_number,
            palm_depth_m=depth.raw_depth_m,
            control_palm_depth_m=depth.depth_m,
            palm_depth_source=depth.source,
            palm_depth_reason=depth.reason,
            palm_depth_evidence_at_monotonic=depth.evidence_at_monotonic,
            palm_depth_age_seconds=depth.age_seconds,
            palm_depth_roi_sample_count=depth.roi_sample_count,
            palm_depth_inlier_count=depth.inlier_count,
            palm_depth_valid_pixel_count=depth.valid_pixel_count,
            palm_depth_radius_px=depth.radius_px,
            vertices_2d=vertices_2d,
            mesh_faces=mesh_faces,
        )

    def _diagnose_predictions(
        self,
        outputs: List[dict],
        image_shape: Optional[tuple[int, int]] = None,
    ) -> dict:
        predictions = [item for item in outputs if "wilor_preds" in item]
        roi_predictions = [
            item
            for item in predictions
            if self._prediction_is_in_operator_roi(item, image_shape)
        ]
        distinct = self._deduplicate_predictions(roi_predictions)
        detector_handedness = getattr(
            self, "detector_handedness", self.handedness
        )
        detector_wants_right = detector_handedness == "right"
        labels = [
            "right" if float(item.get("is_right", 0.0)) > 0.5 else "left"
            for item in predictions
        ]
        matching_count = sum(
            (float(item.get("is_right", 0.0)) > 0.5) == detector_wants_right
            for item in distinct
        )
        if matching_count == 0:
            if not predictions:
                result = "no_candidates"
            elif not distinct:
                result = "outside_operator_roi"
            else:
                result = "handedness_mismatch"
        elif self.strict_single_hand and len(distinct) != 1:
            result = "multiple_hands"
        else:
            result = "candidate_selected"
        return {
            "candidate_count": len(predictions),
            "candidate_bboxes": [
                list(map(float, item["hand_bbox"])) for item in predictions
            ],
            "candidate_in_operator_roi": [
                self._prediction_is_in_operator_roi(item, image_shape)
                for item in predictions
            ],
            "roi_candidate_count": len(roi_predictions),
            "distinct_candidate_count": len(distinct),
            "duplicate_candidate_count": len(roi_predictions) - len(distinct),
            "matching_candidate_count": matching_count,
            "detector_labels": labels,
            "expected_detector_label": detector_handedness,
            "operator_roi": (
                list(self.operator_roi)
                if getattr(self, "operator_roi", None) is not None
                else None
            ),
            "result": result,
        }

    def _select_prediction(
        self,
        outputs: List[dict],
        image_shape: Optional[tuple[int, int]] = None,
    ) -> Optional[dict]:
        predictions = [item for item in outputs if "wilor_preds" in item]
        predictions = self._deduplicate_predictions(
            [
                item
                for item in predictions
                if self._prediction_is_in_operator_roi(item, image_shape)
            ]
        )
        # The fallback preserves compatibility for tests or callers which
        # construct an uninitialized backend via ``object.__new__``.  Normal
        # instances always set detector_handedness in __init__.
        detector_handedness = getattr(
            self, "detector_handedness", self.handedness
        )
        detector_wants_right = detector_handedness == "right"
        matching: List[dict] = [
            item
            for item in predictions
            if (float(item.get("is_right", 0.0)) > 0.5)
            == detector_wants_right
        ]
        if not matching:
            return None
        if self.strict_single_hand and len(predictions) != 1:
            return None
        return max(matching, key=self._bbox_area)

    def _prediction_is_in_operator_roi(
        self,
        item: dict,
        image_shape: Optional[tuple[int, int]],
    ) -> bool:
        roi = getattr(self, "operator_roi", None)
        if roi is None or image_shape is None:
            return True
        height, width = map(float, image_shape)
        if height <= 0.0 or width <= 0.0:
            return False
        try:
            x1, y1, x2, y2 = map(float, item["hand_bbox"])
        except (KeyError, TypeError, ValueError):
            return False
        center_x = 0.5 * (x1 + x2) / width
        center_y = 0.5 * (y1 + y2) / height
        return roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]

    @classmethod
    def _deduplicate_predictions(cls, predictions: List[dict]) -> List[dict]:
        """Suppress only near-identical boxes; separated hands remain distinct."""

        kept: List[dict] = []
        for item in sorted(predictions, key=cls._bbox_area, reverse=True):
            label = float(item.get("is_right", 0.0)) > 0.5
            if any(
                label == (float(existing.get("is_right", 0.0)) > 0.5)
                and cls._bbox_iou(item, existing) >= 0.85
                for existing in kept
            ):
                continue
            kept.append(item)
        return kept

    @staticmethod
    def _bbox_iou(first: dict, second: dict) -> float:
        try:
            ax1, ay1, ax2, ay2 = map(float, first["hand_bbox"])
            bx1, by1, bx2, by2 = map(float, second["hand_bbox"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0

    @staticmethod
    def _bbox_area(item: dict) -> float:
        x1, y1, x2, y2 = map(float, item["hand_bbox"])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def canonicalize_mano(
    keypoints_3d: np.ndarray, global_orient: np.ndarray
) -> np.ndarray:
    """Remove WiLoR's root translation and MANO global orientation."""

    import cv2

    joints = np.asarray(keypoints_3d, dtype=np.float32)
    orientation = np.asarray(global_orient, dtype=np.float32).reshape(3)
    if joints.shape != (21, 3):
        raise ValueError(f"expected keypoints shape (21, 3), got {joints.shape}")
    if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(orientation)):
        raise ValueError("MANO keypoints and orientation must be finite")
    global_rotation, _ = cv2.Rodrigues(orientation)
    return ((joints - joints[0]) @ global_rotation).astype(np.float32)


def canonicalize_for_dex(
    keypoints_3d: np.ndarray, is_right: bool = True
) -> np.ndarray:
    """Match dex-retargeting's official wrist-local MANO convention."""

    joints = np.asarray(keypoints_3d, dtype=np.float32)
    if joints.shape != (21, 3) or not np.all(np.isfinite(joints)):
        raise ValueError("expected finite MANO keypoints with shape (21, 3)")
    joints = joints - joints[0]
    points = joints[[0, 5, 9]]
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered)
    if (
        singular_values[0] < 1e-7
        or singular_values[1] / singular_values[0] < 1e-3
    ):
        raise ValueError("degenerate wrist/index/middle geometry")
    normal = vh[2]
    x_vector = points[0] - points[2]
    x_axis = x_vector - np.dot(x_vector, normal) * normal
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-7:
        raise ValueError("cannot estimate wrist x axis")
    x_axis /= x_norm
    z_axis = np.cross(x_axis, normal)
    if np.dot(z_axis, centered[1] - centered[2]) < 0:
        normal *= -1
        z_axis *= -1
    wrist_frame = np.stack([x_axis, normal, z_axis], axis=1)
    operator_to_mano = np.asarray(
        (
            ((0, 0, -1), (-1, 0, 0), (0, 1, 0))
            if is_right
            else ((0, 0, -1), (1, 0, 0), (0, -1, 0))
        ),
        dtype=np.float32,
    )
    return (joints @ wrist_frame @ operator_to_mano).astype(np.float32)
