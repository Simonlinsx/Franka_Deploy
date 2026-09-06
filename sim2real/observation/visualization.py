"""Lossy, out-of-band visualization of the exact V94 policy point input.

The policy thread only replaces one Python reference and signals a bridge
thread.  Image copies, IPC serialization, Tk/Pillow, and Open3D all stay off the
60 Hz observation/action path.  Visualization is deliberately best-effort:
after its startup handshake, a closed or failed viewer can never fault the
robot-control ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


class V94LiveVisualizationError(RuntimeError):
    """The explicitly requested viewer could not be started."""


@dataclass(frozen=True)
class V94LiveVisualizationSample:
    """One exact-frame visualization candidate produced by the policy owner."""

    sequence: int
    frame_id: int
    captured_realtime_s: float
    color_bgr: np.ndarray
    object_mask: np.ndarray
    pointcloud_xyzrgb_palm: np.ndarray
    pointcloud_valid: np.ndarray
    T_base_palm_at_capture: np.ndarray
    source_valid_points: int
    point_coordinate_frame: str = "policy_palm"
    requested_object_mask_mode: str = "unknown"
    effective_object_mask_mode: str = "unknown"
    effective_provider_mask_publication_mode: str = "unknown"
    effective_provider_recovery_publication_mode: str = "unknown"
    provider_published_mask_valid: Optional[bool] = None
    provider_published_mask_area_px: Optional[int] = None
    provider_published_mask_bbox_xyxy: Optional[
        tuple[int, int, int, int]
    ] = None
    provider_published_mask_source: str = "unknown"
    provider_published_mask_message: str = ""
    provider_online_sam2_status: str = "unknown"
    projector_effective_policy_mask_provenance: str = "unknown"
    projector_effective_policy_mask_source_frame_id: Optional[int] = None
    projector_effective_policy_mask_source_captured_realtime_s: Optional[
        float
    ] = None
    projector_effective_policy_mask_area_px: Optional[int] = None
    projector_effective_policy_mask_bbox_xyxy: Optional[
        tuple[int, int, int, int]
    ] = None
    # Deprecated constructor aliases retained for callers created before the
    # provider/projector provenance split.  They are normalized immediately;
    # artifacts never emit these ambiguous names.
    mask_source: Optional[str] = None
    mask_message: Optional[str] = None
    online_sam2_status: Optional[str] = None

    def __post_init__(self) -> None:
        coordinate_frame = str(self.point_coordinate_frame)
        if coordinate_frame not in (
            "policy_palm",
            "robot_base_via_identity_T_base_palm",
        ):
            raise ValueError(
                "point_coordinate_frame must be policy_palm or "
                "robot_base_via_identity_T_base_palm"
            )
        aliases = (
            (
                "mask_source",
                self.mask_source,
                "provider_published_mask_source",
                self.provider_published_mask_source,
                "unknown",
            ),
            (
                "mask_message",
                self.mask_message,
                "provider_published_mask_message",
                self.provider_published_mask_message,
                "",
            ),
            (
                "online_sam2_status",
                self.online_sam2_status,
                "provider_online_sam2_status",
                self.provider_online_sam2_status,
                "unknown",
            ),
        )
        for alias_name, alias, explicit_name, explicit, default in aliases:
            if alias is None:
                continue
            alias_text = str(alias)
            explicit_text = str(explicit)
            if explicit_text != default and explicit_text != alias_text:
                raise ValueError(
                    f"{alias_name} conflicts with {explicit_name}"
                )
            object.__setattr__(self, explicit_name, alias_text)


@runtime_checkable
class V94LiveVisualizationSink(Protocol):
    """Non-blocking sink used by ``PersistentV94ObservationOwner``."""

    def try_publish(self, sample: V94LiveVisualizationSample) -> bool: ...


@dataclass(frozen=True)
class _MaskViewerPayload:
    sequence: int
    frame_id: int
    color_bgr: np.ndarray
    object_mask: np.ndarray
    source_valid_points: int
    final_policy_points: int


@dataclass(frozen=True)
class _CloudViewerPayload:
    sequence: int
    frame_id: int
    pointcloud_xyzrgb_palm: np.ndarray
    pointcloud_valid: np.ndarray
    T_base_palm_at_capture: np.ndarray


def _policy_cloud_geometry_and_colors(
    pointcloud: np.ndarray, validity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return visible palm-frame XYZ and display-only RGB for either mode."""

    points = np.asarray(pointcloud, dtype=np.float64)
    valid = np.asarray(validity) > 0.5
    if (
        points.ndim != 2
        or points.shape[1] not in (3, 6)
        or valid.shape != (points.shape[0],)
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("visualized policy cloud must be finite XYZ or XYZRGB")
    xyz_palm = points[valid, :3]
    if points.shape[1] == 6:
        colors = np.clip(points[valid, 3:6], 0.0, 1.0)
    else:
        # XYZ checkpoints intentionally discard image color.  A fixed cyan
        # tint keeps geometry visible without implying RGB reached policy.
        colors = np.tile(
            np.asarray([[0.15, 0.75, 1.0]], dtype=np.float64),
            (xyz_palm.shape[0], 1),
        )
    return xyz_palm, colors


def _set_current_thread_affinity(cpus: Sequence[int]) -> None:
    selected = tuple(int(value) for value in cpus)
    if not selected:
        return
    os.sched_setaffinity(0, set(selected))


def _constrain_all_current_threads(cpus: Sequence[int]) -> None:
    selected = set(int(value) for value in cpus)
    if not selected:
        return
    for entry in os.scandir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(entry.name), selected)
        except (FileNotFoundError, ProcessLookupError):
            continue


def _mask_overlay_rgb(
    payload: _MaskViewerPayload,
) -> tuple[np.ndarray, tuple[str, ...]]:
    image = np.ascontiguousarray(payload.color_bgr[..., ::-1]).copy()
    mask = np.asarray(payload.object_mask, dtype=bool)
    if np.any(mask):
        source = image[mask].astype(np.float32)
        green = np.asarray([0.0, 255.0, 0.0], dtype=np.float32)
        image[mask] = np.clip(source * 0.45 + green * 0.55, 0.0, 255.0).astype(np.uint8)
    lines = (
        f"policy candidate seq={payload.sequence} camera_frame={payload.frame_id}",
        f"mask_px={int(np.count_nonzero(mask))} source_points={payload.source_valid_points}",
        f"final_policy_points={payload.final_policy_points}/128  (Q/Esc closes mask only)",
    )
    return image, lines


def _mask_viewer_frame_bgr(payload: _MaskViewerPayload) -> np.ndarray:
    """Return the live-view camera frame with a compact binary-mask inset."""

    frame = np.ascontiguousarray(payload.color_bgr, dtype=np.uint8).copy()
    mask = np.asarray(payload.object_mask, dtype=bool)
    if frame.ndim != 3 or frame.shape[2] != 3 or mask.shape != frame.shape[:2]:
        raise ValueError("recording mask/image shapes differ")
    height, width = frame.shape[:2]
    inset_width = min(max(120, int(round(width * 0.28))), max(1, width - 24))
    inset_height = min(
        max(68, int(round(inset_width * height / max(1, width)))),
        max(1, height - 24),
    )
    y_index = np.minimum(
        (np.arange(inset_height, dtype=np.int64) * height) // inset_height,
        height - 1,
    )
    x_index = np.minimum(
        (np.arange(inset_width, dtype=np.int64) * width) // inset_width,
        width - 1,
    )
    resized = mask[y_index[:, None], x_index[None, :]]
    inset = np.repeat((resized.astype(np.uint8) * 255)[..., None], 3, axis=2)
    x0 = width - inset_width - 12
    y0 = 12
    frame[y0 : y0 + inset_height, x0 : x0 + inset_width] = inset
    border = min(3, inset_height, inset_width)
    green = np.asarray([0, 255, 0], dtype=np.uint8)
    frame[y0 : y0 + border, x0 : x0 + inset_width] = green
    frame[y0 + inset_height - border : y0 + inset_height, x0 : x0 + inset_width] = green
    frame[y0 : y0 + inset_height, x0 : x0 + border] = green
    frame[y0 : y0 + inset_height, x0 + inset_width - border : x0 + inset_width] = green
    return frame


def _recording_frame_bgr(payload: _MaskViewerPayload) -> np.ndarray:
    """Return an unmodified copy of the camera BGR frame for recording."""

    frame = np.ascontiguousarray(payload.color_bgr, dtype=np.uint8).copy()
    mask = np.asarray(payload.object_mask, dtype=bool)
    if frame.ndim != 3 or frame.shape[2] != 3 or mask.shape != frame.shape[:2]:
        raise ValueError("recording mask/image shapes differ")
    return frame


def _recording_mask_frame_bgr(payload: _MaskViewerPayload) -> np.ndarray:
    """Return the exact full-resolution policy mask as a binary BGR frame."""

    frame = np.asarray(payload.color_bgr)
    mask = np.asarray(payload.object_mask, dtype=bool)
    if frame.ndim != 3 or frame.shape[2] != 3 or mask.shape != frame.shape[:2]:
        raise ValueError("recording mask/image shapes differ")
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def mask_video_path_for_recording(record_video_path: Path) -> Path:
    """Derive the sidecar mask-video path from the requested RGB MP4 path."""

    path = Path(record_video_path).expanduser().resolve()
    return path.with_name(f"{path.stem}_mask{path.suffix}")


def _bbox_from_mask(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not np.any(binary):
        return None
    ys, xs = np.nonzero(binary)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _copy_visualization_sample(
    sample: V94LiveVisualizationSample,
) -> V94LiveVisualizationSample:
    """Copy a sample off the policy thread before retaining it for saving."""

    return V94LiveVisualizationSample(
        sequence=int(sample.sequence),
        frame_id=int(sample.frame_id),
        captured_realtime_s=float(sample.captured_realtime_s),
        color_bgr=np.ascontiguousarray(sample.color_bgr, dtype=np.uint8).copy(),
        object_mask=np.ascontiguousarray(sample.object_mask, dtype=bool).copy(),
        pointcloud_xyzrgb_palm=np.ascontiguousarray(
            sample.pointcloud_xyzrgb_palm, dtype=np.float32
        ).copy(),
        pointcloud_valid=np.ascontiguousarray(
            sample.pointcloud_valid, dtype=np.float32
        ).copy(),
        T_base_palm_at_capture=np.ascontiguousarray(
            sample.T_base_palm_at_capture, dtype=np.float64
        ).copy(),
        source_valid_points=int(sample.source_valid_points),
        point_coordinate_frame=str(sample.point_coordinate_frame),
        requested_object_mask_mode=str(sample.requested_object_mask_mode),
        effective_object_mask_mode=str(sample.effective_object_mask_mode),
        effective_provider_mask_publication_mode=str(
            sample.effective_provider_mask_publication_mode
        ),
        effective_provider_recovery_publication_mode=str(
            sample.effective_provider_recovery_publication_mode
        ),
        provider_published_mask_valid=(
            None
            if sample.provider_published_mask_valid is None
            else bool(sample.provider_published_mask_valid)
        ),
        provider_published_mask_area_px=(
            None
            if sample.provider_published_mask_area_px is None
            else int(sample.provider_published_mask_area_px)
        ),
        provider_published_mask_bbox_xyxy=(
            None
            if sample.provider_published_mask_bbox_xyxy is None
            else tuple(
                int(value) for value in sample.provider_published_mask_bbox_xyxy
            )
        ),
        provider_published_mask_source=str(
            sample.provider_published_mask_source
        ),
        provider_published_mask_message=str(
            sample.provider_published_mask_message
        ),
        provider_online_sam2_status=str(sample.provider_online_sam2_status),
        projector_effective_policy_mask_provenance=str(
            sample.projector_effective_policy_mask_provenance
        ),
        projector_effective_policy_mask_source_frame_id=(
            None
            if sample.projector_effective_policy_mask_source_frame_id is None
            else int(sample.projector_effective_policy_mask_source_frame_id)
        ),
        projector_effective_policy_mask_source_captured_realtime_s=(
            None
            if sample.projector_effective_policy_mask_source_captured_realtime_s
            is None
            else float(
                sample.projector_effective_policy_mask_source_captured_realtime_s
            )
        ),
        projector_effective_policy_mask_area_px=(
            None
            if sample.projector_effective_policy_mask_area_px is None
            else int(sample.projector_effective_policy_mask_area_px)
        ),
        projector_effective_policy_mask_bbox_xyxy=(
            None
            if sample.projector_effective_policy_mask_bbox_xyxy is None
            else tuple(
                int(value)
                for value in sample.projector_effective_policy_mask_bbox_xyxy
            )
        ),
    )


def _draw_pointcloud_projections(
    xyz_base: np.ndarray,
    colors: np.ndarray,
    *,
    sequence: int,
    frame_id: int,
) -> Any:
    """Create a dependency-light three-view rendering of the exact cloud."""

    from PIL import Image, ImageDraw, ImageFont

    width, height = 1320, 500
    canvas = Image.new("RGB", (width, height), (13, 16, 21))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15
        )
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
        )
    except BaseException:
        font = ImageFont.load_default()
        title_font = font
    draw.text(
        (16, 10),
        f"Exact final policy cloud in robot_base | seq={sequence} frame={frame_id} "
        f"valid={xyz_base.shape[0]}/128",
        font=title_font,
        fill=(245, 245, 245),
    )
    panels = ((0, 1, "X-Y"), (0, 2, "X-Z"), (1, 2, "Y-Z"))
    panel_width = 420
    panel_top = 52
    panel_height = 420
    for panel_index, (u_axis, v_axis, label) in enumerate(panels):
        left = 12 + panel_index * 436
        top = panel_top
        right = left + panel_width
        bottom = top + panel_height
        draw.rectangle((left, top, right, bottom), outline=(90, 100, 115), width=1)
        draw.text((left + 10, top + 8), label, font=title_font, fill=(235, 235, 235))
        if xyz_base.shape[0] == 0:
            draw.text(
                (left + 120, top + 200),
                "no valid points",
                font=font,
                fill=(230, 100, 100),
            )
            continue
        uv = np.asarray(xyz_base[:, [u_axis, v_axis]], dtype=np.float64)
        lo = uv.min(axis=0)
        hi = uv.max(axis=0)
        center = (lo + hi) * 0.5
        span = max(float(np.max(hi - lo)), 0.002) * 1.15
        normalized = (uv - (center - span * 0.5)) / span
        px = left + 30 + normalized[:, 0] * (panel_width - 60)
        py = bottom - 30 - normalized[:, 1] * (panel_height - 60)
        for x, y, rgb in zip(px, py, colors):
            color = tuple(int(value) for value in np.clip(rgb * 255.0, 0, 255))
            draw.ellipse(
                (float(x) - 3, float(y) - 3, float(x) + 3, float(y) + 3),
                fill=color,
            )
        draw.text(
            (left + 10, bottom - 22),
            f"range={span:.4f}m  center=({center[0]:.4f},{center[1]:.4f})m",
            font=font,
            fill=(185, 195, 205),
        )
    return canvas


def _save_visualization_sample(
    sample: V94LiveVisualizationSample,
    *,
    output_directory: Path,
    selected_bbox_xyxy: Optional[tuple[int, int, int, int]],
) -> Mapping[str, object]:
    """Save the exact retained policy observation after hardware has stopped."""

    from PIL import Image, ImageDraw, ImageFont

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    color_bgr = np.asarray(sample.color_bgr, dtype=np.uint8)
    mask = np.asarray(sample.object_mask, dtype=bool)
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("saved visualization color image must be HxWx3 BGR")
    if mask.shape != color_bgr.shape[:2]:
        raise ValueError("saved visualization mask/image shapes differ")
    color_rgb = np.ascontiguousarray(color_bgr[..., ::-1])
    mask_bbox = _bbox_from_mask(mask)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
    except BaseException:
        font = ImageFont.load_default()

    rgb_image = Image.fromarray(color_rgb)
    bbox_image = rgb_image.copy()
    bbox_draw = ImageDraw.Draw(bbox_image)
    if selected_bbox_xyxy is not None:
        bbox_draw.rectangle(selected_bbox_xyxy, outline=(255, 0, 255), width=3)
    bbox_draw.text(
        (12, 10),
        "magenta: operator-selected bbox",
        font=font,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

    overlay_payload = _MaskViewerPayload(
        sequence=int(sample.sequence),
        frame_id=int(sample.frame_id),
        color_bgr=color_bgr,
        object_mask=mask,
        source_valid_points=int(sample.source_valid_points),
        final_policy_points=int(np.count_nonzero(sample.pointcloud_valid > 0.5)),
    )
    overlay_rgb, lines = _mask_overlay_rgb(overlay_payload)
    overlay_image = Image.fromarray(overlay_rgb)
    overlay_draw = ImageDraw.Draw(overlay_image)
    if selected_bbox_xyxy is not None:
        overlay_draw.rectangle(selected_bbox_xyxy, outline=(255, 0, 255), width=3)
    if mask_bbox is not None:
        overlay_draw.rectangle(mask_bbox, outline=(255, 255, 0), width=3)
    for index, line in enumerate(
        (*lines, "magenta=selected bbox; yellow=final mask bbox; green=final mask")
    ):
        xy = (12, 8 + 23 * index)
        overlay_draw.text(
            xy,
            line,
            font=font,
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    pointcloud = np.asarray(sample.pointcloud_xyzrgb_palm, dtype=np.float32)
    validity = np.asarray(sample.pointcloud_valid, dtype=np.float32)
    xyz_input, colors = _policy_cloud_geometry_and_colors(pointcloud, validity)
    transform = np.asarray(sample.T_base_palm_at_capture, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("saved visualization palm transform must be finite 4x4")
    coordinate_frame = str(sample.point_coordinate_frame)
    if coordinate_frame == "policy_palm":
        xyz_base = xyz_input @ transform[:3, :3].T + transform[:3, 3]
    elif coordinate_frame == "robot_base_via_identity_T_base_palm":
        if not np.allclose(
            transform, np.eye(4, dtype=np.float64), atol=1.0e-12, rtol=0.0
        ):
            raise ValueError(
                "robot-base visualization input requires identity "
                "T_base_palm_at_capture"
            )
        xyz_base = xyz_input.copy()
    else:  # protected by V94LiveVisualizationSample.__post_init__
        raise ValueError(f"unsupported point coordinate frame: {coordinate_frame}")
    cloud_image = _draw_pointcloud_projections(
        xyz_base,
        colors,
        sequence=int(sample.sequence),
        frame_id=int(sample.frame_id),
    )

    rgb_image.save(output / "color_rgb.png")
    bbox_image.save(output / "rgb_with_selected_bbox.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(
        output / "final_policy_mask.png"
    )
    overlay_image.save(output / "final_policy_mask_overlay.png")
    cloud_image.save(output / "final_policy_pointcloud_robot_base.png")
    np.save(output / "final_policy_mask.npy", mask)
    if coordinate_frame == "policy_palm":
        np.save(output / "final_policy_points_palm.npy", pointcloud)
    np.save(output / "final_policy_points_robot_base.npy", xyz_base.astype(np.float32))
    np.save(output / "final_policy_valid.npy", validity)
    np.save(output / "T_base_palm_at_capture.npy", transform)

    projector_bbox = sample.projector_effective_policy_mask_bbox_xyxy
    files: dict[str, str] = {
        "bbox": "rgb_with_selected_bbox.png",
        "mask": "final_policy_mask.png",
        "mask_overlay": "final_policy_mask_overlay.png",
        "pointcloud_visualization": "final_policy_pointcloud_robot_base.png",
        "raw_policy_mask": "final_policy_mask.npy",
        "raw_policy_points_robot_base": "final_policy_points_robot_base.npy",
        "raw_policy_validity": "final_policy_valid.npy",
        "capture_transform": "T_base_palm_at_capture.npy",
    }
    if coordinate_frame == "policy_palm":
        files["raw_policy_points_palm"] = "final_policy_points_palm.npy"

    manifest: dict[str, object] = {
        "schema_version": 2,
        "mask_provenance_schema": "provider_projector_split_v1",
        "content": "exact_last_published_policy_observation",
        "sequence": int(sample.sequence),
        "camera_frame_id": int(sample.frame_id),
        "captured_realtime_s": float(sample.captured_realtime_s),
        "requested_object_mask_mode": str(sample.requested_object_mask_mode),
        "effective_object_mask_mode": str(sample.effective_object_mask_mode),
        "effective_provider_mask_publication_mode": str(
            sample.effective_provider_mask_publication_mode
        ),
        "effective_provider_recovery_publication_mode": str(
            sample.effective_provider_recovery_publication_mode
        ),
        "provider_published_mask_frame_id": int(sample.frame_id),
        "provider_published_mask_coordinate_space": "source_camera_pixels",
        "provider_published_mask_valid": sample.provider_published_mask_valid,
        "provider_published_mask_area_px": (
            None
            if sample.provider_published_mask_area_px is None
            else int(sample.provider_published_mask_area_px)
        ),
        "provider_published_mask_bbox_xyxy": (
            None
            if sample.provider_published_mask_bbox_xyxy is None
            else [
                int(value)
                for value in sample.provider_published_mask_bbox_xyxy
            ]
        ),
        "provider_published_mask_source": str(
            sample.provider_published_mask_source
        ),
        "provider_published_mask_message": str(
            sample.provider_published_mask_message
        ),
        "provider_online_sam2_status": str(
            sample.provider_online_sam2_status
        ),
        "projector_effective_policy_mask_provenance": str(
            sample.projector_effective_policy_mask_provenance
        ),
        "projector_effective_policy_mask_source_frame_id": (
            None
            if sample.projector_effective_policy_mask_source_frame_id is None
            else int(sample.projector_effective_policy_mask_source_frame_id)
        ),
        "projector_effective_policy_mask_source_captured_realtime_s": (
            None
            if sample.projector_effective_policy_mask_source_captured_realtime_s
            is None
            else float(
                sample.projector_effective_policy_mask_source_captured_realtime_s
            )
        ),
        "projector_effective_policy_mask_coordinate_space": "policy_rgbd_pixels",
        "projector_effective_policy_mask_area_px": (
            None
            if sample.projector_effective_policy_mask_area_px is None
            else int(sample.projector_effective_policy_mask_area_px)
        ),
        "projector_effective_policy_mask_bbox_xyxy": (
            None
            if projector_bbox is None
            else [int(value) for value in projector_bbox]
        ),
        "selected_bbox_xyxy": (
            None if selected_bbox_xyxy is None else list(selected_bbox_xyxy)
        ),
        "saved_source_resolution_mask_semantics": (
            "projector_effective_policy_mask_nearest_expanded_for_overlay"
        ),
        "projector_effective_source_resolution_mask_bbox_xyxy": (
            None if mask_bbox is None else list(mask_bbox)
        ),
        "projector_effective_source_resolution_mask_area_px": int(
            np.count_nonzero(mask)
        ),
        "final_mask_bbox_xyxy": None if mask_bbox is None else list(mask_bbox),
        "final_mask_area_px": int(np.count_nonzero(mask)),
        "source_valid_points_before_sampling": int(sample.source_valid_points),
        "policy_point_feature_mode": "xyzrgb" if pointcloud.shape[1] == 6 else "xyz",
        "policy_point_array_shape": list(pointcloud.shape),
        "final_valid_policy_points": int(xyz_base.shape[0]),
        "pointcloud_input_frame": (
            "palm" if coordinate_frame == "policy_palm" else "robot_base"
        ),
        "pointcloud_input_frame_provenance": coordinate_frame,
        "palm_frame_geometry_validated": coordinate_frame == "policy_palm",
        "pointcloud_visualization_frame": "robot_base",
        "robot_base_bounds_m": (
            None
            if xyz_base.shape[0] == 0
            else {
                "minimum": xyz_base.min(axis=0).tolist(),
                "maximum": xyz_base.max(axis=0).tolist(),
                "centroid": xyz_base.mean(axis=0).tolist(),
            }
        ),
        "files": files,
    }
    (output / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _prepare_child(cpu_affinity: tuple[int, ...]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    _set_current_thread_affinity(cpu_affinity)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("DISPLAY/WAYLAND_DISPLAY is not available")


def _send_startup_error(connection: Any, ready_sent: bool, exc: BaseException) -> None:
    detail = f"{type(exc).__name__}: {exc}"
    if not ready_sent:
        try:
            connection.send({"status": "ERROR", "detail": detail})
        except BaseException:
            pass
    else:
        print(
            f"[V94 visualization] viewer stopped: {detail}",
            file=sys.stderr,
            flush=True,
        )


def _mask_viewer_process(
    payload_queue: Any,
    stop_event: Any,
    ready_connection: Any,
    cpu_affinity: tuple[int, ...],
) -> None:
    """Tk/Pillow-only mask child; isolated from Open3D/GLFW."""

    root = None
    ready_sent = False
    try:
        _prepare_child(cpu_affinity)
        import tkinter as tk
        from PIL import Image, ImageDraw, ImageFont, ImageTk

        _constrain_all_current_threads(cpu_affinity)

        mask_window = "V94 final policy mask"
        root = tk.Tk()
        root.title(mask_window)
        root.geometry("848x480+20+40")
        root.resizable(True, True)
        label = tk.Label(root, background="black")
        label.pack(fill=tk.BOTH, expand=True)
        closed = False

        def request_close(_event: Any = None) -> None:
            nonlocal closed
            closed = True

        root.protocol("WM_DELETE_WINDOW", request_close)
        root.bind("q", request_close)
        root.bind("Q", request_close)
        root.bind("<Escape>", request_close)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
            )
        except BaseException:
            font = ImageFont.load_default()
        root.update_idletasks()
        root.update()
        ready_connection.send(
            {
                "status": "READY",
                "viewer": "mask",
                "mask_window": mask_window,
                "cpu_affinity": list(cpu_affinity),
            }
        )
        ready_sent = True
        ready_connection.close()

        latest: Optional[_MaskViewerPayload] = None
        photo = None
        while not stop_event.is_set() and not closed:
            while True:
                try:
                    latest = payload_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                overlay, lines = _mask_overlay_rgb(latest)
                display_payload = _MaskViewerPayload(
                    sequence=latest.sequence,
                    frame_id=latest.frame_id,
                    color_bgr=np.ascontiguousarray(overlay[..., ::-1]),
                    object_mask=latest.object_mask,
                    source_valid_points=latest.source_valid_points,
                    final_policy_points=latest.final_policy_points,
                )
                display_bgr = _mask_viewer_frame_bgr(display_payload)
                image = Image.fromarray(
                    np.ascontiguousarray(display_bgr[..., ::-1]), mode="RGB"
                )
                draw = ImageDraw.Draw(image)
                mask = np.asarray(latest.object_mask, dtype=bool)
                if np.any(mask):
                    ys, xs = np.nonzero(mask)
                    draw.rectangle(
                        (
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max()),
                            int(ys.max()),
                        ),
                        outline=(255, 255, 0),
                        width=2,
                    )
                for index, line in enumerate(lines):
                    xy = (12, 8 + 23 * index)
                    draw.text((xy[0] + 1, xy[1] + 1), line, font=font, fill=(0, 0, 0))
                    draw.text(xy, line, font=font, fill=(255, 255, 255))
                photo = ImageTk.PhotoImage(image=image)
                label.configure(image=photo)
                latest = None
            try:
                root.update_idletasks()
                root.update()
            except BaseException:
                break
            stop_event.wait(0.015)
    except BaseException as exc:
        _send_startup_error(ready_connection, ready_sent, exc)
    finally:
        try:
            ready_connection.close()
        except BaseException:
            pass
        if root is not None:
            try:
                root.destroy()
            except BaseException:
                pass


def _cloud_viewer_process(
    payload_queue: Any,
    stop_event: Any,
    ready_connection: Any,
    cpu_affinity: tuple[int, ...],
) -> None:
    """Open3D-only cloud child; shows exact policy points in robot_base."""

    visualizer = None
    ready_sent = False
    try:
        _prepare_child(cpu_affinity)
        import open3d as o3d

        _constrain_all_current_threads(cpu_affinity)
        cloud_window = "V94 final policy point cloud (robot_base, 128 points)"
        visualizer = o3d.visualization.Visualizer()
        if not bool(
            visualizer.create_window(
                window_name=cloud_window,
                width=850,
                height=700,
                left=890,
                top=40,
                visible=True,
            )
        ):
            raise RuntimeError("Open3D could not create its display window")
        cloud = o3d.geometry.PointCloud()
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.08, origin=(0.0, 0.0, 0.0)
        )
        visualizer.add_geometry(axes)
        render = visualizer.get_render_option()
        if render is not None:
            render.point_size = 8.0
            render.background_color = np.asarray([0.04, 0.04, 0.04])
        ready_connection.send(
            {
                "status": "READY",
                "viewer": "cloud",
                "cloud_window": cloud_window,
                "cpu_affinity": list(cpu_affinity),
            }
        )
        ready_sent = True
        ready_connection.close()

        latest: Optional[_CloudViewerPayload] = None
        first_cloud = True
        cloud_added = False
        while not stop_event.is_set():
            while True:
                try:
                    latest = payload_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                palm = np.asarray(latest.T_base_palm_at_capture, dtype=np.float64)
                xyz_palm, colors = _policy_cloud_geometry_and_colors(
                    latest.pointcloud_xyzrgb_palm,
                    latest.pointcloud_valid,
                )
                xyz_base = xyz_palm @ palm[:3, :3].T + palm[:3, 3]
                cloud.points = o3d.utility.Vector3dVector(xyz_base)
                cloud.colors = o3d.utility.Vector3dVector(colors)
                if not cloud_added:
                    visualizer.add_geometry(cloud, reset_bounding_box=False)
                    cloud_added = True
                else:
                    visualizer.update_geometry(cloud)
                if first_cloud and xyz_base.shape[0] > 0:
                    visualizer.reset_view_point(True)
                    first_cloud = False
                latest = None
            if not bool(visualizer.poll_events()):
                break
            visualizer.update_renderer()
            stop_event.wait(0.015)
    except BaseException as exc:
        _send_startup_error(ready_connection, ready_sent, exc)
    finally:
        try:
            ready_connection.close()
        except BaseException:
            pass
        if visualizer is not None:
            try:
                visualizer.destroy_window()
            except BaseException:
                pass


def _video_recorder_process(
    payload_queue: Any,
    stop_event: Any,
    ready_connection: Any,
    cpu_affinity: tuple[int, ...],
    output_path: str,
    fps: float,
) -> None:
    """Write a lossy observation video outside the robot-control process."""

    writer = None
    mask_writer = None
    ready_sent = False
    frames = 0
    path = Path(output_path)
    mask_path = mask_video_path_for_recording(path)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = "1"
        _set_current_thread_affinity(cpu_affinity)
        import cv2

        _constrain_all_current_threads(cpu_affinity)
        path.parent.mkdir(parents=True, exist_ok=True)
        for candidate in (path, mask_path):
            if candidate.exists():
                raise RuntimeError(f"recording path already exists: {candidate}")
        ready_connection.send(
            {
                "status": "READY",
                "viewer": "video",
                "output_path": str(path),
                "mask_output_path": str(mask_path),
                "fps": float(fps),
                "cpu_affinity": list(cpu_affinity),
            }
        )
        ready_sent = True
        ready_connection.close()
        while not stop_event.is_set():
            try:
                payload = payload_queue.get(timeout=0.050)
            except queue.Empty:
                continue
            frame = _recording_frame_bgr(payload)
            mask_frame = _recording_mask_frame_bgr(payload)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (int(width), int(height)),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"OpenCV could not open MP4 writer: {path}")
                mask_writer = cv2.VideoWriter(
                    str(mask_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (int(width), int(height)),
                )
                if not mask_writer.isOpened():
                    raise RuntimeError(
                        f"OpenCV could not open mask MP4 writer: {mask_path}"
                    )
            writer.write(frame)
            mask_writer.write(mask_frame)
            frames += 1
    except BaseException as exc:
        _send_startup_error(ready_connection, ready_sent, exc)
    finally:
        try:
            ready_connection.close()
        except BaseException:
            pass
        if writer is not None:
            try:
                writer.release()
            except BaseException:
                pass
        if mask_writer is not None:
            try:
                mask_writer.release()
            except BaseException:
                pass
        if ready_sent:
            print(
                f"[V94 video] saved_rgb={path} saved_mask={mask_path} "
                f"frames={frames}",
                flush=True,
            )


class V94LiveVisualizer:
    """Latest-only bridge to isolated OpenCV and Open3D processes."""

    construction_is_inert = True

    def __init__(
        self,
        *,
        update_rate_hz: float = 10.0,
        cpu_affinity: Sequence[int] = (),
        show_live_windows: bool = True,
        save_directory: Optional[Path] = None,
        record_video_path: Optional[Path] = None,
        record_video_rate_hz: Optional[float] = None,
        selected_bbox_xyxy: Optional[Sequence[int]] = None,
        ready_timeout_s: float = 10.0,
        join_timeout_s: float = 2.0,
    ) -> None:
        rate = float(update_rate_hz)
        if not np.isfinite(rate) or not 1.0 <= rate <= 15.0:
            raise ValueError("live visualization rate must be in 1..15 Hz")
        cpus = tuple(int(value) for value in cpu_affinity)
        if any(value < 0 for value in cpus) or len(set(cpus)) != len(cpus):
            raise ValueError("visualization CPU affinity must be unique/nonnegative")
        self.update_rate_hz = rate
        self.cpu_affinity = cpus
        self.show_live_windows = bool(show_live_windows)
        self.save_directory = (
            None if save_directory is None else Path(save_directory).resolve()
        )
        self.record_video_path = (
            None
            if record_video_path is None
            else Path(record_video_path).expanduser().resolve()
        )
        self.record_mask_video_path = (
            None
            if self.record_video_path is None
            else mask_video_path_for_recording(self.record_video_path)
        )
        if (
            self.record_video_path is not None
            and self.record_video_path.suffix.lower() != ".mp4"
        ):
            raise ValueError("record video path must end in .mp4")
        video_rate = (
            rate if record_video_rate_hz is None else float(record_video_rate_hz)
        )
        if not np.isfinite(video_rate) or not 1.0 <= video_rate <= 30.0:
            raise ValueError("record video rate must be in 1..30 Hz")
        self.record_video_rate_hz = video_rate
        if selected_bbox_xyxy is None:
            self.selected_bbox_xyxy = None
        else:
            bbox = tuple(int(value) for value in selected_bbox_xyxy)
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError("selected visualization bbox must be valid xyxy")
            self.selected_bbox_xyxy = bbox
        self.ready_timeout_s = float(ready_timeout_s)
        self.join_timeout_s = float(join_timeout_s)
        self._context = mp.get_context("spawn")
        self._latest: Optional[V94LiveVisualizationSample] = None
        self._wake = threading.Event()
        self._bridge_stop = threading.Event()
        self._process_stop: Any = None
        self._mask_queue: Any = None
        self._cloud_queue: Any = None
        self._video_queue: Any = None
        self._mask_process: Any = None
        self._cloud_process: Any = None
        self._video_process: Any = None
        self._bridge: Optional[threading.Thread] = None
        self._open = False
        self._closed = False
        self._offered = 0
        self._published = 0
        self._mask_published = 0
        self._cloud_published = 0
        self._video_published = 0
        self._dropped = 0
        self._duplicate_frames = 0
        self._last_offered_frame_id: Optional[int] = None
        self._last_viewer_frame_id: Optional[int] = None
        self._last_video_sequence: Optional[int] = None
        self._snapshot_sample: Optional[V94LiveVisualizationSample] = None
        self._snapshot_saved = False
        self._snapshot_save_error: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self._open and not self._closed

    def open(self) -> None:
        if self._open or self._closed:
            raise V94LiveVisualizationError("live visualizer is single-use")
        if not self.show_live_windows and self.record_video_path is None:
            raise V94LiveVisualizationError(
                "background visualization pipeline has no requested output"
            )
        if self.show_live_windows and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            raise V94LiveVisualizationError(
                "live visualization requested but DISPLAY/WAYLAND_DISPLAY is absent"
            )
        self._process_stop = self._context.Event()
        self._mask_queue = (
            self._context.Queue(maxsize=1) if self.show_live_windows else None
        )
        self._cloud_queue = (
            self._context.Queue(maxsize=1) if self.show_live_windows else None
        )
        self._video_queue = (
            None
            if self.record_video_path is None
            else self._context.Queue(maxsize=4)
        )
        mask_parent_ready = None
        mask_child_ready = None
        cloud_parent_ready = None
        cloud_child_ready = None
        if self.show_live_windows:
            mask_parent_ready, mask_child_ready = self._context.Pipe(duplex=False)
            cloud_parent_ready, cloud_child_ready = self._context.Pipe(duplex=False)
        video_parent_ready = None
        video_child_ready = None
        if self.record_video_path is not None:
            video_parent_ready, video_child_ready = self._context.Pipe(duplex=False)
        if self.show_live_windows:
            self._mask_process = self._context.Process(
                target=_mask_viewer_process,
                args=(
                    self._mask_queue,
                    self._process_stop,
                    mask_child_ready,
                    self.cpu_affinity,
                ),
                name="v94-policy-mask-viewer",
                daemon=True,
            )
            self._cloud_process = self._context.Process(
                target=_cloud_viewer_process,
                args=(
                    self._cloud_queue,
                    self._process_stop,
                    cloud_child_ready,
                    self.cpu_affinity,
                ),
                name="v94-policy-cloud-viewer",
                daemon=True,
            )
        if self.record_video_path is not None:
            assert video_child_ready is not None
            self._video_process = self._context.Process(
                target=_video_recorder_process,
                args=(
                    self._video_queue,
                    self._process_stop,
                    video_child_ready,
                    self.cpu_affinity,
                    str(self.record_video_path),
                    self.record_video_rate_hz,
                ),
                name="v94-policy-mask-video-recorder",
                daemon=True,
            )
        try:
            if self._mask_process is not None:
                self._mask_process.start()
            if self._cloud_process is not None:
                self._cloud_process.start()
            if self._video_process is not None:
                self._video_process.start()
            if mask_child_ready is not None:
                mask_child_ready.close()
            if cloud_child_ready is not None:
                cloud_child_ready.close()
            if video_child_ready is not None:
                video_child_ready.close()
            deadline = time.monotonic() + self.ready_timeout_s
            ready_connections = []
            if mask_parent_ready is not None:
                ready_connections.append(("mask", mask_parent_ready))
            if cloud_parent_ready is not None:
                ready_connections.append(("cloud", cloud_parent_ready))
            if video_parent_ready is not None:
                ready_connections.append(("video", video_parent_ready))
            for name, connection in ready_connections:
                remaining = max(0.0, deadline - time.monotonic())
                if not connection.poll(remaining):
                    raise V94LiveVisualizationError(
                        f"{name} output startup timed out before READY"
                    )
                try:
                    reply = connection.recv()
                except EOFError as exc:
                    raise V94LiveVisualizationError(
                        f"{name} output process exited before READY"
                    ) from exc
                if not isinstance(reply, Mapping) or reply.get("status") != "READY":
                    detail = (
                        reply.get("detail", "unknown child startup error")
                        if isinstance(reply, Mapping)
                        else repr(reply)
                    )
                    raise V94LiveVisualizationError(
                        f"{name} visualization is unavailable: {detail}"
                    )
            self._open = True
            self._bridge = threading.Thread(
                target=self._bridge_loop,
                name="v94-visualization-bridge",
                daemon=True,
            )
            self._bridge.start()
            if self.show_live_windows:
                print(
                    "[V94 visualization] READY: exact-frame final policy "
                    f"mask + 128-point cloud at {self.update_rate_hz:g}Hz; "
                    "Q/Esc closes only the viewer"
                    + (
                        ""
                        if self.save_directory is None
                        else f"; final snapshot -> {self.save_directory}"
                    )
                    + (
                        ""
                        if self.record_video_path is None
                        else (
                            f"; RGB video {self.record_video_rate_hz:g}Hz -> "
                            f"{self.record_video_path}; mask video -> "
                            f"{self.record_mask_video_path}"
                        )
                    ),
                    flush=True,
                )
            else:
                print(
                    "[V94 video] READY: background recording only; no live "
                    f"window; RGB {self.record_video_rate_hz:g}Hz -> "
                    f"{self.record_video_path}; mask -> "
                    f"{self.record_mask_video_path}",
                    flush=True,
                )
        except BaseException:
            self._cleanup_resources()
            self._closed = True
            raise
        finally:
            for connection in (
                mask_parent_ready,
                cloud_parent_ready,
                video_parent_ready,
            ):
                if connection is None:
                    continue
                try:
                    connection.close()
                except BaseException:
                    pass

    def try_publish(self, sample: V94LiveVisualizationSample) -> bool:
        """Offer a sample without copying, waiting, queueing, or retrying."""

        if not self.is_open:
            return False
        try:
            frame_id = int(sample.frame_id)
            if self._last_offered_frame_id == frame_id:
                self._duplicate_frames += 1
                # A 60 Hz policy may legitimately consume the same 30 Hz
                # camera frame more than once. Keep those distinct policy
                # sequences in a requested action-aligned recording, while
                # still avoiding redundant GUI redraws.
                if self.record_video_path is None:
                    return False
            else:
                self._offered += 1
            if self._wake.is_set():
                self._dropped += 1
            self._latest = sample
            self._last_offered_frame_id = frame_id
            self._wake.set()
            return True
        except BaseException:
            self._dropped += 1
            return False

    def _bridge_loop(self) -> None:
        try:
            _set_current_thread_affinity(self.cpu_affinity)
        except BaseException:
            pass
        period_s = 1.0 / self.update_rate_hz
        next_viewer_publish_s = 0.0
        while not self._bridge_stop.is_set():
            self._wake.wait(0.050)
            self._wake.clear()
            if self._bridge_stop.is_set():
                break
            now = time.monotonic()
            sample = self._latest
            if sample is None:
                continue
            mask_alive = bool(
                self._mask_process is not None and self._mask_process.is_alive()
            )
            cloud_alive = bool(
                self._cloud_process is not None and self._cloud_process.is_alive()
            )
            video_alive = bool(
                self._video_process is not None and self._video_process.is_alive()
            )
            viewer_due = bool(
                (mask_alive or cloud_alive)
                and self._last_viewer_frame_id != int(sample.frame_id)
                and now >= next_viewer_publish_s
            )
            video_due = bool(
                video_alive
                and self._last_video_sequence != int(sample.sequence)
            )
            if not viewer_due and not video_due:
                continue
            accepted = False
            try:
                retained = _copy_visualization_sample(sample)
                self._snapshot_sample = retained
                mask_payload = _MaskViewerPayload(
                    sequence=int(retained.sequence),
                    frame_id=int(retained.frame_id),
                    color_bgr=retained.color_bgr,
                    object_mask=retained.object_mask,
                    source_valid_points=int(sample.source_valid_points),
                    final_policy_points=int(
                        np.count_nonzero(retained.pointcloud_valid > 0.5)
                    ),
                )
                cloud_payload = _CloudViewerPayload(
                    sequence=int(retained.sequence),
                    frame_id=int(retained.frame_id),
                    pointcloud_xyzrgb_palm=retained.pointcloud_xyzrgb_palm,
                    pointcloud_valid=retained.pointcloud_valid,
                    T_base_palm_at_capture=retained.T_base_palm_at_capture,
                )
                if viewer_due and mask_alive:
                    try:
                        self._mask_queue.put_nowait(mask_payload)
                        self._mask_published += 1
                        accepted = True
                    except queue.Full:
                        self._dropped += 1
                if viewer_due and cloud_alive:
                    try:
                        self._cloud_queue.put_nowait(cloud_payload)
                        self._cloud_published += 1
                        accepted = True
                    except queue.Full:
                        self._dropped += 1
                if video_due:
                    try:
                        self._video_queue.put_nowait(mask_payload)
                        self._video_published += 1
                        accepted = True
                    except queue.Full:
                        self._dropped += 1
            except BaseException:
                self._dropped += 1
                continue
            if not accepted:
                continue
            self._published += 1
            if viewer_due:
                self._last_viewer_frame_id = int(sample.frame_id)
                next_viewer_publish_s = time.monotonic() + period_s
            if video_due:
                self._last_video_sequence = int(sample.sequence)

    def stats(self) -> Mapping[str, object]:
        mask_alive = bool(
            self._mask_process is not None and self._mask_process.is_alive()
        )
        cloud_alive = bool(
            self._cloud_process is not None and self._cloud_process.is_alive()
        )
        video_alive = bool(
            self._video_process is not None and self._video_process.is_alive()
        )
        return {
            "requested": self.show_live_windows,
            "background_output_pipeline_active": True,
            "live_windows_requested": self.show_live_windows,
            "update_rate_hz": self.update_rate_hz,
            "cpu_affinity": list(self.cpu_affinity),
            "offered_unique_frames": self._offered,
            "published_frames": self._published,
            "mask_published_frames": self._mask_published,
            "cloud_published_frames": self._cloud_published,
            "video_published_frames": self._video_published,
            "dropped_frames": self._dropped,
            "duplicate_camera_frames_skipped": self._duplicate_frames,
            "mask_viewer_alive": mask_alive,
            "cloud_viewer_alive": cloud_alive,
            "video_recorder_alive": video_alive,
            "record_video_path": (
                None if self.record_video_path is None else str(self.record_video_path)
            ),
            "record_mask_video_path": (
                None
                if self.record_mask_video_path is None
                else str(self.record_mask_video_path)
            ),
            "record_video_rate_hz": (
                None
                if self.record_video_path is None
                else self.record_video_rate_hz
            ),
            "viewer_alive": (
                self.show_live_windows and mask_alive and cloud_alive
            ),
            "snapshot_directory": (
                None if self.save_directory is None else str(self.save_directory)
            ),
            "snapshot_saved": self._snapshot_saved,
            "snapshot_save_error": self._snapshot_save_error,
        }

    def _save_snapshot(self) -> None:
        if self.save_directory is None or self._snapshot_sample is None:
            return
        try:
            _save_visualization_sample(
                self._snapshot_sample,
                output_directory=self.save_directory,
                selected_bbox_xyxy=self.selected_bbox_xyxy,
            )
            self._snapshot_saved = True
            print(
                f"[V94 visualization] SAVED bbox/mask/pointcloud -> "
                f"{self.save_directory}",
                flush=True,
            )
        except BaseException as exc:
            self._snapshot_save_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[V94 visualization] snapshot save failed: "
                f"{self._snapshot_save_error}",
                file=sys.stderr,
                flush=True,
            )

    def _cleanup_resources(self) -> None:
        self._bridge_stop.set()
        self._wake.set()
        if self._bridge is not None and self._bridge is not threading.current_thread():
            self._bridge.join(timeout=self.join_timeout_s)
        if self._process_stop is not None:
            self._process_stop.set()
        for process in (
            self._mask_process,
            self._cloud_process,
            self._video_process,
        ):
            if process is None:
                continue
            process.join(timeout=self.join_timeout_s)
            if process.is_alive():
                process.terminate()
                process.join(timeout=self.join_timeout_s)
        for payload_queue in (
            self._mask_queue,
            self._cloud_queue,
            self._video_queue,
        ):
            if payload_queue is None:
                continue
            try:
                payload_queue.close()
                payload_queue.cancel_join_thread()
            except BaseException:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._open = False
        try:
            self._cleanup_resources()
        except BaseException:
            pass
        # ``_latest`` is the newest exact policy sample offered by the owner.
        # Copy it only after the hardware run and visualization bridge have
        # stopped, so even a short run or a viewer closed with Q still saves
        # the last observation without adding work to the control path.
        if self._latest is not None:
            try:
                self._snapshot_sample = _copy_visualization_sample(self._latest)
            except BaseException as exc:
                self._snapshot_save_error = f"{type(exc).__name__}: {exc}"
        self._save_snapshot()
        stats = dict(self.stats())
        print(
            "[V94 output] closed: "
            f"published={stats['published_frames']} "
            f"dropped={stats['dropped_frames']} "
            f"duplicate_camera_frames={stats['duplicate_camera_frames_skipped']}",
            flush=True,
        )


__all__ = [
    "V94LiveVisualizationError",
    "V94LiveVisualizationSample",
    "V94LiveVisualizationSink",
    "V94LiveVisualizer",
    "_recording_frame_bgr",
    "_save_visualization_sample",
]
