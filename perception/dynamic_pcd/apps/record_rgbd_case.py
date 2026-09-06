#!/usr/bin/env python3
"""Record one camera-only RGB-D regression case for object-PCD development.

Every synchronized frame returned by the D435 camera API is retained,
including frames on which the current segmentation code would be LOST.  A
separate pre-roll initialization RGB-D frame lets offline replay evaluate every
recorded video frame.  Lossless PNG sequences are the replay source of truth;
``preview.mp4`` is only for quick human inspection.  This module imports no
Franka or Inspire interface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import shutil
import sys
import threading
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from dynamic_pcd.utils.gui_env import (
    repair_gui_env_after_cv2_import,
    setup_gui_env,
)

setup_gui_env()
import cv2  # noqa: E402

repair_gui_env_after_cv2_import()

from dynamic_pcd.calibration.io import resolve_extrinsics  # noqa: E402
from dynamic_pcd.camera.realsense_camera import RealSenseCamera  # noqa: E402
from dynamic_pcd.config import load_config  # noqa: E402
from dynamic_pcd.segmentation.manual import select_roi_bbox  # noqa: E402
from dynamic_pcd.types import RGBDFrame  # noqa: E402
from dynamic_pcd.utils.geometry import clip_bbox  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "d435_default.yaml"
SCHEMA = "dynamic_object_pcd_rgbd_case_v1"
WRITER_QUEUE_CAPACITY = 90
MINIMUM_FREE_HEADROOM_BYTES = 512 * 1024 * 1024
CASE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "static_ball": {
        "duration_s": 5.0,
        "instruction": "Keep the ball completely still for the full recording.",
    },
    "rolling_ball": {
        "duration_s": 8.0,
        "instruction": "After RECORDING appears, roll the ball through the workspace.",
    },
    "rolling_cylinder": {
        "duration_s": 8.0,
        "instruction": "After RECORDING appears, roll the cylinder through the workspace.",
    },
    "hand_occlusion": {
        "duration_s": 10.0,
        "instruction": (
            "Use the RH56/dexterous-hand fingers (not the operator's hand): "
            "keep the object clear for ~2 s, partially occlude it, fully "
            "occlude it for ~1 s, then uncover it. Do not reselect the ROI."
        ),
    },
    "thrown_ball": {
        "duration_s": 12.0,
        "requires_initial_roi": False,
        "instruction": (
            "Keep the ball outside the image for the first ~2 s, throw it "
            "quickly through the camera view, then leave ~2 s of empty view. "
            "No manual ROI is recorded; offline replay must acquire it from "
            "the text prompt."
        ),
    },
}


class RecordingError(RuntimeError):
    pass


@dataclass(frozen=True)
class _WriteItem:
    index: int
    color_bgr: np.ndarray
    depth_raw: np.ndarray
    metadata: Dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_integrity_manifest(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.sha256":
            continue
        lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _frame_record(index: int, frame: RGBDFrame) -> Dict[str, Any]:
    return {
        "index": int(index),
        "color_path": f"color/{index:06d}.png",
        "depth_path": f"depth/{index:06d}.png",
        "frame_id": int(frame.frame_id),
        "sensor_frame_number": int(frame.sensor_frame_number),
        "depth_sensor_frame_number": int(frame.depth_sensor_frame_number),
        "camera_timestamp_s": float(frame.timestamp),
        "depth_timestamp_s": (
            None
            if frame.depth_timestamp_s is None
            else float(frame.depth_timestamp_s)
        ),
        "retrieved_at_s": (
            None if frame.retrieved_at_s is None else float(frame.retrieved_at_s)
        ),
        "retrieved_monotonic_s": (
            None
            if frame.retrieved_monotonic_s is None
            else float(frame.retrieved_monotonic_s)
        ),
        "timestamp_domain": str(frame.timestamp_domain),
        "color_depth_timestamp_skew_s": (
            None
            if frame.color_depth_timestamp_skew_s is None
            else float(frame.color_depth_timestamp_skew_s)
        ),
        "color_depth_epoch_timestamp_skew_s": (
            None
            if frame.color_depth_epoch_timestamp_skew_s is None
            else float(frame.color_depth_epoch_timestamp_skew_s)
        ),
        "host_clock_pair_span_s": (
            None
            if frame.host_clock_pair_span_s is None
            else float(frame.host_clock_pair_span_s)
        ),
        "dropped_queued_framesets_total": int(frame.dropped_queued_framesets),
        "rejected_timestamp_skew_frames_total": int(
            frame.rejected_timestamp_skew_frames
        ),
        "rejected_transport_stale_frames_total": int(
            frame.rejected_transport_stale_frames
        ),
        "capture_diagnostic": _json_safe(frame.capture_diagnostic or {}),
    }


def _write_initialization_frame(
    output: Path,
    frame: RGBDFrame,
    *,
    png_compression: int,
    role: str = "tracker_initialization",
) -> Dict[str, Any]:
    directory = output / "initialization"
    directory.mkdir(parents=True, exist_ok=False)
    color_path = directory / "color.png"
    depth_path = directory / "depth.png"
    png_args = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
    if not cv2.imwrite(str(color_path), frame.color_bgr, png_args):
        raise RecordingError(f"failed to write {color_path}")
    if not cv2.imwrite(str(depth_path), frame.depth_raw, png_args):
        raise RecordingError(f"failed to write {depth_path}")
    record = _frame_record(-1, frame)
    record["role"] = str(role)
    record["color_path"] = "initialization/color.png"
    record["depth_path"] = "initialization/depth.png"
    return record


def _expanded_provider_roi(
    tight_roi_xyxy: np.ndarray,
    *,
    width: int,
    height: int,
    padding_px: int,
) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(tight_roi_xyxy, dtype=np.int32).reshape(4)
    return clip_bbox(
        np.asarray(
            [
                x1 - int(padding_px),
                y1 - int(padding_px),
                x2 + int(padding_px),
                y2 + int(padding_px),
            ],
            dtype=np.int32,
        ),
        int(width),
        int(height),
    ).astype(np.int32)


class _FrameWriter:
    def __init__(
        self,
        output: Path,
        *,
        width: int,
        height: int,
        fps: float,
        png_compression: int,
    ) -> None:
        self.output = output
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.png_compression = int(png_compression)
        self.items: "queue.Queue[Optional[_WriteItem]]" = queue.Queue(
            maxsize=WRITER_QUEUE_CAPACITY
        )
        self.error: Optional[BaseException] = None
        self.preview_available = False
        self.preview_error: Optional[str] = None
        self.written = 0
        self._thread = threading.Thread(
            target=self._run, name="rgbd-case-writer", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def submit(self, item: _WriteItem) -> None:
        if self.error is not None:
            raise RecordingError(f"frame writer failed: {self.error}")
        try:
            self.items.put(item, timeout=1.0)
        except queue.Full as exc:
            raise RecordingError(
                "frame writer queue is full; refusing to drop RGB-D frames"
            ) from exc

    def close(self) -> None:
        while True:
            if self.error is not None:
                break
            try:
                self.items.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=60.0)
        if self._thread.is_alive():
            raise RecordingError("frame writer did not stop within 60 seconds")
        if self.error is not None:
            raise RecordingError(f"frame writer failed: {self.error}")

    def _run(self) -> None:
        video = None
        metadata_handle = None
        try:
            (self.output / "color").mkdir(parents=True, exist_ok=False)
            (self.output / "depth").mkdir(parents=True, exist_ok=False)
            metadata_handle = (self.output / "frames.jsonl").open(
                "w", encoding="utf-8"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                str(self.output / "preview.mp4"),
                fourcc,
                self.fps,
                (self.width, self.height),
            )
            if not video.isOpened():
                self.preview_error = "OpenCV could not create mp4v preview"
                video.release()
                video = None
                try:
                    (self.output / "preview.mp4").unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                self.preview_available = True
            png_args = [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression]
            while True:
                item = self.items.get()
                if item is None:
                    break
                color_path = self.output / item.metadata["color_path"]
                depth_path = self.output / item.metadata["depth_path"]
                if not cv2.imwrite(str(color_path), item.color_bgr, png_args):
                    raise RecordingError(f"failed to write {color_path}")
                if not cv2.imwrite(str(depth_path), item.depth_raw, png_args):
                    raise RecordingError(f"failed to write {depth_path}")
                if video is not None:
                    video.write(item.color_bgr)
                metadata_handle.write(
                    json.dumps(
                        _json_safe(item.metadata),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                self.written += 1
        except BaseException as exc:
            self.error = exc
        finally:
            if metadata_handle is not None:
                metadata_handle.close()
            if video is not None:
                video.release()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record every synchronized aligned D435 RGB-D frame for one "
            "object-PCD regression case; never open Franka/RH56."
        )
    )
    parser.add_argument("--case", choices=tuple(CASE_DEFAULTS), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        help="case directory; default: data/perception_corpus/<case>_<timestamp>",
    )
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--countdown-s", type=float, default=3.0)
    parser.add_argument(
        "--camera-serial",
        help="override camera.serial from the config snapshot",
    )
    parser.add_argument("--width", type=int, help="override camera width")
    parser.add_argument("--height", type=int, help="override camera height")
    parser.add_argument("--fps", type=int, help="override camera frame rate")
    parser.add_argument(
        "--color-exposure",
        type=float,
        help=(
            "manual RGB exposure in the RealSense option units reported by "
            "the device; disables RGB auto exposure"
        ),
    )
    parser.add_argument(
        "--color-gain",
        type=float,
        help="manual RGB gain; disables RGB auto exposure",
    )
    parser.add_argument(
        "--depth-exposure",
        type=float,
        help=(
            "manual stereo-depth exposure in RealSense option units; "
            "disables depth auto exposure"
        ),
    )
    parser.add_argument(
        "--depth-gain",
        type=float,
        help="manual stereo-depth gain; disables depth auto exposure",
    )
    parser.add_argument(
        "--laser-power",
        type=float,
        help="projector laser power in the RealSense option units",
    )
    parser.add_argument(
        "--max-color-depth-timestamp-skew-ms",
        type=float,
        help=(
            "camera-specific RGB-D timestamp tolerance; must remain below "
            "half a frame period"
        ),
    )
    parser.add_argument(
        "--camera-frame-only",
        action="store_true",
        help=(
            "record an uncalibrated camera-frame case and do not reuse the "
            "configured robot extrinsics"
        ),
    )
    parser.add_argument(
        "--object-text",
        default="ball",
        help="text prompt saved for automatic-grounding replay",
    )
    parser.add_argument(
        "--overlay-base-workspace-min-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "minimum robot_base corner of a live-view-only workspace box; "
            "requires --overlay-base-workspace-max-m and calibrated extrinsics"
        ),
    )
    parser.add_argument(
        "--overlay-base-workspace-max-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "maximum robot_base corner of a live-view-only workspace box; "
            "the lossless RGB-D frames remain unannotated"
        ),
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="tight object ROI; omit to select interactively",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--png-compression",
        type=int,
        default=1,
        choices=range(0, 10),
        metavar="0..9",
    )
    return parser


def _validate_roi(
    roi: Sequence[int], *, width: int, height: int
) -> np.ndarray:
    values = np.asarray(tuple(int(value) for value in roi), dtype=np.int32)
    if values.shape != (4,):
        raise ValueError("ROI must contain X1 Y1 X2 Y2")
    x1, y1, x2, y2 = values.tolist()
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must satisfy 0<=X1<X2 and 0<=Y1<Y2")
    if x2 > width or y2 > height:
        raise ValueError(f"ROI {values.tolist()} exceeds {width}x{height}")
    return values


def _preflight_disk(
    parent: Path, *, width: int, height: int, frame_count: int
) -> Dict[str, int]:
    # Lossless PNG normally uses much less space, but reserve the raw RGB+Z16
    # payload plus margin so incompressible scenes cannot fill the filesystem.
    raw_bytes = int(width) * int(height) * 5 * int(frame_count)
    required = int(raw_bytes * 1.15) + MINIMUM_FREE_HEADROOM_BYTES
    free = int(shutil.disk_usage(parent).free)
    if free < required:
        raise RecordingError(
            "insufficient free space for a lossless RGB-D case: "
            f"free={free / 2**30:.2f}GiB required>={required / 2**30:.2f}GiB. "
            "Choose another --output disk or free space first."
        )
    return {"free_bytes_before": free, "reserved_required_bytes": required}


def _draw_status(
    frame: RGBDFrame,
    roi: Optional[np.ndarray],
    lines: Sequence[str],
    workspace_overlay: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    image = frame.color_bgr.copy()
    if workspace_overlay is not None:
        vertices = np.asarray(
            workspace_overlay["projected_vertices_uv"], dtype=np.float64
        ).reshape(8, 2)
        for first, second in workspace_overlay["edge_vertex_indices"]:
            p1 = tuple(np.rint(vertices[int(first)]).astype(int).tolist())
            p2 = tuple(np.rint(vertices[int(second)]).astype(int).tolist())
            accepted, clipped_p1, clipped_p2 = cv2.clipLine(
                (0, 0, image.shape[1], image.shape[0]), p1, p2
            )
            if accepted:
                cv2.line(
                    image,
                    clipped_p1,
                    clipped_p2,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
    if roi is not None:
        x1, y1, x2, y2 = np.asarray(roi, dtype=np.int32).tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    for index, line in enumerate(lines):
        y = 28 + index * 25
        cv2.putText(
            image,
            str(line),
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(line),
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


_WORKSPACE_BOX_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def _project_base_workspace_overlay(
    frame: RGBDFrame,
    T_base_camera: np.ndarray,
    minimum_base_m: Sequence[float],
    maximum_base_m: Sequence[float],
) -> Dict[str, Any]:
    minimum = np.asarray(minimum_base_m, dtype=np.float64).reshape(3)
    maximum = np.asarray(maximum_base_m, dtype=np.float64).reshape(3)
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("workspace overlay bounds must be finite")
    if np.any(maximum <= minimum):
        raise ValueError(
            "workspace overlay maximum must exceed minimum on every axis"
        )
    transform = np.asarray(T_base_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("workspace overlay requires a finite 4x4 T_base_camera")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ValueError("workspace overlay T_base_camera has an invalid last row")
    T_camera_base = np.linalg.inv(transform)
    vertices_base = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    vertices_camera = (
        T_camera_base
        @ np.column_stack(
            (vertices_base, np.ones(vertices_base.shape[0], dtype=np.float64))
        ).T
    ).T[:, :3]
    if np.any(vertices_camera[:, 2] <= 0.0):
        raise ValueError("workspace overlay contains a corner behind the camera")
    intrinsics = frame.intrinsics
    projected = np.column_stack(
        (
            float(intrinsics.fx) * vertices_camera[:, 0] / vertices_camera[:, 2]
            + float(intrinsics.ppx),
            float(intrinsics.fy) * vertices_camera[:, 1] / vertices_camera[:, 2]
            + float(intrinsics.ppy),
        )
    )
    if not np.isfinite(projected).all():
        raise ValueError("workspace overlay projection is non-finite")
    inside = (
        (projected[:, 0] >= 0.0)
        & (projected[:, 0] <= float(intrinsics.width - 1))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] <= float(intrinsics.height - 1))
    )
    if not bool(np.all(inside)):
        raise ValueError(
            "workspace overlay is not fully visible in the commissioned image"
        )
    return {
        "reference_frame": "robot_base",
        "minimum_base_m": minimum.tolist(),
        "maximum_base_m": maximum.tolist(),
        "vertices_base_m": vertices_base.tolist(),
        "vertices_camera_m": vertices_camera.tolist(),
        "projected_vertices_uv": projected.tolist(),
        "edge_vertex_indices": [list(edge) for edge in _WORKSPACE_BOX_EDGES],
        "all_vertices_inside_image": True,
        "camera_depth_min_m": float(np.min(vertices_camera[:, 2])),
        "camera_depth_max_m": float(np.max(vertices_camera[:, 2])),
        "lossless_frames_annotated": False,
    }


def _show(image: np.ndarray, window: str) -> bool:
    cv2.imshow(window, image)
    key = cv2.waitKey(1) & 0xFF
    return key not in (27, ord("q"), ord("Q"))


def record_case(args: argparse.Namespace) -> Path:
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    cfg = load_config(str(config_path))
    camera_cfg = dict(cfg["camera"])
    capture_overrides: Dict[str, Any] = {}
    if getattr(args, "camera_serial", None) is not None:
        serial = str(args.camera_serial).strip()
        if not serial:
            raise ValueError("--camera-serial must not be empty")
        camera_cfg["serial"] = serial
        capture_overrides["camera.serial"] = serial
    for argument, key in (
        (getattr(args, "width", None), "width"),
        (getattr(args, "height", None), "height"),
        (getattr(args, "fps", None), "fps"),
    ):
        if argument is not None:
            value = int(argument)
            if value <= 0:
                raise ValueError(f"--{key} must be positive")
            camera_cfg[key] = value
            capture_overrides[f"camera.{key}"] = value
    color_exposure = getattr(args, "color_exposure", None)
    color_gain = getattr(args, "color_gain", None)
    if color_exposure is not None or color_gain is not None:
        camera_cfg["color_auto_exposure"] = False
        capture_overrides["camera.color_auto_exposure"] = False
    for argument, key in (
        (color_exposure, "color_exposure"),
        (color_gain, "color_gain"),
    ):
        if argument is not None:
            value = float(argument)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"--{key.replace('_', '-')} must be finite and non-negative")
            camera_cfg[key] = value
            capture_overrides[f"camera.{key}"] = value
    depth_exposure = getattr(args, "depth_exposure", None)
    depth_gain = getattr(args, "depth_gain", None)
    if depth_exposure is not None or depth_gain is not None:
        camera_cfg["depth_auto_exposure"] = False
        capture_overrides["camera.depth_auto_exposure"] = False
    for argument, key in (
        (depth_exposure, "depth_exposure"),
        (depth_gain, "depth_gain"),
        (getattr(args, "laser_power", None), "laser_power"),
    ):
        if argument is not None:
            value = float(argument)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"--{key.replace('_', '-')} must be finite and non-negative"
                )
            camera_cfg[key] = value
            capture_overrides[f"camera.{key}"] = value
    timestamp_skew_ms = getattr(
        args, "max_color_depth_timestamp_skew_ms", None
    )
    if timestamp_skew_ms is not None:
        timestamp_skew_s = float(timestamp_skew_ms) * 1.0e-3
        fps_for_limit = int(camera_cfg["fps"])
        half_frame_period_s = 0.5 / float(fps_for_limit)
        if (
            not math.isfinite(timestamp_skew_s)
            or timestamp_skew_s <= 0.0
            or timestamp_skew_s >= half_frame_period_s
        ):
            raise ValueError(
                "--max-color-depth-timestamp-skew-ms must be finite, "
                f"positive, and below {half_frame_period_s * 1000.0:.3f} "
                f"ms at {fps_for_limit} Hz"
            )
        camera_cfg["max_color_depth_timestamp_skew_s"] = timestamp_skew_s
        capture_overrides[
            "camera.max_color_depth_timestamp_skew_s"
        ] = timestamp_skew_s
    cfg["camera"] = camera_cfg
    if bool(getattr(args, "camera_frame_only", False)):
        cfg["extrinsics"] = {
            "calibration_file": None,
            "require_calibration": False,
            "require_quality_pass": False,
            "strict_camera_serial": False,
            "calibrated": False,
            "base_frame": "robot_base",
            "camera_frame": "camera_color_optical_frame",
            "T_base_camera": np.eye(4, dtype=np.float64).tolist(),
        }
        capture_overrides["extrinsics"] = "identity_camera_frame_only"
    # Validate site-specific calibration before opening the camera or asking
    # the operator to perform a physical test case.
    extrinsics = resolve_extrinsics(cfg.get("extrinsics"))
    width = int(camera_cfg["width"])
    height = int(camera_cfg["height"])
    fps = int(camera_cfg["fps"])
    duration_s = (
        float(CASE_DEFAULTS[args.case]["duration_s"])
        if args.duration_s is None
        else float(args.duration_s)
    )
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("--duration-s must be finite and positive")
    if (
        not math.isfinite(float(args.countdown_s))
        or float(args.countdown_s) < 0.0
    ):
        raise ValueError("--countdown-s must be finite and non-negative")
    target_frames = max(1, int(round(duration_s * fps)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        Path("object_pcd_testdata") / f"{args.case}_{stamp}"
        if args.output is None
        else args.output.expanduser()
    ).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    disk = _preflight_disk(
        output.parent,
        width=width,
        height=height,
        frame_count=target_frames,
    )

    stage = output.parent / f".{output.name}.partial"
    if stage.exists():
        raise FileExistsError(f"partial output already exists: {stage}")
    stage.mkdir()
    (stage / "capture_config.json").write_text(
        json.dumps(
            _json_safe(cfg),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    camera = RealSenseCamera(camera_cfg)
    writer: Optional[_FrameWriter] = None
    window = "RGB-D case recorder (q/Esc abort)"
    capture_started_utc = None
    first_frame: Optional[RGBDFrame] = None
    last_frame: Optional[RGBDFrame] = None
    tight_roi: Optional[np.ndarray] = None
    completed = False
    try:
        camera.start()
        expected_serial = str(camera_cfg.get("serial") or "")
        actual_serial = str(camera.device_serial or "")
        if expected_serial and actual_serial != expected_serial:
            raise RecordingError(
                "opened RealSense serial differs from requested camera: "
                f"actual={actual_serial or 'unknown'} expected={expected_serial}"
            )
        selection_frame = camera.get_frame()
        overlay_minimum = getattr(args, "overlay_base_workspace_min_m", None)
        overlay_maximum = getattr(args, "overlay_base_workspace_max_m", None)
        if (overlay_minimum is None) != (overlay_maximum is None):
            raise ValueError(
                "workspace overlay requires both minimum and maximum bounds"
            )
        workspace_overlay: Optional[Dict[str, Any]] = None
        if overlay_minimum is not None:
            if bool(getattr(args, "camera_frame_only", False)):
                raise ValueError(
                    "robot_base workspace overlay is unavailable in "
                    "--camera-frame-only mode"
                )
            if (
                str(extrinsics.reference_frame) != "robot_base"
                or not str(extrinsics.calibration_id or "")
            ):
                raise ValueError(
                    "robot_base workspace overlay requires calibrated "
                    "robot_base extrinsics"
                )
            workspace_overlay = _project_base_workspace_overlay(
                selection_frame,
                np.asarray(extrinsics.T_base_camera, dtype=np.float64),
                overlay_minimum,
                overlay_maximum,
            )
            capture_overrides["live_workspace_overlay"] = {
                "minimum_base_m": workspace_overlay["minimum_base_m"],
                "maximum_base_m": workspace_overlay["maximum_base_m"],
            }
        requires_initial_roi = bool(
            CASE_DEFAULTS[args.case].get("requires_initial_roi", True)
        )
        if not requires_initial_roi:
            if args.roi is not None:
                raise ValueError(
                    f"--case {args.case} is an automatic-entry case and "
                    "must not be given --roi"
                )
            tight_roi = None
            provider_roi = None
        elif args.roi is None:
            tight_roi = select_roi_bbox(
                selection_frame, window_name="Select object before recording"
            )
            if tight_roi is None:
                raise RecordingError("ROI selection was cancelled")
            padding = int(
                cfg.get("tracker", {}).get("interactive_roi_padding_px", 16)
            )
            provider_roi = _expanded_provider_roi(
                tight_roi,
                width=width,
                height=height,
                padding_px=padding,
            )
        else:
            tight_roi = _validate_roi(args.roi, width=width, height=height)
            padding = int(
                cfg.get("tracker", {}).get("interactive_roi_padding_px", 16)
            )
            provider_roi = _expanded_provider_roi(
                tight_roi,
                width=width,
                height=height,
                padding_px=padding,
            )
        if not requires_initial_roi:
            padding = 0
        instruction = str(CASE_DEFAULTS[args.case]["instruction"])
        print(f"[case] {args.case}: {instruction}", flush=True)
        countdown_deadline = time.monotonic() + float(args.countdown_s)
        while time.monotonic() < countdown_deadline:
            frame = camera.get_frame()
            remaining = max(0.0, countdown_deadline - time.monotonic())
            if not args.no_preview and not _show(
                _draw_status(
                    frame,
                    tight_roi,
                    (
                        f"{args.case}: starts in {remaining:.1f}s",
                        instruction,
                        *(
                            ("GREEN BOX: commissioned catch workspace",)
                            if workspace_overlay is not None
                            else ()
                        ),
                    ),
                    workspace_overlay,
                ),
                window,
            ):
                raise KeyboardInterrupt

        # Use one dedicated pre-roll RGB-D frame to initialize GrabCut/tracker
        # state.  It is not part of preview.mp4 or frame_count, so offline
        # replay can evaluate all requested video frames instead of consuming
        # video frame 0 as an initialization-only sample.
        initialization_frame = camera.get_frame()
        initialization_record = _write_initialization_frame(
            stage,
            initialization_frame,
            png_compression=int(args.png_compression),
            role=(
                "automatic_grounding_preroll"
                if not requires_initial_roi
                else "tracker_initialization"
            ),
        )

        writer = _FrameWriter(
            stage,
            width=width,
            height=height,
            fps=float(fps),
            png_compression=int(args.png_compression),
        )
        writer.start()
        capture_started_utc = datetime.now(timezone.utc).isoformat()
        print(
            f"[RECORDING] case={args.case} frames={target_frames} "
            f"duration={duration_s:.1f}s — {instruction}",
            flush=True,
        )
        for index in range(target_frames):
            frame = camera.get_frame()
            if first_frame is None:
                first_frame = frame
            last_frame = frame
            metadata = _frame_record(index, frame)
            writer.submit(
                _WriteItem(
                    index=index,
                    color_bgr=frame.color_bgr,
                    depth_raw=frame.depth_raw,
                    metadata=metadata,
                )
            )
            if not args.no_preview and not _show(
                _draw_status(
                    frame,
                    tight_roi,
                    (
                        f"RECORDING {args.case}",
                        f"{index + 1}/{target_frames}",
                        *(
                            ("KEEP TARGET INSIDE GREEN BOX",)
                            if workspace_overlay is not None
                            else ()
                        ),
                    ),
                    workspace_overlay,
                ),
                window,
            ):
                raise KeyboardInterrupt
        finished_writer = writer
        writer.close()
        writer = None
        preview_path = stage / "preview.mp4"
        preview_available = bool(
            finished_writer.preview_available
            and preview_path.is_file()
            and preview_path.stat().st_size > 0
        )
        preview_error = (
            None
            if preview_available
            else finished_writer.preview_error
            or "preview encoder produced no non-empty file"
        )
        if not preview_available:
            print(
                "[WARN] lossless RGB-D recording is complete, but preview.mp4 "
                f"is unavailable: {preview_error}",
                file=sys.stderr,
                flush=True,
            )
        if first_frame is None or last_frame is None:
            raise RecordingError("no RGB-D frame was recorded")

        manifest = {
            "schema": SCHEMA,
            "case": str(args.case),
            "instruction": instruction,
            "complete": True,
            "camera_only": True,
            "franka_interface_opened": False,
            "rh56_interface_opened": False,
            "robot_command_writes": False,
            "capture_started_utc": capture_started_utc,
            "capture_finished_utc": datetime.now(timezone.utc).isoformat(),
            "requested_duration_s": duration_s,
            "initialization_mode": (
                "automatic_text_grounding_on_recorded_frames"
                if not requires_initial_roi
                else "manual_roi_on_preroll"
            ),
            "object_initially_required_visible": bool(requires_initial_roi),
            "object_text": str(getattr(args, "object_text", "ball")),
            "nominal_fps": fps,
            "frame_count": target_frames,
            "initialization_frame": initialization_record,
            "image_width": width,
            "image_height": height,
            "color_encoding": "lossless_png_bgr8",
            "depth_encoding": "lossless_png_z16_aligned_to_color",
            "preview_available": preview_available,
            "preview_error": preview_error,
            "preview_semantics": (
                "lossy_mp4_for_human_review_only"
                if preview_available
                else "unavailable; lossless color PNG sequence is authoritative"
            ),
            "frame_metadata": "frames.jsonl",
            "capture_config_snapshot": "capture_config.json",
            "integrity_manifest": "MANIFEST.sha256",
            "depth_scale_m_per_unit": float(first_frame.depth_scale),
            "camera_intrinsics": initialization_frame.intrinsics.to_dict(),
            "camera_K": initialization_frame.intrinsics.as_matrix().astype(
                np.float64
            ).tolist(),
            "tight_roi_xyxy": (
                None if tight_roi is None else tight_roi.astype(int).tolist()
            ),
            "provider_initialization_roi_xyxy": (
                None if provider_roi is None else provider_roi.astype(int).tolist()
            ),
            "interactive_roi_padding_px": padding,
            "camera_serial": str(camera.device_serial or ""),
            "camera_name": str(camera.device_name or ""),
            "camera_firmware": str(camera.device_firmware_version or ""),
            "camera_usb_type": str(camera.device_usb_type_descriptor or ""),
            "camera_sdk_version": str(camera.sdk_version or ""),
            "calibration_id": str(extrinsics.calibration_id or ""),
            "reference_frame": str(extrinsics.reference_frame),
            "camera_frame": str(extrinsics.camera_frame),
            "T_base_camera": np.asarray(
                extrinsics.T_base_camera, dtype=np.float64
            ).tolist(),
            "live_workspace_overlay": workspace_overlay,
            "source_config_name": config_path.name,
            "source_config_sha256": _sha256(config_path),
            "capture_config_overrides": capture_overrides,
            "calibration_source_sha256": (
                _sha256(Path(extrinsics.source))
                if Path(extrinsics.source).is_file()
                else None
            ),
            "first_camera_timestamp_s": float(first_frame.timestamp),
            "last_camera_timestamp_s": float(last_frame.timestamp),
            "actual_recorded_sensor_duration_s": max(
                0.0,
                float(last_frame.timestamp) - float(first_frame.timestamp),
            ),
            "effective_accepted_frame_rate_hz": (
                float(target_frames - 1)
                / max(
                    1.0e-9,
                    float(last_frame.timestamp) - float(first_frame.timestamp),
                )
                if target_frames > 1
                else 0.0
            ),
            "first_sensor_frame_number": int(first_frame.sensor_frame_number),
            "last_sensor_frame_number": int(last_frame.sensor_frame_number),
            "recording_color_sensor_frame_gaps": max(
                0,
                int(last_frame.sensor_frame_number)
                - int(first_frame.sensor_frame_number)
                + 1
                - target_frames,
            ),
            "recording_depth_sensor_frame_gaps": max(
                0,
                int(last_frame.depth_sensor_frame_number)
                - int(first_frame.depth_sensor_frame_number)
                + 1
                - target_frames,
            ),
            "dropped_queued_framesets_total": int(
                last_frame.dropped_queued_framesets
            ),
            **disk,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_integrity_manifest(stage)
        stage.rename(output)
        completed = True
        print(
            f"[PASS] saved={output} frames={target_frames} "
            "franka_opened=0 rh56_opened=0 robot_writes=0",
            flush=True,
        )
        return output
    finally:
        if writer is not None:
            try:
                writer.close()
            except BaseException:
                pass
        camera.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if not completed:
            print(
                f"[partial] incomplete files, if any, remain at {stage}",
                file=sys.stderr,
                flush=True,
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        record_case(args)
        return 0
    except KeyboardInterrupt:
        print("[interrupted] camera recording stopped; no robot was opened", file=sys.stderr)
        return 130
    except (OSError, RecordingError, RuntimeError, ValueError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
