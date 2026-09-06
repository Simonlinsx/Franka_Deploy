#!/usr/bin/env python3
"""Capture a fresh calibrated RealSense scene cloud without controlling hardware.

The Franka configuration is supplied explicitly as metadata.  This program
never imports or opens a Franka or Inspire transport.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DYNAMIC_PCD_ROOT = WORKSPACE / "perception"
if str(DYNAMIC_PCD_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMIC_PCD_ROOT))

EXPECTED_CAMERA_SERIAL = "337322072188"
EXPECTED_CALIBRATION_ID = "eye-to-hand-b722bce10485c8a3"
EXPECTED_REFERENCE_FRAME = "robot_base"
SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture several fresh D435 frames, transform their full scene "
            "clouds into robot_base, and save a collision-audit NPZ. "
            "Perception only: no Franka or Inspire connection is opened."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DYNAMIC_PCD_ROOT / "configs/d435_default.yaml",
        help="dynamic_object_pcd config containing the calibrated eye-to-hand transform",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capture-q-rad",
        type=float,
        nargs=7,
        required=True,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help=(
            "Read-only Franka joint snapshot associated with this capture. "
            "The values are recorded verbatim; this program does not query the robot."
        ),
    )
    parser.add_argument("--expected-camera-serial", default=EXPECTED_CAMERA_SERIAL)
    parser.add_argument("--expected-calibration-id", default=EXPECTED_CALIBRATION_ID)
    parser.add_argument("--expected-camera-model", default="D435")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--capture-frames", type=int, default=5)
    parser.add_argument("--capture-timeout-s", type=float, default=15.0)
    parser.add_argument("--frame-timeout-ms", type=int, default=2000)
    parser.add_argument("--scene-stride", type=int, default=2)
    parser.add_argument("--min-points-per-frame", type=int, default=1000)
    parser.add_argument(
        "--camera-release-settle-s",
        type=float,
        default=2.0,
        help=(
            "quiet time after stopping and releasing the RealSense pipeline; "
            "keeps the following CPU-heavy collision stage away from USB teardown"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _scalar(value: Any) -> np.ndarray:
    return np.asarray(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> np.ndarray:
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if args.capture_frames < 1:
        raise ValueError("--capture-frames must be >= 1")
    if args.capture_timeout_s <= 0 or args.frame_timeout_ms <= 0:
        raise ValueError("capture timeouts must be positive")
    if args.scene_stride < 1 or args.min_points_per_frame < 1:
        raise ValueError("--scene-stride and --min-points-per-frame must be >= 1")
    if (
        not np.isfinite(float(args.camera_release_settle_s))
        or args.camera_release_settle_s < 0.0
        or args.camera_release_settle_s > 10.0
    ):
        raise ValueError("--camera-release-settle-s must be within 0..10 seconds")
    q = np.asarray(args.capture_q_rad, dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("--capture-q-rad must contain seven finite values")
    return q


def _validate_calibration(
    cfg: Dict[str, Any],
    extrinsics: Any,
    expected_serial: str,
    expected_calibration_id: str,
) -> None:
    configured_serial = str(cfg.get("camera", {}).get("serial") or "")
    if configured_serial != expected_serial:
        raise RuntimeError(
            f"configured camera serial {configured_serial or 'missing'} != expected "
            f"{expected_serial}"
        )
    if not bool(extrinsics.calibrated):
        raise RuntimeError("eye-to-hand calibration is required")
    if str(extrinsics.reference_frame) != EXPECTED_REFERENCE_FRAME:
        raise RuntimeError(
            f"calibration reference frame {extrinsics.reference_frame!r} != "
            f"{EXPECTED_REFERENCE_FRAME!r}"
        )
    if str(extrinsics.camera_serial or "") != expected_serial:
        raise RuntimeError(
            f"calibration camera serial {extrinsics.camera_serial or 'missing'} != "
            f"expected {expected_serial}"
        )
    if str(extrinsics.calibration_id or "") != expected_calibration_id:
        raise RuntimeError(
            f"calibration id {extrinsics.calibration_id or 'missing'} != expected "
            f"{expected_calibration_id}"
        )
    quality = getattr(extrinsics, "quality_status", None)
    if quality is not None and str(quality) != "pass":
        raise RuntimeError(f"calibration quality is {quality!r}, expected 'pass'")


def _atomic_save_npz(path: Path, overwrite: bool, arrays: Dict[str, np.ndarray]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output}")

    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
        try:
            directory_fd = os.open(str(output.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture(args: argparse.Namespace) -> Path:
    q_capture = _validate_args(args)
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output}")

    from dynamic_pcd.calibration.io import resolve_extrinsics
    from dynamic_pcd.camera.realsense_camera import RealSenseCamera
    from dynamic_pcd.config import load_config
    from dynamic_pcd.pointcloud.extractor import ObjectPointCloudExtractor

    cfg = load_config(str(config_path))
    extrinsics = resolve_extrinsics(cfg.get("extrinsics"))
    _validate_calibration(
        cfg, extrinsics, args.expected_camera_serial, args.expected_calibration_id
    )

    pointcloud_cfg = dict(cfg["pointcloud"])
    pointcloud_cfg["z_min"] = float(cfg["camera"].get("z_min", 0.25))
    pointcloud_cfg["z_max"] = float(cfg["camera"].get("z_max", 1.20))
    extractor = ObjectPointCloudExtractor(
        pointcloud_cfg, T_base_camera=extrinsics.T_base_camera
    )
    camera = RealSenseCamera(cfg["camera"])

    all_points = []
    all_colors = []
    frame_ids = []
    frame_timestamps = []
    last_frame = None
    capture_started_at_unix_s = 0.0
    capture_completed_at_unix_s = 0.0
    actual_serial = ""
    actual_name = ""

    try:
        camera.start()
        actual_serial = str(camera.device_serial or "")
        actual_name = str(camera.device_name or "")
        if actual_serial != args.expected_camera_serial:
            raise RuntimeError(
                f"opened camera serial {actual_serial or 'unknown'} != expected "
                f"{args.expected_camera_serial}"
            )
        if args.expected_camera_model and args.expected_camera_model not in actual_name:
            raise RuntimeError(
                f"opened camera model {actual_name or 'unknown'} does not contain "
                f"{args.expected_camera_model!r}"
            )

        for _ in range(args.warmup_frames):
            camera.get_frame(timeout_ms=args.frame_timeout_ms)

        capture_started_at_unix_s = time.time()
        deadline = time.monotonic() + args.capture_timeout_s
        while len(frame_ids) < args.capture_frames and time.monotonic() < deadline:
            frame = camera.get_frame(timeout_ms=args.frame_timeout_ms)
            scene = extractor.extract_scene(
                frame, exclude_mask=None, stride=args.scene_stride
            )
            points = np.asarray(scene.points, dtype=np.float32)
            colors = np.asarray(scene.colors, dtype=np.float32)
            if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
                raise RuntimeError(
                    f"invalid scene arrays: points={points.shape}, colors={colors.shape}"
                )
            finite = np.all(np.isfinite(points), axis=1) & np.all(
                np.isfinite(colors), axis=1
            )
            points = points[finite]
            colors = colors[finite]
            if len(points) < args.min_points_per_frame:
                print(
                    f"[capture][skip] frame={frame.frame_id} points={len(points)} "
                    f"< {args.min_points_per_frame}"
                )
                continue
            all_points.append(points)
            all_colors.append(np.clip(colors, 0.0, 1.0))
            frame_ids.append(int(frame.frame_id))
            frame_timestamps.append(float(frame.timestamp))
            last_frame = frame
            print(
                f"[capture] accepted={len(frame_ids)}/{args.capture_frames} "
                f"frame={frame.frame_id} points={len(points)}"
            )
        capture_completed_at_unix_s = time.time()
    finally:
        camera.stop()
        # pyrealsense2 owns native pipeline/profile/sensor handles.  Drop the
        # wrapper and force finalizers before the parent starts the multi-minute
        # HPP-FCL stage, then leave a short quiet period for UVC teardown.  The
        # camera metadata needed below was copied to plain strings above.
        del camera
        gc.collect()
        settle_s = float(args.camera_release_settle_s)
        if settle_s > 0.0:
            print(
                f"[camera] pipeline stopped and released; settling USB for "
                f"{settle_s:.1f}s",
                flush=True,
            )
            time.sleep(settle_s)

    if len(frame_ids) != args.capture_frames or last_frame is None:
        raise RuntimeError(
            f"captured only {len(frame_ids)}/{args.capture_frames} valid frames before timeout"
        )

    points = np.concatenate(all_points, axis=0).astype(np.float32, copy=False)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32, copy=False)
    intr = last_frame.intrinsics
    calibration_source = str(extrinsics.source)
    calibration_path = Path(calibration_source)
    calibration_sha256 = (
        _sha256_file(calibration_path) if calibration_path.is_file() else ""
    )
    captured_at_utc = datetime.fromtimestamp(
        capture_completed_at_unix_s, tz=timezone.utc
    ).isoformat().replace("+00:00", "Z")

    arrays = {
        "artifact_type": _scalar("dexgrasp_live_scene"),
        "schema_name": _scalar("dexgrasp_live_scene"),
        "schema_version": _scalar(SCHEMA_VERSION),
        "scene_points": points,
        "scene_colors": colors,
        "reference_frame": _scalar(EXPECTED_REFERENCE_FRAME),
        "T_reference_camera": np.asarray(
            extrinsics.T_base_camera, dtype=np.float64
        ),
        "calibration_id": _scalar(str(extrinsics.calibration_id)),
        "calibration_source": _scalar(calibration_source),
        "calibration_sha256": _scalar(calibration_sha256),
        "camera_serial": _scalar(actual_serial),
        "camera_name": _scalar(actual_name),
        "capture_q_rad": q_capture,
        "capture_q_source": _scalar("cli_asserted"),
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "frame_timestamps_s": np.asarray(frame_timestamps, dtype=np.float64),
        "frame_count": _scalar(len(frame_ids)),
        # Keep both legacy names and explicit wall-clock names for consumers.
        "capture_started_s": _scalar(capture_started_at_unix_s),
        "capture_completed_s": _scalar(capture_completed_at_unix_s),
        "capture_started_at_unix_s": _scalar(capture_started_at_unix_s),
        "capture_completed_at_unix_s": _scalar(capture_completed_at_unix_s),
        "captured_at_unix_s": _scalar(capture_completed_at_unix_s),
        "captured_at_utc": _scalar(captured_at_utc),
        "scene_stride": _scalar(args.scene_stride),
        "points_per_frame": np.asarray(
            [len(value) for value in all_points], dtype=np.int64
        ),
        "config_path": _scalar(str(config_path)),
        "config_sha256": _scalar(_sha256_file(config_path)),
        "depth_scale": _scalar(float(last_frame.depth_scale)),
        "camera_intrinsics": np.asarray(
            [intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy],
            dtype=np.float64,
        ),
        "camera_intrinsics_model": _scalar(str(intr.model)),
        "camera_distortion": np.asarray(intr.distortion, dtype=np.float64),
        "z_range_m": np.asarray(
            [pointcloud_cfg["z_min"], pointcloud_cfg["z_max"]], dtype=np.float64
        ),
    }
    _atomic_save_npz(output, bool(args.overwrite), arrays)
    print(
        f"[saved] {output} points={len(points)} frames={len(frame_ids)} "
        f"frame={EXPECTED_REFERENCE_FRAME} calibration={extrinsics.calibration_id}"
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        "[safety] perception-only: no Franka/Inspire transport is imported or opened; "
        "--capture-q-rad is metadata only"
    )
    try:
        capture(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
