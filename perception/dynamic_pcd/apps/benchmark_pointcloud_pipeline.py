"""Headless CPU benchmark for the live point-cloud publication path.

This intentionally does not open a camera, GUI, ZMQ port, or robot connection.
It measures object back-projection/cleaning/sampling, display-scene cloud
preparation at the configured cadence, and serialization of a representative
published packet.  Camera, segmentation-model, and actual Open3D rendering
latencies still have to pass the live ``--print_fps`` acceptance check.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from dynamic_pcd.config import load_config
from dynamic_pcd.ipc.zmq_pubsub import serialize_object_pcd_packet
from dynamic_pcd.pointcloud.extractor import ObjectPointCloudExtractor
from dynamic_pcd.types import CameraIntrinsics, ObjectPCDPacket, RGBDFrame


def _synthetic_input(width: int, height: int, depth_scale: float = 0.001):
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width * 0.52, height * 0.52
    rx, ry = max(20.0, width * 0.045), max(24.0, height * 0.10)
    mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0)

    depth_raw = np.full((height, width), 800, dtype=np.uint16)
    depth_raw[mask] = 650
    color_bgr = np.empty((height, width, 3), dtype=np.uint8)
    color_bgr[..., 0] = np.asarray(xx % 256, dtype=np.uint8)
    color_bgr[..., 1] = np.asarray(yy % 256, dtype=np.uint8)
    color_bgr[..., 2] = 70
    color_bgr[mask] = [90, 60, 230]

    frame = RGBDFrame(
        color_bgr=color_bgr,
        depth_raw=depth_raw,
        depth_scale=float(depth_scale),
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=0.72 * width,
            fy=1.27 * height,
            ppx=(width - 1) / 2.0,
            ppy=(height - 1) / 2.0,
        ),
        timestamp=0.0,
        frame_id=0,
    )
    return frame, mask.astype(np.uint8)


def _percentile_ms(samples, percentile: float) -> float:
    if not samples:
        return 0.0
    return float(np.percentile(np.asarray(samples) * 1000.0, percentile))


def _make_packet(obj, frame_id: int, history_len: int) -> ObjectPCDPacket:
    history = np.repeat(obj.policy_points[None, ...], history_len, axis=0)
    return ObjectPCDPacket(
        pcd_current=obj.policy_points,
        pcd_history=history,
        center=obj.center,
        velocity=np.zeros(3, dtype=np.float32),
        bbox_xyxy=np.asarray([0, 0, 1, 1], dtype=np.float32),
        timestamp=float(frame_id) / 30.0,
        frame_id=frame_id,
        valid=obj.valid,
        debug={"raw_points": int(len(obj.points))},
        pcd_reference=obj.reference_points,
        reference_frame="robot_base",
        point_frame="object_centered",
        calibration_id="benchmark-only",
        T_base_camera=np.eye(4, dtype=np.float32),
        camera_serial="synthetic",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only benchmark of object/scene PCD preparation and packet serialization"
    )
    parser.add_argument("--config", default="configs/d435_default.yaml")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--scene-stride", type=int, default=None)
    parser.add_argument("--scene-update-every", type=int, default=None)
    parser.add_argument("--min-hz", type=float, default=20.0)
    parser.add_argument(
        "--no-cleaning",
        action="store_true",
        help="Disable voxel/outlier cleaning for an isolated back-projection benchmark",
    )
    parser.add_argument(
        "--no-enforce",
        action="store_true",
        help="Report results without returning a failing exit status below --min-hz",
    )
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be >=1 and --warmup must be >=0")
    if not args.min_hz > 0.0:
        parser.error("--min-hz must be >0")

    cfg = load_config(args.config)
    width = int(cfg["camera"].get("width", 848))
    height = int(cfg["camera"].get("height", 480))
    scene_stride = max(
        1,
        int(
            args.scene_stride
            if args.scene_stride is not None
            else cfg["runtime"].get("scene_stride", 4)
        ),
    )
    scene_every = max(
        1,
        int(
            args.scene_update_every
            if args.scene_update_every is not None
            else cfg["runtime"].get("scene_update_every", 2)
        ),
    )
    pcfg = dict(cfg["pointcloud"])
    pcfg["z_min"] = float(cfg["camera"].get("z_min", 0.25))
    pcfg["z_max"] = float(cfg["camera"].get("z_max", 1.20))
    if args.no_cleaning:
        pcfg["voxel_size"] = 0.0
        pcfg["remove_outliers"] = False

    frame, mask = _synthetic_input(width, height)
    extractor = ObjectPointCloudExtractor(pcfg, T_base_camera=np.eye(4))
    history_len = int(pcfg.get("history_len", 4))
    brightness = float(cfg["runtime"].get("scene_brightness", 0.32))
    color_floor = float(cfg["runtime"].get("scene_color_floor", 0.08))
    stage = defaultdict(list)
    np.random.seed(0)

    total_iterations = args.warmup + args.frames
    measured_start = None
    last_obj = None
    for index in range(total_iterations):
        if index == args.warmup:
            measured_start = time.perf_counter()
        record = index >= args.warmup
        iteration_start = time.perf_counter()

        t0 = time.perf_counter()
        obj = extractor.extract(frame, mask)
        if not obj.valid:
            raise RuntimeError(f"synthetic object extraction failed: {obj.message}")
        if record:
            stage["object"].append(time.perf_counter() - t0)

        if index % scene_every == 0:
            t0 = time.perf_counter()
            scene = extractor.extract_scene(
                frame, exclude_mask=mask, stride=scene_stride
            )
            # Include the NumPy conversion/color work done by the Open3D
            # composite path, but not the platform-dependent renderer itself.
            np.asarray(scene.points, dtype=np.float64)
            rgb = np.asarray(scene.colors, dtype=np.float64)
            np.clip(rgb * brightness + color_floor, 0.0, 1.0)
            np.asarray(obj.points, dtype=np.float64)
            np.empty((len(obj.points), 3), dtype=np.float64)[:] = [1.0, 0.12, 0.02]
            if record:
                stage["scene_prepare"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        packet = _make_packet(obj, index, history_len)
        payload = serialize_object_pcd_packet(packet)
        if not payload:
            raise RuntimeError("packet serialization produced an empty payload")
        if record:
            stage["serialize"].append(time.perf_counter() - t0)
            stage["iteration"].append(time.perf_counter() - iteration_start)
        last_obj = obj

    measured_elapsed = time.perf_counter() - float(measured_start)
    measured_hz = args.frames / max(measured_elapsed, 1e-9)
    status = "PASS" if measured_hz >= args.min_hz else "FAIL"
    print(
        f"[Benchmark] {width}x{height}, frames={args.frames}, warmup={args.warmup}, "
        f"scene_stride={scene_stride}, scene_every={scene_every}, "
        f"cleaning={not args.no_cleaning}"
    )
    for name in ("object", "scene_prepare", "serialize", "iteration"):
        samples = stage[name]
        print(
            f"[Stage] {name:13s} p50={_percentile_ms(samples, 50):7.3f}ms "
            f"p95={_percentile_ms(samples, 95):7.3f}ms n={len(samples)}"
        )
    print(
        f"[Result] cpu_pipeline={measured_hz:.2f}Hz "
        f"target>={args.min_hz:.2f}Hz {status} "
        f"object_points={len(last_obj.points) if last_obj is not None else 0}"
    )
    print(
        "[Scope] CPU PCD + scene-array preparation + exact packet serialization; "
        "run the camera app with --print_fps to accept tracker/GPU/GUI performance."
    )
    return 0 if status == "PASS" or args.no_enforce else 2


if __name__ == "__main__":
    raise SystemExit(main())
