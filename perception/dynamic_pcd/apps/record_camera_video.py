#!/usr/bin/env python3
"""Record the connected RealSense color stream to an MP4 until Ctrl+C."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Optional, Sequence

import cv2
import numpy as np
import pyrealsense2 as rs

from dynamic_pcd.config import load_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "d435_default.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start recording the configured RealSense color camera immediately; "
            "press Ctrl+C once to finalize and save the MP4."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def record(output: Path, config_path: Path) -> Path:
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    partial = output.with_name(f".{output.stem}.partial.mp4")
    if partial.exists():
        raise FileExistsError(f"partial output already exists: {partial}")

    cfg = load_config(str(config_path.expanduser().resolve()))
    camera_cfg = dict(cfg["camera"])
    width = int(camera_cfg["width"])
    height = int(camera_cfg["height"])
    fps = int(camera_cfg["fps"])
    serial = str(camera_cfg.get("serial") or "")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial:
        rs_config.enable_device(serial)
    rs_config.enable_stream(
        rs.stream.color,
        width,
        height,
        rs.format.bgr8,
        fps,
    )

    writer: Optional[cv2.VideoWriter] = None
    started = False
    frame_count = 0
    started_at = time.monotonic()
    stopped_by_operator = False
    try:
        pipeline.start(rs_config)
        started = True
        writer = cv2.VideoWriter(
            str(partial),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create MP4 file: {partial}")

        print(f"[RECORDING] RealSense color video -> {output}", flush=True)
        print("Press Ctrl+C once to stop and save.", flush=True)
        last_status_second = -1
        try:
            while True:
                frameset = pipeline.wait_for_frames(5000)
                color_frame = frameset.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data())
                if image.shape != (height, width, 3):
                    raise RuntimeError(
                        f"unexpected color frame shape: {image.shape}"
                    )
                writer.write(image)
                frame_count += 1
                elapsed_second = int(time.monotonic() - started_at)
                if elapsed_second != last_status_second:
                    print(
                        f"\r[RECORDING] {elapsed_second}s, "
                        f"{frame_count} frames",
                        end="",
                        flush=True,
                    )
                    last_status_second = elapsed_second
        except KeyboardInterrupt:
            stopped_by_operator = True
            print("\n[STOPPING] Ctrl+C received; finalizing MP4 ...", flush=True)
    finally:
        if writer is not None:
            writer.release()
        if started:
            pipeline.stop()

    if not stopped_by_operator:
        raise RuntimeError("recording ended without an operator stop")
    if frame_count <= 0:
        raise RuntimeError("no color frame was recorded")
    if not partial.is_file() or partial.stat().st_size <= 0:
        raise RuntimeError("MP4 encoder produced no data")
    partial.replace(output)
    print(
        f"[SAVED] {output} frames={frame_count} "
        f"size={output.stat().st_size / 2**20:.1f}MiB",
        flush=True,
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        record(args.output, args.config)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
