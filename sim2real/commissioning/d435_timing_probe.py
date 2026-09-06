#!/usr/bin/env python3
"""Strict D435 timing probe with no Franka or RH56 access.

Starting the camera applies the configured emitter, laser-power, preset, and
auto-exposure options.  This command never imports a robot control adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    WORKSPACE_ROOT / "perception" / "configs" / "d435_default.yaml"
)


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 1:
        return {name: float("nan") for name in ("min", "median", "p95", "max")}
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _diagnostic_float(value: Mapping[str, Any], name: str) -> float:
    raw = value.get(name)
    if raw is None:
        return float("nan")
    result = float(raw)
    return result if np.isfinite(result) else float("nan")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure strict D435 arrival/timestamp timing without accessing "
            "Franka or RH56"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--progress-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    duration_s = float(args.duration)
    progress_s = float(args.progress_s)
    timeout_ms = int(args.timeout_ms)
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration must be finite and positive")
    if not np.isfinite(progress_s) or progress_s <= 0.0:
        raise ValueError("progress-s must be finite and positive")
    if timeout_ms <= 0:
        raise ValueError("timeout-ms must be positive")

    source_path = WORKSPACE_ROOT / "perception"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    from dynamic_pcd.camera.realsense_camera import RealSenseCamera
    from dynamic_pcd.config import load_config

    config = load_config(str(args.config.expanduser().resolve()))
    camera = RealSenseCamera(config["camera"])
    records: list[dict[str, Any]] = []
    started_s = time.monotonic()
    next_progress_s = started_s + progress_s
    try:
        camera.start()
        started_s = time.monotonic()
        next_progress_s = started_s + progress_s
        while time.monotonic() - started_s < duration_s:
            frame = camera.get_frame(timeout_ms=timeout_ms)
            diagnostic = dict(frame.capture_diagnostic or {})
            records.append(
                {
                    "frame_id": int(frame.frame_id),
                    "sensor_frame_number": int(frame.sensor_frame_number),
                    "depth_sensor_frame_number": int(
                        frame.depth_sensor_frame_number
                    ),
                    "capture_timestamp_s": float(frame.timestamp),
                    "retrieved_at_s": float(frame.retrieved_at_s),
                    "retrieved_monotonic_s": float(frame.retrieved_monotonic_s),
                    "host_clock_pair_span_s": float(frame.host_clock_pair_span_s),
                    "timestamp_domain": str(frame.timestamp_domain),
                    "rejected_transport_stale_frames": int(
                        frame.rejected_transport_stale_frames
                    ),
                    "drained_pending_framesets_total": int(
                        diagnostic.get("drained_pending_framesets_total", 0)
                    ),
                    "drained_pending_framesets_before_candidate": int(
                        diagnostic.get(
                            "drained_pending_framesets_before_candidate", 0
                        )
                    ),
                    "last_rejected_transport_age_s": (
                        float("nan")
                        if frame.last_rejected_transport_age_s is None
                        else float(frame.last_rejected_transport_age_s)
                    ),
                    "color_frame_timestamp_metadata": _diagnostic_float(
                        diagnostic, "color_frame_timestamp_metadata"
                    ),
                    "color_sensor_timestamp_metadata": _diagnostic_float(
                        diagnostic, "color_sensor_timestamp_metadata"
                    ),
                    "color_backend_timestamp_metadata": _diagnostic_float(
                        diagnostic, "color_backend_timestamp_metadata"
                    ),
                    "color_time_of_arrival_metadata": _diagnostic_float(
                        diagnostic, "color_time_of_arrival_metadata"
                    ),
                    "depth_frame_timestamp_metadata": _diagnostic_float(
                        diagnostic, "depth_frame_timestamp_metadata"
                    ),
                    "depth_sensor_timestamp_metadata": _diagnostic_float(
                        diagnostic, "depth_sensor_timestamp_metadata"
                    ),
                    "depth_backend_timestamp_metadata": _diagnostic_float(
                        diagnostic, "depth_backend_timestamp_metadata"
                    ),
                    "depth_time_of_arrival_metadata": _diagnostic_float(
                        diagnostic, "depth_time_of_arrival_metadata"
                    ),
                    "capture_diagnostic_json": json.dumps(
                        diagnostic, sort_keys=True
                    ),
                }
            )
            now_s = time.monotonic()
            if now_s >= next_progress_s:
                age_ms = (
                    records[-1]["retrieved_at_s"]
                    - records[-1]["capture_timestamp_s"]
                ) * 1000.0
                print(
                    "[D435 timing] "
                    f"elapsed={now_s - started_s:.1f}s frames={len(records)} "
                    f"last_capture_to_retrieval={age_ms:.3f}ms "
                    "transport_stale_rejections="
                    f"{records[-1]['rejected_transport_stale_frames']} "
                    "drained_pending_framesets="
                    f"{records[-1]['drained_pending_framesets_total']}",
                    file=sys.stderr,
                    flush=True,
                )
                next_progress_s += progress_s
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(
            f"[D435 timing failed] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    finally:
        camera.stop()

    if not records:
        print("[D435 timing failed] no frames collected", file=sys.stderr)
        return 2
    payload: dict[str, np.ndarray] = {}
    for name in records[0]:
        if name == "capture_diagnostic_json" or name == "timestamp_domain":
            payload[name] = np.asarray([record[name] for record in records])
        elif name in {
            "frame_id",
            "sensor_frame_number",
            "depth_sensor_frame_number",
            "rejected_transport_stale_frames",
            "drained_pending_framesets_total",
            "drained_pending_framesets_before_candidate",
        }:
            payload[name] = np.asarray(
                [record[name] for record in records], dtype=np.int64
            )
        else:
            payload[name] = np.asarray(
                [record[name] for record in records], dtype=np.float64
            )
    payload.update(
        {
            "audit_schema": np.asarray("d435_strict_timing_probe_v1"),
            "configured_duration_s": np.asarray(duration_s),
            "configured_timeout_ms": np.asarray(timeout_ms, dtype=np.int32),
            "robot_command_writes": np.asarray(False),
            "camera_configuration_writes": np.asarray(True),
        }
    )
    capture_age = payload["retrieved_at_s"] - payload["capture_timestamp_s"]
    monotonic_interarrival = np.diff(payload["retrieved_monotonic_s"])
    sensor_step = np.diff(payload["sensor_frame_number"])
    summary = {
        "result": "PASS",
        "frames": len(records),
        "elapsed_s": float(time.monotonic() - started_s),
        "capture_to_retrieval_s": _finite_stats(capture_age),
        "host_monotonic_interarrival_s": _finite_stats(monotonic_interarrival),
        "sensor_frame_step": _finite_stats(sensor_step),
        "transport_stale_rejections": int(
            payload["rejected_transport_stale_frames"][-1]
        ),
        "drained_pending_framesets": int(
            payload["drained_pending_framesets_total"][-1]
        ),
        "timestamp_domains": sorted(set(payload["timestamp_domain"].tolist())),
        "robot_command_writes": False,
        "camera_configuration_writes": True,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **payload)
        summary["output"] = str(output)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
