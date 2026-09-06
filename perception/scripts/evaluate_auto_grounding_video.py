#!/usr/bin/env python3
"""Evaluate automatic text grounding when an object enters a saved video.

This is camera-only/offline: it never imports or opens Franka/RH56 code.  The
video is paced at its recorded FPS so the asynchronous GroundingDINO path sees
the same capture pressure as a live camera.  Synthetic constant depth is used
only to exercise source-to-current mask replay; no metric from this script is a
3-D calibration claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_pcd.apps.realtime_masked_pcd import (  # noqa: E402
    prompt_service_launcher_path,
    prompt_service_launcher_args,
    validate_prompting_config,
    validate_prompt_service_health,
    wait_for_prompt_target,
)
from dynamic_pcd.config import load_config  # noqa: E402
from dynamic_pcd.segmentation.prompt_replay import RecentRGBDFrameBuffer  # noqa: E402
from dynamic_pcd.segmentation.prompt_runtime import PromptServiceManager  # noqa: E402
from dynamic_pcd.types import CameraIntrinsics, RGBDFrame  # noqa: E402


class PacedVideoCamera:
    def __init__(self, video_path: Path):
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise RuntimeError("video has no valid FPS")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.started_s = time.monotonic()
        self.frame_id = 0
        self.frames_bgr: dict[int, np.ndarray] = {}

    def get_frame(self) -> RGBDFrame:
        target_s = self.started_s + self.frame_id / self.fps
        remaining_s = target_s - time.monotonic()
        if remaining_s > 0.0:
            time.sleep(remaining_s)
        ok, color = self.capture.read()
        if not ok or color is None:
            raise RuntimeError(
                "video ended before automatic grounding acquired the target"
            )
        frame_id = self.frame_id
        self.frame_id += 1
        self.frames_bgr[frame_id] = color.copy()
        return RGBDFrame(
            color_bgr=color,
            depth_raw=np.full(color.shape[:2], 1000, dtype=np.uint16),
            depth_scale=0.001,
            intrinsics=CameraIntrinsics(
                width=self.width,
                height=self.height,
                fx=float(self.width),
                fy=float(self.width),
                ppx=0.5 * self.width,
                ppy=0.5 * self.height,
            ),
            timestamp=frame_id / self.fps,
            frame_id=frame_id,
        )

    def close(self) -> None:
        self.capture.release()


class SearchProvider:
    def __init__(self, camera: PacedVideoCamera, tracker_cfg: dict):
        self.camera = camera
        self.cfg = {"tracker": tracker_cfg}


def _green_ball_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Evaluation-only color proxy for the green ball in the supplied clip."""

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(
        hsv,
        np.asarray([35, 55, 55], dtype=np.uint8),
        np.asarray([100, 255, 255], dtype=np.uint8),
    )
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(raw, 8)
    best = None
    best_area = 0
    height, width = raw.shape
    for label in range(1, count):
        x, y, box_w, box_h, area = (int(v) for v in stats[label])
        if not (80 <= area <= 5000 and y + box_h >= int(0.45 * height)):
            continue
        aspect = box_w / max(1.0, float(box_h))
        fill = area / max(1.0, float(box_w * box_h))
        if not 0.45 <= aspect <= 2.2 or fill < 0.35:
            continue
        if area > best_area:
            best_area = area
            best = labels == label
    return (
        np.zeros((height, width), dtype=bool)
        if best is None
        else np.asarray(best, dtype=bool)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--object-text", default="green ball")
    parser.add_argument(
        "--detector-backend",
        choices=("grounding_dino", "yolo_world"),
        help="override prompting.detector_backend for an offline A/B replay",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        help="override the prompt service request timeout for offline replay",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/d435_default.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    config = load_config(str(args.config.expanduser().resolve()))
    prompt_cfg = config["prompting"]
    if args.detector_backend is not None:
        prompt_cfg["detector_backend"] = str(args.detector_backend)
    if args.request_timeout_s is not None:
        if not np.isfinite(args.request_timeout_s) or args.request_timeout_s <= 0.0:
            parser.error("--request-timeout-s must be finite and positive")
        prompt_cfg["request_timeout_s"] = float(args.request_timeout_s)
    validate_prompting_config(prompt_cfg)
    camera = PacedVideoCamera(video)
    provider = SearchProvider(camera, dict(config["tracker"]))
    frame_buffer = RecentRGBDFrameBuffer(
        retention_s=float(prompt_cfg.get("replay_buffer_s", 2.0)),
        capacity_frames=max(
            2,
            int(
                np.ceil(
                    float(prompt_cfg.get("replay_buffer_s", 2.0)) * camera.fps
                )
            )
            + 2,
        ),
    )
    manager = PromptServiceManager(
        addr=str(prompt_cfg["service_addr"]),
        autostart=bool(prompt_cfg["service_autostart"]),
        startup_timeout_s=float(prompt_cfg["startup_timeout_s"]),
        request_timeout_ms=int(1000.0 * float(prompt_cfg["request_timeout_s"])),
        launcher_path=str(prompt_service_launcher_path(prompt_cfg)),
        # Match the production launcher: YOLO-World text embeddings and the
        # final one-class head must be hot before video/camera time starts.
        # Without the explicit prompt the service can report READY while the
        # first request still pays several seconds of set_classes warm-up and
        # exceeds the deployed 250 ms RPC deadline.
        launcher_args=prompt_service_launcher_args(
            prompt_cfg, prompt=str(args.object_text)
        ),
    )
    try:
        health = manager.start()
        validate_prompt_service_health(prompt_cfg, health)
        acquired_frame, result = wait_for_prompt_target(
            provider=provider,
            prompt_manager=manager,
            prompt=str(args.object_text),
            frame_buffer=frame_buffer,
            box_threshold=float(prompt_cfg["box_threshold"]),
            text_threshold=float(prompt_cfg["text_threshold"]),
            mask_threshold=float(prompt_cfg["mask_threshold"]),
            top_k=int(prompt_cfg["top_k"]),
            search_interval_s=float(prompt_cfg.get("search_interval_s", 0.5)),
            detector_backend=str(
                prompt_cfg.get("detector_backend", "grounding_dino")
            ),
        )
    finally:
        camera.close()
        manager.close()

    entry_ids = [
        frame_id
        for frame_id, image in camera.frames_bgr.items()
        if int(_green_ball_mask(image).sum()) >= 80
    ]
    entry_frame = None if not entry_ids else min(entry_ids)
    detected_mask = np.asarray(result.mask, dtype=bool)
    proxy_mask = _green_ball_mask(acquired_frame.color_bgr)
    intersection = int(np.logical_and(detected_mask, proxy_mask).sum())
    union = int(np.logical_or(detected_mask, proxy_mask).sum())
    proxy_iou = 0.0 if union == 0 else intersection / float(union)
    source_frame = int(
        result.frame_metadata.get(
            "auto_grounding_source_frame_id", acquired_frame.frame_id
        )
    )
    acquired_id = int(acquired_frame.frame_id)
    summary = {
        "video": str(video),
        "fps": camera.fps,
        "object_text": str(args.object_text),
        "detector_backend": str(
            prompt_cfg.get("detector_backend", "grounding_dino")
        ),
        "evaluation_proxy": "green HSV component; diagnostic only",
        "proxy_entry_frame": entry_frame,
        "grounding_source_frame": source_frame,
        "valid_mask_frame": acquired_id,
        "entry_to_valid_mask_frames": (
            None if entry_frame is None else acquired_id - entry_frame
        ),
        "entry_to_valid_mask_s": (
            None
            if entry_frame is None
            else (acquired_id - entry_frame) / camera.fps
        ),
        "detected_mask_area_px": int(detected_mask.sum()),
        "proxy_mask_area_px": int(proxy_mask.sum()),
        "proxy_mask_iou": proxy_iou,
        "bbox_xyxy": np.asarray(result.bbox_xyxy, dtype=np.int32).tolist(),
        "message": str(result.message),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay = acquired_frame.color_bgr.copy()
    overlay[detected_mask] = (
        0.45 * overlay[detected_mask] + 0.55 * np.asarray([255, 0, 255])
    ).astype(np.uint8)
    x1, y1, x2, y2 = (int(v) for v in summary["bbox_xyxy"])
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if not cv2.imwrite(str(output), overlay):
        raise RuntimeError(f"failed to save overlay: {output}")
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**summary, "overlay": str(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
