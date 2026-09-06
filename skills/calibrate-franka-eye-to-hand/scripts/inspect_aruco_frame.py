#!/usr/bin/env python3
"""Capture one sharp RealSense color frame and inventory visible ArUco markers.

This is a read-only commissioning diagnostic.  It never opens the Franka FCI
connection and never commands robot motion.  The raw image is kept separate
from the annotated image so it can be reprocessed without overlay bias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from dynamic_pcd.camera.realsense_camera import RealSenseCamera
from dynamic_pcd.config import load_config

DEFAULT_DICTIONARIES = (
    "DICT_ARUCO_ORIGINAL",
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
)


def _detector(dictionary: Any) -> Any:
    aruco = cv2.aruco
    parameters = aruco.DetectorParameters()
    if hasattr(parameters, "cornerRefinementMethod"):
        # Match the calibration detector's preference order.  This target has
        # a physically damaged corner for which SUBPIX was observed to jump;
        # APRILTAG refinement remained stable during the rigidity audit.
        for method_name in (
            "CORNER_REFINE_APRILTAG",
            "CORNER_REFINE_CONTOUR",
            "CORNER_REFINE_SUBPIX",
        ):
            method = getattr(aruco, method_name, None)
            if method is not None:
                parameters.cornerRefinementMethod = method
                break
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def _detect(detector: Any, gray: np.ndarray) -> Any:
    if hasattr(detector, "detectMarkers"):
        return detector.detectMarkers(gray)
    dictionary, parameters = detector
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def _marker_record(
    dictionary_name: str,
    marker_id: int,
    corners: np.ndarray,
    width: int,
    height: int,
) -> Dict[str, Any]:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    margins = np.column_stack(
        (
            points[:, 0],
            width - 1.0 - points[:, 0],
            points[:, 1],
            height - 1.0 - points[:, 1],
        )
    )
    return {
        "dictionary": dictionary_name,
        "marker_id": int(marker_id),
        "corners_px": points.tolist(),
        "center_px": points.mean(axis=0).tolist(),
        "edge_lengths_px": edge_lengths.tolist(),
        "edge_mean_px": float(edge_lengths.mean()),
        "edge_min_px": float(edge_lengths.min()),
        "edge_max_px": float(edge_lengths.max()),
        "polygon_area_px2": float(abs(cv2.contourArea(points.astype(np.float32)))),
        "minimum_image_margin_px": float(margins.min()),
        "complete_in_image": bool(margins.min() > 0.0),
    }


def inspect_frame(
    image_bgr: np.ndarray, dictionary_names: Iterable[str]
) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    records: List[Dict[str, Any]] = []
    for name in dictionary_names:
        dictionary_id = getattr(cv2.aruco, name, None)
        if dictionary_id is None:
            continue
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        corners, ids, _rejected = _detect(_detector(dictionary), gray)
        if ids is None:
            continue
        for marker_corners, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
            records.append(
                _marker_record(name, int(marker_id), marker_corners, width, height)
            )
    return records


def _draw_records(
    image_bgr: np.ndarray, records: Iterable[Dict[str, Any]]
) -> np.ndarray:
    output = image_bgr.copy()
    for record in records:
        points = np.round(np.asarray(record["corners_px"])).astype(np.int32)
        cv2.polylines(output, [points.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        origin = tuple(points[0].tolist())
        label = "{} ID {} {:.0f}px".format(
            record["dictionary"], record["marker_id"], record["edge_mean_px"]
        )
        cv2.putText(
            output,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--annotated-output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--dictionary", action="append", dest="dictionaries")
    args = parser.parse_args()

    if args.frames < 1:
        raise ValueError("--frames must be at least 1")
    for output in (args.raw_output, args.annotated_output):
        if output.exists():
            raise FileExistsError("refusing to overwrite {}".format(output))
        output.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg["camera"]["serial"] = str(args.camera_serial)
    camera = RealSenseCamera(cfg["camera"])
    best_frame = None
    best_focus = -1.0
    try:
        camera.start()
        if camera.device_serial != str(args.camera_serial):
            raise RuntimeError(
                "requested serial {}, opened {}".format(
                    args.camera_serial, camera.device_serial
                )
            )
        for _ in range(args.frames):
            frame = camera.get_frame()
            gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
            focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if focus > best_focus:
                best_focus = focus
                best_frame = frame
    finally:
        camera.stop()

    if best_frame is None:
        raise RuntimeError("camera produced no valid frames")
    dictionaries = tuple(args.dictionaries or DEFAULT_DICTIONARIES)
    records = inspect_frame(best_frame.color_bgr, dictionaries)
    annotated = _draw_records(best_frame.color_bgr, records)
    if not cv2.imwrite(str(args.raw_output), best_frame.color_bgr):
        raise OSError("failed to write {}".format(args.raw_output))
    if not cv2.imwrite(str(args.annotated_output), annotated):
        raise OSError("failed to write {}".format(args.annotated_output))

    payload = {
        "camera_name": camera.device_name,
        "camera_serial": camera.device_serial,
        "frame_id": int(best_frame.frame_id),
        "depth_scale": float(best_frame.depth_scale),
        "intrinsics": best_frame.intrinsics.to_dict(),
        "image_width": int(best_frame.intrinsics.width),
        "image_height": int(best_frame.intrinsics.height),
        "fx": float(best_frame.intrinsics.fx),
        "fy": float(best_frame.intrinsics.fy),
        "focus_laplacian_variance": best_focus,
        "raw_output": str(args.raw_output.resolve()),
        "annotated_output": str(args.annotated_output.resolve()),
        "detections": records,
    }
    # RealSenseCamera intentionally prints startup diagnostics.  Emit one
    # machine-readable sentinel line so a headless caller never has to guess
    # where a pretty-printed JSON object begins.
    print(
        "ARUCO_FRAME_INSPECTION_JSON="
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
