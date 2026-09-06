from __future__ import annotations

import argparse
import os

from dynamic_pcd.utils.gui_env import (
    repair_gui_env_after_cv2_import,
    setup_gui_env,
)
setup_gui_env()
from datetime import datetime

import cv2
import numpy as np

repair_gui_env_after_cv2_import()

from dynamic_pcd.config import load_config
from dynamic_pcd.camera.realsense_camera import RealSenseCamera, PRESET_MAP
from dynamic_pcd.utils.vis import put_lines


WIN = "D435 Depth Viewer"


def nothing(_):
    pass


def colorize_depth(depth_m: np.ndarray, z_min: float, z_max: float, cmap_id: int, invert: bool = False):
    valid = np.isfinite(depth_m) & (depth_m > z_min) & (depth_m < z_max)
    norm = np.zeros_like(depth_m, dtype=np.float32)
    norm[valid] = (depth_m[valid] - z_min) / max(1e-6, z_max - z_min)
    norm = np.clip(norm, 0, 1)
    if invert:
        norm[valid] = 1.0 - norm[valid]
    u8 = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cmap_id)
    vis[~valid] = (0, 0, 0)
    return vis, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/d435_default.yaml")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--z_min", type=float, default=None)
    parser.add_argument("--z_max", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default="depth_saves")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ccfg = cfg["camera"]
    if args.width is not None:
        ccfg["width"] = args.width
    if args.height is not None:
        ccfg["height"] = args.height
    if args.fps is not None:
        ccfg["fps"] = args.fps
    if args.z_min is not None:
        ccfg["z_min"] = args.z_min
    if args.z_max is not None:
        ccfg["z_max"] = args.z_max

    os.makedirs(args.save_dir, exist_ok=True)

    cam = RealSenseCamera(ccfg)
    cam.start()
    depth_sensor = cam.depth_sensor
    rs = cam.rs

    try:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, int(ccfg["width"]), int(ccfg["height"]))
        cv2.createTrackbar("z_min_mm", WIN, int(float(ccfg["z_min"]) * 1000), 3000, nothing)
        cv2.createTrackbar("z_max_mm", WIN, int(float(ccfg["z_max"]) * 1000), 3000, nothing)
        cv2.createTrackbar("spatial", WIN, int(bool(ccfg.get("spatial_filter", False))), 1, nothing)
        cv2.createTrackbar("temporal", WIN, int(bool(ccfg.get("temporal_filter", False))), 1, nothing)
        cv2.createTrackbar("hole", WIN, int(bool(ccfg.get("hole_filter", False))), 1, nothing)
        cv2.createTrackbar("invert", WIN, 0, 1, nothing)
        colormaps = [
            ("JET", cv2.COLORMAP_JET),
            ("TURBO", cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET),
            ("INFERNO", cv2.COLORMAP_INFERNO),
            ("PLASMA", cv2.COLORMAP_PLASMA),
            ("VIRIDIS", cv2.COLORMAP_VIRIDIS),
            ("BONE", cv2.COLORMAP_BONE),
        ]
        cv2.createTrackbar("colormap", WIN, 1, len(colormaps) - 1, nothing)

        laser_supported = depth_sensor is not None and depth_sensor.supports(rs.option.laser_power)
        if laser_supported:
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            cv2.createTrackbar("laser", WIN, int(ccfg.get("laser_power", 180)), int(rng.max), nothing)
        last_laser = None

        print("[Keys] q/ESC quit | s save | r reset range | 1/2/3 toggle filters")
        while True:
            ccfg["spatial_filter"] = cv2.getTrackbarPos("spatial", WIN) == 1
            ccfg["temporal_filter"] = cv2.getTrackbarPos("temporal", WIN) == 1
            ccfg["hole_filter"] = cv2.getTrackbarPos("hole", WIN) == 1
            if laser_supported:
                laser = cv2.getTrackbarPos("laser", WIN)
                if laser != last_laser:
                    try:
                        depth_sensor.set_option(rs.option.laser_power, float(laser))
                        last_laser = laser
                    except Exception as e:
                        print(f"[WARN] set laser failed: {e}")

            frame = cam.get_frame()
            depth_m = frame.depth_m
            z_min = cv2.getTrackbarPos("z_min_mm", WIN) / 1000.0
            z_max = cv2.getTrackbarPos("z_max_mm", WIN) / 1000.0
            if z_max <= z_min:
                z_max = z_min + 0.001
            cmap_idx = cv2.getTrackbarPos("colormap", WIN)
            invert = cv2.getTrackbarPos("invert", WIN) == 1
            vis, valid = colorize_depth(depth_m, z_min, z_max, colormaps[cmap_idx][1], invert)
            vals = depth_m[valid]
            if vals.size:
                stats = f"p1/med/p99={np.percentile(vals,1)*1000:.0f}/{np.median(vals)*1000:.0f}/{np.percentile(vals,99)*1000:.0f}mm"
            else:
                stats = "no valid depth"
            lines = [
                f"D435 depth {ccfg['width']}x{ccfg['height']}@{ccfg['fps']} frame={frame.frame_id}",
                f"range={z_min:.2f}-{z_max:.2f}m valid={valid.mean()*100:.1f}% {stats}",
                f"filters spatial={int(ccfg['spatial_filter'])} temporal={int(ccfg['temporal_filter'])} hole={int(ccfg['hole_filter'])}",
                f"colormap={colormaps[cmap_idx][0]} laser={last_laser}",
            ]
            vis = put_lines(vis, lines)
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_path = os.path.join(args.save_dir, f"depth_raw_{ts}.png")
                vis_path = os.path.join(args.save_dir, f"depth_vis_{ts}.png")
                npy_path = os.path.join(args.save_dir, f"depth_m_{ts}.npy")
                cv2.imwrite(raw_path, frame.depth_raw)
                cv2.imwrite(vis_path, vis)
                np.save(npy_path, depth_m)
                print(f"[Saved] {raw_path}, {vis_path}, {npy_path}")
            elif key == ord("r"):
                cv2.setTrackbarPos("z_min_mm", WIN, 250)
                cv2.setTrackbarPos("z_max_mm", WIN, 1200)
            elif key == ord("1"):
                cv2.setTrackbarPos("spatial", WIN, 1 - cv2.getTrackbarPos("spatial", WIN))
            elif key == ord("2"):
                cv2.setTrackbarPos("temporal", WIN, 1 - cv2.getTrackbarPos("temporal", WIN))
            elif key == ord("3"):
                cv2.setTrackbarPos("hole", WIN, 1 - cv2.getTrackbarPos("hole", WIN))
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
