from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dynamic_pcd.config import load_config
from dynamic_pcd.camera.realsense_camera import RealSenseCamera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/d435_default.yaml")
    parser.add_argument("--out", type=str, default="d435_sample_colored.ply")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cam = RealSenseCamera(cfg["camera"])
    cam.start()
    try:
        frame = cam.get_frame()
        import open3d as o3d
        depth = frame.depth_m
        h, w = depth.shape
        vv, uu = np.indices((h, w))
        z = depth.reshape(-1)
        valid = np.isfinite(z) & (z > float(cfg["camera"].get("z_min", 0.25))) & (z < float(cfg["camera"].get("z_max", 1.2)))
        u = uu.reshape(-1).astype(np.float32)
        v = vv.reshape(-1).astype(np.float32)
        x = (u - frame.intrinsics.ppx) / frame.intrinsics.fx * z
        y = (v - frame.intrinsics.ppy) / frame.intrinsics.fy * z
        pts = np.stack([x, y, z], axis=1)[valid]
        colors = frame.color_bgr.reshape(-1, 3)[valid][:, ::-1].astype(np.float32) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_point_cloud(args.out, pcd)
        print(f"[Saved] {args.out}, points={len(pts)}")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
