# Tuning guide

## First settings for dynamic grasping

```yaml
camera:
  width: 848
  height: 480
  fps: 30
  z_min: 0.25
  z_max: 1.20
  spatial_filter: false
  temporal_filter: false
  hole_filter: false

tracker:
  roi_scale: 1.8
  depth_tolerance: 0.08
  min_area: 80

pointcloud:
  erode_kernel: 3
  voxel_size: 0.003
  num_points: 1024
```

## If mask includes table/background

- Use `init_method: grabcut`.
- Make the initial ROI tight around the object.
- Decrease `depth_tolerance`, e.g. `0.08 -> 0.04`.
- Increase `erode_kernel`, e.g. `3 -> 5`.
- Add `workspace_min/max` in base frame.

## If tracking is lost

- Increase `roi_scale`, e.g. `1.8 -> 2.5`.
- Increase `depth_tolerance`, e.g. `0.08 -> 0.12`.
- Decrease `min_area` for small objects.
- Enable SAM2 low-frequency re-init only after ROI-depth works.

## If point cloud is noisy

- Keep temporal depth filter off for moving objects.
- Use `erode_kernel: 3-5` to remove edge depth shadows.
- Use `voxel_size: 0.003-0.005`.
- Use statistical outlier removal.
- Avoid reflective/transparent objects for first tests.

## If FPS is low

- Run without Open3D viewer: `--no_show_pcd`.
- Reduce visualization FPS by not displaying every frame.
- Use `num_points: 1024` first.
- Use D435 `848x480@30` before trying 60 FPS.
- Do not run every-frame SAM2.
