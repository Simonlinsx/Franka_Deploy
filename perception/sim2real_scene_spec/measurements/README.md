# Measurements

这个目录保存相机只读测量结果。当前三份 `table_plane_roi_*.yaml` 是独立原始结果，`table_plane_summary.yaml` 是经审核的临时共识表面；工具参数仍没有被臆测为实测值。

建议每次测量使用不同文件名并保留原始结果，例如：

```text
table_plane_left_20260717.yaml
table_plane_center_20260717.yaml
table_plane_right_20260717.yaml
```

不要用 `--force` 覆盖唯一一份原始测量。当前汇总已把路径、时间和 SHA-256 写入 `scene_manifest.json`；相机、台面或机器人基座改变后必须生成新的一组文件和新汇总，不能复用旧平面。

桌面工具只访问 D435：

```bash
python ../tools/measure_table_plane.py --self-test

python ../tools/measure_table_plane.py \
  --select-roi \
  --frames 30 \
  --output table_plane_center_YYYYMMDD.yaml \
  --confirm-camera-only
```

正式采用前检查输出中的 `review_checklist`。尤其注意 `sampled_inlier_bounds_robot_base_m` 只是采样覆盖，不是桌面物理边界。
