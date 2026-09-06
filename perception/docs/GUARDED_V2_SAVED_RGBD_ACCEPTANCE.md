# guarded_v2 保存 RGB-D → 正式 128 点验收

这个验收入口补齐旧 RGB-only 回放不能覆盖的 3-D 边界：它读取相机诊断保存的无损
RGB/Z16、逐帧内参、depth scale、固定相机外参和历史 provider mask，但只有 frame 0 的
历史 mask 会成为输入。后续帧不复用历史 mask：

```text
frame 0 保存 mask
  -> 当前 ObjectPCDProvider.initialize_from_mask
  -> provider-owned 本地 SAM2 temporal manager.initialize
frames 1..N-1 保存 RGB-D
  -> 当前 guarded_sam2_primary / unified_three_evidence
  -> 最终发布 mask（invalid 时为空）
  -> 同一帧保存的 Z16
  -> sim2real.observation.model.MaskedRGBDProjector
  -> 固定 128 行 policy point frame
```

CLI 默认不会注入测试 fixture；它复用 provider 与真实 RGB replay 相同的单 worker service
owner，并且 production 验收只允许本次 validator 的 manager 自行启动且拥有的 child
service。即使地址是 `127.0.0.1:5558`，已有服务也会被当成 foreign service 直接拒绝；运行
前应确保该地址空闲。child 启动事件记录 validator PID/child PID/ownership，health 必须返回
实际加载 checkpoint/model-config 的 SHA-256。

## 运行 production 模式

GPU/SAM2 环境稳定后运行：

```bash
cd /home/qiaoguanren/code/franka

SOURCE=dexgrasp/runs/real_mask_alignment_red_ball_20260808-190406.npz
REPORT=/tmp/guarded-v2-saved-rgbd-red-ball.json

.venv/bin/python \
  perception/dynamic_pcd/apps/validate_saved_rgbd_npz.py \
  "$SOURCE" \
  --config perception/configs/d435_default.yaml \
  --rerun-current-provider-guarded-v2 \
  --output "$REPORT"
```

`--output` 使用 exclusive + atomic 发布；目标已存在时拒绝覆盖。默认退出码验收同时要求：

- formal projector `fresh` fraction 为 `1.0`；
- post-seed provider valid fraction 至少 `0.95`；
- 每个 post-seed frame 都有审计到的 temporal `track` 调用；
- 每帧 128 个输出 validity 均有效（full-128 fraction 为 `1.0`）。

最终 JSON 只会在参数、metrics 和全部 checks 计算完成后原子发布，包含
`acceptance_profile/thresholds/production_defaults_used/checks/accepted`。另行给出
`accepted_geometry`、`accepted_pipeline_identity` 与
`accepted_end_to_end_cadence`；顶层 `accepted` 要求三项同时成立，避免把单纯 3-D
projector 绿色误写成 end-to-end production 绿色。

阈值可分别用 `--minimum-fresh-fraction`、
`--minimum-provider-valid-fraction`、`--minimum-sam2-track-call-fraction` 和
`--minimum-full-128-fraction` 调整。production 结论应保留默认严格的 service/full-128
阈值；任一非默认阈值都会自动变为 diagnostic。启用 fixed-sphere shape completion、关闭
mask depth-deviation，或改变默认 point-feature/config 同样是 diagnostic，即使数值通过也
不会得到 production `accepted=true`。

validator 本身先用硬编码 SHA
`b49398c2ac206a252c5a79e93611423e1b96c294b428d98cbe76156b7438d541`
锁定 `guarded_v2_real_rgb_replay_cases.json` acceptance contract，再从该不可自洽修改的
contract 读取并固定 config
`13146712ff82e21e439ef414b4e9d5152fd5ef2d5d54491cdafbdf36049bacd8`、checkpoint
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`、model config
`f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d`；不能通过同时修改
config 与“期望值”绕过。production evidence 只接受 SHA-256
`769179e512615bda5fca49406790a2c23fc9dadde23201e038a1f208c63e542b`
的 60-frame legacy schema-v1 real capture。schema v1 是显式 pinned legacy 证据，不会被
伪称为新 capture schema。archive 必须包含非空且与 active config 完全一致的 camera
serial/calibration、真实 RGB/Z16/scale/K/transform/sensor metadata，并证明 capture 时没有
Franka/RH56 interface 或 robot command write。原 archive 的
`requested_object_mask_mode=guarded`、`effective_mask_publication_mode=adaptive_fusion` 只作
历史一致性参考；本文结论来自当前 `guarded_v2` rerun，绝不把历史 mask 冒充 guarded_v2 GT。

## 产物字段

顶层摘要包括：

- `effective_mask_publication_mode=guarded_sam2_primary` 与
  `effective_recovery_publication_mode=unified_three_evidence`；
- `frame0_temporal_sam2_initialized`；
- provider valid/source/SAM2 status/publication-guard status 与 compute 百分位；
- `sam2_temporal_service` 的 owner、local address、backend、逐操作审计、frame 0 initialize
  次数和 post-seed track coverage；
- projector fresh/full-128 fraction、source point 数、output valid point 数、compute 百分位和
  effective-mask provenance；
- camera serial、calibration ID、源 SHA-256，以及 active config calibration 是否与 archive
  一致。
- canonical effective-config、checkpoint、model-config digest；SAM2 health 的 backend、
  loaded/checkpoint/model config/device/GPU/image-size/AMP/TF32/fill-hole identity；全部 formal
  projector 参数以及逐帧 current-frame mask provenance。
- abstract camera ID、color/depth sensor number、device timestamp 与 retrieved wall cadence
  分开报告。20 Hz publication gate 使用 `camera_retrieved_at_s`：p95≤50 ms、max≤100 ms、
  `>50 ms` fraction≤2%；30→20 采样天然出现的 33/67 ms sensor timestamp 与 sensor gap 只
  作透明 non-gating diagnostic。

`frame_records` 对初始化帧和每个 tracking 帧逐项记录：

- `provider_valid`、`provider_source`、`provider_candidate_source`、`provider_message`；
- `provider_compute_ms`、`provider_timings_ms`、SAM2/guard/tracking-commit 状态；
- `pcd_status`、`pcd_fresh`、`pcd_source_valid_points`、
  `pcd_output_valid_points`、`pcd_compute_ms`；
- projector effective mask 的 kind/source frame/timestamp/area/bbox provenance；
- 同 frame 的 temporal initialize/track 调用次数；
- 与历史 mask 的 IoU（仅 consistency diagnostic，不是 GT，也不参与 provider 输入）。

## 明确的硬件边界

这个入口总是向 provider 显式传入 `_SavedNPZCamera`。它不会构造 RealSense camera，也不会
import/启动 Franka 或 RH56 owner；产物固定写入：

```text
camera_interface_opened=false
camera_hardware_interface_opened=false
realsense_interface_opened=false
franka_interface_opened=false
rh56_interface_opened=false
hardware_interfaces_opened=false
hardware_writes=false
```

本地 SAM2/GPU 是计算服务，不是相机或执行器接口。单测可通过私有 Python 注入点替换为
fake temporal service；这种产物会标记
`actual_local_sam2_service_evaluated=false` 且 health-identity check 失败，不能当作 GPU
acceptance。仅 loopback 地址也不够；backend/checkpoint/model-config/device identity 任一错误
都会使 production evidence fail closed。

## 其他模式与限制

- 无 mode flag：只把历史 provider mask 送到 formal projector。
- `--rerun-current-provider-adaptive`：frame 0 mask 初始化当前 adaptive provider，明确禁用
  SAM2，用于确定性回归，不是 guarded_v2 production 证据。
- `--rerun-current-provider-guarded-v2`：本文的 production mask owner + real saved RGB-D +
  formal projector 路径。

源 NPZ 只保存当时 60 个有效 publication，因此能验证当前路径在这些连续真实 RGB-D 上的
identity/depth/projector 一致性，却不能重建当时未保存的 rejected、完全遮挡或 recovery gap。
frame 0 trusted mask 也绕过 text grounding。archive 没有 capture-time `T_base_palm`，所以点坐标
只验到 `robot_base`，不能宣称 palm-frame observation replay 或机器人闭环验收。
