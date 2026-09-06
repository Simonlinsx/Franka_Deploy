# guarded_v2 阶段验收：真实视频离线 Mask Benchmark

这个工具只读取保存好的 RGB 视频、二值 Mask 和可选的逐帧耗时记录。它不会导入或
打开 RealSense、Franka、RH56 接口。它适合在修改 guarded_v2 后反复运行同一组回归，
避免每次都靠真机肉眼判断。

## 单命令运行

历史 long-VOS/SAM2 结果已经配置成 baseline：

```bash
cd /home/qiaoguanren/code/franka

OUT="/tmp/guarded_v2-benchmark-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python perception/scripts/benchmark_guarded_v2_masks.py \
  --manifest perception/configs/guarded_v2_benchmark_long_vos_baseline.json \
  --output "$OUT"
```

退出码 `0` 表示所有 case 满足阈值；退出码 `1` 表示程序正常完成但至少一项验收失败；
退出码 `2` 表示输入、帧对齐或文件本身错误。输出包括：

```text
$OUT/summary.json                         # 汇总以及逐 case checks
$OUT/REPORT.md                            # 一页表格
$OUT/<case>/<candidate>/summary.json      # 该 case 的完整证据
$OUT/<case>/<candidate>/frames.csv        # 逐帧证据
$OUT/<case>/<candidate>/worst_*.png       # 最差污染/跳变帧
$OUT/<case>/<candidate>/sparse_gt_*.png   # 人工稀疏参考帧 TP/FP/FN 叠图
```

生产候选不需要改 manifest。按下面的固定接口导出，然后加 `--candidate-root`：

```text
guarded_v2_exports/
├── summary.json
├── fast_green_ball_entry/masks/*.png
├── fast_green_ball_entry/states.jsonl
├── fast_green_ball_entry/summary.json
├── rolling_green_ball/masks/*.png
├── rolling_red_cylinder/masks/*.png
├── static_green_ball_rh56/masks/*.png
└── rh56_heavy_occlusion/masks/*.png
```

production candidate-root 必须来自 replay runner，而不是任意同长度 mask 目录。root 与
五个 case 的 `summary.json`、`states.jsonl` 都是必填；states 每帧只能出现一次，必须按
顺序同时给出精确 `frame_index/source_frame_index/latency_evaluable`。seed 后每个 tick
必须有有限非负 `compute_processing_ms`，缺失、重复、NaN、越界或只留一个 latency 样本
都会直接拒绝。正式验收命令为：

```bash
.venv/bin/python perception/scripts/benchmark_guarded_v2_masks.py \
  --manifest perception/configs/guarded_v2_benchmark_long_vos_baseline.json \
  --candidate-root /path/to/guarded_v2_exports \
  --candidate-name guarded_v2 \
  --output /tmp/guarded_v2-production
```

PNG 必须命名为精确的零起点 `000000.png...`。evaluator 重新计算 30→20 全部 source
indices、source SHA、每张 mask/states SHA、per-case 与 root 聚合 digest；删除中间帧再补
尾帧、移动索引或事后改 mask 都会直接失败。production 还要求恰好固定五 case、realtime
schedule、`guarded_v2 + guarded_sam2_primary + unified_three_evidence`、默认 config/image
size/depth 且 `diagnostic_only=false`。

当前阶段验收链固定为 config
`13146712ff82e21e439ef414b4e9d5152fd5ef2d5d54491cdafbdf36049bacd8` → real-RGB replay manifest
`b49398c2ac206a252c5a79e93611423e1b96c294b428d98cbe76156b7438d541` → benchmark manifest
`886c5202b15fcf40f2aac2d981d072ebc4b26ce09256db7047fc2d82ac76b233`。人工 sparse GT manifest
独立保持 `84aaa5b2017aea1e80a0f52938274e9d691ebc9c46f0eeaa739038adc06f4cbd`；SAM2 checkpoint 与
model-config digest 不随本次 bootstrap 配置修复改变。

单 case 调试必须同时使用 `--diagnostic-partial --case NAME`；旧任意目录只能显式使用
`--diagnostic-legacy-candidate`。两者即使所有数值 checks 通过，也只写
`diagnostic_checks_passed=true`，顶层 `passed` 仍为 false。

旧真实 RGB 视频的当前生产 `guarded_v2` 候选可以直接生成，无需人工整理 mask：

```bash
CANDIDATE="/tmp/guarded-v2-real-rgb-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python perception/scripts/replay_guarded_v2_real_rgb.py \
  --candidate-root "$CANDIDATE"
```

具体 seed provenance、中性合成 depth 的证据边界和硬件零访问说明见
`docs/GUARDED_V2_REAL_RGB_REPLAY.md`。

## 固定真实 case 与 20 Hz 口径

原视频均为 848×480、30 Hz，位于 U 盘的 `test_videos/session_01`。20 Hz 取帧公式为：

```text
source_frame(n) = start + floor(1.5 * n + 0.5)
```

即相对起点依次取 `0, 2, 3, 5, 6, 8, ...`，与之前生成的精确 20 Hz MKV 逐像素一致。

| case | 原视频 | start | 目的 |
|---|---|---:|---|
| `fast_green_ball_entry` | `realsense_20260725_223026.mp4` | 0 | 小球从画外高速进入；完整视频 30→20 Hz |
| `rolling_green_ball` | `realsense_20260725_221851.mp4` | 60 | 滚动绿球 |
| `rolling_red_cylinder` | `realsense_20260725_222343.mp4` | 90 | 滚动红圆柱及离场 |
| `static_green_ball_rh56` | `realsense_20260725_223222.mp4` | 75 | 静态小球，RH56 靠近/遮挡 |
| `rh56_heavy_occlusion` | `realsense_20260725_223251.mp4` | 32 | RH56 重遮挡与接触漂移 |

高速入场和滚动绿球录像在真正目标出现前包含其他绿色运动区域。固定 case 因此还记录了
人工核对的首个目标可见源帧（分别为 `38` 和 `68`）；更早帧按“目标不在画面内”计算，
用于检查 false positive。这个边界只属于离线标注，不会传给候选 tracker。

## 指标与验收阈值

颜色代理只取“确定属于目标”的 HSV 像素，并用时间连续的连通域避免跳到背景同色物体。
因此 `proxy_recall` 是保守可见目标召回，不是人工标注 IoU；`contamination_proxy` 是候选
Mask 落在颜色代理膨胀区域之外的比例，是污染风险代理，不等同于真实 false-positive。

为避免只用 HSV proxy 循环论证，manifest 还引用
`configs/guarded_v2_sparse_ground_truth.json`。其中保存 28 个经过原图和放大叠图逐帧
人工复核的 848×480、0/255 可见目标 Mask：高速入场 6 帧、滚动球 5 帧、滚动圆柱
5 帧、静态球 + RH56 6 帧、重遮挡 6 帧。每条记录包含源视频、源帧、标注方法、复核
备注和 Mask SHA-256；源视频本身也有 SHA-256。运行时任一 digest、帧率取样关系、PNG
二值性或尺寸不符都会直接报错。高速入场的稀疏标签由独立的人工框 GrabCut 草稿复核，
不是拿被评估的历史 SAM2 mask 复制成标签。

稀疏标签只标当前真正可见的目标像素：RH56 挡住的部分不会补全；重遮挡录像没有完全
遮挡帧，所以没有伪造全空 GT。报告额外给出真实 `IoU/recall/precision/contamination`，
而原有 proxy 指标继续负责 dense 时序覆盖：

| case | sparse IoU p05 ≥ | recall p05 ≥ | precision p05 ≥ | contamination p95 ≤ |
|---|---:|---:|---:|---:|
| 高速入场 | 0.65 | 0.75 | 0.75 | 0.25 |
| 滚动球 | 0.75 | 0.85 | 0.85 | 0.15 |
| 滚动圆柱 | 0.72 | 0.82 | 0.82 | 0.18 |
| 静态球 + RH56 | 0.70 | 0.78 | 0.85 | 0.15 |
| RH56 重遮挡 | 0.60 | 0.65 | 0.85 | 0.15 |

有足量人工 sparse GT 时，HSV proxy 的 recall/contamination 只保留为诊断项，不再覆盖
真实 IoU/recall/precision 后决定 PASS；正式 coverage 改由下面的人工可见区间逐帧非空率
和连续断流门槛给出。没有足量人工 GT 的旧 case 仍让 proxy quality 参与 gate，保持旧
manifest 行为。

HSV 连通域在高速运动模糊时也可能选中同色干扰物，因此它不能单独作为“真实位移”权威。
evaluator 支持额外的、整份 JSON 由 SHA-256 固定的
`reviewed_visible_bbox_track_v1`：它必须绑定源视频 SHA、完整 exact-20-Hz source indices，
并为每个 reviewed-visible sampled frame 提供人工复核 bbox 与 centroid。完整轨迹启用后，
正式门控为 centroid error p95≤0.35、max≤0.75，以及连续位移 residual p95≤0.40；三者都
按该帧可见目标 bbox 对角线归一化，原 HSV jump 只保留为诊断。没有配置完整轨迹时，旧
HSV jump 门控原样保留。

先生成不会被 evaluator 接受的候选草稿和 12 帧 contact sheets：

```bash
.venv/bin/python perception/scripts/build_reviewed_motion_track_draft.py \
  --candidate-root /path/to/guarded_v2_exports \
  --case fast_green_ball_entry \
  --output /tmp/fast_green_ball_entry_reviewed_motion_draft.json
```

已有 sparse GT 的帧沿用其人工复核几何；其余记录明确写为
`review_status=needs_human_review`。在人工逐格校正并确认全部记录之前，不得把这个草稿
路径/SHA 写进 benchmark manifest；配置了缺帧、重复、乱序、错误视频/JSON hash 或任何
未复核记录时 evaluator 都会 fail closed。

为避免 HSV proxy 的单帧丢失被误当成真实目标离场，五个固定 case 还在主 manifest 中
保存了人工复核的 `reviewed_visible_source_intervals`、
`reviewed_absent_source_intervals`，以及确有遮挡后重现的
`reviewed_reappearance_source_frames`。报告会独立硬检查：

- 人工确认可见区间内最长连续空 Mask：无遮挡动态 case 不超过 1 tick，RH56 遮挡 case
  不超过 2 tick；
- 人工确认可见区间的逐帧非空 coverage 分别不得低于每个 case 的原 dense coverage
  门槛；
- 重现到首个非空 Mask 不超过 3 个 20 Hz tick；
- 人工确认离场区间的非空 Mask 比例不超过 0.05；
- 有逐帧计算记录时，超过 50 ms 的比例不超过 0.01。

这些指标不会从颜色 proxy 推导，也不会在缺少逐帧 timing 时根据 p50/p95 猜测
`>50 ms` 比例；证据缺失会按阈值不可用而 fail closed。旧 manifest 不含这些字段时仍按
原指标运行，保持兼容。

高速入场 source `0..37` 已并入人工 absent；静态球三个重现点分别由 sampled source
`245/290/308` 的显式人工 absent 帧前导。reappearance 只有在前导 reviewed-absent 帧
确实输出空 mask 后才开始计 tick，持续 hallucination 不再得到 0-tick PASS。滚动绿球没有
诚实的全空人工区间，因此其 proxy absence 明确为 non-gating diagnostic。稀疏 GT manifest
本身也由主 manifest 固定 SHA 和完整五-case 集合；未知 threshold key 或空 checks fail closed。

- `target_coverage`：有可见代理的帧中，候选至少召回 20% 代理像素的比例。
- `proxy_recall p05`：最差 5% 附近的可见目标召回。
- `contamination p95`：最差 5% 附近的外来 Mask 比例。
- `bbox growth p95`：候选 bbox 面积相对前 12 个干净帧中位框的增长。
- `jump p95`：扣除目标颜色代理真实移动后，候选中心残余跳变除以参考框对角线。
- `latency p95`：逐帧处理耗时；20 Hz 的目标阈值保留约 5–10 ms 给相机、点云和 policy。

| case | coverage ≥ | recall p05 ≥ | contamination p95 ≤ | bbox growth p95 ≤ | jump p95 ≤ | latency p95 ≤ |
|---|---:|---:|---:|---:|---:|---:|
| 高速入场 | 0.98 | 0.70 | 0.35 | 1.60 | 0.40 | 45 ms |
| 滚动球 | 0.98 | 0.75 | 0.30 | 1.60 | 0.35 | 45 ms |
| 滚动圆柱 | 0.95 | 0.65 | 0.35 | 1.80 | 0.45 | 45 ms |
| 静态球 + RH56 | 0.95 | 0.65 | 0.30 | 1.60 | 0.35 | 40 ms |
| RH56 重遮挡 | 0.95 | 0.60 | 0.30 | 1.60 | 0.40 | 40 ms |

高速入场、滚动球、滚动圆柱还检查物体不可见期间的非空 Mask 比例不超过 0.05，防止目标
离场后仍把桌面或手当成物体。

`rh56_heavy_occlusion` 经逐帧人工复核以及保守颜色代理核对后，218 个评估帧中目标始终
至少部分可见（最小代理面积仍为 61 px），不存在可诚实标注为“目标完全不可见”的连续
区间。因此该 case 不设置 `absent_false_positive_max`，报告中的 `Absent frames=0`、
`Absent FP=n/a` 只表示录像没有这个测试条件，不能作为完全遮挡 fail-closed 证据。
完全空 mask 的 fail-closed 性质只由 exact-empty 单元测试覆盖，仍需要另录一段目标确实
完全离开画面或被完全遮挡的真实录像才能完成真实证据闭环。

## 能证明什么，不能证明什么

这组旧素材只有 RGB MP4，没有逐帧原始 D435 depth、时间戳和相机内参。因此它可以稳定
回归 2D identity、可见部分 coverage、手部污染、bbox 膨胀、时序跳变以及保存下来的模型
耗时，但不能复现或证明：

- RGB/depth 对齐与 transport age；
- depth 空洞和深度带筛选；
- 2D→3D projector、机器人基座外参；
- workspace/support-plane crop；
- 最终 128 点 policy point cloud 的几何精度和年龄。

所以这个 benchmark PASS 是进入 RGB-D perception-only 测试的必要条件，不是真机闭环的
充分条件。3D 部分仍需用保存的原始 RGB-D case 单独验收；当前 production mask owner →
真实 Z16 → formal 128 点路径见
[`GUARDED_V2_SAVED_RGBD_ACCEPTANCE.md`](GUARDED_V2_SAVED_RGBD_ACCEPTANCE.md)。
