# guarded_v2 旧真实 RGB 回放候选

这个 runner 把旧的真实测试 MP4 按固定的 30→20 Hz 序列送入当前生产
`ObjectPCDProvider + temporal SAM2 + guarded_v2 最终发布守卫`，导出可直接交给
`benchmark_guarded_v2_masks.py` 的候选：

```text
candidate-root/
├── summary.json
├── fast_green_ball_entry/masks/*.png
├── fast_green_ball_entry/states.jsonl
├── fast_green_ball_entry/summary.json
├── rolling_green_ball/masks/*.png
├── rolling_red_cylinder/masks/*.png
├── static_green_ball_rh56/masks/*.png
└── rh56_heavy_occlusion/masks/*.png
```

它不会打开 D435、Franka 或 RH56。旧 MP4 没有 depth，所以 runner 显式使用全图
`1.0 m` 的中性合成深度，仅用于验证：

- 2-D mask identity、漂移和手部污染；
- SAM2-primary/LOST/recovery 与最终 fail-closed 的生产状态机；
- 不含 20 Hz 等待时间的单帧计算耗时。

它不能证明 depth 对齐、深度空洞、3-D 点云、相机外参、support plane/workspace crop
或控制安全。它目前只回放 `ObjectPCDProvider` 内部路径，没有启动部署外层的
object-text prompt owner，因此也不能证明目标丢失/离场后的类别重检与 auto-reacquire。
每个 `summary.json` 都会固定写入这些限制，不能把本回放称为 3-D 或完整自动
grounding 验收。

## 生成候选

```bash
cd /home/qiaoguanren/code/franka

CANDIDATE="/tmp/guarded-v2-real-rgb-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python perception/scripts/replay_guarded_v2_real_rgb.py \
  --candidate-root "$CANDIDATE"
```

默认严格按 20 Hz wall schedule 送帧，但 schedule wait 位于计时区间外。只调试文件接口时
可加 `--as-fast-as-possible`；该模式固定标记 diagnostic。多个 `--case NAME`、
`--sam2-image-size`、非默认 config、非 `1.0 m` depth 或 semantic-direct 都会写入
`production_acceptance_eligible=false`，不能形成 production PASS。

为了区分“SAM2 本身跟踪失败”和“adaptive/recovery gate 拒绝了正确
SAM2 mask”，可做一次仅诊断 A/B：

```bash
.venv/bin/python perception/scripts/replay_guarded_v2_real_rgb.py \
  --candidate-root /tmp/guarded-v2-semantic-direct \
  --as-fast-as-possible \
  --diagnostic-semantic-sam2-direct
```

默认候选会显式覆盖为 `guarded_sam2_primary`，与部署端
`--object-mask-mode guarded_v2` 的有效 provider 模式一致，不再使用旧的
`adaptive_fusion` 默认值。上述诊断开关则让 temporal SAM2 mask 直接发布。它的
`summary.json` 固定标记 `diagnostic_only_not_production_acceptance=true`，
即使该 A/B 表现更好也不能当作 guarded_v2 生产验收 PASS。

`states.jsonl` 对 pre-seed 和 bbox/SAM2 初始化帧写入
`latency_evaluable=false`，只对 seed 之后真正进入 rollout 的 provider tick 写入
`latency_evaluable=true`。candidate-root benchmark 使用 `compute_processing_ms`，因此
20 Hz latency 不包含调度 sleep、pre-seed 的 0 ms 占位值或一次性初始化耗时。

生产 manifest 对五个 source MP4 的 SHA-256/FPS/848×480/frame-count 全部固定；高速入场
的 seed mask video 也固定同样四项，并强制 `seed.frame_index == seed_source_frame`，禁止
未来帧 lookahead。每个 case summary 记录原始 config digest、canonical effective config
digest、SAM2 checkpoint/model-config digest，以及逐 mask SHA、states SHA 和聚合 digest；
root summary 再固定五个 per-case summary/digest 链。任一文件、索引或摘要变化都会被下游
evaluator 拒绝。

本阶段 production provenance 固定为：

- `d435_default.yaml`：`13146712ff82e21e439ef414b4e9d5152fd5ef2d5d54491cdafbdf36049bacd8`；
- `guarded_v2_real_rgb_replay_cases.json`：`b49398c2ac206a252c5a79e93611423e1b96c294b428d98cbe76156b7438d541`；
- `guarded_v2_benchmark_long_vos_baseline.json`：`886c5202b15fcf40f2aac2d981d072ebc4b26ce09256db7047fc2d82ac76b233`；
- sparse GT manifest 保持 `84aaa5b2017aea1e80a0f52938274e9d691ebc9c46f0eeaa739038adc06f4cbd`。

任何上游配置或 manifest 改动都必须从 config 开始重新计算整条链，不能只改下游期望值。

高速入场 case 保留 source frame 38 为真实首次可见时刻，但 seed 严格复用已验证自动
grounding 在 source frame 39 产生的二值 mask（1 个 30 Hz 帧的检出延迟）；此前同色视频
中的动态绿块明确视为背景。滚动绿球从 source frame 69 的真实球建立 seed，不复用已知错误的
long-VOS 右上角背景 mask。所有 `states.jsonl` 均记录原始 source frame、seed provenance、
最终 mask source、SAM2 状态和逐阶段耗时。

## 运行验收

```bash
REPORT="/tmp/guarded-v2-report-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python perception/scripts/benchmark_guarded_v2_masks.py \
  --manifest perception/configs/guarded_v2_benchmark_long_vos_baseline.json \
  --candidate-root "$CANDIDATE" \
  --candidate-name guarded_v2_production_rgb_neutral_depth \
  --output "$REPORT"
```

不要只看 `valid_fraction`：错误目标也可能持续输出 `valid=True`。正式判断使用 benchmark
的 coverage、proxy recall、contamination、bbox growth、jump 和 latency，并检查
`worst_*.png`。
