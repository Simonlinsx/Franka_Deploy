# AnyDexGrasp + Franka FR3 + Inspire RH56

这里实现 D435 点云 → AnyDexGrasp 候选 → 自动可执行候选选择 → 实时预览 →
FR3 + RH56 执行。

> 当前安装的是 Bambu PLA 适配器。本流程只执行低速、无接触、无载荷、
> 无 lift 的空抓往返；接触抓取和提起物体仍不在本入口范围内。

## 日常使用：只运行一条命令

```bash
cd /home/qiaoguanren/code/franka/dexgrasp
./scripts/run_anydex_grasp_once.sh
```

程序会在终端集中确认一次现场安全条件，然后自动完成：

1. RH56 张开并回到 default；
2. Franka 低速回到 default；
3. 打开 D435 窗口，由用户框选物体 bbox；
4. SAM2 分割物体并生成 object point cloud；
5. GPU AnyDexGrasp 生成 candidates；
6. 按分数顺序自动检查 RH56 寄存器范围、Franka IK、关节限位和自碰撞，
   选择第一个真正可执行的 candidate；
7. 自动生成新鲜场景碰撞审计并打开实时预览；
8. 前台执行并在结束时停止 Franka、禁用 RH56。

`SNAPSHOT`、`CANDIDATE`、审计目录和 `SESSION` 仍会保存用于复现，但都是内部
文件，不需要用户复制、设置或再次运行命令。

当前一体化执行后端仍是低速无接触空抓。完整的接触抓取 + lift 使用相同入口：

```bash
./scripts/run_anydex_grasp_once.sh --execution-mode lift --lift-height-m 0.05
```

但在 profile 填入物体质量/质心/惯量和适配器承载批准前，它会在任何真机动作
之前拒绝；D435 点云只能计算几何，不能测出物体质量。

只检查完整编排、不连接设备：

```bash
./scripts/run_anydex_grasp_once.sh --dry-run
```

## 自动候选选择为什么不能直接取第 0 个

AnyDex 分数只评价抓取，不代表安装后的 FR3 一定能到达。当前保存结果的
candidate 0 为：

```text
hand targets = [798, 798, 798, 798, 978, 450]
```

`q6=450` 已包含在现有 RL sim2real 真机证据覆盖的 `416..1000` 范围内。
失败原因实际是 candidate 0 的 FR3 目标姿态不可达，而不是 q6。自动选择器会继续
检查后续候选；这组数据中最高分的可执行结果是 candidate 32，candidate 41 也
通过。direct profile 接受的范围是前五轴 `0..1000`、q6 `416..1000`，不再
要求为每个新六轴向量单独跑 Stage1/Stage2。

## 复位

只复位 RH56（会写寄存器）：

```bash
PROFILE=configs/fr3_rh56_v7_sim2real_supervised.json

./scripts/reset_installed_rh56_open.sh run \
  --config "$PROFILE" \
  --confirm-installed RH56_INSTALLED_ON_FR3 \
  --confirm-24v-cutoff RH56_24V_CUTOFF_READY \
  --confirm-franka-stop FR3_STOP_READY \
  --confirm-workspace-clear INSTALLED_AIR_WORKSPACE_CLEAR \
  --confirm-no-contact PLA_LOW_SPEED_NO_CONTACT \
  --confirm-reset-open RH56_RESET_OPEN
```

RH56 已张开且禁用后，只复位 Franka（会移动机械臂）：

```bash
./scripts/reset_franka_default.sh \
  --config "$PROFILE" \
  --confirm-installed RH56_INSTALLED_ON_FR3 \
  --confirm-hand-open RH56_OPEN_DISABLED \
  --confirm-workspace-clear FR3_RH56_CURRENT_TO_DEFAULT_SWEEP_CLEAR \
  --confirm-stop-ready FR3_STOP_READY \
  --confirm-pla-low-speed PLA_LOW_SPEED_UNLOADED_ONLY
```

## 只读检查

下面只读设备状态，不创建 Franka 控制器，也不写 RH56 寄存器：

```bash
./scripts/run_control_preflight.sh \
  --config configs/fr3_rh56_v7_sim2real_supervised.json \
  --read-hardware
```

## 常见问题

| 现象 | 说明 |
| --- | --- |
| `q6=450 has no exact commissioning evidence` | 不应再出现；确认使用 `fr3_rh56_v7_sim2real_supervised.json`，不要使用旧 V7 profile |
| 自动选择跳过 candidate 0 | 正常；AnyDex 分数最高不等于 FR3 IK 可达，程序会继续检查后续候选 |
| viewer 没有 `READY` | 执行器不会启动；检查 D435、`DISPLAY` 和 Open3D |
| 实时点云与 object cloud 错位 | 不执行，重新采集/检查 eye-to-hand 标定 |
| `STOP UNCONFIRMED` | 立即使用 Franka 物理停止并切断 RH56 24 V |

## 当前设备与证据

- D435：`337322072188`，`848x480@30`；
- eye-to-hand 标定：`eye-to-hand-b722bce10485c8a3`；
- direct profile：`configs/fr3_rh56_v7_sim2real_supervised.json`；
- q6 sim2real profile：`configs/fr3_rh56_v94_seq286_20hz_commissioned.json`；
- q6/coupled evidence：`runs/rh56_policy_seq286_20hz_coupled_20260724_codex01.json`。

更多设计与历史信息见 [README_LEGACY.md](README_LEGACY.md)、
[连续 telemetry 合同](docs/continuous_telemetry_contract.md) 和
[安装工具碰撞审计](docs/installed_collision_backend.md)。
