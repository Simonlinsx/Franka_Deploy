# RH56 candidate 51 验收状态与最小分阶段链

本文件只陈述当前仓库中可复核的证据。`PASS`、配置中的范围声明、离线单元测试和
AnyDex 候选都不能互相替代。这里的命令分成“完全离线”“只读真机”和“会产生
RH56 动作”三类；任何动作命令都必须在当次现场重新确认后运行。

## 1. 当前可证明的状态

绑定快照：

- `runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz`
- file SHA-256：`92c44aff86e73229c7c2a454c0cf3ffce714580d1895dc98d98759e3711b6395`
- candidate index：`51`（source `54`、type `7`、score `0.9209079`）
- RH56 六轴寄存器目标：`[0, 358, 799, 911, 922, 646]`
- 轴顺序：`pinky, ring, middle, index, thumb_bend, thumb_rotate`

当前 V7 profile 的 file SHA-256 是
`132c69a63fd1264b8f49582ac090acb91c11f9ae9a04f71e476e9e220b0e610b`，其
RH56 字段仍为：

```text
thumb_rotate_validated_realtime_range = [900, 1000]
six_axis_coupled_closure_commissioned = false
commissioned_air_closure_targets = []
```

因此 candidate 51 的 `q6=646` 和五弯曲目标都**未验收、未解锁**。

仓库内两份 q6=900 历史记录也不能作为新的 Stage 1 PASS：

| evidence | 记录结果 | 当前校验结果 |
|---|---|---|
| `runs/rh56_q6_900_commissioning_20260721.json` | 正向到 900；旧回程在 925 超时；`status=fail` | `LOCKED`：结果失败、旧 source hash/绑定 |
| `runs/rh56_q6_900_commissioning_v2_20260721.json` | 正向到 900；旧回程在 950 超时；`status=fail` | `LOCKED`：结果失败、旧 source hash/绑定 |

二者都证明当时故障后最终 `ANGLE_SET=[-1]*6`，但“安全停机成功”不等于“运动验收
成功”。当前实现为 q6=900 保留了后来真机观察过的直接开端回程；这个特例不会扩展
给更低 q6。

candidate 51 当前 canonical no-contact hand path（25 units 步长）有 191 个状态：
q6 正向 15 步、五弯曲正向 83 步、五弯曲反向 78 步、q6 反向 14 步，path
SHA-256 为
`eba19ef585cd6760649e4f9a882ff3f69d321b9c189e8cb43ef00cc8438e85e9`。
这只是确定性命令路径，不是碰撞或真机通过证据。

## 2. 完全离线可立即完成

准确检查 candidate 51（不会导入硬件控制模块）：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

./scripts/run_control_preflight.sh \
  --config configs/fr3_rh56_v7_commissioning.json \
  --snapshot runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --selected-index 51 \
  --strict-full-grasp
```

当前预期退出码为 `3`，并明确列出 q6 超范围和 exact target 无证据；这个退出码是
正确的 fail-closed 结果。

复核历史 JSON（同样完全离线）：

```bash
./scripts/commission_installed_rh56.sh verify \
  --evidence runs/rh56_q6_900_commissioning_20260721.json

./scripts/commission_installed_rh56.sh verify \
  --evidence runs/rh56_q6_900_commissioning_v2_20260721.json
```

两条都应打印 `LOCKED`，不能通过改 JSON、复制旧 hash 或直接修改 profile 来消除。

## 3. 只读真机检查

下面命令读取 Franka 一帧和 RH56 寄存器，不创建 Franka controller，也不写 RH56：

```bash
./scripts/run_control_preflight.sh \
  --config configs/fr3_rh56_v7_commissioning.json \
  --snapshot runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --selected-index 51 \
  --read-hardware
```

它不能替代动作前的 fresh confirmation；若只读状态不是 Franka Idle、Desk 载荷不是
当前 0.607 kg/CoM/惯量、RH56 有故障或数字目标仍在活动，则不得进入下一阶段。

## 4. 会动作的最小安全分阶段链

每一阶段都是独立 run，必须使用新的输出文件，并在该次运行前重新确认：RH56 已牢固
安装、24 V 可立即切断、Franka 可立即停止、全扫掠空间无人无物、线缆固定且有余量。
程序不移动 Franka，只持续只读核对其 Idle/负载状态。

### Stage 1：只复验 q6=900 与已观察的直接开端回程

```bash
./scripts/commission_installed_rh56.sh run \
  --config configs/fr3_rh56_v7_commissioning.json \
  --output runs/rh56_q6_900_stage1_direct_FRESH.json \
  --target-q6 900 \
  --q6-step 25 \
  --max-axis-current-ma 400 \
  --confirm-installed RH56_INSTALLED_ON_FR3 \
  --confirm-24v-cutoff RH56_24V_CUTOFF_READY \
  --confirm-franka-stop FR3_STOP_READY \
  --confirm-workspace-clear INSTALLED_AIR_WORKSPACE_CLEAR \
  --confirm-no-contact PLA_LOW_SPEED_NO_CONTACT
```

只有终端和 JSON 同时满足 `status=pass`、`q6_sweep_pass=true`、
`q6_return_pass=true`、`reopened_and_verified=true`、`disabled_verified=true`，且人工
检查 telemetry 无异常后，Stage 1 才完成。随后离线运行：

```bash
./scripts/commission_installed_rh56.sh verify \
  --evidence runs/rh56_q6_900_stage1_direct_FRESH.json \
  --config configs/fr3_rh56_v7_commissioning.json
```

失败、Ctrl-C 或“最终已 disable”都不能升级 q6 范围。

### Stage 2：验收 q6=646、candidate 51 精确六轴空中闭合及完整回程

Stage 1 通过并人工审核后，重新张开、重新清场、重新确认，再运行：

```bash
./scripts/commission_installed_rh56.sh run \
  --config configs/fr3_rh56_v7_commissioning.json \
  --output runs/rh56_candidate51_exact_air_FRESH.json \
  --target-q6 646 \
  --q6-step 25 \
  --stage1-evidence runs/rh56_q6_900_stage1_direct_FRESH.json \
  --coupled-air-close \
  --bend-targets 0 358 799 911 922 \
  --max-axis-current-ma 400 \
  --confirm-installed RH56_INSTALLED_ON_FR3 \
  --confirm-24v-cutoff RH56_24V_CUTOFF_READY \
  --confirm-franka-stop FR3_STOP_READY \
  --confirm-workspace-clear INSTALLED_AIR_WORKSPACE_CLEAR \
  --confirm-no-contact PLA_LOW_SPEED_NO_CONTACT \
  --confirm-wide-q6 RH56_Q6_WIDE_RANGE_COMMISSIONING \
  --confirm-coupled-closure RH56_COUPLED_AIR_CLOSURE \
  --confirm-exact-air-target RH56_EXACT_CANDIDATE_AIR_TARGET
```

`--stage1-evidence` 是代码级硬依赖：缺失、`status=fail`、非 q6-only、目标不是
900、配置谱系不符、文件被改或任一 bound source hash 过期，都会在导入硬件模块前
退出。Stage 2 的 q6 正向为 `975,950,...,650,646`；完成弯曲轴反向张开后，q6
不得从 646 直接跳到 1000，而是 `696,721,...,996,1000`。

这一阶段按
`pinky -> ring -> middle -> index -> thumb_bend` 每次只变一轴闭合，并沿 canonical
反向路径全部张开；空抓中任一 status 3 都是失败。通过后：

```bash
./scripts/commission_installed_rh56.sh verify \
  --evidence runs/rh56_candidate51_exact_air_FRESH.json \
  --config configs/fr3_rh56_v7_commissioning.json \
  --require-coupled

./scripts/commission_installed_rh56.sh propose-config-update \
  --evidence runs/rh56_candidate51_exact_air_FRESH.json \
  --config configs/fr3_rh56_v7_commissioning.json \
  --require-coupled

./scripts/commission_installed_rh56.sh materialize-config-update \
  --evidence runs/rh56_candidate51_exact_air_FRESH.json \
  --config configs/fr3_rh56_v7_commissioning.json \
  --output-config configs/fr3_rh56_v7_candidate51_commissioned.json

./scripts/commission_installed_rh56.sh verify-applied \
  --evidence runs/rh56_candidate51_exact_air_FRESH.json \
  --config configs/fr3_rh56_v7_candidate51_commissioned.json \
  --require-coupled
```

`materialize-config-update` 只接受 coupled PASS，并创建同目录、只读、不可覆盖的新
profile；原 commissioning profile 不变。后续 plan/audit/control 必须显式传入这个
新 profile。不能提前把 `[0,358,799,911,922,646]` 写进 profile，也不能用 Stage 1
记录物化 coupled 配置。

## 5. RH56 验收后仍未自动完成的事项

两阶段通过只证明“安装态、低速、无载荷、完全无接触”的 RH56 执行器路径。配置
hash 会改变，因此必须重新生成 candidate 51 joint plan；随后在当前姿态重新采集并
过滤场景，在 120 秒窗口内生成新的 schema-v2 installed-air audit。该审计必须覆盖
同一 canonical hand path 的全部 dense link sweep/feedback tube，并通过 FR3、V7、
open/closed RH56、bound object cloud 的要求。

`air-grasp` 运行时会在全部输出已 disable 时调用
`bind_audited_no_contact_execution_path`。当前驱动会在任何 q6 数字运动 I/O 前拒绝：

- audit q6 不在 profile 的 validated range；
- 运行计划 q6 与绑定 path target 不同；
- runtime forward waypoints 与绑定 path 不同；
- closure 时完整 runtime path hash 与 sidecar 不同。

每次完整空抓仍需要 fresh workspace/立即停止确认；旧场景、旧 joint plan、旧 audit
都不能复用。当前 PLA V7 只允许无载荷空抓，不允许把这条链解释成物体承载抓取或
lift 已验收。真正 loaded grasp 仍需承载适配器、相应 Desk 动力学和 loaded-mode
碰撞/接触验收。
