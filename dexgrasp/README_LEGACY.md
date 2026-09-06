# 历史详细流程（归档，2026-07-22）

> 警告：本文档是旧的调试/设计记录，包含已过期的 candidate、evidence、
> 状态结论和拆分命令。它不是当前真机操作手册，不得直接复制其中的
> 动作命令。当前唯一操作入口见项目根目录的 `README.md`。

# AnyDexGrasp 真机感知、可视化与门禁控制（旧版）

本目录把已经标定的 Intel RealSense D435、目标分割点云、抓取候选生成和
Open3D 可视化串成一条流水线，并提供独立验收及安装后的 fail-closed 控制层。
感知/可视化程序本身仍然**不控制机器人**。当前可以在
`robot_base` 场景点云中同时查看：

- 去除目标后的场景点云；
- 分割并清理后的 object point cloud；
- 抓取原点、分数和 canonical grasp pose；
- 选中抓取对应的 Inspire Hand-R 13-link URDF/STL 网格。

当前结论：D435 capture、官方 AnyDex representation + Inspire decision 推理、
候选去重、13-link FK/mesh、V7 安装变换、确定性 FR3 IK/joint-plan、HPP-FCL
installed-tool 审计以及分阶段执行器都已完成离线回归。预览默认显示与正式
`air-grasp` 相同的 `80 mm final-air + 10 mm pregrasp` 合同，不再把 contact pose
错当成实际空抓目标；可选绿色手模型由实际六轴 `ANGLE_ACT` 通过官方 XLS/URDF
重建。

安装后的真机空抓目前仍是 **LOCKED**：代码改变后，历史 Stage1、Stage2 和
installed-air artifacts 的源码 SHA 绑定均已失效；必须在当前代码冻结后重新做
`q6=900` Stage1、所选候选 exact Stage2，并生成新的 commissioned profile 与
120 秒内的新鲜 installed-air audit。当前实物适配器是 Bambu PLA，所以即使空抓
全部通过，也只允许低速、无接触、无载荷、无 lift。不能用旧 evidence、离线
PASS、裸法兰参数或确认 token 绕过这些门禁。

## 先看这里：最终操作地图

| 层级 | 当前状态 | 入口 |
| --- | --- | --- |
| 官方抓取生成、候选列表、点云/目标预览 | 可立即运行；只使用 D435/GPU/文件 | 第 3.1 节 |
| Franka/RH56 状态读取 | 允许只读；不创建控制器、不写寄存器 | `run_control_preflight.sh --read-hardware` |
| 恢复验收、回 default、新鲜审计、实时预览和空抓 | 推荐使用合并入口；不再手填六轴目标或中间证据路径 | “简化流程（推荐）”一节 |
| 安装态回 default | 脚本已完成；会产生动作，只能由现场操作者明确运行 | 第 0.1 节 |
| exact RH56 验收 + fresh air audit + 空抓 round trip | 代码已完成；历史 evidence 失效，需按第 8.2/8.5 节重新生成全部 PASS | `commission_installed_rh56.sh`、`prepare_fresh_installed_air_audit.sh` |
| 接触抓取、lift、setdown | 当前 PLA 配置硬锁；没有可执行真机命令 | 第 8.6 节 |

需要逐阶段定位问题时，仍可按以下拆分流程工作；日常真机操作优先使用下方
“简化流程（推荐）”，且每个默认 no-overwrite 的产物都使用唯一文件名：

1. 新拍一帧并运行官方推理，先加 `--no-preview`，不要猜 candidate index。
2. 用 `list_grasp_candidates.sh` 重列候选；设置本轮唯一 `CANDIDATE`。
3. 用 `run_live_pipeline_preview.sh --execution-mode air` 检查 object cloud、
   approach、final-air/pregrasp 和目标 mesh；`contact` 只用于感知参考。
4. 若之后需要真机空抓：先运行两个 default 脚本；再在代码冻结后重做 Stage1、
   exact Stage2、materialize profile，并从当前只读 q 生成 fresh full-air audit。
5. 只有 dry-run 为 `EVIDENCE-PASS` 且无 blocker，才由现场操作者运行第 8.5 节
   的正式 `air-grasp` 命令。它会自动 round trip 回程，但不会接触或抬起物体。
6. 可选 `pregrasp` 是独立分支；一旦执行，live q 已变化，之前的 full-air audit
   不能继续使用。必须回 default、重新读 q、重新生成 fresh audit。

只读检查示例（不会创建 Franka control handle，也不会写 RH56）：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

./scripts/run_control_preflight.sh \
  --config configs/fr3_rh56_v7_commissioning.json \
  --read-hardware
```

## 简化流程（推荐）

下面把原来的恢复、Stage2 验收、profile 生成、两台设备复位、新鲜审计、
telemetry、预览和正式空抓收敛成三个入口。现场仍需逐项确认命令中的安全 token；
这些 token 是当次现场事实，不是可永久保存的配置。

### A. 一条命令完成中断恢复和 exact Stage2 验收

当前这份中断证据已经绑定原始 control config、official snapshot、candidate index、
Stage1 evidence、`q6-step` 和六轴 `hand_targets`。入口会依次执行安全回程、exact
Stage2、`--require-coupled` 验证、commissioned profile 生成和 applied 验证：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

./scripts/recover_and_commission_rh56.sh run \
  --failed-stage2 "$PWD/runs/rh56_candidate0_exact_stage2_20260722_151314.json" \
  --output-dir "$PWD/runs/candidate0_recover_commission_$(date +%Y%m%d_%H%M%S)" \
  --confirm-installed RH56_INSTALLED_ON_FR3 \
  --confirm-24v-cutoff RH56_24V_CUTOFF_READY \
  --confirm-franka-stop FR3_STOP_READY \
  --confirm-workspace-clear INSTALLED_AIR_WORKSPACE_CLEAR \
  --confirm-no-contact PLA_LOW_SPEED_NO_CONTACT \
  --confirm-recovery RH56_INTERRUPTED_OPEN_RECOVERY \
  --confirm-wide-q6 RH56_Q6_WIDE_RANGE_COMMISSIONING \
  --confirm-coupled-closure RH56_COUPLED_AIR_CLOSURE \
  --confirm-exact-air-target RH56_EXACT_CANDIDATE_AIR_TARGET
```

这里没有 `--hand-targets`、`--target-q6`、`--snapshot`、`--candidate-index` 或
`--config` 覆盖项；脚本从不可变的失败证据自动选择并核对原来的 candidate 及其
六轴目标，不再要求手工抄写 `[798,798,798,798,978,450]`。不要在闭合后的自动
回程阶段主动中断，除非现场确实需要紧急停止。

成功结束时会打印两个真实绝对路径：`PROFILE=...` 和 `RECEIPT=...`。把终端实际
打印的 `RECEIPT=...` **整行复制到当前 shell 执行**；B 会从这份 PASS receipt
自动、严格派生 verified profile、official snapshot 和 candidate index，不再手工
传这三个值。若本轮 receipt 及其绑定文件仍完整，以后可直接从 B 步开始，不要重复做 A。

### B. 一条 `prepare` 命令完成两台设备复位和执行准备

确认 `$RECEIPT` 是 A 步实际打印并在当前 shell 中设置的值后运行：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

AIR_RUN="$PWD/runs/candidate0_air_$(date +%Y%m%d_%H%M%S)"

./scripts/run_installed_air_grasp.sh prepare \
  --commission-receipt "$RECEIPT" \
  --output-dir "$AIR_RUN" \
  --confirm-installed RH56_INSTALLED_ON_FR3 \
  --confirm-24v-cutoff RH56_24V_CUTOFF_READY \
  --confirm-franka-stop FR3_STOP_READY \
  --confirm-rh56-workspace-clear INSTALLED_AIR_WORKSPACE_CLEAR \
  --confirm-no-contact PLA_LOW_SPEED_NO_CONTACT \
  --confirm-rh56-reset-open RH56_RESET_OPEN \
  --confirm-hand-open RH56_OPEN_DISABLED \
  --confirm-default-sweep-clear FR3_RH56_CURRENT_TO_DEFAULT_SWEEP_CLEAR \
  --confirm-pla-low-speed PLA_LOW_SPEED_UNLOADED_ONLY \
  --confirm-stationary-q CURRENT_Q_READ_ONLY_AND_STATIONARY
```

入口会先以 `O_NOFOLLOW` 只读打开 receipt，核对 schema/kind/integrity/PASS、全部
input/output/workflow-source hash，再独立执行 Stage2 `--require-coupled` 和 derived
profile applied 验证；receipt 本身不是 motion authority。receipt 模式禁止再混入
`--config`、`--snapshot` 或 `--selected-index` 覆盖项。随后入口按顺序自动完成：
RH56 张开并禁用、Franka 回安装态 default、原生 telemetry 构建、一次新鲜 Franka
状态读取、D435 installed-tool audit、正式执行器 dry-run、manifest 和不可覆盖
session 封装；不需要再选择候选或输入任何 `hand_targets`。

成功时终端会打印 `SESSION=/真实/绝对路径/installed_air_grasp_session.json`。
把终端实际打印的 `SESSION=...` **整行复制到当前 shell 执行**；不要使用示例路径。
新鲜审计只有 120 秒有效期，因此 B 成功后应立即执行 C。若过期或 session 已使用，
用新的 `AIR_RUN` 重新执行 B。

### C. 一条 `run` 命令打开实时窗口并执行完整空抓 round trip

确认 `$SESSION` 来自 B 步终端实际打印的路径后运行：

```bash
./scripts/run_installed_air_grasp.sh run \
  --session "$SESSION" \
  --confirm-workspace-clear FR3_RH56_WORKSPACE_CLEAR \
  --confirm-immediate-stop IMMEDIATE_STOP_AND_24V_CUT_READY \
  --confirm-air-grasp FR3_RH56_AIR_GRASP_NO_CONTACT_NO_LIFT \
  --confirm-q6-preshape RH56_Q6_PRESHAPE_VERIFIED \
  --confirm-installed-collision-model INSTALLED_TOOL_COLLISIONS_VERIFIED
```

`run` 会先打开只读 Open3D/D435 窗口，实时显示场景点云、保存的 object cloud、
选中的目标 grasp/目标手 mesh、实测当前 EE 和由六轴反馈重建的当前手 mesh；只有窗口、
D435 和原生 telemetry reader 全部发布 `READY` 后，才启动前台执行器。窗口未就绪、
提前退出或 session/hash/freshness 不一致时，机器人执行器不会启动。

这条路径严格限定为当前 Bambu PLA 适配器的**低速、无接触、无载荷、无 lift 空抓**。
它不是物体接触抓取命令，也不会抬起物体；接触和 lift 仍然锁定。

执行期间按 `Ctrl+C` 时，编排器先把 `SIGINT` 转发给正式执行器并等待其既有的
Franka stop/RH56 disable 清理，再关闭预览窗口；恢复验收入口同样只转发
`SIGINT` 并等待子流程清理，不会用 `SIGTERM`/`SIGKILL` 强杀硬件执行器。
`Ctrl+C` 本身不等于已经安全停止：只有日志明确确认 stop/disable 才算完成；若打印
`STOP UNCONFIRMED`，立即使用 Franka 物理停止并切断 RH56 24 V。

## 安全边界

`apps/capture_object_grasps.py`、`apps/visualize_snapshot.py` 和
`apps/live_pipeline_preview.py` 只使用相机/快照/可选 pose JSON，不导入
Franka/Inspire 控制包。运行 capture/viewer 不需要给灵巧手接 24 V；
标定 YAML 中保存机器人信息只是数据 provenance，不表示程序会连接机器人。

只有以下显式入口可能产生动作，且都要求完整的精确确认 token：

- `run_rh56_bench_demo.sh`：只连接已固定在治具上的 RH56；
- `run_franka_unmounted_default.sh`：只连接空法兰 Franka；
- `reset_installed_rh56_open.sh run`：只张开并禁用已安装 RH56，Franka 全程只读；
- `reset_franka_default.sh`：只在 RH56 已张开/禁用后低速移动 Franka 到安装态 default；
- `commission_installed_rh56.sh run`：安装态 RH56 的分阶段无接触验收，不移动 Franka；
- `recover_and_commission_rh56.sh run`：按失败证据恢复 RH56，并自动完成 exact
  Stage2 验收和 profile 生成；Franka 保持只读；
- `run_installed_air_grasp.sh prepare|run`：合并安装态复位、新鲜审计、实时预览和
  联合空抓；`prepare` 和 `run` 都可能产生真机动作；
- `execute_control_sequence.sh default|pregrasp|air-grasp|grasp|grasp-lift`：安装后的
  联合控制；`grasp-lift` 已有独立 fail-closed 执行合同，但当前 PLA 配置仍被锁定。

感知候选、IK、可达性和离线碰撞结果都不可直接当成安全运动目标；它们不证明
接触稳定性，也不证明下发后的真机行为。只有绑定当前配置、当前关节状态和新鲜
场景的完整 sidecar 才能进入后续门禁，而且仍需现场确认 token。

## 0. 未安装状态的独立验收（已跑通）

在 RH56/适配器尚未装到 Franka 时，必须分开执行，不能同时运动。

RH56 必须机械固定在治具上，24 V 可立即切断且手周围清空：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

./scripts/run_rh56_bench_demo.sh \
  --confirm-clamped RH56_MECHANICALLY_CLAMPED \
  --confirm-24v-cutoff RH56_24V_CUTOFF_READY \
  --confirm-workspace-clear RH56_WORKSPACE_CLEAR
```

动作顺序为：全开 → `q6=900` 预成形 →
`[700,700,700,700,800,900]` 部分闭合 → 六轴禁用 → 重新全开 → 六轴禁用。
速度为 40，力阈值为 80 g；临时寄存器设置会在停止后恢复。2026-07-17
真机验收最终回读 `q6=986`、`ANGLE_SET=[-1]*6`、无故障、六轴电流均为 0。

确认法兰为空、Franka 周围清空且可以立即停止后，移动到 default joint pose：

```bash
./scripts/run_franka_unmounted_default.sh \
  --confirm-flange-empty FR3_FLANGE_EMPTY \
  --confirm-workspace-clear FR3_WORKSPACE_CLEAR \
  --confirm-stop-ready FR3_STOP_READY
```

该入口在创建控制器前强制核对 `F_T_EE=I`、`m_ee=0`、零 CoM/惯量、Idle、
错误/接触/碰撞和关节限位。目标为
`[0,0,0,-pi/2,0,pi/2,0] rad`，最大关节速度 `0.05 rad/s`。本次真机最终
最大误差为 `0.000094 rad`。

这些数值只适用于“空法兰”。安装适配器/RH56 后必须重新配置并验证末端动力学。

### 0.1 已安装 RH56 后的可重复 default

安装态不要再使用上面的“空法兰”命令。可重复 default 固定为两步：先把 RH56
恢复为六轴张开且输出禁用，再只移动 Franka 到 V7 profile 的
`franka.default_q_rad`。两条命令都不会执行 grasp。

第一步只移动 RH56；Franka 只建立 read-only Idle/load gate，不创建控制器：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

./scripts/reset_installed_rh56_open.sh run \
  --config configs/fr3_rh56_v7_commissioning.json \
  --confirm-installed RH56_INSTALLED_ON_FR3 \
  --confirm-24v-cutoff RH56_24V_CUTOFF_READY \
  --confirm-franka-stop FR3_STOP_READY \
  --confirm-workspace-clear INSTALLED_AIR_WORKSPACE_CLEAR \
  --confirm-no-contact PLA_LOW_SPEED_NO_CONTACT \
  --confirm-reset-open RH56_RESET_OPEN
```

成功条件是六轴 `ANGLE_ACT>=980`、`ANGLE_SET=[-1]*6`、状态为 idle
（寄存器值 `2`，或该固件的 idle/unknown 回读 `0xFF`）、电流不超过
100 mA，且所有临时速度/力设置均恢复为 `[1000]*6`/`[500]*6`。这里的
`RH56_24V_CUTOFF_READY` 表示现场能在异常持续运动或打印
`STOP UNCONFIRMED` 时立即断电，**不是要求每次正常结束后断 24 V**。

若 q6 已接近张开端、但仍低于验收阈值（例如 `ANGLE_ACT=973`），直接发送
`1000` 只有 27 units，可能被执行器近端死区吞掉并保持 `STATUS=2`。复位脚本会
在已验收的 `900..1000` 范围内先低速回退到约 `920`，确认六轴
disable/idle 后，再从新的实时反馈锚点回到 `1000`。因此这类复位过程中 q6
短暂向远离张开端的方向运动是预期行为；只有第一步打印 `[PASS]` 后，才执行下面
的 Franka default 命令。

第二步在整条“当前位姿 → default”扫掠空间、已安装手和黑色线缆都确认无碰撞
风险后，只移动 Franka：

```bash
./scripts/reset_franka_default.sh \
  --config configs/fr3_rh56_v7_commissioning.json \
  --confirm-installed RH56_INSTALLED_ON_FR3 \
  --confirm-hand-open RH56_OPEN_DISABLED \
  --confirm-workspace-clear FR3_RH56_CURRENT_TO_DEFAULT_SWEEP_CLEAR \
  --confirm-stop-ready FR3_STOP_READY \
  --confirm-pla-low-speed PLA_LOW_SPEED_UNLOADED_ONLY
```

该入口先连续两次**只读**核验 RH56 的上述 default 状态，再保持串口独占，防止
其他进程在 Franka 运动期间改变手型；随后核对 Desk 中的
`F_T_EE`、0.607 kg、CoM、惯量、零额外 load、Idle/error/contact/joint margin，
最后复用正式实时控制环，以不超过 `0.05 rad/s`、单段不超过 `0.20 rad` 回到
配置中的 cable-friendly default。它不调用 `set_EE`/`set_load`，不写 RH56
寄存器。`Ctrl+C` 会进入同一个 Franka stop + 连续静止回读流程；只有打印
`STOP UNCONFIRMED` 时才需要立即使用 Franka 物理停止。

只查看目标和静态配置、不连接任何硬件：

```bash
./scripts/reset_franka_default.sh --dry-run
```

`FR3_RH56_CURRENT_TO_DEFAULT_SWEEP_CLEAR` 是每次运行的现场扫掠空间确认；该
default 复位不会把 `default_path_collision_verified` 改成 true，也不会解锁
pregrasp、grasp 或 lift。
`execute_control_sequence.sh default` 仍受 installed-tool 路径审计门限，不是上述两个
专用 reset 入口的替代品；当前 profile 下它的 dry-run 显示 LOCKED 是预期行为。

## 1. 最快查看已验证的真机结果

推荐使用现有 `dynamic` Python。以下窗口已在当前桌面环境验证可以打开：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

DEXGRASP_SHELL_PYTHON=/home/qiaoguanren/anaconda3/envs/dynamic/bin/python \
  ./scripts/run_snapshot_viewer.sh \
  runs/d435_pink_cylinder_geometric_inspire_type4.npz \
  --hand-mesh-resolution full
```

窗口中场景点云被调暗，object point cloud 为橙色，箭头表示 canonical
抓取的 `+X` approach，坐标轴显示选中 pose，彩色几何体为 13 个 Inspire
link。`full` 使用官方 STL；显卡/窗口较慢时可改成 `simplified`。

原始旧快照不含 hand pose，可显式附加固定 type 4，并保存 enriched 快照：

```bash
DEXGRASP_SHELL_PYTHON=/home/qiaoguanren/anaconda3/envs/dynamic/bin/python \
  ./scripts/run_snapshot_viewer.sh \
  runs/d435_pink_cylinder_geometric.npz \
  --diagnostic-inspire-type 4 \
  --hand-mesh-resolution full \
  --save-enriched runs/d435_pink_cylinder_geometric_inspire_type4.npz
```

这里的 type 4 是人为固定的诊断值，不是网络输出。viewer 需要桌面
`DISPLAY`；远程纯 headless 环境只适合保存快照或跑测试。

## 2. 环境检查

两个环境彼此独立。相机、快照和 viewer 不要求 Torch/CUDA：

```bash
/home/qiaoguanren/anaconda3/envs/dynamic/bin/python \
  scripts/check_anydex_env.py --profile shell
```

当前该检查结果为 `READY`：Python 3.10、Open3D 0.19、OpenCV、
`pyrealsense2` 和 SciPy 均可导入。详细的官方环境安装说明见
[ENVIRONMENT.md](ENVIRONMENT.md)。

无相机 smoke test：

```bash
../.venv/bin/python apps/synthetic_demo.py \
  --output runs/synthetic_demo.npz
```

## 3. 真实 D435 capture

默认配置来自同级项目：

```text
/home/qiaoguanren/code/franka/perception/configs/d435_default.yaml
```

它绑定 D435 序列号 `337322072188`、`848x480@30`，并严格加载已经通过
质量检查的 eye-to-hand 标定：

```text
perception/configs/calibrations/fr3_d435_eye_to_hand.yaml
calibration_id = eye-to-hand-b722bce10485c8a3
```

启动一次真实 capture 和几何联调：

```bash
DEXGRASP_SHELL_PYTHON=/home/qiaoguanren/anaconda3/envs/dynamic/bin/python \
  ./scripts/run_capture.sh \
  --backend geometric \
  --geometric-inspire-type 4 \
  --hand-mesh-resolution full \
  --output runs/d435_geometric_type4.npz
```

未给 `--roi` 时会弹出交互框选；也可传 `--roi X1 Y1 X2 Y2` 使用像素 ROI。
程序等待连续稳定的有效帧，然后：

1. 从目标 mask 得到清理后的 `object_points`；
2. 用同一深度帧提取 `scene_points`，并用 mask 排除目标；
3. 把两者都转换到标定的 `robot_base`；
4. 生成抓取、保存严格 NPZ，并打开 Open3D viewer。

只采集和保存、不打开窗口时追加 `--no-viewer`。如果移动了相机、换了序列号
或改变相机安装，必须重新标定，不能沿用当前 `T_robot_base_camera`。

### 3.1 抓取生成后持续显示实时场景、目标手和位姿误差

`capture_object_grasps.py` 的神经网络/几何抓取仍是**一次生成、保存快照**；没有
必要为每个 D435 帧重复推理。新增的 perception-only viewer 会重新打开同一台
D435，持续刷新标定到 `robot_base` 的场景，同时保留快照中的 object cloud、目标
grasp pose 和目标 Inspire 13-link mesh。它不导入 Franka 或 Inspire 驱动。

正式入口会明确切换两个互不污染的 Python 环境，并一次完成：

1. `dynamic` 环境打开 D435，按 ROI/SAM2 分割，保存**不含抓取候选**的 raw NPZ；
2. 独立 official Python 3.8/CUDA 环境运行 representation + Inspire decision，保存
   provenance 完整的 official NPZ；
3. 回到 `dynamic` 环境重新打开 D435，显示实时场景、保存的 object cloud、所选
   grasp pose 和 13-link hand mesh。

首次对一个新快照运行正式抓取生成时，先不打开预览，让终端完整打印去重后的
候选表：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

DEXGRASP_DYNAMIC_PYTHON=/home/qiaoguanren/anaconda3/envs/dynamic/bin/python \
./scripts/run_grasp_generation_live_preview.sh \
  runs/d435_YYYYMMDD_HHMMSS_official.npz \
  --raw-output runs/d435_YYYYMMDD_HHMMSS_raw.npz \
  --roi 250 100 620 450 \
  --sam2 \
  --device cuda:0 \
  --top-k 64 \
  --trust-official-checkpoints \
  --no-preview
```

把示例 ROI 换成当前物体的像素范围；完全省略 `--roi` 时会先交互框选。
`--sam2`、ROI、`--selected-index` 都由同一个入口分别传给正确阶段，不再尝试在
`dynamic` 环境载入 MinkowskiEngine。脚本在打开相机前会：核对 9 个官方权重的
`MANIFEST.sha256`、激活 `scripts/activate_official_runtime.sh`，并检查 CUDA、
MinkowskiEngine、pointnet2 和 knn。任一阶段失败时不会发布半写的最终 NPZ；已有
输出默认拒绝覆盖，需要明确传 `--overwrite`。
`--selected-index` 只选择本次内存中的预览候选，不改写 official NPZ 内保存的默认
selection；它还必须小于 `--top-k`，并在推理后再次按实际去重候选数检查。
推理结束后终端会逐行打印实际返回的全部候选，包括 `index`、`score`、`type`、
`canonical_xyz_m`、在 `robot_base` 中的 approach 单位向量、六轴
`hand_targets` 和单列 `q6`；先根据这张表选 index，再核对窗口中的方向与位置。
`--lift-preview-m` 必须是严格大于零的有限值，与五阶段 preview contract 一致。
新快照不得预设 candidate 0，也不得沿用旧快照的 candidate 51。选定后建立一个唯一
`CANDIDATE=<本次实际 index>` 变量，后续 plan、commission、audit、viewer 和 executor
必须全部复用该值。

终端输出丢失后，不需要重新推理；下面的纯离线入口会从已有 official NPZ 重列
所有固定 index、score、type、canonical xyz/approach 和六轴 target：

```bash
DEXGRASP_SHELL_PYTHON=/home/qiaoguanren/anaconda3/envs/dynamic/bin/python \
./scripts/list_grasp_candidates.sh \
  runs/d435_YYYYMMDD_HHMMSS_official.npz
```

查看列表后，把最终选中的 index 写入 `CANDIDATE=<本次实际 index>`；若只想在列表中
标记某一项，可追加 `--selected-index "$CANDIDATE"`。这个命令不改 NPZ。

只看完整的三个阶段命令、验证路径和权重哈希，不打开相机/GPU/窗口：

```bash
./scripts/run_grasp_generation_live_preview.sh \
  runs/d435_YYYYMMDD_HHMMSS_official.npz \
  --raw-output runs/d435_YYYYMMDD_HHMMSS_raw.npz \
  --sam2 \
  --device cuda:0 \
  --top-k 64 \
  --trust-official-checkpoints \
  --dry-run
```

生成 official NPZ 但暂不打开预览时加 `--no-preview`。raw NPZ 与 official NPZ
保留同一帧的 scene/object、相机序列号、标定 ID 和 `T_robot_base_camera`；workflow
在发布文件前逐项比较，且要求 official 输出含完整 checkpoint/source provenance、
hand pose 和 `[K,6]` Inspire target。这个入口只连接 D435；不会启动 Franka 或
Inspire 进程。两个文件都先写随机临时名并完整回读校验；raw 先发布，内容自包含的
official 最后发布并作为 commit marker。默认模式用原子的 no-clobber 创建消除
`exists→replace` 竞态；`--overwrite` 只在新文件全部通过后才替换旧文件，若第二个
文件发布失败会恢复原 raw/official 对。强制断电或 `SIGKILL` 不可能让两个独立路径
具备单次 rename 的事务语义，因此下游只应把 official 路径的出现/更新视为本轮提交
完成，不能把孤立的 raw 路径当成推理完成标志。

对已有的当前 official candidate 51 直接打开实时窗口：

```bash
DEXGRASP_SHELL_PYTHON=/home/qiaoguanren/anaconda3/envs/dynamic/bin/python \
  ./scripts/run_live_pipeline_preview.sh \
  runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --selected-index 51 \
  --source realsense \
  --execution-mode air \
  --hand-mesh-resolution simplified \
  --lift-preview-m 0.05
```

窗口中调暗点云是当前 D435 scene，橙色点是生成候选时保存的 object cloud，彩色
link 是目标闭合手型；较大的坐标轴是所选误差目标，额外 lift 坐标轴位于
`grasp + [0,0,0.05] m`。`--execution-mode air` 也是默认值，它严格复用正式审计的
`80 mm` 后退 final-air 和额外 `10 mm` pregrasp，所以窗口中的目标 mesh/坐标轴与
air executor 一致。若只想查看网络预测的 nominal contact pose，显式改成
`--execution-mode contact`；该模式不解锁当前 PLA 的接触抓取。lift 坐标轴仍只是
`+robot_base Z` 的锁定可视化提案。
如果实时物体点与橙色 object cloud 明显错位，说明物体/相机/标定已经变化，应重新
分割并生成抓取，不能继续使用旧 pose。

只校验五阶段 default→pregrasp→grasp→close→lift 数值，不打开相机或 Open3D：

```bash
./scripts/run_live_pipeline_preview.sh \
  runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --selected-index 51 \
  --execution-mode air \
  --lift-preview-m 0.05 \
  --validate-only
```

可选的 `--pose-state /path/current_ee.json` 会叠加当前 EE frame、当前 hand-source
frame、连到目标的误差线，并且仅在收到新的 waypoint/stage 边界样本时，
在终端打印位置误差（mm）和 SO(3) geodesic 旋转误差（deg）。这不是连续实时
telemetry。加 `--show-current-hand-mesh` 后，彩色 link 仍是目标手型，绿色 link 则由
同一 JSON 的六轴 `hand.angles`（即 `ANGLE_ACT`）通过 checksum 固定的官方
`driver_routine_to_angle.xls` 离散查表，再经官方 URDF FK 重建；没有做线性插值，
也不会拿目标手型冒充当前手型。这里的“当前”是**六个执行器反馈驱动的模型重建**，
不是 12 个独立关节传感器或视觉测得的物理真值，机构间隙、柔顺、接触变形和模型误差
仍然不可见。能离线画出某个 `0..1000` 反馈值也不代表该值已经通过真机验收，更不构成
运动授权。JSON schema 为：

```json
{
  "schema_version": 1,
  "reference_frame": "robot_base",
  "timestamp_unix_s": 1784620800.125,
  "T_reference_EE": [
    [1, 0, 0, 0.50],
    [0, 1, 0, 0.00],
    [0, 0, 1, 0.40],
    [0, 0, 0, 1]
  ],
  "hand": {
    "angles": [1000, 1000, 997, 1000, 1000, 985],
    "angle_targets": [-1, -1, -1, -1, -1, -1]
  },
  "stage": "moving_pregrasp",
  "source": "single_owner_non_realtime_publisher",
  "sequence": 42
}
```

在执行器已用 `--pose-state-output` 发布上述状态时，viewer 命令可写成：

```bash
CANDIDATE=SELECTED_INDEX

./scripts/run_live_pipeline_preview.sh \
  /absolute/path/to/official_snapshot.npz \
  --selected-index "$CANDIDATE" \
  --source realsense \
  --execution-mode air \
  --pose-state /tmp/fr3_rh56_waypoint_state.json \
  --pose-max-age-s 2.0 \
  --error-target auto \
  --show-current-hand-mesh \
  --hand-mesh-resolution simplified
```

wrapper 只在请求绿色反馈 mesh 时加入隔离的纯 Python `xlrd` 路径，并由代码核对
官方 workbook SHA-256；缺依赖或 hash 不一致会直接失败，不会退化成近似模型。

发布者必须用临时文件加原子 rename，并在 FCI 的非实时边界低频发布；viewer 默认
拒绝超过 `0.50 s` 的样本。不要为了生成这个文件再启动第二个 Franka client，也
不要在 1 kHz callback 内做 JSON、磁盘 I/O 或打印。没有 `--pose-state` 时实时点云
和目标仍正常显示，但 current/error 会明确报告 `UNAVAILABLE`，不会用目标冒充当前
状态；样本文件缺失、格式错误、时间戳过旧或明显来自未来时，当前 EE/hand frame、
误差线和绿色反馈 mesh 会一起隐藏。样本恢复后才重新显示，窗口不会保留旧位姿冒充
当前状态。只有六轴反馈字段缺失而 EE 样本仍有效时，才仅隐藏绿色手模型。

默认 `--execution-mode air` 的静态 pregrasp、final-air 和目标 mesh 已与当前
air-audit/executor 共享同一公式；显式 `contact` 才显示 10 cm pregrasp/nominal
contact 参考。真机执行时，`--pose-state-output` 仍会在每个 waypoint/stage 边界
发布 artifact 中的精确 target，viewer 的 `--error-target auto` 会优先用该 target
计算误差。两种模式都只是显示合同，不构成运动授权。

上述 `--pose-state` 仍只是 waypoint/stage 边界样本，不能称为连续机器人状态。正式
执行器现在已经接入 opt-in 的原生 fixed-POD telemetry：Franka 侧把 producer 融合进
**已有唯一 FCI owner** 的同一次 `ActiveControlBase.readOnce()`，不会补做第二次
`readOnce()`；RH56 侧只旁路发布**已有唯一串口 owner** 完成寄存器、目标、包络、
状态、故障以及该阶段适用的电流/接触策略校验后的六轴反馈，也不会增加一次
串口读取。实现和 fake/synthetic、跨
Python ABI 测试已经离线通过；这不等于完成了新一轮真机 A/B，也不构成运动授权。

两端必须分别构建，因为 executor 使用仓库控制环境的 CPython 3.9 ABI，而带
Open3D/RealSense 的 viewer 使用 dynamic 环境的 CPython 3.10 ABI：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp
./scripts/build_continuous_telemetry.sh
```

默认输出分别是：

```text
reader / viewer (Python 3.10): /tmp/anydex-native-telemetry-viewer-py310/python
producer / executor (Python 3.9): /tmp/anydex-franka-telemetry-producer-py39/python
control-side test reader (Python 3.9): /tmp/anydex-native-telemetry-control-py39/python
```

不能把 reader 目录传给 executor，也不能把 producer 目录传给 viewer。viewer 不会
通过 `Robot.read_once()` 或第二个串口 client 补采状态，也不会把 synthetic slot、
目标 pose 或过期样本冒充 measured feedback。完整的 manifest 创建、两终端 viewer /
executor 命令见第 8.5 节。

连续 viewer 的只读接口是：

```bash
TELEMETRY_MAP=/tmp/fr3_rh56_UNIQUE_SESSION.map
TELEMETRY_MANIFEST=/tmp/fr3_rh56_UNIQUE_SESSION.manifest.json
READER_PYTHON_DIR=/tmp/anydex-native-telemetry-viewer-py310/python

./scripts/run_live_pipeline_preview.sh \
  /absolute/path/to/official_snapshot.npz \
  --control-config /absolute/path/to/control_config.json \
  --selected-index "$CANDIDATE" \
  --source realsense \
  --execution-mode air \
  --continuous-telemetry "$TELEMETRY_MAP" \
  --telemetry-session-manifest "$TELEMETRY_MANIFEST" \
  --continuous-telemetry-python-dir "$READER_PYTHON_DIR" \
  --telemetry-wait-seconds 60 \
  --arm-max-age-s 0.25 \
  --hand-max-age-s 0.75 \
  --error-target auto \
  --show-current-hand-mesh \
  --hand-mesh-resolution simplified
```

`UNIQUE_SESSION` 每轮都必须替换成新的值；mapping 路径必须原本不存在，禁止复用上轮
文件。启动顺序是：先在终端 A 运行上面的 viewer；若 executor 尚未创建 mapping，终端会显示
`[telemetry] waiting ...`，viewer 最多只读等待 60 秒。然后在终端 B 启动与**同一个
manifest/run UUID** 绑定、且已单独完成审计与授权的 executor。producer 用 `O_EXCL`
创建并 commit mapping 后，viewer 会自动只读接入，不需要重启。需要更长等待可增大
`--telemetry-wait-seconds`；设为 `0` 表示只尝试一次。

等待期间只会重试“路径尚不存在”和“初始化尚未 commit”。producer 赢得 `O_EXCL`
之后到 `ftruncate(2112)` 之间可能短暂看见零字节普通文件；reader 只对此窗口给最多
`0.50 s` 的初始化 grace。非零错误尺寸、持续零字节、权限错误、ABI/layout 不兼容、
identity/hash 不一致、`mmap` 失败等都直接退出，不会被 60 秒等待掩盖。viewer 从不调用
initialize/create/truncate/write API；等待时按 `Ctrl-C` 会走正常 viewer 清理。

`--continuous-telemetry` 与旧的 `--pose-state` 互斥，并且必须和
`--telemetry-session-manifest` 成对出现。manifest 锁定 run UUID、execution/snapshot/
config/calibration/producer build 五个 SHA-256、candidate 和目标位姿；任一不一致都拒绝
整个 mapping。arm 和 hand 分别做 realtime/monotonic freshness 检查：hand 过期时仍可
显示新鲜 EE，但隐藏绿色手 mesh；arm 过期时全部 current 几何都隐藏。默认 Franka
发布 decimation 是 20；在 1 kHz active control loop 中名义约 `50 Hz`。RH56 发布节奏由
已有串口 owner 的校验读取决定；在手反馈校验活跃的阶段实际通常约 `5–8 Hz`，不是额外
定时轮询或硬实时保证。
绿色 hand 只是六个 `ANGLE_ACT` 驱动的官方 URDF/XLS 模型重建，不是视觉测得的物理
表面，也看不到机构间隙、柔顺或接触变形。

在不导入 native module、不打开 mapping、相机或任何驱动的情况下，可先给同一命令加
`--validate-only` 验证 snapshot/config/manifest/target pose 是否完全一致。native reader
还会固定核对 ABI schema digest，并且正常模式只调用
`TelemetryReader.open_read_only(...)`。完整合同见
[`docs/continuous_telemetry_contract.md`](docs/continuous_telemetry_contract.md)。producer
只在 executor 已通过原有离线、live、审计和 Desk 门后创建；telemetry 本身不会放宽
任何门。当前集成状态是**代码接入且离线验证完成，尚未在本轮由本文档工作直接控制真机
验证**；目标 pose/mesh 和实时点云显示不受这个限制。

## 4. geometric commissioning 的含义

`GeometricGraspBackend` 使用目标点云的 PCA/OBB 生成确定性的候选，用来验证
分割、标定、frame、pose 方向和 viewer。它的 backend 名为
`geometric_demo`，不是 AnyDexGrasp 神经网络，也不代表真实抓取成功率。

几何 backend 不预测 Inspire semantic type。capture 默认把
`--geometric-inspire-type 4` 附加到所有候选，只为检查官方映射、URDF FK 和
手模型放置。日志与快照 `model_name` 都会标记：

```text
diagnostic fixed Inspire type 4 (not decision-model output)
```

因此，看到 type 4 或一只合理摆放的手，不能声称“网络识别为 type 4”。只有
官方 representation + Inspire decision backend 的结果才是网络产生的 type。

## 5. Inspire 13-link mesh、FK 和 type 1..8

viewer 从官方 checkout 读取：

- `generate_mesh_and_pointcloud/inspire_urdf/urdf-five3/robots/urdf-five3.urdf`；
- `urdf-five3/meshes/*.STL`（13 个 link）；
- `width_12Dangle_6Dangle.json`。

映射表的 `12d` 是按 URDF 顺序排列的 12 个关节弧度，用于 FK；`6d` 是
Inspire 执行器寄存器命令，绝不能当弧度。`6d` 顺序是
`little, ring, middle, index, thumb_bend, thumb_rotate`。

| ID | 官方映射名 |
| ---: | --- |
| 1 | `Ring` |
| 2 | `Prismatic_2_Finger` |
| 3 | `Prismatic_3_Finger` |
| 4 | `Large_Diameter` |
| 5 | `Medium_Wrap` |
| 6 | `Tripod` |
| 7 | `Sphere_3_Finger` |
| 8 | `Distal_Type` |

映射宽度最终限制在 `0.025–0.10 m`。当前只渲染选中候选的 13 个 link；可用
`--no-hand-mesh` 关闭 link mesh，只看点云、箭头和 pose frame。

## 6. official AnyDex backend

代码路径已经实现：object cloud 从 `robot_base` 变回 RealSense optical frame，
运行官方 representation network，再用 8 个 Inspire decision head 选择 type 和
depth，最后把 canonical/hand pose 变回 snapshot reference frame。整个过程没有
机器人控制。

源码固定在官方 commit `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`；当前
representation checkpoint 和 8 个 decision `.pth` 已下载，并通过
`weights/MANIFEST.sha256` 校验。需要重新准备资产时：

```bash
./scripts/fetch_official_source.sh
python scripts/download_official_weights.py
```

先在专用环境检查，不要把 `dynamic` 环境误当官方环境：

```bash
source scripts/activate_official_runtime.sh
python scripts/check_anydex_env.py --profile official
```

上游推荐栈是 Python 3.8、PyTorch 1.13.x、CUDA 11.x、MinkowskiEngine 0.5.x，
以及同环境编译的 `pointnet2`/`knn`。`dynamic` 仍只是相机/可视化环境；官方推理
使用独立的 `dexgrasp/.venv-anydex-official` 兼容环境。

对已经保存的 D435 snapshot 运行离线推理：

```bash
DEXGRASP_OFFICIAL_PYTHON=/path/to/anydex-python \
  ./scripts/run_official_snapshot.sh \
  runs/d435_pink_cylinder_geometric.npz \
  --output runs/d435_pink_cylinder_official.npz \
  --device cuda:0 \
  --top-k 10 \
  --trust-official-checkpoints
```

默认 checkpoint 是 `weights/logs/model/checkpoint.tar.18`，decision 目录是
`weights/logs/model/inspire_model/obj140`。官方 decision `.pth` 是 pickle-backed
模块对象；只有确认文件来自官方目录且校验通过后才使用
`--trust-official-checkpoints`。生成后再用 snapshot viewer 查看 13-link 手模型。

当前正式离线输出是
`runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz`，文件
SHA-256 为
`92c44aff86e73229c7c2a454c0cf3ffce714580d1895dc98d98759e3711b6395`。
它绑定官方 commit `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`、representation
checkpoint 和 8 个 decision checkpoint，共保存 84 个去重候选。当前离线选用
candidate 51（原始 source index 54，type 7，score 约 0.9209），RH56 target 为
`[0,358,799,911,922,646]`。这些 provenance 证明“结果来自正式网络”，不等于
授权机器人运动。

该 NPZ 内保存的默认 selection 仍是 candidate 0；查看当前控制所绑定的
candidate 51 时必须显式覆盖 selection（只改变内存视图，不修改快照）：

```bash
DEXGRASP_SHELL_PYTHON=/home/qiaoguanren/anaconda3/envs/dynamic/bin/python \
  ./scripts/run_snapshot_viewer.sh \
  runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --selected-index 51 \
  --hand-mesh-resolution full
```

## 7. frame、pose、单位和 snapshot 约定

- 点、平移、width 和 depth 一律为米；URDF `12d` 关节量为弧度。
- `T_A_B` 把 B frame 坐标映射到 A frame。
- 真机快照的 `reference_frame` 是 `robot_base`；
  `T_reference_camera` 即标定的 `T_robot_base_camera`。
- RealSense optical frame 是 `+x` 向右、`+y` 向下、`+z` 向前。
- `T_reference_grasp` 把 canonical grasp local frame 映射到 reference；旋转矩阵
  第 0 列/局部 `+X` 是 approach，第 1 列是开合方向，第 2 列是高度方向。
- `T_reference_hand` 是 Inspire palm/mesh pose，必须和 canonical pose 分开理解。
- snapshot 是严格 v1 `.npz`，只含数值/文本数组，并用 `allow_pickle=False`
  读取；同时保存 frame、相机序列号、标定 ID、模型和 checkpoint provenance。

## 8. 安装后联合控制与适配器门禁

真机使用的 V7 适配器资产位于
`assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl`，SHA-256 为
`7fdd3dd06bd8dafed445f6a6910315edd3073bd1b9cd9015a951b6e415b947e1`。
它来自用户提供的 `FR3_RH56_adapter_V7_bundle.zip`；V7 把 RH56 径向 M3
孔系与螺母腔一起旋转了 45°，不能再用旧 V2/V6 网格代替真机碰撞外形。
当前默认控制配置是 `configs/fr3_rh56_v7_commissioning.json`；旧 V2 配置仅保留作
历史记录，不再被控制入口默认加载。
几何尺寸为：

- 法兰圆盘 `Ø70 × 10 mm`；
- 插柱 `Ø37.6 × 7.8 mm`；
- 碰撞包络轴向总长 `17.8 mm`；
- Franka 法兰面到 RH56 安装座面的坐标偏移是 `10 mm`，不是 `17.8 mm`。

7.8 mm 插柱进入 RH56 插孔，所以不能把总外形高度直接加到 TCP。控制坐标链为：

```text
T_robot_base_EE = T_robot_base_hand_source @ inverse(T_EE_hand_source)
T_EE_hand_source = inverse(F_T_EE) @ T_F_hand_source
```

当前安装关系可由 V7 STEP、官方 AnyDex RH56 source datum 和用户确认的右手方向
直接组成，不需要先用相机拟合主值：

```text
assembled_yaw = -45 deg
seating_to_hand_source_origin = [0, 0, 0] m
F_T_EE = identity

T_EE_hand =
[[0,  0.707106781187, 0.707106781187, 0],
 [0, -0.707106781187, 0.707106781187, 0],
 [1,  0,              0,              0.010],
 [0,  0,              0,              1]]
```

这里 `10 mm` 是 FR3 法兰面到 V7 肩面/AnyDex source 原点的距离。官方 mesh
生成器中的 `7.8 mm` 是安装金属套筒的预留：它把 `Link111` 白壳向指尖方向移动，
同时 V7 的 7.8 mm 插轴进入这个接口。因此既不能把它再加到 TCP，也不能从
10 mm 中再减一次。纯离线通用合成工具仍可用于其他安装：

```bash
./scripts/compose_mount_transform.sh \
  --F-T-EE R00 R01 R02 TX R10 R11 R12 TY R20 R21 R22 TZ 0 0 0 1 \
  --assembled-yaw-deg MEASURED_YAW_DEG \
  --seating-to-source-origin-mm DX DY DZ \
  --mark-measured \
  --output runs/fr3_rh56_mount_measurement.json
```

它不会连接硬件，也不会把上游 UR 的 44 mm TCP 或 14 mm 经验插入量混入安装
变换。14 mm 只能作为单独的抓取执行偏置；当前默认仍为 0 且未验收。

### 8.1 用 D435 独立复核 `T_EE_hand`

`register_installed_hand.sh` 是可选的只读静态复核程序：Franka 只调用
`Robot.read_once()`，不创建控制器；程序不导入或打开 RH56 串口。它读取标定后的
D435 点云和当前 `O_T_EE`，只拟合官方 URDF 中不随手指运动的 `Link111`，并计算：

```text
T_EE_hand = inverse(T_robot_base_EE) @ T_robot_base_hand_source
```

运行前确保没有其他程序占用 D435/FCI，并让已安装 RH56 的白色手掌/腕部外壳完整
出现在相机画面内。相机不能移动，否则 eye-to-hand 标定失效。运行：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

./scripts/register_installed_hand.sh \
  --output-prefix runs/fr3_rh56_static_registration \
  --overwrite
```

在 ROI 窗口中只框选刚性白色手掌/腕部外壳，排除手指、Franka 法兰、金属适配器、
黑色线缆和背景，然后按 Enter/Space。程序默认采 3 个独立 batch，共 36 帧；在
`-45°` 附近和其 `+180°` 歧义支路搜索安装 yaw，并检查残差、batch 一致性以及采集
前后的 Franka 位姿漂移。输出：

- `.json`：候选 `T_EE_hand`、yaw、座面到 source 原点、全部阈值和 provenance；
- `.npz`：观测点云、拟合后的 Link111 点和各坐标变换；
- Open3D：橙色为实测点，青色为拟合后的 Link111 mesh。

该程序只生成视觉 candidate，不覆盖由 CAD/source datum 得到的 V7 主值。即使显示
`PASS-CANDIDATE`，仍须人工核对 overlay；它也不会把
`installed_collision_model_verified`、`default_path_collision_verified` 或
`full_execution_ready` 改为 true。`OFFICIAL_SOURCE_MESH_OFFSET_M` 在配准模型中只
应用一次，不能再额外叠加上游 UR 的 44 mm TCP。

安装后的预检（默认完全离线）：

```bash
./scripts/execute_control_sequence.sh inspect \
  --snapshot runs/d435_pink_cylinder_geometric_inspire_type4.npz

# 可选，只读两台设备；不创建控制器、不写 RH56 寄存器
./scripts/run_control_preflight.sh \
  --read-hardware \
  --snapshot runs/d435_pink_cylinder_geometric_inspire_type4.npz
```

### 8.2 安装状态下验收 q6 与六轴无接触耦合

`commission_installed_rh56.sh` 专门生成 RH56 第六轴/六轴耦合验收证据。它不移动
Franka：RH56 串口连接后的第一条硬件操作先双写并验证六轴 disable；然后才构造
Franka 只读客户端，安装 watchdog，并在后续每次 RH56 组合反馈读取的前后调用
`Robot.read_once()`，持续核对 Franka `Idle`、错误/接触/碰撞、`F_T_EE`、
Desk 中的 0.607 kg/CoM/惯量、`m_load=0` 和
`m_total=0.607 kg`。缺少任一精确 token 时，程序在导入 `pylibfranka` 和 RH56
串口模块之前退出。

正式连接顺序也采用 fail-closed：RH56 串口连接后的第一个硬件动作必定是双写并
验证六轴 disable，之后才连接 Franka；安装载荷只读门禁通过后，Franka watchdog
才以 one-shot 方式绑定到 RH56。随后每次 RH56 组合反馈读取，以及 reviewed open
helper 的每个运动轮询，都会先执行新的 Franka `read_once()` 门禁。门禁中途失败时，
当前 RH56 数字目标立即被全六轴 disable，状态机随后停止 Franka 并再次验证 RH56
disable；异常之后不会发送下一个数字 waypoint。

本入口只适用于当前 Bambu PLA V7 的**低速、无载荷、完全无接触空中验收**。
先移走目标和其他可能碰到手指的物体，确认已安装 RH56 不会扫到桌面、机械臂、
线缆或人员。第一次只验收现有保守范围 `q6=900..1000`：

**当前仓库内已有的 Stage1/Stage2 JSON 都不能复用。** 它们正确地检测到所绑定的
Franka/Inspire driver、hand path 或 planner 源码 SHA 已变化并返回 `LOCKED`；不要重签、
改 hash 或复制旧 PASS。下面每个输出名都应换成本轮唯一名字，待代码冻结后重新运行。

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

./scripts/commission_installed_rh56.sh run \
  --output runs/rh56_q6_900_stage1_YYYYMMDD_HHMMSS.json \
  --target-q6 900 \
  --q6-step 25 \
  --max-axis-current-ma 400 \
  --confirm-installed RH56_INSTALLED_ON_FR3 \
  --confirm-24v-cutoff RH56_24V_CUTOFF_READY \
  --confirm-franka-stop FR3_STOP_READY \
  --confirm-workspace-clear INSTALLED_AIR_WORKSPACE_CLEAR \
  --confirm-no-contact PLA_LOW_SPEED_NO_CONTACT
```

程序连接后首先双写并多样本验证六轴 `ANGLE_SET=-1`、Idle、低空载电流；在此之前
不会读取或修改 speed/force。随后全六轴张开并验证，将 q6 按
`1000 -> 975 -> ... -> 900` 的低速小步路径移动，最后先安全张开五个弯曲轴，
再使用仅在 q6=900 Stage 1 真机观察过的直接开端回程。这个特例绝不用于更低 q6；
所有 `q6<900` 验收都使用 `+50` 反向启动、随后不超过 `q6-step` 的 canonical
小步回程（不放宽官方 q6-open helper 的 885 起点门禁）。
正向和反向每一步都要求：六轴 batch `ANGLE_SET` 精确读回、五个
弯曲轴保持张开/Idle、q6 方向正确、连续稳定到位、无 ERROR/故障/接触、温度低于
60 °C，并逐样本比较每轴电流与 `min(设备 CURRENT_LIMIT, 400 mA)`。总电流只记录
为 telemetry，不再使用任意的主机总和阈值代替设备逐轴保护。任何通信、反馈、
电流或超时异常都会锁停并两次写入/读回 `ANGLE_SET=[-1]*6`；若停机无法确认，
终端与 evidence 都会明确标记 `STOP UNCONFIRMED`，此时应立即切断 24 V。

在上一步 evidence 人工检查通过后，才可用额外 wide-range token 分阶段扩展到官方
候选所需的较低 q6。正式执行不会再把“做过一次任意耦合闭合”的布尔值当成授权；
它要求 profile 中存在与候选**六个寄存器逐值完全相等**的
`commissioned_air_closure_targets` 证据。旧快照 candidate 51 的
`[0, 358, 799, 911, 922, 646]` 只是一条历史诊断记录，不能复制给新快照。
Stage 2 只能在 Stage 1 的终端与 JSON 已人工审计、手重新全开、工作区再次确认后
执行。CLI 直接从同一份 official snapshot 和 candidate index 读取六轴
`hand_targets`，不再要求人工抄写 q6 或五个 bend target；snapshot 路径/hash、候选
index/score/type/pose 和六轴值都会绑定进 evidence，并在验收结束及后续 verify 时
重新核对。推荐先在候选表和预览窗口中确定 index，再显式填写
`CANDIDATE`：

```bash
CANDIDATE=SELECTED_INDEX
SNAPSHOT=/absolute/path/to/this_run_official_snapshot.npz
STAGE1=/absolute/path/to/rh56_q6_900_stage1_YYYYMMDD_HHMMSS.json
STAGE2=runs/rh56_candidate${CANDIDATE}_exact_stage2_YYYYMMDD_HHMMSS.json

./scripts/commission_installed_rh56.sh run \
  --output "$STAGE2" \
  --snapshot "$SNAPSHOT" \
  --candidate-index "$CANDIDATE" \
  --q6-step 25 \
  --stage1-evidence "$STAGE1" \
  --coupled-air-close \
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

如果确实要自动选择官方 decision score 最高的候选，可省略
`--candidate-index`；代码会重新计算最大 score，同分时取最小 index，并在运动前打印
`[snapshot target] index=... targets=[...]`。显式传 `--candidate-index 0` 就是选择第一项。
这里的“最高分”只表示官方模型分数，不包含 Franka IK、安装工具/场景碰撞或真机安全，
因此仍必须先看预览，并通过后续 joint-plan 与 fresh installed-tool audit；不能把它称为
“最安全”或“最可执行”的候选。snapshot 模式禁止同时传 `--target-q6` 或
`--bend-targets`，避免同一 evidence 混入两套不一致目标。

任何 `target-q6<900` 都把上述正式 Stage 1 PASS JSON 作为代码级硬依赖：缺失、
失败、目标错误、非 q6-only、内容/hash 被改、配置谱系不符或绑定源码已变化，都会在
导入硬件模块前退出。工具支持完整寄存器范围 `target-q6=0..1000`；目标为 `1000`
时会记录精确的数值 1000 endpoint（不拿 999 代替）并继续验收五个 bend 轴，但不会
请求第六轴旋转。目标低于 1000 时按小步下降；即使目标是 0，也会从 1000
开始按 10–50 单位的小步路径下降，绝不会一次跳到 0。建议按逐步扩大的多个 run
人工检查外观、走线和 evidence，而不是第一次就扫完整范围。普通耦合验收只接受
五个 `800..950` 的弯曲目标；超出该范围时必须额外提供上面的 exact-target token，
并且 evidence 只登记本次实际到达的那个六轴向量，不登记一个可泛化的矩形范围。
闭合采用确定性的逐轴路径：先 q6 小步到位，再按 pinky→ring→middle→index→thumb
bend 顺序，每次只改变一个轴且每步不超过 `q6-step`。成功返回使用
`rh56_hand_path` canonical v3：每个轴换向后的第一条张开命令至少跨 50 units
（避免固件对 25-unit 首条反向命令 idle/no-motion），后续恢复 `q6-step` 小步；先
逐轴全开五个弯曲轴，再把 q6 小步回到 1000，最后六轴 disable。evidence 会绑定
hand-path 源文件并重新生成 canonical 路径核对请求与 telemetry phase，不再接受
旧版 forward waypoint 的机械反序。
任一轴返回力接触状态 3 都判失败，不能把碰到物体或机械限位当成验收通过。

每个 run 通过 no-replace 原子操作创建一个默认只读的 JSON，记录 profile/adapter hash、Franka
连续只读门禁次数/最后成功状态/失败信息、
只读状态、RH56 设备寄存器、请求与实际 q6 范围、每一步完整反馈、最终重新张开、
最终 `ANGLE_SET=[-1]*6` 以及 coupled closure 结果。证据还绑定官方
`inspire_hand_routine_to_angle-use.xlsx`、`driver_routine_to_angle.xls` 和
`recover_inspire_hand_to_stl.py`、实际 RH56 register API、Franka driver 和配置校验模块
的 SHA-256；执行器验收不会猜测 open pose 的 URDF FK。JSON 自带 canonical payload
SHA-256，但它只是**防意外篡改的 checksum，不是可信签名，也不能阻止文件所有者重新
生成或替换记录**。因此必须人工审计，并在外部独立保留终端打印的整文件 SHA-256；
`motion_authorized` 永远为 `false`。

先对**尚未修改的** profile 验证 evidence：

```bash
BASE_PROFILE=configs/fr3_rh56_v7_commissioning.json
STAGE2=/absolute/path/to/rh56_candidateSELECTED_INDEX_exact_stage2_YYYYMMDD_HHMMSS.json

./scripts/commission_installed_rh56.sh verify \
  --evidence "$STAGE2" \
  --config "$BASE_PROFILE" \
  --require-coupled

./scripts/commission_installed_rh56.sh propose-config-update \
  --evidence "$STAGE2" \
  --config "$BASE_PROFILE" \
  --require-coupled
```

第二条命令只打印以下三个字段的 evidence-derived proposal，**不会修改文件**：

```text
inspire.thumb_rotate_validated_realtime_range
inspire.six_axis_coupled_closure_commissioned
inspire.commissioned_air_closure_targets
```

人工审计 JSON 后，推荐把精确三字段更新物化为一个**新的、只读、不可覆盖**的 sibling
profile；原 commissioning profile 和 Stage 1/2 的 hash lineage 保持不变，相对 adapter
路径也保持同一含义：

```bash
CANDIDATE=SELECTED_INDEX
RUN_TAG=YYYYMMDD_HHMMSS
BASE_PROFILE=configs/fr3_rh56_v7_commissioning.json
STAGE2=/absolute/path/to/rh56_candidate${CANDIDATE}_exact_stage2_${RUN_TAG}.json
PROFILE=configs/fr3_rh56_v7_candidate${CANDIDATE}_${RUN_TAG}_commissioned.json

./scripts/commission_installed_rh56.sh materialize-config-update \
  --evidence "$STAGE2" \
  --config "$BASE_PROFILE" \
  --output-config "$PROFILE"

./scripts/commission_installed_rh56.sh verify-applied \
  --evidence "$STAGE2" \
  --config "$PROFILE" \
  --require-coupled
```

`materialize-config-update` 自身强制 coupled PASS、拒绝覆盖源文件或已有输出，并拒绝把
新 profile 写到其他目录。后续 plan/audit/control 命令必须显式使用打印出的新
`--config` 路径；不能提前手改三个字段，也不能把 Stage 1 evidence 当成 coupled PASS。

commissioning evidence 只证明记录到的低速、无接触范围，不证明物体夹持、PLA
承载、手—场景连续碰撞或 AnyDex 抓取成功，也不能单独解锁联合执行。

联合状态机严格执行：

```text
六轴张开并验证
  -> Franka default_q
  -> pregrasp 到位并稳定
  -> grasp 到位并稳定
  -> q6 单轴预成形（其余五轴保持全开）
  -> 五个弯曲轴按已验收 exact waypoint 路径逐轴闭合并保持
  -> 有界 hold
  -> 五个弯曲轴沿 exact waypoint 反向张开
  -> q6 沿预成形 waypoint 反向回到 1000
  -> Franka stop + RH56 ANGLE_SET=[-1]*6
```

### 8.3 确定性生成所选候选的空抓关节计划

下面的入口完全离线，不打开相机、不连接 Franka，也不访问 RH56 串口。它从官方
snapshot 的指定候选、V7 profile 的 `T_EE_hand`/空抓距离以及官方 FR3 URDF 重新
求解 IK；候选冗余关节固定为 `q7=1.3525 rad`。规划起点必须绑定现场已核对的
关节读数。V7 profile 的 `franka.default_q_rad` 已改为 2026-07-18 用户报告的 FCI
静止读回值 `[-0.1118436,-0.1207545,0.0739457,-1.7431009,0.0463540,
1.6809169,0.8117281]`，从而不会为了套用旧 default 先把带腕部线缆的 `q7`
人为转到 0。profile 同时记录 `source_kind=user_reported_fci_stationary_readback`、
安装组合、日期、线缆原因和 `motion_authorized=false`；由于没有独立 evidence 文件，
其验证范围明确只限这个静止姿态，不证明运动路径。planner 按 `0.005 rad` 最大步长
显式采样 `start -> default -> pregrasp -> final-air`，并检查官方 SRDF
排除规则下的裸 FR3 自碰撞：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

PROFILE=/absolute/path/to/selected_candidate_commissioned_profile.json
SNAPSHOT=/absolute/path/to/official_snapshot.npz
CANDIDATE=SELECTED_INDEX
CURRENT_Q=(Q1 Q2 Q3 Q4 Q5 Q6 Q7)
RUN_TAG=YYYYMMDD_HHMMSS
PLAN=runs/candidate${CANDIDATE}_installed_air_joint_plan_${RUN_TAG}.json

./scripts/plan_installed_air_candidate.sh \
  --snapshot "$SNAPSHOT" \
  --config "$PROFILE" \
  --candidate-index "$CANDIDATE" \
  --q7-rad 1.3525 \
  --start-q-rad "${CURRENT_Q[@]}" \
  --output "$PLAN"
```

输出 JSON 原子创建且拒绝覆盖，绑定 snapshot/config/URDF 的 SHA-256，并记录候选、
目标位姿、`q_start/q_default/q_pregrasp/q_final_air`、独立 FK residual、实际最大关节步长、
`q7` 总转角和裸臂自碰撞结果。相同输入会得到逐字节一致的 manifest。所有占位符必须
替换成本轮同一个 snapshot/candidate、Stage 2 物化出的 profile，以及刚刚只读取得且
随后保持静止的七轴 `CURRENT_Q`。历史 candidate 51 的一次记录恰好从 profile default
开始，`q7` 从 `0.8117281` 到 `1.3525 rad`、转角 `+0.5407719 rad`；这组数值不是新一轮
规划的默认输入。
manifest 明确要求
现场另行确认黑色腕部线缆的余量。这个 planner **不检查**
scene、V7 adapter 或 RH56 mesh，且始终写入 `motion_authorized=false`；必须再把这些
waypoint 交给 installed-tool audit，不能把该 JSON 直接用于真机执行。

### 8.4 只把 Franka 移到 pregrasp（不执行抓取）

需要先单独观察机械臂/已安装张开手的 approach 时，使用正式的 `pregrasp`
子命令。它不截断或“借用”一个失败的 full-grasp audit，而只接受 schema-v2
`fr3_rh56_pregrasp_only_collision_audit`：artifact 必须绑定 config、官方 snapshot
及 candidate、V7 STL、`T_EE_hand`、AnyDex object point cloud、FK joint-plan，以及
下面这一条 canonical named prefix 和专用 `prefix_contract_sha256`：

```text
current
  -> default_transit_0 ...（可选）
  -> default
  -> approach_transit_0 ...（可选）
  -> pregrasp
```

硬门禁覆盖连续 tracking tube 下的 FR3 self、adapter↔FR3、open-RH56↔FR3、
open-RH56↔adapter，以及 adapter/open-RH56↔绑定 object cloud；任何碰撞、间隙不足、
coverage 缺失或 hash 不一致都不能由 operator token 绕过。单视角 full-scene 中的
机器人/线缆回波只作 hash-bound advisory；每次实际运行仍必须重新给出精确的
workspace-clear token，表示人员、线缆和环境在**整条扫掠空间**内均已现场确认。
artifact 保留真实的 scene `captured_at`、`created_at`、生成时 age 和 profile 建议的
`120 s` freshness；超出建议值会明确打印 advisory warning，但不会把这份含自体回波的
full-scene cloud 升格为运行门禁。schema-v2 明确拆开两种 q：scene NPZ 内的
`capture_q_rad` 只作 hash-bound 场景来源记录；prefix 的 `current` 则独立绑定
joint-plan 的 `q_start_rad` 及其 SHA-256。两者可以不同，artifact 会记录 L∞ 差值；
真正执行前仍必须把 Franka **实时读取**的 q 与 prefix `current` 比较，任一轴误差
超过 `0.002 rad` 就在运动前硬拒绝。object cloud、prefix path、live q、
config/CAD hash 仍是不可由 token 绕过的硬门。

本小节下面的 candidate 51 文件是一个**只到 pregrasp 的历史绑定样例**；它不等于
第 8.5 节的新鲜 full-air artifact，也不能改 candidate index、snapshot、profile 或
起始 q 后继续使用。新一轮完整空抓应直接使用第 8.5 节的 `CANDIDATE` 变量和 fresh
audit 流程，不要从这里复制硬编码的 51。

下面是 2026-07-21 在第一次通信质量停止后重新只读取得 q，并据此重新绑定的纯离线
生成命令。它们不导入
libfranka/Inspire 驱动，也不会连接相机、机械臂或串口；生成出的
`motion_authorized` 始终为 `false`。两个输出文件已经存在，generator 会拒绝覆盖；
原样复现时应先换一个新的 `--output` 文件名：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

./scripts/plan_installed_air_candidate.sh \
  --snapshot runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --config configs/fr3_rh56_v7_commissioning.json \
  --candidate-index 51 \
  --q7-rad 1.3525 \
  --start-q-rad -0.114031 -0.1182116 0.0736167 -1.7442204 0.042868 1.6763846 0.8131912 \
  --max-joint-step-rad 0.005 \
  --output runs/candidate51_installed_air_joint_plan_after_rate_stop_20260721.json

./scripts/generate_pregrasp_only_audit.sh generate \
  --config configs/fr3_rh56_v7_commissioning.json \
  --snapshot runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --filtered-scene runs/live_scene_installed_filtered_candidate51_cable_fixed_20260721.npz \
  --joint-plan runs/candidate51_installed_air_joint_plan_after_rate_stop_20260721.json \
  --adapter assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl \
  --anydex-root third_party/AnyDexGrasp \
  --candidate-index 51 \
  --max-q-tracking-error-rad 0.002 \
  --scene-clearance-margin-m 0.005 \
  --robot-clearance-margin-m 0.002 \
  --object-clearance-margin-m 0.002 \
  --output runs/candidate51_pregrasp_only_audit_after_rate_stop_20260721.json

./scripts/generate_pregrasp_only_audit.sh verify \
  --artifact runs/candidate51_pregrasp_only_audit_after_rate_stop_20260721.json \
  --verify-files \
  --require-pass
```

然后做 executor 离线预演（同样不会导入驱动或连接设备）：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

SNAPSHOT=runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz
PREFIX_AUDIT=runs/candidate51_pregrasp_only_audit_after_rate_stop_20260721.json

./scripts/execute_control_sequence.sh pregrasp \
  --snapshot "$SNAPSHOT" \
  --pregrasp-only-audit "$PREFIX_AUDIT" \
  --selected-index 51 \
  --dry-run
```

只有预演显示 `EVIDENCE-PASS` 且没有 blocker 后，才运行一次低速真机 prefix：

```bash
./scripts/execute_control_sequence.sh pregrasp \
  --snapshot "$SNAPSHOT" \
  --pregrasp-only-audit "$PREFIX_AUDIT" \
  --selected-index 51 \
  --confirm-workspace-clear FR3_RH56_WORKSPACE_CLEAR \
  --confirm-immediate-stop IMMEDIATE_STOP_AND_24V_CUT_READY \
  --confirm-pregrasp-only FR3_RH56_PREGRASP_ONLY
```

连接顺序为 RH56 connect→立即双写/验证 `[−1]*6`→Franka connect/read-only gate；
正式 executor 强制使用 libfranka `RealtimeConfig.kEnforce`，如果进程不能取得实时
调度权限，会在创建任何 Franka 控制 handle、发送任何运动命令之前失败；
随后六轴张开并验证，再次 disable/idle 验证后才允许 Franka 运动。Franka 每个控制
周期继续执行状态/tracking watchdog。状态机到 `PREGRASP_VERIFIED` 后只允许
`arm.stop()` + RH56 disable cleanup；这个 plan 类型没有 grasp pose、q6 target、
bend target 或 lift 字段，Ctrl-C、到位失败和任一 watchdog fault 也走同一停止路径。
cleanup 必须连续两次写入并回读六轴 `[−1]*6`，而且每次物理反馈采样结束都会再次
读取 target；任何非 disable 回读都会拒绝进入 Franka 运动。

正式 executor 还会在所有离线门禁和 token 通过之后、导入真机驱动之前，以及完成
硬件只读校验后紧邻第一次运动之前，各直接只读检查一次 `/proc/net/tcp` 与
`/proc/net/tcp6`。只要存在任何到配置中 Franka IP 的
`ESTABLISHED TCP :443` 连接（通常是仍打开的 Desk/Chrome 页面），本次运动就会
fail-closed。检查不会打开网络 socket、不会发包、不会关闭连接，也不会终止浏览器；
`inspect` 和 `--dry-run` 完全不读取这项实时主机状态。开始正式 FCI 控制前应关闭所有
连接该机器人的 Desk 标签页/窗口，等待 `ESTABLISHED` 连接消失。可用下面的只读命令
自行确认；没有输出才表示这项门禁已满足：

```bash
ss -Htn state established | rg '172\.16\.0\.2:443'
```

如果内核 TCP 表不可读或内容无法解析，executor 同样拒绝导入/连接硬件，而不会把
“无法检查”误当作“没有连接”。该门禁只消除 Desk HTTPS 与 1 kHz FCI 共用专用链路
的可控竞争源，并不能替代 PREEMPT_RT、实时权限、低延迟控制循环或物理网络诊断。

Franka cleanup 也不再把 `Robot.stop()` 正常返回直接写成 `stop confirmed`。每次正常
结束、异常或 Ctrl-C 都会**先无条件调用** `Robot.stop()`，随后只读
`Robot.read_once()`：在 `2.0 s` 内必须取得至少 3 个连续样本，且每个样本均为
`RobotMode.Idle`、无 `current_errors`、四组 joint/Cartesian contact/collision 全部为
零，并且 `max(abs(dq)) <= 0.020 rad/s`。样本间隔为 `20 ms`；中间任一样本不满足会
将连续计数清零。停止命令异常、读回超时、字段缺失、持续非 Idle/碰撞或速度未降到
阈值内，都会输出 `STOP UNCONFIRMED`。这个读回门禁位于停止命令之后，因此不会阻止
或推迟 `Robot.stop()` 本身；看到 `STOP UNCONFIRMED` 时仍按现场流程立即停止 Franka
并切断 RH56 24 V。

`control_command_success_rate` 是 libfranka 给出的最近 100 条控制命令滚动成功率，
所以 `0.939999998` 表示真实的约 `0.94`，不是可用浮点 epsilon 当成 `0.95` 的显示
误差；但同一短暂丢包会连续影响约 `100 ms` 的重叠样本，也不能把单个低样本直接
当作独立故障重复计数。当前项目门禁为：每个新的 control handle 先预热 `0.10 s`；
任一样本低于项目保守 severe floor `0.80` 时立即停止；否则必须收满独立的 `0.50 s`
时间加权滑动窗口，窗口平均低于 `0.95` 才停止。每个 joint/Cartesian segment 都
重新建立 watchdog，而且配置会拒绝短于“预热 + 完整窗口”的 segment，避免用频繁
切段绕过门禁。RobotMode、current errors、接触/碰撞、关节边界、安装工具/负载、
连续 tracking error、控制周期和 wall deadline 仍逐样本 fail-closed，不经过该去抖。

2026-07-21 的第二次真机尝试记录到 `actual=0.790000021`；这不是阈值显示误差，
而是严重的实时周期丢失。复查发现旧实现把完整 `O_T_EE/F_T_EE` 刚体分解以及
末端质量、质心、惯量和 load 比较全部放进 1 kHz 循环，离线基准约
`504 us/sample`，已经耗尽官方建议的 `<500 us` read-to-write 预算。当前实现把这些
配置/安装来源检查放在每个 control handle 前，并在每段结束后完整复检；实时循环仍
逐样本检查 RobotMode、current errors、四组 contact/collision、q 有限性/关节余量、
audit-bound tracking tube、周期/deadline 和通信成功率。当前同机基准为动态 validator
约 `18 us/sample`，含 Cartesian command sampler 约 `24 us/sample`；安全阈值没有
降低。可离线复测，命令不会导入或连接硬件：

```bash
./scripts/benchmark_franka_hotloop.py --iterations 20000
```

每个真实 control handle 还会在 cleanup 时输出
`[Franka/FCI timing] kind=... samples=... max_read_to_write=...us over_500us=...`。
计时只在循环中维护整数最大值和计数，不打印、不用于新增停机门；通信 hard floor、
滑动窗口及 Franka 自身 communication reflex 仍负责 fail-closed。如果优化后仍不稳，
应根据这一行区分本机 read-to-write 超时与外部链路问题，不能继续降低阈值。

当前 rebound artifact 的 prefix 共 197 个采样点；scene capture q 与 prefix current q
的 L∞ 差为 `0.0062866 rad`，按 schema-v2 作为 advisory provenance 保留。六项硬间隙
均通过：FR3 self `195.35 mm`、adapter↔FR3 `55.26 mm`、open-RH56↔adapter
`40.37 mm`、adapter↔object `162.33 mm`、open-RH56↔FR3 `57.69 mm`、
open-RH56↔object `53.41 mm`。单视角 full-scene 的 open-RH56 回波间隙
`-17.11 mm` 仍只作现场清空提示，不能替代 workspace-clear 确认。

### 8.5 在当前静止 q 生成新鲜 full-air 审计证据

完整空抓不要手工串多条容易超时的命令。下面的入口先离线生成 joint-plan，用一次
bootstrap D435 capture/filter 只为构造完全相同的请求，并预计算 8 项与场景无关的
FR3/V7/RH56 mesh 检查；随后第二次采集最终 fresh scene，从这次采集才开始计算原有
`120 s` freshness，并只在窗口内执行 10 项 scene/object 检查和原 schema-v2 合成。
它不导入 Franka/Inspire 驱动，不发送任何运动命令。
`--capture-q-rad` 必须来自刚取得的只读 Franka 静止状态，并且从读取到相机采集期间
机械臂不能移动。这个 q 仍只是显式 provenance；正式 executor 会再次读取 live q，
超过 `0.002 rad` 仍然硬拒绝。NPZ 中固定写入的
`capture_q_source=cli_asserted` 是较弱的人工声明，并非自动 FCI 测量；full generator
会要求并 hash-bind 这个精确标签，任何声称自动测量的伪标签都会被拒绝。D435 进程
不会混入 `pylibfranka`。

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

PROFILE=configs/fr3_rh56_v7_selected_commissioned.json
SNAPSHOT=runs/d435_YYYYMMDD_HHMMSS_official.npz
CANDIDATE=SELECTED_INDEX
RUN=runs/candidate${CANDIDATE}_fresh_YYYYMMDD_HHMMSS

./scripts/prepare_fresh_installed_air_audit.sh \
  --config "$PROFILE" \
  --snapshot "$SNAPSHOT" \
  --candidate-index "$CANDIDATE" \
  --capture-q-rad Q1 Q2 Q3 Q4 Q5 Q6 Q7 \
  --confirm-stationary-q CURRENT_Q_READ_ONLY_AND_STATIONARY \
  --output-dir "$RUN"
```

先追加 `--dry-run` 可只校验输入并打印七步命令；该模式不创建目录、不打开 D435，
也不连接任何机器人，若 commissioning 尚未满足则返回 3。

中间文件 `$RUN/static_collision_cache.json` 使用原子 no-replace 发布、目录 fsync，且
权限固定为 `0444`。它精确绑定 native backend identity、config/snapshot/joint-plan
hash、V7 mesh/安装变换、完整 q path、13 个 RH56 mesh、所有手型 FK waypoint 和
dense feedback tube；scene/object 点与任何时间戳明确不在缓存内。最终 audit 的
backend 名称会明确写成 `static-cache-plus-fresh-...`，不会冒充 18 项均为本次原生
计算；8 项静态 check 各自记录 cache path、整文件 SHA-256、canonical payload
SHA-256，10 项动态 check 各自标记为最终 fresh scene/object 计算。两部分仍交给原来
同一个 policy evaluator，任何缺项、重项、顺序错误、cache/query hash 不一致都会
fail closed；executor 的 120 秒与 live-q 门没有放宽。

2026-07-21 用 candidate 51 的真实旧 D435 filtered scene（56,068 点）、214 个
canonical FR3 path sample 和 full RH56 mesh 做了纯离线计时：最终代码的静态 8 项
cache 为 `278.59 s`（发生在最终 capture 前）；新进程严格验 cache 并计算动态 10 项为
`15.94 s`，含 query 构造为 `21.69 s`，应用总墙钟 `23.46 s`。另一次完整 stale
schema-v2 构造/校验/写盘为 `18.00 s`；即使把两个 whole-app 时间保守相加也是
`41.47 s < 120 s`。旧 scene 始终被判 stale、返回 3，没有通过回拨时间生成 PASS，
正式流程仍以第二次真实 capture 时间和 audit 后 expiry check 为准。

每次必须使用新的 `RUN`；该入口没有 overwrite。成功输出位于
`$RUN/installed_air_audit_v2.json`，并打印精确过期时间和剩余秒数；过期后只能重新
采集。若 profile 的 q6 范围或 exact 六轴 target 尚未验收，流程仍可留下 plan、原始
scene、过滤 scene 和 tamper-evident receipt 供诊断，但会在 audit 阶段明确输出
`AUDIT LOCKED`、退出码 3，绝不会伪造 PASS；完成验收后必须重新采集，不能复用这份
已过期场景。

这里的 `PROFILE` 必须是 Stage 2 evidence 经 `materialize-config-update` 生成并由
`verify-applied --require-coupled` 验过的新 profile；继续使用原始
`fr3_rh56_v7_commissioning.json` 会按设计在 q6 commissioning 门处退出 3。

两类点云不能互相替代：官方 snapshot 的 `object_points` 绑定 AnyDex grasp pose、
候选和 RH56 target；新鲜 D435 full scene 只提供当前可见环境障碍。过滤器在删除目标
回波前要求旧 object cloud 与新观测满足 median `<=8 mm`、p95 `<=15 mm`、15 mm 内
coverage `>=80%`。不通过表示目标、相机或标定发生变化，应重新运行官方 AnyDex
感知；程序不会用 ICP 偷偷移动旧 grasp pose。即使通过，单视角不可见空间也不升格
为 authoritative，执行时仍需 fresh workspace-clear token。

full-audit 使用 ROS Python 3.10；本机若不能系统 `import xlrd`，wrapper 只加入
`/home/qiaoguanren/anaconda3/pkgs/xlrd-2.0.1-pyhd3eb1b0_0/site-packages` 这一纯 Python
目录。可用 `DEXGRASP_XLRD_SITE` 覆盖，但不会引入整套 Conda 包；官方 XLS mapping
工作簿仍按原路径和 SHA-256 绑定。

在开始最终 fresh capture 的 `120 s` 窗口**之前**先完成 continuous telemetry 双 ABI
构建，避免在 artifact 已开始计时后再编译：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp
./scripts/build_continuous_telemetry.sh
```

该脚本不打开 Franka、RH56、D435 或 GUI；它构建 Python 3.9 producer、Python 3.10
只读 reader，并运行 fake control 与 3.9→3.10 文件 ABI 互读测试。离线测试通过不代表
真机运动已经验收。

完整执行还必须提供 schema-v2 `installed-tool audit` sidecar。它绑定原始 snapshot
文件 SHA、selected candidate、V7 CAD/`T_EE_hand`、官方 URDF/mapping、相机 scene、
current/capture q，以及 `current -> transit... -> default -> pregrasp -> grasp`
整条关节路径。官方感知 NPZ 不会被改写；sidecar 通过 `verify_files + require_pass`
后，程序只在内存中给绑定的 candidate 附加碰撞证据。artifact 中
`motion_authorized` 永远是 `false`，实际运动仍需独立确认 token。

先做不会连接硬件的分阶段预演：

```bash
PROFILE=/absolute/path/to/selected_candidate_commissioned_profile.json
SNAPSHOT=/absolute/path/to/official_snapshot.npz
AUDIT=/absolute/path/to/installed_tool_audit_v2.json
CANDIDATE=SELECTED_INDEX

./scripts/execute_control_sequence.sh air-grasp \
  --config "$PROFILE" \
  --snapshot "$SNAPSHOT" \
  --installed-tool-audit "$AUDIT" \
  --selected-index "$CANDIDATE" \
  --trajectory-mode audited-joint \
  --dry-run
```

这条 dry-run 会同时打印 `[plan/nominal-contact-reference]` 和
`[plan/air-execution-contract]`；前者只解释网络原始接触参考，后者才是空抓审计和
执行所绑定的 `80 mm + 10 mm` 坐标。两组值不会再都标成含糊的 `grasp`。

当且仅当这份新鲜 schema-v2 air artifact 和 profile 都显示 `EVIDENCE-PASS`、
执行门无 blocker，并在现场再次确认立即停止、24 V 与整个扫掠空间后，
正式低速空抓命令模板为：
必须等 `prepare_fresh_installed_air_audit.sh` 完成最终 D435 capture 并退出后才启动
viewer，否则两个进程会争用同一台相机。artifact 仅有 `120 s` freshness，
所以 viewer 与下面的 executor 应在 audit PASS 后紧接着启动，不要留着旧
viewer 或隔几分钟再执行。

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

PROFILE=/absolute/path/to/selected_candidate_commissioned_profile.json
SNAPSHOT=/absolute/path/to/official_snapshot.npz
AUDIT=/absolute/path/to/fresh_installed_air_audit_v2.json
CANDIDATE=SELECTED_INDEX
SESSION_TAG=air_${CANDIDATE}_$(date +%Y%m%d_%H%M%S)
TELEMETRY_MAP=/tmp/fr3_rh56_${SESSION_TAG}.map
TELEMETRY_MANIFEST=/tmp/fr3_rh56_${SESSION_TAG}.manifest.json
READER_PYTHON_DIR=/tmp/anydex-native-telemetry-viewer-py310/python
PRODUCER_PYTHON_DIR=/tmp/anydex-franka-telemetry-producer-py39/python
PRODUCER_SO=$PRODUCER_PYTHON_DIR/_anydex_franka_telemetry.cpython-39-x86_64-linux-gnu.so

# 为本轮新建 immutable manifest；输出中的 SESSION_TAG 要原样复制到终端 B
./scripts/telemetry_session_manifest.sh create \
  --snapshot "$SNAPSHOT" \
  --config "$PROFILE" \
  --audit-artifact "$AUDIT" \
  --producer-build "$PRODUCER_SO" \
  --command air-grasp \
  --selected-index "$CANDIDATE" \
  --output "$TELEMETRY_MANIFEST"

echo "SESSION_TAG=$SESSION_TAG"

# 终端 A：只连接 D435，并只读等待 executor 创建 telemetry mapping
DEXGRASP_SHELL_PYTHON=/home/qiaoguanren/anaconda3/envs/dynamic/bin/python \
./scripts/run_live_pipeline_preview.sh "$SNAPSHOT" \
  --control-config "$PROFILE" \
  --selected-index "$CANDIDATE" \
  --source realsense \
  --execution-mode air \
  --continuous-telemetry "$TELEMETRY_MAP" \
  --telemetry-session-manifest "$TELEMETRY_MANIFEST" \
  --continuous-telemetry-python-dir "$READER_PYTHON_DIR" \
  --telemetry-wait-seconds 60 \
  --arm-max-age-s 0.25 \
  --hand-max-age-s 0.75 \
  --error-target auto \
  --show-current-hand-mesh \
  --hand-mesh-resolution simplified
```

终端 A 出现 `[telemetry] waiting ...`（或窗口已创建）后，立即在终端 B 运行；不要等满
60 秒，也不要在命令中使用 `SELECTED_INDEX` 字面量：

```bash
cd /home/qiaoguanren/code/franka/dexgrasp

PROFILE=/absolute/path/to/selected_candidate_commissioned_profile.json
SNAPSHOT=/absolute/path/to/official_snapshot.npz
AUDIT=/absolute/path/to/fresh_installed_air_audit_v2.json
CANDIDATE=SELECTED_INDEX
SESSION_TAG=PASTE_THE_EXACT_SESSION_TAG_PRINTED_BY_TERMINAL_A
TELEMETRY_MAP=/tmp/fr3_rh56_${SESSION_TAG}.map
TELEMETRY_MANIFEST=/tmp/fr3_rh56_${SESSION_TAG}.manifest.json
PRODUCER_PYTHON_DIR=/tmp/anydex-franka-telemetry-producer-py39/python

./scripts/execute_control_sequence.sh air-grasp \
  --config "$PROFILE" \
  --snapshot "$SNAPSHOT" \
  --installed-tool-audit "$AUDIT" \
  --selected-index "$CANDIDATE" \
  --trajectory-mode audited-joint \
  --hold-seconds 2.0 \
  --continuous-telemetry "$TELEMETRY_MAP" \
  --telemetry-session-manifest "$TELEMETRY_MANIFEST" \
  --continuous-telemetry-python-dir "$PRODUCER_PYTHON_DIR" \
  --confirm-workspace-clear FR3_RH56_WORKSPACE_CLEAR \
  --confirm-immediate-stop IMMEDIATE_STOP_AND_24V_CUT_READY \
  --confirm-air-grasp FR3_RH56_AIR_GRASP_NO_CONTACT_NO_LIFT \
  --confirm-q6-preshape RH56_Q6_PRESHAPE_VERIFIED \
  --confirm-installed-collision-model INSTALLED_TOOL_COLLISIONS_VERIFIED
```

两端的 `PROFILE`、`SNAPSHOT`、`AUDIT`、`CANDIDATE` 和 `SESSION_TAG` 必须完全相同。
每轮都生成新的 tag；不要删除后复用旧 `.map`。viewer 必须先启动，它只读等待；executor
随后用 `O_EXCL` 创建 mapping。生产者只会在原执行门全部通过并且既有 Franka/RH56
连接建立后才启动，manifest 自身的 `motion_authorized` 永远是 `false`。

窗口中的绿色 EE 来自 Franka `O_T_EE` measured feedback；绿色手 mesh 是六个 RH56
`ANGLE_ACT` 经 checksum 固定的官方 XLS/URDF 重建，不是视觉测得的实体表面。默认 active
Franka loop 名义约 `50 Hz` 发布；在手反馈校验活跃的阶段，RH56 随唯一串口 owner 的
已验证读取通常约 `5–8 Hz`。
手反馈过期时只隐藏绿色手；臂反馈过期时隐藏全部 current 几何，不会保留旧模型冒充实时。
正常 `grasp`/`air-grasp` 的 bounded hold 也不是裸 `sleep`：同一个 RH56
owner 约每 `0.20 s` 读取一次完整六轴反馈，并在同一次读取的前后复用
Franka 只读安全门。它会核对 exact `ANGLE_SET`、逐轴电流上限、状态、
`ANGLE_ACT` 和路径 envelope；空抓还要求六轴 `STATUS=2` 且零接触。
`--hold-seconds 0` 也至少做一次这个只读验证。

上面只授权 artifact 绑定的**无接触、无载荷、不抬升** round trip；不得把
`--confirm-air-grasp` 或这份 artifact 用于接触抓取或 lift。占位路径必须替换成
实际 PASS 文件，否则只会得到离线输入错误。

`grasp` 只接受 `mode=loaded_grasp` 的接触型 artifact；`air-grasp` 只接受
`mode=air_grasp` 的无接触 artifact，两者的确认 token 也不能互换。空抓的 planned
hand/EEF pose 必须从候选接触 pose 沿 `-approach` 方向后退；闭合 RH56 对物体和
场景都必须保持正间隙，序列不包含接触或 lift。当前实物 V7 是 Bambu PLA，因此
loaded `grasp` 始终拒绝，只有低速、无接触、无载荷的 `air-grasp` 可以进入后续
门禁。

`audited-joint` 模式按 artifact 中每个命名 waypoint 顺序调用关节控制，不复用
Cartesian 插值路径；pregrasp/grasp 还会核对绑定的 EEF pose。Franka 控制循环每个
周期和每段终点都检查 q tracking bound。碰撞后端必须证明离散区间之间的连续
包络、±tracking-error 管道、每项保守运动界，并使用体素化 observed-scene 及
`unknown_space=occupied` 策略；缺少任意一项时 artifact 不能 PASS，硬件保持
LOCKED。

当前 checked-in profile 已记录 RH56/适配器安装完成，并于 2026-07-18 通过
FCI + RH56 联合只读预检。Desk 与本地门禁使用的末端参数是：

```text
mass       = 0.607 kg
F_x_Cee    = [0.000, 0.000, 0.076] m
I_ee       = diag(0.001510, 0.001690, 0.000442) kg m^2
m_load     = 0 kg
m_total    = 0.607 kg
F_T_EE     = identity
```

`F_T_EE=identity` 只表示 Desk 当前把 EE 控制 frame 放在法兰中心；它不是 RH56
手掌或 AnyDex hand-source TCP。运行时门禁会同时拒绝错误的 EE 动力学、非零
`m_load` 或与 `m_ee` 不一致的 `m_total`，防止 RH56 被作为 EE 与 payload 重复
计入。FR3 官方法兰的偏心销、V7 唯一偏心销槽、ROT45 RH56 孔系以及已确认的
右手安装方向共同给出 yaw=`-45 deg (-pi/4)`：腕到指尖沿 `F +Z`，食指到小指
方向沿 `(F +X + F +Y)/sqrt(2)`。V7 profile 已记录并验算该安装变换；当前
`default`/`grasp` 仍保持 fail-closed，尚须完成并核对：

- 绑定当前关节状态与新鲜场景的 default、transit、pregrasp、final-air 连续碰撞；
- `q6` 与五个弯曲轴 exact target 的无接触真机验收；
- 与实际逐轴命令完全同源、同 hash 的 RH56 link-sweep 碰撞审计。

用户提供的 3MF 是 Bambu PLA 切片任务；其随附确认包只允许塑料件用于低速、
无载荷的装配/间隙/走线检查。带载抓取前应换用铝制适配器并重新配置末端动力学。

### 8.6 独立 loaded lift 合同与当前锁定状态

> **当前 HARD LOCK：Bambu PLA 适配器配置不得执行接触 `grasp` 或
> `grasp-lift`。本节当前可运行的只有带 `--dry-run` 的离线门禁检查；后面的正式形态
> 仅用于说明未来需要满足的接口，不是删除 `--dry-run` 的授权。**

`air-grasp` 按定义仍是 no-contact/no-lift；普通 `grasp` 仍只到 `HOLDING`，两者都
不能自动升级为抬升。新增的 `grasp-lift` 是单独的 schema-v1 round-trip 合同，只有
同时绑定通过的 schema-v2 `loaded_grasp` 基础审计和独立 `loaded-lift audit` 才存在：

```text
open → default/transit → pregrasp → grasp/settle → q6 preshape → close/contact hold
→ set_load + readback → lift-transit/lift/settle → bounded hold
→ exact reverse lift path → setdown/settle → disable hand → clear set_load
```

独立 artifact 必须绑定可承载材料审批、watertight payload mesh、payload 质量/质心/
惯量、`T_hand_payload`、保持/抗滑移 evidence、闭合 13-link FK/feedback envelope、
完整 outbound + exact reverse joint path，以及 robot/adapter/hand/payload/scene 的连续
碰撞和 tracking/hand/payload uncertainty。普通 loaded-grasp sidecar 不能代替它。
可承载 profile 还必须用 path + SHA-256 精确锁定 material、retention、
mass-properties、FK manifest、权威 collision report 和 loaded-recovery procedure，
并锁定 audit generator 及 FK/collision/mass-properties backend 的完整身份。任一文件、
算法实现、配置或恢复入口不匹配都会在硬件导入前 fail-closed。
当前 Bambu PLA profile、未验收 installed collision model、未验收六轴接触闭合以及
缺失的 loaded-lift 配置都会分别形成 blocker，因此当前仍**没有可执行的真机 lift**。

可先用已有快照做纯离线预演；下面命令会打印完整阶段和当前
profile 的具体 blocker，不导入或连接硬件：

```bash
./scripts/execute_control_sequence.sh grasp-lift \
  --snapshot runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz \
  --selected-index 51 \
  --trajectory-mode audited-joint \
  --pose-state-output /tmp/fr3_rh56_waypoint_state.json \
  --dry-run
```

不要在这个 blocker 示例中填不存在的 `/absolute/path/...` 占位文件；那会是
命令行输入错误，而不是对当前 profile 的离线门禁验证。

只有未来更换可承载适配器、重新配置动力学，可承载 profile 与两份 artifact 都显示
`EVIDENCE-PASS`、专用 loaded recovery 也完成验收、硬件 gate 无 blocker，并重新确认
现场条件后，命令形态才会是在同一模板上去掉 `--dry-run` 并补齐七个精确 token：

```bash
# FUTURE INTERFACE ONLY — current PLA configuration must not run this block
CANDIDATE=SELECTED_INDEX

./scripts/execute_control_sequence.sh grasp-lift \
  --config /absolute/path/to/load_rated_profile.json \
  --snapshot /absolute/path/to/official_snapshot.npz \
  --installed-tool-audit /absolute/path/to/loaded_grasp_audit_v2.json \
  --loaded-lift-audit /absolute/path/to/loaded_lift_round_trip_v1.json \
  --selected-index "$CANDIDATE" \
  --trajectory-mode audited-joint \
  --lift-hold-seconds 1.0 \
  --pose-state-output /tmp/fr3_rh56_waypoint_state.json \
  --confirm-workspace-clear FR3_RH56_WORKSPACE_CLEAR \
  --confirm-immediate-stop IMMEDIATE_STOP_AND_24V_CUT_READY \
  --confirm-loaded-lift-round-trip FR3_RH56_LOADED_LIFT_ROUND_TRIP \
  --confirm-setdown-support LOAD_SETDOWN_SUPPORT_READY \
  --confirm-load-rated-tool LOAD_RATED_ADAPTER_AND_PAYLOAD_VERIFIED \
  --confirm-q6-preshape RH56_Q6_PRESHAPE_VERIFIED \
  --confirm-installed-collision-model INSTALLED_TOOL_COLLISIONS_VERIFIED
```

这是一份未来 PASS 配置的命令模板，不是对当前 PLA 件的授权；当前 checked-in profile
运行它仍会在导入硬件前返回 LOCKED。

未来若同时启用 continuous telemetry，session manifest 的 `--command` 必须是
`grasp-lift`，`--audit-artifact` 必须绑定独立的
`loaded_lift_round_trip_v1.json`；不能拿普通 loaded-grasp 或 air-grasp audit 代替。
viewer 应显示 `--execution-mode contact`，executor 仍分别使用 Python 3.10 reader 与
Python 3.9 producer 目录、全新的 `/tmp` mapping，并先启动只读 viewer。telemetry 只增加
可观测性，不会解锁当前 PLA gate，也不会替代 loaded collision、retention、setdown 或
recovery evidence。

`--pose-state-output` 只在真机执行时由**已有的唯一 FCI/串口 owner**在已验证的
waypoint/stage 边界原子发布 JSON，覆盖 default、transit、pregrasp、grasp、close、
lift 和 setdown；dry-run 不创建该文件。它是 waypoint-rate 状态，不是连续 1 kHz
stream，也不会在 realtime callback 中读设备、写盘或打印。第 3.1 节的 viewer 可用
同一路径作为 `--pose-state`；它会用精确 target 选择误差目标，并可把六轴
`ANGLE_ACT` 重建为绿色当前手模型，同时继续忽略仅供日志的 q、力、电流和错误字段。

lift 顶端的 bounded hold 不是裸 `sleep`：同一 owner 每 `0.20 s` 重新只读验证 Franka
Idle/末端与 active payload provenance，并复核 RH56 exact numeric target 和最少接触轴。
带载期间的失败语义与普通 cleanup 不同：一旦物体可能离开支撑面，Ctrl-C、通信、
tracking、hold readback 或 pose-state 发布失败都会先停止 Franka，并故意保留 RH56
数值抓持及 Franka external-load 配置；程序会输出 `LOADED RECOVERY REQUIRED`，绝不
自动松手。操作者必须先物理支撑/放回载荷；直接断 24 V 会让手失去保持力。
当前 repo **还没有**可执行的 loaded-recovery CLI；现有
`recover_installed_rh56` 只允许 PLA 低速无载操作，不能用于带载恢复。因此专用恢复流程、
其权威 evidence 与 profile pinning 未完成也是未来解锁带载执行的硬 blocker；
在此之前不得尝试真机 lift。只有 exact reverse 到审计 setdown、q/FK/EEF settle 和接触 hold
全部验证后，正常路径才会先禁用手、再清除 `set_load`。

## 9. 当前验证结果与限制

- 自动化回归在 `dexgrasp/` 目录由
  `PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests`
  重跑；本次最终结果为 `703 passed, 11 skipped`。
  覆盖 snapshot/frame、官方候选绑定、Inspire 映射、12-joint FK、13-link mesh、
  适配器 provenance/连续碰撞包络、mount frame、联合状态机、Franka/RH56
  fail-closed 驱动、exact commissioning evidence、离线执行门禁和独立验收 CLI。
- 真机样例：frame 6，`scene=15525`、`object=1170`、7 个 geometric 候选；
  serial 和 calibration ID 与上述配置一致。
- `run_capture.sh --help`、`run_snapshot_viewer.sh --help`、
  `run_official_snapshot.sh --help`、`run_grasp_generation_live_preview.sh --help`
  以及环境检查 CLI 均已实际通过。
- 已生成并绑定官方 AnyDex 表示的 84 个去重候选；当前选择 candidate 51
  （source 54/type 7，score 约 0.9209），六轴目标为
  `[0,358,799,911,922,646]`。这不等于真机抓取已授权。
- 安装态碰撞审计使用 Pinocchio + HPP-FCL、官方 FR3 collision meshes、V7 STL、
  RH56 exact FK/meshes 和带连续区间/tracking tube 的场景点检查；不再使用旧的
  `collision_free=True` 兼容字段作为证据。单视角点云与自体回波过滤仍是条件证据，
  不能覆盖相机不可见空间。
- 2026-07-21 fresh-scene 回归采用 `20 mm` 硬上限的“模型距离 + 同部件外观调色板
  + 连通图”自回波过滤；FR3/scene 改善到 `+2.453 mm`，但 open RH56/scene 仍为
  `-17.107 mm`（`Link111`/`Link44`）。最后残留距 capture-state Link111
  `20.429 mm`，没有扩大上限删除它，因此该场景保持 LOCKED，不能真机执行。
- geometric backend 同样没有完整手模型—场景碰撞、可达性或抓取质量验证。
- 分割结果依赖 ROI、深度覆盖与相机不移动；目标点云质量差时 pose 也会失真。
- 当前 V7 profile 的 q6 仍只验收到 `900..1000`，且 exact target 列表为空；所以
  candidate 51 的 q6=646 与六轴闭合均保持 LOCKED。只有两阶段真机验收 evidence、
  三字段 profile 更新、重新生成的 joint-plan、两次新鲜场景审计全部通过后，才可
  进入低速、无接触、无载荷的正式空抓序列。
- 最新 candidate 51 full-air 离线诊断覆盖 211 个 Franka path samples 和 191 个
  RH56 command waypoints。离散 waypoint 没有发现 mesh 穿透，但全路径名义 self
  clearance 最小值只有 `1.047271 mm`，低于固定的 `2 mm` robot margin；此外
  `Link11/Link22` 的区间 81→82 在旧的全反馈包络下只能证明
  `7.916749 - 42.424913 = -34.508164 mm`。因此当前结论是“无法证明连续安全间隙”，
  不是 PASS，也不能靠 operator token、降低 margin 或忽略该 pair 解锁。
