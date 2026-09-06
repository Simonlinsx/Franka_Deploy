# RealSense → MANO → Inspire RH56BFX-2R

这套 example 把本机已有的四部分连成一条实时链路：

```text
RealSense 对齐 RGB-D
  → WiLoR-mini（MANO pose、shape、778 顶点、21 关节点）
  → geometric retargeting（默认）或 dexsuite vector retargeting（实验预览）
  → 六轴标定与限速
  → RH56 USB-RS485 批量 ANGLE_SET
```

默认只做预览，绝不会打开串口。真机输出除了两个 motion flag，还强制要求显式
给出相机序列号、串口、标定文件、retargeter 和开放轴。当前只允许 geometric
retargeter 驱动真机；Dex 输出尚未完成人体手势语义标定，只能用于实验预览，CLI
会拒绝 `--retargeter dex --enable-hardware`。

## 已使用的本机资源

- WiLoR-mini 源码和权重：
  `/home/qiaoguanren/下载/WiLoR_OL/wilor_mini`
- dex-retargeting 0.5.0 源码、Inspire URDF 和配置：
  `/home/qiaoguanren/桌面/brainco/dex-retargeting`
- RH56 直连协议驱动：`examples/inspire_rh56_test.py`
- 运行环境：模块内 `examples/inspire_mano_pipeline/.venv`，继承 `dynamic` Conda 环境的
  Torch 2.5.1+cu121、OpenCV、NumPy 和 pyrealsense2。

启动脚本会清除 ROS Humble 的 `PYTHONPATH`，并让隔离环境的
Pinocchio/EigenPy 动态库优先，避免 ABI 冲突。

## 当前验证边界

截至本文更新时，环境导入、离线协议、retargeting 数值、模拟 watchdog、
RealSense 真人右手五指预览、单食指全量程真机运行，以及五指窄行程第一次 ACTIVE
和旧版软件电流停机都已有日志证据。旧逻辑曾分别在五轴合计 `505/611/632 mA`、
六轴合计 `619 mA` 时锁存退出；这些帧的单轴峰值不超过 `220 mA`、故障码全 0，
说明 `stream_max_current_ma` / `stream_max_total_current_ma` 与正常多轴并发运动电流
重叠，不能作为正式六轴实时控制的硬故障阈值。当前六轴 profile 显式使用
`active_current_policy=monitor_only`，把 `400/600 mA` 保留为 ACTIVE 遥测 warning；
启动前的空闲电流检查仍然 fail closed，并且只有回读确认每个所选轴的设备
`CURRENT_LIMIT` 已启用才允许 armed，固件 CURRENT_LIMIT 和状态/故障反馈继续负责
硬保护。其他没有显式设为 `monitor_only` 的 profile 默认仍为 `fault`，不能从六轴
策略外推。因此目前是“五指/六轴命令链已打通、旧停机原因已定位”，还不能表述成
分级真机 commissioning 全部完成。

- D435 未镜像实时流已经识别到物理右手，有效掌心深度约 `0.24..0.35 m`，MANO
  目标会随手势变化；这只证明实时视觉/retargeting 预览链路工作，不等同于真机
  逐指方向和量程已经标定；
- RH56 的 CH340 串口已在 runtime 屏蔽 BRLTTY 后稳定枚举为
  `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`；动作前只读状态为故障码 0、
  电流 0 mA、温度 `28..32 °C`、五个弯曲轴实测 `1000`；
- open1000 单食指 20 秒运行进入 ACTIVE 237 帧，实际发送食指目标 `922..1000`、
  实测角度 `970..1000`、峰值电流 121 mA、最高 34 °C，其他轴始终为 `-1`；
  `shutdown_status.json` 的 session 匹配、`stop_confirmed=true`、最终六轴为 `-1`；
- 另一次默认 `disable` 运行在 ACTIVE 127 帧后移手，最后有效目标后约 `0.794 s`
  进入 `fault_latched`；食指实测 `930..992`、峰值 117 mA、最高 34 °C、故障码全 0，
  session 匹配且最终六轴 `-1`，因此 ACTIVE 后 tracking watchdog 已有真机证据；
- 三轮真人数据表明旧 `q_closed=1.47` 严重低估闭合量程；食指已经重标定为
  `q_open=0.05`、`q_closed=0.65`（命令软范围仍为 `800..1000`），离线测试通过，
  但该增益调整后的真机量程和 `no-hand-policy=open` 回到 `1000` 仍待复验；
- 五指 full-range 预览共接受 279 个右手帧，深度有效 `260/279`（`93.19%`），五个
  弯曲轴的预览目标都覆盖 `0..1000`，新拇指掌面角也覆盖完整弯曲范围；
- 五指窄行程第一次真机运行进入 ACTIVE 18 帧，worker 对五个弯曲轴都发送了数值
  目标（四指最低 `946`、拇指最低 `956`），任一轴故障帧为 0；五轴启动电流分别为
  `123/101/117/109/55 mA`，合计 `505 mA`，触发当时 `500 mA` 的总门后进入
  `fault_latched`，最终六轴 `-1`、session 匹配且停机回读为 safe。该运行证明五轴
  输出生效，也证明旧软件总电流阈值会把正常同步启动误判成硬故障；它不能作为
  窄行程动作范围通过证据；
- 随后的五轴 `611/632 mA` 和六轴 `619 mA` 运行重复了相同模式：多个状态正常的
  执行器同时运动，任一轴都没有达到 profile 的单轴 warning 值，设备故障寄存器也
  始终为 0。`619 mA` 六轴运行中第六轴只贡献 `23 mA`，根因不是第六轴故障，而是
  对正常并发电流求和后用单个样本硬停；该次退出还因释放后 STATUS 未及时回到 idle
  而记录为 `STOP_UNCONFIRMED`，不能作为通过证据。这些旧日志只用于说明策略变更，
  必须用新的 monitor-only token 重新完成 20 秒验收；
- 五指窄行程、200 档和 300 档 profile 已加入并必须逐档验证，不能因预览或第一次
  ACTIVE 就直接跳到 200/300 档；
- 常规五指 profile 仍禁用 `thumb_rotate`；第六轴已经完成
  `1000 -> 0 -> 1000` 额定全行程单轴验收。在此基础上新增了精确受限的六轴实时
  profile：五个弯曲轴保留 `0..1000` 全量程，第六轴只开放张开侧 `1000..900`，
  并用独立低速门控制。旧的旋转轴单轴和 `900..800` 六轴窄行程 profile 仍保留作
  commissioning 历史与排障入口，不应再当作正式实时命令。Dex 真机输出仍保持禁用；
- 六轴联合实时会话仍应先完成本文给出的 20 秒现场验收，再进入不限时运行。单轴
  全行程通过证明执行轴、方向和协议路径正常，不自动等同于六轴 MANO 耦合动作已经
  完成现场验收。

## 当前推荐使用流程：五指实时跟随

下面是本机当前应使用的最短完整 SOP。不要同时运行两个 RH56 控制进程；每次真机
运行前都要确认 24 V 已上电、手周围清空、操作者能立即切断 24 V，并从张开的真人
右手开始。五指运行固定控制 `pinky,ring,middle,index,thumb_bend`，不会控制
`thumb_rotate`。需要控制第六轴时，直接使用后文“六轴实时联合控制（open1000）”的
精确 profile 和确认 token，不要在五指命令后手工追加轴名。

### 0. 设置本机参数

```bash
cd /home/qiaoguanren/code/franka
CAMERA_SERIAL=337322072188
PORT=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
FIVE_AXES=pinky,ring,middle,index,thumb_bend
```

### 1. 首次安装后或代码更新后跑离线检查

```bash
examples/setup_inspire_mano_env.sh       # 环境已经建好时可跳过
examples/test_inspire_mano_pipeline.sh
examples/check_inspire_mano_devices.sh --gpu
```

测试必须通过；本机当前基线为 `204/204`，并额外通过 RH56 V1.09 协议帧自检。设备
检查中应看到 D435 序列号、`cuda:0` 可用以及稳定的 RH56 by-id 串口。

### 2. 真机只读检查并低速张开五个弯曲轴

```bash
python3 examples/inspire_rh56_test.py status --port "$PORT"

python3 examples/inspire_rh56_test.py open \
  --port "$PORT" --speed 80 --force-limit 80 --motion-timeout 20 \
  --confirm-movement

python3 examples/inspire_rh56_test.py status --port "$PORT"
```

第二次 `status` 必须显示六个 `ANGLE_SET` 全为 `-1`、五个弯曲轴接近 `1000`、故障
全 0、空闲电流接近 0 且温度低于 `60 °C`。`open` 会产生动作，但完成后会恢复
`ANGLE_SET=-1`；拇指旋转轴不会改变。

### 3. 先开 GPU 五指预览窗口，不连接机械手

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_five_finger_fullrange_realtime200.json \
  --retargeter geometric --axes "$FIVE_AXES" \
  --no-hand-policy disable \
  --output-dir /tmp/inspire-mano-five-preview --overwrite-output
```

这条命令没有 `--enable-hardware`，不会打开串口。窗口应同时显示 camera image、MANO
mesh、21 点骨架和五指 `T`。把单只右手完整放在画面中央约 `25..40 cm` 处，依次做
“完全张开 → 逐指弯曲 → 握拳 → 完全张开”；五个 `T` 应各自从接近 `1000` 降低，
完全动作时能够接近 `0`。若显示 `NO HAND`，先看实际相机画面而不是改阈值；本次
现场曾因相机只拍到空桌面而连续 0 检测。按 `q`、`Esc` 或 Ctrl-C 退出预览。

预览退出后检查日志：

```bash
python3 examples/analyze_inspire_mano_log.py \
  /tmp/inspire-mano-five-preview/mano_retarget.jsonl
```

只有右手检测、深度有效率和五轴方向/范围都正常时才进入真机。

### 4. 第一档：五指窄行程真机

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_five_finger_commissioning_open1000.json \
  --retargeter geometric \
  --enable-hardware --confirm-hardware-motion \
  --confirm-wide-range RH56_WIDE_RANGE \
  --port "$PORT" --axes "$FIVE_AXES" \
  --no-hand-policy disable --duration 20 \
  --output-dir /tmp/inspire-mano-five-hardware-narrow --overwrite-output
```

先让真人右手保持完全张开；连续 5 个稳定、深度有效的 MANO 帧后才会从
`WAITING_FOR_TRACKING` 进入 `ACTIVE`。然后只做缓慢、小幅弯曲。此档四指目标限制为
`800..1000`，拇指弯曲限制为 `850..1000`，speed 为 `120`。该五轴 profile 的
`active_current_policy` 仍为 `fault`：单轴 `500 mA`、所选五轴绝对电流和 `600 mA`
任一超限都会锁存停机。窗口中的 `T/S/A` 分别是识别目标、worker 实际发送目标和
机械手实际角度。

### 5. 每档结束后做日志验收

```bash
python3 examples/analyze_inspire_mano_log.py \
  /tmp/inspire-mano-five-hardware-narrow/mano_retarget.jsonl
python3 examples/inspire_rh56_test.py status --port "$PORT"
```

窄行程只有同时满足以下条件才算通过：

- `ever_active=True`，五个 selected axes 都有实际发送目标且五个 `ANGLE_ACT` 都随
  正确方向变化；
- 六轴故障始终为 0，最高温度 `<60 °C`，单轴和五轴总电流都未触发该 profile 的
  `fault` 门槛；
- `shutdown.verification_status=safe`、`stop_confirmed=true`、session 匹配且最终六轴
  `ANGLE_SET` 全为 `-1`；
- 现场没有自碰、卡滞、反向或异常声音。

若报告为 `fault_latched`，先保留 run 目录并查明原因，不能直接提高速度或量程。
本机第一次五轴运行的 `505 mA > 500 mA` 是正常五轴启动电流触发了过紧的旧硬停
逻辑；后续 `611/632/619 mA` 且故障码为 0 的并发动作进一步确认了这一点。正式六轴
profile 因此把 `400/600 mA` 改成 monitor-only warning；本节五轴 profile 仍保留
旧 `fault` 策略，若触发仍会退出。任何 `fault_latched` 都应同时检查设备
ERROR/STATUS、温度、tracking timeout、串口和停机确认原因。

### 6. 窄行程通过后再逐档提速

窄行程完整通过后，使用本 README 后面的“[五指真机分级命令](#五指真机分级命令当前进度)”
先运行 full-range 200 档；200 档另一次运行也通过后，才运行 300 档。200 档四指/
拇指软件速率为 `200/160 units/s`；300 档为 `300/200 units/s`。GPU 有效右手推理
约 `52 ms`，此前 speed/rate 200 的实测表明主要延迟来自运动速率限制，因此 300 档
用于缩短全行程时间，而不是关闭滤波或 watchdog。

### 7. 停止与无手行为

- 正常停止：窗口按 `q`/`Esc`，或终端 Ctrl-C；程序会先停在当前位置，再双重写入
  并回读六轴 `-1`。
- 紧急情况：立即切断 24 V；`-1` 不是物理急停，也不会切断电源。
- 五指真机固定使用 `--no-hand-policy disable`。ACTIVE 后丢失有效新目标超过
  `0.75 s` 会锁存停机并退出；未进入 ACTIVE 时没有手只会保持等待，不会运动。
- 只有控制进程已经退出、串口不再被占用时，才可单独执行：

  ```bash
  python3 examples/inspire_rh56_test.py disable --port "$PORT"
  ```

## 实时稳定性排查与拇指第二自由度

这一节用于解决两类现场问题：画面里明明有手却反复失去控制，以及在保留安全门的
前提下启用 `thumb_rotate`。下面所有预览命令都不打开串口；带
`--enable-hardware` 的命令会产生真实动作。

### 为什么会不稳定

- **掌心深度缺失**：D435 在手掌贴近画面边缘、手指遮挡掌心、表面反光或深度纹理
  不足时会产生空洞。程序会对掌心做鲁棒采样，并最多短时保留约 `0.12 s` 的最近
  实测深度用于显示；状态为 `DEPTH HELD ... / CONTROL INHIBITED` 时不会把它当成新的
  真机控制证据，也不会刷新 `0.75 s` watchdog。测得但过近/过远的深度同样不会被
  旧值掩盖。
- **`MULTIPLE HANDS`**：相机同时看到操作者右手和 Inspire 右手时，严格单手门会拒绝
  当前帧。不要在真机模式加 `--allow-multiple-right-hands`（CLI 也会拒绝）；应使用
  operator ROI，把机械手排除在控制区域外。
- **USB 2 链路**：`lsusb -t` 显示相机为 `480M`，或
  `Usb Type Descriptor: 2.1`，表示 D435 没有建立 SuperSpeed 链路。此时
  `848x480@30` RGB-D 可能无法启动或持续丢帧。换用主板直连 USB 3.x 端口和
  SuperSpeed 数据线，直到显示 `5000M` 或更高以及 USB `3.x`。
- **有效手约 20 FPS**：D435 输入是 30 FPS，但当前 GPU 上有效右手的 WiLoR 推理约
  `48..52 ms`，所以新的 MANO 目标约为 20 FPS。提高串口控制频率不会凭空产生更多
  视觉目标；CPU 更慢，真机模式必须显式使用 `--device cuda:N`。

### 六轴实时联合控制（open1000）

当前正式六轴 profile 是
`inspire_rh56bfx_right_six_dof_open1000_realtime.json`。它只允许精确的六轴顺序
`pinky,ring,middle,index,thumb_bend,thumb_rotate`，不能删轴、换序或与其他 profile
混用。先统一设置本机变量：

```bash
cd /home/qiaoguanren/code/franka
CAMERA_SERIAL=337322072188
PORT=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
OPERATOR_ROI=0.35,0.10,0.98,0.98
SIX_AXES=pinky,ring,middle,index,thumb_bend,thumb_rotate
PROFILE=examples/inspire_mano_pipeline/inspire_rh56bfx_right_six_dof_open1000_realtime.json
```

这个 profile 的范围和速度有意不对称：

- 小拇指、无名指、中指、食指和拇指弯曲保留 `1000=张开`、`0=闭合` 的完整范围；
  四指软件限速为 `200 units/s`，拇指弯曲为 `160 units/s`；
- `thumb_rotate` 在额定单轴全行程验收中已经证明能够走 `1000 -> 0 -> 1000`，但
  当前 geometric 第六维是“拇指指尖相对食指 MCP 的对掌距离”代理量，不是纯粹的
  解剖旋转角。为避免弯曲和对掌耦合造成意外大动作，实时命令只映射到张开侧
  `1000..900`；该轴单独使用 `SPEED_SET=80` 和软件限速 `40 units/s`。不要把配置
  手改为 `0..1000`，也不要用五指的 `SPEED_SET=200` 覆盖第六轴；
- 六轴都使用 `median_window=3`、`EMA alpha=0.60`。ACTIVE 遥测 warning 参考值为
  单轴 `400 mA`、所选六轴绝对电流和 `600 mA`，force 为 `80 g`。warning 会写入
  日志并累计 `active_current_over_limit_sample_count` / `active_current_warning_event_count`，
  但不会单独锁存停机；该行为由显式 `active_current_policy=monitor_only` 开启。启动时
  必须成功回读六轴设备 `CURRENT_LIMIT`，且每个所选轴都为 `1..1500 mA` 的有效启用
  值，否则 fail closed。固件 CURRENT_LIMIT、ERROR/STATUS 和温度反馈才属于运行中的
  硬保护信号。

#### 1. 先开六轴实时预览窗口，不连接 Inspire

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --operator-roi "$OPERATOR_ROI" --device cuda:0 \
  --calibration "$PROFILE" --retargeter geometric \
  --axes "$SIX_AXES" --no-hand-policy open \
  --output-dir /tmp/inspire-mano-six-open1000-preview --overwrite-output
```

这条命令没有 `--enable-hardware`，只打开 camera image、MANO mesh、21 点骨架和六轴
目标的实时窗口，不会打开串口。先让真人右手完全张开，再缓慢逐指弯曲、握拳、张开
并做小幅对掌。五个弯曲轴应能覆盖完整 `0..1000`，第六轴只应在 `900..1000` 内
变化。无手时预览显示六轴 `CALIBRATED_OPEN=[1000,1000,1000,1000,1000,1000]`。
若第六轴方向、连续性或画面跟踪不对，不进入真机。

#### 2. 第一次六轴真机联合运行固定 20 秒

先确认 24 V 已上电、手周围清空、能立即断电；确保没有其他串口控制进程，并用
`status` 确认六个 `ANGLE_SET` 全为 `-1`、六轴均在张开端附近、故障为 0、空闲
电流接近 0、温度低于 `60 °C`。然后从完全张开的真人右手开始：

```bash
python3 examples/inspire_rh56_test.py open \
  --port "$PORT" --include-thumb-rotate \
  --speed 40 --force-limit 80 --motion-timeout 20 \
  --confirm-movement

python3 examples/inspire_rh56_test.py status --port "$PORT"

examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --operator-roi "$OPERATOR_ROI" --device cuda:0 \
  --calibration "$PROFILE" --retargeter geometric \
  --enable-hardware --confirm-hardware-motion \
  --confirm-wide-range RH56_WIDE_RANGE \
  --confirm-six-dof-motion RH56_SIX_DOF_REALTIME \
  --confirm-current-monitor-only RH56_ACTIVE_CURRENT_MONITOR_ONLY \
  --port "$PORT" --axes "$SIX_AXES" \
  --no-hand-policy open --confirm-no-hand-open CALIBRATED_OPEN \
  --duration 20 \
  --output-dir /tmp/inspire-mano-six-open1000-hardware-monitor-20s \
  --overwrite-output
```

`open --include-thumb-rotate` 先张开五个弯曲轴，再把第六轴从已验证的开侧范围单程
低速移到命令 `1000`；它不会先闭合或返回原位置。第六轴反馈稳定达到至少 `980`
后会释放并回读六个 `ANGLE_SET=-1`。本机命令 `1000` 的实测稳定反馈为 `983`，经
profile 的 `+15` 反馈偏置映射为接近 `1000`，属于正常张开端。

连续 5 个新鲜、深度实测且目标稳定的 MANO 帧后才会进入 `ACTIVE`。第一次 20 秒
只做缓慢、可随时停下的动作，确认窗口中六个轴的 `T/S/A` 同向变化；尤其观察
`thumb_rotate` 只能在 `900..1000` 内低速移动，不能把真人拇指一次压到极限。
`RH56_ACTIVE_CURRENT_MONITOR_ONLY` 明确确认操作者知道 host `400/600 mA` 只告警、
不会单独停机；缺少该精确 token 时 CLI 会拒绝 monitor-only 真机输出。

结束后验收本次日志和只读状态：

```bash
python3 examples/analyze_inspire_mano_log.py \
  /tmp/inspire-mano-six-open1000-hardware-monitor-20s/mano_retarget.jsonl
python3 examples/inspire_rh56_test.py status --port "$PORT"
```

只有 `ever_active=True`、六轴均有实际发送记录、方向和现场动作正确、设备无 ERROR/
异常 STATUS/过温，并结合 warning 前后的电流与 `T/S/A` 确认没有卡滞，且
`shutdown_status.json` 同时给出 `stop_confirmed=true`、
`physical_stop_verified=true`、session 匹配和最终六轴 `ANGLE_SET=-1`，这次 20 秒
验收才算通过。还要确认 `active_current_policy=monitor_only`、
`verified_device_current_limits` 六个值均有效，并记录
`active_current_over_limit_sample_count`、`active_current_warning_event_count`、
`active_peak_abs_currents` 和 `active_max_selected_total_current_ma`；warning 次数不是
自动通过条件，必须结合现场是否存在持续受力、卡滞或异常声音判断。

#### 3. 20 秒验收通过后不限时实时控制

不限时命令与上面完全相同，只移除 `--duration 20`。不指定 `--output-dir` 时程序会
自动创建带时间戳的新目录，避免覆盖验收证据：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --operator-roi "$OPERATOR_ROI" --device cuda:0 \
  --calibration "$PROFILE" --retargeter geometric \
  --enable-hardware --confirm-hardware-motion \
  --confirm-wide-range RH56_WIDE_RANGE \
  --confirm-six-dof-motion RH56_SIX_DOF_REALTIME \
  --confirm-current-monitor-only RH56_ACTIVE_CURRENT_MONITOR_ONLY \
  --port "$PORT" --axes "$SIX_AXES" \
  --no-hand-policy open --confirm-no-hand-open CALIBRATED_OPEN
```

窗口按 `q`/`Esc` 或终端 Ctrl-C 正常结束。`--no-hand-policy open` 不会在启动时凭空
拉动机械手：必须先有 5 帧有效 MANO 并进入过 `ACTIVE`；之后在相机仍持续产出新帧
的前提下，连续约 `0.30 s` 没有有效右手或深度无效，才会把六轴限速回退到
`[1000,1000,1000,1000,1000,1000]`。重新看到右手后仍需重新满足稳定帧门槛。
如果相机、推理或控制目标本身停顿超过 `0.75 s`，watchdog 仍会锁存故障、释放
输出并退出，fallback 不能绕过 watchdog。

正常、故障和 Ctrl-C 路径在回读六轴 `ANGLE_SET=-1` 后，还会连续采样实际角度、
电缸位置、状态、电流、温度和故障，确认机械反馈与电流都已稳定，才设置
`physical_stop_verified=true` 并恢复原来的 SPEED/FORCE 设置。因此这里的
`stop_confirmed=true` 同时表示目标已释放且物理/电气反馈已停止；若出现
`STOP UNCONFIRMED`、`physical_stop_verified=false` 或程序没有生成停机文件，立即
切断 24 V，不要重启控制命令尝试“恢复”。

### 旧 `900..800` commissioning 预览（历史与排障）

以下内容保留第六轴从“不动”排查到单程探测、窄行程和额定全行程验收的现场依据。
正式实时使用上面的 `inspire_rh56bfx_right_six_dof_open1000_realtime.json`；除非正在
复现历史问题，不要再把旧 `900..800` profile 当作当前入口。

本机当前建议的 ROI 是 `0.35,0.10,0.98,0.98`。四个数字分别是归一化的
`x1,y1,x2,y2`；检测框**中心**落在矩形内才会作为操作者手。窗口会画出 ROI。若操作者
手显示 `HAND OUTSIDE OPERATOR ROI`，先在无硬件预览中小幅调整边界，不要扩大到重新
包含机械手。

```bash
cd /home/qiaoguanren/code/franka
CAMERA_SERIAL=337322072188
PORT=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
OPERATOR_ROI=0.35,0.10,0.98,0.98
SIX_AXES=pinky,ring,middle,index,thumb_bend,thumb_rotate

examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --operator-roi "$OPERATOR_ROI" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_six_dof_commissioning_900_800.json \
  --retargeter geometric --axes "$SIX_AXES" \
  --no-hand-policy disable \
  --output-dir /tmp/inspire-mano-six-preview --overwrite-output
```

这条命令没有 `--enable-hardware`，不会连接 Inspire。保持单只真人右手完整位于 ROI
内部，依次做张开、握拳和拇指向掌心对掌动作。确认窗口持续为 `TRACKING OK`，并观察
`thumb_rotate` 目标随对掌动作连续变化；按 `q`、`Esc` 或 Ctrl-C 退出。然后检查：

```bash
python3 examples/analyze_inspire_mano_log.py \
  /tmp/inspire-mano-six-preview/mano_retarget.jsonl
```

### 为什么之前关闭拇指旋转轴

RH56 的拇指有 `thumb_bend` 和 `thumb_rotate` 两个执行轴。之前只启用弯曲轴，是因为：

- `thumb_rotate` 更容易造成拇指与掌面或食指自碰；
- 旧配置中的 `300..700` 只是未验证的占位范围，而这台真机禁用状态下的实测角度约
  为 `887`，直接启用旧范围会被 preflight 拒绝；绕过检查则可能产生很大的首次动作；
- MANO 第六维使用拇指对掌几何量，虽然预览中会变化，但还需要现场确认机械方向、
  安全范围以及它与 `thumb_bend` 的耦合。

因此当时的 commissioning 配置只允许旋转轴在观测位置附近走 `900..800`。随后
额定全行程单轴验收已经通过，但 geometric 第六维仍不是纯旋转角；当前正式实时
profile 因而改用已张开真机对应的 `1000..900` 受限范围，而不是直接实时开放
`0..1000`。

### 拇指第二自由度的安全分阶段流程（commissioning 记录）

每个真机阶段开始前都必须确认：24 V 已上电、手周围清空、能立即切断 24 V、没有
其他控制进程占用串口。任一阶段出现反向、自碰、卡滞、异常声音、电流/温度故障或
`STOP UNCONFIRMED`，立即停止并切断 24 V，不能继续下一阶段。

#### 1. 张开五指后验证旋转轴的六寄存器批量控制

先张开五个弯曲轴，再确认六个 `ANGLE_SET` 全为 `-1`、五个弯曲轴实际角均不低于
`950`、故障全 0、温度低于 `60 °C`，并记录 `thumb_rotate` 的 `ANGLE_ACT`。实测中，
单独写第六个角度寄存器时，负方向 `885 -> 865` 曾到达 `859`，但正向返回没有到达；
另一次正向 `855 -> 875` 虽然目标回读正确，电机电流始终为 `0 mA`，实际角仍为
`855`。因此不能把“寄存器写入成功”当作第六轴运动成功。

正式实时控制会从 `ANGLE_SET=1486` 一次写六个寄存器。批量 `850 -> 870` 真机测试
同样无运动：固件把目标转换为 `POS_SET=593`，而 `POS_ACT=601`，8 个位置计数的
误差被直接判为 `STATUS=2`，电流仍为 `0 mA`。这排除了单寄存器写法的影响。

多组反馈可近似拟合为 `POS_ACT ~= 1791 - 1.4*ANGLE_ACT`，目标映射约为
`POS_SET ~= 1811 - 1.4*ANGLE_SET`，两者相差约 20 个位置计数，也就是 14--15 个
角度单位。旧恢复路径把当前 `ANGLE_ACT` 重写为 `ANGLE_SET` 时会产生反向目标；实测
末尾的 `11 mA/STATUS=1` 就来自这一偏置。只写六轴 `-1` 也不会立即清除固件内部的
`POS_SET`：首次单程探测结束后，目标虽已全为 `-1`，第六轴仍以约 `25 mA` 继续向
旧位置目标运动。因此正常结束不能只依赖 `-1`，也不能重写原始反馈角度。

单程 `+30` 探测使用正式控制的六寄存器帧
`[-1, -1, -1, -1, -1, target]`，但不会自动回到旧起点：

```bash
python3 examples/inspire_rh56_test.py status --port "$PORT"

python3 examples/inspire_rh56_test.py open \
  --port "$PORT" --speed 80 --force-limit 80 --motion-timeout 20 \
  --confirm-movement

python3 examples/inspire_rh56_test.py status --port "$PORT"

python3 examples/inspire_rh56_test.py probe \
  --port "$PORT" --joint thumb_rotate --include-thumb-rotate \
  --delta 30 --speed 40 --force-limit 80 --motion-timeout 2 \
  --confirm-movement

python3 examples/inspire_rh56_test.py status --port "$PORT"
```

该探测已在真机通过：起点 `ANGLE_ACT/POS_ACT=846/607`，命令
`ANGLE_SET/POS_SET=876/585`；约 0.8 秒后反馈到 `856/592`，随后到 `857/591`，
峰值电流 `42 mA`，方向正确、故障为 0。这证明第六轴电机、正向驱动和六寄存器
批量控制都正常，之前不动是命令/反馈域偏置与固件到位窗共同造成的。

脚本会拒绝
弯曲轴未完全张开、旋转目标超出 `800..900`、六轴目标不全为 `-1` 或静止电流异常
的情况。`probe` 在角度变化至少 3 或电缸位置变化至少 5、且连续两个样本确认后立即
结束；轮询期间还会检查方向、电流、力、温度、故障以及其余五轴没有明显漂移。
检测到运动后，脚本用本机实测的 `ANGLE_SET = ANGLE_ACT + 15` 更新一次中性位置目标，
再两次写入并回读六轴 `ANGLE_SET=-1`；不发送回程。无响应或故障时不重新激励，直接
写六轴 `-1`。

两份第六轴窄行程 profile 现在都配置
`feedback_to_command_offset_units=15`，只用于实时控制启动时的内部 rate-limit seed
和正常退出的中性保持，不会叠加到 MANO target。该转换只允许在本机已验证的
`ANGLE_ACT=840..870` 内使用，超界不会静默钳位；其他 profile 和五个弯曲轴的偏置
均为 0。日志中的第六轴发送值 `S` 与反馈值 `A` 因此预期相差约 15，不应再要求两者
数值完全相等。

#### 1b. 肉眼确认第六轴的低速可视往返

只有五个弯曲轴均保持完全张开、六轴目标全为 `-1`、第六轴反馈位于
`840..870`、电流为 0 且状态到位时才能执行：

```bash
python3 examples/inspire_rh56_test.py visual-cycle \
  --port "$PORT" --joint thumb_rotate --include-thumb-rotate \
  --speed 40 --force-limit 80 --motion-timeout 10 \
  --confirm-movement --confirm-visual-cycle

python3 examples/inspire_rh56_test.py status --port "$PORT"
```

该动作固定执行两轮 `ANGLE_SET 855 <-> 885`，对应预计反馈约 `840 <-> 870`，
每个端点连续 3 个稳定样本后停留 `0.7 s`，最后用启动时冻结的
`initial_ANGLE_ACT+15` 返回原姿态，再双重写入六轴 `-1`。按官方名义角范围做线性
估算，单次侧摆约 2 度；在拇指弯曲完全张开时，模型估算指尖位移约 3.8 mm。应从
掌心斜侧或拇指侧面观察拇指根部的侧摆/对掌运动，而不是观察弯曲关节。

该动作硬编码 speed `40`、force `80 g`、单轴/总电流门 `400/600 mA`，并监控
其余五轴的状态与反馈漂移。任一端点超时、方向错误、故障、过温或过流后都不会继续
下一端点或自动回程，而是直接尝试写六轴 `-1`。出现 `VISUAL-CYCLE RECOVERY FAILED`
时立即切断 24 V。

#### 1c. 第六轴张开到闭合的额定全行程验收

前一项窄行程已经确认方向正常后，如需肉眼确认完整动作，使用专用命令；不要用通用
`sweep` 代替：

```bash
python3 examples/inspire_rh56_test.py thumb-full-cycle \
  --port "$PORT" --joint thumb_rotate --include-thumb-rotate \
  --speed 40 --force-limit 80 --motion-timeout 120 \
  --confirm-movement \
  --confirm-rated-thumb-cycle RH56_FULL_RANGE_1000_0_1000

python3 examples/inspire_rh56_test.py status --port "$PORT"
```

官方定义第六轴 `1000=张开`、`0=闭合`。该命令只执行一轮
`1000 -> 750 -> 500 -> 250 -> 0 -> 250 -> 500 -> 750 -> 1000`，即逻辑上的
`张开 -> 闭合 -> 张开`，最后停在张开端，并将六轴 `ANGLE_SET` 全部释放为 `-1`。
启动门禁要求五个弯曲轴 `ANGLE_ACT>=980`、第六轴处于已确认的 `840..870` 起始窗、
两次预检反馈稳定、所有目标均为 `-1`、无故障、温度低于 `50 °C`且静止电流正常。

动作固定为 speed `40`、force `80 g`；每个端点要连续 3 个稳定且电流回落到
单轴/总计 `100/200 mA` 以内的样本才算到位，运行中
监控方向、六轴状态/故障/温度、第六轴 `400 mA` 和六轴合计 `600 mA` 电流门，以及
其余五轴反馈漂移。异常后不会发送数字回程目标，只会尝试批量写六轴 `-1`。释放后
还会观察 0.75 秒的六轴反馈；只有最后连续样本同时满足 `ANGLE_ACT/POS_ACT` 稳定、
状态空闲、无故障、温度正常且电流回到 `100/200 mA` 静止门限后，才恢复原速度和
力阈值。若出现 `FULL-CYCLE MOTION STOP UNCONFIRMED`，立即切断 24 V。

#### 2. 旋转轴单轴窄行程，20 秒

只有点动通过后才运行。该 profile 只启用 `thumb_rotate=800..900`，speed `60`、
软件速率 `40 units/s`、force `80`，ACTIVE 单轴/总电流 `fault` 门都是 `400 mA`。
它属于内容受限的
commissioning profile，因此不需要 wide-range token，但必须限制在 `1..60 s`；这里
固定运行 20 秒：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --operator-roi "$OPERATOR_ROI" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_thumb_rotate_commissioning_900_800.json \
  --retargeter geometric \
  --enable-hardware --confirm-hardware-motion \
  --port "$PORT" --axes thumb_rotate \
  --no-hand-policy disable --duration 20 \
  --output-dir /tmp/inspire-mano-thumb-rotate-hardware-narrow \
  --overwrite-output
```

开始时保持真人右手稳定；连续 5 个新鲜、深度实测且目标稳定的 MANO 帧后才会进入
`ACTIVE`。只做缓慢、小幅对掌动作。结束后验收日志和状态：

```bash
python3 examples/analyze_inspire_mano_log.py \
  /tmp/inspire-mano-thumb-rotate-hardware-narrow/mano_retarget.jsonl
python3 examples/inspire_rh56_test.py status --port "$PORT"
```

必须确认 `thumb_rotate` 的 `T/S/A` 同向变化、无故障/过流/过温，且
`shutdown.verification_status=safe`、`stop_confirmed=true`、最终六轴均为 `-1`。

#### 3. 六轴联合窄行程，20 秒

只有旋转轴单轴阶段完整通过后，才先张开五个弯曲轴；`open` 不会改变旋转轴：

```bash
python3 examples/inspire_rh56_test.py open \
  --port "$PORT" --speed 80 --force-limit 80 --motion-timeout 20 \
  --confirm-movement
python3 examples/inspire_rh56_test.py status --port "$PORT"
```

状态必须满足六轴目标全为 `-1`、五个弯曲轴接近 `1000`、旋转轴仍在 `800..900`、
故障为 0、温度低于 `60 °C`。然后运行六轴窄行程；它不是 tokenless commissioning，
所以必须显式给出精确的 wide-range token：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --operator-roi "$OPERATOR_ROI" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_six_dof_commissioning_900_800.json \
  --retargeter geometric \
  --enable-hardware --confirm-hardware-motion \
  --confirm-wide-range RH56_WIDE_RANGE \
  --port "$PORT" --axes "$SIX_AXES" \
  --no-hand-policy disable --duration 20 \
  --output-dir /tmp/inspire-mano-six-hardware-narrow --overwrite-output
```

该档四个非拇指弯曲轴只走 `800..1000`，拇指弯曲只走 `850..1000`，旋转轴只走
`800..900`；speed `80`，各轴软件速率分别不超过 `80/60/40 units/s`，ACTIVE 单轴/
所选六轴总电流 `fault` 门为 `500/600 mA`。只做小幅缓慢动作。按与单轴阶段相同的标准
分析 `/tmp/inspire-mano-six-hardware-narrow/mano_retarget.jsonl`；六个轴必须逐一验证
方向、`T/S/A`、电流、温度、故障以及最终六轴 `-1`。

目前仍没有把 `thumb_rotate=0..1000` 用于 MANO 实时控制的 profile，也没有六轴统一
高速 profile。正式六轴 profile 不是“第六轴全范围/高速”：它只开放 `1000..900`，
并把该轴固定为 `SPEED_SET=80`、`40 units/s`。不要把旋转轴改成旧的 `300..700` 或
`0..1000`，不要提高 force、修改设备 `CURRENT_LIMIT` 或隐藏遥测 warning，也不要把
五指的 speed 200/300 强行套到第六轴。
扩大旋转范围必须另建逐档 profile 并逐档留存日志证据。

### 无手与故障时会发生什么

- 本 commissioning 记录中的 `thumb_rotate_...900_800` 和
  `six_dof_commissioning_900_800` 命令固定使用 `--no-hand-policy disable`。正式
  `six_dof_open1000_realtime` profile 则要求完整四枚确认：wide-range、six-DOF、
  `--confirm-current-monitor-only RH56_ACTIVE_CURRENT_MONITOR_ONLY` 和
  `--confirm-no-hand-open CALIBRATED_OPEN`；未进入 `ACTIVE` 时仍保持等待。
- `ACTIVE` 后，最后一帧真实深度证据超过 `0.75 s`、串口异常、执行器 ERROR/异常
  STATUS 或温度达到 `60 °C`，都会锁存故障并退出，不能自动重新解锁。旧
  commissioning profile 的 `active_current_policy=fault` 仍会在 host 电流门超限时
  退出；只有上面的正式 `six_dof_open1000_realtime` 精确内容使用 `monitor_only`。
  两种策略的启动前空闲电流都 fail closed，设备 `CURRENT_LIMIT` 都由固件执行。
- 正常退出和故障路径会写入并回读六轴 `ANGLE_SET=-1`，随后继续观察角度、位置、
  状态和电流，确认物理/电气反馈已经稳定后才报告安全停机。`-1` 本身仍不会切断
  24 V。
- 若看到 `STOP UNCONFIRMED`，软件无法确认目标释放或释放后的物理停机，必须立即
  切断 24 V；不要先重启程序或再次发送运动命令。

### 优化优先级

1. 先保证 D435 为 USB 3.x，并把真人手完整放在画面和有效深度范围内。
2. 用 operator ROI 排除机械手，坚持单只物理右手；不要放宽多手安全门。
3. 使用 `cuda:0` 并减少其他 GPU 负载。当前视觉约 20 FPS，提高 `control_hz` 的收益
   很小；如需继续优化，再评估检测裁剪、FP16 和窗口绘制异步化。
4. 在日志中分别检查 `NO HAND`、`MULTIPLE HANDS`、`HAND OUTSIDE OPERATOR ROI`、
   depth reason 和推理延迟，先修复真正的丢帧来源，再调滤波。
5. Inspire 速度最后再调：先通过单轴和六轴窄行程，再逐档提高软件速率与
   `SPEED_SET`。不要通过延长 watchdog、关闭深度门、提高 force、修改设备
   `CURRENT_LIMIT` 或隐藏电流 warning 来换取“稳定”。

## 1. 建立环境

```bash
cd /home/qiaoguanren/code/franka
examples/setup_inspire_mano_env.sh
```

环境安装在 `examples/inspire_mano_pipeline/.venv`，不会修改项目原有 `.venv`。
脚本会先检查四个 WiLoR 权重、dex 的 Inspire 配置/URDF，安装后再验证 Torch、
RealSense、Pinocchio 和 dex-retargeting 能在同一进程导入。资源不在上述默认路径时：

```bash
WILOR_ROOT=/path/to/wilor_mini \
DEX_ROOT=/path/to/dex-retargeting \
examples/setup_inspire_mano_env.sh
```

本机已验证组合为 Python 3.10.20、Torch 2.5.1+cu121、NumPy 2.2.6、
OpenCV 4.13.0、pyrealsense2 2.58.2、dex-retargeting 0.5.0 和 Pinocchio 4.1.0。

## 2. 离线测试

```bash
examples/test_inspire_mano_pipeline.sh
```

测试包括：

- MANO 根姿态消除和 dexsuite 腕部坐标系的刚体变换不变性；
- Inspire 六轴优化器构建、关节顺序和软限位映射；
- RH56 数值增大为张开方向、常规配置的五个弯曲轴张开值为 `900`、open1000 首次
  联调配置的食指张开值为 `1000`，禁用轴使用 `-1`；
- 模拟 tracking timeout 后锁存停机；
- 稀疏检测帧不会跨 timeout 累积并误解锁；
- 只有一个线程访问串口，最后一条写命令必须是六轴 `-1`；
- 临时 SPEED_SET/FORCE_SET 写入后必须完整六轴回读一致，错读会在 ACTIVE 前
  fail closed；
- 串口打开失败不会误报停机已确认，未选轴在运动/停机阶段始终保持 `-1`；
- 官方 dex 人手帧的 Inspire 六轴 golden 输出；
- 真机 CLI 必须显式选择 `cuda:N`、commissioning/量程、retargeter、轴和串口；
- open1000 单食指 profile、无手回到 `1000` 的模拟 stream 和六轴双重 `-1`
  `disable` 路径；
- 正式六轴 open1000 profile 的精确内容门、四枚确认 token、各轴独立硬件速度、
  第六轴反馈/命令偏置端点处理，以及释放输出后的物理/电气停机确认；
- 日志分析会区分期望目标、按开放轴屏蔽后的目标、worker 实际发送目标，汇总实际
  角度、电流、温度和故障反馈，比较 raw→filtered 六轴范围/抖动，并校验真机停机
  文件；
- 五个弯曲轴的 median/EMA 只作用于配置列出的轴、拇指掌面角的刚体/尺度不变性，
  `active_current_policy=fault` 的旧超限停机、`monitor_only` 的 warning 计数与继续
  运行、设备 CURRENT_LIMIT 回读门，以及启动前空闲电流、设备 ERROR/异常 STATUS、
  过温和通信故障的 fail-closed 路径；
- RH56 官方 V1.09 协议帧自检。

当前测试脚本会自动发现本目录下全部测试；本机最近一次为 `204/204 passed`，具体
数量仍以脚本当次汇总为准。随后还会执行 RH56 V1.09 协议帧自检（协议自检不计入
unittest 数量）。

本地 `assets/bo.jpg` 从手掌和拇指朝向判断是一张**没有镜像的物理右手**照片，
但当前 bundled detector 会把它误分为 `left/class0`。因此这张图片只能作为
“右手被 detector 误分后应安全拒绝”的回归样本，不能作为成功的右手 smoke：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source image \
  --image /home/qiaoguanren/下载/WiLoR_OL/assets/bo.jpg \
  --retargeter geometric \
  --device cuda:0 --headless --max-frames 5
```

WiLoR detector 的类别定义是 `left/class0`、`right/class1`；当前 pipeline 对未镜像
RealSense 中的物理右手只接受 `right/class1`。`bo.jpg` 的 `class0` 输出是该样本的
检测器误分类，不能据此反推照片是左手。上面的运行应记录 `handedness_mismatch`，
不会产生可用于控制的右手 MANO 目标。只有静态图片本身确实经过水平镜像时才加
`--mirror-input`，此时 pipeline 会相应交换期望 detector label；不要为了让某张误分
图片通过而滥用此选项。`--mirror-input` 不允许用于 RealSense，也不能代替真人右手
测试。对确实镜像的静态图片，pipeline 会在 canonicalize/retarget 前把 3D MANO
几何恢复到物理手约定；bbox、2D joints 和 mesh 投影仍留在镜像画面的坐标系中，
以保证预览叠加正确。

## 3. RealSense 实时预览

先确认相机被系统枚举：

```bash
examples/check_inspire_mano_devices.sh --camera-only --gpu
rs-enumerate-devices -s
CAMERA_SERIAL="替换为上条命令显示的 D435 序列号"
```

`--gpu` 会用 pipeline 虚拟环境检查 `torch.cuda.is_available()`；只想确认相机时可以
省略它。CUDA 检查失败时先确认 `nvidia-smi` 和是否有其他 GPU 作业，再按脚本提示
处理 `nvidia_uvm`，不要在有 CUDA 作业运行时卸载模块。

带窗口运行，按 `q` 或 `Esc` 退出：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --retargeter geometric --no-hand-policy open --device cuda:0
```

窗口同时显示 RealSense camera image、有效检测时的半透明 MANO surface（778 个
顶点、1538 个三角面）、21 点手骨架和六轴目标。第三行会直接显示 `TRACKING OK`、
`NO HAND`、`WRONG HAND / RIGHT REQUIRED`、`MULTIPLE HANDS`、`DEPTH TOO NEAR/FAR`
或 `DEPTH INVALID`，用于区分漏检、手别和深度门控。这里的
`--no-hand-policy open` 只让**预览目标**在无手或深度无效时显示
`CALIBRATED_OPEN`；预览模式不会打开串口。省略该参数时默认策略是 `disable`。
不要加 `--headless`，否则只写日志/截图而不会创建窗口；按 `q` 或 `Esc` 正常退出。

本机 848×480 的实测推理延迟与画面内容强相关：CUDA GPU 检测到有效右手时
p50 `49.12 ms`、p95 `53.59 ms`（约 `20 FPS`），无有效手约 `10.24 ms`；CPU
检测到有效右手时 p50 `998.31 ms`、p95 `1044.39 ms`（约 `1 FPS`），无有效手约
`119.15 ms`（约 `8 FPS`）。因此实时 mesh/控制必须优先显式使用
`--device cuda:0`；CPU 的“空画面 8 FPS”不能视为可实时控制。真机 CLI 会拒绝
`--device auto`、`--device cpu` 和未带编号的 `--device cuda`，必须显式给出
`--device cuda:N`；预览模式不受这个门槛影响。实际速度仍会随手数、检测框大小、
GPU 负载和是否保存输出变化。

无窗口运行 20 秒：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --retargeter geometric --device cuda:0 --headless --duration 20
```

省略 `--retargeter` 时也默认使用 `geometric`。只接受单只物理右手；在未镜像
RealSense 中，它对应 WiLoR 的 `right/class1` detector label。画面还会显示掌心
深度、推理 FPS 和醒目的 `PREVIEW ONLY`。每次运行的输出位于
`examples/inspire_mano_pipeline/output/<时间>/`：

- `latest.jpg`：最新叠加画面；
- `first_detection.jpg`：首次有效右手检测；
- `first_detection_mano.npz`：首次有效帧的 MANO pose、shape、778 顶点、原始/腕部
  坐标系 21 关节点；
- `run_metadata.json`：相机、模型、标定、环境、开放轴和 session ID；
- `shutdown_status.json`（真机模式）：最终六轴回读、目标释放、物理/电气停机确认、
  最后反馈样本、状态和退出码；
- `mano_retarget.jsonl`：每个相机帧的检测状态；有效帧另含 MANO
  pose/shape/21 点、qpos、六轴目标、实际发送值和硬件反馈。漏检帧也会记录，便于
  计算检测率和复盘 watchdog；同时记录 RGB/Depth 帧号、设备时间戳、主机收帧
  时间、align/copy 时间和从收帧到落日志的 pipeline age。

模型权重、MANO、Inspire URDF/config 的本次验证 SHA-256 与 dex-retargeting commit
记录在 `examples/inspire_mano_pipeline/asset_manifest.json`。

离线汇总检测率、推理与 pipeline age 的 p50/p95、RGB-D 时间戳/帧号同步、深度、
最长连续漏检、各轴范围/相邻帧跳变、目标提交接受率、是否进入过 ACTIVE，以及
实际发送目标的观测次数：

```bash
python3 examples/analyze_inspire_mano_log.py \
  examples/inspire_mano_pipeline/output/替换为本次时间目录/mano_retarget.jsonl
# 机器可读结果在命令末尾加 --json
```

若同目录存在 `shutdown_status.json`，分析器会自动检查：

- `stop_confirmed` 是否为 `true`；
- `physical_stop_verified` 是否为 `true`；
- `final_angle_targets` 是否六轴全为 `-1`；
- shutdown session 是否与帧日志一致。

只有报告中的 `shutdown.verification_status=safe` 才构成软件停机回读证据。JSONL
记录的是 worker 最近一次发送目标的采样，不等同于逐条串口写事务计数。

WiLoR 接收 RealSense/OpenCV 原始 `uint8 BGR`，不要再转一次 RGB。RealSense
深度用于掌心距离和有效性检查；手指姿态来自 WiLoR 的 MANO 模型。

### 五指 geometric 预览与三个分级 profile

下面三个 profile 都选择
`pinky,ring,middle,index,thumb_bend`，`thumb_rotate` 始终禁用。它们已经具备可加载
配置、离线数值测试和无串口预览入口；五轴窄行程已经进入过 ACTIVE 并发送五轴
目标，但第一次运行因旧硬停逻辑在合计 `505 mA` 时退出；后续五轴正常并发动作又
记录到 `611/632 mA`，任一轴均未达到单轴门且 ERROR 为 0。三份五轴 profile 目前仍
保持默认 `active_current_policy=fault`；`monitor_only` 只用于精确受限的正式六轴
profile。仍需按顺序比较三份预览和真机档位，不要把第一次 ACTIVE 表述成 RH56
五指 commissioning 已全部跑通：

| profile | 用途 | RH56 目标范围 | 速度 / 软件速率 | ACTIVE 电流 fault 门 |
| --- | --- | --- | --- | --- |
| `inspire_rh56bfx_right_five_finger_commissioning_open1000.json` | 五轴首次动作前的窄行程预览/commissioning 起点 | 四个非拇指弯曲轴 `800..1000`；拇指弯曲 `850..1000` | `SPEED_SET=120`；四指 `120 units/s`，拇指 `80 units/s` | 任一轴 `500 mA`；所选五轴绝对电流和 `600 mA` |
| `inspire_rh56bfx_right_five_finger_fullrange_realtime200.json` | 窄行程现场通过后的全量程中间档 | 五个弯曲轴 `0..1000` | `SPEED_SET=200`；四指 `200 units/s`，拇指 `160 units/s` | 任一轴 `600 mA`；所选五轴绝对电流和 `600 mA` |
| `inspire_rh56bfx_right_five_finger_fullrange_realtime.json` | 窄行程和 200 档都通过后的 300 档 | 五个弯曲轴 `0..1000` | `SPEED_SET=300`；四指 `300 units/s`，拇指 `200 units/s` | 任一轴 `600 mA`；所选五轴绝对电流和 `600 mA` |

表中的五轴门仍会在任一采样超限时锁存退出。它们与正式六轴 profile 的
`monitor_only` warning 不是同一策略；启动 preflight 仍要求空闲电流低于 profile 的
`preflight_max_idle_current_ma`，不合格时不会 armed。

三份配置都只对这五个弯曲轴使用 `median_window=3` 和 `EMA alpha=0.65`：先取最近
三个有效 MANO qpos 的逐轴中值，再计算
`filtered = 0.65 * median + 0.35 * previous`，最后应用各轴端点迟滞。原始和滤波后的
qpos/目标都会写进 JSONL，日志分析器可逐轴比较 raw→filtered 的范围与相邻跳变；
滤波不会启用或改写 `thumb_rotate`。

拇指弯曲不再使用“指根到指尖的链条直线度”。该量在真人拇指收向掌心时变化很小。
现在使用腕点、食指/小拇指 MCP 构成的掌面局部坐标系，将 MANO 拇指第一段
`joint 1 → joint 2` 投影到掌面的横向/纵向基底，并计算掌面内角；约 `35°` 作为张开
端，约 `47°` 作为闭合端，再映射到拇指弯曲 qpos `0..0.60`。该角只依赖归一化相对
向量和点积，因此不受手在相机前整体平移、旋转或等比例缩放影响；它不是 RH56
拇指旋转轴的控制量。

以下三条都是**带窗口、无串口**的精确预览命令。五指依次完全张开、逐指弯曲、
握拳、再完全张开；无有效右手时显式使用 `disable`，不合成“自动张开”目标：

```bash
FIVE_AXES=pinky,ring,middle,index,thumb_bend

examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_five_finger_commissioning_open1000.json \
  --retargeter geometric --axes "$FIVE_AXES" \
  --no-hand-policy disable --duration 30 \
  --output-dir /tmp/inspire-mano-five-preview-narrow --overwrite-output

examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_five_finger_fullrange_realtime200.json \
  --retargeter geometric --axes "$FIVE_AXES" \
  --no-hand-policy disable --duration 30 \
  --output-dir /tmp/inspire-mano-five-preview-200 --overwrite-output

examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_five_finger_fullrange_realtime.json \
  --retargeter geometric --axes "$FIVE_AXES" \
  --no-hand-policy disable --duration 30 \
  --output-dir /tmp/inspire-mano-five-preview-300 --overwrite-output
```

预览窗口下方的柱和数字是滤波后的目标；预览不连接 RH56，因此不会显示串口反馈。
真机窗口进入硬件模式后，顶端每个所选轴还会显示 `T/S/A`：`T` 是当前滤波后的
retarget 目标，`S` 是串口 worker 经过软件速率限制后最近实际写出的目标，`A` 是
最近读取的 `ANGLE_ACT`。三者更新源不同，短时不相等是正常的；判断“真机有没有
跟上”要看 `S` 和 `A`，不能只看 `T` 瞬间到达 `0`。

### Dex 实验预览

Dex vector optimizer 当前只用于比较预览，不得连接真机：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --retargeter dex --device cuda:0 --headless --duration 20
```

必须先用真人的张开/闭合及逐指动作确认 Dex 六轴方向、零位和范围；在此之前，CLI
会拒绝 Dex 硬件输出。

## 4. 首次真机联调

本节的单食指阶段已经通过，五指窄行程完成了第一次 ACTIVE 与安全保护验证，但
五指窄行程、200 档和 300 档仍待在其 `active_current_policy=fault` 门下逐级完成；
正式六轴的 monitor-only 策略不改变本节五轴流程。真机前必须满足：24 V 已上电、
手周围清空、
可以立即切断 24 V、同一序列号相机的 geometric 预览已由真人验证稳定，并且下面
的状态检查无故障：

```bash
CAMERA_SERIAL="替换为 rs-enumerate-devices -s 显示的 D435 序列号"
PORT=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
examples/check_inspire_mano_devices.sh
python3 examples/inspire_rh56_test.py status --port "$PORT"
```

若 `lsusb` 能看到 `1a86:7523`，但没有 `/dev/ttyUSB*`，先检查是否被 Ubuntu 的
BRLTTY 误识别。没有使用盲文设备时，可用 runtime mask 阻止它在重新插拔时被 udev
再次拉起，再重新加载驱动：

```bash
systemctl status brltty-udev.service --no-pager
sudo systemctl mask --runtime --now brltty-udev.service
sudo modprobe -r ch341
sudo modprobe ch341
```

随后重新插拔 RH56 USB，并再次查看 `/dev/serial/by-id/`。`--runtime` mask 位于
`/run`，重启后自动失效；若需在重启前恢复，可执行
`sudo systemctl unmask --runtime brltty-udev.service`。如果确实使用盲文设备，不要
屏蔽 BRLTTY。本机已用上述 runtime mask 恢复串口并完成状态读取与单食指窄范围
ACTIVE 运行；runtime mask 当前仍在。重新插拔或重启后的历史状态不能替代新的现场
检查，因此每次仍要等稳定的 `$PORT` 出现并重新通过 `status` 后再执行运动命令。

### 阶段 0：同一相机、无硬件预览

先保持 `PREVIEW ONLY`，由真人依次做“完全张开 → 只弯食指 → 握拳 → 完全张开”，
确认 geometric 的食指目标在张开时接近 `900`、弯曲时下降，且掌深持续有效：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --retargeter geometric --device cuda:0 --headless --duration 20
```

先分析这次预览日志；检测/深度或方向不正确时，不进入真机阶段。

### 阶段 1：geometric commissioning，仅食指

六个 `ANGLE_SET` 必须全为 `-1`。如果刚用 `inspire_rh56_test.py open` 将手张开，
食指实际角通常接近协议张开端 `1000`；此时首次只开放食指，并使用对应的 open1000
窄范围 commissioning 配置（`800..1000`，低速、低力）。只读 `status` 中食指实际
角也必须已经位于该范围，否则程序会拒绝 armed，不会自动把手拉进范围：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_commissioning_open1000.json \
  --retargeter geometric \
  --enable-hardware --confirm-hardware-motion \
  --port "$PORT" --axes index \
  --duration 20
```

如果这台手已经单独标定在 `700..900` 且食指实际角位于该区间，才把上面配置替换为
`inspire_rh56bfx_right_commissioning.json`。两份配置都是待现场验证的 commissioning
起点；文件名或通过软件安全门不等于已经完成真机方向/量程验证。

运行中先小幅弯曲食指，再把手移出画面以验证 tracking timeout。退出后立即分析该
run 目录：

```bash
RUN_DIR="examples/inspire_mano_pipeline/output/替换为本次时间目录"
python3 examples/analyze_inspire_mano_log.py "$RUN_DIR/mano_retarget.jsonl"
```

必须同时看到 `ever_active=True`、实际发送目标观测数大于 0，并且停机证据为
`status=safe`，再结合现场观察判断这一阶段通过。本机已经取得正常到时退出时的
ACTIVE、单食指运动和安全停机证据；随后又取得 ACTIVE 后移手约 `0.794 s` 触发
tracking timeout、进入 `fault_latched` 且最终六轴 `-1` 的真机证据，默认
`disable` commissioning 已通过。

只有在上述默认 `disable` 联调已经通过后，才可测试“跟踪丢失时回到标定张开位”。
真机模式还必须显式给出精确确认 token：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_commissioning_open1000.json \
  --retargeter geometric \
  --enable-hardware --confirm-hardware-motion \
  --port "$PORT" --axes index \
  --no-hand-policy open --confirm-no-hand-open CALIBRATED_OPEN \
  --duration 20
```

这个硬件 fallback 只允许 content-safe commissioning 配置和 `--axes index`，且
启动回读的食指必须在所选配置 `command_open` 的 ±30 units 内，否则程序拒绝运行；
上面的 open1000 配置对应 `command_open=1000`，其他五轴始终是 `-1`。它也不会在
启动后凭空驱动一只闭合的手：必须先由有效右手 MANO 进入 ACTIVE，随后连续无手/
深度无效至少 `0.30 s` 才向所选 `command_open` 限速回退，重新检测到手后还要经过
稳定帧门槛才恢复 MANO 控制。相机或推理线程停顿仍由 watchdog 停机，不能用
fallback 绕过。

`CALIBRATED_OPEN` 指所选标定文件里的张开目标：open1000 配置是 `1000`，旧的保守
配置是 `900`。即使协议定义 `1000` 为完全张开，open1000 配置也只用于已经通过
`open` 和只读状态确认的这台手，不表示 pipeline 真机跟随已经验证。

### 五指真机分级命令（当前进度）

本小节是继续 commissioning 的执行顺序。五指窄行程第一次运行已经证明五轴目标
能进入串口 worker；旧硬停逻辑在正常合计 `505 mA` 时过早退出，因此没有走完
窄行程。后续 `611/632 mA` 五轴记录说明现有总门容易在并发闭合时触发，但这些五轴
profile 当前仍是 `active_current_policy=fault`，仍需按现有门槛逐级复跑。五指宽
量程、`thumb_rotate` 和 Dex 真机输出都不属于当前已
通过阶段；不要从这次部分证据直接跳到五轴 200/300 档。每一档开始前都要重新确认
现场可立即切断 24 V，并把五个弯曲轴张开后检查六轴目标、故障、温度、电流和实际角：

```bash
FIVE_AXES=pinky,ring,middle,index,thumb_bend
python3 examples/inspire_rh56_test.py open \
  --port "$PORT" --speed 80 --force-limit 80 --motion-timeout 20 \
  --confirm-movement
python3 examples/inspire_rh56_test.py status --port "$PORT"
```

只有状态回读中六个 `ANGLE_SET` 全为 `-1`、五个弯曲轴已在所用配置的软范围内、
故障全 0、温度 `<60 °C` 且空闲电流合格时，才能执行下一条运动命令。虽然第一份
配置是窄行程，软件的 tokenless commissioning 门只覆盖单食指；五轴命令仍必须
显式给出 `--confirm-wide-range RH56_WIDE_RANGE`。

第一档仅为窄行程：四个非拇指弯曲轴最多从 `1000` 降到 `800`，拇指弯曲最多降到
`850`。先做小幅同步弯曲，不要直接握紧：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_five_finger_commissioning_open1000.json \
  --retargeter geometric \
  --enable-hardware --confirm-hardware-motion \
  --confirm-wide-range RH56_WIDE_RANGE \
  --port "$PORT" --axes "$FIVE_AXES" \
  --no-hand-policy disable --duration 20 \
  --output-dir /tmp/inspire-mano-five-hardware-narrow --overwrite-output
```

只有窄行程的方向、`T/S/A`、五轴电流、温度、故障和最终六轴 `-1` 都经日志与现场
复核通过后，才可另一次运行 200 档：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_five_finger_fullrange_realtime200.json \
  --retargeter geometric \
  --enable-hardware --confirm-hardware-motion \
  --confirm-wide-range RH56_WIDE_RANGE \
  --port "$PORT" --axes "$FIVE_AXES" \
  --no-hand-policy disable --duration 20 \
  --output-dir /tmp/inspire-mano-five-hardware-200 --overwrite-output
```

300 档只能在 200 档另一次运行也通过后使用：

```bash
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --calibration examples/inspire_mano_pipeline/inspire_rh56bfx_right_five_finger_fullrange_realtime.json \
  --retargeter geometric \
  --enable-hardware --confirm-hardware-motion \
  --confirm-wide-range RH56_WIDE_RANGE \
  --port "$PORT" --axes "$FIVE_AXES" \
  --no-hand-policy disable --duration 20 \
  --output-dir /tmp/inspire-mano-five-hardware-300 --overwrite-output
```

这三档真机命令都必须使用 `--no-hand-policy disable`。五轴配置不满足仅食指
fallback-open 的安全门，不能加 `--no-hand-policy open` 或
`--confirm-no-hand-open`。`disable` 表示无有效右手/深度时不再提交新 MANO 目标；
连续失去新目标超过 `0.75 s` 后 watchdog 锁存停机并执行最终六轴 `-1`，它不会把
五指自动拉回 `1000`。

ACTIVE 时每次反馈都会统计任一执行器的绝对电流和**所选五轴绝对电流之和**。
窄行程 `fault` 门为单轴 `500 mA`、五轴和 `600 mA`；200/300 档两项都是 `600 mA`。
任一采样超限仍会锁存退出。正式六轴 profile 的 monitor-only 例外不能套用到本节
五轴命令；不要给其他 profile 手工添加 monitor-only token 绕过内容门。

每一档结束后分别分析对应 `/tmp/.../mano_retarget.jsonl`，并要求同目录
`shutdown_status.json` 给出 session 匹配、`stop_confirmed=true` 和最终六轴 `-1`；
只有实际完成这些检查后，才能把该档记录为真机证据。

## 安全机制

- 真机启动时检查全部 `ANGLE_SET=-1`、故障为 0、温度 `<60 °C`、空闲状态和
  实际角度范围；`preflight_max_idle_current_ma` 仍是 fail-closed 门，空闲电流不合格
  时不会 armed；
- 连续 3 帧（commissioning 为 5 帧）有效、相邻间隔不超过 `0.25 s`，且 selected
  axes 的相邻目标跳变不超过配置阈值（commissioning 为 25 units）的单只右手 MANO
  才进入 ACTIVE；
- 所有六轴在一个 `<6h>` 协议帧中更新，并按轴限制每秒最大变化量；
- ACTIVE 反馈统计任一轴绝对电流和所选轴绝对电流之和；
  `active_current_policy=fault` 时两个阈值仍会锁存停机，只有精确六轴 profile 的
  `monitor_only` 才把它们作为 telemetry warning，超过后记录但不因该采样单独停机；
- RH56 设备的 `CURRENT_LIMIT` 寄存器及固件保护保持生效，worker 对非零 ERROR、
  ACTIVE 不允许的 STATUS、温度 `>=60 °C`、串口或反馈失败仍锁存故障并走最终六轴
  `-1` 停机路径。软件 warning 不替代设备硬保护；
- 默认 `disable` 策略下跟踪丢失不会提交目标；相机或推理停顿超过 `0.75 s` 后立即
  锁存停机，不能因下一帧恢复而自行重启；
- 硬件 `open` 策略只允许两种精确内容：已确认的 open1000 单食指 commissioning，
  或本文正式 `six_dof_open1000_realtime` 六轴 profile。六轴模式还必须提供
  `RH56_WIDE_RANGE`、`RH56_SIX_DOF_REALTIME`、`RH56_ACTIVE_CURRENT_MONITOR_ONLY`
  和 `CALIBRATED_OPEN` 四个精确 token；
  两种模式都要求启动角接近标定张开端并且曾进入 ACTIVE，才允许无手时回到所选
  `command_open`；
- 串口由一个 worker 独占，视觉线程只能覆盖“最新目标”；
- Linux 串口启用独占锁，第二个进程不能同时控制同一只手；
- Ctrl-C、正常退出、异常和 tracking timeout 都会把最后一条写命令设为六轴 `-1`
  并回读，再通过多次实际角度、位置、状态和电流反馈确认物理/电气停止；只有这两层
  都通过才报告 `stop_confirmed=true`、`physical_stop_verified=true`；
- 若出现 `STOP UNCONFIRMED`，软件无法保证手已停，必须立即切断 24 V。

`-1` 不是硬件急停，也不会切断电机电源；物理断电始终是最终急停手段。

## 标定文件

- `inspire_rh56bfx_right_commissioning_open1000.json`：手已用协议 `open` 张开后的
  食指窄范围 `800..1000`；基于三轮真人日志，食指 MANO 标定端为
  `q_open=0.05`、`q_closed=0.65`。在 80 units/s 下走完整 200 units 至少需 2.5 秒，
  量程复验时完全弯曲应稳定保持至少 3 秒；
- `inspire_rh56bfx_right_commissioning.json`：仅用于已确认食指在 `700..900` 的保守
  首次动作；
- `inspire_rh56bfx_right_five_finger_commissioning_open1000.json`：五个弯曲轴的
  open1000 窄行程起点，四个非拇指轴 `800..1000`、拇指弯曲 `850..1000`，
  speed `120`、force `80`、ACTIVE 单轴/五轴和 `fault` 门为 `500/600 mA`；
- `inspire_rh56bfx_right_five_finger_fullrange_realtime200.json`：仅在窄行程通过后
  使用的五轴 `0..1000` 中间档，speed `200`、force `80`，四指/拇指软件速率分别
  为 `200/160 units/s`，ACTIVE 单轴/总和 `fault` 门均为 `600 mA`；
- `inspire_rh56bfx_right_five_finger_fullrange_realtime.json`：仅在 200 档通过后使用
  的五轴 `0..1000` 300 档，speed `300`、force `80`，四指/拇指软件速率分别为
  `300/200 units/s`，ACTIVE 单轴/总和 `fault` 门均为 `600 mA`；
- `inspire_rh56bfx_right_six_dof_open1000_realtime.json`：第六轴额定全行程单轴验收
  后的正式六轴入口；五个弯曲轴为 `0..1000`，使用 `SPEED_SET=200`，四指/拇指
  弯曲软件速率为 `200/160 units/s`；`thumb_rotate` 仅为张开侧 `1000..900`，使用
  独立 `SPEED_SET=80` 和 `40 units/s`。六轴 filter 为 median3 + EMA 0.60，force
  `80`、`active_current_policy=monitor_only`，ACTIVE 单轴/六轴和 warning 为
  `400/600 mA`；只允许精确六轴内容和四枚确认 token；
- `inspire_rh56bfx_right.json`：五指弯曲轴保守软范围约 `100..900`；
- dexsuite qpos 顺序固定映射为
  `pinky, ring, middle, index, thumb_bend, thumb_rotate`；
- 真机协议方向为 qpos 越大、越闭合，RH56 指令越小。

不同 RH56BFX 真机和拇指机构可能存在偏差。正式使用前应逐轴记录完全张开、完全
闭合和无自碰范围，再调整各轴的 `q_open/q_closed`、`command_open/command_closed`
和速率。常规保守配置的五个弯曲轴 `command_open=900`；open1000 commissioning
把所选轴张开端设为 `1000`。三份五轴 profile 都使用 `median3 + EMA 0.65` 并禁用
`thumb_rotate`；正式六轴 profile 单独使用 median3 + EMA 0.60，并只开放第六轴
`1000..900`。五轴各档和六轴 20 秒联合运行仍分别按上述验收条件留存现场证据，
不能用第六轴单轴全行程通过替代联合 MANO 控制验收。
