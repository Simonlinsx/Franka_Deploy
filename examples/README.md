# Examples

## Inspire RH56BFX-2R（USB-RS485）

[`inspire_rh56_test.py`](./inspire_rh56_test.py) 是 Inspire RH56BFX/DFX 灵巧手的 Linux 命令行测试程序，只依赖 Python 3 标准库。通信帧和寄存器地址来自官方《RH56 系列灵巧手用户手册 V1.09》。

Inspire 官网公开下载页目前主要提供 Windows 调试软件、驱动和手册，没有公开的 Linux/Python SDK 示例。因此，这个脚本是按官方通信协议编写的本地 example，不是厂商官方 SDK。

### 安全和准备

运行任何会动作的命令前：

- 使用 `24 V DC ±10%` 供电，电源的输出能力建议不低于 `2 A`。
- 清空手指和手掌周围，人员不要触碰运动部件，全程看护并保证可以立即切断 24 V。
- 关闭厂商调试软件或其他串口程序，同一时间只让一个程序控制手。
- 先执行下面的 `self-test` 和 `status`，再执行张开、点动或扫描。

官方角度指令的含义是：

- `ANGLE_SET=1000`：手指弯曲轴完全张开。
- `ANGLE_SET=0`：手指弯曲轴完全闭合。
- `ANGLE_SET=-1`：不执行新动作，保持当前机械位置。

脚本只通过角度寄存器控制，不直接写 `POS_SET`。拇指旋转轴默认不参与五指扫描，以减少自碰风险。

### 设置串口

在项目根目录运行：

```bash
cd /home/qiaoguanren/code/franka
PORT=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
```

`/dev/serial/by-id/...` 比 `/dev/ttyUSB0` 稳定，推荐固定使用。如果系统中只有一个串口，也可省略 `--port` 让脚本自动检测。手的工厂默认参数是 ID `1`、波特率 `115200`。

### 1. 离线协议自检

```bash
python3 examples/inspire_rh56_test.py self-test
```

该命令不访问串口、不会让手动作。它会将读帧、写帧和响应解析结果与官方 V1.09 手册中的字节示例比较。

### 2. 读取状态（只读）

```bash
python3 examples/inspire_rh56_test.py status --port "$PORT"
```

输出会包含 ID、波特率、角度目标、角度反馈、电缸位置、电流、温度、状态和故障位。重点检查：

- 六个轴的“故障”都为“正常”。
- `STATUS=2` 表示位置到位。`STATUS=255` 不在手册已定义的状态列表中；脚本只会在 `ANGLE_SET=-1` 的空闲场景接受它，并结合故障位、电流和位置判断。
- “系统电压寄存器原始值”只是未换算的寄存器数值，不要将它直接当成伏特值。

需要查看原始收发帧时，在命令后加 `--debug`。

### 3. 无动作写入检查

```bash
python3 examples/inspire_rh56_test.py write-check --port "$PORT"
```

该命令只在小拇指的 `ANGLE_SET` 已经是 `-1` 时，重写一次 `-1` 并验证 ACK 和回读，用于在不发起动作的情况下检查写通信。如果当前目标不是 `-1`，脚本会拒绝执行。

### 4. 六轴取消动作指令

需要让控制器停止追踪现有角度目标时，可写入六轴 `ANGLE_SET=-1`：

```bash
python3 examples/inspire_rh56_test.py disable --port "$PORT"
```

`disable` 打开串口后不会先等待完整状态快照，而是立即发送六轴 `-1`，随后再次
写入并回读确认两次。它不会把手移动到某个新位置，机械上会保持当前位置；它也
不是断电或硬件急停。若手仍持续用力、方向异常或软件无法回读确认，立即切断 24 V。

### 5. 先将手张开

下面这条独立协议命令此前已用于将 RH56BFX-2R 张开；它只验证 `open` 流程，不构成
RealSense/MANO 实时 pipeline 的真机验证：

```bash
python3 examples/inspire_rh56_test.py open --port "$PORT" \
  --speed 300 --force-limit 500 --motion-timeout 20 \
  --confirm-movement
```

脚本会从小拇指到拇指弯曲轴逐个张开，同时监视 `ANGLE_ACT`、`POS_ACT`、电流、温度、状态和故障。每个手指到位后会将它的 `ANGLE_SET` 设回 `-1`：手指保持张开的机械位置，但不再保留一条持续的 `1000` 动作指令。拇指旋转轴不会被改动。

建议在后续扫描前先执行一次这条 `open` 命令，再用 `status` 确认六个 `ANGLE_SET` 都是 `-1`。如果拇指旋转轴不是 `-1`，扫描会为安全起见拒绝启动。

### 6. 单指小幅点动

先用风险较低的小幅动作检查指定轴。例如，食指移动 `20` 个协议单位后返回原位：

```bash
python3 examples/inspire_rh56_test.py nudge --port "$PORT" \
  --joint index --delta 20 --speed 100 --force-limit 100 \
  --motion-timeout 20 --confirm-movement
```

`--delta` 允许范围为 `20..100`；`nudge` 的速度允许范围为 `1..300`，力阈值为 `1..300 g`。脚本会在动作后返回初始位置，并恢复原来的目标、速度和力阈值。

### 7. 单指全行程测试

先单独测一根手指。例如，无名指按 `0 -> 1000 -> 初始位置` 运动：

```bash
python3 examples/inspire_rh56_test.py sweep --port "$PORT" \
  --sweep-only --joint ring \
  --speed 100 --force-limit 100 --motion-timeout 20 \
  --joint-pause 2 --confirm-full-sweep
```

注意，这里的 `0` 是完全闭合，`1000` 是完全张开。这是真正的端到端动作，必须全程看护。`--confirm-full-sweep` 表示已确认周围清空且可以立即断电。

### 8. 五指轮流全行程测试

单指测试正常后，再执行五个弯曲轴的轮流测试：

```bash
python3 examples/inspire_rh56_test.py sweep --port "$PORT" \
  --speed 100 --force-limit 100 --motion-timeout 20 \
  --joint-pause 2 --confirm-full-sweep
```

默认顺序为小拇指、无名指、中指、食指、拇指弯曲。每个轴都会单独执行 `0 -> 1000 -> 该轴初始位置`，恢复后才测下一个轴。拇指旋转轴默认不动。

`sweep` 只在六个 `ANGLE_SET` 全为 `-1`、所有轴无故障、温度低于 `60 °C`且各轴状态可接受时启动。全行程测试的速度范围被限制为 `100..300`，力阈值为 `1..300 g`，单段超时不能小于 `15 s`。

### 拇指旋转轴

官方定义拇指旋转轴 `1000=张开`、`0=闭合`。先完成窄行程方向确认；五个弯曲轴
全部张开且可立即切断 24 V 后，使用专用的受监督额定全行程验收：

```bash
python3 examples/inspire_rh56_test.py thumb-full-cycle --port "$PORT" \
  --joint thumb_rotate --include-thumb-rotate \
  --speed 40 --force-limit 80 --motion-timeout 120 \
  --confirm-movement \
  --confirm-rated-thumb-cycle RH56_FULL_RANGE_1000_0_1000
```

动作以 `250` 单位分段，固定执行一次 `1000 -> 0 -> 1000` 并停在张开端。脚本要求五个弯曲轴反馈均
不小于 `980`，持续监控方向、故障、温度、其他轴漂移及 `400/600 mA` 单轴/总电流
门，结束时写六轴 `ANGLE_SET=-1` 并继续验证实际反馈已停稳。出现
`FULL-CYCLE MOTION STOP UNCONFIRMED` 时立即切断 24 V。

### 轴名称

| `--joint` 参数 | 机械轴 |
| --- | --- |
| `pinky` | 小拇指弯曲 |
| `ring` | 无名指弯曲 |
| `middle` | 中指弯曲 |
| `index` | 食指弯曲 |
| `thumb_bend` | 拇指弯曲 |
| `thumb_rotate` | 拇指旋转 |

### 异常处理

如果手未按预期运动、运动方向不对、持续用力或脚本报恢复失败，立即切断 24 V，不要反复重试全行程命令。

#### 没有出现串口设备

先检查：

```bash
lsusb
ls -l /dev/serial/by-id/ /dev/ttyUSB* 2>/dev/null
```

CH340 通常显示为 USB ID `1a86:7523`。Ubuntu 上 `brltty` 可能会抢占该设备，导致 `/dev/ttyUSB0` 没有出现。确认是这种情况后，可重新加载 CH341 驱动：

```bash
sudo systemctl mask --runtime --now brltty-udev.service
sudo modprobe -r ch341
sudo modprobe ch341
```

然后重新插拔 USB-RS485 适配器并再检查设备节点。runtime mask 重启后自动失效；
若需立即恢复，执行 `sudo systemctl unmask --runtime brltty-udev.service`。使用盲文
设备的机器不要屏蔽 BRLTTY。

#### 串口权限不足

```bash
groups
sudo usermod -aG dialout "$USER"
```

加入 `dialout` 后需要退出当前登录会话并重新登录。

#### 数值有变化，但手指几乎没动

- 同时看 `ANGLE_ACT`、`POS_ACT` 和电流，不要只看角度反馈。
- 如果 `POS_ACT` 明显变化且有短暂电流，但外部指节不动，停止测试并检查机械传动，必要时联系厂商。
- 如果动作时完全没有电流，检查 24 V 在负载下是否稳定，以及电源限流值是否足够。

#### 故障或保护停止

`STATUS=3/5/6/7` 分别代表力控到位、电流保护停止、堵转停止和故障停止。出现这些状态或任意非零 `ERROR` 时，不要继续扫描；先用 `status` 记录完整状态，然后检查机械干涉、供电和温度。

### 官方资料

- [Inspire RH56BFX 系列产品页](https://www.inspire-robots.com/dexterous%20hands/rh56bfx-series/)
- [RH56 系列灵巧手用户手册 V1.09（PDF）](https://www.inspire-robots.com/d/file/p/2023/11-17/%E5%9B%A0%E6%97%B6%E6%9C%BA%E5%99%A8%E4%BA%BA%E4%BB%BF%E4%BA%BA%E4%BA%94%E6%8C%87%E7%81%B5%E5%B7%A7%E6%89%8B--RH56%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8CV1.09cn%20.pdf)

## RealSense → MANO → Inspire 实时跟随

[`realsense_mano_inspire.py`](./realsense_mano_inspire.py) 已将 RealSense、
WiLoR-mini/MANO、dex-retargeting 的 Inspire 模型和上面的 RH56 串口驱动连接起来。
默认是不会打开串口的预览模式；真机模式带有连续有效帧 arming、每轴软限位与速率
限制、独占串口线程、故障/温度监视、tracking timeout 锁存和最终六轴 `-1` 回读。
当前已取得 open1000 单食指全量程真机证据、ACTIVE 后 tracking loss 锁存和最终
六轴 `-1` 安全停机证据；五指 full-range 预览也已覆盖五轴 `0..1000`。拇指旋转轴
已通过独立的 `1000 -> 0 -> 1000` 额定全行程验收，并已加入新的六轴实时 profile。
其中五个弯曲轴保留原有全量程，拇指旋转先使用保守的张开侧 `1000..900` 范围；
这不代表六轴实时联合跟随已经完成真机验收，首次运行仍须全程看护。

完整的环境安装、离线测试、RealSense 预览、逐步真机联调和标定方法见：

- [RealSense → MANO → Inspire pipeline README](./inspire_mano_pipeline/README.md)
- [当前推荐的五指使用 SOP](./inspire_mano_pipeline/README.md#当前推荐使用流程五指实时跟随)

最常用的预览命令：

```bash
cd /home/qiaoguanren/code/franka
CAMERA_SERIAL="替换为 rs-enumerate-devices -s 显示的 D435 序列号"
examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" \
  --retargeter geometric --no-hand-policy open --device cuda:0
```

该命令不带 `--headless`，会打开 camera image + 半透明 MANO mesh + 21 点骨架的实时
窗口；按 `q` 或 `Esc` 退出，且不会打开 RH56 串口。本机 848×480 实测中，有效右手
在 CUDA GPU 上约 `20 FPS`，在 CPU 上仅约 `1 FPS`；CPU 空画面虽约 `8 FPS`，不能
据此认为 CPU 足以实时控制。

### 六轴实时控制（正式命令）

六轴 profile 为
[`inspire_rh56bfx_right_six_dof_open1000_realtime.json`](./inspire_mano_pipeline/inspire_rh56bfx_right_six_dof_open1000_realtime.json)。
下面的命令固定使用本机 D435 序列号和 USB-RS485 串口，控制小拇指、无名指、中指、
食指、拇指弯曲和拇指旋转六个轴；它显式启用 CUDA，并在失去有效右手后回到标定的
六轴张开目标：

```bash
cd /home/qiaoguanren/code/franka
CAMERA_SERIAL=337322072188
PORT=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
SIX_AXES=pinky,ring,middle,index,thumb_bend,thumb_rotate
PROFILE=examples/inspire_mano_pipeline/inspire_rh56bfx_right_six_dof_open1000_realtime.json

python3 examples/inspire_rh56_test.py open \
  --port "$PORT" --include-thumb-rotate \
  --speed 40 --force-limit 80 --motion-timeout 20 \
  --confirm-movement

examples/run_inspire_mano_pipeline.sh \
  --source realsense --camera-serial "$CAMERA_SERIAL" --device cuda:0 \
  --operator-roi 0.35,0.10,0.98,0.98 \
  --calibration "$PROFILE" --retargeter geometric \
  --enable-hardware --confirm-hardware-motion --port "$PORT" \
  --axes "$SIX_AXES" \
  --confirm-wide-range RH56_WIDE_RANGE \
  --confirm-six-dof-motion RH56_SIX_DOF_REALTIME \
  --confirm-current-monitor-only RH56_ACTIVE_CURRENT_MONITOR_ONLY \
  --no-hand-policy open --confirm-no-hand-open CALIBRATED_OPEN
```

命令故意不带 `--duration`，因此会持续运行，直到按 `q`/`Esc`、按 `Ctrl-C` 或安全门
锁存停机。前置 `open --include-thumb-rotate` 只把第六轴从已验证的开侧范围单程低速
移到命令 `1000`，不会执行闭合循环；成功后六轴目标均回到 `-1`。启动前六个轴都
必须处于张开端附近，且 24 V 必须可立即切断。无手自动
张开只在 pipeline 已经用连续有效 MANO 帧进入 ACTIVE 后生效；相机或推理线程超时
仍会触发 watchdog 停机。

当前拇指旋转映射是由 MANO 几何关系估计的耦合信号，不是纯粹的拇指偏航角，所以
第六轴暂限为 `ANGLE_SET=1000..900`，硬件速度 `80`，软件目标最大变化率
`40 unit/s`；五个弯曲轴的硬件速度为 `200`，其中四指最大变化率为
`200 unit/s`、拇指弯曲为 `160 unit/s`。确认实时方向、耦合和自碰间隙前不要擅自
把第六轴扩大到 `0..1000`。标定、状态机、停机验证和日志验收细节见
[pipeline 六轴实时联合控制说明](./inspire_mano_pipeline/README.md#六轴实时联合控制open1000)。

该六轴 profile 显式设置 `safety.active_current_policy=monitor_only`，因此真机命令必须
同时给出精确确认串
`--confirm-current-monitor-only RH56_ACTIVE_CURRENT_MONITOR_ONLY`；这表示操作者明确
接受 ACTIVE 期间主机不会仅凭 `CURRENT` 遥测阈值锁存停机。进入 ACTIVE 前仍要求
每轴空闲电流不超过 `100 mA`；ACTIVE 后的 `400 mA/轴` 和六轴绝对值之和
`600 mA` 只记录 warning。真正的电流硬保护仍由手内 `CURRENT_LIMIT` 及其
`STATUS/ERROR` 反馈负责；程序不会修改设备已保存的 `CURRENT_LIMIT`。遥测告警和
板载保护都不能替代可立即切断的 24 V 物理急停。

### 五指分级预览 profile

五个弯曲轴现在有三份按风险递增的 profile；它们都禁用 `thumb_rotate`，并使用
`median_window=3`、`EMA alpha=0.65` 和逐轴端点迟滞。当前已有 full-range 预览和
窄行程第一次 ACTIVE 证据；那次运行由旧版主机总电流硬门在 `505 mA` 锁存停机。
这三份五指 profile 没有显式选择 `monitor_only`，因此使用 schema 默认的
`active_current_policy=fault`：ACTIVE 电流阈值仍会锁存停机。窄行程、200 和 300
档仍需逐级完成现场验收，进入 ACTIVE 前也都保留每轴 `300 mA` 的空闲电流硬门：

| profile | 用途 | 关键限制 |
| --- | --- | --- |
| `inspire_rh56bfx_right_five_finger_commissioning_open1000.json` | 首次五轴窄行程 | 四指 `800..1000`、拇指弯曲 `850..1000`；speed `120`；ACTIVE fault 门 `500/600 mA`（单轴/总和） |
| `inspire_rh56bfx_right_five_finger_fullrange_realtime200.json` | 窄行程通过后的全量程 200 档 | 五轴 `0..1000`；speed `200`；ACTIVE fault 门 `600/600 mA`（单轴/总和） |
| `inspire_rh56bfx_right_five_finger_fullrange_realtime.json` | 200 档通过后的全量程 300 档 | 五轴 `0..1000`；speed `300`；ACTIVE fault 门 `600/600 mA`（单轴/总和） |

三条精确预览命令如下；它们均显式使用 `no-hand-policy=disable`，没有有效右手时不
合成张开目标，也不会打开串口：

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

新的拇指弯曲信号使用 MANO 拇指第一段在掌面局部横向/纵向基底中的角度，约
`35°..47°` 映射到 qpos `0..0.60`；它比真人拇指收向掌心时几乎不变的“整条拇指
直线度”更适合功能性弯曲，并且对整体刚体变换和等比例缩放不敏感。

真机窗口中的 `T/S/A` 分别是滤波后的 retarget 目标、worker 限速后最近写出的目标
和最新 `ANGLE_ACT`；`T` 到达端点不代表机械手已经到位，应同时看 `S` 与 `A`。
进入 ACTIVE 前仍会用 profile 的空闲电流阈值拒绝异常启动；ACTIVE 期间任一轴
`CURRENT` 或所选五轴绝对值之和超过 profile 阈值时会按默认 `fault` 策略锁存停机；
设备自身的 `CURRENT_LIMIT` 触发状态、其他不安全 `STATUS` 或非零 `ERROR` 同样会
锁存停机。只有显式配置 `monitor_only` 并提供专用确认 token 的 profile 才把 ACTIVE
阈值降为 warning。五指真机必须保持 `--no-hand-policy disable`，丢失有效新目标超过
`0.75 s` 时由 watchdog 停机；这些保护均不能替代 24 V 物理急停。

完整 SOP、三档真机命令、`RH56_WIDE_RANGE` token、逐档前置状态检查和日志验收
条件见[详细 pipeline README](./inspire_mano_pipeline/README.md#当前推荐使用流程五指实时跟随)。
