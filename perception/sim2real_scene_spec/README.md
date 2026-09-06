# FR3 + D435 真机场景规格包

这个目录是把当前真机工作站对齐到仿真的**可追溯快照**。它不会声称“有这些文件就自动完成 sim2real”：目前相机模型、相机到机器人基座的位姿和黑布工作面的临时平面已经可用，但桌面实体边界、真实 EEF 工具、机器人硬件版本和动力学/时延仍需测量。

本包先在离线状态下从现有 repo、标定数据集和验证报告整理，随后只读访问 D435，对三个分离的空台面 ROI 做了重复平面测量。整个打包与测量过程没有导入、连接或命令机械臂。

## 一眼看懂当前状态

| 项目 | 状态 | 可直接用于仿真吗 |
|---|---|---|
| 世界坐标 | 约定 `world == robot_base` | 是；这是仿真约定，不是房间测量 |
| D435 彩色内参 | 848×480@30 Hz，完整 `K` | 是 |
| D435 深度比例 | `0.0010000000475 m/raw_unit` | 是 |
| 相机外参 | `T_base_camera_color_optical`，质量 `pass` | 是 |
| 相机光学坐标轴 | OpenCV/ROS optical：右、下、前 | 是 |
| FR3 本体 | 本机有 `franka_description 1.0.1` 候选 | 仅原型；需确认 `fr3`/`fr3v2` |
| EEF/工具/TCP/负载 | 未知 | 否，接触和碰撞训练的硬阻塞项 |
| 黑布工作面平面 | 3 个 ROI 实测；参考高度约 `0.0380 m` | 可用于视觉/非接触原型；不是安全碰撞面 |
| 桌面边界、厚度、支撑 | 未知 | 否，碰撞场景的硬阻塞项 |
| 物体尺寸、质量、摩擦 | 未知 | 否，动力学迁移的硬阻塞项 |
| 相机/控制时延与深度噪声 | 未测 | 否，动态策略迁移的硬阻塞项 |

权威机器可读入口是 [`scene_manifest.json`](scene_manifest.json)。每组数据都有：

- `evidence_class`: `measured / derived / configured / reported / assumed / unknown`；
- `units`；
- `source_refs`；
- 来源时间与 SHA-256（适用时）。

`configured` 的安全工作区不是实测桌面尺寸，`assumed` 的 `world == robot_base` 也不是物理房间坐标。

## 文件结构

```text
sim2real_scene_spec/
├── scene_manifest.json                 # 权威场景规格
├── schemas/scene_manifest.schema.json  # JSON Schema（结构校验）
├── tools/validate_spec.py              # 标准库语义/矩阵/校验和检查
├── tools/measure_table_plane.py        # 只读 D435 桌面平面测量，不连接机器人
├── tools/generate_fr3_candidate_urdf.sh# 从锁定 ROS 包生成裸臂候选 URDF
├── robot/asset_lock.json               # FR3 外部资产版本与哈希
├── measurements/table_plane_summary.yaml # 三个 ROI 的临时共识平面
├── measurements/table_plane_roi_*.yaml # 三份相机只读原始测量
├── measurements/README.md              # 实测结果放置与审核规则
├── evidence/current_d435_prompt.jpg    # 当前真实场景 RGB，仅定性参考
├── evidence/final_v2_validation_001.png# 标定验证 RGB，仅定性参考
├── source/calibration/                 # 标定、验证、原始20姿态数据和审计快照
└── source/runtime/d435_default.yaml     # 打包时的运行配置快照
```

没有复制 SAM/分割权重，也没有复制约 118 MB 的 `franka_description` 网格。机器人模型是明确锁版本的**外部依赖**，不是用旧 Panda 模型代替 FR3。

## 先验证数据包

在本目录运行：

```bash
python tools/validate_spec.py
```

它会验证：

- 所有打包来源文件的 SHA-256；
- `T_base_camera` 与逆矩阵互逆；
- 旋转矩阵属于 SO(3)，四元数为单位四元数；
- OpenCV optical 到 `+x right, +y up, -z forward` 相机坐标的派生姿态；
- `K`、FOV、光轴向量与分辨率一致；
- FR3 候选关节限制和 hover 边界的基本有效性；
- 所有 `source_refs` 可解析。

检查原始绝对路径仍与快照相同：

```bash
python tools/validate_spec.py --check-originals
```

在 CI 中要求完整场景时用：

```bash
python tools/validate_spec.py --require-complete
```

桌面实体边界和 EEF 未测完前，最后一个命令**应当失败**；这是防止把半成品场景误用于 sim2real 训练。

## 推荐的仿真搭建顺序

### 1. 固定世界坐标

将仿真 `world` 定义为 `robot_base`，因此：

```text
T_world_robot_base = I
```

这样不需要知道机械臂相对实验室地板的绝对位姿，策略、点云和真机 FCI `O_T_EE` 都能在同一坐标语义下工作。如果要建整个房间，再单独测 `T_room_robot_base`，不要改本包的标定外参。

### 2. 导入正确的 FR3，而不是 Panda 近似

本机检测到：

```text
ROS package: ros-humble-franka-description
version:     1.0.1-3jammy.20260422.111300
root:        /opt/ros/humble/share/franka_description
```

先从机器人铭牌/Desk/采购记录确认物理硬件对应 `fr3` 还是 `fr3v2`。确认后可生成不带臆测工具的裸臂 URDF：

```bash
bash tools/generate_fr3_candidate_urdf.sh fr3 /tmp/fr3_core.urdf
# 或者，只有在硬件版本确认后：
bash tools/generate_fr3_candidate_urdf.sh fr3v2 /tmp/fr3v2_core.urdf
```

生成文件仍通过 `package://franka_description/...` 引用外部网格。不要把 Xacro 默认 `franka_hand` 和默认 TCP 当成当前真机工具；当前 EEF 的类型、法兰变换、质量、惯量和碰撞体尚未测量。

### 3. 放置相机

权威外参为：

```text
p_base = T_base_camera_color_optical * p_camera_optical
```

其中 optical 坐标为：

```text
+x: 图像向右
+y: 图像向下
+z: 光轴向前
```

矩阵、逆矩阵、四元数、相机光轴在 base 下的方向都在 manifest 的 `camera.extrinsics`。

对于本地相机坐标为 `+x right, +y up, -z forward` 的 API（例如常见 USD/Blender/MuJoCo 相机约定），使用：

```text
T_base_camera_sim = T_base_camera_optical * diag(1, -1, -1, 1)
```

manifest 已给出该矩阵，以及 `xyzw` 和 `wxyz` 两种四元数顺序。仍要核对目标引擎的坐标手性、矩阵乘法方向和四元数顺序。

对于 look-at API：

```text
eye     = camera.extrinsics.translation_base_m
forward = optical_axes_in_base.forward_plus_z
up      = -optical_axes_in_base.down_plus_y
target  = eye + forward
```

### 4. 复制精确投影，而不是只填一个 FOV

使用 manifest 的完整 `K`：

```text
fx = 603.7852172851562 px
fy = 603.3232421875 px
cx = 435.23577880859375 px
cy = 246.64630126953125 px
width = 848, height = 480
```

主点相对图像中心有偏移。只使用对称 FOV 会丢失这个偏移；优先设置 off-axis projection。对称 FOV `70.156° × 43.385°` 仅用于不支持 `K` 的近似渲染。

标定记录的畸变系数为 0，模型名为 `inverse_brown_conrady`。深度先对齐到 color，然后按 color 内参反投影。真机管线把深度值作为 optical `z`，不是沿像素射线的欧氏距离；仿真深度若是 range，需要先转换成 z-depth。

`0.25–1.20 m` 是当前感知裁剪范围，不是 D435 的完整物理量程，也不等同于渲染器 near/far clip。

### 5. 黑布表面已测，但添加碰撞体前仍要测桌面实体

三个分离 ROI 的只读 D435 测量得到临时共识平面（`robot_base`）：

```text
-0.000750314*x - 0.002711797*y + 0.999996042*z - 0.037502718 = 0
```

在参考 `xy=[0.631033, 0.013883] m` 处，黑布表面 `z=0.038014 m`。三次局部参考高度范围为 `1.953 mm`，样本标准差 `0.977 mm`；最差局部拟合 p95 残差为 `4.410 mm`，局部法向与共识法向最多相差 `1.45°`。因此它可用于相机画面、点云和非接触运动学场景的初步对齐，但黑布褶皱和传感器/外参误差要求碰撞仿真另加保守余量。

权威数值与原始结果分别在 `measurements/table_plane_summary.yaml` 和三份 `table_plane_roi_*.yaml`。其中 observed coverage 只是相机采样覆盖，不是桌面物理边界、自由空间或机器人安全工作区。桌面四角、厚度、支撑结构、桌布可压缩性仍需卷尺/点云边缘测量。

本包提供可重复执行的只读 D435 测量工具。它不会导入或连接机械臂 API。场景或相机移动后，保持机器人静止、清空一块黑色台面，然后重新选择 ROI：

```bash
python tools/measure_table_plane.py \
  --select-roi \
  --frames 30 \
  --output measurements/table_plane_measurement.yaml \
  --confirm-camera-only
```

无 GUI 时显式给像素边界，坐标为 `x0 y0 x1 y1`：

```bash
python tools/measure_table_plane.py \
  --roi X0 Y0 X1 Y1 \
  --frames 30 \
  --output measurements/table_plane_measurement.yaml \
  --confirm-camera-only
```

不要照抄示例 ROI；每次应看当前画面选择只包含裸露黑色台面的区域。工具会：

1. 只打开标定序列号 `337322072188`；
2. 校验 848×480 color profile 和内参；
3. 对齐 depth 到 color；
4. 选择 ROI 内的暗色、有效深度点；
5. 用 `T_base_camera` 转到 `robot_base`；
6. 用近水平约束 RANSAC + 迭代 SVD 拟合；
7. 输出归一化 `a*x+b*y+c*z+d=0`、参考 XY 的 z、高度倾角、RMS/P95/max、点数、覆盖范围和测量时间。

先做纯离线检查，不访问硬件：

```bash
python tools/measure_table_plane.py --self-test
```

每次仍应在至少三个彼此分离的台面 ROI 重复测量。拟合点的 XY bounds 只是“看到了哪里”，不是物理桌面边界；桌面四角、厚度和支撑结构仍需卷尺/点云角点测量。

### 6. 建模真实 EEF 和任务物体

必须补齐：

- `T_flange_tool` 和 `T_flange_tcp`；
- 工具/手爪 CAD、开合关节、碰撞体；
- 工具质量、质心、惯量；
- 相机标定板当前是否已拆除；
- 圆柱实际直径/高度范围、质量、质心、惯量、表面摩擦。

标定文件里的 `T_ee_target` 只描述标定期间临时固定的 ArUco 板，绝不能当成 TCP 或工具变换。

### 7. 做 render-to-real / depth-to-real 验证

在场景固定后，同步保存一组：

- 7 个关节角；
- EEF 位姿；
- 原始 color + aligned depth + 相机内参；
- 仿真同姿态渲染。

至少比较：

- 机器人轮廓和桌面边界的像素重合；
- 已知 3D 点投影误差；
- 桌面深度残差；
- 物体中心在 `robot_base` 下的误差；
- 遮挡顺序与 EEF 碰撞距离。

现有手眼训练残差 p95 为约 `2.65 mm / 0.483°`，独立闭环检查约 `0.905 mm / 0.408°`；这些是质量证据，不是完整 6-DoF 协方差。另有单平面深度/PnP 差异 `9.994 mm`，不能据此直接对全部深度加固定修正。

## Sim2real 训练还需要什么

如果策略输入主要是 object point cloud，而不是 RGB，优先级通常是：

1. 坐标系、机器人运动学、真实 EEF、桌面碰撞；
2. D435 的遮挡、边缘飞点、缺失深度、采样/体素化；
3. 感知和控制时延、控制频率、动作保持；
4. 物体几何/质量/摩擦的任务分布；
5. 视觉材质、曝光、灯光和背景随机化。

`pink cylinder` 是语义 prompt，不定义一个物理物体。训练时可以随机化一族粉色圆柱，但上线前仍要测量真实任务物体的参数范围。

不要把标定残差、桌面拟合残差或一张照片直接当作 domain-randomization 分布。先测量重复性与系统误差，再选择有物理依据的随机化范围。

## 已知风险

- `franka_description` 候选关节位置限制比当前 hover 控制源中的 gate 常量更保守；两者语义尚未核对。仿真先使用更保守的 description limits，并在真机版本确认后统一。
- `fr3` 与 `fr3v2` 的运动学在本机包中相同，但惯量/视觉资产并不完全相同，不能仅凭运动学一致就任选。
- 当前 RGB 参考图没有同步深度和关节角，只能用于理解布局/外观，不能做严格像素对齐。
- 相机 optical pose 已知，但 D435 外壳与支架 CAD、相机 body pose 未测；如它们参与碰撞，需要额外建模。

剩余测量项及验收建议见 [`TODO_MEASUREMENTS.md`](TODO_MEASUREMENTS.md)。
