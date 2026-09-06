# 真机测量 TODO（按阻塞程度排序）

这里的“完成”是指：数值、单位、坐标系、时间、采集方法、重复性和原始证据都已记录，并通过 `tools/validate_spec.py --require-complete` 所对应的人工审核。不要只把一个手填数字写进 manifest。

## P0：搭建碰撞与运动学场景前必须完成

### 1. 桌面平面与边界

- [x] 保持相机、机器人基座、桌面固定不动（2026-07-17 相机只读测量）。
- [x] 选择三块彼此分离、图像中无物体的黑色台面区域。
- [x] 用 `tools/measure_table_plane.py` 对三块 ROI 分别采集 30 帧。
- [x] 每次记录 `plane_abcd`、参考 XY 的 z、倾角、RMS/P95/max 和 inlier 数。
- [x] 比较三次法向量和高度的重复性，保留三份 YAML 并生成 `table_plane_summary.yaml`。
- [ ] 用台面四角/边缘的 base-frame 3D 点或物理测量确定桌面 polygon。
- [ ] 测量桌面厚度、圆角以及桌布相对硬桌面的厚度/可压缩性。
- [ ] 分开建模黑色工作台与机器人金属支撑台；不要假设它们共面。
- [ ] 将最终台面碰撞体与真实点云叠加审核。

要填的 manifest 字段：

```text
scene.tabletop.plane_in_robot_base              # 已填临时黑布表面
scene.tabletop.height_z_at_reference_xy_m       # 已填
scene.tabletop.reference_xy_m                   # 已填
scene.tabletop.corners_robot_base_m
scene.tabletop.length_m / width_m / thickness_m
scene.tabletop.collision_geometry
```

### 2. 确认 FR3 硬件版本

- [ ] 从机器人铭牌、Desk 或采购记录抄录型号、物理序列号、硬件 revision。
- [ ] 确认使用 `fr3` 还是 `fr3v2` description。
- [ ] 核对 7 个关节的位置/速度/力矩限制语义。
- [ ] 解释并统一 `franka_description` limits 与 hover 控制源码 gate limits 的差异。
- [ ] 记录最终 URDF/Xacro、所有 mesh 和参数文件的版本/哈希。

不要使用 Panda URDF 作为 FR3 的无声明替代。

### 3. 真实 EEF/工具/TCP/负载

- [ ] 确认法兰上当前安装的工具/手爪型号和序列号。
- [ ] 获取或制作工具 CAD/URDF、关节、visual/collision mesh。
- [ ] 测量 `T_flange_tool` 和控制使用的 `T_flange_tcp`。
- [ ] 记录质量、质心、惯量或供应商数据。
- [ ] 测量工具最外层扫掠碰撞体，包括电缆、软管和临时支架。
- [ ] 明确标定板是否已经从运行时 EEF 移除；如未移除，必须加入碰撞模型并重新评估安全工作区。
- [ ] 核对真机 `O_T_EE` 中的 EE 定义与仿真 link/TCP 完全一致。

标定文件中的 `T_ee_target` 只属于 ArUco 板，不完成以上测量。

### 4. 同步参考状态

- [ ] 场景固定后保存时间同步的 7 关节角、`O_T_EE`、color、aligned depth、内参、depth scale。
- [ ] 记录相机/机器人时间戳来源和同步方式。
- [ ] 保存一份无遮挡空场景和一份目标物体场景。
- [ ] 在同一关节状态渲染仿真图并保存对比报告。

要填：

```text
robot.reference_joint_configuration
timing_and_sensor_model.measured_timestamp_offset_color_depth_s
```

## P1：动力学和动态策略 sim2real 前必须完成

### 5. 任务物体族

- [ ] 不只测一个粉色圆柱；定义 prompt 对应的物体实例范围和歧义处理。
- [ ] 对实际会出现的圆柱测直径、高度、质量、质心/惯量近似。
- [ ] 测表面材质、与桌面/夹爪的静摩擦和动摩擦。
- [ ] 记录颜色、反光、纹理、制造误差和损伤范围。
- [ ] 保存 CAD 或参数化 primitive 定义，并给每个实例唯一 ID。

### 6. D435 深度噪声与缺失模型

- [ ] 在实际工作距离的多个距离/入射角采集静止平面深度。
- [ ] 分别统计偏差、标准差、空间相关性、时间相关性和 invalid rate。
- [ ] 单独统计物体轮廓附近的飞点、遮挡边和孔洞。
- [ ] 比较 emitter/laser power/preset 与最终运行配置一致时的结果。
- [ ] 不从当前 `9.994 mm` 单平面观测直接生成全局修正。
- [ ] 把传感器噪声与分割/跟踪误差分开建模。

### 7. 延迟、频率和控制系统识别

- [ ] 在真实发布路径测 camera capture → mask → PCD packet 的端到端延迟分布。
- [ ] 确认 object point cloud 的持续频率而非瞬时峰值达到任务要求（用户要求至少 20 Hz）。
- [ ] 测丢帧、长尾延迟和重新检测期间的 fail-closed 行为。
- [ ] 测 policy inference、命令传输、机器人响应延迟和动作保持时间。
- [ ] 记录控制器类型、增益、限速、滤波和安全限制。
- [ ] 在仿真中复现实际 observation/action timing，而不只复现平均频率。

## P2：视觉与环境随机化前建议完成

### 8. 相机外壳/支架和固定障碍物

- [ ] 获取 D435 外壳模型和 color optical → body frame 变换。
- [ ] 测相机支架相对 optical frame 的几何和碰撞体。
- [ ] 测试相机/支架是否进入机械臂扫掠空间。
- [ ] 建模急停、桌边、机器人支撑台以及不可移动的大型障碍物。

### 9. 光照和材质

- [ ] 在典型时间段记录相机曝光、白平衡和 RGB 直方图。
- [ ] 测或估计主要灯源位置、色温、亮度变化范围。
- [ ] 扫描黑色桌布的皱褶/高低起伏，而不是只建无限理想平面。
- [ ] 将背景杂物作为随机遮挡/干扰物分布，而不是固定复制一张图。

### 10. 工作空间定义

- [ ] 重新记录用户所述 0.40×0.30×0.40 m 捕获体积的中心 EEF 位姿。
- [ ] 明确 width/height/depth 分别对应 `robot_base` 哪个轴或相机哪个轴。
- [ ] 区分四种集合：感知视野、训练目标分布、机器人可达集、已验证无碰撞运动区。
- [ ] 不把 `hover.object_workspace_*` 当成桌面范围或全机器人可达空间。

## 验收产物

完成后应至少有：

```text
measurements/table_plane_roi_*.yaml
measurements/table_corners.yaml
measurements/robot_identity.yaml
measurements/eef_tool.yaml
measurements/reference_snapshot/...
measurements/object_family.yaml
measurements/depth_noise.yaml
measurements/latency.yaml
```

然后更新 `scene_manifest.json`：将相应字段从 `unknown` 改为 `measured`，加入来源、单位、时间和 SHA-256，并再次运行：

```bash
python tools/validate_spec.py
python tools/validate_spec.py --require-complete
```
