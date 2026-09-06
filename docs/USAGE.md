# Low-level FR3 teleoperation and Cartesian-control examples

This is a low-level operator notebook under the `robot_control` domain. It is
not the entry point for policy deployment, AnyDex grasping, motion-planning
demos, or object perception. See `README.md` and `docs/MODULE_MAP.md` for the
maintained module map and task commands.

## GELLO -> FR3 bare-flange teleop flow

# Safety/prep:
# - Desk: unlock joints, recover errors if needed, activate FCI.
# - Desk end-effector configuration: no Franka Hand; payload matches the real hardware.
# - Keep GELLO in the calibrated neutral pose before starting the arm controller.
# - Do not run another FCI script while the ROS2 arm controller is active.
# - Do not start franka_gripper_manager while the physical Hand is removed.


# Terminal 1: GELLO publisher
cd /home/qiaoguanren/code/gello_software/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch franka_gello_state_publisher main.launch.py config_file:=franka_gello_single.yaml


# Terminal 2: verify GELLO before connecting to FR3
cd /home/qiaoguanren/code/gello_software/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 topic echo --once /gello/joint_states


# Terminal 3: move FR3 near the same neutral q before starting teleop
cd /home/qiaoguanren/code/franka
conda deactivate
source .venv/bin/activate

python examples/move_to_q.py --ip 172.16.0.2 \
  --preset gello-neutral \
  --max-delta 1.2 \
  --max-velocity 0.05 \
  --min-duration 25 \
  --execute \
  --enforce-realtime


# Terminal 4: FR3 bare-flange arm controller
cd /home/qiaoguanren/code/gello_software/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  robot_config_file:=example_fr3_config.yaml


# Terminal 5: verify the arm controller
cd /home/qiaoguanren/code/gello_software/ros2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 control list_controllers

# Expected:
# - Moving GELLO joints changes /gello/joint_states and the FR3 follows via joint_impedance_controller.
# - joint_impedance_controller is active.
# - No /fr3_gripper node or action is started.


# FR3 EEF Cartesian motion
cd /home/qiaoguanren/code/franka
conda deactivate
source .venv/bin/activate

# Dry-run: move end-effector +1 cm in base-frame z, preserving orientation.
python examples/move_eef.py --ip 172.16.0.2 --relative 0 0 0.01

# Execute after Desk -> Unlock joints -> Execution -> Activate FCI.
python examples/move_eef.py --ip 172.16.0.2 \
  --relative 0 0 0.01 \
  --execute \
  --enforce-realtime

# Absolute target xyz in the robot base frame, preserving current orientation.
python examples/move_eef.py --ip 172.16.0.2 \
  --target 0.30 0.00 0.50 \
  --max-distance 0.08 \
  --execute \
  --enforce-realtime

# Diagnostic: rotate EEF orientation with one slow trajectory, preserving xyz.
# Use this before debugging policy rotation if Cartesian orientation control
# triggers a discontinuity reflex.
python examples/move_eef_rotation.py --ip 172.16.0.2 \
  --relative-rotvec 0 0 0.02 \
  --frame base \
  --max-angle 0.05 \
  --max-angular-velocity 0.01 \
  --min-duration 5

# Execute after Desk -> recover if needed -> Unlock joints -> Execution -> Activate FCI.
python examples/move_eef_rotation.py --ip 172.16.0.2 \
  --relative-rotvec 0 0 0.02 \
  --frame base \
  --max-angle 0.05 \
  --max-angular-velocity 0.01 \
  --min-duration 5 \
  --startup-hold 0.50 \
  --execute \
  --enforce-realtime



# FR3 EEF policy-env inference loop
cd /home/qiaoguanren/code/franka
conda deactivate
source .venv/bin/activate

# Dry-run a 10 Hz handcrafted policy. This only reads the robot once and simulates targets.
python examples/run_eef_policy_env.py --ip 172.16.0.2 \
  --policy up \
  --policy-hz 10 \
  --duration 3 \
  --max-velocity 0.02

# Execute: policy runs at 10 Hz, Cartesian velocity commands stream at FCI rate.
python examples/run_eef_policy_env.py --ip 172.16.0.2 \
  --policy up \
  --policy-hz 10 \
  --duration 3 \
  --max-velocity 0.02 \
  --script-velocity 0.01 \
  --max-radius 0.05 \
  --execute \
  --enforce-realtime

# Policy action remains 7D: [dx, dy, dz, drx, dry, drz, gripper].
# dx/dy/dz are EEF position deltas in meters. drx/dry/drz are rotation-vector
# deltas in radians. With the current bare-flange setup, the final gripper value
# is ignored; do not pass --use-gripper.
#
# policy-hz is the policy inference frequency. Policy deltas are converted to
# velocity targets, then --max-acceleration and --max-angular-acceleration ramp
# those targets smoothly while Cartesian velocity commands stream to FCI at 1 kHz.

# Rotate around base-frame z with a scripted angular velocity.
python examples/run_eef_policy_env.py --ip 172.16.0.2 \
  --policy rz-plus \
  --policy-hz 10 \
  --duration 2 \
  --max-angular-velocity 0.10 \
  --max-angular-acceleration 0.05 \
  --script-angular-velocity 0.05 \
  --max-rotation-radius 0.30

# If executing Cartesian velocity control, keep the startup hold unless you have
# a reason to remove it. It streams zero velocity briefly before policy actions.
python examples/run_eef_policy_env.py --ip 172.16.0.2 \
  --policy rz-plus \
  --policy-hz 10 \
  --duration 2 \
  --max-angular-velocity 0.05 \
  --max-angular-acceleration 0.02 \
  --script-angular-velocity 0.02 \
  --max-rotation-radius 0.10 \
  --startup-hold 0.50 \
  --settle-time 1.5 \
  --execute \
  --enforce-realtime

# Other handcrafted policies:
# hold, up, down, x-plus, x-minus, y-plus, y-minus, sine-z, circle-xy,
# rx-plus, rx-minus, ry-plus, ry-minus, rz-plus, rz-minus.

# FR3 EEF policy-env with Cartesian impedance torque control
cd /home/qiaoguanren/code/franka
conda deactivate
source .venv/bin/activate

# Dry-run only: prints conservative stiffness/damping settings, no torque control.
python examples/run_eef_impedance_policy_env.py --ip 172.16.0.2 \
  --policy hold \
  --duration 2 \
  --trans-stiffness 50 50 50 \
  --rot-stiffness 3 3 3 \
  --damping-ratio 1.0

# First real torque-control test: hold current EEF pose with low stiffness.
# Execute after Desk -> recover if needed -> Unlock joints -> Execution -> Activate FCI.
python examples/run_eef_impedance_policy_env.py --ip 172.16.0.2 \
  --policy hold \
  --duration 2 \
  --startup-hold 0.5 \
  --trans-stiffness 50 50 50 \
  --rot-stiffness 3 3 3 \
  --damping-ratio 1.0 \
  --target-filter-tau 0.20 \
  --settle-time 1.0 \
  --max-force 15 \
  --max-task-torque 3 \
  --max-delta-tau 0.5 \
  --execute \
  --enforce-realtime

# Small impedance action test: 10 Hz policy updates an impedance target.
python examples/run_eef_impedance_policy_env.py --ip 172.16.0.2 \
  --policy up \
  --policy-hz 10 \
  --duration 2 \
  --max-velocity 0.01 \
  --script-velocity 0.003 \
  --max-radius 0.03 \
  --startup-hold 0.5 \
  --trans-stiffness 50 50 50 \
  --rot-stiffness 3 3 3 \
  --damping-ratio 1.0 \
  --target-filter-tau 0.20 \
  --settle-time 1.0 \
  --max-force 15 \
  --max-task-torque 3 \
  --max-delta-tau 0.5 \
  --execute \
  --enforce-realtime
