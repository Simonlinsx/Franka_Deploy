# Franka 20 Hz Controller Contract

Status: accepted for the V225/V226 Inspire route on 2026-07-28.

## Decision

Use this command path:

```text
20 Hz policy joint delta
  -> held absolute joint target
  -> 100 Hz first-order target filter at 1 kHz
  -> 6 Hz critically damped joint-position interpolation at 1 kHz
  -> libfranka joint-position rate limiter as a final guard
  -> Franka internal joint-impedance controller
```

Do not use the older custom `0.5 rad/s`, `4 rad/s^2`, `120 rad/s^3` V94
shaper in the accepted route. Keep it only as an archived ablation. Do not send
the raw 20 Hz held target directly into libfranka's limiter either: the limiter
is a safety guard, not a motion generator for discontinuous steps.

The executable NumPy reference is
`shaper/franka_v225_interpolated_reference.py`. The batched simulator version
is `BatchedLibfrankaJointPositionInterpolator` in
`franka_command_shaper.py`.

## Policy-Level Mapping

The policy runs once every `0.05 s`. Its first seven outputs are normalized
incremental Franka commands:

```text
a_arm = clip(action[0:7], -1, 1)
raw = previous_policy_target + 0.045 * a_arm
held_target = 0.40 * raw + 0.60 * previous_policy_target
```

Thus the effective maximum target increment is `0.018 rad` per policy tick,
or an average target slope of `0.36 rad/s`. Clamp the result to:

```text
measured_q - 0.05 <= held_target <= measured_q + 0.05
safe_joint_lower <= held_target <= safe_joint_upper
```

The previous policy target is the previously accepted held target. It is not
replaced with measured `q` on every update.

## 1 kHz Motion Generator

Hold the latest policy target for 50 servo packets. At packet `k`, with
`dt=0.001 s`:

```text
lp_alpha = dt / (dt + 1 / (2*pi*100))
u_k = lp_alpha * held_target + (1-lp_alpha) * u_(k-1)

omega = 2*pi*6
a_star = omega^2 * (u_k-q_d) - 2*1.0*omega*dq_d
ddq_d = ddq_d_prev + clip((a_star-ddq_d_prev)/dt, +/-250.0) * dt
ddq_d = clip(ddq_d, +/-5.0)
dq_d = dq_d + ddq_d * dt
q_candidate = q_d + dq_d * dt
```

Send `q_candidate` as the next `franka::JointPositions` command. With the
deployment/simulator-shared derivative envelope, the simulated 18 mrad step
reaches about 4.28 mrad after 50 ms, 15.17 mrad after 100 ms, and 18.002 mrad
after 300 ms.

## libfranka Call

The exact-match real path applies the 100 Hz filter above, so disable a second
libfranka low-pass filter while retaining the official rate limiter:

```cpp
robot.control(
    motion_generator_callback,
    franka::ControllerMode::kJointImpedance,
    true,                         // final rate limiter enabled
    franka::kMaxCutoffFrequency   // built-in low-pass disabled
);
```

Set every optional argument explicitly. Defaults differ across libfranka
versions. Do not combine this path with the old custom shaper or another
velocity/acceleration/jerk filter.

Before opening the joint-position control session, the real deployment also
sets the collision behavior to the values from libfranka's official
`generate_joint_position_motion_external_control_loop` example:

```text
joint torque thresholds = [20, 20, 18, 18, 16, 14, 12] Nm
Cartesian thresholds    = [20, 20, 20, 25, 25, 25] N/Nm
```

The same arrays are used for lower/upper acceleration and nominal thresholds.
This setting does not reshape the V225 command trajectory; it makes collision
reflex behavior explicit instead of inheriting mutable Desk/session values.

## Initialization And Timing

1. Move to the common reset pose and let the arm settle.
2. At a new FCI control-session boundary, initialize the held target and
   interpolator from the last desired/commanded joint position reported by
   libfranka (`q_d`; for joint motion generation it represents the accepted
   desired command). Initialize desired velocity and acceleration from
   the reported desired state, or to zero only after the arm has settled.
3. Do not reinitialize interpolation state from measured `q` inside the 1 kHz
   callback.
4. Run policy inference outside the real-time callback. Atomically publish a
   new held target at 20 Hz; on a missed policy deadline, continue holding the
   previous target.
5. Log policy action, held target, generated target, `q_d`, measured `q`,
   desired derivatives, `control_command_success_rate`, and safety events with
   timestamps.

The Isaac Lab physics loop is 120 Hz. It advances virtual 1 kHz packets with an
`8, 8, 9` repeating schedule, exactly 50 packets over one 20 Hz policy period.
The real controller simply advances once per actual FCI callback period.

## Inspire Command

The remaining six policy outputs are absolute semantic motor targets in this
order:

```text
[thumb_yaw, thumb_flex, index, middle, ring, pinky]
```

Map `[-1,1]` to semantic closure fraction `[0,1]`, apply the 20 Hz target EMA
`alpha=0.737856`, and limit the virtual joint-target change to `0.30 rad` per
policy tick before converting to RH56 register order:

```text
[pinky, ring, middle, index, thumb_flex, thumb_yaw]
```

The six hand targets are updated at 20 Hz. Do not command the URDF follower
joints independently on hardware.

## Evidence

All rows use the same retained V163 epoch-1260 teacher on a 60 mm static sphere,
64 deterministic trials:

| Execution path | Formal success | Lift | Stable hold |
| --- | ---: | ---: | ---: |
| No shaper V212 | 57/64 (89.06%) | 64/64 | 61/64 |
| Old custom V213 | 0/64 | 6/64 | 2/64 |
| Raw 20 Hz step into limiter V224 | 0/64 | 0/64 | 0/64 |
| Interpolated + limiter V225 | 59/64 (92.19%) | 64/64 | 60/64 |
| Independent V226 limiter audit | 60/64 (93.75%) | 64/64 | 63/64 |

In the V226 audit, the maximum final-limiter correction over the complete
rollout was `1.1e-19 rad`, numerically zero. The accepted behavior therefore
comes from generating a continuous trajectory before the limiter, not from the
limiter repairing an invalid 20 Hz step.

The old V163 checkpoint is only a controller compatibility probe. It is not the
final Full-DR sphere/cube student. Zero-shot cube success was `5/64`: cube grasp
and lift were above 96%, but stable retention did not generalize. The V226
sphere/cube teacher and student must therefore be trained under this frozen
controller contract.

The step response explains the policy regression. For the same 18 mrad held
target, the old custom shaper moves only 0.59 mrad at 50 ms and 2.37 mrad at
100 ms, then overshoots to 19.84 mrad at 300 ms and returns to 15.43 mrad at
1 s. The revised interpolator reaches 4.28, 15.17, and 18.002 mrad at 50, 100,
and 300 ms. Its lower initial derivative must be reproduced in simulation
before training or deploying a matching checkpoint. The old shaper still adds
a different phase lag and braking response; it is not interchangeable with
this controller.
