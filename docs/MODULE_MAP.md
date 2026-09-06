# Module map

Use this page to decide where a new file belongs.

| Question answered by the code | Owning module | Examples |
| --- | --- | --- |
| Which checkpoint/config is admitted and when does policy inference run? | `sim2real` | checkpoint loading, task launcher, observation/action scheduling, deployment CLI |
| How are checkpoint inference, policy evidence, and rate modes represented? | `sim2real.policy` | rolling NumPy policy, exact I/O recorder, 20/30 Hz contracts |
| What are the stable observation/action data schemas? | `sim2real.contracts` | general observation types, V94 contract, transactional action mapping |
| How are camera frames converted into policy RGB-D, points, and previews? | `sim2real.observation` | task camera profiles, ROI, projection, history, visualization, capture |
| How is a camera/device/profile explicitly calibrated or commissioned? | `sim2real.commissioning` | D435 timing, RH56 force/dynamics identification, supervised mapping probes |
| Which grasp pose or RH56 hand configuration should be used? | `dexgrasp` | AnyDex candidates, installed-hand geometry, grasp/lift workflow |
| Where should the arm move and when should interception/closure occur? | `motion_planning` | FK/IK, trajectory generation, moving-object fit, intercept and closure state machine |
| How is an approved command delivered safely to FR3/RH56? | `robot_control` | pylibfranka session, serial transport, watchdog, stop verification, action ledger |
| What is the auditable NumPy reference for Franka mapping/shaping? | `robot_control.reference` | V225 interpolator and historical V94 shaper regressions |
| What object is visible and what 3D points describe it? | `perception` (`dynamic_pcd`) | D435, grounding, SAM2, mask tracking, depth filtering, point-cloud publication |

## Dependency direction

```text
                            sim2real
                   task/deployment orchestration
                  /         |          |         \
                 v          v          v          v
       motion_planning   dexgrasp   perception   robot_control
          trajectories    grasps   (`dynamic_pcd`) hardware I/O
```

- `sim2real` may compose every lower module.
- `motion_planning` is pure computation and must not open hardware or cameras.
- perception must not import deployment or robot-control policy.
- `robot_control` accepts validated targets; it does not choose a task, grasp,
  or model.
- diagnostics may inspect all layers but production code must not import
  diagnostics.

## Canonical paths

The flat implementation aliases have been retired after maintained callers,
tests, shell wrappers, and documentation migrated. The stable public commands
are `sim2real.deployment`, `sim2real.tasks`, `sim2real.perception`, and
`sim2real.validate`.

Runtime scheduling and composition live in `sim2real.runtime`; camera-to-policy
construction lives in `sim2real.observation`.

Offline audit, replay, comparison, and export implementations live in
`sim2real.diagnostics`; calibration and physical probes live in
`sim2real.commissioning`.

Importing any owner package must remain hardware-inert; individual
write-capable commands retain their explicit confirmation gates.

FR3 persistent-session/backend code and RH56 transport/actuator/watchdog code
live under `robot_control.franka` and `robot_control.rh56`.

`sim2real.policy` and `sim2real.contracts` own inference support, I/O evidence,
rate modes, and stable observation/action schemas.

## Non-source directories

`data/checkpoints/`, `data/perception_corpus/`, and `data/runs/` contain active
models and runtime evidence. The internal `dexgrasp/runs` compatibility link
is retained for external operator tooling; the old root aliases are retired.
`data/` also contains curated fixtures and demos, `third_party/` contains
independent external checkouts, and `data/archives/` contains inactive
historical material. None of these are modules of the maintained integration
stack.
