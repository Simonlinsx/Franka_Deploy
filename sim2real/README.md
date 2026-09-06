# FR3 + Inspire RH56 sim-to-real runtime

`sim2real` is the orchestration layer for checkpoint admission, policy
inference, task composition, and supervised runtime scheduling. Pure
kinematics/planning lives in `motion_planning`, hardware transports live behind
`robot_control`, grasp semantics live in `dexgrasp`, and mask/point-cloud
perception lives in `perception`.

The stable user entry points are `sim2real.tasks`, `sim2real.deployment`,
`sim2real.perception`, and `sim2real.validate`. Implementation modules are
grouped below `tasks/`, `deployment/`, `observation/`, `runtime/`,
`diagnostics/`, and `commissioning/`.

## Runtime contract

- Policy and controller frequency must match the checkpoint metadata.
- The commissioned task profile owns `q_home`, camera calibration, action
  mapping, joint/force limits, and the maximum supervised step count.
- The policy RGB-D resolution is `424x240` for the maintained tabletop and
  thrown tasks.
- `guarded_v2` is the maintained object-mask/point-cloud path.
- A real run requires `--execute --yes-i-am-supervising`; keep the emergency
  stop ready throughout reset and execution.
- Missing, stale, ambiguous, or out-of-envelope observations remain
  fail-closed unless a task-specific committed capture transaction explicitly
  owns the continuation.

## Canonical commands

Run commands from the workspace root:

```bash
cd /home/qiaoguanren/code/franka
```

The complete categorized command list is in
[`../docs/COMMANDS.md`](../docs/COMMANDS.md).

### Task configuration and perception

```bash
# Show the accepted task/mode combinations.
.venv/bin/python -m sim2real.tasks --help

# Materialize or inspect a task configuration without hardware writes.
./perception/scripts/run_tabletop_task.sh config
./perception/scripts/run_thrown_object_task.sh config

# Camera/GPU-only perception checks.
./perception/scripts/run_tabletop_task.sh perception --help
./perception/scripts/run_thrown_object_task.sh perception --help
```

### Adaptive tabletop demos

List scenarios:

```bash
./perception/scripts/run_tabletop_motion_demo.sh --help
```

Representative supervised run:

```bash
./perception/scripts/run_tabletop_motion_demo.sh \
  sphere-posy-board-collision \
  --live-visualization \
  --execute \
  --yes-i-am-supervising
```

The supported names distinguish object shape, direction, and speed; examples
include `cylinder-posy-low`, `sphere-negy-high`, and
`sphere-posy-board-collision`. The online planner remains responsible for the
live intercept; the scenario is not a fixed-time grasp replay.

### Thrown-object checkpoints

The wrappers verify checkpoint hashes and delegate to `sim2real.tasks`:

```bash
# V57 compatibility bundle.
./perception/scripts/run_thrown_checkpoint_test.sh

# V60 candidate bundle.
./perception/scripts/run_thrown_v60_checkpoint_test.sh

# One explicitly selected V61 expert; no router.
./perception/scripts/run_thrown_v61_sixexpert_test.sh
```

Example V61 read-only admission and 40-tick supervised test:

```bash
THROWN_V61_EXPERT=base-forward \
THROWN_V61_OBJECT_TEXT="small patterned beanbag toy" \
./perception/scripts/run_thrown_v61_sixexpert_test.sh runtime-admit

THROWN_V61_EXPERT=base-forward \
THROWN_V61_OBJECT_TEXT="small patterned beanbag toy" \
./perception/scripts/run_thrown_v61_sixexpert_test.sh test-40
```

Accepted expert names are `base-forward`, `base-negative-y`,
`base-positive-y`, `high-forward`, `high-negative-y`, and `high-positive-y`.
Use the wrapper's help text for reset, perception, shadow, and admission modes.

## Direct deployment API

Use this only when a task wrapper does not already own the configuration:

```bash
.venv/bin/python -m sim2real.deployment \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --profile /absolute/path/to/commissioned-profile.json \
  --pcd-config /absolute/path/to/task-camera.yaml \
  --policy-rate-hz 20 \
  --policy-rgbd-resolution 424x240 \
  --object-mask-mode guarded_v2 \
  --steps 20 \
  --run-id descriptive-safe-run-id \
  --object-text "object description" \
  --record-video dexgrasp/runs/descriptive-safe-run-id.mp4 \
  --record-policy-io \
  --live-visualization \
  --execute \
  --yes-i-am-supervising
```

Prefer a wrapper because it binds the checkpoint, task, reset pose, and
commissioning profile together. Do not copy a profile from another task only
to bypass an admission error.

## Observation and action conventions

The policy input contains RGB-D-derived object points plus robot state/history
according to the verified bundle contract. Object point coordinates are
explicitly labeled as `policy_palm` or `robot_base_via_identity_T_base_palm`.

The 13-D action is:

```text
action[0:7]   Franka joint target semantics owned by the selected contract
action[7:13]  [thumb_rotation, thumb_bending, index, middle, ring, little]
```

RH56 hardware registers use
`[little, ring, middle, index, thumb_bending, thumb_rotation]`. Reordering is
centralized in the action adapter; task and policy modules must not duplicate
it.

## Output files

Runs are written under `dexgrasp/runs/`:

- `<run-id>.mp4`: RGB recording;
- `<run-id>_mask.mp4`: separate final-mask recording when enabled;
- `<run-id>_observation_visualization/`: final mask/point-cloud snapshots;
- `v94_supervised_real_<run-id>.json`: transaction and failure audit;
- `v94_supervised_real_<run-id>_rh56_force.csv`: RH56 feedback;
- `<run-id>_policy_io.npz`: exact policy I/O when requested.

These files are runtime evidence and are intentionally ignored by Git.

## Common failures

| Message | Meaning / next check |
| --- | --- |
| `control_dt differs` | Checkpoint rate and `--policy-rate-hz` disagree |
| `q_home` or reset-profile mismatch | Use the task's reset command/profile; do not widen the takeover envelope |
| `object tracker ROI initialization failed` | Reacquire a box with valid in-range depth and visible object pixels |
| `no fresh policy point cloud` / `stale_palm` | Inspect the saved mask/video and camera timing before changing safety holds |
| `D435 formal publication stalled` | Inspect USB/frame timestamp diagnostics and camera ownership |
| `RH56 ... timeout/fault` | Stop, run read-only status, and follow the commissioned recovery procedure |
| `compiled profile/envelope differs` | Rebuild and offline-test the native Franka servo against the selected profile |
| `No space left on device` | Archive old run/test data to a dedicated data volume |

Read-only hardware preflight:

```bash
.venv/bin/python dexgrasp/apps/control_preflight.py --read-hardware
```

Do not repeatedly issue reset/clear-error writes after an RH56 or Franka fault.
Use the matching commissioned recovery application and inspect its refusal.

## Code ownership

| Module | Responsibility |
| --- | --- |
| `deployment/` and `deploy.py` | Admission, reset, lease, action audits, and deployment CLI |
| `tasks/` | Task-specific configuration, contracts, and command composition |
| `observation/` | Camera profiles, RGB-D/point-cloud construction, ROI, preview, and visualization |
| `runtime/` | Bounded execution, observation/action, and device-owner scheduling |
| `diagnostics/` | Offline saved-data audits, comparisons, replays, and exporters |
| `commissioning/` | Explicit calibration, identification, timing, and physical probe tools |
| `policy/` | NumPy inference, policy-I/O evidence, and policy-rate contracts |
| `contracts/` | General observations plus V94 action/runtime contracts |
| `replay/` | Replay data models, validation/loading, and transactional policy implementation |
| `closed_loop/` | Authorization, immutable commands, dual-ACK ledger, mapper, and thread ownership |
| `configs/` | Shared sim-to-real defaults and identified dynamics evidence |
| `perception.py` | Stable camera-only perception command |
| `action_replay.py` | Compatibility facade for `replay/` |
| `closed_loop_core.py` | Compatibility facade for `closed_loop/` |
| `../motion_planning/` | Kinematics, trajectories, and tabletop interception |
| `../robot_control/` | FR3/RH56 sessions, transports, watchdogs, and safety commits |

New modules and refactors follow the workspace rules in
[`../docs/REPOSITORY_ARCHITECTURE.md`](../docs/REPOSITORY_ARCHITECTURE.md).

## Design documents

- [`CLOSED_LOOP_CONTROL_DESIGN.md`](../docs/sim2real/CLOSED_LOOP_CONTROL_DESIGN.md)
- [`COMMISSIONING_PREFLIGHT.md`](../docs/sim2real/COMMISSIONING_PREFLIGHT.md)
- [`FRANKA_PYLIBFRANKA_BACKEND.md`](../docs/sim2real/FRANKA_PYLIBFRANKA_BACKEND.md)
- [`V94_RH56_ACTION_MAPPING_AUDIT.md`](../docs/sim2real/V94_RH56_ACTION_MAPPING_AUDIT.md)
