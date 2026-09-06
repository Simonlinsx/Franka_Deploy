# Franka dynamic manipulation workspace

This workspace contains the real-robot integration for an FR3, an Inspire
RH56 hand, and a fixed RealSense D435.  Maintained code is organized by domain;
recorded data, checkpoints, external repositories, and run output are kept
outside that source boundary.

## Start here

| Area | Responsibility | Primary documentation |
| --- | --- | --- |
| `sim2real/` | Checkpoint admission, policy inference, task composition, supervised deployment, and runtime scheduling | [`sim2real/README.md`](sim2real/README.md) |
| `dexgrasp/` | Grasp candidates, RH56 hand geometry, installed-hand tooling, and grasp execution workflows | [`dexgrasp/README.md`](dexgrasp/README.md) |
| `motion_planning/` | Hardware-inert kinematics, moving-object interception, and trajectory planning | [`motion_planning/README.md`](motion_planning/README.md) |
| `robot_control/` | FR3/RH56 transport, persistent sessions, watchdogs, authorization, and fail-closed commits | [`robot_control/README.md`](robot_control/README.md) |
| `perception/` | RealSense capture, object-mask tracking, guarded point-cloud publication, replay, and perception evaluation | [`perception/README.md`](perception/README.md) |

The dependency and naming rules for new code are documented in
[`docs/REPOSITORY_ARCHITECTURE.md`](docs/REPOSITORY_ARCHITECTURE.md).
The concise ownership map is in [`docs/MODULE_MAP.md`](docs/MODULE_MAP.md).
Maintained commands are collected in [`docs/COMMANDS.md`](docs/COMMANDS.md).
Low-level operator notes and older examples remain in
[`docs/USAGE.md`](docs/USAGE.md).

## Canonical task entry points

Use the task wrappers instead of invoking internal `v94_*` implementation
modules directly.

The stable Python APIs are `sim2real.deployment`, `sim2real.tasks`,
`sim2real.observation`, `motion_planning`, and `robot_control`. Retired flat
implementation imports have been removed; use the domain package paths.

```bash
# Tabletop task configuration/perception/deployment.
./perception/scripts/run_tabletop_task.sh

# Adaptive tabletop motion-planning demos.
./perception/scripts/run_tabletop_motion_demo.sh --help

# Thrown-object task.
./perception/scripts/run_thrown_object_task.sh

# Versioned checkpoint bundles retained for compatibility.
./perception/scripts/run_thrown_checkpoint_test.sh
./perception/scripts/run_thrown_v60_checkpoint_test.sh
./perception/scripts/run_thrown_v61_sixexpert_test.sh
```

Commands that can write to hardware retain their explicit supervision and
commissioning confirmations.  Repository cleanup must not bypass those
contracts.

## Workspace layout outside source

Generated assets and external repositories are separated from the five source
domains:

- `data/checkpoints/`: active model checkpoints;
- `data/perception_corpus/`: recorded RGB-D and replay products;
- `data/runs/`: deployment videos, masks, telemetry, and audits;
- `data/`: also contains curated test fixtures and demo evidence used by
  maintained tools;
- `third_party/`: independent SAM2, BrainCo SDK, and YOLO-World checkouts;
- `data/archives/`: historical bundles, old release packages, and extraction
  metadata that are not runtime inputs;
- `docs/controller_reference/`: commissioned and historical controller
  contracts.

Large contents are ignored by the integration repository. Reproducible runtime
inputs must be referenced by content hashes in the existing
manifests/configuration, not copied into source directories. Maintained
commands use the canonical `data/...` paths. The internal `dexgrasp/runs`
link remains temporarily because deployment output paths are also consumed by
external operator tooling.

## Development checks

Run the smallest relevant test group while editing, then the owning package's
suite before using a changed runtime on hardware.  Typical commands are:

```bash
# Use a development Python environment that includes pytest.
# sim2real
python -m pytest -q sim2real/tests

# perception package
PYTHONPATH=perception python -m pytest -q \
  perception/tests

# AnyDex/hardware integration package
PYTHONPATH=dexgrasp/src python -m pytest -q dexgrasp/tests
```

The public repository history starts from the organized source snapshot.
Local virtual environments, checkpoints, captures, run output, external
checkouts, credentials, and crash dumps remain outside Git. Hardware commands
must still be reviewed against the commissioned profile and run under direct
supervision; cloning this repository does not commission a robot or camera.
