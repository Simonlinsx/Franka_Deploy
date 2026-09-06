# Command guide

Run maintained commands from the workspace root:

```bash
cd /home/qiaoguanren/code/franka
```

## Discover task and deployment options

```bash
.venv/bin/python -m sim2real.tasks --help
.venv/bin/python -m sim2real.deployment --help
.venv/bin/python -m sim2real.perception --help
.venv/bin/python -m sim2real.validate --help
```

## Materialize sealed task configurations

These commands do not command the robot:

```bash
.venv/bin/python -m sim2real.tasks tabletop config
.venv/bin/python -m sim2real.tasks thrown_object config
.venv/bin/python -m sim2real.tasks thrown_object_v60 config
.venv/bin/python -m sim2real.tasks thrown_object_v61 config
```

Equivalent shell wrappers:

```bash
./perception/scripts/run_tabletop_task.sh config
./perception/scripts/run_thrown_object_task.sh config
```

## Camera-only perception

```bash
./perception/scripts/run_tabletop_task.sh perception --help
./perception/scripts/run_thrown_object_task.sh perception --help
.venv/bin/python -m sim2real.observation.capture --help
.venv/bin/python -m sim2real.observation.live_preview --help
```

## Motion-planning demos

```bash
./perception/scripts/run_tabletop_motion_demo.sh --help
```

Example command construction; add the existing explicit execution and
supervision flags only after the selected scenario's preflight passes:

```bash
./perception/scripts/run_tabletop_motion_demo.sh \
  sphere-posy-board-collision \
  --live-visualization
```

## Checkpoint admission and offline evidence

```bash
.venv/bin/python -m sim2real.deployment.verify --help
.venv/bin/python -m sim2real.deployment.preflight --help
.venv/bin/python -m sim2real.diagnostics.audit_thrown_v60_candidate --help
.venv/bin/python -m sim2real.diagnostics.audit_thrown_v61_bundle --help
```

## Commissioning and diagnostics

```bash
.venv/bin/python -m sim2real.commissioning.d435_timing_probe --help
.venv/bin/python -m sim2real.commissioning.rh56_v94_microprobe --help
.venv/bin/python -m sim2real.diagnostics.compare_policy_io --help
.venv/bin/python -m sim2real.diagnostics.replay_v94 --help
```

Commissioning commands that can touch hardware keep their required
confirmation flags; `--help` is the safe way to inspect them.

## Development checks

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/qiaoguanren/anaconda3/lib/python3.9/site-packages:/home/qiaoguanren/code/franka \
.venv/bin/python -m pytest -q -p no:cacheprovider sim2real/tests

PYTHONPATH=perception python -m pytest -q perception/tests
PYTHONPATH=dexgrasp/src python -m pytest -q dexgrasp/tests
```

Generated recordings, masks, telemetry, and reports belong under `data/runs`
(the existing `dexgrasp/runs` compatibility link may still be used by older
operator scripts).
