# Tasks

This package owns task profiles and command composition for tabletop and
thrown-object workflows.

```bash
.venv/bin/python -m sim2real.tasks --help
.venv/bin/python -m sim2real.tasks tabletop config
.venv/bin/python -m sim2real.tasks thrown_object config
```

- `launcher.py` materializes sealed task configurations and selects a public
  entry point.
- `tabletop.py` exposes tabletop command composition.
- `tabletop_demo.py` is the adaptive tabletop demo entry point.
- `thrown.py` exposes the thrown-task API.
- `thrown_contract.py` validates thrown-task camera/reset/curriculum data.
- `thrown_shadow.py` runs read-only triggered policy inference.
- `ballistics.py` contains hardware-free trajectory contracts.

Motion prediction and interception remain in `motion_planning`; this package
only chooses and validates a task.
