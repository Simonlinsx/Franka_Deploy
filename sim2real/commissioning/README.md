# Commissioning tools

This package contains explicit camera timing checks, RH56 calibration and
identification utilities, and supervised physical mapping probes. Importing a
module does not open hardware. Commands that can write hardware retain their
existing confirmation and fail-closed admission checks.

Canonical module paths are `sim2real.commissioning.<tool>`, for example:

```bash
.venv/bin/python -m sim2real.commissioning.d435_timing_probe --help
```

The package is separate from:

- `sim2real.diagnostics`, which only analyzes saved data;
- `sim2real.runtime`, which owns live observation/action scheduling;
- `robot_control`, which owns reusable FR3/RH56 device boundaries.
