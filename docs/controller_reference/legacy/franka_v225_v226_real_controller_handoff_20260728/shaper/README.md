# Franka V225/V226 Real-Controller Handoff

Use these files to align the FR3 deployment controller with the accepted
Isaac Lab execution contract:

- `franka_v225_interpolated_reference.hpp`: dependency-free C++17 command
  mapping and stateful 1 kHz motion generator.
- `franka_v225_libfranka_example.cpp`: libfranka callback and explicit
  filter/rate-limiter arguments.
- `franka_v225_controller_config.yaml`: single source of controller and
  Inspire action parameters.
- `franka_v225_interpolated_reference.py`: NumPy numerical oracle.
- `franka_v225_cpp_regression.cpp`: standalone C++ regression executable.
- `franka_v225_regression_vectors.json`: committed packet-level expected values.
- `../docs/franka_v225_controller_contract.md`: rationale, initialization,
  timing, evidence, and required logs.

Compile the dependency-free regression program with:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
  shaper/franka_v225_cpp_regression.cpp \
  -o /tmp/franka_v225_cpp_regression
/tmp/franka_v225_cpp_regression
```

The first joint's positive 18 mrad command should be approximately 8.93 mrad
after 50 packets, 15.58 mrad after 100 packets, and 17.995 mrad after 300
packets. Run the following for simulator, NumPy, action-mapping, and C++
packet-by-packet regressions:

```bash
pytest -q tests/test_franka_command_shaper.py \
  tests/test_franka_v225_cpp_reference.py
```

Do not deploy `franka_v94_shaper_reference.py`; it is retained only as the
rejected historical ablation. Do not stack another low-pass, velocity,
acceleration, or jerk shaper around the accepted V225/V226 generator.
