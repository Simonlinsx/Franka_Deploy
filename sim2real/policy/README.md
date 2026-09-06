# Policy runtime

- `__init__.py`: checkpoint-backed rolling policy and action-controller
  parameters, preserving the public `sim2real.policy` API;
- `io_recorder.py`: exact policy observation/action evidence writer;
- `rate_mode.py`: validated checkpoint/runtime rate contracts.

This package performs no hardware I/O. Device command authorization remains in
the supervised runtime and `robot_control`.
