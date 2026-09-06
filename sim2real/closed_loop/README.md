# Closed-loop transaction core

This package is hardware-inert. It separates:

- `authorization.py`: run authorization, interlocks, and sticky faults;
- `commands.py`: immutable command/proposal/commit records and sample/hold;
- `ledger.py`: exact Franka/RH56 dual-ack commit barrier;
- `mapper.py`: rollback-safe policy-action target mapping;
- `ownership.py`: single-thread endpoint ownership;
- `validation.py`: internal immutable value validators.

Use `sim2real.closed_loop_core` as the stable compatibility import. Device I/O
belongs to `robot_control`; this package must remain safe to import offline.
