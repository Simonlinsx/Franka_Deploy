# Franka device boundary

- `session.py`: persistent ownership, targets, telemetry, and envelopes;
- `backend.py`: lazy pylibfranka adapter;
- `native_session.py`: supervised native-servo process protocol.

Importing the package never opens Franka. Hardware access still requires the
existing deployment admission and explicit runtime authorization.
