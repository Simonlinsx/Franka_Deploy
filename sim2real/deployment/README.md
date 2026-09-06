# Deployment

This package owns checkpoint/bundle admission, commissioned profile checks,
execution reset, hardware lease acquisition, action-mapping audits, and the
supervised deployment runner.

The public command is:

```bash
.venv/bin/python -m sim2real.deployment --help
```

Offline-only verification commands are:

```bash
.venv/bin/python -m sim2real.deployment.verify --help
.venv/bin/python -m sim2real.deployment.preflight --help
```

`runner.py` composes the runtime but does not own device transports. FR3 and
RH56 access remains under `robot_control`; scheduling and transaction owners
remain under `sim2real.runtime`.
