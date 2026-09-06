# Franka persistent backend audit and boundary

This is the canonical backend note under `docs/sim2real/`.

This document records the hardware-free review behind
`franka_pylibfranka_backend.py`. It does not authorize motion and is not
commissioning evidence.

## Reviewed local implementations

- `examples/move_to_q.py`, `examples/default_pos.py` and the Cartesian examples
  use the synchronous pylibfranka active-control API:
  `Robot.start_*_control()`, then repeated `readOnce()` / `writeOnce()`, followed
  by a command whose `motion_finished` flag is true.
- `dexgrasp/src/anydex_pipeline/franka_sequence_driver.py` already uses the same
  API with one owner, realtime enforcement, one read/one write cadence and an
  unconditional `Robot.stop()` plus fresh idle-state verification.
- `dexgrasp/native/franka_tap` is a telemetry adapter, not a controller. Its
  `read_once_tapped(control)` replaces the owner's `control.readOnce()` and must
  never be added as a second read. It also requires a pre-control state/time
  synchronization step.
- The installed wheel is pylibfranka 0.21.2. Its public module exposes
  `Robot`, `ActiveControlBase`, `JointPositions`, `ControllerMode`,
  `RealtimeConfig` and the libfranka exception classes. `Robot` has no reviewed
  explicit `close()` operation in this Python API.

The synchronous API can therefore own one persistent nominal 1 kHz joint
position session. A separate C++ helper is not required to express the current
control contract. This is an API conclusion only: measured realtime behavior,
read-to-write timing and command success still require physical commissioning.
The low-rate asynchronous position handler is not suitable because it does not
provide the required measured state and safety validation on every FCI cycle.

## Delayed lifecycle

Production wiring is:

```text
construct PylibfrankaBackendFactory (no import or connection)
        |
FrankaPersistentSession validates run-scoped authorization,
preflight token, interlocks and initial target
        |
session calls the single-use factory
        |
factory lazily imports pylibfranka and constructs
verifies the audited pylibfranka 0.21.2 API, then constructs
Robot(ip, RealtimeConfig.kEnforce)
        |
backend creates exactly one joint-position ActiveControlBase
        |
owner thread: readOnce -> validate/shape/sample-hold -> writeOnce
        |
motion_finished hold -> unconditional Robot.stop()
        |
fresh Robot.read_once idle/rest verification -> release references
```

The backend constructor is sealed so an already-open `Robot` cannot be wrapped
through the production class directly. The factory is single-use, and a failed
import, connection or control start is terminal for that run. There is no
automatic reconnection or `automatic_error_recovery()` call.

As in the reviewed sequence driver, cyclic garbage collection is collected and
disabled immediately before starting the active handle, then restored during
all start-failure and close paths. Python reference counting remains active.
This removes one avoidable pause source but is not evidence that the full
session meets its commissioned read-to-write deadline.

The factory always selects `RealtimeConfig.kEnforce`. It does not configure
collision thresholds, payload, end-effector transforms, impedance or any
motion limit. Those values must come from and be verified against the exact
commissioning profile. In particular, this adapter does not fill the current
V94 profile's uncommissioned acceleration, jerk or tracking fields.

## 1 kHz owner and 60 Hz sample-hold

`FrankaPersistentSession` is the only active-handle owner. On every FCI cycle it
performs exactly one blocking active read, consumes the returned state for all
dynamic checks and the pose ring, samples the latest sequenced policy target,
and performs exactly one active write. A 60 Hz target sequence is expected to
repeat for roughly 16 or 17 nominal 1 ms FCI cycles. The adapter accepts equal
sequence numbers for this hold, accepts only the next integer on an update, and
rejects regressions and gaps.

The adapter does not currently attach `franka_tap`. The sim2real pose ring is
already populated from the exact active-handle state, so no second telemetry
read is needed. If shared-memory telemetry is later required, integration must
add an audited pre-handle synchronization boundary and route the existing read
through `read_once_tapped`; calling both it and `readOnce()` in one cycle is
forbidden.

## RobotState field boundary

The adapter returns the raw pylibfranka `RobotState` without allocating a
second state object in the hot loop. `FrankaPersistentSession` consumes and
validates these exact fields:

| Session meaning | pylibfranka field |
| --- | --- |
| Joint position / velocity | `q`, `dq` |
| Base-to-EEF / flange-to-EEF transforms | `O_T_EE`, `F_T_EE` |
| Installed end-effector dynamics | `m_ee`, `F_x_Cee`, `I_ee` |
| External-load dynamics | `m_load`, `F_x_Cload`, `I_load` |
| Total mass | `m_total` |
| Mode and errors | `robot_mode`, `current_errors` |
| Contact/collision evidence | `joint_contact`, `joint_collision`, `cartesian_contact`, `cartesian_collision` |
| Communication quality | `control_command_success_rate` |
| Robot-owned freshness clock | `time` |

The 16-value transforms and 9-value inertias retain libfranka's column-major
layout. `current_errors` retains the native pylibfranka `Errors` object so its
native boolean any-error test remains available. No state is inferred from a
command acknowledgement. `RobotState.time` must increase strictly across the
active loop and post-stop verification, so repeated cached objects cannot count
as consecutive fresh stop samples.

## Stop and exception semantics

- `finish(q)` sends one final `JointPositions(q)` with `motion_finished=True`.
  It is not treated as physical-rest evidence.
- `request_stop()` calls `Robot.stop()` once on both normal and fault cleanup,
  even if `finish()` succeeded.
- The stop attempt is recorded before invoking the binding. If `Robot.stop()`
  throws, fresh post-stop reads remain available; the session reports the stop
  failure and cannot claim a clean stop.
- Stop verification is owned by `FrankaPersistentSession`: it requires the
  profile's consecutive fresh Idle, low-velocity, error-free and
  contact/collision-free samples.
- `close()` sends no command and performs no recovery. After stop verification
  it releases the active-control and Robot references because the reviewed
  Python API has no explicit close method.
- Binding exceptions are wrapped once as `PylibfrankaBackendError` with a stable
  operation (`backend_factory`, `start_control`, `active_read`, `active_write`,
  `motion_finish`, `robot_stop` or `post_stop_read`) and retain the original
  exception as `__cause__`. `KeyboardInterrupt` and `SystemExit` propagate to
  the session, whose `finally` block still performs stop cleanup.

## Remaining physical evidence

The fake tests prove lifecycle and boundary behavior only. Before this factory
can be supplied to a motion-bearing entry point, commissioning must still bind
all currently missing profile values and demonstrate at least:

- sustained persistent FCI rate and read-to-write deadline on the deployed RT
  host;
- command-success, velocity, acceleration, jerk and tracking guards under the
  exact installed tool/load;
- collision/workspace/path limits and physical deadman/E-stop behavior;
- unconditional stop followed by consecutive fresh idle/rest samples; and
- exact Franka/RH56 dual-device sequence acknowledgement at the intended policy
  update rate.
