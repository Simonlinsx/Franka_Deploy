# V94 native Franka supervised-servo protocol (version 4)

This protocol is an execution boundary, not a formal C2 authorization. The
native process owns the only Franka `Robot` and active-control handle. Its peer
owns policy inference and the RH56 transaction. Creating or importing the
protocol library cannot open hardware.

## Transport and inherited authority

The parent passes two already-connected Unix-domain descriptors through
`exec`. The child accepts no socket path and does not listen:

- the **critical fd** is `AF_UNIX/SOCK_SEQPACKET`, full duplex, and carries
  `ARM`, `HEARTBEAT`, `TARGET`, `STOP`, `HELLO`, `IPC_READY`, `ACTION_READY`,
  `ACK`, `FAULT`, and `STOP_PROOF`;
- the **telemetry fd** is a connected `AF_UNIX/SOCK_DGRAM`, child-to-parent
  only, and carries `STATE` records. It is always nonblocking and a full queue
  increments a drop counter instead of delaying the servo.

The child checks `SO_TYPE`, `SO_DOMAIN`, and `SO_PEERCRED` for both descriptors.
Each peer UID must equal the servo effective UID and its peer PID must equal the
parent PID captured before `PR_SET_PDEATHSIG`. `HELLO.process_id` is the exec'd
child PID; the parent must compare it to the exact `Popen.pid` (and retain a
pidfd when available). Same-UID alone is not an identity proof. `SIGTERM` and
Linux `PR_SET_PDEATHSIG` are an out-of-band stop path independent of either
queue.

The child generates a 128-bit session nonce and sends `HELLO` as critical child
packet sequence 1. Every later packet carries that nonce. Parent-critical,
child-critical, and child-telemetry each have an independent sequence starting
at 1 and increasing by exactly one. A missing, duplicate, reordered,
malformed, truncated, oversized, or unknown critical packet is terminal.

There is no possible reply after a peer has actually closed its only socket.
On disconnect the child still performs the full physical-stop path and emits a
single stop summary to stderr/exit status; `FAULT` and `STOP_PROOF` sends are
best effort in that case. On a connected timeout or requested stop, both are
normal critical protocol replies.

## Fixed wire ABI

The ABI is explicitly encoded little-endian, IEEE-754 binary64, and fixed
width. The C++ declarations are packed layout sentinels, not permission to send
an arbitrary native struct. No native pointer, `bool`, `size_t`, enum storage,
or padding crosses the socket. Header size is exactly 56 bytes:

```text
u32 magic = 0x46343956              # bytes "V94F" on little endian
u16 version = 4
u16 kind
u32 payload_bytes                   # must equal the kind's exact size
u32 flags = 0
u64 packet_sequence                 # per-direction, starts at one
u64 monotonic_ns                    # CLOCK_MONOTONIC
u8  session_nonce[16]
u32 crc32
u32 reserved = 0
```

CRC-32 uses polynomial `0xedb88320`, initial state `0xffffffff`, and final
XOR `0xffffffff`. It covers the complete header and payload with the four CRC
bytes set to zero. The receiver must consume one complete seqpacket and reject
`MSG_TRUNC`; stream framing is forbidden. Maximum packet size is 1024 bytes.

The authoritative structs and compile-time sizes are in
`include/anydex/v94_franka_servo/protocol.hpp`. The version-4 payload sizes are:

| Kind | Direction | Value | Bytes |
| --- | --- | ---: | ---: |
| `ARM` | parent → child | `0x001` | 616 |
| `HEARTBEAT` | parent → child | `0x002` | 8 |
| `TARGET` | parent → child | `0x003` | 80 |
| `STOP` | parent → child | `0x004` | 16 |
| `HELLO` | child → parent | `0x101` | 164 |
| `IPC_READY` | child critical fd → parent | `0x102` | 152 |
| `STATE` | child telemetry fd → parent | `0x103` | 764 |
| `ACK` | child → parent | `0x104` | 248 |
| `FAULT` | child → parent | `0x105` | 200 |
| `STOP_PROOF` | child critical fd → parent | `0x106` | 488 |
| `ACTION_READY` | child critical fd → parent | `0x107` | 184 |

All reserved bytes must be zero. Hashes are raw digest bytes, not hexadecimal
text. Fixed character arrays are zero-filled and `detail_bytes` determines the
non-NUL diagnostic prefix.

## State machine

```text
child validates inherited fd
  -> HELLO (no Robot exists)
  -> receive one valid ARM before handshake deadline
  -> construct Robot(ip, RealtimeConfig::kEnforce)
  -> explicitly set SCHED_FIFO at the kernel maximum priority and prove
     one-logical-CPU affinity by exact syscall readback
  -> same-Robot fresh read-only static/q_home/rest preflight
  -> IPC_READY
  -> start the unique joint-position active handle
  -> at least 100 consecutive healthy nonzero-period measured-q writes
  -> ACTION_READY
  -> readOnce -> bounded IPC drain/validation/shaper -> writeOnce
  -> STATE every 16 control cycles (nominally about 62.5 Hz, best effort);
     ACK after first successful write for TARGET N
  -> common finish/Robot.stop/fresh Idle+dq verification
  -> STOP_PROOF
```

`TARGET.target_sequence` and `observation_sequence` both start at one and
increment exactly. A new target is
accepted only when no prior target is awaiting its first successful write.
`ACK(N)` therefore proves that a command shaped toward target N was accepted by
`writeOnce`; it includes the original produced time and age at that write.
Packet receipt alone never produces an ACK. The target is held
between policy updates (20 or 60 Hz according to the admitted checkpoint). Target timestamps use the same host `CLOCK_MONOTONIC` as
the header and are checked for future skew and age.

`ARM.controller_mode` selects one of two explicit target-validation contracts:

- `0` (`legacy`) retains the compiled 0.020 rad previous-held-target guard;
- `1` (`qd_g015`) disables that high-level previous-target comparison and the
  home-centered 1.21 rad episode radius, because each target is relative to
  the coherent shaper state observed by the policy rather than to the previous
  target. Compiled absolute safe-joint intervals, timestamp/sequence checks,
  tracking/contact/reflex checks, the persistent trajectory generator, and
  libfranka's final rate limiter remain unchanged.

Every version-4 `STATE` appends four explicit seven-double vectors:
`shaper_q_d_rad`, `shaper_dq_d_rad_s`, `shaper_ddq_d_rad_s2`, and
`held_q_cmd_rad`. They and `controller_state29` are committed after the final
position limiter from the same successful control-cycle write. Consequently
`commanded_q_rad == shaper_q_d_rad`, and the 29D vector can be independently
recomputed from the four vectors plus that STATE's measured q. Legacy policy
consumers may continue to use the unchanged commanded-q and 29D fields.

The parent may generate an action only from a Franka `STATE` no older than
25 ms. An age in `(25, 50]` ms is a side-effect-free, no-stage retry so one
best-effort telemetry loss does not terminate the episode; an age above 50 ms
is terminal. The independent camera-capture/pose skew remains 25 ms.

Header time is diagnostic rather than a cross-process liveness oracle. A
`TARGET` still requires both a fresh header and its independent fresh
`produced_monotonic_ns`. `HEARTBEAT` liveness is measured only from the native
child's receipt times; the child records timeout before draining the socket so
an old queued heartbeat cannot revive an expired supervisor. Its nonce, CRC,
and exact heartbeat/packet sequences remain mandatory. An authenticated exact
`STOP` is honored regardless of sender-header age because stopping is the
fail-safe interpretation.

The active loop performs the blocking `readOnce` first. Only then may it drain
a fixed maximum number of critical packets within a fixed time slice, validate
the state and target, shape the held target, and call `writeOnce`. Packet count,
IPC time, validation, and shaping all fit inside the final 800-microsecond
pre-write boundary. `STATE` is a best-effort nonblocking post-write datagram.
`ACK`, `FAULT`, and connected `STOP_PROOF` are critical; backpressure on a
critical send is a terminal fault.

Sequence one is refused until the child has observed and answered at least 100
consecutive healthy, nonzero control periods (approximately 100 ms). FCI is
nominally 1 kHz, but the native child does not add a tighter single-period or
consecutive-long-period stop policy: every positive returned period through
21 ms is admitted. The returned period includes the recovered current packet,
so 21 ms represents 20 missing packets and does not relax the 20-packet FCI
fail-stop or uncontrolled-continuation bound. A zero period after bootstrap or
a returned period above 21 ms is internally inconsistent and terminal;
libfranka remains authoritative for actual communication failure.
Every gate sample also requires read-to-write <=0.8 ms, command success rate
>=0.99, and clear
errors/contact/collision. `ACTION_READY` is sent only after this gate;
receiving `TARGET(1)` earlier is a terminal protocol fault. `IPC_READY` proves
the same-Robot independent static/q-home preflight and carries the exact
SCHED_FIFO policy, maximum priority, current logical CPU, and affinity-count
readback. Production refuses before opening the active handle unless that
affinity count is one and the Python parent confirms the expected CPU.
`IPC_READY` still does not authorize an action. The single persistent shaper
uses a velocity-aware acceleration bound: it starts reducing acceleration
before the 0.50 rad/s boundary, so clipping velocity cannot create a hidden
jerk discontinuity. The final call remains libfranka's official
joint-position `limitRate`, parameterized with the same 0.50/5/250 envelope.
Release regressions differentiate the actual emitted position sequence and
cover a full 0.060 rad q_d-g015 target, dropped/recovered commands, and every
admitted returned 1--21 ms FCI period (0--20 missing packets).

## Hard safety ceilings

The native binary compiles ceilings that an `ARM` packet cannot relax:

- command velocity 0.50 rad/s;
- measured joint-velocity fault boundary 0.70 rad/s; this is an independent
  tracking-overshoot guard and does not relax the 0.50 rad/s command ceiling;
- command acceleration 5.0 rad/s²;
- command jerk 250 rad/s³;
- start error from V94 q-home 0.01 rad L-infinity;
- legacy high-level target-to-target delta 0.020 rad L-infinity; q_d-g015
  omits this previous-target comparison because its command is relative to the
  published shaper q_d;
- legacy complete episode delta 1.21 rad L-infinity; q_d-g015 does not apply a
  home-centered episode radius. Both modes retain the same compiled absolute
  safe-joint intervals and independently velocity/acceleration/jerk-limited
  1 kHz command trajectory;
- measured/command tracking error 0.01 rad;
- TARGET packet age 50 ms, checked from its production timestamp;
- post-first-target receive gap 500 ms, allowing a bounded recoverable
  observation/USB scheduling gap while converging to and holding the last
  accepted command; the independent 100 ms parent heartbeat remains active,
  and the next target must still satisfy the 50 ms packet-age check;
- parent-process heartbeat receipt gap 100 ms;
- first target after `ACTION_READY` 5 s (covers bounded late RH56 startup);
- at most 720 policy targets in one operator-supervised session;
- control-state period 1 ms nominal; all positive returned periods through
  21 ms are admitted without an extra consecutive-period rule; this is still
  at most 20 missing packets because the returned period includes the current
  recovered packet; zero after bootstrap or a returned period above 21 ms is
  terminal;
- active session at most 15 seconds (12 seconds for 720 targets at 60 Hz plus
  three seconds for bootstrap, handoff, and final acknowledged hold/stop).

Before the active handle, the child validates the ARM-bound flange transform,
end-effector mass/COM/inertia, zero external load, safe joint intervals,
q-home error, rest velocity, mode, errors, contacts, and collisions against a
fresh `Robot.readOnce` state. The active loop performs only bounded dynamic
checks and uses the accepted V225/V226 command path: a stateful 100 Hz target
filter, 6 Hz critically damped interpolation, and libfranka's official
joint-position `limitRate` as the final guard. The archived V94 custom
jerk-limited shaper is not constructed by the production loop.

## Terminal cleanup and proof

Requested stop, SIGINT/SIGTERM/PDEATHSIG, peer disconnect,
heartbeat/target/session timeout, protocol fault, state fault, timing fault,
libfranka exception, and critical reply backpressure all enter the same cleanup
scope. A normal request or watchdog exit may attempt a terminal command when
the latest state remains healthy. A state/FCI fault preempts directly with
`Robot.stop()` and must not claim a successful finish:

1. when permitted by the terminal classification, attempt one
   `motion_finished=true` write of the last safe command;
2. call `Robot.stop()` unconditionally after a Robot was constructed;
3. release the active handle;
4. read fresh Robot states and require three consecutive strictly newer
   error/contact/collision-free samples with every `|dq| <= 0.01 rad/s`.
   Normal cleanup requires `Idle`. After an explicitly successful
   `Robot.stop()`, the exact communication-only status (`current`, `last`, and
   `only_comm` bits, no other bit) may be accepted in `Reflex` or `Idle`;
5. return all attempts and direct evidence in `STOP_PROOF`.

`STOP_PROOF` includes the pre-stop robot-time freshness anchor, both active
handle and Robot-backend release flags, all attempt flags, plus the three raw accepted summaries
(`robot_time_ms`, mode, status flags, and all seven measured `dq`) so the parent
can recompute the verdict. It describes physical cleanup independently from
whether the run faulted. A run can therefore fail while still proving a
successful stop.
