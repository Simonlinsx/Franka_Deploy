# V94 operator-supervised closed-loop control design

This document is an implementation contract, not motion authorization.  The
public deployment entry point remains hardware-inert unless execution and live
operator supervision are explicitly requested.  Offline admission, the
hardware lease and reset must pass before either actuator owner is opened.

## Ownership and rates

The executor needs five independently supervised roles:

1. The camera producer retains the current D435 capture/tracker path.
2. The policy supervisor consumes timestamp-checked camera, Franka and RH56
   snapshots at the selected 20 or 60 Hz.  It owns policy history and the
   persistent action mapper.
3. A single Franka owner creates exactly one `Robot` and one persistent joint
   position control handle.  Its 1 kHz read-once/write-once loop publishes the
   state returned by that handle and samples-and-holds the latest policy target.
   The policy process must never create a concurrent `FrankaStateReader`.
4. A single RH56 owner owns the only serial context for the whole run.  One
   ordered queue serializes target writes/readback, 20 Hz feedback, fault checks
   and emergency disable.  Policy observation reads only its timestamped cache.
   The read-only `InspireStateReader` thread must not coexist with it.
5. An interlock supervisor samples the commissioned external deadman and E-stop,
   latches any failure, and wakes both actuator stop paths.  Audit serialization
   stays off the 1 kHz path.

The Franka state used for proprioception and camera-pose interpolation must be
the state already returned by the active control handle.  A fused 1 kHz pose
history avoids a second FCI connection/read and gives the camera projector a
better capture-time pose than the current 60 Hz read-only history.

## Command transaction

For each accepted logical observation:

1. Read `previous_executed_action13` from the commit ledger and build proprio.
2. Run the checkpoint and all post-forward freshness/deadline checks.
3. Transactionally map the action without yet advancing mapper state.
4. Stage one immutable command containing the raw/executed action, Franka target,
   RH56 target, observation time and monotonically increasing sequence.
5. Publish it to the Franka sample/hold, wait for the matching Franka ACK, then
   queue it directly to the RH56 owner.  There is no parent-side serial
   `command window`; no second command may be staged while this one is
   uncommitted.
6. Franka acknowledges after the first successful `writeOnce` carrying that
   sequence.  RH56 acknowledges after its verified six-register target write
   and exact readback.  A held 20 Hz physical target is not rewritten.
7. Only when both acknowledge the same sequence does the ledger expose that
   command's `executed_policy_action13` as the next observation's previous
   action and commit the persistent mapper target.

Any partial write with unknown application state, wrong sequence, skipped
command or commit timeout is terminal:
latch the fault, stop Franka, disable RH56, verify both physical stop states and
exit.  Intermediate hand targets must not be silently dropped by a latest-only
queue.  The RH56 physical target is scheduled at 20 Hz in both policy modes;
60 Hz policy steps still commit their exact logical action sequence while held
hand targets avoid redundant numeric writes.

The hardware-free implementation of this transaction and safety latch is in
`closed_loop_core.py`.  It intentionally cannot open a device.

## Franka 1 kHz loop

The existing `FrankaSequenceDriver` already supplies strong state checks, static
tool/load provenance, command-success monitoring, stop verification and an
allocation-light read/write cadence.  Its public motion methods create finite,
precomputed segments and therefore cannot directly execute the policy.  A
persistent session must preserve the same checks while adding:

- one handle for the full authorized episode;
- a bounded command-age watchdog (sample/hold only between fresh policy ticks);
- continuous online target shaping with commissioned joint velocity,
  acceleration and jerk bounds—never a 60 Hz position step at one FCI cycle;
- per-cycle safe-joint-margin, tracking-error, contact/collision, robot-mode,
  error and communication-quality checks;
- a wall-clock deadline and immediate fault wakeup; and
- a guaranteed handle exit followed by `Robot.stop()` and consecutive fresh
  idle/low-velocity verification on every termination path.

The policy's nominal arm target rate is 0.18 rad/s while the current installed
profile is commissioned only to 0.05 rad/s.  Silently shaping it down changes
the closed-loop dynamics and is not an alignment fix.  The run must remain
locked until either the policy-side rate is changed and retrained/revalidated,
or supervised hardware commissioning accepts the required envelope.  Online
trajectory shaping also needs explicit acceleration/jerk and tracking-error
limits, which the current profile does not provide.

## RH56 loop

The continuous RH56 owner reuses the verified-write, fault, current,
temperature and disable logic and enforces the following under the exact
six-axis action range:

- one batch `ANGLE_SET` write plus readback fits the command commit deadline;
- 20 Hz feedback and 20 Hz changed-target writes share one ordered serial
  stream without overlapping transport access;
- speed/force/current limits and contact behavior are accepted;
- every bend axis and thumb rotation range commanded by V94 is commissioned;
- a write/read timeout immediately latches and executes two disable passes;
- disabled readback, low currents, idle statuses and non-moving feedback are
  verified before the process releases the serial context.

The hardware-agnostic transaction is implemented in
`rh56_transactional_actuator.py`; lifecycle, scheduling and cached feedback are
implemented in `rh56_watchdog_owner.py`.  The owner starts disarmed, binds one
thread for the life of the transport and acknowledges the RH56 side of the
action ledger only after the target transaction is verified.  Complete safety
feedback runs independently on an absolute 20 Hz schedule: transaction time is
not added to the next period, and missed slots are skipped rather than replayed.
A deterministic fake covers ordered queuing, late ACKs, partial/wrong readback,
stale feedback, watchdog and stop-verification failures without opening a
device.

The installed profile currently validates thumb rotation only over 900..1000,
while V94 mapping can request 0..1000, and six-axis coupled policy motion remains
uncommissioned.  These are hard blockers.

## Authorization and stop semantics

Importing modules, offline verification, D435 configuration and device reads do
not authorize motion.  The eventual CLI must default to audit/read-only mode.
After all offline and read-only preflight gates pass it must display a unique run
ID, requested duration, rate/position envelope and checkpoint/config hashes.  A
trusted boundary may create a short-lived `MotionAuthorization` only after the
operator explicitly approves that exact scope.  `--execute` by itself is not
sufficient.

Before arming, require a fresh healthy E-stop sample, asserted physical deadman,
verified stationary Franka, verified disabled RH56 and the exact reset scene.
Authorization expiry, deadman release, E-stop opening, stale action heartbeat,
sensor/freshness failure, policy exception, target-limit violation, FCI failure
or RH56 failure all latch the session terminally.  A fault cannot be reset in
process.  Recovery requires verified Franka stop, verified RH56 disable, process
restart, a new run ID and new operator approval.

## Prior-action and mutable-state rules

The read-only preview correctly feeds the training reset value because it never
executes a command.  Active control must instead use the last *dual-acknowledged*
`executed_policy_action13`.  Policy proposals, clipped targets that were not
published, and a one-sided device write never advance it.

`V94ActionMapper.map()` currently mutates its arm/hand targets immediately.  The
active executor needs propose/commit behavior (or an equivalent rollback-safe
wrapper), because a post-forward age/deadline rejection must not accumulate an
unexecuted target.  The mapper must persist across committed ticks; recreating it
from measured state each tick, as the read-only preview deliberately does, is
not the simulator's closed-loop target semantics.  Mapper target bounds must
also be contracted by the same commissioned joint-limit margin enforced by the
FCI loop.

## Gates still open

The current `deployment_safety` audit remains locked.  In addition to perception
and replay items being optimized separately, the control-specific blockers are:

- `hardware_writes_enabled` and explicit per-run operator authorization;
- policy reset path and workspace collision verification;
- installed payload mass/CoM/inertia and FR3 hardware revision verification;
- installed adapter/RH56 collision model and removal of the low-speed-unloaded
  restriction;
- policy nominal 0.18 rad/s versus commissioned 0.05 rad/s, plus missing online
  acceleration/jerk/tracking limits;
- RH56 six-axis and fingertip-FK commissioning, including the full thumb range;
- verified 60 Hz dual-device command/feedback latency and maximum inter-action
  gap under simultaneous camera/model load;
- an independent physical hold/deadman and emergency-stop acceptance test; and
- collision/path commissioning evidence for the training reset.  The dedicated
  `fr3_rh56_v94_commissioning.json` now records the exact V94 `q_home` while the
  older general V7 profile remains unchanged, so the configuration mismatch is
  gone.  Its provenance is still read-only and explicitly says the reset path,
  workspace collision and motion authorization are unverified; matching a pose
  must not be mistaken for permission to move to or away from it.

The first motion-bearing validation must be a separately authorized, short,
low-envelope hold/rate-tracking commissioning run—not task-level policy motion.
Only after its telemetry, stop paths and action transaction pass should a second
explicit authorization permit a bounded closed-loop inference test.

## Readiness levels are not interchangeable

"Ready to commission" must never be reported as "ready for the task."  Use the
following three motion-bearing levels, each with a new run ID and a separate,
scope-specific operator authorization:

### C1: supervised control commissioning only

This level permits no checkpoint action.  It covers a very short stationary
hold and then separately bounded, deterministic per-axis/rate probes.  Before C1,
the exact installed payload and collision envelope, physical deadman/E-stop,
single-owner device sessions, 1 kHz command shaping, command watchdog, telemetry
and both verified stop paths must already pass without motion in tests/preflight.
The approved C1 envelope must stay within the *currently* commissioned limits;
it cannot be used to bypass the 0.05 versus 0.18 rad/s policy mismatch.  RH56
axes/ranges are enabled one commissioned stage at a time.  Passing C1 establishes
control plumbing and measured limits only.

### C2: bounded closed-loop policy test

This is the first level that may consume V94 actions.  It additionally requires
the observation and point-cloud contract, capture-to-action latency, camera/robot
time alignment, persistent transactional mapper, exact previous-action replay,
60 Hz dual-device commit rate, maximum inter-action gap, reset/path/workspace
collision checks, and policy-versus-commissioned velocity envelope all to pass.
It must use a short timeout, restricted workspace/target envelope, no payload
grasp/lift, immediate interlock stop and a fresh explicit authorization.  Passing
C2 means this one bounded real scene is technically eligible for a supervised
closed-loop experiment; it is not production readiness or generalized task
success.

### C3: task/production readiness

This level requires the remaining distribution-randomization stress acceptance,
representative object/placement/lighting trials, grasp/contact/payload dynamics,
collision and cable-clearance coverage across the task workspace, repeated stop
and fault-injection tests, recovery procedures, task success metrics and reviewed
telemetry for failure cases.  The current bundle status explicitly says DR stress
is pending, so C3 cannot be claimed even after one successful C2 run.

The authorization scope implemented in `closed_loop_core.py` is for C2 V94
closed-loop use.  A future C1 commissioning tool must use a distinct scope and
must not import or invoke the checkpoint execution path.  This repository does
not currently provide either motion-bearing CLI.
