# Fused Franka/RH56 telemetry producer

`_anydex_franka_telemetry` is the executor-side CPython 3.9 producer for the
fixed-POD ABI in `native/telemetry`. Its telemetry API is opt-in observability,
not a motion-authority mechanism. The same exact-ABI extension also exposes one
deliberately narrow native motion primitive for a caller-owned active Cartesian
control handle; that primitive is described separately below.

## Single-owner integration

- During an active Franka loop, `read_once_tapped(control)` performs the one
  `ActiveControlBase.readOnce()` that the existing driver would perform, returns
  the same state/period pair to that driver, and publishes the decimated measured
  state. It never performs a second FCI read and never creates a second Robot or
  control handle.
- At non-realtime stage/boundary points,
  `synchronize_and_publish_arm(...)` may publish a state that the same owner has
  already read and validated. It does not read hardware.
- RH56 samples enter through `publish_hand(...)` only after the existing unique
  serial owner's identity-bound validated-feedback observer has accepted the
  complete six-axis readback. Installing telemetry adds no register read and no
  serial client.
- Stage name/epoch changes are published by the sequence state machine. Arm and
  hand retain independent atomic streams and freshness.

The default arm decimation is 20, nominally about 50 Hz inside a 1 kHz active
control loop. During stages with active hand-feedback verification, RH56 cadence
is inherited from existing validated serial operations and is normally about
5–8 Hz; it is not a new polling guarantee.

## Exact ABI build

From the `dexgrasp` root, the normal dual-environment build is:

```bash
./scripts/build_continuous_telemetry.sh
```

The producer output is:

```text
/tmp/anydex-franka-telemetry-producer-py39/python/
  _anydex_franka_telemetry.cpython-39-x86_64-linux-gnu.so
```

The build verifies the exact installed dependencies before linking:

- pylibfranka extension SHA-256
  `53ebc14276df92d0687e1673e43df17f0a4a347dedd9a0c08bb01f152e9f2610`;
- hashed libfranka DSO SHA-256
  `956d2f7e85e3c4e127899734a170dff7c91f17f560a4fa2147739631ad721a3d`;
- libfranka header source commit
  `9f9304ec0ac897eff3219a67f612b959948535e2`, with no tracked edits;
- pybind11 3.0.1 from the pinned hash in `requirements-build.txt`.

At runtime the adapter rechecks the loaded module path/hash and the mapped
libfranka DSO identity. The build exercises adapter-first and pylibfranka-first
imports, exact C++ layout sentinels, a fake control round trip, and a CPython 3.9
producer → CPython 3.10 reader mapping. These checks open no hardware; they are
not a claim of completed real-device validation.

## Native bounded Cartesian segment

`run_bounded_cartesian_segment(control, planned_start, target, config)` moves
the deadline-sensitive `readOnce` → safety checks → minimum-jerk interpolation
→ `writeOnce` path into one C++ call while the Python GIL is released. It only
accepts an already-created, uniquely owned `ActiveControlBase`; it cannot
construct a Robot, start a control handle, change controller/load/collision
configuration, or call `Robot::stop`.

Inputs are validated before the first control read. The configuration schema is
exact (unknown or missing fields fail), and it cannot weaken these communication
gates:

- at least 100 completed positive-period exact-start hold writes before motion;
- success rate at least 0.95 within at most 0.5 s before arming;
- post-qualification hard floor at least 0.80;
- time-weighted success window no longer than 0.5 s at threshold 0.95;
- read-to-write telemetry budget no greater than 500000 ns.

Every active sample also checks robot mode, all current errors, all four
contact/collision vectors, finite joint positions with commissioned margins,
the full measured rigid pose, workspace containment, and the control period.
The target is independently checked against segment and minimum-jerk speed
bounds. After the trajectory, the exact target is held until measured pose and
joint velocity remain within the supplied convergence bounds for the required
settle interval; timeout is fail-closed. A zero-displacement probe uses the same
path by setting `target == planned_start`.

The result is a fixed-schema telemetry mapping. A native safety failure raises
`NativeCartesianSegmentError`, stops issuing writes immediately, and preserves
the same fixed telemetry through `last_native_cartesian_segment_telemetry()`.
The caller remains responsible for its existing verified stop/cleanup path.

`tests/test_native_cartesian_segment.py` uses only an in-process fake
`ActiveControlBase`. It covers a successful minimum-jerk segment, exact hold,
startup non-qualification, post-arm hard-floor loss, the complete 0.5 s quality
window, contact, endpoint timeout, and rejection of weakened gates.

## Runtime lifecycle

The executor delays importing this extension until all pre-existing offline,
live, collision/audit and Desk gates have passed and both unique device owners
are connected. It then:

1. replays the immutable session manifest and exact producer binary;
2. creates a new nonexistent mapping with `O_EXCL`;
3. installs the fused Franka tap and validated RH56 observer;
4. publishes only measured source codes and validated stage transitions;
5. performs normal safe device cleanup before detaching telemetry.

The viewer uses a different CPython 3.10 read-only module from
`/tmp/anydex-native-telemetry-viewer-py310/python`. The two Python directories
are not interchangeable. Every run needs a new `/tmp` mapping path, manifest and
run UUID; stale mappings are never reclaimed. See
[`../../docs/continuous_telemetry_contract.md`](../../docs/continuous_telemetry_contract.md)
and README section 8.5 for the two-terminal workflow.
