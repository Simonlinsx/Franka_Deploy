# Continuous telemetry viewer contract

Status: **the native transport, strict read-only viewer, fused Franka producer,
validated RH56 feedback adapter, immutable session manifest, and executor wiring
are implemented and offline-tested**. The opt-in producer starts only inside a
formal executor after its existing offline/live/audit/Desk gates and after the
single Franka and RH56 owners already exist. No new hardware motion was performed
to validate this documentation update, so offline/fake coverage must not be
reported as a completed real-device A/B.

The viewer must never start an FCI or RH56 client. The producer also does not
create a second client:

- Franka publication is fused into the existing owner's
  `ActiveControlBase.readOnce()` through `read_once_tapped(control)`. It replaces
  that one read for the active loop; it does not supplement it with another read.
  Already-read validated boundary states may be published separately without a
  new device read.
- RH56 publication is a passive, run-identity-bound observer on the existing
  serial owner's validated feedback path. A sample is emitted only after the
  owner's register, target, envelope, status, fault, and the phase-applicable
  current/contact policy have accepted it. Installing the observer adds no
  serial read.
- The default arm decimation is 20, nominally about 50 Hz in a 1 kHz active
  control loop. During stages with active hand-feedback verification, RH56
  cadence follows the existing validated serial operations and is typically
  about 5–8 Hz; it is not a new timer or a hard real-time guarantee.
- Ordinary `grasp`/`air-grasp` bounded holds use that same unique RH56 owner for
  a read-only verification about every 0.20 s (and at least once for a zero
  duration). The verifier checks the close-time per-axis current caps, exact
  target, status/angle/envelope and, for air mode, all-six settled with zero
  contact. Its surrounding Franka read-only safety gate also refreshes the arm
  stream; no second device client or background polling thread is created.

Telemetry is observability only. Its manifest always records
`motion_authorized=false` and cannot bypass any motion gate.

## Trust and display rules

- A viewer instance is constructed for one exact `run_uuid` and five expected
  SHA-256 values: execution contract, source snapshot, control config,
  calibration, and producer build. A mismatch rejects the entire publication.
- `stage.epoch` changes whenever stage/target authority changes. Reusing an epoch
  with another stage name, rolling it backward, or mixing stream epochs rejects
  the publication.
- Each native arm/hand slot is read through the core's stable double-slot
  protocol. `NO_DATA`, `CONTENDED`, malformed fields, stream sequence rollback,
  or mutation under one stream sequence hides only that stream.
- Arm/hand stage epoch, exact stage name, FNV-1a name hash, and (when both are
  nonzero) bundle sequence must match before their samples may be combined into
  a green hand mesh. The fresh arm EE can still be displayed alone.
- Arm and hand timestamps are checked independently with both realtime and
  monotonic clocks. A fresh arm can remain visible when the hand is stale. A
  fresh hand without a fresh arm cannot be placed geometrically, so its green
  mesh remains hidden.
- The EE overlay accepts only a measured Franka `RobotState.O_T_EE` sample. It is
  labeled **measured EE feedback**, not ground-truth Cartesian pose.
- The RH56 overlay accepts only measured `ANGLE_ACT` register readback. The green
  mesh is labeled **official kinematic model reconstruction from six actuator
  registers**. It is not a measured surface and not 12 independent joint sensing.
- UUIDs and hashes prevent accidental cross-run display; they are not
  cryptographic authentication and never authorize motion.

Visibility is intentionally asymmetric:

| Arm stream | Hand stream | EE/error overlay | Green hand mesh |
| --- | --- | --- | --- |
| fresh | fresh, same run/epoch | shown | shown as model reconstruction |
| fresh | missing/stale/replayed | shown | hidden |
| missing/stale/replayed | fresh | hidden | hidden (no fresh wrist transform) |
| envelope torn or run/hash mismatch | any | hidden | hidden |

## Native reader mapping

The transport is fixed-POD/shared-memory. The pybind reader exposes three plain
Python mappings to the hardware-free `NativeContinuousTelemetryAdapter`; neither
the writer nor viewer serializes or polls JSON.

```json
{
  "run_uuid": "12345678-1234-5678-9234-567812345678",
  "execution_contract_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_snapshot_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "control_config_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "calibration_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "producer_build_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  "created_unix_ns": 1784656800000000000,
  "created_monotonic_ns": 123000000000,
  "producer_name": "reviewed-single-owner-producer",
  "robot_id": "fr3-rh56"
}
```

An `OK` arm read is:

```json
{
  "available": true,
  "code": 0,
  "sequence": 8101,
  "attempts": 1,
  "sample": {
    "timestamp_unix_ns": 1784656800124000000,
    "timestamp_monotonic_ns": 123455789000,
    "stage": {"name": "moving_pregrasp", "epoch": 3, "name_hash64": 123},
    "bundle_sequence": 42,
    "source": "franka_robot_state.O_T_EE",
    "source_code": 1,
    "measurement_kind": "measured",
    "measurement_kind_code": 1,
    "O_T_EE": ["16 finite values in libfranka column-major order"],
    "q": ["7 finite values or null"],
    "dq": ["7 finite values or null"],
    "control_command_success_rate": 0.99
  }
}
```

An `OK` hand read has the same outer five fields and this sample payload:

```json
{
  "timestamp_unix_ns": 1784656800100000000,
  "timestamp_monotonic_ns": 123431789000,
  "stage": {"name": "moving_pregrasp", "epoch": 3, "name_hash64": 123},
  "bundle_sequence": 42,
  "source": "inspire_rh56.ANGLE_ACT",
  "source_code": 2,
  "measurement_kind": "measured",
  "measurement_kind_code": 1,
  "angles": [1000, 1000, 997, 1000, 1000, 985],
  "angle_targets": [-1, -1, -1, -1, -1, -1],
  "current_mA": [0, 0, 0, 0, 0, 0],
  "force_g": [-10, -10, -10, -10, -10, -10],
  "temperature_c": [30, 30, 30, 30, 30, 30],
  "status": [2, 2, 2, 2, 2, 2],
  "errors": [0, 0, 0, 0, 0, 0]
}
```

The illustrative string placeholders and `name_hash64: 123` above are not valid
samples. Production mappings contain numeric arrays and the exact FNV-1a 64-bit
hash of the untruncated, non-NUL UTF-8 stage name. The adapter rejects missing or
unknown fields instead of repairing them. Read codes are `OK=0`, `NO_DATA=1`, and
`CONTENDED=2`; `available` is true if and only if the code is `OK`.

## Viewer interface

### Build and immutable session identity

The two extensions intentionally use different CPython ABIs. Build both before
starting a fresh audit window:

```bash
cd /home/qiaoguanren/code/franka/dexgrasp
./scripts/build_continuous_telemetry.sh
```

Default outputs are:

```text
viewer read-only module, CPython 3.10: /tmp/anydex-native-telemetry-viewer-py310/python
executor producer module, CPython 3.9: /tmp/anydex-franka-telemetry-producer-py39/python
control-side test reader, CPython 3.9: /tmp/anydex-native-telemetry-control-py39/python
```

The build runs the native contract tests, fake `ActiveControlBase` round trip,
and a CPython 3.9 producer → CPython 3.10 read-only reader file test. It opens no
robot, hand, camera, controller or GUI. Passing it is not hardware validation.

Each execution needs a newly generated manifest and a fresh, nonexistent `/tmp`
mapping path. For an air-grasp run:

```bash
PRODUCER_PYTHON_DIR=/tmp/anydex-franka-telemetry-producer-py39/python
PRODUCER_SO=$PRODUCER_PYTHON_DIR/_anydex_franka_telemetry.cpython-39-x86_64-linux-gnu.so

./scripts/telemetry_session_manifest.sh create \
  --snapshot "$SNAPSHOT" \
  --config "$PROFILE" \
  --audit-artifact "$AUDIT" \
  --producer-build "$PRODUCER_SO" \
  --command air-grasp \
  --selected-index "$CANDIDATE" \
  --output "$TELEMETRY_MANIFEST"
```

The command is hardware-free. `pregrasp`, `grasp`, and `grasp-lift` require
their own command-specific audit types; one command's manifest cannot be reused
for another. The producer binary path and SHA-256 are part of the identity, so a
rebuild requires a new manifest.

### Read-only window

Continuous mode is separate from the existing waypoint JSON mode. It takes the
native mapping path plus one immutable manifest; individual hashes are not
accepted from loose command-line strings:

```bash
./scripts/run_live_pipeline_preview.sh \
  /absolute/path/to/official_snapshot.npz \
  --control-config /absolute/path/to/control_config.json \
  --selected-index "$CANDIDATE" \
  --source snapshot \
  --execution-mode air \
  --continuous-telemetry /tmp/fr3_rh56_UNIQUE_SESSION.map \
  --telemetry-session-manifest /tmp/fr3_rh56_UNIQUE_SESSION.manifest.json \
  --continuous-telemetry-python-dir /tmp/anydex-native-telemetry-viewer-py310/python \
  --telemetry-wait-seconds 60 \
  --arm-max-age-s 0.25 \
  --hand-max-age-s 0.75 \
  --show-current-hand-mesh
```

`--continuous-telemetry` and `--telemetry-session-manifest` are required
together. Continuous mode and legacy `--pose-state` are mutually exclusive.
The manifest binds the exact run UUID, execution contract, snapshot, control
config, calibration, producer build, selected candidate and target poses. The
viewer replays all bound file hashes before opening the mapping. The reader also
requires ABI major 1 and exact schema digest
`22347780a4caac337192480aff948c65b01e9ce2ab4fd1d1ae97e8a4c0317bb5`.

Append `--validate-only` to validate the snapshot/config/manifest and their
target-pose agreement. That path does not import the native transport, Open3D,
RealSense, libfranka, or an RH56 driver, and does not open the mapping. Normal
continuous mode imports only `_anydex_telemetry` and calls
`TelemetryReader.open_read_only(...)`; it never calls mapping initialization,
writer APIs, or either device API.

The intended launch order is viewer first, executor second. The viewer receives
the **CPython 3.10 reader** directory; the executor receives the **CPython 3.9
producer** directory. Because the producer creates the fresh mapping with
`O_EXCL` only when that executor run starts,
the viewer may wait up to `--telemetry-wait-seconds` using a monotonic deadline
and at most 100 ms between read-only open attempts. It retries only native
`InitCode::kOpenFailed` with `ENOENT` and `InitCode::kLayoutNotReady` (plus a
direct Python `FileNotFoundError`). There is one narrowly bounded exception:
after the producer wins `O_EXCL` but before `ftruncate(2112)`, a zero-byte regular
file may be visible. Only that exact state gets at most 0.50 s initialization
grace, bounded by the overall wait deadline. A nonzero incompatible size and a
persistent zero-byte file fail immediately/after that short grace; they do not
inherit the full 60 s wait. Permission, ABI/layout, mmap, lock-free, identity,
and all unknown failures are immediate errors. A wait value of zero makes one
open attempt. `KeyboardInterrupt` is not caught by this retry loop. The wait
path never calls `initialize_mapping`, creates a file, truncates it, or writes
bytes.

Never reuse a previous mapping. A stale writer claim is deliberately not
stolen, and the executor rejects any path that already exists. Generate a new
session tag, manifest/run UUID and `/tmp/...map` path for every run.

The native hot loop must not serialize JSON, allocate, print, lock a mutex, call
the filesystem, or perform a syscall. The fused native tap only copies the
already-returned Franka state into fixed POD and atomically publishes it. All
mapping initialization, manifest/hash work, Python conversion, scene rendering
and viewer I/O remain outside the 1 kHz callback. RH56 sampling remains with the
existing single serial owner. No second robot/hand client is permitted.

Implementation and offline tests live in:

- `src/anydex_pipeline/continuous_telemetry.py`
- `src/anydex_pipeline/continuous_telemetry_runtime.py`
- `src/anydex_pipeline/telemetry_session_manifest.py`
- `src/anydex_pipeline/franka_sequence_driver.py`
- `src/anydex_pipeline/inspire_sequence_driver.py`
- `apps/live_pipeline_preview.py`
- `apps/execute_control_sequence.py`
- `apps/telemetry_session_manifest.py`
- `native/franka_tap/`
- `scripts/build_continuous_telemetry.sh`
- `tests/test_continuous_telemetry.py`
- `tests/test_continuous_telemetry_runtime.py`
- `tests/test_telemetry_session_manifest.py`
- `tests/test_pipeline_preview.py`

The synthetic native binding fixture deliberately labels its source and
measurement kind `synthetic_test`; the production adapter rejects those slots
instead of displaying them as measured feedback. In formal execution the fused
producer publishes only reviewed measured-source codes from the already-existing
single Franka and RH56 owners. The viewer hides stale/replayed/incoherent streams;
the green hand remains an official six-axis kinematic reconstruction, not a
measured physical surface. No viewer-side workaround may open a second device
connection.
