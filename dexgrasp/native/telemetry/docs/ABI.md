# AnyDex native telemetry ABI v1

This document is the normative contract for the fixed shared-memory wire
layout in `include/anydex/telemetry/telemetry.hpp`. It is an offline telemetry
transport only. The library imports no Franka, RH56 serial, RealSense, or
control API. The repository contains a separate production adapter in
`native/franka_tap`; keeping it separate preserves this core's device-independent
ABI and makes the exact pylibfranka/libfranka dependency explicit.

## Canonical schema identity

The ABI header stores SHA-256
`22347780a4caac337192480aff948c65b01e9ce2ab4fd1d1ae97e8a4c0317bb5`.
It is the digest of the following exact UTF-8 byte sequence, without a trailing
newline:

```text
anydex.telemetry.v1;endian=little;layout=2112;bootstrap=64@0;header=384@64;arm_stream=960@448;hand_stream=704@1408;arm_sample=384;hand_sample=256;atomic=u64-always-lock-free;slots=2;matrix=O_T_EE-f64x16-column-major;hand_order=pinky,ring,middle,index,thumb_bend,thumb_rotate;source=0:unknown,1:franka_robot_state.O_T_EE,2:inspire_rh56.ANGLE_ACT,32767:synthetic_test;kind=0:unknown,1:measured,2:commanded,3:derived,32767:synthetic_test;stage_hash=fnv1a64-utf8;bundle=0-unbundled
```

Every reader checks the magic, endianness, ABI version, all structure sizes and
offsets, this schema digest, non-zero session UUID and provenance hashes, and
the runtime lock-free status of every shared `std::atomic<uint64_t>`. A mismatch
fails at open time; there is no compatibility guess or partial read.

## Layout and concurrency

The mapping is exactly 2,112 bytes and has four cache-line-aligned regions:

| Offset | Size | Region |
|---:|---:|---|
| 0 | 64 | atomic initialization bootstrap |
| 64 | 384 | immutable session/provenance header |
| 448 | 960 | arm stream control and two slots |
| 1,408 | 704 | hand stream control and two slots |

Each stream has exactly one claimed writer and any number of readers. Arm and
hand claims, sequence numbers and timestamps are independent. A writer claim
left behind by a crashed process is deliberately not stolen: create a new
session UUID and mapping instead of silently accepting mixed provenance.

Production creation is one-shot: the executor selects a fresh nonexistent
`/tmp` path and the producer creates it with `O_EXCL`. A reader is allowed to
wait for an absent or not-yet-committed mapping. The read-only viewer gives only
the exact zero-byte regular-file window between `O_EXCL` and `ftruncate(2112)` a
maximum 0.50 s initialization grace; nonzero incompatible sizes fail
immediately. A path from an earlier run must never be truncated or reused.

Payloads are encoded into lock-free atomic 64-bit words. Each stream uses two
slots with an odd/even guard and a separately published latest sequence. This
avoids the undefined C++ data race of a traditional seqlock over plain bytes.
A reader accepts a sample only when both guard observations and the latest
sequence agree. It retries a bounded number of times and otherwise returns
`CONTENDED`; it never returns a possibly torn or invariant-invalid payload.

The exact read codes are:

- `0 OK`: `available == true`, sample and sequence are valid.
- `1 NO_DATA`: `available == false`, the stream has never published.
- `2 CONTENDED`: `available == false`, no stable latest slot was obtained
  within `max_attempts`.

## Hot-path contract

`ArmWriter::publish`, `HandWriter::publish`, `TelemetryReader::read_arm`, and
`TelemetryReader::read_hand` are `noexcept`. After mapping and writer-claim
initialization they perform no heap allocation, mutex operation, filesystem or
clock syscall, logging, or Python call. Callers must provide already-captured
timestamps and already-filled fixed payloads. The unit test instruments global
allocation across 10,000 publish/read pairs; the stress tool checks complete
cross-field invariants under concurrent writers/readers.

Mapping create/open/close, writer claim object allocation, and Python
conversion are initialization or non-real-time reader operations and are not
part of this hot-path guarantee.

## Provenance and stage identity

The immutable header contains:

- RFC-4122 `run_uuid` bytes;
- `execution_contract_sha256`;
- `source_snapshot_sha256`;
- `control_config_sha256`;
- `calibration_sha256`;
- `producer_build_sha256`;
- creation monotonic and Unix timestamps;
- fixed, complete UTF-8 producer and robot identifiers.

Zero UUID/hashes, missing timestamps, truncation, embedded NUL, malformed UTF-8
or non-zero bytes after the terminator fail closed. Each payload repeats a
stage epoch and a complete UTF-8 stage name of at most 31 bytes. Its hash is
FNV-1a 64-bit over the exact UTF-8 bytes (offset basis
`14695981039346656037`, prime `1099511628211`, no trailing NUL). Readers may
combine arm and hand only when stage epoch and name hash match.

`bundle_sequence == 0` means the stream is asynchronous. When both streams use
non-zero bundle sequences, equality is additionally required before treating
the pair as a coherent bundle. A mismatch does not invalidate either stream's
independent sample.

## Payload conventions

`ArmSample` is 384 bytes. `O_T_EE` is a measured 16-double transform in
libfranka column-major order; Python reconstructs it with
`reshape((4, 4), order="F")`. Optional measured `q`, `dq`, and command success
rate are described by validity bits.

`HandSample` is 256 bytes. Every six-element field has the fixed order
`pinky, ring, middle, index, thumb_bend, thumb_rotate`. Production current-hand
feedback is accepted only with source `inspire_rh56.ANGLE_ACT` and measurement
kind `measured`.

Production arm feedback is accepted only with source
`franka_robot_state.O_T_EE` and measurement kind `measured`. The pybind test
publisher forces both source and kind to `synthetic_test`; it cannot forge a
production measurement.

## Python reader mapping

The module is `_anydex_telemetry`. It is intentionally limited to mapping
initialization, read-only reads, and explicitly named synthetic-test publishes:

```python
import _anydex_telemetry as nt

nt.initialize_mapping("/tmp/anydex-unique-test.map", provenance_dict)
reader = nt.TelemetryReader.open_read_only("/tmp/anydex-unique-test.map")
provenance = reader.header()
arm_result = reader.read_arm(max_attempts=8)
hand_result = reader.read_hand(max_attempts=8)
```

`header()` returns exactly these keys:

```text
run_uuid, execution_contract_sha256, source_snapshot_sha256,
control_config_sha256, calibration_sha256, producer_build_sha256,
created_monotonic_ns, created_unix_ns, producer_name, robot_id
```

Each `read_*()` result has exactly `available`, `code`, `sequence`, `attempts`,
and `sample`. An available sample has these common keys:

```text
timestamp_monotonic_ns, timestamp_unix_ns, stage, bundle_sequence,
source, source_code, measurement_kind, measurement_kind_code
```

`stage` has exactly `name`, `epoch`, and `name_hash64`. Arm adds `O_T_EE`, `q`,
`dq`, and `control_command_success_rate`. Hand adds `angles`, `angle_targets`,
`current_mA`, `force_g`, `temperature_c`, `status`, and `errors`. Optional
fields are `None` when their validity bit is absent.

No binding in this core accepts a pylibfranka `ActiveControlBase`, opens an RH56
port, or starts a camera. The implemented `native/franka_tap` adapter is a
separate extension built against and runtime-checked against the exact installed
pylibfranka module, hashed libfranka DSO and pinned libfranka headers. It replaces
the existing owner's active-loop `readOnce()` with one fused
`read_once_tapped(control)` operation; it never adds a second FCI read. RH56
samples arrive only from the already-connected driver's identity-bound validated
feedback observer and add no serial read.

The normal viewer never calls `initialize_mapping`; it imports only the CPython
3.10 `_anydex_telemetry` reader and calls `TelemetryReader.open_read_only(...)`.
The executor imports the distinct CPython 3.9 `_anydex_franka_telemetry`
producer. An immutable session manifest binds the exact producer binary SHA-256,
run UUID, execution/snapshot/config/calibration hashes, candidate and target
poses. None of these identity fields authorize motion.
