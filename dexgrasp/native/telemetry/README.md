# Native continuous telemetry transport

This directory provides the offline-verified, fixed-ABI shared-memory transport
for continuous FR3/RH56 pose feedback. The core deliberately imports no robot,
hand, camera or control API. The repository's separate
[`../franka_tap`](../franka_tap) component is the executor-side production
adapter: it fuses Franka publication into the existing owner's `readOnce()` and
accepts already-validated RH56 feedback from the existing serial owner. See
[docs/ABI.md](docs/ABI.md) for the exact schema and memory model, and
[`../../docs/continuous_telemetry_contract.md`](../../docs/continuous_telemetry_contract.md)
for lifecycle and viewer rules.

## Offline build and verification

The host already provides `/usr/bin/cmake`, `/usr/bin/g++-9`, Python 3.9
development files and `/usr/lib/cmake/pybind11`; no download or installation is
needed. From `dexgrasp`:

```bash
./native/telemetry/build_offline.sh
```

The script builds in `/tmp/anydex-native-telemetry-build` by default and runs
the native and Python contract tests. To repeat the race and timing checks:

```bash
/tmp/anydex-native-telemetry-build/anydex_telemetry_stress 500000 4
/tmp/anydex-native-telemetry-build/anydex_telemetry_benchmark 1000000
```

Both commands are synthetic and hardware-free. Override `BUILD_DIR`, `CXX` or
`PYTHON` in the environment if needed.

For the complete executor/viewer workflow, build the two intentionally separate
CPython ABIs from the repository root:

```bash
./scripts/build_continuous_telemetry.sh
```

This produces a CPython 3.9 producer for the control environment at
`/tmp/anydex-franka-telemetry-producer-py39/python` and a CPython 3.10 read-only
viewer module at `/tmp/anydex-native-telemetry-viewer-py310/python`, then checks
a 3.9-produced mapping with the 3.10 reader. These directories are not
interchangeable. The build is device-free and does not authorize motion.

## Native producer-facing API

Include `anydex/telemetry/telemetry.hpp`, create or open one mapping during
initialization, and claim at most one writer per stream. Fill an `ArmSample` or
`HandSample` outside the transport and call `publish()`. The payload, timestamp,
source and stage are caller-owned; the transport only validates and atomically
publishes them. Writer objects must be destroyed before their mapping owner.

The real-time functions are bounded, `noexcept`, allocation-free and
syscall-free after initialization. A producer must never put file creation,
UUID/hash computation, string construction, Python conversion or clock access
inside its control callback.

The transport core itself remains device-independent. The production adapter is
kept in `native/franka_tap` so its exact pylibfranka/libfranka ABI dependency is
explicit and hash-checked rather than leaking into this library. At runtime the
executor creates a fresh mapping only after its existing motion gates and unique
owners are established; the viewer opens that mapping read-only. Building either
component alone does not connect hardware or produce feedback.
