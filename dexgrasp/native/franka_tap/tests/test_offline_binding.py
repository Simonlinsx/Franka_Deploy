#!/usr/bin/env python3
"""Offline ABI/readback test; uses a C++ fake control and no hardware API."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import time
import uuid

import pylibfranka
import pylibfranka._pylibfranka as pylibfranka_native

import _anydex_franka_telemetry as producer_module
import _anydex_telemetry as reader_module


def _provenance() -> dict:
    now_mono = time.monotonic_ns()
    now_unix = time.time_ns()
    return {
        "run_uuid": str(uuid.uuid4()),
        "execution_contract_sha256": "11" * 32,
        "source_snapshot_sha256": "22" * 32,
        "control_config_sha256": "33" * 32,
        "calibration_sha256": "44" * 32,
        "producer_build_sha256": "55" * 32,
        "created_monotonic_ns": now_mono,
        "created_unix_ns": now_unix,
        "producer_name": "offline-fake-only",
        "robot_id": "offline-fake",
    }


def _actual_loaded_libfranka() -> tuple[str, ...]:
    expected = Path(os.environ["ANYDEX_EXPECTED_LIBFRANKA"]).resolve()
    loaded = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        path = line.rsplit(None, 1)[-1]
        if "/" not in path:
            continue
        candidate = Path(path)
        if candidate.name.startswith("libfranka-"):
            loaded.add(str(candidate.resolve()))
    assert loaded == {str(expected)}, (loaded, expected)
    return tuple(sorted(loaded))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_build_dependencies() -> None:
    metadata = producer_module.BUILD_DEPENDENCIES
    assert set(metadata) == {"schema_version", "pylibfranka", "libfranka"}
    assert metadata["schema_version"] == 1
    wheel_path = Path(metadata["pylibfranka"]["path"]).resolve()
    library_path = Path(metadata["libfranka"]["path"]).resolve()
    assert wheel_path == Path(pylibfranka_native.__file__).resolve()
    assert library_path == Path(os.environ["ANYDEX_EXPECTED_LIBFRANKA"]).resolve()
    assert _sha256(wheel_path) == metadata["pylibfranka"]["sha256"]
    assert _sha256(library_path) == metadata["libfranka"]["sha256"]


def main() -> None:
    assert reader_module.platform_is_supported_lock_free()
    _verify_build_dependencies()
    with tempfile.TemporaryDirectory(prefix="anydex-franka-tap-offline-") as temp:
        mapping = Path(temp) / "telemetry.mmap"
        provenance = _provenance()
        producer = producer_module.NativeTelemetryProducer.create(
            str(mapping), provenance, 0xA11CE, 0xB00B5, 2
        )
        producer.set_stage("moving_pregrasp", 1, 0)

        fake = producer_module.make_fake_control_for_offline_test(1000)
        assert isinstance(fake, pylibfranka.ActiveControlBase)
        first_state, _ = fake.readOnce()
        assert isinstance(first_state, pylibfranka.RobotState)
        assert abs(first_state.time.to_sec() - 1.001) < 1.0e-12
        assert abs(first_state.O_T_EE[12] - 0.001001) < 1.0e-15
        assert abs(first_state.q[0] - 0.0001001) < 1.0e-15
        unix_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        assert producer.synchronize_and_publish_arm(
            first_state, unix_ns, monotonic_ns
        ) == 1

        tapped_state, tapped_period = producer.read_once_tapped(fake)
        assert isinstance(tapped_state, pylibfranka.RobotState)
        assert isinstance(tapped_period, pylibfranka.Duration)
        # Exact sentinel values cross the external pybind11 module boundary.
        # Checking only Python types would not catch a mismatched 0.21.2 C++
        # RobotState/Duration layout.
        assert abs(tapped_state.time.to_sec() - 1.002) < 1.0e-12
        assert abs(tapped_state.O_T_EE[12] - 0.001002) < 1.0e-15
        assert abs(tapped_state.q[0] - 0.0001002) < 1.0e-15
        assert abs(tapped_period.to_sec() - 0.001) < 1.0e-12
        producer.publish_hand(
            [1000, 990, 980, 970, 960, 950],
            [-1, -1, -1, -1, -1, -1],
            [1, 2, 3, 4, 5, 6],
            [-10, -11, -12, -13, -14, -15],
            [30, 31, 32, 33, 34, 35],
            [2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0],
            time.time_ns(),
            time.monotonic_ns(),
        )

        reader = reader_module.TelemetryReader.open_read_only(str(mapping))
        assert reader.header() == provenance
        arm = reader.read_arm()
        hand = reader.read_hand()
        assert arm["available"] and arm["sequence"] == 2
        assert arm["sample"]["source_code"] == 1
        assert arm["sample"]["measurement_kind_code"] == 1
        assert arm["sample"]["stage"]["name"] == "moving_pregrasp"
        assert arm["sample"]["stage"]["epoch"] == 1
        assert len(arm["sample"]["O_T_EE"]) == 16
        assert hand["available"] and hand["sequence"] == 1
        assert hand["sample"]["source_code"] == 2
        assert hand["sample"]["measurement_kind_code"] == 1
        assert hand["sample"]["angles"] == [1000, 990, 980, 970, 960, 950]
        assert hand["sample"]["angle_targets"] == [-1] * 6
        assert hand["sample"]["force_g"] == [-10, -11, -12, -13, -14, -15]
        assert hand["sample"]["stage"] == arm["sample"]["stage"]
        assert _actual_loaded_libfranka()
        producer.close()
        assert producer.closed

    print("offline fake-control ABI and native reader round-trip passed")


if __name__ == "__main__":
    main()
