#!/usr/bin/env python3
"""Create a measured-looking offline fixture with the C++ fake controller."""

from pathlib import Path
import sys

import _anydex_franka_telemetry as native


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: publish_cross_version_fixture.py MAP")
    path = Path(sys.argv[1])
    provenance = {
        "run_uuid": "12345678-1234-5678-9234-567812345678",
        "execution_contract_sha256": "11" * 32,
        "source_snapshot_sha256": "22" * 32,
        "control_config_sha256": "33" * 32,
        "calibration_sha256": "44" * 32,
        "producer_build_sha256": "55" * 32,
        "created_monotonic_ns": 9_000_000_000,
        "created_unix_ns": 9_000_000_000,
        "producer_name": "offline-cross-version",
        "robot_id": "offline-fake",
    }
    producer = native.NativeTelemetryProducer.create(
        str(path), provenance, 0x1111, 0x2222, 1
    )
    producer.set_stage("moving_grasp", 7, 0)
    fake = native.make_fake_control_for_offline_test(1000)
    state, _ = fake.readOnce()
    producer.synchronize_and_publish_arm(
        state, 10_000_000_000, 10_000_000_000
    )
    producer.publish_hand(
        [900, 800, 700, 600, 500, 950],
        [900, 800, 700, 600, 500, 950],
        [1, 2, 3, 4, 5, 6],
        None,
        [30, 31, 32, 33, 34, 35],
        [2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0],
        10_000_000_000,
        10_000_000_000,
    )
    producer.close()


if __name__ == "__main__":
    main()

