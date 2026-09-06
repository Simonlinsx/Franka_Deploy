#!/usr/bin/env python3
"""Offline-only contract test for the native telemetry Python binding."""

from __future__ import annotations

import gc
from pathlib import Path
import sys
import tempfile

import _anydex_telemetry as telemetry


PROVENANCE_KEYS = {
    "run_uuid",
    "execution_contract_sha256",
    "source_snapshot_sha256",
    "control_config_sha256",
    "calibration_sha256",
    "producer_build_sha256",
    "created_monotonic_ns",
    "created_unix_ns",
    "producer_name",
    "robot_id",
}
READ_KEYS = {"available", "code", "sequence", "attempts", "sample"}
COMMON_SAMPLE_KEYS = {
    "timestamp_monotonic_ns",
    "timestamp_unix_ns",
    "stage",
    "bundle_sequence",
    "source",
    "source_code",
    "measurement_kind",
    "measurement_kind_code",
}


def _provenance() -> dict[str, object]:
    return {
        "run_uuid": "00112233-4455-4677-8899-aabbccddeeff",
        "execution_contract_sha256": "01" * 32,
        "source_snapshot_sha256": "02" * 32,
        "control_config_sha256": "03" * 32,
        "calibration_sha256": "04" * 32,
        "producer_build_sha256": "05" * 32,
        "created_monotonic_ns": 100,
        "created_unix_ns": 1_700_000_000_000_000_000,
        "producer_name": "python-binding-test",
        "robot_id": "offline-fixture",
    }


def main() -> int:
    assert telemetry.ABI_MAJOR == 1
    assert telemetry.ABI_MINOR == 0
    assert (
        telemetry.ABI_SCHEMA_SHA256
        == "22347780a4caac337192480aff948c65b01e9ce2ab4fd1d1ae97e8a4c0317bb5"
    )
    assert telemetry.READ_OK == 0
    assert telemetry.READ_NO_DATA == 1
    assert telemetry.READ_CONTENDED == 2
    assert telemetry.platform_is_supported_lock_free() is True
    assert "pylibfranka" not in sys.modules
    assert "serial" not in sys.modules
    assert "pyrealsense2" not in sys.modules

    with tempfile.TemporaryDirectory(prefix="anydex-telemetry-python-") as root:
        initialized_path = Path(root) / "initialized-only.shm"
        assert (
            set(
                telemetry.initialize_mapping(
                    str(initialized_path), _provenance()
                )
            )
            == PROVENANCE_KEYS
        )
        initialized_reader = telemetry.TelemetryReader.open_read_only(
            str(initialized_path)
        )
        assert initialized_reader.read_arm()["code"] == telemetry.READ_NO_DATA
        assert initialized_reader.read_hand()["code"] == telemetry.READ_NO_DATA

        path = Path(root) / "telemetry.shm"
        session = telemetry.TelemetryTestSession.create(
            str(path), _provenance(), 0xA101, 0xB101
        )
        assert set(session.header()) == PROVENANCE_KEYS
        assert session.header()["run_uuid"] == _provenance()["run_uuid"]

        empty_arm = session.read_arm()
        empty_hand = session.read_hand()
        assert set(empty_arm) == READ_KEYS
        assert empty_arm == {
            "available": False,
            "code": telemetry.READ_NO_DATA,
            "sequence": 0,
            "attempts": 1,
            "sample": None,
        }
        assert empty_hand == empty_arm

        pose = [float(index) for index in range(16)]
        q = [index / 10.0 for index in range(7)]
        dq = [-value for value in q]
        arm_publish = session.publish_test_arm(
            {
                "timestamp_monotonic_ns": 1_000,
                "timestamp_unix_ns": 2_000,
                "stage_name": "pregrasp",
                "stage_epoch": 7,
                "bundle_sequence": 11,
                "producer_cycle": 123,
                "O_T_EE": pose,
                "q": q,
                "dq": dq,
                "control_command_success_rate": 0.99,
            }
        )
        assert arm_publish == {"code": 0, "sequence": 1}

        hand_publish = session.publish_test_hand(
            {
                "timestamp_monotonic_ns": 1_010,
                "timestamp_unix_ns": 2_010,
                "stage_name": "pregrasp",
                "stage_epoch": 7,
                "bundle_sequence": 11,
                "producer_poll": 456,
                "angles": [1000, 999, 998, 997, 996, 995],
                "angle_targets": [-1, -1, -1, -1, -1, -1],
                "current_mA": [0, 1, 2, 3, 4, 5],
                "force_g": [-10, -9, -8, -7, -6, -5],
                "temperature_c": [30, 31, 32, 33, 34, 35],
                "status": [0, 1, 2, 3, 4, 5],
                "errors": [0, 0, 0, 0, 0, 0],
            }
        )
        assert hand_publish == {"code": 0, "sequence": 1}

        reader = telemetry.TelemetryReader.open_read_only(str(path))
        assert set(reader.header()) == PROVENANCE_KEYS
        arm = reader.read_arm()
        hand = reader.read_hand()
        assert set(arm) == READ_KEYS and arm["available"] is True
        assert set(hand) == READ_KEYS and hand["available"] is True
        assert arm["sequence"] == 1 and hand["sequence"] == 1
        assert set(arm["sample"]) == COMMON_SAMPLE_KEYS | {
            "O_T_EE",
            "q",
            "dq",
            "control_command_success_rate",
        }
        assert set(hand["sample"]) == COMMON_SAMPLE_KEYS | {
            "angles",
            "angle_targets",
            "current_mA",
            "force_g",
            "temperature_c",
            "status",
            "errors",
        }
        assert arm["sample"]["O_T_EE"] == pose
        assert arm["sample"]["q"] == q
        assert arm["sample"]["dq"] == dq
        assert arm["sample"]["source"] == "synthetic_test"
        assert arm["sample"]["source_code"] == 32767
        assert arm["sample"]["measurement_kind"] == "synthetic_test"
        assert arm["sample"]["stage"] == {
            "name": "pregrasp",
            "epoch": 7,
            "name_hash64": 0x2B65809F1FE32BFD,
        }
        assert hand["sample"]["angles"] == [1000, 999, 998, 997, 996, 995]
        assert hand["sample"]["angle_targets"] == [-1] * 6

        # The streams are intentionally independent.
        assert session.publish_test_arm(
            {
                "timestamp_monotonic_ns": 1_100,
                "timestamp_unix_ns": 2_100,
                "stage_name": "pregrasp",
                "stage_epoch": 7,
                "O_T_EE": pose,
            }
        ) == {"code": 0, "sequence": 2}
        assert reader.read_arm()["sequence"] == 2
        assert reader.read_hand()["sequence"] == 1

        try:
            telemetry.TelemetryTestSession.create(
                str(path), _provenance(), 0xA102, 0xB102
            )
        except RuntimeError as exc:
            assert "code=2" in str(exc)
        else:
            raise AssertionError("exclusive creation unexpectedly succeeded")

        session = None
        gc.collect()
        assert reader.read_arm()["available"] is True

    print(
        "python telemetry binding passed; hardware_modules_imported=false "
        "reader_schema=strict"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
