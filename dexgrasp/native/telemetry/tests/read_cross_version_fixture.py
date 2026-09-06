#!/usr/bin/env python3
"""Read the Python-3.9 producer fixture with the viewer's Python binding."""

from pathlib import Path
import sys

import _anydex_telemetry as native


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: read_cross_version_fixture.py MAP")
    reader = native.TelemetryReader.open_read_only(str(Path(sys.argv[1])))
    header = reader.header()
    assert header["run_uuid"] == "12345678-1234-5678-9234-567812345678"
    assert header["producer_name"] == "offline-cross-version"
    arm = reader.read_arm()
    hand = reader.read_hand()
    assert arm["available"] and arm["sequence"] == 1
    assert arm["sample"]["stage"]["name"] == "moving_grasp"
    assert arm["sample"]["stage"]["epoch"] == 7
    assert abs(arm["sample"]["O_T_EE"][12] - 0.001001) < 1.0e-15
    assert hand["available"] and hand["sequence"] == 1
    assert hand["sample"]["angles"] == [900, 800, 700, 600, 500, 950]
    assert hand["sample"]["force_g"] is None
    print("Python-3.9 producer -> viewer reader cross-version ABI passed")


if __name__ == "__main__":
    main()

