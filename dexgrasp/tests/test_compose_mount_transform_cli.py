from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from anydex_pipeline.control_frames import T_MOUNT_SOURCE_AXIS_BASIS


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps/compose_mount_transform.py"
IDENTITY = [
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
]


def _run(*arguments: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(APP)] + list(arguments),
        cwd=str(ROOT),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _inline_identity() -> list:
    return ["--F-T-EE"] + [str(value) for value in IDENTITY]


def test_identity_ee_zero_yaw_uses_10mm_seating_plane_and_basis_P():
    result = _run(
        *_inline_identity(),
        "--assembled-yaw-deg",
        "0",
        "--seating-to-source-origin-mm",
        "0",
        "0",
        "0",
    )
    assert result.returncode == 0, result.stderr
    artifact = json.loads(result.stdout)
    expected = np.array(T_MOUNT_SOURCE_AXIS_BASIS, copy=True)
    expected[2, 3] = 0.010
    np.testing.assert_allclose(
        np.asarray(artifact["T_F_hand_source"]).reshape(4, 4), expected
    )
    np.testing.assert_allclose(
        np.asarray(artifact["T_EE_hand_source"]).reshape(4, 4), expected
    )
    assert artifact["matrix_convention"]["serialization"].startswith("row-major")
    basis = artifact["provenance"]["anydex_source_axis_basis"]
    assert basis["symbol"] == "P"
    np.testing.assert_array_equal(
        np.asarray(basis["T_mount_source_row_major"]).reshape(4, 4),
        T_MOUNT_SOURCE_AXIS_BASIS,
    )
    excluded = artifact["provenance"]["excluded_from_composition"]
    assert [(item["value_mm"], item["included"]) for item in excluded] == [
        (44.0, False),
        (14.0, False),
    ]
    status = artifact["measurement_status"]
    assert status["mount_datum_captured"] is False
    assert status["commissioned"] is False
    assert status["full_execution_ready"] is False


def test_yaw_and_measured_seating_offset_are_composed_in_mount_axes():
    result = _run(
        *_inline_identity(),
        "--assembled-yaw-deg",
        "90",
        "--seating-to-source-origin-mm",
        "1",
        "-2",
        "3",
        "--adapter-seat-mm",
        "12",
    )
    assert result.returncode == 0, result.stderr
    artifact = json.loads(result.stdout)
    actual = np.asarray(artifact["T_F_hand_source"]).reshape(4, 4)
    Rz = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(
        actual[:3, :3], Rz @ T_MOUNT_SOURCE_AXIS_BASIS[:3, :3], atol=1e-12
    )
    np.testing.assert_allclose(actual[:3, 3], [0.001, -0.002, 0.015])


def test_rejects_invalid_or_ambiguous_inputs():
    too_short = _run(
        "--F-T-EE",
        *(["0"] * 15),
        "--assembled-yaw-deg",
        "0",
        "--seating-to-source-origin-mm",
        "0",
        "0",
        "0",
    )
    assert too_short.returncode != 0

    non_rigid = IDENTITY.copy()
    non_rigid[0] = 2
    bad_matrix = _run(
        "--F-T-EE",
        *[str(value) for value in non_rigid],
        "--assembled-yaw-deg",
        "0",
        "--seating-to-source-origin-mm",
        "0",
        "0",
        "0",
    )
    assert bad_matrix.returncode != 0
    assert "F_T_EE" in bad_matrix.stderr

    non_finite = _run(
        *_inline_identity(),
        "--assembled-yaw-deg",
        "nan",
        "--seating-to-source-origin-mm",
        "0",
        "0",
        "0",
    )
    assert non_finite.returncode != 0
    assert "finite" in non_finite.stderr


def test_output_artifact_is_fail_closed_and_mark_measured_is_narrow(tmp_path):
    output = tmp_path / "mount_measurement.json"
    result = _run(
        *_inline_identity(),
        "--assembled-yaw-deg",
        "0",
        "--seating-to-source-origin-mm",
        "0",
        "0",
        "0",
        "--mark-measured",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact == json.loads(result.stdout)
    assert artifact["artifact_type"] == "fr3_rh56_anydex_mount_transform_measurement"
    status = artifact["measurement_status"]
    assert status["mount_datum_captured"] is True
    assert status["commissioned"] is False
    assert status["collision_model_validated"] is False
    assert status["payload_mass_properties_validated"] is False
    assert status["full_execution_ready"] is False


def test_json_input_supports_a_dotted_key(tmp_path):
    source = tmp_path / "state.json"
    source.write_text(
        json.dumps({"robot": {"F_T_EE": IDENTITY}}), encoding="utf-8"
    )
    result = _run(
        "--F-T-EE-json",
        str(source),
        "--F-T-EE-json-key",
        "robot.F_T_EE",
        "--assembled-yaw-deg",
        "0",
        "--seating-to-source-origin-mm",
        "0",
        "0",
        "0",
    )
    assert result.returncode == 0, result.stderr
    artifact = json.loads(result.stdout)
    np.testing.assert_array_equal(artifact["inputs"]["F_T_EE"], IDENTITY)
