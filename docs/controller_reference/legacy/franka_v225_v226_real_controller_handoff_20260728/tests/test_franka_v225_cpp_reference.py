from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_cpp_reference_matches_committed_numpy_regression(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")

    source = ROOT / "shaper" / "franka_v225_cpp_regression.cpp"
    executable = tmp_path / "franka_v225_cpp_regression"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-pedantic",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        cwd=ROOT,
    )
    output = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, cwd=ROOT
    ).stdout

    parsed: dict[str, np.ndarray] = {}
    for line in output.strip().splitlines():
        name, *values = line.split(",")
        parsed[name] = np.asarray(values, dtype=np.float64)

    reference = json.loads(
        (ROOT / "shaper" / "franka_v225_regression_vectors.json").read_text()
    )
    tolerance = float(reference["absolute_tolerance_rad"])
    np.testing.assert_allclose(
        parsed["held_target"], reference["held_target_q_rad"], rtol=0.0, atol=tolerance
    )
    for packet, expected in reference["generated_q_rad"].items():
        np.testing.assert_allclose(
            parsed[packet], expected, rtol=0.0, atol=tolerance
        )
