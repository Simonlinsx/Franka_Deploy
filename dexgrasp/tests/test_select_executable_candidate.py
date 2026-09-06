from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps/select_executable_candidate.py"
SNAPSHOT = ROOT / "runs/d435_20260722_124733_official.npz"
CONFIG = ROOT / "configs/fr3_rh56_v7_sim2real_supervised.json"
URDF = Path("/home/qiaoguanren/code/libfranka/test/fr3.urdf")


def _load_app():
    spec = importlib.util.spec_from_file_location(
        "test_select_executable_candidate_app", APP
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not SNAPSHOT.is_file() or not CONFIG.is_file() or not URDF.is_file(),
    reason="candidate auto-selection integration inputs are unavailable",
)
def test_current_snapshot_auto_selects_highest_scoring_executable_candidate(tmp_path):
    app = _load_app()
    output = tmp_path / "selected-plan.json"
    args = app.build_parser().parse_args(
        [
            "--snapshot", str(SNAPSHOT),
            "--config", str(CONFIG),
            "--output", str(output),
        ]
    )

    selected = app.select_candidate(args)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert selected == 32
    assert artifact["candidate"]["index"] == 32
    assert artifact["candidate"]["hand_targets"] == [0, 0, 485, 951, 917, 740]
    assert artifact["fk_residual"]["passed"] is True
    assert artifact["collision_check"]["bare_fr3_self_collision_free"] is True
    assert artifact["motion_authorized"] is False
