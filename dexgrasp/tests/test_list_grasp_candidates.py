from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from anydex_pipeline.candidate_summary import candidate_summary_lines
from anydex_pipeline.snapshot import GraspCandidates, VisualizationSnapshot, save_snapshot_npz


ROOT = Path(__file__).resolve().parents[1]


def _snapshot() -> VisualizationSnapshot:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 2, axis=0)
    poses[0, :3, 3] = [0.5, 0.1, 0.2]
    poses[1, :3, 3] = [0.6, 0.2, 0.3]
    grasps = GraspCandidates(
        canonical_poses=poses,
        scores=np.asarray([0.8, 0.9], dtype=np.float32),
        type_ids=np.asarray([4, 7], dtype=np.int32),
        collision_free=np.asarray([False, False], dtype=np.bool_),
        collision_checked=np.asarray([False, False], dtype=np.bool_),
        selected_index=0,
        hand_poses=poses.copy(),
        hand_angles=np.asarray(
            [[700, 710, 720, 730, 850, 900], [0, 358, 799, 911, 922, 646]],
            dtype=np.float32,
        ),
        widths_m=np.asarray([0.05, 0.06], dtype=np.float32),
        depths_m=np.asarray([0.02, 0.03], dtype=np.float32),
    )
    return VisualizationSnapshot(
        scene_points=np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32),
        scene_colors=np.asarray([[0.2, 0.2, 0.2]], dtype=np.float32),
        object_points=np.asarray([[0.5, 0.1, 0.2]], dtype=np.float32),
        object_colors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        grasps=grasps,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
        frame_id=1,
        timestamp_s=1.0,
    )


def _load_app():
    spec = importlib.util.spec_from_file_location(
        "test_list_grasp_candidates_app", ROOT / "apps/list_grasp_candidates.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_summary_preserves_indices_and_marks_selection():
    lines = candidate_summary_lines(_snapshot(), selected_index=1)

    assert "returned=2" in lines[0]
    assert " index=000" in lines[1]
    assert "index=001" in lines[2]
    assert lines[2].startswith("[candidate]*")
    assert "hand_targets=[0,358,799,911,922,646]" in lines[2]
    assert "q6=646" in lines[2]


def test_offline_cli_lists_existing_snapshot_without_hardware(tmp_path, capsys):
    path = save_snapshot_npz(tmp_path / "candidates.npz", _snapshot())
    app = _load_app()

    result = app.main([str(path), "--selected-index", "1"])

    assert result == 0
    output = capsys.readouterr().out
    assert "OFFLINE LIST ONLY" in output
    assert "index=001" in output
    assert "pylibfranka" not in sys.modules
