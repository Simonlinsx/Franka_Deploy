from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anydex_pipeline.snapshot import (
    GraspCandidates,
    VisualizationSnapshot,
    load_snapshot_npz,
    save_snapshot_npz,
)
from anydex_pipeline.types import PointCloudObservation


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _raw_snapshot():
    scene = np.asarray([[0.1, 0.0, 0.3], [0.2, 0.0, 0.3]], dtype=np.float32)
    obj = np.stack(
        [
            np.linspace(0.0, 0.019, 20),
            np.zeros(20),
            np.full(20, 0.4),
        ],
        axis=1,
    ).astype(np.float32)
    empty = GraspCandidates(
        canonical_poses=np.empty((0, 4, 4)),
        scores=np.empty((0,), dtype=np.float32),
        selected_index=-1,
    )
    return VisualizationSnapshot(
        scene_points=scene,
        scene_colors=np.full_like(scene, 0.2),
        object_points=obj,
        object_colors=np.full_like(obj, 0.8),
        grasps=empty,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
        frame_id=7,
        timestamp_s=42.0,
        calibration_id="calibration-test",
        camera_serial="camera-test",
        model_name="raw calibrated D435 object capture (no inference)",
    )


def _official_snapshot(raw, count=2):
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, 0, 3] = np.arange(count) * 0.01
    grasps = GraspCandidates(
        canonical_poses=poses,
        scores=np.linspace(0.9, 0.8, count, dtype=np.float32),
        type_ids=np.full(count, 4, dtype=np.int32),
        collision_free=np.zeros(count, dtype=np.bool_),
        collision_checked=np.zeros(count, dtype=np.bool_),
        selected_index=0,
        hand_poses=poses.copy(),
        hand_angles=np.full((count, 6), 900.0, dtype=np.float32),
        widths_m=np.full(count, 0.05, dtype=np.float32),
        depths_m=np.full(count, 0.02, dtype=np.float32),
        source_indices=np.arange(count, dtype=np.int64),
    )
    return replace(
        raw,
        grasps=grasps,
        model_name="AnyDexGrasp official representation + Inspire obj140 decision",
        checkpoint_sha256="a" * 64,
        representation_checkpoint_sha256="a" * 64,
        decision_checkpoint_sha256s=tuple(f"{index + 1:064x}" for index in range(8)),
        official_source_commit="c" * 40,
        inference_points=raw.object_points.copy(),
    )


def _args(app, tmp_path, *extra):
    return app.build_parser().parse_args(
        [
            str(tmp_path / "official.npz"),
            "--raw-output",
            str(tmp_path / "raw.npz"),
            "--dynamic-python",
            sys.executable,
            "--trust-official-checkpoints",
            *extra,
        ]
    )


def test_raw_capture_snapshot_contains_only_calibrated_clouds():
    app = _load("apps/capture_raw_object_snapshot.py", "test_raw_capture_app")
    points = np.stack(
        [np.linspace(0, 0.03, 20), np.zeros(20), np.ones(20)], axis=1
    ).astype(np.float32)
    observation = PointCloudObservation(
        scene_points=points,
        scene_colors=np.full_like(points, 0.2),
        object_points=points,
        object_colors=np.full_like(points, 0.8),
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
        frame_id=9,
        timestamp_s=10.0,
        calibration_id="cal",
        camera_serial="serial",
    )

    snapshot = app.raw_snapshot_from_observation(observation)

    assert snapshot.grasps.count == 0
    assert snapshot.grasps.selected_index == -1
    assert snapshot.model_name.startswith("raw calibrated D435")
    assert snapshot.reference_frame == "robot_base"


def test_workflow_commands_keep_roi_sam2_gpu_and_selected_index(tmp_path, monkeypatch):
    app = _load("apps/grasp_generation_workflow.py", "test_workflow_commands")
    monkeypatch.setattr(app, "_verify_weight_manifest", lambda *_args: None)
    args = _args(
        app,
        tmp_path,
        "--roi", "10", "20", "300", "400",
        "--sam2",
        "--device", "cuda:0",
        "--top-k", "8",
        "--selected-index", "3",
    )
    paths = app.validate_request(args)
    commands = app.build_stage_commands(
        args, paths, tmp_path / ".raw.tmp.npz", tmp_path / ".official.tmp.npz"
    )

    capture = commands["dynamic_capture"]
    inference = commands["official_inference"]
    preview = commands["dynamic_preview"]
    assert capture[capture.index("--roi") + 1 : capture.index("--roi") + 5] == [
        "10", "20", "300", "400"
    ]
    assert "--sam2" in capture
    assert "activate_official_runtime.sh" in " ".join(inference)
    assert inference[inference.index("--device") + 1] == "cuda:0"
    assert inference[inference.index("--top-k") + 1] == "8"
    assert preview[preview.index("--selected-index") + 1] == "3"
    assert preview[preview.index("--source") + 1] == "realsense"
    assert preview[preview.index("--execution-mode") + 1] == "air"


def test_dry_run_accepts_existing_regular_outputs_without_modifying_them(
    tmp_path, monkeypatch
):
    app = _load("apps/grasp_generation_workflow.py", "test_dry_run_existing")
    monkeypatch.setattr(app, "_verify_weight_manifest", lambda *_args: None)
    raw = tmp_path / "raw.npz"
    official = tmp_path / "official.npz"
    raw.write_bytes(b"existing-raw")
    official.write_bytes(b"existing-official")
    args = _args(app, tmp_path, "--dry-run")

    returned = app.run_workflow(
        args,
        stage_runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("dry-run invoked a stage")
        ),
    )

    assert returned == (raw.resolve(), official.resolve())
    assert raw.read_bytes() == b"existing-raw"
    assert official.read_bytes() == b"existing-official"


def test_dry_run_still_rejects_output_directory_and_same_output(tmp_path, monkeypatch):
    app = _load("apps/grasp_generation_workflow.py", "test_dry_run_invalid_output")
    monkeypatch.setattr(app, "_verify_weight_manifest", lambda *_args: None)
    (tmp_path / "raw.npz").mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        app.run_workflow(_args(app, tmp_path, "--dry-run"))

    same = tmp_path / "same.npz"
    with pytest.raises(ValueError, match="must be different"):
        app.run_workflow(
            app.build_parser().parse_args(
                [
                    str(same),
                    "--raw-output",
                    str(same),
                    "--dynamic-python",
                    sys.executable,
                    "--trust-official-checkpoints",
                    "--dry-run",
                ]
            )
        )


def test_weight_manifest_binds_representation_and_all_eight_heads(tmp_path):
    app = _load("apps/grasp_generation_workflow.py", "test_workflow_manifest")
    checkpoint = tmp_path / "representation.ckpt"
    checkpoint.write_bytes(b"representation")
    decisions = []
    for index in range(8):
        path = tmp_path / f"head-{index + 1}.pth"
        path.write_bytes(f"head-{index + 1}".encode())
        decisions.append(path)
    files = [checkpoint, *decisions]
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            for path in files
        )
        + "\n",
        encoding="utf-8",
    )

    app._verify_weight_manifest(manifest, checkpoint, decisions)
    decisions[-1].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        app._verify_weight_manifest(manifest, checkpoint, decisions)


@pytest.mark.parametrize(
    "extra, message",
    [
        (("--roi", "20", "0", "10", "30"), "X2>X1"),
        (("--top-k", "2", "--selected-index", "2"), "smaller than --top-k"),
        (("--device", "cpu"), "must select CUDA"),
        (("--lift-preview-m", "0"), "finite and positive"),
        (("--lift-preview-m", "-0.01"), "finite and positive"),
    ],
)
def test_workflow_rejects_invalid_request_before_any_stage(
    tmp_path, monkeypatch, extra, message
):
    app = _load(
        "apps/grasp_generation_workflow.py",
        "test_invalid_" + str(abs(hash(extra))),
    )
    calls = []
    monkeypatch.setattr(app, "_verify_weight_manifest", lambda *_args: calls.append("manifest"))
    args = _args(app, tmp_path, *extra)

    with pytest.raises(ValueError, match=message):
        app.run_workflow(args, stage_runner=lambda *_args: calls.append("stage"))

    assert calls == []


def test_workflow_commits_valid_raw_and_official_then_previews(tmp_path, monkeypatch):
    app = _load("apps/grasp_generation_workflow.py", "test_workflow_commit")
    args = _args(app, tmp_path, "--top-k", "2", "--selected-index", "1")
    paths = {
        "official": (tmp_path / "official.npz").resolve(),
        "raw": (tmp_path / "raw.npz").resolve(),
        "dynamic_python": Path(sys.executable).resolve(),
        "activation": ROOT / "scripts/activate_official_runtime.sh",
        "camera_config": ROOT.parent / "perception/configs/d435_default.yaml",
        "control_config": ROOT / "configs/fr3_rh56_v7_commissioning.json",
        "checkpoint": ROOT / "weights/logs/model/checkpoint.tar.18",
        "model_dir": ROOT / "weights/logs/model/inspire_model/obj140",
        "upstream": ROOT / "third_party/AnyDexGrasp",
    }
    monkeypatch.setattr(app, "validate_request", lambda _args: paths)
    stages = []

    def runner(name, command):
        stages.append(name)
        if name == "dynamic_capture":
            output = Path(command[command.index("--output") + 1])
            save_snapshot_npz(output, _raw_snapshot())
        elif name == "official_inference":
            output = Path(command[command.index("--output") + 1])
            save_snapshot_npz(output, _official_snapshot(_raw_snapshot()))

    raw_path, official_path = app.run_workflow(args, stage_runner=runner)

    assert stages == [
        "official_preflight",
        "dynamic_capture",
        "official_inference",
        "dynamic_preview",
    ]
    assert raw_path == paths["raw"] and official_path == paths["official"]
    assert load_snapshot_npz(raw_path).grasps.count == 0
    assert load_snapshot_npz(official_path).grasps.count == 2
    assert not list(tmp_path.glob(".*.tmp.npz"))


def test_failed_inference_leaves_no_partial_outputs(tmp_path, monkeypatch):
    app = _load("apps/grasp_generation_workflow.py", "test_workflow_failure")
    args = _args(app, tmp_path)
    paths = {
        "official": (tmp_path / "official.npz").resolve(),
        "raw": (tmp_path / "raw.npz").resolve(),
        "dynamic_python": Path(sys.executable).resolve(),
        "activation": ROOT / "scripts/activate_official_runtime.sh",
        "camera_config": ROOT.parent / "perception/configs/d435_default.yaml",
        "control_config": ROOT / "configs/fr3_rh56_v7_commissioning.json",
        "checkpoint": ROOT / "weights/logs/model/checkpoint.tar.18",
        "model_dir": ROOT / "weights/logs/model/inspire_model/obj140",
        "upstream": ROOT / "third_party/AnyDexGrasp",
    }
    monkeypatch.setattr(app, "validate_request", lambda _args: paths)

    def runner(name, command):
        if name == "dynamic_capture":
            save_snapshot_npz(
                Path(command[command.index("--output") + 1]), _raw_snapshot()
            )
        if name == "official_inference":
            raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        app.run_workflow(args, stage_runner=runner)

    assert not paths["raw"].exists()
    assert not paths["official"].exists()
    assert not list(tmp_path.glob(".*.tmp.npz"))


def test_candidate_summary_prints_all_requested_fields():
    app = _load("apps/grasp_generation_workflow.py", "test_candidate_summary")
    snapshot = _official_snapshot(_raw_snapshot(), count=2)

    lines = app.candidate_summary_lines(
        snapshot,
        selected_index=1,
        requested_top_k=64,
    )

    assert lines[0] == (
        "[candidates] returned=2 requested_top_k=64 selected=1 frame=robot_base"
    )
    assert len(lines) == 3
    assert "index=000" in lines[1]
    assert "score=0.900000" in lines[1]
    assert "type=4" in lines[1]
    assert "canonical_xyz_m=[+0.00000,+0.00000,+0.00000]" in lines[1]
    assert "approach_robot_base=[+1.00000,+0.00000,+0.00000]" in lines[1]
    assert "hand_targets=[900,900,900,900,900,900] q6=900" in lines[1]
    assert lines[2].startswith("[candidate]* index=001")


def test_actual_candidate_count_failure_still_prints_candidate_list(
    tmp_path, monkeypatch, capsys
):
    app = _load("apps/grasp_generation_workflow.py", "test_candidate_count_summary")
    args = _args(app, tmp_path, "--top-k", "3", "--selected-index", "2")
    paths = {
        "official": (tmp_path / "official.npz").resolve(),
        "raw": (tmp_path / "raw.npz").resolve(),
        "dynamic_python": Path(sys.executable).resolve(),
        "activation": ROOT / "scripts/activate_official_runtime.sh",
        "camera_config": ROOT.parent / "perception/configs/d435_default.yaml",
        "control_config": ROOT / "configs/fr3_rh56_v7_commissioning.json",
        "checkpoint": ROOT / "weights/logs/model/checkpoint.tar.18",
        "model_dir": ROOT / "weights/logs/model/inspire_model/obj140",
        "upstream": ROOT / "third_party/AnyDexGrasp",
    }
    monkeypatch.setattr(app, "validate_request", lambda _args: paths)

    def runner(name, command):
        if name == "dynamic_capture":
            save_snapshot_npz(
                Path(command[command.index("--output") + 1]), _raw_snapshot()
            )
        elif name == "official_inference":
            save_snapshot_npz(
                Path(command[command.index("--output") + 1]),
                _official_snapshot(_raw_snapshot(), count=2),
            )

    with pytest.raises(ValueError, match="outside official candidates 0..1"):
        app.run_workflow(args, stage_runner=runner)

    output = capsys.readouterr().out
    assert "[candidates] returned=2 requested_top_k=3 selected=2" in output
    assert "index=000" in output and "index=001" in output
    assert not paths["raw"].exists()
    assert not paths["official"].exists()


def test_no_overwrite_publication_race_never_clobbers_or_leaves_half_pair(
    tmp_path, monkeypatch
):
    app = _load("apps/grasp_generation_workflow.py", "test_no_clobber_race")
    args = _args(app, tmp_path, "--top-k", "2", "--selected-index", "0")
    paths = {
        "official": (tmp_path / "official.npz").resolve(),
        "raw": (tmp_path / "raw.npz").resolve(),
        "dynamic_python": Path(sys.executable).resolve(),
        "activation": ROOT / "scripts/activate_official_runtime.sh",
        "camera_config": ROOT.parent / "perception/configs/d435_default.yaml",
        "control_config": ROOT / "configs/fr3_rh56_v7_commissioning.json",
        "checkpoint": ROOT / "weights/logs/model/checkpoint.tar.18",
        "model_dir": ROOT / "weights/logs/model/inspire_model/obj140",
        "upstream": ROOT / "third_party/AnyDexGrasp",
    }
    monkeypatch.setattr(app, "validate_request", lambda _args: paths)

    def runner(name, command):
        if name == "dynamic_capture":
            save_snapshot_npz(
                Path(command[command.index("--output") + 1]), _raw_snapshot()
            )
        elif name == "official_inference":
            save_snapshot_npz(
                Path(command[command.index("--output") + 1]),
                _official_snapshot(_raw_snapshot(), count=2),
            )
            # Simulate another process winning the output-name race after the
            # workflow's initial non-existence check.
            paths["official"].write_bytes(b"concurrent-owner")

    with pytest.raises(FileExistsError):
        app.run_workflow(args, stage_runner=runner)

    assert paths["official"].read_bytes() == b"concurrent-owner"
    assert not paths["raw"].exists()
    assert not list(tmp_path.glob(".*.tmp.npz"))


def test_overwrite_publication_failure_restores_previous_pair(tmp_path, monkeypatch):
    app = _load("apps/grasp_generation_workflow.py", "test_overwrite_rollback")
    args = _args(
        app,
        tmp_path,
        "--top-k",
        "2",
        "--selected-index",
        "0",
        "--overwrite",
    )
    paths = {
        "official": (tmp_path / "official.npz").resolve(),
        "raw": (tmp_path / "raw.npz").resolve(),
        "dynamic_python": Path(sys.executable).resolve(),
        "activation": ROOT / "scripts/activate_official_runtime.sh",
        "camera_config": ROOT.parent / "perception/configs/d435_default.yaml",
        "control_config": ROOT / "configs/fr3_rh56_v7_commissioning.json",
        "checkpoint": ROOT / "weights/logs/model/checkpoint.tar.18",
        "model_dir": ROOT / "weights/logs/model/inspire_model/obj140",
        "upstream": ROOT / "third_party/AnyDexGrasp",
    }
    paths["raw"].write_bytes(b"previous-raw")
    paths["official"].write_bytes(b"previous-official")
    monkeypatch.setattr(app, "validate_request", lambda _args: paths)
    official_temp = None

    def runner(name, command):
        nonlocal official_temp
        if name == "dynamic_capture":
            save_snapshot_npz(
                Path(command[command.index("--output") + 1]), _raw_snapshot()
            )
        elif name == "official_inference":
            official_temp = Path(command[command.index("--output") + 1])
            save_snapshot_npz(
                official_temp, _official_snapshot(_raw_snapshot(), count=2)
            )

    real_replace = app.os.replace

    def fail_official_commit(source, destination):
        if official_temp is not None and Path(source) == official_temp:
            raise OSError("injected second-file publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(app.os, "replace", fail_official_commit)

    with pytest.raises(OSError, match="injected second-file"):
        app.run_workflow(args, stage_runner=runner)

    assert paths["raw"].read_bytes() == b"previous-raw"
    assert paths["official"].read_bytes() == b"previous-official"
    assert not list(tmp_path.glob(".*.workflow-backup"))
    assert not list(tmp_path.glob(".*.tmp.npz"))


def test_overwrite_never_moves_a_directory_that_appears_before_commit(tmp_path):
    app = _load("apps/grasp_generation_workflow.py", "test_overwrite_late_directory")
    raw_temp = tmp_path / ".raw.tmp.npz"
    official_temp = tmp_path / ".official.tmp.npz"
    raw_output = tmp_path / "raw.npz"
    official_output = tmp_path / "official.npz"
    raw_temp.write_bytes(b"new-raw")
    official_temp.write_bytes(b"new-official")
    official_output.mkdir()

    with pytest.raises(ValueError, match="became a non-file"):
        app._publish_outputs(
            raw_temp=raw_temp,
            raw_output=raw_output,
            official_temp=official_temp,
            official_output=official_output,
            overwrite=True,
            token="test-token",
        )

    assert official_output.is_dir()
    assert not raw_output.exists()
    assert raw_temp.read_bytes() == b"new-raw"
    assert official_temp.read_bytes() == b"new-official"
