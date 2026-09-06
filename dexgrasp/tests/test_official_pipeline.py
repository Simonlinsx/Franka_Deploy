from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import anydex_pipeline.official_pipeline as official_pipeline
from anydex_pipeline.inspire_decision import InspireDecisionBatch
from anydex_pipeline.official_backend import (
    OFFICIAL_NO_FLIP_FEATURE_WIDTH,
    RepresentationFeatureLayout,
    RepresentationGraspBatch,
)
from anydex_pipeline.official_pipeline import (
    OfficialAnyDexBackend,
    _suppress_near_duplicate_candidates,
)
from anydex_pipeline.snapshot_bridge import make_snapshot
from anydex_pipeline.types import PointCloudObservation


class _Representation:
    def __init__(self, batch):
        self.batch = batch

    def infer(self, *_args, **_kwargs):
        return self.batch


class _Decision:
    def __init__(self, batch):
        self.batch = batch
        self.kwargs = None

    def infer(self, *_args, **kwargs):
        self.kwargs = kwargs
        return self.batch


def _rotation_z(degrees):
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def _candidate(
    *,
    score,
    source_index,
    grasp_type=4,
    q6=950.0,
    translation_m=0.0,
    rotation_deg=0.0,
    hand_translation_m=None,
    width_m=0.05,
    depth_m=0.02,
    source_commit="c" * 40,
):
    canonical = np.eye(4)
    canonical[:3, :3] = _rotation_z(rotation_deg)
    canonical[0, 3] = translation_m
    hand = canonical.copy()
    hand[0, 3] = (
        translation_m if hand_translation_m is None else hand_translation_m
    )
    return official_pipeline.GraspCandidate(
        T_reference_grasp=canonical,
        T_reference_hand=hand,
        hand_angles=np.asarray([700, 700, 700, 700, 800, q6]),
        score=score,
        width_m=width_m,
        depth_m=depth_m,
        collision_free=False,
        grasp_type_id=grasp_type,
        source_index=source_index,
        metadata={
            "representation_checkpoint_sha256": "a" * 64,
            "decision_checkpoint_sha256s": tuple(
                f"{index + 1:064x}" for index in range(8)
            ),
            "official_source_commit": source_commit,
            "pose_frame_before_calibration": "camera_color_optical_frame",
            "collision_checked": False,
            "collision_check": "not_run",
        },
    )


def test_near_duplicate_suppression_is_stable_and_keeps_highest_score_metadata():
    lower = _candidate(
        score=0.8,
        source_index=11,
        translation_m=0.0005,
        hand_translation_m=0.0005,
        rotation_deg=0.5,
        width_m=0.0505,
        depth_m=0.0205,
    )
    winner = _candidate(score=0.9, source_index=10)
    result = _suppress_near_duplicate_candidates((lower, winner))
    assert len(result) == 1
    assert result[0].source_index == 10
    assert result[0].score == 0.9
    assert result[0].metadata["near_duplicate_cluster_size"] == 2
    assert result[0].metadata["near_duplicate_source_indices"] == (10, 11)
    assert result[0].metadata["near_duplicate_decision_ranks"] == (1, 0)


def test_near_duplicate_suppression_never_crosses_type_target_or_provenance():
    base = _candidate(score=0.9, source_index=1)
    different_type = _candidate(score=0.8, source_index=2, grasp_type=5)
    different_target = _candidate(score=0.7, source_index=3, q6=949.0)
    different_source = _candidate(
        score=0.6, source_index=4, source_commit="d" * 40
    )
    result = _suppress_near_duplicate_candidates(
        (different_source, different_target, different_type, base)
    )
    assert [candidate.source_index for candidate in result] == [1, 2, 3, 4]


def test_near_duplicate_suppression_requires_both_hand_and_canonical_pose_near():
    base = _candidate(score=0.9, source_index=1)
    hand_far = _candidate(
        score=0.8, source_index=2, hand_translation_m=0.0011
    )
    canonical_far = _candidate(score=0.7, source_index=3, translation_m=0.0011)
    rotation_far = _candidate(score=0.6, source_index=4, rotation_deg=1.1)
    result = _suppress_near_duplicate_candidates(
        (base, hand_far, canonical_far, rotation_far)
    )
    assert len(result) == 4


def test_official_candidates_are_unchecked_and_carry_all_model_provenance(monkeypatch):
    grasp = np.zeros((1, 17), dtype=np.float64)
    grasp[0, 0:4] = (0.7, 0.05, 0.03, 0.02)
    grasp[0, 4:13] = np.eye(3).reshape(-1)
    grasp[0, 13:16] = (0.0, 0.0, 0.5)
    representation = RepresentationGraspBatch(
        grasps=grasp,
        features=np.zeros((1, OFFICIAL_NO_FLIP_FEATURE_WIDTH), dtype=np.float32),
        voxel_points_camera=np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32),
        feature_layout=(
            RepresentationFeatureLayout.OFFICIAL_RIGHT_HAND_NO_FLIP_V1
        ),
    )
    decision = InspireDecisionBatch(
        source_indices=np.asarray([0], dtype=np.int64),
        grasp_types=np.asarray([4], dtype=np.int32),
        depth_offsets_m=np.asarray([0.01], dtype=np.float32),
        scores=np.asarray([0.9], dtype=np.float32),
    )
    mapped = SimpleNamespace(
        angles=np.asarray([[700, 700, 700, 700, 800, 950]], dtype=np.float64),
        widths=np.asarray([0.05]),
        depths=np.asarray([0.03]),
        grasp_types=np.asarray([4]),
        pose_matrices=lambda apply_depth: np.eye(4, dtype=np.float64)[None],
    )
    monkeypatch.setattr(official_pipeline, "load_inspire_mapping", lambda _path: {})
    monkeypatch.setattr(
        official_pipeline,
        "map_two_finger_grasps",
        lambda *_args, **_kwargs: mapped,
    )

    backend = OfficialAnyDexBackend.__new__(OfficialAnyDexBackend)
    backend.representation = _Representation(representation)
    backend.decision = _Decision(decision)
    backend.representation_top_k = 10
    backend.top_k = 3
    backend.decision_score_threshold = None
    backend.mapping_path = None
    representation_hash = "a" * 64
    decision_hashes = tuple(f"{index + 1:064x}" for index in range(8))
    source_commit = "c" * 40
    backend._model_provenance = lambda: (
        representation_hash,
        decision_hashes,
        source_commit,
    )

    points = np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32)
    observation = PointCloudObservation(
        scene_points=points,
        object_points=points,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
    )
    result = backend.infer(observation)

    assert backend.decision.kwargs["top_k"] == 3 * 16
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.collision_free is False
    assert candidate.metadata["collision_checked"] is False
    assert candidate.metadata["decision_checkpoint_sha256s"] == decision_hashes
    snapshot = make_snapshot(observation, result)
    assert not snapshot.grasps.collision_checked[0]
    assert not snapshot.grasps.collision_free[0]
    assert snapshot.representation_checkpoint_sha256 == representation_hash
    assert snapshot.decision_checkpoint_sha256s == decision_hashes
    assert snapshot.official_source_commit == source_commit


def test_decision_checkpoint_discovery_requires_exactly_one_file_per_head(tmp_path):
    root = tmp_path / "480"
    for grasp_type in range(1, 9):
        directory = root / str(grasp_type)
        directory.mkdir(parents=True)
        (directory / f"type-{grasp_type}.pth").write_bytes(b"model")

    paths = official_pipeline._decision_checkpoint_paths(tmp_path)
    assert len(paths) == 8
    assert [path.parent.name for path in paths] == [str(value) for value in range(1, 9)]

    (root / "8" / "duplicate.pth").write_bytes(b"duplicate")
    try:
        official_pipeline._decision_checkpoint_paths(tmp_path)
    except ValueError as exc:
        assert "exactly one decision checkpoint" in str(exc)
    else:
        raise AssertionError("duplicate decision checkpoint was accepted")
