from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from anydex_pipeline.installed_tool_audit import (
    CollisionBackendIdentity,
    CollisionObservation,
)
from anydex_pipeline.installed_tool_static_cache import (
    DYNAMIC_AIR_CHECK_IDS,
    STATIC_AIR_CHECK_IDS,
    StaticCacheCombinedBackend,
    build_static_cache_artifact,
    load_static_cache_artifact,
    write_static_cache_artifact,
)


def _file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(("fixed-" + name).encode("ascii"))
    return path


def _query(tmp_path: Path):
    adapter = _file(tmp_path, "adapter.stl")
    config = _file(tmp_path, "config.json")
    snapshot = _file(tmp_path, "snapshot.npz")
    plan = _file(tmp_path, "plan.json")
    urdf = _file(tmp_path, "hand.urdf")
    mapping = _file(tmp_path, "mapping.json")
    _file(tmp_path, "driver_routine_to_angle.xls")
    meshes = {
        "Link{:02d}".format(index): _file(
            tmp_path, "link{:02d}.stl".format(index)
        )
        for index in range(13)
    }
    transforms = {name: np.eye(4) for name in meshes}
    model = SimpleNamespace(
        urdf_path=urdf,
        mapping_path=mapping,
        mesh_resolution="full",
        joints=(
            SimpleNamespace(name="j0", parent="Link00", child="Link01"),
        ),
    )
    request = SimpleNamespace(
        mode="air_grasp",
        hand_model=model,
        joint_plan_source_path=plan,
        adapter_stl_path=adapter,
        control_config_path=config,
        snapshot_source_path=snapshot,
        selected_candidate_index=51,
        selected_hand_targets=np.asarray([0, 1, 2, 3, 4, 900]),
        selected_canonical_pose_base=np.eye(4),
        selected_hand_pose_base=np.eye(4),
        T_EE_hand=np.eye(4),
        max_q_tracking_error_rad=0.002,
        hand_self_clearance_margin_m=0.0,
        hand_arrival_tolerance_units=20,
    )
    return SimpleNamespace(
        request=request,
        q_path_rad=np.zeros((3, 7)),
        path_segments=({"name": "one", "start_index": 0, "end_index": 2},),
        adapter_sha256="a" * 64,
        hand_link_mesh_paths=meshes,
        T_hand_open_link_visual=transforms,
        T_hand_closed_link_visual=transforms,
        T_hand_waypoint_link_visual=(transforms, transforms),
        hand_dense_interval_q12_rad=(np.zeros((2, 12)),),
        hand_interval_feedback_tube_q12_rad=(np.zeros(12),),
        hand_interval_feedback_q12_lower_rad=(np.zeros(12),),
        hand_interval_feedback_q12_upper_rad=(np.zeros(12),),
    )


def _observation(check_id: str) -> CollisionObservation:
    coverage = (
        (0, 1)
        if check_id.startswith("rh56_execution_")
        else ((2,) if "final" in check_id else (0, 1, 2))
    )
    return CollisionObservation(
        check_id=check_id,
        authoritative=check_id in STATIC_AIR_CHECK_IDS,
        tested_sample_indices=coverage,
        minimum_signed_distance_m=0.1,
        details={"source": "fake deterministic geometry"},
    )


class _Backend:
    identity = CollisionBackendIdentity(
        name="native-test",
        version="1",
        implementation_sha256="1" * 64,
        configuration_sha256="2" * 64,
    )

    def evaluate_checks(self, query, check_ids):
        return {check_id: _observation(check_id) for check_id in check_ids}


def test_static_cache_is_no_replace_read_only_and_exactly_bound(tmp_path):
    query = _query(tmp_path)
    backend = _Backend()
    artifact = build_static_cache_artifact(query, backend)
    destination = tmp_path / "cache.json"
    write_static_cache_artifact(destination, artifact)
    assert destination.stat().st_mode & 0o777 == 0o444
    assert tuple(load_static_cache_artifact(destination, query, backend.identity)) == (
        STATIC_AIR_CHECK_IDS
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_static_cache_artifact(destination, artifact)

    query.q_path_rad[0, 0] = 0.001
    with pytest.raises(ValueError, match="exact current query"):
        load_static_cache_artifact(destination, query, backend.identity)


def test_combiner_identity_and_each_check_disclose_its_phase(tmp_path):
    query = _query(tmp_path)
    backend = _Backend()
    destination = tmp_path / "cache.json"
    write_static_cache_artifact(
        destination, build_static_cache_artifact(query, backend)
    )
    combined = StaticCacheCombinedBackend(backend, destination)
    assert combined.identity.name.startswith("static-cache-plus-fresh-")
    assert combined.identity != backend.identity
    observations = combined.evaluate(query)
    assert set(observations) == set(STATIC_AIR_CHECK_IDS + DYNAMIC_AIR_CHECK_IDS)
    for check_id in STATIC_AIR_CHECK_IDS:
        provenance = observations[check_id].details["split_evaluation_provenance"]
        assert provenance["phase"] == "precomputed_static_mesh"
        assert provenance["static_cache"]["path"] == str(destination.resolve())
        assert len(provenance["static_cache"]["file_sha256"]) == 64
        assert len(provenance["static_cache"]["payload_sha256"]) == 64
    for check_id in DYNAMIC_AIR_CHECK_IDS:
        provenance = observations[check_id].details["split_evaluation_provenance"]
        assert provenance["phase"] == "fresh_scene_object"
        assert provenance["static_cache"] is None
