from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from anydex_pipeline.inspire_hand_model import InspireHandModel
from anydex_pipeline.inspire_open_configuration import (
    OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
)
from anydex_pipeline.execution_audit_gate import bind_installed_tool_audit
from anydex_pipeline.installed_tool_audit import (
    CAPTURED_POINT_CHECK_IDS,
    CHECK_SPECS,
    CollisionBackendIdentity,
    CollisionObservation,
    InstalledToolAuditRequest,
    RUNTIME_WORKSPACE_CLEAR_CONDITION_ID,
    V7_ADAPTER_SHA256,
    V7_T_EE_HAND,
    build_joint_path,
    check_specs_for_mode,
    load_installed_tool_audit,
    run_installed_tool_audit,
    validate_installed_tool_audit,
    write_installed_tool_audit,
)
from anydex_pipeline.snapshot import (
    GraspCandidates,
    VisualizationSnapshot,
    load_snapshot_npz,
    save_snapshot_npz,
)
from anydex_pipeline.rh56_hand_path import (
    build_rh56_no_contact_execution_path,
    loaded_hand_execution_path_not_applicable,
    rh56_feedback_envelope_policy,
)


ROOT = Path(__file__).resolve().parents[1]
ANYDEX_ROOT = ROOT / "third_party/AnyDexGrasp"
ADAPTER = ROOT / "assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl"


def _array_sha256(value):
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(
        ("dtype=<f8;shape={};".format(",".join(map(str, array.shape)))).encode(
            "ascii"
        )
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reseal_artifact(value):
    artifact = json.loads(json.dumps(value))
    artifact.pop("artifact_sha256", None)
    encoded = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    artifact["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    return artifact


class CompleteBackend:
    identity = CollisionBackendIdentity(
        name="fake-hppfcl",
        version="test-1",
        implementation_sha256="a" * 64,
        configuration_sha256="b" * 64,
    )

    def __init__(
        self, *, authoritative=True, omit=None, truncate=None, continuous=True,
        point_non_authoritative=False, distances=None,
    ):
        self.authoritative = authoritative
        self.omit = omit
        self.truncate = truncate
        self.continuous = continuous
        self.point_non_authoritative = point_non_authoritative
        self.distances = dict(distances or {})
        self.calls = 0
        self.last_query = None

    def evaluate(self, query):
        self.calls += 1
        self.last_query = query
        final = len(query.q_path_rad) - 1
        result = {}
        for spec in check_specs_for_mode(query.request.mode):
            if spec.check_id == self.omit:
                continue
            if spec.coverage == "full_path":
                indices = tuple(range(len(query.q_path_rad)))
            elif spec.coverage == "hand_execution_path":
                indices = tuple(range(len(query.T_hand_waypoint_link_visual)))
            else:
                indices = (final,)
            if spec.check_id == self.truncate:
                indices = indices[:-1]
            object_contact = spec.expectation == "object_contact"
            result[spec.check_id] = CollisionObservation(
                check_id=spec.check_id,
                authoritative=(
                    self.authoritative
                    and not (
                        self.point_non_authoritative
                        and spec.check_id in CAPTURED_POINT_CHECK_IDS
                    )
                ),
                tested_sample_indices=indices,
                minimum_signed_distance_m=float(
                    self.distances.get(
                        spec.check_id, 0.001 if object_contact else 0.020
                    )
                ),
                observed_pairs=("Link1 / object",) if object_contact else (),
                details={
                    "engine": "unit-test",
                    "max_q_tracking_error_rad": float(
                        query.request.max_q_tracking_error_rad
                    ),
                    "joint_tracking_uncertainty_applied": self.continuous,
                    "continuous_segment_envelope_verified": self.continuous,
                    "conservative_motion_bound_m": 0.001,
                    "minimum_distance_is_after_motion_bound": self.continuous,
                    "observed_scene_scope": query.request.observed_scene_scope,
                    "scene_voxel_resolution_m": float(
                        query.request.scene_voxel_resolution_m
                    ),
                    "unknown_space_policy": query.request.unknown_space_policy,
                    "unknown_space_policy_applied": (
                        self.continuous
                        and not (
                            self.point_non_authoritative
                            and spec.check_id in CAPTURED_POINT_CHECK_IDS
                        )
                    ),
                    "continuous_inter_waypoint_collision_claimed": (
                        self.continuous
                        if spec.coverage == "hand_execution_path"
                        else None
                    ),
                    "hand_interval_envelope_method": (
                        "official XLS every 1 register; accumulated absolute q12 "
                        "variation times configuration-independent URDF serial-chain "
                        "link radii; all-six arrival feedback tube; adaptive exact-mesh "
                        "q12 feedback-box subdivision for otherwise unresolved self-pairs; "
                        "final arm tracking tube"
                        if spec.coverage == "hand_execution_path"
                        else None
                    ),
                    "hand_arrival_tolerance_units": (
                        int(query.request.hand_arrival_tolerance_units)
                        if spec.coverage == "hand_execution_path"
                        else None
                    ),
                    "feedback_envelope_policy_sha256": (
                        rh56_feedback_envelope_policy(
                            query.request.hand_arrival_tolerance_units
                        )["sha256"]
                        if spec.coverage == "hand_execution_path"
                        else None
                    ),
                    "hand_self_clearance_margin_m": (
                        float(query.request.hand_self_clearance_margin_m)
                        if spec.margin_policy == "hand_self"
                        else None
                    ),
                    "all_nonadjacent_hand_link_pairs_checked": (
                        True if spec.margin_policy == "hand_self" else None
                    ),
                },
            )
        return result


class SplitCacheBackend(CompleteBackend):
    native_identity = CollisionBackendIdentity(
        name="native-test-hppfcl",
        version="1",
        implementation_sha256="c" * 64,
        configuration_sha256="d" * 64,
    )
    cache = {
        "path": "/tmp/test-static-cache.json",
        "file_sha256": "e" * 64,
        "payload_sha256": "f" * 64,
    }
    static_ids = tuple(
        spec.check_id
        for spec in check_specs_for_mode("air_grasp")
        if spec.check_id not in CAPTURED_POINT_CHECK_IDS
    )
    dynamic_ids = tuple(
        spec.check_id
        for spec in check_specs_for_mode("air_grasp")
        if spec.check_id in CAPTURED_POINT_CHECK_IDS
    )
    configuration = {
        "combiner_contract": "cached-static-8-plus-fresh-point-10-v1",
        "native_backend": {
            "name": native_identity.name,
            "version": native_identity.version,
            "implementation_sha256": native_identity.implementation_sha256,
            "configuration_sha256": native_identity.configuration_sha256,
        },
        "static_cache": cache,
        "static_check_ids": list(static_ids),
        "fresh_dynamic_check_ids": list(dynamic_ids),
    }
    identity = CollisionBackendIdentity(
        name="static-cache-plus-fresh-pinocchio-hppfcl-fr3-v7-rh56",
        version="1",
        implementation_sha256="1" * 64,
        configuration_sha256=hashlib.sha256(
            json.dumps(
                configuration,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    )

    def __init__(self):
        super().__init__(point_non_authoritative=True)

    def evaluate(self, query):
        raw = super().evaluate(query)
        result = {}
        for check_id, observation in raw.items():
            details = dict(observation.details or {})
            is_static = check_id in self.static_ids
            details["split_evaluation_provenance"] = {
                "phase": (
                    "precomputed_static_mesh" if is_static else "fresh_scene_object"
                ),
                "combiner_contract": "cached-static-8-plus-fresh-point-10-v1",
                "native_backend": dict(self.configuration["native_backend"]),
                "static_cache": dict(self.cache) if is_static else None,
            }
            result[check_id] = CollisionObservation(
                check_id=observation.check_id,
                authoritative=observation.authoritative,
                tested_sample_indices=observation.tested_sample_indices,
                minimum_signed_distance_m=observation.minimum_signed_distance_m,
                observed_pairs=observation.observed_pairs,
                details=details,
            )
        return result


@pytest.fixture(scope="module")
def hand_model():
    return InspireHandModel.from_anydex_root(ANYDEX_ROOT, mesh_resolution="simplified")


def make_request(tmp_path, hand_model, **updates):
    config = tmp_path / "control.json"
    config.write_text('{"fixture":true}\n', encoding="utf-8")
    snapshot = tmp_path / "official_snapshot.npz"
    snapshot.write_bytes(b"strict immutable snapshot fixture")
    scene = tmp_path / "fresh_scene.npz"
    scene.write_bytes(b"strict fresh scene fixture")
    object_source = tmp_path / "object.npz"
    object_source.write_bytes(b"strict object fixture")
    scene_points = np.asarray(
        [[0.5, 0.0, 0.1], [0.6, 0.1, 0.2]], dtype=np.float32
    )
    filter_evidence = tmp_path / "fresh_scene.npz.evidence.json"
    filter_evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "installed_scene_return_filter_evidence",
                "filtered_scene_points_sha256": _array_sha256(scene_points),
                "object_points_sha256": _array_sha256(
                    np.asarray(
                        [[0.55, 0, 0.15], [0.56, 0.01, 0.16]],
                        dtype=np.float32,
                    )
                ),
                "installed_return_filter": {
                    "method": (
                        "official FR3 visual shell/V7/open-RH56 triangle mesh "
                        "versus inflated point cube"
                    ),
                    "fr3_geometry_model": "official_fr3_visual_triangle_meshes",
                    "fr3_visual_meshes": [
                        {
                            "link": "link{}".format(index),
                            "path": str(
                                Path(
                                    "/opt/ros/humble/share/franka_description/meshes/"
                                    "robot_arms/fr3/visual/link{}.dae".format(index)
                                ).resolve()
                            ),
                            "sha256": _file_sha256(
                                "/opt/ros/humble/share/franka_description/meshes/"
                                "robot_arms/fr3/visual/link{}.dae".format(index)
                            ),
                        }
                        for index in range(8)
                    ],
                    "point_half_extent_m": 0.0025,
                    "inflation_margin_m": 0.002,
                    "candidate_mesh_point_tests": 10,
                    "reason_counts": {"FR3_visual_link2": 1},
                    "residual_self_return_filter": {
                        "enabled": True,
                        "failure": "",
                        "method": (
                            "strict HPP-FCL point-to-mesh neighborhood AND "
                            "same-component normalized-sRGB seed palette AND "
                            "bounded same-component 3-D geodesic"
                        ),
                        "distance_method": (
                            "HPP-FCL triangle-mesh to 1nm sphere, radius corrected"
                        ),
                        "maximum_model_surface_distance_m": 0.010,
                        "hard_maximum_model_surface_distance_m": 0.020,
                        "calibration_alignment_maximum_m": 0.008,
                        "calibration_guard_m": 0.002,
                        "calibration_bound_formula": (
                            "min(hard_maximum, object_to_live_maximum + "
                            "calibration_guard)"
                        ),
                        "connectivity_radius_m": 0.010,
                        "color_space": "normalized_sRGB_euclidean",
                        "palette_color_max_l2": 0.18,
                        "maximum_geodesic_m": 0.040,
                        "model_candidate_test_count": 2,
                        "model_candidate_indices_sha256": "a" * 64,
                        "model_candidate_surface_distances_sha256": "b" * 64,
                        "exact_seed_count": 1,
                        "model_neighborhood_candidate_count": 1,
                        "appearance_palette_candidate_count": 1,
                        "connected_candidate_count": 0,
                        "residual_removed_count": 0,
                        "residual_removed_indices_sha256": "c" * 64,
                        "residual_removed_surface_distance_min_m": None,
                        "residual_removed_surface_distance_max_m": None,
                        "reason_counts": {},
                    },
                },
                "object_alignment_and_filter": {
                    "method": "bidirectional_cKDTree_with_expanded_object_AABB",
                    "object_point_count": 2,
                    "live_scene_point_count": 2,
                    "object_to_live_median_m": 0.004,
                    "object_to_live_p95_m": 0.007,
                    "object_to_live_maximum_m": 0.008,
                    "alignment_coverage_distance_m": 0.015,
                    "alignment_coverage_fraction": 1.0,
                    "object_return_max_distance_m": 0.015,
                    "expanded_object_aabb_min_m": [0.535, -0.015, 0.135],
                    "expanded_object_aabb_max_m": [0.575, 0.025, 0.175],
                    "object_return_count_before_installed_precedence": 1,
                    "alignment_median_max_m": 0.008,
                    "alignment_p95_max_m": 0.015,
                    "alignment_minimum_coverage": 0.8,
                    "passed": True,
                    "failures": [],
                },
                "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "adapter_sha256": V7_ADAPTER_SHA256,
                "open_hand_fk_commissioned": True,
                "unknown_space_policy_applied": False,
                "authoritative_for_unseen_camera_space": False,
                "motion_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    loaded_grasp_eef = np.linalg.inv(V7_T_EE_HAND)
    loaded_pregrasp_eef = loaded_grasp_eef.copy()
    loaded_pregrasp_eef[0, 3] -= 0.10
    values = dict(
        mode="loaded_grasp",
        adapter_stl_path=ADAPTER,
        T_EE_hand=V7_T_EE_HAND,
        control_config_path=config,
        snapshot_source_path=snapshot,
        snapshot_schema_version=2,
        snapshot_reference_frame="robot_base",
        snapshot_frame_id=42,
        snapshot_timestamp_s=99.0,
        snapshot_calibration_id="test-calibration",
        snapshot_camera_serial="test-camera",
        snapshot_model_name="official AnyDexGrasp fixture",
        snapshot_representation_checkpoint_sha256="c" * 64,
        snapshot_decision_checkpoint_sha256s=tuple(
            "{:064x}".format(index + 1) for index in range(8)
        ),
        snapshot_official_source_commit="d" * 40,
        selected_candidate_index=3,
        selected_canonical_pose_base=np.eye(4),
        selected_hand_pose_base=np.eye(4),
        selected_hand_targets=np.asarray([600, 610, 620, 630, 700, 900]),
        plan_hand_pose_base=np.eye(4),
        plan_pregrasp_pose_base_EE=loaded_pregrasp_eef,
        plan_grasp_pose_base_EE=loaded_grasp_eef,
        air_retreat_distance_m=0.0,
        pregrasp_distance_m=0.10,
        scene_source_path=scene,
        scene_points_base=scene_points,
        scene_capture_q_rad=np.asarray([0, 0, 0, -1.57, 0, 1.57, 0]),
        scene_captured_at_s=99.0,
        current_q_rad=np.asarray([0.01, 0, 0, -1.57, 0, 1.57, 0]),
        current_q_captured_at_s=99.9,
        default_q_rad=np.asarray([0, 0, 0, -1.57, 0, 1.57, 0]),
        pregrasp_q_rad=np.asarray([0.10, 0.05, 0, -1.50, 0, 1.65, 0.02]),
        grasp_q_rad=np.asarray([0.12, 0.06, 0.01, -1.45, 0, 1.68, 0.03]),
        object_source_path=object_source,
        object_points_base=np.asarray(
            [[0.55, 0, 0.15], [0.56, 0.01, 0.16]], dtype=np.float32
        ),
        hand_model=hand_model,
        open_hand_joint_positions_rad=OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
        open_actuator_targets=np.full(6, 1000),
        closure_hand_joint_positions_rad=np.linspace(0.0, 0.5, 12),
        closure_actuator_targets=np.asarray([600, 610, 620, 630, 700, 900]),
        hand_execution_path=build_rh56_no_contact_execution_path(
            (600, 610, 620, 630, 700, 900)
        ).as_dict(),
        audit_started_at_s=100.0,
        max_joint_step_rad=0.02,
        scene_filter_evidence_path=filter_evidence,
    )
    values.update(updates)
    if values["mode"] == "loaded_grasp" and "hand_execution_path" not in updates:
        values["hand_execution_path"] = loaded_hand_execution_path_not_applicable()
    if values["mode"] == "air_grasp" and "joint_plan_source_path" not in updates:
        joint_plan = tmp_path / "installed_air_joint_plan.json"
        joint_plan.write_text('{"fixture":"air joint plan"}\n', encoding="utf-8")
        values["joint_plan_source_path"] = joint_plan
    if values["mode"] == "air_grasp" and "scene_clearance_margin_m" not in updates:
        values["scene_clearance_margin_m"] = 0.002
    return InstalledToolAuditRequest(**values)


def test_complete_audit_binds_every_input_and_round_trips(tmp_path, hand_model):
    request = make_request(tmp_path, hand_model)
    backend = CompleteBackend()
    artifact = run_installed_tool_audit(request, backend)

    assert artifact["decision"]["passed"] is True
    assert artifact["decision"]["motion_authorized"] is False
    assert artifact["bindings"]["adapter"]["sha256"] == V7_ADAPTER_SHA256
    assert artifact["bindings"]["hand_model"]["link_count"] == 13
    assert len(artifact["bindings"]["hand_model"]["links"]) == 13
    assert artifact["bindings"]["hand_model"]["closure_actuator_targets"] == [
        600, 610, 620, 630, 700, 900
    ]
    assert [item["check_id"] for item in artifact["checks"]] == [
        item.check_id for item in CHECK_SPECS
    ]
    assert backend.calls == 1
    assert backend.last_query.q_path_rad.shape[1:] == (7,)
    assert tuple(backend.last_query.hand_link_mesh_paths) == tuple(
        link.name for link in hand_model.links
    )

    output = write_installed_tool_audit(tmp_path / "audit.json", artifact)
    loaded = load_installed_tool_audit(output, verify_files=True, require_pass=True)
    assert loaded == artifact


def test_air_hand_self_uses_zero_intersection_margin_only(tmp_path, hand_model):
    retreat = 0.08
    planned_hand = np.eye(4)
    planned_hand[0, 3] = -retreat
    grasp_eef = planned_hand @ np.linalg.inv(V7_T_EE_HAND)
    pregrasp_eef = grasp_eef.copy()
    pregrasp_eef[0, 3] -= 0.01
    request = make_request(
        tmp_path,
        hand_model,
        mode="air_grasp",
        plan_hand_pose_base=planned_hand,
        plan_grasp_pose_base_EE=grasp_eef,
        plan_pregrasp_pose_base_EE=pregrasp_eef,
        air_retreat_distance_m=retreat,
        pregrasp_distance_m=0.01,
    )
    backend = CompleteBackend(
        distances={"rh56_execution_self_final": 0.001047}
    )
    artifact = run_installed_tool_audit(request, backend)

    assert artifact["policies"]["robot"] == 0.002
    assert artifact["policies"]["hand_self"] == 0.0
    self_check = next(
        item
        for item in artifact["checks"]
        if item["check_id"] == "rh56_execution_self_final"
    )
    assert self_check["passed"] is True
    assert self_check["minimum_signed_distance_m"] == pytest.approx(0.001047)

    robot_blocked = run_installed_tool_audit(
        request,
        CompleteBackend(distances={"rh56_execution_fr3_final": 0.001047}),
    )
    robot_check = next(
        item
        for item in robot_blocked["checks"]
        if item["check_id"] == "rh56_execution_fr3_final"
    )
    assert robot_check["passed"] is False


def test_joint_path_has_all_stages_and_respects_step():
    current = np.asarray([0, 0, 0, -1.57, 0, 1.57, 0.0])
    default = current.copy()
    pre = current + np.asarray([0.10, 0, 0, 0.02, 0, 0, 0])
    grasp = pre + np.asarray([0.01, 0.03, 0, 0, 0, 0, 0])
    approach_transit = default + np.asarray([0.02, 0.01, 0, 0, 0, 0, 0])
    path, segments = build_joint_path(
        current,
        default,
        pre,
        grasp,
        max_joint_step_rad=0.02,
        default_transit_q_rad=(
            current + np.asarray([0.04, 0, 0, 0, 0, 0, 0.8]),
            current + np.asarray([0.04, 0, 0, 0, 0, 0, 0.0]),
        ),
        approach_transit_q_rad=(approach_transit,),
    )
    assert np.array_equal(path[segments[0]["start_index"]], current)
    assert len(segments) == 6
    assert np.array_equal(path[segments[2]["end_index"]], default)
    assert segments[3]["name"] == "default_to_approach_transit_0"
    assert np.array_equal(path[segments[3]["end_index"]], approach_transit)
    assert np.array_equal(path[segments[4]["end_index"]], pre)
    assert np.array_equal(path[segments[5]["end_index"]], grasp)
    assert np.max(np.abs(np.diff(path, axis=0))) <= 0.02 + 1e-12


@pytest.mark.parametrize("mode", ["missing", "truncated", "non_authoritative"])
def test_incomplete_or_non_authoritative_backend_fails_closed(
    tmp_path, hand_model, mode
):
    check_id = "rh56_open_scene_path"
    backend = CompleteBackend(
        omit=check_id if mode == "missing" else None,
        truncate=check_id if mode == "truncated" else None,
        authoritative=mode != "non_authoritative",
    )
    artifact = run_installed_tool_audit(make_request(tmp_path, hand_model), backend)
    assert artifact["decision"]["passed"] is False
    assert artifact["decision"]["motion_authorized"] is False
    with pytest.raises(ValueError, match="did not pass"):
        validate_installed_tool_audit(artifact, require_pass=True)


def test_nominal_discrete_checks_without_continuous_tracking_tube_fail_closed(
    tmp_path, hand_model
):
    artifact = run_installed_tool_audit(
        make_request(tmp_path, hand_model), CompleteBackend(continuous=False)
    )
    assert artifact["decision"]["passed"] is False
    reasons = " ".join(artifact["decision"]["reasons"])
    assert "tracking uncertainty" in reasons
    assert "continuous joint-path envelope" in reasons
    with pytest.raises(ValueError, match="did not pass"):
        validate_installed_tool_audit(artifact, require_pass=True)


def test_wrong_adapter_and_stale_scene_skip_backend(tmp_path, hand_model):
    wrong_adapter = tmp_path / "adapter.stl"
    wrong_adapter.write_bytes(b"not the commissioned V7")
    backend = CompleteBackend()
    request = make_request(
        tmp_path,
        hand_model,
        adapter_stl_path=wrong_adapter,
        scene_captured_at_s=90.0,
    )
    artifact = run_installed_tool_audit(request, backend)
    assert backend.calls == 0
    assert artifact["decision"]["passed"] is False
    failures = " ".join(artifact["decision"]["precondition_failures"])
    assert "not the commissioned V7" in failures
    assert "freshness" in failures


def test_air_grasp_has_distinct_retreat_and_no_contact_check(tmp_path, hand_model):
    retreat = 0.15
    planned_hand = np.eye(4)
    planned_hand[0, 3] = -retreat
    grasp_eef = planned_hand @ np.linalg.inv(V7_T_EE_HAND)
    pregrasp_eef = grasp_eef.copy()
    pregrasp_eef[0, 3] -= 0.10
    request = make_request(
        tmp_path,
        hand_model,
        mode="air_grasp",
        plan_hand_pose_base=planned_hand,
        plan_grasp_pose_base_EE=grasp_eef,
        plan_pregrasp_pose_base_EE=pregrasp_eef,
        air_retreat_distance_m=retreat,
    )
    artifact = run_installed_tool_audit(request, CompleteBackend())

    assert artifact["mode"] == "air_grasp"
    assert artifact["decision"]["passed"] is True
    assert artifact["bindings"]["execution_plan"]["contact_and_lift_forbidden"] is True
    closed_object_check = next(
        item
        for item in artifact["checks"]
        if item["check_id"] == "rh56_closed_object_all_links_final"
    )
    assert closed_object_check["expectation"] == "clear"
    assert not any(item["expectation"] == "object_contact" for item in artifact["checks"])


def test_air_point_cloud_limit_is_explicit_runtime_condition(tmp_path, hand_model):
    retreat = 0.08
    pregrasp_distance = 0.01
    planned_hand = np.eye(4)
    planned_hand[0, 3] = -retreat
    grasp_eef = planned_hand @ np.linalg.inv(V7_T_EE_HAND)
    pregrasp_eef = grasp_eef.copy()
    pregrasp_eef[0, 3] -= pregrasp_distance
    request = make_request(
        tmp_path,
        hand_model,
        mode="air_grasp",
        plan_hand_pose_base=planned_hand,
        plan_grasp_pose_base_EE=grasp_eef,
        plan_pregrasp_pose_base_EE=pregrasp_eef,
        air_retreat_distance_m=retreat,
        pregrasp_distance_m=pregrasp_distance,
    )
    artifact = run_installed_tool_audit(
        request, CompleteBackend(point_non_authoritative=True)
    )

    assert artifact["decision"]["passed"] is True
    assert artifact["decision"]["pass_kind"] == "conditional_air"
    assert artifact["decision"]["motion_authorized"] is False
    assert artifact["decision"]["runtime_operator_workspace_clear_required"] is True
    for check in artifact["checks"]:
        if check["check_id"] in CAPTURED_POINT_CHECK_IDS:
            assert check["authoritative"] is False
            assert check["runtime_condition_ids"] == [
                RUNTIME_WORKSPACE_CLEAR_CONDITION_ID
            ]
            assert check["passed"] is True
        else:
            assert check["authoritative"] is True


def test_split_backend_provenance_is_part_of_the_strict_schema(tmp_path, hand_model):
    retreat = 0.08
    planned_hand = np.eye(4)
    planned_hand[0, 3] = -retreat
    grasp_eef = planned_hand @ np.linalg.inv(V7_T_EE_HAND)
    pregrasp_eef = grasp_eef.copy()
    pregrasp_eef[0, 3] -= 0.01
    request = make_request(
        tmp_path,
        hand_model,
        mode="air_grasp",
        plan_hand_pose_base=planned_hand,
        plan_grasp_pose_base_EE=grasp_eef,
        plan_pregrasp_pose_base_EE=pregrasp_eef,
        air_retreat_distance_m=retreat,
        pregrasp_distance_m=0.01,
    )
    artifact = run_installed_tool_audit(request, SplitCacheBackend())
    assert artifact["decision"]["passed"] is True
    assert artifact["collision_backend"]["name"].startswith(
        "static-cache-plus-fresh-"
    )

    tampered = json.loads(json.dumps(artifact))
    static_check = next(
        item
        for item in tampered["checks"]
        if item["check_id"] == "fr3_self_path"
    )
    static_check["details"]["split_evaluation_provenance"]["phase"] = (
        "fresh_scene_object"
    )
    tampered = _reseal_artifact(tampered)
    with pytest.raises(ValueError, match="wrong split phase"):
        validate_installed_tool_audit(tampered)


def test_loaded_point_cloud_limit_never_becomes_conditional_pass(tmp_path, hand_model):
    artifact = run_installed_tool_audit(
        make_request(tmp_path, hand_model),
        CompleteBackend(point_non_authoritative=True),
    )
    assert artifact["decision"]["passed"] is False
    assert artifact["decision"]["pass_kind"] == "failed"
    assert artifact["decision"]["runtime_operator_workspace_clear_required"] is False


@pytest.mark.parametrize(
    "tamper",
    [
        "object_hash",
        "alignment_passed",
        "alignment_p95",
        "residual_hard_cap",
        "residual_candidate_hash",
    ],
)
def test_air_filter_object_alignment_tamper_fails_before_backend(
    tmp_path, hand_model, tamper
):
    retreat = 0.08
    pregrasp_distance = 0.01
    planned_hand = np.eye(4)
    planned_hand[0, 3] = -retreat
    grasp_eef = planned_hand @ np.linalg.inv(V7_T_EE_HAND)
    pregrasp_eef = grasp_eef.copy()
    pregrasp_eef[0, 3] -= pregrasp_distance
    request = make_request(
        tmp_path,
        hand_model,
        mode="air_grasp",
        plan_hand_pose_base=planned_hand,
        plan_grasp_pose_base_EE=grasp_eef,
        plan_pregrasp_pose_base_EE=pregrasp_eef,
        air_retreat_distance_m=retreat,
        pregrasp_distance_m=pregrasp_distance,
    )
    evidence_path = request.scene_filter_evidence_path
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if tamper == "object_hash":
        payload["object_points_sha256"] = "0" * 64
    elif tamper == "alignment_passed":
        payload["object_alignment_and_filter"]["passed"] = False
        payload["object_alignment_and_filter"]["failures"] = ["moved"]
    elif tamper == "alignment_p95":
        payload["object_alignment_and_filter"]["object_to_live_p95_m"] = 0.016
    elif tamper == "residual_hard_cap":
        payload["installed_return_filter"]["residual_self_return_filter"][
            "hard_maximum_model_surface_distance_m"
        ] = 0.021
    else:
        payload["installed_return_filter"]["residual_self_return_filter"][
            "model_candidate_indices_sha256"
        ] = "0" * 63 + "z"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    backend = CompleteBackend(point_non_authoritative=True)
    artifact = run_installed_tool_audit(request, backend)
    assert backend.calls == 0
    assert artifact["decision"]["passed"] is False
    failures = " ".join(artifact["decision"]["precondition_failures"])
    if tamper.startswith("residual_"):
        assert "installed-scene FR3 visual filter policy is invalid" in failures
    else:
        assert "installed-scene filter evidence policy/provenance is invalid" in failures


def test_loaded_and_air_pose_contracts_cannot_be_relabelled(tmp_path, hand_model):
    artifact = run_installed_tool_audit(
        make_request(tmp_path, hand_model), CompleteBackend()
    )
    tampered = json.loads(json.dumps(artifact))
    tampered["mode"] = "air_grasp"
    with pytest.raises(ValueError, match="artifact_sha256"):
        validate_installed_tool_audit(tampered)


def test_strict_schema_digest_and_bound_file_revalidation(tmp_path, hand_model):
    artifact = run_installed_tool_audit(
        make_request(tmp_path, hand_model), CompleteBackend()
    )
    tampered = json.loads(json.dumps(artifact))
    tampered["unexpected"] = True
    with pytest.raises(ValueError, match="keys differ"):
        validate_installed_tool_audit(tampered)

    tampered = json.loads(json.dumps(artifact))
    tampered["bindings"]["joint_path"]["samples_rad"][0][0] += 0.1
    with pytest.raises(ValueError, match="artifact_sha256"):
        validate_installed_tool_audit(tampered)

    output = write_installed_tool_audit(tmp_path / "audit.json", artifact)
    Path(artifact["bindings"]["scene"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="scene source no longer matches"):
        load_installed_tool_audit(output, verify_files=True)


@pytest.mark.parametrize(
    "tamper,expected_message",
    [
        ("arrival_tolerance", "dense hand interval mapping binding"),
        ("policy_hash", "dense hand interval mapping binding"),
        ("feedback_value", "dense hand feedback tube binding"),
        ("feedback_hash", "dense hand feedback tube hash"),
        ("q12_envelope_hash", "dense hand feedback q12 envelope hash"),
        ("missing_feedback_hash", "keys differ"),
    ],
)
def test_air_dense_hand_feedback_tube_binding_rejects_tamper(
    tmp_path, hand_model, tamper, expected_message
):
    retreat = 0.08
    planned_hand = np.eye(4)
    planned_hand[0, 3] = -retreat
    grasp_eef = planned_hand @ np.linalg.inv(V7_T_EE_HAND)
    pregrasp_eef = grasp_eef.copy()
    pregrasp_eef[0, 3] -= 0.01
    artifact = run_installed_tool_audit(
        make_request(
            tmp_path,
            hand_model,
            mode="air_grasp",
            plan_hand_pose_base=planned_hand,
            plan_grasp_pose_base_EE=grasp_eef,
            plan_pregrasp_pose_base_EE=pregrasp_eef,
            air_retreat_distance_m=retreat,
            pregrasp_distance_m=0.01,
        ),
        CompleteBackend(),
    )
    assert artifact["decision"]["passed"] is True
    mapping = artifact["bindings"]["hand_model"][
        "execution_dense_interval_mapping"
    ]
    if tamper == "arrival_tolerance":
        mapping["arrival_tolerance_units"] += 1
    elif tamper == "policy_hash":
        mapping["feedback_envelope_policy_sha256"] = "0" * 64
    elif tamper == "feedback_value":
        mapping["feedback_tube_q12_rad"][0][0] += 1.0e-6
    elif tamper == "feedback_hash":
        mapping["feedback_tubes_sha256"] = "0" * 64
    elif tamper == "q12_envelope_hash":
        mapping["feedback_q12_envelopes_sha256"] = "0" * 64
    else:
        del mapping["feedback_tubes_sha256"]

    with pytest.raises(ValueError, match=expected_message):
        validate_installed_tool_audit(_reseal_artifact(artifact))


def test_loaded_dense_hand_feedback_tube_fields_are_strictly_not_applicable(
    tmp_path, hand_model
):
    artifact = run_installed_tool_audit(
        make_request(tmp_path, hand_model), CompleteBackend()
    )
    mapping = artifact["bindings"]["hand_model"][
        "execution_dense_interval_mapping"
    ]
    assert mapping == {
        "algorithm": "not_applicable_loaded_grasp_v1",
        "arrival_tolerance_units": None,
        "feedback_envelope_policy_sha256": "",
        "interval_sample_counts": [],
        "q12_intervals_sha256": "",
        "feedback_tube_q12_rad": [],
        "feedback_tubes_sha256": "",
        "feedback_q12_envelopes_sha256": "",
    }
    mapping["arrival_tolerance_units"] = 0
    with pytest.raises(ValueError, match="loaded audit dense hand interval"):
        validate_installed_tool_audit(_reseal_artifact(artifact))


def test_feedback_envelope_policy_is_canonical_and_resealed_tamper_fails(
    tmp_path, hand_model
):
    artifact = run_installed_tool_audit(
        make_request(tmp_path, hand_model), CompleteBackend()
    )
    assert artifact["policies"]["hand_feedback_envelope"] == (
        rh56_feedback_envelope_policy(20)
    )
    artifact["policies"]["hand_feedback_envelope"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="feedback-envelope policy/hash"):
        validate_installed_tool_audit(_reseal_artifact(artifact))


def test_execution_sidecar_overlay_binds_immutable_selected_candidate(tmp_path):
    full_model = InspireHandModel.from_anydex_root(
        ANYDEX_ROOT, mesh_resolution="full"
    )
    request = make_request(tmp_path, full_model)
    count = 4
    canonical = np.repeat(np.eye(4)[None, :, :], count, axis=0)
    canonical[:, 0, 3] = np.linspace(0.0, 0.03, count)
    hand_poses = canonical.copy()
    canonical[3] = request.selected_canonical_pose_base
    hand_poses[3] = request.selected_hand_pose_base
    hand_angles = np.tile(np.asarray([800, 800, 800, 800, 800, 950]), (count, 1))
    hand_angles[3] = request.selected_hand_targets
    snapshot = VisualizationSnapshot(
        scene_points=request.scene_points_base.astype(np.float32),
        scene_colors=np.zeros((2, 3), dtype=np.float32),
        object_points=request.object_points_base.astype(np.float32),
        object_colors=np.ones((2, 3), dtype=np.float32),
        grasps=GraspCandidates(
            canonical_poses=canonical,
            scores=np.asarray([0.1, 0.2, 0.3, 0.9], dtype=np.float32),
            selected_index=3,
            hand_poses=hand_poses,
            hand_angles=hand_angles,
        ),
        reference_frame=request.snapshot_reference_frame,
        T_reference_camera=np.eye(4),
        frame_id=request.snapshot_frame_id,
        timestamp_s=request.snapshot_timestamp_s,
        calibration_id=request.snapshot_calibration_id,
        camera_serial=request.snapshot_camera_serial,
        model_name=request.snapshot_model_name,
        representation_checkpoint_sha256=request.snapshot_representation_checkpoint_sha256,
        decision_checkpoint_sha256s=request.snapshot_decision_checkpoint_sha256s,
        official_source_commit=request.snapshot_official_source_commit,
    )
    save_snapshot_npz(request.snapshot_source_path, snapshot)
    loaded_snapshot = load_snapshot_npz(request.snapshot_source_path)
    np.savez_compressed(
        request.scene_source_path,
        filtered_scene_points=np.asarray(request.scene_points_base, dtype=np.float32),
        scene_excludes_object=np.asarray(True, dtype=np.bool_),
        reference_frame=np.asarray("robot_base"),
        capture_q_rad=np.asarray(request.scene_capture_q_rad, dtype=np.float64),
        captured_at_unix_s=np.asarray(request.scene_captured_at_s, dtype=np.float64),
        calibration_id=np.asarray(request.snapshot_calibration_id),
        camera_serial=np.asarray(request.snapshot_camera_serial),
    )
    artifact = run_installed_tool_audit(request, CompleteBackend())
    audit_path = write_installed_tool_audit(tmp_path / "bound_audit.json", artifact)
    config = {
        "tool": {
            "adapter_asset": str(ADAPTER),
            "T_EE_hand": V7_T_EE_HAND.tolist(),
        },
        "franka": {"default_q_rad": request.default_q_rad.tolist()},
        "grasp": {
            "pregrasp_distance_m": 0.10,
            "air_retreat_distance_m": 0.08,
            "air_pregrasp_distance_m": 0.01,
            "air_audit_observed_scene_margin_m": 0.002,
            "air_audit_rh56_self_clearance_margin_m": 0.0,
        },
    }
    plan = SimpleNamespace(
        T_reference_EE_pregrasp=request.plan_pregrasp_pose_base_EE,
        T_reference_EE_grasp=request.plan_grasp_pose_base_EE,
    )

    binding = bind_installed_tool_audit(
        audit_path,
        config=config,
        config_path=request.control_config_path,
        snapshot=loaded_snapshot,
        snapshot_path=request.snapshot_source_path,
        plan=plan,
        workspace_root=ROOT.parent,
        now_s=100.0,
    )
    assert binding.passed
    assert artifact["decision"]["motion_authorized"] is False
    assert not bool(loaded_snapshot.grasps.collision_checked[3])
    assert bool(binding.evidence_snapshot.grasps.collision_checked[3])
    assert bool(binding.evidence_snapshot.grasps.collision_free[3])

    later_offline_replay = bind_installed_tool_audit(
        audit_path,
        config=config,
        config_path=request.control_config_path,
        snapshot=loaded_snapshot,
        snapshot_path=request.snapshot_source_path,
        plan=plan,
        workspace_root=ROOT.parent,
        now_s=103.0,
    )
    assert later_offline_replay.passed
