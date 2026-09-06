from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from anydex_pipeline.pregrasp_only_audit import validate_pregrasp_only_audit

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "apps/generate_pregrasp_only_audit.py"
REAL_AUDIT = (
    ROOT
    / "runs/candidate51_pregrasp_only_audit_current_fr3_limits_20260726.json"
)


def _load_app():
    spec = importlib.util.spec_from_file_location(
        "test_generate_pregrasp_only_audit_app", APP_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app = _load_app()


def test_generator_cli_defaults_bind_cable_fixed_prefix_inputs():
    args = app.build_parser().parse_args(
        ["generate", "--output", "/tmp/pregrasp-prefix.json"]
    )
    assert args.command == "generate"
    assert args.candidate_index == 51
    assert args.filtered_scene.name == (
        "live_scene_installed_filtered_candidate51_cable_fixed_20260721.npz"
    )
    assert args.joint_plan.name == (
        "candidate51_installed_air_joint_plan_current_fr3_limits_20260726.json"
    )
    assert args.max_q_tracking_error_rad == 0.002
    assert args.robot_clearance_margin_m == 0.002
    assert args.object_clearance_margin_m == 0.002


def test_generator_treats_scene_capture_q_difference_as_advisory_provenance():
    capture_q = np.asarray([0.0, 0.1, 0.2, -1.7, 0.0, 1.6, 0.8])
    prefix_q = capture_q.copy()
    prefix_q[5] += 0.0062866

    assert app._advisory_scene_prefix_q_delta(
        capture_q, prefix_q
    ) == pytest.approx(0.0062866)


@pytest.mark.parametrize(
    "capture,current",
    [
        ([0.0] * 6, [0.0] * 7),
        ([0.0] * 7, [0.0] * 6),
        ([0.0] * 6 + [float("nan")], [0.0] * 7),
    ],
)
def test_generator_rejects_malformed_scene_or_prefix_q(capture, current):
    with pytest.raises(ValueError, match="finite 7-vectors"):
        app._advisory_scene_prefix_q_delta(capture, current)


class _FakeBackend:
    def __init__(self):
        self.config = SimpleNamespace(adapter_mesh_scale=0.001)
        self.pairs = []

    def _load_mesh(self, path, scale=None):
        return (str(path), scale)

    def _robot_state(self, q):
        transform = np.eye(4)
        transform[0, 3] = float(q[0])
        return {}, transform

    def _object(self, geometry, transform):
        return geometry, np.asarray(transform)

    def _hand_objects(self, geometries, transforms, T_base_hand):
        return {
            name: (geometry, np.asarray(T_base_hand) @ transforms[name])
            for name, geometry in geometries.items()
        }

    def _mesh_pair_distance(self, adapter, hand, *, pair, sample_index):
        self.pairs.append((pair, int(sample_index)))
        name = pair.split(" / ", 1)[1]
        distance = 0.040 if name == "Link5" else 0.100
        return SimpleNamespace(
            distance_m=distance,
            pair=pair,
            sample_index=int(sample_index),
            nearest_point_a=(0.0, 0.0, 0.0),
            nearest_point_b=(distance, 0.0, 0.0),
            intersecting=False,
            penetration_depth_trustworthy=True,
        )


def test_open_hand_adapter_check_is_full_path_authoritative_and_excludes_only_mount():
    names = (
        "Link1",
        "Link11",
        "Link111",
        "Link2",
        "Link22",
        "Link3",
        "Link33",
        "Link4",
        "Link44",
        "Link5",
        "Link51",
        "Link52",
        "Link53",
    )
    backend = _FakeBackend()
    q_path = np.zeros((3, 7), dtype=np.float64)
    q_path[:, 0] = [0.0, 0.1, 0.2]
    result = app._open_hand_adapter_check(
        backend=backend,
        q_path=q_path,
        adapter_path=Path("/tmp/adapter.stl"),
        T_EE_hand=np.eye(4),
        hand_mesh_paths={name: Path("/tmp/{}.stl".format(name)) for name in names},
        open_transforms={name: np.eye(4) for name in names},
        max_q_tracking_error_rad=0.002,
    )
    assert result["check_id"] == "rh56_open_adapter_path"
    assert result["authoritative"] is True
    assert result["coverage"] == "full_path"
    assert result["tested_sample_indices"] == [0, 1, 2]
    assert result["expected_sample_count"] == 3
    assert result["minimum_signed_distance_m"] == 0.040
    assert result["observed_pairs"] == []
    assert len(backend.pairs) == 3 * 12
    assert all("Link111" not in pair for pair, _ in backend.pairs)
    details = result["details"]
    assert details["fixed_mount_pair_exclusions"] == ["Link111"]
    assert details["checked_non_mount_link_count"] == 12
    assert details["rigid_relative_transform_invariant"] is True
    assert details["continuous_segment_envelope_verified"] is True
    assert details["joint_tracking_uncertainty_applied"] is True
    assert details["conservative_motion_bound_m"] == 0.0


def test_verify_subcommand_exposes_file_and_pass_replay_gates():
    args = app.build_parser().parse_args(
        [
            "verify",
            "--artifact",
            "/tmp/prefix.json",
            "--verify-files",
            "--require-pass",
        ]
    )
    assert args.command == "verify"
    assert args.verify_files is True
    assert args.require_pass is True


def _canonical_json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reseal(artifact):
    result = copy.deepcopy(artifact)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _canonical_json_sha256(result)
    return result


def _real_artifact():
    return json.loads(REAL_AUDIT.read_text(encoding="utf-8"))


@pytest.mark.skipif(not REAL_AUDIT.is_file(), reason="bound real prefix audit missing")
def test_real_prefix_artifact_replays_all_files_and_passes():
    artifact = validate_pregrasp_only_audit(
        _real_artifact(), verify_files=True, require_pass=True
    )
    checks = {item["check_id"]: item for item in artifact["checks"]}
    assert artifact["decision"]["passed"] is True
    assert artifact["motion_authorized"] is False
    assert checks["rh56_open_adapter_path"]["authoritative"] is True
    assert checks["rh56_open_adapter_path"]["minimum_signed_distance_m"] > 0.040
    assert checks["rh56_open_adapter_path"]["details"][
        "fixed_mount_pair_exclusions"
    ] == ["Link111"]
    assert checks["rh56_open_scene_path"]["advisory_only"] is True
    assert checks["rh56_open_scene_path"]["minimum_signed_distance_m"] < 0.0
    assert checks["adapter_object_path"]["required_for_decision"] is True
    assert checks["rh56_open_object_path"]["required_for_decision"] is True
    scene_binding = artifact["bindings"]["filtered_scene"]
    assert scene_binding["age_at_artifact_creation_s"] > 1900.0
    assert scene_binding["configured_recommended_max_scene_age_s"] == 120.0
    assert scene_binding["age_exceeded_recommended_max"] is True
    assert scene_binding["freshness_role"] == "advisory_provenance_only"
    assert scene_binding["freshness_warning"]


@pytest.mark.skipif(not REAL_AUDIT.is_file(), reason="bound real prefix audit missing")
@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "fixed_mount_pair_exclusions",
            ["Link111", "Link5"],
            "required_geometry_passed",
        ),
        ("checked_non_mount_link_count", 11, "required_geometry_passed"),
        (
            "rigid_relative_transform_invariant",
            False,
            "required_geometry_passed",
        ),
        ("geometry_model", "convex approximation", "required_geometry_passed"),
        ("conservative_motion_bound_m", 0.001, "required_geometry_passed"),
        ("evaluated_pair_count", 12, "required_geometry_passed"),
    ],
)
def test_open_adapter_validator_rejects_resealed_scope_or_tracking_tamper(
    field, value, match
):
    artifact = _real_artifact()
    check = next(
        item for item in artifact["checks"] if item["check_id"] == "rh56_open_adapter_path"
    )
    check["details"][field] = value
    with pytest.raises(ValueError, match=match):
        validate_pregrasp_only_audit(_reseal(artifact), require_pass=True)


@pytest.mark.skipif(not REAL_AUDIT.is_file(), reason="bound real prefix audit missing")
def test_scene_collision_remains_advisory_but_required_object_clearance_cannot_be_forged():
    artifact = _real_artifact()
    scene = next(
        item for item in artifact["checks"] if item["check_id"] == "rh56_open_scene_path"
    )
    scene["minimum_signed_distance_m"] = -10.0
    scene["observed_pairs"] = ["Link111 / scene", "Link5 / scene"]
    validate_pregrasp_only_audit(_reseal(artifact), require_pass=True)

    object_tamper = _real_artifact()
    object_check = next(
        item
        for item in object_tamper["checks"]
        if item["check_id"] == "rh56_open_object_path"
    )
    object_check["minimum_signed_distance_m"] = 0.0
    with pytest.raises(ValueError, match="required_geometry_passed"):
        validate_pregrasp_only_audit(_reseal(object_tamper), require_pass=True)


@pytest.mark.skipif(not REAL_AUDIT.is_file(), reason="bound real prefix audit missing")
def test_bound_object_authority_basis_is_hash_protected_and_exactly_scoped():
    artifact = _real_artifact()
    for check in artifact["checks"]:
        if check["check_id"] in ("adapter_object_path", "rh56_open_object_path"):
            assert check["authority_basis"] == (
                "authoritative_for_exact_bound_anydex_object_point_cube_union"
            )
            assert check["authoritative_for_bound_object_cloud"] is True
            assert check["authoritative"] is False
    target = next(
        item
        for item in artifact["checks"]
        if item["check_id"] == "adapter_object_path"
    )
    target["authority_basis"] = "authoritative_for_all_unseen_object_space"
    with pytest.raises(ValueError, match="authority_basis"):
        validate_pregrasp_only_audit(_reseal(artifact), require_pass=True)

    boolean_tamper = _real_artifact()
    target = next(
        item
        for item in boolean_tamper["checks"]
        if item["check_id"] == "rh56_open_object_path"
    )
    target["authoritative_for_bound_object_cloud"] = False
    with pytest.raises(ValueError, match="authoritative_for_bound_object_cloud"):
        validate_pregrasp_only_audit(_reseal(boolean_tamper), require_pass=True)


@pytest.mark.skipif(not REAL_AUDIT.is_file(), reason="bound real prefix audit missing")
@pytest.mark.parametrize(
    "field,value,match",
    [
        ("age_at_artifact_creation_s", 0.0, "age_at_artifact_creation"),
        ("configured_recommended_max_scene_age_s", 3600.0, "recommended age"),
        ("age_exceeded_recommended_max", False, "warning boolean"),
        ("freshness_role", "runtime_gate", "advisory-only"),
        ("freshness_warning", "", "warning text"),
    ],
)
def test_scene_age_provenance_is_exact_and_tamper_evident_but_not_a_pass_gate(
    field, value, match
):
    artifact = _real_artifact()
    # The authentic artifact is older than the recommendation yet remains a
    # valid geometry artifact because full-scene returns are advisory only.
    validate_pregrasp_only_audit(artifact, require_pass=True)
    artifact["bindings"]["filtered_scene"][field] = value
    with pytest.raises(ValueError, match=match):
        validate_pregrasp_only_audit(_reseal(artifact), require_pass=True)
