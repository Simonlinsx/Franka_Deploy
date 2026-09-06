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

from anydex_pipeline.pregrasp_only_audit import (
    SCHEMA_VERSION,
    build_pregrasp_joint_path,
    pregrasp_prefix_contract_sha256,
    validate_pregrasp_only_audit,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_AUDIT = (
    ROOT
    / "runs/candidate51_pregrasp_only_audit_current_fr3_limits_20260726.json"
)
EXECUTE_APP = ROOT / "apps/execute_control_sequence.py"


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


def _array_sha256(value):
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    shape = ",".join(str(item) for item in array.shape)
    digest = hashlib.sha256()
    digest.update(("dtype=<f8;shape={};".format(shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reseal(artifact):
    result = copy.deepcopy(artifact)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _canonical_json_sha256(result)
    return result


def _upgrade_fixture_to_v2():
    """Normalize the immutable source evidence only inside this test fixture.

    This deliberately does not depend on a newly generated v2 run artifact.
    All original path/check/object evidence remains unchanged; only the new
    explicit provenance fields and the root digest are populated.
    """

    artifact = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    artifact["schema_version"] = SCHEMA_VERSION
    prefix = artifact["bindings"]["joint_prefix"]
    current_q = np.asarray(prefix["waypoints"][0]["q_rad"], dtype=np.float64)
    prefix["current_q_sha256"] = _array_sha256(current_q)
    prefix["current_q_source"] = "joint_plan_manifest.q_start_rad"
    prefix["runtime_live_q_gate_required"] = True

    scene = artifact["bindings"]["filtered_scene"]
    capture_q = np.asarray(scene["capture_q_rad"], dtype=np.float64)
    scene["capture_q_role"] = "advisory_scene_provenance_only"
    scene["capture_to_prefix_current_linf_rad"] = float(
        np.max(np.abs(capture_q - current_q))
    )
    scene["capture_matches_prefix_current"] = bool(
        np.array_equal(capture_q, current_q)
    )
    return _reseal(artifact)


def _with_independent_scene_q(artifact, scene_q):
    result = copy.deepcopy(artifact)
    scene_q = np.asarray(scene_q, dtype=np.float64)
    prefix_q = np.asarray(
        result["bindings"]["joint_prefix"]["waypoints"][0]["q_rad"],
        dtype=np.float64,
    )
    scene = result["bindings"]["filtered_scene"]
    scene["capture_q_rad"] = scene_q.tolist()
    scene["capture_q_sha256"] = _array_sha256(scene_q)
    scene["capture_to_prefix_current_linf_rad"] = float(
        np.max(np.abs(scene_q - prefix_q))
    )
    scene["capture_matches_prefix_current"] = bool(
        np.array_equal(scene_q, prefix_q)
    )
    return _reseal(result)


@pytest.fixture
def v2_artifact():
    if not SOURCE_AUDIT.is_file():
        pytest.skip("prefix fixture is unavailable")
    artifact = _upgrade_fixture_to_v2()
    validate_pregrasp_only_audit(artifact, require_pass=True)
    return artifact


def test_advisory_scene_q_can_differ_without_changing_prefix_or_hard_checks(
    v2_artifact,
):
    prefix_before = copy.deepcopy(v2_artifact["bindings"]["joint_prefix"])
    checks_before = copy.deepcopy(v2_artifact["checks"])
    capture_q = np.asarray(
        prefix_before["waypoints"][0]["q_rad"], dtype=np.float64
    )
    capture_q[0] += 0.025

    artifact = _with_independent_scene_q(v2_artifact, capture_q)
    validated = validate_pregrasp_only_audit(artifact, require_pass=True)

    scene = validated["bindings"]["filtered_scene"]
    assert scene["capture_q_role"] == "advisory_scene_provenance_only"
    assert scene["capture_matches_prefix_current"] is False
    assert scene["capture_to_prefix_current_linf_rad"] == pytest.approx(0.025)
    assert validated["bindings"]["joint_prefix"] == prefix_before
    assert validated["checks"] == checks_before
    assert validated["decision"]["required_geometry_passed"] is True


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda scene: scene.__setitem__("capture_q_sha256", "0" * 64),
            "capture q hash",
        ),
        (
            lambda scene: scene.__setitem__("capture_q_role", "execution_q"),
            "capture q role",
        ),
        (
            lambda scene: scene.__setitem__(
                "capture_to_prefix_current_linf_rad",
                float(scene["capture_to_prefix_current_linf_rad"]) + 1.0e-6,
            ),
            "scene-to-prefix current-q distance",
        ),
        (
            lambda scene: scene.__setitem__("capture_matches_prefix_current", True),
            "match flag",
        ),
    ],
)
def test_advisory_scene_q_provenance_fields_are_tamper_evident(
    v2_artifact, mutation, match
):
    capture_q = np.asarray(
        v2_artifact["bindings"]["filtered_scene"]["capture_q_rad"],
        dtype=np.float64,
    )
    capture_q[0] += 0.025
    artifact = _with_independent_scene_q(v2_artifact, capture_q)
    mutation(artifact["bindings"]["filtered_scene"])
    with pytest.raises(ValueError, match=match):
        validate_pregrasp_only_audit(_reseal(artifact), require_pass=True)


def test_decoupled_scene_q_still_requires_exact_npz_hash_and_capture_replay(
    v2_artifact, tmp_path
):
    source = Path(v2_artifact["bindings"]["filtered_scene"]["path"])
    with np.load(str(source), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    capture_q = np.asarray(arrays["capture_q_rad"], dtype=np.float64).copy()
    capture_q[0] += 0.025
    arrays["capture_q_rad"] = capture_q
    replay_scene = tmp_path / "independent_scene_q.npz"
    np.savez_compressed(str(replay_scene), **arrays)

    artifact = _with_independent_scene_q(v2_artifact, capture_q)
    scene = artifact["bindings"]["filtered_scene"]
    scene["path"] = str(replay_scene.resolve())
    scene["sha256"] = _file_sha256(replay_scene)
    artifact = _reseal(artifact)
    validate_pregrasp_only_audit(
        artifact, verify_files=True, require_pass=True
    )

    # A self-consistent JSON rewrite cannot impersonate the independently
    # hashed scene capture stored in the bound NPZ.
    forged_q = capture_q.copy()
    forged_q[1] += 0.01
    forged = _with_independent_scene_q(artifact, forged_q)
    with pytest.raises(ValueError, match="filtered-scene capture binding replay"):
        validate_pregrasp_only_audit(
            forged, verify_files=True, require_pass=True
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("current_q_sha256", "0" * 64, "current-q hash"),
        ("current_q_source", "filtered_scene.capture_q_rad", "joint plan"),
        ("runtime_live_q_gate_required", False, "live-q gate"),
    ],
)
def test_prefix_current_authority_fields_remain_fail_closed(
    v2_artifact, field, value, match
):
    artifact = copy.deepcopy(v2_artifact)
    artifact["bindings"]["joint_prefix"][field] = value
    with pytest.raises(ValueError, match=match):
        validate_pregrasp_only_audit(_reseal(artifact), require_pass=True)


def test_fully_rehashed_prefix_current_still_must_match_joint_plan(v2_artifact):
    artifact = copy.deepcopy(v2_artifact)
    prefix = artifact["bindings"]["joint_prefix"]
    prefix["waypoints"][0]["q_rad"][0] += 0.001
    waypoints = tuple(
        (item["name"], item["q_rad"]) for item in prefix["waypoints"]
    )
    samples, segments = build_pregrasp_joint_path(
        waypoints, prefix["max_joint_step_rad"]
    )
    assert len(samples) == prefix["sample_count"]
    prefix["current_q_sha256"] = _array_sha256(waypoints[0][1])
    prefix["samples_rad"] = samples.tolist()
    prefix["samples_sha256"] = _array_sha256(samples)
    prefix["segments"] = [dict(item) for item in segments]
    prefix["prefix_contract_sha256"] = pregrasp_prefix_contract_sha256(
        waypoints=waypoints,
        pregrasp_pose=prefix["pregrasp_pose_base_EE"],
        max_joint_step_rad=prefix["max_joint_step_rad"],
        max_q_tracking_error_rad=prefix["max_q_tracking_error_rad"],
        samples_sha256=prefix["samples_sha256"],
    )
    scene_q = np.asarray(
        artifact["bindings"]["filtered_scene"]["capture_q_rad"],
        dtype=np.float64,
    )
    artifact = _with_independent_scene_q(artifact, scene_q)

    # The internal prefix hashes are now self-consistent.  File replay must
    # still bind its first waypoint exactly to joint_plan.q_start_rad.
    with pytest.raises(ValueError, match="joint-plan start q"):
        validate_pregrasp_only_audit(
            artifact, verify_files=True, require_pass=True
        )


def test_object_cloud_replay_remains_hard_with_independent_scene_q(v2_artifact):
    capture_q = np.asarray(
        v2_artifact["bindings"]["filtered_scene"]["capture_q_rad"],
        dtype=np.float64,
    )
    capture_q[0] += 0.025
    artifact = _with_independent_scene_q(v2_artifact, capture_q)
    artifact["bindings"]["object_cloud"]["points_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound AnyDex object cloud replay"):
        validate_pregrasp_only_audit(
            _reseal(artifact), verify_files=True, require_pass=True
        )


def _load_execute_app():
    name = "test_pregrasp_scene_q_contract_execute_app"
    spec = importlib.util.spec_from_file_location(name, EXECUTE_APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_live_q_uses_prefix_current_and_never_advisory_scene_q(
    monkeypatch, capsys
):
    app = _load_execute_app()
    prefix_q = np.asarray([0.01, 0.0, 0.0, -1.57, 0.0, 1.57, 0.8])
    scene_q = prefix_q.copy()
    scene_q[0] += 0.2
    state = SimpleNamespace(q=prefix_q.copy())

    class Arm:
        robot = SimpleNamespace(read_once=lambda: state)

        @staticmethod
        def _validate_state(value, *, require_idle, enforce_success):
            assert require_idle is True
            assert enforce_success is False

    artifact = {
        "bindings": {
            "joint_prefix": {
                "waypoints": [
                    {"name": "current", "q_rad": prefix_q.tolist()}
                ],
                "max_q_tracking_error_rad": 0.002,
            },
            "filtered_scene": {
                "capture_q_rad": scene_q.tolist(),
                "capture_to_prefix_current_linf_rad": 0.2,
                "capture_matches_prefix_current": False,
                "captured_at_s": 1.0,
            },
        },
        "policies": {"max_scene_age_s": 2.0},
    }
    monkeypatch.setattr(app.time, "time", lambda: 10_000.0)
    app._verify_live_q_matches_pregrasp_audit(Arm(), artifact)
    assert "is not an execution gate" in capsys.readouterr().out

    state.q = scene_q
    with pytest.raises(RuntimeError, match="moved since pregrasp-only audit"):
        app._verify_live_q_matches_pregrasp_audit(Arm(), artifact)
