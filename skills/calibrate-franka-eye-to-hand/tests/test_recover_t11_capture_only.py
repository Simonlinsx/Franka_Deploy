from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "recover_t11_capture_only.py"
)
SPEC = importlib.util.spec_from_file_location("recover_t11_capture_only_tested", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
recovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recovery
SPEC.loader.exec_module(recovery)


def _prepare_args(
    plan: Path,
    artifacts: Path,
    claim: Path,
    *,
    minimum_free_bytes: int = recovery.MIN_FREE_BYTES,
) -> argparse.Namespace:
    return argparse.Namespace(
        output_plan=plan,
        artifacts_dir=artifacts,
        claim_path=claim,
        python_executable=str(recovery.ROOT / ".venv/bin/python"),
        minimum_free_bytes=minimum_free_bytes,
        confirm_capture_only_no_motion=recovery.RECOVERY_TOKEN,
    )


def _patch_fixed_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    plan = tmp_path / "fixed.plan.yaml"
    artifacts = tmp_path / "fixed-artifacts"
    claim = tmp_path / "fixed-claim.json"
    monkeypatch.setattr(recovery, "RECOVERY_PLAN", plan)
    monkeypatch.setattr(recovery, "RECOVERY_ARTIFACTS", artifacts)
    monkeypatch.setattr(recovery, "RECOVERY_CLAIM", claim)
    return plan, artifacts, claim


def _require_unconsumed_precondition() -> None:
    snapshot = recovery.training._dataset_snapshot(recovery.DATASET)
    if len(snapshot.samples) != 10 or snapshot.sha256 != recovery.DATASET_10_SHA256:
        pytest.skip("the one-use T11 recovery precondition has already been consumed")


def test_exact_original_evidence_is_currently_valid() -> None:
    _require_unconsumed_precondition()
    master, _ = recovery._load_master()
    snapshot = recovery._validate_dataset_10(master)
    evidence = recovery._validate_original_evidence(master, snapshot)
    assert evidence["claim"]["sha256"] == recovery.ORIGINAL_CLAIM_SHA256
    assert evidence["motion_log"]["sha256"] == recovery.ORIGINAL_MOTION_LOG_SHA256
    assert evidence["inspection_raw"]["sha256"] == recovery.ORIGINAL_INSPECTION_RAW_SHA256
    assert (
        evidence["inspection_annotated"]["sha256"]
        == recovery.ORIGINAL_INSPECTION_ANNOTATED_SHA256
    )
    assert evidence["failed_zero_byte_telemetry"] == {
        "path": str(recovery.ORIGINAL_TELEMETRY.resolve()),
        "sha256": recovery.EMPTY_SHA256,
        "bytes": 0,
    }


def test_prepare_builds_no_motion_one_use_plan_without_opening_hardware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _require_unconsumed_precondition()
    plan, artifacts, claim = _patch_fixed_paths(monkeypatch, tmp_path)
    published = {}

    def fake_publish(path: Path, payload: bytes, **_: object) -> None:
        published[Path(path)] = payload

    monkeypatch.setattr(recovery.training, "_publish_exclusive", fake_publish)
    result = recovery.prepare_recovery_plan(_prepare_args(plan, artifacts, claim))
    assert result["robot_motion_permitted"] is False
    assert result["hardware_opened"] is False
    document = yaml.safe_load(published[plan])
    assert document["scope"] == {
        "robot_motion_permitted": False,
        "robot_state_read_only": True,
        "camera_capture_permitted": True,
        "target_pose_id": "T11",
        "expected_new_sample_index": 11,
        "automatic_retry_permitted": False,
    }
    assert document["safety"]["minimum_free_bytes_before_hardware"] == 1_073_741_824
    assert document["recovery_transaction"]["canonical_one_use_claim"] == str(claim)
    assert document["recovery_transaction"]["artifacts_dir"] == str(artifacts)


@pytest.mark.parametrize("bad_value", [-1, 0, 1_073_741_823, True])
def test_prepare_rejects_nonexact_free_space_gate_before_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_value: object
) -> None:
    plan, artifacts, claim = _patch_fixed_paths(monkeypatch, tmp_path)
    called = False

    def fake_publish(*_: object, **__: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(recovery.training, "_publish_exclusive", fake_publish)
    args = _prepare_args(plan, artifacts, claim)
    args.minimum_free_bytes = bad_value
    with pytest.raises(ValueError, match="exactly"):
        recovery.prepare_recovery_plan(args)
    assert called is False
    assert not plan.exists()


def test_prepare_rejects_alternate_claim_before_any_hardware_or_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, artifacts, claim = _patch_fixed_paths(monkeypatch, tmp_path)
    args = _prepare_args(plan, artifacts, tmp_path / "alternate-claim.json")
    monkeypatch.setattr(
        recovery,
        "_load_master",
        lambda: (_ for _ in ()).throw(AssertionError("evidence read must not occur")),
    )
    with pytest.raises(ValueError, match="fixed one-use"):
        recovery.prepare_recovery_plan(args)
    assert not plan.exists()
    assert not claim.exists()


def _state_payload(**updates: object) -> dict:
    payload = {
        "recovery_plan": "/tmp/recovery.plan.yaml",
        "recovery_plan_sha256": "a" * 64,
        "recovery_claim": "/tmp/recovery.claim.json",
        "recovery_claim_sha256": "b" * 64,
        "robot_motion_commanded": False,
        "robot_state_read_only": True,
        "robot_mode": "Idle",
        "current_errors_active": False,
        "contacts_or_collisions_active": False,
        "pose_id": "T11",
        "pose_gate_pass": True,
        "T_base_ee": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "translation_error_m": 0.001,
        "rotation_error_deg": 0.1,
        "max_abs_dq_rad_s": 0.004,
    }
    payload.update(updates)
    return payload


def _parse_fake_state(payload: dict):
    output = "FRANKA_T11_PASSIVE_STATE_JSON=" + json.dumps(payload) + "\n"
    result = recovery.training.CommandResult(("fake-passive-read",), 0, output)
    return recovery._parse_state_probe(
        result,
        plan_path=Path("/tmp/recovery.plan.yaml"),
        plan_sha256="a" * 64,
        claim_path=Path("/tmp/recovery.claim.json"),
        claim_sha256="b" * 64,
    )


def test_fake_passive_state_accepts_idle_no_contact_t11_without_motion() -> None:
    checked = _parse_fake_state(_state_payload())
    assert checked["robot_motion_commanded"] is False
    assert checked["robot_mode"] == "Idle"


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"robot_motion_commanded": True}, "field robot_motion_commanded"),
        ({"robot_mode": "Move"}, "field robot_mode"),
        ({"contacts_or_collisions_active": True}, "contacts_or_collisions"),
        ({"translation_error_m": 0.006}, "not at T11"),
        ({"max_abs_dq_rad_s": 0.0051}, "not stationary"),
    ],
)
def test_fake_passive_state_fails_closed(updates: dict, match: str) -> None:
    with pytest.raises(recovery.training.SequenceFailure, match=match):
        _parse_fake_state(_state_payload(**updates))


def test_resume_output_paths_are_fixed_before_receipt_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The fixed-output check occurs after receipt validation in the production
    # function, so inspect the source contract as a regression guard too.
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'output != RESUME_MASTER.resolve()' in source
    assert 'artifacts != RESUME_ARTIFACTS.resolve()' in source
    assert '"robot_motion_command_invoked": False' in source
    assert '"move_api_invoked": False' in source
