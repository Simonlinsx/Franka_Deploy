import hashlib
import json
from pathlib import Path

import pytest

from dynamic_pcd.apps import validate_saved_rgbd_npz as saved_validator
from dynamic_pcd.evaluation.guarded_v2_real_rgb_replay import (
    RealRGBReplayError,
    replay_manifest,
)
from dynamic_pcd.evaluation.guarded_v2_source_provenance import (
    CONTRACT_MANIFEST_SHA256_FIELD,
    CONTRACT_PATH_FIELD,
    CONTRACT_SOURCE_SET_SHA256_FIELD,
    GuardedV2SourceProvenanceError,
    REQUIRED_SOURCE_SPECS,
    SOURCE_MANIFEST_SCHEMA,
    source_set_payload,
    validate_contract_source_provenance,
    validate_source_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_workspace(tmp_path: Path) -> tuple[Path, Path, dict]:
    workspace = tmp_path / "workspace"
    for spec in REQUIRED_SOURCE_SPECS:
        source = workspace / spec.path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"producer={spec.path}\n", encoding="utf-8")
    source_set = source_set_payload(workspace)
    manifest = (
        workspace
        / "perception"
        / "configs"
        / "guarded_v2_implementation_sources.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "description": "deterministic test fixture",
        "workspace_root": ".",
        "files": source_set["files"],
        "source_set_sha256": hashlib.sha256(
            json.dumps(
                source_set,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return workspace, manifest, payload


def test_source_manifest_is_exact_sorted_acyclic_and_byte_valid(tmp_path):
    workspace, manifest, payload = _synthetic_workspace(tmp_path)

    result = validate_source_manifest(manifest, workspace_root=workspace)

    expected_paths = sorted(spec.path for spec in REQUIRED_SOURCE_SPECS)
    assert [record["path"] for record in payload["files"]] == expected_paths
    assert all(
        not record["path"].startswith("perception/configs/")
        for record in payload["files"]
    )
    assert result["validated"] is True
    assert result["manifest_sha256"] == _sha256(manifest)
    assert result["source_set_sha256"] == payload["source_set_sha256"]
    assert result["file_count"] == len(REQUIRED_SOURCE_SPECS)


def test_source_manifest_rejects_a_changed_producer_byte(tmp_path):
    workspace, manifest, _payload = _synthetic_workspace(tmp_path)
    changed = workspace / REQUIRED_SOURCE_SPECS[0].path
    changed.write_text("changed after commissioning\n", encoding="utf-8")

    with pytest.raises(
        GuardedV2SourceProvenanceError,
        match="source (size|SHA-256) mismatch",
    ):
        validate_source_manifest(manifest, workspace_root=workspace)


def test_contract_resolves_relative_manifest_and_checks_both_digests(tmp_path):
    workspace, manifest, payload = _synthetic_workspace(tmp_path)
    contract = {
        CONTRACT_PATH_FIELD: manifest.name,
        CONTRACT_MANIFEST_SHA256_FIELD: _sha256(manifest),
        CONTRACT_SOURCE_SET_SHA256_FIELD: payload["source_set_sha256"],
    }

    result = validate_contract_source_provenance(
        contract,
        contract_base=manifest.parent,
        workspace_root=workspace,
        required=True,
    )

    assert result is not None
    assert result["validated"] is True
    assert result["manifest_sha256"] == contract[
        CONTRACT_MANIFEST_SHA256_FIELD
    ]


def test_contract_rejects_an_incomplete_source_pin_trio(tmp_path):
    workspace, manifest, _payload = _synthetic_workspace(tmp_path)
    with pytest.raises(
        GuardedV2SourceProvenanceError,
        match="contract is incomplete",
    ):
        validate_contract_source_provenance(
            {CONTRACT_PATH_FIELD: manifest.name},
            contract_base=manifest.parent,
            workspace_root=workspace,
            required=False,
        )


def test_replay_refuses_bad_source_contract_before_creating_output(tmp_path):
    manifest = tmp_path / "replay.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "guarded_v2_real_rgb_replay_v1",
                "production_acceptance_contract": {
                    CONTRACT_PATH_FIELD: "source.json"
                },
                "cases": [{"name": "synthetic"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "candidate"

    with pytest.raises(RealRGBReplayError, match="contract is incomplete"):
        replay_manifest(
            manifest,
            output,
            config_path=tmp_path / "unused.yaml",
            realtime=False,
        )

    assert not output.exists()


def test_saved_rgbd_fixed_contract_propagates_source_provenance(
    tmp_path, monkeypatch
):
    contract_path = tmp_path / "replay_contract.json"
    contract = {
        "required_case_names": ["a"],
        "default_config_sha256": "1" * 64,
        "default_neutral_depth_m": 1.0,
        "default_sam2_image_size": 512,
        "checkpoint_sha256": "2" * 64,
        "model_config_sha256": "3" * 64,
        CONTRACT_PATH_FIELD: "implementation.json",
        CONTRACT_MANIFEST_SHA256_FIELD: "4" * 64,
        CONTRACT_SOURCE_SET_SHA256_FIELD: "5" * 64,
    }
    contract_path.write_text(
        json.dumps({"production_acceptance_contract": contract}),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps({"expected_replay_manifest_sha256": _sha256(contract_path)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(saved_validator, "ACCEPTANCE_CONTRACT_PATH", contract_path)
    monkeypatch.setattr(saved_validator, "BENCHMARK_CONTRACT_PATH", benchmark_path)
    calls = []

    def fake_validate(candidate, **kwargs):
        calls.append((candidate, kwargs))
        return {"validated": True, "source_set_sha256": "5" * 64}

    monkeypatch.setattr(
        saved_validator, "validate_contract_source_provenance", fake_validate
    )

    loaded = saved_validator._load_fixed_acceptance_contract()

    assert loaded["implementation_source_provenance"]["validated"] is True
    assert loaded["sha256_pin_source"]["path"] == str(benchmark_path)
    assert calls == [
        (
            contract,
            {
                "contract_base": contract_path.parent,
                "workspace_root": saved_validator.WORKSPACE,
                "required": False,
            },
        )
    ]
