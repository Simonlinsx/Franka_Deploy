"""Hardware-independent audit for the packaged V61 six-expert handoff.

Deployment in this workspace selects exactly one of the six checkpoints; no
automatic router is used.  This audit binds the separately supplied runtime
alignment, verifies its golden vectors and constructs each selected policy.
It never opens a camera or robot interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from sim2real.deployment.bundle import load_checkpoint_safely
from sim2real.policy import RollingStudentPolicy
from sim2real.tasks.ballistics import (
    PACKAGED_SOURCE_SHA256,
    deployable_thrown_v35_future_contract,
    deployable_visual_ballistic_future_contract,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (
    WORKSPACE_ROOT
    / "data/checkpoints/thrown"
    / "thrown_v61_sixexpert_visualflight_perceptiondr025_cmp31p9_demo_candidate_20260815"
    / "thrown_v61_sixexpert_visualflight_perceptiondr025_cmp31p9_demo_candidate_20260815"
)
DEFAULT_ALIGNMENT = DEFAULT_BUNDLE.parent / "runtime_alignment/runtime_alignment"
EXPERT_PATHS = {
    "base-forward": "checkpoint.pt",
    "base-negative-y": "checkpoints/base/side_negative_y.pt",
    "base-positive-y": "checkpoints/base/side_positive_y.pt",
    "high-forward": "checkpoints/high/forward.pt",
    "high-negative-y": "checkpoints/high/side_negative_y.pt",
    "high-positive-y": "checkpoints/high/side_positive_y.pt",
}
EXPECTED_Q_HOME = np.asarray(
    [
        0.6292979717254639,
        -0.8440930247306824,
        -0.008244000375270844,
        -2.096261978149414,
        -0.40424200892448425,
        1.8299169540405273,
        -1.7469099760055542,
    ],
    dtype=np.float64,
)
DEPLOYMENT_BLOCKERS = (
    "packaged q_home is proprioception metadata, not autonomous real-reset authorization",
)

ALIGNMENT_EXPERT_KEYS = {
    "base-forward": "base/forward",
    "base-negative-y": "base/side_negative_y",
    "base-positive-y": "base/side_positive_y",
    "high-forward": "high/forward",
    "high-negative-y": "high/side_negative_y",
    "high-positive-y": "high/side_positive_y",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _artifact_hashes(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            path_value = value.get("path")
            digest_value = value.get("sha256")
            if isinstance(path_value, str) and isinstance(digest_value, str):
                path = (root / path_value).resolve()
                if root != path and root not in path.parents:
                    raise ValueError(f"artifact escapes bundle root: {path_value}")
                if not path.is_file():
                    raise ValueError(f"missing bundle artifact: {path_value}")
                actual = _sha256(path)
                if actual != digest_value.lower():
                    raise ValueError(f"bundle artifact SHA-256 differs: {path_value}")
                records.append(
                    {"path": path_value, "sha256": actual, "bytes": path.stat().st_size}
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest.get("artifacts"))
    return records


def _audit_runtime_alignment(root: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    alignment = Path(root).expanduser().resolve(strict=True)
    source_manifest_path = alignment / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, dict) or source_manifest.get("contract") != (
        "thrown_v61_17d_runtime_alignment_v1"
    ):
        raise ValueError("unexpected V61 runtime-alignment contract")
    packaged = _mapping(
        source_manifest.get("packaged_files"), "runtime packaged_files"
    )
    verified: dict[str, str] = {}
    for relative, expected in packaged.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("runtime packaged file records must be string mappings")
        path = (alignment / relative).resolve()
        if alignment != path and alignment not in path.parents:
            raise ValueError(f"runtime artifact escapes alignment root: {relative}")
        if not path.is_file() or _sha256(path) != expected.lower():
            raise ValueError(f"runtime alignment SHA-256 differs: {relative}")
        verified[relative] = expected.lower()
    if verified.get("thrown_ballistic_contracts.py") != PACKAGED_SOURCE_SHA256:
        raise ValueError("NumPy runtime is not bound to the packaged contract source")

    contracts = json.loads(
        (alignment / "expert_contracts.json").read_text(encoding="utf-8")
    )
    if not isinstance(contracts, dict):
        raise ValueError("expert_contracts.json root must be a mapping")
    golden = np.load(alignment / "golden_vectors.npz", allow_pickle=False)
    actual_v1 = deployable_thrown_v35_future_contract(
        golden["predicted_compact_privileged"], golden["proprio_seq"][:, -1]
    )
    actual_v2 = deployable_visual_ballistic_future_contract(
        golden["pointcloud_seq"], golden["valid_seq"], golden["proprio_seq"]
    )
    if not np.array_equal(actual_v1, golden["expected_v1"]):
        raise ValueError("NumPy thrown v1 contract differs from frozen golden")
    v2_max_abs = float(np.max(np.abs(actual_v2 - golden["expected_v2"])))
    if v2_max_abs > 2.1e-6:
        raise ValueError(
            f"NumPy visual-ballistic v2 differs from golden: {v2_max_abs}"
        )
    return (
        {
            "root": str(alignment),
            "contract": source_manifest["contract"],
            "source_manifest_sha256": _sha256(source_manifest_path),
            "packaged_files_verified": len(verified),
            "reference_source_sha256": PACKAGED_SOURCE_SHA256,
            "golden_v1_byte_equal": True,
            "golden_v2_max_abs": v2_max_abs,
        },
        contracts,
    )


def audit_bundle(
    root: Path,
    *,
    selected_expert: Optional[str] = None,
    only_selected: bool = False,
    alignment_root: Path = DEFAULT_ALIGNMENT,
) -> dict[str, Any]:
    bundle = Path(root).expanduser().resolve(strict=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("V61 manifest root must be a mapping")
    if manifest.get("bundle_contract") != (
        "inspire_thrown_v61_speed600_xyz_h16_visual_six_expert_v1"
    ):
        raise ValueError("unexpected V61 bundle contract")
    artifacts = _artifact_hashes(bundle, manifest)
    alignment_report, alignment_contracts = _audit_runtime_alignment(alignment_root)
    manifest_artifacts = _mapping(manifest.get("artifacts"), "manifest.artifacts")
    main = _mapping(manifest_artifacts.get("checkpoint"), "checkpoint artifact")
    expected_checkpoint_hashes = {
        "base-forward": str(main.get("sha256", "")).lower(),
        "base-negative-y": str(
            _mapping(
                _mapping(manifest_artifacts.get("router_experts"), "router_experts").get(
                    "negative_y"
                ),
                "negative_y expert",
            ).get("sha256", "")
        ).lower(),
        "base-positive-y": str(
            _mapping(
                _mapping(manifest_artifacts.get("router_experts"), "router_experts").get(
                    "positive_y"
                ),
                "positive_y expert",
            ).get("sha256", "")
        ).lower(),
        "high-forward": str(
            _mapping(
                _mapping(
                    manifest_artifacts.get("trajectory_router_experts"),
                    "trajectory_router_experts",
                ).get("forward"),
                "high forward expert",
            ).get("sha256", "")
        ).lower(),
        "high-negative-y": str(
            _mapping(
                _mapping(
                    manifest_artifacts.get("trajectory_router_experts"),
                    "trajectory_router_experts",
                ).get("negative_y"),
                "high negative_y expert",
            ).get("sha256", "")
        ).lower(),
        "high-positive-y": str(
            _mapping(
                _mapping(
                    manifest_artifacts.get("trajectory_router_experts"),
                    "trajectory_router_experts",
                ).get("positive_y"),
                "high positive_y expert",
            ).get("sha256", "")
        ).lower(),
    }
    experts: dict[str, Any] = {}
    reference_keys: Optional[tuple[str, ...]] = None
    if only_selected and selected_expert is None:
        raise ValueError("only_selected requires an explicit expert")
    selected_paths = (
        {selected_expert: EXPERT_PATHS[selected_expert]}
        if only_selected and selected_expert is not None
        else EXPERT_PATHS
    )
    for name, relative in selected_paths.items():
        path = (bundle / relative).resolve()
        actual_sha = _sha256(path)
        if actual_sha != expected_checkpoint_hashes[name]:
            raise ValueError(f"{name} checkpoint SHA-256 differs from manifest")
        checkpoint = load_checkpoint_safely(path.read_bytes())
        q_home = np.asarray(
            checkpoint.metadata.get("default_arm_position_rad"), dtype=np.float64
        )
        if q_home.shape != (7,) or not np.array_equal(q_home, EXPECTED_Q_HOME):
            raise ValueError(f"{name} q_home differs from the V61 bundle contract")
        spec = checkpoint.spec
        expected_spec = {
            "history": 16,
            "num_object_points": 128,
            "point_feature_dim": 3,
            "proprio_dim": 96,
            "action_dim": 13,
            "action_chunk_size": 1,
            "temporal_encoder": "lstm",
            "temporal_hidden_dim": 256,
        }
        for key, expected in expected_spec.items():
            if spec.get(key) != expected:
                raise ValueError(
                    f"{name} spec {key} mismatch: {spec.get(key)!r}!={expected!r}"
                )
        if spec.get("analytic_future_contract_action_adapter_enabled") is not True:
            raise ValueError(f"{name} does not enable its analytic action adapter")
        alignment_key = ALIGNMENT_EXPERT_KEYS[name]
        alignment_contract = _mapping(
            alignment_contracts.get(alignment_key),
            f"runtime alignment expert {alignment_key}",
        )
        if alignment_contract.get("checkpoint_sha256") != actual_sha:
            raise ValueError(f"{name} runtime-alignment checkpoint SHA differs")
        if alignment_contract.get("analytic_future_contract") != spec.get(
            "analytic_future_contract"
        ):
            raise ValueError(f"{name} runtime-alignment 17-D contract differs")
        keys = tuple(sorted(checkpoint.model_state_dict))
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"{name} model key set differs from base-forward")
        controller = _mapping(
            checkpoint.metadata.get("action_controller"),
            f"{name} action_controller",
        )
        arm = _mapping(controller.get("arm"), f"{name} action_controller.arm")
        hand = _mapping(controller.get("hand"), f"{name} action_controller.hand")
        for actual, expected, label in (
            (controller.get("policy_frequency_hz"), 20.0, "policy_frequency_hz"),
            (arm.get("delta_scale_rad_per_policy_step"), 0.045, "arm delta"),
            (arm.get("moving_average"), 0.4, "arm EMA"),
            (hand.get("moving_average"), 0.737856, "hand EMA"),
            (hand.get("max_target_delta_rad_per_policy_step"), 0.3, "hand delta"),
        ):
            if float(actual) != float(expected):
                raise ValueError(f"{name} {label} differs from the V61 contract")
        policy = RollingStudentPolicy(checkpoint)
        if policy.expected_model_parameter_count != 2_705_387:
            raise ValueError(f"{name} runtime parameter contract differs")
        experts[name] = {
            "path": str(path),
            "sha256": actual_sha,
            "iteration_or_epoch": checkpoint.iteration,
            "parameters": int(
                sum(value.size for value in checkpoint.model_state_dict.values())
            ),
            "analytic_future_contract": spec.get("analytic_future_contract"),
            "selected": name == selected_expert,
        }

    return {
        "accepted": True,
        "hardware_writes": False,
        "bundle": str(bundle),
        "bundle_manifest_sha256": _sha256(manifest_path),
        "bundle_contract": manifest["bundle_contract"],
        "bundle_status": manifest.get("status"),
        "artifact_hashes_verified": len(artifacts),
        "checkpoint_contract": {
            "history": 16,
            "pointcloud": [16, 128, 3],
            "proprioception": [16, 96],
            "action": [13],
            "policy_rate_hz": 20.0,
            "q_home_rad": EXPECTED_Q_HOME.tolist(),
        },
        "experts": experts,
        "loaded_checkpoint_count": len(experts),
        "selected_expert": selected_expert,
        "runtime_alignment": alignment_report,
        "runtime_inference_ready": True,
        "routing_mode": "manual_single_expert",
        "automatic_router_used": False,
        "robot_reset_authorized": False,
        "runtime_blockers": [],
        "deployment_blockers": list(DEPLOYMENT_BLOCKERS),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--expert", choices=tuple(EXPERT_PATHS), default=None)
    parser.add_argument("--only-selected", action="store_true")
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--require-runtime-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = audit_bundle(
            args.bundle,
            selected_expert=args.expert,
            only_selected=args.only_selected,
            alignment_root=args.alignment,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"V61 bundle audit: FAIL: {exc}")
        return 1
    print(json.dumps(report, sort_keys=True))
    return int(bool(args.require_runtime_ready and not report["runtime_inference_ready"]))


if __name__ == "__main__":
    raise SystemExit(main())
