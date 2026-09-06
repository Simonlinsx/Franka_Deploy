"""Fail-closed provenance for the guarded-v2 perception implementation.

Configuration, model, and replay-input hashes do not identify the Python code
that produced a mask or point cloud.  This module validates a separate,
deterministic manifest of the production implementation sources.  The source
manifest deliberately does not contain itself, any runtime configuration, or
either acceptance manifest; this keeps the hash dependency graph acyclic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


SOURCE_MANIFEST_SCHEMA = "guarded_v2_implementation_sources_v1"
SOURCE_SET_SCHEMA = "guarded_v2_implementation_source_set_v1"
CONTRACT_PATH_FIELD = "implementation_source_manifest"
CONTRACT_MANIFEST_SHA256_FIELD = "implementation_source_manifest_sha256"
CONTRACT_SOURCE_SET_SHA256_FIELD = "implementation_source_set_sha256"
CONTRACT_FIELDS = frozenset(
    {
        CONTRACT_PATH_FIELD,
        CONTRACT_MANIFEST_SHA256_FIELD,
        CONTRACT_SOURCE_SET_SHA256_FIELD,
    }
)


@dataclass(frozen=True)
class GuardedV2SourceSpec:
    path: str
    role: str


# This is intentionally an exact source set, not a caller-controlled list.
# Adding or removing a producer requires a reviewed code change and a new
# source manifest.  Paths are workspace-relative and sorted in the emitted
# manifest, regardless of the presentation order below.
REQUIRED_SOURCE_SPECS = (
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/apps/sam2_video_service.py",
        "sam2_service",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/apps/validate_saved_rgbd_npz.py",
        "saved_rgbd_validator",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/camera/realsense_camera.py",
        "rgbd_camera",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/config.py",
        "configuration_loader",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/evaluation/guarded_v2_offline.py",
        "offline_evaluator",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/evaluation/guarded_v2_real_rgb_replay.py",
        "real_rgb_replay",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/evaluation/guarded_v2_source_provenance.py",
        "source_provenance_validator",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/pointcloud/extractor.py",
        "pointcloud_extractor",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/provider/object_pcd_provider.py",
        "object_pcd_provider",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/provider/state.py",
        "object_pcd_provider_state",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/segmentation/adaptive_color_depth_tracker.py",
        "adaptive_tracker",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/segmentation/sam2_video_backend.py",
        "sam2_backend",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/segmentation/sam2_video_client.py",
        "sam2_client",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/segmentation/sam2_video_protocol.py",
        "sam2_protocol",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/segmentation/sam2_video_runtime.py",
        "sam2_runtime",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/types.py",
        "perception_types",
    ),
    GuardedV2SourceSpec(
        "perception/dynamic_pcd/utils/geometry.py",
        "geometry",
    ),
    GuardedV2SourceSpec(
        "perception/scripts/benchmark_guarded_v2_masks.py",
        "offline_evaluator_cli",
    ),
    GuardedV2SourceSpec(
        "perception/scripts/replay_guarded_v2_real_rgb.py",
        "real_rgb_replay_cli",
    ),
    GuardedV2SourceSpec(
        "sim2real/observation/model.py",
        "formal_policy_pointcloud_projector",
    ),
)


class GuardedV2SourceProvenanceError(ValueError):
    """Raised when implementation-source provenance is absent or invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and text == text.lower() and all(
        character in "0123456789abcdef" for character in text
    )


def _resolved_workspace_source(workspace_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or path.as_posix() != relative_path:
        raise GuardedV2SourceProvenanceError(
            f"implementation source path must be normalized and relative: {relative_path!r}"
        )
    resolved_root = workspace_root.expanduser().resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise GuardedV2SourceProvenanceError(
            f"implementation source escapes workspace: {relative_path!r}"
        ) from exc
    if not resolved.is_file():
        raise GuardedV2SourceProvenanceError(
            f"implementation source is missing: {relative_path}"
        )
    return resolved


def source_set_payload(workspace_root: Path) -> dict[str, Any]:
    """Build the canonical source-set payload from the exact required files."""

    records = []
    for spec in sorted(REQUIRED_SOURCE_SPECS, key=lambda item: item.path):
        source = _resolved_workspace_source(workspace_root, spec.path)
        records.append(
            {
                "path": spec.path,
                "role": spec.role,
                "sha256": _sha256(source),
                "size_bytes": int(source.stat().st_size),
            }
        )
    return {"schema": SOURCE_SET_SCHEMA, "files": records}


def source_set_sha256(workspace_root: Path) -> str:
    return _canonical_json_sha256(source_set_payload(workspace_root))


def validate_source_manifest(
    manifest_path: Path,
    *,
    workspace_root: Path,
    expected_manifest_sha256: Optional[str] = None,
    expected_source_set_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Validate manifest bytes, exact membership, and every producer byte."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise GuardedV2SourceProvenanceError(
            f"implementation source manifest is missing: {manifest}"
        )
    actual_manifest_sha256 = _sha256(manifest)
    if expected_manifest_sha256 is not None:
        if not _valid_sha256(expected_manifest_sha256):
            raise GuardedV2SourceProvenanceError(
                "implementation source manifest contract SHA-256 is invalid"
            )
        if actual_manifest_sha256 != str(expected_manifest_sha256):
            raise GuardedV2SourceProvenanceError(
                "implementation source manifest SHA-256 mismatch: "
                f"expected={expected_manifest_sha256}, actual={actual_manifest_sha256}"
            )

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GuardedV2SourceProvenanceError(
            f"cannot read implementation source manifest: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise GuardedV2SourceProvenanceError(
            f"implementation source manifest schema must be {SOURCE_MANIFEST_SCHEMA!r}"
        )
    if payload.get("workspace_root") != ".":
        raise GuardedV2SourceProvenanceError(
            "implementation source manifest workspace_root must be '.'"
        )
    records = payload.get("files")
    if not isinstance(records, list):
        raise GuardedV2SourceProvenanceError(
            "implementation source manifest files must be a list"
        )
    expected_specs = {
        spec.path: spec.role for spec in REQUIRED_SOURCE_SPECS
    }
    paths = [record.get("path") for record in records if isinstance(record, dict)]
    if len(paths) != len(records) or any(not isinstance(path, str) for path in paths):
        raise GuardedV2SourceProvenanceError(
            "every implementation source record needs a string path"
        )
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise GuardedV2SourceProvenanceError(
            "implementation source records must have unique sorted paths"
        )
    if set(paths) != set(expected_specs):
        missing = sorted(set(expected_specs) - set(paths))
        extra = sorted(set(paths) - set(expected_specs))
        raise GuardedV2SourceProvenanceError(
            f"implementation source membership mismatch: missing={missing}, extra={extra}"
        )

    # Config/replay manifests and this source manifest itself are forbidden to
    # make the provenance dependency graph explicit and acyclic.
    manifest_root = Path(workspace_root).expanduser().resolve()
    try:
        manifest_relative = manifest.relative_to(manifest_root).as_posix()
    except ValueError:
        manifest_relative = None
    for record in records:
        relative = str(record["path"])
        if relative == manifest_relative or relative.startswith(
            "perception/configs/"
        ):
            raise GuardedV2SourceProvenanceError(
                f"cyclic/non-source manifest entry is forbidden: {relative}"
            )
        if record.get("role") != expected_specs[relative]:
            raise GuardedV2SourceProvenanceError(
                f"implementation source role mismatch for {relative}"
            )
        declared_sha256 = record.get("sha256")
        if not _valid_sha256(declared_sha256):
            raise GuardedV2SourceProvenanceError(
                f"invalid source SHA-256 for {relative}"
            )
        declared_size = record.get("size_bytes")
        if isinstance(declared_size, bool) or not isinstance(declared_size, int):
            raise GuardedV2SourceProvenanceError(
                f"invalid source size for {relative}"
            )
        source = _resolved_workspace_source(workspace_root, relative)
        actual_size = int(source.stat().st_size)
        if actual_size != declared_size:
            raise GuardedV2SourceProvenanceError(
                f"implementation source size mismatch for {relative}: "
                f"expected={declared_size}, actual={actual_size}"
            )
        actual_sha256 = _sha256(source)
        if actual_sha256 != declared_sha256:
            raise GuardedV2SourceProvenanceError(
                f"implementation source SHA-256 mismatch for {relative}: "
                f"expected={declared_sha256}, actual={actual_sha256}"
            )

    source_payload = {"schema": SOURCE_SET_SCHEMA, "files": records}
    actual_source_set_sha256 = _canonical_json_sha256(source_payload)
    declared_source_set_sha256 = payload.get("source_set_sha256")
    if not _valid_sha256(declared_source_set_sha256):
        raise GuardedV2SourceProvenanceError(
            "implementation source_set_sha256 is invalid"
        )
    if actual_source_set_sha256 != declared_source_set_sha256:
        raise GuardedV2SourceProvenanceError(
            "implementation source-set digest does not match manifest records"
        )
    if expected_source_set_sha256 is not None:
        if not _valid_sha256(expected_source_set_sha256):
            raise GuardedV2SourceProvenanceError(
                "implementation source-set contract SHA-256 is invalid"
            )
        if actual_source_set_sha256 != str(expected_source_set_sha256):
            raise GuardedV2SourceProvenanceError(
                "implementation source-set SHA-256 mismatch: "
                f"expected={expected_source_set_sha256}, "
                f"actual={actual_source_set_sha256}"
            )

    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "manifest": (
            manifest_relative if manifest_relative is not None else str(manifest)
        ),
        "manifest_sha256": actual_manifest_sha256,
        "source_set_sha256": actual_source_set_sha256,
        "file_count": len(records),
        "files": records,
        "validated": True,
    }


def validate_contract_source_provenance(
    contract: Mapping[str, Any],
    *,
    contract_base: Path,
    workspace_root: Path,
    required: bool,
) -> Optional[dict[str, Any]]:
    """Resolve and validate the source-manifest fields of an outer contract."""

    supplied = CONTRACT_FIELDS.intersection(contract)
    if not supplied:
        if required:
            raise GuardedV2SourceProvenanceError(
                "production contract does not pin implementation sources"
            )
        return None
    if supplied != CONTRACT_FIELDS:
        missing = sorted(CONTRACT_FIELDS - supplied)
        raise GuardedV2SourceProvenanceError(
            f"implementation source contract is incomplete; missing={missing}"
        )
    manifest_ref = contract[CONTRACT_PATH_FIELD]
    if not isinstance(manifest_ref, str) or not manifest_ref:
        raise GuardedV2SourceProvenanceError(
            "implementation_source_manifest must be a non-empty relative path"
        )
    relative = Path(manifest_ref)
    if relative.is_absolute():
        raise GuardedV2SourceProvenanceError(
            "implementation_source_manifest must be relative to its contract"
        )
    manifest = (Path(contract_base).expanduser().resolve() / relative).resolve()
    return validate_source_manifest(
        manifest,
        workspace_root=workspace_root,
        expected_manifest_sha256=str(contract[CONTRACT_MANIFEST_SHA256_FIELD]),
        expected_source_set_sha256=str(contract[CONTRACT_SOURCE_SET_SHA256_FIELD]),
    )
