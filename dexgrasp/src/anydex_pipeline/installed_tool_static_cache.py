"""Tamper-evident scene-independent cache for installed-tool air audits.

The cache contains only collision observations whose two sides are immutable
triangle-mesh models (FR3, V7 adapter, and RH56).  Checks involving either
the captured scene or the saved object point cloud are deliberately excluded
and must be recomputed after a fresh capture.  A cache is accepted only when
an exact query binding and the native backend identity still match.

This module performs geometry and file I/O only.  It imports no hardware
transport and a cache artifact never authorizes motion.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .installed_tool_audit import (
    CollisionBackendIdentity,
    CollisionObservation,
    InstalledToolCollisionQuery,
    check_specs_for_mode,
)
from .rh56_hand_path import rh56_feedback_envelope_policy


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "fr3_v7_rh56_static_collision_observation_cache"

STATIC_AIR_CHECK_IDS: Tuple[str, ...] = (
    "fr3_self_path",
    "adapter_fr3_path",
    "rh56_open_fr3_path",
    "rh56_closed_fr3_final",
    "rh56_closed_adapter_final",
    "rh56_execution_fr3_final",
    "rh56_execution_adapter_final",
    "rh56_execution_self_final",
)

DYNAMIC_AIR_CHECK_IDS: Tuple[str, ...] = (
    "fr3_scene_path",
    "adapter_scene_path",
    "adapter_object_path",
    "rh56_open_scene_path",
    "rh56_open_object_path",
    "rh56_closed_scene_final",
    "rh56_closed_object_noncontact_final",
    "rh56_closed_object_all_links_final",
    "rh56_execution_scene_final",
    "rh56_execution_object_final",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_binding(value: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    shape = [int(item) for item in array.shape]
    digest = hashlib.sha256()
    digest.update(
        ("dtype=<f8;shape={};".format(",".join(map(str, shape)))).encode(
            "ascii"
        )
    )
    digest.update(array.tobytes(order="C"))
    return {"dtype": "<f8", "shape": shape, "sha256": digest.hexdigest()}


def _file_binding(path: Path) -> Mapping[str, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("static cache input is missing: {}".format(source))
    return {"path": str(source), "sha256": _sha256_file(source)}


def _identity_binding(identity: CollisionBackendIdentity) -> Mapping[str, str]:
    if not isinstance(identity, CollisionBackendIdentity):
        raise TypeError("identity must be CollisionBackendIdentity")
    return {
        "name": identity.name,
        "version": identity.version,
        "implementation_sha256": identity.implementation_sha256,
        "configuration_sha256": identity.configuration_sha256,
    }


def _transform_map_binding(values: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    return {
        str(name): _array_binding(np.asarray(values[name], dtype=np.float64))
        for name in sorted(values)
    }


def static_query_binding(
    query: InstalledToolCollisionQuery,
    identity: CollisionBackendIdentity,
) -> Mapping[str, Any]:
    """Return an exhaustive binding for every input used by static checks."""

    request = query.request
    if request.mode != "air_grasp":
        raise ValueError("static observation caching is only defined for air_grasp")
    model = request.hand_model
    plan_path = request.joint_plan_source_path
    if plan_path is None:
        raise ValueError("static cache requires a bound joint-plan manifest")
    hand_files = {
        "urdf": _file_binding(model.urdf_path),
        "mapping": _file_binding(model.mapping_path),
    }
    driver_mapping = model.mapping_path.parent / "driver_routine_to_angle.xls"
    hand_files["official_driver_mapping"] = _file_binding(driver_mapping)
    binding: Dict[str, Any] = {
        "binding_version": 1,
        "mode": request.mode,
        "cache_contract": {
            "implementation_path": str(Path(__file__).resolve()),
            "implementation_sha256": _sha256_file(Path(__file__).resolve()),
            "partition": "static-mesh-8-plus-fresh-point-10-v1",
        },
        "backend": _identity_binding(identity),
        "immutable_inputs": {
            "adapter": _file_binding(request.adapter_stl_path),
            "control_config": _file_binding(request.control_config_path),
            "official_snapshot": _file_binding(request.snapshot_source_path),
            "joint_plan": _file_binding(plan_path),
            "hand_model": hand_files,
            "hand_link_meshes": {
                str(name): _file_binding(path)
                for name, path in sorted(query.hand_link_mesh_paths.items())
            },
        },
        "candidate": {
            "selected_index": int(request.selected_candidate_index),
            "selected_targets": [
                int(value) for value in request.selected_hand_targets
            ],
            "selected_canonical_pose": _array_binding(
                request.selected_canonical_pose_base
            ),
            "selected_hand_pose": _array_binding(request.selected_hand_pose_base),
        },
        "mount_and_path": {
            "adapter_sha256_from_query": str(query.adapter_sha256),
            "T_EE_hand": _array_binding(request.T_EE_hand),
            "q_path": _array_binding(query.q_path_rad),
            "path_segments_sha256": _json_sha256(list(query.path_segments)),
            "max_q_tracking_error_rad": float(
                request.max_q_tracking_error_rad
            ),
            "hand_self_clearance_margin_m": float(
                request.hand_self_clearance_margin_m
            ),
        },
        "rh56_geometry": {
            "mesh_resolution": str(model.mesh_resolution),
            "feedback_envelope_policy_sha256": (
                rh56_feedback_envelope_policy(
                    request.hand_arrival_tolerance_units
                )["sha256"]
            ),
            "joint_topology": [
                {
                    "name": str(joint.name),
                    "parent": str(joint.parent),
                    "child": str(joint.child),
                }
                for joint in model.joints
            ],
            "open_link_visual": _transform_map_binding(
                query.T_hand_open_link_visual
            ),
            "closed_link_visual": _transform_map_binding(
                query.T_hand_closed_link_visual
            ),
            "waypoint_link_visual": [
                _transform_map_binding(item)
                for item in query.T_hand_waypoint_link_visual
            ],
            "dense_interval_q12": [
                _array_binding(item)
                for item in query.hand_dense_interval_q12_rad
            ],
            "feedback_tube_q12": [
                _array_binding(item)
                for item in query.hand_interval_feedback_tube_q12_rad
            ],
            "feedback_lower_q12": [
                _array_binding(item)
                for item in query.hand_interval_feedback_q12_lower_rad
            ],
            "feedback_upper_q12": [
                _array_binding(item)
                for item in query.hand_interval_feedback_q12_upper_rad
            ],
        },
        "excluded_fresh_inputs": {
            "scene_points": True,
            "object_points": True,
            "capture_timestamp": True,
            "audit_timestamp": True,
            "scene_filter_evidence": True,
        },
    }
    if len(binding["immutable_inputs"]["hand_link_meshes"]) != 13:
        raise ValueError("static cache must bind exactly 13 RH56 link meshes")
    return binding


def _observation_to_json(value: CollisionObservation) -> Mapping[str, Any]:
    return {
        "check_id": value.check_id,
        "authoritative": bool(value.authoritative),
        "tested_sample_indices": list(value.tested_sample_indices),
        "minimum_signed_distance_m": float(value.minimum_signed_distance_m),
        "observed_pairs": list(value.observed_pairs),
        "details": value.details,
    }


def _observation_from_json(value: Mapping[str, Any]) -> CollisionObservation:
    expected = {
        "check_id",
        "authoritative",
        "tested_sample_indices",
        "minimum_signed_distance_m",
        "observed_pairs",
        "details",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("static cache observation schema is invalid")
    return CollisionObservation(
        check_id=str(value["check_id"]),
        authoritative=value["authoritative"],
        tested_sample_indices=tuple(value["tested_sample_indices"]),
        minimum_signed_distance_m=float(value["minimum_signed_distance_m"]),
        observed_pairs=tuple(value["observed_pairs"]),
        details=value["details"],
    )


def build_static_cache_artifact(
    query: InstalledToolCollisionQuery,
    backend: Any,
) -> Mapping[str, Any]:
    """Evaluate and seal the scene-independent half of an air audit."""

    identity = backend.identity
    before = static_query_binding(query, identity)
    observations = backend.evaluate_checks(query, STATIC_AIR_CHECK_IDS)
    after = static_query_binding(query, identity)
    if before != after:
        raise RuntimeError("static collision inputs changed during evaluation")
    if tuple(observations) != STATIC_AIR_CHECK_IDS:
        raise RuntimeError("static backend returned incomplete or reordered checks")
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "binding": before,
        "binding_sha256": _json_sha256(before),
        "static_check_ids": list(STATIC_AIR_CHECK_IDS),
        "dynamic_check_ids_excluded": list(DYNAMIC_AIR_CHECK_IDS),
        "observations": [
            _observation_to_json(observations[item])
            for item in STATIC_AIR_CHECK_IDS
        ],
        "motion_authorized": False,
    }
    payload["integrity"] = {
        "algorithm": "sha256-canonical-json-without-integrity",
        "payload_sha256": _json_sha256(payload),
    }
    return payload


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key {!r}".format(key))
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant {!r} is forbidden".format(value))


def validate_static_cache_artifact(
    payload: Mapping[str, Any],
    query: InstalledToolCollisionQuery,
    identity: CollisionBackendIdentity,
) -> Mapping[str, CollisionObservation]:
    expected_root = {
        "schema_version",
        "artifact_type",
        "binding",
        "binding_sha256",
        "static_check_ids",
        "dynamic_check_ids_excluded",
        "observations",
        "motion_authorized",
        "integrity",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_root:
        raise ValueError("static cache root schema is invalid")
    integrity = payload["integrity"]
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {"algorithm", "payload_sha256"}
        or integrity.get("algorithm")
        != "sha256-canonical-json-without-integrity"
    ):
        raise ValueError("static cache integrity schema is invalid")
    unsealed = dict(payload)
    del unsealed["integrity"]
    if integrity.get("payload_sha256") != _json_sha256(unsealed):
        raise ValueError("static cache self checksum is invalid")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["artifact_type"] != ARTIFACT_TYPE
        or payload["motion_authorized"] is not False
        or payload["static_check_ids"] != list(STATIC_AIR_CHECK_IDS)
        or payload["dynamic_check_ids_excluded"]
        != list(DYNAMIC_AIR_CHECK_IDS)
    ):
        raise ValueError("static cache type, authority, or partition is invalid")
    required_ids = tuple(
        item.check_id for item in check_specs_for_mode("air_grasp")
    )
    if set(STATIC_AIR_CHECK_IDS).intersection(DYNAMIC_AIR_CHECK_IDS) or set(
        STATIC_AIR_CHECK_IDS
    ).union(DYNAMIC_AIR_CHECK_IDS) != set(required_ids):
        raise RuntimeError("compiled static/dynamic partition is incomplete")
    expected_binding = static_query_binding(query, identity)
    if payload["binding_sha256"] != _json_sha256(payload["binding"]):
        raise ValueError("static cache binding checksum is invalid")
    if payload["binding"] != expected_binding:
        raise ValueError("static cache does not match the exact current query")
    values = payload["observations"]
    if not isinstance(values, list) or len(values) != len(STATIC_AIR_CHECK_IDS):
        raise ValueError("static cache observation coverage is incomplete")
    observations = tuple(_observation_from_json(value) for value in values)
    if tuple(item.check_id for item in observations) != STATIC_AIR_CHECK_IDS:
        raise ValueError("static cache observations are missing or reordered")
    if any(item.authoritative is not True for item in observations):
        raise ValueError("static mesh observation must remain authoritative")
    return {item.check_id: item for item in observations}


def load_static_cache_artifact(
    path: Path,
    query: InstalledToolCollisionQuery,
    identity: CollisionBackendIdentity,
) -> Mapping[str, CollisionObservation]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("cannot load static collision cache: {}".format(exc)) from exc
    return validate_static_cache_artifact(payload, query, identity)


def write_static_cache_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            "static cache already exists; refusing to replace {}".format(
                destination
            )
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = destination.with_name(
        ".{}.{}.tmp".format(destination.name, os.getpid())
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails if destination appeared
        # concurrently; unlike os.replace it can never overwrite evidence.
        os.link(str(temporary), str(destination))
        temporary.unlink()
        directory_fd = os.open(
            str(destination.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


class StaticCacheCombinedBackend:
    """Supply cached static observations plus freshly evaluated point checks."""

    def __init__(self, backend: Any, cache_path: Path) -> None:
        self._backend = backend
        self._cache_path = Path(cache_path).expanduser().resolve()
        if not self._cache_path.is_file():
            raise FileNotFoundError(
                "static collision cache is missing: {}".format(self._cache_path)
            )
        self._cache_file_sha256 = _sha256_file(self._cache_path)
        try:
            header = json.loads(
                self._cache_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_constant,
            )
            self._cache_payload_sha256 = str(
                header["integrity"]["payload_sha256"]
            )
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("static collision cache header is invalid: {}".format(exc)) from exc
        if (
            len(self._cache_payload_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self._cache_payload_sha256
            )
        ):
            raise ValueError("static collision cache payload SHA-256 is invalid")
        native = _identity_binding(self._backend.identity)
        configuration = {
            "combiner_contract": "cached-static-8-plus-fresh-point-10-v1",
            "native_backend": native,
            "static_cache": {
                "path": str(self._cache_path),
                "file_sha256": self._cache_file_sha256,
                "payload_sha256": self._cache_payload_sha256,
            },
            "static_check_ids": list(STATIC_AIR_CHECK_IDS),
            "fresh_dynamic_check_ids": list(DYNAMIC_AIR_CHECK_IDS),
        }
        self._identity = CollisionBackendIdentity(
            name="static-cache-plus-fresh-pinocchio-hppfcl-fr3-v7-rh56",
            version="1",
            implementation_sha256=_sha256_file(Path(__file__).resolve()),
            configuration_sha256=_json_sha256(configuration),
        )
        self._provenance = configuration

    @property
    def identity(self) -> CollisionBackendIdentity:
        return self._identity

    def _with_phase(
        self,
        observation: CollisionObservation,
        *,
        phase: str,
    ) -> CollisionObservation:
        details = dict(observation.details or {})
        details["split_evaluation_provenance"] = {
            "phase": phase,
            "combiner_contract": self._provenance["combiner_contract"],
            "native_backend": dict(self._provenance["native_backend"]),
            "static_cache": (
                dict(self._provenance["static_cache"])
                if phase == "precomputed_static_mesh"
                else None
            ),
        }
        return CollisionObservation(
            check_id=observation.check_id,
            authoritative=observation.authoritative,
            tested_sample_indices=observation.tested_sample_indices,
            minimum_signed_distance_m=observation.minimum_signed_distance_m,
            observed_pairs=observation.observed_pairs,
            details=details,
        )

    def evaluate(
        self, query: InstalledToolCollisionQuery
    ) -> Mapping[str, CollisionObservation]:
        before = _sha256_file(self._cache_path)
        if before != self._cache_file_sha256:
            raise ValueError("static collision cache changed before composition")
        static_raw = load_static_cache_artifact(
            self._cache_path, query, self._backend.identity
        )
        static = {
            check_id: self._with_phase(
                observation, phase="precomputed_static_mesh"
            )
            for check_id, observation in static_raw.items()
        }
        dynamic_raw = self._backend.evaluate_checks(query, DYNAMIC_AIR_CHECK_IDS)
        dynamic = {
            check_id: self._with_phase(
                observation, phase="fresh_scene_object"
            )
            for check_id, observation in dynamic_raw.items()
        }
        after = _sha256_file(self._cache_path)
        if after != before:
            raise ValueError("static collision cache changed during composition")
        if tuple(dynamic) != DYNAMIC_AIR_CHECK_IDS:
            raise RuntimeError("dynamic backend returned incomplete or reordered checks")
        overlap = set(static).intersection(dynamic)
        if overlap:
            raise RuntimeError("static/dynamic collision observations overlap")
        required = tuple(
            item.check_id for item in check_specs_for_mode(query.request.mode)
        )
        combined = {**static, **dynamic}
        if set(combined) != set(required):
            raise RuntimeError("combined collision observations are incomplete")
        return {check_id: combined[check_id] for check_id in required}


class StaticCacheBuildingBackend:
    """Capture the exact query produced by the normal audit request builder."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.artifact: Mapping[str, Any] | None = None

    @property
    def identity(self) -> CollisionBackendIdentity:
        return self._backend.identity

    def evaluate(
        self, query: InstalledToolCollisionQuery
    ) -> Mapping[str, CollisionObservation]:
        if self.artifact is not None:
            raise RuntimeError("static cache backend was evaluated more than once")
        self.artifact = build_static_cache_artifact(query, self._backend)
        observations = tuple(
            _observation_from_json(value)
            for value in self.artifact["observations"]
        )
        return {item.check_id: item for item in observations}


__all__ = [
    "ARTIFACT_TYPE",
    "DYNAMIC_AIR_CHECK_IDS",
    "SCHEMA_VERSION",
    "STATIC_AIR_CHECK_IDS",
    "StaticCacheBuildingBackend",
    "StaticCacheCombinedBackend",
    "build_static_cache_artifact",
    "load_static_cache_artifact",
    "static_query_binding",
    "validate_static_cache_artifact",
    "write_static_cache_artifact",
]
