"""Fail-closed, hardware-independent installed FR3/RH56 collision audit.

The module is intentionally split from robot execution.  It binds every
collision claim to immutable inputs (the exact V7 adapter, mount transform,
fresh scene, capture/current joints, sampled path, and all 13 Inspire link
meshes) and asks an injected collision backend only for geometric
observations.  Pass/fail is derived here from fixed policies and complete
sample coverage.  Unavailable or incomplete evidence always fails.  Captured
point-cloud checks never gain geometric authority: only an unloaded air audit
may pass them conditionally, with an explicit runtime workspace-clear token;
the loaded/contact audit remains fail-closed on non-authoritative evidence.

The JSON artifact is a replay/audit contract, not a motion command.  Importing
this module opens no camera, serial port, FCI connection, or GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from .control_plan import validate_rigid_transform
from .inspire_hand_model import InspireHandModel
from .inspire_open_configuration import (
    OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
    official_open_configuration_provenance,
)
from .joint_path_sampling import (
    CANONICAL_JOINT_SAMPLING_ALGORITHM,
    canonical_joint_path_samples,
)
from .rh56_actuator_mapping import OfficialRH56ActuatorMapper
from .rh56_hand_path import (
    dense_hand_interval_sha256,
    dense_hand_interval_tubes_sha256,
    feedback_q12_envelopes_sha256,
    loaded_hand_execution_path_not_applicable,
    rh56_feedback_envelope_policy,
    rh56_hand_interval_feedback_q12_envelopes,
    rh56_hand_interval_q12_paths_and_feedback_tubes,
    validate_loaded_hand_execution_path_not_applicable,
    validate_rh56_feedback_envelope_policy,
    validate_rh56_hand_execution_path,
)


SCHEMA_VERSION = 2
ARTIFACT_TYPE = "fr3_rh56_installed_tool_collision_audit"
RUNTIME_WORKSPACE_CLEAR_CONDITION_ID = (
    "runtime_operator_workspace_clear_required"
)
RUNTIME_WORKSPACE_CLEAR_TOKEN = "FR3_RH56_WORKSPACE_CLEAR"
V7_ADAPTER_SHA256 = (
    "7fdd3dd06bd8dafed445f6a6910315edd3073bd1b9cd9015a951b6e415b947e1"
)
V7_T_EE_HAND = np.asarray(
    [
        [0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0],
        [0.0, -math.sqrt(0.5), math.sqrt(0.5), 0.0],
        [1.0, 0.0, 0.0, 0.010],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

FR3_JOINT_LIMITS = np.asarray(
    [
        [-2.7437, 2.7437],
        [-1.7837, 1.7837],
        [-2.9007, 2.9007],
        [-3.0421, -0.1518],
        [-2.8065, 2.8065],
        [0.5445, 4.5169],
        [-3.0159, 3.0159],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CollisionCheckSpec:
    check_id: str
    scope: str
    expectation: str
    coverage: str
    margin_policy: str


LOADED_GRASP_CHECK_SPECS: Tuple[CollisionCheckSpec, ...] = (
    CollisionCheckSpec(
        "fr3_self_path", "fr3_vs_fr3", "clear", "full_path", "robot"
    ),
    CollisionCheckSpec(
        "fr3_scene_path", "fr3_vs_scene", "clear", "full_path", "scene"
    ),
    CollisionCheckSpec(
        "adapter_fr3_path", "adapter_vs_fr3", "clear", "full_path", "robot"
    ),
    CollisionCheckSpec(
        "adapter_scene_path", "adapter_vs_scene", "clear", "full_path", "scene"
    ),
    CollisionCheckSpec(
        "adapter_object_path",
        "adapter_vs_object",
        "clear",
        "full_path",
        "object_approach",
    ),
    CollisionCheckSpec(
        "rh56_open_fr3_path", "open_rh56_vs_fr3", "clear", "full_path", "robot"
    ),
    CollisionCheckSpec(
        "rh56_open_scene_path", "open_rh56_vs_scene", "clear", "full_path", "scene"
    ),
    CollisionCheckSpec(
        "rh56_open_object_path",
        "open_rh56_vs_object",
        "clear",
        "full_path",
        "object_approach",
    ),
    CollisionCheckSpec(
        "rh56_closed_fr3_final", "closed_rh56_vs_fr3", "clear", "final", "robot"
    ),
    CollisionCheckSpec(
        "rh56_closed_adapter_final",
        "closed_rh56_vs_adapter",
        "clear",
        "final",
        "robot",
    ),
    CollisionCheckSpec(
        "rh56_closed_scene_final", "closed_rh56_vs_scene", "clear", "final", "scene"
    ),
    CollisionCheckSpec(
        "rh56_closed_object_noncontact_final",
        "closed_rh56_noncontact_links_vs_object",
        "clear",
        "final",
        "object_approach",
    ),
    CollisionCheckSpec(
        "rh56_closed_object_contact_final",
        "closed_rh56_vs_object",
        "object_contact",
        "final",
        "object_contact",
    ),
)
AIR_GRASP_CHECK_SPECS: Tuple[CollisionCheckSpec, ...] = (
    *LOADED_GRASP_CHECK_SPECS[:-1],
    CollisionCheckSpec(
        "rh56_closed_object_all_links_final",
        "closed_rh56_vs_object",
        "clear",
        "final",
        "object_approach",
    ),
    CollisionCheckSpec(
        "rh56_execution_fr3_final", "rh56_execution_vs_fr3", "clear",
        "hand_execution_path", "robot",
    ),
    CollisionCheckSpec(
        "rh56_execution_adapter_final", "rh56_execution_vs_adapter", "clear",
        "hand_execution_path", "robot",
    ),
    CollisionCheckSpec(
        "rh56_execution_scene_final", "rh56_execution_vs_scene", "clear",
        "hand_execution_path", "scene",
    ),
    CollisionCheckSpec(
        "rh56_execution_object_final", "rh56_execution_vs_object", "clear",
        "hand_execution_path", "object_approach",
    ),
    CollisionCheckSpec(
        "rh56_execution_self_final", "rh56_execution_self", "clear",
        "hand_execution_path", "hand_self",
    ),
)
# Backwards import spelling means the loaded/contact semantics only.
CHECK_SPECS = LOADED_GRASP_CHECK_SPECS
_CHECK_BY_ID = {
    item.check_id: item
    for item in (*LOADED_GRASP_CHECK_SPECS, *AIR_GRASP_CHECK_SPECS)
}

# These checks use captured point unions.  Their geometric result is useful,
# but a single camera view cannot prove that occluded workspace is empty.
# Only air_grasp may carry this limitation as an explicit runtime operator
# condition; loaded/contact execution may never do so.
CAPTURED_POINT_CHECK_IDS = frozenset(
    {
        "fr3_scene_path",
        "adapter_scene_path",
        "adapter_object_path",
        "rh56_open_scene_path",
        "rh56_open_object_path",
        "rh56_closed_scene_final",
        "rh56_closed_object_noncontact_final",
        "rh56_closed_object_contact_final",
        "rh56_closed_object_all_links_final",
        "rh56_execution_scene_final",
        "rh56_execution_object_final",
    }
)


def check_specs_for_mode(mode: str) -> Tuple[CollisionCheckSpec, ...]:
    if mode == "loaded_grasp":
        return LOADED_GRASP_CHECK_SPECS
    if mode == "air_grasp":
        return AIR_GRASP_CHECK_SPECS
    raise ValueError("audit mode must be loaded_grasp or air_grasp")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8")
    contiguous = np.ascontiguousarray(array)
    shape = ",".join(str(item) for item in contiguous.shape)
    digest = hashlib.sha256()
    digest.update(("dtype=<f8;shape={};".format(shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


V7_T_EE_HAND_SHA256 = _array_sha256(V7_T_EE_HAND)


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("{} must be a lowercase SHA-256 hex digest".format(name))
    return text


def _finite_vector(value: Sequence[float], length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError("{} must be a finite {}-vector".format(name, length))
    return vector.copy()


def _fr3_q(value: Sequence[float], name: str) -> np.ndarray:
    q = _finite_vector(value, 7, name)
    if np.any(q < FR3_JOINT_LIMITS[:, 0]) or np.any(q > FR3_JOINT_LIMITS[:, 1]):
        raise ValueError("{} is outside the FR3 joint limits".format(name))
    return q


def _points(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("{} must have shape (N,3) and be non-empty".format(name))
    if not np.all(np.isfinite(points)):
        raise ValueError("{} contains NaN or infinity".format(name))
    return points.copy()


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or (numeric < 0.0 if allow_zero else numeric <= 0.0):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError("{} must be finite and {}".format(name, comparator))
    return numeric


@dataclass(frozen=True)
class CollisionBackendIdentity:
    name: str
    version: str
    implementation_sha256: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        if not str(self.name).strip() or not str(self.version).strip():
            raise ValueError("collision backend name/version must be non-empty")
        _sha256_text(self.implementation_sha256, "implementation_sha256")
        _sha256_text(self.configuration_sha256, "configuration_sha256")


@dataclass(frozen=True)
class CollisionObservation:
    """Raw backend observation; policy pass/fail is deliberately absent."""

    check_id: str
    authoritative: bool
    tested_sample_indices: Tuple[int, ...]
    minimum_signed_distance_m: float
    observed_pairs: Tuple[str, ...] = ()
    details: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.check_id not in _CHECK_BY_ID:
            raise ValueError("unknown installed-tool check {!r}".format(self.check_id))
        if not isinstance(self.authoritative, (bool, np.bool_)):
            raise ValueError("authoritative must be boolean")
        indices = tuple(int(item) for item in self.tested_sample_indices)
        if any(item < 0 for item in indices) or len(set(indices)) != len(indices):
            raise ValueError("tested_sample_indices must be unique and non-negative")
        if indices != tuple(sorted(indices)):
            raise ValueError("tested_sample_indices must be sorted")
        distance = float(self.minimum_signed_distance_m)
        if not np.isfinite(distance):
            raise ValueError("minimum_signed_distance_m must be finite")
        pairs = tuple(str(item).strip() for item in self.observed_pairs)
        if any(not item for item in pairs):
            raise ValueError("observed_pairs entries must be non-empty")
        if self.details is not None:
            # This both checks JSON compatibility and rejects NaN/Infinity.
            json.dumps(self.details, allow_nan=False, sort_keys=True)
        object.__setattr__(self, "tested_sample_indices", indices)
        object.__setattr__(self, "minimum_signed_distance_m", distance)
        object.__setattr__(self, "observed_pairs", pairs)


@dataclass(frozen=True)
class InstalledToolAuditRequest:
    mode: str
    adapter_stl_path: Path
    T_EE_hand: np.ndarray
    control_config_path: Path
    snapshot_source_path: Path
    snapshot_schema_version: int
    snapshot_reference_frame: str
    snapshot_frame_id: int
    snapshot_timestamp_s: float
    snapshot_calibration_id: str
    snapshot_camera_serial: str
    snapshot_model_name: str
    snapshot_representation_checkpoint_sha256: str
    snapshot_decision_checkpoint_sha256s: Tuple[str, ...]
    snapshot_official_source_commit: str
    selected_candidate_index: int
    selected_canonical_pose_base: np.ndarray
    selected_hand_pose_base: np.ndarray
    selected_hand_targets: np.ndarray
    plan_hand_pose_base: np.ndarray
    plan_pregrasp_pose_base_EE: np.ndarray
    plan_grasp_pose_base_EE: np.ndarray
    air_retreat_distance_m: float
    pregrasp_distance_m: float
    scene_source_path: Path
    scene_points_base: np.ndarray
    scene_capture_q_rad: np.ndarray
    scene_captured_at_s: float
    current_q_rad: np.ndarray
    current_q_captured_at_s: float
    default_q_rad: np.ndarray
    pregrasp_q_rad: np.ndarray
    grasp_q_rad: np.ndarray
    object_source_path: Path
    object_points_base: np.ndarray
    hand_model: InspireHandModel
    open_hand_joint_positions_rad: np.ndarray
    open_actuator_targets: np.ndarray
    closure_hand_joint_positions_rad: np.ndarray
    closure_actuator_targets: np.ndarray
    audit_started_at_s: float
    hand_execution_path: Optional[Mapping[str, Any]] = None
    hand_waypoint_joint_positions_rad: Tuple[np.ndarray, ...] = ()
    hand_arrival_tolerance_units: int = 20
    default_transit_q_rad: Tuple[np.ndarray, ...] = ()
    approach_transit_q_rad: Tuple[np.ndarray, ...] = ()
    joint_plan_source_path: Optional[Path] = None
    scene_filter_evidence_path: Optional[Path] = None
    scene_excludes_object: bool = True
    allowed_object_contact_links: Tuple[str, ...] = (
        "Link11",
        "Link22",
        "Link33",
        "Link44",
        "Link53",
    )
    max_scene_age_s: float = 2.0
    max_current_q_age_s: float = 0.25
    max_joint_step_rad: float = 0.02
    max_q_tracking_error_rad: float = 0.002
    scene_voxel_resolution_m: float = 0.005
    observed_scene_scope: str = "calibrated_camera_frustum_voxel_grid"
    unknown_space_policy: str = "occupied"
    scene_clearance_margin_m: float = 0.005
    robot_clearance_margin_m: float = 0.002
    hand_self_clearance_margin_m: float = 0.0
    object_contact_max_distance_m: float = 0.002
    object_max_penetration_m: float = 0.003
    object_approach_clearance_margin_m: float = 0.002

    def __post_init__(self) -> None:
        if self.mode not in ("loaded_grasp", "air_grasp"):
            raise ValueError("mode must be loaded_grasp or air_grasp")
        object.__setattr__(self, "adapter_stl_path", Path(self.adapter_stl_path).expanduser().resolve())
        object.__setattr__(self, "control_config_path", Path(self.control_config_path).expanduser().resolve())
        object.__setattr__(self, "snapshot_source_path", Path(self.snapshot_source_path).expanduser().resolve())
        object.__setattr__(self, "scene_source_path", Path(self.scene_source_path).expanduser().resolve())
        object.__setattr__(self, "object_source_path", Path(self.object_source_path).expanduser().resolve())
        if self.scene_filter_evidence_path is not None:
            object.__setattr__(
                self,
                "scene_filter_evidence_path",
                Path(self.scene_filter_evidence_path).expanduser().resolve(),
            )
        if self.joint_plan_source_path is not None:
            object.__setattr__(
                self,
                "joint_plan_source_path",
                Path(self.joint_plan_source_path).expanduser().resolve(),
            )
        object.__setattr__(self, "T_EE_hand", validate_rigid_transform(self.T_EE_hand, "T_EE_hand"))
        if type(self.snapshot_schema_version) is not int or self.snapshot_schema_version != 2:
            raise ValueError("snapshot_schema_version must be exactly 2")
        if str(self.snapshot_reference_frame) != "robot_base":
            raise ValueError("snapshot_reference_frame must be robot_base")
        if type(self.snapshot_frame_id) is not int or self.snapshot_frame_id < 0:
            raise ValueError("snapshot_frame_id must be a non-negative integer")
        for name in (
            "snapshot_calibration_id",
            "snapshot_camera_serial",
            "snapshot_model_name",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError("{} must be a string".format(name))
        representation_hash = str(self.snapshot_representation_checkpoint_sha256)
        if representation_hash:
            _sha256_text(representation_hash, "snapshot_representation_checkpoint_sha256")
        decision_hashes = tuple(str(item) for item in self.snapshot_decision_checkpoint_sha256s)
        if len(decision_hashes) not in (0, 8):
            raise ValueError("snapshot_decision_checkpoint_sha256s must be empty or contain 8 hashes")
        for index, digest in enumerate(decision_hashes):
            _sha256_text(digest, "snapshot_decision_checkpoint_sha256s[{}]".format(index))
        object.__setattr__(self, "snapshot_decision_checkpoint_sha256s", decision_hashes)
        source_commit = str(self.snapshot_official_source_commit)
        if source_commit and (
            len(source_commit) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in source_commit)
        ):
            raise ValueError("snapshot_official_source_commit must be a 40- or 64-hex object ID")
        if type(self.selected_candidate_index) is not int or self.selected_candidate_index < 0:
            raise ValueError("selected_candidate_index must be a non-negative integer")
        object.__setattr__(
            self,
            "selected_canonical_pose_base",
            validate_rigid_transform(self.selected_canonical_pose_base, "selected_canonical_pose_base"),
        )
        object.__setattr__(
            self,
            "selected_hand_pose_base",
            validate_rigid_transform(self.selected_hand_pose_base, "selected_hand_pose_base"),
        )
        object.__setattr__(
            self,
            "plan_hand_pose_base",
            validate_rigid_transform(self.plan_hand_pose_base, "plan_hand_pose_base"),
        )
        object.__setattr__(
            self,
            "plan_pregrasp_pose_base_EE",
            validate_rigid_transform(self.plan_pregrasp_pose_base_EE, "plan_pregrasp_pose_base_EE"),
        )
        object.__setattr__(
            self,
            "plan_grasp_pose_base_EE",
            validate_rigid_transform(self.plan_grasp_pose_base_EE, "plan_grasp_pose_base_EE"),
        )
        selected_targets = _finite_vector(self.selected_hand_targets, 6, "selected_hand_targets")
        if (
            np.any(selected_targets < 0.0)
            or np.any(selected_targets > 1000.0)
            or not np.array_equal(selected_targets, np.rint(selected_targets))
        ):
            raise ValueError("selected_hand_targets must be integer registers in [0,1000]")
        object.__setattr__(self, "selected_hand_targets", selected_targets)
        composed_hand_pose = self.plan_grasp_pose_base_EE @ self.T_EE_hand
        if not np.allclose(
            composed_hand_pose,
            self.plan_hand_pose_base,
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "plan_hand_pose_base must equal plan_grasp_pose_base_EE @ T_EE_hand"
            )
        retreat = float(self.air_retreat_distance_m)
        if not np.isfinite(retreat) or retreat < 0.0:
            raise ValueError("air_retreat_distance_m must be finite and non-negative")
        approach = self.selected_canonical_pose_base[:3, 0]
        expected_hand = self.selected_hand_pose_base.copy()
        expected_hand[:3, 3] -= retreat * approach
        if not np.allclose(
            self.plan_hand_pose_base,
            expected_hand,
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "planned hand pose must be selected hand pose retreated along -approach"
            )
        if self.mode == "loaded_grasp" and retreat != 0.0:
            raise ValueError("loaded_grasp air_retreat_distance_m must be zero")
        if self.mode == "air_grasp" and retreat <= 0.0:
            raise ValueError("air_grasp requires a positive retreat distance")
        object.__setattr__(self, "air_retreat_distance_m", retreat)
        pregrasp_distance = float(self.pregrasp_distance_m)
        if not np.isfinite(pregrasp_distance) or pregrasp_distance <= 0.0:
            raise ValueError("pregrasp_distance_m must be finite and positive")
        expected_pregrasp = self.plan_grasp_pose_base_EE.copy()
        expected_pregrasp[:3, 3] -= pregrasp_distance * approach
        if not np.allclose(
            self.plan_pregrasp_pose_base_EE,
            expected_pregrasp,
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "plan pregrasp pose must be the planned grasp pose retreated "
                "by pregrasp_distance_m along -approach"
            )
        object.__setattr__(self, "pregrasp_distance_m", pregrasp_distance)
        object.__setattr__(self, "scene_points_base", _points(self.scene_points_base, "scene_points_base"))
        object.__setattr__(self, "object_points_base", _points(self.object_points_base, "object_points_base"))
        for name in (
            "scene_capture_q_rad",
            "current_q_rad",
            "default_q_rad",
            "pregrasp_q_rad",
            "grasp_q_rad",
        ):
            object.__setattr__(self, name, _fr3_q(getattr(self, name), name))
        transit = tuple(
            _fr3_q(value, "default_transit_q_rad[{}]".format(index))
            for index, value in enumerate(self.default_transit_q_rad)
        )
        object.__setattr__(self, "default_transit_q_rad", transit)
        approach_transit = tuple(
            _fr3_q(value, "approach_transit_q_rad[{}]".format(index))
            for index, value in enumerate(self.approach_transit_q_rad)
        )
        object.__setattr__(self, "approach_transit_q_rad", approach_transit)
        object.__setattr__(self, "open_hand_joint_positions_rad", _finite_vector(self.open_hand_joint_positions_rad, 12, "open_hand_joint_positions_rad"))
        object.__setattr__(self, "closure_hand_joint_positions_rad", _finite_vector(self.closure_hand_joint_positions_rad, 12, "closure_hand_joint_positions_rad"))
        open_targets = _finite_vector(self.open_actuator_targets, 6, "open_actuator_targets")
        if not np.array_equal(open_targets, np.full(6, 1000.0)):
            raise ValueError("open_actuator_targets must be exactly [1000]*6")
        object.__setattr__(self, "open_actuator_targets", open_targets)
        targets = _finite_vector(self.closure_actuator_targets, 6, "closure_actuator_targets")
        if (
            np.any(targets < 0.0)
            or np.any(targets > 1000.0)
            or not np.array_equal(targets, np.rint(targets))
        ):
            raise ValueError("closure_actuator_targets must be integer registers in [0,1000]")
        object.__setattr__(self, "closure_actuator_targets", targets)
        if not np.array_equal(selected_targets, targets):
            raise ValueError("selected_hand_targets must exactly equal closure_actuator_targets")
        if not isinstance(self.hand_model, InspireHandModel):
            raise TypeError("hand_model must be an InspireHandModel")
        if self.mode == "air_grasp":
            if self.hand_execution_path is None:
                raise ValueError("air grasp requires a canonical hand execution path")
            hand_path = validate_rh56_hand_execution_path(self.hand_execution_path)
            if hand_path.target != tuple(int(value) for value in targets):
                raise ValueError("hand execution path target differs from closure targets")
            mapper = OfficialRH56ActuatorMapper(
                self.hand_model.mapping_path.parent / "driver_routine_to_angle.xls"
            )
            hand_joints = tuple(
                mapper.to_joint_positions_rad(item.actuator_configuration)
                for item in hand_path.waypoints
            )
            normalized_hand_path = hand_path.as_dict()
        else:
            if self.hand_execution_path is not None:
                validate_loaded_hand_execution_path_not_applicable(
                    self.hand_execution_path
                )
            normalized_hand_path = loaded_hand_execution_path_not_applicable()
            hand_joints = ()
        object.__setattr__(self, "hand_execution_path", normalized_hand_path)
        object.__setattr__(self, "hand_waypoint_joint_positions_rad", hand_joints)
        tolerance_units = self.hand_arrival_tolerance_units
        if (
            isinstance(tolerance_units, (bool, np.bool_))
            or int(tolerance_units) != float(tolerance_units)
            or not 0 <= int(tolerance_units) <= 100
        ):
            raise ValueError("hand_arrival_tolerance_units must be an integer in [0,100]")
        object.__setattr__(self, "hand_arrival_tolerance_units", int(tolerance_units))
        if self.scene_excludes_object is not True:
            raise ValueError(
                "scene_excludes_object must be true; object contact has a separate policy"
            )
        allowed_links = tuple(str(item).strip() for item in self.allowed_object_contact_links)
        model_links = {link.name for link in self.hand_model.links}
        if not allowed_links or len(set(allowed_links)) != len(allowed_links):
            raise ValueError("allowed_object_contact_links must be non-empty and unique")
        if any(not item or item not in model_links for item in allowed_links):
            raise ValueError("allowed_object_contact_links contains an unknown hand link")
        object.__setattr__(self, "allowed_object_contact_links", allowed_links)
        for name in (
            "snapshot_timestamp_s",
            "scene_captured_at_s",
            "current_q_captured_at_s",
            "audit_started_at_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("{} must be finite and non-negative".format(name))
            object.__setattr__(self, name, value)
        for name in (
            "max_scene_age_s",
            "max_current_q_age_s",
            "max_joint_step_rad",
            "max_q_tracking_error_rad",
            "scene_clearance_margin_m",
            "robot_clearance_margin_m",
            "hand_self_clearance_margin_m",
            "object_contact_max_distance_m",
            "object_max_penetration_m",
            "object_approach_clearance_margin_m",
            "scene_voxel_resolution_m",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name, allow_zero=name.endswith("margin_m")))
        if self.max_joint_step_rad > 0.10:
            raise ValueError("max_joint_step_rad must not exceed 0.10 rad")
        if self.max_q_tracking_error_rad > 0.01:
            raise ValueError("max_q_tracking_error_rad must not exceed 0.01 rad")
        if self.observed_scene_scope != "calibrated_camera_frustum_voxel_grid":
            raise ValueError(
                "observed_scene_scope must be calibrated_camera_frustum_voxel_grid"
            )
        if self.unknown_space_policy != "occupied":
            raise ValueError("unknown_space_policy must be occupied")


@dataclass(frozen=True)
class InstalledToolCollisionQuery:
    request: InstalledToolAuditRequest
    q_path_rad: np.ndarray
    path_segments: Tuple[Mapping[str, Any], ...]
    adapter_sha256: str
    hand_link_mesh_paths: Mapping[str, Path]
    T_hand_open_link_visual: Mapping[str, np.ndarray]
    T_hand_closed_link_visual: Mapping[str, np.ndarray]
    T_hand_waypoint_link_visual: Tuple[Mapping[str, np.ndarray], ...]
    hand_dense_interval_q12_rad: Tuple[np.ndarray, ...]
    hand_interval_feedback_tube_q12_rad: Tuple[np.ndarray, ...]
    hand_interval_feedback_q12_lower_rad: Tuple[np.ndarray, ...]
    hand_interval_feedback_q12_upper_rad: Tuple[np.ndarray, ...]


class InstalledToolCollisionBackend(Protocol):
    @property
    def identity(self) -> CollisionBackendIdentity:
        ...

    def evaluate(
        self, query: InstalledToolCollisionQuery
    ) -> Mapping[str, CollisionObservation]:
        ...


def build_joint_path(
    current_q_rad: Sequence[float],
    default_q_rad: Sequence[float],
    pregrasp_q_rad: Sequence[float],
    grasp_q_rad: Sequence[float],
    *,
    max_joint_step_rad: float,
    default_transit_q_rad: Sequence[Sequence[float]] = (),
    approach_transit_q_rad: Sequence[Sequence[float]] = (),
) -> Tuple[np.ndarray, Tuple[Mapping[str, Any], ...]]:
    """Sample the exact audited joint polyline without boundary gaps.

    ``default_transit_q_rad`` is deliberately explicit.  Installed-tool
    clearance can depend on a bent transit path, so a collision audit of a
    direct current-to-default chord must never be reused for an execution that
    travels through different waypoints (or vice versa).
    """

    maximum_step = _positive(max_joint_step_rad, "max_joint_step_rad")
    ordered = [("current", _fr3_q(current_q_rad, "current_q_rad"))]
    ordered.extend(
        (
            "default_transit_{}".format(index),
            _fr3_q(value, "default_transit_q_rad[{}]".format(index)),
        )
        for index, value in enumerate(default_transit_q_rad)
    )
    ordered.append(("default", _fr3_q(default_q_rad, "default_q_rad")))
    ordered.extend(
        (
            "approach_transit_{}".format(index),
            _fr3_q(value, "approach_transit_q_rad[{}]".format(index)),
        )
        for index, value in enumerate(approach_transit_q_rad)
    )
    ordered.extend(
        (
            ("pregrasp", _fr3_q(pregrasp_q_rad, "pregrasp_q_rad")),
            ("grasp", _fr3_q(grasp_q_rad, "grasp_q_rad")),
        )
    )
    waypoints = tuple(
        (
            "{}_to_{}".format(start_name, end_name),
            start,
            end,
        )
        for (start_name, start), (end_name, end) in zip(ordered[:-1], ordered[1:])
    )
    path, interval_counts = canonical_joint_path_samples(
        tuple(item[1] for item in ordered), maximum_step
    )
    segments = []
    cursor = 0
    for (name, _start, _end), intervals in zip(waypoints, interval_counts):
        start_index = cursor
        cursor += int(intervals)
        segments.append(
            {
                "name": name,
                "start_index": int(start_index),
                "end_index": int(cursor),
            }
        )
    if len(path) < 4:
        raise AssertionError("joint path unexpectedly lost a waypoint")
    observed = float(np.max(np.abs(np.diff(path, axis=0))))
    if observed > maximum_step + 1e-12:
        raise AssertionError("joint path sampling exceeded max_joint_step_rad")
    return path, tuple(segments)


def _bind_hand_model(
    model: InspireHandModel,
    open_joints: np.ndarray,
    closed_joints: np.ndarray,
    hand_waypoint_joints: Sequence[np.ndarray],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Path],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Tuple[Dict[str, np.ndarray], ...],
]:
    if len(model.links) != 13 or len({link.name for link in model.links}) != 13:
        raise ValueError("Inspire hand model must contain exactly 13 unique links")
    mesh_directory = model.urdf_path.parent.parent / (
        "meshes" if model.mesh_resolution == "full" else "meshes_simplified"
    )
    paths: Dict[str, Path] = {}
    link_entries = []
    for link in model.links:
        path = (mesh_directory / link.mesh_filename).resolve()
        if not path.is_file():
            raise FileNotFoundError("Inspire link mesh not found: {}".format(path))
        paths[link.name] = path
        link_entries.append(
            {
                "name": link.name,
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        )
    open_transforms = model.link_mesh_transforms(np.eye(4), open_joints)
    closed_transforms = model.link_mesh_transforms(np.eye(4), closed_joints)
    waypoint_transforms = tuple(
        model.link_mesh_transforms(np.eye(4), joints)
        for joints in hand_waypoint_joints
    )
    ordered_open = np.stack([open_transforms[item["name"]] for item in link_entries])
    ordered_closed = np.stack([closed_transforms[item["name"]] for item in link_entries])
    waypoint_fk_hashes = [
        _array_sha256(np.stack([transforms[item["name"]] for item in link_entries]))
        for transforms in waypoint_transforms
    ]
    if not np.array_equal(open_joints, OFFICIAL_OPEN_JOINT_POSITIONS_RAD):
        raise ValueError(
            "open RH56 q12 must equal the official [1000]*6 actuator conversion"
        )
    anydex_root = model.mapping_path.parents[2]
    open_provenance = official_open_configuration_provenance(anydex_root)
    binding = {
        "urdf_path": str(model.urdf_path),
        "urdf_sha256": _sha256_file(model.urdf_path),
        "mapping_path": str(model.mapping_path),
        "mapping_sha256": _sha256_file(model.mapping_path),
        "mesh_resolution": model.mesh_resolution,
        "link_count": 13,
        "links": link_entries,
        "open_joint_positions_rad": open_joints.tolist(),
        "open_fk_sha256": _array_sha256(ordered_open),
        "open_configuration_provenance": open_provenance,
        "closure_joint_positions_rad": closed_joints.tolist(),
        "closure_fk_sha256": _array_sha256(ordered_closed),
        "execution_waypoint_joint_positions_rad": [
            np.asarray(value, dtype=np.float64).tolist()
            for value in hand_waypoint_joints
        ],
        "execution_waypoint_fk_sha256s": waypoint_fk_hashes,
        "actuator_mapping_provenance": OfficialRH56ActuatorMapper(
            model.mapping_path.parent / "driver_routine_to_angle.xls"
        ).provenance(),
    }
    return binding, paths, open_transforms, closed_transforms, waypoint_transforms


def _expected_indices(
    spec: CollisionCheckSpec, path_length: int, hand_path_length: int
) -> Tuple[int, ...]:
    if spec.coverage == "full_path":
        return tuple(range(path_length))
    if spec.coverage == "final":
        return (path_length - 1,)
    if spec.coverage == "hand_execution_path":
        return tuple(range(hand_path_length))
    raise AssertionError("unknown coverage policy")


def _scene_filter_evidence_binding(
    path: Optional[Path],
    *,
    scene_points: np.ndarray,
    object_points: np.ndarray,
    snapshot_sha256: str,
    adapter_sha256: str,
    expected_point_half_extent_m: float,
    expected_inflation_margin_m: float,
) -> Tuple[Optional[Mapping[str, Any]], str]:
    """Validate the replayable filter sidecar without promoting its authority."""

    if path is None:
        return None, "air audit requires installed-scene filter evidence"
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, "cannot load installed-scene filter evidence: {}".format(exc)
    required = {
        "schema_version",
        "artifact_type",
        "filtered_scene_points_sha256",
        "object_points_sha256",
        "installed_return_filter",
        "object_alignment_and_filter",
        "snapshot_sha256",
        "adapter_sha256",
        "open_hand_fk_commissioned",
        "unknown_space_policy_applied",
        "authoritative_for_unseen_camera_space",
        "motion_authorized",
    }
    missing = sorted(required - set(payload))
    if missing:
        return None, "installed-scene filter evidence is missing {}".format(missing)
    installed_filter = payload.get("installed_return_filter")
    if not isinstance(installed_filter, Mapping):
        return None, "installed-scene filter evidence has no filter policy"
    installed_filter_required = {
        "method",
        "fr3_geometry_model",
        "fr3_visual_meshes",
        "point_half_extent_m",
        "inflation_margin_m",
        "candidate_mesh_point_tests",
        "reason_counts",
        "residual_self_return_filter",
    }
    if set(installed_filter) != installed_filter_required:
        return None, "installed-scene FR3 visual filter schema is invalid"
    visual_meshes = installed_filter.get("fr3_visual_meshes")
    if not isinstance(visual_meshes, list) or len(visual_meshes) != 8:
        return None, "installed-scene FR3 visual mesh provenance is invalid"
    for index, entry in enumerate(visual_meshes):
        if not isinstance(entry, Mapping) or set(entry) != {"link", "path", "sha256"}:
            return None, "installed-scene FR3 visual mesh provenance is invalid"
        mesh_path = Path(str(entry.get("path", ""))).expanduser().resolve()
        if (
            entry.get("link") != "link{}".format(index)
            or not mesh_path.is_file()
            or entry.get("sha256") != _sha256_file(mesh_path)
        ):
            return None, "installed-scene FR3 visual mesh provenance is invalid"
    point_half_extent = installed_filter.get("point_half_extent_m")
    candidate_tests = installed_filter.get("candidate_mesh_point_tests")
    reason_counts = installed_filter.get("reason_counts")
    residual = installed_filter.get("residual_self_return_filter")
    residual_required = {
        "enabled",
        "failure",
        "method",
        "distance_method",
        "maximum_model_surface_distance_m",
        "hard_maximum_model_surface_distance_m",
        "calibration_alignment_maximum_m",
        "calibration_guard_m",
        "calibration_bound_formula",
        "connectivity_radius_m",
        "color_space",
        "palette_color_max_l2",
        "maximum_geodesic_m",
        "model_candidate_test_count",
        "model_candidate_indices_sha256",
        "model_candidate_surface_distances_sha256",
        "exact_seed_count",
        "model_neighborhood_candidate_count",
        "appearance_palette_candidate_count",
        "connected_candidate_count",
        "residual_removed_count",
        "residual_removed_indices_sha256",
        "residual_removed_surface_distance_min_m",
        "residual_removed_surface_distance_max_m",
        "reason_counts",
    }
    residual_valid = isinstance(residual, Mapping) and set(residual) == residual_required
    if residual_valid:
        residual_numeric = (
            "maximum_model_surface_distance_m",
            "hard_maximum_model_surface_distance_m",
            "calibration_alignment_maximum_m",
            "calibration_guard_m",
            "connectivity_radius_m",
            "palette_color_max_l2",
            "maximum_geodesic_m",
        )
        residual_counts = (
            "model_candidate_test_count",
            "exact_seed_count",
            "model_neighborhood_candidate_count",
            "appearance_palette_candidate_count",
            "connected_candidate_count",
            "residual_removed_count",
        )
        residual_valid = (
            residual.get("enabled") is True
            and residual.get("failure") == ""
            and residual.get("method")
            == "strict HPP-FCL point-to-mesh neighborhood AND same-component normalized-sRGB seed palette AND bounded same-component 3-D geodesic"
            and residual.get("distance_method")
            == "HPP-FCL triangle-mesh to 1nm sphere, radius corrected"
            and residual.get("color_space") == "normalized_sRGB_euclidean"
            and residual.get("calibration_bound_formula")
            == "min(hard_maximum, object_to_live_maximum + calibration_guard)"
            and all(
                not isinstance(residual.get(key), bool)
                and isinstance(residual.get(key), (int, float))
                and np.isfinite(float(residual[key]))
                and float(residual[key]) > 0.0
                for key in residual_numeric
            )
            and np.isclose(float(residual["hard_maximum_model_surface_distance_m"]), 0.020, atol=1e-15, rtol=0.0)
            and np.isclose(float(residual["calibration_guard_m"]), 0.002, atol=1e-15, rtol=0.0)
            and np.isclose(
                float(residual["maximum_model_surface_distance_m"]),
                min(
                    float(residual["hard_maximum_model_surface_distance_m"]),
                    float(residual["calibration_alignment_maximum_m"])
                    + float(residual["calibration_guard_m"]),
                ),
                atol=1e-15, rtol=0.0,
            )
            and np.isclose(
                float(residual["connectivity_radius_m"]), 0.010,
                atol=1e-15, rtol=0.0,
            )
            and np.isclose(
                float(residual["palette_color_max_l2"]), 0.18,
                atol=1e-15, rtol=0.0,
            )
            and np.isclose(
                float(residual["maximum_geodesic_m"]), 0.040,
                atol=1e-15, rtol=0.0,
            )
            and all(type(residual.get(key)) is int and residual[key] >= 0 for key in residual_counts)
            and residual["residual_removed_count"]
            <= residual["appearance_palette_candidate_count"]
            <= residual["model_neighborhood_candidate_count"]
            and residual["connected_candidate_count"]
            <= residual["model_neighborhood_candidate_count"]
            and isinstance(residual.get("reason_counts"), Mapping)
            and all(
                isinstance(key, str)
                and bool(key)
                and type(value) is int
                and value > 0
                for key, value in residual["reason_counts"].items()
            )
            and sum(residual["reason_counts"].values())
            == residual["residual_removed_count"]
            and all(
                isinstance(residual.get(key), str)
                and len(residual[key]) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in residual[key]
                )
                for key in (
                    "model_candidate_indices_sha256",
                    "model_candidate_surface_distances_sha256",
                    "residual_removed_indices_sha256",
                )
            )
        )
        lower_distance = residual.get("residual_removed_surface_distance_min_m")
        upper_distance = residual.get("residual_removed_surface_distance_max_m")
        if residual.get("residual_removed_count") == 0:
            residual_valid = residual_valid and lower_distance is None and upper_distance is None
        else:
            residual_valid = residual_valid and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(float(value))
                for value in (lower_distance, upper_distance)
            )
            if residual_valid:
                residual_valid = (
                    float(lower_distance) <= float(upper_distance)
                    and float(upper_distance)
                    <= float(residual["maximum_model_surface_distance_m"]) + 1e-12
                )
    if (
        installed_filter.get("fr3_geometry_model")
        != "official_fr3_visual_triangle_meshes"
        or installed_filter.get("method")
        != "official FR3 visual shell/V7/open-RH56 triangle mesh versus inflated point cube"
        or isinstance(point_half_extent, bool)
        or not isinstance(point_half_extent, (int, float))
        or not np.isfinite(float(point_half_extent))
        or float(point_half_extent) <= 0.0
        or not np.isclose(
            float(point_half_extent),
            float(expected_point_half_extent_m),
            atol=1e-15,
            rtol=0.0,
        )
        or type(candidate_tests) is not int
        or candidate_tests < 0
        or not isinstance(reason_counts, Mapping)
        or any(
            not isinstance(key, str)
            or not key
            or type(value) is not int
            or value <= 0
            for key, value in reason_counts.items()
        )
        or not residual_valid
    ):
        return None, "installed-scene FR3 visual filter policy is invalid"
    alignment = payload.get("object_alignment_and_filter")
    if not isinstance(alignment, Mapping):
        return None, "installed-scene filter evidence has no object alignment"
    margin = installed_filter.get("inflation_margin_m")
    alignment_required = {
        "method",
        "object_point_count",
        "live_scene_point_count",
        "object_to_live_median_m",
        "object_to_live_p95_m",
        "object_to_live_maximum_m",
        "alignment_coverage_distance_m",
        "alignment_coverage_fraction",
        "object_return_max_distance_m",
        "expanded_object_aabb_min_m",
        "expanded_object_aabb_max_m",
        "object_return_count_before_installed_precedence",
        "alignment_median_max_m",
        "alignment_p95_max_m",
        "alignment_minimum_coverage",
        "passed",
        "failures",
    }
    if set(alignment) != alignment_required:
        return None, "installed-scene object alignment schema is invalid"
    numeric_alignment = (
        "object_to_live_median_m",
        "object_to_live_p95_m",
        "object_to_live_maximum_m",
        "alignment_coverage_distance_m",
        "alignment_coverage_fraction",
        "object_return_max_distance_m",
        "alignment_median_max_m",
        "alignment_p95_max_m",
        "alignment_minimum_coverage",
    )
    if any(
        isinstance(alignment.get(key), bool)
        or not isinstance(alignment.get(key), (int, float))
        or not np.isfinite(float(alignment[key]))
        or float(alignment[key]) < 0.0
        for key in numeric_alignment
    ):
        return None, "installed-scene object alignment numbers are invalid"
    try:
        aabb_min = _finite_vector(
            alignment["expanded_object_aabb_min_m"], 3, "object alignment AABB min"
        )
        aabb_max = _finite_vector(
            alignment["expanded_object_aabb_max_m"], 3, "object alignment AABB max"
        )
    except (TypeError, ValueError):
        return None, "installed-scene object alignment AABB is invalid"
    alignment_counts_valid = (
        type(alignment.get("object_point_count")) is int
        and alignment["object_point_count"] == len(object_points)
        and type(alignment.get("live_scene_point_count")) is int
        and alignment["live_scene_point_count"] > 0
        and type(alignment.get("object_return_count_before_installed_precedence"))
        is int
        and alignment["object_return_count_before_installed_precedence"] >= 0
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type")
        != "installed_scene_return_filter_evidence"
        or payload.get("snapshot_sha256") != snapshot_sha256
        or payload.get("adapter_sha256") != adapter_sha256
        or payload.get("open_hand_fk_commissioned") is not True
        or payload.get("unknown_space_policy_applied") is not False
        or payload.get("authoritative_for_unseen_camera_space") is not False
        or payload.get("motion_authorized") is not False
        or isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not np.isfinite(float(margin))
        or not 0.0 <= float(margin) <= 0.02
        or not np.isclose(
            float(margin),
            float(expected_inflation_margin_m),
            atol=1e-15,
            rtol=0.0,
        )
        or payload.get("object_points_sha256") != _array_sha256(object_points)
        or alignment.get("passed") is not True
        or alignment.get("failures") != []
        or float(alignment["object_to_live_median_m"]) > 0.008
        or float(alignment["object_to_live_p95_m"]) > 0.015
        or float(alignment["alignment_coverage_fraction"]) < 0.80
        or float(alignment["alignment_median_max_m"]) > 0.008
        or float(alignment["alignment_p95_max_m"]) > 0.015
        or float(alignment["alignment_minimum_coverage"]) < 0.80
        or alignment.get("method")
        != "bidirectional_cKDTree_with_expanded_object_AABB"
        or not alignment_counts_valid
        or float(alignment["alignment_coverage_fraction"]) > 1.0
        or float(alignment["alignment_minimum_coverage"]) > 1.0
        or float(alignment["object_to_live_median_m"])
        > float(alignment["alignment_median_max_m"])
        or float(alignment["object_to_live_p95_m"])
        > float(alignment["alignment_p95_max_m"])
        or float(alignment["object_to_live_p95_m"])
        < float(alignment["object_to_live_median_m"])
        or float(alignment["object_to_live_maximum_m"])
        < float(alignment["object_to_live_p95_m"])
        or not np.isclose(
            float(residual["calibration_alignment_maximum_m"]),
            float(alignment["object_to_live_maximum_m"]),
            atol=1e-15,
            rtol=0.0,
        )
        or np.any(aabb_min > aabb_max)
    ):
        return None, "installed-scene filter evidence policy/provenance is invalid"
    if payload["filtered_scene_points_sha256"] != _array_sha256(scene_points):
        return None, "installed-scene filter points differ from the audited scene"
    return (
        {
            "path": str(source),
            "sha256": _sha256_file(source),
            "sha256_after_audit": "",
            "artifact_type": payload["artifact_type"],
            "filtered_scene_points_sha256": payload[
                "filtered_scene_points_sha256"
            ],
            "object_points_sha256": payload["object_points_sha256"],
            "object_alignment_and_filter": json.loads(
                json.dumps(
                    dict(alignment),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            ),
            "installed_return_filter": json.loads(
                json.dumps(
                    dict(installed_filter),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            ),
            "installed_inflation_margin_m": float(margin),
            "unknown_space_policy_applied": False,
            "authoritative_for_unseen_camera_space": False,
            "motion_authorized": False,
        },
        "",
    )


def _evaluate_observation(
    spec: CollisionCheckSpec,
    observation: Optional[CollisionObservation],
    path_length: int,
    hand_path_length: int,
    policies: Mapping[str, Any],
    *,
    mode: str,
) -> Dict[str, Any]:
    expected = _expected_indices(spec, path_length, hand_path_length)
    failures = []
    runtime_condition_ids = []
    conditional_point_evidence = bool(
        mode == "air_grasp"
        and spec.check_id in CAPTURED_POINT_CHECK_IDS
        and policies["runtime_operator_workspace_clear"]["required"] is True
    )
    if observation is None:
        failures.append("backend did not return this required check")
        authoritative = False
        tested: Tuple[int, ...] = ()
        distance: Optional[float] = None
        pairs: Tuple[str, ...] = ()
        details: Mapping[str, Any] = {"backend_status": "missing"}
    else:
        authoritative = bool(observation.authoritative)
        tested = observation.tested_sample_indices
        distance = float(observation.minimum_signed_distance_m)
        pairs = observation.observed_pairs
        details = json.loads(
            json.dumps(
                dict(observation.details or {}),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        if observation.check_id != spec.check_id:
            failures.append("backend check_id does not match requested check")
    if not authoritative:
        if conditional_point_evidence:
            runtime_condition_ids.append(RUNTIME_WORKSPACE_CLEAR_CONDITION_ID)
        else:
            failures.append("observation is not authoritative")
    if tested != expected:
        failures.append("sample coverage is incomplete or out of order")

    if distance is None:
        failures.append("minimum signed distance is unavailable")
    elif spec.expectation == "clear":
        margin = float(policies[spec.margin_policy])
        if distance <= margin:
            failures.append(
                "minimum signed distance {:.9f}m does not exceed {:.9f}m margin".format(
                    distance, margin
                )
            )
        if pairs:
            failures.append("collision pairs were observed in a clear-required check")
    elif spec.expectation == "object_contact":
        if distance > float(policies["object_contact_max_distance_m"]):
            failures.append("final hand is farther than the object-contact threshold")
        if distance < -float(policies["object_max_penetration_m"]):
            failures.append("final hand/object penetration exceeds the allowed bound")
    else:
        raise AssertionError("unknown collision expectation")

    reported_tracking_error = details.get("max_q_tracking_error_rad")
    if (
        isinstance(reported_tracking_error, bool)
        or not isinstance(reported_tracking_error, (int, float))
        or not np.isfinite(float(reported_tracking_error))
        or not np.isclose(
            float(reported_tracking_error),
            float(policies["max_q_tracking_error_rad"]),
            atol=1e-15,
            rtol=0.0,
        )
    ):
        failures.append("backend did not bind the configured joint tracking uncertainty")
    if details.get("joint_tracking_uncertainty_applied") is not True:
        failures.append("backend did not apply joint tracking uncertainty conservatively")
    if details.get("continuous_segment_envelope_verified") is not True:
        failures.append("backend did not verify the continuous joint-path envelope")
    conservative_motion_bound = details.get("conservative_motion_bound_m")
    if (
        isinstance(conservative_motion_bound, bool)
        or not isinstance(conservative_motion_bound, (int, float))
        or not np.isfinite(float(conservative_motion_bound))
        or float(conservative_motion_bound) < 0.0
    ):
        failures.append("backend did not report a conservative per-check motion bound")
    if details.get("minimum_distance_is_after_motion_bound") is not True:
        failures.append("reported distance does not conservatively include the motion bound")
    if spec.coverage == "hand_execution_path":
        if details.get("continuous_inter_waypoint_collision_claimed") is not True:
            failures.append("backend did not verify the continuous RH56 command intervals")
        if details.get("hand_interval_envelope_method") != (
            "official XLS every 1 register; accumulated absolute q12 variation "
            "times configuration-independent URDF serial-chain link radii; "
            "all-six arrival feedback tube; adaptive exact-mesh q12 feedback-box "
            "subdivision for otherwise unresolved self-pairs; final arm tracking tube"
        ):
            failures.append("backend RH56 interval-envelope method is missing or unsupported")
        expected_tolerance = policies.get("hand_arrival_tolerance_units")
        if (
            type(details.get("hand_arrival_tolerance_units")) is not int
            or details.get("hand_arrival_tolerance_units") != expected_tolerance
        ):
            failures.append("backend RH56 feedback tube tolerance differs from policy")
        expected_feedback_policy = policies.get("hand_feedback_envelope")
        if (
            not isinstance(expected_feedback_policy, Mapping)
            or details.get("feedback_envelope_policy_sha256")
            != expected_feedback_policy.get("sha256")
        ):
            failures.append(
                "backend RH56 feedback-envelope policy hash differs from audit policy"
            )
    if spec.margin_policy == "hand_self":
        reported_margin = details.get("hand_self_clearance_margin_m")
        if (
            isinstance(reported_margin, bool)
            or not isinstance(reported_margin, (int, float))
            or not np.isfinite(float(reported_margin))
            or not np.isclose(
                float(reported_margin),
                float(policies["hand_self"]),
                atol=1e-15,
                rtol=0.0,
            )
        ):
            failures.append("backend RH56 self check did not bind its clearance policy")
        if details.get("all_nonadjacent_hand_link_pairs_checked") is not True:
            failures.append("backend did not check every non-adjacent RH56 link pair")
    if spec.margin_policy == "scene":
        observed_policy = policies["observed_scene_policy"]
        if details.get("observed_scene_scope") != observed_policy["scope"]:
            failures.append("backend observed-scene scope differs from policy")
        voxel_resolution = details.get("scene_voxel_resolution_m")
        if (
            isinstance(voxel_resolution, bool)
            or not isinstance(voxel_resolution, (int, float))
            or not np.isclose(
                float(voxel_resolution),
                float(observed_policy["voxel_resolution_m"]),
                atol=1e-15,
                rtol=0.0,
            )
        ):
            failures.append("backend scene voxel resolution differs from policy")
        if details.get("unknown_space_policy") != observed_policy["unknown_space_policy"]:
            failures.append("backend unknown-space policy differs from policy")
        if details.get("unknown_space_policy_applied") is not True:
            if conditional_point_evidence:
                runtime_condition_ids.append(
                    RUNTIME_WORKSPACE_CLEAR_CONDITION_ID
                )
            else:
                failures.append(
                    "backend did not conservatively apply the unknown-space policy"
                )

    runtime_condition_ids = list(dict.fromkeys(runtime_condition_ids))
    passed = not failures
    pass_basis = (
        "failed"
        if not passed
        else (
            RUNTIME_WORKSPACE_CLEAR_CONDITION_ID
            if runtime_condition_ids
            else "authoritative_geometry"
        )
    )

    return {
        "check_id": spec.check_id,
        "scope": spec.scope,
        "expectation": spec.expectation,
        "coverage": spec.coverage,
        "authoritative": authoritative,
        "tested_sample_indices": list(tested),
        "expected_sample_count": len(expected),
        "minimum_signed_distance_m": distance,
        "observed_pairs": list(pairs),
        "details": details,
        "runtime_condition_ids": runtime_condition_ids,
        "pass_basis": pass_basis,
        "passed": passed,
        "failures": failures,
    }


def build_installed_tool_collision_query(
    request: InstalledToolAuditRequest,
    *,
    adapter_sha256: Optional[str] = None,
) -> Tuple[InstalledToolCollisionQuery, Dict[str, Any]]:
    """Build the exact backend query without evaluating freshness or motion.

    Separating deterministic query construction lets the scene-independent
    half of an air audit be computed before the final fresh capture.  The
    returned query still binds the complete path and RH56 sweep; this helper
    does not evaluate a collision policy and never authorizes execution.
    """

    if not isinstance(request, InstalledToolAuditRequest):
        raise TypeError("request must be an InstalledToolAuditRequest")
    digest = (
        _sha256_file(request.adapter_stl_path)
        if adapter_sha256 is None
        else _sha256_text(adapter_sha256, "adapter_sha256")
    )
    q_path, path_segments = build_joint_path(
        request.current_q_rad,
        request.default_q_rad,
        request.pregrasp_q_rad,
        request.grasp_q_rad,
        max_joint_step_rad=request.max_joint_step_rad,
        default_transit_q_rad=request.default_transit_q_rad,
        approach_transit_q_rad=request.approach_transit_q_rad,
    )
    (
        hand_binding,
        mesh_paths,
        open_transforms,
        closed_transforms,
        hand_waypoint_transforms,
    ) = _bind_hand_model(
        request.hand_model,
        request.open_hand_joint_positions_rad,
        request.closure_hand_joint_positions_rad,
        request.hand_waypoint_joint_positions_rad,
    )
    if request.mode == "air_grasp":
        canonical_hand_path = validate_rh56_hand_execution_path(
            request.hand_execution_path
        )
        hand_mapper = OfficialRH56ActuatorMapper(
            request.hand_model.mapping_path.parent / "driver_routine_to_angle.xls"
        )
        dense_hand_intervals, hand_feedback_tubes = (
            rh56_hand_interval_q12_paths_and_feedback_tubes(
                canonical_hand_path,
                hand_mapper,
                request.hand_arrival_tolerance_units,
            )
        )
        hand_feedback_lower, hand_feedback_upper = (
            rh56_hand_interval_feedback_q12_envelopes(
                canonical_hand_path,
                hand_mapper,
                request.hand_arrival_tolerance_units,
            )
        )
        hand_binding["execution_dense_interval_mapping"] = {
            "algorithm": "official_xls_every_integer_register_v1",
            "arrival_tolerance_units": request.hand_arrival_tolerance_units,
            "feedback_envelope_policy_sha256": rh56_feedback_envelope_policy(
                request.hand_arrival_tolerance_units
            )["sha256"],
            "interval_sample_counts": [len(item) for item in dense_hand_intervals],
            "q12_intervals_sha256": dense_hand_interval_sha256(
                dense_hand_intervals
            ),
            "feedback_tube_q12_rad": [item.tolist() for item in hand_feedback_tubes],
            "feedback_tubes_sha256": dense_hand_interval_tubes_sha256(
                hand_feedback_tubes
            ),
            "feedback_q12_envelopes_sha256": feedback_q12_envelopes_sha256(
                hand_feedback_lower, hand_feedback_upper
            ),
        }
    else:
        dense_hand_intervals = ()
        hand_feedback_tubes = ()
        hand_feedback_lower = ()
        hand_feedback_upper = ()
        hand_binding["execution_dense_interval_mapping"] = {
            "algorithm": "not_applicable_loaded_grasp_v1",
            "arrival_tolerance_units": None,
            "feedback_envelope_policy_sha256": "",
            "interval_sample_counts": [],
            "q12_intervals_sha256": "",
            "feedback_tube_q12_rad": [],
            "feedback_tubes_sha256": "",
            "feedback_q12_envelopes_sha256": "",
        }
    query = InstalledToolCollisionQuery(
        request=request,
        q_path_rad=q_path.copy(),
        path_segments=path_segments,
        adapter_sha256=digest,
        hand_link_mesh_paths=mesh_paths,
        T_hand_open_link_visual=open_transforms,
        T_hand_closed_link_visual=closed_transforms,
        T_hand_waypoint_link_visual=hand_waypoint_transforms,
        hand_dense_interval_q12_rad=dense_hand_intervals,
        hand_interval_feedback_tube_q12_rad=hand_feedback_tubes,
        hand_interval_feedback_q12_lower_rad=hand_feedback_lower,
        hand_interval_feedback_q12_upper_rad=hand_feedback_upper,
    )
    return query, hand_binding


def run_installed_tool_audit(
    request: InstalledToolAuditRequest,
    backend: InstalledToolCollisionBackend,
) -> Dict[str, Any]:
    """Run an injected offline backend and return one strict JSON artifact."""

    if not isinstance(request, InstalledToolAuditRequest):
        raise TypeError("request must be an InstalledToolAuditRequest")
    identity = backend.identity
    if not isinstance(identity, CollisionBackendIdentity):
        raise TypeError("backend.identity must be CollisionBackendIdentity")
    check_specs = check_specs_for_mode(request.mode)
    check_ids = {item.check_id for item in check_specs}
    for path in (
        request.adapter_stl_path,
        request.control_config_path,
        request.snapshot_source_path,
        request.scene_source_path,
        request.object_source_path,
    ):
        if not path.is_file():
            raise FileNotFoundError("audit input file not found: {}".format(path))

    adapter_before = _sha256_file(request.adapter_stl_path)
    config_before = _sha256_file(request.control_config_path)
    snapshot_before = _sha256_file(request.snapshot_source_path)
    scene_before = _sha256_file(request.scene_source_path)
    object_before = _sha256_file(request.object_source_path)
    joint_plan_before = ""
    joint_plan_error = ""
    if request.mode == "air_grasp":
        if request.joint_plan_source_path is None:
            joint_plan_error = "air audit requires a bound joint-plan manifest"
        elif not request.joint_plan_source_path.is_file():
            joint_plan_error = "air audit joint-plan manifest is missing"
        else:
            joint_plan_before = _sha256_file(request.joint_plan_source_path)
    filter_binding: Optional[Dict[str, Any]] = None
    filter_error = ""
    if request.mode == "air_grasp":
        candidate_binding, filter_error = _scene_filter_evidence_binding(
            request.scene_filter_evidence_path,
            scene_points=request.scene_points_base,
            object_points=request.object_points_base,
            snapshot_sha256=snapshot_before,
            adapter_sha256=adapter_before,
            expected_point_half_extent_m=(
                0.5 * float(request.scene_voxel_resolution_m)
            ),
            expected_inflation_margin_m=float(
                request.scene_clearance_margin_m
            ),
        )
        if candidate_binding is not None:
            filter_binding = dict(candidate_binding)
    query, hand_binding = build_installed_tool_collision_query(
        request, adapter_sha256=adapter_before
    )
    q_path = query.q_path_rad
    path_segments = query.path_segments

    scene_age = request.audit_started_at_s - request.scene_captured_at_s
    current_age = request.audit_started_at_s - request.current_q_captured_at_s
    precondition_failures = []
    if adapter_before != V7_ADAPTER_SHA256:
        precondition_failures.append("adapter STL is not the commissioned V7 asset")
    transform_hash = _array_sha256(request.T_EE_hand)
    if transform_hash != V7_T_EE_HAND_SHA256:
        precondition_failures.append("T_EE_hand does not match the commissioned V7 transform")
    if scene_age < -0.05 or scene_age > request.max_scene_age_s:
        precondition_failures.append("scene capture is outside the freshness window")
    if current_age < -0.05 or current_age > request.max_current_q_age_s:
        precondition_failures.append("current q is outside the freshness window")
    if filter_error:
        precondition_failures.append(filter_error)
    if joint_plan_error:
        precondition_failures.append(joint_plan_error)

    backend_error = ""
    observations: Mapping[str, CollisionObservation] = {}
    if not precondition_failures:
        try:
            candidate = backend.evaluate(query)
            if not isinstance(candidate, Mapping):
                raise TypeError("backend evaluate() result must be a mapping")
            unknown = sorted(set(candidate) - check_ids)
            if unknown:
                raise ValueError("backend returned unknown checks: {}".format(unknown))
            for check_id, observation in candidate.items():
                if not isinstance(observation, CollisionObservation):
                    raise TypeError("backend check {} is not CollisionObservation".format(check_id))
                if observation.check_id != check_id:
                    raise ValueError("backend mapping key/check_id mismatch for {}".format(check_id))
            observations = candidate
        except Exception as exc:  # Backend failure must become a closed artifact.
            backend_error = "{}: {}".format(type(exc).__name__, exc)
            observations = {}

    adapter_after = _sha256_file(request.adapter_stl_path)
    config_after = _sha256_file(request.control_config_path)
    snapshot_after = _sha256_file(request.snapshot_source_path)
    scene_after = _sha256_file(request.scene_source_path)
    object_after = _sha256_file(request.object_source_path)
    joint_plan_after = (
        _sha256_file(request.joint_plan_source_path)
        if joint_plan_before and request.joint_plan_source_path is not None
        else ""
    )
    if filter_binding is not None:
        filter_binding["sha256_after_audit"] = _sha256_file(
            Path(filter_binding["path"])
        )
    if adapter_after != adapter_before:
        precondition_failures.append("adapter STL changed during audit")
    if config_after != config_before:
        precondition_failures.append("control config changed during audit")
    if snapshot_after != snapshot_before:
        precondition_failures.append("snapshot changed during audit")
    if scene_after != scene_before:
        precondition_failures.append("scene source changed during audit")
    if object_after != object_before:
        precondition_failures.append("object source changed during audit")
    if joint_plan_after != joint_plan_before:
        precondition_failures.append("joint-plan manifest changed during audit")
    if (
        filter_binding is not None
        and filter_binding["sha256_after_audit"] != filter_binding["sha256"]
    ):
        precondition_failures.append("installed-scene filter evidence changed during audit")
    if backend_error:
        precondition_failures.append("collision backend failed: {}".format(backend_error))

    policies = {
        "scene": float(request.scene_clearance_margin_m),
        "robot": float(request.robot_clearance_margin_m),
        "hand_self": float(request.hand_self_clearance_margin_m),
        "object_contact_max_distance_m": float(request.object_contact_max_distance_m),
        "object_max_penetration_m": float(request.object_max_penetration_m),
        "object_approach": float(request.object_approach_clearance_margin_m),
        "allowed_object_contact_links": list(request.allowed_object_contact_links),
        "max_scene_age_s": float(request.max_scene_age_s),
        "max_current_q_age_s": float(request.max_current_q_age_s),
        "max_joint_step_rad": float(request.max_joint_step_rad),
        "max_q_tracking_error_rad": float(request.max_q_tracking_error_rad),
        "hand_arrival_tolerance_units": int(
            request.hand_arrival_tolerance_units
        ),
        "hand_feedback_envelope": rh56_feedback_envelope_policy(
            request.hand_arrival_tolerance_units
        ),
        "continuous_path_policy": {
            "max_interval_joint_delta_rad": float(request.max_joint_step_rad),
            "runtime_max_tracking_error_rad": float(request.max_q_tracking_error_rad),
            "continuous_segment_envelope_required": True,
            "per_check_conservative_motion_bound_required": True,
        },
        "observed_scene_policy": {
            "scope": request.observed_scene_scope,
            "voxel_resolution_m": float(request.scene_voxel_resolution_m),
            "unknown_space_policy": request.unknown_space_policy,
        },
        "runtime_operator_workspace_clear": {
            "condition_id": RUNTIME_WORKSPACE_CLEAR_CONDITION_ID,
            "required": request.mode == "air_grasp",
            "eligible_mode": "air_grasp",
            "executor_confirmation_token": RUNTIME_WORKSPACE_CLEAR_TOKEN,
            "covers": [
                "unobserved_camera_space",
                "scene_changes_after_capture",
            ],
            "point_cloud_authority_promoted": False,
            "installed_return_filter_authority_promoted": False,
        },
    }
    checks = [
        _evaluate_observation(
            spec,
            observations.get(spec.check_id),
            len(q_path),
            len(request.hand_waypoint_joint_positions_rad),
            policies,
            mode=request.mode,
        )
        for spec in check_specs
    ]
    all_checks_passed = all(item["passed"] for item in checks)
    runtime_workspace_clear_required = request.mode == "air_grasp"
    decision_passed = not precondition_failures and all_checks_passed
    reasons = list(precondition_failures)
    reasons.extend(
        "{}: {}".format(item["check_id"], failure)
        for item in checks
        for failure in item["failures"]
    )

    waypoint_items = [
        {"name": "current", "q_rad": request.current_q_rad.tolist()}
    ]
    waypoint_items.extend(
        {
            "name": "default_transit_{}".format(index),
            "q_rad": value.tolist(),
        }
        for index, value in enumerate(request.default_transit_q_rad)
    )
    waypoint_items.extend(
        (
            {"name": "default", "q_rad": request.default_q_rad.tolist()},
        )
    )
    waypoint_items.extend(
        {
            "name": "approach_transit_{}".format(index),
            "q_rad": value.tolist(),
        }
        for index, value in enumerate(request.approach_transit_q_rad)
    )
    waypoint_items.extend(
        (
            {"name": "pregrasp", "q_rad": request.pregrasp_q_rad.tolist()},
            {"name": "grasp", "q_rad": request.grasp_q_rad.tolist()},
        )
    )
    maximum_observed_step = float(np.max(np.abs(np.diff(q_path, axis=0))))
    artifact: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "mode": request.mode,
        "created_at_s": float(request.audit_started_at_s),
        "bindings": {
            "control_profile": {
                "path": str(request.control_config_path),
                "sha256": config_before,
                "sha256_after_audit": config_after,
            },
            "adapter": {
                "path": str(request.adapter_stl_path),
                "sha256": adapter_before,
                "sha256_after_audit": adapter_after,
                "required_v7_sha256": V7_ADAPTER_SHA256,
            },
            "mount_transform": {
                "convention": "T_EE_hand_maps_hand_source_into_fr3_link8",
                "T_EE_hand": request.T_EE_hand.tolist(),
                "sha256": transform_hash,
                "required_v7_sha256": V7_T_EE_HAND_SHA256,
            },
            "snapshot": {
                "path": str(request.snapshot_source_path),
                "sha256": snapshot_before,
                "sha256_after_audit": snapshot_after,
                "schema_version": int(request.snapshot_schema_version),
                "reference_frame": request.snapshot_reference_frame,
                "frame_id": int(request.snapshot_frame_id),
                "timestamp_s": float(request.snapshot_timestamp_s),
                "calibration_id": request.snapshot_calibration_id,
                "camera_serial": request.snapshot_camera_serial,
                "model_name": request.snapshot_model_name,
                "representation_checkpoint_sha256": request.snapshot_representation_checkpoint_sha256,
                "decision_checkpoint_sha256s": list(request.snapshot_decision_checkpoint_sha256s),
                "official_source_commit": request.snapshot_official_source_commit,
                "selected_candidate": {
                    "index": int(request.selected_candidate_index),
                    "canonical_pose_base": request.selected_canonical_pose_base.tolist(),
                    "canonical_pose_sha256": _array_sha256(request.selected_canonical_pose_base),
                    "hand_pose_base": request.selected_hand_pose_base.tolist(),
                    "hand_pose_sha256": _array_sha256(request.selected_hand_pose_base),
                    "hand_targets": request.selected_hand_targets.tolist(),
                    "hand_targets_sha256": _array_sha256(request.selected_hand_targets),
                },
            },
            "execution_plan": {
                "trajectory_contract": "joint_waypoint_polyline_v1",
                "planned_hand_pose_base": request.plan_hand_pose_base.tolist(),
                "planned_hand_pose_sha256": _array_sha256(request.plan_hand_pose_base),
                "pregrasp_pose_base_EE": request.plan_pregrasp_pose_base_EE.tolist(),
                "pregrasp_pose_sha256": _array_sha256(request.plan_pregrasp_pose_base_EE),
                "grasp_pose_base_EE": request.plan_grasp_pose_base_EE.tolist(),
                "grasp_pose_sha256": _array_sha256(request.plan_grasp_pose_base_EE),
                "air_retreat_distance_m": float(request.air_retreat_distance_m),
                "pregrasp_distance_m": float(request.pregrasp_distance_m),
                "pregrasp_distance_profile_key": (
                    "grasp.air_pregrasp_distance_m"
                    if request.mode == "air_grasp"
                    else "grasp.pregrasp_distance_m"
                ),
                "contact_and_lift_forbidden": request.mode == "air_grasp",
            },
            "joint_plan_manifest": {
                "required": request.mode == "air_grasp",
                "path": (
                    str(request.joint_plan_source_path)
                    if request.joint_plan_source_path is not None
                    else ""
                ),
                "sha256": joint_plan_before,
                "sha256_after_audit": joint_plan_after,
                "validation_error": joint_plan_error,
            },
            "scene": {
                "path": str(request.scene_source_path),
                "sha256": scene_before,
                "sha256_after_audit": scene_after,
                "points_sha256": _array_sha256(request.scene_points_base),
                "point_count": int(len(request.scene_points_base)),
                "reference_frame": "robot_base",
                "scene_excludes_object": True,
                "capture_q_rad": request.scene_capture_q_rad.tolist(),
                "capture_q_sha256": _array_sha256(request.scene_capture_q_rad),
                "captured_at_s": float(request.scene_captured_at_s),
                "age_at_audit_s": float(scene_age),
                "preprocessing": (
                    dict(filter_binding, required=True, validation_error="")
                    if filter_binding is not None
                    else {
                        "required": request.mode == "air_grasp",
                        "path": "",
                        "sha256": "",
                        "sha256_after_audit": "",
                        "artifact_type": "",
                        "filtered_scene_points_sha256": "",
                        "object_points_sha256": "",
                        "object_alignment_and_filter": {},
                        "installed_return_filter": {},
                        "installed_inflation_margin_m": None,
                        "unknown_space_policy_applied": False,
                        "authoritative_for_unseen_camera_space": False,
                        "motion_authorized": False,
                        "validation_error": filter_error,
                    }
                ),
            },
            "object": {
                "path": str(request.object_source_path),
                "sha256": object_before,
                "sha256_after_audit": object_after,
                "points_sha256": _array_sha256(request.object_points_base),
                "point_count": int(len(request.object_points_base)),
                "reference_frame": "robot_base",
            },
            "joint_path": {
                "sampling_algorithm": CANONICAL_JOINT_SAMPLING_ALGORITHM,
                "waypoints": waypoint_items,
                "current_q_captured_at_s": float(request.current_q_captured_at_s),
                "current_q_age_at_audit_s": float(current_age),
                "segments": list(path_segments),
                "sample_count": int(len(q_path)),
                "samples_rad": q_path.tolist(),
                "sha256": _array_sha256(q_path),
                "maximum_observed_joint_step_rad": maximum_observed_step,
            },
            "hand_execution_path": dict(request.hand_execution_path),
            "hand_model": dict(
                hand_binding,
                open_actuator_targets=request.open_actuator_targets.tolist(),
                open_actuator_targets_sha256=_array_sha256(request.open_actuator_targets),
                closure_actuator_targets=request.closure_actuator_targets.tolist(),
                closure_actuator_targets_sha256=_array_sha256(request.closure_actuator_targets),
            ),
        },
        "policies": policies,
        "collision_backend": {
            "name": identity.name,
            "version": identity.version,
            "implementation_sha256": identity.implementation_sha256,
            "configuration_sha256": identity.configuration_sha256,
            "error": backend_error,
        },
        "checks": checks,
        "decision": {
            "passed": bool(decision_passed),
            "pass_kind": (
                "conditional_air"
                if decision_passed and runtime_workspace_clear_required
                else ("authoritative" if decision_passed else "failed")
            ),
            "motion_authorized": False,
            "all_required_checks_passed": bool(all_checks_passed),
            "runtime_operator_workspace_clear_required": bool(
                runtime_workspace_clear_required
            ),
            "runtime_conditions_satisfied_in_artifact": False,
            "precondition_failures": precondition_failures,
            "reasons": reasons,
            "meaning": "offline collision evidence only; never a motion command",
        },
    }
    artifact["artifact_sha256"] = _json_sha256(artifact)
    validate_installed_tool_audit(artifact)
    return artifact


def _expect_keys(value: Any, expected: Sequence[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("{} must be a JSON object".format(name))
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise ValueError(
            "{} keys differ: missing={} unknown={}".format(
                name, sorted(expected_set - actual), sorted(actual - expected_set)
            )
        )
    return value


def validate_installed_tool_audit(
    artifact: Mapping[str, Any],
    *,
    verify_files: bool = False,
    require_pass: bool = False,
) -> Mapping[str, Any]:
    """Validate exact schema, hashes, path reconstruction, and decision logic."""

    root = _expect_keys(
        artifact,
        (
            "schema_version",
            "artifact_type",
            "mode",
            "created_at_s",
            "bindings",
            "policies",
            "collision_backend",
            "checks",
            "decision",
            "artifact_sha256",
        ),
        "artifact",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported installed-tool audit schema_version")
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("artifact_type is not an installed-tool collision audit")
    mode = root["mode"]
    check_specs = check_specs_for_mode(mode)
    if not np.isfinite(float(root["created_at_s"])):
        raise ValueError("created_at_s must be finite")
    supplied_digest = _sha256_text(root["artifact_sha256"], "artifact_sha256")
    unsigned = dict(root)
    del unsigned["artifact_sha256"]
    if _json_sha256(unsigned) != supplied_digest:
        raise ValueError("artifact_sha256 does not match artifact content")

    bindings = _expect_keys(
        root["bindings"],
        (
            "control_profile", "adapter", "mount_transform", "snapshot",
            "execution_plan", "joint_plan_manifest", "scene", "object",
            "joint_path", "hand_execution_path", "hand_model",
        ),
        "bindings",
    )
    profile = _expect_keys(
        bindings["control_profile"],
        ("path", "sha256", "sha256_after_audit"),
        "bindings.control_profile",
    )
    for key in ("sha256", "sha256_after_audit"):
        _sha256_text(profile[key], "bindings.control_profile." + key)
    if verify_files:
        profile_path = Path(profile["path"])
        if not profile_path.is_file() or _sha256_file(profile_path) != profile["sha256"]:
            raise ValueError("control config file no longer matches artifact")
    adapter = _expect_keys(
        bindings["adapter"],
        ("path", "sha256", "sha256_after_audit", "required_v7_sha256"),
        "bindings.adapter",
    )
    if adapter["required_v7_sha256"] != V7_ADAPTER_SHA256:
        raise ValueError("artifact does not require the commissioned V7 adapter")
    _sha256_text(adapter["sha256"], "bindings.adapter.sha256")
    _sha256_text(adapter["sha256_after_audit"], "bindings.adapter.sha256_after_audit")
    if verify_files:
        adapter_path = Path(adapter["path"])
        if not adapter_path.is_file() or _sha256_file(adapter_path) != adapter["sha256"]:
            raise ValueError("adapter file no longer matches artifact")

    mount = _expect_keys(
        bindings["mount_transform"],
        ("convention", "T_EE_hand", "sha256", "required_v7_sha256"),
        "bindings.mount_transform",
    )
    if mount["convention"] != "T_EE_hand_maps_hand_source_into_fr3_link8":
        raise ValueError("unsupported mount transform convention")
    transform = validate_rigid_transform(np.asarray(mount["T_EE_hand"]), "T_EE_hand")
    if _array_sha256(transform) != mount["sha256"]:
        raise ValueError("T_EE_hand SHA-256 mismatch")
    if mount["required_v7_sha256"] != V7_T_EE_HAND_SHA256:
        raise ValueError("artifact has wrong commissioned V7 T_EE_hand binding")

    snapshot = _expect_keys(
        bindings["snapshot"],
        (
            "path", "sha256", "sha256_after_audit", "schema_version",
            "reference_frame", "frame_id", "timestamp_s", "calibration_id",
            "camera_serial", "model_name", "representation_checkpoint_sha256",
            "decision_checkpoint_sha256s", "official_source_commit",
            "selected_candidate",
        ),
        "bindings.snapshot",
    )
    for key in ("sha256", "sha256_after_audit"):
        _sha256_text(snapshot[key], "bindings.snapshot." + key)
    if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 2:
        raise ValueError("bound snapshot schema_version must be exactly 2")
    if snapshot["reference_frame"] != "robot_base":
        raise ValueError("bound snapshot reference frame must be robot_base")
    if type(snapshot["frame_id"]) is not int or snapshot["frame_id"] < 0:
        raise ValueError("bound snapshot frame_id must be a non-negative integer")
    if isinstance(snapshot["timestamp_s"], bool) or not isinstance(snapshot["timestamp_s"], (int, float)) or not np.isfinite(float(snapshot["timestamp_s"])) or float(snapshot["timestamp_s"]) < 0.0:
        raise ValueError("bound snapshot timestamp_s must be finite and non-negative")
    for key in ("calibration_id", "camera_serial", "model_name", "representation_checkpoint_sha256", "official_source_commit"):
        if not isinstance(snapshot[key], str):
            raise ValueError("bound snapshot {} must be a string".format(key))
    if snapshot["representation_checkpoint_sha256"]:
        _sha256_text(snapshot["representation_checkpoint_sha256"], "bindings.snapshot.representation_checkpoint_sha256")
    decision_hashes = snapshot["decision_checkpoint_sha256s"]
    if not isinstance(decision_hashes, list) or len(decision_hashes) not in (0, 8):
        raise ValueError("bound decision checkpoint hashes must be empty or contain exactly 8 entries")
    for index, digest in enumerate(decision_hashes):
        _sha256_text(digest, "bindings.snapshot.decision_checkpoint_sha256s[{}]".format(index))
    source_commit = snapshot["official_source_commit"]
    if source_commit and (
        len(source_commit) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("bound official source commit must be 40- or 64-hex")
    candidate = _expect_keys(
        snapshot["selected_candidate"],
        (
            "index", "canonical_pose_base", "canonical_pose_sha256",
            "hand_pose_base", "hand_pose_sha256", "hand_targets",
            "hand_targets_sha256",
        ),
        "bindings.snapshot.selected_candidate",
    )
    if type(candidate["index"]) is not int or candidate["index"] < 0:
        raise ValueError("bound selected candidate index must be non-negative")
    canonical_pose = validate_rigid_transform(np.asarray(candidate["canonical_pose_base"]), "selected canonical pose")
    hand_pose = validate_rigid_transform(np.asarray(candidate["hand_pose_base"]), "selected hand pose")
    for value, key in (
        (canonical_pose, "canonical_pose_sha256"),
        (hand_pose, "hand_pose_sha256"),
    ):
        _sha256_text(candidate[key], "bindings.snapshot.selected_candidate." + key)
        if _array_sha256(value) != candidate[key]:
            raise ValueError("selected candidate {} mismatch".format(key))
    selected_targets = _finite_vector(candidate["hand_targets"], 6, "selected candidate hand targets")
    _sha256_text(candidate["hand_targets_sha256"], "selected candidate hand targets SHA-256")
    if (
        np.any(selected_targets < 0.0)
        or np.any(selected_targets > 1000.0)
        or not np.array_equal(selected_targets, np.rint(selected_targets))
        or _array_sha256(selected_targets) != candidate["hand_targets_sha256"]
    ):
        raise ValueError("selected candidate hand targets are invalid")
    if verify_files:
        snapshot_path = Path(snapshot["path"])
        if not snapshot_path.is_file() or _sha256_file(snapshot_path) != snapshot["sha256"]:
            raise ValueError("snapshot file no longer matches artifact")

    execution_plan = _expect_keys(
        bindings["execution_plan"],
        (
            "trajectory_contract", "planned_hand_pose_base",
            "planned_hand_pose_sha256", "pregrasp_pose_base_EE",
            "pregrasp_pose_sha256", "grasp_pose_base_EE", "grasp_pose_sha256",
            "air_retreat_distance_m", "pregrasp_distance_m",
            "pregrasp_distance_profile_key", "contact_and_lift_forbidden",
        ),
        "bindings.execution_plan",
    )
    if execution_plan["trajectory_contract"] != "joint_waypoint_polyline_v1":
        raise ValueError("unsupported execution trajectory contract")
    plan_poses = {}
    for pose_key, hash_key in (
        ("planned_hand_pose_base", "planned_hand_pose_sha256"),
        ("pregrasp_pose_base_EE", "pregrasp_pose_sha256"),
        ("grasp_pose_base_EE", "grasp_pose_sha256"),
    ):
        pose = validate_rigid_transform(np.asarray(execution_plan[pose_key]), pose_key)
        plan_poses[pose_key] = pose
        _sha256_text(execution_plan[hash_key], hash_key)
        if _array_sha256(pose) != execution_plan[hash_key]:
            raise ValueError("execution plan {} SHA-256 mismatch".format(pose_key))
    retreat = execution_plan["air_retreat_distance_m"]
    if isinstance(retreat, bool) or not isinstance(retreat, (int, float)) or not np.isfinite(float(retreat)) or float(retreat) < 0.0:
        raise ValueError("execution plan air retreat must be finite and non-negative")
    expected_forbidden = mode == "air_grasp"
    if execution_plan["contact_and_lift_forbidden"] is not expected_forbidden:
        raise ValueError("execution plan contact/lift policy differs from audit mode")
    if mode == "loaded_grasp" and float(retreat) != 0.0:
        raise ValueError("loaded grasp cannot contain an air retreat")
    if mode == "air_grasp" and float(retreat) <= 0.0:
        raise ValueError("air grasp requires a positive retreat")
    pregrasp_distance = execution_plan["pregrasp_distance_m"]
    if (
        isinstance(pregrasp_distance, bool)
        or not isinstance(pregrasp_distance, (int, float))
        or not np.isfinite(float(pregrasp_distance))
        or float(pregrasp_distance) <= 0.0
    ):
        raise ValueError("execution plan pregrasp distance must be finite and positive")
    expected_profile_key = (
        "grasp.air_pregrasp_distance_m"
        if mode == "air_grasp"
        else "grasp.pregrasp_distance_m"
    )
    if execution_plan["pregrasp_distance_profile_key"] != expected_profile_key:
        raise ValueError("execution plan pregrasp profile key differs from audit mode")
    composed_hand = plan_poses["grasp_pose_base_EE"] @ transform
    if not np.allclose(
        composed_hand,
        plan_poses["planned_hand_pose_base"],
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError("bound planned hand/EEF poses are inconsistent with T_EE_hand")
    approach = canonical_pose[:3, 0]
    expected_planned_hand = hand_pose.copy()
    expected_planned_hand[:3, 3] -= float(retreat) * approach
    if not np.allclose(
        expected_planned_hand,
        plan_poses["planned_hand_pose_base"],
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError("bound planned hand pose is not the required -approach retreat")
    expected_pregrasp = plan_poses["grasp_pose_base_EE"].copy()
    expected_pregrasp[:3, 3] -= float(pregrasp_distance) * approach
    if not np.allclose(
        expected_pregrasp,
        plan_poses["pregrasp_pose_base_EE"],
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError("bound pregrasp pose does not match the profile distance")

    joint_plan_manifest = _expect_keys(
        bindings["joint_plan_manifest"],
        ("required", "path", "sha256", "sha256_after_audit", "validation_error"),
        "bindings.joint_plan_manifest",
    )
    if joint_plan_manifest["required"] is not (mode == "air_grasp"):
        raise ValueError("joint-plan manifest requirement differs from audit mode")
    if not isinstance(joint_plan_manifest["validation_error"], str):
        raise ValueError("joint-plan manifest validation_error must be a string")
    if joint_plan_manifest["path"]:
        for key in ("sha256", "sha256_after_audit"):
            _sha256_text(
                joint_plan_manifest[key], "bindings.joint_plan_manifest." + key
            )
        if joint_plan_manifest["validation_error"]:
            raise ValueError("bound joint-plan manifest cannot also contain an error")
        if verify_files:
            manifest_path = Path(joint_plan_manifest["path"])
            if (
                not manifest_path.is_file()
                or _sha256_file(manifest_path) != joint_plan_manifest["sha256"]
            ):
                raise ValueError("joint-plan manifest no longer matches artifact")
    elif mode == "air_grasp" and not joint_plan_manifest["validation_error"]:
        raise ValueError("air audit has neither a joint-plan manifest nor an error")

    scene = _expect_keys(
        bindings["scene"],
        (
            "path", "sha256", "sha256_after_audit", "points_sha256", "point_count", "reference_frame", "scene_excludes_object",
            "capture_q_rad", "capture_q_sha256", "captured_at_s", "age_at_audit_s",
            "preprocessing",
        ),
        "bindings.scene",
    )
    for key in ("sha256", "sha256_after_audit", "points_sha256", "capture_q_sha256"):
        _sha256_text(scene[key], "bindings.scene." + key)
    if scene["reference_frame"] != "robot_base" or scene["scene_excludes_object"] is not True or type(scene["point_count"]) is not int or scene["point_count"] <= 0:
        raise ValueError("scene binding has invalid frame or point count")
    capture_q = _fr3_q(scene["capture_q_rad"], "scene.capture_q_rad")
    if _array_sha256(capture_q) != scene["capture_q_sha256"]:
        raise ValueError("scene capture q SHA-256 mismatch")
    if verify_files:
        path = Path(scene["path"])
        if not path.is_file() or _sha256_file(path) != scene["sha256"]:
            raise ValueError("scene source no longer matches artifact")
    preprocessing = _expect_keys(
        scene["preprocessing"],
        (
            "required", "path", "sha256", "sha256_after_audit",
            "artifact_type", "filtered_scene_points_sha256",
            "object_points_sha256", "object_alignment_and_filter",
            "installed_return_filter",
            "installed_inflation_margin_m", "unknown_space_policy_applied",
            "authoritative_for_unseen_camera_space", "motion_authorized",
            "validation_error",
        ),
        "bindings.scene.preprocessing",
    )
    if preprocessing["required"] is not (mode == "air_grasp"):
        raise ValueError("scene preprocessing requirement differs from audit mode")
    if not isinstance(preprocessing["validation_error"], str):
        raise ValueError("scene preprocessing validation_error must be a string")
    if preprocessing["path"]:
        for key in (
            "sha256",
            "sha256_after_audit",
            "filtered_scene_points_sha256",
            "object_points_sha256",
        ):
            _sha256_text(preprocessing[key], "bindings.scene.preprocessing." + key)
        margin = preprocessing["installed_inflation_margin_m"]
        installed_filter = _expect_keys(
            preprocessing["installed_return_filter"],
            (
                "method", "fr3_geometry_model", "fr3_visual_meshes",
                "point_half_extent_m", "inflation_margin_m",
                "candidate_mesh_point_tests", "reason_counts",
                "residual_self_return_filter",
            ),
            "bindings.scene.preprocessing.installed_return_filter",
        )
        visual_meshes = installed_filter["fr3_visual_meshes"]
        if not isinstance(visual_meshes, list) or len(visual_meshes) != 8:
            raise ValueError("scene preprocessing FR3 visual mesh list is invalid")
        for index, entry in enumerate(visual_meshes):
            mesh = _expect_keys(
                entry,
                ("link", "path", "sha256"),
                "scene preprocessing FR3 visual mesh {}".format(index),
            )
            _sha256_text(mesh["sha256"], "FR3 visual mesh SHA-256")
            if mesh["link"] != "link{}".format(index):
                raise ValueError("scene preprocessing FR3 visual mesh order is invalid")
            if verify_files:
                mesh_path = Path(mesh["path"])
                if not mesh_path.is_file() or _sha256_file(mesh_path) != mesh["sha256"]:
                    raise ValueError("FR3 visual mesh no longer matches filter evidence")
        point_half_extent = installed_filter["point_half_extent_m"]
        candidate_tests = installed_filter["candidate_mesh_point_tests"]
        reason_counts = installed_filter["reason_counts"]
        if (
            installed_filter["method"]
            != "official FR3 visual shell/V7/open-RH56 triangle mesh versus inflated point cube"
            or installed_filter["fr3_geometry_model"]
            != "official_fr3_visual_triangle_meshes"
            or isinstance(point_half_extent, bool)
            or not isinstance(point_half_extent, (int, float))
            or not np.isfinite(float(point_half_extent))
            or float(point_half_extent) <= 0.0
            or type(candidate_tests) is not int
            or candidate_tests < 0
            or not isinstance(reason_counts, dict)
            or any(
                not isinstance(key, str)
                or not key
                or type(value) is not int
                or value <= 0
                for key, value in reason_counts.items()
            )
        ):
            raise ValueError("scene preprocessing FR3 visual filter is invalid")
        residual = _expect_keys(
            installed_filter["residual_self_return_filter"],
            (
                "enabled", "failure", "method", "distance_method",
                "maximum_model_surface_distance_m",
                "hard_maximum_model_surface_distance_m",
                "calibration_alignment_maximum_m", "calibration_guard_m",
                "calibration_bound_formula", "connectivity_radius_m",
                "color_space", "palette_color_max_l2",
                "maximum_geodesic_m", "model_candidate_test_count",
                "model_candidate_indices_sha256",
                "model_candidate_surface_distances_sha256",
                "exact_seed_count", "model_neighborhood_candidate_count",
                "appearance_palette_candidate_count",
                "connected_candidate_count", "residual_removed_count",
                "residual_removed_indices_sha256",
                "residual_removed_surface_distance_min_m",
                "residual_removed_surface_distance_max_m", "reason_counts",
            ),
            "bindings.scene.preprocessing.installed_return_filter.residual_self_return_filter",
        )
        residual_counts = (
            "model_candidate_test_count", "exact_seed_count",
            "model_neighborhood_candidate_count",
            "appearance_palette_candidate_count", "connected_candidate_count",
            "residual_removed_count",
        )
        for key in (
            "model_candidate_indices_sha256",
            "model_candidate_surface_distances_sha256",
            "residual_removed_indices_sha256",
        ):
            _sha256_text(residual[key], "residual self-filter " + key)
        residual_reason_counts = residual["reason_counts"]
        residual_minimum = residual["residual_removed_surface_distance_min_m"]
        residual_maximum = residual["residual_removed_surface_distance_max_m"]
        residual_valid = (
            residual["enabled"] is True
            and residual["failure"] == ""
            and residual["method"]
            == "strict HPP-FCL point-to-mesh neighborhood AND same-component normalized-sRGB seed palette AND bounded same-component 3-D geodesic"
            and residual["distance_method"]
            == "HPP-FCL triangle-mesh to 1nm sphere, radius corrected"
            and residual["calibration_bound_formula"]
            == "min(hard_maximum, object_to_live_maximum + calibration_guard)"
            and residual["color_space"] == "normalized_sRGB_euclidean"
            and np.isclose(float(residual["hard_maximum_model_surface_distance_m"]), 0.020, atol=1e-15, rtol=0.0)
            and np.isclose(float(residual["calibration_guard_m"]), 0.002, atol=1e-15, rtol=0.0)
            and np.isclose(float(residual["connectivity_radius_m"]), 0.010, atol=1e-15, rtol=0.0)
            and np.isclose(float(residual["palette_color_max_l2"]), 0.18, atol=1e-15, rtol=0.0)
            and np.isclose(float(residual["maximum_geodesic_m"]), 0.040, atol=1e-15, rtol=0.0)
            and np.isclose(
                float(residual["maximum_model_surface_distance_m"]),
                min(
                    float(residual["hard_maximum_model_surface_distance_m"]),
                    float(residual["calibration_alignment_maximum_m"])
                    + float(residual["calibration_guard_m"]),
                ),
                atol=1e-15, rtol=0.0,
            )
            and all(type(residual[key]) is int and residual[key] >= 0 for key in residual_counts)
            and residual["residual_removed_count"]
            <= residual["appearance_palette_candidate_count"]
            <= residual["model_neighborhood_candidate_count"]
            and residual["connected_candidate_count"]
            <= residual["model_neighborhood_candidate_count"]
            and isinstance(residual_reason_counts, dict)
            and all(
                isinstance(key, str) and key and type(value) is int and value > 0
                for key, value in residual_reason_counts.items()
            )
            and sum(residual_reason_counts.values())
            == residual["residual_removed_count"]
        )
        if residual["residual_removed_count"] == 0:
            residual_valid = residual_valid and residual_minimum is None and residual_maximum is None
        else:
            residual_valid = residual_valid and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(float(value))
                for value in (residual_minimum, residual_maximum)
            )
            if residual_valid:
                residual_valid = (
                    float(residual_minimum) <= float(residual_maximum)
                    and float(residual_maximum)
                    <= float(residual["maximum_model_surface_distance_m"]) + 1e-12
                )
        if not residual_valid:
            raise ValueError("scene preprocessing residual self filter is invalid")
        alignment = _expect_keys(
            preprocessing["object_alignment_and_filter"],
            (
                "method", "object_point_count", "live_scene_point_count",
                "object_to_live_median_m", "object_to_live_p95_m",
                "object_to_live_maximum_m", "alignment_coverage_distance_m",
                "alignment_coverage_fraction", "object_return_max_distance_m",
                "expanded_object_aabb_min_m", "expanded_object_aabb_max_m",
                "object_return_count_before_installed_precedence",
                "alignment_median_max_m", "alignment_p95_max_m",
                "alignment_minimum_coverage", "passed", "failures",
            ),
            "bindings.scene.preprocessing.object_alignment_and_filter",
        )
        alignment_numbers = (
            "object_to_live_median_m", "object_to_live_p95_m",
            "object_to_live_maximum_m", "alignment_coverage_distance_m",
            "alignment_coverage_fraction", "object_return_max_distance_m",
            "alignment_median_max_m", "alignment_p95_max_m",
            "alignment_minimum_coverage",
        )
        if any(
            isinstance(alignment[key], bool)
            or not isinstance(alignment[key], (int, float))
            or not np.isfinite(float(alignment[key]))
            or float(alignment[key]) < 0.0
            for key in alignment_numbers
        ):
            raise ValueError("scene preprocessing object alignment numbers are invalid")
        for key in ("object_point_count", "live_scene_point_count"):
            if type(alignment[key]) is not int or alignment[key] <= 0:
                raise ValueError("scene preprocessing object alignment counts are invalid")
        if (
            type(alignment["object_return_count_before_installed_precedence"])
            is not int
            or alignment["object_return_count_before_installed_precedence"] < 0
        ):
            raise ValueError("scene preprocessing object-return count is invalid")
        aabb_min = _finite_vector(
            alignment["expanded_object_aabb_min_m"], 3, "object alignment AABB min"
        )
        aabb_max = _finite_vector(
            alignment["expanded_object_aabb_max_m"], 3, "object alignment AABB max"
        )
        if (
            preprocessing["artifact_type"]
            != "installed_scene_return_filter_evidence"
            or preprocessing["unknown_space_policy_applied"] is not False
            or preprocessing["authoritative_for_unseen_camera_space"] is not False
            or preprocessing["motion_authorized"] is not False
            or preprocessing["validation_error"]
            or preprocessing["filtered_scene_points_sha256"] != scene["points_sha256"]
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not np.isfinite(float(margin))
            or not 0.0 <= float(margin) <= 0.02
            or not np.isclose(
                float(installed_filter["inflation_margin_m"]),
                float(margin),
                atol=1e-15,
                rtol=0.0,
            )
            or alignment["method"]
            != "bidirectional_cKDTree_with_expanded_object_AABB"
            or alignment["passed"] is not True
            or alignment["failures"] != []
            or float(alignment["object_to_live_median_m"]) > 0.008
            or float(alignment["object_to_live_p95_m"]) > 0.015
            or float(alignment["alignment_coverage_fraction"]) < 0.80
            or float(alignment["alignment_coverage_fraction"]) > 1.0
            or float(alignment["alignment_median_max_m"]) > 0.008
            or float(alignment["alignment_p95_max_m"]) > 0.015
            or float(alignment["alignment_minimum_coverage"]) < 0.80
            or float(alignment["alignment_minimum_coverage"]) > 1.0
            or float(alignment["object_to_live_median_m"])
            > float(alignment["alignment_median_max_m"])
            or float(alignment["object_to_live_p95_m"])
            > float(alignment["alignment_p95_max_m"])
            or float(alignment["object_to_live_p95_m"])
            < float(alignment["object_to_live_median_m"])
            or float(alignment["object_to_live_maximum_m"])
            < float(alignment["object_to_live_p95_m"])
            or not np.isclose(
                float(residual["calibration_alignment_maximum_m"]),
                float(alignment["object_to_live_maximum_m"]),
                atol=1e-15,
                rtol=0.0,
            )
            or np.any(aabb_min > aabb_max)
        ):
            raise ValueError("scene preprocessing evidence binding is invalid")
        if verify_files:
            filter_path = Path(preprocessing["path"])
            if (
                not filter_path.is_file()
                or _sha256_file(filter_path) != preprocessing["sha256"]
            ):
                raise ValueError("scene filter evidence no longer matches artifact")
    elif mode == "air_grasp" and not preprocessing["validation_error"]:
        raise ValueError("air scene preprocessing has neither evidence nor an error")

    object_binding = _expect_keys(
        bindings["object"],
        ("path", "sha256", "sha256_after_audit", "points_sha256", "point_count", "reference_frame"),
        "bindings.object",
    )
    for key in ("sha256", "sha256_after_audit", "points_sha256"):
        _sha256_text(object_binding[key], "bindings.object." + key)
    if object_binding["reference_frame"] != "robot_base" or type(object_binding["point_count"]) is not int or object_binding["point_count"] <= 0:
        raise ValueError("object binding has invalid frame or point count")
    if preprocessing["path"] and (
        preprocessing["object_points_sha256"] != object_binding["points_sha256"]
        or preprocessing["object_alignment_and_filter"]["object_point_count"]
        != object_binding["point_count"]
    ):
        raise ValueError("scene preprocessing object binding differs from audit object")
    if verify_files:
        path = Path(object_binding["path"])
        if not path.is_file() or _sha256_file(path) != object_binding["sha256"]:
            raise ValueError("object source no longer matches artifact")

    path_binding = _expect_keys(
        bindings["joint_path"],
        (
            "sampling_algorithm", "waypoints", "current_q_captured_at_s", "current_q_age_at_audit_s",
            "segments", "sample_count", "samples_rad", "sha256",
            "maximum_observed_joint_step_rad",
        ),
        "bindings.joint_path",
    )
    if (
        path_binding["sampling_algorithm"]
        != CANONICAL_JOINT_SAMPLING_ALGORITHM
    ):
        raise ValueError("unsupported joint path sampling algorithm")
    if mode == "air_grasp":
        hand_execution_path = validate_rh56_hand_execution_path(
            bindings["hand_execution_path"]
        )
    else:
        validate_loaded_hand_execution_path_not_applicable(
            bindings["hand_execution_path"]
        )
        hand_execution_path = None
    waypoint_items = path_binding["waypoints"]
    if not isinstance(waypoint_items, list) or len(waypoint_items) < 4:
        raise ValueError("joint path waypoints must be a list with at least four entries")
    waypoint_names = []
    waypoint_values = []
    for index, waypoint in enumerate(waypoint_items):
        item = _expect_keys(
            waypoint,
            ("name", "q_rad"),
            "bindings.joint_path.waypoints[{}]".format(index),
        )
        if not isinstance(item["name"], str) or not item["name"]:
            raise ValueError("joint waypoint name must be a non-empty string")
        waypoint_names.append(item["name"])
        waypoint_values.append(
            _fr3_q(item["q_rad"], "joint waypoint {}".format(index))
        )
    if len(set(waypoint_names)) != len(waypoint_names):
        raise ValueError("joint waypoint names must be unique")
    if waypoint_names[0] != "current" or waypoint_names[-2:] != [
        "pregrasp", "grasp"
    ] or waypoint_names.count("default") != 1:
        raise ValueError(
            "joint waypoints must be current, optional default_transit_N, "
            "default, optional approach_transit_N, pregrasp, grasp"
        )
    default_index = waypoint_names.index("default")
    expected_transit_names = [
        "default_transit_{}".format(index)
        for index in range(default_index - 1)
    ]
    if waypoint_names[1:default_index] != expected_transit_names:
        raise ValueError("default transit waypoint names are missing or out of order")
    expected_approach_names = [
        "approach_transit_{}".format(index)
        for index in range(len(waypoint_names) - default_index - 3)
    ]
    if waypoint_names[default_index + 1 : -2] != expected_approach_names:
        raise ValueError("approach transit waypoint names are missing or out of order")
    policies = _expect_keys(
        root["policies"],
        (
            "scene", "robot", "hand_self", "object_contact_max_distance_m",
            "object_max_penetration_m", "object_approach",
            "allowed_object_contact_links", "max_scene_age_s",
            "max_current_q_age_s", "max_joint_step_rad",
            "max_q_tracking_error_rad",
            "hand_arrival_tolerance_units",
            "hand_feedback_envelope",
            "continuous_path_policy", "observed_scene_policy",
            "runtime_operator_workspace_clear",
        ),
        "policies",
    )
    for key in (
        "scene", "robot", "hand_self", "object_contact_max_distance_m",
        "object_max_penetration_m", "object_approach", "max_scene_age_s",
        "max_current_q_age_s", "max_joint_step_rad",
        "max_q_tracking_error_rad",
    ):
        if isinstance(policies[key], bool) or not isinstance(policies[key], (int, float)):
            raise ValueError("policy {} must be a JSON number".format(key))
        numeric = float(policies[key])
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError("policy {} must be finite and non-negative".format(key))
    if not np.isclose(float(policies["hand_self"]), 0.0, atol=1e-15, rtol=0.0):
        raise ValueError("RH56 self-clearance policy must be exactly 0 m")
    continuous_policy = _expect_keys(
        policies["continuous_path_policy"],
        (
            "max_interval_joint_delta_rad",
            "runtime_max_tracking_error_rad",
            "continuous_segment_envelope_required",
            "per_check_conservative_motion_bound_required",
        ),
        "policies.continuous_path_policy",
    )
    for key in (
        "max_interval_joint_delta_rad",
        "runtime_max_tracking_error_rad",
    ):
        value = continuous_policy[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError("continuous path policy {} is invalid".format(key))
    if (
        continuous_policy["continuous_segment_envelope_required"] is not True
        or continuous_policy["per_check_conservative_motion_bound_required"] is not True
        or not np.isclose(
            float(continuous_policy["max_interval_joint_delta_rad"]),
            float(policies["max_joint_step_rad"]),
            atol=1e-15,
            rtol=0.0,
        )
        or not np.isclose(
            float(continuous_policy["runtime_max_tracking_error_rad"]),
            float(policies["max_q_tracking_error_rad"]),
            atol=1e-15,
            rtol=0.0,
        )
    ):
        raise ValueError("continuous path policy is inconsistent")
    observed_policy = _expect_keys(
        policies["observed_scene_policy"],
        ("scope", "voxel_resolution_m", "unknown_space_policy"),
        "policies.observed_scene_policy",
    )
    if (
        observed_policy["scope"] != "calibrated_camera_frustum_voxel_grid"
        or observed_policy["unknown_space_policy"] != "occupied"
        or isinstance(observed_policy["voxel_resolution_m"], bool)
        or not isinstance(observed_policy["voxel_resolution_m"], (int, float))
        or not np.isfinite(float(observed_policy["voxel_resolution_m"]))
        or float(observed_policy["voxel_resolution_m"]) <= 0.0
    ):
        raise ValueError("observed scene policy is invalid")
    if preprocessing["path"]:
        installed_filter = preprocessing["installed_return_filter"]
        if (
            not np.isclose(
                float(installed_filter["point_half_extent_m"]),
                0.5 * float(observed_policy["voxel_resolution_m"]),
                atol=1e-15,
                rtol=0.0,
            )
            or not np.isclose(
                float(installed_filter["inflation_margin_m"]),
                float(policies["scene"]),
                atol=1e-15,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "scene preprocessing voxel/inflation policy differs from audit"
            )
    runtime_workspace_policy = _expect_keys(
        policies["runtime_operator_workspace_clear"],
        (
            "condition_id", "required", "eligible_mode",
            "executor_confirmation_token", "covers",
            "point_cloud_authority_promoted",
            "installed_return_filter_authority_promoted",
        ),
        "policies.runtime_operator_workspace_clear",
    )
    if (
        runtime_workspace_policy["condition_id"]
        != RUNTIME_WORKSPACE_CLEAR_CONDITION_ID
        or runtime_workspace_policy["required"] is not (mode == "air_grasp")
        or runtime_workspace_policy["eligible_mode"] != "air_grasp"
        or runtime_workspace_policy["executor_confirmation_token"]
        != RUNTIME_WORKSPACE_CLEAR_TOKEN
        or runtime_workspace_policy["covers"]
        != ["unobserved_camera_space", "scene_changes_after_capture"]
        or runtime_workspace_policy["point_cloud_authority_promoted"] is not False
        or runtime_workspace_policy[
            "installed_return_filter_authority_promoted"
        ]
        is not False
    ):
        raise ValueError("runtime operator workspace-clear policy is invalid")
    allowed_contact_links = policies["allowed_object_contact_links"]
    if not isinstance(allowed_contact_links, list) or not allowed_contact_links or len(set(allowed_contact_links)) != len(allowed_contact_links):
        raise ValueError("allowed_object_contact_links policy must be a unique non-empty list")
    if (
        float(policies["max_scene_age_s"]) <= 0.0
        or float(policies["max_current_q_age_s"]) <= 0.0
        or not 0.0 < float(policies["max_joint_step_rad"]) <= 0.10
        or not 0.0 < float(policies["max_q_tracking_error_rad"]) <= 0.01
        or type(policies["hand_arrival_tolerance_units"]) is not int
        or not 0 <= policies["hand_arrival_tolerance_units"] <= 100
    ):
        raise ValueError(
            "freshness windows, joint sampling step, and tracking error must be positive"
        )
    validate_rh56_feedback_envelope_policy(
        policies["hand_feedback_envelope"],
        policies["hand_arrival_tolerance_units"],
    )
    rebuilt, rebuilt_segments = build_joint_path(
        waypoint_values[0], waypoint_values[default_index],
        waypoint_values[-2], waypoint_values[-1],
        max_joint_step_rad=float(policies["max_joint_step_rad"]),
        default_transit_q_rad=waypoint_values[1:default_index],
        approach_transit_q_rad=waypoint_values[default_index + 1 : -2],
    )
    stored_path = np.asarray(path_binding["samples_rad"], dtype=np.float64)
    if stored_path.shape != rebuilt.shape or not np.array_equal(stored_path, rebuilt):
        raise ValueError("joint path samples do not reconstruct from waypoints/policy")
    if path_binding["segments"] != list(rebuilt_segments):
        raise ValueError("joint path segment indices are inconsistent")
    if type(path_binding["sample_count"]) is not int or path_binding["sample_count"] != len(rebuilt):
        raise ValueError("joint path sample_count is inconsistent")
    if _array_sha256(rebuilt) != path_binding["sha256"]:
        raise ValueError("joint path SHA-256 mismatch")
    observed_step = float(np.max(np.abs(np.diff(rebuilt, axis=0))))
    if not np.isclose(
        float(path_binding["maximum_observed_joint_step_rad"]),
        observed_step,
        atol=1e-15,
        rtol=0.0,
    ):
        raise ValueError("maximum observed joint step is inconsistent")

    hand = _expect_keys(
        bindings["hand_model"],
        (
            "urdf_path", "urdf_sha256", "mapping_path", "mapping_sha256",
            "mesh_resolution", "link_count", "links", "open_joint_positions_rad",
            "open_fk_sha256", "open_configuration_provenance",
            "closure_joint_positions_rad", "closure_fk_sha256",
            "execution_waypoint_joint_positions_rad",
            "execution_waypoint_fk_sha256s", "actuator_mapping_provenance",
            "open_actuator_targets", "open_actuator_targets_sha256",
            "closure_actuator_targets", "closure_actuator_targets_sha256",
            "execution_dense_interval_mapping",
        ),
        "bindings.hand_model",
    )
    if hand["mesh_resolution"] not in ("full", "simplified") or hand["link_count"] != 13:
        raise ValueError("hand model must bind exactly 13 full/simplified link meshes")
    if not isinstance(hand["links"], list) or len(hand["links"]) != 13:
        raise ValueError("hand model link list must contain exactly 13 entries")
    link_names = set()
    for index, entry in enumerate(hand["links"]):
        _expect_keys(entry, ("name", "path", "sha256"), "hand link {}".format(index))
        if not isinstance(entry["name"], str) or not entry["name"].strip():
            raise ValueError("hand link name must be non-empty")
        link_names.add(entry["name"])
        _sha256_text(entry["sha256"], "hand link sha256")
        if verify_files:
            link_path = Path(entry["path"])
            if not link_path.is_file() or _sha256_file(link_path) != entry["sha256"]:
                raise ValueError("hand link mesh no longer matches artifact")
    if len(link_names) != 13:
        raise ValueError("hand mesh link names must be unique")
    if any(item not in link_names for item in allowed_contact_links):
        raise ValueError("allowed object-contact policy references an unknown hand link")
    for key in ("urdf_sha256", "mapping_sha256", "open_fk_sha256", "closure_fk_sha256", "open_actuator_targets_sha256", "closure_actuator_targets_sha256"):
        _sha256_text(hand[key], "bindings.hand_model." + key)
    open_q = _finite_vector(hand["open_joint_positions_rad"], 12, "open hand q12")
    if not np.array_equal(open_q, OFFICIAL_OPEN_JOINT_POSITIONS_RAD):
        raise ValueError("open hand q12 is not the official [1000]*6 conversion")
    mapper_provenance = _expect_keys(
        hand["actuator_mapping_provenance"],
        (
            "algorithm", "driver_workbook_path", "driver_workbook_sha256",
            "integer_indexing", "interpolation",
        ),
        "bindings.hand_model.actuator_mapping_provenance",
    )
    mapper = OfficialRH56ActuatorMapper(mapper_provenance["driver_workbook_path"])
    if mapper.provenance() != mapper_provenance:
        raise ValueError("official actuator mapping provenance changed")
    waypoint_q12 = hand["execution_waypoint_joint_positions_rad"]
    waypoint_fk_hashes = hand["execution_waypoint_fk_sha256s"]
    hand_execution_waypoints = (
        () if hand_execution_path is None else hand_execution_path.waypoints
    )
    if (
        not isinstance(waypoint_q12, list)
        or not isinstance(waypoint_fk_hashes, list)
        or len(waypoint_q12) != len(hand_execution_waypoints)
        or len(waypoint_fk_hashes) != len(hand_execution_waypoints)
    ):
        raise ValueError("hand waypoint mapping/FK coverage is incomplete")
    for index, (item, stored_q12, fk_hash) in enumerate(
        zip(hand_execution_waypoints, waypoint_q12, waypoint_fk_hashes)
    ):
        expected_q12 = mapper.to_joint_positions_rad(item.actuator_configuration)
        if not np.array_equal(
            _finite_vector(stored_q12, 12, "hand waypoint q12"), expected_q12
        ):
            raise ValueError("hand waypoint {} differs from official XLS mapping".format(index))
        _sha256_text(fk_hash, "hand waypoint FK hash")
    dense_mapping = _expect_keys(
        hand["execution_dense_interval_mapping"],
        (
            "algorithm",
            "arrival_tolerance_units",
            "feedback_envelope_policy_sha256",
            "interval_sample_counts",
            "q12_intervals_sha256",
            "feedback_tube_q12_rad",
            "feedback_tubes_sha256",
            "feedback_q12_envelopes_sha256",
        ),
        "bindings.hand_model.execution_dense_interval_mapping",
    )
    if mode == "air_grasp":
        if hand_execution_path is None:
            raise ValueError("air audit has no canonical hand execution path")
        expected_tolerance = policies["hand_arrival_tolerance_units"]
        expected_dense_intervals, expected_feedback_tubes = (
            rh56_hand_interval_q12_paths_and_feedback_tubes(
                hand_execution_path, mapper, expected_tolerance
            )
        )
        expected_feedback_lower, expected_feedback_upper = (
            rh56_hand_interval_feedback_q12_envelopes(
                hand_execution_path, mapper, expected_tolerance
            )
        )
        expected_feedback_policy = rh56_feedback_envelope_policy(
            expected_tolerance
        )
        expected_dense_counts = [len(item) for item in expected_dense_intervals]
        stored_feedback_tubes = dense_mapping["feedback_tube_q12_rad"]
        if (
            dense_mapping["algorithm"]
            != "official_xls_every_integer_register_v1"
            or type(dense_mapping["arrival_tolerance_units"]) is not int
            or dense_mapping["arrival_tolerance_units"] != expected_tolerance
            or dense_mapping["feedback_envelope_policy_sha256"]
            != expected_feedback_policy["sha256"]
            or dense_mapping["interval_sample_counts"] != expected_dense_counts
            or dense_mapping["q12_intervals_sha256"]
            != dense_hand_interval_sha256(expected_dense_intervals)
            or not isinstance(stored_feedback_tubes, list)
            or len(stored_feedback_tubes) != len(expected_feedback_tubes)
        ):
            raise ValueError("dense hand interval mapping binding is invalid")
        for index, (stored, expected) in enumerate(
            zip(stored_feedback_tubes, expected_feedback_tubes)
        ):
            actual = _finite_vector(
                stored,
                12,
                "dense hand feedback tube {}".format(index),
            )
            if np.any(actual < 0.0) or not np.array_equal(actual, expected):
                raise ValueError("dense hand feedback tube binding is invalid")
        _sha256_text(
            dense_mapping["q12_intervals_sha256"],
            "dense hand interval mapping hash",
        )
        expected_tube_hash = dense_hand_interval_tubes_sha256(
            expected_feedback_tubes
        )
        _sha256_text(
            dense_mapping["feedback_tubes_sha256"],
            "dense hand feedback tube hash",
        )
        if dense_mapping["feedback_tubes_sha256"] != expected_tube_hash:
            raise ValueError("dense hand feedback tube hash is invalid")
        _sha256_text(
            dense_mapping["feedback_envelope_policy_sha256"],
            "dense hand feedback-envelope policy hash",
        )
        _sha256_text(
            dense_mapping["feedback_q12_envelopes_sha256"],
            "dense hand feedback q12 envelope hash",
        )
        if dense_mapping["feedback_q12_envelopes_sha256"] != (
            feedback_q12_envelopes_sha256(
                expected_feedback_lower, expected_feedback_upper
            )
        ):
            raise ValueError("dense hand feedback q12 envelope hash is invalid")
    elif (
        dense_mapping["algorithm"] != "not_applicable_loaded_grasp_v1"
        or dense_mapping["arrival_tolerance_units"] is not None
        or dense_mapping["feedback_envelope_policy_sha256"] != ""
        or dense_mapping["interval_sample_counts"] != []
        or dense_mapping["q12_intervals_sha256"] != ""
        or dense_mapping["feedback_tube_q12_rad"] != []
        or dense_mapping["feedback_tubes_sha256"] != ""
        or dense_mapping["feedback_q12_envelopes_sha256"] != ""
    ):
        raise ValueError("loaded audit dense hand interval mapping must be empty")
    open_provenance = _expect_keys(
        hand["open_configuration_provenance"],
        ("method", "actuator_targets", "joint_positions_rad", "sources"),
        "bindings.hand_model.open_configuration_provenance",
    )
    if (
        not isinstance(open_provenance["method"], str)
        or not np.array_equal(
            _finite_vector(
                open_provenance["actuator_targets"], 6, "open provenance targets"
            ),
            np.full(6, 1000.0),
        )
        or not np.array_equal(
            _finite_vector(
                open_provenance["joint_positions_rad"],
                12,
                "open provenance q12",
            ),
            open_q,
        )
        or not isinstance(open_provenance["sources"], dict)
        or set(open_provenance["sources"])
        != {"generator", "routine_workbook", "driver_workbook"}
    ):
        raise ValueError("open hand provenance is invalid")
    for name, source in open_provenance["sources"].items():
        item = _expect_keys(
            source,
            ("path", "sha256"),
            "open provenance source {}".format(name),
        )
        _sha256_text(item["sha256"], "open provenance source hash")
        if verify_files:
            source_path = Path(item["path"])
            if not source_path.is_file() or _sha256_file(source_path) != item["sha256"]:
                raise ValueError("open hand provenance source no longer matches")
    closed_q = _finite_vector(hand["closure_joint_positions_rad"], 12, "closed hand q12")
    open_targets = _finite_vector(hand["open_actuator_targets"], 6, "open actuator q6")
    if not np.array_equal(open_targets, np.full(6, 1000.0)) or _array_sha256(open_targets) != hand["open_actuator_targets_sha256"]:
        raise ValueError("open actuator q6 binding must be exactly [1000]*6")
    targets = _finite_vector(hand["closure_actuator_targets"], 6, "closure actuator q6")
    if (
        np.any(targets < 0)
        or np.any(targets > 1000)
        or not np.array_equal(targets, np.rint(targets))
        or _array_sha256(targets) != hand["closure_actuator_targets_sha256"]
    ):
        raise ValueError("closure actuator q6 binding is invalid")
    if not np.array_equal(targets, selected_targets):
        raise ValueError(
            "selected candidate hand targets differ from closure actuator targets"
        )
    if verify_files:
        for path_key, hash_key in (("urdf_path", "urdf_sha256"), ("mapping_path", "mapping_sha256")):
            path = Path(hand[path_key])
            if not path.is_file() or _sha256_file(path) != hand[hash_key]:
                raise ValueError("hand model source no longer matches artifact")
        replay_model = InspireHandModel(
            hand["urdf_path"],
            hand["mapping_path"],
            mesh_resolution=hand["mesh_resolution"],
        )
        replay_names = [link.name for link in replay_model.links]
        if replay_names != [entry["name"] for entry in hand["links"]]:
            raise ValueError("hand link order differs from the bound URDF")
        replay_open = replay_model.link_mesh_transforms(np.eye(4), open_q)
        replay_closed = replay_model.link_mesh_transforms(np.eye(4), closed_q)
        open_stack = np.stack([replay_open[name] for name in replay_names])
        closed_stack = np.stack([replay_closed[name] for name in replay_names])
        if _array_sha256(open_stack) != hand["open_fk_sha256"]:
            raise ValueError("open hand FK does not match bound q12/URDF")
        if _array_sha256(closed_stack) != hand["closure_fk_sha256"]:
            raise ValueError("closed hand FK does not match bound q12/URDF")

    backend = _expect_keys(
        root["collision_backend"],
        ("name", "version", "implementation_sha256", "configuration_sha256", "error"),
        "collision_backend",
    )
    CollisionBackendIdentity(
        backend["name"], backend["version"], backend["implementation_sha256"], backend["configuration_sha256"]
    )
    if not isinstance(backend["error"], str):
        raise ValueError("collision_backend.error must be a string")

    expected_preconditions = []
    if adapter["sha256"] != V7_ADAPTER_SHA256:
        expected_preconditions.append("adapter STL is not the commissioned V7 asset")
    if mount["sha256"] != V7_T_EE_HAND_SHA256:
        expected_preconditions.append("T_EE_hand does not match the commissioned V7 transform")
    scene_age = float(scene["age_at_audit_s"])
    current_age = float(path_binding["current_q_age_at_audit_s"])
    if not np.isclose(
        float(root["created_at_s"]) - float(scene["captured_at_s"]),
        scene_age,
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("scene age is inconsistent with audit/capture timestamps")
    if not np.isclose(
        float(root["created_at_s"]) - float(path_binding["current_q_captured_at_s"]),
        current_age,
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("current q age is inconsistent with audit/capture timestamps")
    if scene_age < -0.05 or scene_age > float(policies["max_scene_age_s"]):
        expected_preconditions.append("scene capture is outside the freshness window")
    if current_age < -0.05 or current_age > float(policies["max_current_q_age_s"]):
        expected_preconditions.append("current q is outside the freshness window")
    if preprocessing["validation_error"]:
        expected_preconditions.append(preprocessing["validation_error"])
    if joint_plan_manifest["validation_error"]:
        expected_preconditions.append(joint_plan_manifest["validation_error"])
    if adapter["sha256_after_audit"] != adapter["sha256"]:
        expected_preconditions.append("adapter STL changed during audit")
    if profile["sha256_after_audit"] != profile["sha256"]:
        expected_preconditions.append("control config changed during audit")
    if snapshot["sha256_after_audit"] != snapshot["sha256"]:
        expected_preconditions.append("snapshot changed during audit")
    if scene["sha256_after_audit"] != scene["sha256"]:
        expected_preconditions.append("scene source changed during audit")
    if object_binding["sha256_after_audit"] != object_binding["sha256"]:
        expected_preconditions.append("object source changed during audit")
    if (
        joint_plan_manifest["path"]
        and joint_plan_manifest["sha256_after_audit"]
        != joint_plan_manifest["sha256"]
    ):
        expected_preconditions.append("joint-plan manifest changed during audit")
    if (
        preprocessing["path"]
        and preprocessing["sha256_after_audit"] != preprocessing["sha256"]
    ):
        expected_preconditions.append(
            "installed-scene filter evidence changed during audit"
        )
    if backend["error"]:
        expected_preconditions.append("collision backend failed: {}".format(backend["error"]))

    if not isinstance(root["checks"], list) or len(root["checks"]) != len(check_specs):
        raise ValueError("checks must contain every required installed-tool check exactly once")
    recomputed_checks = []
    split_backend_name = (
        "static-cache-plus-fresh-pinocchio-hppfcl-fr3-v7-rh56"
    )
    split_backend = backend["name"] == split_backend_name
    if split_backend and (mode != "air_grasp" or backend["version"] != "1"):
        raise ValueError("static-cache collision backend is only valid for air_grasp v1")
    split_static_ids = tuple(
        spec.check_id
        for spec in check_specs_for_mode("air_grasp")
        if spec.check_id not in CAPTURED_POINT_CHECK_IDS
    )
    split_dynamic_ids = tuple(
        spec.check_id
        for spec in check_specs_for_mode("air_grasp")
        if spec.check_id in CAPTURED_POINT_CHECK_IDS
    )
    split_native_backend = None
    split_static_cache = None
    for spec, check in zip(check_specs, root["checks"]):
        item = _expect_keys(
            check,
            (
                "check_id", "scope", "expectation", "coverage", "authoritative",
                "tested_sample_indices", "expected_sample_count",
                "minimum_signed_distance_m", "observed_pairs", "details",
                "runtime_condition_ids", "pass_basis", "passed", "failures",
            ),
            "check",
        )
        if item["check_id"] != spec.check_id or item["scope"] != spec.scope or item["expectation"] != spec.expectation or item["coverage"] != spec.coverage:
            raise ValueError("collision checks are missing, reordered, or have altered semantics")
        if type(item["authoritative"]) is not bool or type(item["passed"]) is not bool:
            raise ValueError("collision check authority/pass fields must be booleans")
        if type(item["expected_sample_count"]) is not int:
            raise ValueError("collision expected_sample_count must be an integer")
        if not isinstance(item["tested_sample_indices"], list) or not isinstance(item["observed_pairs"], list) or not isinstance(item["runtime_condition_ids"], list) or not isinstance(item["pass_basis"], str) or not isinstance(item["failures"], list) or not isinstance(item["details"], dict):
            raise ValueError("collision check evidence fields have wrong JSON types")
        if split_backend and item["minimum_signed_distance_m"] is not None:
            provenance = _expect_keys(
                item["details"].get("split_evaluation_provenance"),
                (
                    "phase",
                    "combiner_contract",
                    "native_backend",
                    "static_cache",
                ),
                "split_evaluation_provenance",
            )
            if (
                provenance["combiner_contract"]
                != "cached-static-8-plus-fresh-point-10-v1"
            ):
                raise ValueError("split collision combiner contract is invalid")
            native = _expect_keys(
                provenance["native_backend"],
                (
                    "name",
                    "version",
                    "implementation_sha256",
                    "configuration_sha256",
                ),
                "split native backend",
            )
            CollisionBackendIdentity(
                native["name"],
                native["version"],
                native["implementation_sha256"],
                native["configuration_sha256"],
            )
            if split_native_backend is None:
                split_native_backend = dict(native)
            elif dict(native) != split_native_backend:
                raise ValueError("split checks bind different native backends")
            if spec.check_id in split_static_ids:
                if provenance["phase"] != "precomputed_static_mesh":
                    raise ValueError("static mesh check has the wrong split phase")
                cache = _expect_keys(
                    provenance["static_cache"],
                    ("path", "file_sha256", "payload_sha256"),
                    "split static cache",
                )
                if not isinstance(cache["path"], str) or not cache["path"]:
                    raise ValueError("split static cache path is invalid")
                _sha256_text(cache["file_sha256"], "split cache file sha256")
                _sha256_text(
                    cache["payload_sha256"], "split cache payload sha256"
                )
                if split_static_cache is None:
                    split_static_cache = dict(cache)
                elif dict(cache) != split_static_cache:
                    raise ValueError("static checks bind different cache artifacts")
            elif spec.check_id in split_dynamic_ids:
                if (
                    provenance["phase"] != "fresh_scene_object"
                    or provenance["static_cache"] is not None
                ):
                    raise ValueError("fresh point check has the wrong split phase")
            else:
                raise ValueError("split check is outside the exact 8+10 partition")
        observation = None
        if item["minimum_signed_distance_m"] is not None:
            observation = CollisionObservation(
                check_id=spec.check_id,
                authoritative=item["authoritative"],
                tested_sample_indices=tuple(item["tested_sample_indices"]),
                minimum_signed_distance_m=item["minimum_signed_distance_m"],
                observed_pairs=tuple(item["observed_pairs"]),
                details=item["details"],
            )
        rebuilt_check = _evaluate_observation(
            spec,
            observation,
            len(rebuilt),
            len(bindings["hand_execution_path"]["waypoints"]),
            policies,
            mode=mode,
        )
        if item != rebuilt_check:
            raise ValueError("collision check {} does not match policy evaluation".format(spec.check_id))
        recomputed_checks.append(rebuilt_check)

    if split_backend and split_native_backend is not None:
        if split_static_cache is None:
            raise ValueError("split backend has no bound static cache provenance")
        split_configuration = {
            "combiner_contract": "cached-static-8-plus-fresh-point-10-v1",
            "native_backend": split_native_backend,
            "static_cache": split_static_cache,
            "static_check_ids": list(split_static_ids),
            "fresh_dynamic_check_ids": list(split_dynamic_ids),
        }
        if _json_sha256(split_configuration) != backend["configuration_sha256"]:
            raise ValueError("split backend configuration hash is invalid")

    decision = _expect_keys(
        root["decision"],
        (
            "passed", "pass_kind", "motion_authorized",
            "all_required_checks_passed",
            "runtime_operator_workspace_clear_required",
            "runtime_conditions_satisfied_in_artifact",
            "precondition_failures", "reasons", "meaning",
        ),
        "decision",
    )
    if decision["motion_authorized"] is not False:
        raise ValueError("offline audit must never authorize motion")
    if not isinstance(decision["precondition_failures"], list) or not isinstance(decision["reasons"], list):
        raise ValueError("decision failure lists must be arrays")
    if decision["precondition_failures"] != expected_preconditions:
        raise ValueError("decision precondition failures are inconsistent with bindings")
    checks_pass = all(item["passed"] for item in recomputed_checks)
    expected_pass = not expected_preconditions and checks_pass
    if type(decision["passed"]) is not bool or decision["passed"] != expected_pass:
        raise ValueError("decision.passed is inconsistent with preconditions/checks")
    if decision["all_required_checks_passed"] != checks_pass:
        raise ValueError("decision all_required_checks_passed is inconsistent")
    if type(decision["all_required_checks_passed"]) is not bool:
        raise ValueError("decision all_required_checks_passed must be boolean")
    expected_runtime_condition = mode == "air_grasp"
    if (
        decision["runtime_operator_workspace_clear_required"]
        is not expected_runtime_condition
        or decision["runtime_conditions_satisfied_in_artifact"] is not False
    ):
        raise ValueError("decision runtime workspace-clear condition is inconsistent")
    expected_pass_kind = (
        "conditional_air"
        if expected_pass and expected_runtime_condition
        else ("authoritative" if expected_pass else "failed")
    )
    if decision["pass_kind"] != expected_pass_kind:
        raise ValueError("decision pass_kind is inconsistent")
    expected_reasons = list(expected_preconditions)
    expected_reasons.extend(
        "{}: {}".format(item["check_id"], failure)
        for item in recomputed_checks
        for failure in item["failures"]
    )
    if decision["reasons"] != expected_reasons:
        raise ValueError("decision reasons are inconsistent with checks")
    if decision["meaning"] != "offline collision evidence only; never a motion command":
        raise ValueError("decision meaning was altered")
    if require_pass and not decision["passed"]:
        raise ValueError("installed-tool audit did not pass: {}".format(decision["reasons"]))
    return artifact


def write_installed_tool_audit(path: Path, artifact: Mapping[str, Any]) -> Path:
    """Atomically write a validated, finite JSON artifact."""

    validate_installed_tool_audit(artifact)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(".{}.{}.tmp".format(output.name, os.getpid()))
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(output))
    return output


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key {!r}".format(key))
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant {!r} is forbidden".format(value))


def load_installed_tool_audit(
    path: Path, *, verify_files: bool = False, require_pass: bool = False
) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        artifact = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("cannot load installed-tool audit {}: {}".format(source, exc)) from exc
    return validate_installed_tool_audit(
        artifact, verify_files=verify_files, require_pass=require_pass
    )


__all__ = [
    "AIR_GRASP_CHECK_SPECS",
    "ARTIFACT_TYPE",
    "CHECK_SPECS",
    "CollisionBackendIdentity",
    "CollisionObservation",
    "InstalledToolAuditRequest",
    "InstalledToolCollisionBackend",
    "InstalledToolCollisionQuery",
    "LOADED_GRASP_CHECK_SPECS",
    "SCHEMA_VERSION",
    "V7_ADAPTER_SHA256",
    "V7_T_EE_HAND",
    "V7_T_EE_HAND_SHA256",
    "build_installed_tool_collision_query",
    "build_joint_path",
    "check_specs_for_mode",
    "load_installed_tool_audit",
    "run_installed_tool_audit",
    "validate_installed_tool_audit",
    "write_installed_tool_audit",
]
