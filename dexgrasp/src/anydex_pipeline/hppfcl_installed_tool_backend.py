"""Pinocchio/HPP-FCL backend for the installed FR3 + V7 + RH56 audit.

This module performs geometry queries only.  It never imports libfranka,
opens a serial port, reads a camera, or sends a motion command.

The implementation deliberately models the two kinds of input differently:

* FR3, V7 and all thirteen RH56 links use their triangle meshes.  Collision
  checks use the official FR3 collision meshes; depth-return filtering uses
  the official visual DAE shells so the white outer housing is represented.
  Clearances are exact HPP-FCL mesh distances.  When two triangle soups intersect,
  HPP-FCL reliably establishes collision but does not provide a trustworthy
  solid penetration depth; the reported clearance is therefore zero and the
  collision pair is recorded (which is sufficient to fail every ``clear``
  policy).
* Scene/object samples are represented as conservative, axis-aligned cubes.
  Mesh-versus-cube queries provide a genuine signed distance.  This is exact
  for the *bound cube union*, but a single captured point cloud cannot prove
  that unobserved/occluded workspace is empty.  Point-cloud observations are
  consequently non-authoritative.  A loaded/contact audit fails closed; an
  unloaded air audit may only pass conditionally behind the separate exact
  runtime workspace-clear token.  There is no switch that promotes point
  evidence to geometric authority.

The checked path is the sampled joint path in
``InstalledToolCollisionQuery.q_path_rad``.  Evidence from this backend is
made continuous by subtracting a configuration-independent serial-chain
influence-radius bound for every interval and for the configured joint
tracking tube.  It is authoritative only for an executor that follows that
piecewise-linear joint path and enforces the same tracking bound.  It is not
evidence for a Cartesian interpolation between the same endpoint poses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .installed_tool_audit import (
    CollisionBackendIdentity,
    CollisionObservation,
    InstalledToolCollisionQuery,
    check_specs_for_mode,
)
from .rh56_hand_path import rh56_feedback_envelope_policy


DEFAULT_FR3_URDF = Path("/home/qiaoguanren/code/libfranka/test/fr3.urdf")
DEFAULT_FRANKA_DESCRIPTION_SHARE = Path(
    "/opt/ros/humble/share/franka_description"
)
DEFAULT_FR3_SRDF_XACRO = (
    DEFAULT_FRANKA_DESCRIPTION_SHARE / "robots/common/franka_arm.srdf.xacro"
)

# Pairs disabled by the official Franka SRDF.  The source file is hashed into
# the backend identity; these names are intentionally explicit so an upstream
# SRDF change cannot silently alter the audit policy.
OFFICIAL_FRANKA_DISABLED_SELF_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("link0", "link1"),
    ("link0", "link2"),
    ("link0", "link3"),
    ("link0", "link4"),
    ("link1", "link2"),
    ("link1", "link3"),
    ("link1", "link4"),
    ("link2", "link3"),
    ("link2", "link4"),
    ("link2", "link6"),
    ("link3", "link4"),
    ("link3", "link5"),
    ("link3", "link6"),
    ("link3", "link7"),
    ("link4", "link5"),
    ("link4", "link6"),
    ("link4", "link7"),
    ("link5", "link6"),
    ("link5", "link7"),
    ("link6", "link7"),
)

# The adapter is rigidly bolted at link7/link8.  Those installation surfaces
# are not obstacle pairs.  link8 has no collision STL in the official model.
ADAPTER_FR3_MOUNT_EXCLUSIONS: Tuple[str, ...] = ("link7", "link8")

# Link111 contains the RH56 wrist/mount shell.  Its designed seating/spigot
# interface overlaps the adapter representation and is not an obstacle pair.
RH56_ADAPTER_MOUNT_EXCLUSIONS: Tuple[str, ...] = ("Link111",)

# V7 STL frame A: +Z is the flange axis and its pin feature establishes this
# 135 degree yaw in the FR3 EE frame.  The mesh origin is on the flange face.
T_EE_ADAPTER = np.asarray(
    [
        [-math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0],
        [math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _transform_sha256(value: np.ndarray) -> str:
    matrix = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(b"dtype=<f8;shape=4,4;")
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class HppFclInstalledToolConfig:
    """Immutable geometry and evidence policy for the native backend."""

    fr3_urdf_path: Path = DEFAULT_FR3_URDF
    franka_description_share: Path = DEFAULT_FRANKA_DESCRIPTION_SHARE
    fr3_srdf_xacro_path: Path = DEFAULT_FR3_SRDF_XACRO
    adapter_mesh_scale: float = 0.001
    # Default request policy uses 5 mm voxels; the conservative cube is one
    # full voxel wide around each retained voxel centre.
    scene_point_half_extent_m: float = 0.0025
    object_point_half_extent_m: float = 0.0015
    max_q_tracking_error_rad: float = 0.002
    # A captured point cloud is not a proof that occluded space is empty.  The
    # field is fixed false in __post_init__ and is included here to make the
    # fail-closed evidence policy visible in the identity hash.
    point_cloud_observations_authoritative: bool = False
    # The backend subtracts a serial-chain influence-radius bound covering
    # both the allowed tracking tube and every interval between q samples.
    joint_tracking_uncertainty_applied: bool = True
    continuous_segment_envelope_verified: bool = True

    def __post_init__(self) -> None:
        for name in (
            "fr3_urdf_path",
            "franka_description_share",
            "fr3_srdf_xacro_path",
        ):
            path = Path(getattr(self, name)).expanduser().resolve()
            object.__setattr__(self, name, path)
        for name in (
            "adapter_mesh_scale",
            "scene_point_half_extent_m",
            "object_point_half_extent_m",
            "max_q_tracking_error_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))
            object.__setattr__(self, name, value)
        if self.point_cloud_observations_authoritative is not False:
            raise ValueError(
                "a captured point cloud cannot be promoted to authoritative "
                "workspace evidence"
            )
        if self.joint_tracking_uncertainty_applied is not True:
            raise ValueError(
                "joint tracking uncertainty coverage cannot be disabled"
            )
        if self.continuous_segment_envelope_verified is not True:
            raise ValueError(
                "continuous segment envelope verification cannot be disabled"
            )


@dataclass
class _PointCubeUnion:
    points: np.ndarray
    half_extent_m: float
    geometry: Any
    objects: List[Any]
    manager: Any


@dataclass(frozen=True)
class _DistanceEvidence:
    distance_m: float
    pair: str
    sample_index: int
    nearest_point_a: Optional[Tuple[float, float, float]]
    nearest_point_b: Optional[Tuple[float, float, float]]
    intersecting: bool
    penetration_depth_trustworthy: bool


@dataclass(frozen=True)
class _AdaptiveHandBoxResult:
    """Fail-closed clearance proof for one RH56 self-pair feedback box."""

    certified: bool
    lower_bound_m: float
    evaluated_nodes: int
    maximum_depth: int
    minimum_sampled_distance_m: float
    reason: str


@dataclass(frozen=True)
class InstalledReturnClassification:
    """Replayable point indices classified as installed-tool depth returns."""

    removed_indices: np.ndarray
    reason_labels: Tuple[str, ...]
    candidate_test_count: int
    point_half_extent_m: float
    inflation_margin_m: float
    fr3_geometry_model: str
    fr3_mesh_provenance: Tuple[Tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        indices = np.asarray(self.removed_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) != len(self.reason_labels):
            raise ValueError("installed return indices/reasons have inconsistent lengths")
        if len(indices) and (
            np.any(indices < 0)
            or not np.array_equal(indices, np.unique(indices))
        ):
            raise ValueError("installed return indices must be sorted and unique")
        if any(not str(item).strip() for item in self.reason_labels):
            raise ValueError("installed return reason labels must be non-empty")
        if self.fr3_geometry_model != "official_fr3_visual_triangle_meshes":
            raise ValueError("installed-return FR3 filter must use official visual meshes")
        provenance = tuple(tuple(item) for item in self.fr3_mesh_provenance)
        if len(provenance) != 8 or any(len(item) != 3 for item in provenance):
            raise ValueError("FR3 visual mesh provenance must contain eight entries")
        if tuple(item[0] for item in provenance) != tuple(
            "link{}".format(index) for index in range(8)
        ):
            raise ValueError("FR3 visual mesh provenance must be ordered link0..link7")
        for _, path, digest in provenance:
            if not str(path) or len(str(digest)) != 64:
                raise ValueError("FR3 visual mesh provenance is malformed")
        object.__setattr__(self, "removed_indices", indices)
        object.__setattr__(self, "reason_labels", tuple(self.reason_labels))
        object.__setattr__(self, "fr3_mesh_provenance", provenance)


@dataclass(frozen=True)
class InstalledReturnNeighborhood:
    """Points inside a strict capture-state installed-mesh distance bound.

    Proximity alone never authorizes deleting a scene point.  The scene
    filter combines these candidates with appearance and connectivity to an
    exact-intersection seed.
    """

    candidate_indices: np.ndarray
    reason_labels: Tuple[str, ...]
    surface_distances_m: np.ndarray
    candidate_test_count: int
    maximum_surface_distance_m: float
    distance_method: str

    def __post_init__(self) -> None:
        indices = np.asarray(self.candidate_indices, dtype=np.int64)
        distances = np.asarray(self.surface_distances_m, dtype=np.float64)
        maximum = float(self.maximum_surface_distance_m)
        if (
            indices.ndim != 1
            or distances.shape != indices.shape
            or len(indices) != len(self.reason_labels)
        ):
            raise ValueError("installed neighborhood arrays have inconsistent lengths")
        if len(indices) and (
            np.any(indices < 0)
            or not np.array_equal(indices, np.unique(indices))
            or not np.all(np.isfinite(distances))
            or np.any(distances > maximum + 1e-9)
        ):
            raise ValueError("installed neighborhood candidates violate their bound")
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum installed surface distance must be positive")
        if self.distance_method != "HPP-FCL triangle-mesh to 1nm sphere, radius corrected":
            raise ValueError("installed neighborhood distance method is invalid")
        if any(not str(item).strip() for item in self.reason_labels):
            raise ValueError("installed neighborhood reason labels must be non-empty")
        object.__setattr__(self, "candidate_indices", indices)
        object.__setattr__(self, "surface_distances_m", distances)
        object.__setattr__(self, "reason_labels", tuple(self.reason_labels))


def _finite_points(points: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1:] != (3,) or len(value) == 0:
        raise ValueError("{} must be a non-empty (N,3) array".format(name))
    if not np.all(np.isfinite(value)):
        raise ValueError("{} contains NaN or infinity".format(name))
    return value


class HppFclInstalledToolBackend:
    """Exact-mesh, hardware-free backend for ``run_installed_tool_audit``."""

    def __init__(
        self,
        config: Optional[HppFclInstalledToolConfig] = None,
        *,
        pinocchio_module: Any = None,
        hppfcl_module: Any = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or HppFclInstalledToolConfig()
        # Reporting is intentionally not part of the backend identity: it
        # changes terminal visibility only, never geometry or evidence.
        self._progress_callback = progress_callback
        for path in (
            self.config.fr3_urdf_path,
            self.config.franka_description_share,
            self.config.fr3_srdf_xacro_path,
        ):
            if not path.exists():
                raise FileNotFoundError("native collision input not found: {}".format(path))

        if pinocchio_module is None or hppfcl_module is None:
            try:
                import pinocchio as native_pinocchio
                import hppfcl as native_hppfcl
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "Pinocchio/HPP-FCL could not be imported.  On this machine "
                    "use /usr/bin/python3 (ROS Humble Python 3.10), not the "
                    "Python 3.9 project virtualenv."
                ) from exc
            if pinocchio_module is None:
                pinocchio_module = native_pinocchio
            if hppfcl_module is None:
                hppfcl_module = native_hppfcl
        self.pin = pinocchio_module
        self.fcl = hppfcl_module

        try:
            model, collision_model, visual_model = self.pin.buildModelsFromUrdf(
                str(self.config.fr3_urdf_path),
                [str(self.config.franka_description_share.parent)],
            )
        except Exception as exc:
            raise RuntimeError("failed to load the official FR3 collision model") from exc
        if (
            int(model.nq) != 7
            or len(collision_model.geometryObjects) != 8
            or len(visual_model.geometryObjects) != 8
        ):
            raise RuntimeError(
                "unexpected FR3 geometry topology: nq={}, collision_meshes={}, "
                "visual_meshes={}".format(
                    model.nq,
                    len(collision_model.geometryObjects),
                    len(visual_model.geometryObjects),
                )
            )
        self.model = model
        self.collision_model = collision_model
        self.visual_model = visual_model
        self._robot_link_names = tuple(
            self.model.frames[item.parentFrame].name
            for item in self.collision_model.geometryObjects
        )
        if self._robot_link_names != tuple("link{}".format(i) for i in range(8)):
            raise RuntimeError(
                "FR3 collision meshes are not ordered link0..link7: {}".format(
                    self._robot_link_names
                )
            )
        self._robot_geometries = tuple(
            item.geometry for item in self.collision_model.geometryObjects
        )
        self._robot_visual_link_names = tuple(
            self.model.frames[item.parentFrame].name
            for item in self.visual_model.geometryObjects
        )
        if self._robot_visual_link_names != tuple(
            "link{}".format(i) for i in range(8)
        ):
            raise RuntimeError(
                "FR3 visual meshes are not ordered link0..link7: {}".format(
                    self._robot_visual_link_names
                )
            )
        self._robot_visual_geometries = tuple(
            item.geometry for item in self.visual_model.geometryObjects
        )
        self._robot_visual_mesh_provenance = tuple(
            (
                name,
                str(Path(item.meshPath).expanduser().resolve()),
                _sha256_file(Path(item.meshPath).expanduser().resolve()),
            )
            for name, item in zip(
                self._robot_visual_link_names, self.visual_model.geometryObjects
            )
        )
        self._link8_frame_id = int(self.model.getFrameId("link8"))
        if self._link8_frame_id >= len(self.model.frames):
            raise RuntimeError("FR3 URDF has no link8 frame")
        self._T_joint7_EE = self._matrix_from_se3(
            self.model.frames[self._link8_frame_id].placement
        )
        self._robot_influences = {
            name: self._joint_influence_radii(
                int(item.parentJoint),
                self._matrix_from_se3(item.placement),
                item.geometry,
            )
            for name, item in zip(
                self._robot_link_names, self.collision_model.geometryObjects
            )
        }
        self._robot_parent_joints = {
            name: int(item.parentJoint)
            for name, item in zip(
                self._robot_link_names, self.collision_model.geometryObjects
            )
        }

        disabled = {tuple(sorted(item)) for item in OFFICIAL_FRANKA_DISABLED_SELF_PAIRS}
        self._self_pairs = tuple(
            (first, second)
            for first in range(8)
            for second in range(first + 1, 8)
            if tuple(sorted((self._robot_link_names[first], self._robot_link_names[second])))
            not in disabled
        )
        if not self._self_pairs:
            raise RuntimeError("official SRDF exclusions removed every FR3 self pair")

        implementation_path = Path(__file__).resolve()
        config_binding = {
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self.config).items()
            },
            "fr3_urdf_sha256": _sha256_file(self.config.fr3_urdf_path),
            "fr3_srdf_xacro_sha256": _sha256_file(
                self.config.fr3_srdf_xacro_path
            ),
            "fr3_collision_meshes": [
                {
                    "link": name,
                    "path": str(Path(item.meshPath).resolve()),
                    "sha256": _sha256_file(Path(item.meshPath).resolve()),
                }
                for name, item in zip(
                    self._robot_link_names, self.collision_model.geometryObjects
                )
            ],
            # A set is useful for membership above, but its iteration order is
            # process-randomized.  Sort before hashing so an immutable backend
            # has the same identity in the precompute and fresh subprocesses.
            "official_disabled_self_pairs": [
                list(item) for item in sorted(disabled)
            ],
            "adapter_fr3_mount_exclusions": list(ADAPTER_FR3_MOUNT_EXCLUSIONS),
            "rh56_adapter_mount_exclusions": list(RH56_ADAPTER_MOUNT_EXCLUSIONS),
            "T_EE_adapter_sha256": _transform_sha256(T_EE_ADAPTER),
            "path_semantics": "piecewise_linear_in_joint_space_over_bound_q_samples",
        }
        self._identity = CollisionBackendIdentity(
            name="pinocchio-hppfcl-fr3-v7-rh56",
            version="1",
            implementation_sha256=_sha256_file(implementation_path),
            configuration_sha256=_json_sha256(config_binding),
        )
        self._identity_details = config_binding

    @property
    def identity(self) -> CollisionBackendIdentity:
        return self._identity

    def _tf(self, matrix: np.ndarray) -> Any:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError("collision transform must be finite (4,4)")
        return self.fcl.Transform3f(value[:3, :3], value[:3, 3])

    @staticmethod
    def _matrix_from_se3(value: Any) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(value.rotation, dtype=np.float64)
        matrix[:3, 3] = np.asarray(value.translation, dtype=np.float64).reshape(3)
        return matrix

    def _object(self, geometry: Any, matrix: np.ndarray) -> Any:
        result = self.fcl.CollisionObject(geometry, self._tf(matrix))
        result.computeAABB()
        return result

    def _load_mesh(self, path: Path, *, scale: float = 1.0) -> Any:
        mesh_path = Path(path).expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError("collision mesh not found: {}".format(mesh_path))
        loader = self.fcl.MeshLoader()
        geometry = loader.load(
            str(mesh_path), np.full(3, float(scale), dtype=np.float64)
        )
        geometry.computeLocalAABB()
        if int(geometry.num_tris) <= 0 or int(geometry.num_vertices) <= 0:
            raise RuntimeError("collision mesh is empty: {}".format(mesh_path))
        return geometry

    @staticmethod
    def _mesh_origin_radius(geometry: Any) -> float:
        """Maximum vertex norm in a mesh's local coordinates."""

        try:
            vertices = np.asarray(geometry.vertices(), dtype=np.float64)
        except Exception as exc:
            raise RuntimeError("collision geometry does not expose mesh vertices") from exc
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
            raise RuntimeError("collision mesh has no finite (N,3) vertices")
        if not np.all(np.isfinite(vertices)):
            raise RuntimeError("collision mesh vertices contain NaN or infinity")
        return float(np.max(np.linalg.norm(vertices, axis=1)))

    def _joint_influence_radii(
        self,
        parent_joint: int,
        T_parent_geometry: np.ndarray,
        geometry: Any,
    ) -> np.ndarray:
        """Configuration-independent point-motion radii for FR3 joints.

        For a point rigidly attached below joint ``k``, its distance from an
        upstream joint ``j`` is bounded by the sum of intervening URDF
        translation norms plus the local geometry radius.  Rotations preserve
        every term's norm, so the bound is valid over the entire joint path.
        """

        parent = int(parent_joint)
        if parent < 0 or parent > 7:
            raise ValueError("FR3 collision parent joint must be in [0,7]")
        local = np.asarray(T_parent_geometry, dtype=np.float64)
        if local.shape != (4, 4) or not np.all(np.isfinite(local)):
            raise ValueError("local collision transform must be finite (4,4)")
        terminal_radius = float(np.linalg.norm(local[:3, 3])) + self._mesh_origin_radius(
            geometry
        )
        radii = np.zeros(7, dtype=np.float64)
        for joint in range(1, parent + 1):
            intervening = sum(
                float(
                    np.linalg.norm(
                        np.asarray(self.model.jointPlacements[index].translation)
                    )
                )
                for index in range(joint + 1, parent + 1)
            )
            radii[joint - 1] = terminal_radius + intervening
        return radii

    def _hand_joint_influence_radii(
        self, hand_model: Any, hand_geometries: Mapping[str, Any]
    ) -> Mapping[str, np.ndarray]:
        """Configuration-independent radii for each RH56 link and 12 URDF joints."""

        joints = tuple(hand_model.joints)
        if len(joints) != 12:
            raise ValueError("RH56 model must contain exactly 12 revolute joints")
        joint_by_child = {joint.child: (index, joint) for index, joint in enumerate(joints)}
        visual_by_link = {link.name: link for link in hand_model.links}
        result: Dict[str, np.ndarray] = {}
        for link_name, geometry in hand_geometries.items():
            chain = []
            cursor = link_name
            while cursor in joint_by_child:
                index, joint = joint_by_child[cursor]
                chain.append((index, joint))
                cursor = joint.parent
            chain.reverse()
            terminal = self._mesh_origin_radius(geometry) + float(
                np.linalg.norm(visual_by_link[link_name].T_link_visual[:3, 3])
            )
            radii = np.zeros(12, dtype=np.float64)
            for chain_index, (joint_index, _joint) in enumerate(chain):
                descendant = sum(
                    float(np.linalg.norm(item.T_parent_joint[:3, 3]))
                    for _index, item in chain[chain_index + 1 :]
                )
                radii[joint_index] = terminal + descendant
            result[link_name] = radii
        return result

    def _adaptive_hand_self_feedback_box(
        self,
        *,
        hand_model: Any,
        first_name: str,
        second_name: str,
        first_geometry: Any,
        second_geometry: Any,
        lower_q12_rad: np.ndarray,
        upper_q12_rad: np.ndarray,
        pair_influence_radii_m: np.ndarray,
        required_clearance_m: float,
        maximum_nodes: int = 4095,
        maximum_depth: int = 32,
    ) -> _AdaptiveHandBoxResult:
        """Prove a self-pair feedback box by adaptive q12 subdivision.

        Each box centre is evaluated with the exact triangle meshes.  For a
        revolute joint, every point at most ``R`` from the axis moves by at
        most ``2 R sin(delta/2)`` over an angular displacement ``delta``.
        Summing those per-joint displacements gives a configuration-independent
        lower distance bound for the whole box.  A box is accepted only when
        that lower bound exceeds the unchanged policy margin; otherwise it is
        bisected along the joint with the largest bound contribution.

        Reaching either resource limit is an unresolved result, never PASS.
        The input feedback envelope is already a component-wise enlargement
        of the exact official-mapping register set, so subdivision cannot omit
        a reachable state or weaken the configured arrival tolerance.
        """

        lower = np.asarray(lower_q12_rad, dtype=np.float64)
        upper = np.asarray(upper_q12_rad, dtype=np.float64)
        radii = np.asarray(pair_influence_radii_m, dtype=np.float64)
        margin = float(required_clearance_m)
        if (
            lower.shape != (12,)
            or upper.shape != (12,)
            or radii.shape != (12,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or not np.all(np.isfinite(radii))
            or np.any(lower > upper)
            or np.any(radii < 0.0)
            or not np.isfinite(margin)
            or margin < 0.0
        ):
            raise ValueError("invalid adaptive RH56 feedback box")
        if maximum_nodes < 1 or maximum_depth < 0:
            raise ValueError("adaptive RH56 subdivision limits are invalid")

        # Joints outside both serial chains cannot change this pair.  Pinning
        # them to the box midpoint reduces numerical work without shrinking
        # the reachable transforms of either tested link.
        inactive = radii <= 0.0
        midpoint = 0.5 * (lower + upper)
        lower = lower.copy()
        upper = upper.copy()
        lower[inactive] = midpoint[inactive]
        upper[inactive] = midpoint[inactive]

        stack = [(lower, upper, 0)]
        evaluated = 0
        deepest = 0
        minimum_sampled = float("inf")
        minimum_leaf_bound = float("inf")
        while stack:
            if evaluated >= int(maximum_nodes):
                return _AdaptiveHandBoxResult(
                    False,
                    float("-inf"),
                    evaluated,
                    deepest,
                    minimum_sampled,
                    "node_limit",
                )
            box_lower, box_upper, depth = stack.pop()
            deepest = max(deepest, int(depth))
            q12 = 0.5 * (box_lower + box_upper)
            transforms = hand_model.link_mesh_transforms(np.eye(4), q12)
            first_object = self._object(
                first_geometry, transforms[first_name]
            )
            second_object = self._object(
                second_geometry, transforms[second_name]
            )
            evidence = self._mesh_pair_distance(
                first_object,
                second_object,
                pair="{} / {}".format(first_name, second_name),
                sample_index=-1,
            )
            evaluated += 1
            minimum_sampled = min(minimum_sampled, evidence.distance_m)

            half_width = 0.5 * (box_upper - box_lower)
            # A rotation interval wider than pi has the universal 2R chord
            # bound.  RH56 calibrated ranges are much smaller, but clipping
            # keeps the proof valid if a future mapping changes unexpectedly.
            bounded_delta = np.minimum(half_width, np.pi)
            contributions = 2.0 * radii * np.sin(0.5 * bounded_delta)
            displacement_bound = float(np.sum(contributions))
            leaf_lower_bound = float(evidence.distance_m - displacement_bound)
            if leaf_lower_bound > margin:
                minimum_leaf_bound = min(minimum_leaf_bound, leaf_lower_bound)
                continue

            split_joint = int(np.argmax(contributions))
            if (
                depth >= int(maximum_depth)
                or contributions[split_joint] <= 1.0e-12
                or box_upper[split_joint] - box_lower[split_joint] <= 1.0e-12
            ):
                return _AdaptiveHandBoxResult(
                    False,
                    leaf_lower_bound,
                    evaluated,
                    deepest,
                    minimum_sampled,
                    "unresolved_clearance",
                )
            split = 0.5 * (
                box_lower[split_joint] + box_upper[split_joint]
            )
            first_upper = box_upper.copy()
            first_upper[split_joint] = split
            second_lower = box_lower.copy()
            second_lower[split_joint] = split
            stack.append((second_lower, box_upper.copy(), depth + 1))
            stack.append((box_lower.copy(), first_upper, depth + 1))

        if not np.isfinite(minimum_leaf_bound):
            return _AdaptiveHandBoxResult(
                False,
                float("-inf"),
                evaluated,
                deepest,
                minimum_sampled,
                "no_certified_leaf",
            )
        return _AdaptiveHandBoxResult(
            True,
            minimum_leaf_bound,
            evaluated,
            deepest,
            minimum_sampled,
            "certified",
        )

    @staticmethod
    def _maximum_influence(values: Iterable[np.ndarray]) -> np.ndarray:
        items = [np.asarray(item, dtype=np.float64) for item in values]
        if not items or any(item.shape != (7,) for item in items):
            raise RuntimeError("influence set must contain finite 7-vectors")
        stack = np.stack(items, axis=0)
        if not np.all(np.isfinite(stack)) or np.any(stack < 0.0):
            raise RuntimeError("joint influence radii must be finite and non-negative")
        return np.max(stack, axis=0)

    def _continuous_motion_bound(
        self,
        influence_radii_m: np.ndarray,
        q_path_rad: np.ndarray,
        *,
        coverage: str,
    ) -> float:
        """Hausdorff displacement bound for interpolation plus tracking.

        Any point on a body at an intermediate linear-q state is within
        ``2 R_j sin(delta_j/2)`` of the same body at its nearest sampled
        endpoint.  ``delta_j`` includes half the interval width and the full
        configured tracking error.  Summing over revolute joints is a strict
        triangle-inequality bound.  For two moving bodies callers pass the sum
        of their influence radii.
        """

        radii = np.asarray(influence_radii_m, dtype=np.float64)
        if radii.shape != (7,) or not np.all(np.isfinite(radii)) or np.any(radii < 0):
            raise ValueError("influence_radii_m must be a non-negative 7-vector")
        tracking = float(self.config.max_q_tracking_error_rad)
        if coverage == "final":
            deltas = np.full((1, 7), tracking, dtype=np.float64)
        elif coverage == "full_path":
            path = np.asarray(q_path_rad, dtype=np.float64)
            if len(path) < 2:
                deltas = np.full((1, 7), tracking, dtype=np.float64)
            else:
                deltas = 0.5 * np.abs(np.diff(path, axis=0)) + tracking
        else:
            raise ValueError("unknown collision coverage {!r}".format(coverage))
        per_interval = np.sum(
            2.0 * radii[None, :] * np.sin(0.5 * deltas), axis=1
        )
        return float(np.max(per_interval))

    @staticmethod
    def _relative_influence_radii(
        first: np.ndarray,
        first_parent_joint: int,
        second: np.ndarray,
        second_parent_joint: int,
    ) -> np.ndarray:
        """Bound relative, rather than world, motion of two serial-chain bodies.

        Every joint at or above the nearest common ancestor applies the same
        rigid transform to both bodies and cannot change their mutual distance.
        Removing those common terms is both rigorous and much tighter for the
        fixed adapter/hand stack near link6/link7.
        """

        result = np.asarray(first, dtype=np.float64) + np.asarray(
            second, dtype=np.float64
        )
        common_parent = min(int(first_parent_joint), int(second_parent_joint))
        if common_parent > 0:
            result[:common_parent] = 0.0
        return result

    @staticmethod
    def _pair_influence_radii(
        check_id: str,
        pair: str,
        *,
        robot: Mapping[str, np.ndarray],
        robot_parents: Mapping[str, int],
        adapter: np.ndarray,
        hand_open: Mapping[str, np.ndarray],
        hand_closed: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        names = tuple(item.strip() for item in pair.split(" / "))
        if len(names) != 2:
            raise RuntimeError("unexpected collision pair label {!r}".format(pair))
        first, second = names
        if check_id == "fr3_self_path":
            return HppFclInstalledToolBackend._relative_influence_radii(
                robot[first], robot_parents[first], robot[second], robot_parents[second]
            )
        if check_id == "fr3_scene_path":
            return robot[first]
        if check_id == "adapter_fr3_path":
            return HppFclInstalledToolBackend._relative_influence_radii(
                adapter, 7, robot[second], robot_parents[second]
            )
        if check_id in ("adapter_scene_path", "adapter_object_path"):
            return adapter
        if check_id == "rh56_open_fr3_path":
            return HppFclInstalledToolBackend._relative_influence_radii(
                hand_open[first], 7, robot[second], robot_parents[second]
            )
        if check_id in ("rh56_open_scene_path", "rh56_open_object_path"):
            return hand_open[first]
        if check_id == "rh56_closed_fr3_final":
            return HppFclInstalledToolBackend._relative_influence_radii(
                hand_closed[first], 7, robot[second], robot_parents[second]
            )
        if check_id == "rh56_closed_adapter_final":
            return HppFclInstalledToolBackend._relative_influence_radii(
                hand_closed[first], 7, adapter, 7
            )
        if check_id in (
            "rh56_closed_scene_final",
            "rh56_closed_object_noncontact_final",
            "rh56_closed_object_contact_final",
            "rh56_closed_object_all_links_final",
            "rh56_execution_scene_final",
            "rh56_execution_object_final",
            "rh56_execution_scene_final",
            "rh56_execution_object_final",
        ):
            return hand_closed[first]
        raise RuntimeError("no motion-influence rule for {}".format(check_id))

    def _conservative_distance_evidence(
        self,
        items: Sequence[_DistanceEvidence],
        q_path_rad: np.ndarray,
        *,
        check_id: str,
        coverage: str,
        robot_influences: Mapping[str, np.ndarray],
        robot_parent_joints: Mapping[str, int],
        adapter_influence: np.ndarray,
        open_influences: Mapping[str, np.ndarray],
        closed_influences: Mapping[str, np.ndarray],
    ) -> Tuple[float, float, np.ndarray, _DistanceEvidence, Tuple[int, int]]:
        """Return the tightest pair/interval-specific conservative lower bound."""

        if not items:
            raise RuntimeError("collision check has no distance evidence")

        def influence(item: _DistanceEvidence) -> np.ndarray:
            return self._pair_influence_radii(
                check_id,
                item.pair,
                robot=robot_influences,
                robot_parents=robot_parent_joints,
                adapter=adapter_influence,
                hand_open=open_influences,
                hand_closed=closed_influences,
            )

        best: Optional[Tuple[float, float, np.ndarray, _DistanceEvidence, Tuple[int, int]]] = None
        if coverage == "final":
            for item in items:
                radii = influence(item)
                bound = self._continuous_motion_bound(
                    radii, q_path_rad, coverage="final"
                )
                candidate = float(item.distance_m - bound)
                value = (candidate, bound, radii, item, (item.sample_index, item.sample_index))
                if best is None or candidate < best[0]:
                    best = value
        elif coverage == "full_path":
            by_pair: Dict[str, Dict[int, _DistanceEvidence]] = {}
            for item in items:
                by_pair.setdefault(item.pair, {})[item.sample_index] = item
            if len(q_path_rad) == 1:
                for item in items:
                    radii = influence(item)
                    bound = self._continuous_motion_bound(
                        radii, q_path_rad, coverage="final"
                    )
                    candidate = float(item.distance_m - bound)
                    value = (candidate, bound, radii, item, (0, 0))
                    if best is None or candidate < best[0]:
                        best = value
            else:
                for pair, samples in by_pair.items():
                    expected = set(range(len(q_path_rad)))
                    if set(samples) != expected:
                        raise RuntimeError(
                            "pair {} lacks full q-sample coverage".format(pair)
                        )
                    radii = influence(samples[0])
                    for index in range(len(q_path_rad) - 1):
                        first = samples[index]
                        second = samples[index + 1]
                        nominal = first if first.distance_m <= second.distance_m else second
                        bound = self._continuous_motion_bound(
                            radii,
                            q_path_rad[index : index + 2],
                            coverage="full_path",
                        )
                        candidate = float(nominal.distance_m - bound)
                        value = (candidate, bound, radii, nominal, (index, index + 1))
                        if best is None or candidate < best[0]:
                            best = value
        else:
            raise RuntimeError("unknown coverage {!r}".format(coverage))
        if best is None:
            raise RuntimeError("continuous collision envelope produced no result")
        return best

    def _point_union(self, points: np.ndarray, half_extent: float) -> _PointCubeUnion:
        values = _finite_points(points, "point cloud")
        side = 2.0 * float(half_extent)
        geometry = self.fcl.Box(side, side, side)
        objects = [
            self.fcl.CollisionObject(
                geometry,
                self.fcl.Transform3f(np.eye(3), point),
            )
            for point in values
        ]
        manager = self.fcl.DynamicAABBTreeCollisionManager()
        manager.registerObjects(objects)
        manager.setup()
        return _PointCubeUnion(values, float(half_extent), geometry, objects, manager)

    def _raw_distance(self, first: Any, second: Any) -> Tuple[float, Any]:
        request = self.fcl.DistanceRequest(True)
        request.enable_nearest_points = True
        result = self.fcl.DistanceResult()
        distance = float(
            self.fcl.distance(first, second, request, result)
        )
        return distance, result

    @staticmethod
    def _nearest(result: Any, which: int) -> Optional[Tuple[float, float, float]]:
        try:
            value = (
                result.getNearestPoint1()
                if which == 1
                else result.getNearestPoint2()
            )
            point = np.asarray(value, dtype=np.float64).reshape(3)
        except Exception:
            return None
        if not np.all(np.isfinite(point)):
            return None
        return tuple(float(item) for item in point)

    def _mesh_pair_distance(
        self,
        first: Any,
        second: Any,
        *,
        pair: str,
        sample_index: int,
    ) -> _DistanceEvidence:
        distance, result = self._raw_distance(first, second)
        collision_request = self.fcl.CollisionRequest()
        collision_request.enable_contact = False
        collision_result = self.fcl.CollisionResult()
        collision = bool(
            self.fcl.collide(
                first, second, collision_request, collision_result
            )
        )
        # Triangle-soup penetration depth is not a solid signed-distance
        # result.  Preserve exact positive clearance and exact collision state,
        # but never invent a negative depth.
        reported = 0.0 if collision else max(0.0, distance)
        return _DistanceEvidence(
            distance_m=float(reported),
            pair=pair,
            sample_index=int(sample_index),
            nearest_point_a=self._nearest(result, 1),
            nearest_point_b=self._nearest(result, 2),
            intersecting=collision,
            penetration_depth_trustworthy=not collision,
        )

    def _mesh_point_union_distance(
        self,
        mesh_object: Any,
        union: _PointCubeUnion,
        *,
        pair: str,
        sample_index: int,
    ) -> _DistanceEvidence:
        callback = self.fcl.DistanceCallBackDefault()
        callback.data.request.enable_nearest_points = True
        union.manager.distance(mesh_object, callback)
        result = callback.data.result
        distance = float(result.min_distance)
        nearest_a = self._nearest(result, 1)
        nearest_b = self._nearest(result, 2)

        # Broad-phase distance callbacks may terminate at the first overlap.
        # If an overlap exists, evaluate every cube whose AABB can intersect
        # the mesh AABB.  This keeps the returned negative number a true
        # minimum over the bound cube union without scanning distant points.
        if distance <= 0.0:
            mesh_object.computeAABB()
            bounds = mesh_object.getAABB()
            lower = np.asarray(bounds.min_, dtype=np.float64) - union.half_extent_m
            upper = np.asarray(bounds.max_, dtype=np.float64) + union.half_extent_m
            candidates = np.flatnonzero(
                np.all(union.points >= lower, axis=1)
                & np.all(union.points <= upper, axis=1)
            )
            if len(candidates) == 0:
                raise RuntimeError(
                    "broad phase reported overlap without an AABB candidate"
                )
            best_result = result
            best = float("inf")
            for index in candidates:
                candidate_distance, candidate_result = self._raw_distance(
                    mesh_object, union.objects[int(index)]
                )
                if candidate_distance < best:
                    best = candidate_distance
                    best_result = candidate_result
            distance = float(best)
            nearest_a = self._nearest(best_result, 1)
            nearest_b = self._nearest(best_result, 2)

        return _DistanceEvidence(
            distance_m=distance,
            pair=pair,
            sample_index=int(sample_index),
            nearest_point_a=nearest_a,
            nearest_point_b=nearest_b,
            intersecting=distance <= 0.0,
            penetration_depth_trustworthy=True,
        )

    @staticmethod
    def _minimum(items: Iterable[_DistanceEvidence]) -> _DistanceEvidence:
        values = list(items)
        if not values:
            raise RuntimeError("collision check has no geometry pairs")
        return min(values, key=lambda item: item.distance_m)

    def _robot_state(
        self, q: np.ndarray
    ) -> Tuple[Mapping[str, Any], np.ndarray]:
        value = np.asarray(q, dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("FR3 q sample must be a finite 7-vector")
        data = self.model.createData()
        geometry_data = self.pin.GeometryData(self.collision_model)
        self.pin.forwardKinematics(self.model, data, value)
        self.pin.updateFramePlacements(self.model, data)
        self.pin.updateGeometryPlacements(
            self.model, data, self.collision_model, geometry_data
        )
        objects = {
            name: self._object(geometry, self._matrix_from_se3(placement))
            for name, geometry, placement in zip(
                self._robot_link_names,
                self._robot_geometries,
                geometry_data.oMg,
            )
        }
        T_base_EE = self._matrix_from_se3(data.oMf[self._link8_frame_id])
        return objects, T_base_EE

    def _robot_visual_state(self, q: np.ndarray) -> Mapping[str, Any]:
        """Place the official FR3 visual shell meshes for depth-return filtering."""

        value = np.asarray(q, dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("FR3 q sample must be a finite 7-vector")
        data = self.model.createData()
        geometry_data = self.pin.GeometryData(self.visual_model)
        self.pin.forwardKinematics(self.model, data, value)
        self.pin.updateFramePlacements(self.model, data)
        self.pin.updateGeometryPlacements(
            self.model, data, self.visual_model, geometry_data
        )
        return {
            name: self._object(geometry, self._matrix_from_se3(placement))
            for name, geometry, placement in zip(
                self._robot_visual_link_names,
                self._robot_visual_geometries,
                geometry_data.oMg,
            )
        }

    def _hand_objects(
        self,
        geometries: Mapping[str, Any],
        transforms: Mapping[str, np.ndarray],
        T_base_hand: np.ndarray,
    ) -> Mapping[str, Any]:
        if set(geometries) != set(transforms):
            raise ValueError("RH56 mesh/FK link sets differ")
        if len(geometries) != 13:
            raise ValueError("RH56 collision backend requires all 13 link meshes")
        return {
            name: self._object(geometries[name], T_base_hand @ transforms[name])
            for name in sorted(geometries)
        }

    def _capture_installed_objects(
        self,
        capture_q_rad: np.ndarray,
        *,
        adapter_stl_path: Path,
        T_EE_hand: np.ndarray,
        hand_link_mesh_paths: Mapping[str, Path],
        T_hand_open_link_visual: Mapping[str, np.ndarray],
    ) -> Tuple[Tuple[str, Any], ...]:
        """Build every installed mesh at one bound, hardware-free FR3 state."""

        capture_q = np.asarray(capture_q_rad, dtype=np.float64)
        _, T_base_EE = self._robot_state(capture_q)
        robot_visual = self._robot_visual_state(capture_q)
        adapter_geometry = self._load_mesh(
            Path(adapter_stl_path), scale=self.config.adapter_mesh_scale
        )
        installed_objects: List[Tuple[str, Any]] = [
            ("FR3_visual_{}".format(name), robot_visual[name])
            for name in self._robot_visual_link_names
        ]
        installed_objects.append(
            (
                "V7_adapter",
                self._object(adapter_geometry, T_base_EE @ T_EE_ADAPTER),
            )
        )
        hand_geometries = {
            name: self._load_mesh(path)
            for name, path in hand_link_mesh_paths.items()
        }
        if len(hand_geometries) != 13:
            raise ValueError("installed return filter requires all 13 RH56 meshes")
        hand_objects = self._hand_objects(
            hand_geometries,
            T_hand_open_link_visual,
            T_base_EE @ np.asarray(T_EE_hand, dtype=np.float64),
        )
        installed_objects.extend(
            ("RH56_{}".format(name), hand_objects[name])
            for name in sorted(hand_objects)
        )
        return tuple(installed_objects)

    def classify_installed_depth_returns(
        self,
        scene_points_base: np.ndarray,
        capture_q_rad: np.ndarray,
        *,
        adapter_stl_path: Path,
        T_EE_hand: np.ndarray,
        hand_link_mesh_paths: Mapping[str, Path],
        T_hand_open_link_visual: Mapping[str, np.ndarray],
        point_half_extent_m: Optional[float] = None,
        inflation_margin_m: float = 0.002,
    ) -> InstalledReturnClassification:
        """Classify only points intersecting the capture-state installed meshes.

        The tested volume for one depth sample is an axis-aligned cube with
        half extent ``point_half_extent_m + inflation_margin_m``.  FR3 link
        meshes, the scaled V7 mesh, and every open RH56 link are evaluated at
        the exact bound ``capture_q_rad``.  No camera, robot or serial device is
        opened by this method.
        """

        points = _finite_points(scene_points_base, "scene_points_base")
        half_extent = (
            self.config.scene_point_half_extent_m
            if point_half_extent_m is None
            else float(point_half_extent_m)
        )
        margin = float(inflation_margin_m)
        if not np.isfinite(half_extent) or half_extent <= 0.0:
            raise ValueError("point_half_extent_m must be finite and positive")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("inflation_margin_m must be finite and non-negative")
        effective_half_extent = half_extent + margin

        installed_objects = self._capture_installed_objects(
            capture_q_rad,
            adapter_stl_path=adapter_stl_path,
            T_EE_hand=T_EE_hand,
            hand_link_mesh_paths=hand_link_mesh_paths,
            T_hand_open_link_visual=T_hand_open_link_visual,
        )

        cube = self.fcl.Box(
            2.0 * effective_half_extent,
            2.0 * effective_half_extent,
            2.0 * effective_half_extent,
        )
        removed = np.zeros(len(points), dtype=np.bool_)
        reasons = np.full(len(points), "", dtype=object)
        candidate_test_count = 0
        for label, mesh_object in installed_objects:
            mesh_object.computeAABB()
            bounds = mesh_object.getAABB()
            lower = np.asarray(bounds.min_, dtype=np.float64) - effective_half_extent
            upper = np.asarray(bounds.max_, dtype=np.float64) + effective_half_extent
            candidates = np.flatnonzero(
                (~removed)
                & np.all(points >= lower, axis=1)
                & np.all(points <= upper, axis=1)
            )
            candidate_test_count += int(len(candidates))
            for index in candidates:
                point_object = self.fcl.CollisionObject(
                    cube,
                    self.fcl.Transform3f(
                        np.eye(3), points[int(index)]
                    ),
                )
                distance, _ = self._raw_distance(mesh_object, point_object)
                if distance <= 0.0:
                    removed[int(index)] = True
                    reasons[int(index)] = label
        indices = np.flatnonzero(removed).astype(np.int64)
        return InstalledReturnClassification(
            removed_indices=indices,
            reason_labels=tuple(str(reasons[index]) for index in indices),
            candidate_test_count=candidate_test_count,
            point_half_extent_m=half_extent,
            inflation_margin_m=margin,
            fr3_geometry_model="official_fr3_visual_triangle_meshes",
            fr3_mesh_provenance=self._robot_visual_mesh_provenance,
        )

    def installed_depth_return_neighborhood(
        self,
        scene_points_base: np.ndarray,
        capture_q_rad: np.ndarray,
        *,
        adapter_stl_path: Path,
        T_EE_hand: np.ndarray,
        hand_link_mesh_paths: Mapping[str, Path],
        T_hand_open_link_visual: Mapping[str, np.ndarray],
        maximum_surface_distance_m: float,
    ) -> InstalledReturnNeighborhood:
        """Measure points near capture-state installed meshes with HPP-FCL.

        The broad phase only selects points whose centres lie in an expanded
        mesh AABB.  Every selected point is then measured against the triangle
        mesh using a 1 nm sphere; the sphere radius is added back to report a
        point-to-mesh distance.  Only distances no greater than the requested
        cap are returned.
        """

        points = _finite_points(scene_points_base, "scene_points_base")
        maximum = float(maximum_surface_distance_m)
        if not np.isfinite(maximum) or not 0.0 < maximum <= 0.02:
            raise ValueError("maximum_surface_distance_m must be in (0,0.02]")
        installed_objects = self._capture_installed_objects(
            capture_q_rad,
            adapter_stl_path=adapter_stl_path,
            T_EE_hand=T_EE_hand,
            hand_link_mesh_paths=hand_link_mesh_paths,
            T_hand_open_link_visual=T_hand_open_link_visual,
        )
        probe_radius = 1e-9
        probe_geometry = self.fcl.Sphere(probe_radius)
        best = np.full(len(points), np.inf, dtype=np.float64)
        labels = np.full(len(points), "", dtype=object)
        candidate_test_count = 0
        for label, mesh_object in installed_objects:
            mesh_object.computeAABB()
            bounds = mesh_object.getAABB()
            lower = np.asarray(bounds.min_, dtype=np.float64) - maximum
            upper = np.asarray(bounds.max_, dtype=np.float64) + maximum
            candidates = np.flatnonzero(
                np.all(points >= lower, axis=1) & np.all(points <= upper, axis=1)
            )
            candidate_test_count += int(len(candidates))
            for index in candidates:
                point_object = self.fcl.CollisionObject(
                    probe_geometry,
                    self.fcl.Transform3f(np.eye(3), points[int(index)]),
                )
                distance, _ = self._raw_distance(mesh_object, point_object)
                corrected = float(distance) + probe_radius
                if corrected < best[int(index)]:
                    best[int(index)] = corrected
                    labels[int(index)] = label
        selected = np.flatnonzero(best <= maximum + 1e-12).astype(np.int64)
        return InstalledReturnNeighborhood(
            candidate_indices=selected,
            reason_labels=tuple(str(labels[index]) for index in selected),
            surface_distances_m=best[selected],
            candidate_test_count=candidate_test_count,
            maximum_surface_distance_m=maximum,
            distance_method=(
                "HPP-FCL triangle-mesh to 1nm sphere, radius corrected"
            ),
        )

    @staticmethod
    def _observation_details(
        minimum: _DistanceEvidence,
        *,
        geometry_model: str,
        pair_count: int,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        details: Dict[str, Any] = {
            "engine": "Pinocchio + HPP-FCL",
            "geometry_model": geometry_model,
            "evaluated_pair_count": int(pair_count),
            "minimum_pair": minimum.pair,
            "minimum_sample_index": int(minimum.sample_index),
            "nearest_point_a_base_m": (
                None
                if minimum.nearest_point_a is None
                else list(minimum.nearest_point_a)
            ),
            "nearest_point_b_base_m": (
                None
                if minimum.nearest_point_b is None
                else list(minimum.nearest_point_b)
            ),
            "intersecting": bool(minimum.intersecting),
            "penetration_depth_trustworthy": bool(
                minimum.penetration_depth_trustworthy
            ),
            "path_semantics": "piecewise linear in joint space over the bound q samples",
            "authoritative_for": "the bound joint-waypoint path only",
            "not_authoritative_for": "Cartesian pose interpolation or any other realized path",
        }
        if extra:
            details.update(extra)
        return details

    def _progress(self, message: str) -> None:
        callback = self._progress_callback
        if callback is not None:
            callback(str(message))

    @staticmethod
    def _progress_milestones(total: int) -> frozenset[int]:
        """Return bounded 10% progress milestones as zero-based indices."""

        count = int(total)
        if count <= 0:
            return frozenset()
        return frozenset(
            min(count - 1, max(0, int(math.ceil(count * fraction)) - 1))
            for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )

    def evaluate(
        self, query: InstalledToolCollisionQuery
    ) -> Mapping[str, CollisionObservation]:
        return self.evaluate_checks(
            query,
            tuple(
                item.check_id
                for item in check_specs_for_mode(query.request.mode)
            ),
        )

    def evaluate_checks(
        self,
        query: InstalledToolCollisionQuery,
        check_ids: Sequence[str],
    ) -> Mapping[str, CollisionObservation]:
        """Evaluate an exact, fail-closed subset of the normal audit checks.

        This is used by the two-phase cache workflow.  It does not relax a
        policy or change an observation: the caller must still provide every
        check required by :func:`run_installed_tool_audit` before an audit can
        pass.  Rejecting duplicates and unknown IDs keeps a malformed split
        from silently losing coverage.
        """

        all_specs = check_specs_for_mode(query.request.mode)
        requested = tuple(str(value) for value in check_ids)
        if not requested:
            raise ValueError("check_ids must not be empty")
        if len(set(requested)) != len(requested):
            raise ValueError("check_ids must be unique")
        known = {item.check_id for item in all_specs}
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValueError("unknown check_ids: {}".format(unknown))
        requested_set = frozenset(requested)
        check_specs = tuple(
            item for item in all_specs if item.check_id in requested_set
        )

        q_path = np.asarray(query.q_path_rad, dtype=np.float64)
        if q_path.ndim != 2 or q_path.shape[1:] != (7,) or len(q_path) == 0:
            raise ValueError("query.q_path_rad must have shape (N,7)")
        if not np.all(np.isfinite(q_path)):
            raise ValueError("query.q_path_rad contains NaN or infinity")

        hand_waypoint_count = len(query.T_hand_waypoint_link_visual)
        self._progress(
            "start checks={} arm_samples={} hand_waypoints={}".format(
                len(check_specs), len(q_path), hand_waypoint_count
            )
        )

        request = query.request
        requested_scene_voxel = float(request.scene_voxel_resolution_m)
        if not np.isclose(
            2.0 * self.config.scene_point_half_extent_m,
            requested_scene_voxel,
            atol=1e-15,
            rtol=0.0,
        ):
            raise ValueError(
                "scene cube width does not match request.scene_voxel_resolution_m"
            )
        if not np.isclose(
            float(request.max_q_tracking_error_rad),
            self.config.max_q_tracking_error_rad,
            atol=1e-15,
            rtol=0.0,
        ):
            raise ValueError(
                "backend max_q_tracking_error_rad differs from the audit request"
            )
        scene_check_ids = {
            "fr3_scene_path",
            "adapter_scene_path",
            "rh56_open_scene_path",
            "rh56_closed_scene_final",
            "rh56_execution_scene_final",
        }
        object_check_ids = {
            "adapter_object_path",
            "rh56_open_object_path",
            "rh56_closed_object_noncontact_final",
            "rh56_closed_object_contact_final",
            "rh56_closed_object_all_links_final",
            "rh56_execution_object_final",
        }
        scene = (
            self._point_union(
                request.scene_points_base,
                self.config.scene_point_half_extent_m,
            )
            if requested_set & scene_check_ids
            else None
        )
        object_union = (
            self._point_union(
                request.object_points_base,
                self.config.object_point_half_extent_m,
            )
            if requested_set & object_check_ids
            else None
        )
        adapter_geometry = self._load_mesh(
            request.adapter_stl_path, scale=self.config.adapter_mesh_scale
        )
        hand_geometries = {
            name: self._load_mesh(path)
            for name, path in query.hand_link_mesh_paths.items()
        }
        if len(hand_geometries) != 13:
            raise ValueError("query must bind exactly 13 RH56 meshes")
        hand_joint_influences = self._hand_joint_influence_radii(
            request.hand_model, hand_geometries
        )
        hand_mesh_resolution = str(
            getattr(getattr(request, "hand_model", None), "mesh_resolution", "unknown")
        )

        adapter_influence = self._joint_influence_radii(
            7,
            self._T_joint7_EE @ T_EE_ADAPTER,
            adapter_geometry,
        )
        T_joint7_hand = self._T_joint7_EE @ np.asarray(
            request.T_EE_hand, dtype=np.float64
        )
        open_influences = {
            name: self._joint_influence_radii(
                7,
                T_joint7_hand
                @ np.asarray(query.T_hand_open_link_visual[name], dtype=np.float64),
                geometry,
            )
            for name, geometry in hand_geometries.items()
        }
        closed_influences = {
            name: self._joint_influence_radii(
                7,
                T_joint7_hand
                @ np.asarray(query.T_hand_closed_link_visual[name], dtype=np.float64),
                geometry,
            )
            for name, geometry in hand_geometries.items()
        }
        waypoint_influences = tuple(
            {
                name: self._joint_influence_radii(
                    7,
                    T_joint7_hand @ np.asarray(transforms[name], dtype=np.float64),
                    geometry,
                )
                for name, geometry in hand_geometries.items()
            }
            for transforms in query.T_hand_waypoint_link_visual
        )
        evidence: Dict[str, List[_DistanceEvidence]] = {
            item.check_id: [] for item in check_specs
        }
        observed: Dict[str, set[str]] = {
            item.check_id: set() for item in check_specs
        }

        allowed_contact = set(request.allowed_object_contact_links)
        all_hand_links = set(hand_geometries)
        if not allowed_contact or not allowed_contact.issubset(all_hand_links):
            raise ValueError("allowed object contact links do not match RH56 meshes")

        final_index = len(q_path) - 1
        arm_progress = self._progress_milestones(len(q_path))
        for sample_index, q in enumerate(q_path):
            robot, T_base_EE = self._robot_state(q)
            adapter = self._object(adapter_geometry, T_base_EE @ T_EE_ADAPTER)
            T_base_hand = T_base_EE @ np.asarray(request.T_EE_hand, dtype=np.float64)
            hand_open = self._hand_objects(
                hand_geometries,
                query.T_hand_open_link_visual,
                T_base_hand,
            )

            if "fr3_self_path" in requested_set:
                for first, second in self._self_pairs:
                    item = self._mesh_pair_distance(
                        robot[self._robot_link_names[first]],
                        robot[self._robot_link_names[second]],
                        pair="{} / {}".format(
                            self._robot_link_names[first],
                            self._robot_link_names[second],
                        ),
                        sample_index=sample_index,
                    )
                    evidence["fr3_self_path"].append(item)
                    if item.intersecting:
                        observed["fr3_self_path"].add(item.pair)

            if "fr3_scene_path" in requested_set:
                assert scene is not None
                for name, robot_object in robot.items():
                    item = self._mesh_point_union_distance(
                        robot_object,
                        scene,
                        pair="{} / scene".format(name),
                        sample_index=sample_index,
                    )
                    evidence["fr3_scene_path"].append(item)
                    if item.intersecting:
                        observed["fr3_scene_path"].add(item.pair)

            if "adapter_fr3_path" in requested_set:
                for name, robot_object in robot.items():
                    if name in ADAPTER_FR3_MOUNT_EXCLUSIONS:
                        continue
                    item = self._mesh_pair_distance(
                        adapter,
                        robot_object,
                        pair="adapter / {}".format(name),
                        sample_index=sample_index,
                    )
                    evidence["adapter_fr3_path"].append(item)
                    if item.intersecting:
                        observed["adapter_fr3_path"].add(item.pair)

            for check_id, union, label in (
                ("adapter_scene_path", scene, "scene"),
                ("adapter_object_path", object_union, "object"),
            ):
                if check_id not in requested_set:
                    continue
                assert union is not None
                item = self._mesh_point_union_distance(
                    adapter,
                    union,
                    pair="adapter / {}".format(label),
                    sample_index=sample_index,
                )
                evidence[check_id].append(item)
                if item.intersecting:
                    observed[check_id].add(item.pair)

            for hand_name, hand_object in hand_open.items():
                if "rh56_open_fr3_path" in requested_set:
                    for robot_name, robot_object in robot.items():
                        # Link111 and link7 meet through the fixed mount stack.
                        if hand_name == "Link111" and robot_name == "link7":
                            continue
                        item = self._mesh_pair_distance(
                            hand_object,
                            robot_object,
                            pair="{} / {}".format(hand_name, robot_name),
                            sample_index=sample_index,
                        )
                        evidence["rh56_open_fr3_path"].append(item)
                        if item.intersecting:
                            observed["rh56_open_fr3_path"].add(item.pair)

                for check_id, union, label in (
                    ("rh56_open_scene_path", scene, "scene"),
                    ("rh56_open_object_path", object_union, "object"),
                ):
                    if check_id not in requested_set:
                        continue
                    assert union is not None
                    item = self._mesh_point_union_distance(
                        hand_object,
                        union,
                        pair="{} / {}".format(hand_name, label),
                        sample_index=sample_index,
                    )
                    evidence[check_id].append(item)
                    if item.intersecting:
                        observed[check_id].add(item.pair)

            if sample_index in arm_progress:
                self._progress(
                    "arm_path {}/{} ({:.0f}%)".format(
                        sample_index + 1,
                        len(q_path),
                        100.0 * float(sample_index + 1) / float(len(q_path)),
                    )
                )

            if sample_index != final_index:
                continue

            hand_closed = self._hand_objects(
                hand_geometries,
                query.T_hand_closed_link_visual,
                T_base_hand,
            )
            for hand_name, hand_object in hand_closed.items():
                if "rh56_closed_fr3_final" in requested_set:
                    for robot_name, robot_object in robot.items():
                        if hand_name == "Link111" and robot_name == "link7":
                            continue
                        item = self._mesh_pair_distance(
                            hand_object,
                            robot_object,
                            pair="{} / {}".format(hand_name, robot_name),
                            sample_index=sample_index,
                        )
                        evidence["rh56_closed_fr3_final"].append(item)
                        if item.intersecting:
                            observed["rh56_closed_fr3_final"].add(item.pair)

                if (
                    "rh56_closed_adapter_final" in requested_set
                    and hand_name not in RH56_ADAPTER_MOUNT_EXCLUSIONS
                ):
                    item = self._mesh_pair_distance(
                        hand_object,
                        adapter,
                        pair="{} / adapter".format(hand_name),
                        sample_index=sample_index,
                    )
                    evidence["rh56_closed_adapter_final"].append(item)
                    if item.intersecting:
                        observed["rh56_closed_adapter_final"].add(item.pair)

                if "rh56_closed_scene_final" in requested_set:
                    assert scene is not None
                    item = self._mesh_point_union_distance(
                        hand_object,
                        scene,
                        pair="{} / scene".format(hand_name),
                        sample_index=sample_index,
                    )
                    evidence["rh56_closed_scene_final"].append(item)
                    if item.intersecting:
                        observed["rh56_closed_scene_final"].add(item.pair)

                if (
                    "rh56_closed_object_noncontact_final" in requested_set
                    and hand_name not in allowed_contact
                ):
                    assert object_union is not None
                    item = self._mesh_point_union_distance(
                        hand_object,
                        object_union,
                        pair="{} / object".format(hand_name),
                        sample_index=sample_index,
                    )
                    evidence["rh56_closed_object_noncontact_final"].append(item)
                    if item.intersecting:
                        observed["rh56_closed_object_noncontact_final"].add(item.pair)
                elif (
                    "rh56_closed_object_contact_final" in requested_set
                    and request.mode == "loaded_grasp"
                    and hand_name in allowed_contact
                ):
                    assert object_union is not None
                    item = self._mesh_point_union_distance(
                        hand_object,
                        object_union,
                        pair="{} / object".format(hand_name),
                        sample_index=sample_index,
                    )
                    evidence["rh56_closed_object_contact_final"].append(item)
                    if item.intersecting:
                        observed["rh56_closed_object_contact_final"].add(item.pair)
                if (
                    "rh56_closed_object_all_links_final" in requested_set
                    and request.mode == "air_grasp"
                ):
                    assert object_union is not None
                    item = self._mesh_point_union_distance(
                        hand_object,
                        object_union,
                        pair="{} / object".format(hand_name),
                        sample_index=sample_index,
                    )
                    evidence["rh56_closed_object_all_links_final"].append(item)
                    if item.intersecting:
                        observed["rh56_closed_object_all_links_final"].add(item.pair)

        execution_check_ids = {
            "rh56_execution_fr3_final",
            "rh56_execution_adapter_final",
            "rh56_execution_scene_final",
            "rh56_execution_object_final",
            "rh56_execution_self_final",
        }
        if request.mode == "air_grasp" and requested_set & execution_check_ids:
            if len(query.T_hand_waypoint_link_visual) == 0:
                raise ValueError("air audit has no RH56 execution waypoint transforms")
            final_robot, final_T_base_EE = self._robot_state(q_path[-1])
            final_adapter = self._object(
                adapter_geometry, final_T_base_EE @ T_EE_ADAPTER
            )
            final_T_base_hand = final_T_base_EE @ np.asarray(
                request.T_EE_hand, dtype=np.float64
            )
            adjacent = {
                frozenset((joint.parent, joint.child))
                for joint in request.hand_model.joints
            }
            hand_names = tuple(hand_geometries)
            hand_progress = self._progress_milestones(
                len(query.T_hand_waypoint_link_visual)
            )
            for hand_index, transforms in enumerate(
                query.T_hand_waypoint_link_visual
            ):
                hand_objects = self._hand_objects(
                    hand_geometries, transforms, final_T_base_hand
                )
                for hand_name, hand_object in hand_objects.items():
                    if "rh56_execution_fr3_final" in requested_set:
                        for robot_name, robot_object in final_robot.items():
                            if hand_name == "Link111" and robot_name == "link7":
                                continue
                            item = self._mesh_pair_distance(
                                hand_object,
                                robot_object,
                                pair="{} / {}".format(hand_name, robot_name),
                                sample_index=hand_index,
                            )
                            evidence["rh56_execution_fr3_final"].append(item)
                            if item.intersecting:
                                observed["rh56_execution_fr3_final"].add(item.pair)
                    if (
                        "rh56_execution_adapter_final" in requested_set
                        and hand_name not in RH56_ADAPTER_MOUNT_EXCLUSIONS
                    ):
                        item = self._mesh_pair_distance(
                            hand_object,
                            final_adapter,
                            pair="{} / adapter".format(hand_name),
                            sample_index=hand_index,
                        )
                        evidence["rh56_execution_adapter_final"].append(item)
                        if item.intersecting:
                            observed["rh56_execution_adapter_final"].add(item.pair)
                    for check_id, union, label in (
                        ("rh56_execution_scene_final", scene, "scene"),
                        ("rh56_execution_object_final", object_union, "object"),
                    ):
                        if check_id not in requested_set:
                            continue
                        assert union is not None
                        item = self._mesh_point_union_distance(
                            hand_object,
                            union,
                            pair="{} / {}".format(hand_name, label),
                            sample_index=hand_index,
                        )
                        evidence[check_id].append(item)
                        if item.intersecting:
                            observed[check_id].add(item.pair)
                if "rh56_execution_self_final" in requested_set:
                    for first_index, first_name in enumerate(hand_names):
                        for second_name in hand_names[first_index + 1 :]:
                            if frozenset((first_name, second_name)) in adjacent:
                                continue
                            item = self._mesh_pair_distance(
                                hand_objects[first_name],
                                hand_objects[second_name],
                                pair="{} / {}".format(first_name, second_name),
                                sample_index=hand_index,
                            )
                            evidence["rh56_execution_self_final"].append(item)
                            if item.intersecting:
                                observed["rh56_execution_self_final"].add(item.pair)
                if hand_index in hand_progress:
                    self._progress(
                        "hand_path {}/{} ({:.0f}%)".format(
                            hand_index + 1,
                            len(query.T_hand_waypoint_link_visual),
                            100.0
                            * float(hand_index + 1)
                            / float(len(query.T_hand_waypoint_link_visual)),
                        )
                    )

        path_indices = tuple(range(len(q_path)))
        results: Dict[str, CollisionObservation] = {}
        adaptive_self_cache: Dict[
            Tuple[str, str, bytes, bytes, float], _AdaptiveHandBoxResult
        ] = {}
        point_checks = {
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
        mesh_intersection_unbounded = {
            "fr3_self_path",
            "adapter_fr3_path",
            "rh56_open_fr3_path",
            "rh56_closed_fr3_final",
            "rh56_closed_adapter_final",
            "rh56_execution_fr3_final",
            "rh56_execution_adapter_final",
            "rh56_execution_self_final",
        }
        for spec_index, spec in enumerate(check_specs):
            self._progress(
                "finalize_check {}/{} {}".format(
                    spec_index + 1, len(check_specs), spec.check_id
                )
            )
            items = evidence[spec.check_id]
            nominal_minimum = self._minimum(items)
            decisive_hand_motion_bound = 0.0
            adaptive_self_attempted = 0
            adaptive_self_certified = 0
            adaptive_self_unresolved = 0
            adaptive_self_nodes = 0
            adaptive_self_records: List[Mapping[str, Any]] = []
            if spec.coverage == "hand_execution_path":
                best = None
                by_pair: Dict[str, Dict[int, _DistanceEvidence]] = {}
                for item in items:
                    by_pair.setdefault(item.pair, {})[item.sample_index] = item
                interval_paths = query.hand_dense_interval_q12_rad
                feedback_tubes = query.hand_interval_feedback_tube_q12_rad
                feedback_lowers = tuple(
                    getattr(
                        query, "hand_interval_feedback_q12_lower_rad", ()
                    )
                )
                feedback_uppers = tuple(
                    getattr(
                        query, "hand_interval_feedback_q12_upper_rad", ()
                    )
                )
                adaptive_boxes_available = bool(feedback_lowers or feedback_uppers)
                if (
                    len(interval_paths) != len(waypoint_influences) - 1
                    or len(feedback_tubes) != len(interval_paths)
                ):
                    raise RuntimeError("RH56 dense interval/tube coverage is incomplete")
                if adaptive_boxes_available and (
                    len(feedback_lowers) != len(interval_paths)
                    or len(feedback_uppers) != len(interval_paths)
                ):
                    raise RuntimeError(
                        "RH56 feedback q12 box coverage is incomplete"
                    )
                expected_hand_indices = set(range(len(waypoint_influences)))
                for pair, samples in by_pair.items():
                    if set(samples) != expected_hand_indices:
                        raise RuntimeError("RH56 pair lacks full waypoint coverage")
                    names = tuple(part.strip() for part in pair.split(" / "))
                    if len(names) != 2:
                        raise RuntimeError("unexpected RH56 execution pair label")
                    first, second = names
                    hand_radii = hand_joint_influences[first].copy()
                    if spec.check_id == "rh56_execution_self_final":
                        hand_radii += hand_joint_influences[second]
                    for interval_index, dense_q12 in enumerate(interval_paths):
                        nominal_variation = np.sum(
                            np.abs(np.diff(dense_q12, axis=0)), axis=0
                        )
                        hand_delta = nominal_variation + np.asarray(
                            feedback_tubes[interval_index], dtype=np.float64
                        )
                        hand_bound = float(np.dot(hand_radii, hand_delta))
                        start_item = samples[interval_index]
                        end_item = samples[interval_index + 1]
                        minimum_item = (
                            start_item
                            if start_item.distance_m <= end_item.distance_m
                            else end_item
                        )
                        if spec.check_id == "rh56_execution_fr3_final":
                            radii = self._maximum_influence(
                                self._relative_influence_radii(
                                    waypoint_influences[index][first], 7,
                                    self._robot_influences[second],
                                    self._robot_parent_joints[second],
                                )
                                for index in (interval_index, interval_index + 1)
                            )
                        elif spec.check_id in (
                            "rh56_execution_scene_final",
                            "rh56_execution_object_final",
                        ):
                            radii = self._maximum_influence(
                                waypoint_influences[index][first]
                                for index in (interval_index, interval_index + 1)
                            )
                        elif spec.check_id in (
                            "rh56_execution_adapter_final",
                            "rh56_execution_self_final",
                        ):
                            radii = np.zeros(7, dtype=np.float64)
                        else:
                            raise RuntimeError("unknown RH56 execution check")
                        arm_bound = self._continuous_motion_bound(
                            radii, q_path, coverage="final"
                        )
                        bound = float(hand_bound + arm_bound)
                        candidate = float(minimum_item.distance_m - bound)
                        if (
                            spec.check_id == "rh56_execution_self_final"
                            and adaptive_boxes_available
                            and candidate
                            <= float(request.hand_self_clearance_margin_m)
                            # A nominal endpoint at/below the margin is already
                            # a direct exact-mesh blocker.  Refinement must not
                            # hide or reinterpret that witness.
                            and minimum_item.distance_m
                            > float(request.hand_self_clearance_margin_m)
                        ):
                            lower_box = np.ascontiguousarray(
                                np.asarray(
                                    feedback_lowers[interval_index],
                                    dtype="<f8",
                                )
                            )
                            upper_box = np.ascontiguousarray(
                                np.asarray(
                                    feedback_uppers[interval_index],
                                    dtype="<f8",
                                )
                            )
                            cache_key = (
                                first,
                                second,
                                lower_box.tobytes(order="C"),
                                upper_box.tobytes(order="C"),
                                float(request.hand_self_clearance_margin_m),
                            )
                            refined = adaptive_self_cache.get(cache_key)
                            cache_hit = refined is not None
                            if refined is None:
                                refined = self._adaptive_hand_self_feedback_box(
                                    hand_model=request.hand_model,
                                    first_name=first,
                                    second_name=second,
                                    first_geometry=hand_geometries[first],
                                    second_geometry=hand_geometries[second],
                                    lower_q12_rad=lower_box,
                                    upper_q12_rad=upper_box,
                                    pair_influence_radii_m=hand_radii,
                                    required_clearance_m=float(
                                        request.hand_self_clearance_margin_m
                                    ),
                                )
                                adaptive_self_cache[cache_key] = refined
                            adaptive_self_attempted += 1
                            if not cache_hit:
                                adaptive_self_nodes += refined.evaluated_nodes
                            if refined.certified:
                                adaptive_self_certified += 1
                                coarse_candidate = candidate
                                candidate = min(
                                    float(refined.lower_bound_m),
                                    float(minimum_item.distance_m),
                                )
                                hand_bound = max(
                                    0.0,
                                    float(minimum_item.distance_m) - candidate,
                                )
                                bound = float(hand_bound + arm_bound)
                            else:
                                adaptive_self_unresolved += 1
                                coarse_candidate = candidate
                            adaptive_self_records.append(
                                {
                                    "pair": pair,
                                    "interval_sample_indices": [
                                        interval_index,
                                        interval_index + 1,
                                    ],
                                    "coarse_lower_bound_m": float(
                                        coarse_candidate
                                    ),
                                    "certified": bool(refined.certified),
                                    "refined_lower_bound_m": (
                                        float(refined.lower_bound_m)
                                        if refined.certified
                                        else None
                                    ),
                                    "evaluated_nodes": int(
                                        refined.evaluated_nodes
                                    ),
                                    "maximum_depth": int(refined.maximum_depth),
                                    "minimum_sampled_distance_m": float(
                                        refined.minimum_sampled_distance_m
                                    ),
                                    "reason": refined.reason,
                                    "cache_hit": bool(cache_hit),
                                }
                            )
                        value = (
                            candidate, bound, radii, minimum_item,
                            (interval_index, interval_index + 1), hand_bound,
                        )
                        if best is None or candidate < best[0]:
                            best = value
                if best is None:
                    raise RuntimeError("RH56 execution check has no evidence")
                (
                    conservative_distance,
                    motion_bound,
                    decisive_influence,
                    minimum,
                    decisive_interval,
                    decisive_hand_motion_bound,
                ) = best
            else:
                (
                    conservative_distance,
                    motion_bound,
                    decisive_influence,
                    minimum,
                    decisive_interval,
                ) = self._conservative_distance_evidence(
                    items,
                    q_path,
                    check_id=spec.check_id,
                    coverage=spec.coverage,
                    robot_influences=self._robot_influences,
                    robot_parent_joints=self._robot_parent_joints,
                    adapter_influence=adapter_influence,
                    open_influences=open_influences,
                    closed_influences=closed_influences,
                )
            unresolved_adaptive_records = [
                item
                for item in adaptive_self_records
                if not bool(item["certified"])
            ]
            certified_adaptive_records = sorted(
                (
                    item
                    for item in adaptive_self_records
                    if bool(item["certified"])
                ),
                key=lambda item: item["coarse_lower_bound_m"],
            )
            representative_adaptive_records = (
                unresolved_adaptive_records + certified_adaptive_records
            )[:16]
            authoritative = True
            authority_reasons: List[str] = []
            if spec.check_id in point_checks:
                authoritative = self.config.point_cloud_observations_authoritative
                authority_reasons.append(
                    "single captured point cloud does not prove occluded space empty"
                )
            if spec.check_id.startswith("rh56_") and hand_mesh_resolution != "full":
                authoritative = False
                authority_reasons.append(
                    "RH56 simplified/unknown mesh is not a conservative full-link envelope"
                )
            if (
                spec.check_id == "rh56_closed_object_contact_final"
                and minimum.intersecting
                and not minimum.penetration_depth_trustworthy
            ):
                authoritative = False
                authority_reasons.append(
                    "object penetration depth was not trustworthy"
                )
            # Intersecting mesh soups have an exact collision result but no
            # certified penetration magnitude.  For clear checks this still
            # authoritatively fails: zero clearance and an observed pair are
            # sufficient, so no authority downgrade is needed.
            details_extra: Dict[str, Any] = {
                "authority_limitations": authority_reasons,
                "point_cloud_observations_authoritative": bool(
                    self.config.point_cloud_observations_authoritative
                ),
                "scene_point_half_extent_m": float(
                    self.config.scene_point_half_extent_m
                ),
                "object_point_half_extent_m": float(
                    self.config.object_point_half_extent_m
                ),
                "adapter_mesh_scale": float(self.config.adapter_mesh_scale),
                "rh56_mesh_resolution": hand_mesh_resolution,
                "max_q_tracking_error_rad": float(
                    self.config.max_q_tracking_error_rad
                ),
                "joint_tracking_uncertainty_applied": bool(
                    self.config.joint_tracking_uncertainty_applied
                ),
                "continuous_segment_envelope_verified": bool(
                    self.config.continuous_segment_envelope_verified
                ),
                "conservative_motion_bound_m": float(motion_bound),
                "minimum_distance_is_after_motion_bound": True,
                "nominal_minimum_signed_distance_m": float(
                    nominal_minimum.distance_m
                ),
                "nominal_minimum_pair": nominal_minimum.pair,
                "nominal_minimum_sample_index": int(
                    nominal_minimum.sample_index
                ),
                "nominal_minimum_intersecting": bool(
                    nominal_minimum.intersecting
                ),
                "decisive_nominal_distance_m": float(minimum.distance_m),
                "conservative_decisive_pair": minimum.pair,
                "decisive_interval_sample_indices": list(decisive_interval),
                "motion_bound_method": (
                    "serial-chain influence radii; 2*R*sin(delta_q/2), "
                    "nearest interval endpoint plus tracking tube"
                ),
                "motion_influence_radii_m": decisive_influence.tolist(),
                "observed_scene_scope": str(request.observed_scene_scope),
                "scene_voxel_resolution_m": float(
                    request.scene_voxel_resolution_m
                ),
                "unknown_space_policy": str(request.unknown_space_policy),
                "unknown_space_policy_applied": False,
            }
            if spec.coverage == "hand_execution_path":
                details_extra.update(
                    {
                        "hand_waypoint_count": len(waypoint_influences),
                        "hand_waypoint_discrete_collision_checked": True,
                        "continuous_inter_waypoint_collision_claimed": True,
                        "hand_interval_envelope_method": (
                            "official XLS every 1 register; accumulated absolute q12 "
                            "variation times configuration-independent URDF serial-chain "
                            "link radii; all-six arrival feedback tube; adaptive exact-mesh "
                            "q12 feedback-box subdivision for otherwise unresolved self-pairs; "
                            "final arm tracking tube"
                        ),
                        "decisive_hand_motion_bound_m": float(
                            decisive_hand_motion_bound
                        ),
                        "hand_arrival_tolerance_units": int(
                            request.hand_arrival_tolerance_units
                        ),
                        "feedback_envelope_policy_sha256": (
                            rh56_feedback_envelope_policy(
                                request.hand_arrival_tolerance_units
                            )["sha256"]
                        ),
                        "arm_configuration": "fixed final audited q with tracking tube",
                        "path_semantics": (
                            "discrete bound RH56 actuator waypoints at fixed final arm q"
                        ),
                        "authoritative_for": (
                            "continuous commanded RH56 intervals plus configured feedback tube"
                        ),
                        "not_authoritative_for": (
                            "unlisted actuator paths or feedback outside the bound tolerance"
                        ),
                        "adaptive_self_feedback_box": {
                            "available": bool(adaptive_boxes_available),
                            "attempted_interval_pairs": int(
                                adaptive_self_attempted
                            ),
                            "certified_interval_pairs": int(
                                adaptive_self_certified
                            ),
                            "unresolved_interval_pairs": int(
                                adaptive_self_unresolved
                            ),
                            "evaluated_nodes": int(adaptive_self_nodes),
                            "method": (
                                "component-wise official-mapping feedback q12 box; "
                                "exact HPP-FCL centre distances; adaptive bisection with "
                                "serial-chain revolute chord lower bounds"
                            ),
                            "policy_margin_unchanged": True,
                            "representative_refinements": (
                                representative_adaptive_records
                            ),
                        },
                    }
                )
            if spec.check_id == "rh56_execution_self_final":
                adjacent_pairs = {
                    frozenset((joint.parent, joint.child))
                    for joint in request.hand_model.joints
                }
                tested_hand_self_pairs = [
                    [first_name, second_name]
                    for first_index, first_name in enumerate(hand_names)
                    for second_name in hand_names[first_index + 1 :]
                    if frozenset((first_name, second_name)) not in adjacent_pairs
                ]
                details_extra.update(
                    {
                        "hand_self_clearance_margin_m": float(
                            request.hand_self_clearance_margin_m
                        ),
                        "all_nonadjacent_hand_link_pairs_checked": True,
                        "hand_self_pair_policy": (
                            "all distinct RH56 links except direct URDF parent-child pairs"
                        ),
                        "tested_hand_self_pairs": tested_hand_self_pairs,
                    }
                )
            if spec.check_id == "fr3_self_path":
                details_extra.update(
                    {
                        "official_srdf_disabled_pairs": [
                            list(item) for item in OFFICIAL_FRANKA_DISABLED_SELF_PAIRS
                        ],
                        "tested_self_pairs": [
                            [self._robot_link_names[a], self._robot_link_names[b]]
                            for a, b in self._self_pairs
                        ],
                    }
                )
            if spec.check_id == "adapter_fr3_path":
                details_extra["fixed_mount_pair_exclusions"] = list(
                    ADAPTER_FR3_MOUNT_EXCLUSIONS
                )
            if spec.check_id == "rh56_closed_adapter_final":
                details_extra["fixed_mount_pair_exclusions"] = list(
                    RH56_ADAPTER_MOUNT_EXCLUSIONS
                )
            if spec.check_id in mesh_intersection_unbounded and minimum.intersecting:
                details_extra[
                    "mesh_intersection_distance_semantics"
                ] = "exact collision, reported clearance 0; no triangle-soup penetration claim"
            if spec.coverage == "full_path":
                indices = path_indices
            elif spec.coverage == "hand_execution_path":
                indices = tuple(range(len(waypoint_influences)))
            else:
                indices = (final_index,)
            results[spec.check_id] = CollisionObservation(
                check_id=spec.check_id,
                authoritative=bool(authoritative),
                tested_sample_indices=indices,
                minimum_signed_distance_m=conservative_distance,
                observed_pairs=tuple(sorted(observed[spec.check_id])),
                details=self._observation_details(
                    minimum,
                    geometry_model=(
                        "exact triangle meshes"
                        if spec.check_id not in point_checks
                        else "exact mesh versus conservative captured-point cube union"
                    ),
                    pair_count=len(items),
                    extra=details_extra,
                ),
            )
        self._progress("complete checks={}".format(len(results)))
        return results


__all__ = [
    "ADAPTER_FR3_MOUNT_EXCLUSIONS",
    "DEFAULT_FR3_SRDF_XACRO",
    "DEFAULT_FR3_URDF",
    "DEFAULT_FRANKA_DESCRIPTION_SHARE",
    "HppFclInstalledToolBackend",
    "HppFclInstalledToolConfig",
    "InstalledReturnClassification",
    "InstalledReturnNeighborhood",
    "OFFICIAL_FRANKA_DISABLED_SELF_PAIRS",
    "RH56_ADAPTER_MOUNT_EXCLUSIONS",
    "T_EE_ADAPTER",
]
