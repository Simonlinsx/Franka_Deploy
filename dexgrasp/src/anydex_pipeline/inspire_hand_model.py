"""Inspire Hand-R URDF forward kinematics and selected-grasp mesh overlay.

The official AnyDexGrasp release ships the Inspire URDF, per-link STL files,
and a ``width_12Dangle_6Dangle.json`` table, but not the generated aggregate
hand meshes.  This module reconstructs the aggregate pose without PyBullet:

* the table's ``12d`` values are revolute-joint radians in URDF order;
* the table's ``6d`` values are Inspire register commands and are never used
  as radians;
* the two translations used by the official mesh generator are reproduced
  before the hand pose is applied.

Open3D remains a lazy dependency.  Importing this module parses no mesh and
does not create a renderer, camera connection, serial connection, or robot
connection.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from .inspire_mapping import (
    INSPIRE_GRASP_TYPES,
    MAX_GRASP_WIDTH,
    MIN_GRASP_WIDTH,
    load_inspire_mapping,
)
from .snapshot import VisualizationSnapshot


# The official generator applies both translations to the FK-combined mesh:
#   wrist shell -> wrist centre: [0.04123, 0.00804, -0.01796]
#   reserve the mounting ring:   [0.00780, 0,       0]
#
# The AnyDex source origin is at the mounting end of that ring.  Consequently
# an RH56 adapter shoulder that supports the bottom of the metal sleeve is the
# source origin; the 7.8 mm ring/spigot length moves Link111 away from it and
# must not be added to the adapter shoulder transform a second time.
OFFICIAL_WRIST_CENTER_OFFSET_M = np.asarray(
    [0.04123, 0.00804, -0.01796], dtype=np.float64
)
OFFICIAL_MOUNTING_RING_OFFSET_M = np.asarray(
    [0.00780, 0.0, 0.0], dtype=np.float64
)
OFFICIAL_SOURCE_MESH_OFFSET_M = (
    OFFICIAL_WRIST_CENTER_OFFSET_M + OFFICIAL_MOUNTING_RING_OFFSET_M
)
for _official_offset in (
    OFFICIAL_WRIST_CENTER_OFFSET_M,
    OFFICIAL_MOUNTING_RING_OFFSET_M,
    OFFICIAL_SOURCE_MESH_OFFSET_M,
):
    _official_offset.setflags(write=False)


_LINK_COLORS: Mapping[str, Tuple[float, float, float]] = {
    "Link111": (0.34, 0.38, 0.45),
    "Link1": (0.20, 0.55, 0.95),
    "Link11": (0.48, 0.76, 1.00),
    "Link2": (0.18, 0.72, 0.45),
    "Link22": (0.50, 0.90, 0.67),
    "Link3": (0.95, 0.67, 0.14),
    "Link33": (1.00, 0.85, 0.42),
    "Link4": (0.78, 0.31, 0.78),
    "Link44": (0.94, 0.59, 0.94),
    "Link5": (0.92, 0.35, 0.14),
    "Link51": (1.00, 0.50, 0.20),
    "Link52": (1.00, 0.65, 0.30),
    "Link53": (1.00, 0.79, 0.50),
}


@dataclass(frozen=True)
class LinkVisual:
    name: str
    mesh_filename: str
    T_link_visual: np.ndarray


@dataclass(frozen=True)
class RevoluteJoint:
    name: str
    parent: str
    child: str
    T_parent_joint: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class InspireHandConfiguration:
    """One official discrete hand configuration selected by type and width."""

    grasp_type_id: int
    grasp_type_name: str
    width_m: float
    width_key_cm: str
    joint_positions_rad: np.ndarray
    actuator_registers: np.ndarray


def _numbers(text: Optional[str], *, count: int, default: Sequence[float]) -> np.ndarray:
    if text is None:
        values = np.asarray(default, dtype=np.float64)
    else:
        values = np.asarray([float(item) for item in text.split()], dtype=np.float64)
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError(f"expected {count} finite numeric values, got {text!r}")
    return values


def rotation_from_rpy(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis RPY rotation: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""

    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def rotation_about_axis(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """Rodrigues rotation around a URDF joint axis."""

    vector = np.asarray(axis, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("joint axis must be a finite vector of shape (3,)")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("joint axis must be non-zero")
    vector = vector / norm
    x, y, z = vector
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    return identity + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)


def _transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_from_rpy(rpy)
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def _axis_transform(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_about_axis(axis, angle_rad)
    return result


def _validate_pose(value: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite (4,4) transform")
    if not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return pose.copy()


class InspireHandModel:
    """Parsed official URDF plus cached high-resolution per-link STL meshes."""

    def __init__(
        self,
        urdf_path: str | Path,
        mapping_path: str | Path,
        *,
        mesh_resolution: str = "full",
    ) -> None:
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.mapping_path = Path(mapping_path).expanduser().resolve()
        if mesh_resolution not in ("full", "simplified"):
            raise ValueError("mesh_resolution must be 'full' or 'simplified'")
        self.mesh_resolution = mesh_resolution
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Inspire URDF not found: {self.urdf_path}")
        self.mapping = load_inspire_mapping(self.mapping_path)
        self.links, self.joints, self.root_link = self._parse_urdf(self.urdf_path)
        self._mesh_templates: Optional[Dict[str, Any]] = None

    @classmethod
    def from_anydex_root(
        cls,
        anydex_root: str | Path,
        *,
        mesh_resolution: str = "full",
    ) -> "InspireHandModel":
        root = Path(anydex_root).expanduser().resolve()
        hand_root = root / "generate_mesh_and_pointcloud/inspire_urdf"
        return cls(
            hand_root / "urdf-five3/robots/urdf-five3.urdf",
            hand_root / "width_12Dangle_6Dangle.json",
            mesh_resolution=mesh_resolution,
        )

    @staticmethod
    def _parse_urdf(
        path: Path,
    ) -> tuple[Tuple[LinkVisual, ...], Tuple[RevoluteJoint, ...], str]:
        root = ET.parse(path).getroot()
        links = []
        link_names = []
        for element in root.findall("link"):
            name = element.attrib.get("name", "")
            if not name:
                raise ValueError("URDF link is missing its name")
            visual = element.find("visual")
            if visual is None:
                raise ValueError(f"URDF link {name} has no visual")
            origin = visual.find("origin")
            xyz = _numbers(
                None if origin is None else origin.attrib.get("xyz"),
                count=3,
                default=(0, 0, 0),
            )
            rpy = _numbers(
                None if origin is None else origin.attrib.get("rpy"),
                count=3,
                default=(0, 0, 0),
            )
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.attrib.get("filename"):
                raise ValueError(f"URDF link {name} has no visual mesh filename")
            links.append(
                LinkVisual(
                    name=name,
                    mesh_filename=Path(mesh.attrib["filename"]).name,
                    T_link_visual=_transform(xyz, rpy),
                )
            )
            link_names.append(name)

        joints = []
        child_names = set()
        for element in root.findall("joint"):
            joint_type = element.attrib.get("type")
            if joint_type != "revolute":
                raise ValueError(
                    f"unsupported Inspire joint type {joint_type!r}; expected revolute"
                )
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError("URDF joint is missing parent or child")
            parent_name = parent.attrib.get("link", "")
            child_name = child.attrib.get("link", "")
            origin = element.find("origin")
            axis = element.find("axis")
            xyz = _numbers(
                None if origin is None else origin.attrib.get("xyz"),
                count=3,
                default=(0, 0, 0),
            )
            rpy = _numbers(
                None if origin is None else origin.attrib.get("rpy"),
                count=3,
                default=(0, 0, 0),
            )
            axis_xyz = _numbers(
                None if axis is None else axis.attrib.get("xyz"),
                count=3,
                default=(1, 0, 0),
            )
            axis_norm = float(np.linalg.norm(axis_xyz))
            if axis_norm <= 1e-12:
                raise ValueError(f"URDF joint {child_name} has a zero axis")
            joints.append(
                RevoluteJoint(
                    name=element.attrib.get("name", child_name),
                    parent=parent_name,
                    child=child_name,
                    T_parent_joint=_transform(xyz, rpy),
                    axis=axis_xyz / axis_norm,
                )
            )
            child_names.add(child_name)

        roots = sorted(set(link_names) - child_names)
        if len(roots) != 1:
            raise ValueError(f"expected one URDF root link, got {roots}")
        if len(links) != 13 or len(joints) != 12:
            raise ValueError(
                f"unexpected Inspire URDF topology: {len(links)} links, {len(joints)} joints"
            )
        return tuple(links), tuple(joints), roots[0]

    def configuration(
        self, grasp_type_id: int, width_m: float
    ) -> InspireHandConfiguration:
        """Resolve the exact JSON row used by the official AnyDex mapper."""

        if isinstance(grasp_type_id, (bool, np.bool_)):
            raise ValueError("grasp_type_id must be an integer in [1,8]")
        numeric_type = float(grasp_type_id)
        if not np.isfinite(numeric_type) or not numeric_type.is_integer():
            raise ValueError("grasp_type_id must be an integer in [1,8]")
        grasp_type = int(numeric_type)
        if grasp_type not in INSPIRE_GRASP_TYPES:
            raise ValueError(f"unsupported Inspire grasp type {grasp_type}; expected 1..8")
        if not np.isfinite(width_m) or float(width_m) < 0:
            raise ValueError("width_m must be finite and non-negative")
        type_name, (type_minimum, type_maximum) = INSPIRE_GRASP_TYPES[grasp_type]
        minimum = max(type_minimum, MIN_GRASP_WIDTH)
        maximum = min(type_maximum, MAX_GRASP_WIDTH)
        mapped_width = float(np.clip(width_m, minimum, maximum))
        width_key = str(np.round(mapped_width * 100.0, 1))
        try:
            entry = self.mapping[type_name][width_key]
        except KeyError as exc:
            raise KeyError(
                f"mapping has no {type_name} configuration at {width_key} cm"
            ) from exc
        joints = np.asarray(entry.get("12d"), dtype=np.float64)
        actuators = np.asarray(entry.get("6d"), dtype=np.float64)
        if joints.shape != (12,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"{type_name}/{width_key} 12d must be 12 finite radians")
        if actuators.shape != (6,) or not np.all(np.isfinite(actuators)):
            raise ValueError(f"{type_name}/{width_key} 6d must have six finite values")
        # Do not clamp to URDF limits.  The official generator uses resetJointState
        # and its calibrated table intentionally contains a few out-of-limit values.
        return InspireHandConfiguration(
            grasp_type_id=grasp_type,
            grasp_type_name=type_name,
            width_m=mapped_width,
            width_key_cm=width_key,
            joint_positions_rad=joints.copy(),
            actuator_registers=actuators.copy(),
        )

    def forward_kinematics(
        self, joint_positions_rad: Sequence[float]
    ) -> Dict[str, np.ndarray]:
        """Return ``T_urdf_root_link`` for the root and all 12 child links."""

        positions = np.asarray(joint_positions_rad, dtype=np.float64)
        if positions.shape != (len(self.joints),) or not np.all(np.isfinite(positions)):
            raise ValueError(
                f"joint_positions_rad must contain {len(self.joints)} finite values"
            )
        transforms: Dict[str, np.ndarray] = {
            self.root_link: np.eye(4, dtype=np.float64)
        }
        pending = list(zip(self.joints, positions.tolist()))
        while pending:
            progressed = False
            remaining = []
            for joint, value in pending:
                parent_pose = transforms.get(joint.parent)
                if parent_pose is None:
                    remaining.append((joint, value))
                    continue
                transforms[joint.child] = (
                    parent_pose
                    @ joint.T_parent_joint
                    @ _axis_transform(joint.axis, float(value))
                )
                progressed = True
            if not progressed:
                unresolved = [joint.name for joint, _ in remaining]
                raise ValueError(f"URDF joint tree cannot be resolved: {unresolved}")
            pending = remaining
        return transforms

    def link_mesh_transforms(
        self,
        T_reference_hand: np.ndarray,
        joint_positions_rad: Sequence[float],
    ) -> Dict[str, np.ndarray]:
        """Return final ``T_reference_link_visual`` matrices for all links."""

        hand_pose = _validate_pose(T_reference_hand, "T_reference_hand")
        fk = self.forward_kinematics(joint_positions_rad)
        source_offset = np.eye(4, dtype=np.float64)
        source_offset[:3, 3] = OFFICIAL_SOURCE_MESH_OFFSET_M
        output = {}
        for link in self.links:
            output[link.name] = (
                hand_pose @ source_offset @ fk[link.name] @ link.T_link_visual
            )
        return output

    def _mesh_directory(self) -> Path:
        package_root = self.urdf_path.parent.parent
        directory = "meshes" if self.mesh_resolution == "full" else "meshes_simplified"
        return package_root / directory

    def _load_mesh_templates(self) -> Dict[str, Any]:
        if self._mesh_templates is not None:
            return self._mesh_templates
        import open3d as o3d

        directory = self._mesh_directory()
        templates: Dict[str, Any] = {}
        for link in self.links:
            path = directory / link.mesh_filename
            mesh = o3d.io.read_triangle_mesh(str(path))
            if mesh.is_empty() or len(mesh.triangles) == 0:
                raise ValueError(f"failed to load Inspire link mesh: {path}")
            templates[link.name] = mesh
        self._mesh_templates = templates
        return templates

    def build_link_meshes(
        self,
        T_reference_hand: np.ndarray,
        joint_positions_rad: Sequence[float],
    ) -> list[Any]:
        """Build 13 separately colored legacy Open3D meshes in reference frame."""

        templates = self._load_mesh_templates()
        transforms = self.link_mesh_transforms(T_reference_hand, joint_positions_rad)
        meshes = []
        for link in self.links:
            mesh = copy.deepcopy(templates[link.name])
            mesh.transform(transforms[link.name])
            mesh.paint_uniform_color(_LINK_COLORS.get(link.name, (0.75, 0.78, 0.82)))
            mesh.compute_vertex_normals()
            meshes.append(mesh)
        return meshes

    def build_selected_snapshot_links(
        self, snapshot: VisualizationSnapshot, selected_index: int
    ) -> list[Any]:
        """Open3D callback for one selected snapshot candidate."""

        grasps = snapshot.grasps
        selected = int(selected_index)
        if selected < 0 or selected >= grasps.count:
            raise ValueError(f"selected_index={selected} is outside snapshot candidates")
        if grasps.hand_poses is None:
            raise ValueError(
                "snapshot has no Inspire hand pose; run the official backend or "
                "apply an explicitly diagnostic Inspire type to this snapshot"
            )
        if grasps.widths_m is None:
            raise ValueError("snapshot has no grasp width for Inspire mesh lookup")
        grasp_type = int(np.asarray(grasps.type_ids)[selected])
        width_m = float(np.asarray(grasps.widths_m)[selected])
        config = self.configuration(grasp_type, width_m)
        return self.build_link_meshes(
            np.asarray(grasps.hand_poses, dtype=np.float64)[selected],
            config.joint_positions_rad,
        )


class SelectedInspireHandLinkMeshBuilder:
    """Callable adapter consumed by :func:`build_open3d_geometries`."""

    def __init__(
        self,
        anydex_root: str | Path,
        *,
        mesh_resolution: str = "full",
    ) -> None:
        self.model = InspireHandModel.from_anydex_root(
            anydex_root, mesh_resolution=mesh_resolution
        )

    def __call__(
        self, snapshot: VisualizationSnapshot, selected_index: int
    ) -> list[Any]:
        return self.model.build_selected_snapshot_links(snapshot, selected_index)
