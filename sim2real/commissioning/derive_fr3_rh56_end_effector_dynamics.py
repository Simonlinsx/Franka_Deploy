#!/usr/bin/env python3
"""Derive a reproducible FR3 + V7 adapter + RH56 dynamics candidate.

This is an offline CAD/URDF calculation.  It never imports pylibfranka and
never opens a robot or hand interface.  The result is a candidate for Franka
Desk end-effector commissioning, not a substitute for weighing the installed
assembly or validating the result on the stationary robot.

The AnyDex RH56 URDF contains per-link inertial properties whose total mass is
not the manufacturer's nominal hand mass.  We therefore preserve its relative
mass distribution while scaling all link masses/inertias to the supplied
nominal hand mass.  The remaining installed mass is assigned to the V7 adapter
mesh (including its fasteners), using the mesh's geometric mass distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Iterable
from xml.etree import ElementTree

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "dexgrasp/configs/fr3_rh56_v94_commissioning.json"
DEFAULT_URDF = (
    ROOT
    / "dexgrasp/third_party/AnyDexGrasp/generate_mesh_and_pointcloud"
    / "inspire_urdf/urdf-five3/robots/urdf-five3.urdf"
)
DEFAULT_ADAPTER = (
    ROOT / "dexgrasp/assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl"
)

# Exact transforms applied by AnyDex's official RH56 mesh generator after the
# URDF FK-combined mesh is assembled.  The resulting frame is the hand-source
# frame consumed by tool.T_EE_hand in the commissioning profile.
OFFICIAL_SOURCE_MESH_OFFSET_M = np.asarray(
    [0.04123 + 0.00780, 0.00804, -0.01796], dtype=np.float64
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vector(text: str) -> np.ndarray:
    value = np.asarray([float(item) for item in text.split()], dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid 3-vector: {text!r}")
    return value


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = _rpy_rotation(rpy)
    value[:3, 3] = xyz
    return value


def _parallel_axis(mass: float, displacement: np.ndarray) -> np.ndarray:
    return float(mass) * (
        float(displacement @ displacement) * np.eye(3)
        - np.outer(displacement, displacement)
    )


def _combine(
    components: Iterable[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[float, np.ndarray, np.ndarray]:
    items = tuple(components)
    mass = float(sum(item[0] for item in items))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("combined mass must be positive")
    center = sum(item[0] * item[1] for item in items) / mass
    inertia = sum(
        item[2] + _parallel_axis(item[0], item[1] - center) for item in items
    )
    inertia = 0.5 * (inertia + inertia.T)
    if float(np.min(np.linalg.eigvalsh(inertia))) <= 0.0:
        raise ValueError("combined inertia is not positive definite")
    return mass, center, inertia


def _urdf_hand_properties(urdf: Path) -> tuple[float, np.ndarray, np.ndarray]:
    robot = ElementTree.parse(urdf).getroot()
    links = robot.findall("link")
    joints = {}
    for joint in robot.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        origin = joint.find("origin")
        if child is None or parent is None or origin is None:
            raise ValueError("URDF joint is missing child/parent/origin")
        joints[child.get("link")] = (
            parent.get("link"),
            _transform(_vector(origin.get("xyz")), _vector(origin.get("rpy"))),
        )

    children = set(joints)
    roots = [link.get("name") for link in links if link.get("name") not in children]
    if len(roots) != 1:
        raise ValueError(f"expected one URDF root link, got {roots}")
    link_transforms = {roots[0]: np.eye(4, dtype=np.float64)}
    while len(link_transforms) < len(links):
        before = len(link_transforms)
        for child, (parent, transform) in joints.items():
            if child not in link_transforms and parent in link_transforms:
                # q=0 is the canonical open-pose candidate.  A Franka Desk
                # end-effector model is fixed, while RH56 link poses vary;
                # the dominant base-link mass makes this approximation stable.
                link_transforms[child] = link_transforms[parent] @ transform
        if len(link_transforms) == before:
            raise ValueError("URDF joint tree could not be resolved")

    components = []
    for link in links:
        name = link.get("name")
        inertial = link.find("inertial")
        if name not in link_transforms or inertial is None:
            raise ValueError(f"URDF link {name!r} has no inertial data")
        origin = inertial.find("origin")
        mass_node = inertial.find("mass")
        inertia_node = inertial.find("inertia")
        if origin is None or mass_node is None or inertia_node is None:
            raise ValueError(f"URDF link {name!r} has incomplete inertial data")
        mass = float(mass_node.get("value"))
        local = _transform(_vector(origin.get("xyz")), _vector(origin.get("rpy")))
        link_to_root = link_transforms[name]
        rotation = link_to_root[:3, :3] @ local[:3, :3]
        center = (link_to_root @ np.r_[local[:3, 3], 1.0])[:3]
        ixx = float(inertia_node.get("ixx"))
        ixy = float(inertia_node.get("ixy"))
        ixz = float(inertia_node.get("ixz"))
        iyy = float(inertia_node.get("iyy"))
        iyz = float(inertia_node.get("iyz"))
        izz = float(inertia_node.get("izz"))
        inertia = np.asarray(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
            dtype=np.float64,
        )
        components.append((mass, center, rotation @ inertia @ rotation.T))
    return _combine(components)


def _binary_stl_properties(stl: Path) -> tuple[float, np.ndarray, np.ndarray]:
    raw = stl.read_bytes()
    if len(raw) < 84:
        raise ValueError("adapter STL is truncated")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * triangle_count:
        raise ValueError("adapter STL must be binary with an exact triangle count")
    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
    )
    vertices = np.frombuffer(raw, dtype=dtype, offset=84, count=triangle_count)[
        "vertices"
    ].astype(np.float64)
    # The commissioned V7 STL is in millimetres and already uses the Franka
    # flange frame (z=0 at the flange face).
    vertices *= 1.0e-3
    signed_volumes = np.linalg.det(vertices) / 6.0
    volume = float(np.sum(signed_volumes))
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("adapter STL has invalid signed volume")
    center = np.sum(
        signed_volumes[:, None] * np.sum(vertices, axis=1) / 4.0, axis=0
    ) / volume

    vertex_sum = np.sum(vertices, axis=1)
    vertex_square_sum = np.einsum("tki,tkj->tij", vertices, vertices)
    second_moment_per_tetra = (
        np.einsum("ti,tj->tij", vertex_sum, vertex_sum) + vertex_square_sum
    ) / 20.0
    second_moment_per_unit_mass = np.sum(
        signed_volumes[:, None, None] * second_moment_per_tetra, axis=0
    ) / volume
    inertia_at_origin = (
        np.trace(second_moment_per_unit_mass) * np.eye(3)
        - second_moment_per_unit_mass
    )
    inertia_at_center_per_unit_mass = inertia_at_origin - (
        float(center @ center) * np.eye(3) - np.outer(center, center)
    )
    inertia_at_center_per_unit_mass = 0.5 * (
        inertia_at_center_per_unit_mass + inertia_at_center_per_unit_mass.T
    )
    if float(np.min(np.linalg.eigvalsh(inertia_at_center_per_unit_mass))) <= 0.0:
        raise ValueError("adapter STL inertia is not positive definite")
    return volume, center, inertia_at_center_per_unit_mass


def derive(
    *,
    profile_path: Path,
    urdf_path: Path,
    adapter_path: Path,
    assembly_mass_kg: float,
    nominal_hand_mass_kg: float,
) -> dict:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    franka = profile["franka"]
    tool = profile["tool"]
    F_T_EE = np.asarray(franka["expected_F_T_EE"], dtype=np.float64)
    T_EE_hand = np.asarray(tool["T_EE_hand"], dtype=np.float64)
    if F_T_EE.shape != (4, 4) or T_EE_hand.shape != (4, 4):
        raise ValueError("profile transforms must be 4x4")
    T_F_hand = F_T_EE @ T_EE_hand

    assembly_mass = float(assembly_mass_kg)
    hand_mass = float(nominal_hand_mass_kg)
    adapter_mass = assembly_mass - hand_mass
    if not (math.isfinite(assembly_mass) and math.isfinite(hand_mass)):
        raise ValueError("masses must be finite")
    if hand_mass <= 0.0 or adapter_mass <= 0.0:
        raise ValueError("assembly mass must exceed the positive hand mass")

    urdf_mass, urdf_center, urdf_inertia = _urdf_hand_properties(urdf_path)
    hand_scale = hand_mass / urdf_mass
    source_center = urdf_center + OFFICIAL_SOURCE_MESH_OFFSET_M
    hand_center_F = (
        T_F_hand @ np.r_[source_center, 1.0]
    )[:3]
    hand_inertia_F = (
        T_F_hand[:3, :3]
        @ (urdf_inertia * hand_scale)
        @ T_F_hand[:3, :3].T
    )

    adapter_volume, adapter_center_F, adapter_inertia_per_kg = (
        _binary_stl_properties(adapter_path)
    )
    adapter_inertia_F = adapter_inertia_per_kg * adapter_mass
    combined_mass, combined_center_F, combined_inertia_F = _combine(
        (
            (hand_mass, hand_center_F, hand_inertia_F),
            (adapter_mass, adapter_center_F, adapter_inertia_F),
        )
    )
    current = franka["expected_end_effector"]
    current_center = np.asarray(current["F_x_Cee_m"], dtype=np.float64)
    current_inertia = np.asarray(current["inertia_kg_m2"], dtype=np.float64)
    return {
        "schema_version": 1,
        "classification": "offline_cad_urdf_candidate_not_physical_identification",
        "warning": (
            "Do not deploy until the installed assembly mass is confirmed, the values "
            "are entered as the selected end effector in Franka Desk, and a stationary "
            "read-only RobotState verifies the exact parameters and low residual wrench."
        ),
        "assumptions": {
            "canonical_rh56_pose": "URDF q=0 (open-pose fixed EE approximation)",
            "urdf_mass_distribution_scaled_uniformly_to_nominal_hand_mass": True,
            "adapter_and_fastener_mass_uses_v7_stl_geometric_distribution": True,
            "external_load_mass_kg": 0.0,
            "F_T_EE": F_T_EE.tolist(),
        },
        "inputs": {
            "assembly_mass_kg": assembly_mass,
            "nominal_hand_mass_kg": hand_mass,
            "adapter_and_fastener_mass_kg": adapter_mass,
            "profile": str(profile_path.resolve()),
            "profile_sha256": _sha256(profile_path),
            "rh56_urdf": str(urdf_path.resolve()),
            "rh56_urdf_sha256": _sha256(urdf_path),
            "rh56_urdf_raw_mass_kg": urdf_mass,
            "rh56_urdf_mass_scale": hand_scale,
            "adapter_stl": str(adapter_path.resolve()),
            "adapter_stl_sha256": _sha256(adapter_path),
            "adapter_stl_signed_volume_m3": adapter_volume,
            "official_source_mesh_offset_m": OFFICIAL_SOURCE_MESH_OFFSET_M.tolist(),
            "T_F_hand_source": T_F_hand.tolist(),
        },
        "components": {
            "rh56": {
                "mass_kg": hand_mass,
                "F_x_C_m": hand_center_F.tolist(),
                "inertia_at_com_kg_m2": hand_inertia_F.tolist(),
            },
            "v7_adapter_and_fasteners": {
                "mass_kg": adapter_mass,
                "F_x_C_m": adapter_center_F.tolist(),
                "inertia_at_com_kg_m2": adapter_inertia_F.tolist(),
            },
        },
        "candidate_end_effector": {
            "mass_kg": combined_mass,
            "F_x_Cee_m": combined_center_F.tolist(),
            "inertia_kg_m2": combined_inertia_F.tolist(),
        },
        "difference_from_current_profile": {
            "mass_kg": combined_mass - float(current["mass_kg"]),
            "F_x_Cee_m": (combined_center_F - current_center).tolist(),
            "F_x_Cee_l2_m": float(np.linalg.norm(combined_center_F - current_center)),
            "inertia_kg_m2": (combined_inertia_F - current_inertia).tolist(),
        },
        "desk_column_major": {
            "mass_kg": combined_mass,
            "center_of_mass_m": combined_center_F.tolist(),
            "inertia_kg_m2": combined_inertia_F.reshape(9, order="F").tolist(),
            "transformation": F_T_EE.reshape(16, order="F").tolist(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--rh56-urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--adapter-stl", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--assembly-mass-kg", type=float, required=True)
    parser.add_argument("--nominal-hand-mass-kg", type=float, default=0.54)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = derive(
        profile_path=args.profile.expanduser().resolve(),
        urdf_path=args.rh56_urdf.expanduser().resolve(),
        adapter_path=args.adapter_stl.expanduser().resolve(),
        assembly_mass_kg=args.assembly_mass_kg,
        nominal_hand_mass_kg=args.nominal_hand_mass_kg,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
        print(f"[SAVED] {destination}")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
