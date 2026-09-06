"""Auditable official RH56 all-six-open URDF configuration.

The released AnyDexGrasp mapping JSON contains grasp-specific closed poses;
it does not contain the hardware command ``[1000] * 6``.  The same official
checkout does include the actuator-to-URDF generator and its two calibration
workbooks.  Applying that generator at 1000 gives the constants below.

Keeping this as a small, explicit datum avoids silently treating an arbitrary
grasp row as the open hand during installed-tool collision checks.  The
provenance helper binds all three upstream sources by SHA-256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping

import numpy as np


OFFICIAL_OPEN_ACTUATOR_TARGETS = np.asarray(
    [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float64
)

# Joint order is the official urdf-five3 order:
# index(2), middle(2), ring(2), little(2), thumb rotate, thumb bend(3).
# Four bend axes use get_angle(...): actuator==1000 -> (0, 0).
# Thumb values are the exact generator result at actuator 1000 using the
# vendored driver workbook.  See recover_inspire_hand_to_stl.py.
OFFICIAL_OPEN_JOINT_POSITIONS_RAD = np.asarray(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2999999995533857,
        -0.28579147703794766,
        0.19315849157054144,
        -0.22522725568942426,
    ],
    dtype=np.float64,
)

_GENERATOR_RELATIVE = Path(
    "generate_mesh_and_pointcloud/recover_inspire_hand_to_stl.py"
)
_ROUTINE_RELATIVE = Path(
    "generate_mesh_and_pointcloud/inspire_urdf/"
    "inspire_hand_routine_to_angle-use.xlsx"
)
_DRIVER_RELATIVE = Path(
    "generate_mesh_and_pointcloud/inspire_urdf/driver_routine_to_angle.xls"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_open_configuration_provenance(
    anydex_root: str | Path,
) -> Mapping[str, object]:
    """Return immutable source paths/hashes for the all-open FK datum."""

    root = Path(anydex_root).expanduser().resolve()
    sources: Dict[str, Path] = {
        "generator": (root / _GENERATOR_RELATIVE).resolve(),
        "routine_workbook": (root / _ROUTINE_RELATIVE).resolve(),
        "driver_workbook": (root / _DRIVER_RELATIVE).resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "official Inspire open-configuration source is missing: "
            + ", ".join(missing)
        )
    return {
        "method": (
            "official recover_inspire_hand_to_stl.py actuator-to-12D "
            "generator evaluated at [1000]*6"
        ),
        "actuator_targets": OFFICIAL_OPEN_ACTUATOR_TARGETS.tolist(),
        "joint_positions_rad": OFFICIAL_OPEN_JOINT_POSITIONS_RAD.tolist(),
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
    }


__all__ = [
    "OFFICIAL_OPEN_ACTUATOR_TARGETS",
    "OFFICIAL_OPEN_JOINT_POSITIONS_RAD",
    "official_open_configuration_provenance",
]
