"""Official RH56 six-register to URDF twelve-joint conversion.

This is a small, hardware-free extraction of ``read_excel_6Dangle_to_12Dangle``
from the vendored AnyDexGrasp release.  It deliberately reads the released
``driver_routine_to_angle.xls`` calibration table and applies the same integer
indexing and formulae; it does not interpolate or invent a linear 6D-to-12D
mapping.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


DRIVER_WORKBOOK_SHA256 = (
    "23ca934b1092ce1a42e46f7bd7edc1ac0e3cacb98efe397ab9d9db1c938e7bdb"
)
MAPPING_ALGORITHM = "anydex_recover_inspire_hand_to_stl_driver_xls_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actuators(value: Sequence[int]) -> Tuple[int, ...]:
    raw = np.asarray(tuple(value), dtype=np.float64)
    if (
        raw.shape != (6,)
        or not np.all(np.isfinite(raw))
        or not np.array_equal(raw, np.rint(raw))
        or np.any(raw < 0)
        or np.any(raw > 1000)
    ):
        raise ValueError("RH56 actuator registers must be six integers in [0,1000]")
    return tuple(int(item) for item in raw)


def _anti_rate(rate1: Tuple[float, float], rate2: Tuple[float, float], value: float) -> float:
    return float(
        rate2[1]
        - (float(value) - rate1[0]) / (rate1[1] - rate1[0])
        * (rate2[1] - rate2[0])
    )


class OfficialRH56ActuatorMapper:
    """Immutable view of the official actuator calibration workbook."""

    def __init__(self, driver_workbook_path: str | Path) -> None:
        path = Path(driver_workbook_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("official RH56 driver workbook not found: {}".format(path))
        digest = _sha256(path)
        if digest != DRIVER_WORKBOOK_SHA256:
            raise ValueError(
                "RH56 driver workbook checksum mismatch: expected={}, actual={}".format(
                    DRIVER_WORKBOOK_SHA256, digest
                )
            )
        try:
            import xlrd
        except ImportError as exc:  # Fail closed; never substitute an approximation.
            raise RuntimeError("xlrd is required to read the official RH56 .xls mapping") from exc
        sheet = xlrd.open_workbook(filename=str(path)).sheet_by_index(0)
        columns = tuple(
            np.asarray(sheet.col_values(index)[2:], dtype=np.float64)
            for index in range(7)
        )
        if any(column.shape != (2001,) for column in columns):
            raise ValueError("official RH56 driver workbook has unexpected dimensions")
        if any(not np.all(np.isfinite(column)) for column in columns):
            raise ValueError("official RH56 driver workbook contains non-finite data")
        self.path = path
        self.sha256 = digest
        self._columns = columns

    @classmethod
    def from_anydex_root(cls, anydex_root: str | Path) -> "OfficialRH56ActuatorMapper":
        root = Path(anydex_root).expanduser().resolve()
        return cls(
            root
            / "generate_mesh_and_pointcloud/inspire_urdf/driver_routine_to_angle.xls"
        )

    def to_joint_positions_rad(self, actuator_registers: Sequence[int]) -> np.ndarray:
        """Return official URDF order: four fingers (2 each), thumb rotate/bend (4)."""

        little, ring, middle, index, thumb_bend, thumb_rotate = _actuators(
            actuator_registers
        )
        bend_col1 = self._columns[5]
        bend_col2 = self._columns[6]

        def finger(value: int) -> Tuple[float, float]:
            # This special case is explicit in the official get_angle helper.
            if value == 1000:
                return (0.0, 0.0)
            scope = int(float(value) * (1857 - 20) / 1000)
            return (
                _anti_rate((178.66, 93.97), (0.0, -1.511), bend_col1[scope]),
                _anti_rate((180.78, 88.7), (0.0, -1.5416), bend_col2[scope])
                + 0.25,
            )

        thumb_scope = int(float(thumb_bend) * (1606 - 53) / 1000)
        thumb_bending = (
            (self._columns[1][thumb_scope] - 139.27)
            / (139.27 - 110.53)
            * 0.433
            + 0.13,
            (self._columns[2][thumb_scope] - 188.08)
            / (154.27 - 188.08)
            * (0.179 + 0.267)
            - 0.237,
            (self._columns[3][thumb_scope] - 169.94)
            / (147.53 - 169.94)
            * 0.503
            - 0.7,
        )
        rotation_scope = int(float(thumb_rotate) * (2000 - 400) / 1000) + 400
        thumb_rotation = _anti_rate(
            (185.5631219, 106.72277927615),
            (-1.3, 0.3),
            self._columns[4][rotation_scope],
        )
        index_joints = finger(index)
        middle_joints = finger(middle)
        ring_joints = finger(ring)
        little_joints = finger(little)
        result = np.asarray(
            index_joints
            + middle_joints
            + ring_joints
            + little_joints
            + (thumb_rotation,)
            + thumb_bending,
            dtype=np.float64,
        )
        if result.shape != (12,) or not np.all(np.isfinite(result)):
            raise RuntimeError("official RH56 mapping produced invalid joint values")
        return result

    def feedback_to_joint_positions_rad(
        self, angle_act_registers: Sequence[int]
    ) -> np.ndarray:
        """Reconstruct q12 from six RH56 ``ANGLE_ACT`` registers.

        The released driver exposes commanded and actual angle in the same
        six-axis 0..1000 coordinate, so the checksum-pinned discrete workbook
        conversion can be applied to actual-angle feedback without inventing
        an interpolation.  The returned q12 is nevertheless a kinematic model
        reconstruction: it is not 12 independently measured joint angles and
        cannot observe backlash, compliance, or contact deformation.
        """

        return self.to_joint_positions_rad(angle_act_registers)

    def provenance(self) -> dict:
        return {
            "algorithm": MAPPING_ALGORITHM,
            "driver_workbook_path": str(self.path),
            "driver_workbook_sha256": self.sha256,
            "integer_indexing": True,
            "interpolation": False,
        }


__all__ = [
    "DRIVER_WORKBOOK_SHA256",
    "MAPPING_ALGORITHM",
    "OfficialRH56ActuatorMapper",
]
