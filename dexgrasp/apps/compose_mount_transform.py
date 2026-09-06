#!/usr/bin/env python3
"""Compose the measured FR3/adapter/Inspire mount transform offline.

This utility performs matrix arithmetic only.  It intentionally imports no
Franka or Inspire driver and cannot connect to either device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anydex_pipeline.control_frames import (
    T_MOUNT_SOURCE_AXIS_BASIS,
    compose_T_EE_hand_source,
    compose_T_flange_hand_source,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "fr3_rh56_anydex_mount_transform_measurement"


def _row_major_4x4(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (16,):
        array = array.reshape((4, 4), order="C")
    elif array.shape != (4, 4):
        raise ValueError(
            "{} must be 16 row-major values or a nested 4x4 array".format(name)
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return array


def _json_value_at_key(document: Any, dotted_key: str) -> Any:
    value = document
    for component in dotted_key.split("."):
        if not component:
            raise ValueError("F_T_EE JSON key must not contain empty components")
        if not isinstance(value, Mapping) or component not in value:
            raise KeyError("F_T_EE JSON key {!r} was not found".format(dotted_key))
        value = value[component]
    return value


def _load_F_T_EE(args: argparse.Namespace) -> np.ndarray:
    if args.F_T_EE_values is not None:
        return _row_major_4x4(args.F_T_EE_values, "F_T_EE")

    path = args.F_T_EE_json.expanduser()
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    value = _json_value_at_key(document, args.F_T_EE_json_key)
    return _row_major_4x4(value, "F_T_EE JSON value")


def _flat_row_major(matrix: np.ndarray) -> list:
    return np.asarray(matrix, dtype=np.float64).reshape(16, order="C").tolist()


def _artifact(
    *,
    F_T_EE: np.ndarray,
    T_F_hand_source: np.ndarray,
    T_EE_hand_source: np.ndarray,
    adapter_seat_mm: float,
    assembled_yaw_deg: float,
    seating_to_source_origin_mm: Sequence[float],
    mark_measured: bool,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "matrix_convention": {
            "transform": "T_A_B maps coordinates in frame B into frame A",
            "serialization": "row-major homogeneous 4x4 flattened to 16 values",
            "units": "metres in transforms; CLI mechanical inputs are millimetres",
        },
        "inputs": {
            "F_T_EE": _flat_row_major(F_T_EE),
            "adapter_face_to_seating_plane_mm": float(adapter_seat_mm),
            "assembled_yaw_deg": float(assembled_yaw_deg),
            "seating_to_source_origin_mm": [
                float(value) for value in seating_to_source_origin_mm
            ],
        },
        "T_F_hand_source": _flat_row_major(T_F_hand_source),
        "T_EE_hand_source": _flat_row_major(T_EE_hand_source),
        "provenance": {
            "composer": "anydex_pipeline.control_frames",
            "anydex_source_axis_basis": {
                "symbol": "P",
                "T_mount_source_row_major": _flat_row_major(
                    T_MOUNT_SOURCE_AXIS_BASIS
                ),
                "axis_mapping": [
                    "source +X maps to mount +Z",
                    "source +Y maps to mount +X",
                    "source +Z maps to mount +Y",
                ],
            },
            "excluded_from_composition": [
                {
                    "name": "upstream_UR_flange_to_TCP_offset",
                    "value_mm": 44.0,
                    "reason": "specific to the upstream UR mounting setup",
                    "included": False,
                },
                {
                    "name": "upstream_empirical_approach_insertion",
                    "value_mm": 14.0,
                    "reason": "grasp execution bias, not a rigid mount transform",
                    "included": False,
                },
            ],
        },
        "measurement_status": {
            "mount_datum_captured": bool(mark_measured),
            "commissioned": False,
            "collision_model_validated": False,
            "payload_mass_properties_validated": False,
            "full_execution_ready": False,
            "scope": (
                "operator marked the yaw and seating/source datum as measured"
                if mark_measured
                else "numerical composition only; mount datum not marked measured"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only composition of T_F_hand_source and T_EE_hand_source. "
            "No robot or hand driver is imported."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--F-T-EE",
        dest="F_T_EE_values",
        type=float,
        nargs=16,
        metavar=(
            "R00",
            "R01",
            "R02",
            "TX",
            "R10",
            "R11",
            "R12",
            "TY",
            "R20",
            "R21",
            "R22",
            "TZ",
            "H0",
            "H1",
            "H2",
            "H3",
        ),
        help="F_T_EE as 16 row-major homogeneous-matrix values",
    )
    source.add_argument(
        "--F-T-EE-json",
        dest="F_T_EE_json",
        type=Path,
        help="JSON file containing F_T_EE",
    )
    parser.add_argument(
        "--F-T-EE-json-key",
        default="F_T_EE",
        help="Dotted JSON key used with --F-T-EE-json (default: F_T_EE)",
    )
    parser.add_argument(
        "--assembled-yaw-deg",
        type=float,
        required=True,
        help="Measured hand yaw around the adapter seating-frame +Z axis",
    )
    parser.add_argument(
        "--seating-to-source-origin-mm",
        type=float,
        nargs=3,
        required=True,
        metavar=("DX", "DY", "DZ"),
        help="Measured source-frame origin offset, expressed in seating axes",
    )
    parser.add_argument(
        "--adapter-seat-mm",
        type=float,
        default=10.0,
        help="FR3 face to RH56 seating plane distance (default: 10.0 mm)",
    )
    parser.add_argument(
        "--mark-measured",
        action="store_true",
        help=(
            "Record that yaw and seating/source datum were physically measured; "
            "does not mark the system commissioned or execution-ready"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON measurement artifact",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        F_T_EE = _load_F_T_EE(args)
        offset_mm = np.asarray(
            args.seating_to_source_origin_mm, dtype=np.float64
        )
        if offset_mm.shape != (3,) or not np.all(np.isfinite(offset_mm)):
            raise ValueError(
                "seating-to-source-origin must contain three finite values"
            )
        if not np.isfinite(args.adapter_seat_mm):
            raise ValueError("adapter-seat-mm must be finite")
        if not np.isfinite(args.assembled_yaw_deg):
            raise ValueError("assembled-yaw-deg must be finite")

        T_F_hand_source = compose_T_flange_hand_source(
            fr3_face_to_rh56_seating_plane_m=args.adapter_seat_mm / 1000.0,
            assembled_yaw_rad=np.deg2rad(args.assembled_yaw_deg),
            seating_to_source_origin_m=offset_mm / 1000.0,
        )
        T_EE_hand_source = compose_T_EE_hand_source(
            F_T_EE, T_F_hand_source
        )
        artifact = _artifact(
            F_T_EE=F_T_EE,
            T_F_hand_source=T_F_hand_source,
            T_EE_hand_source=T_EE_hand_source,
            adapter_seat_mm=args.adapter_seat_mm,
            assembled_yaw_deg=args.assembled_yaw_deg,
            seating_to_source_origin_mm=offset_mm.tolist(),
            mark_measured=args.mark_measured,
        )
        rendered = json.dumps(
            artifact, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        if args.output is not None:
            args.output.expanduser().write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
