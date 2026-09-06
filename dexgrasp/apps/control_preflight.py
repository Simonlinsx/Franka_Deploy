#!/usr/bin/env python3
"""Report FR3/RH56 grasp-control readiness; hardware access is opt-in/read-only."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Iterable, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydex_pipeline.control_config import (
    control_readiness,
    load_control_config,
    verify_adapter_assets,
)
from anydex_pipeline.snapshot import load_snapshot_npz


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"


def _matrix(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64).reshape((4, 4), order="F")


def _format_matrix(matrix: np.ndarray) -> str:
    rows = []
    for row in np.asarray(matrix, dtype=np.float64):
        rows.append("  [" + ", ".join(f"{value: .8f}" for value in row) + "]")
    return "[\n" + ",\n".join(rows) + "\n]"


def _print_blockers(title: str, blockers: tuple[str, ...]) -> None:
    print(f"[{title}] {'READY' if not blockers else 'LOCKED'}")
    for item in blockers:
        print(f"  - {item}")


def _print_snapshot(snapshot: Any) -> None:
    selected = int(snapshot.grasps.selected_index)
    print(
        f"[snapshot] frame={snapshot.frame_id} reference={snapshot.reference_frame} "
        f"calibration={snapshot.calibration_id} camera={snapshot.camera_serial}"
    )
    print(f"[snapshot] model={snapshot.model_name}")
    if 0 <= selected < snapshot.grasps.count:
        print(
            f"[snapshot] selected={selected} score="
            f"{float(snapshot.grasps.scores[selected]):.5f}"
        )
        if snapshot.grasps.hand_angles is not None:
            targets = np.asarray(snapshot.grasps.hand_angles[selected], dtype=np.float64)
            print(f"[snapshot] Inspire target={np.rint(targets).astype(int).tolist()}")


def _read_franka(config: dict[str, Any]) -> None:
    try:
        import pylibfranka
    except ImportError as exc:
        raise RuntimeError(
            "pylibfranka is unavailable; run with /home/qiaoguanren/code/franka/.venv/bin/python"
        ) from exc
    robot = pylibfranka.Robot(
        str(config["franka"]["ip"]), pylibfranka.RealtimeConfig.kIgnore
    )
    state = robot.read_once()
    print(f"[Franka/read-only] robot_mode={state.robot_mode}")
    print(
        "[Franka/read-only] q="
        + np.array2string(np.asarray(state.q), precision=7, separator=", ")
    )
    print("[Franka/read-only] O_T_EE=\n" + _format_matrix(_matrix(state.O_T_EE)))
    print("[Franka/read-only] F_T_EE=\n" + _format_matrix(_matrix(state.F_T_EE)))
    print(
        f"[Franka/read-only] m_ee={float(state.m_ee):.6f}kg "
        f"m_load={float(state.m_load):.6f}kg m_total={float(state.m_total):.6f}kg"
    )
    print(
        "[Franka/read-only] F_x_Cee="
        + np.array2string(np.asarray(state.F_x_Cee), precision=8, separator=", ")
    )
    print(
        "[Franka/read-only] I_ee(column-major)="
        + np.array2string(np.asarray(state.I_ee), precision=10, separator=", ")
    )
    for field, unit in (
        ("tau_ext_hat_filtered", "Nm"),
        ("O_F_ext_hat_K", "N/Nm"),
    ):
        values = getattr(state, field, None)
        if values is not None:
            print(
                f"[Franka/read-only] {field}({unit})="
                + np.array2string(
                    np.asarray(values), precision=7, separator=", "
                )
            )
    print("[Franka/read-only] no motion command was created")


def _read_inspire(config: dict[str, Any]) -> None:
    from examples.inspire_rh56_test import LinuxSerial, RH56Hand

    settings = config["inspire"]
    with LinuxSerial(
        str(settings["port"]), int(settings["baud"]), timeout=0.5
    ) as transport:
        hand = RH56Hand(transport, int(settings["hand_id"]))
        snapshot = hand.snapshot()
    print(
        f"[Inspire/read-only] angles={list(snapshot['angles'])} "
        f"targets={list(snapshot['angle_targets'])}"
    )
    print(
        f"[Inspire/read-only] status={list(snapshot['statuses'])} "
        f"errors={list(snapshot['errors'])} current_mA={list(snapshot['currents'])}"
    )
    print("[Inspire/read-only] no register was written")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the adapter/control profile and list every execution blocker. "
            "Default operation is offline and cannot move either device."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument(
        "--selected-index",
        type=int,
        default=None,
        help=(
            "Offline-only candidate override used for readiness reporting; "
            "does not modify the snapshot"
        ),
    )
    parser.add_argument(
        "--read-hardware",
        action="store_true",
        help="Connect to Franka and RH56 for one read-only state snapshot; sends no motion/write",
    )
    parser.add_argument(
        "--strict-full-grasp",
        action="store_true",
        help="Return a non-zero exit code while any full-grasp blocker remains",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config, config_path = load_control_config(args.config)
    assets = verify_adapter_assets(config, config_path)
    print(f"[config] {config_path}")
    print(
        f"[adapter] verified mesh={assets.mesh_path} sha256={assets.mesh_sha256}"
    )
    tool = config["tool"]
    print(
        "[adapter] disk=Ø70x10mm spigot=Ø37.6x7.8mm "
        f"mount-plane={float(tool['fr3_face_to_rh56_seating_plane_m']) * 1000:.1f}mm "
        f"collision-envelope={float(tool['adapter_total_axial_envelope_m']) * 1000:.1f}mm"
    )

    snapshot = None
    if args.snapshot is not None:
        snapshot = load_snapshot_npz(args.snapshot)
        if args.selected_index is not None:
            selected = int(args.selected_index)
            if not 0 <= selected < snapshot.grasps.count:
                raise ValueError(
                    "--selected-index is outside the snapshot candidate range"
                )
            snapshot = replace(
                snapshot,
                grasps=replace(snapshot.grasps, selected_index=selected),
            )
        _print_snapshot(snapshot)
    elif args.selected_index is not None:
        raise ValueError("--selected-index requires --snapshot")
    readiness = control_readiness(config, snapshot)
    _print_blockers("default-pose motion", readiness.default_motion_blockers)
    _print_blockers("full grasp", readiness.full_grasp_blockers)

    if args.read_hardware:
        print("[hardware] beginning read-only preflight; no controller or hand write is created")
        _read_franka(config)
        _read_inspire(config)

    if args.strict_full_grasp and not readiness.full_grasp_ready:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
