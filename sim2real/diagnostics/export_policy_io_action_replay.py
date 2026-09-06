#!/usr/bin/env python3
"""Export a successful simulator policy-I/O trace as an exact-target replay ZIP.

The real replay runtime consumes a small, pickle-free bundle rather than the
full simulator diagnostic archive.  This exporter deliberately retains both
the normalized policy action (for the previous-action observation chain) and
the simulator's exact Franka/RH56 actuator targets (for trajectory fidelity).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Optional, Sequence
import zipfile

import numpy as np

from sim2real.action_replay import CANONICAL_ACTION_ORDER, load_replay_actions


MAX_TARGET_TICK_DELTA_RAD = 0.020
MAX_INITIAL_TARGET_ERROR_RAD = 0.010


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _required_array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    *,
    shape_tail: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if name not in archive:
        raise ValueError(f"simulator policy-I/O is missing {name}")
    value = np.asarray(archive[name], dtype=dtype)
    if value.ndim != 1 + len(shape_tail) or value.shape[1:] != shape_tail:
        raise ValueError(
            f"{name} must have shape [K,{','.join(map(str, shape_tail))}]"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(value)


def export_policy_io_action_replay(source: Path, output: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"simulator policy-I/O is missing: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite replay bundle: {output}")

    source_bytes = source.read_bytes()
    source_sha256 = _sha256(source_bytes)
    try:
        with np.load(io.BytesIO(source_bytes), allow_pickle=False) as archive:
            actions = _required_array(
                archive,
                "output_action_sent_to_env",
                shape_tail=(13,),
                dtype=np.dtype(np.float32),
            )
            franka_targets = _required_array(
                archive,
                "sim_arm_joint_target_rad",
                shape_tail=(7,),
                dtype=np.dtype(np.float32),
            )
            rh56_targets = _required_array(
                archive,
                "rh56_angle_set_register_order",
                shape_tail=(6,),
                dtype=np.dtype(np.float64),
            )
            arm_position = _required_array(
                archive,
                "sim_arm_joint_position_rad",
                shape_tail=(7,),
                dtype=np.dtype(np.float32),
            )
            time_s = np.asarray(archive["time_s"], dtype=np.float64)
            success = np.asarray(archive["sim_success"], dtype=np.bool_)
            stable_hold = np.asarray(archive["sim_stable_hold"], dtype=np.bool_)
            checkpoint_sha256 = ""
            if "constant__checkpoint_sha256" in archive:
                checkpoint_sha256 = str(
                    np.asarray(archive["constant__checkpoint_sha256"]).reshape(())
                )
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid simulator policy-I/O NPZ: {exc}") from exc

    count = int(actions.shape[0])
    if count < 1 or any(
        value.shape[0] != count
        for value in (franka_targets, rh56_targets, arm_position)
    ):
        raise ValueError("simulator replay arrays have inconsistent frame counts")
    if time_s.shape != (count,) or success.shape != (count,) or stable_hold.shape != (
        count,
    ):
        raise ValueError("time/success/stable-hold arrays have inconsistent shapes")
    if not np.all(np.isfinite(time_s)) or not np.allclose(
        np.diff(time_s), 0.05, atol=1.0e-6, rtol=0.0
    ):
        raise ValueError("simulator replay must be a uniform 20 Hz sequence")
    if not bool(success[-1]) or not bool(np.any(stable_hold)):
        raise ValueError("simulator replay is not a successful stable-hold trace")
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        raise ValueError("simulator actions are outside normalized [-1,1]")
    if not np.all(rh56_targets == np.rint(rh56_targets)) or np.any(
        (rh56_targets < 0.0) | (rh56_targets > 1000.0)
    ):
        raise ValueError("simulator RH56 targets are not integer registers in [0,1000]")
    maximum_tick_delta = float(
        np.max(np.abs(np.diff(franka_targets.astype(np.float64), axis=0)), initial=0.0)
    )
    if maximum_tick_delta > MAX_TARGET_TICK_DELTA_RAD + 1.0e-9:
        raise ValueError(
            "simulator Franka target exceeds the supervised 0.020 rad/tick bound: "
            f"{maximum_tick_delta:.9f}"
        )
    initial_target_error = float(
        np.max(
            np.abs(
                franka_targets[0].astype(np.float64)
                - arm_position[0].astype(np.float64)
            )
        )
    )
    if initial_target_error > MAX_INITIAL_TARGET_ERROR_RAD:
        raise ValueError(
            "simulator first target differs from its initial arm state by "
            f"{initial_target_error:.9f} rad"
        )

    payload = io.BytesIO()
    np.savez(
        payload,
        time_s=np.ascontiguousarray(time_s, dtype=np.float64),
        policy_action=actions,
        franka_joint_target_rad=franka_targets,
        inspire_angle_set_register_order=np.asarray(
            np.rint(rh56_targets), dtype=np.int32
        ),
    )
    payload_bytes = payload.getvalue()
    metadata = {
        "schema": "sim_policy_io_exact_target_replay_v1",
        "control_hz": 20.0,
        "control_dt_s": 0.05,
        "frames": count,
        "action_contract": (
            "simulator normalized policy action plus exact Franka/RH56 targets"
        ),
        "inspire_policy_order": list(CANONICAL_ACTION_ORDER[7:]),
        "inspire_register_order": [
            "little",
            "ring",
            "middle",
            "index",
            "thumb_bending",
            "thumb_rotation",
        ],
        "recommended_replay_fields": {
            "franka": "franka_joint_target_rad",
            "inspire": "inspire_angle_set_register_order",
        },
        "source_policy_io": str(source),
        "source_policy_io_sha256": source_sha256,
        "source_checkpoint_sha256": checkpoint_sha256 or None,
        "source_final_success": True,
        "source_first_stable_hold_frame": int(np.flatnonzero(stable_hold)[0]),
        "maximum_franka_target_tick_delta_rad": maximum_tick_delta,
        "initial_franka_target_error_rad": initial_target_error,
        "payload_npz_sha256": _sha256(payload_bytes),
    }
    metadata_bytes = json.dumps(
        metadata, indent=2, sort_keys=True, ensure_ascii=True
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("replay/metadata.json", metadata_bytes)
            bundle.writestr("replay/data.npz", payload_bytes)
        sequence = load_replay_actions(output, expected_policy_rate_hz=20)
        if sequence.action_count != count:
            raise RuntimeError("written replay bundle failed frame-count verification")
    except BaseException:
        output.unlink(missing_ok=True)
        raise

    return {
        "output": str(output),
        "output_sha256": _sha256(output.read_bytes()),
        "source": str(source),
        "source_sha256": source_sha256,
        "frames": count,
        "first_stable_hold_frame": int(np.flatnonzero(stable_hold)[0]),
        "maximum_franka_target_tick_delta_rad": maximum_tick_delta,
        "initial_franka_target_error_rad": initial_target_error,
        "exact_franka_targets": True,
        "exact_rh56_targets": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one successful simulator policy_io.npz as a validated "
            "20 Hz exact-target real replay bundle."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        print(
            json.dumps(
                export_policy_io_action_replay(args.source, args.output),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"replay export: REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
