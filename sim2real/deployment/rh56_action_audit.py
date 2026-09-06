#!/usr/bin/env python3
"""Offline audit of the V94 policy-to-RH56 action mapping.

This module intentionally has no hardware adapter imports.  It reads only the
transferred deployment ZIP, the local commissioning JSON files, and an
optional read-only shadow NPZ.  In particular, it cannot open the RH56 serial
port or write ``ANGLE_SET``.

The audit separates two claims which must not be conflated:

* mathematical alignment means that policy axes, filtering, clipping,
  quantisation, register permutation, and dual-ACK state advancement agree
  with the transferred simulator contract; and
* physical commissioning means that every installed actuator has actually
  been observed moving in the expected direction and range under bounded
  commands.

Only the first claim can be established offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.closed_loop_core import (  # type: ignore[no-redef]
        ClosedLoopCommand,
        ClosedLoopProtocolError,
        ExecutedActionLedger,
        TransactionalV94ActionMapper,
    )
    from sim2real.deployment.bundle import DeployBundle  # type: ignore[no-redef]
    from sim2real.contracts.actions import V94ActionMapper  # type: ignore[no-redef]
    from sim2real.contracts.v94 import (  # type: ignore[no-redef]
        INITIAL_PREVIOUS_ACTION13,
        POLICY_HAND_ORDER,
        Q_HAND_SEMANTIC_CLOSE_RAD,
        REGISTER_HAND_ORDER,
        V94Contract,
    )
    from sim2real.v94_kinematics import (  # type: ignore[no-redef]
        RH56FeedbackMapper,
    )
else:
    from sim2real.closed_loop_core import (
        ClosedLoopCommand,
        ClosedLoopProtocolError,
        ExecutedActionLedger,
        TransactionalV94ActionMapper,
    )
    from .bundle import DeployBundle
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import (
        INITIAL_PREVIOUS_ACTION13,
        POLICY_HAND_ORDER,
        Q_HAND_SEMANTIC_CLOSE_RAD,
        REGISTER_HAND_ORDER,
        V94Contract,
    )
    from sim2real.v94_kinematics import RH56FeedbackMapper


from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE
DEFAULT_DEPLOY_CONFIG = (
    Path(__file__).resolve().parents[1] / "v94_deploy_config.json"
)
RESET_REFERENCE = (
    "alignment/reset_idle_open/initial_observations_and_student_response.npz"
)
CLOSED_REFERENCE = (
    "alignment/student_closed_loop/closed_loop_observation_action_pairs.npz"
)
FLOAT_TOLERANCE = 5.0e-7
MAX_LOCAL_JSON_BYTES = 1024 * 1024
MAX_SHADOW_BYTES = 512 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{name} not found: {source}")
    if source.stat().st_size > MAX_LOCAL_JSON_BYTES:
        raise ValueError(f"{name} is unexpectedly large")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return source, value


def _maximum_error(actual: np.ndarray, expected: np.ndarray, name: str) -> float:
    first = np.asarray(actual)
    second = np.asarray(expected)
    if first.shape != second.shape:
        raise ValueError(f"{name} shape mismatch: {first.shape}!={second.shape}")
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        raise ValueError(f"{name} contains non-finite values")
    return float(
        np.max(np.abs(first.astype(np.float64) - second.astype(np.float64)))
    )


def _policy_to_register_order(values: np.ndarray) -> np.ndarray:
    policy = np.asarray(values)
    if policy.shape[-1] != 6:
        raise ValueError("policy-order values must end in six axes")
    indices = tuple(POLICY_HAND_ORDER.index(name) for name in REGISTER_HAND_ORDER)
    return policy[..., indices]


def _independent_hand_rollout(
    actions13: np.ndarray,
    *,
    initial_target_q_policy_order_rad: np.ndarray,
    q_close_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reimplement the transferred hand equations without using the mapper."""

    actions = np.asarray(actions13, dtype=np.float32)
    previous = np.asarray(initial_target_q_policy_order_rad, dtype=np.float32).copy()
    q_close = np.asarray(q_close_rad, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 13:
        raise ValueError("actions13 must have shape [steps,13]")
    if previous.shape != (6,) or q_close.shape != (6,):
        raise ValueError("RH56 semantic target vectors must have shape (6,)")

    executed_rows = []
    target_rows = []
    register_rows = []
    for raw in actions:
        executed = np.clip(
            raw, np.float32(-1.0), np.float32(1.0)
        ).astype(np.float32, copy=True)
        fraction = np.clip(
            (executed[7:] + np.float32(1.0)) * np.float32(0.5),
            np.float32(0.0),
            np.float32(1.0),
        )
        absolute_raw_target = fraction * q_close
        filtered = (
            np.float32(0.20) * absolute_raw_target
            + np.float32(0.80) * previous
        )
        delta = np.clip(
            filtered - previous, np.float32(-0.05), np.float32(0.05)
        )
        previous = np.clip(
            previous + delta, np.float32(0.0), q_close
        ).astype(np.float32, copy=True)
        register_policy = np.rint(
            np.float32(1000.0)
            * (np.float32(1.0) - previous / q_close)
        ).astype(np.int32)
        executed_rows.append(executed)
        target_rows.append(previous.copy())
        register_rows.append(_policy_to_register_order(register_policy))
    return (
        np.stack(executed_rows),
        np.stack(target_rows),
        np.stack(register_rows).astype(np.int32),
    )


def _audit_reference_stream(
    name: str,
    values: Mapping[str, np.ndarray],
    contract: V94Contract,
) -> dict[str, Any]:
    actions = np.asarray(values["policy_action13"])
    initial_hand = np.asarray(values["rh56_virtual_q_policy_order_rad"])[0]
    independent_executed, independent_targets, independent_registers = (
        _independent_hand_rollout(
            actions,
            initial_target_q_policy_order_rad=initial_hand,
            q_close_rad=contract.q_hand_close_rad,
        )
    )

    mapper = V94ActionMapper(
        initial_arm_target_q_rad=np.asarray(values["franka_measured_q_rad"])[0],
        initial_hand_target_q_policy_order_rad=initial_hand,
        joint_limits_rad=contract.joint_limits_rad,
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    production_targets = []
    production_registers = []
    for index, action in enumerate(actions):
        mapped = mapper.map(
            action,
            measured_q_rad=np.asarray(values["franka_measured_q_rad"])[index],
        )
        production_targets.append(mapped.rh56_target_q_policy_order_rad)
        production_registers.append(mapped.rh56_angle_set_register_order)
    production_targets_array = np.stack(production_targets)
    production_registers_array = np.stack(production_registers)

    stored_targets = np.asarray(values["rh56_target_q_policy_order_rad"])
    stored_registers = np.asarray(values["rh56_angle_set_register_order"])
    errors = {
        "independent_equations_vs_stored_target_max_abs_rad": _maximum_error(
            independent_targets, stored_targets, f"{name} independent target"
        ),
        "production_mapper_vs_stored_target_max_abs_rad": _maximum_error(
            production_targets_array, stored_targets, f"{name} production target"
        ),
        "production_vs_independent_target_max_abs_rad": _maximum_error(
            production_targets_array,
            independent_targets,
            f"{name} production/independent target",
        ),
    }
    if max(errors.values()) > FLOAT_TOLERANCE:
        raise ValueError(f"{name} RH56 float mapping exceeds tolerance: {errors}")
    if not np.array_equal(independent_executed, np.clip(actions, -1.0, 1.0)):
        raise ValueError(f"{name} action clipping mismatch")
    if not np.array_equal(independent_registers, stored_registers):
        raise ValueError(f"{name} independent RH56 register mapping mismatch")
    if not np.array_equal(production_registers_array, stored_registers):
        raise ValueError(f"{name} production RH56 register mapping mismatch")

    previous = np.concatenate(
        [initial_hand[None, :], independent_targets[:-1]], axis=0
    )
    delta = independent_targets.astype(np.float64) - previous.astype(np.float64)
    return {
        "name": name,
        "rows": int(actions.shape[0]),
        **errors,
        "independent_registers_exact": True,
        "production_registers_exact": True,
        "maximum_virtual_target_step_abs_rad_by_policy_axis": np.max(
            np.abs(delta), axis=0
        ).tolist(),
        "stored_register_min_by_manufacturer_axis": np.min(
            stored_registers, axis=0
        ).astype(int).tolist(),
        "stored_register_max_by_manufacturer_axis": np.max(
            stored_registers, axis=0
        ).astype(int).tolist(),
    }


def _audit_axis_permutation(contract: V94Contract) -> list[dict[str, Any]]:
    results = []
    half = np.float32(0.5) * contract.q_hand_close_rad
    register_index_by_name = {
        name: index for index, name in enumerate(REGISTER_HAND_ORDER)
    }
    for policy_index, name in enumerate(POLICY_HAND_ORDER):
        action = np.zeros(13, dtype=np.float32)
        action[7 + policy_index] = np.float32(1.0)
        mapper = V94ActionMapper(
            initial_arm_target_q_rad=np.zeros(7, dtype=np.float32),
            initial_hand_target_q_policy_order_rad=half,
            joint_limits_rad=np.asarray([[-3.0, 3.0]] * 7, dtype=np.float32),
            q_hand_close_rad=contract.q_hand_close_rad,
        )
        mapped = mapper.map(action, measured_q_rad=np.zeros(7, dtype=np.float32))
        changed_q = np.flatnonzero(
            np.abs(mapped.rh56_target_q_policy_order_rad - half) > 1.0e-8
        )
        half_registers = _policy_to_register_order(
            np.rint(
                np.float32(1000.0)
                * (np.float32(1.0) - half / contract.q_hand_close_rad)
            ).astype(np.int32)
        )
        changed_register = np.flatnonzero(
            mapped.rh56_angle_set_register_order != half_registers
        )
        expected_register_index = register_index_by_name[name]
        if changed_q.tolist() != [policy_index]:
            raise ValueError(f"one-hot policy action cross-coupled virtual axis {name}")
        if changed_register.tolist() != [expected_register_index]:
            raise ValueError(f"one-hot policy action reached wrong register for {name}")
        if not (
            mapped.rh56_target_q_policy_order_rad[policy_index]
            > half[policy_index]
            and mapped.rh56_angle_set_register_order[expected_register_index]
            < half_registers[expected_register_index]
        ):
            raise ValueError(f"positive action direction is inverted for {name}")
        results.append(
            {
                "policy_action_index": 7 + policy_index,
                "policy_axis": name,
                "register_index": expected_register_index,
                "register_axis": REGISTER_HAND_ORDER[expected_register_index],
                "semantic_close_rad": float(contract.q_hand_close_rad[policy_index]),
                "positive_action_effect": (
                    "larger virtual semantic q; smaller nominal ANGLE_SET"
                ),
            }
        )
    return results


def _audit_absolute_semantics(contract: V94Contract) -> dict[str, Any]:
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=np.zeros(7, dtype=np.float32),
        initial_hand_target_q_policy_order_rad=np.zeros(6, dtype=np.float32),
        joint_limits_rad=np.asarray([[-3.0, 3.0]] * 7, dtype=np.float32),
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    neutral = np.zeros(13, dtype=np.float32)
    first = mapper.map(neutral, measured_q_rad=np.zeros(7, dtype=np.float32))
    last = first
    for _ in range(199):
        last = mapper.map(neutral, measured_q_rad=np.zeros(7, dtype=np.float32))
    expected = np.float32(0.5) * contract.q_hand_close_rad
    error = _maximum_error(
        last.rh56_target_q_policy_order_rad,
        expected,
        "repeated neutral absolute hand target",
    )
    if error > 2.0e-6:
        raise ValueError("hand action behaves like an increment instead of an absolute target")
    if np.any(first.rh56_target_q_policy_order_rad > np.float32(0.050001)):
        raise ValueError("first hand target step exceeds 0.05 rad")

    clipped_mapper = V94ActionMapper(
        initial_arm_target_q_rad=np.zeros(7, dtype=np.float32),
        initial_hand_target_q_policy_order_rad=np.zeros(6, dtype=np.float32),
        joint_limits_rad=np.asarray([[-3.0, 3.0]] * 7, dtype=np.float32),
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    clipped = clipped_mapper.map(
        np.asarray([0.0] * 7 + [2.0, -2.0, 2.0, -2.0, 2.0, -2.0]),
        measured_q_rad=np.zeros(7, dtype=np.float32),
    )
    expected_clipped = np.asarray(
        [0.0] * 7 + [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        dtype=np.float32,
    )
    if not np.array_equal(clipped.executed_policy_action13, expected_clipped):
        raise ValueError("policy action clipping is not exact")
    return {
        "hand_action_kind": "absolute semantic closure setpoint",
        "filter_alpha": 0.20,
        "maximum_virtual_target_step_rad_per_tick": 0.05,
        "register_quantization": "numpy.rint to nearest integer",
        "repeated_zero_action_converges_to_fraction": 0.5,
        "repeated_zero_action_target_max_abs_error_rad": error,
        "policy_clip_to_minus1_plus1_verified": True,
    }


def _audit_feedback_inverse(contract: V94Contract) -> dict[str, Any]:
    fractions = np.asarray(
        [0.137, 0.421, 0.679, 0.893, 0.254, 0.764], dtype=np.float32
    )
    target = fractions * contract.q_hand_close_rad
    register_policy = np.rint(
        np.float32(1000.0)
        * (np.float32(1.0) - target / contract.q_hand_close_rad)
    ).astype(np.int32)
    register_order = _policy_to_register_order(register_policy).astype(np.int32)
    recovered = RH56FeedbackMapper(
        q_hand_close_rad=contract.q_hand_close_rad
    ).map(register_order)
    error = _maximum_error(
        recovered.q_policy_order_rad, target, "command/feedback virtual q round trip"
    )
    maximum_quantisation_error = float(np.max(contract.q_hand_close_rad) / 2000.0)
    if error > maximum_quantisation_error + 1.0e-7:
        raise ValueError("command and feedback register permutations do not invert")
    return {
        "default_open_endpoint_register": 1000,
        "default_close_endpoint_register": 0,
        "round_trip_max_abs_error_rad": error,
        "round_trip_quantisation_bound_rad": maximum_quantisation_error,
        "register_permutation_inverse_verified": True,
    }


def _audit_dual_ack(contract: V94Contract) -> dict[str, Any]:
    mapper = TransactionalV94ActionMapper(
        initial_arm_target_q_rad=np.zeros(7, dtype=np.float32),
        initial_hand_target_q_policy_order_rad=np.zeros(6, dtype=np.float32),
        joint_limits_rad=np.asarray([[-3.0, 3.0]] * 7, dtype=np.float32),
        control_dt_s=contract.control_dt_s,
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.02)
    action = np.asarray([0.0] * 7 + [1.0] * 6, dtype=np.float32)
    proposal = mapper.propose(
        1, action, measured_q_rad=np.zeros(7, dtype=np.float32)
    )
    command = ClosedLoopCommand(
        sequence=1,
        produced_monotonic_s=1.0,
        observation_realtime_s=2.0,
        previous_executed_action13_used=ledger.previous_executed_action13(),
        raw_policy_action13=proposal.mapped.raw_policy_action13,
        executed_policy_action13=proposal.mapped.executed_policy_action13,
        franka_target_q_rad=proposal.mapped.franka_target_q_rad,
        rh56_angle_set_register_order=(
            proposal.mapped.rh56_angle_set_register_order
        ),
    )
    ledger.stage(command, now_monotonic_s=1.001)
    ledger.acknowledge("rh56", sequence=1, now_monotonic_s=1.002)
    if not np.array_equal(
        ledger.previous_executed_action13(), INITIAL_PREVIOUS_ACTION13
    ):
        raise ValueError("previous action advanced after only RH56 ACK")
    try:
        mapper.commit(proposal, action_ledger=ledger)
    except ClosedLoopProtocolError:
        pass
    else:
        raise ValueError("mapper target committed before dual ACK")
    ledger.acknowledge("franka", sequence=1, now_monotonic_s=1.003)
    if not np.array_equal(ledger.previous_executed_action13(), action):
        raise ValueError("previous action did not advance after dual ACK")
    mapper.commit(proposal, action_ledger=ledger)
    _, committed_hand = mapper.committed_targets()
    if not np.array_equal(
        committed_hand, proposal.mapped.rh56_target_q_policy_order_rad
    ):
        raise ValueError("mapper target did not commit after dual ACK")
    return {
        "previous_action_initial": INITIAL_PREVIOUS_ACTION13.tolist(),
        "single_device_ack_does_not_advance_previous_action": True,
        "single_device_ack_does_not_commit_mapper_target": True,
        "both_franka_and_rh56_ack_required": True,
        "dual_ack_advances_previous_action_and_target": True,
        "scope": "offline software protocol; not real 60 Hz transport evidence",
    }


def _audit_shadow(
    path: str | Path,
    contract: V94Contract,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"shadow NPZ not found: {source}")
    if source.stat().st_size <= 0 or source.stat().st_size > MAX_SHADOW_BYTES:
        raise ValueError("shadow NPZ has an unsafe file size")
    required = (
        "raw_policy_action13",
        "executed_policy_action13",
        "franka_q_rad",
        "rh56_virtual_q_policy_order_rad",
        "rh56_target_q_policy_order_rad",
        "rh56_proposed_angle_set_register_order",
        "proposal_mode",
    )
    try:
        with np.load(source, allow_pickle=False) as archive:
            missing = [name for name in required if name not in archive.files]
            if missing:
                raise ValueError(f"shadow NPZ is missing mapping fields: {missing}")
            values = {name: archive[name].copy() for name in required}
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid no-pickle shadow NPZ: {exc}") from exc
    actions = np.asarray(values["raw_policy_action13"])
    steps = int(actions.shape[0])
    modes = np.asarray(values["proposal_mode"])
    if modes.shape != (steps,) or np.any(modes != "one_step_from_measured_idle"):
        raise ValueError(
            "shadow mapping audit requires one_step_from_measured_idle proposals"
        )
    expected_executed = []
    expected_targets = []
    expected_registers = []
    for index in range(steps):
        mapper = V94ActionMapper(
            initial_arm_target_q_rad=np.asarray(values["franka_q_rad"])[index],
            initial_hand_target_q_policy_order_rad=np.asarray(
                values["rh56_virtual_q_policy_order_rad"]
            )[index],
            joint_limits_rad=contract.joint_limits_rad,
            q_hand_close_rad=contract.q_hand_close_rad,
        )
        mapped = mapper.map(
            actions[index],
            measured_q_rad=np.asarray(values["franka_q_rad"])[index],
        )
        expected_executed.append(mapped.executed_policy_action13)
        expected_targets.append(mapped.rh56_target_q_policy_order_rad)
        expected_registers.append(mapped.rh56_angle_set_register_order)
    executed_error = _maximum_error(
        np.stack(expected_executed),
        np.asarray(values["executed_policy_action13"]),
        "shadow executed action",
    )
    target_error = _maximum_error(
        np.stack(expected_targets),
        np.asarray(values["rh56_target_q_policy_order_rad"]),
        "shadow RH56 target",
    )
    registers = np.stack(expected_registers)
    register_exact = np.array_equal(
        registers, np.asarray(values["rh56_proposed_angle_set_register_order"])
    )
    if max(executed_error, target_error) > FLOAT_TOLERANCE or not register_exact:
        raise ValueError("shadow action mapping does not replay exactly")
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "rows": steps,
        "proposal_mode": "one_step_from_measured_idle",
        "executed_action_max_abs_error": executed_error,
        "virtual_target_max_abs_error_rad": target_error,
        "register_targets_exact": register_exact,
        "register_min_by_manufacturer_axis": np.min(registers, axis=0)
        .astype(int)
        .tolist(),
        "register_max_by_manufacturer_axis": np.max(registers, axis=0)
        .astype(int)
        .tolist(),
        "scope": (
            "read-only one-step proposals; not accumulated physical closed-loop "
            "tracking evidence"
        ),
    }


def audit_v94_rh56_action_mapping(
    *,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    deploy_config_path: str | Path = DEFAULT_DEPLOY_CONFIG,
    commissioning_profile_path: Optional[str | Path] = None,
    shadow_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return a fail-closed, JSON-serialisable, hardware-free audit report."""

    bundle = DeployBundle(bundle_path)
    bundle_verification = bundle.verify()
    contract = V94Contract.from_bundle(bundle)
    manifest_alignment = bundle.manifest.get("hardware_alignment")
    if not isinstance(manifest_alignment, Mapping):
        raise ValueError("bundle hardware_alignment is missing")
    manifest_policy_order = tuple(manifest_alignment.get("rh56_policy_order", ()))
    manifest_register_order = tuple(
        manifest_alignment.get("rh56_register_order", ())
    )
    if manifest_policy_order != POLICY_HAND_ORDER:
        raise ValueError("bundle and runtime RH56 policy orders disagree")
    if manifest_register_order != REGISTER_HAND_ORDER:
        raise ValueError("bundle and runtime RH56 register orders disagree")
    model_output = bundle.manifest.get("model", {}).get("output", {})
    if not isinstance(model_output, Mapping) or model_output.get("rh56_7_12") != (
        "absolute six-motor target in policy order"
    ):
        raise ValueError("bundle does not declare absolute RH56 action semantics")

    reset = bundle.load_npz(RESET_REFERENCE)
    closed = bundle.load_npz(CLOSED_REFERENCE)
    stream_reports = [
        _audit_reference_stream("reset_idle_open", reset, contract),
        _audit_reference_stream("student_closed_loop", closed, contract),
    ]
    closed_previous = np.asarray(closed["proprio_history67"])[1:, -1, 54:67]
    closed_prior_action = np.asarray(closed["policy_action13"][:-1])
    if not np.array_equal(closed_previous, closed_prior_action):
        raise ValueError("simulator previous-action observation is not prior action")
    reset_previous = np.asarray(reset["proprio67"])[0, 54:67]
    if not np.array_equal(reset_previous, INITIAL_PREVIOUS_ACTION13):
        raise ValueError("reset previous-action seed disagrees with V94 contract")

    deploy_path, deploy = _load_json(deploy_config_path, "deploy config")
    raw_profile = deploy.get("commissioning_profile")
    if not isinstance(raw_profile, str) or not raw_profile.strip():
        raise ValueError("deploy config commissioning_profile is missing")
    profile_path = (
        (deploy_path.parent / raw_profile).resolve()
        if commissioning_profile_path is None
        else Path(commissioning_profile_path).expanduser().resolve()
    )
    profile_path, profile = _load_json(profile_path, "commissioning profile")
    inspire = profile.get("inspire")
    execution = deploy.get("execution")
    if not isinstance(inspire, Mapping) or not isinstance(execution, Mapping):
        raise ValueError("commissioning RH56/execution sections are missing")
    thumb_range = inspire.get("thumb_rotate_validated_realtime_range")
    if (
        not isinstance(thumb_range, list)
        or len(thumb_range) != 2
        or any(isinstance(item, bool) for item in thumb_range)
    ):
        raise ValueError("thumb rotation commissioned range is invalid")
    thumb_range_int = [int(item) for item in thumb_range]
    six_axis_commissioned = inspire.get("six_axis_coupled_closure_commissioned")
    policy_motion_commissioned = execution.get(
        "rh56_six_axis_policy_motion_commissioned"
    )
    if not isinstance(six_axis_commissioned, bool) or not isinstance(
        policy_motion_commissioned, bool
    ):
        raise ValueError("RH56 commissioning flags must be boolean")
    physical_mapping_confirmed = bool(
        six_axis_commissioned
        and policy_motion_commissioned
        and thumb_range_int[0] <= 0
        and thumb_range_int[1] >= 1000
    )

    shadow = None if shadow_path is None else _audit_shadow(shadow_path, contract)
    axis_mapping = _audit_axis_permutation(contract)
    absolute_semantics = _audit_absolute_semantics(contract)
    feedback_inverse = _audit_feedback_inverse(contract)
    dual_ack = _audit_dual_ack(contract)

    return {
        "audit": "v94_rh56_action_mapping_offline_v1",
        "result": "PASS",
        "offline_only": True,
        "hardware_access": False,
        "hardware_writes": False,
        "motion_authorized": False,
        "bundle": {
            "path": str(bundle.path),
            "sha256": _sha256(bundle.path),
            "checked_manifest_hashes": bundle_verification.checked_files,
            "checkpoint_sha256": bundle_verification.primary_checkpoint_sha256,
            "status": contract.bundle_status,
        },
        "semantic_contract": {
            "action_indices": "7:13",
            "policy_order": list(POLICY_HAND_ORDER),
            "register_order": list(REGISTER_HAND_ORDER),
            "q_semantic_close_rad_policy_order": contract.q_hand_close_rad.tolist(),
            "minus_one_meaning": "fully open semantic target",
            "zero_meaning": "half-closed semantic target",
            "plus_one_meaning": "fully closed semantic target",
            "nominal_register_direction": "1000=open, 0=closed",
            "absolute_not_incremental": True,
        },
        "axis_mapping": axis_mapping,
        "absolute_filter_clip_rounding": absolute_semantics,
        "feedback_command_inverse": feedback_inverse,
        "simulator_reference_streams": stream_reports,
        "simulator_previous_action_chain_exact": True,
        "dual_ack_protocol": dual_ack,
        "optional_real_shadow": shadow,
        "mathematical_mapping_confirmed": True,
        "physical_mapping_confirmed": physical_mapping_confirmed,
        "physical_commissioning": {
            "profile_path": str(profile_path),
            "profile_sha256": _sha256(profile_path),
            "profile_mode": profile.get("mode"),
            "six_axis_coupled_closure_commissioned": six_axis_commissioned,
            "six_axis_policy_motion_commissioned": policy_motion_commissioned,
            "thumb_rotate_validated_realtime_range": thumb_range_int,
            "microprobe_required_axes": list(REGISTER_HAND_ORDER),
            "especially_unresolved": (
                "thumb_rotation physical direction and range outside the "
                f"commissioned interval {thumb_range_int[0]}..{thumb_range_int[1]}"
            ),
            "required_observation": (
                "one axis at a time, unloaded and low speed: command/readback "
                "identity, ANGLE_ACT direction, motion range, tracking, current, "
                "temperature, status, and verified all-six disable"
            ),
        },
        "conclusion": (
            "The software mapping is exact against both transferred simulator "
            "streams and the optional real shadow proposal log.  Physical RH56 "
            "direction/range is not yet fully commissioned and cannot be inferred "
            "from an offline replay."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V94 policy-to-RH56 semantics and register mapping without "
            "opening any hardware device."
        )
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--deploy-config", type=Path, default=DEFAULT_DEPLOY_CONFIG
    )
    parser.add_argument("--shadow", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_v94_rh56_action_mapping(
            bundle_path=args.bundle,
            deploy_config_path=args.deploy_config,
            shadow_path=args.shadow,
        )
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print("V94 RH56 action mapping offline audit: PASS")
            print(json.dumps(report, indent=2, sort_keys=True))
            print("hardware: untouched (offline audit only)")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(
            f"V94 RH56 action mapping offline audit: FAIL: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
