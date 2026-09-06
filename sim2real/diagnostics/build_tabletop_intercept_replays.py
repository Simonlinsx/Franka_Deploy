#!/usr/bin/env python3
"""Build table-clear, perception-triggered tabletop demo replay bundles."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable
import zipfile

import numpy as np

from sim2real.action_replay import CANONICAL_ACTION_ORDER, load_replay_actions
from sim2real.deployment.bundle import DeployBundle
from motion_planning.online_tabletop import (
    cartesian_joint_correction,
    panda_T_base_policy_palm,
    pose_preserving_joint_target,
)
from sim2real.contracts.v94 import V94Contract
from sim2real.v94_kinematics import RH56FingertipKinematics

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    WORKSPACE_ROOT
    / "dexgrasp/runs/tabletop-v364-cylinder-posy-low-20260818-190615_policy_io.npz"
)
from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE
DEFAULT_SPHERE_SOURCE = (
    WORKSPACE_ROOT
    / "data/test_fixtures/inspire_v205_xyz20hz_static_sphere_replay_20260727"
    / "static_sphere_success_replay_20hz.npz"
)
DEFAULT_SPHERE_TRACE = (
    WORKSPACE_ROOT
    / "data/test_fixtures/inspire_v205_xyz20hz_static_sphere_replay_20260727"
    / "success_attempt_005_env_001_idx_000.trace.json"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "dexgrasp/runs/replay_inputs/tabletop_online_planner_v4"
)
REGISTER_ORDER = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bending",
    "thumb_rotation",
)


def _rotation_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


_PANDA_ORIGINS = (
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
)


def _T_base_palm(q_rad: np.ndarray, contract: V94Contract) -> np.ndarray:
    return panda_T_base_policy_palm(q_rad, contract.T_flange_policy_palm)


def _densify_joint_path(
    waypoints: Iterable[np.ndarray], *, maximum_tick_delta_rad: float
) -> np.ndarray:
    source = [np.asarray(item, dtype=np.float64) for item in waypoints]
    if not source or any(item.shape != (7,) for item in source):
        raise ValueError("joint path must contain [7] waypoints")
    output = [source[0].copy()]
    for start, stop in zip(source, source[1:]):
        steps = max(
            1,
            int(
                np.ceil(
                    float(np.max(np.abs(stop - start)))
                    / maximum_tick_delta_rad
                )
            ),
        )
        for index in range(1, steps + 1):
            output.append(start + (stop - start) * (index / steps))
    return np.asarray(output, dtype=np.float32)


def _preshape(count: int, target: np.ndarray) -> np.ndarray:
    if count < 2:
        raise ValueError("preposition path is too short")
    start = np.full(6, 1000, dtype=np.float64)
    target_value = np.asarray(target, dtype=np.float64)
    transition = min(40, count - 1)
    output = []
    for index in range(count):
        fraction = min(1.0, index / transition)
        output.append(np.rint(start + fraction * (target_value - start)))
    return np.asarray(output, dtype=np.int32)


def _minimum_jerk_path(
    start: np.ndarray,
    stop: np.ndarray,
    *,
    intervals: int,
) -> np.ndarray:
    if intervals < 1:
        raise ValueError("minimum-jerk path requires at least one interval")
    source = np.asarray(start, dtype=np.float64)
    target = np.asarray(stop, dtype=np.float64)
    if source.shape != target.shape or not np.all(np.isfinite([source, target])):
        raise ValueError("minimum-jerk endpoints must have equal finite shapes")
    output = []
    for index in range(intervals + 1):
        fraction = index / intervals
        blend = fraction**3 * (
            10.0 - 15.0 * fraction + 6.0 * fraction**2
        )
        output.append(source + blend * (target - source))
    return np.asarray(output, dtype=np.float64)


def _rotation_geodesic_midpoint(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Return the SO(3) midpoint without adding a SciPy runtime dependency."""

    source = np.asarray(first, dtype=np.float64)
    target = np.asarray(second, dtype=np.float64)
    if source.shape != (3, 3) or target.shape != (3, 3) or not (
        np.all(np.isfinite(source)) and np.all(np.isfinite(target))
    ):
        raise ValueError("rotation midpoint inputs must be finite [3,3]")
    # For rotations separated by less than pi, the polar factor of R0 + R1
    # is their unique geodesic midpoint.  The determinant repair keeps the
    # result in SO(3) if numerical SVD signs differ.
    left, _singular, right = np.linalg.svd(source + target)
    midpoint = left @ right
    if np.linalg.det(midpoint) < 0.0:
        left[:, -1] *= -1.0
        midpoint = left @ right
    return midpoint


def _source_object_center_base(
    *,
    points_history: np.ndarray,
    valid_history: np.ndarray,
    source_q: np.ndarray,
    source_index: int,
    contract: V94Contract,
) -> np.ndarray:
    points = np.asarray(points_history[source_index, -1], dtype=np.float64)
    valid = np.asarray(valid_history[source_index, -1]) >= 0.5
    if np.count_nonzero(valid) < 16:
        raise ValueError("source catch frame has too few valid object points")
    center_palm = np.mean(points[valid, :3], axis=0)
    homogeneous = np.concatenate([center_palm, [1.0]])
    return (_T_base_palm(source_q[source_index], contract) @ homogeneous)[:3]


def _audit_plan(
    *,
    q_targets: np.ndarray,
    hand_targets: np.ndarray,
    contract: V94Contract,
    tabletop_plane: np.ndarray,
    minimum_fingertip_clearance_m: float,
    maximum_target_tick_delta_rad: float = 0.015,
) -> dict[str, object]:
    if q_targets.shape[0] != hand_targets.shape[0]:
        raise ValueError("Franka/RH56 target counts differ")
    first_error = float(np.max(np.abs(q_targets[0] - contract.q_home_rad)))
    maximum_tick = float(np.max(np.abs(np.diff(q_targets, axis=0)), initial=0.0))
    lower = contract.joint_limits_rad[:, 0] + 0.05
    upper = contract.joint_limits_rad[:, 1] - 0.05
    if first_error > 1.0e-6:
        raise ValueError("planned first target is not exact q_home")
    if maximum_tick > float(maximum_target_tick_delta_rad) + 1.0e-7:
        raise ValueError(
            "planned Franka target exceeds the target-stream envelope"
        )
    if np.any(q_targets < lower) or np.any(q_targets > upper):
        raise ValueError("planned Franka target violates joint-limit margin")
    fingertip_model = RH56FingertipKinematics(contract)
    normal = tabletop_plane[:3]
    normal_norm = float(np.linalg.norm(normal))
    if not np.isclose(normal_norm, 1.0, atol=1.0e-3, rtol=0.0):
        raise ValueError("tabletop plane normal must be unit length")
    minimum_tip = float("inf")
    minimum_tip_index = -1
    minimum_palm = float("inf")
    for index, (q, hand) in enumerate(zip(q_targets, hand_targets)):
        palm = _T_base_palm(q, contract)
        tips = fingertip_model.positions_base(
            angle_act_register_order=hand,
            T_base_palm=palm,
        )
        tip_clearance = float(np.min(tips @ normal + tabletop_plane[3]))
        if tip_clearance < minimum_tip:
            minimum_tip = tip_clearance
            minimum_tip_index = int(index)
        minimum_palm = min(
            minimum_palm,
            float(np.dot(palm[:3, 3], normal) + tabletop_plane[3]),
        )
    if minimum_tip < minimum_fingertip_clearance_m:
        raise ValueError(
            "planned fingertip/table clearance is unsafe: "
            f"{minimum_tip:.6f}m < {minimum_fingertip_clearance_m:.6f}m "
            f"at index={minimum_tip_index}"
        )
    if minimum_palm < 0.12:
        raise ValueError("planned palm/table clearance is below 0.12m")
    return {
        "first_target_q_home_error_rad": first_error,
        "maximum_franka_tick_delta_rad": maximum_tick,
        "maximum_target_tick_delta_envelope_rad": float(
            maximum_target_tick_delta_rad
        ),
        "joint_limit_margin_rad": 0.05,
        "minimum_fingertip_table_clearance_m": minimum_tip,
        "required_fingertip_table_clearance_m": minimum_fingertip_clearance_m,
        "minimum_palm_table_clearance_m": minimum_palm,
        "tabletop_plane_base": tabletop_plane.tolist(),
    }


def _scenario_specs():
    for shape, text in (
        ("cylinder", "large pink foam cylinder"),
        ("sphere", "small ball"),
    ):
        for direction_name, direction in (("posy", 1.0), ("negy", -1.0)):
            for speed_name, speed_range in (
                ("low", (0.02, 0.20)),
                ("high", (0.20, 0.40)),
            ):
                yield {
                    "name": f"{shape}_{direction_name}_{speed_name}",
                    "object_shape": shape,
                    "object_text": text,
                    "direction": direction,
                    "speed_range": speed_range,
                    "collision_replan": False,
                    "grasp_vertical_offset_m": (
                        -0.035 if shape == "sphere" else -0.030
                    ),
                    "max_linear_prediction_s": 0.40,
                    "closure_trigger_ttc_s": (
                        0.75 if shape == "sphere" else 0.55
                    ),
                }
    yield {
        "name": "sphere_posy_board_collision",
        "object_shape": "sphere",
        "object_text": "small ball",
        "direction": 1.0,
        "speed_range": (0.02, 0.40),
        "collision_replan": True,
        "grasp_vertical_offset_m": -0.035,
        # Replan against the post-impact velocity every frame, but send the
        # arm toward a downstream interception point instead of making it
        # chase a point that stays only just ahead of the ball.  Workspace,
        # joint, geometry, and capture gates still bound every proposal.
        "max_linear_prediction_s": 1.00,
        # The board-collision run changes direction close to the hand.  The
        # commissioned RH56 trace needs roughly two 20 Hz ticks before the
        # fingers visibly respond, so lead this one scenario by 100 ms.  Keep
        # the ordinary straight-line cases at 0.55 s.
        "closure_trigger_ttc_s": 0.75,
    }


def build_bundles(
    *,
    source_path: Path,
    deploy_bundle_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    deploy = DeployBundle(deploy_bundle_path)
    deploy.verify()
    contract = V94Contract.from_bundle(deploy).with_runtime_policy_rate_hz(20)
    scene = deploy.read_json("calibration/scene_manifest.json")
    tabletop_plane = np.asarray(
        scene["scene"]["tabletop"]["plane_in_robot_base"],
        dtype=np.float64,
    )
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as archive:
        q_source = archive["real_measured_franka_q_rad"].copy()
        q_target_source = archive["real_franka_target_q_rad"].copy()
        hand_source = archive["rh56_angle_set_register_order"].copy()
        action_source = archive["output_action_sent_to_env"].copy()
        points = archive["input_pointcloud_history_metric"].copy()
        valid = archive["input_pointcloud_valid_history"].copy()
    preposition_source_index = 47
    catch_source_index = 55
    lift_source_end_index = 60
    q_waypoints = [contract.q_home_rad.astype(np.float64)]
    q_waypoints.extend(q_source[1 : preposition_source_index + 1])
    preposition_q = _densify_joint_path(
        q_waypoints,
        maximum_tick_delta_rad=0.015,
    )
    preposition_hand = _preshape(
        len(preposition_q),
        np.clip(hand_source[46], 0, 1000),
    )
    closure_hand = np.clip(hand_source[47:57], 0, 1000).astype(np.int32)
    closure_q = np.repeat(preposition_q[-1][None, :], len(closure_hand), axis=0)
    lift_q = _densify_joint_path(
        q_source[preposition_source_index : lift_source_end_index + 1],
        maximum_tick_delta_rad=0.015,
    )[1:]
    grasp_hand = np.clip(hand_source[56], 0, 1000).astype(np.int32)
    lift_hand = np.repeat(grasp_hand[None, :], len(lift_q), axis=0)
    hold_q = np.repeat(lift_q[-1][None, :], 12, axis=0)
    hold_hand = np.repeat(grasp_hand[None, :], 12, axis=0)
    q_targets = np.concatenate(
        [preposition_q, closure_q, lift_q, hold_q], axis=0
    ).astype(np.float32)
    hand_targets = np.concatenate(
        [preposition_hand, closure_hand, lift_hand, hold_hand], axis=0
    ).astype(np.int32)
    preposition_end_index = len(preposition_q) - 1
    safety = _audit_plan(
        q_targets=q_targets,
        hand_targets=hand_targets,
        contract=contract,
        tabletop_plane=tabletop_plane,
        minimum_fingertip_clearance_m=0.020,
    )
    catch_center_base = _source_object_center_base(
        points_history=points,
        valid_history=valid,
        source_q=q_source,
        source_index=catch_source_index,
        contract=contract,
    )
    # Build a genuine interception primitive from the successful grasp pose,
    # not from the source policy's approach timing.  The source recording was
    # an assisted stationary pickup: its grasp pose and RH56 closure are useful,
    # while its approach phase is not a moving-object demonstration.
    online_template_start_index = 4
    # The commissioned Franka target speed is 0.5 rad/s (0.025 rad/tick).
    # The former 30-tick minimum-jerk approach asked the hand to close with a
    # measured/target error of 0.436 rad.  Fifty approach intervals reduce the
    # conservative 0.5 rad/s actuator-model error below 0.02 rad *before* the
    # first closure target.  The following 14-tick visual-servo closure lets
    # the physical RH56 finish closing before lift.
    online_template_catch_index = 54
    online_closure_end_index = 67
    online_lift_end_index = 79
    online_frame_count = 98
    nominal_table_lift_m = 0.012
    source_reference_grasp_q = q_target_source[preposition_source_index].astype(
        np.float64
    ).copy()
    if not np.all(np.isfinite(source_reference_grasp_q)):
        raise ValueError("source grasp target contains NaN/inf")
    source_reference_grasp_q += cartesian_joint_correction(
        source_reference_grasp_q,
        np.asarray([0.0, 0.0, nominal_table_lift_m]),
        T_flange_policy_palm=contract.T_flange_policy_palm,
        damping=0.04,
    )
    # The source is a successful static grasp, so its hand closure remains a
    # useful primitive.  Its far wrist pose is not copied verbatim: use a
    # shorter, table-clear palm target midway between q_home orientation and
    # the source orientation.  Runtime still recomputes XY interception every
    # 50 ms; this is only the nominal dynamic approach, not a preposition.
    home_palm = _T_base_palm(contract.q_home_rad, contract)
    source_reference_palm = _T_base_palm(source_reference_grasp_q, contract)
    dynamic_reference_rotation = _rotation_geodesic_midpoint(
        home_palm[:3, :3], source_reference_palm[:3, :3]
    )
    dynamic_reference_translation_m = np.asarray(
        [0.010, -0.010, 0.030], dtype=np.float64
    )
    reference_grasp_q = pose_preserving_joint_target(
        source_reference_grasp_q,
        dynamic_reference_translation_m,
        T_flange_policy_palm=contract.T_flange_policy_palm,
        joint_limits_rad=contract.joint_limits_rad,
        joint_limit_margin_rad=0.050,
        damping=0.04,
        target_rotation_base=dynamic_reference_rotation,
    )
    grasp_lift_m = 0.120
    reference_lift_q = pose_preserving_joint_target(
        reference_grasp_q,
        np.asarray([0.0, 0.0, grasp_lift_m]),
        T_flange_policy_palm=contract.T_flange_policy_palm,
        joint_limits_rad=contract.joint_limits_rad,
        joint_limit_margin_rad=0.050,
        damping=0.04,
    )
    approach_q = _minimum_jerk_path(
        contract.q_home_rad,
        reference_grasp_q,
        intervals=(
            online_template_catch_index - online_template_start_index
        ),
    )
    lift_q = _minimum_jerk_path(
        reference_grasp_q,
        reference_lift_q,
        intervals=(online_lift_end_index - online_closure_end_index),
    )
    online_q = np.empty((online_frame_count, 7), dtype=np.float64)
    online_q[:online_template_start_index] = contract.q_home_rad
    online_q[
        online_template_start_index : online_template_catch_index + 1
    ] = approach_q
    online_q[
        online_template_catch_index + 1 : online_closure_end_index + 1
    ] = reference_grasp_q
    online_q[
        online_closure_end_index + 1 : online_lift_end_index + 1
    ] = lift_q[1:]
    online_q[online_lift_end_index + 1 :] = reference_lift_q
    online_q = online_q.astype(np.float32)

    pregrasp_hand = np.clip(hand_source[46], 0, 1000).astype(np.int32)
    online_hand = _preshape(
        online_template_catch_index,
        pregrasp_hand,
    )
    source_closure_hand = np.clip(
        hand_source[47:57], 0, 1000
    ).astype(np.int32)
    # Four held targets let the physical RH56 finish the successful source
    # closure before the arm begins to lift.
    closure_hand = np.concatenate(
        [
            source_closure_hand,
            np.repeat(source_closure_hand[-1][None, :], 4, axis=0),
        ],
        axis=0,
    )
    if len(closure_hand) != (
        online_closure_end_index - online_template_catch_index + 1
    ):
        raise ValueError("source RH56 closure does not match the online phase")
    closed_hand = closure_hand[-1]
    online_hand = np.concatenate(
        [
            online_hand,
            closure_hand,
            np.repeat(
                closed_hand[None, :],
                online_frame_count - online_closure_end_index - 1,
                axis=0,
            ),
        ],
        axis=0,
    ).astype(np.int32)
    if len(online_hand) != online_frame_count or np.any(online_hand < 0) or (
        np.any(online_hand > 1000)
    ):
        raise ValueError("online RH56 target stream is invalid")
    online_actions = action_source[:online_frame_count].astype(np.float32).copy()
    if len(online_actions) < online_frame_count:
        online_actions = np.concatenate(
            [
                online_actions,
                np.repeat(
                    action_source[-1][None, :],
                    online_frame_count - len(online_actions),
                    axis=0,
                ).astype(np.float32),
            ],
            axis=0,
        )
    online_safety = _audit_plan(
        q_targets=online_q,
        hand_targets=online_hand,
        contract=contract,
        tabletop_plane=tabletop_plane,
        minimum_fingertip_clearance_m=0.020,
        maximum_target_tick_delta_rad=0.10,
    )
    online_reference_center_base = _source_object_center_base(
        points_history=points,
        valid_history=valid,
        source_q=q_source,
        source_index=preposition_source_index,
        contract=contract,
    )

    # The real cylinder recording supplies a proven hardware closure, but its
    # long-object palm offset puts a small sphere at the outside of the hand.
    # For sphere scenarios, reuse only the invariant grasp geometry and RH56
    # shape from a successful simulation with the same q_home and actuator
    # mapping.  The arm trajectory remains a fresh 20 Hz visual intercept.
    sphere_source_bytes = DEFAULT_SPHERE_SOURCE.read_bytes()
    sphere_trace_bytes = DEFAULT_SPHERE_TRACE.read_bytes()
    with np.load(io.BytesIO(sphere_source_bytes), allow_pickle=False) as sphere:
        sphere_q = sphere["franka_joint_target_rad"].copy()
        sphere_hand_source = sphere[
            "inspire_angle_set_register_order"
        ].copy()
    sphere_trace_payload = json.loads(sphere_trace_bytes)
    sphere_trace = sphere_trace_payload.get("trace")
    if not isinstance(sphere_trace, list) or len(sphere_trace) != len(sphere_q):
        raise ValueError("successful sphere trace does not match its actuator replay")
    if not np.allclose(
        sphere_q[0], contract.q_home_rad, atol=1.0e-6, rtol=0.0
    ):
        raise ValueError("successful sphere replay uses a different q_home")
    sphere_contact_index = next(
        (
            index
            for index, sample in enumerate(sphere_trace)
            if bool(sample.get("strict_true_grasp"))
            and not bool(sample.get("lifted"))
        ),
        None,
    )
    if sphere_contact_index is None or sphere_contact_index < 2:
        raise ValueError("successful sphere replay has no pre-lift grasp evidence")
    sphere_contact_q = np.asarray(
        sphere_q[sphere_contact_index], dtype=np.float64
    )
    sphere_contact_palm = _T_base_palm(sphere_contact_q, contract)
    sphere_object_from_palm_base = np.asarray(
        sphere_trace[sphere_contact_index].get("object_minus_palm"),
        dtype=np.float64,
    )
    if sphere_object_from_palm_base.shape != (3,) or not np.all(
        np.isfinite(sphere_object_from_palm_base)
    ):
        raise ValueError("successful sphere contact geometry is malformed")
    sphere_object_in_palm = (
        sphere_contact_palm[:3, :3].T @ sphere_object_from_palm_base
    )
    # The simulator contact grazed the table.  Raise the whole demonstrated
    # grasp by the minimum audited 20 mm reserve; never lower the fingertips
    # to reproduce simulator-only contact.
    sphere_reference_palm_position = (
        online_reference_center_base
        - sphere_contact_palm[:3, :3] @ sphere_object_in_palm
        + np.asarray([0.0, 0.0, 0.020])
    )
    sphere_reference_grasp_q = pose_preserving_joint_target(
        sphere_contact_q,
        sphere_reference_palm_position - sphere_contact_palm[:3, 3],
        T_flange_policy_palm=contract.T_flange_policy_palm,
        joint_limits_rad=contract.joint_limits_rad,
        joint_limit_margin_rad=0.050,
        damping=0.04,
        target_rotation_base=sphere_contact_palm[:3, :3],
    )
    sphere_reference_palm = _T_base_palm(
        sphere_reference_grasp_q, contract
    )
    sphere_capture_object_offset_palm_m = (
        sphere_reference_palm[:3, :3].T
        @ (
            online_reference_center_base
            - sphere_reference_palm[:3, 3]
        )
    )
    sphere_reference_lift_q = pose_preserving_joint_target(
        sphere_reference_grasp_q,
        np.asarray([0.0, 0.0, grasp_lift_m]),
        T_flange_policy_palm=contract.T_flange_policy_palm,
        joint_limits_rad=contract.joint_limits_rad,
        joint_limit_margin_rad=0.050,
        damping=0.04,
    )
    sphere_open_approach_lift_m = 0.030
    # Keep every flexion axis visibly open during the visual approach.  Only
    # thumb opposition is prepared in the air, matching the real cylinder
    # pregrasp.  The previous v5 draft drove the hand to source index 62
    # before capture; that pose is already almost closed and made the hand
    # wait around the empty capture volume for roughly two seconds.
    sphere_open_preshape = np.asarray(
        [1000, 1000, 1000, 1000, 1000, 0], dtype=np.int32
    )
    sphere_closed_hand = np.clip(
        sphere_hand_source[sphere_contact_index], 0, 1000
    ).astype(np.int32)
    sphere_closure_count = (
        online_closure_end_index - online_template_catch_index + 1
    )
    closure_fraction = (
        np.arange(1, sphere_closure_count + 1, dtype=np.float64)
        / sphere_closure_count
    )
    sphere_closure_blend = closure_fraction**3 * (
        10.0 - 15.0 * closure_fraction + 6.0 * closure_fraction**2
    )
    sphere_closure = np.rint(
        sphere_open_preshape[None, :]
        + sphere_closure_blend[:, None]
        * (sphere_closed_hand - sphere_open_preshape)[None, :]
    ).astype(np.int32)
    sphere_closure_max_tick_delta = np.max(
        np.abs(
            np.diff(
                np.concatenate(
                    [sphere_open_preshape[None, :], sphere_closure], axis=0
                ).astype(np.int32),
                axis=0,
            )
        ),
        axis=0,
    ).astype(np.int32)
    sphere_reference_approach_q = pose_preserving_joint_target(
        sphere_reference_grasp_q,
        np.asarray([0.0, 0.0, sphere_open_approach_lift_m]),
        T_flange_policy_palm=contract.T_flange_policy_palm,
        joint_limits_rad=contract.joint_limits_rad,
        joint_limit_margin_rad=0.050,
        damping=0.04,
    )
    sphere_approach_q = _minimum_jerk_path(
        contract.q_home_rad,
        sphere_reference_approach_q,
        intervals=(
            online_template_catch_index - online_template_start_index - 1
        ),
    )
    # An open RH56 at the final ball height would cross the calibrated table.
    # Keep the palm 30 mm high until the fingers have curled into a wide bowl,
    # then descend only over closure blend 0.55..0.80.  This couples arm and
    # hand motion so neither "close in empty air" nor open-finger table contact
    # is used as the workaround for the other.
    descent_input = np.clip((sphere_closure_blend - 0.55) / 0.25, 0.0, 1.0)
    descent_blend = descent_input**3 * (
        10.0 - 15.0 * descent_input + 6.0 * descent_input**2
    )
    sphere_closure_arm_q = np.asarray(
        [
            pose_preserving_joint_target(
                sphere_reference_grasp_q,
                np.asarray(
                    [
                        0.0,
                        0.0,
                        sphere_open_approach_lift_m * (1.0 - blend),
                    ]
                ),
                T_flange_policy_palm=contract.T_flange_policy_palm,
                joint_limits_rad=contract.joint_limits_rad,
                joint_limit_margin_rad=0.050,
                damping=0.04,
            )
            for blend in descent_blend
        ],
        dtype=np.float64,
    )
    sphere_lift_q = _minimum_jerk_path(
        sphere_reference_grasp_q,
        sphere_reference_lift_q,
        intervals=(online_lift_end_index - online_closure_end_index),
    )
    sphere_online_q = np.empty_like(online_q, dtype=np.float64)
    sphere_online_q[:online_template_start_index] = contract.q_home_rad
    sphere_online_q[
        online_template_start_index : online_template_catch_index
    ] = sphere_approach_q
    sphere_online_q[
        online_template_catch_index : online_closure_end_index + 1
    ] = sphere_closure_arm_q
    sphere_online_q[
        online_closure_end_index + 1 : online_lift_end_index + 1
    ] = sphere_lift_q[1:]
    sphere_online_q[online_lift_end_index + 1 :] = sphere_reference_lift_q
    sphere_online_q = sphere_online_q.astype(np.float32)
    sphere_preshape = _preshape(
        online_template_catch_index,
        sphere_open_preshape,
    )
    # Begin this minimum-jerk flexion only after the live spatial/TTC gate.
    # Fourteen 20 Hz targets span 0.70 s; the 0.75 s sphere gate leaves one
    # additional tick for the identified RH56 execution delay.  Stop exactly
    # at the first strict successful sphere grasp and hold it during lift;
    # never replay the source's later post-lift finger motion as "closure".
    sphere_online_hand = np.concatenate(
        [
            sphere_preshape,
            sphere_closure,
            np.repeat(
                sphere_closed_hand[None, :],
                online_frame_count - online_closure_end_index - 1,
                axis=0,
            ),
        ],
        axis=0,
    ).astype(np.int32)
    sphere_online_safety = _audit_plan(
        q_targets=sphere_online_q,
        hand_targets=sphere_online_hand,
        contract=contract,
        tabletop_plane=tabletop_plane,
        minimum_fingertip_clearance_m=0.020,
        maximum_target_tick_delta_rad=0.10,
    )
    preposition_palm = _T_base_palm(q_targets[preposition_end_index], contract)
    center_palm = (
        np.linalg.inv(preposition_palm)
        @ np.concatenate([catch_center_base, [1.0]])
    )[:3]
    actions = np.zeros((len(q_targets), 13), dtype=np.float32)
    time_s = np.arange(len(q_targets), dtype=np.float64) * 0.05
    output_directory.mkdir(parents=True, exist_ok=True)
    precheck_q = np.concatenate(
        [preposition_q, np.repeat(preposition_q[-1][None, :], 12, axis=0)],
        axis=0,
    ).astype(np.float32)
    precheck_hand = np.concatenate(
        [
            preposition_hand,
            np.repeat(preposition_hand[-1][None, :], 12, axis=0),
        ],
        axis=0,
    ).astype(np.int32)
    precheck_npz = io.BytesIO()
    np.savez(
        precheck_npz,
        time_s=np.arange(len(precheck_q), dtype=np.float64) * 0.05,
        policy_action=np.zeros((len(precheck_q), 13), dtype=np.float32),
        franka_joint_target_rad=precheck_q,
        inspire_angle_set_register_order=precheck_hand,
    )
    precheck_metadata = {
        "kind": "tabletop_preposition_clearance_check_v1",
        "control_hz": 20.0,
        "control_dt_s": 0.05,
        "frames": len(precheck_q),
        "action_contract": "exact actuator targets; preposition and hold only",
        "inspire_policy_order": list(CANONICAL_ACTION_ORDER[7:]),
        "inspire_register_order": list(REGISTER_ORDER),
        "recommended_replay_fields": {
            "franka": "franka_joint_target_rad",
            "inspire": "inspire_angle_set_register_order",
        },
        "source_real_policy_io": str(source_path.resolve()),
        "source_real_policy_io_sha256": source_sha256,
        "safety_audit": safety,
        "prohibited_phases": ["visual_trigger", "finger_closure", "lift"],
    }
    precheck_path = output_directory / "preposition_check.zip"
    with zipfile.ZipFile(precheck_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "replay/metadata.json",
            json.dumps(precheck_metadata, indent=2, sort_keys=True),
        )
        bundle.writestr("replay/data.npz", precheck_npz.getvalue())
    precheck_loaded = load_replay_actions(
        precheck_path, expected_policy_rate_hz=20
    )
    bundles = []
    for scenario in _scenario_specs():
        sphere_scenario = scenario["object_shape"] == "sphere"
        scenario_q = sphere_online_q if sphere_scenario else online_q
        scenario_hand = sphere_online_hand if sphere_scenario else online_hand
        scenario_safety = sphere_online_safety if sphere_scenario else online_safety
        axis_base = np.asarray(
            [0.0, float(scenario["direction"]), 0.0], dtype=np.float64
        )
        npz = io.BytesIO()
        np.savez(
            npz,
            time_s=np.arange(len(scenario_q), dtype=np.float64) * 0.05,
            policy_action=online_actions,
            franka_joint_target_rad=scenario_q,
            inspire_angle_set_register_order=scenario_hand,
        )
        speed_min, speed_max = scenario["speed_range"]
        metadata = {
            "kind": (
                "tabletop_online_visual_intercept_plan_v5"
                if sphere_scenario
                else "tabletop_online_visual_intercept_plan_v4"
            ),
            "control_hz": 20.0,
            "control_dt_s": 0.05,
            "frames": len(scenario_q),
            "action_contract": (
                "successful grasp pose and RH56 closure with transactional "
                "20 Hz visual intercept, closure tracking, and lift"
            ),
            "inspire_policy_order": list(CANONICAL_ACTION_ORDER[7:]),
            "inspire_register_order": list(REGISTER_ORDER),
            "recommended_replay_fields": {
                "franka": "franka_joint_target_rad",
                "inspire": "inspire_angle_set_register_order",
            },
            "scenario": scenario,
            "source_real_policy_io": str(source_path.resolve()),
            "source_real_policy_io_sha256": source_sha256,
            "source_indices": {
                "online_start": online_template_start_index,
                "source_contact": preposition_source_index,
                "source_lift_end": lift_source_end_index,
                "online_contact": online_template_catch_index,
                "online_closure_end": online_closure_end_index,
                "online_lift_end": online_lift_end_index,
            },
            "intercept_center_base_m": online_reference_center_base.tolist(),
            "safety_audit": scenario_safety,
            "nominal_table_clearance_lift_m": nominal_table_lift_m,
            "tabletop_online_planner": {
                "version": (
                    "tabletop_online_intercept_planner_v5"
                    if sphere_scenario
                    else "tabletop_online_intercept_planner_v4"
                ),
                "control_dt_s": 0.05,
                "template_start_index": online_template_start_index,
                "template_catch_index": online_template_catch_index,
                "closure_end_index": online_closure_end_index,
                "lift_end_index": online_lift_end_index,
                "fit_sample_count": 5,
                "motion_axis_base": axis_base.tolist(),
                "reference_intercept_center_base_m": (
                    online_reference_center_base.tolist()
                ),
                "speed_min_m_s": speed_min,
                "speed_max_m_s": speed_max,
                "heading_cos_min": 0.8660254037844386,
                "max_fit_residual_m": 0.015,
                # Receding-horizon visual servo: every 50 ms observation
                # refreshes the target, so projecting all the way to the
                # nominal catch tick only magnifies a transient velocity fit.
                # A 0.40 s horizon still leads a 0.40 m/s object by 160 mm,
                # while remaining inside the commissioned XY workspace.
                "max_linear_prediction_s": float(
                    scenario["max_linear_prediction_s"]
                ),
                "max_position_correction_xy_m": 0.210,
                "max_position_correction_z_m": 0.015,
                "correction_filter_alpha": 0.50,
                "dls_damping": 0.04,
                "max_joint_correction_rad": 0.65,
                "max_joint_correction_step_rad": 0.050, # 0.050
                "max_target_step_rad": 0.100,
                "max_orientation_correction_rad": 0.120,
                "joint_limit_margin_rad": 0.050,
                "tabletop_plane_base": tabletop_plane.tolist(),
                # Runtime correction may use part of the nominal 26 mm
                # clearance reserve, but it may never command a fingertip
                # closer than 15 mm to the calibrated tabletop.
                "minimum_fingertip_clearance_m": 0.015,
                "grasp_vertical_offset_m": (
                    -0.015
                    if sphere_scenario
                    else float(scenario["grasp_vertical_offset_m"])
                ),
                "grasp_lift_m": grasp_lift_m,
                # Captured tabletop runtime evidence measures about 118 ms
                # from RGB-D exposure to the projected policy point cloud.
                # Add the rounded 120 ms observation/command delay to every
                # receding-horizon intercept.  This is recomputed from fresh
                # velocity every tick, including after a board collision; it
                # is not a fixed spatial or fixed-time catch target.
                "contact_prediction_latency_s": (
                    0.120 if sphere_scenario else 0.050
                ),
                "collision_replan": bool(scenario["collision_replan"]),
                # The action source is a visual planner, not checkpoint
                # inference.  It may therefore use a measured speed from the
                # adjacent low/high band while retaining this scenario's
                # midpoint only as the no-fit prior.
                "adaptive_speed_max_m_s": 0.45,
                # The native 1 kHz controller remains the physical
                # velocity/acceleration/jerk authority.  This only bounds how
                # quickly the 20 Hz target may converge toward the live IK
                # goal; it never widens max_target_step_rad.
                "approach_target_step_rad": 0.10,
                # Finish the demonstrated open-hand preshape in about 0.9 s.
                # Its largest 20 Hz register delta is 75, still far below the
                # demonstrated closure segment's commissioned maximum 397.
                "preshape_phase_step": 3, # 3
                # The measured RH56 needs about 0.10 s before the first
                # bending response and about 0.45 s to reach the demonstrated
                # contact posture.  Lead the same sealed closure primitive by
                # that physical response time instead of waiting until only
                # 0.35 s remains.
                "closure_trigger_ttc_s": float(
                    scenario["closure_trigger_ttc_s"]
                ),
                "closure_cross_track_m": 0.06,
                "closure_z_tolerance_m": 0.035,
                "closure_past_tolerance_m": 0.03,
                "capture_arm_error_rad": 0.10,
                # Keep two degrees of joint-space reserve inside the
                # commissioned correction envelope instead of chasing its
                # exact boundary.
                "joint_correction_reserve_rad": 0.035,
            },
        }
        if sphere_scenario:
            metadata["sphere_grasp_source"] = {
                "replay": str(DEFAULT_SPHERE_SOURCE.resolve()),
                "replay_sha256": hashlib.sha256(
                    sphere_source_bytes
                ).hexdigest(),
                "trace": str(DEFAULT_SPHERE_TRACE.resolve()),
                "trace_sha256": hashlib.sha256(
                    sphere_trace_bytes
                ).hexdigest(),
                "first_strict_pre_lift_grasp_index": int(
                    sphere_contact_index
                ),
                "safety_lift_m": 0.020,
                "open_preshape_register_order": (
                    sphere_open_preshape.tolist()
                ),
                "closed_grasp_register_order": sphere_closed_hand.tolist(),
                "closure_duration_s": float(
                    sphere_closure_count * 0.05
                ),
                "maximum_closure_target_delta_per_tick_register_order": (
                    sphere_closure_max_tick_delta.tolist()
                ),
            }
            metadata["tabletop_online_planner"].update(
                {
                    "capture_object_offset_palm_m": (
                        sphere_capture_object_offset_palm_m.tolist()
                    ),
                    "arm_tracking_speed_rad_s": 0.50,
                    "minimum_intercept_horizon_s": 0.15,
                    "open_approach_lift_m": sphere_open_approach_lift_m,
                }
            )
        path = output_directory / f"{scenario['name']}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(
                "replay/metadata.json",
                json.dumps(metadata, indent=2, sort_keys=True),
            )
            bundle.writestr("replay/data.npz", npz.getvalue())
        loaded = load_replay_actions(path, expected_policy_rate_hz=20)
        if loaded.tabletop_online_planner is None:
            raise RuntimeError("generated replay lost its online planner contract")
        bundles.append(
            {
                "name": scenario["name"],
                "object_text": scenario["object_text"],
                "path": str(path.resolve()),
                "sha256": loaded.sha256,
                "frames": loaded.action_count,
                "collision_replan": scenario["collision_replan"],
            }
        )
    result = {
        "kind": "tabletop_online_motion_planner_manifest_v5",
        "hardware_writes": False,
        "source_real_policy_io": str(source_path.resolve()),
        "source_real_policy_io_sha256": source_sha256,
        "preposition_check": {
            "path": str(precheck_path.resolve()),
            "sha256": precheck_loaded.sha256,
            "frames": precheck_loaded.action_count,
            "finger_closure": False,
            "lift": False,
        },
        "preposition_end_index": preposition_end_index,
        "online_template_start_index": online_template_start_index,
        "online_template_catch_index": online_template_catch_index,
        "online_closure_end_index": online_closure_end_index,
        "online_lift_end_index": online_lift_end_index,
        "online_template_frames": len(online_q),
        "source_reference_grasp_q_rad": source_reference_grasp_q.tolist(),
        "dynamic_reference_grasp_q_rad": reference_grasp_q.tolist(),
        "dynamic_reference_max_q_home_delta_rad": float(
            np.max(np.abs(reference_grasp_q - contract.q_home_rad))
        ),
        "dynamic_reference_palm_translation_from_source_m": (
            dynamic_reference_translation_m.tolist()
        ),
        "sphere_reference_grasp_q_rad": sphere_reference_grasp_q.tolist(),
        "sphere_reference_approach_q_rad": (
            sphere_reference_approach_q.tolist()
        ),
        "sphere_open_approach_lift_m": sphere_open_approach_lift_m,
        "sphere_reference_max_q_home_delta_rad": float(
            np.max(
                np.abs(sphere_reference_grasp_q - contract.q_home_rad)
            )
        ),
        "sphere_capture_object_offset_palm_m": (
            sphere_capture_object_offset_palm_m.tolist()
        ),
        "sphere_grasp_source": {
            "replay": str(DEFAULT_SPHERE_SOURCE.resolve()),
            "replay_sha256": hashlib.sha256(sphere_source_bytes).hexdigest(),
            "trace": str(DEFAULT_SPHERE_TRACE.resolve()),
            "trace_sha256": hashlib.sha256(sphere_trace_bytes).hexdigest(),
            "first_strict_pre_lift_grasp_index": int(sphere_contact_index),
            "safety_lift_m": 0.020,
            "open_preshape_register_order": sphere_open_preshape.tolist(),
            "closed_grasp_register_order": sphere_closed_hand.tolist(),
            "closure_duration_s": float(sphere_closure_count * 0.05),
            "maximum_closure_target_delta_per_tick_register_order": (
                sphere_closure_max_tick_delta.tolist()
            ),
        },
        "intercept_center_base_m": online_reference_center_base.tolist(),
        "intercept_center_palm_m": center_palm.tolist(),
        "safety_audit": safety,
        "online_safety_audit": online_safety,
        "sphere_online_safety_audit": sphere_online_safety,
        "bundles": bundles,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policy-io", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--deploy-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_bundles(
        source_path=args.source_policy_io.expanduser().resolve(),
        deploy_bundle_path=args.deploy_bundle.expanduser().resolve(),
        output_directory=args.output_directory.expanduser().resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
