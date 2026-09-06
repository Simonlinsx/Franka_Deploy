from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from anydex_pipeline.control_sequence import (
    AuditedJointSequencePlan,
    AuditedLoadedLiftSequencePlan,
    AuditedPregraspSequencePlan,
    DefaultSequencePlan,
    LoadedLiftPayload,
    SequenceExecutionError,
    SequencePlan,
    SequencePlanError,
    SequenceState,
    SequenceStateError,
    StagedGraspSequence,
)
from anydex_pipeline.pregrasp_only_audit import (
    build_pregrasp_joint_path,
    pregrasp_prefix_contract_sha256,
)


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def _plan(*, preshape: bool = True, eligible: bool = True) -> SequencePlan:
    targets = (700, 710, 720, 730, 800, 900)
    return SequencePlan.from_arrays(
        default_q=(0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7),
        pregrasp_pose=_pose(0.4),
        grasp_pose=_pose(0.5),
        hand_target6=targets,
        execution_eligible=eligible,
        thumb_preshape_target=targets[5] if preshape else None,
        ineligibility_reason="collision check failed" if not eligible else "",
    )


def _air_plan() -> AuditedJointSequencePlan:
    q = np.asarray([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])
    pregrasp = q + np.asarray([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    grasp = q + np.asarray([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return AuditedJointSequencePlan(
        execution_eligible=True,
        mode="air_grasp",
        contact_and_lift_forbidden=True,
        audit_schema_version=2,
        joint_pose_binding_verified=True,
        joint_waypoints=(
            ("current", q),
            ("default", q),
            ("pregrasp", pregrasp),
            ("grasp", grasp),
        ),
        pregrasp_pose=_pose(0.4),
        grasp_pose=_pose(0.5),
        hand_target6=(700, 710, 720, 730, 800, 900),
        max_q_tracking_error_rad=0.002,
        thumb_preshape_target=900,
    )


class FakeArm:
    def __init__(self, log, *, settled=(True, True), fail_on=None):
        self.log = log
        self.settled = list(settled)
        self.fail_on = fail_on
        self.loaded_time_laws = []

    def move_joints(self, target_q):
        self.log.append(("arm.move_joints", tuple(float(v) for v in target_q)))
        if self.fail_on == "move_joints":
            raise RuntimeError("default move failed")
        return np.asarray(target_q, dtype=np.float64)

    def move_pose(self, target):
        label = "pregrasp" if np.isclose(target[0, 3], 0.4) else "grasp"
        self.log.append(("arm.move_pose", label))
        if self.fail_on == label:
            raise RuntimeError(f"{label} move failed")

    def move_loaded_joints(self, target_q, **time_law):
        self.loaded_time_laws.append(dict(time_law))
        return self.move_joints(target_q)

    def verify_settled(self, target, tolerances):
        label = "pregrasp" if np.isclose(target[0, 3], 0.4) else "grasp"
        self.log.append(("arm.verify_settled", label))
        if self.fail_on == f"verify_{label}":
            raise RuntimeError(f"{label} verification failed")
        return self.settled.pop(0)

    def verify_audited_joint_settled(
        self, target_q, target_pose, q_tolerance_rad, tolerances
    ):
        if np.isclose(target_pose[0, 3], 0.4):
            label = "pregrasp"
        elif np.isclose(target_pose[0, 3], 0.7):
            label = "lift"
        else:
            label = "grasp"
        self.log.append(
            (
                "arm.verify_audited_joint_settled",
                label,
                tuple(float(value) for value in target_q),
                float(q_tolerance_rad),
            )
        )
        if self.fail_on == f"verify_{label}":
            raise RuntimeError(f"{label} verification failed")
        return self.settled.pop(0)

    def stop(self):
        self.log.append(("arm.stop",))
        if self.fail_on == "stop":
            raise RuntimeError("arm stop acknowledgement missing")

    def apply_external_load_and_verify(self, payload):
        self.log.append(
            ("arm.apply_external_load_and_verify", float(payload.mass_kg))
        )
        if self.fail_on == "apply_load":
            raise RuntimeError("external load verification failed")

    def clear_external_load_and_verify(self):
        self.log.append(("arm.clear_external_load_and_verify",))
        if self.fail_on == "clear_load":
            raise RuntimeError("external load clear verification failed")

    def verify_idle_state(self):
        self.log.append(("arm.verify_idle_state",))
        if self.fail_on == "verify_idle_state":
            raise RuntimeError("loaded idle verification failed")


class FakeHand:
    def __init__(self, log, *, fail_on=None):
        self.log = log
        self.fail_on = fail_on
        self.angle_tolerance = 20
        self.hold_targets = None
        self.snapshot_angles = None
        self.snapshot_statuses = (2, 2, 2, 2, 2, 2)
        self.snapshot_contact_axes = ()

    def open_and_verify(self, targets):
        self.log.append(("hand.open_and_verify", tuple(targets)))
        if self.fail_on == "open":
            raise RuntimeError("open failed")

    def preshape_thumb(self, target_q6):
        self.log.append(("hand.preshape_thumb", int(target_q6)))
        if self.fail_on == "preshape":
            raise RuntimeError("preshape failed")

    def close_bends_and_hold(self, targets):
        self.log.append(("hand.close_bends_and_hold", tuple(targets)))
        if self.fail_on == "close":
            raise RuntimeError("close failed after partial motion")
        self.hold_targets = tuple(targets)

    def close_bends_no_contact_and_hold(self, targets):
        self.log.append(("hand.close_bends_no_contact_and_hold", tuple(targets)))
        if self.fail_on == "close":
            raise RuntimeError("no-contact close failed after partial motion")
        self.hold_targets = tuple(targets)

    def verify_bounded_hold(self, targets, *, no_contact):
        expected = tuple(targets)
        self.log.append(
            ("hand.verify_bounded_hold", expected, bool(no_contact))
        )
        if self.fail_on == "verify_bounded_hold":
            raise RuntimeError("bounded hold read failed")
        if self.hold_targets != expected:
            raise RuntimeError("bounded hold ANGLE_SET changed unexpectedly")
        angles = (
            self.hold_targets
            if self.snapshot_angles is None
            else self.snapshot_angles
        )
        if no_contact and (
            self.snapshot_statuses != (2, 2, 2, 2, 2, 2)
            or self.snapshot_contact_axes
        ):
            raise RuntimeError("air bounded hold detected forbidden contact")
        if not no_contact and any(
            status not in (2, 3) for status in self.snapshot_statuses
        ):
            raise RuntimeError("bounded hold axis is neither settled nor in contact")
        if any(
            status == 2 and abs(actual - target) > self.angle_tolerance
            for actual, target, status in zip(
                angles, expected, self.snapshot_statuses
            )
        ):
            raise RuntimeError(
                "bounded hold settled ANGLE_ACT is outside tolerance"
            )
        if abs(angles[5] - expected[5]) > self.angle_tolerance:
            raise RuntimeError(
                "bounded hold thumb rotation is outside target tolerance"
            )

    def return_no_contact_hand_to_open(self):
        self.log.append(("hand.return_no_contact_hand_to_open",))
        if self.fail_on == "return":
            raise RuntimeError("monitored reverse failed")

    def verify_loaded_hold(self, targets, minimum_contact_axes):
        self.log.append(
            (
                "hand.verify_loaded_hold",
                tuple(int(value) for value in targets),
                int(minimum_contact_axes),
            )
        )
        if self.fail_on == "verify_loaded_hold":
            raise RuntimeError("loaded hold verification failed")

    def disable_and_verify(self):
        self.log.append(("hand.disable_and_verify",))
        if self.fail_on == "disable":
            raise RuntimeError("hand disable readback missing")


def _pregrasp_only_plan(*, eligible=True, tamper_hash=False):
    current = np.asarray([0.01, 0, 0, -1.57, 0, 1.57, 0.81])
    transit = current + np.asarray([0.01, 0, 0, 0, 0, 0, 0])
    default = transit + np.asarray([0.01, 0, 0, 0, 0, 0, 0])
    approach = default + np.asarray([0.01, 0.01, 0, 0, 0, 0, 0])
    pregrasp = approach + np.asarray([0.01, 0.01, 0, 0, 0, 0, 0])
    waypoints = (
        ("current", current),
        ("default_transit_0", transit),
        ("default", default),
        ("approach_transit_0", approach),
        ("pregrasp", pregrasp),
    )
    samples, _ = build_pregrasp_joint_path(waypoints, 0.01)
    import hashlib

    raw = np.ascontiguousarray(np.asarray(samples, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(
        ("dtype=<f8;shape={};".format(",".join(map(str, raw.shape)))).encode("ascii")
    )
    digest.update(raw.tobytes(order="C"))
    samples_hash = digest.hexdigest()
    prefix_hash = pregrasp_prefix_contract_sha256(
        waypoints=waypoints,
        pregrasp_pose=_pose(0.4),
        max_joint_step_rad=0.01,
        max_q_tracking_error_rad=0.002,
        samples_sha256=samples_hash,
    )
    if tamper_hash:
        prefix_hash = "0" * 64
    return AuditedPregraspSequencePlan(
        execution_eligible=eligible,
        audit_schema_version=2,
        audit_artifact_sha256="a" * 64,
        prefix_contract_sha256=prefix_hash,
        joint_path_samples_sha256=samples_hash,
        joint_waypoints=waypoints,
        pregrasp_pose=_pose(0.4),
        max_joint_step_rad=0.01,
        max_q_tracking_error_rad=0.002,
        ineligibility_reason="prefix collision failed" if not eligible else "",
    )


def test_pregrasp_only_sequence_disables_hand_before_arm_and_never_reaches_grasp():
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))
    plan = _pregrasp_only_plan()

    assert sequence.run_to_pregrasp_joint_waypoints(plan) == SequenceState.PREGRASP_VERIFIED
    names = [item[0] for item in log]
    assert names[:2] == ["hand.open_and_verify", "hand.disable_and_verify"]
    assert names.count("arm.move_joints") == 4
    assert names[-1] == "arm.verify_audited_joint_settled"
    assert not any(name in names for name in (
        "arm.move_pose",
        "hand.preshape_thumb",
        "hand.close_bends_and_hold",
        "hand.close_bends_no_contact_and_hold",
    ))
    assert sequence.abort("pregrasp-only success cleanup") == SequenceState.STOPPED
    assert log[-2:] == [("arm.stop",), ("hand.disable_and_verify",)]


def test_pregrasp_only_invalid_prefix_hash_is_rejected_before_driver_calls():
    log = []
    with pytest.raises(SequencePlanError, match="prefix contract"):
        _pregrasp_only_plan(tamper_hash=True)
    assert log == []


def test_pregrasp_only_midway_failure_dual_stops_and_never_commands_hand_again():
    log = []
    sequence = StagedGraspSequence(
        FakeArm(log, fail_on="move_joints"), FakeHand(log)
    )
    with pytest.raises(SequenceExecutionError) as raised:
        sequence.run_to_pregrasp_joint_waypoints(_pregrasp_only_plan())
    assert raised.value.stop_confirmed
    assert [item[0] for item in log][-2:] == ["arm.stop", "hand.disable_and_verify"]
    assert not any(item[0].startswith("hand.close") for item in log)


def test_run_full_has_exact_staged_order_and_state_history():
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))

    assert sequence.run_full(_plan()) == SequenceState.HOLDING
    assert log == [
        ("hand.open_and_verify", (1000, 1000, 1000, 1000, 1000, 1000)),
        ("arm.move_joints", (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7)),
        ("arm.move_pose", "pregrasp"),
        ("arm.verify_settled", "pregrasp"),
        ("arm.move_pose", "grasp"),
        ("arm.verify_settled", "grasp"),
        ("hand.preshape_thumb", 900),
        ("hand.close_bends_and_hold", (700, 710, 720, 730, 800, 900)),
    ]


def test_bounded_hold_refresh_reuses_hand_owner_and_exact_close_target():
    log = []
    hand = FakeHand(log)
    sequence = StagedGraspSequence(FakeArm(log), hand)

    assert sequence.run_full(_plan()) == SequenceState.HOLDING
    assert sequence.bounded_hold_targets == (700, 710, 720, 730, 800, 900)
    assert sequence.verify_bounded_holding() == SequenceState.HOLDING
    assert log[-1] == (
        "hand.verify_bounded_hold",
        (700, 710, 720, 730, 800, 900),
        False,
    )
    assert [item[0] for item in log].count("hand.verify_bounded_hold") == 1


def test_contact_grasp_hold_accepts_contact_axis_but_checks_settled_axes():
    log = []
    hand = FakeHand(log)
    sequence = StagedGraspSequence(FakeArm(log), hand)
    sequence.run_full(_plan())
    hand.snapshot_statuses = (3, 2, 2, 2, 2, 2)
    hand.snapshot_contact_axes = ("pinky",)
    hand.snapshot_angles = (850, 710, 720, 730, 800, 900)

    assert sequence.verify_bounded_holding() == SequenceState.HOLDING


@pytest.mark.parametrize(
    ("statuses", "angles", "message"),
    [
        (
            (1, 2, 2, 2, 2, 2),
            (700, 710, 720, 730, 800, 900),
            "neither settled nor in contact",
        ),
        (
            (2, 2, 2, 2, 2, 2),
            (721, 710, 720, 730, 800, 900),
            "settled ANGLE_ACT is outside tolerance",
        ),
        (
            (3, 2, 2, 2, 2, 3),
            (850, 710, 720, 730, 800, 921),
            "thumb rotation is outside target tolerance",
        ),
    ],
)
def test_contact_grasp_bounded_hold_rejects_unsettled_or_drifted_axis(
    statuses, angles, message
):
    log = []
    hand = FakeHand(log)
    sequence = StagedGraspSequence(FakeArm(log), hand)
    sequence.run_full(_plan())
    hand.snapshot_statuses = statuses
    hand.snapshot_angles = angles

    with pytest.raises(SequenceExecutionError, match=message):
        sequence.verify_bounded_holding()

    assert sequence.state == SequenceState.FAULT_LATCHED
    assert log[-2:] == [("arm.stop",), ("hand.disable_and_verify",)]


def test_bounded_hold_target_drift_faults_and_runs_coordinated_cleanup():
    log = []
    hand = FakeHand(log)
    sequence = StagedGraspSequence(FakeArm(log), hand)
    sequence.run_full(_plan())
    hand.hold_targets = (701, 710, 720, 730, 800, 900)

    with pytest.raises(SequenceExecutionError, match="ANGLE_SET changed") as raised:
        sequence.verify_bounded_holding()

    assert raised.value.failure_state == SequenceState.HOLDING
    assert raised.value.stop_confirmed is True
    assert sequence.state == SequenceState.FAULT_LATCHED
    assert log[-3:] == [
        (
            "hand.verify_bounded_hold",
            (700, 710, 720, 730, 800, 900),
            False,
        ),
        ("arm.stop",),
        ("hand.disable_and_verify",),
    ]


@pytest.mark.parametrize(
    ("statuses", "contact_axes"),
    [
        ((2, 3, 2, 2, 2, 2), ()),
        ((2, 1, 2, 2, 2, 2), ()),
        ((2, 2, 2, 2, 2, 2), ("ring",)),
    ],
)
def test_air_bounded_hold_rejects_status_or_contact_axis(
    statuses, contact_axes
):
    log = []
    hand = FakeHand(log)
    sequence = StagedGraspSequence(FakeArm(log), hand)
    sequence.run_full_joint_waypoints(_air_plan())
    hand.snapshot_statuses = statuses
    hand.snapshot_contact_axes = contact_axes

    with pytest.raises(SequenceExecutionError, match="forbidden contact") as raised:
        sequence.verify_bounded_holding()

    assert raised.value.failure_state == SequenceState.HOLDING
    assert raised.value.stop_confirmed is True
    assert sequence.state == SequenceState.FAULT_LATCHED
    assert log[-2:] == [("arm.stop",), ("hand.disable_and_verify",)]


def test_bounded_hold_monitor_is_rejected_outside_holding_without_read():
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))

    with pytest.raises(SequenceStateError, match="requires HOLDING"):
        sequence.verify_bounded_holding()

    assert log == []


def test_audited_joint_mode_dispatches_every_bound_waypoint_in_order():
    current = np.asarray([0.01, 0, 0, -1.57, 0, 1.57, 0.81])
    transit_a = np.asarray([0.04, 0, 0, -1.57, 0, 1.57, 0.81])
    transit_b = np.asarray([0.04, 0, 0, -1.57, 0, 1.57, 0.0])
    default = np.asarray([0, 0, 0, -1.57, 0, 1.57, 0])
    pregrasp = default + np.asarray([0.1, 0.05, 0, 0.02, 0, 0, 0])
    grasp = pregrasp + np.asarray([0.01, 0.02, 0, 0.01, 0, 0, 0])
    plan = AuditedJointSequencePlan(
        execution_eligible=True,
        mode="loaded_grasp",
        contact_and_lift_forbidden=False,
        audit_schema_version=2,
        joint_pose_binding_verified=True,
        joint_waypoints=(
            ("current", current),
            ("default_transit_0", transit_a),
            ("default_transit_1", transit_b),
            ("default", default),
            ("pregrasp", pregrasp),
            ("grasp", grasp),
        ),
        pregrasp_pose=_pose(0.4),
        grasp_pose=_pose(0.5),
        hand_target6=(700, 710, 720, 730, 800, 900),
        max_q_tracking_error_rad=0.002,
        thumb_preshape_target=900,
    )
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))

    assert sequence.run_full_joint_waypoints(plan) == SequenceState.HOLDING
    sent = [entry[1] for entry in log if entry[0] == "arm.move_joints"]
    assert sent == [
        tuple(transit_a),
        tuple(transit_b),
        tuple(default),
        tuple(pregrasp),
        tuple(grasp),
    ]
    assert not any(entry[0] == "arm.move_pose" for entry in log)
    assert [
        entry[1]
        for entry in log
        if entry[0] == "arm.verify_audited_joint_settled"
    ] == [
        "pregrasp",
        "grasp",
    ]


def test_audited_default_prefix_stops_before_pregrasp_and_can_be_aborted():
    current = np.asarray([0.01, 0, 0, -1.57, 0, 1.57, 0.81])
    transit = np.asarray([0.04, 0, 0, -1.57, 0, 1.57, 0.81])
    default = np.asarray([0, 0, 0, -1.57, 0, 1.57, 0])
    pregrasp = default + np.asarray([0.1, 0.05, 0, 0.02, 0, 0, 0])
    grasp = pregrasp + np.asarray([0.01, 0.02, 0, 0.01, 0, 0, 0])
    plan = AuditedJointSequencePlan(
        execution_eligible=True,
        mode="air_grasp",
        contact_and_lift_forbidden=True,
        audit_schema_version=2,
        joint_pose_binding_verified=True,
        joint_waypoints=(
            ("current", current),
            ("default_transit_0", transit),
            ("default", default),
            ("pregrasp", pregrasp),
            ("grasp", grasp),
        ),
        pregrasp_pose=_pose(0.4),
        grasp_pose=_pose(0.5),
        hand_target6=(700, 710, 720, 730, 800, 900),
        max_q_tracking_error_rad=0.002,
        thumb_preshape_target=900,
    )
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))

    assert (
        sequence.run_to_default_joint_waypoints(plan)
        == SequenceState.DEFAULT_VERIFIED
    )
    assert [entry for entry in log if entry[0] == "arm.move_joints"] == [
        ("arm.move_joints", tuple(transit)),
        ("arm.move_joints", tuple(default)),
    ]
    assert not any(
        entry[0] == "arm.verify_audited_joint_settled" for entry in log
    )
    assert not any(entry[0].startswith("hand.preshape") for entry in log)
    assert not any(entry[0].startswith("hand.close") for entry in log)

    assert sequence.abort("staged recapture") == SequenceState.STOPPED
    assert log[-2:] == [
        ("arm.stop",),
        ("hand.disable_and_verify",),
    ]


def test_audited_approach_transits_are_dispatched_after_default():
    current = np.asarray([0.01, 0, 0, -1.57, 0, 1.57, 0.81])
    default = np.asarray([0, 0, 0, -1.57, 0, 1.57, 0])
    approach_a = default + np.asarray([0.05, 0.02, 0, 0, 0, 0, 0.05])
    approach_b = default + np.asarray([0.10, 0.04, 0, 0, 0, 0, 0.10])
    pregrasp = default + np.asarray([0.15, 0.06, 0, 0, 0, 0, 0.15])
    grasp = pregrasp + np.asarray([0.01, 0.01, 0, 0, 0, 0, 0])
    plan = AuditedJointSequencePlan(
        execution_eligible=True,
        mode="air_grasp",
        contact_and_lift_forbidden=True,
        audit_schema_version=2,
        joint_pose_binding_verified=True,
        joint_waypoints=(
            ("current", current),
            ("default", default),
            ("approach_transit_0", approach_a),
            ("approach_transit_1", approach_b),
            ("pregrasp", pregrasp),
            ("grasp", grasp),
        ),
        pregrasp_pose=_pose(0.4),
        grasp_pose=_pose(0.5),
        hand_target6=(700, 710, 720, 730, 800, 900),
        max_q_tracking_error_rad=0.002,
        thumb_preshape_target=900,
    )
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))

    assert sequence.run_full_joint_waypoints(plan) == SequenceState.HOLDING
    assert [entry[1] for entry in log if entry[0] == "arm.move_joints"] == [
        tuple(default),
        tuple(approach_a),
        tuple(approach_b),
        tuple(pregrasp),
        tuple(grasp),
    ]
    assert log[-1] == (
        "hand.close_bends_no_contact_and_hold",
        (700, 710, 720, 730, 800, 900),
    )


def test_air_grasp_closure_has_no_contact_or_lift_arm_stage():
    current = np.asarray([0.0, 0, 0, -1.57, 0, 1.57, 0.0])
    default = current.copy()
    pregrasp = current + np.asarray([0.05, 0, 0, 0, 0, 0, 0])
    air_grasp = current + np.asarray([0.10, 0, 0, 0, 0, 0, 0])
    plan = AuditedJointSequencePlan(
        execution_eligible=True,
        mode="air_grasp",
        contact_and_lift_forbidden=True,
        audit_schema_version=2,
        joint_pose_binding_verified=True,
        joint_waypoints=(
            ("current", current),
            ("default", default),
            ("pregrasp", pregrasp),
            ("grasp", air_grasp),
        ),
        pregrasp_pose=_pose(0.4),
        grasp_pose=_pose(0.5),
        hand_target6=(700, 710, 720, 730, 800, 900),
        max_q_tracking_error_rad=0.002,
        thumb_preshape_target=900,
    )
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))
    assert sequence.run_full_joint_waypoints(plan) == SequenceState.HOLDING
    close_index = next(
        index
        for index, item in enumerate(log)
        if item[0] == "hand.close_bends_no_contact_and_hold"
    )
    assert not any(item[0].startswith("arm.move") for item in log[close_index + 1 :])
    assert not any(item[0] == "arm.move_pose" for item in log)
    assert sequence.state_history == [
        SequenceState.DISARMED,
        SequenceState.OPENING_HAND,
        SequenceState.OPEN_VERIFIED,
        SequenceState.MOVING_FRANKA_DEFAULT,
        SequenceState.DEFAULT_VERIFIED,
        SequenceState.MOVING_PREGRASP,
        SequenceState.PREGRASP_VERIFIED,
        SequenceState.MOVING_GRASP,
        SequenceState.EEF_SETTLING,
        SequenceState.THUMB_PRESHAPE,
        SequenceState.CLOSING_BENDS,
        SequenceState.HOLDING,
    ]

    assert sequence.return_air_hand_to_open() == SequenceState.RETURNED_OPEN
    assert log[-1] == ("hand.return_no_contact_hand_to_open",)
    assert sequence.state_history[-2:] == [
        SequenceState.REOPENING_HAND,
        SequenceState.RETURNED_OPEN,
    ]


def test_air_reverse_failure_stops_arm_then_verifies_hand_disable():
    q = np.asarray([0.0, 0, 0, -1.57, 0, 1.57, 0.0])
    plan = AuditedJointSequencePlan(
        execution_eligible=True,
        mode="air_grasp",
        contact_and_lift_forbidden=True,
        audit_schema_version=2,
        joint_pose_binding_verified=True,
        joint_waypoints=(("current", q), ("default", q), ("pregrasp", q), ("grasp", q)),
        pregrasp_pose=_pose(0.4),
        grasp_pose=_pose(0.5),
        hand_target6=(700, 710, 720, 730, 800, 900),
        max_q_tracking_error_rad=0.002,
        thumb_preshape_target=900,
    )
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log, fail_on="return"))
    assert sequence.run_full_joint_waypoints(plan) == SequenceState.HOLDING

    with pytest.raises(SequenceExecutionError) as raised:
        sequence.return_air_hand_to_open()

    return_index = log.index(("hand.return_no_contact_hand_to_open",))
    assert log[return_index + 1 :] == [
        ("arm.stop",),
        ("hand.disable_and_verify",),
    ]
    assert raised.value.failure_state == SequenceState.REOPENING_HAND
    assert sequence.state == SequenceState.FAULT_LATCHED


def test_run_to_default_can_be_followed_by_full_without_reopening():
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))
    plan = _plan(preshape=False)

    assert sequence.run_to_default(plan) == SequenceState.DEFAULT_VERIFIED
    assert sequence.run_full(plan) == SequenceState.HOLDING
    assert [entry[0] for entry in log].count("hand.open_and_verify") == 1
    assert not any(entry[0] == "hand.preshape_thumb" for entry in log)


def test_default_stage_has_independent_execution_permission():
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))
    default = DefaultSequencePlan(
        default_execution_eligible=True,
        default_q=np.asarray([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]),
    )

    assert sequence.run_to_default(default) == SequenceState.DEFAULT_VERIFIED
    assert [entry[0] for entry in log] == [
        "hand.open_and_verify",
        "arm.move_joints",
    ]


def test_ineligible_default_stage_is_rejected_before_driver_calls():
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))
    default = DefaultSequencePlan(
        default_execution_eligible=False,
        default_q=np.zeros(7),
        ineligibility_reason="tool dynamics unverified",
    )
    with pytest.raises(SequencePlanError, match="tool dynamics unverified"):
        sequence.run_to_default(default)
    assert log == []


def test_failed_grasp_arrival_never_calls_preshape_or_close():
    log = []
    sequence = StagedGraspSequence(
        FakeArm(log, settled=(True, False)), FakeHand(log)
    )

    with pytest.raises(SequenceExecutionError) as raised:
        sequence.run_full(_plan())

    assert sequence.state == SequenceState.FAULT_LATCHED
    assert raised.value.stop_confirmed
    assert "grasp EEF settle verification failed" in str(raised.value.original_error)
    names = [entry[0] for entry in log]
    assert "hand.preshape_thumb" not in names
    assert "hand.close_bends_and_hold" not in names
    assert names[-2:] == ["arm.stop", "hand.disable_and_verify"]


def test_partial_close_failure_stops_without_arm_retreat():
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log, fail_on="close"))

    with pytest.raises(SequenceExecutionError) as raised:
        sequence.run_full(_plan(preshape=False))

    close_index = next(
        index for index, entry in enumerate(log) if entry[0] == "hand.close_bends_and_hold"
    )
    assert log[close_index + 1 :] == [("arm.stop",), ("hand.disable_and_verify",)]
    assert raised.value.failure_state == SequenceState.CLOSING_BENDS
    assert raised.value.stop_confirmed


def test_air_no_contact_failure_stops_arm_then_verifies_hand_disable():
    q = np.asarray([0.0, 0, 0, -1.57, 0, 1.57, 0.0])
    plan = AuditedJointSequencePlan(
        execution_eligible=True,
        mode="air_grasp",
        contact_and_lift_forbidden=True,
        audit_schema_version=2,
        joint_pose_binding_verified=True,
        joint_waypoints=(
            ("current", q),
            ("default", q),
            ("pregrasp", q),
            ("grasp", q),
        ),
        pregrasp_pose=_pose(0.4),
        grasp_pose=_pose(0.5),
        hand_target6=(700, 710, 720, 730, 800, 900),
        max_q_tracking_error_rad=0.002,
        thumb_preshape_target=900,
    )
    log = []
    sequence = StagedGraspSequence(
        FakeArm(log), FakeHand(log, fail_on="close")
    )

    with pytest.raises(SequenceExecutionError) as raised:
        sequence.run_full_joint_waypoints(plan)

    close_index = next(
        index
        for index, entry in enumerate(log)
        if entry[0] == "hand.close_bends_no_contact_and_hold"
    )
    assert log[close_index + 1 :] == [
        ("arm.stop",),
        ("hand.disable_and_verify",),
    ]
    assert raised.value.failure_state == SequenceState.CLOSING_BENDS
    assert raised.value.stop_confirmed


def test_mid_hand_franka_watchdog_fault_causes_dual_stop_and_no_later_target():
    class WatchdogFaultHand(FakeHand):
        def preshape_thumb(self, target_q6):
            self.log.append(("hand.numeric_target", int(target_q6)))
            # The real RH56 driver immediately batch-disables before this
            # exception reaches the sequencer (covered by its driver test).
            self.log.append(("hand.emergency_disable",))
            raise RuntimeError("continuous Franka read-only gate lost Idle")

    log = []
    sequence = StagedGraspSequence(FakeArm(log), WatchdogFaultHand(log))

    with pytest.raises(SequenceExecutionError, match="lost Idle") as raised:
        sequence.run_full(_plan())

    numeric_index = log.index(("hand.numeric_target", 900))
    assert log[numeric_index + 1 :] == [
        ("hand.emergency_disable",),
        ("arm.stop",),
        ("hand.disable_and_verify",),
    ]
    assert raised.value.failure_state == SequenceState.THUMB_PRESHAPE
    assert raised.value.stop_confirmed


def test_abort_from_holding_stops_arm_before_disabling_hand():
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))
    sequence.run_full(_plan(preshape=False))
    before_abort = len(log)

    assert sequence.abort("operator requested stop") == SequenceState.STOPPED
    assert log[before_abort:] == [("arm.stop",), ("hand.disable_and_verify",)]
    assert sequence.state_history[-2:] == [
        SequenceState.STOPPING,
        SequenceState.STOPPED,
    ]


def test_original_failure_and_unconfirmed_stop_are_both_preserved():
    log = []
    sequence = StagedGraspSequence(
        FakeArm(log, fail_on="grasp"), FakeHand(log, fail_on="disable")
    )

    with pytest.raises(SequenceExecutionError) as raised:
        sequence.run_full(_plan())

    error = raised.value
    assert str(error.original_error) == "grasp move failed"
    assert not error.stop_confirmed
    assert [str(item) for item in error.stop_errors] == [
        "hand disable readback missing"
    ]
    assert "STOP UNCONFIRMED" in str(error)
    assert log[-2:] == [("arm.stop",), ("hand.disable_and_verify",)]


def _duck_plan(**updates):
    values = dict(
        execution_eligible=True,
        default_q=np.zeros(7),
        pregrasp_pose=_pose(0.4),
        grasp_pose=_pose(0.5),
        hand_target6=(700, 700, 700, 700, 700, 900),
        thumb_preshape_target=900,
    )
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "plan",
    [
        _plan(eligible=False),
        _duck_plan(hand_target6=(700, 700, 700, 700, 700, 1001)),
        _duck_plan(hand_target6=(700, 700, 700, 700, 700)),
        _duck_plan(thumb_preshape_target=899),
    ],
)
def test_ineligible_or_invalid_targets_are_rejected_before_driver_calls(plan):
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))

    with pytest.raises(SequencePlanError):
        sequence.run_full(plan)

    assert sequence.state == SequenceState.DISARMED
    assert log == []


def _loaded_lift_plan(*, eligible=True, **updates):
    current = np.asarray([0.01, 0, 0, -1.57, 0, 1.57, 0.81])
    default = current + np.asarray([0.01, 0, 0, 0, 0, 0, 0])
    pregrasp = default + np.asarray([0.01, 0, 0, 0, 0, 0, 0])
    grasp = pregrasp + np.asarray([0.01, 0, 0, 0, 0, 0, 0])
    lift_transit = grasp + np.asarray([0.01, 0, 0, 0, 0, 0, 0])
    lift = lift_transit + np.asarray([0.01, 0, 0, 0, 0, 0, 0])
    base = AuditedJointSequencePlan(
        execution_eligible=eligible,
        mode="loaded_grasp",
        contact_and_lift_forbidden=False,
        audit_schema_version=2,
        joint_pose_binding_verified=True,
        joint_waypoints=(
            ("current", current),
            ("default", default),
            ("pregrasp", pregrasp),
            ("grasp", grasp),
        ),
        pregrasp_pose=_pose(0.4),
        grasp_pose=_pose(0.5),
        hand_target6=(700, 710, 720, 730, 800, 900),
        max_q_tracking_error_rad=0.002,
        thumb_preshape_target=900,
    )
    payload = LoadedLiftPayload(
        mass_kg=0.25,
        F_x_Cload_m=np.asarray([0.0, 0.0, 0.08]),
        I_load_kg_m2=np.diag([0.001, 0.001, 0.001]),
        binding_sha256="b" * 64,
    )
    values = dict(
        execution_eligible=eligible,
        loaded_lift_audit_schema_version=1,
        loaded_lift_binding_verified=True,
        loaded_lift_artifact_sha256="a" * 64,
        grasp_plan=base,
        lift_waypoints=(
            ("grasp", grasp),
            ("lift_transit_0", lift_transit),
            ("lift", lift),
        ),
        lift_pose=_pose(0.7),
        payload=payload,
        minimum_contact_axes=2,
        round_trip_setdown_required=True,
        max_q_tracking_error_rad=0.002,
        time_law_max_joint_velocity_rad_s=0.05,
        time_law_max_joint_acceleration_rad_s2=0.10,
        time_law_max_dynamic_segment_rad=0.20,
        time_law_min_segment_duration_s=6.0,
    )
    values.update(updates)
    return AuditedLoadedLiftSequencePlan(**values)


def test_loaded_lift_requires_separate_plan_type_before_any_driver_call():
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))

    with pytest.raises(SequencePlanError, match="AuditedLoadedLiftSequencePlan"):
        sequence.run_loaded_lift_joint_waypoints(_loaded_lift_plan().grasp_plan)

    assert log == []
    assert sequence.state == SequenceState.DISARMED


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"time_law_max_joint_velocity_rad_s": 0.21}, "velocity"),
        ({"time_law_max_joint_acceleration_rad_s2": 0.51}, "acceleration"),
        ({"time_law_max_dynamic_segment_rad": 0.36}, "segment"),
        (
            {"settle_tolerances": SimpleNamespace(
                position_m=100.0,
                orientation_rad=0.05,
                linear_speed_m_s=0.01,
                angular_speed_rad_s=0.05,
                stable_seconds=0.30,
            )},
            "position_m",
        ),
    ],
)
def test_loaded_lift_plan_rejects_unbounded_time_law_or_settle(updates, message):
    with pytest.raises(SequencePlanError, match=message):
        _loaded_lift_plan(**updates)


@pytest.mark.parametrize("mass", [True, False, "0.25", b"0.25"])
def test_loaded_payload_rejects_boolean_or_text_mass(mass):
    with pytest.raises(SequencePlanError, match="numeric scalar"):
        LoadedLiftPayload(
            mass_kg=mass,
            F_x_Cload_m=np.asarray([0.0, 0.0, 0.08]),
            I_load_kg_m2=np.diag([0.001, 0.001, 0.001]),
            binding_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_q_tracking_error_rad", "0.002"),
        ("max_q_tracking_error_rad", True),
        ("time_law_max_joint_velocity_rad_s", "0.05"),
        ("time_law_max_joint_acceleration_rad_s2", False),
        ("time_law_max_dynamic_segment_rad", "0.20"),
        ("time_law_min_segment_duration_s", True),
    ],
)
def test_loaded_lift_plan_rejects_boolean_or_text_numeric_fields(field, value):
    with pytest.raises(SequencePlanError):
        _loaded_lift_plan(**{field: value})


def test_loaded_lift_round_trip_reverses_exact_suffix_before_release():
    plan = _loaded_lift_plan()
    log = []
    arm = FakeArm(log, settled=(True, True, True, True))
    sequence = StagedGraspSequence(
        arm, FakeHand(log)
    )

    assert (
        sequence.run_loaded_lift_joint_waypoints(plan)
        == SequenceState.LIFTED_HOLDING
    )
    assert sequence.requires_manual_load_recovery
    close_index = next(
        index
        for index, item in enumerate(log)
        if item[0] == "hand.close_bends_and_hold"
    )
    apply_index = log.index(("arm.apply_external_load_and_verify", 0.25))
    sent = [item[1] for item in log if item[0] == "arm.move_joints"]
    assert close_index < apply_index
    assert sent == [
        tuple(plan.grasp_plan.joint_waypoints[1][1]),
        tuple(plan.grasp_plan.joint_waypoints[2][1]),
        tuple(plan.grasp_plan.joint_waypoints[3][1]),
        tuple(plan.lift_waypoints[1][1]),
        tuple(plan.lift_waypoints[2][1]),
    ]

    before_return = len(log)
    assert (
        sequence.return_loaded_lift_to_setdown(plan)
        == SequenceState.SETDOWN_COMPLETE
    )
    assert not sequence.requires_manual_load_recovery
    returned = [
        item[1]
        for item in log[before_return:]
        if item[0] == "arm.move_joints"
    ]
    assert returned == [
        tuple(plan.lift_waypoints[1][1]),
        tuple(plan.lift_waypoints[0][1]),
    ]
    assert arm.loaded_time_laws == [
        {
            "max_joint_velocity_rad_s": 0.05,
            "max_joint_acceleration_rad_s2": 0.10,
            "max_dynamic_segment_rad": 0.20,
            "min_segment_duration_s": 6.0,
        }
    ] * 4
    disable_index = max(
        index for index, item in enumerate(log) if item[0] == "hand.disable_and_verify"
    )
    clear_index = log.index(("arm.clear_external_load_and_verify",))
    assert disable_index < clear_index
    assert sequence.state_history[-4:] == [
        SequenceState.LOWERING_LOAD,
        SequenceState.SETDOWN_SETTLING,
        SequenceState.SETDOWN_HOLDING,
        SequenceState.SETDOWN_COMPLETE,
    ]


def test_loaded_round_trip_reports_every_waypoint_and_stage_boundary():
    plan = _loaded_lift_plan()
    log = []
    observed = []

    def observer(stage, target_q, target_pose, hand_targets):
        observed.append((stage, target_q, target_pose, hand_targets))

    sequence = StagedGraspSequence(
        FakeArm(log, settled=(True, True, True, True)),
        FakeHand(log),
        boundary_observer=observer,
    )
    sequence.run_loaded_lift_joint_waypoints(plan)
    sequence.return_loaded_lift_to_setdown(plan)

    assert [item[0] for item in observed] == [
        "open",
        "default",
        "pregrasp",
        "pregrasp",
        "grasp",
        "grasp",
        "thumb_preshape",
        "close",
        "load_applied",
        "lift_transit_0",
        "lift",
        "lift",
        "setdown_lift_transit_0",
        "setdown_grasp",
        "setdown",
        "setdown_complete",
    ]
    lift = observed[11]
    np.testing.assert_array_equal(lift[1], plan.lift_waypoints[-1][1])
    np.testing.assert_array_equal(lift[2], plan.lift_pose)
    assert lift[3] == plan.grasp_plan.hand_target6
    assert observed[-1][3] == (-1, -1, -1, -1, -1, -1)


def test_boundary_observer_failure_during_lift_preserves_loaded_hold():
    plan = _loaded_lift_plan()
    log = []

    def observer(stage, _target_q, _target_pose, _hand_targets):
        if stage == "lift_transit_0":
            raise OSError("pose-state publish failed")

    sequence = StagedGraspSequence(
        FakeArm(log, settled=(True, True)),
        FakeHand(log),
        boundary_observer=observer,
    )

    with pytest.raises(SequenceExecutionError, match="LOADED HOLD") as raised:
        sequence.run_loaded_lift_joint_waypoints(plan)

    assert raised.value.manual_load_recovery_required
    assert ("arm.stop",) in log
    assert not any(item[0] == "hand.disable_and_verify" for item in log)
    assert not any(item[0] == "arm.clear_external_load_and_verify" for item in log)


def test_transition_observer_reports_state_before_each_stage_and_default_is_unchanged():
    log = []
    observed = []
    sequence = StagedGraspSequence(
        FakeArm(log),
        FakeHand(log),
        transition_observer=observed.append,
    )

    plan = DefaultSequencePlan(
        default_execution_eligible=True,
        default_q=np.asarray([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]),
    )
    sequence.run_to_default(plan)
    sequence.abort("test cleanup")

    assert observed == [
        SequenceState.OPENING_HAND,
        SequenceState.OPEN_VERIFIED,
        SequenceState.MOVING_FRANKA_DEFAULT,
        SequenceState.DEFAULT_VERIFIED,
        SequenceState.STOPPING,
        SequenceState.STOPPED,
    ]


def test_transition_observer_failure_fail_latches_but_cannot_block_stop_cleanup():
    log = []

    def observer(state):
        if state == SequenceState.OPEN_VERIFIED:
            raise OSError("telemetry stage update failed")

    sequence = StagedGraspSequence(
        FakeArm(log), FakeHand(log), transition_observer=observer
    )

    with pytest.raises(SequenceExecutionError, match="stage update failed"):
        sequence.run_to_default(
            DefaultSequencePlan(
                default_execution_eligible=True,
                default_q=np.asarray(
                    [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]
                ),
            )
        )

    assert sequence.state == SequenceState.FAULT_LATCHED
    assert ("arm.stop",) in log
    assert ("hand.disable_and_verify",) in log


def test_transition_observer_failure_on_stopping_is_best_effort_after_safe_stop():
    log = []

    def observer(state):
        if state == SequenceState.STOPPING:
            raise OSError("display already gone")

    sequence = StagedGraspSequence(
        FakeArm(log), FakeHand(log), transition_observer=observer
    )
    sequence.run_to_default(
        DefaultSequencePlan(
            default_execution_eligible=True,
            default_q=np.asarray([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]),
        )
    )

    assert sequence.abort("test cleanup") == SequenceState.STOPPED
    assert ("arm.stop",) in log
    assert ("hand.disable_and_verify",) in log
    assert isinstance(sequence._transition_observer_error, OSError)


def test_loaded_lift_motion_failure_stops_arm_but_never_disables_hold():
    plan = _loaded_lift_plan()

    class FailAtLiftArm(FakeArm):
        def move_joints(self, target_q):
            self.log.append(("arm.move_joints", tuple(float(v) for v in target_q)))
            if np.array_equal(np.asarray(target_q), plan.lift_waypoints[-1][1]):
                raise RuntimeError("lift motion failed")
            return np.asarray(target_q, dtype=np.float64)

    log = []
    sequence = StagedGraspSequence(
        FailAtLiftArm(log, settled=(True, True)), FakeHand(log)
    )

    with pytest.raises(SequenceExecutionError, match="LOADED HOLD") as raised:
        sequence.run_loaded_lift_joint_waypoints(plan)

    assert raised.value.manual_load_recovery_required
    assert sequence.requires_manual_load_recovery
    assert sequence.state == SequenceState.FAULT_LATCHED
    assert ("arm.stop",) in log
    assert not any(item[0] == "hand.disable_and_verify" for item in log)
    assert not any(item[0] == "arm.clear_external_load_and_verify" for item in log)
    assert log[-1][0] == "hand.verify_loaded_hold"


def test_loaded_lift_abort_is_fault_containment_and_preserves_hold():
    plan = _loaded_lift_plan()
    log = []
    sequence = StagedGraspSequence(
        FakeArm(log, settled=(True, True, True)), FakeHand(log)
    )
    sequence.run_loaded_lift_joint_waypoints(plan)
    before_abort = len(log)

    with pytest.raises(SequenceExecutionError) as raised:
        sequence.abort("operator interrupt")

    assert raised.value.manual_load_recovery_required
    assert log[before_abort:][0] == ("arm.stop",)
    assert log[before_abort:][1][0] == "hand.verify_loaded_hold"
    assert not any(
        item[0] in ("hand.disable_and_verify", "arm.clear_external_load_and_verify")
        for item in log[before_abort:]
    )


def test_loaded_lift_hold_monitor_refreshes_arm_and_hand_proofs():
    plan = _loaded_lift_plan()
    log = []
    sequence = StagedGraspSequence(
        FakeArm(log, settled=(True, True, True)), FakeHand(log)
    )
    sequence.run_loaded_lift_joint_waypoints(plan)
    before = len(log)

    assert (
        sequence.verify_loaded_lift_holding()
        == SequenceState.LIFTED_HOLDING
    )
    assert log[before:] == [
        ("arm.verify_idle_state",),
        (
            "hand.verify_loaded_hold",
            plan.grasp_plan.hand_target6,
            plan.minimum_contact_axes,
        ),
    ]


def test_loaded_lift_hold_monitor_failure_preserves_numeric_hold():
    plan = _loaded_lift_plan()
    log = []
    sequence = StagedGraspSequence(
        FakeArm(log, settled=(True, True, True), fail_on="verify_idle_state"),
        FakeHand(log),
    )
    sequence.run_loaded_lift_joint_waypoints(plan)

    with pytest.raises(SequenceExecutionError, match="LOADED HOLD") as raised:
        sequence.verify_loaded_lift_holding()

    assert raised.value.manual_load_recovery_required
    assert not any(item[0] == "hand.disable_and_verify" for item in log)
    assert not any(item[0] == "arm.clear_external_load_and_verify" for item in log)


def test_loaded_hold_failure_before_first_lift_uses_normal_disable():
    plan = _loaded_lift_plan()
    log = []
    sequence = StagedGraspSequence(
        FakeArm(log, settled=(True, True)),
        FakeHand(log, fail_on="verify_loaded_hold"),
    )

    with pytest.raises(SequenceExecutionError) as raised:
        sequence.run_loaded_lift_joint_waypoints(plan)

    assert not raised.value.manual_load_recovery_required
    assert not sequence.requires_manual_load_recovery
    assert log[-2:] == [("arm.stop",), ("hand.disable_and_verify",)]


def test_duck_typed_plan_attributes_are_supported():
    class ExternalPlan:
        execution_eligible = True
        franka_default_q = np.zeros(7)
        T_base_eef_pregrasp = _pose(0.4)
        T_base_eef_grasp = _pose(0.5)
        inspire_targets = (800, 800, 800, 800, 850, 950)
        requires_thumb_preshape = True

    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))

    assert sequence.run_full(ExternalPlan()) == SequenceState.HOLDING
    assert ("hand.preshape_thumb", 950) in log


def test_stage_based_grasp_execution_plan_is_supported_without_import_coupling():
    def stage(name, **values):
        return SimpleNamespace(name=name, **values)

    external = SimpleNamespace(
        execution_eligible=True,
        execution_blockers=(),
        stages=(
            stage("INSPIRE_OPEN", inspire_angles=np.full(6, 1000.0)),
            stage("FRANKA_DEFAULT", franka_q=np.zeros(7)),
            stage("FRANKA_PREGRASP", T_reference_EE=_pose(0.4)),
            stage("FRANKA_GRASP", T_reference_EE=_pose(0.5)),
            stage(
                "EEF_SETTLE_GATE",
                T_reference_EE=_pose(0.5),
                is_verification_gate=True,
            ),
            stage(
                "THUMB_PRESHAPE",
                inspire_angles=(1000, 1000, 1000, 1000, 1000, 900),
            ),
            stage(
                "INSPIRE_CLOSE",
                inspire_angles=(700, 710, 720, 730, 800, 900),
            ),
        ),
    )
    log = []
    sequence = StagedGraspSequence(FakeArm(log), FakeHand(log))

    assert sequence.run_full(external) == SequenceState.HOLDING
    assert ("hand.preshape_thumb", 900) in log
