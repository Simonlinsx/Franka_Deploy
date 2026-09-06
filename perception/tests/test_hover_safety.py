from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from dynamic_pcd.apps.hover_over_object import (
    FR3_JOINT_LIMITS,
    MotionCancelled,
    PerceptionWatchdog,
    StableTarget,
    analyze_stability,
    execute_hover_one_shot,
    execute_waypoints,
    plan_hover_waypoints,
    stable_enough,
    validate_hover_config,
    validate_perception_packet,
    validate_robot_state,
)
from dynamic_pcd.calibration.io import ResolvedExtrinsics
from dynamic_pcd.types import ObjectPCDPacket


CALIBRATION_ID = "eye-to-hand-test"
CAMERA_SERIAL = "camera-test"


def _packet(*, now_s: float = 100.0) -> ObjectPCDPacket:
    points = np.zeros((300, 3), dtype=np.float32)
    points[:, 0] = np.linspace(0.54, 0.56, len(points))
    points[:, 1] = np.linspace(0.05, 0.07, len(points))
    points[:, 2] = np.linspace(0.10, 0.20, len(points))
    return ObjectPCDPacket(
        pcd_current=points.copy(),
        pcd_history=None,
        center=np.asarray([0.55, 0.06, 0.15], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        bbox_xyxy=np.asarray([10, 20, 30, 40], dtype=np.int32),
        timestamp=now_s - 0.05,
        frame_id=7,
        valid=True,
        debug={"raw_points": len(points)},
        pcd_reference=points,
        reference_frame="robot_base",
        point_frame="robot_base",
        calibration_id=CALIBRATION_ID,
        T_base_camera=np.eye(4, dtype=np.float32),
        camera_serial=CAMERA_SERIAL,
    )


def _validate(packet: ObjectPCDPacket, *, now_s: float = 100.0):
    return validate_perception_packet(
        packet,
        expected_calibration_id=CALIBRATION_ID,
        expected_camera_serial=CAMERA_SERIAL,
        expected_T_base_camera=np.eye(4),
        max_age_s=0.15,
        min_raw_points=200,
        now_s=now_s,
    )


def test_packet_gate_accepts_fresh_absolute_calibrated_cloud():
    packet = _packet()

    center, z_p95 = _validate(packet)

    np.testing.assert_allclose(center, [0.55, 0.06, 0.15], atol=1e-7)
    assert z_p95 == pytest.approx(
        float(np.percentile(packet.pcd_reference[:, 2], 95))
    )


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"valid": False}, "packet.valid"),
        ({"reference_frame": "camera_color_optical_frame"}, "reference_frame"),
        ({"calibration_id": "wrong"}, "calibration_id mismatch"),
        ({"camera_serial": "wrong"}, "camera serial mismatch"),
        ({"timestamp": 99.0}, "packet age"),
        ({"timestamp": 100.2}, "packet age"),
        ({"debug": {"raw_points": 199}}, "raw point count"),
        ({"pcd_reference": np.zeros((9, 3))}, "no usable"),
        ({"center": np.asarray([np.nan, 0.0, 0.0])}, "three finite"),
        (
            {"T_base_camera": np.eye(4) + np.diag([0.0, 0.0, 0.0, 0.1])},
            "last row",
        ),
    ],
)
def test_packet_gate_rejects_bad_provenance_or_data(changed, message):
    with pytest.raises(ValueError, match=message):
        _validate(replace(_packet(), **changed))


def test_packet_gate_rejects_missing_center_as_invalid_input():
    with pytest.raises(ValueError, match="packet.center"):
        _validate(replace(_packet(), center=None))


def _stability_config():
    return {
        "stable_frames": 30,
        "stable_seconds": 1.0,
        "max_center_p95": 0.005,
        "max_center_step": 0.010,
        "max_fitted_speed": 0.010,
    }


def test_stationary_target_passes_all_stability_gates():
    timestamps = np.linspace(20.0, 21.2, 31)
    phase = np.linspace(0.0, 2.0 * np.pi, len(timestamps))
    centers = np.column_stack(
        [
            0.55 + 0.0008 * np.sin(phase),
            0.06 + 0.0005 * np.cos(phase),
            0.12 + 0.0003 * np.sin(2.0 * phase),
        ]
    )
    target = analyze_stability(
        centers, np.full(len(centers), 0.16), timestamps
    )

    assert stable_enough(target, _stability_config())
    assert target.sample_count == 31
    assert target.duration_s == pytest.approx(1.2)
    assert target.center_p95_m < 0.005
    assert target.max_step_m < 0.010
    assert target.fitted_speed_mps < 0.010


def test_moving_target_is_rejected_by_stability_gate():
    timestamps = np.linspace(20.0, 21.2, 31)
    elapsed = timestamps - timestamps[0]
    centers = np.column_stack(
        [0.53 + 0.02 * elapsed, np.full(31, 0.06), np.full(31, 0.12)]
    )

    target = analyze_stability(
        centers, np.full(len(centers), 0.16), timestamps
    )

    assert target.fitted_speed_mps == pytest.approx(0.02, rel=1e-6)
    assert not stable_enough(target, _stability_config())


def test_stability_analysis_rejects_nonadvancing_timestamps():
    centers = np.tile([0.55, 0.06, 0.12], (3, 1))

    with pytest.raises(ValueError, match=r"not .*increasing"):
        analyze_stability(centers, np.full(3, 0.16), np.ones(3))


def _hover_config():
    return {
        "object_workspace_min": [0.505, 0.008, -0.05],
        "object_workspace_max": [0.588, 0.104, 0.30],
        "eef_workspace_min": [0.480, -0.020, 0.300],
        "eef_workspace_max": [0.610, 0.130, 0.360],
        "target_z_min": 0.320,
        "target_z_max": 0.345,
        "hover_clearance": 0.150,
        "max_segment_distance": 0.030,
    }


def _descent_hover_config():
    """A commissioned test envelope that explicitly permits guarded descent."""

    return {
        **_hover_config(),
        "eef_workspace_min": [0.480, -0.020, 0.200],
        "target_z_min": 0.200,
        "hover_clearance": 0.100,
        "descent_clearance_guard": 0.015,
        "descent_velocity": 0.005,
        "max_descent_xy_error": 0.005,
        "max_top_drift": 0.010,
    }


def _stable_target(center=(0.56, 0.08, 0.10), z_p95=0.16):
    return StableTarget(
        center=np.asarray(center, dtype=np.float64),
        z_p95=float(z_p95),
        sample_count=31,
        duration_s=1.2,
        center_p95_m=0.001,
        max_step_m=0.001,
        fitted_speed_mps=0.001,
    )


def test_hover_plan_rises_then_translates_in_bounded_segments():
    current = np.asarray([0.52, 0.02, 0.31])

    goal, waypoints = plan_hover_waypoints(
        current, _stable_target(), _hover_config()
    )

    np.testing.assert_allclose(goal, [0.56, 0.08, 0.32])
    assert waypoints
    np.testing.assert_allclose(waypoints[0][:2], current[:2])
    chain = [current, *waypoints]
    for start, end in zip(chain, chain[1:]):
        assert np.linalg.norm(end - start) <= 0.030000001
        assert end[2] >= start[2] - 1e-12
    np.testing.assert_allclose(waypoints[-1], goal)


def test_hover_plan_never_descends_from_an_already_high_eef():
    current = np.asarray([0.52, 0.02, 0.34])

    goal, waypoints = plan_hover_waypoints(
        current, _stable_target(z_p95=0.10), _hover_config()
    )

    assert goal[2] == pytest.approx(current[2])
    assert all(point[2] == pytest.approx(current[2]) for point in waypoints)


def test_guarded_descent_plan_aligns_high_then_descends_vertically():
    current = np.asarray([0.52, 0.02, 0.31])
    target = _stable_target(center=(0.56, 0.08, 0.10), z_p95=0.16)
    cfg = _descent_hover_config()

    goal, waypoints = plan_hover_waypoints(
        current,
        target,
        cfg,
        allow_descent=True,
    )

    expected_goal_z = target.z_p95 + 0.100 + cfg["descent_clearance_guard"]
    np.testing.assert_allclose(goal, [0.56, 0.08, expected_goal_z])
    chain = np.asarray([current, *waypoints])
    deltas = np.diff(chain, axis=0)
    descending = np.flatnonzero(deltas[:, 2] < -1e-12)
    assert len(descending) > 0
    first_descent = int(descending[0])

    # Transit rises first, completes XY alignment at >=0.320m, and only then
    # permits pure-z descent over the frozen object target.
    assert np.all(chain[: first_descent + 1, 2] >= current[2] - 1e-12)
    np.testing.assert_allclose(chain[first_descent, :2], goal[:2], atol=1e-12)
    assert chain[first_descent, 2] >= 0.320
    np.testing.assert_allclose(deltas[descending, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(chain[-1], goal, atol=1e-12)


def test_descent_is_still_disabled_unless_explicitly_requested():
    current = np.asarray([0.52, 0.02, 0.34])

    goal, waypoints = plan_hover_waypoints(
        current,
        _stable_target(z_p95=0.10),
        _descent_hover_config(),
    )

    assert goal[2] == pytest.approx(current[2])
    chain = np.asarray([current, *waypoints])
    assert np.all(np.diff(chain[:, 2]) >= -1e-12)


@pytest.mark.parametrize(
    ("current", "target", "message"),
    [
        ([0.52, 0.02, 0.31], _stable_target(center=(0.70, 0.08, 0.10)), "object workspace"),
        ([0.52, 0.02, 0.29], _stable_target(), "current EEF"),
        ([0.52, 0.02, 0.31], _stable_target(z_p95=0.25), "exceeds commissioned maximum"),
    ],
)
def test_hover_plan_rejects_uncommissioned_or_low_clearance_motion(
    current, target, message
):
    with pytest.raises(ValueError, match=message):
        plan_hover_waypoints(current, target, _hover_config())


class _OnePacketSubscriber:
    def __init__(self, packet):
        self.packet = packet

    def recv_latest(self, timeout_ms=0):
        packet, self.packet = self.packet, None
        return packet


def _resolved_extrinsics():
    return ResolvedExtrinsics(
        T_base_camera=np.eye(4, dtype=np.float32),
        calibrated=True,
        source="test",
        calibration_id=CALIBRATION_ID,
        base_frame="robot_base",
        camera_frame="camera_color_optical_frame",
        camera_serial=CAMERA_SERIAL,
        quality_status="pass",
    )


def test_motion_watchdog_rejects_target_drift():
    packet = _packet(now_s=time.time())
    packet.center = np.asarray([0.58, 0.06, 0.15])
    packet.pcd_reference[:, 0] += 0.03
    cfg = {
        "motion_max_packet_age": 0.25,
        "min_raw_points": 200,
        "max_target_drift": 0.020,
    }
    watchdog = PerceptionWatchdog(
        _OnePacketSubscriber(packet),
        _resolved_extrinsics(),
        cfg,
        frozen_center=np.asarray([0.55, 0.06, 0.15]),
    )

    with pytest.raises(RuntimeError, match="target drifted"):
        watchdog.check()


def _safe_robot_state(**changes):
    class _NoErrors:
        pass

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [0.54, 0.06, 0.33]
    values = {
        "robot_mode": "RobotMode.Idle",
        "current_errors": _NoErrors(),
        "cartesian_contact": np.zeros(6),
        "cartesian_collision": np.zeros(6),
        "joint_contact": np.zeros(7),
        "joint_collision": np.zeros(7),
        "control_command_success_rate": 1.0,
        "q": np.mean(FR3_JOINT_LIMITS, axis=1),
        "O_T_EE": pose.reshape(-1, order="F"),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_robot_state_gate_rejects_contact_and_near_joint_limit():
    contact = np.zeros(6)
    contact[2] = 1.0
    with pytest.raises(RuntimeError, match="cartesian_contact"):
        validate_robot_state(
            _safe_robot_state(cartesian_contact=contact),
            require_idle=True,
            min_joint_margin=0.05,
            min_success_rate=0.95,
        )

    near_limit = np.mean(FR3_JOINT_LIMITS, axis=1)
    near_limit[0] = FR3_JOINT_LIMITS[0, 0] + 0.01
    with pytest.raises(RuntimeError, match="joint 1"):
        validate_robot_state(
            _safe_robot_state(q=near_limit),
            require_idle=True,
            min_joint_margin=0.05,
            min_success_rate=0.95,
        )


def test_idle_zero_control_success_rate_is_allowed_but_move_zero_is_rejected():
    validate_robot_state(
        _safe_robot_state(
            robot_mode="RobotMode.Idle",
            control_command_success_rate=0.0,
        ),
        require_idle=True,
        min_joint_margin=0.05,
        min_success_rate=0.95,
    )

    with pytest.raises(RuntimeError, match="control command success rate"):
        validate_robot_state(
            _safe_robot_state(
                robot_mode="RobotMode.Move",
                control_command_success_rate=0.0,
            ),
            require_idle=False,
            min_joint_margin=0.05,
            min_success_rate=0.95,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"control_command_success_rate": np.nan},
        {"q": np.full(7, np.nan)},
        {"cartesian_contact": np.full(6, np.nan)},
        {"O_T_EE": np.full(16, np.nan)},
        {"current_errors": None},
    ],
)
def test_robot_state_gate_rejects_nonfinite_or_missing_safety_state(changes):
    with pytest.raises(RuntimeError):
        validate_robot_state(
            _safe_robot_state(**changes),
            require_idle=True,
            min_joint_margin=0.05,
            min_success_rate=0.95,
        )


def test_hover_config_rejects_silent_safety_envelope_expansion():
    cfg = {
        **_hover_config(),
        **_stability_config(),
        "max_packet_age": 0.15,
        "motion_max_packet_age": 0.25,
        "max_target_drift": 0.020,
        "min_raw_points": 200,
        "max_velocity": 0.010,
        "min_segment_duration": 3.0,
        "min_joint_margin": 0.05,
        "min_control_success_rate": 0.95,
        "max_final_error": 0.015,
    }
    validate_hover_config(cfg)
    expanded = dict(cfg)
    expanded["eef_workspace_max"] = [0.70, 0.13, 0.36]
    with pytest.raises(ValueError, match="commissioned envelope"):
        validate_hover_config(expanded)


def _pose_vector(xyz):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return pose.reshape(-1, order="F").tolist()


def _execution_config(**changes):
    cfg = {
        **_hover_config(),
        **_stability_config(),
        "max_packet_age": 0.15,
        "motion_max_packet_age": 0.25,
        "max_target_drift": 0.020,
        "min_raw_points": 200,
        "max_velocity": 0.010,
        "min_segment_duration": 3.0,
        "min_joint_margin": 0.05,
        "min_control_success_rate": 0.95,
        "max_final_error": 0.015,
    }
    cfg.update(changes)
    return cfg


def _descent_execution_config(**changes):
    cfg = {
        **_execution_config(),
        **_descent_hover_config(),
    }
    cfg.update(changes)
    return cfg


class _FreshPacketSubscriber:
    """A deterministic perception source that never becomes stale in a test."""

    def recv_latest(self, timeout_ms=0):
        return _packet(now_s=time.time())


class _FakePeriod:
    def __init__(self, seconds):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds


class _FakeCartesianPose:
    def __init__(self, pose):
        self.pose = list(pose)
        self.motion_finished = False


class _FakePylibfranka:
    ControllerMode = SimpleNamespace(CartesianImpedance="cartesian-impedance")
    CartesianPose = _FakeCartesianPose


class _FakeControl:
    def __init__(self, robot):
        self.robot = robot
        self.written_xyz = []

    def readOnce(self):
        return (
            _safe_robot_state(
                robot_mode="RobotMode.Move",
                O_T_EE=_pose_vector(self.robot.xyz),
                control_command_success_rate=self.robot.next_success_rate(),
            ),
            _FakePeriod(self.robot.next_period()),
        )

    def writeOnce(self, command):
        xyz = np.asarray(command.pose[12:15], dtype=np.float64)
        self.written_xyz.append(xyz.copy())
        if command.motion_finished:
            self.robot.finish_segment(xyz)


class _FakeRobot:
    """Minimal pylibfranka double with configurable control and tracking faults."""

    def __init__(
        self,
        xyz,
        *,
        periods=(0.02,),
        final_offsets=(),
        active_success_rates=(1.0,),
    ):
        self.xyz = np.asarray(xyz, dtype=np.float64).copy()
        self.periods = list(periods)
        self.period_index = 0
        self.active_success_rates = list(active_success_rates)
        self.success_rate_index = 0
        self.final_offsets = [
            np.asarray(offset, dtype=np.float64) for offset in final_offsets
        ]
        self.controls = []
        self.segment_targets = []
        self.stop_calls = 0

    def next_period(self):
        index = min(self.period_index, len(self.periods) - 1)
        self.period_index += 1
        return self.periods[index]

    def next_success_rate(self):
        index = min(
            self.success_rate_index, len(self.active_success_rates) - 1
        )
        self.success_rate_index += 1
        return self.active_success_rates[index]

    def read_once(self):
        return _safe_robot_state(O_T_EE=_pose_vector(self.xyz))

    def start_cartesian_pose_control(self, mode):
        assert mode == _FakePylibfranka.ControllerMode.CartesianImpedance
        control = _FakeControl(self)
        self.controls.append(control)
        return control

    def finish_segment(self, target):
        segment_index = len(self.segment_targets)
        self.segment_targets.append(np.asarray(target, dtype=np.float64).copy())
        offset = (
            self.final_offsets[segment_index]
            if segment_index < len(self.final_offsets)
            else np.zeros(3, dtype=np.float64)
        )
        self.xyz = np.asarray(target, dtype=np.float64) + offset

    def stop(self):
        self.stop_calls += 1


def _execute(robot, waypoints, *, config=None, **execution_kwargs):
    return execute_waypoints(
        robot,
        _FakePylibfranka,
        [np.asarray(point, dtype=np.float64) for point in waypoints],
        _FreshPacketSubscriber(),
        _resolved_extrinsics(),
        _execution_config() if config is None else config,
        frozen_center=np.asarray([0.55, 0.06, 0.15], dtype=np.float64),
        **execution_kwargs,
    )


@pytest.mark.parametrize(
    ("periods", "message"),
    [
        ([np.nan], "unsafe Franka control period"),
        ([-0.001], "unsafe Franka control period"),
        ([0.020001], "unsafe Franka control period"),
        ([0.0, 0.0], "repeated zero Franka control period"),
    ],
)
def test_execute_rejects_unsafe_control_period_and_stops_robot(periods, message):
    robot = _FakeRobot([0.54, 0.06, 0.33], periods=periods)

    with pytest.raises(RuntimeError, match=message):
        _execute(robot, [[0.541, 0.06, 0.33]])

    assert robot.stop_calls == 1
    assert robot.segment_targets == []


def test_execute_allows_initial_zero_success_rate_then_enforces_it():
    robot = _FakeRobot(
        [0.54, 0.06, 0.33],
        periods=[0.02],
        active_success_rates=[0.0, 1.0],
    )

    final_xyz = _execute(robot, [[0.541, 0.06, 0.33]])

    np.testing.assert_allclose(final_xyz, [0.541, 0.06, 0.33], atol=1e-12)
    assert robot.stop_calls == 0


def test_execute_rejects_persistent_zero_success_rate_after_warmup():
    robot = _FakeRobot(
        [0.54, 0.06, 0.33],
        periods=[0.02],
        active_success_rates=[0.0],
    )

    with pytest.raises(RuntimeError, match="control command success rate"):
        _execute(robot, [[0.541, 0.06, 0.33]])

    assert robot.stop_calls == 1
    assert robot.segment_targets == []


def test_execute_subdivides_from_each_actual_robot_start():
    robot = _FakeRobot([0.50, 0.02, 0.33])

    final_xyz = _execute(robot, [[0.585, 0.02, 0.33]])

    targets = np.asarray(robot.segment_targets)
    np.testing.assert_allclose(
        targets,
        [
            [0.530, 0.02, 0.33],
            [0.560, 0.02, 0.33],
            [0.585, 0.02, 0.33],
        ],
        atol=1e-9,
    )
    starts = np.vstack([[0.50, 0.02, 0.33], targets[:-1]])
    assert np.all(np.linalg.norm(targets - starts, axis=1) <= 0.030000001)
    np.testing.assert_allclose(final_xyz, targets[-1])


def test_execute_clamps_a_downward_waypoint_to_actual_start_height():
    start = np.asarray([0.54, 0.06, 0.335])
    robot = _FakeRobot(start)

    final_xyz = _execute(robot, [[0.545, 0.06, 0.320]])

    assert len(robot.segment_targets) == 1
    assert robot.segment_targets[0][2] == pytest.approx(start[2])
    assert all(
        command_xyz[2] >= start[2] - 1e-12
        for command_xyz in robot.controls[0].written_xyz
    )
    assert final_xyz[2] == pytest.approx(start[2])


def test_guarded_descent_corrects_xy_at_safe_height_before_lowering_z():
    start = np.asarray([0.54, 0.06, 0.32])
    robot = _FakeRobot(start)
    floor_z = 0.275

    final_xyz = _execute(
        robot,
        [[0.55, 0.06, floor_z]],
        config=_descent_execution_config(),
        allow_descent=True,
        frozen_z_p95=0.20,
        minimum_descent_z=floor_z,
    )

    targets = np.asarray(robot.segment_targets)
    assert len(targets) >= 2
    # The first command fixes XY without reducing height. Every subsequent
    # lowering command is vertical and remains exactly over the frozen target.
    np.testing.assert_allclose(targets[0], [0.55, 0.06, start[2]], atol=1e-12)
    lowering = np.flatnonzero(np.diff(np.r_[start[2], targets[:, 2]]) < -1e-12)
    assert len(lowering) > 0
    np.testing.assert_allclose(
        targets[lowering, :2],
        np.tile([0.55, 0.06], (len(lowering), 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(final_xyz, [0.55, 0.06, floor_z], atol=1e-12)


def test_guarded_descent_treats_submillimetre_z_readback_delta_as_horizontal():
    """Regression: planning/readback noise must not misclassify XY transit."""

    start = np.asarray([0.54, 0.06, 0.33])
    robot = _FakeRobot(start)
    floor_z = 0.275

    final_xyz = _execute(
        robot,
        [[0.545, 0.06, start[2] - 1e-6]],
        config=_descent_execution_config(),
        allow_descent=True,
        frozen_z_p95=0.20,
        minimum_descent_z=floor_z,
    )

    assert len(robot.segment_targets) == 1
    np.testing.assert_allclose(
        robot.segment_targets[0], [0.545, 0.06, start[2]], atol=1e-12
    )
    np.testing.assert_allclose(final_xyz, robot.segment_targets[0], atol=1e-12)


def test_guarded_descent_preserves_height_for_high_xy_transit_readback_error():
    """A millimetre-scale Z offset must not turn XY transit into descent."""

    start = np.asarray([0.54, 0.06, 0.3315])
    robot = _FakeRobot(start)
    floor_z = 0.275

    final_xyz = _execute(
        robot,
        [[0.545, 0.06, 0.3300]],
        config=_descent_execution_config(),
        allow_descent=True,
        frozen_z_p95=0.20,
        minimum_descent_z=floor_z,
    )

    assert len(robot.segment_targets) == 1
    np.testing.assert_allclose(
        robot.segment_targets[0], [0.545, 0.06, start[2]], atol=1e-12
    )
    np.testing.assert_allclose(final_xyz, robot.segment_targets[0], atol=1e-12)


def test_guarded_descent_rejects_waypoint_below_requested_floor():
    robot = _FakeRobot([0.55, 0.06, 0.32])
    floor_z = 0.275

    with pytest.raises(RuntimeError, match="violates the guarded z floor"):
        _execute(
            robot,
            [[0.55, 0.06, floor_z - 0.001]],
            config=_descent_execution_config(),
            allow_descent=True,
            frozen_z_p95=0.20,
            minimum_descent_z=floor_z,
        )

    assert robot.segment_targets == []


def test_execute_rejects_post_segment_tracking_error_without_continuing():
    robot = _FakeRobot(
        [0.54, 0.06, 0.33],
        final_offsets=[[0.016, 0.0, 0.0]],
    )

    with pytest.raises(RuntimeError, match="waypoint tracking error .*not continuing"):
        _execute(
            robot,
            [
                [0.545, 0.06, 0.33],
                [0.550, 0.06, 0.33],
            ],
        )

    assert len(robot.controls) == 1
    assert len(robot.segment_targets) == 1


def test_execute_cooperative_cancel_stops_active_robot_control():
    cancel_event = threading.Event()

    class _CancellingControl(_FakeControl):
        def readOnce(self):
            state_and_period = super().readOnce()
            cancel_event.set()
            return state_and_period

    class _CancellingRobot(_FakeRobot):
        def start_cartesian_pose_control(self, mode):
            assert mode == _FakePylibfranka.ControllerMode.CartesianImpedance
            control = _CancellingControl(self)
            self.controls.append(control)
            return control

    robot = _CancellingRobot([0.54, 0.06, 0.33])

    with pytest.raises(MotionCancelled, match="cancelled by operator"):
        _execute(
            robot,
            [[0.55, 0.06, 0.33]],
            cancel_event=cancel_event,
        )

    assert robot.stop_calls == 1
    assert robot.segment_targets == []
    assert len(robot.controls) == 1


def test_execute_keyboard_interrupt_stops_active_robot_control():
    class _InterruptedControl(_FakeControl):
        def readOnce(self):
            raise KeyboardInterrupt

    class _InterruptedRobot(_FakeRobot):
        def start_cartesian_pose_control(self, mode):
            assert mode == _FakePylibfranka.ControllerMode.CartesianImpedance
            control = _InterruptedControl(self)
            self.controls.append(control)
            return control

    robot = _InterruptedRobot([0.54, 0.06, 0.33])

    with pytest.raises(KeyboardInterrupt):
        _execute(robot, [[0.55, 0.06, 0.33]])

    assert robot.stop_calls == 1
    assert robot.segment_targets == []


def test_one_shot_callable_rises_then_moves_xy_then_descends():
    start = np.asarray([0.52, 0.02, 0.22], dtype=np.float64)
    target = _stable_target(center=(0.55, 0.06, 0.15), z_p95=0.195)
    robot = _FakeRobot(start)

    result = execute_hover_one_shot(
        robot,
        _FakePylibfranka,
        _FreshPacketSubscriber(),
        _resolved_extrinsics(),
        _descent_execution_config(),
        target,
        allow_descent=True,
    )

    targets = np.asarray(robot.segment_targets)
    assert len(targets) > 2
    np.testing.assert_allclose(result.current_xyz, start, atol=1e-12)
    np.testing.assert_allclose(result.goal_xyz, [0.55, 0.06, 0.31], atol=1e-12)
    np.testing.assert_allclose(result.final_xyz, result.goal_xyz, atol=1e-12)
    assert result.waypoint_count > 0

    xy_moved = np.linalg.norm(targets[:, :2] - start[:2], axis=1) > 1e-9
    first_xy = int(np.flatnonzero(xy_moved)[0])
    np.testing.assert_allclose(
        targets[:first_xy, :2],
        np.tile(start[:2], (first_xy, 1)),
        atol=1e-12,
    )
    assert np.all(np.diff(np.r_[start[2], targets[:first_xy, 2]]) > 0.0)
    assert targets[first_xy - 1, 2] >= 0.320 - 1e-12
    assert targets[first_xy, 2] >= 0.320 - 1e-12

    lowering = np.flatnonzero(np.diff(np.r_[start[2], targets[:, 2]]) < -1e-9)
    assert len(lowering) > 0
    np.testing.assert_allclose(
        targets[lowering, :2],
        np.tile(target.center[:2], (len(lowering), 1)),
        atol=1e-12,
    )


def test_one_shot_callable_rejects_unstable_target_before_robot_motion():
    robot = _FakeRobot([0.52, 0.02, 0.32])
    unstable = replace(_stable_target(), sample_count=2)

    with pytest.raises(ValueError, match="requires a commissioned stable target"):
        execute_hover_one_shot(
            robot,
            _FakePylibfranka,
            _FreshPacketSubscriber(),
            _resolved_extrinsics(),
            _execution_config(),
            unstable,
        )

    assert robot.controls == []
    assert robot.stop_calls == 0


def test_one_shot_callable_rejects_unapproved_calibration_before_robot_read():
    class _NoReadRobot(_FakeRobot):
        def read_once(self):
            raise AssertionError("robot state must not be read")

    robot = _NoReadRobot([0.52, 0.02, 0.32])
    failed_quality = replace(_resolved_extrinsics(), quality_status="fail")

    with pytest.raises(RuntimeError, match="calibration quality status pass"):
        execute_hover_one_shot(
            robot,
            _FakePylibfranka,
            _FreshPacketSubscriber(),
            failed_quality,
            _execution_config(),
            _stable_target(),
        )

    assert robot.controls == []
    assert robot.stop_calls == 0
