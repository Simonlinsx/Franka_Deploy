from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.contracts import (
    FrankaObservation,
    InspireObservation,
    ObjectPCDObservation,
    ObservationSpec,
    active_error_names,
    assemble_observation,
    franka_column_major_pose,
)


def _flat_identity(xyz=(0.0, 0.0, 0.0)):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = xyz
    return matrix.reshape(16, order="F")


class _Errors:
    joint_reflex = False
    cartesian_reflex = False

    def __bool__(self):
        return self.joint_reflex or self.cartesian_reflex


def _robot_state(**overrides):
    values = {
        "q": np.asarray([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]),
        "dq": np.zeros(7),
        "q_d": np.asarray([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]),
        "dq_d": np.zeros(7),
        "tau_J": np.zeros(7),
        "tau_ext_hat_filtered": np.zeros(7),
        "O_T_EE": _flat_identity((0.5, 0.0, 0.4)),
        "O_T_EE_d": _flat_identity((0.5, 0.0, 0.4)),
        "O_dP_EE_d": np.zeros(6),
        "O_F_ext_hat_K": np.zeros(6),
        "joint_contact": np.zeros(7),
        "joint_collision": np.zeros(7),
        "cartesian_contact": np.zeros(6),
        "cartesian_collision": np.zeros(6),
        "robot_mode": "RobotMode.kIdle",
        "current_errors": _Errors(),
        "control_command_success_rate": 1.0,
        "F_T_EE": _flat_identity(),
        "F_x_Cee": np.asarray([0.0, 0.0, 0.076]),
        "I_ee": np.diag([0.00151, 0.00169, 0.000442]).reshape(9, order="F"),
        "m_ee": 0.607,
        "m_load": 0.0,
        "m_total": 0.607,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _hand(timestamp=100.02, **overrides):
    snapshot = {
        "angle_targets": (-1,) * 6,
        "angles": (1000,) * 6,
        "positions": (0,) * 6,
        "forces": (0,) * 6,
        "currents": (0,) * 6,
        "errors": (0,) * 6,
        "statuses": (2,) * 6,
        "temperatures": (25,) * 6,
    }
    snapshot.update(overrides)
    return InspireObservation.from_snapshot(snapshot, timestamp)


def _packet(timestamp=100.0, **overrides):
    values = {
        "timestamp": timestamp,
        "frame_id": 12,
        "valid": True,
        "pcd_current": np.zeros((4, 3), dtype=np.float32),
        "pcd_history": np.zeros((2, 4, 3), dtype=np.float32),
        "pcd_reference": np.tile(np.asarray([0.5, 0.0, 0.1], dtype=np.float32), (4, 1)),
        "center": np.asarray([0.5, 0.0, 0.1], dtype=np.float32),
        "velocity": np.zeros(3, dtype=np.float32),
        "bbox_xyxy": np.asarray([1, 2, 3, 4], dtype=np.float32),
        "reference_frame": "robot_base",
        "point_frame": "robot_base",
        "calibration_id": "calibration-test",
        "camera_serial": "camera-test",
        "T_base_camera": np.eye(4, dtype=np.float32),
        "debug": {"raw_points": 300},
        "message": "ok",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _spec(**overrides):
    values = {
        "object_history_shape": (2, 4, 3),
        "object_current_shape": (4, 3),
        "calibration_id": "calibration-test",
        "camera_serial": "camera-test",
        "max_object_age_s": 0.15,
        "min_raw_points": 200,
        "expected_T_base_camera": np.eye(4),
        "max_capture_span_s": 0.5,
        "joint_limits_rad": np.asarray(
            [
                [-2.9, 2.9],
                [-1.8, 1.8],
                [-2.9, 2.9],
                [-3.0, -0.1],
                [-2.8, 2.8],
                [0.4, 4.6],
                [-3.0, 3.0],
            ]
        ),
        "expected_F_T_EE": np.eye(4),
        "expected_m_ee_kg": 0.607,
        "expected_F_x_Cee_m": np.asarray([0.0, 0.0, 0.076]),
        "expected_I_ee_kg_m2": np.diag([0.00151, 0.00169, 0.000442]),
    }
    values.update(overrides)
    return ObservationSpec(**values)


def test_franka_pose_is_column_major():
    matrix = franka_column_major_pose(_flat_identity((0.4, -0.1, 0.3)))
    np.testing.assert_allclose(matrix[:3, 3], [0.4, -0.1, 0.3])


def test_franka_pose_rejects_row_major_translation_layout():
    matrix = np.eye(4)
    matrix[:3, 3] = [0.4, -0.1, 0.3]
    with pytest.raises(ValueError, match="bottom row"):
        franka_column_major_pose(matrix.reshape(16, order="C"))


def test_false_python_error_fields_are_not_reported_as_unknown_error():
    assert active_error_names(SimpleNamespace(one=False, two=False)) == ()


def test_valid_observation_and_npz_contract():
    robot = FrankaObservation.from_state(_robot_state(), 100.01)
    hand = _hand()
    obj = ObjectPCDObservation.from_packet(_packet(), received_at_s=100.05)
    sample = assemble_observation(
        captured_at_s=100.06,
        spec=_spec(),
        franka=robot,
        inspire=hand,
        object_pcd=obj,
    )
    assert sample.valid
    assert sample.motion_safe
    assert sample.invalid_reasons == ()
    payload = sample.to_npz_payload()
    assert payload["robot_q"].shape == (7,)
    assert payload["hand_angles"].shape == (6,)
    assert payload["object_pcd_history"].shape == (2, 4, 3)
    assert payload["robot_q"].dtype == np.float32


def test_stale_wrong_frame_packet_fails_closed():
    robot = FrankaObservation.from_state(_robot_state(), 100.2)
    obj = ObjectPCDObservation.from_packet(
        _packet(reference_frame="camera_color_optical_frame"),
        received_at_s=100.2,
    )
    sample = assemble_observation(
        captured_at_s=100.2,
        spec=_spec(require_inspire=False),
        franka=robot,
        inspire=None,
        object_pcd=obj,
    )
    assert not sample.valid
    assert any("stale" in reason for reason in sample.invalid_reasons)
    assert any("reference_frame" in reason for reason in sample.invalid_reasons)


def test_faults_are_motion_blockers_without_corrupting_observation_data():
    errors = _Errors()
    errors.joint_reflex = True
    robot = FrankaObservation.from_state(
        _robot_state(current_errors=errors, joint_contact=np.ones(7)), 100.01
    )
    hand = _hand(errors=(0, 0, 1, 0, 0, 0))
    obj = ObjectPCDObservation.from_packet(_packet(), received_at_s=100.05)
    sample = assemble_observation(
        captured_at_s=100.06,
        spec=_spec(),
        franka=robot,
        inspire=hand,
        object_pcd=obj,
    )
    assert sample.valid
    assert not sample.motion_safe
    assert any("current_errors" in reason for reason in sample.motion_blockers)
    assert any("contact" in reason for reason in sample.motion_blockers)
    assert any("Inspire errors" in reason for reason in sample.motion_blockers)


def test_invalid_object_packet_must_not_retain_old_payload():
    with pytest.raises(ValueError, match="retained payload"):
        ObjectPCDObservation.from_packet(
            _packet(valid=False, center=np.zeros(3)), received_at_s=100.01
        )


def test_clean_invalid_object_packet_becomes_invalid_observation():
    packet = _packet(
        valid=False,
        pcd_current=None,
        pcd_history=None,
        pcd_reference=None,
        center=None,
        velocity=None,
        bbox_xyxy=None,
    )
    obj = ObjectPCDObservation.from_packet(packet, received_at_s=100.01)
    sample = assemble_observation(
        captured_at_s=100.02,
        spec=_spec(require_franka=False, require_inspire=False),
        franka=None,
        inspire=None,
        object_pcd=obj,
    )
    assert not sample.valid
    assert sample.invalid_reasons == ("object PCD packet is invalid",)
