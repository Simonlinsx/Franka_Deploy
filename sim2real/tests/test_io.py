from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.io import (
    FrankaStateReader,
    InspireStateReader,
    ObjectPCDReader,
    _require_trusted_pcd_endpoint,
)


def _flat_identity():
    return np.eye(4).reshape(16, order="F")


def _state():
    errors = SimpleNamespace(joint_reflex=False)
    return SimpleNamespace(
        q=np.asarray([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]),
        dq=np.zeros(7),
        q_d=np.asarray([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]),
        dq_d=np.zeros(7),
        tau_J=np.zeros(7),
        tau_ext_hat_filtered=np.zeros(7),
        O_T_EE=_flat_identity(),
        O_T_EE_d=_flat_identity(),
        O_dP_EE_d=np.zeros(6),
        O_F_ext_hat_K=np.zeros(6),
        joint_contact=np.zeros(7),
        joint_collision=np.zeros(7),
        cartesian_contact=np.zeros(6),
        cartesian_collision=np.zeros(6),
        robot_mode="idle",
        current_errors=errors,
        control_command_success_rate=1.0,
        F_T_EE=_flat_identity(),
        F_x_Cee=np.asarray([0.0, 0.0, 0.076]),
        I_ee=np.diag([0.00151, 0.00169, 0.000442]).reshape(9, order="F"),
        m_ee=0.607,
        m_load=0.0,
        m_total=0.607,
    )


class _Robot:
    def __init__(self):
        self.reads = 0

    def read_once(self):
        self.reads += 1
        return _state()


class _Hand:
    def __init__(self):
        self.snapshots = 0

    def snapshot(self):
        self.snapshots += 1
        return {
            "angle_targets": (-1,) * 6,
            "angles": (1000,) * 6,
            "positions": (0,) * 6,
            "forces": (0,) * 6,
            "currents": (0,) * 6,
            "errors": (0,) * 6,
            "statuses": (2,) * 6,
            "temperatures": (25,) * 6,
        }


class _Subscriber:
    def __init__(self, packet):
        self.packet = packet
        self.closed = False

    def recv_latest(self, timeout_ms):
        assert timeout_ms == 25
        return self.packet

    def close(self):
        self.closed = True


def test_injected_state_readers_are_read_only():
    robot = _Robot()
    arm_reader = FrankaStateReader("unused", robot=robot)
    arm = arm_reader.read()
    assert robot.reads == 1
    assert arm.q.shape == (7,)

    hand = _Hand()
    hand_reader = InspireStateReader(port=None, hand=hand)
    observed = hand_reader.read()
    assert hand.snapshots == 1
    np.testing.assert_array_equal(observed.angle_targets, [-1] * 6)


def test_compact_policy_snapshot_reads_three_contiguous_blocks():
    api = SimpleNamespace(
        REG_ANGLE_SET=1486,
        REG_POS_ACT=1534,
        REG_ANGLE_ACT=1546,
        REG_FORCE_ACT=1582,
        REG_CURRENT=1594,
        REG_ERROR=1606,
        REG_STATUS=1612,
        REG_TEMP=1618,
    )
    targets = (-1, -1, -1, -1, -1, -1)
    positions = (10, 11, 12, 13, 14, 15)
    angles = (1000, 999, 998, 997, 996, 995)
    forces = (1, 2, 3, 4, 5, 6)
    currents = (-6, -5, -4, -3, -2, -1)
    safety = (
        struct.pack("<6h", *forces)
        + struct.pack("<6h", *currents)
        + bytes([0, 1, 0, 0, 0, 0])
        + bytes([2, 3, 2, 2, 2, 2])
        + bytes([25, 26, 27, 28, 29, 30])
    )

    class Hand:
        def __init__(self):
            self.reads = []

        def read(self, address, length, retries):
            self.reads.append((address, length, retries))
            values = {
                (1486, 12): struct.pack("<6h", *targets),
                (1534, 24): struct.pack("<12h", *(positions + angles)),
                (1582, 42): safety,
            }
            return values[(address, length)]

    hand = Hand()
    reader = InspireStateReader(
        port=None,
        hand=hand,
        api=api,
        snapshot_mode="compact_policy",
    )
    observed = reader.read()
    assert hand.reads == [(1486, 12, 0), (1534, 24, 0), (1582, 42, 0)]
    np.testing.assert_array_equal(observed.angle_targets, targets)
    np.testing.assert_array_equal(observed.positions, positions)
    np.testing.assert_array_equal(observed.angles, angles)
    np.testing.assert_array_equal(observed.forces, forces)
    np.testing.assert_array_equal(observed.currents, currents)
    np.testing.assert_array_equal(observed.errors, [0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(observed.statuses, [2, 3, 2, 2, 2, 2])
    np.testing.assert_array_equal(observed.temperatures_c, [25, 26, 27, 28, 29, 30])


def test_compact_policy_snapshot_rejects_register_map_drift():
    api = SimpleNamespace(
        REG_ANGLE_SET=1486,
        REG_POS_ACT=1534,
        REG_ANGLE_ACT=1547,
        REG_FORCE_ACT=1582,
        REG_CURRENT=1594,
        REG_ERROR=1606,
        REG_STATUS=1612,
        REG_TEMP=1618,
    )
    reader = InspireStateReader(
        port=None,
        hand=SimpleNamespace(),
        api=api,
        snapshot_mode="compact_policy",
    )
    with pytest.raises(RuntimeError, match="not contiguous"):
        reader.read()


def test_inspire_reader_rejects_unknown_snapshot_mode():
    with pytest.raises(ValueError, match="snapshot_mode"):
        InspireStateReader(port=None, hand=_Hand(), snapshot_mode="fastish")


def test_object_reader_uses_latest_packet_and_closes_subscriber():
    packet = SimpleNamespace(
        timestamp=10.0,
        frame_id=1,
        valid=False,
        pcd_current=None,
        pcd_history=None,
        pcd_reference=None,
        center=None,
        velocity=None,
        bbox_xyxy=None,
        reference_frame="robot_base",
        point_frame="robot_base",
        calibration_id="cal",
        camera_serial="serial",
        T_base_camera=np.eye(4),
        debug={"raw_points": 0},
        message="lost",
    )
    subscriber = _Subscriber(packet)
    reader = ObjectPCDReader(
        "tcp://127.0.0.1:5556", timeout_ms=25, subscriber=subscriber
    )
    observed = reader.read()
    assert observed is not None
    assert not observed.valid
    reader.close()
    assert subscriber.closed


@pytest.mark.parametrize(
    "addr",
    [
        "tcp://192.168.1.5:5556",
        "tcp://example.com:5556",
        "udp://127.0.0.1:5556",
    ],
)
def test_pickle_subscriber_rejects_untrusted_endpoint(addr):
    with pytest.raises(ValueError, match="pickle"):
        _require_trusted_pcd_endpoint(addr)


@pytest.mark.parametrize(
    "addr", ["tcp://127.0.0.1:5556", "tcp://localhost:5556", "ipc:///tmp/pcd.sock"]
)
def test_pickle_subscriber_accepts_local_endpoint(addr):
    _require_trusted_pcd_endpoint(addr)
