from pathlib import Path

import numpy as np

from sim2real.deployment.rh56_action_audit import (
    audit_v94_rh56_action_mapping,
)
from sim2real.deployment.bundle import DeployBundle
from sim2real.contracts.actions import V94ActionMapper
from sim2real.contracts.v94 import (
    POLICY_HAND_ORDER,
    REGISTER_HAND_ORDER,
    V94Contract,
)
from sim2real.v94_kinematics import RH56FeedbackMapper


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data/test_fixtures/sim2real/deploy.zip"
DEPLOY_CONFIG = ROOT / "sim2real" / "v94_deploy_config.json"


def _mapper(initial_hand):
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    return V94ActionMapper(
        initial_arm_target_q_rad=np.zeros(7, dtype=np.float32),
        initial_hand_target_q_policy_order_rad=np.asarray(
            initial_hand, dtype=np.float32
        ),
        joint_limits_rad=np.asarray([[-3.0, 3.0]] * 7, dtype=np.float32),
        q_hand_close_rad=contract.q_hand_close_rad,
    ), contract


def test_offline_audit_proves_math_but_not_physical_commissioning():
    report = audit_v94_rh56_action_mapping(
        bundle_path=BUNDLE,
        deploy_config_path=DEPLOY_CONFIG,
    )
    assert report["result"] == "PASS"
    assert report["offline_only"] is True
    assert report["hardware_access"] is False
    assert report["hardware_writes"] is False
    assert report["mathematical_mapping_confirmed"] is True
    assert report["physical_mapping_confirmed"] is False
    assert report["semantic_contract"]["absolute_not_incremental"] is True
    assert tuple(report["semantic_contract"]["policy_order"]) == POLICY_HAND_ORDER
    assert tuple(report["semantic_contract"]["register_order"]) == REGISTER_HAND_ORDER
    assert all(
        stream["independent_registers_exact"]
        and stream["production_registers_exact"]
        and stream["independent_equations_vs_stored_target_max_abs_rad"] == 0.0
        for stream in report["simulator_reference_streams"]
    )
    assert report["simulator_previous_action_chain_exact"] is True
    assert report["dual_ack_protocol"][
        "single_device_ack_does_not_advance_previous_action"
    ] is True
    assert report["dual_ack_protocol"][
        "dual_ack_advances_previous_action_and_target"
    ] is True


def test_each_policy_axis_reaches_only_its_named_manufacturer_register():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    initial = np.float32(0.5) * contract.q_hand_close_rad
    register_index = {name: index for index, name in enumerate(REGISTER_HAND_ORDER)}
    base_mapper, _ = _mapper(initial)
    base = base_mapper.map(
        np.zeros(13, dtype=np.float32), measured_q_rad=np.zeros(7)
    )
    for policy_index, name in enumerate(POLICY_HAND_ORDER):
        mapper, _ = _mapper(initial)
        action = np.zeros(13, dtype=np.float32)
        action[7 + policy_index] = 1.0
        mapped = mapper.map(action, measured_q_rad=np.zeros(7))
        changed_virtual = np.flatnonzero(
            mapped.rh56_target_q_policy_order_rad
            != base.rh56_target_q_policy_order_rad
        )
        changed_register = np.flatnonzero(
            mapped.rh56_angle_set_register_order
            != base.rh56_angle_set_register_order
        )
        assert changed_virtual.tolist() == [policy_index]
        assert changed_register.tolist() == [register_index[name]]
        assert (
            mapped.rh56_angle_set_register_order[register_index[name]]
            < base.rh56_angle_set_register_order[register_index[name]]
        )


def test_neutral_hand_action_is_absolute_half_close_not_an_increment():
    mapper, contract = _mapper(np.zeros(6, dtype=np.float32))
    neutral = np.zeros(13, dtype=np.float32)
    mapped = None
    for _ in range(200):
        mapped = mapper.map(neutral, measured_q_rad=np.zeros(7))
    assert mapped is not None
    np.testing.assert_allclose(
        mapped.rh56_target_q_policy_order_rad,
        np.float32(0.5) * contract.q_hand_close_rad,
        atol=2.0e-6,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        mapped.rh56_angle_set_register_order,
        np.full(6, 500, dtype=np.int32),
    )


def test_command_and_feedback_orders_are_inverse_with_quantisation_bound():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    fractions = np.asarray(
        [0.123, 0.456, 0.789, 0.234, 0.567, 0.891], dtype=np.float32
    )
    target = fractions * contract.q_hand_close_rad
    register_policy = np.rint(
        np.float32(1000.0)
        * (np.float32(1.0) - target / contract.q_hand_close_rad)
    ).astype(np.int32)
    by_name = dict(zip(POLICY_HAND_ORDER, register_policy.tolist()))
    registers = np.asarray(
        [by_name[name] for name in REGISTER_HAND_ORDER], dtype=np.int32
    )
    recovered = RH56FeedbackMapper(
        q_hand_close_rad=contract.q_hand_close_rad
    ).map(registers)
    np.testing.assert_allclose(
        recovered.q_policy_order_rad,
        target,
        atol=float(np.max(contract.q_hand_close_rad) / 2000.0) + 1.0e-7,
        rtol=0.0,
    )
