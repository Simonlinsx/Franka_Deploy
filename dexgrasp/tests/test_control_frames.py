from __future__ import annotations

import numpy as np

from anydex_pipeline.control_frames import (
    T_MOUNT_SOURCE_AXIS_BASIS,
    compose_T_EE_hand_source,
    compose_T_flange_hand_source,
    compose_T_reference_EE,
)


def test_source_axis_basis_maps_wrist_to_fingers_onto_adapter_axis():
    rotation = T_MOUNT_SOURCE_AXIS_BASIS[:3, :3]
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(rotation @ [0, 1, 0], [1, 0, 0])
    np.testing.assert_allclose(rotation @ [0, 0, 1], [0, 1, 0])


def test_adapter_mount_plane_adds_10mm_not_total_17p8mm():
    transform = compose_T_flange_hand_source(
        fr3_face_to_rh56_seating_plane_m=0.010,
        assembled_yaw_rad=0.0,
        seating_to_source_origin_m=[0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(transform[:3, 3], [0.0, 0.0, 0.010])
    np.testing.assert_allclose(transform[:3, :3], T_MOUNT_SOURCE_AXIS_BASIS[:3, :3])


def test_installed_v7_right_hand_transform_is_directly_composable():
    T_F_hand = compose_T_flange_hand_source(
        fr3_face_to_rh56_seating_plane_m=0.010,
        assembled_yaw_rad=-np.pi / 4.0,
        seating_to_source_origin_m=[0.0, 0.0, 0.0],
    )
    root_half = np.sqrt(0.5)
    np.testing.assert_allclose(
        T_F_hand,
        [
            [0.0, root_half, root_half, 0.0],
            [0.0, -root_half, root_half, 0.0],
            [1.0, 0.0, 0.0, 0.010],
            [0.0, 0.0, 0.0, 1.0],
        ],
        atol=1e-12,
    )
    # AnyDex source +X is wrist-to-fingertips; +Z is index-to-pinky.
    np.testing.assert_allclose(T_F_hand[:3, :3] @ [1, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(
        T_F_hand[:3, :3] @ [0, 0, 1],
        [root_half, root_half, 0],
        atol=1e-12,
    )


def test_franka_configured_ee_is_kept_explicit_in_chain():
    F_T_EE = np.eye(4)
    F_T_EE[2, 3] = 0.020
    T_F_hand = compose_T_flange_hand_source(
        fr3_face_to_rh56_seating_plane_m=0.010,
        assembled_yaw_rad=np.pi / 2.0,
        seating_to_source_origin_m=[0.001, -0.002, 0.003],
    )
    T_EE_hand = compose_T_EE_hand_source(F_T_EE, T_F_hand)
    np.testing.assert_allclose(F_T_EE @ T_EE_hand, T_F_hand, atol=1e-12)

    T_base_hand = np.eye(4)
    T_base_hand[:3, 3] = [0.6, 0.2, 0.1]
    T_base_EE = compose_T_reference_EE(T_base_hand, T_EE_hand)
    np.testing.assert_allclose(T_base_EE @ T_EE_hand, T_base_hand, atol=1e-12)
