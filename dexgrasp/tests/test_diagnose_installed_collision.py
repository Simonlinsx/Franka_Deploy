from pathlib import Path

import numpy as np

from apps import diagnose_installed_collision as app


def test_diagnostic_backend_request_binds_hand_feedback_uncertainties():
    hand = object()
    request = app._diagnostic_collision_request(
        mode="air_grasp",
        scene=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        object_points=np.asarray([[0.4, 0.5, 0.6]], dtype=np.float64),
        adapter_stl_path=Path("/tmp/adapter.stl"),
        max_q_tracking_error_rad=0.002,
        hand_arrival_tolerance_units=25,
        hand_model=hand,
    )
    assert request.hand_arrival_tolerance_units == 25
    assert request.hand_self_clearance_margin_m == 0.0
    assert request.q6_reverse_hysteresis_tolerance_units == 30
    assert request.hand_model is hand
