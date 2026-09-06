from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from anydex_pipeline.attached_cable_returns import (
    IDENTITY_CONFIRMATION_TOKEN,
    RIGID_ROUTE_CONFIRMATION_TOKEN,
    AttachedCableCapture,
    audit_attached_cable_returns,
)


ROOT = Path(__file__).resolve().parents[1]
FRESH2 = ROOT / "runs/live_scene_installed_current_fresh2_20260721.npz"
STAGE1 = ROOT / "runs/live_scene_installed_stage1_fresh_20260721.npz"


INTRINSICS = np.asarray([640.0, 480.0, 400.0, 400.0, 320.0, 240.0])
ROUTE_HAND = np.asarray([[-0.030, 0.0, 0.0], [0.030, 0.0, 0.0]])


def _T(x=0.0, y=0.0, z=0.0):
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = [x, y, z]
    return value


def _synthetic_capture(
    capture_id: str,
    T_base_hand: np.ndarray,
    *,
    component_hand: Optional[np.ndarray] = None,
    intrinsics: np.ndarray = INTRINSICS,
    bright: bool = False,
) -> AttachedCableCapture:
    if component_hand is None:
        x = np.linspace(-0.030, 0.030, 25)
        component_hand = np.column_stack(
            (x, 0.0005 * np.sin(x * 200.0), np.zeros_like(x))
        )
    component_base = (
        T_base_hand[:3, :3] @ np.asarray(component_hand).T
    ).T + T_base_hand[:3, 3]
    # Unlabelled points exercise the exact-index scope without influencing the
    # component proof.
    points = np.vstack((component_base, [[0.25, 0.20, 1.10], [-0.2, 0.1, 1.2]]))
    color = 0.9 if bright else 0.2
    colors = np.vstack(
        (
            np.full((len(component_base), 3), color, dtype=np.float64),
            np.asarray([[0.9, 0.3, 0.2], [0.1, 0.8, 0.1]]),
        )
    )
    return AttachedCableCapture(
        capture_id=capture_id,
        scene_points_base=points,
        scene_colors_srgb=colors,
        component_indices=np.arange(len(component_base), dtype=np.int64),
        T_base_hand=T_base_hand,
        T_base_camera=np.eye(4),
        camera_intrinsics=intrinsics,
        captured_at_unix_s=1000.0 + (1.0 if capture_id == "b" else 0.0),
    )


def _audit(captures, **overrides):
    values = {
        "route_centerline_hand_m": ROUTE_HAND,
        "route_outer_radius_m": 0.006,
        "identity_evidence_sha256": "1" * 64,
        "official_parent_link": "Link111",
        "official_mesh_relation_evidence_sha256": "2" * 64,
        "operator_identity_confirmation": IDENTITY_CONFIRMATION_TOKEN,
        "operator_rigid_route_confirmation": RIGID_ROUTE_CONFIRMATION_TOKEN,
    }
    values.update(overrides)
    return audit_attached_cable_returns(captures, **values)


def test_distinct_pose_rigid_dark_component_can_authorize_only_exact_return_exclusion():
    first = _synthetic_capture("a", _T(z=1.0))
    second = _synthetic_capture("b", _T(x=0.050, z=1.0))
    result = _audit([first, second])
    assert result.exclusion_authorized is True
    assert result.evidence["attached_return_exclusion_authorized"] is True
    assert result.evidence["excited_pair_count"] == 1
    assert result.evidence["moving_cable_collision_envelope_required"] is True
    assert result.evidence["motion_authorized"] is False
    assert "exact component_indices" in result.evidence["exclusion_scope"]


def test_same_pose_repeated_capture_cannot_prove_attachment():
    result = _audit(
        [_synthetic_capture("a", _T(z=1.0)), _synthetic_capture("b", _T(z=1.0))]
    )
    assert result.exclusion_authorized is False
    assert result.evidence["excited_pair_count"] == 0
    assert any("no capture pair" in item for item in result.evidence["failures"])


def test_static_dark_obstacle_does_not_pass_as_attached_after_hand_moves():
    first = _synthetic_capture("a", _T(z=1.0))
    first_hand = first.scene_points_base[first.component_indices] - np.asarray(
        [0.0, 0.0, 1.0]
    )
    # Keep the returns static in base while claiming the hand moved.  This
    # changes their hand-frame route and must fail even with both tokens.
    second = _synthetic_capture(
        "b", _T(x=0.050, z=1.0), component_hand=first_hand - [0.050, 0.0, 0.0]
    )
    result = _audit([first, second])
    assert result.exclusion_authorized is False
    assert any(
        "leaves the measured hand-frame cable route" in item
        or "does not beat static-obstacle model" in item
        for item in result.evidence["failures"]
    )


def test_camera_border_clipping_fails_even_when_comotion_is_perfect():
    border_intrinsics = INTRINSICS.copy()
    border_intrinsics[5] = 5.0
    result = _audit(
        [
            _synthetic_capture("a", _T(z=1.0), intrinsics=border_intrinsics),
            _synthetic_capture("b", _T(x=0.050, z=1.0), intrinsics=border_intrinsics),
        ]
    )
    assert result.exclusion_authorized is False
    assert any("clipped by camera border" in item for item in result.evidence["failures"])


def test_tokens_and_dark_appearance_are_independent_fail_closed_gates():
    captures = [
        _synthetic_capture("a", _T(z=1.0), bright=True),
        _synthetic_capture("b", _T(x=0.050, z=1.0), bright=True),
    ]
    result = _audit(
        captures,
        official_parent_link="Link44",
        operator_identity_confirmation="",
        operator_rigid_route_confirmation="",
    )
    assert result.exclusion_authorized is False
    failures = result.evidence["failures"]
    assert any("identity confirmation" in item for item in failures)
    assert any("rigid-route confirmation" in item for item in failures)
    assert any("Link111" in item for item in failures)
    assert any("not consistently dark" in item for item in failures)


@pytest.mark.skipif(
    not FRESH2.is_file() or not STAGE1.is_file(),
    reason="bound 2026-07-21 captures are unavailable",
)
def test_current_two_real_captures_cannot_certify_the_top_border_dark_return():
    """Lock the observed limitation: same pose and row-zero truncation."""

    T_fresh2_base_hand = np.asarray(
        [
            [0.02377458, 0.99626714, -0.08298533, 0.52645163],
            [0.02187607, -0.08350736, -0.99626701, -0.02017176],
            [-0.99947797, 0.02187044, -0.02377976, 0.59756681],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    T_stage1_base_hand = np.asarray(
        [
            [0.02377776, 0.99626726, -0.08298292, 0.52645142],
            [0.02187596, -0.08350503, -0.99626721, -0.02017256],
            [-0.99947789, 0.02187367, -0.02377987, 0.59756867],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    specs = (
        (
            "fresh2",
            FRESH2,
            np.asarray([0.559451, -0.050183, 0.466236]),
            T_fresh2_base_hand,
        ),
        (
            "stage1",
            STAGE1,
            np.asarray([0.556241, -0.052979, 0.464946]),
            T_stage1_base_hand,
        ),
    )
    captures = []
    for capture_id, path, witness, T_base_hand in specs:
        with np.load(path, allow_pickle=False) as archive:
            points = np.asarray(archive["scene_points"], dtype=np.float64)
            indices = np.flatnonzero(
                np.linalg.norm(points - witness[None, :], axis=1) <= 0.012
            ).astype(np.int64)
            captures.append(
                AttachedCableCapture(
                    capture_id=capture_id,
                    scene_points_base=points,
                    scene_colors_srgb=archive["scene_colors"],
                    component_indices=indices,
                    T_base_hand=T_base_hand,
                    T_base_camera=archive["T_reference_camera"],
                    camera_intrinsics=archive["camera_intrinsics"],
                    captured_at_unix_s=float(archive["captured_at_unix_s"]),
                )
            )
    # This is an analysis corridor, not a claimed measured route.  Supplying
    # even the affirmative tokens cannot overcome the geometric evidence.
    observed_corridor = np.asarray(
        [[0.132, 0.015, 0.032], [0.132, 0.042, 0.032]], dtype=np.float64
    )
    result = _audit(
        captures,
        route_centerline_hand_m=observed_corridor,
        route_outer_radius_m=0.010,
    )
    assert result.exclusion_authorized is False
    assert result.evidence["excited_pair_count"] == 0
    assert all(
        item["component_camera_border_margin_px"] < 2.0
        for item in result.evidence["captures"]
    )
    assert any("clipped by camera border" in item for item in result.evidence["failures"])
    assert any("no capture pair" in item for item in result.evidence["failures"])
    assert result.evidence["motion_authorized"] is False
