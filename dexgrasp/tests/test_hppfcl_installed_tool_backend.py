from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from anydex_pipeline.hppfcl_installed_tool_backend import (
    ADAPTER_FR3_MOUNT_EXCLUSIONS,
    DEFAULT_FR3_URDF,
    DEFAULT_FRANKA_DESCRIPTION_SHARE,
    HppFclInstalledToolConfig,
    RH56_ADAPTER_MOUNT_EXCLUSIONS,
    T_EE_ADAPTER,
)


ROOT = Path(__file__).resolve().parents[1]
ROS_PYTHON_PATHS = (
    "/opt/ros/humble/lib/python3.10/site-packages",
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
)


def _native_runtime_environment():
    environment = dict(os.environ)
    entries = list(ROS_PYTHON_PATHS)
    if environment.get("PYTHONPATH"):
        entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _native_runtime_available():
    if not Path("/usr/bin/python3").is_file():
        return False
    result = subprocess.run(
        ["/usr/bin/python3", "-c", "import pinocchio, hppfcl"],
        env=_native_runtime_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    return result.returncode == 0


def test_fixed_mount_exclusions_and_adapter_frame_are_explicit():
    assert ADAPTER_FR3_MOUNT_EXCLUSIONS == ("link7", "link8")
    assert RH56_ADAPTER_MOUNT_EXCLUSIONS == ("Link111",)
    np.testing.assert_allclose(T_EE_ADAPTER[:3, 2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        T_EE_ADAPTER[:3, :3].T @ T_EE_ADAPTER[:3, :3],
        np.eye(3),
        atol=1e-12,
    )
    assert np.isclose(np.linalg.det(T_EE_ADAPTER[:3, :3]), 1.0)


@pytest.mark.parametrize(
    "updates, match",
    [
        ({"point_cloud_observations_authoritative": True}, "cannot be promoted"),
        ({"joint_tracking_uncertainty_applied": False}, "cannot be disabled"),
        ({"continuous_segment_envelope_verified": False}, "cannot be disabled"),
    ],
)
def test_unsupported_authority_promotions_are_locked(updates, match):
    with pytest.raises(ValueError, match=match):
        HppFclInstalledToolConfig(**updates)


@pytest.mark.skipif(
    not _native_runtime_available()
    or not DEFAULT_FR3_URDF.is_file()
    or not DEFAULT_FRANKA_DESCRIPTION_SHARE.is_dir(),
    reason="native ROS Humble FR3 collision runtime is not installed",
)
def test_native_backend_loads_real_meshes_and_returns_every_check():
    """Run with the ABI-compatible ROS Python, not the project Python 3.9."""

    program = r"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from anydex_pipeline.hppfcl_installed_tool_backend import HppFclInstalledToolBackend
from anydex_pipeline.installed_tool_audit import CHECK_SPECS, V7_T_EE_HAND, check_specs_for_mode
from anydex_pipeline.inspire_hand_model import InspireHandModel
from anydex_pipeline.inspire_open_configuration import OFFICIAL_OPEN_JOINT_POSITIONS_RAD
from anydex_pipeline.installed_scene_filter import filter_installed_scene

root = Path(__import__('os').environ['DEXGRASP_TEST_ROOT'])
hand = InspireHandModel.from_anydex_root(
    root / 'third_party/AnyDexGrasp', mesh_resolution='simplified'
)
transforms = hand.link_mesh_transforms(
    np.eye(4), OFFICIAL_OPEN_JOINT_POSITIONS_RAD
)
mesh_dir = hand.urdf_path.parent.parent / 'meshes_simplified'
paths = {link.name: (mesh_dir / link.mesh_filename).resolve() for link in hand.links}
request = SimpleNamespace(
    mode='loaded_grasp',
    scene_points_base=np.asarray([[5.0, 5.0, 5.0]]),
    object_points_base=np.asarray([[5.1, 5.0, 5.0]]),
    adapter_stl_path=root / 'assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl',
    T_EE_hand=V7_T_EE_HAND,
    allowed_object_contact_links=('Link11', 'Link22', 'Link33', 'Link44', 'Link53'),
    max_q_tracking_error_rad=0.002,
    hand_self_clearance_margin_m=0.0,
    hand_arrival_tolerance_units=25,
    q6_reverse_hysteresis_tolerance_units=30,
    scene_voxel_resolution_m=0.005,
    observed_scene_scope='calibrated_camera_frustum_voxel_grid',
    unknown_space_policy='occupied',
    hand_model=hand,
)
query = SimpleNamespace(
    request=request,
    q_path_rad=np.asarray([[0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]]),
    hand_link_mesh_paths=paths,
    T_hand_open_link_visual=transforms,
    T_hand_closed_link_visual=transforms,
    T_hand_waypoint_link_visual=(transforms, transforms),
    hand_dense_interval_q12_rad=(
        np.vstack((OFFICIAL_OPEN_JOINT_POSITIONS_RAD, OFFICIAL_OPEN_JOINT_POSITIONS_RAD)),
    ),
    hand_interval_feedback_tube_q12_rad=(np.zeros(12, dtype=np.float64),),
)
backend = HppFclInstalledToolBackend()
assert len(backend.visual_model.geometryObjects) == 8
assert backend._robot_visual_link_names == tuple('link{}'.format(i) for i in range(8))
observations = backend.evaluate(query)
assert tuple(observations) == tuple(item.check_id for item in CHECK_SPECS)
assert observations['fr3_self_path'].authoritative is True
assert observations['adapter_fr3_path'].authoritative is True
assert observations['fr3_scene_path'].authoritative is False
assert observations['adapter_object_path'].authoritative is False
assert all(item.details['joint_tracking_uncertainty_applied'] is True for item in observations.values())
assert all(item.details['continuous_segment_envelope_verified'] is True for item in observations.values())
assert all(item.details['minimum_distance_is_after_motion_bound'] is True for item in observations.values())
assert all(item.details['conservative_motion_bound_m'] >= 0.0 for item in observations.values())
assert observations['fr3_scene_path'].details['conservative_motion_bound_m'] > 0.0
assert observations['fr3_self_path'].minimum_signed_distance_m > 0.0
assert observations['adapter_fr3_path'].details['fixed_mount_pair_exclusions'] == ['link7', 'link8']
assert observations['rh56_closed_adapter_final'].details['fixed_mount_pair_exclusions'] == ['Link111']
request.mode = 'air_grasp'
air = backend.evaluate(query)
assert tuple(air) == tuple(item.check_id for item in check_specs_for_mode('air_grasp'))
assert 'rh56_closed_object_contact_final' not in air
assert 'rh56_closed_object_all_links_final' in air
assert air['rh56_closed_object_all_links_final'].authoritative is False
for check_id in ('rh56_execution_scene_final', 'rh56_execution_object_final'):
    assert air[check_id].authoritative is False
    assert (
        'single captured point cloud does not prove occluded space empty'
        in air[check_id].details['authority_limitations']
    )
    assert air[check_id].details['geometry_model'] == (
        'exact mesh versus conservative captured-point cube union'
    )
filtered = filter_installed_scene(
    backend,
    np.asarray([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [6.0, 6.0, 6.0]]),
    np.asarray([[5.1, 5.0, 5.0]]),
    np.asarray([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]),
    adapter_stl_path=request.adapter_stl_path,
    T_EE_hand=V7_T_EE_HAND,
    hand_link_mesh_paths=paths,
    T_hand_open_link_visual=transforms,
)
assert filtered.object_return_indices.tolist() == [1]
assert filtered.installed_return_indices.tolist() == []
assert filtered.kept_original_indices.tolist() == [0, 2]
assert filtered.evidence['motion_authorized'] is False
assert filtered.evidence['installed_return_filter']['fr3_geometry_model'] == 'official_fr3_visual_triangle_meshes'
assert len(filtered.evidence['installed_return_filter']['fr3_visual_meshes']) == 8

# A replayed white-shell return that the coarse collision meshes missed must
# be classified by the official visual DAE shell at the exact capture q.
capture_q = np.asarray([
    -0.1118436, -0.1207545, 0.0739457, -1.7431009,
    0.0463540, 1.6809169, 0.8117281,
])
white_shell = np.asarray([0.04816544055938721, 0.05993962287902832, 0.3090685307979584])
classified = backend.classify_installed_depth_returns(
    np.asarray([white_shell, [5.0, 5.0, 5.0]]),
    capture_q,
    adapter_stl_path=request.adapter_stl_path,
    T_EE_hand=V7_T_EE_HAND,
    hand_link_mesh_paths=paths,
    T_hand_open_link_visual=transforms,
    point_half_extent_m=0.0025,
    inflation_margin_m=0.002,
)
assert classified.removed_indices.tolist() == [0]
assert classified.reason_labels == ('FR3_visual_link2',)
assert classified.fr3_geometry_model == 'official_fr3_visual_triangle_meshes'
assert len(classified.fr3_mesh_provenance) == 8
neighborhood = backend.installed_depth_return_neighborhood(
    np.asarray([white_shell, [5.0, 5.0, 5.0]]),
    capture_q,
    adapter_stl_path=request.adapter_stl_path,
    T_EE_hand=V7_T_EE_HAND,
    hand_link_mesh_paths=paths,
    T_hand_open_link_visual=transforms,
    maximum_surface_distance_m=0.02,
)
assert neighborhood.candidate_indices.tolist() == [0]
assert neighborhood.reason_labels == ('FR3_visual_link2',)
assert neighborhood.surface_distances_m[0] <= 0.02
assert neighborhood.distance_method == 'HPP-FCL triangle-mesh to 1nm sphere, radius corrected'
"""
    environment = _native_runtime_environment()
    environment["DEXGRASP_TEST_ROOT"] = str(ROOT)
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment["PYTHONPATH"]
    result = subprocess.run(
        ["/usr/bin/python3", "-c", program],
        cwd=str(ROOT),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
