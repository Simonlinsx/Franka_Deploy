import numpy as np

from anydex_pipeline.snapshot_bridge import make_snapshot
from anydex_pipeline.types import GraspCandidate, GraspResult, PointCloudObservation


def test_make_snapshot_keeps_canonical_and_hand_poses_separate():
    points = np.asarray([[0.0, 0.0, 0.5], [0.01, 0.0, 0.5]], dtype=np.float32)
    observation = PointCloudObservation(
        scene_points=points,
        object_points=points,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
        frame_id=2,
        timestamp_s=1.5,
    )
    hand_pose = np.eye(4)
    hand_pose[:3, 3] = (0.1, 0.2, 0.3)
    candidate = GraspCandidate(
        T_reference_grasp=np.eye(4),
        T_reference_hand=hand_pose,
        hand_angles=np.arange(6),
        score=0.8,
    )
    result = GraspResult(
        candidates=(candidate,),
        backend_name="test",
        reference_frame="robot_base",
    )
    snapshot = make_snapshot(observation, result)
    np.testing.assert_array_equal(snapshot.grasps.canonical_poses[0], np.eye(4))
    np.testing.assert_array_equal(snapshot.grasps.hand_poses[0], hand_pose)
    np.testing.assert_array_equal(snapshot.grasps.hand_angles[0], np.arange(6))
    assert not snapshot.grasps.collision_checked[0]
    assert not snapshot.grasps.collision_free[0]


def test_make_snapshot_rejects_non_boolean_collision_claims():
    points = np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32)
    observation = PointCloudObservation(
        scene_points=points,
        object_points=points,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
    )
    candidate = GraspCandidate(
        T_reference_grasp=np.eye(4),
        score=0.8,
        collision_free=1,
    )
    result = GraspResult(
        candidates=(candidate,),
        backend_name="test",
        reference_frame="robot_base",
    )
    with np.testing.assert_raises_regex(ValueError, "collision_free must be boolean"):
        make_snapshot(observation, result)
