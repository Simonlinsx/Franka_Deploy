import numpy as np

from anydex_pipeline.backends.geometric import GeometricGraspBackend
from anydex_pipeline.frames import change_pose_reference, inverse_transform, transform_points
from anydex_pipeline.types import PointCloudObservation


def _box_points(seed=4):
    rng = np.random.default_rng(seed)
    points = rng.uniform((-0.04, -0.025, -0.015), (0.04, 0.025, 0.015), (1500, 3))
    points += np.array((0.55, -0.08, 0.18))
    return points.astype(np.float32)


def test_frame_round_trip():
    angle = 0.37
    transform = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 0.2],
            [np.sin(angle), np.cos(angle), 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    points = _box_points()[:20]
    recovered = transform_points(inverse_transform(transform), transform_points(transform, points))
    np.testing.assert_allclose(recovered, points, atol=1e-6)
    np.testing.assert_allclose(
        change_pose_reference(transform, inverse_transform(transform)), np.eye(4), atol=1e-7
    )


def test_geometric_backend_returns_valid_ranked_poses():
    points = _box_points()
    observation = PointCloudObservation(
        scene_points=points,
        object_points=points,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
    )
    result = GeometricGraspBackend(top_k=5).infer(observation)
    assert result.backend_name == "geometric_demo"
    assert len(result.candidates) == 5
    scores = [candidate.score for candidate in result.candidates]
    assert scores == sorted(scores, reverse=True)
    for candidate in result.candidates:
        rotation = candidate.T_reference_grasp[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        assert np.linalg.det(rotation) > 0.999
        assert 0.025 <= candidate.width_m <= 0.10
