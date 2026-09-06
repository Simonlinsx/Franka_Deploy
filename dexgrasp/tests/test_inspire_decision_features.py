import numpy as np

import pytest

from anydex_pipeline.inspire_decision import OfficialInspireDecisionBackend
from anydex_pipeline.official_backend import (
    OFFICIAL_NO_FLIP_FEATURE_WIDTH,
    RepresentationFeatureLayout,
    RepresentationGraspBatch,
    _official_right_hand_orientation_mask,
)


def _batch(features):
    count = features.shape[0]
    grasps = np.zeros((count, 17), dtype=np.float32)
    points = np.zeros((max(count, 1), 3), dtype=np.float32)
    return RepresentationGraspBatch(
        grasps=grasps,
        features=features,
        voxel_points_camera=points,
        feature_layout=(
            RepresentationFeatureLayout.OFFICIAL_RIGHT_HAND_NO_FLIP_V1
        ),
    )


def test_decision_features_rotate_each_240_half_by_angle_times_depth_count():
    features = np.zeros((1, OFFICIAL_NO_FLIP_FEATURE_WIDTH), dtype=np.float32)
    features[0, 240:480] = np.arange(240)
    features[0, 480:720] = 1000 + np.arange(240)
    features[0, -2] = 2
    features[0, -1] = 0.03
    rotated, depths = OfficialInspireDecisionBackend._decision_features(
        _batch(features)
    )
    np.testing.assert_array_equal(rotated[0, :240], np.roll(np.arange(240), -10))
    np.testing.assert_array_equal(
        rotated[0, 240:], np.roll(1000 + np.arange(240), -10)
    )
    np.testing.assert_array_equal(depths, (3,))


def test_representation_batch_rejects_ambiguous_feature_layout_string():
    features = np.zeros((1, OFFICIAL_NO_FLIP_FEATURE_WIDTH), dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported representation feature layout"):
        RepresentationGraspBatch(
            grasps=np.zeros((1, 17), dtype=np.float32),
            features=features,
            voxel_points_camera=np.zeros((1, 3), dtype=np.float32),
            feature_layout="official_right_hand_no_flip_v1",
        )


def test_representation_batch_rejects_wrong_feature_width():
    with pytest.raises(ValueError, match="must have shape"):
        _batch(np.zeros((1, OFFICIAL_NO_FLIP_FEATURE_WIDTH - 1), dtype=np.float32))


def test_right_hand_orientation_filter_matches_upstream_flip_rejection():
    predictions = np.zeros((3, 15), dtype=np.float32)
    predictions[:, 3:12] = np.eye(3, dtype=np.float32).reshape(1, 9)
    predictions[1, 3:12] = np.diag((1.0, -1.0, -1.0)).reshape(9)
    predictions[2, 3:12] = np.diag((1.0, 0.0, 1.0)).reshape(9)
    np.testing.assert_array_equal(
        _official_right_hand_orientation_mask(predictions),
        (True, False, True),
    )


def test_right_hand_orientation_filter_rejects_unknown_prediction_layout():
    with pytest.raises(ValueError, match="shape"):
        _official_right_hand_orientation_mask(np.zeros((1, 17), dtype=np.float32))


@pytest.mark.parametrize(
    ("angle_class", "depth_m", "message"),
    ((24.0, 0.02, "angle class"), (2.0, 0.05, "depth class")),
)
def test_decision_features_reject_out_of_range_tail_values(
    angle_class, depth_m, message
):
    features = np.zeros((1, OFFICIAL_NO_FLIP_FEATURE_WIDTH), dtype=np.float32)
    features[0, -2] = angle_class
    features[0, -1] = depth_m
    with pytest.raises(ValueError, match=message):
        OfficialInspireDecisionBackend._decision_features(_batch(features))
