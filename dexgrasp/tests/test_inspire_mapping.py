import json

import numpy as np
import pytest

from anydex_pipeline.inspire_mapping import (
    InspireGraspBatch,
    load_inspire_mapping,
    map_two_finger_grasps,
)


def _grasp(
    *,
    score=0.8,
    width=0.025,
    depth=0.02,
    rotation=np.eye(3),
    translation=(0.0, 0.0, 0.0),
    object_id=-1.0,
):
    result = np.zeros(17, dtype=np.float64)
    result[0:4] = [score, width, 0.03, depth]
    result[4:13] = np.asarray(rotation).reshape(9)
    result[13:16] = translation
    result[16] = object_id
    return result


def _entry(translation, rotation, angles):
    return {
        "translation": list(translation),
        "rotation": np.asarray(rotation).tolist(),
        "6d": list(angles),
    }


def test_mapping_matches_official_transform_and_array_layout():
    rotation_two_finger = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_offset = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    translation_two_finger = np.array([0.4, -0.2, 0.7])
    translation_offset = np.array([0.10, 0.02, -0.03])
    angles = [1000, 1000, 1000, 472, 523, 87]
    mapping = {
        "Ring": {
            "2.5": _entry(translation_offset, rotation_offset, angles),
        }
    }
    two_finger = _grasp(
        width=0.001,  # Official mapping clamps this to 0.025 m / key "2.5".
        rotation=rotation_two_finger,
        translation=translation_two_finger,
        object_id=42,
    )

    result = map_two_finger_grasps(
        two_finger,
        1,
        mapping,
        scores=0.95,
        depth_offsets=0.03,
    )

    expected_two_finger = np.eye(4)
    expected_two_finger[:3, :3] = rotation_two_finger
    expected_two_finger[:3, 3] = translation_two_finger
    expected_offset = np.eye(4)
    expected_offset[:3, :3] = rotation_offset
    expected_offset[:3, 3] = translation_offset
    expected_palm = expected_two_finger @ np.linalg.inv(expected_offset)

    assert isinstance(result, InspireGraspBatch)
    assert len(result) == 1
    np.testing.assert_allclose(result.rotation_matrices[0], expected_palm[:3, :3])
    np.testing.assert_allclose(result.translations[0], expected_palm[:3, 3])
    np.testing.assert_allclose(result.angles[0], angles)
    np.testing.assert_allclose(result.widths, [0.025])
    np.testing.assert_allclose(result.depths, [0.05])
    np.testing.assert_allclose(result.scores, [0.95])

    expected_at_depth = expected_palm[:3, 3] + rotation_two_finger[:, 0] * 0.05
    np.testing.assert_allclose(result.translations_at_depth[0], expected_at_depth)
    np.testing.assert_allclose(
        result.pose_matrices(apply_depth=True)[0, :3, 3], expected_at_depth
    )

    official = result.to_official_array()
    assert official.shape == (1, 23)
    np.testing.assert_allclose(official[0, :3], [0.95, 0.05, 1.0])
    np.testing.assert_allclose(official[0, 3:12], expected_palm[:3, :3].reshape(9))
    np.testing.assert_allclose(official[0, 12:15], expected_palm[:3, 3])
    np.testing.assert_allclose(official[0, 15:21], angles)
    np.testing.assert_allclose(official[0, 21:], [42.0, 0.025])


def test_load_directory_and_map_multiple_types(tmp_path):
    data = {
        "Ring": {"3.0": _entry([0, 0, 0], np.eye(3), [1, 2, 3, 4, 5, 6])},
        "Tripod": {"10.0": _entry([0, 0, 0], np.eye(3), [6, 5, 4, 3, 2, 1])},
    }
    json_path = tmp_path / "width_12Dangle_6Dangle.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_inspire_mapping(tmp_path)
    result = map_two_finger_grasps(
        np.stack([_grasp(width=0.03), _grasp(width=0.2)]),
        [1, 6],
        loaded,
    )

    np.testing.assert_allclose(result.widths, [0.03, 0.10])
    np.testing.assert_allclose(
        result.angles,
        [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]],
    )
    np.testing.assert_allclose(result.scores, [0.8, 0.8])


@pytest.mark.parametrize("invalid_type", [0, 9, 1.5, np.nan])
def test_invalid_inspire_type_is_rejected(invalid_type):
    with pytest.raises(ValueError, match="grasp_type|type"):
        map_two_finger_grasps(
            _grasp(),
            invalid_type,
            {"Ring": {"2.5": _entry([0, 0, 0], np.eye(3), [0] * 6)}},
        )
