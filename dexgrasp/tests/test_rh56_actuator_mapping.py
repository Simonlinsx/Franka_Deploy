import json
from pathlib import Path

import numpy as np

from anydex_pipeline.inspire_open_configuration import (
    OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
)
from anydex_pipeline.rh56_actuator_mapping import (
    DRIVER_WORKBOOK_SHA256,
    OfficialRH56ActuatorMapper,
)


ANYDEX_ROOT = Path(__file__).parents[1] / "third_party/AnyDexGrasp"


def test_official_mapper_reproduces_every_non_ring_release_json_row_exactly():
    mapper = OfficialRH56ActuatorMapper.from_anydex_root(ANYDEX_ROOT)
    mapping_path = (
        ANYDEX_ROOT
        / "generate_mesh_and_pointcloud/inspire_urdf/width_12Dangle_6Dangle.json"
    )
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    tested = 0
    # The upstream generator deliberately overrides only grasp-type zero's q6
    # with a pose-specific constant.  Every other row is a direct workbook
    # conversion and therefore provides the appropriate regression oracle.
    for grasp_type, rows in payload.items():
        if grasp_type == "Ring":
            continue
        for row in rows.values():
            actual = mapper.to_joint_positions_rad(row["6d"])
            expected = np.asarray(row["12d"], dtype=np.float64)
            assert np.array_equal(actual, expected)
            tested += 1
    assert tested > 100
    assert mapper.sha256 == DRIVER_WORKBOOK_SHA256


def test_official_mapper_reproduces_bound_all_open_configuration():
    mapper = OfficialRH56ActuatorMapper.from_anydex_root(ANYDEX_ROOT)
    assert np.array_equal(
        mapper.to_joint_positions_rad([1000] * 6),
        OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
    )
    assert np.array_equal(
        mapper.feedback_to_joint_positions_rad([1000] * 6),
        OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
    )


def test_official_mapper_rejects_non_integer_or_out_of_range_registers():
    mapper = OfficialRH56ActuatorMapper.from_anydex_root(ANYDEX_ROOT)
    for target in ([1000] * 5, [1000] * 5 + [1001], [1000] * 5 + [1.5]):
        try:
            mapper.to_joint_positions_rad(target)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid RH56 target was accepted")
