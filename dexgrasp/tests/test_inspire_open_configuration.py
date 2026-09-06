from pathlib import Path

import numpy as np

from anydex_pipeline.inspire_hand_model import InspireHandModel
from anydex_pipeline.inspire_open_configuration import (
    OFFICIAL_OPEN_ACTUATOR_TARGETS,
    OFFICIAL_OPEN_JOINT_POSITIONS_RAD,
    official_open_configuration_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
ANYDEX = ROOT / "third_party/AnyDexGrasp"


def test_official_open_configuration_has_exact_topology_and_finite_fk():
    np.testing.assert_array_equal(
        OFFICIAL_OPEN_ACTUATOR_TARGETS, np.full(6, 1000.0)
    )
    assert OFFICIAL_OPEN_JOINT_POSITIONS_RAD.shape == (12,)
    assert np.all(np.isfinite(OFFICIAL_OPEN_JOINT_POSITIONS_RAD))
    np.testing.assert_array_equal(OFFICIAL_OPEN_JOINT_POSITIONS_RAD[:8], 0.0)

    model = InspireHandModel.from_anydex_root(ANYDEX, mesh_resolution="full")
    transforms = model.link_mesh_transforms(
        np.eye(4), OFFICIAL_OPEN_JOINT_POSITIONS_RAD
    )
    assert set(transforms) == {link.name for link in model.links}
    assert len(transforms) == 13
    assert all(np.all(np.isfinite(value)) for value in transforms.values())


def test_open_configuration_provenance_binds_all_official_sources():
    evidence = official_open_configuration_provenance(ANYDEX)
    assert evidence["actuator_targets"] == [1000.0] * 6
    assert len(evidence["joint_positions_rad"]) == 12
    assert set(evidence["sources"]) == {
        "generator",
        "routine_workbook",
        "driver_workbook",
    }
    for item in evidence["sources"].values():
        assert Path(item["path"]).is_file()
        assert len(item["sha256"]) == 64


def test_constants_match_official_driver_workbook_when_xlrd_is_available():
    try:
        import xlrd
    except ImportError:
        return

    source = ANYDEX / (
        "generate_mesh_and_pointcloud/inspire_urdf/driver_routine_to_angle.xls"
    )
    sheet = xlrd.open_workbook(str(source)).sheet_by_index(0)
    thumb_1 = sheet.col_values(1)[2:]
    thumb_2 = sheet.col_values(2)[2:]
    thumb_3 = sheet.col_values(3)[2:]
    thumb_rotation = sheet.col_values(4)[2:]
    bend_index = int(1000.0 * (1606 - 53) / 1000.0)
    rotation_index = int(1000.0 * (2000 - 400) / 1000.0) + 400
    expected = np.zeros(12, dtype=np.float64)
    expected[8] = 0.3 - (
        (thumb_rotation[rotation_index] - 185.5631219)
        / (106.72277927615 - 185.5631219)
        * (0.3 - (-1.3))
    )
    expected[9] = (
        (thumb_1[bend_index] - 139.27) / (139.27 - 110.53) * 0.433
        + 0.13
    )
    expected[10] = (
        (thumb_2[bend_index] - 188.08)
        / (154.27 - 188.08)
        * (0.179 + 0.267)
        - 0.237
    )
    expected[11] = (
        (thumb_3[bend_index] - 169.94)
        / (147.53 - 169.94)
        * 0.503
        - 0.7
    )
    np.testing.assert_allclose(
        OFFICIAL_OPEN_JOINT_POSITIONS_RAD, expected, atol=1e-12, rtol=0.0
    )
