from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim2real.deployment import runner as cli
from sim2real.rh56_profile_contract import (
    load_commissioned_rh56_force_set_g,
    load_v94_rh56_profile_command_bounds,
)


WORKSPACE = Path(__file__).resolve().parents[2]
LEGACY_V94_PROFILE = (
    WORKSPACE / "dexgrasp/configs/fr3_rh56_v94_commissioning.json"
)
V57_THROWN_PROFILE = (
    WORKSPACE
    / "dexgrasp/configs/fr3_rh56_v57_thrown_alpha0p5_20hz_commissioned.json"
)
V60_THROWN_PROFILE = (
    WORKSPACE
    / "dexgrasp/configs/fr3_rh56_v60_palmcatch_first_motion.json"
)
V61_THROWN_PROFILE = (
    WORKSPACE
    / "dexgrasp/configs/fr3_rh56_v61_sixexpert_40tick.json"
)


def _configured_profile(config_path: Path) -> Path:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    return (config_path.parent / value["commissioning_profile"]).resolve()


def test_all_v94_deployment_defaults_select_the_evidence_derived_profile() -> None:
    expected = (
        WORKSPACE
        / "dexgrasp/configs/fr3_rh56_v94_seq286_20hz_commissioned.json"
    ).resolve()
    assert cli.DEFAULT_PROFILE.resolve() == expected
    assert _configured_profile(WORKSPACE / "sim2real/config.json") == expected
    assert (
        _configured_profile(WORKSPACE / "sim2real/v94_deploy_config.json")
        == expected
    )


def test_profile_derives_one_consistent_live_q6_contract() -> None:
    bounds = load_v94_rh56_profile_command_bounds(cli.DEFAULT_PROFILE)
    assert bounds.minimum_angle_set_register_order == (0, 0, 0, 0, 0, 0)
    assert bounds.maximum_angle_set_register_order == (1000,) * 6
    assert bounds.feedback_to_command_valid_min == (0, 0, 0, 0, 0, 0)
    assert bounds.feedback_to_command_valid_max == (1000,) * 6
    assert bounds.commissioned_exact_targets == (
        (61, 17, 524, 758, 422, 416),
    )

    request = cli.SupervisedRequest(
        run_id="DRY_RUN",
        steps=1,
        bundle=cli.DEFAULT_BUNDLE,
        profile=cli.DEFAULT_PROFILE,
        pcd_config=cli.DEFAULT_PCD_CONFIG,
        execute=False,
    )
    guards = cli._summary(request)["hard_guards"]
    assert tuple(guards["rh56_min_angle_set_register_order"]) == (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert tuple(guards["rh56_feedback_to_command_valid_min"]) == (
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_v57_thrown_profile_reuses_the_commissioned_rh56_bounds() -> None:
    expected = load_v94_rh56_profile_command_bounds(cli.DEFAULT_PROFILE)
    actual = load_v94_rh56_profile_command_bounds(V57_THROWN_PROFILE)
    assert actual == expected
    assert load_commissioned_rh56_force_set_g(cli.DEFAULT_PROFILE) == 80
    assert load_commissioned_rh56_force_set_g(V57_THROWN_PROFILE) == 500


def test_v60_single_tick_profile_reuses_bounds_and_force500() -> None:
    expected = load_v94_rh56_profile_command_bounds(cli.DEFAULT_PROFILE)
    actual = load_v94_rh56_profile_command_bounds(V60_THROWN_PROFILE)
    assert actual == expected
    assert load_commissioned_rh56_force_set_g(V60_THROWN_PROFILE) == 500


def test_v61_forty_tick_profile_reuses_bounds_and_force500() -> None:
    expected = load_v94_rh56_profile_command_bounds(cli.DEFAULT_PROFILE)
    actual = load_v94_rh56_profile_command_bounds(V61_THROWN_PROFILE)
    assert actual == expected
    assert load_commissioned_rh56_force_set_g(V61_THROWN_PROFILE) == 500


def test_force_set_is_task_scoped_and_fails_closed_on_profile_edits() -> None:
    ordinary = json.loads(cli.DEFAULT_PROFILE.read_text(encoding="utf-8"))
    ordinary["inspire"]["force_limit_g"] = 500
    with pytest.raises(ValueError, match="must remain 80 g"):
        load_commissioned_rh56_force_set_g(ordinary)

    thrown = json.loads(V57_THROWN_PROFILE.read_text(encoding="utf-8"))
    thrown["inspire"]["force_limit_g"] = 80
    with pytest.raises(ValueError, match="differs from its commissioned 500 g"):
        load_commissioned_rh56_force_set_g(thrown)

    thrown = json.loads(V57_THROWN_PROFILE.read_text(encoding="utf-8"))
    thrown["inspire"]["force_set_commissioning"][
        "free_space_comparison_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="comparison evidence differs"):
        load_commissioned_rh56_force_set_g(thrown)


def test_unlisted_profile_id_cannot_spoof_commissioned_rh56_bounds() -> None:
    value = json.loads(V57_THROWN_PROFILE.read_text(encoding="utf-8"))
    value["profile_id"] = "unreviewed-copy"
    with pytest.raises(ValueError, match="explicitly commissioned"):
        load_v94_rh56_profile_command_bounds(value)


def test_legacy_uncommissioned_v94_profile_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="coupled RH56 closure is not commissioned"):
        load_v94_rh56_profile_command_bounds(LEGACY_V94_PROFILE)
