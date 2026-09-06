from __future__ import annotations

import pytest

from sim2real.policy.rate_mode import resolve_policy_rate_mode


def test_60hz_mode_preserves_existing_runtime_contract() -> None:
    mode = resolve_policy_rate_mode(60)

    assert mode.name == "60hz"
    assert mode.control_dt_s == pytest.approx(1.0 / 60.0)
    assert mode.policy_ticks_per_rh56_update == 3
    assert mode.rh56_max_register_delta_per_update == (
        137,
        143,
        158,
        158,
        251,
        120,
    )
    assert mode.camera_max_policy_actions_per_frame == 6
    assert mode.maximum_supervised_steps == 720


def test_20hz_mode_updates_rh56_each_policy_tick() -> None:
    mode = resolve_policy_rate_mode("20")

    assert mode.name == "20hz"
    assert mode.control_dt_s == pytest.approx(0.05)
    assert mode.policy_ticks_per_rh56_update == 1
    assert mode.rh56_max_register_delta_per_update == (
        46,
        48,
        53,
        53,
        84,
        40,
    )
    # Point-cloud latency DR is 0..2 ticks: original use plus two reuses.
    assert mode.camera_max_policy_actions_per_frame == 3
    assert mode.maximum_supervised_steps == 240


@pytest.mark.parametrize("value", (0, 10, 30, 40, 120, True, "20hz"))
def test_policy_rate_mode_rejects_every_unreviewed_rate(value: object) -> None:
    with pytest.raises(ValueError, match="exactly 20 or 60"):
        resolve_policy_rate_mode(value)
