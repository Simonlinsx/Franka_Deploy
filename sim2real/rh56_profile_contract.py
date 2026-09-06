"""Fail-closed RH56 command bounds derived from a selected V94 profile.

The deployment must use one profile as the source of truth for policy
clipping, feedback bootstrap, emergency hold, reset, and audit output.  This
module is hardware-inert and performs only strict JSON/profile validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


V94_PROFILE_ID = "fr3_rh56_v94_reset_locked_v1"
V57_THROWN_PROFILE_ID = "fr3_rh56_v57_thrown_alpha0p5_first_motion_v1"
V60_THROWN_SINGLE_TICK_PROFILE_ID = "fr3_rh56_v60_palmcatch_single_tick_v1"
V61_THROWN_40_TICK_PROFILE_ID = "fr3_rh56_v61_sixexpert_40tick_v1"
COMMISSIONED_RH56_PROFILE_IDS = frozenset(
    {
        V94_PROFILE_ID,
        V57_THROWN_PROFILE_ID,
        V60_THROWN_SINGLE_TICK_PROFILE_ID,
        V61_THROWN_40_TICK_PROFILE_ID,
    }
)
V94_POLICY_CONTRACT = "v94_inspire_semantic_13d"
REGISTER_MINIMUM = 0
REGISTER_MAXIMUM = 1000
V94_FORCE_SET_G = 80
V57_THROWN_FORCE_SET_G = 500
V57_THROWN_FORCE_COMMISSIONING_CONTRACT = (
    "v57_thrown_rh56_force500_sim_alignment_v1"
)
V60_THROWN_FORCE_COMMISSIONING_CONTRACT = (
    "v60_thrown_rh56_force500_single_tick_v1"
)
V61_THROWN_FORCE_COMMISSIONING_CONTRACT = (
    "v61_thrown_rh56_force500_40tick_v1"
)


@dataclass(frozen=True)
class RH56ProfileCommandBounds:
    """Manufacturer-order command and feedback-bootstrap intervals."""

    minimum_angle_set_register_order: tuple[int, ...]
    maximum_angle_set_register_order: tuple[int, ...]
    feedback_to_command_valid_min: tuple[int, ...]
    feedback_to_command_valid_max: tuple[int, ...]
    commissioned_exact_targets: tuple[tuple[int, ...], ...]


def _six_ints(value: Any, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 6
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must contain exactly six integers")
    result = tuple(int(item) for item in value)
    if any(
        item < REGISTER_MINIMUM or item > REGISTER_MAXIMUM for item in result
    ):
        raise ValueError(f"{name} is outside the RH56 register domain")
    return result


def _profile_object(profile: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(profile, Mapping):
        return dict(profile)
    source = Path(profile).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load RH56 commissioning profile {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("RH56 commissioning profile root must be an object")
    return value


def load_v94_rh56_profile_command_bounds(
    profile: Mapping[str, Any] | str | Path,
    *,
    feedback_to_command_offset_units: Sequence[int] = (0, 0, 0, 0, 0, 15),
) -> RH56ProfileCommandBounds:
    """Validate a commissioned V94 profile and derive every live interval.

    A legacy V7 profile, an uncommissioned V94 profile, or a profile with no
    exact coupled target is rejected.  This prevents an apparently harmless
    ``--profile`` change from silently mixing a V94 reset pose with unrelated
    RH56 limits.
    """

    value = _profile_object(profile)
    if value.get("profile_id") not in COMMISSIONED_RH56_PROFILE_IDS:
        raise ValueError(
            "selected profile is not an explicitly commissioned RH56 deployment profile"
        )
    if value.get("policy_contract") != V94_POLICY_CONTRACT:
        raise ValueError("selected profile does not declare the V94 policy contract")
    inspire = value.get("inspire")
    if not isinstance(inspire, Mapping):
        raise ValueError("selected V94 profile has no inspire object")
    q6_range = inspire.get("thumb_rotate_validated_realtime_range")
    if (
        not isinstance(q6_range, list)
        or len(q6_range) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in q6_range)
    ):
        raise ValueError(
            "inspire.thumb_rotate_validated_realtime_range must contain two integers"
        )
    q6_lower, q6_upper = (int(item) for item in q6_range)
    if not REGISTER_MINIMUM <= q6_lower < q6_upper <= REGISTER_MAXIMUM:
        raise ValueError("V94 q6 commissioned command range is invalid")
    open_targets = _six_ints(inspire.get("open_targets"), "inspire.open_targets")
    if open_targets != (REGISTER_MAXIMUM,) * 6 or open_targets[5] != q6_upper:
        raise ValueError("V94 RH56 open targets differ from the commissioned upper endpoint")
    if inspire.get("six_axis_coupled_closure_commissioned") is not True:
        raise ValueError("V94 six-axis coupled RH56 closure is not commissioned")

    raw_targets = inspire.get("commissioned_air_closure_targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("V94 profile has no commissioned exact RH56 closure target")
    exact_targets = tuple(
        _six_ints(target, f"inspire.commissioned_air_closure_targets[{index}]")
        for index, target in enumerate(raw_targets)
    )
    if any(not q6_lower <= target[5] <= q6_upper for target in exact_targets):
        raise ValueError("commissioned exact RH56 target lies outside the q6 range")

    offsets = _six_ints(
        feedback_to_command_offset_units,
        "feedback_to_command_offset_units",
    )
    command_minimum = (0, 0, 0, 0, 0, q6_lower)
    command_maximum = (1000, 1000, 1000, 1000, 1000, q6_upper)
    feedback_minimum = tuple(
        max(REGISTER_MINIMUM, lower - offset)
        for lower, offset in zip(command_minimum, offsets)
    )
    feedback_maximum = tuple(
        min(
            REGISTER_MAXIMUM,
            upper if offset >= 0 else upper - offset,
        )
        for upper, offset in zip(command_maximum, offsets)
    )
    return RH56ProfileCommandBounds(
        minimum_angle_set_register_order=command_minimum,
        maximum_angle_set_register_order=command_maximum,
        feedback_to_command_valid_min=feedback_minimum,
        feedback_to_command_valid_max=feedback_maximum,
        commissioned_exact_targets=exact_targets,
    )


def load_commissioned_rh56_force_set_g(
    profile: Mapping[str, Any] | str | Path,
) -> int:
    """Return the exact task-scoped RH56 force threshold or fail closed.

    The ordinary V94/tabletop profile retains the original 80 g threshold.
    The V57 thrown-object profile is separately commissioned at 500 g to match
    the simulator/identification contract. Merely editing ``force_limit_g``
    in another profile cannot widen the live hardware setting.
    """

    value = _profile_object(profile)
    profile_id = value.get("profile_id")
    inspire = value.get("inspire")
    if not isinstance(inspire, Mapping):
        raise ValueError("selected RH56 profile has no inspire object")
    raw_force = inspire.get("force_limit_g")
    if isinstance(raw_force, bool) or not isinstance(raw_force, int):
        raise ValueError("inspire.force_limit_g must be an integer")
    force_set_g = int(raw_force)
    if profile_id == V94_PROFILE_ID:
        if force_set_g != V94_FORCE_SET_G:
            raise ValueError(
                f"ordinary V94 RH56 force must remain {V94_FORCE_SET_G} g"
            )
        return force_set_g
    if profile_id in (
        V57_THROWN_PROFILE_ID,
        V60_THROWN_SINGLE_TICK_PROFILE_ID,
        V61_THROWN_40_TICK_PROFILE_ID,
    ):
        if force_set_g != V57_THROWN_FORCE_SET_G:
            raise ValueError(
                "V57 thrown RH56 force differs from its commissioned "
                f"{V57_THROWN_FORCE_SET_G} g setting"
            )
        evidence = inspire.get("force_set_commissioning")
        if not isinstance(evidence, Mapping):
            raise ValueError("V57 thrown RH56 force has no commissioning evidence")
        expected_contract = {
            V57_THROWN_PROFILE_ID: V57_THROWN_FORCE_COMMISSIONING_CONTRACT,
            V60_THROWN_SINGLE_TICK_PROFILE_ID: (
                V60_THROWN_FORCE_COMMISSIONING_CONTRACT
            ),
            V61_THROWN_40_TICK_PROFILE_ID: V61_THROWN_FORCE_COMMISSIONING_CONTRACT,
        }[profile_id]
        if evidence.get("contract") != expected_contract:
            raise ValueError("thrown RH56 force commissioning contract differs")
        if evidence.get("free_space_comparison_sha256") != (
            "752b285041264273c7c98af25392a1e94cb3977b36277898b7e14bf7ab05199b"
        ):
            raise ValueError("V57 thrown RH56 force comparison evidence differs")
        return force_set_g
    raise ValueError(
        "selected profile is not an explicitly commissioned RH56 deployment profile"
    )
