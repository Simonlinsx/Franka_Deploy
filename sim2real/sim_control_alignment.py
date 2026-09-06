"""Installed actuator settings for the v202/v205 deployment contract.

The simulator contract expresses actuator limits in SI units, while RH56
``SPEED_SET`` is a manufacturer register without a reviewed rad/s conversion.
The Franka limits can therefore be matched directly.  RH56 retains the
successful-v205 calibration as provenance.  Deployment uses the uniform 600
register-speed profile matched by the completed six-axis free-space response
identification.
"""

from __future__ import annotations

from typing import Any, Mapping


SIM_CONTROL_CONTRACT_VERSION = "inspire_v202_20hz_adapter_v7_20260727"
SIM_CONTROL_CONTRACT_PATH = (
    "docs/controller_reference/contracts/inspire_v202_sim_control_contract.yaml"
)
SIM_CONTROL_CONTRACT_SHA256 = (
    "9934815923ba758d9a089fb0605b5473822fe30454718456b0a96c2bf6a0610a"
)

SIM_POLICY_RATE_HZ = 20.0
SIM_FRANKA_EFFECTIVE_TARGET_RATE_RAD_S = 0.36
SIM_FRANKA_REPLAY_MAX_VELOCITY_RAD_S = 0.405
SIM_FRANKA_REPLAY_MAX_ACCELERATION_RAD_S2 = 3.59
SIM_FRANKA_REPLAY_MAX_JERK_RAD_S3 = 102.76

REAL_FRANKA_MAX_COMMAND_VELOCITY_RAD_S = 0.50
REAL_FRANKA_MAX_COMMAND_ACCELERATION_RAD_S2 = 5.0
REAL_FRANKA_MAX_COMMAND_JERK_RAD_S3 = 250.0
FRANKA_CONTROLLER_CONTRACT = "franka_v225_v226_interpolated_joint_position_20hz"
FRANKA_CONTROLLER_CONFIG_PATH = (
    "docs/controller_reference/legacy/"
    "franka_v225_v226_real_controller_handoff_20260728/"
    "shaper/franka_v225_controller_config.yaml"
)
FRANKA_CONTROLLER_CONFIG_SHA256 = (
    "c42efe4564bcabe3dc30713c37e713ba369ae124ba280fed91ee598c302ae061"
)

# Register order: little, ring, middle, index, thumb bending, thumb rotation.
# SPEED_SET accepts values in [0,1000].  Recent same-checkpoint simulation and
# real traces showed that the real hand's 20 Hz virtual-joint displacement at
# SPEED_SET=1000 was materially faster than the nominal simulated response.
# Same-checkpoint traces measured an aggregate virtual-joint-speed p95 of about
# 1.33rad/s in simulation, 1.42rad/s at SPEED_SET=400, and 1.72rad/s at
# SPEED_SET=600.  Deployment now uses 600 for checkpoints trained against this
# identified profile.  The per-axis identification is stored in
# RH56_IDENTIFIED_DYNAMICS_PATH and is the active dynamics reference.
# FORCE_SET and the independent current/stop monitors remain unchanged and
# continue to own the contact envelope.  This affects physical response only;
# policy filtering and the checkpoint's semantic per-tick target limit remain
# exact.
RH56_SPEED_SET_REGISTER_ORDER = (600, 600, 600, 600, 600, 600)
RH56_IDENTIFIED_DYNAMICS_PATH = "sim2real/rh56_speed600_identified_dynamics.yaml"
RH56_IDENTIFIED_DYNAMICS_SHA256 = (
    "85dd9bd93b16d6c810f6ad03523f4b4979c48ca0d2b1be6be64305665cb7e371"
)
RH56_THUMB_ROTATION_CALIBRATION = {
    "previous_speed_set": 120,
    "real_replay_p95_rate_units_s": 457.0,
    "sim_replay_p95_rate_units_s": 1246.0,
    "scaled_speed_set": 327.2,
    "replay_calibrated_speed_set": 330,
    "selected_speed_set": 600,
    "selection_reason": "speed600_checkpoint_and_identified_response_alignment",
    "scope": "installed-hand v205 unloaded/mixed-motion replay estimate",
}


def sim_control_alignment_summary() -> Mapping[str, Any]:
    """Return JSON-safe provenance for every real execution audit."""

    return {
        "contract_version": SIM_CONTROL_CONTRACT_VERSION,
        "source_path": SIM_CONTROL_CONTRACT_PATH,
        "source_sha256": SIM_CONTROL_CONTRACT_SHA256,
        "policy_rate_hz": SIM_POLICY_RATE_HZ,
        "franka": {
            "controller_contract": FRANKA_CONTROLLER_CONTRACT,
            "controller_config_path": FRANKA_CONTROLLER_CONFIG_PATH,
            "controller_config_sha256": FRANKA_CONTROLLER_CONFIG_SHA256,
            "command_path": (
                "20hz_held_target->100hz_first_order_filter->6hz_critical_"
                "damping_1khz_interpolator->libfranka_official_limitRate->"
                "joint_impedance"
            ),
            "sim_effective_target_rate_rad_s": (
                SIM_FRANKA_EFFECTIVE_TARGET_RATE_RAD_S
            ),
            "sim_success_replay_max_velocity_rad_s": (
                SIM_FRANKA_REPLAY_MAX_VELOCITY_RAD_S
            ),
            "sim_success_replay_max_acceleration_rad_s2": (
                SIM_FRANKA_REPLAY_MAX_ACCELERATION_RAD_S2
            ),
            "sim_success_replay_max_jerk_rad_s3": (
                SIM_FRANKA_REPLAY_MAX_JERK_RAD_S3
            ),
            "real_command_velocity_guard_rad_s": (
                REAL_FRANKA_MAX_COMMAND_VELOCITY_RAD_S
            ),
            "real_interpolator_acceleration_limit_rad_s2": (
                REAL_FRANKA_MAX_COMMAND_ACCELERATION_RAD_S2
            ),
            "real_interpolator_jerk_limit_rad_s3": (
                REAL_FRANKA_MAX_COMMAND_JERK_RAD_S3
            ),
            "target_lowpass_cutoff_hz": 100.0,
            "tracking_natural_frequency_hz": 6.0,
            "tracking_damping_ratio": 1.0,
            "libfranka_official_rate_limiter": True,
            "libfranka_extra_lowpass": False,
            "alignment": "accepted_v225_v226_real_controller_contract",
        },
        "rh56": {
            "response_profile": "uniform_speed_600_identified_free_space",
            "active_dynamics_identification": (
                "rh56_speed600_free_space_step_identification_v1"
            ),
            "speed600_identified_dynamics_reference_path": (
                RH56_IDENTIFIED_DYNAMICS_PATH
            ),
            "speed600_identified_dynamics_reference_sha256": (
                RH56_IDENTIFIED_DYNAMICS_SHA256
            ),
            "register_order": [
                "little",
                "ring",
                "middle",
                "index",
                "thumb_bending",
                "thumb_rotation",
            ],
            "speed_set": list(RH56_SPEED_SET_REGISTER_ORDER),
            "thumb_rotation_calibration": dict(
                RH56_THUMB_ROTATION_CALIBRATION
            ),
        },
    }


__all__ = [
    "REAL_FRANKA_MAX_COMMAND_ACCELERATION_RAD_S2",
    "REAL_FRANKA_MAX_COMMAND_JERK_RAD_S3",
    "REAL_FRANKA_MAX_COMMAND_VELOCITY_RAD_S",
    "FRANKA_CONTROLLER_CONFIG_PATH",
    "FRANKA_CONTROLLER_CONFIG_SHA256",
    "FRANKA_CONTROLLER_CONTRACT",
    "RH56_SPEED_SET_REGISTER_ORDER",
    "SIM_CONTROL_CONTRACT_PATH",
    "SIM_CONTROL_CONTRACT_SHA256",
    "SIM_CONTROL_CONTRACT_VERSION",
    "sim_control_alignment_summary",
]
