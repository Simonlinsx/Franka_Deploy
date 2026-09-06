"""FR3 + Inspire RH56 sim2real observation and deployment package.

The stable public commands are ``python -m sim2real.validate`` for offline
validation and ``python -m sim2real.deploy`` for operator-supervised hardware
deployment.  Importing this package never opens a hardware interface.
"""

from .contracts import (
    FRANKA_DOF,
    INSPIRE_AXES,
    OBSERVATION_SCHEMA_VERSION,
    FrankaObservation,
    InspireObservation,
    ObjectPCDObservation,
    ObservationSample,
    ObservationSpec,
    assemble_observation,
    franka_column_major_pose,
)
from sim2real.policy.rate_mode import (
    PolicyRateMode,
    SUPPORTED_POLICY_RATES_HZ,
    resolve_policy_rate_mode,
)

__all__ = [
    "FRANKA_DOF",
    "INSPIRE_AXES",
    "OBSERVATION_SCHEMA_VERSION",
    "FrankaObservation",
    "InspireObservation",
    "ObjectPCDObservation",
    "ObservationSample",
    "ObservationSpec",
    "PolicyRateMode",
    "SUPPORTED_POLICY_RATES_HZ",
    "assemble_observation",
    "franka_column_major_pose",
    "resolve_policy_rate_mode",
]
