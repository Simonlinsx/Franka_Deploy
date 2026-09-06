"""Checks for the domain-oriented public module layout."""

from motion_planning.kinematics import (
    cartesian_joint_correction as modular_cartesian_joint_correction,
    panda_T_base_policy_palm as modular_panda_fk,
    pose_preserving_joint_target as modular_pose_target,
)
from robot_control.franka import PylibfrankaBackendFactory
from robot_control.rh56 import RH56WatchdogOwner
from robot_control.safety import ClosedLoopSafetyGate
from sim2real.deploy import DeploymentRequest as PublicDeploymentRequest
from sim2real.deployment import DeploymentRequest
from motion_planning.online_tabletop import (
    cartesian_joint_correction as planner_cartesian_joint_correction,
    panda_T_base_policy_palm as planner_panda_fk,
    pose_preserving_joint_target as planner_pose_target,
)
from sim2real.tasks import build_task_command


def test_motion_planning_public_and_implementation_imports_share_functions() -> None:
    assert planner_panda_fk is modular_panda_fk
    assert planner_cartesian_joint_correction is modular_cartesian_joint_correction
    assert planner_pose_target is modular_pose_target


def test_domain_public_apis_import_without_constructing_hardware() -> None:
    assert PylibfrankaBackendFactory.__name__ == "PylibfrankaBackendFactory"
    assert RH56WatchdogOwner.__name__ == "RH56WatchdogOwner"
    assert ClosedLoopSafetyGate.__name__ == "ClosedLoopSafetyGate"
    assert DeploymentRequest.__name__ == "DeploymentRequest"
    assert DeploymentRequest is PublicDeploymentRequest
    assert build_task_command.__name__ == "build_task_command"
