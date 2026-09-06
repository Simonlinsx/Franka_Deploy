#include "anydex/v94_franka_servo/libfranka_backend.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <franka/active_control_base.h>
#include <franka/control_types.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#include <research_interface/robot/service_types.h>

namespace anydex::v94_franka_servo {
namespace {

template <std::size_t N>
bool any_active(const std::array<double, N>& values) noexcept {
  return std::any_of(values.begin(), values.end(),
                     [](const double value) { return value != 0.0; });
}

RobotModeCode mode_code(const franka::RobotMode mode) noexcept {
  switch (mode) {
    case franka::RobotMode::kOther:
      return RobotModeCode::kOther;
    case franka::RobotMode::kIdle:
      return RobotModeCode::kIdle;
    case franka::RobotMode::kMove:
      return RobotModeCode::kMove;
    case franka::RobotMode::kGuiding:
      return RobotModeCode::kGuiding;
    case franka::RobotMode::kReflex:
      return RobotModeCode::kReflex;
    case franka::RobotMode::kUserStopped:
      return RobotModeCode::kUserStopped;
    case franka::RobotMode::kAutomaticErrorRecovery:
      return RobotModeCode::kAutomaticErrorRecovery;
  }
  return RobotModeCode::kUnknown;
}

std::uint32_t status_flags(const franka::RobotState& state) noexcept {
  std::uint32_t flags = 0U;
  const std::string current_errors =
      static_cast<std::string>(state.current_errors);
  const std::string last_motion_errors =
      static_cast<std::string>(state.last_motion_errors);
  if (static_cast<bool>(state.current_errors)) {
    flags |= kStateHasCurrentErrors;
  }
  if (static_cast<bool>(state.last_motion_errors)) {
    flags |= kStateHasLastMotionErrors;
  }
  if (any_active(state.joint_contact)) {
    flags |= kStateHasJointContact;
  }
  if (any_active(state.joint_collision)) {
    flags |= kStateHasJointCollision;
  }
  if (any_active(state.cartesian_contact)) {
    flags |= kStateHasCartesianContact;
  }
  if (any_active(state.cartesian_collision)) {
    flags |= kStateHasCartesianCollision;
  }
  if (
      current_errors == "[\"communication_constraints_violation\"]" &&
      last_motion_errors == "[\"communication_constraints_violation\"]") {
    flags |= kStateHasOnlyCommunicationConstraintsViolation;
  }
  return flags;
}

RobotSample convert_sample(const franka::RobotState& state,
                           const franka::Duration period) {
  const std::uint64_t period_ms = period.toMSec();
  if (period_ms > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("libfranka period does not fit uint32 milliseconds");
  }
  RobotSample output{};
  output.q_rad = state.q;
  output.dq_rad_s = state.dq;
  output.desired_q_rad = state.q_d;
  output.desired_dq_rad_s = state.dq_d;
  output.desired_ddq_rad_s2 = state.ddq_d;
  output.O_T_EE = state.O_T_EE;
  output.F_T_EE = state.F_T_EE;
  output.end_effector_mass_kg = state.m_ee;
  output.end_effector_com_m = state.F_x_Cee;
  output.end_effector_inertia_kg_m2 = state.I_ee;
  output.external_load_mass_kg = state.m_load;
  output.robot_time_ms = state.time.toMSec();
  output.control_period_ms = static_cast<std::uint32_t>(period_ms);
  output.mode = mode_code(state.robot_mode);
  output.status_flags = status_flags(state);
  output.current_errors_text = static_cast<std::string>(state.current_errors);
  output.last_motion_errors_text =
      static_cast<std::string>(state.last_motion_errors);
  output.control_command_success_rate = state.control_command_success_rate;
  return output;
}

class LibfrankaActiveSession final : public ActiveRobotSession {
 public:
  explicit LibfrankaActiveSession(
      franka::Robot& robot,
      std::unique_ptr<franka::ActiveControlBase> active) noexcept
      : robot_(robot), active_(std::move(active)) {}

  RobotSample read_once() override {
    auto value = active_->readOnce();
    return convert_sample(value.first, value.second);
  }

  std::array<double, 7> limit_joint_position_command(
      const std::array<double, 7>& candidate_q_rad,
      const std::array<double, 7>& reference_q_rad,
      const std::array<double, 7>& reference_dq_rad_s,
      const std::array<double, 7>& reference_ddq_rad_s2) override {
    std::array<double, 7> upper_velocity_limits =
        robot_.getUpperJointVelocityLimits(reference_q_rad);
    std::array<double, 7> lower_velocity_limits =
        robot_.getLowerJointVelocityLimits(reference_q_rad);
    std::array<double, 7> maximum_acceleration{};
    std::array<double, 7> maximum_jerk{};
    for (std::size_t joint = 0U; joint < upper_velocity_limits.size(); ++joint) {
      upper_velocity_limits[joint] = std::min(
          upper_velocity_limits[joint],
          HardSafetyLimits::kMaximumCommandVelocityRadS);
      lower_velocity_limits[joint] = std::max(
          lower_velocity_limits[joint],
          -HardSafetyLimits::kMaximumCommandVelocityRadS);
      maximum_acceleration[joint] =
          HardSafetyLimits::kMaximumCommandAccelerationRadS2;
      maximum_jerk[joint] = HardSafetyLimits::kMaximumCommandJerkRadS3;
    }
    return franka::limitRate(
        upper_velocity_limits, lower_velocity_limits,
        maximum_acceleration, maximum_jerk,
        candidate_q_rad, reference_q_rad, reference_dq_rad_s,
        reference_ddq_rad_s2);
  }

  void write_once(const std::array<double, 7>& q_rad,
                  const bool motion_finished) override {
    franka::JointPositions command(q_rad);
    if (motion_finished) {
      command = franka::MotionFinished(command);
    }
    active_->writeOnce(command);
  }

 private:
  franka::Robot& robot_;
  std::unique_ptr<franka::ActiveControlBase> active_;
};

class LibfrankaBackend final : public RobotBackend {
 public:
  explicit LibfrankaBackend(const std::string& robot_address)
      : robot_(robot_address, franka::RealtimeConfig::kEnforce) {}

  RobotSample read_once() override {
    return convert_sample(robot_.readOnce(), franka::Duration(0U));
  }

  std::unique_ptr<ActiveRobotSession> start_joint_position_control() override {
    // Set an explicit, reproducible two-level behavior immediately before
    // taking motion-generator ownership.  The lower official-example values
    // publish contact state early enough for the 1 kHz motion generator to
    // decelerate.  The higher installed-RH56 values remain the controller-side
    // collision/reflex boundary.
    robot_.setCollisionBehavior(
        HardSafetyLimits::kContactTorqueThresholdNm,
        HardSafetyLimits::kCollisionTorqueThresholdNm,
        HardSafetyLimits::kContactTorqueThresholdNm,
        HardSafetyLimits::kCollisionTorqueThresholdNm,
        HardSafetyLimits::kContactForceThresholdN,
        HardSafetyLimits::kCollisionForceThresholdN,
        HardSafetyLimits::kContactForceThresholdN,
        HardSafetyLimits::kCollisionForceThresholdN);
    return std::make_unique<LibfrankaActiveSession>(robot_,
        robot_.startJointPositionControl(
            research_interface::robot::Move::ControllerMode::kJointImpedance));
  }

  void stop() override { robot_.stop(); }

 private:
  franka::Robot robot_;
};

}  // namespace

std::unique_ptr<RobotBackend> LibfrankaBackendFactory::create_enforced(
    const std::string& robot_address) {
  return std::make_unique<LibfrankaBackend>(robot_address);
}

}  // namespace anydex::v94_franka_servo
