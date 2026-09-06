#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "anydex/v94_franka_servo/protocol.hpp"
#include "anydex/v94_franka_servo/safety_limits.hpp"

namespace anydex::v94_franka_servo {

struct RobotSample final {
  std::array<double, 7> q_rad{};
  std::array<double, 7> dq_rad_s{};
  // For a joint-position motion generator these are the exact commanded
  // position history and derivatives maintained by FCI.  In particular, FCI
  // updates them with its constant-acceleration extrapolator when command
  // packets are missed.  The next wire command must be rate-limited against
  // these values rather than against a workstation-side trajectory copy.
  std::array<double, 7> desired_q_rad{};
  std::array<double, 7> desired_dq_rad_s{};
  std::array<double, 7> desired_ddq_rad_s2{};
  std::array<double, 16> O_T_EE{};
  std::array<double, 16> F_T_EE{};
  double end_effector_mass_kg{0.0};
  std::array<double, 3> end_effector_com_m{};
  std::array<double, 9> end_effector_inertia_kg_m2{};
  double external_load_mass_kg{0.0};
  std::uint64_t robot_time_ms{0U};
  std::uint32_t control_period_ms{0U};
  RobotModeCode mode{RobotModeCode::kUnknown};
  std::uint32_t status_flags{0U};
  std::string current_errors_text{"[]"};
  std::string last_motion_errors_text{"[]"};
  double control_command_success_rate{0.0};
};

class ActiveRobotSession {
 public:
  virtual ~ActiveRobotSession() = default;
  ActiveRobotSession() = default;
  ActiveRobotSession(const ActiveRobotSession&) = delete;
  ActiveRobotSession& operator=(const ActiveRobotSession&) = delete;

  virtual RobotSample read_once() = 0;
  // Apply libfranka's official position-command rate limiter against the
  // exact FCI desired history. The default is a dependency-free equivalent
  // envelope for fake/offline backends; production overrides it with the
  // position-dependent official libfranka implementation.
  virtual std::array<double, 7> limit_joint_position_command(
      const std::array<double, 7>& candidate_q_rad,
      const std::array<double, 7>& reference_q_rad,
      const std::array<double, 7>& reference_dq_rad_s,
      const std::array<double, 7>& reference_ddq_rad_s2) {
    constexpr double kDtS = 0.001;
    std::array<double, 7> limited{};
    for (std::size_t joint = 0U; joint < limited.size(); ++joint) {
      const double requested_velocity =
          (candidate_q_rad[joint] - reference_q_rad[joint]) / kDtS;
      const double requested_acceleration =
          (requested_velocity - reference_dq_rad_s[joint]) / kDtS;
      const double requested_jerk =
          (requested_acceleration - reference_ddq_rad_s2[joint]) / kDtS;
      const double jerk_limited_acceleration =
          reference_ddq_rad_s2[joint] +
          std::clamp(
              requested_jerk,
              -HardSafetyLimits::kMaximumCommandJerkRadS3,
              HardSafetyLimits::kMaximumCommandJerkRadS3) *
              kDtS;
      const double acceleration_to_velocity_gain =
          HardSafetyLimits::kMaximumCommandJerkRadS3 /
          HardSafetyLimits::kMaximumCommandAccelerationRadS2;
      const double safe_maximum_acceleration = std::min(
          acceleration_to_velocity_gain *
              (HardSafetyLimits::kMaximumCommandVelocityRadS -
               reference_dq_rad_s[joint]),
          HardSafetyLimits::kMaximumCommandAccelerationRadS2);
      const double safe_minimum_acceleration = std::max(
          acceleration_to_velocity_gain *
              (-HardSafetyLimits::kMaximumCommandVelocityRadS -
               reference_dq_rad_s[joint]),
          -HardSafetyLimits::kMaximumCommandAccelerationRadS2);
      const double acceleration = std::clamp(
          jerk_limited_acceleration, safe_minimum_acceleration,
          safe_maximum_acceleration);
      const double velocity = reference_dq_rad_s[joint] + acceleration * kDtS;
      limited[joint] = reference_q_rad[joint] + velocity * kDtS;
    }
    return limited;
  }
  virtual void write_once(const std::array<double, 7>& q_rad,
                          bool motion_finished) = 0;
};

class RobotBackend {
 public:
  virtual ~RobotBackend() = default;
  RobotBackend() = default;
  RobotBackend(const RobotBackend&) = delete;
  RobotBackend& operator=(const RobotBackend&) = delete;

  virtual RobotSample read_once() = 0;
  virtual std::unique_ptr<ActiveRobotSession> start_joint_position_control() = 0;
  virtual void stop() = 0;
};

class RobotBackendFactory {
 public:
  virtual ~RobotBackendFactory() = default;
  virtual std::unique_ptr<RobotBackend> create_enforced(
      const std::string& robot_address) = 0;
};

}  // namespace anydex::v94_franka_servo
