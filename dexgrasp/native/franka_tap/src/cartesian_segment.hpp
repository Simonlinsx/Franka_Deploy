#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <franka/active_control_base.h>

namespace anydex::franka_native {

// This contract intentionally mirrors the reviewed calibration driver instead
// of exposing permissive defaults.  The binding validates every field before
// releasing the GIL and refuses values that weaken the fixed communication
// gates documented below.
struct CartesianSegmentConfig {
  std::array<double, 3> workspace_min_m{};
  std::array<double, 3> workspace_max_m{};
  std::array<double, 7> joint_lower_rad{};
  std::array<double, 7> joint_upper_rad{};
  double joint_limit_margin_rad{};

  double duration_s{};
  double min_cartesian_duration_s{};
  double max_cartesian_speed_m_s{};
  double max_angular_speed_rad_s{};
  double max_segment_translation_m{};
  double max_segment_rotation_rad{};

  double endpoint_timeout_s{};
  double settle_time_s{};
  double translation_arrival_tolerance_m{};
  double rotation_arrival_tolerance_rad{};
  double settle_max_dq_rad_s{};

  double min_control_success_rate{};
  double control_success_hard_floor{};
  double control_success_evaluation_window_s{};
  double startup_deadline_s{};
  std::uint32_t startup_min_positive_writes{};

  double min_control_period_s{};
  double max_control_period_s{};
  std::uint64_t read_to_write_budget_ns{};
  double wall_deadline_slack_s{};
  double wall_deadline_fraction{};
};

enum class CartesianSegmentFailureCode : std::uint32_t {
  kNone = 0,
  kControlIo = 1,
  kWallDeadline = 2,
  kControlPeriod = 3,
  kRobotMode = 4,
  kRobotErrors = 5,
  kContactOrCollision = 6,
  kJointBounds = 7,
  kStatePose = 8,
  kWorkspace = 9,
  kStartTracking = 10,
  kStartupQualification = 11,
  kControlSuccessHardFloor = 12,
  kControlSuccessWindow = 13,
  kEndpointConvergence = 14,
  kInternalCapacity = 15,
};

struct CartesianSegmentTelemetry {
  CartesianSegmentFailureCode failure_code{CartesianSegmentFailureCode::kNone};
  std::uint64_t reads{0};
  std::uint64_t writes{0};
  std::uint64_t positive_period_writes{0};
  std::uint64_t read_to_write_overruns{0};
  std::uint64_t max_read_to_write_ns{0};
  std::uint64_t complete_success_windows{0};
  bool success_qualified{false};
  bool motion_finished_written{false};
  double qualification_control_time_s{0.0};
  double qualification_wall_time_s{0.0};
  double qualification_rate{0.0};
  double trajectory_control_time_s{0.0};
  double endpoint_control_time_s{0.0};
  double endpoint_stable_time_s{0.0};
  double latest_success_rate{0.0};
  double minimum_postqualification_success_rate{1.0};
  double latest_complete_window_average{0.0};
  double final_translation_error_m{0.0};
  double final_rotation_error_rad{0.0};
  double final_max_abs_dq_rad_s{0.0};
};

class CartesianSegmentError final : public std::runtime_error {
 public:
  CartesianSegmentError(CartesianSegmentFailureCode code,
                        CartesianSegmentTelemetry telemetry,
                        const std::string& message);

  CartesianSegmentFailureCode code() const noexcept;
  const CartesianSegmentTelemetry& telemetry() const noexcept;

 private:
  CartesianSegmentFailureCode code_;
  CartesianSegmentTelemetry telemetry_;
};

// Validate all non-hardware inputs.  This is public so the binding can reject a
// malformed or weakened contract before it releases the GIL or touches a
// control handle.
void ValidateCartesianSegmentContract(
    const std::array<double, 16>& planned_start,
    const std::array<double, 16>& target,
    const CartesianSegmentConfig& config);

// Execute exactly one already-bounded segment on the caller-owned active
// Cartesian pose control handle.  This function never constructs a Robot,
// starts a second control handle, changes collision behavior/load/EE state, or
// calls stop().  On every failure it ceases writing immediately and throws a
// CartesianSegmentError carrying the fixed telemetry accumulated so far.
CartesianSegmentTelemetry RunBoundedCartesianSegment(
    franka::ActiveControlBase& control,
    const std::array<double, 16>& planned_start,
    const std::array<double, 16>& target,
    const CartesianSegmentConfig& config);

const char* CartesianSegmentFailureCodeName(
    CartesianSegmentFailureCode code) noexcept;

}  // namespace anydex::franka_native
