#include "cartesian_segment.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <utility>

#include <franka/control_types.h>
#include <franka/robot_state.h>

namespace anydex::franka_native {
namespace {

constexpr double kAbsoluteTolerance = 1.0e-6;
constexpr double kPoseTolerance = 1.0e-6;
constexpr double kMinimumJerkPeakVelocityFactor = 1.875;
constexpr double kRequiredMinimumSuccessRate = 0.95;
constexpr double kRequiredHardFloor = 0.80;
constexpr double kMaximumSuccessWindowSeconds = 0.50;
constexpr double kMaximumStartupDeadlineSeconds = 0.50;
constexpr std::uint32_t kRequiredStartupPositiveWrites = 100;
constexpr std::uint64_t kMaximumReadToWriteBudgetNs = 500000;
constexpr std::size_t kSuccessRunCapacity = 4096;
constexpr std::uint64_t kMaximumControlCycles = 2000000;

using Clock = std::chrono::steady_clock;
using Pose = std::array<double, 16>;

std::size_t pose_index(std::size_t row, std::size_t column) noexcept {
  return column * 4U + row;
}

bool finite(double value) noexcept { return std::isfinite(value); }

double elapsed_seconds(Clock::time_point start, Clock::time_point end) noexcept {
  return std::chrono::duration<double>(end - start).count();
}

std::uint64_t elapsed_nanoseconds(Clock::time_point start,
                                  Clock::time_point end) noexcept {
  const auto value =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return value > 0 ? static_cast<std::uint64_t>(value) : 0U;
}

[[noreturn]] void invalid_contract(const std::string& message) {
  throw std::invalid_argument("native Cartesian segment contract: " + message);
}

[[noreturn]] void fail(CartesianSegmentFailureCode code,
                       CartesianSegmentTelemetry telemetry,
                       const std::string& message) {
  telemetry.failure_code = code;
  throw CartesianSegmentError(code, telemetry, message);
}

void require_finite_positive(double value, const char* name) {
  if (!finite(value) || value <= 0.0) {
    invalid_contract(std::string(name) + " must be finite and positive");
  }
}

double rotation_determinant(const Pose& pose) noexcept {
  const double a = pose[pose_index(0, 0)];
  const double b = pose[pose_index(0, 1)];
  const double c = pose[pose_index(0, 2)];
  const double d = pose[pose_index(1, 0)];
  const double e = pose[pose_index(1, 1)];
  const double f = pose[pose_index(1, 2)];
  const double g = pose[pose_index(2, 0)];
  const double h = pose[pose_index(2, 1)];
  const double i = pose[pose_index(2, 2)];
  return a * (e * i - f * h) - b * (d * i - f * g) +
         c * (d * h - e * g);
}

bool valid_pose(const Pose& pose) noexcept {
  for (double value : pose) {
    if (!finite(value)) {
      return false;
    }
  }
  if (std::abs(pose[3]) > kPoseTolerance ||
      std::abs(pose[7]) > kPoseTolerance ||
      std::abs(pose[11]) > kPoseTolerance ||
      std::abs(pose[15] - 1.0) > kPoseTolerance) {
    return false;
  }
  for (std::size_t first = 0; first < 3; ++first) {
    for (std::size_t second = 0; second < 3; ++second) {
      double dot = 0.0;
      for (std::size_t row = 0; row < 3; ++row) {
        dot += pose[pose_index(row, first)] * pose[pose_index(row, second)];
      }
      const double expected = first == second ? 1.0 : 0.0;
      if (std::abs(dot - expected) > kPoseTolerance) {
        return false;
      }
    }
  }
  return std::abs(rotation_determinant(pose) - 1.0) <= kPoseTolerance;
}

struct Quaternion {
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

Quaternion normalized(Quaternion value) {
  const double norm = std::sqrt(value.w * value.w + value.x * value.x +
                                value.y * value.y + value.z * value.z);
  if (!finite(norm) || norm <= std::numeric_limits<double>::epsilon()) {
    invalid_contract("rotation produced an invalid quaternion");
  }
  value.w /= norm;
  value.x /= norm;
  value.y /= norm;
  value.z /= norm;
  return value;
}

Quaternion quaternion_from_pose(const Pose& pose) {
  const double r00 = pose[pose_index(0, 0)];
  const double r11 = pose[pose_index(1, 1)];
  const double r22 = pose[pose_index(2, 2)];
  Quaternion result{};
  const double trace = r00 + r11 + r22;
  if (trace > 0.0) {
    const double scale = 2.0 * std::sqrt(trace + 1.0);
    result.w = 0.25 * scale;
    result.x = (pose[pose_index(2, 1)] - pose[pose_index(1, 2)]) / scale;
    result.y = (pose[pose_index(0, 2)] - pose[pose_index(2, 0)]) / scale;
    result.z = (pose[pose_index(1, 0)] - pose[pose_index(0, 1)]) / scale;
  } else if (r00 > r11 && r00 > r22) {
    const double scale =
        2.0 * std::sqrt(std::max(0.0, 1.0 + r00 - r11 - r22));
    if (scale <= std::numeric_limits<double>::epsilon()) {
      invalid_contract("rotation quaternion branch was singular");
    }
    result.w = (pose[pose_index(2, 1)] - pose[pose_index(1, 2)]) / scale;
    result.x = 0.25 * scale;
    result.y = (pose[pose_index(0, 1)] + pose[pose_index(1, 0)]) / scale;
    result.z = (pose[pose_index(0, 2)] + pose[pose_index(2, 0)]) / scale;
  } else if (r11 > r22) {
    const double scale =
        2.0 * std::sqrt(std::max(0.0, 1.0 + r11 - r00 - r22));
    if (scale <= std::numeric_limits<double>::epsilon()) {
      invalid_contract("rotation quaternion branch was singular");
    }
    result.w = (pose[pose_index(0, 2)] - pose[pose_index(2, 0)]) / scale;
    result.x = (pose[pose_index(0, 1)] + pose[pose_index(1, 0)]) / scale;
    result.y = 0.25 * scale;
    result.z = (pose[pose_index(1, 2)] + pose[pose_index(2, 1)]) / scale;
  } else {
    const double scale =
        2.0 * std::sqrt(std::max(0.0, 1.0 + r22 - r00 - r11));
    if (scale <= std::numeric_limits<double>::epsilon()) {
      invalid_contract("rotation quaternion branch was singular");
    }
    result.w = (pose[pose_index(1, 0)] - pose[pose_index(0, 1)]) / scale;
    result.x = (pose[pose_index(0, 2)] + pose[pose_index(2, 0)]) / scale;
    result.y = (pose[pose_index(1, 2)] + pose[pose_index(2, 1)]) / scale;
    result.z = 0.25 * scale;
  }
  return normalized(result);
}

double quaternion_dot(const Quaternion& first,
                      const Quaternion& second) noexcept {
  return first.w * second.w + first.x * second.x + first.y * second.y +
         first.z * second.z;
}

double rotation_error(const Pose& first, const Pose& second) {
  const Quaternion first_q = quaternion_from_pose(first);
  const Quaternion second_q = quaternion_from_pose(second);
  const double dot = std::clamp(std::abs(quaternion_dot(first_q, second_q)),
                                0.0, 1.0);
  return 2.0 * std::acos(dot);
}

double translation_error(const Pose& first, const Pose& second) noexcept {
  const double dx = first[12] - second[12];
  const double dy = first[13] - second[13];
  const double dz = first[14] - second[14];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

Quaternion slerp(Quaternion first, Quaternion second, double alpha) {
  double dot = quaternion_dot(first, second);
  if (dot < 0.0) {
    second.w = -second.w;
    second.x = -second.x;
    second.y = -second.y;
    second.z = -second.z;
    dot = -dot;
  }
  dot = std::clamp(dot, -1.0, 1.0);
  if (dot > 0.9995) {
    return normalized({first.w + alpha * (second.w - first.w),
                       first.x + alpha * (second.x - first.x),
                       first.y + alpha * (second.y - first.y),
                       first.z + alpha * (second.z - first.z)});
  }
  const double angle = std::acos(dot);
  const double sine = std::sin(angle);
  if (std::abs(sine) <= std::numeric_limits<double>::epsilon()) {
    return first;
  }
  const double first_weight = std::sin((1.0 - alpha) * angle) / sine;
  const double second_weight = std::sin(alpha * angle) / sine;
  return normalized({first_weight * first.w + second_weight * second.w,
                     first_weight * first.x + second_weight * second.x,
                     first_weight * first.y + second_weight * second.y,
                     first_weight * first.z + second_weight * second.z});
}

void quaternion_to_pose_rotation(const Quaternion& q, Pose* output) noexcept {
  const double xx = q.x * q.x;
  const double yy = q.y * q.y;
  const double zz = q.z * q.z;
  const double xy = q.x * q.y;
  const double xz = q.x * q.z;
  const double yz = q.y * q.z;
  const double wx = q.w * q.x;
  const double wy = q.w * q.y;
  const double wz = q.w * q.z;
  (*output)[pose_index(0, 0)] = 1.0 - 2.0 * (yy + zz);
  (*output)[pose_index(1, 0)] = 2.0 * (xy + wz);
  (*output)[pose_index(2, 0)] = 2.0 * (xz - wy);
  (*output)[pose_index(0, 1)] = 2.0 * (xy - wz);
  (*output)[pose_index(1, 1)] = 1.0 - 2.0 * (xx + zz);
  (*output)[pose_index(2, 1)] = 2.0 * (yz + wx);
  (*output)[pose_index(0, 2)] = 2.0 * (xz + wy);
  (*output)[pose_index(1, 2)] = 2.0 * (yz - wx);
  (*output)[pose_index(2, 2)] = 1.0 - 2.0 * (xx + yy);
}

double minimum_jerk_blend(double alpha) noexcept {
  const double bounded = std::clamp(alpha, 0.0, 1.0);
  return bounded * bounded * bounded *
         (10.0 + bounded * (-15.0 + 6.0 * bounded));
}

class PreparedInterpolation final {
 public:
  PreparedInterpolation(const Pose& start, const Pose& target)
      : start_(start),
        target_(target),
        start_q_(quaternion_from_pose(start)),
        target_q_(quaternion_from_pose(target)) {}

  void sample(double alpha, Pose* output) const {
    const double blend = minimum_jerk_blend(alpha);
    output->fill(0.0);
    quaternion_to_pose_rotation(slerp(start_q_, target_q_, blend), output);
    (*output)[12] = start_[12] + blend * (target_[12] - start_[12]);
    (*output)[13] = start_[13] + blend * (target_[13] - start_[13]);
    (*output)[14] = start_[14] + blend * (target_[14] - start_[14]);
    (*output)[15] = 1.0;
  }

 private:
  Pose start_;
  Pose target_;
  Quaternion start_q_;
  Quaternion target_q_;
};

bool in_workspace(const Pose& pose,
                  const CartesianSegmentConfig& config) noexcept {
  for (std::size_t index = 0; index < 3; ++index) {
    const double value = pose[12 + index];
    if (!finite(value) || value < config.workspace_min_m[index] ||
        value > config.workspace_max_m[index]) {
      return false;
    }
  }
  return true;
}

void validate_live_pose(const Pose& pose,
                        const CartesianSegmentConfig& config,
                        CartesianSegmentTelemetry telemetry) {
  if (!valid_pose(pose)) {
    fail(CartesianSegmentFailureCode::kStatePose, telemetry,
         "Franka O_T_EE is not a finite rigid transform");
  }
  if (!in_workspace(pose, config)) {
    std::ostringstream message;
    message << "measured EEF translation left workspace: [" << pose[12] << ", "
            << pose[13] << ", " << pose[14] << "]";
    fail(CartesianSegmentFailureCode::kWorkspace, telemetry, message.str());
  }
}

template <std::size_t N>
void validate_flag_vector(const std::array<double, N>& values,
                          const char* name,
                          CartesianSegmentTelemetry telemetry) {
  for (std::size_t index = 0; index < N; ++index) {
    const double value = values[index];
    if (!finite(value) || value < 0.0 || value > 1.0) {
      fail(CartesianSegmentFailureCode::kContactOrCollision, telemetry,
           std::string("Franka ") + name + " contains an invalid value");
    }
    if (value > 0.5) {
      std::ostringstream message;
      message << "Franka reports " << name << " at index " << index
              << ": " << value;
      fail(CartesianSegmentFailureCode::kContactOrCollision, telemetry,
           message.str());
    }
  }
}

double validate_live_state(const franka::RobotState& state,
                           const CartesianSegmentConfig& config,
                           CartesianSegmentTelemetry telemetry) {
  if (state.robot_mode != franka::RobotMode::kMove &&
      state.robot_mode != franka::RobotMode::kIdle) {
    fail(CartesianSegmentFailureCode::kRobotMode, telemetry,
         "Franka mode became unsafe during native Cartesian motion");
  }
  if (static_cast<bool>(state.current_errors)) {
    fail(CartesianSegmentFailureCode::kRobotErrors, telemetry,
         "Franka current_errors became active: " +
             static_cast<std::string>(state.current_errors));
  }
  validate_flag_vector(state.cartesian_contact, "cartesian_contact", telemetry);
  validate_flag_vector(state.cartesian_collision, "cartesian_collision", telemetry);
  validate_flag_vector(state.joint_contact, "joint_contact", telemetry);
  validate_flag_vector(state.joint_collision, "joint_collision", telemetry);
  for (std::size_t index = 0; index < state.q.size(); ++index) {
    const double value = state.q[index];
    if (!finite(value)) {
      fail(CartesianSegmentFailureCode::kJointBounds, telemetry,
           "Franka q contains NaN or infinity");
    }
    const double lower =
        config.joint_lower_rad[index] + config.joint_limit_margin_rad;
    const double upper =
        config.joint_upper_rad[index] - config.joint_limit_margin_rad;
    if (value < lower || value > upper) {
      std::ostringstream message;
      message << "Franka joint " << (index + 1U)
              << " left commissioned margin: actual=" << value << ", interval=["
              << lower << ", " << upper << "]";
      fail(CartesianSegmentFailureCode::kJointBounds, telemetry, message.str());
    }
  }
  validate_live_pose(state.O_T_EE, config, telemetry);
  const double success = state.control_command_success_rate;
  if (!finite(success) || success < 0.0 || success > 1.0) {
    fail(CartesianSegmentFailureCode::kControlSuccessHardFloor, telemetry,
         "Franka control_command_success_rate is outside [0, 1]");
  }
  return success;
}

class SuccessWindow final {
 public:
  explicit SuccessWindow(double duration_limit_s)
      : duration_limit_s_(duration_limit_s) {}

  double observe(double success,
                 double duration_s,
                 CartesianSegmentTelemetry telemetry) {
    if (count_ > 0U) {
      const std::size_t last = (head_ + count_ - 1U) % runs_.size();
      if (runs_[last].success == success) {
        runs_[last].duration_s += duration_s;
      } else {
        append(success, duration_s, telemetry);
      }
    } else {
      append(success, duration_s, telemetry);
    }
    total_duration_s_ += duration_s;
    weighted_success_ += success * duration_s;

    double excess = total_duration_s_ - duration_limit_s_;
    while (excess > 1.0e-12 && count_ > 0U) {
      SuccessRun& oldest = runs_[head_];
      const double removed = std::min(oldest.duration_s, excess);
      oldest.duration_s -= removed;
      total_duration_s_ -= removed;
      weighted_success_ -= oldest.success * removed;
      excess -= removed;
      if (oldest.duration_s <= 1.0e-12) {
        head_ = (head_ + 1U) % runs_.size();
        --count_;
      }
    }
    if (total_duration_s_ + 1.0e-12 < duration_limit_s_) {
      return -1.0;
    }
    return weighted_success_ / total_duration_s_;
  }

  double minimum() const noexcept {
    double result = 1.0;
    for (std::size_t offset = 0; offset < count_; ++offset) {
      result = std::min(result, runs_[(head_ + offset) % runs_.size()].success);
    }
    return result;
  }

 private:
  struct SuccessRun {
    double success{0.0};
    double duration_s{0.0};
  };

  void append(double success,
              double duration_s,
              CartesianSegmentTelemetry telemetry) {
    if (count_ == runs_.size()) {
      fail(CartesianSegmentFailureCode::kInternalCapacity, telemetry,
           "native control-success window exceeded fixed run capacity");
    }
    const std::size_t index = (head_ + count_) % runs_.size();
    runs_[index] = {success, duration_s};
    ++count_;
  }

  std::array<SuccessRun, kSuccessRunCapacity> runs_{};
  std::size_t head_{0};
  std::size_t count_{0};
  double duration_limit_s_{};
  double total_duration_s_{0.0};
  double weighted_success_{0.0};
};

void update_write_timing(Clock::time_point cycle_start,
                         const CartesianSegmentConfig& config,
                         CartesianSegmentTelemetry* telemetry) noexcept {
  const std::uint64_t elapsed = elapsed_nanoseconds(cycle_start, Clock::now());
  telemetry->max_read_to_write_ns =
      std::max(telemetry->max_read_to_write_ns, elapsed);
  if (elapsed > config.read_to_write_budget_ns) {
    ++telemetry->read_to_write_overruns;
  }
}

void write_pose(franka::ActiveControlBase& control,
                const Pose& pose,
                bool finished,
                Clock::time_point cycle_start,
                const CartesianSegmentConfig& config,
                CartesianSegmentTelemetry* telemetry) {
  franka::CartesianPose command(pose);
  command.motion_finished = finished;
  try {
    control.writeOnce(command);
  } catch (const std::exception& error) {
    fail(CartesianSegmentFailureCode::kControlIo, *telemetry,
         std::string("Franka Cartesian writeOnce failed: ") + error.what());
  }
  ++telemetry->writes;
  if (finished) {
    telemetry->motion_finished_written = true;
  }
  update_write_timing(cycle_start, config, telemetry);
}

std::pair<franka::RobotState, franka::Duration> read_state(
    franka::ActiveControlBase& control,
    CartesianSegmentTelemetry telemetry) {
  try {
    return control.readOnce();
  } catch (const std::exception& error) {
    fail(CartesianSegmentFailureCode::kControlIo, telemetry,
         std::string("Franka Cartesian readOnce failed: ") + error.what());
  }
}

void require_start_tracking(const Pose& reference,
                            const Pose& candidate,
                            const CartesianSegmentConfig& config,
                            CartesianSegmentTelemetry telemetry,
                            const char* label) {
  const double translation = translation_error(reference, candidate);
  const double rotation = rotation_error(reference, candidate);
  if (translation > config.translation_arrival_tolerance_m ||
      rotation > config.rotation_arrival_tolerance_rad) {
    std::ostringstream message;
    message << label << " exceeds start-tracking bounds: translation="
            << translation << "m, rotation=" << rotation << "rad";
    fail(CartesianSegmentFailureCode::kStartTracking, telemetry, message.str());
  }
}

}  // namespace

CartesianSegmentError::CartesianSegmentError(
    CartesianSegmentFailureCode code,
    CartesianSegmentTelemetry telemetry,
    const std::string& message)
    : std::runtime_error(message), code_(code), telemetry_(telemetry) {}

CartesianSegmentFailureCode CartesianSegmentError::code() const noexcept {
  return code_;
}

const CartesianSegmentTelemetry& CartesianSegmentError::telemetry() const noexcept {
  return telemetry_;
}

const char* CartesianSegmentFailureCodeName(
    CartesianSegmentFailureCode code) noexcept {
  switch (code) {
    case CartesianSegmentFailureCode::kNone:
      return "none";
    case CartesianSegmentFailureCode::kControlIo:
      return "control_io";
    case CartesianSegmentFailureCode::kWallDeadline:
      return "wall_deadline";
    case CartesianSegmentFailureCode::kControlPeriod:
      return "control_period";
    case CartesianSegmentFailureCode::kRobotMode:
      return "robot_mode";
    case CartesianSegmentFailureCode::kRobotErrors:
      return "robot_errors";
    case CartesianSegmentFailureCode::kContactOrCollision:
      return "contact_or_collision";
    case CartesianSegmentFailureCode::kJointBounds:
      return "joint_bounds";
    case CartesianSegmentFailureCode::kStatePose:
      return "state_pose";
    case CartesianSegmentFailureCode::kWorkspace:
      return "workspace";
    case CartesianSegmentFailureCode::kStartTracking:
      return "start_tracking";
    case CartesianSegmentFailureCode::kStartupQualification:
      return "startup_qualification";
    case CartesianSegmentFailureCode::kControlSuccessHardFloor:
      return "control_success_hard_floor";
    case CartesianSegmentFailureCode::kControlSuccessWindow:
      return "control_success_window";
    case CartesianSegmentFailureCode::kEndpointConvergence:
      return "endpoint_convergence";
    case CartesianSegmentFailureCode::kInternalCapacity:
      return "internal_capacity";
  }
  return "unknown";
}

void ValidateCartesianSegmentContract(
    const Pose& planned_start,
    const Pose& target,
    const CartesianSegmentConfig& config) {
  if (!valid_pose(planned_start) || !valid_pose(target)) {
    invalid_contract("planned_start and target must be finite rigid transforms");
  }
  for (std::size_t index = 0; index < 3; ++index) {
    if (!finite(config.workspace_min_m[index]) ||
        !finite(config.workspace_max_m[index]) ||
        config.workspace_min_m[index] >= config.workspace_max_m[index]) {
      invalid_contract("workspace bounds must be finite and ordered");
    }
  }
  if (!in_workspace(planned_start, config) || !in_workspace(target, config)) {
    invalid_contract("planned_start and target must lie inside the workspace");
  }
  require_finite_positive(config.joint_limit_margin_rad,
                          "joint_limit_margin_rad");
  for (std::size_t index = 0; index < 7; ++index) {
    const double lower = config.joint_lower_rad[index];
    const double upper = config.joint_upper_rad[index];
    if (!finite(lower) || !finite(upper) || lower >= upper ||
        2.0 * config.joint_limit_margin_rad >= upper - lower) {
      invalid_contract("joint bounds must leave a non-empty commissioned margin");
    }
  }

  require_finite_positive(config.duration_s, "duration_s");
  require_finite_positive(config.min_cartesian_duration_s,
                          "min_cartesian_duration_s");
  require_finite_positive(config.max_cartesian_speed_m_s,
                          "max_cartesian_speed_m_s");
  require_finite_positive(config.max_angular_speed_rad_s,
                          "max_angular_speed_rad_s");
  require_finite_positive(config.max_segment_translation_m,
                          "max_segment_translation_m");
  require_finite_positive(config.max_segment_rotation_rad,
                          "max_segment_rotation_rad");
  const double translation = translation_error(planned_start, target);
  const double rotation = rotation_error(planned_start, target);
  if (translation > config.max_segment_translation_m + 1.0e-12 ||
      rotation > config.max_segment_rotation_rad + 1.0e-12) {
    invalid_contract("target exceeds the single-segment translation/rotation bound");
  }
  const double required_duration =
      std::max({config.min_cartesian_duration_s,
                kMinimumJerkPeakVelocityFactor * translation /
                    config.max_cartesian_speed_m_s,
                kMinimumJerkPeakVelocityFactor * rotation /
                    config.max_angular_speed_rad_s});
  if (config.duration_s + 1.0e-12 < required_duration) {
    invalid_contract("duration_s exceeds the commissioned minimum-jerk speed bound");
  }

  require_finite_positive(config.endpoint_timeout_s, "endpoint_timeout_s");
  require_finite_positive(config.settle_time_s, "settle_time_s");
  require_finite_positive(config.translation_arrival_tolerance_m,
                          "translation_arrival_tolerance_m");
  require_finite_positive(config.rotation_arrival_tolerance_rad,
                          "rotation_arrival_tolerance_rad");
  require_finite_positive(config.settle_max_dq_rad_s,
                          "settle_max_dq_rad_s");
  if (config.endpoint_timeout_s + 1.0e-12 < config.settle_time_s) {
    invalid_contract("endpoint_timeout_s must cover settle_time_s");
  }

  if (!finite(config.min_control_success_rate) ||
      config.min_control_success_rate <
          kRequiredMinimumSuccessRate - kAbsoluteTolerance ||
      config.min_control_success_rate > 1.0) {
    invalid_contract("min_control_success_rate may not be weaker than 0.95");
  }
  if (!finite(config.control_success_hard_floor) ||
      config.control_success_hard_floor <
          kRequiredHardFloor - kAbsoluteTolerance ||
      config.control_success_hard_floor > config.min_control_success_rate) {
    invalid_contract("control_success_hard_floor may not be weaker than 0.80");
  }
  require_finite_positive(config.control_success_evaluation_window_s,
                          "control_success_evaluation_window_s");
  if (config.control_success_evaluation_window_s >
      kMaximumSuccessWindowSeconds + 1.0e-12) {
    invalid_contract("control-success window may not exceed 0.5s");
  }
  require_finite_positive(config.startup_deadline_s, "startup_deadline_s");
  if (config.startup_deadline_s >
      kMaximumStartupDeadlineSeconds + 1.0e-12) {
    invalid_contract("startup qualification deadline may not exceed 0.5s");
  }
  if (config.startup_min_positive_writes <
      kRequiredStartupPositiveWrites) {
    invalid_contract("startup qualification requires at least 100 positive writes");
  }

  require_finite_positive(config.min_control_period_s, "min_control_period_s");
  require_finite_positive(config.max_control_period_s, "max_control_period_s");
  if (config.min_control_period_s >= config.max_control_period_s) {
    invalid_contract("control period bounds must be ordered");
  }
  if (config.read_to_write_budget_ns == 0U ||
      config.read_to_write_budget_ns > kMaximumReadToWriteBudgetNs) {
    invalid_contract("read_to_write_budget_ns may not exceed 500000");
  }
  require_finite_positive(config.wall_deadline_slack_s,
                          "wall_deadline_slack_s");
  require_finite_positive(config.wall_deadline_fraction,
                          "wall_deadline_fraction");
}

CartesianSegmentTelemetry RunBoundedCartesianSegment(
    franka::ActiveControlBase& control,
    const Pose& planned_start,
    const Pose& target,
    const CartesianSegmentConfig& config) {
  ValidateCartesianSegmentContract(planned_start, target, config);

  CartesianSegmentTelemetry telemetry{};
  SuccessWindow success_window(config.control_success_evaluation_window_s);
  const Clock::time_point wall_start = Clock::now();
  const double nominal_total = config.startup_deadline_s + config.duration_s +
                               config.endpoint_timeout_s;
  const double wall_limit_s =
      nominal_total +
      std::max(config.wall_deadline_slack_s,
               config.wall_deadline_fraction * nominal_total);
  bool allow_initial_zero_period = true;
  bool initialized = false;
  Pose active_start{};
  Pose command{};
  PreparedInterpolation* interpolation_ptr = nullptr;
  alignas(PreparedInterpolation) std::array<std::byte, sizeof(PreparedInterpolation)>
      interpolation_storage{};
  double trajectory_elapsed = 0.0;
  double endpoint_elapsed = 0.0;
  double endpoint_stable_elapsed = 0.0;
  Clock::time_point endpoint_wall_start{};
  bool endpoint_started = false;

  try {
    for (;;) {
      if (telemetry.reads >= kMaximumControlCycles) {
        fail(CartesianSegmentFailureCode::kInternalCapacity, telemetry,
             "native Cartesian segment exceeded maximum control cycles");
      }
      auto state_and_period = read_state(control, telemetry);
      ++telemetry.reads;
      const Clock::time_point cycle_start = Clock::now();
      const double wall_elapsed = elapsed_seconds(wall_start, cycle_start);
      if (!finite(wall_elapsed) || wall_elapsed > wall_limit_s) {
        fail(CartesianSegmentFailureCode::kWallDeadline, telemetry,
             "native Cartesian segment exceeded wall-clock deadline");
      }

      const double dt = state_and_period.second.toSec();
      if (!finite(dt) || dt < 0.0 || dt > config.max_control_period_s ||
          (dt > 0.0 && dt <= config.min_control_period_s)) {
        std::ostringstream message;
        message << "unsafe Franka control period dt=" << dt << "s";
        fail(CartesianSegmentFailureCode::kControlPeriod, telemetry,
             message.str());
      }
      if (dt == 0.0) {
        if (!allow_initial_zero_period) {
          fail(CartesianSegmentFailureCode::kControlPeriod, telemetry,
               "repeated zero Franka control period");
        }
        allow_initial_zero_period = false;
      } else {
        allow_initial_zero_period = false;
      }

      franka::RobotState& state = state_and_period.first;
      const double success = validate_live_state(state, config, telemetry);
      telemetry.latest_success_rate = success;
      if (!initialized) {
        if (!valid_pose(state.O_T_EE_c)) {
          fail(CartesianSegmentFailureCode::kStatePose, telemetry,
               "Franka O_T_EE_c is not a finite rigid transform");
        }
        if (!in_workspace(state.O_T_EE_c, config)) {
          fail(CartesianSegmentFailureCode::kWorkspace, telemetry,
               "active commanded Cartesian start lies outside workspace");
        }
        require_start_tracking(planned_start, state.O_T_EE_c, config, telemetry,
                               "active commanded versus planned start");
        require_start_tracking(planned_start, state.O_T_EE, config, telemetry,
                               "active measured versus planned start");
        require_start_tracking(state.O_T_EE, state.O_T_EE_c, config, telemetry,
                               "active commanded versus measured start");
        active_start = state.O_T_EE_c;
        interpolation_ptr = new (interpolation_storage.data())
            PreparedInterpolation(active_start, target);
        initialized = true;
      }

      if (!telemetry.success_qualified) {
        if (dt > 0.0) {
          telemetry.qualification_control_time_s += dt;
        }
        telemetry.qualification_wall_time_s = wall_elapsed;

        // A success sample describes commands sent before this read.  Require
        // at least N already-completed positive-period hold writes, then send
        // one final exact-start hold on the qualifying read before arming.
        const bool candidate =
            telemetry.positive_period_writes >=
                config.startup_min_positive_writes &&
            dt > 0.0 &&
            success >=
                config.min_control_success_rate - kAbsoluteTolerance &&
            telemetry.qualification_control_time_s <=
                config.startup_deadline_s + kAbsoluteTolerance &&
            telemetry.qualification_wall_time_s <=
                config.startup_deadline_s + kAbsoluteTolerance;
        if (!candidate &&
            (telemetry.qualification_control_time_s + 1.0e-12 >=
                 config.startup_deadline_s ||
             telemetry.qualification_wall_time_s + 1.0e-12 >=
                 config.startup_deadline_s)) {
          std::ostringstream message;
          message << "Franka control-success startup qualification timed out: "
                  << "latest=" << success
                  << ", threshold=" << config.min_control_success_rate
                  << ", positive_writes="
                  << telemetry.positive_period_writes
                  << ", control_time="
                  << telemetry.qualification_control_time_s
                  << "s, wall_time=" << telemetry.qualification_wall_time_s
                  << "s";
          fail(CartesianSegmentFailureCode::kStartupQualification, telemetry,
               message.str());
        }
        command = active_start;
        if (!in_workspace(command, config)) {
          fail(CartesianSegmentFailureCode::kWorkspace, telemetry,
               "startup hold command lies outside workspace");
        }
        write_pose(control, command, false, cycle_start, config, &telemetry);
        if (dt > 0.0) {
          ++telemetry.positive_period_writes;
        }
        if (candidate) {
          telemetry.success_qualified = true;
          telemetry.qualification_rate = success;
          telemetry.qualification_wall_time_s =
              elapsed_seconds(wall_start, Clock::now());
          if (telemetry.qualification_wall_time_s >
              config.startup_deadline_s + kAbsoluteTolerance) {
            fail(CartesianSegmentFailureCode::kStartupQualification, telemetry,
                 "startup qualification exceeded deadline during hold write");
          }
        }
        continue;
      }

      telemetry.minimum_postqualification_success_rate =
          std::min(telemetry.minimum_postqualification_success_rate, success);
      if (success <
          config.control_success_hard_floor - kAbsoluteTolerance) {
        std::ostringstream message;
        message << "Franka control success crossed hard floor: actual="
                << success << ", hard_floor="
                << config.control_success_hard_floor;
        fail(CartesianSegmentFailureCode::kControlSuccessHardFloor, telemetry,
             message.str());
      }
      if (dt > 0.0) {
        const double average = success_window.observe(success, dt, telemetry);
        if (average >= 0.0) {
          ++telemetry.complete_success_windows;
          telemetry.latest_complete_window_average = average;
          if (average <
              config.min_control_success_rate - kAbsoluteTolerance) {
            std::ostringstream message;
            message << "Franka control-success window average is below threshold: "
                    << "average=" << average
                    << ", minimum=" << success_window.minimum()
                    << ", latest=" << success
                    << ", threshold=" << config.min_control_success_rate
                    << ", window="
                    << config.control_success_evaluation_window_s << "s";
            fail(CartesianSegmentFailureCode::kControlSuccessWindow, telemetry,
                 message.str());
          }
        }
      }

      if (trajectory_elapsed < config.duration_s) {
        if (dt > 0.0) {
          trajectory_elapsed += dt;
        }
        telemetry.trajectory_control_time_s = trajectory_elapsed;
        interpolation_ptr->sample(
            std::min(1.0, trajectory_elapsed / config.duration_s), &command);
      } else {
        if (!endpoint_started) {
          endpoint_started = true;
          endpoint_wall_start = cycle_start;
        }
        if (dt > 0.0) {
          endpoint_elapsed += dt;
        }
        telemetry.endpoint_control_time_s = endpoint_elapsed;
        telemetry.final_translation_error_m =
            translation_error(state.O_T_EE, target);
        telemetry.final_rotation_error_rad =
            rotation_error(state.O_T_EE, target);
        double max_abs_dq = 0.0;
        for (double velocity : state.dq) {
          if (!finite(velocity)) {
            fail(CartesianSegmentFailureCode::kEndpointConvergence, telemetry,
                 "Franka dq contains NaN or infinity at endpoint");
          }
          max_abs_dq = std::max(max_abs_dq, std::abs(velocity));
        }
        telemetry.final_max_abs_dq_rad_s = max_abs_dq;
        const bool endpoint_good =
            telemetry.final_translation_error_m <=
                config.translation_arrival_tolerance_m &&
            telemetry.final_rotation_error_rad <=
                config.rotation_arrival_tolerance_rad &&
            max_abs_dq <= config.settle_max_dq_rad_s;
        if (endpoint_good) {
          endpoint_stable_elapsed += dt;
          telemetry.endpoint_stable_time_s = endpoint_stable_elapsed;
          if (endpoint_stable_elapsed + 1.0e-12 >= config.settle_time_s) {
            break;
          }
        } else {
          endpoint_stable_elapsed = 0.0;
          telemetry.endpoint_stable_time_s = 0.0;
        }
        const double endpoint_wall_elapsed =
            elapsed_seconds(endpoint_wall_start, cycle_start);
        if (endpoint_elapsed + 1.0e-12 >= config.endpoint_timeout_s ||
            endpoint_wall_elapsed + 1.0e-12 >= config.endpoint_timeout_s) {
          std::ostringstream message;
          message << "Franka Cartesian endpoint convergence timed out: "
                  << "translation=" << telemetry.final_translation_error_m
                  << "m, rotation=" << telemetry.final_rotation_error_rad
                  << "rad, max_dq=" << telemetry.final_max_abs_dq_rad_s
                  << "rad/s, stable=" << endpoint_stable_elapsed
                  << "s, timeout=" << config.endpoint_timeout_s << "s";
          fail(CartesianSegmentFailureCode::kEndpointConvergence, telemetry,
               message.str());
        }
        command = target;
      }

      if (!valid_pose(command)) {
        fail(CartesianSegmentFailureCode::kStatePose, telemetry,
             "native Cartesian interpolation produced an invalid pose");
      }
      if (!in_workspace(command, config)) {
        fail(CartesianSegmentFailureCode::kWorkspace, telemetry,
             "native Cartesian command left workspace");
      }
      write_pose(control, command, false, cycle_start, config, &telemetry);
    }

    const Clock::time_point final_write_start = Clock::now();
    write_pose(control, target, true, final_write_start, config, &telemetry);
  } catch (const CartesianSegmentError&) {
    if (interpolation_ptr != nullptr) {
      interpolation_ptr->~PreparedInterpolation();
    }
    throw;
  } catch (const std::exception& error) {
    if (interpolation_ptr != nullptr) {
      interpolation_ptr->~PreparedInterpolation();
    }
    fail(CartesianSegmentFailureCode::kControlIo, telemetry,
         std::string("native Cartesian control failed: ") + error.what());
  }
  if (interpolation_ptr != nullptr) {
    interpolation_ptr->~PreparedInterpolation();
  }
  return telemetry;
}

}  // namespace anydex::franka_native
