// Standalone C++17 reference for the accepted V225/V226 Franka command path.
//
// This header intentionally has no libfranka or Eigen dependency. The real
// controller should run InterpolatedJointPositionGenerator::Step() once per
// 1 kHz FCI callback and pass state().q_rad to franka::JointPositions. Enable
// libfranka's rate limiter as the final guard and disable its extra low-pass
// filter, as shown in franka_v225_libfranka_example.cpp.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace simtoolreal::franka_v225 {

using Vector7d = std::array<double, 7>;
using Vector7f = std::array<float, 7>;

inline constexpr double kPolicyDtS = 0.05;
inline constexpr double kServoDtS = 0.001;
inline constexpr double kTargetLowpassCutoffHz = 100.0;
inline constexpr double kTrackingNaturalFrequencyHz = 6.0;
inline constexpr double kTrackingDampingRatio = 1.0;
inline constexpr double kMaxAccelerationRadS2 = 5.0;
inline constexpr double kMaxJerkRadS3 = 250.0;

inline constexpr float kArmActionGainRad = 0.045F;
inline constexpr float kArmTargetAlpha = 0.40F;
inline constexpr float kArmMaxTargetStepRad = 0.045F;
inline constexpr float kArmMeasuredEnvelopeRad = 0.05F;

inline constexpr Vector7d kSafeJointLowerRad = {
    -2.6937, -1.7337, -2.8507, -2.9921, -2.7565, 0.5945, -2.9659};
inline constexpr Vector7d kSafeJointUpperRad = {
    2.6937, 1.7337, 2.8507, -0.2018, 2.7565, 4.4669, 2.9659};

template <typename T>
constexpr T Clamp(const T value, const T lower, const T upper) {
  return std::min(std::max(value, lower), upper);
}

inline Vector7f MapPolicyActionToHeldTarget(
    const Vector7f& previous_target_q_rad,
    const Vector7f& measured_q_rad,
    const Vector7f& action_arm,
    const Vector7d& joint_lower_rad = kSafeJointLowerRad,
    const Vector7d& joint_upper_rad = kSafeJointUpperRad) {
  Vector7f next_target{};
  for (std::size_t joint = 0; joint < next_target.size(); ++joint) {
    const float action = Clamp(action_arm[joint], -1.0F, 1.0F);
    const float raw_target =
        previous_target_q_rad[joint] + kArmActionGainRad * action;
    const float filtered_target =
        kArmTargetAlpha * raw_target +
        (1.0F - kArmTargetAlpha) * previous_target_q_rad[joint];
    const float target_delta = Clamp(
        filtered_target - previous_target_q_rad[joint],
        -kArmMaxTargetStepRad,
        kArmMaxTargetStepRad);
    const float candidate = previous_target_q_rad[joint] + target_delta;
    const float safe_lower = std::max(
        static_cast<float>(joint_lower_rad[joint]),
        measured_q_rad[joint] - kArmMeasuredEnvelopeRad);
    const float safe_upper = std::min(
        static_cast<float>(joint_upper_rad[joint]),
        measured_q_rad[joint] + kArmMeasuredEnvelopeRad);
    next_target[joint] = Clamp(candidate, safe_lower, safe_upper);
  }
  return next_target;
}

inline Vector7d ToDouble(const Vector7f& value) {
  Vector7d result{};
  for (std::size_t joint = 0; joint < result.size(); ++joint) {
    result[joint] = static_cast<double>(value[joint]);
  }
  return result;
}

struct DesiredCommandState {
  Vector7d q_rad{};
  Vector7d dq_rad_s{};
  Vector7d ddq_rad_s2{};
  Vector7d filtered_target_rad{};
};

class InterpolatedJointPositionGenerator {
 public:
  explicit InterpolatedJointPositionGenerator(
      const Vector7d& start_q_rad,
      const Vector7d& start_dq_rad_s = {},
      const Vector7d& start_ddq_rad_s2 = {},
      const double cutoff_frequency_hz = kTargetLowpassCutoffHz,
      const double natural_frequency_hz = kTrackingNaturalFrequencyHz,
      const double damping_ratio = kTrackingDampingRatio)
      : cutoff_frequency_hz_(cutoff_frequency_hz),
        omega_rad_s_(2.0 * Pi() * natural_frequency_hz),
        damping_ratio_(damping_ratio) {
    if (!(cutoff_frequency_hz > 0.0) ||
        !(natural_frequency_hz > 0.0) || !(damping_ratio > 0.0)) {
      throw std::invalid_argument(
          "filter frequency, tracking frequency, and damping must be positive");
    }
    state_.q_rad = ClampToSafeLimits(start_q_rad);
    state_.dq_rad_s = start_dq_rad_s;
    state_.ddq_rad_s2 = start_ddq_rad_s2;
    state_.filtered_target_rad = state_.q_rad;
  }

  const DesiredCommandState& state() const noexcept { return state_; }

  const DesiredCommandState& Step(
      const Vector7d& held_target_q_rad,
      const double dt_s = kServoDtS) {
    if (!(dt_s > 0.0) || !std::isfinite(dt_s)) {
      throw std::invalid_argument("dt_s must be finite and positive");
    }

    const Vector7d held = ClampToSafeLimits(held_target_q_rad);
    const double lowpass_gain =
        dt_s / (dt_s + 1.0 / (2.0 * Pi() * cutoff_frequency_hz_));
    for (std::size_t joint = 0; joint < held.size(); ++joint) {
      state_.filtered_target_rad[joint] =
          lowpass_gain * held[joint] +
          (1.0 - lowpass_gain) * state_.filtered_target_rad[joint];

      const double desired_acceleration =
          omega_rad_s_ * omega_rad_s_ *
              (state_.filtered_target_rad[joint] - state_.q_rad[joint]) -
          2.0 * damping_ratio_ * omega_rad_s_ * state_.dq_rad_s[joint];
      const double acceleration_delta = Clamp(
          desired_acceleration - state_.ddq_rad_s2[joint],
          -kMaxJerkRadS3 * dt_s,
          kMaxJerkRadS3 * dt_s);
      const double acceleration = Clamp(
          state_.ddq_rad_s2[joint] + acceleration_delta,
          -kMaxAccelerationRadS2,
          kMaxAccelerationRadS2);
      const double velocity = state_.dq_rad_s[joint] + acceleration * dt_s;
      const double position = state_.q_rad[joint] + velocity * dt_s;

      state_.q_rad[joint] = Clamp(
          position, kSafeJointLowerRad[joint], kSafeJointUpperRad[joint]);
      state_.dq_rad_s[joint] = velocity;
      state_.ddq_rad_s2[joint] = acceleration;
    }
    return state_;
  }

  const DesiredCommandState& Advance(
      const Vector7d& held_target_q_rad,
      const int packets,
      const double dt_s = kServoDtS) {
    if (packets <= 0) {
      throw std::invalid_argument("packets must be positive");
    }
    for (int packet = 0; packet < packets; ++packet) {
      Step(held_target_q_rad, dt_s);
    }
    return state_;
  }

 private:
  static constexpr double Pi() { return 3.14159265358979323846; }

  static Vector7d ClampToSafeLimits(const Vector7d& value) {
    Vector7d result{};
    for (std::size_t joint = 0; joint < result.size(); ++joint) {
      if (!std::isfinite(value[joint])) {
        throw std::invalid_argument("joint vector contains a non-finite value");
      }
      result[joint] = Clamp(
          value[joint], kSafeJointLowerRad[joint], kSafeJointUpperRad[joint]);
    }
    return result;
  }

  DesiredCommandState state_{};
  double cutoff_frequency_hz_;
  double omega_rad_s_;
  double damping_ratio_;
};

}  // namespace simtoolreal::franka_v225
