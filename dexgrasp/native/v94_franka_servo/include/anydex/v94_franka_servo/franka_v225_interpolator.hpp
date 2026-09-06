#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "anydex/v94_franka_servo/safety_limits.hpp"

namespace anydex::v94_franka_servo {

// Accepted V225/V226 real-controller contract.  A policy target is held by
// the parent at 20 Hz; this generator is advanced once for every returned FCI
// period.  libfranka's official joint-position rate limiter is applied after
// this generator by the hardware backend.
struct InterpolatedCommand final {
  std::array<double, 7> q{};
  std::array<double, 7> dq{};
  std::array<double, 7> ddq{};
  std::array<double, 7> filtered_target{};
};

// FCI stores the desired motion-generator history as binary32 values and
// libfranka promotes those values back to double in RobotState.  Several of
// the decimal hard limits below are not exactly representable as binary32:
// clamping an outgoing double command to the exact hard limit can therefore
// round *outside* that same limit on the next FCI read.
//
// Keep the physical HardSafetyLimits unchanged and derive a separate command
// interval whose endpoints are the first binary32 values strictly inside the
// hard interval.  Only internally generated actuator commands use this
// contracted-by-at-most-one-float-ulp interval.  Target admission, measured
// state validation, and returned FCI-history validation remain against the
// unchanged physical hard limits.
class SendableJointLimits final {
 public:
  static const std::array<double, 7>& lower() noexcept {
    static const std::array<double, 7> limits = make_lower();
    return limits;
  }

  static const std::array<double, 7>& upper() noexcept {
    static const std::array<double, 7> limits = make_upper();
    return limits;
  }

  static std::array<double, 7> clamp(
      const std::array<double, 7>& value) {
    std::array<double, 7> result{};
    for (std::size_t joint = 0U; joint < result.size(); ++joint) {
      if (!std::isfinite(value[joint])) {
        throw std::invalid_argument("sendable joint command is non-finite");
      }
      result[joint] =
          std::clamp(value[joint], lower()[joint], upper()[joint]);
    }
    return result;
  }

 private:
  static double first_float_strictly_above(const double lower_bound) noexcept {
    float candidate = static_cast<float>(lower_bound);
    if (static_cast<double>(candidate) <= lower_bound) {
      candidate = std::nextafter(
          candidate, std::numeric_limits<float>::infinity());
    }
    return static_cast<double>(candidate);
  }

  static double first_float_strictly_below(const double upper_bound) noexcept {
    float candidate = static_cast<float>(upper_bound);
    if (static_cast<double>(candidate) >= upper_bound) {
      candidate = std::nextafter(
          candidate, -std::numeric_limits<float>::infinity());
    }
    return static_cast<double>(candidate);
  }

  static std::array<double, 7> make_lower() noexcept {
    std::array<double, 7> result{};
    for (std::size_t joint = 0U; joint < result.size(); ++joint) {
      result[joint] =
          first_float_strictly_above(HardSafetyLimits::kSafeJointLower[joint]);
    }
    return result;
  }

  static std::array<double, 7> make_upper() noexcept {
    std::array<double, 7> result{};
    for (std::size_t joint = 0U; joint < result.size(); ++joint) {
      result[joint] =
          first_float_strictly_below(HardSafetyLimits::kSafeJointUpper[joint]);
    }
    return result;
  }
};

class FrankaV225Interpolator final {
 public:
  static constexpr double kTargetLowpassCutoffHz = 100.0;
  static constexpr double kTrackingNaturalFrequencyHz = 6.0;
  static constexpr double kTrackingDampingRatio = 1.0;

  FrankaV225Interpolator(const std::array<double, 7>& start_q,
                         const std::array<double, 7>& start_dq,
                         const std::array<double, 7>& start_ddq) {
    state_.q = SendableJointLimits::clamp(start_q);
    state_.dq = start_dq;
    state_.ddq = start_ddq;
    state_.filtered_target = state_.q;
  }

  const InterpolatedCommand& state() const noexcept { return state_; }

  // Contact is not a new policy target.  Re-anchor only the low-pass target
  // to the exact returned desired position while preserving q/dq/ddq.  The
  // normal jerk-limited critically damped step then decelerates smoothly;
  // clearing contact resumes toward the latest held policy target without a
  // discontinuity or a second trajectory generator.
  void hold_filtered_target_at(
      const std::array<double, 7>& desired_q) {
    state_.filtered_target = SendableJointLimits::clamp(desired_q);
  }

  // Synchronize only the desired command state with the exact FCI history.
  // The filtered policy target is deliberately retained.  This is not a
  // measured-q re-anchor: it handles a dropped command packet while keeping
  // the accepted interpolator's temporal state continuous.
  void synchronize_desired_history(const std::array<double, 7>& q,
                                   const std::array<double, 7>& dq,
                                   const std::array<double, 7>& ddq) {
    state_.q = SendableJointLimits::clamp(q);
    state_.dq = require_finite(dq);
    state_.ddq = require_finite(ddq);
  }

  const InterpolatedCommand& step(
      const std::array<double, 7>& held_target,
      const double dt_s) {
    if (!(dt_s > 0.0) || !std::isfinite(dt_s)) {
      throw std::invalid_argument("V225 interpolator dt must be finite and positive");
    }
    const std::array<double, 7> held = SendableJointLimits::clamp(held_target);
    constexpr double kPi = 3.14159265358979323846;
    const double lowpass_gain =
        dt_s / (dt_s + 1.0 / (2.0 * kPi * kTargetLowpassCutoffHz));
    const double omega = 2.0 * kPi * kTrackingNaturalFrequencyHz;
    for (std::size_t joint = 0U; joint < state_.q.size(); ++joint) {
      state_.filtered_target[joint] =
          lowpass_gain * held[joint] +
          (1.0 - lowpass_gain) * state_.filtered_target[joint];
      const double desired_acceleration =
          omega * omega *
              (state_.filtered_target[joint] - state_.q[joint]) -
          2.0 * kTrackingDampingRatio * omega * state_.dq[joint];
      const double acceleration_delta = std::clamp(
          desired_acceleration - state_.ddq[joint],
          -HardSafetyLimits::kMaximumCommandJerkRadS3 * dt_s,
          HardSafetyLimits::kMaximumCommandJerkRadS3 * dt_s);
      const double jerk_limited_acceleration = std::clamp(
          state_.ddq[joint] + acceleration_delta,
          -HardSafetyLimits::kMaximumCommandAccelerationRadS2,
          HardSafetyLimits::kMaximumCommandAccelerationRadS2);
      // A final hard velocity clip would make the emitted position's
      // backward-Euler acceleration jump when dq reaches the ceiling (for a
      // 60 mrad step the old path produced about 2500 rad/s^3 despite the
      // advertised 250 rad/s^3 limit).  Start reducing acceleration before
      // the boundary using the same invariant used by libfranka::limitRate.
      const double acceleration_to_velocity_gain =
          HardSafetyLimits::kMaximumCommandJerkRadS3 /
          HardSafetyLimits::kMaximumCommandAccelerationRadS2;
      const double safe_maximum_acceleration = std::min(
          acceleration_to_velocity_gain *
              (HardSafetyLimits::kMaximumCommandVelocityRadS -
               state_.dq[joint]),
          HardSafetyLimits::kMaximumCommandAccelerationRadS2);
      const double safe_minimum_acceleration = std::max(
          acceleration_to_velocity_gain *
              (-HardSafetyLimits::kMaximumCommandVelocityRadS -
               state_.dq[joint]),
          -HardSafetyLimits::kMaximumCommandAccelerationRadS2);
      const double acceleration = std::clamp(
          jerk_limited_acceleration, safe_minimum_acceleration,
          safe_maximum_acceleration);
      const double velocity = std::clamp(
          state_.dq[joint] + acceleration * dt_s,
          -HardSafetyLimits::kMaximumCommandVelocityRadS,
          HardSafetyLimits::kMaximumCommandVelocityRadS);
      const double position = state_.q[joint] + velocity * dt_s;
      state_.q[joint] = std::clamp(
          position, SendableJointLimits::lower()[joint],
          SendableJointLimits::upper()[joint]);
      state_.dq[joint] = velocity;
      state_.ddq[joint] = acceleration;
    }
    return state_;
  }

 private:
  static std::array<double, 7> require_finite(
      const std::array<double, 7>& value) {
    for (const double element : value) {
      if (!std::isfinite(element)) {
        throw std::invalid_argument("V225 interpolator state is non-finite");
      }
    }
    return value;
  }

  InterpolatedCommand state_{};
};

}  // namespace anydex::v94_franka_servo
