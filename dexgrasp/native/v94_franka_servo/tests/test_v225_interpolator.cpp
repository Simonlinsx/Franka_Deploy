#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "anydex/v94_franka_servo/franka_v225_interpolator.hpp"

namespace servo = anydex::v94_franka_servo;

int main() {
  std::array<double, 7> start{};
  std::array<double, 7> target{};
  // Production target payloads come from the float32 policy mapper.
  target[0] = static_cast<double>(0.045F * 0.40F);
  servo::FrankaV225Interpolator generator(start, {}, {});
  double previous_ddq = 0.0;
  double q1 = 0.0;
  double q50 = 0.0;
  double q100 = 0.0;
  double q300 = 0.0;
  for (int packet = 1; packet <= 300; ++packet) {
    const servo::InterpolatedCommand state = generator.step(target, 0.001);
    // The lower derivative envelope introduces a bounded 12.1 urad settling
    // overshoot.  Keep it explicitly regression-bounded instead of claiming
    // strict monotonicity.
    assert(state.q[0] >= -1.0e-15);
    assert(state.q[0] <= target[0] + 1.3e-5);
    assert(std::abs(state.ddq[0]) <=
           servo::HardSafetyLimits::kMaximumCommandAccelerationRadS2 +
               1.0e-12);
    const double jerk = (state.ddq[0] - previous_ddq) / 0.001;
    assert(std::abs(jerk) <=
           servo::HardSafetyLimits::kMaximumCommandJerkRadS3 + 1.0e-9);
    previous_ddq = state.ddq[0];
    if (packet == 1) {
      q1 = state.q[0];
    } else if (packet == 50) {
      q50 = state.q[0];
    } else if (packet == 100) {
      q100 = state.q[0];
    } else if (packet == 300) {
      q300 = state.q[0];
    }
  }
  assert(std::abs(q1 - 0.00000025) < 1.0e-12);
  assert(std::abs(q50 - 0.004284888723940) < 1.0e-12);
  assert(std::abs(q100 - 0.015174792163919) < 1.0e-12);
  assert(std::abs(q300 - 0.018001712903260) < 1.0e-12);

  // A lower Franka contact flag re-anchors only the filtered target.  The
  // generator must brake through the same jerk/acceleration envelope, then
  // resume smoothly toward the still-held policy target after contact clears.
  servo::FrankaV225Interpolator contact_generator(start, {}, {});
  servo::InterpolatedCommand contact_state{};
  for (int packet = 0; packet < 40; ++packet) {
    contact_state = contact_generator.step(target, 0.001);
  }
  assert(contact_state.dq[0] > 0.0);
  const double contact_q = contact_state.q[0];
  double contact_previous_ddq = contact_state.ddq[0];
  for (int packet = 0; packet < 300; ++packet) {
    contact_generator.hold_filtered_target_at(contact_generator.state().q);
    contact_state = contact_generator.step(contact_generator.state().q, 0.001);
    const double jerk = (contact_state.ddq[0] - contact_previous_ddq) / 0.001;
    assert(std::abs(contact_state.ddq[0]) <=
           servo::HardSafetyLimits::kMaximumCommandAccelerationRadS2 +
               1.0e-12);
    assert(std::abs(jerk) <=
           servo::HardSafetyLimits::kMaximumCommandJerkRadS3 + 1.0e-9);
    contact_previous_ddq = contact_state.ddq[0];
  }
  assert(std::abs(contact_state.dq[0]) < 1.0e-4);
  assert(contact_state.q[0] >= contact_q);
  assert(contact_state.q[0] - contact_q < 0.01);
  const double held_q = contact_state.q[0];
  for (int packet = 0; packet < 50; ++packet) {
    contact_state = contact_generator.step(target, 0.001);
  }
  assert(contact_state.q[0] > held_q);

  // The q_d-g015 adapter may publish a full 60 mrad target.  Validate the
  // derivatives of the actual emitted positions, not only the interpolator's
  // stored ddq, through the velocity-bound transition that used to create a
  // hidden ~2500 rad/s^3 spike.
  std::array<double, 7> large_target{};
  large_target[0] = 0.060;
  servo::FrankaV225Interpolator large_generator(start, {}, {});
  double emitted_q = start[0];
  double emitted_dq = 0.0;
  double emitted_ddq = 0.0;
  for (int packet = 0; packet < 500; ++packet) {
    const servo::InterpolatedCommand state =
        large_generator.step(large_target, 0.001);
    const double next_dq = (state.q[0] - emitted_q) / 0.001;
    const double next_ddq = (next_dq - emitted_dq) / 0.001;
    const double next_jerk = (next_ddq - emitted_ddq) / 0.001;
    assert(std::abs(next_dq) <=
           servo::HardSafetyLimits::kMaximumCommandVelocityRadS + 1.0e-9);
    assert(std::abs(next_ddq) <=
           servo::HardSafetyLimits::kMaximumCommandAccelerationRadS2 +
               1.0e-6);
    assert(std::abs(next_jerk) <=
           servo::HardSafetyLimits::kMaximumCommandJerkRadS3 + 1.0e-3);
    emitted_q = state.q[0];
    emitted_dq = next_dq;
    emitted_ddq = next_ddq;
  }

  // Every internally sendable boundary must survive the controller's
  // binary32 desired-history storage without crossing the unchanged physical
  // hard interval.  Also prove that it is the nearest such float: the next
  // float toward the physical boundary is no longer strictly inside.
  const auto& sendable_lower = servo::SendableJointLimits::lower();
  const auto& sendable_upper = servo::SendableJointLimits::upper();
  for (std::size_t joint = 0U; joint < sendable_lower.size(); ++joint) {
    assert(sendable_lower[joint] >
           servo::HardSafetyLimits::kSafeJointLower[joint]);
    assert(sendable_upper[joint] <
           servo::HardSafetyLimits::kSafeJointUpper[joint]);
    assert(static_cast<double>(static_cast<float>(sendable_lower[joint])) ==
           sendable_lower[joint]);
    assert(static_cast<double>(static_cast<float>(sendable_upper[joint])) ==
           sendable_upper[joint]);

    const float lower_toward_boundary = std::nextafter(
        static_cast<float>(sendable_lower[joint]),
        -std::numeric_limits<float>::infinity());
    const float upper_toward_boundary = std::nextafter(
        static_cast<float>(sendable_upper[joint]),
        std::numeric_limits<float>::infinity());
    assert(static_cast<double>(lower_toward_boundary) <=
           servo::HardSafetyLimits::kSafeJointLower[joint]);
    assert(static_cast<double>(upper_toward_boundary) >=
           servo::HardSafetyLimits::kSafeJointUpper[joint]);
  }

  // Exact-double endpoints and integration overshoot are both contracted to
  // the canonical sendable interval on all seven axes.  This is the same
  // clamp used after the production active-session limiter.
  const std::array<double, 7> clamped_lower =
      servo::SendableJointLimits::clamp(
          servo::HardSafetyLimits::kSafeJointLower);
  const std::array<double, 7> clamped_upper =
      servo::SendableJointLimits::clamp(
          servo::HardSafetyLimits::kSafeJointUpper);
  assert(clamped_lower == sendable_lower);
  assert(clamped_upper == sendable_upper);

  std::array<double, 7> lower_outward_dq{};
  std::array<double, 7> lower_outward_ddq{};
  std::array<double, 7> upper_outward_dq{};
  std::array<double, 7> upper_outward_ddq{};
  lower_outward_dq.fill(
      -servo::HardSafetyLimits::kMaximumCommandVelocityRadS);
  lower_outward_ddq.fill(
      -servo::HardSafetyLimits::kMaximumCommandAccelerationRadS2);
  upper_outward_dq.fill(
      servo::HardSafetyLimits::kMaximumCommandVelocityRadS);
  upper_outward_ddq.fill(
      servo::HardSafetyLimits::kMaximumCommandAccelerationRadS2);
  servo::FrankaV225Interpolator lower_boundary_generator(
      servo::HardSafetyLimits::kSafeJointLower, lower_outward_dq,
      lower_outward_ddq);
  servo::FrankaV225Interpolator upper_boundary_generator(
      servo::HardSafetyLimits::kSafeJointUpper, upper_outward_dq,
      upper_outward_ddq);
  const servo::InterpolatedCommand lower_boundary_state =
      lower_boundary_generator.step(
          servo::HardSafetyLimits::kSafeJointLower, 0.001);
  const servo::InterpolatedCommand upper_boundary_state =
      upper_boundary_generator.step(
          servo::HardSafetyLimits::kSafeJointUpper, 0.001);
  assert(lower_boundary_state.q == sendable_lower);
  assert(upper_boundary_state.q == sendable_upper);

  std::cout << "V225 interpolator frozen step response passed\n";
  return 0;
}
