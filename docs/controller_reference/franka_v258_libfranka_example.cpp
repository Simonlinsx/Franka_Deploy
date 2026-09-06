// Real-controller integration skeleton for the V258 29D Student contract.
// Copy this ownership/timing pattern into the deployment controller; policy
// inference itself must remain outside the 1 kHz real-time callback.

#include "franka_v258_interpolated_reference.hpp"

#include <array>
#include <atomic>
#include <cstdint>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/robot.h>

namespace v258 = simtoolreal::franka_v258;

class HeldTargetMailbox {
 public:
  explicit HeldTargetMailbox(const v258::Vector7d& initial_target) {
    Publish(initial_target);
  }

  void Publish(const v258::Vector7d& target) noexcept {
    sequence_.fetch_add(1, std::memory_order_acq_rel);
    for (std::size_t joint = 0; joint < 7; ++joint) {
      target_[joint].store(target[joint], std::memory_order_relaxed);
    }
    sequence_.fetch_add(1, std::memory_order_release);
  }

  v258::Vector7d ReadStable(const v258::Vector7d& fallback) const noexcept {
    v258::Vector7d snapshot{};
    for (int attempt = 0; attempt < 3; ++attempt) {
      const std::uint64_t before = sequence_.load(std::memory_order_acquire);
      if ((before & 1U) != 0U) {
        continue;
      }
      for (std::size_t joint = 0; joint < 7; ++joint) {
        snapshot[joint] = target_[joint].load(std::memory_order_relaxed);
      }
      const std::uint64_t after = sequence_.load(std::memory_order_acquire);
      if (before == after) {
        return snapshot;
      }
    }
    return fallback;
  }

 private:
  std::array<std::atomic<double>, 7> target_{};
  std::atomic<std::uint64_t> sequence_{0};
};

struct ControllerSnapshot {
  v258::Vector7d held_target{};
  v258::Vector7d q_d{};
  v258::Vector7d dq_d{};
  v258::Vector7d ddq_d{};
  v258::Vector7d measured_q{};
};

class ControllerSnapshotMailbox {
 public:
  void Publish(const ControllerSnapshot& snapshot) noexcept {
    sequence_.fetch_add(1, std::memory_order_acq_rel);
    for (std::size_t joint = 0; joint < 7; ++joint) {
      held_target_[joint].store(
          snapshot.held_target[joint], std::memory_order_relaxed);
      q_d_[joint].store(snapshot.q_d[joint], std::memory_order_relaxed);
      dq_d_[joint].store(snapshot.dq_d[joint], std::memory_order_relaxed);
      ddq_d_[joint].store(snapshot.ddq_d[joint], std::memory_order_relaxed);
      measured_q_[joint].store(
          snapshot.measured_q[joint], std::memory_order_relaxed);
    }
    sequence_.fetch_add(1, std::memory_order_release);
  }

  bool ReadStable(ControllerSnapshot* output) const noexcept {
    for (int attempt = 0; attempt < 3; ++attempt) {
      const std::uint64_t before = sequence_.load(std::memory_order_acquire);
      if ((before & 1U) != 0U) {
        continue;
      }
      for (std::size_t joint = 0; joint < 7; ++joint) {
        output->held_target[joint] =
            held_target_[joint].load(std::memory_order_relaxed);
        output->q_d[joint] = q_d_[joint].load(std::memory_order_relaxed);
        output->dq_d[joint] = dq_d_[joint].load(std::memory_order_relaxed);
        output->ddq_d[joint] = ddq_d_[joint].load(std::memory_order_relaxed);
        output->measured_q[joint] =
            measured_q_[joint].load(std::memory_order_relaxed);
      }
      const std::uint64_t after = sequence_.load(std::memory_order_acquire);
      if (before == after) {
        return true;
      }
    }
    return false;
  }

 private:
  std::array<std::atomic<double>, 7> held_target_{};
  std::array<std::atomic<double>, 7> q_d_{};
  std::array<std::atomic<double>, 7> dq_d_{};
  std::array<std::atomic<double>, 7> ddq_d_{};
  std::array<std::atomic<double>, 7> measured_q_{};
  std::atomic<std::uint64_t> sequence_{0};
};

void RunV258Control(franka::Robot& robot) {
  const franka::RobotState initial_state = robot.readOnce();
  // V258 initializes all software state from measured q after settling.
  const v258::Vector7d start_q = initial_state.q;
  v258::InterpolatedJointPositionGenerator generator(start_q);
  HeldTargetMailbox policy_target(start_q);
  ControllerSnapshotMailbox controller_snapshot;

  // The 20 Hz policy thread should perform this exact order:
  // 1. ReadStable() from controller_snapshot.
  // 2. BuildControllerStateObservation(..., execution_alpha=1.0).
  // 3. Append those 29 values after the 67D deployable proprioception.
  // 4. Run Student inference and obtain action[0:7].
  // 5. MapPolicyActionToHeldTarget(previous held_target, measured_q, action).
  // 6. Publish the new held target. A missed deadline holds the old target.

  auto motion_generator =
      [&generator, &policy_target, &controller_snapshot](
          const franka::RobotState& robot_state,
          franka::Duration period) -> franka::JointPositions {
    if (period.toSec() <= 0.0) {
      return franka::JointPositions(robot_state.q_d);
    }
    const v258::Vector7d held =
        policy_target.ReadStable(generator.state().filtered_target_rad);
    const auto& generated = generator.Step(held, period.toSec());
    controller_snapshot.Publish(
        ControllerSnapshot{
            held,
            generated.q_rad,
            generated.dq_rad_s,
            generated.ddq_rad_s2,
            robot_state.q});
    return franka::JointPositions(generated.q_rad);
  };

  robot.control(
      motion_generator,
      franka::ControllerMode::kJointImpedance,
      true,                         // retain official final rate limiter
      franka::kMaxCutoffFrequency   // avoid stacking a second low-pass
  );
}
