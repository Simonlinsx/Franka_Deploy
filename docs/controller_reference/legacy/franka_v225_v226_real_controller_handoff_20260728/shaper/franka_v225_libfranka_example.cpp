// Integration skeleton for libfranka. This file is intentionally not part of
// the Isaac Lab build; copy the callback pattern into the real controller.

#include "franka_v225_interpolated_reference.hpp"

#include <array>
#include <atomic>
#include <cstdint>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/robot.h>

namespace v225 = simtoolreal::franka_v225;

// A single policy producer and a single 1 kHz consumer use this versioned
// mailbox. Atomic elements avoid a C++ data race; the sequence counter prevents
// the callback from accepting a partially published target.
class HeldTargetMailbox {
 public:
  explicit HeldTargetMailbox(const v225::Vector7d& initial_target) {
    for (std::size_t joint = 0; joint < target_.size(); ++joint) {
      target_[joint].store(initial_target[joint], std::memory_order_relaxed);
    }
  }

  void Publish(const v225::Vector7d& target) noexcept {
    sequence_.fetch_add(1, std::memory_order_acq_rel);  // odd: write active
    for (std::size_t joint = 0; joint < target_.size(); ++joint) {
      target_[joint].store(target[joint], std::memory_order_relaxed);
    }
    sequence_.fetch_add(1, std::memory_order_release);  // even: complete
  }

  v225::Vector7d ReadStable(const v225::Vector7d& fallback) const noexcept {
    v225::Vector7d snapshot{};
    for (int attempt = 0; attempt < 3; ++attempt) {
      const std::uint64_t before = sequence_.load(std::memory_order_acquire);
      if ((before & 1U) != 0U) {
        continue;
      }
      for (std::size_t joint = 0; joint < target_.size(); ++joint) {
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

void RunV225Control(franka::Robot& robot) {
  const franka::RobotState initial_state = robot.readOnce();
  const v225::Vector7d start_q = initial_state.q_d;
  v225::InterpolatedJointPositionGenerator generator(start_q);
  HeldTargetMailbox policy_target(start_q);

  // The policy thread must run outside this callback. Every 50 ms it should:
  // 1. read measured q and the previously accepted policy target;
  // 2. call MapPolicyActionToHeldTarget(...);
  // 3. call policy_target.Publish(...).
  // A missed policy deadline naturally holds the last published target.

  auto motion_generator =
      [&generator, &policy_target](
          const franka::RobotState& state,
          franka::Duration period) -> franka::JointPositions {
    if (period.toSec() <= 0.0) {
      return franka::JointPositions(state.q_d);
    }
    const v225::Vector7d held =
        policy_target.ReadStable(generator.state().filtered_target_rad);
    const auto& command = generator.Step(held, period.toSec());
    return franka::JointPositions(command.q_rad);
  };

  robot.control(
      motion_generator,
      franka::ControllerMode::kJointImpedance,
      true,                         // final official rate limiter enabled
      franka::kMaxCutoffFrequency   // no second low-pass filter
  );
}
