#pragma once

#include <array>
#include <csignal>
#include <cstdint>
#include <string>

#include "anydex/v94_franka_servo/backend.hpp"
#include "anydex/v94_franka_servo/channel.hpp"
#include "anydex/v94_franka_servo/protocol.hpp"

namespace anydex::v94_franka_servo {

class ServoClock {
 public:
  virtual ~ServoClock() = default;
  virtual std::uint64_t monotonic_ns() noexcept = 0;
  virtual std::uint64_t realtime_ns() noexcept = 0;
};

class PosixServoClock final : public ServoClock {
 public:
  std::uint64_t monotonic_ns() noexcept override;
  std::uint64_t realtime_ns() noexcept override;
};

struct RealtimeSchedulerProofData final {
  std::uint32_t policy{0U};
  std::uint32_t priority{0U};
  std::uint32_t cpu{0U};
  std::uint32_t affinity_cpu_count{0U};
};

// Test-only injection seam around the POSIX scheduling syscalls.  Production
// leaves this null and always executes sched_setscheduler followed by an exact
// readback proof before the active control handle is opened.
using RealtimeSchedulerConfigurator = bool (*)(
    RealtimeSchedulerProofData* proof,
    int* system_errno,
    void* context) noexcept;

struct ServoProcessConfig final {
  std::string robot_address;
  pid_t expected_parent_pid{0};
  std::array<std::uint8_t, kSessionNonceBytes> session_nonce{};
  HelloPayload hello{};
  std::uint64_t handshake_timeout_ns{3000000000ULL};
  const volatile std::sig_atomic_t* signal_stop_requested{nullptr};
  // Production enables this after taskset has restricted the child to one
  // logical CPU. Fake lifecycle tests leave it disabled and exercise the wire
  // proof independently.
  bool require_realtime_scheduler_proof{false};
  RealtimeSchedulerConfigurator realtime_scheduler_configurator{nullptr};
  void* realtime_scheduler_context{nullptr};
};

struct ServoRunResult final {
  int exit_code{1};
  FaultCode terminal_fault{FaultCode::kInternal};
  StopProofPayload stop_proof{};
  bool stop_proof_delivered{false};
};

ServoRunResult run_servo_process(const ServoProcessConfig& config,
                                 InheritedChannels* channels,
                                 RobotBackendFactory* backend_factory,
                                 ServoClock* clock) noexcept;

}  // namespace anydex::v94_franka_servo
