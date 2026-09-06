#include "anydex/v94_franka_servo/servo_core.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <poll.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "anydex/v94_franka_servo/safety_limits.hpp"

namespace anydex::v94_franka_servo {
bool desired_history_ambiguity_regression_for_test() noexcept;
bool desired_history_closed_form_regression_for_test() noexcept;
bool desired_history_near_zero_roundoff_regression_for_test() noexcept;
bool recoverable_fci_history_regression_for_test() noexcept;
bool mode_aware_episode_envelope_regression_for_test() noexcept;
bool jerk_shaper_multi_period_forward_invariant_regression_for_test() noexcept;
bool jerk_shaper_worst_case_wcet_regression_for_test() noexcept;
}  // namespace anydex::v94_franka_servo

namespace servo = anydex::v94_franka_servo;

static_assert(servo::HardSafetyLimits::kMaximumTargetCount == 720U);
static_assert(servo::HardSafetyLimits::kMaximumSessionDurationNs ==
              15000000000ULL);
static_assert(servo::HardSafetyLimits::kMaximumEpisodeDeltaRad == 1.21);
static_assert(servo::HardSafetyLimits::kMaximumCommandVelocityRadS == 0.50);
static_assert(servo::HardSafetyLimits::kMaximumMeasuredVelocityRadS == 0.70);
static_assert(
    servo::HardSafetyLimits::kMaximumUncontrolledContinuationPackets == 20U);
static_assert(
    servo::HardSafetyLimits::kFciFailStopDroppedPacketBound == 20U);
static_assert(
    servo::HardSafetyLimits::kMaximumRecoverableControlPeriodMs == 21U);
static_assert(servo::HardSafetyLimits::kMaximumTargetAgeNs == 50000000U);
static_assert(servo::HardSafetyLimits::kMaximumInterTargetTimeoutNs ==
              500000000U);

namespace {

// Independent copy of the seven doubles emitted by the production Python
// codec (deploy q_home float32 promoted to the ARM wire's float64 fields).
// Do not source this fixture from HardSafetyLimits: that previously let both
// sides of the C++ test share the same wrong rounded constants.
constexpr std::array<double, 7> kProductionPythonArmQHome{
    0x0.0p+0, -0x1.2353f80000000p-1, 0x0.0p+0,
    -0x1.67ae140000000p+1, 0x0.0p+0, 0x1.84bc6a0000000p+1,
    0x1.7b645a0000000p-1};

std::uint64_t monotonic_ns() {
  servo::PosixServoClock clock;
  return clock.monotonic_ns();
}

double float_round_trip(const double value) {
  return static_cast<double>(static_cast<float>(value));
}

std::uint32_t float_bits(const float value) {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float float_from_bits(const std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <std::size_t N>
std::array<double, N> float_round_trip(
    const std::array<double, N>& values) {
  std::array<double, N> output{};
  for (std::size_t index = 0U; index < N; ++index) {
    output[index] = float_round_trip(values[index]);
  }
  return output;
}

std::array<std::uint8_t, servo::kSha256Bytes> parse_sha(const char* text) {
  auto nibble = [](const char value) -> std::uint8_t {
    if (value >= '0' && value <= '9') {
      return static_cast<std::uint8_t>(value - '0');
    }
    return static_cast<std::uint8_t>(value - 'a' + 10);
  };
  std::array<std::uint8_t, servo::kSha256Bytes> output{};
  for (std::size_t index = 0U; index < output.size(); ++index) {
    output[index] = static_cast<std::uint8_t>(
        (nibble(text[index * 2U]) << 4U) | nibble(text[index * 2U + 1U]));
  }
  return output;
}

servo::RobotSample sample(const std::uint64_t time_ms,
                          const std::uint32_t period_ms,
                          const servo::RobotModeCode mode,
                          const std::array<double, 7>& q) {
  servo::RobotSample value{};
  value.q_rad = q;
  value.desired_q_rad = q;
  value.O_T_EE = servo::HardSafetyLimits::kExpectedFTee;
  value.F_T_EE = servo::HardSafetyLimits::kExpectedFTee;
  value.end_effector_mass_kg =
      servo::HardSafetyLimits::kExpectedEndEffectorMassKg;
  value.end_effector_com_m =
      servo::HardSafetyLimits::kExpectedEndEffectorComM;
  value.end_effector_inertia_kg_m2 =
      servo::HardSafetyLimits::kExpectedEndEffectorInertiaKgM2;
  value.external_load_mass_kg = 0.0;
  value.robot_time_ms = time_ms;
  value.control_period_ms = period_ms;
  value.mode = mode;
  value.status_flags = 0U;
  value.control_command_success_rate = 1.0;
  return value;
}

struct FakeShared final {
  std::array<double, 7> q = servo::HardSafetyLimits::kQHome;
  std::array<double, 7> desired_q = servo::HardSafetyLimits::kQHome;
  std::array<double, 7> desired_dq{};
  std::array<double, 7> desired_ddq{};
  std::array<double, 7> pending_q{};
  std::array<double, 7> pending_dq{};
  std::array<double, 7> pending_ddq{};
  std::uint64_t robot_time_ms{1000U};
  std::uint64_t active_reads{0U};
  std::uint64_t active_writes{0U};
  bool pending_command_valid{false};
  bool pending_command_drop{false};
  bool stopped{false};
  bool finish_seen{false};
  std::uint32_t initial_injected_period_ms{0U};
  std::uint32_t post_motion_injected_period_ms{0U};
  bool post_motion_long_period_injected{false};
  bool drop_first_moving_command{false};
  bool first_moving_command_dropped{false};
  bool drop_post_motion_long_period_command{false};
  bool post_motion_long_period_command_dropped{false};
  bool record_post_motion_recovery_write{false};
  bool inject_dynamic_state_error{false};
  bool inject_float_history_mismatch{false};
  bool inject_consecutive_recoverable_periods{false};
  bool stop_reports_reflex{false};
  std::uint32_t stop_calls{0U};
  std::uint32_t stop_settle_reads_remaining{0U};
  std::uint64_t moving_write_count{0U};
  std::array<double, 7> prior_written_q = servo::HardSafetyLimits::kQHome;
  double maximum_command_velocity_rad_s{0.0};
  double maximum_command_acceleration_rad_s2{0.0};
  double maximum_command_jerk_rad_s3{0.0};
  double post_motion_recovery_max_velocity_rad_s{0.0};
  double post_motion_recovery_max_acceleration_rad_s2{0.0};
  double post_motion_recovery_max_jerk_rad_s3{0.0};
  double post_motion_extrapolated_q_gap_rad{0.0};
  double post_motion_stale_history_acceleration_rad_s2{0.0};
  std::uint32_t real_drop_observed_q0_bits{0U};
  std::uint32_t real_drop_accepted_q0_bits{0U};
};

class FakeActive final : public servo::ActiveRobotSession {
 public:
  explicit FakeActive(std::shared_ptr<FakeShared> shared)
      : shared_(std::move(shared)) {}

  servo::RobotSample read_once() override {
    std::this_thread::sleep_for(std::chrono::microseconds(500));
    ++shared_->active_reads;
    std::uint32_t period_ms = shared_->active_reads == 1U ? 0U : 1U;
    if (shared_->initial_injected_period_ms != 0U &&
        shared_->active_reads == 50U) {
      period_ms = shared_->initial_injected_period_ms;
    }
    if (shared_->inject_consecutive_recoverable_periods &&
        (shared_->active_reads == 2U || shared_->active_reads == 3U)) {
      period_ms = 3U;
    }
    const bool inject_post_motion_long_period =
        shared_->post_motion_injected_period_ms != 0U &&
        !shared_->post_motion_long_period_injected &&
        shared_->moving_write_count >= 70U;
    if (inject_post_motion_long_period) {
      period_ms = shared_->post_motion_injected_period_ms;
      shared_->post_motion_long_period_injected = true;
      shared_->record_post_motion_recovery_write = true;
    }
    // ActiveControl reports period=0 on its first read even though the robot
    // timestamp is newer than the preceding static read.  Thereafter the
    // RobotState time jump must agree with the returned multi-ms period.
    shared_->robot_time_ms += period_ms == 0U ? 1U : period_ms;

    // A write is accepted or dropped only when the next FCI state is formed.
    // For returned period p, an accepted command is advanced p-1 packets while
    // a dropped command advances the previously confirmed state p packets.
    // Keeping this transition here (rather than eagerly in write_once) makes
    // the fake reproduce the real first-moving-packet fault exactly.
    constexpr double kCommandPeriodS = 0.001;
    std::uint32_t extrapolation_packets = 0U;
    bool command_dropped = false;
    if (period_ms != 0U) {
      assert(shared_->pending_command_valid);
      command_dropped = shared_->pending_command_drop;
      if (inject_post_motion_long_period &&
          shared_->drop_post_motion_long_period_command) {
        command_dropped = true;
        shared_->post_motion_long_period_command_dropped = true;
      }
      if (!command_dropped) {
        shared_->desired_q = shared_->pending_q;
        shared_->desired_dq = shared_->pending_dq;
        shared_->desired_ddq = shared_->pending_ddq;
        extrapolation_packets = period_ms - 1U;
      } else {
        extrapolation_packets = period_ms;
      }
      shared_->pending_command_valid = false;
      shared_->pending_command_drop = false;
    }
    const std::array<double, 7> pre_extrapolation_q = shared_->desired_q;
    const std::array<double, 7> pre_extrapolation_dq = shared_->desired_dq;
    const std::array<double, 7> pre_extrapolation_ddq = shared_->desired_ddq;
    for (std::uint32_t missed = 0U; missed < extrapolation_packets; ++missed) {
      for (std::size_t index = 0U; index < shared_->desired_q.size(); ++index) {
        shared_->desired_dq[index] +=
            shared_->desired_ddq[index] * kCommandPeriodS;
        shared_->desired_q[index] +=
            shared_->desired_dq[index] * kCommandPeriodS;
      }
    }
    if (shared_->record_post_motion_recovery_write) {
      for (std::size_t index = 0U; index < shared_->desired_q.size(); ++index) {
        shared_->post_motion_extrapolated_q_gap_rad = std::max(
            shared_->post_motion_extrapolated_q_gap_rad,
            std::abs(shared_->desired_q[index] -
                     shared_->prior_written_q[index]));

        // This is the next 1 ms semi-implicit command a workstation-side
        // trajectory copy would emit if it missed FCI's intervening states.
        // Differentiating it against FCI's already-extrapolated history must
        // violate the acceleration contract, proving that this fake catches
        // the stale-history implementation rather than merely exercising a
        // nonzero period field.
        const double stale_velocity =
            pre_extrapolation_dq[index] +
            pre_extrapolation_ddq[index] * kCommandPeriodS;
        const double stale_command =
            pre_extrapolation_q[index] + stale_velocity * kCommandPeriodS;
        const double fci_seen_stale_velocity =
            (stale_command - shared_->desired_q[index]) / kCommandPeriodS;
        const double fci_seen_stale_acceleration =
            (fci_seen_stale_velocity - shared_->desired_dq[index]) /
            kCommandPeriodS;
        shared_->post_motion_stale_history_acceleration_rad_s2 = std::max(
            shared_->post_motion_stale_history_acceleration_rad_s2,
            std::abs(fci_seen_stale_acceleration));
      }
    }
    shared_->q = shared_->desired_q;
    // Match the pinned research_interface ABI: FCI and the fake keep their
    // derivative history in double, but RobotState transports these fields as
    // floatarray<float> and libfranka merely promotes them back to double.
    auto value = sample(shared_->robot_time_ms, period_ms,
                        servo::RobotModeCode::kMove,
                        float_round_trip(shared_->q));
    value.desired_q_rad = float_round_trip(shared_->desired_q);
    value.desired_dq_rad_s = float_round_trip(shared_->desired_dq);
    value.desired_ddq_rad_s2 = float_round_trip(shared_->desired_ddq);
    if (command_dropped) {
      value.control_command_success_rate = 0.99;
    }
    if (shared_->inject_float_history_mismatch &&
        shared_->active_reads == 2U) {
      // Production accepts at most one binary32 ULP of desired-state
      // round-trip disagreement.  Inject exactly two ULP on J6 without
      // changing the fake FCI's internal double history; the servo must fault
      // before issuing the second write.
      float shifted = static_cast<float>(value.desired_q_rad[5]);
      shifted = std::nextafterf(shifted,
                                std::numeric_limits<float>::infinity());
      shifted = std::nextafterf(shifted,
                                std::numeric_limits<float>::infinity());
      value.desired_q_rad[5] = static_cast<double>(shifted);
    }
    if (shared_->inject_dynamic_state_error &&
        shared_->active_reads == 2U) {
      value.mode = servo::RobotModeCode::kReflex;
      value.status_flags = servo::kStateHasCurrentErrors |
                           servo::kStateHasLastMotionErrors;
      value.current_errors_text =
          "[\"joint_motion_generator_velocity_discontinuity\"]";
      value.last_motion_errors_text =
          "[\"joint_motion_generator_velocity_discontinuity\"]";
    }
    return value;
  }

  void write_once(const std::array<double, 7>& q_rad,
                  const bool motion_finished) override {
    if (shared_->pending_command_valid) {
      throw std::runtime_error(
          "fake FCI received a second command before returning a state");
    }
    double command_step = 0.0;
    double maximum_velocity = 0.0;
    double maximum_acceleration = 0.0;
    double maximum_jerk = 0.0;
    std::array<double, 7> next_dq{};
    std::array<double, 7> next_ddq{};
    constexpr double kCommandPeriodS = 0.001;
    for (std::size_t index = 0U; index < q_rad.size(); ++index) {
      command_step = std::max(
          command_step,
          std::abs(q_rad[index] - shared_->prior_written_q[index]));
      next_dq[index] =
          (q_rad[index] - shared_->desired_q[index]) / kCommandPeriodS;
      next_ddq[index] =
          (next_dq[index] - shared_->desired_dq[index]) / kCommandPeriodS;
      const double jerk =
          (next_ddq[index] - shared_->desired_ddq[index]) / kCommandPeriodS;
      maximum_velocity = std::max(maximum_velocity, std::abs(next_dq[index]));
      maximum_acceleration =
          std::max(maximum_acceleration, std::abs(next_ddq[index]));
      maximum_jerk = std::max(maximum_jerk, std::abs(jerk));
    }
    constexpr double kDerivativeTolerance = 1.0e-5;
    shared_->maximum_command_velocity_rad_s =
        std::max(shared_->maximum_command_velocity_rad_s, maximum_velocity);
    shared_->maximum_command_acceleration_rad_s2 = std::max(
        shared_->maximum_command_acceleration_rad_s2, maximum_acceleration);
    shared_->maximum_command_jerk_rad_s3 =
        std::max(shared_->maximum_command_jerk_rad_s3, maximum_jerk);
    if (maximum_velocity >
            servo::HardSafetyLimits::kMaximumCommandVelocityRadS +
                kDerivativeTolerance ||
        maximum_acceleration >
            servo::HardSafetyLimits::kMaximumCommandAccelerationRadS2 +
                kDerivativeTolerance ||
        maximum_jerk > servo::HardSafetyLimits::kMaximumCommandJerkRadS3 +
                           kDerivativeTolerance) {
      throw std::runtime_error(
          "fake FCI rejected discontinuous backward-Euler derivatives: v=" +
          std::to_string(maximum_velocity) + " a=" +
          std::to_string(maximum_acceleration) + " j=" +
          std::to_string(maximum_jerk));
    }
    if (shared_->record_post_motion_recovery_write) {
      shared_->post_motion_recovery_max_velocity_rad_s = maximum_velocity;
      shared_->post_motion_recovery_max_acceleration_rad_s2 =
          maximum_acceleration;
      shared_->post_motion_recovery_max_jerk_rad_s3 = maximum_jerk;
      shared_->record_post_motion_recovery_write = false;
    }
    shared_->prior_written_q = q_rad;
    if (command_step > 0.0) {
      ++shared_->moving_write_count;
    }
    shared_->pending_q = q_rad;
    shared_->pending_dq = next_dq;
    shared_->pending_ddq = next_ddq;
    shared_->pending_command_valid = true;
    if (shared_->drop_first_moving_command && command_step > 0.0 &&
        !shared_->first_moving_command_dropped) {
      shared_->pending_command_drop = true;
      shared_->first_moving_command_dropped = true;
      shared_->real_drop_observed_q0_bits =
          float_bits(static_cast<float>(shared_->desired_q[0]));
      shared_->real_drop_accepted_q0_bits =
          float_bits(static_cast<float>(q_rad[0]));
    }
    ++shared_->active_writes;
    shared_->finish_seen = shared_->finish_seen || motion_finished;
  }

 private:
  std::shared_ptr<FakeShared> shared_;
};

class FakeBackend final : public servo::RobotBackend {
 public:
  explicit FakeBackend(std::shared_ptr<FakeShared> shared)
      : shared_(std::move(shared)) {}

  servo::RobotSample read_once() override {
    if (!shared_->stopped && !shared_->finish_seen) {
      return sample(shared_->robot_time_ms, 0U, servo::RobotModeCode::kIdle,
                    shared_->q);
    }
    ++shared_->robot_time_ms;
    if (shared_->stop_settle_reads_remaining > 0U) {
      --shared_->stop_settle_reads_remaining;
      return sample(shared_->robot_time_ms, 1U, servo::RobotModeCode::kMove,
                    shared_->q);
    }
    auto stopped_sample = sample(
        shared_->robot_time_ms, 1U,
        shared_->stop_reports_reflex ? servo::RobotModeCode::kReflex
                                     : servo::RobotModeCode::kIdle,
        shared_->q);
    if (shared_->stop_reports_reflex) {
      stopped_sample.status_flags =
          servo::kStateHasCurrentErrors |
          servo::kStateHasLastMotionErrors |
          servo::kStateHasOnlyCommunicationConstraintsViolation;
      stopped_sample.current_errors_text =
          "[\"communication_constraints_violation\"]";
      stopped_sample.last_motion_errors_text =
          "[\"communication_constraints_violation\"]";
    }
    return stopped_sample;
  }

  std::unique_ptr<servo::ActiveRobotSession> start_joint_position_control()
      override {
    return std::make_unique<FakeActive>(shared_);
  }

  void stop() override {
    ++shared_->stop_calls;
    shared_->stopped = true;
  }

 private:
  std::shared_ptr<FakeShared> shared_;
};

class FakeFactory final : public servo::RobotBackendFactory {
 public:
  FakeFactory() : shared(std::make_shared<FakeShared>()) {}

  std::unique_ptr<servo::RobotBackend> create_enforced(
      const std::string& robot_address) override {
    if (robot_address != "fake-only") {
      throw std::runtime_error("unexpected fake address");
    }
    return std::make_unique<FakeBackend>(shared);
  }

  std::shared_ptr<FakeShared> shared;
};

class FakeSocketMetadata final : public servo::SocketMetadataProvider {
 public:
  FakeSocketMetadata(const int critical_fd, const int telemetry_fd)
      : critical_fd_(critical_fd), telemetry_fd_(telemetry_fd) {}

  bool read_socket_metadata(const int fd,
                            int* socket_type,
                            int* socket_domain,
                            pid_t* peer_pid,
                            uid_t* peer_uid,
                            int* system_errno) const noexcept override {
    if (fd != critical_fd_ && fd != telemetry_fd_) {
      *system_errno = EBADF;
      return false;
    }
    *socket_type = fd == critical_fd_ ? SOCK_SEQPACKET : SOCK_DGRAM;
    *socket_domain = AF_UNIX;
    *peer_pid = ::getpid();
    *peer_uid = ::geteuid();
    return true;
  }

 private:
  int critical_fd_;
  int telemetry_fd_;
};

struct FakeRealtimeSchedulerContext final {
  bool succeed{true};
  std::uint32_t cpu{11U};
  int failure_errno{EPERM};
};

bool configure_fake_realtime_scheduler(
    servo::RealtimeSchedulerProofData* proof,
    int* system_errno,
    void* opaque) noexcept {
  auto* context = static_cast<FakeRealtimeSchedulerContext*>(opaque);
  if (proof == nullptr || system_errno == nullptr || context == nullptr) {
    return false;
  }
  if (!context->succeed) {
    *system_errno = context->failure_errno;
    return false;
  }
  const int maximum_fifo_priority = ::sched_get_priority_max(SCHED_FIFO);
  if (maximum_fifo_priority <= 0) {
    *system_errno = errno;
    return false;
  }
  proof->policy = static_cast<std::uint32_t>(SCHED_FIFO);
  proof->priority = static_cast<std::uint32_t>(maximum_fifo_priority);
  proof->cpu = context->cpu;
  proof->affinity_cpu_count = 1U;
  *system_errno = 0;
  return true;
}

template <typename Payload>
void send_payload_at(
    const int fd,
    const servo::MessageKind kind,
    const std::uint64_t sequence,
    const std::uint64_t packet_monotonic_ns,
    const std::array<std::uint8_t, servo::kSessionNonceBytes>& nonce,
    const Payload& payload) {
  servo::EncodedPacket packet{};
  assert(servo::encode_payload(kind, sequence, packet_monotonic_ns, nonce,
                               payload, &packet) == servo::CodecError::kNone);
  assert(::send(fd, packet.bytes.data(), packet.size, MSG_NOSIGNAL) ==
         static_cast<ssize_t>(packet.size));
}

template <typename Payload>
void send_payload(const int fd,
                  const servo::MessageKind kind,
                  const std::uint64_t sequence,
                  const std::array<std::uint8_t, servo::kSessionNonceBytes>& nonce,
                  const Payload& payload) {
  send_payload_at(fd, kind, sequence, monotonic_ns(), nonce, payload);
}

servo::DecodedPacket receive_packet(
    const int fd,
    const std::uint64_t expected_sequence,
    const std::array<std::uint8_t, servo::kSessionNonceBytes>& nonce,
    std::array<std::uint8_t, servo::kMaximumPacketBytes>* storage,
    const int timeout_ms = 3000) {
  struct pollfd descriptor {};
  descriptor.fd = fd;
  descriptor.events = POLLIN;
  const int poll_result = ::poll(&descriptor, 1U, timeout_ms);
  if (poll_result != 1) {
    std::cerr << "timed out waiting for child packet sequence "
              << expected_sequence << ", poll_result=" << poll_result
              << ", revents=" << descriptor.revents << '\n';
    std::abort();
  }
  const ssize_t received = ::recv(fd, storage->data(), storage->size(), 0);
  assert(received > 0);
  servo::DecodedPacket decoded{};
  assert(servo::decode_packet(storage->data(), static_cast<std::size_t>(received),
                              expected_sequence, nonce,
                              &decoded) == servo::CodecError::kNone);
  return decoded;
}

servo::ArmPayload arm_payload(
    const servo::ControllerMode controller_mode =
        servo::ControllerMode::kLegacy) {
  servo::ArmPayload arm{};
  const auto profile = parse_sha(servo::HardSafetyLimits::kProfileSha256);
  const auto envelope = parse_sha(servo::HardSafetyLimits::kEnvelopeSha256);
  std::memcpy(arm.profile_sha256, profile.data(), profile.size());
  std::memcpy(arm.envelope_sha256, envelope.data(), envelope.size());
  std::memset(arm.permit_sha256, 0x11, sizeof(arm.permit_sha256));
  std::memset(arm.run_id_sha256, 0x22, sizeof(arm.run_id_sha256));
  std::memset(arm.authorization_id, 0x33, sizeof(arm.authorization_id));
  const std::uint64_t now = monotonic_ns();
  arm.authorization_issued_monotonic_ns = now - 1000000U;
  arm.authorization_expires_monotonic_ns = now + 4000000000ULL;
  arm.heartbeat_timeout_ns =
      servo::HardSafetyLimits::kMaximumHeartbeatTimeoutNs;
  arm.first_target_timeout_ns =
      servo::HardSafetyLimits::kMaximumFirstTargetTimeoutNs;
  arm.target_timeout_ns =
      servo::HardSafetyLimits::kMaximumInterTargetTimeoutNs;
  arm.maximum_session_duration_ns =
      servo::HardSafetyLimits::kMaximumSessionDurationNs;
  std::copy(kProductionPythonArmQHome.begin(),
            kProductionPythonArmQHome.end(), arm.q_home_rad);
  std::copy(servo::HardSafetyLimits::kSafeJointLower.begin(),
            servo::HardSafetyLimits::kSafeJointLower.end(),
            arm.safe_joint_lower_rad);
  std::copy(servo::HardSafetyLimits::kSafeJointUpper.begin(),
            servo::HardSafetyLimits::kSafeJointUpper.end(),
            arm.safe_joint_upper_rad);
  std::copy(servo::HardSafetyLimits::kExpectedFTee.begin(),
            servo::HardSafetyLimits::kExpectedFTee.end(), arm.expected_F_T_EE);
  arm.expected_end_effector_mass_kg =
      servo::HardSafetyLimits::kExpectedEndEffectorMassKg;
  std::copy(servo::HardSafetyLimits::kExpectedEndEffectorComM.begin(),
            servo::HardSafetyLimits::kExpectedEndEffectorComM.end(),
            arm.expected_end_effector_com_m);
  std::copy(servo::HardSafetyLimits::kExpectedEndEffectorInertiaKgM2.begin(),
            servo::HardSafetyLimits::kExpectedEndEffectorInertiaKgM2.end(),
            arm.expected_end_effector_inertia_kg_m2);
  arm.expected_external_load_mass_kg = 0.0;
  arm.maximum_target_count = 2U;
  arm.controller_mode = static_cast<std::uint32_t>(controller_mode);
  return arm;
}

}  // namespace

int main(int argc, char** argv) {
  const bool exercise_delayed_direct_stop =
      argc == 2 && std::string(argv[1]) == "--direct-stop-settle";
  const bool exercise_isolated_15ms_period =
      argc == 2 && std::string(argv[1]) == "--isolated-15ms-period";
  const bool exercise_isolated_20ms_period =
      argc == 2 && std::string(argv[1]) == "--isolated-20ms-period";
  const bool exercise_isolated_21ms_period =
      argc == 2 && std::string(argv[1]) == "--isolated-21ms-period";
  const bool exercise_isolated_22ms_period =
      argc == 2 && std::string(argv[1]) == "--isolated-22ms-period";
  const bool exercise_isolated_4ms_extrapolation =
      argc == 2 &&
      std::string(argv[1]) == "--isolated-4ms-extrapolation";
  const bool exercise_consecutive_recoverable_periods =
      argc == 2 &&
      std::string(argv[1]) == "--consecutive-recoverable-periods";
  const bool exercise_dynamic_state_diagnostic =
      argc == 2 && std::string(argv[1]) == "--dynamic-state-diagnostic";
  const bool exercise_float_history_mismatch =
      argc == 2 && std::string(argv[1]) == "--float-history-mismatch";
  const bool exercise_real_first_motion_drop =
      argc == 2 && std::string(argv[1]) == "--real-first-motion-drop";
  const bool exercise_isolated_4ms_drop =
      argc == 2 && std::string(argv[1]) == "--isolated-4ms-drop";
  const bool exercise_isolated_20ms_drop =
      argc == 2 && std::string(argv[1]) == "--isolated-20ms-drop";
  const bool exercise_realtime_scheduler_proof =
      argc == 2 && std::string(argv[1]) == "--realtime-scheduler-proof";
  const bool exercise_realtime_scheduler_syscall_failure =
      argc == 2 &&
      std::string(argv[1]) == "--realtime-scheduler-syscall-failure";
  const bool exercise_desired_history_ambiguity =
      argc == 2 &&
      std::string(argv[1]) == "--desired-history-ambiguity";
  const bool exercise_desired_history_closed_form =
      argc == 2 &&
      std::string(argv[1]) == "--desired-history-closed-form";
  const bool exercise_desired_history_near_zero_roundoff =
      argc == 2 &&
      std::string(argv[1]) == "--desired-history-near-zero-roundoff";
  const bool exercise_recoverable_fci_history =
      argc == 2 &&
      std::string(argv[1]) == "--recoverable-fci-history";
  const bool exercise_mode_aware_episode_envelope =
      argc == 2 &&
      std::string(argv[1]) == "--mode-aware-episode-envelope";
  const bool exercise_jerk_shaper_multi_period_forward_invariant =
      argc == 2 &&
      std::string(argv[1]) ==
          "--jerk-shaper-multi-period-forward-invariant";
  const bool exercise_jerk_shaper_worst_case_wcet =
      argc == 2 &&
      std::string(argv[1]) == "--jerk-shaper-worst-case-wcet";
  const bool exercise_qd_g015_large_target =
      argc == 2 && std::string(argv[1]) == "--qd-g015-large-target";
  const bool exercise_legacy_large_target_rejected =
      argc == 2 &&
      std::string(argv[1]) == "--legacy-large-target-rejected";
  if (argc > 2 ||
      (argc == 2 && !exercise_delayed_direct_stop &&
       !exercise_isolated_15ms_period &&
       !exercise_isolated_20ms_period &&
       !exercise_isolated_21ms_period &&
       !exercise_isolated_22ms_period &&
       !exercise_isolated_4ms_extrapolation &&
       !exercise_consecutive_recoverable_periods &&
       !exercise_dynamic_state_diagnostic &&
       !exercise_float_history_mismatch &&
       !exercise_real_first_motion_drop &&
       !exercise_isolated_4ms_drop &&
       !exercise_isolated_20ms_drop &&
       !exercise_realtime_scheduler_proof &&
       !exercise_realtime_scheduler_syscall_failure &&
       !exercise_desired_history_ambiguity &&
       !exercise_desired_history_closed_form &&
       !exercise_desired_history_near_zero_roundoff &&
       !exercise_recoverable_fci_history &&
       !exercise_mode_aware_episode_envelope &&
       !exercise_jerk_shaper_multi_period_forward_invariant &&
       !exercise_jerk_shaper_worst_case_wcet &&
       !exercise_qd_g015_large_target &&
       !exercise_legacy_large_target_rejected)) {
    return 2;
  }
  if (exercise_desired_history_ambiguity) {
    assert(servo::desired_history_ambiguity_regression_for_test());
    std::cout << "v94 desired-history ambiguity rejection passed\n";
    return 0;
  }
  if (exercise_desired_history_closed_form) {
    assert(servo::desired_history_closed_form_regression_for_test());
    std::cout <<
        "v94 desired-history p4/p10/p21 closed-form oracle passed\n";
    return 0;
  }
  if (exercise_desired_history_near_zero_roundoff) {
    assert(servo::desired_history_near_zero_roundoff_regression_for_test());
    std::cout << "v94 desired-history near-zero roundoff passed\n";
    return 0;
  }
  if (exercise_recoverable_fci_history) {
    assert(servo::recoverable_fci_history_regression_for_test());
    std::cout << "v94 recoverable FCI history derivative transition passed\n";
    return 0;
  }
  if (exercise_mode_aware_episode_envelope) {
    assert(servo::mode_aware_episode_envelope_regression_for_test());
    std::cout <<
        "v94 mode-aware legacy/qd-g015 episode envelope regression passed\n";
    return 0;
  }
  if (exercise_jerk_shaper_multi_period_forward_invariant) {
    assert(
        servo::jerk_shaper_multi_period_forward_invariant_regression_for_test());
    std::cout <<
        "v94 jerk shaper 1..20 packet FCI fail-stop horizon sweep passed\n";
    return 0;
  }
  if (exercise_jerk_shaper_worst_case_wcet) {
    assert(servo::jerk_shaper_worst_case_wcet_regression_for_test());
    return 0;
  }
  int critical[2]{};
  int telemetry[2]{};
  assert(::socketpair(AF_UNIX, SOCK_SEQPACKET, 0, critical) == 0);
  assert(::socketpair(AF_UNIX, SOCK_DGRAM, 0, telemetry) == 0);

  std::array<std::uint8_t, servo::kSessionNonceBytes> nonce{};
  for (std::size_t index = 0U; index < nonce.size(); ++index) {
    nonce[index] = static_cast<std::uint8_t>(0xA0U + index);
  }
  servo::HelloPayload hello{};
  hello.process_id = static_cast<std::uint32_t>(::getpid());
  hello.state_decimation = servo::HardSafetyLimits::kStateDecimation;
  hello.protocol_version = servo::kProtocolVersion;
  hello.safety_limits_schema = 3U;
  hello.maximum_command_velocity_rad_s = 0.50;
  hello.maximum_command_acceleration_rad_s2 = 5.0;
  hello.maximum_command_jerk_rad_s3 = 250.0;
  hello.maximum_start_error_rad = 0.01;
  hello.maximum_tick_target_delta_rad = 0.020;
  hello.maximum_episode_delta_rad = 1.21;
  hello.maximum_tracking_error_rad = 0.01;
  hello.maximum_read_to_write_s = 0.0008;

  servo::ServoProcessConfig config{};
  config.robot_address = "fake-only";
  config.expected_parent_pid = ::getpid();
  config.session_nonce = nonce;
  config.hello = hello;
  FakeRealtimeSchedulerContext realtime_scheduler_context{};
  if (exercise_realtime_scheduler_proof ||
      exercise_realtime_scheduler_syscall_failure) {
    config.require_realtime_scheduler_proof = true;
    config.realtime_scheduler_configurator =
        &configure_fake_realtime_scheduler;
    realtime_scheduler_context.succeed =
        !exercise_realtime_scheduler_syscall_failure;
    config.realtime_scheduler_context = &realtime_scheduler_context;
  }
  FakeSocketMetadata fake_socket_metadata(critical[1], telemetry[1]);
  servo::InheritedChannels child_channels(critical[1], telemetry[1],
                                           &fake_socket_metadata);
  servo::PosixServoClock clock;
  FakeFactory factory;
  constexpr std::uint32_t kRealObservedQ0Bits = 945916694U;
  if (exercise_real_first_motion_drop) {
    const double real_observed_q0 =
        static_cast<double>(float_from_bits(kRealObservedQ0Bits));
    factory.shared->q[0] = real_observed_q0;
    factory.shared->desired_q[0] = real_observed_q0;
    factory.shared->prior_written_q[0] = real_observed_q0;
  }
  factory.shared->stop_settle_reads_remaining =
      exercise_delayed_direct_stop ? 50U : 0U;
  factory.shared->initial_injected_period_ms =
      exercise_isolated_15ms_period
          ? 15U
          : (exercise_isolated_20ms_period
                 ? 20U
                 : (exercise_isolated_21ms_period
                        ? 21U
                        : (exercise_isolated_22ms_period ? 22U : 0U)));
  factory.shared->post_motion_injected_period_ms =
      (exercise_isolated_20ms_period || exercise_isolated_20ms_drop)
          ? 20U
          : ((exercise_isolated_4ms_extrapolation ||
              exercise_isolated_4ms_drop)
                 ? 4U
                 : 0U);
  factory.shared->drop_first_moving_command =
      exercise_real_first_motion_drop;
  factory.shared->drop_post_motion_long_period_command =
      exercise_isolated_4ms_drop || exercise_isolated_20ms_drop;
  factory.shared->inject_dynamic_state_error =
      exercise_dynamic_state_diagnostic;
  factory.shared->inject_float_history_mismatch =
      exercise_float_history_mismatch;
  factory.shared->inject_consecutive_recoverable_periods =
      exercise_consecutive_recoverable_periods;
  factory.shared->stop_reports_reflex =
      !exercise_dynamic_state_diagnostic &&
      !exercise_float_history_mismatch;
  servo::ServoRunResult run_result{};
  std::thread child([&] {
    run_result =
        servo::run_servo_process(config, &child_channels, &factory, &clock);
  });

  std::array<std::uint8_t, servo::kMaximumPacketBytes> storage{};
  auto decoded = receive_packet(critical[0], 1U, nonce, &storage);
  assert(decoded.header.kind ==
         static_cast<std::uint16_t>(servo::MessageKind::kHello));
  servo::HelloPayload received_hello{};
  assert(servo::copy_payload(decoded, &received_hello) == servo::CodecError::kNone);
  assert(received_hello.process_id == static_cast<std::uint32_t>(::getpid()));

  std::uint64_t parent_sequence = 1U;
  const servo::ArmPayload arm = arm_payload(
      exercise_qd_g015_large_target ? servo::ControllerMode::kQdG015
                                    : servo::ControllerMode::kLegacy);
  send_payload(critical[0], servo::MessageKind::kArm, parent_sequence++, nonce,
               arm);
  decoded = receive_packet(critical[0], 2U, nonce, &storage);
  if (exercise_realtime_scheduler_syscall_failure) {
    assert(decoded.header.kind ==
           static_cast<std::uint16_t>(servo::MessageKind::kFault));
    servo::FaultPayload scheduler_fault{};
    assert(servo::copy_payload(decoded, &scheduler_fault) ==
           servo::CodecError::kNone);
    assert(scheduler_fault.fault_code ==
           static_cast<std::uint32_t>(servo::FaultCode::kStaticPreflight));
    assert(scheduler_fault.system_errno == EPERM);
    const std::string fault_detail(
        scheduler_fault.detail,
        scheduler_fault.detail +
            std::min<std::size_t>(scheduler_fault.detail_bytes,
                                  sizeof(scheduler_fault.detail)));
    assert(fault_detail.find(
               "could not set and inspect native realtime scheduler proof") !=
           std::string::npos);
    decoded = receive_packet(critical[0], 3U, nonce, &storage);
    assert(decoded.header.kind ==
           static_cast<std::uint16_t>(servo::MessageKind::kStopProof));
    child.join();
    assert(run_result.exit_code == 1);
    assert(run_result.terminal_fault == servo::FaultCode::kStaticPreflight);
    assert(factory.shared->active_reads == 0U);
    assert(factory.shared->active_writes == 0U);
    assert(factory.shared->stop_calls == 0U);
    ::close(critical[0]);
    ::close(critical[1]);
    ::close(telemetry[0]);
    ::close(telemetry[1]);
    std::cout <<
        "v94 native servo realtime scheduler syscall rejection passed\n";
    return 0;
  }
  assert(decoded.header.kind ==
         static_cast<std::uint16_t>(servo::MessageKind::kIpcReady));
  servo::IpcReadyPayload ipc_ready{};
  assert(servo::copy_payload(decoded, &ipc_ready) == servo::CodecError::kNone);
  if (exercise_realtime_scheduler_proof) {
    assert(ipc_ready.realtime_scheduler_policy ==
           static_cast<std::uint32_t>(SCHED_FIFO));
    assert(ipc_ready.realtime_scheduler_priority ==
           static_cast<std::uint32_t>(
               ::sched_get_priority_max(SCHED_FIFO)));
    assert(ipc_ready.realtime_cpu == realtime_scheduler_context.cpu);
    assert(ipc_ready.realtime_affinity_cpu_count == 1U);
  }

  if (exercise_isolated_22ms_period ||
      exercise_dynamic_state_diagnostic ||
      exercise_float_history_mismatch) {
    decoded = receive_packet(critical[0], 3U, nonce, &storage);
    assert(decoded.header.kind ==
           static_cast<std::uint16_t>(servo::MessageKind::kFault));
    servo::FaultPayload fault{};
    assert(servo::copy_payload(decoded, &fault) == servo::CodecError::kNone);
    const auto expected_fault = exercise_isolated_22ms_period
                                    ? servo::FaultCode::kControlPeriod
                                    : servo::FaultCode::kDynamicState;
    assert(fault.fault_code == static_cast<std::uint32_t>(expected_fault));
    const std::string fault_detail(
        fault.detail,
        fault.detail + std::min<std::size_t>(fault.detail_bytes,
                                             sizeof(fault.detail)));
    if (exercise_isolated_22ms_period) {
      assert(fault_detail.find(
                 "outside FCI recoverable returned range 1..21 ms") !=
             std::string::npos);
      assert(fault_detail.find("actual=22ms") != std::string::npos);
    } else if (exercise_dynamic_state_diagnostic) {
      assert(fault_detail.find("mode=Reflex(5)") != std::string::npos);
      assert(fault_detail.find("flags=33[current,last]") !=
             std::string::npos);
      assert(fault_detail.find(
                 "joint_motion_generator_velocity_discontinuity") !=
             std::string::npos);
      assert(fault_detail.find("jcontact") == std::string::npos);
      assert(fault_detail.find("jcollision") == std::string::npos);
    } else {
      assert(fault_detail.find(
                 "FCI desired history matched neither accepted nor dropped "
                 "field=q_d axis=5") !=
             std::string::npos);
      assert(fault_detail.find("observed_bits=") != std::string::npos);
      assert(fault_detail.find("accepted_bits=") != std::string::npos);
      assert(fault_detail.find("dropped_bits=") != std::string::npos);
    }

    decoded = receive_packet(critical[0], 4U, nonce, &storage);
    assert(decoded.header.kind ==
           static_cast<std::uint16_t>(servo::MessageKind::kStopProof));
    servo::StopProofPayload proof{};
    assert(servo::copy_payload(decoded, &proof) == servo::CodecError::kNone);
    child.join();

    assert(run_result.exit_code == 1);
    assert(run_result.terminal_fault == expected_fault);
    assert(run_result.stop_proof_delivered);
    assert(proof.terminal_fault_code ==
           static_cast<std::uint32_t>(expected_fault));
    assert(proof.fault_reply_delivered);
    assert(proof.active_read_count ==
           (exercise_isolated_22ms_period ? 50U : 2U));
    assert(proof.active_write_count ==
           (exercise_isolated_22ms_period ? 49U : 1U));
    assert(proof.maximum_control_period_ms ==
           (exercise_isolated_22ms_period ? 22U : 1U));
    assert(!proof.finish_attempted && !proof.finish_succeeded);
    assert(proof.robot_stop_attempted && proof.robot_stop_succeeded);
    assert(proof.active_handle_released && proof.robot_backend_released);
    assert(proof.idle_dq_verified);
    assert(factory.shared->stopped && factory.shared->stop_calls == 1U);

    ::close(critical[0]);
    ::close(critical[1]);
    ::close(telemetry[0]);
    ::close(telemetry[1]);
    if (exercise_isolated_22ms_period) {
      for (std::size_t index = 0U; index < 3U; ++index) {
        assert(proof.verified_samples[index].robot_mode ==
               static_cast<std::uint32_t>(servo::RobotModeCode::kReflex));
        assert(proof.verified_samples[index].status_flags ==
               (servo::kStateHasCurrentErrors |
                servo::kStateHasLastMotionErrors |
                servo::kStateHasOnlyCommunicationConstraintsViolation));
      }
      std::cout <<
          "v94 native servo 22ms returned-period rejection proof passed\n";
    } else if (exercise_dynamic_state_diagnostic) {
      std::cout << "v94 native servo dynamic-state diagnostic passed\n";
    } else {
      std::cout <<
          "v94 native servo 2-ULP desired-history rejection passed\n";
    }
    return 0;
  }

  std::uint64_t heartbeat_sequence = 0U;
  std::uint64_t child_sequence = 3U;
  while (true) {
    struct pollfd descriptor {};
    descriptor.fd = critical[0];
    descriptor.events = POLLIN;
    const int ready = ::poll(&descriptor, 1U, 10);
    if (ready == 1) {
      decoded = receive_packet(critical[0], child_sequence++, nonce, &storage);
      if (decoded.header.kind == static_cast<std::uint16_t>(
                                     servo::MessageKind::kActionReady)) {
        servo::ActionReadyPayload action_ready{};
        assert(servo::copy_payload(decoded, &action_ready) ==
               servo::CodecError::kNone);
        assert(action_ready.consecutive_healthy_cycles ==
               servo::HardSafetyLimits::kHealthyCyclesBeforeAction);
        assert(action_ready.healthy_hold_robot_time_ms >= 100U);
        assert(action_ready.minimum_control_command_success_rate >= 0.99);
        assert(action_ready.latest_control_command_success_rate >= 0.99);
        assert(action_ready.active_read_count == action_ready.active_write_count);
        assert(action_ready.active_read_count ==
               servo::HardSafetyLimits::kHealthyCyclesBeforeAction + 1U);
        assert(action_ready.status_flags == 0U);
        if (exercise_isolated_15ms_period) {
          assert(action_ready.maximum_control_period_ms == 15U);
          assert(action_ready.cumulative_missed_robot_states == 14U);
        } else if (exercise_isolated_20ms_period) {
          assert(action_ready.maximum_control_period_ms == 20U);
          assert(action_ready.cumulative_missed_robot_states == 19U);
        } else if (exercise_isolated_21ms_period) {
          assert(action_ready.maximum_control_period_ms == 21U);
          assert(action_ready.cumulative_missed_robot_states == 20U);
        } else if (exercise_consecutive_recoverable_periods) {
          assert(action_ready.maximum_control_period_ms == 3U);
          assert(action_ready.cumulative_missed_robot_states == 4U);
        } else {
          assert(action_ready.maximum_control_period_ms == 1U);
          assert(action_ready.cumulative_missed_robot_states == 0U);
        }
        break;
      }
    }
    servo::HeartbeatPayload heartbeat{};
    heartbeat.heartbeat_sequence = ++heartbeat_sequence;
    send_payload(critical[0], servo::MessageKind::kHeartbeat,
                 parent_sequence++, nonce, heartbeat);
  }

  // Header creation and the kernel write are not atomic in the Python parent.
  // Reproduce a >50 ms GIL/scheduler pause after encoding but before send.
  // This remains below the independent 100 ms process-liveness watchdog, and
  // TARGET freshness will still have its own strict 50 ms deadline.
  servo::HeartbeatPayload delayed_heartbeat{};
  delayed_heartbeat.heartbeat_sequence = ++heartbeat_sequence;
  servo::EncodedPacket delayed_heartbeat_packet{};
  assert(servo::encode_payload(
             servo::MessageKind::kHeartbeat, parent_sequence++, monotonic_ns(),
             nonce, delayed_heartbeat, &delayed_heartbeat_packet) ==
         servo::CodecError::kNone);
  std::this_thread::sleep_for(std::chrono::milliseconds(65));
  assert(::send(critical[0], delayed_heartbeat_packet.bytes.data(),
                delayed_heartbeat_packet.size, MSG_NOSIGNAL) ==
         static_cast<ssize_t>(delayed_heartbeat_packet.size));

  servo::TargetPayload target{};
  target.target_sequence = 1U;
  target.observation_sequence = 1U;
  target.produced_monotonic_ns = monotonic_ns();
  std::copy(servo::HardSafetyLimits::kQHome.begin(),
            servo::HardSafetyLimits::kQHome.end(), target.target_q_rad);
  if (exercise_real_first_motion_drop) {
    target.target_q_rad[0] =
        static_cast<double>(float_from_bits(kRealObservedQ0Bits)) + 0.001;
  } else if (exercise_qd_g015_large_target) {
    // Exercise the real TARGET admission path with a q_d target that is
    // outside the legacy home-centered episode envelope while remaining
    // inside the absolute commissioned joint limits.
    target.target_q_rad[0] =
        servo::HardSafetyLimits::kQHome[0] +
        servo::HardSafetyLimits::kMaximumEpisodeDeltaRad + 0.01;
    assert(std::abs(target.target_q_rad[0] -
                    servo::HardSafetyLimits::kQHome[0]) >
           servo::HardSafetyLimits::kMaximumEpisodeDeltaRad);
    assert(target.target_q_rad[0] <
           servo::HardSafetyLimits::kSafeJointUpper[0]);
  } else if (exercise_legacy_large_target_rejected) {
    target.target_q_rad[5] -= 0.060;
  } else {
    // Exercise the real float-quantization magnitude (J6 is near +3 rad), not
    // only J1 near zero where binary32 position spacing is artificially tiny.
    target.target_q_rad[5] -= 0.001;
  }
  send_payload(critical[0], servo::MessageKind::kTarget, parent_sequence++,
               nonce, target);
  decoded = receive_packet(critical[0], child_sequence++, nonce, &storage);
  if (decoded.header.kind ==
      static_cast<std::uint16_t>(servo::MessageKind::kFault)) {
    servo::FaultPayload unexpected_fault{};
    assert(servo::copy_payload(decoded, &unexpected_fault) ==
           servo::CodecError::kNone);
    const std::string fault_detail(
        unexpected_fault.detail,
        unexpected_fault.detail +
            std::min<std::size_t>(unexpected_fault.detail_bytes,
                                  sizeof(unexpected_fault.detail)));
    if (exercise_legacy_large_target_rejected) {
      assert(unexpected_fault.fault_code == static_cast<std::uint32_t>(
                                                 servo::FaultCode::kTargetValidation));
      assert(fault_detail.find("legacy V94 tick envelope") !=
             std::string::npos);
      decoded = receive_packet(critical[0], child_sequence++, nonce, &storage);
      assert(decoded.header.kind == static_cast<std::uint16_t>(
                                        servo::MessageKind::kStopProof));
      servo::StopProofPayload rejected_proof{};
      assert(servo::copy_payload(decoded, &rejected_proof) ==
             servo::CodecError::kNone);
      child.join();
      assert(run_result.exit_code == 1);
      assert(run_result.terminal_fault == servo::FaultCode::kTargetValidation);
      assert(rejected_proof.terminal_fault_code == static_cast<std::uint32_t>(
                                                        servo::FaultCode::kTargetValidation));
      ::close(critical[0]);
      ::close(critical[1]);
      ::close(telemetry[0]);
      ::close(telemetry[1]);
      std::cout << "v94 legacy previous-target guard rejection passed\n";
      return 0;
    }
    std::cerr << "unexpected fault before first ACK: " << fault_detail << '\n';
  }
  assert(decoded.header.kind ==
         static_cast<std::uint16_t>(servo::MessageKind::kAck));
  servo::AckPayload ack{};
  assert(servo::copy_payload(decoded, &ack) == servo::CodecError::kNone);
  assert(ack.target_sequence == 1U && ack.observation_sequence == 1U);
  assert(ack.target_produced_monotonic_ns == target.produced_monotonic_ns);

  // Deliberately cross the independent 50 ms TARGET packet-age ceiling
  // between two fresh target packets.  This models one rejected 30 Hz camera
  // frames plus a short USB scheduling gap: the active command is held, and a
  // newly timestamped target can resume before the distinct 500 ms
  // inter-target watchdog expires.
  servo::HeartbeatPayload inter_target_heartbeat{};
  inter_target_heartbeat.heartbeat_sequence = ++heartbeat_sequence;
  send_payload(critical[0], servo::MessageKind::kHeartbeat,
               parent_sequence++, nonce, inter_target_heartbeat);
  // Deliberately exceed the former 100 ms watchdog to regress the real
  // 126 ms no-stage hold that stopped an otherwise healthy supervised run,
  // while keeping the independent 100 ms parent-liveness heartbeat healthy.
  std::this_thread::sleep_for(std::chrono::milliseconds(75));
  inter_target_heartbeat.heartbeat_sequence = ++heartbeat_sequence;
  send_payload(critical[0], servo::MessageKind::kHeartbeat,
               parent_sequence++, nonce, inter_target_heartbeat);
  std::this_thread::sleep_for(std::chrono::milliseconds(75));
  target.target_sequence = 2U;
  target.observation_sequence = 2U;
  target.produced_monotonic_ns = monotonic_ns();
  if (exercise_real_first_motion_drop) {
    target.target_q_rad[0] += 0.001;
  } else if (!exercise_qd_g015_large_target) {
    target.target_q_rad[5] -= 0.001;
  }
  send_payload(critical[0], servo::MessageKind::kTarget, parent_sequence++,
               nonce, target);
  decoded = receive_packet(critical[0], child_sequence++, nonce, &storage);
  if (decoded.header.kind ==
      static_cast<std::uint16_t>(servo::MessageKind::kFault)) {
    servo::FaultPayload unexpected_fault{};
    assert(servo::copy_payload(decoded, &unexpected_fault) ==
           servo::CodecError::kNone);
    std::cerr << "unexpected fault before second ACK: "
              << std::string(
                     unexpected_fault.detail,
                     unexpected_fault.detail +
                         std::min<std::size_t>(unexpected_fault.detail_bytes,
                                              sizeof(unexpected_fault.detail)))
              << '\n';
  }
  assert(decoded.header.kind ==
         static_cast<std::uint16_t>(servo::MessageKind::kAck));
  assert(servo::copy_payload(decoded, &ack) == servo::CodecError::kNone);
  assert(ack.target_sequence == 2U && ack.observation_sequence == 2U);
  assert(ack.target_produced_monotonic_ns == target.produced_monotonic_ns);

  servo::StopPayload stop{};
  stop.reason_code = static_cast<std::uint32_t>(servo::StopReason::kRequested);
  stop.requested_monotonic_ns = monotonic_ns();
  // STOP is fail-safe and must remain actionable even if its user-space header
  // was encoded before a long scheduler pause.
  send_payload_at(
      critical[0], servo::MessageKind::kStop, parent_sequence++,
      monotonic_ns() - servo::HardSafetyLimits::kMaximumTargetAgeNs - 1000000U,
      nonce, stop);
  decoded = receive_packet(critical[0], child_sequence++, nonce, &storage);
  assert(decoded.header.kind ==
         static_cast<std::uint16_t>(servo::MessageKind::kStopProof));
  servo::StopProofPayload proof{};
  assert(servo::copy_payload(decoded, &proof) == servo::CodecError::kNone);
  child.join();

  assert(run_result.exit_code == 0);
  assert(run_result.terminal_fault == servo::FaultCode::kNone);
  assert(run_result.stop_proof_delivered);
  assert(!proof.finish_attempted && !proof.finish_succeeded);
  assert(proof.robot_stop_attempted && proof.robot_stop_succeeded);
  assert(proof.active_handle_released && proof.robot_backend_released);
  assert(proof.idle_dq_verified);
  assert(proof.stop_consecutive_idle_samples == 3U);
  if (exercise_delayed_direct_stop) {
    assert(proof.stop_verification_samples == 53U);
  } else {
    assert(proof.stop_verification_samples == 3U);
  }
  if (exercise_isolated_20ms_period ||
      exercise_isolated_4ms_extrapolation ||
      exercise_isolated_4ms_drop || exercise_isolated_20ms_drop) {
    assert(factory.shared->post_motion_long_period_injected);
    assert(factory.shared->post_motion_extrapolated_q_gap_rad > 1.0e-6);
    assert(factory.shared->post_motion_stale_history_acceleration_rad_s2 >
           servo::HardSafetyLimits::kMaximumCommandAccelerationRadS2);
    assert(factory.shared->post_motion_recovery_max_velocity_rad_s <=
           servo::HardSafetyLimits::kMaximumCommandVelocityRadS + 1.0e-5);
    assert(factory.shared->post_motion_recovery_max_acceleration_rad_s2 <=
           servo::HardSafetyLimits::kMaximumCommandAccelerationRadS2 +
               1.0e-5);
    assert(factory.shared->post_motion_recovery_max_jerk_rad_s3 <=
           servo::HardSafetyLimits::kMaximumCommandJerkRadS3 + 1.0e-5);
  }
  if (exercise_real_first_motion_drop) {
    assert(factory.shared->first_moving_command_dropped);
    assert(factory.shared->real_drop_observed_q0_bits ==
           kRealObservedQ0Bits);
    const float observed_q0 =
        float_from_bits(factory.shared->real_drop_observed_q0_bits);
    const float accepted_q0 =
        float_from_bits(factory.shared->real_drop_accepted_q0_bits);
    assert(accepted_q0 > observed_q0);
    assert(accepted_q0 < observed_q0 + 0.001F);
  }
  if (exercise_isolated_4ms_drop || exercise_isolated_20ms_drop) {
    assert(factory.shared->post_motion_long_period_command_dropped);
  }
  assert(proof.target_ack_count == 2U);
  assert(factory.shared->moving_write_count >= 40U);
  assert(factory.shared->maximum_command_velocity_rad_s <=
         servo::HardSafetyLimits::kMaximumCommandVelocityRadS + 1.0e-5);
  assert(factory.shared->maximum_command_acceleration_rad_s2 <=
         servo::HardSafetyLimits::kMaximumCommandAccelerationRadS2 + 1.0e-5);
  assert(factory.shared->maximum_command_jerk_rad_s3 <=
         servo::HardSafetyLimits::kMaximumCommandJerkRadS3 + 1.0e-5);
  assert(proof.maximum_control_period_ms ==
         (exercise_isolated_21ms_period
              ? 21U
              : ((exercise_isolated_20ms_period || exercise_isolated_20ms_drop)
              ? 20U
              : (exercise_isolated_15ms_period
                     ? 15U
                     : (exercise_consecutive_recoverable_periods
                            ? 3U
                            : ((exercise_isolated_4ms_extrapolation ||
                                exercise_isolated_4ms_drop)
                                   ? 4U
                                   : 1U))))));
  assert(!factory.shared->finish_seen && factory.shared->stopped);
  assert(factory.shared->stop_calls == 1U);
  for (std::size_t index = 0U; index < 3U; ++index) {
    assert(proof.verified_samples[index].robot_mode ==
           static_cast<std::uint32_t>(servo::RobotModeCode::kReflex));
    assert(proof.verified_samples[index].status_flags ==
           (servo::kStateHasCurrentErrors |
            servo::kStateHasLastMotionErrors |
            servo::kStateHasOnlyCommunicationConstraintsViolation));
    if (index != 0U) {
      assert(proof.verified_samples[index].robot_time_ms >
             proof.verified_samples[index - 1U].robot_time_ms);
    }
  }

  ::close(critical[0]);
  ::close(critical[1]);
  ::close(telemetry[0]);
  ::close(telemetry[1]);
  std::cout << "v94 native servo fake lifecycle/ACK/stop-proof passed\n";
  return 0;
}
