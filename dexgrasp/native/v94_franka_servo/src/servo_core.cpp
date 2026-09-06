#include "anydex/v94_franka_servo/servo_core.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <poll.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <time.h>
#include <utility>
#include <vector>

#include "anydex/v94_franka_servo/franka_v225_interpolator.hpp"
#include "anydex/v94_franka_servo/safety_limits.hpp"

namespace anydex::v94_franka_servo {
namespace {

constexpr std::uint32_t kHealthStaticVerified = 1U << 0U;
constexpr std::uint32_t kHealthCurrentCycleHealthy = 1U << 1U;
constexpr std::uint32_t kHealthActionReady = 1U << 2U;

class ServoFault final : public std::runtime_error {
 public:
  ServoFault(const FaultCode code,
             const bool terminal_finish_allowed,
             const std::string& detail,
             const int system_errno = 0)
      : std::runtime_error(detail),
        code(code),
        terminal_finish_allowed(terminal_finish_allowed),
        system_errno(system_errno) {}

  FaultCode code;
  bool terminal_finish_allowed;
  int system_errno;
};

bool configure_and_read_posix_realtime_scheduler(
    RealtimeSchedulerProofData* proof,
    int* system_errno,
    void*) noexcept {
  if (proof == nullptr || system_errno == nullptr) {
    return false;
  }
  *proof = RealtimeSchedulerProofData{};
  *system_errno = 0;
  errno = 0;
  const int maximum_fifo_priority = ::sched_get_priority_max(SCHED_FIFO);
  if (maximum_fifo_priority <= 0) {
    *system_errno = errno;
    return false;
  }
  struct sched_param requested {};
  requested.sched_priority = maximum_fifo_priority;
  // libfranka's enforced Robot constructor also requests realtime scheduling.
  // Repeat it here deliberately: this binds the claim to this exact native
  // child and makes success/readback part of IPC_READY rather than an
  // undocumented library side effect.
  if (::sched_setscheduler(0, SCHED_FIFO, &requested) != 0) {
    *system_errno = errno;
    return false;
  }
  const int policy = ::sched_getscheduler(0);
  struct sched_param observed {};
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  const int cpu = ::sched_getcpu();
  if (policy < 0 || ::sched_getparam(0, &observed) != 0 ||
      ::sched_getaffinity(0, sizeof(affinity), &affinity) != 0 || cpu < 0) {
    *system_errno = errno;
    return false;
  }
  proof->policy = static_cast<std::uint32_t>(policy);
  proof->priority = static_cast<std::uint32_t>(observed.sched_priority);
  proof->cpu = static_cast<std::uint32_t>(cpu);
  proof->affinity_cpu_count =
      static_cast<std::uint32_t>(CPU_COUNT(&affinity));
  if (!CPU_ISSET(cpu, &affinity)) {
    proof->affinity_cpu_count = 0U;
  }
  return true;
}

RealtimeSchedulerProofData require_realtime_scheduler_proof(
    const ServoProcessConfig& config) {
  RealtimeSchedulerProofData proof{};
  if (!config.require_realtime_scheduler_proof) {
    return proof;
  }
  int system_errno = 0;
  const RealtimeSchedulerConfigurator configurator =
      config.realtime_scheduler_configurator != nullptr
          ? config.realtime_scheduler_configurator
          : &configure_and_read_posix_realtime_scheduler;
  if (!configurator(&proof, &system_errno,
                    config.realtime_scheduler_context)) {
    throw ServoFault(
        FaultCode::kStaticPreflight, false,
        "could not set and inspect native realtime scheduler proof",
        system_errno);
  }
  errno = 0;
  const int maximum_fifo_priority = ::sched_get_priority_max(SCHED_FIFO);
  if (maximum_fifo_priority <= 0) {
    throw ServoFault(FaultCode::kStaticPreflight, false,
                     "could not resolve maximum SCHED_FIFO priority",
                     errno);
  }
  if (proof.policy != static_cast<std::uint32_t>(SCHED_FIFO) ||
      proof.priority != static_cast<std::uint32_t>(maximum_fifo_priority) ||
      proof.affinity_cpu_count != 1U) {
    throw ServoFault(
        FaultCode::kStaticPreflight, false,
        "native realtime scheduler proof failed: policy=" +
            std::to_string(proof.policy) +
            " priority=" + std::to_string(proof.priority) +
            " expected_priority=" + std::to_string(maximum_fifo_priority) +
            " cpu=" + std::to_string(proof.cpu) +
            " affinity_count=" +
            std::to_string(proof.affinity_cpu_count));
  }
  return proof;
}

bool finite(const double value) noexcept { return std::isfinite(value); }

template <std::size_t N>
bool finite_array(const std::array<double, N>& values) noexcept {
  for (const double value : values) {
    if (!finite(value)) {
      return false;
    }
  }
  return true;
}

template <std::size_t N>
double max_abs(const std::array<double, N>& values) noexcept {
  double maximum = 0.0;
  for (const double value : values) {
    maximum = std::max(maximum, std::abs(value));
  }
  return maximum;
}

template <std::size_t N>
double max_abs_difference(const std::array<double, N>& left,
                          const std::array<double, N>& right) noexcept {
  double maximum = 0.0;
  for (std::size_t index = 0U; index < N; ++index) {
    maximum = std::max(maximum, std::abs(left[index] - right[index]));
  }
  return maximum;
}

bool previous_target_delta_allowed(
    const ControllerMode mode,
    const std::array<double, 7>& target,
    const std::array<double, 7>& previous_target) noexcept {
  return mode == ControllerMode::kQdG015 ||
         max_abs_difference(target, previous_target) <=
             HardSafetyLimits::kMaximumTickTargetDeltaRad;
}

bool episode_delta_allowed(
    const ControllerMode mode,
    const std::array<double, 7>& value,
    const std::array<double, 7>& start_q) noexcept {
  return mode == ControllerMode::kQdG015 ||
         max_abs_difference(value, start_q) <=
             HardSafetyLimits::kMaximumEpisodeDeltaRad;
}

template <std::size_t N>
std::array<double, N> copy_wire_doubles(const double (&values)[N]) noexcept {
  std::array<double, N> output{};
  std::copy_n(values, N, output.begin());
  return output;
}

template <std::size_t N>
void copy_to_wire(const std::array<double, N>& source,
                  double (&destination)[N]) noexcept {
  std::copy(source.begin(), source.end(), destination);
}

std::array<float, 29> build_controller_state29(
    const std::array<double, 7>& held_target,
    const InterpolatedCommand& desired,
    const std::array<double, 7>& measured_q) noexcept {
  // Exact controller-state normalization retained by the legacy and V75 lanes.
  constexpr double kTargetErrorScaleRad = 0.018;
  constexpr double kTrackingErrorScaleRad = 0.05;
  std::array<float, 29> result{};
  for (std::size_t joint = 0U; joint < 7U; ++joint) {
    result[joint] = static_cast<float>(std::clamp(
        (held_target[joint] - desired.q[joint]) / kTargetErrorScaleRad,
        -4.0, 4.0));
    result[7U + joint] = static_cast<float>(std::clamp(
        (desired.q[joint] - measured_q[joint]) / kTrackingErrorScaleRad,
        -4.0, 4.0));
    result[14U + joint] = static_cast<float>(std::clamp(
        desired.dq[joint] / HardSafetyLimits::kMaximumCommandVelocityRadS,
        -1.0, 1.0));
    result[21U + joint] = static_cast<float>(std::clamp(
        desired.ddq[joint] /
            HardSafetyLimits::kMaximumCommandAccelerationRadS2,
        -1.0, 1.0));
  }
  result[28] = 1.0F;
  return result;
}

bool all_zero(const std::uint8_t* values, const std::size_t count) noexcept {
  std::uint8_t combined = 0U;
  for (std::size_t index = 0U; index < count; ++index) {
    combined = static_cast<std::uint8_t>(combined | values[index]);
  }
  return combined == 0U;
}

std::uint8_t hex_nibble(const char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  throw std::invalid_argument("invalid compiled digest");
}

std::array<std::uint8_t, kSha256Bytes> parse_sha256(const char* text) {
  std::array<std::uint8_t, kSha256Bytes> output{};
  for (std::size_t index = 0U; index < output.size(); ++index) {
    output[index] = static_cast<std::uint8_t>(
        (hex_nibble(text[index * 2U]) << 4U) |
        hex_nibble(text[index * 2U + 1U]));
  }
  if (text[output.size() * 2U] != '\0') {
    throw std::invalid_argument("compiled digest has wrong length");
  }
  return output;
}

bool bytes_equal(const std::uint8_t* left,
                 const std::uint8_t* right,
                 const std::size_t count) noexcept {
  std::uint8_t difference = 0U;
  for (std::size_t index = 0U; index < count; ++index) {
    difference = static_cast<std::uint8_t>(difference | (left[index] ^ right[index]));
  }
  return difference == 0U;
}

bool state_is_clear(const RobotSample& sample) noexcept {
  return sample.status_flags == 0U &&
         (sample.status_flags & ~kKnownStateStatusMask) == 0U;
}

constexpr std::uint32_t kContactStatusMask =
    kStateHasJointContact | kStateHasCartesianContact;

bool state_has_contact(const RobotSample& sample) noexcept {
  return (sample.status_flags & kContactStatusMask) != 0U;
}

bool state_is_dynamic_motion_safe(const RobotSample& sample) noexcept {
  // Contact-only flags are an intentional input to the compliant hold path.
  // Errors, collision flags, communication faults and unknown bits remain
  // terminal and are never masked here.
  return (sample.status_flags & ~kContactStatusMask) == 0U &&
         (sample.status_flags & ~kKnownStateStatusMask) == 0U;
}

bool sample_vectors_are_finite(const RobotSample& sample) noexcept {
  return finite_array(sample.q_rad) && finite_array(sample.dq_rad_s) &&
         finite_array(sample.desired_q_rad) &&
         finite_array(sample.desired_dq_rad_s) &&
         finite_array(sample.desired_ddq_rad_s2) &&
         finite_array(sample.O_T_EE) && finite(sample.control_command_success_rate);
}

const char* robot_mode_name(const RobotModeCode mode) noexcept {
  switch (mode) {
    case RobotModeCode::kIdle:
      return "Idle";
    case RobotModeCode::kMove:
      return "Move";
    case RobotModeCode::kOther:
      return "Other";
    case RobotModeCode::kGuiding:
      return "Guiding";
    case RobotModeCode::kReflex:
      return "Reflex";
    case RobotModeCode::kUserStopped:
      return "UserStopped";
    case RobotModeCode::kAutomaticErrorRecovery:
      return "AutomaticErrorRecovery";
    case RobotModeCode::kUnknown:
      return "Unknown";
  }
  return "Unknown";
}

void append_status_name(std::string* output,
                        const std::uint32_t flags,
                        const std::uint32_t flag,
                        const char* name) {
  if ((flags & flag) == 0U) {
    return;
  }
  if (!output->empty()) {
    output->append(",");
  }
  output->append(name);
}

std::string status_names(const std::uint32_t flags) {
  std::string output;
  append_status_name(&output, flags, kStateHasCurrentErrors, "current");
  append_status_name(&output, flags, kStateHasLastMotionErrors, "last");
  append_status_name(&output, flags, kStateHasJointContact, "jcontact");
  append_status_name(&output, flags, kStateHasJointCollision, "jcollision");
  append_status_name(&output, flags, kStateHasCartesianContact, "ccontact");
  append_status_name(&output, flags, kStateHasCartesianCollision, "ccollision");
  append_status_name(&output, flags,
                     kStateHasOnlyCommunicationConstraintsViolation,
                     "comm_only");
  const std::uint32_t unknown = flags & ~kKnownStateStatusMask;
  if (unknown != 0U) {
    if (!output.empty()) {
      output.append(",");
    }
    output.append("unknown=");
    output.append(std::to_string(unknown));
  }
  return output;
}

std::string dynamic_state_diagnostic(
    const RobotSample& sample,
    const std::array<double, 7>& previous_command,
    const std::array<double, 7>& prior_command,
    const std::uint32_t maximum_control_period_ms) {
  std::string detail = "dyn mode=";
  detail.append(robot_mode_name(sample.mode));
  detail.push_back('(');
  detail.append(std::to_string(static_cast<std::uint32_t>(sample.mode)));
  detail.append(") flags=");
  detail.append(std::to_string(sample.status_flags));
  const std::string names = status_names(sample.status_flags);
  if (!names.empty()) {
    detail.push_back('[');
    detail.append(names);
    detail.push_back(']');
  }
  detail.append(" p=");
  detail.append(std::to_string(sample.control_period_ms));
  detail.append(" maxp=");
  detail.append(std::to_string(maximum_control_period_ms));
  // Preserve the controller's primary error text near the front of the
  // bounded fault payload; the desired-state diagnostics below explain the
  // workstation/FCI history mismatch without displacing that root cause.
  if ((sample.status_flags & kStateHasCurrentErrors) != 0U) {
    detail.append(" current=");
    detail.append(sample.current_errors_text);
  }
  detail.append(" dcmd=");
  detail.append(std::to_string(
      max_abs_difference(previous_command, prior_command)));
  detail.append(" qdgap=");
  detail.append(std::to_string(
      max_abs_difference(sample.desired_q_rad, previous_command)));
  detail.append(" dqd=");
  detail.append(std::to_string(max_abs(sample.desired_dq_rad_s)));
  detail.append(" ddqd=");
  detail.append(std::to_string(max_abs(sample.desired_ddq_rad_s2)));
  if ((sample.status_flags & kStateHasLastMotionErrors) != 0U) {
    detail.append(" last=");
    detail.append(sample.last_motion_errors_text);
  }
  if (!sample_vectors_are_finite(sample)) {
    detail.append(" nonfinite_state_vector");
  }
  return detail;
}

void require_arm_contract(const ArmPayload& arm,
                          const std::uint64_t now_ns) {
  const auto profile = parse_sha256(HardSafetyLimits::kProfileSha256);
  const auto envelope = parse_sha256(HardSafetyLimits::kEnvelopeSha256);
  if (!bytes_equal(arm.profile_sha256, profile.data(), profile.size()) ||
      !bytes_equal(arm.envelope_sha256, envelope.data(), envelope.size())) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM profile/envelope digest differs from compiled V94 contract");
  }
  if (all_zero(arm.permit_sha256, sizeof(arm.permit_sha256)) ||
      all_zero(arm.run_id_sha256, sizeof(arm.run_id_sha256)) ||
      all_zero(arm.authorization_id, sizeof(arm.authorization_id))) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM run/permit/authorization binding is zero");
  }
  if (!(arm.authorization_issued_monotonic_ns <= now_ns &&
        now_ns < arm.authorization_expires_monotonic_ns)) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM authorization is not active");
  }
  if (arm.heartbeat_timeout_ns !=
          HardSafetyLimits::kMaximumHeartbeatTimeoutNs ||
      arm.first_target_timeout_ns !=
          HardSafetyLimits::kMaximumFirstTargetTimeoutNs ||
      arm.target_timeout_ns !=
          HardSafetyLimits::kMaximumInterTargetTimeoutNs ||
      arm.maximum_session_duration_ns !=
          HardSafetyLimits::kMaximumSessionDurationNs) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM watchdog/session values differ from fixed V94 values");
  }
  if (!all_zero(arm.reserved, sizeof(arm.reserved))) {
    throw ServoFault(FaultCode::kProtocol, false,
                     "ARM reserved bytes are not zero");
  }
  if (arm.controller_mode !=
          static_cast<std::uint32_t>(ControllerMode::kLegacy) &&
      arm.controller_mode !=
          static_cast<std::uint32_t>(ControllerMode::kQdG015)) {
    throw ServoFault(FaultCode::kProtocol, false,
                     "ARM controller_mode is unknown");
  }
  if (arm.maximum_target_count == 0U ||
      arm.maximum_target_count > HardSafetyLimits::kMaximumTargetCount) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM maximum_target_count is outside 1..720");
  }
  const auto q_home = copy_wire_doubles(arm.q_home_rad);
  const auto lower = copy_wire_doubles(arm.safe_joint_lower_rad);
  const auto upper = copy_wire_doubles(arm.safe_joint_upper_rad);
  const auto F_T_EE = copy_wire_doubles(arm.expected_F_T_EE);
  const auto com = copy_wire_doubles(arm.expected_end_effector_com_m);
  const auto inertia = copy_wire_doubles(arm.expected_end_effector_inertia_kg_m2);
  if (max_abs_difference(q_home, HardSafetyLimits::kQHome) > 1.0e-12) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM q_home differs from compiled V94 values");
  }
  if (max_abs_difference(lower, HardSafetyLimits::kSafeJointLower) > 1.0e-12) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM safe lower limits differ from compiled V94 values");
  }
  if (max_abs_difference(upper, HardSafetyLimits::kSafeJointUpper) > 1.0e-12) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM safe upper limits differ from compiled V94 values");
  }
  if (max_abs_difference(F_T_EE, HardSafetyLimits::kExpectedFTee) > 1.0e-12) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM F_T_EE differs from compiled V94 values");
  }
  if (max_abs_difference(com, HardSafetyLimits::kExpectedEndEffectorComM) >
      1.0e-12) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM end-effector COM differs from compiled V94 values");
  }
  if (max_abs_difference(inertia,
                         HardSafetyLimits::kExpectedEndEffectorInertiaKgM2) >
      1.0e-12) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM end-effector inertia differs from compiled V94 values");
  }
  if (!finite(arm.expected_end_effector_mass_kg) ||
      std::abs(arm.expected_end_effector_mass_kg -
               HardSafetyLimits::kExpectedEndEffectorMassKg) > 1.0e-12) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM end-effector mass differs from compiled V94 value");
  }
  if (!finite(arm.expected_external_load_mass_kg) ||
      std::abs(arm.expected_external_load_mass_kg) > 1.0e-12) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "ARM external-load mass differs from compiled V94 value");
  }
}

void require_preflight(const RobotSample& sample, const ArmPayload& arm) {
  if (sample.mode != RobotModeCode::kIdle || !state_is_clear(sample) ||
      !sample_vectors_are_finite(sample) ||
      max_abs(sample.dq_rad_s) > HardSafetyLimits::kStopMaximumVelocityRadS) {
    throw ServoFault(FaultCode::kStaticPreflight, false,
                     "fresh Franka preflight is not healthy Idle/rest");
  }
  const auto q_home = copy_wire_doubles(arm.q_home_rad);
  if (max_abs_difference(sample.q_rad, q_home) >
      HardSafetyLimits::kMaximumStartErrorRad) {
    throw ServoFault(FaultCode::kStaticPreflight, false,
                     "fresh Franka q differs from V94 q_home");
  }
  const auto expected_F_T_EE = copy_wire_doubles(arm.expected_F_T_EE);
  const auto expected_com = copy_wire_doubles(arm.expected_end_effector_com_m);
  const auto expected_inertia =
      copy_wire_doubles(arm.expected_end_effector_inertia_kg_m2);
  if (!finite_array(sample.F_T_EE) ||
      max_abs_difference(sample.F_T_EE, expected_F_T_EE) > 1.0e-8 ||
      !finite(sample.end_effector_mass_kg) ||
      std::abs(sample.end_effector_mass_kg -
               arm.expected_end_effector_mass_kg) > 1.0e-5 ||
      !finite_array(sample.end_effector_com_m) ||
      max_abs_difference(sample.end_effector_com_m, expected_com) > 1.0e-6 ||
      !finite_array(sample.end_effector_inertia_kg_m2) ||
      max_abs_difference(sample.end_effector_inertia_kg_m2,
                         expected_inertia) > 1.0e-6 ||
      !finite(sample.external_load_mass_kg) ||
      std::abs(sample.external_load_mass_kg -
               arm.expected_external_load_mass_kg) > 1.0e-5) {
    throw ServoFault(FaultCode::kStaticPreflight, false,
                     "same-Robot static tool/load provenance mismatch");
  }
}

void require_dynamic_state(const RobotSample& sample,
                           const std::array<double, 7>& previous_command,
                           const std::array<double, 7>& prior_command,
                           const std::array<double, 7>& start_q,
                           const ControllerMode controller_mode,
                           const std::uint32_t maximum_control_period_ms) {
  if (sample.mode != RobotModeCode::kMove ||
      !state_is_dynamic_motion_safe(sample) ||
      !sample_vectors_are_finite(sample)) {
    throw ServoFault(FaultCode::kDynamicState, false,
                     dynamic_state_diagnostic(sample, previous_command,
                                              prior_command,
                                              maximum_control_period_ms));
  }
  double maximum_measured_velocity = 0.0;
  std::size_t maximum_measured_velocity_axis = 0U;
  for (std::size_t index = 0U; index < sample.dq_rad_s.size(); ++index) {
    const double magnitude = std::abs(sample.dq_rad_s[index]);
    if (magnitude > maximum_measured_velocity) {
      maximum_measured_velocity = magnitude;
      maximum_measured_velocity_axis = index;
    }
  }
  if (maximum_measured_velocity >
      HardSafetyLimits::kMaximumMeasuredVelocityRadS) {
    throw ServoFault(FaultCode::kDynamicState, false,
                     "measured Franka velocity exceeds 0.70 rad/s: axis=" +
                         std::to_string(maximum_measured_velocity_axis + 1U) +
                         " max_abs_dq=" +
                         std::to_string(maximum_measured_velocity) +
                         "rad/s");
  }
  // desired_dq/ddq are returned FCI history, not a fresh nominal V225
  // generator output.  After one command packet is dropped, FCI legitimately
  // extrapolates the last accepted state before libfranka's official
  // position-dependent limitRate path recovers it on the next command.  Do
  // not apply the lower deployment tuning limits (0.50/5.0) to that returned
  // history: doing so faults before the official limiter can recover.  The
  // vectors are already required finite above; measured velocity, desired
  // position, tracking, controller-error and Franka reflex checks remain
  // enforced here and by libfranka.  The home-centered episode check remains
  // additionally enabled for legacy mode only.
  for (std::size_t index = 0U; index < 7U; ++index) {
    if (sample.q_rad[index] < HardSafetyLimits::kSafeJointLower[index] ||
        sample.q_rad[index] > HardSafetyLimits::kSafeJointUpper[index]) {
      throw ServoFault(FaultCode::kDynamicState, false,
                       "measured Franka q left compiled safe joint interval");
    }
    if (sample.desired_q_rad[index] <
            HardSafetyLimits::kSafeJointLower[index] ||
        sample.desired_q_rad[index] >
            HardSafetyLimits::kSafeJointUpper[index]) {
      throw ServoFault(FaultCode::kDynamicState, false,
                       "FCI desired q left compiled safe joint interval");
    }
  }
  if (!episode_delta_allowed(controller_mode, sample.desired_q_rad, start_q)) {
    throw ServoFault(FaultCode::kDynamicState, false,
                     "legacy FCI desired q left home-centered episode envelope");
  }
  if (max_abs_difference(sample.q_rad, previous_command) >
      HardSafetyLimits::kMaximumTrackingErrorRad) {
    throw ServoFault(FaultCode::kDynamicState, false,
                     "measured Franka q differs from prior safe command");
  }
}

constexpr double kCommandPacketPeriodS = 0.001;
constexpr double kDesiredHistoryQNumericalEquivalenceRad = 1.0e-12;
constexpr double kDesiredHistoryDqNumericalEquivalenceRadS = 1.0e-9;
constexpr double kDesiredHistoryDdqNumericalEquivalenceRadS2 = 1.0e-6;

struct ExactDesiredHistory final {
  std::array<double, 7> q{};
  std::array<double, 7> dq{};
  std::array<double, 7> ddq{};
};

std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool float_round_trip_value_matches(const double observed,
                                    const double expected,
                                    const double absolute_tolerance) noexcept {
  const float expected_float = static_cast<float>(expected);
  const float observed_float = static_cast<float>(observed);
  const float lower_float =
      std::nextafterf(expected_float,
                      -std::numeric_limits<float>::infinity());
  const float upper_float =
      std::nextafterf(expected_float,
                      std::numeric_limits<float>::infinity());
  return observed == static_cast<double>(observed_float) &&
         (observed_float == expected_float || observed_float == lower_float ||
          observed_float == upper_float ||
          std::abs(observed - expected) <= absolute_tolerance);
}

bool float_round_trip_array_matches(
    const std::array<double, 7>& observed,
    const std::array<double, 7>& expected,
    const double absolute_tolerance) noexcept {
  for (std::size_t index = 0U; index < observed.size(); ++index) {
    if (!float_round_trip_value_matches(observed[index], expected[index],
                                        absolute_tolerance)) {
      return false;
    }
  }
  return true;
}

bool exact_float_round_trip_array_matches(
    const std::array<double, 7>& observed,
    const std::array<double, 7>& expected) noexcept {
  for (std::size_t index = 0U; index < observed.size(); ++index) {
    const float expected_float = static_cast<float>(expected[index]);
    const float observed_float = static_cast<float>(observed[index]);
    if (observed[index] != static_cast<double>(observed_float) ||
        observed_float != expected_float) {
      return false;
    }
  }
  return true;
}

bool desired_history_matches(const RobotSample& sample,
                             const ExactDesiredHistory& expected) noexcept {
  // RobotState transports the desired fields through binary32.  A one-ULP
  // rule alone becomes pathologically strict near zero: the installed FCI
  // returned a harmless 1.39e-14 rad/s^2 operation-order residue that is tens
  // of thousands of near-zero-magnitude float ULPs.  These absolute supplements
  // are still far below the command/tracking envelopes.  If both the accepted
  // and dropped candidates pass, the distinct-candidate ambiguity guard below
  // remains authoritative.
  return float_round_trip_array_matches(
             sample.desired_q_rad, expected.q,
             kDesiredHistoryQNumericalEquivalenceRad) &&
         float_round_trip_array_matches(
             sample.desired_dq_rad_s, expected.dq,
             kDesiredHistoryDqNumericalEquivalenceRadS) &&
         float_round_trip_array_matches(sample.desired_ddq_rad_s2,
                                        expected.ddq,
                                        kDesiredHistoryDdqNumericalEquivalenceRadS2);
}

bool desired_history_exactly_matches(
    const RobotSample& sample,
    const ExactDesiredHistory& expected) noexcept {
  return exact_float_round_trip_array_matches(sample.desired_q_rad,
                                               expected.q) &&
         exact_float_round_trip_array_matches(sample.desired_dq_rad_s,
                                               expected.dq) &&
         exact_float_round_trip_array_matches(sample.desired_ddq_rad_s2,
                                               expected.ddq);
}

bool exact_histories_equal(const ExactDesiredHistory& left,
                           const ExactDesiredHistory& right) noexcept {
  return left.q == right.q && left.dq == right.dq &&
         left.ddq == right.ddq;
}

bool exact_histories_numerically_equivalent(
    const ExactDesiredHistory& left,
    const ExactDesiredHistory& right) noexcept {
  // If accepted C and dropped E(H) describe the same constant-acceleration
  // continuation, different operation ordering can leave only double-roundoff
  // residue.  These bounds contribute at most 1e-6 rad/s^2 and 1e-3 rad/s^3
  // when differentiated over the 1 ms command period.
  return max_abs_difference(left.q, right.q) <=
             kDesiredHistoryQNumericalEquivalenceRad &&
         max_abs_difference(left.dq, right.dq) <=
             kDesiredHistoryDqNumericalEquivalenceRadS &&
         max_abs_difference(left.ddq, right.ddq) <=
             kDesiredHistoryDdqNumericalEquivalenceRadS2;
}

std::string desired_history_mismatch_detail(
    const RobotSample& sample,
    const ExactDesiredHistory& accepted,
    const ExactDesiredHistory& dropped) {
  auto first_mismatch =
      [](const char* field, const std::array<double, 7>& observed,
         const std::array<double, 7>& accepted_values,
         const std::array<double, 7>& dropped_values,
         const double absolute_tolerance) -> std::string {
    for (std::size_t index = 0U; index < observed.size(); ++index) {
      if (float_round_trip_value_matches(observed[index],
                                         accepted_values[index],
                                         absolute_tolerance) ||
          float_round_trip_value_matches(observed[index],
                                         dropped_values[index],
                                         absolute_tolerance)) {
        continue;
      }
      return std::string(" field=") + field +
             " axis=" + std::to_string(index) +
             " observed_bits=" +
             std::to_string(float_bits(static_cast<float>(observed[index]))) +
             " accepted_bits=" +
             std::to_string(
                 float_bits(static_cast<float>(accepted_values[index]))) +
             " dropped_bits=" +
             std::to_string(
                 float_bits(static_cast<float>(dropped_values[index])));
    }
    return {};
  };
  std::string detail =
      first_mismatch("q_d", sample.desired_q_rad, accepted.q, dropped.q,
                     kDesiredHistoryQNumericalEquivalenceRad);
  if (detail.empty()) {
    detail = first_mismatch("dq_d", sample.desired_dq_rad_s, accepted.dq,
                            dropped.dq,
                            kDesiredHistoryDqNumericalEquivalenceRadS);
  }
  if (detail.empty()) {
    detail = first_mismatch("ddq_d", sample.desired_ddq_rad_s2, accepted.ddq,
                            dropped.ddq,
                            kDesiredHistoryDdqNumericalEquivalenceRadS2);
  }
  if (detail.empty()) {
    detail = " tuple_components_match_different_candidates";
  }
  return "FCI desired history matched neither accepted nor dropped" + detail;
}

ExactDesiredHistory extrapolate_constant_acceleration(
    ExactDesiredHistory history,
    const std::uint32_t packet_count) {
  for (std::uint32_t missed = 0U; missed < packet_count; ++missed) {
    for (std::size_t index = 0U; index < history.q.size(); ++index) {
      history.dq[index] += history.ddq[index] * kCommandPacketPeriodS;
      history.q[index] += history.dq[index] * kCommandPacketPeriodS;
    }
  }
  return history;
}

ExactDesiredHistory select_matching_desired_history(
    const RobotSample& sample,
    const ExactDesiredHistory& accepted,
    const ExactDesiredHistory& dropped) {
  const bool accepted_matches = desired_history_matches(sample, accepted);
  const bool dropped_matches = desired_history_matches(sample, dropped);
  if (!accepted_matches && !dropped_matches) {
    throw ServoFault(
        FaultCode::kDynamicState, false,
        desired_history_mismatch_detail(sample, accepted, dropped));
  }
  if (accepted_matches && dropped_matches &&
      !exact_histories_equal(accepted, dropped) &&
      !exact_histories_numerically_equivalent(accepted, dropped)) {
    throw ServoFault(
        FaultCode::kDynamicState, false,
        "FCI desired history ambiguously matches distinct accepted/dropped "
        "candidates");
  }
  if (accepted_matches != dropped_matches) {
    return accepted_matches ? accepted : dropped;
  }

  // Both candidates are now proven equal or numerically equivalent.  An
  // exact binary32 match may break only this harmless tie: it must never
  // override the ambiguity guard above merely because the other distinct
  // candidate is one permitted telemetry ULP away.
  const bool accepted_exact =
      desired_history_exactly_matches(sample, accepted);
  const bool dropped_exact = desired_history_exactly_matches(sample, dropped);
  if (accepted_exact != dropped_exact) {
    return accepted_exact ? accepted : dropped;
  }
  return accepted;
}

ExactDesiredHistory select_returned_desired_history(
    const RobotSample& sample,
    const ExactDesiredHistory& last_confirmed,
    const ExactDesiredHistory& last_sent) {
  // For a returned period p, FCI either accepted the command sent after the
  // preceding state and then extrapolated it for p-1 missing packets, or it
  // dropped that command and extrapolated the preceding confirmed state for p
  // packets.  RobotState transports only binary32 desired fields, so those
  // fields select one of the two exact double histories but never become the
  // numerical integration anchor.
  const ExactDesiredHistory accepted = extrapolate_constant_acceleration(
      last_sent, sample.control_period_ms - 1U);
  // At the maximum returned period, p-1 already consumes the complete
  // 20-packet FCI loss horizon.  A dropped last_sent candidate would require
  // p=21 missing packets and cannot be a recoverable returned state.  Require
  // the recovered packet to prove the accepted history instead of silently
  // extending uncontrolled continuation beyond the frozen 20-packet bound.
  if (sample.control_period_ms >
      HardSafetyLimits::kMaximumUncontrolledContinuationPackets) {
    if (!desired_history_matches(sample, accepted)) {
      throw ServoFault(
          FaultCode::kDynamicState, false,
          "maximum-period FCI recovery did not match accepted desired history");
    }
    return accepted;
  }
  const ExactDesiredHistory dropped = extrapolate_constant_acceleration(
      last_confirmed, sample.control_period_ms);
  return select_matching_desired_history(sample, accepted, dropped);
}

void require_exact_history_envelope(
    const ExactDesiredHistory& history,
    const std::array<double, 7>& start_q,
    const ControllerMode controller_mode = ControllerMode::kLegacy) {
  constexpr double kDerivativeNumericalTolerance = 1.0e-6;
  if (!finite_array(history.q) || !finite_array(history.dq) ||
      !finite_array(history.ddq) ||
      max_abs(history.dq) >
          HardSafetyLimits::kMaximumCommandVelocityRadS +
              kDerivativeNumericalTolerance ||
      max_abs(history.ddq) >
          HardSafetyLimits::kMaximumCommandAccelerationRadS2 +
              kDerivativeNumericalTolerance) {
    throw ServoFault(FaultCode::kDynamicState, false,
                     "exact FCI history left derivative envelope");
  }
  for (std::size_t index = 0U; index < history.q.size(); ++index) {
    if (history.q[index] < HardSafetyLimits::kSafeJointLower[index] ||
        history.q[index] > HardSafetyLimits::kSafeJointUpper[index]) {
      throw ServoFault(FaultCode::kDynamicState, false,
                       "exact FCI history left safe joint interval");
    }
  }
  if (!episode_delta_allowed(controller_mode, history.q, start_q)) {
    throw ServoFault(FaultCode::kDynamicState, false,
                     "legacy exact FCI history left home-centered episode envelope");
  }
}

void require_recoverable_fci_history(
    const ExactDesiredHistory& history,
    const std::array<double, 7>& start_q,
    const ControllerMode controller_mode = ControllerMode::kLegacy) {
  // This validates an exact history reconstructed from RobotState or from
  // libfranka::limitRate.  Its derivatives can temporarily be outside the
  // lower V225 tuning envelope after an accepted/dropped packet transition;
  // libfranka's official limiter is the authority that returns them safely.
  // Keep all non-derivative hard envelopes fail-closed.
  if (!finite_array(history.q) || !finite_array(history.dq) ||
      !finite_array(history.ddq)) {
    throw ServoFault(FaultCode::kDynamicState, false,
                     "recoverable FCI desired history is non-finite");
  }
  for (std::size_t index = 0U; index < history.q.size(); ++index) {
    if (history.q[index] < HardSafetyLimits::kSafeJointLower[index] ||
        history.q[index] > HardSafetyLimits::kSafeJointUpper[index]) {
      throw ServoFault(
          FaultCode::kDynamicState, false,
          "recoverable FCI desired history left safe joint interval: axis=" +
              std::to_string(index + 1U) + " q=" +
              std::to_string(history.q[index]));
    }
  }
  const double episode_error = max_abs_difference(history.q, start_q);
  if (!episode_delta_allowed(controller_mode, history.q, start_q)) {
    throw ServoFault(
        FaultCode::kDynamicState, false,
        "legacy recoverable FCI history left home-centered episode envelope: "
        "Linf=" +
            std::to_string(episode_error));
  }
}

struct ShapedCommand final {
  std::array<double, 7> q{};
  std::array<double, 7> dq{};
  std::array<double, 7> ddq{};
};

// A first-order velocity control barrier is intentionally used in the 1 kHz
// path instead of repeatedly solving a stopping trajectory.  For outward
// velocity v and emitted acceleration a, require
//
//   a <= K (V - v),  and symmetrically a >= K (-V - v).
//
// Because the emitted velocity is v1=v0+a*dt, the acceleration interval below
// is the exact algebraic solution in terms of the returned v0.  K*H*dt<=1
// proves that holding any emitted acceleration for all H=20 missing packets
// up to the FCI fail-stop boundary cannot cross |v|=V.  A returned recovery
// period is R=H+1<=21 ms because it also includes the current packet.  The
// second compile-time inequality proves that, after any such recovery, one
// maximum-jerk command can re-enter the same barrier.  Thus every
// accepted command independently survives a fresh fail-stop loss window,
// without inventing another user-space braking trajectory after FCI has
// already stopped.
constexpr double kVelocityBarrierGainPerS = 1.0;
constexpr double kVelocityBarrierDenominator =
    1.0 + kVelocityBarrierGainPerS * kCommandPacketPeriodS;
static_assert(
    kVelocityBarrierGainPerS *
            static_cast<double>(
                HardSafetyLimits::kMaximumUncontrolledContinuationPackets) *
            kCommandPacketPeriodS <=
        1.0);
static_assert(
    kVelocityBarrierGainPerS * kVelocityBarrierGainPerS *
            static_cast<double>(
                HardSafetyLimits::kMaximumUncontrolledContinuationPackets +
                1U) *
            HardSafetyLimits::kMaximumCommandVelocityRadS <=
        HardSafetyLimits::kMaximumCommandJerkRadS3 *
            kVelocityBarrierDenominator);

// The frozen independent discrete oracle below fixes the barrier-domain
// maximum at 0.04890004 rad (v=+/-0.50, a=0, including every 0..20 packet loss
// tail followed by maximum-jerk/-A braking), and a deterministic domain sweep
// rejects a larger value. Keep policy targets 0.060 rad
// inside the raw command boundary.  The emitted trajectory separately stays
// 0.010 rad inside that boundary for measured tracking error, leaving
// 0.050 rad between a saturated target and the emitted-command surface.
constexpr double kPositionTargetReserveRad = 0.060;
constexpr double kPositionTrackingReserveRad =
    HardSafetyLimits::kMaximumTrackingErrorRad;
// This first-order position/velocity barrier starts braking before the
// target edge.  It is intersected at every possible 1..21 packet
// semi-implicit continuation below, so an early recovery is still inside the
// same O(1) feasible surface and a full 20-loss tail remains inside the
// tracking-contracted command boundary.
// Gains above 1/s can make a jerk-limited acceleration interval empty during
// a long nominal approach (the deterministic K=2 counterexample reaches
// q=0.8963515, v=0.1991618, a=0.0008382 before the 21-packet surface asks for
// more than 10 rad/s^3 of braking jerk).  K=1/s is covered by the long
// home-to-edge and loss-injection regressions below.
constexpr double kPositionBarrierGainPerS = 1.0;
constexpr double kPositionAccelerationBarrierGainPerS = 1.0;

double maximum_jerk_limited_outward_excursion(
    const double oriented_velocity,
    const double oriented_acceleration) {
  const double velocity = oriented_velocity;
  const double acceleration = std::clamp(
      oriented_acceleration,
      -HardSafetyLimits::kMaximumCommandAccelerationRadS2,
      HardSafetyLimits::kMaximumCommandAccelerationRadS2);
  const double acceleration_step =
      HardSafetyLimits::kMaximumCommandJerkRadS3 *
      kCommandPacketPeriodS;
  const auto ramp_velocity =
      [&](const std::uint32_t packets) {
        const double count = static_cast<double>(packets);
        return velocity +
               kCommandPacketPeriodS *
                   (count * acceleration -
                    acceleration_step * count * (count + 1.0) * 0.5);
      };
  const auto ramp_position =
      [&](const double packets) {
        return kCommandPacketPeriodS *
               (packets * velocity +
                kCommandPacketPeriodS *
                    (acceleration * packets * (packets + 1.0) * 0.5 -
                     acceleration_step * packets * (packets + 1.0) *
                         (packets + 2.0) / 6.0));
      };
  // Number of full maximum-braking-jerk steps whose acceleration remains
  // above -A. The next and all later steps use exactly -A.
  const std::uint32_t ramp_packets = static_cast<std::uint32_t>(std::max(
      0.0,
      std::floor(
          (acceleration +
           HardSafetyLimits::kMaximumCommandAccelerationRadS2) /
          acceleration_step)));
  double maximum_excursion = 0.0;
  auto consider_ramp_packet =
      [&](const long long packet) {
        if (packet < 0 ||
            packet > static_cast<long long>(ramp_packets)) {
          return;
        }
        maximum_excursion = std::max(
            maximum_excursion,
            ramp_position(static_cast<double>(packet)));
      };
  consider_ramp_packet(0);
  consider_ramp_packet(ramp_packets);
  // D(n)-D(n-1)=dt*v(n), so an integer maximum can occur only beside a
  // velocity root or at an endpoint.  This covers v<0,a>0 trajectories that
  // first move away, reverse toward the boundary, and later brake again.
  const double quadratic_a =
      -0.5 * kCommandPacketPeriodS * acceleration_step;
  const double quadratic_b =
      kCommandPacketPeriodS *
      (acceleration - 0.5 * acceleration_step);
  const double discriminant =
      quadratic_b * quadratic_b -
      4.0 * quadratic_a * velocity;
  if (discriminant >= 0.0) {
    const double root_scale = 2.0 * quadratic_a;
    const double root_delta = std::sqrt(discriminant);
    for (const double root :
         {(-quadratic_b - root_delta) / root_scale,
          (-quadratic_b + root_delta) / root_scale}) {
      if (!finite(root)) {
        continue;
      }
      const long long center =
          static_cast<long long>(std::floor(root));
      for (long long offset = -2; offset <= 2; ++offset) {
        consider_ramp_packet(center + offset);
      }
    }
  }
  const double after_ramp_velocity = ramp_velocity(ramp_packets);
  const double after_ramp_position =
      ramp_position(static_cast<double>(ramp_packets));
  if (after_ramp_velocity > 0.0) {
    const double decrement =
        HardSafetyLimits::kMaximumCommandAccelerationRadS2 *
        kCommandPacketPeriodS;
    const double optimum =
        after_ramp_velocity / decrement - 0.5;
    const long long center =
        static_cast<long long>(std::floor(optimum));
    for (long long offset = -2; offset <= 2; ++offset) {
      const long long packet = std::max(0LL, center + offset);
      const double count = static_cast<double>(packet);
      const double continuation =
          kCommandPacketPeriodS *
          (count * after_ramp_velocity -
           decrement * count * (count + 1.0) * 0.5);
      maximum_excursion =
          std::max(maximum_excursion,
                   after_ramp_position + continuation);
    }
  }
  return std::max(0.0, maximum_excursion);
}

// Archived V94 ablation. It is retained only for the old offline regression
// entry points below and is never constructed by run_servo_process().
class ArchivedJerkLimitedShaper final {
 public:
  explicit ArchivedJerkLimitedShaper(const std::array<double, 7>& start_q)
      : start_q_(start_q) {}

  ShapedCommand step(const std::array<double, 7>& target,
                     const ExactDesiredHistory& reference,
                     const std::uint32_t returned_control_period_ms) {
    ShapedCommand next{};
    for (std::size_t index = 0U; index < 7U; ++index) {
      const double episode_lower =
          start_q_[index] - HardSafetyLimits::kMaximumEpisodeDeltaRad;
      const double episode_upper =
          start_q_[index] + HardSafetyLimits::kMaximumEpisodeDeltaRad;
      const double raw_position_lower =
          std::max(HardSafetyLimits::kSafeJointLower[index], episode_lower);
      const double raw_position_upper =
          std::min(HardSafetyLimits::kSafeJointUpper[index], episode_upper);
      const double command_position_lower =
          raw_position_lower + kPositionTrackingReserveRad;
      const double command_position_upper =
          raw_position_upper - kPositionTrackingReserveRad;
      const double target_position_lower =
          raw_position_lower + kPositionTargetReserveRad;
      const double target_position_upper =
          raw_position_upper - kPositionTargetReserveRad;
      // Policy admission remains against the unchanged raw envelope.  Only
      // the actuator reference is contracted by the independently proven
      // loss-tail stopping plus measured-tracking reserve above.
      const double bounded_target = std::clamp(
          target[index], target_position_lower, target_position_upper);
      const double error = bounded_target - reference.q[index];
      const double stopping_velocity =
          std::sqrt(2.0 * HardSafetyLimits::kMaximumCommandAccelerationRadS2 *
                    std::abs(error));
      const double desired_velocity = std::copysign(
          std::min(HardSafetyLimits::kMaximumCommandVelocityRadS,
                   stopping_velocity),
          error);
      const double desired_acceleration =
          (desired_velocity - reference.dq[index]) /
          kCommandPacketPeriodS;
      double lower = std::max(
          -HardSafetyLimits::kMaximumCommandAccelerationRadS2,
          reference.ddq[index] -
              HardSafetyLimits::kMaximumCommandJerkRadS3 *
                  kCommandPacketPeriodS);
      double upper = std::min(
          HardSafetyLimits::kMaximumCommandAccelerationRadS2,
          reference.ddq[index] +
              HardSafetyLimits::kMaximumCommandJerkRadS3 *
                  kCommandPacketPeriodS);

      constexpr double kDtSquared =
          kCommandPacketPeriodS * kCommandPacketPeriodS;
      // Prove position and episode bounds for the command itself and every
      // constant-acceleration packet through the complete recovery horizon.
      // At step k, semi-implicit integration gives
      //   q(k)=q0+k*v0*dt+a*dt^2*k*(k+1)/2.
      // Intersecting every k (rather than only the endpoint) also covers an
      // interior turning point where velocity changes sign.
      for (std::uint32_t packet = 1U;
           packet <=
               HardSafetyLimits::kMaximumUncontrolledContinuationPackets + 1U;
           ++packet) {
        const double steps = static_cast<double>(packet);
        const double base_position =
            reference.q[index] +
            steps * reference.dq[index] * kCommandPacketPeriodS;
        const double acceleration_position_coefficient =
            0.5 * steps * (steps + 1.0) * kDtSquared;
        lower = std::max(
            lower,
            (command_position_lower - base_position) /
                acceleration_position_coefficient);
        upper = std::min(
            upper,
            (command_position_upper - base_position) /
                acceleration_position_coefficient);
        const double barrier_acceleration_coefficient =
            steps * kCommandPacketPeriodS +
            kPositionBarrierGainPerS *
                acceleration_position_coefficient;
        lower = std::max(
            lower,
            (-kPositionBarrierGainPerS *
                 (reference.q[index] - command_position_lower) -
             reference.dq[index] *
                 (1.0 + kPositionBarrierGainPerS * steps *
                            kCommandPacketPeriodS)) /
                barrier_acceleration_coefficient);
        upper = std::min(
            upper,
            (kPositionBarrierGainPerS *
                 (command_position_upper - reference.q[index]) -
             reference.dq[index] *
                 (1.0 + kPositionBarrierGainPerS * steps *
                            kCommandPacketPeriodS)) /
                barrier_acceleration_coefficient);

        // The first-order surface alone says how much outward velocity is
        // admissible but does not exclude an already-large outward
        // acceleration that cannot be reversed within the jerk limit.  The
        // second (acceleration) surface is the continuous-time HOCBF
        //
        //   a <= L1*L2*(upper-q) - (L1+L2)*v
        //
        // and its sign-symmetric lower form.  Substitution of the exact
        // semi-implicit q(k),v(k) keeps the bound affine in the one candidate
        // acceleration; no iterative solve is present in the servo loop.
        const double acceleration_barrier_position_gain =
            kPositionBarrierGainPerS *
            kPositionAccelerationBarrierGainPerS;
        const double acceleration_barrier_velocity_gain =
            kPositionBarrierGainPerS +
            kPositionAccelerationBarrierGainPerS;
        const double acceleration_barrier_coefficient =
            1.0 +
            acceleration_barrier_position_gain *
                acceleration_position_coefficient +
            acceleration_barrier_velocity_gain * steps *
                kCommandPacketPeriodS;
        lower = std::max(
            lower,
            (-acceleration_barrier_position_gain *
                 (reference.q[index] - command_position_lower +
                  steps * reference.dq[index] *
                      kCommandPacketPeriodS) -
             acceleration_barrier_velocity_gain *
                 reference.dq[index]) /
                acceleration_barrier_coefficient);
        upper = std::min(
            upper,
            (acceleration_barrier_position_gain *
                 (command_position_upper - reference.q[index] -
                  steps * reference.dq[index] *
                      kCommandPacketPeriodS) -
             acceleration_barrier_velocity_gain *
                 reference.dq[index]) /
                acceleration_barrier_coefficient);
      }
      lower = std::max(
          lower,
          kVelocityBarrierGainPerS *
              (-HardSafetyLimits::kMaximumCommandVelocityRadS -
               reference.dq[index]) /
              kVelocityBarrierDenominator);
      upper = std::min(
          upper,
          kVelocityBarrierGainPerS *
              (HardSafetyLimits::kMaximumCommandVelocityRadS -
               reference.dq[index]) /
              kVelocityBarrierDenominator);
      if (lower > upper + 1.0e-12) {
        throw ServoFault(
            FaultCode::kDynamicState, false,
            "no safe jerk-limited command exists: axis=" +
                std::to_string(index + 1U) +
                " period_ms=" +
                std::to_string(returned_control_period_ms) +
                " target=" + std::to_string(target[index]) +
                " q=" + std::to_string(reference.q[index]) +
                " dq=" + std::to_string(reference.dq[index]) +
                " ddq=" + std::to_string(reference.ddq[index]) +
                " lower=" + std::to_string(lower) +
                " upper=" + std::to_string(upper));
      }
      if (lower > upper) {
        // The independently computed bounds can cross by a few ulps.  Keep
        // std::clamp's lo<=hi precondition explicit while treating only this
        // already-tolerated numerical overlap as a singleton interval.
        const double collapsed = upper + 0.5 * (lower - upper);
        lower = collapsed;
        upper = collapsed;
      }
      const double next_acceleration =
          std::clamp(desired_acceleration, lower, upper);
      const double next_velocity =
          reference.dq[index] +
          next_acceleration * kCommandPacketPeriodS;
      // FCI differentiates joint-position commands with backward Euler at a
      // fixed 1 ms sample time.  This semi-implicit update therefore makes
      // the derivatives seen by FCI exactly next_velocity/next_acceleration.
      next.q[index] =
          reference.q[index] + next_velocity * kCommandPacketPeriodS;
      // Store the derivatives reconstructed from the exact double position
      // that is actually sent, matching FCI's backward-Euler arithmetic.
      next.dq[index] =
          (next.q[index] - reference.q[index]) / kCommandPacketPeriodS;
      next.ddq[index] =
          (next.dq[index] - reference.dq[index]) / kCommandPacketPeriodS;
      const double next_jerk =
          (next.ddq[index] - reference.ddq[index]) /
          kCommandPacketPeriodS;
      constexpr double kDerivativeNumericalTolerance = 1.0e-5;
      if (std::abs(next.dq[index]) >
              HardSafetyLimits::kMaximumCommandVelocityRadS +
                  kDerivativeNumericalTolerance ||
          std::abs(next.ddq[index]) >
              HardSafetyLimits::kMaximumCommandAccelerationRadS2 +
                  kDerivativeNumericalTolerance ||
          std::abs(next_jerk) >
              HardSafetyLimits::kMaximumCommandJerkRadS3 +
                  kDerivativeNumericalTolerance) {
        throw ServoFault(FaultCode::kInternal, false,
                         "exact shaped command violates derivative envelope");
      }
    }
    return next;
  }

 private:
  std::array<double, 7> start_q_{};
};

struct RuntimeState final {
  std::uint64_t critical_out_sequence{1U};
  std::uint64_t telemetry_out_sequence{1U};
  std::uint64_t critical_in_sequence{1U};
  std::uint64_t heartbeat_sequence{0U};
  std::uint64_t last_heartbeat_ns{0U};
  std::uint64_t action_ready_ns{0U};
  std::uint64_t last_target_received_ns{0U};
  std::uint64_t last_target_sequence{0U};
  std::uint64_t last_observation_sequence{0U};
  std::uint64_t target_ack_count{0U};
  std::uint64_t telemetry_drop_count{0U};
  std::uint64_t active_reads{0U};
  std::uint64_t active_writes{0U};
  std::uint64_t cycle{0U};
  std::uint64_t maximum_read_to_write_ns{0U};
  std::uint32_t maximum_control_period_ms{0U};
  std::uint32_t missed_robot_states{0U};
  std::uint32_t consecutive_healthy_cycles{0U};
  double minimum_healthy_success_rate{1.0};
  bool action_ready{false};
  bool target_pending_ack{false};
  bool fault_reply_delivered{false};
  std::array<double, 7> start_q{};
  std::array<double, 7> prior_command{};
  std::array<double, 7> last_command{};
  std::array<double, 7> last_command_velocity{};
  std::array<double, 7> last_command_acceleration{};
  std::array<double, 7> held_target{};
  std::array<double, 7> shaper_q_d_rad{};
  std::array<double, 7> shaper_dq_d_rad_s{};
  std::array<double, 7> shaper_ddq_d_rad_s2{};
  std::array<double, 7> held_q_cmd_rad{};
  std::array<float, 29> controller_state29{};
  TargetPayload pending_target{};
  RobotSample latest_sample{};
};

void copy_detail(char* output,
                 const std::size_t capacity,
                 const std::string& detail) noexcept {
  std::memset(output, 0, capacity);
  const std::size_t bytes =
      capacity == 0U ? 0U : std::min(capacity - 1U, detail.size());
  if (bytes != 0U) {
    std::memcpy(output, detail.data(), bytes);
  }
}

template <typename Payload>
void send_critical_or_fault(InheritedChannels* channels,
                            RuntimeState* state,
                            ServoClock* clock,
                            const std::array<std::uint8_t, kSessionNonceBytes>& nonce,
                            const MessageKind kind,
                            const Payload& payload) {
  EncodedPacket packet{};
  const CodecError encoded =
      encode_payload(kind, state->critical_out_sequence, clock->monotonic_ns(),
                     nonce, payload, &packet);
  if (encoded != CodecError::kNone) {
    throw ServoFault(FaultCode::kInternal, false,
                     std::string("critical encode failed: ") +
                         codec_error_name(encoded));
  }
  int system_errno = 0;
  const IoStatus sent = channels->send_critical(packet, &system_errno);
  if (sent != IoStatus::kOk) {
    throw ServoFault(FaultCode::kCriticalReplyBlocked, false,
                     "critical seqpacket send failed: errno=" +
                         std::to_string(system_errno) + " fd=" +
                         std::to_string(channels->critical_fd()),
                     system_errno);
  }
  ++state->critical_out_sequence;
}

bool send_fault_best_effort(
    InheritedChannels* channels,
    RuntimeState* state,
    ServoClock* clock,
    const std::array<std::uint8_t, kSessionNonceBytes>& nonce,
    const FaultCode fault_code,
    const int system_errno,
    const std::string& detail) noexcept {
  if (channels == nullptr || state == nullptr || clock == nullptr) {
    return false;
  }
  FaultPayload payload{};
  payload.fault_code = static_cast<std::uint32_t>(fault_code);
  payload.detail_bytes = static_cast<std::uint32_t>(
      std::min(detail.size(),
               static_cast<std::size_t>(kFaultDetailBytes - 1U)));
  payload.control_cycle = state->cycle;
  payload.target_sequence = state->last_target_sequence;
  payload.robot_time_ms = state->latest_sample.robot_time_ms;
  payload.system_errno = system_errno;
  copy_detail(payload.detail, sizeof(payload.detail), detail);
  EncodedPacket packet{};
  if (encode_payload(MessageKind::kFault, state->critical_out_sequence,
                     clock->monotonic_ns(), nonce, payload,
                     &packet) != CodecError::kNone) {
    return false;
  }
  int send_errno = 0;
  if (channels->send_critical(packet, &send_errno) != IoStatus::kOk) {
    return false;
  }
  ++state->critical_out_sequence;
  return true;
}

ArmPayload wait_for_arm(const ServoProcessConfig& config,
                        InheritedChannels* channels,
                        ServoClock* clock,
                        RuntimeState* state) {
  const std::uint64_t start = clock->monotonic_ns();
  while (clock->monotonic_ns() - start < config.handshake_timeout_ns) {
    struct pollfd descriptor {};
    descriptor.fd = channels->critical_fd();
    descriptor.events = POLLIN | POLLERR | POLLHUP;
    const int polled = ::poll(&descriptor, 1U, 10);
    if (polled < 0 && errno != EINTR) {
      throw ServoFault(FaultCode::kProtocol, false,
                       "ARM poll failed", errno);
    }
    if (polled <= 0) {
      continue;
    }
    const ReceiveResult received = channels->receive_critical();
    if (received.status == IoStatus::kPeerClosed) {
      throw ServoFault(FaultCode::kPeerDisconnected, false,
                       "critical peer disconnected before ARM");
    }
    if (received.status != IoStatus::kOk) {
      throw ServoFault(FaultCode::kProtocol, false,
                       "invalid critical receive before ARM",
                       received.system_errno);
    }
    DecodedPacket decoded{};
    const CodecError codec = decode_packet(
        received.bytes.data(), received.size, state->critical_in_sequence,
        config.session_nonce, &decoded);
    if (codec != CodecError::kNone ||
        decoded.header.kind != static_cast<std::uint16_t>(MessageKind::kArm)) {
      throw ServoFault(FaultCode::kProtocol, false,
                       std::string("expected one ARM packet: ") +
                           codec_error_name(codec));
    }
    ++state->critical_in_sequence;
    ArmPayload arm{};
    if (copy_payload(decoded, &arm) != CodecError::kNone) {
      throw ServoFault(FaultCode::kProtocol, false,
                       "ARM payload decode failed");
    }
    require_arm_contract(arm, clock->monotonic_ns());
    state->last_heartbeat_ns = clock->monotonic_ns();
    return arm;
  }
  throw ServoFault(FaultCode::kAuthorization, false, "ARM handshake timed out");
}

struct DrainResult final {
  bool stop_requested{false};
  StopReason stop_reason{StopReason::kRequested};
};

DrainResult drain_critical(
    const ArmPayload& arm,
    const ServoProcessConfig& config,
    InheritedChannels* channels,
    ServoClock* clock,
    RuntimeState* state) {
  DrainResult result{};
  const std::uint64_t drain_start = clock->monotonic_ns();
  for (std::uint32_t count = 0U;
       count < HardSafetyLimits::kMaximumCriticalPacketsPerCycle; ++count) {
    if (clock->monotonic_ns() - drain_start >
        HardSafetyLimits::kMaximumIpcDrainNs) {
      throw ServoFault(FaultCode::kReadToWriteDeadline, false,
                       "critical IPC drain exceeded 100 us");
    }
    const ReceiveResult received = channels->receive_critical();
    if (received.status == IoStatus::kWouldBlock) {
      break;
    }
    if (received.status == IoStatus::kPeerClosed) {
      throw ServoFault(FaultCode::kPeerDisconnected, false,
                       "critical peer disconnected");
    }
    if (received.status != IoStatus::kOk) {
      throw ServoFault(FaultCode::kProtocol, false,
                       "critical recvmsg failed or truncated",
                       received.system_errno);
    }
    DecodedPacket decoded{};
    const CodecError codec = decode_packet(
        received.bytes.data(), received.size, state->critical_in_sequence,
        config.session_nonce, &decoded);
    if (codec != CodecError::kNone) {
      throw ServoFault(FaultCode::kProtocol, false,
                       std::string("critical packet rejected: ") +
                           codec_error_name(codec));
    }
    ++state->critical_in_sequence;
    const std::uint64_t now = clock->monotonic_ns();
    const MessageKind kind = static_cast<MessageKind>(decoded.header.kind);
    const bool header_is_future =
        decoded.header.monotonic_ns >
        now + HardSafetyLimits::kMaximumFutureTargetSkewNs;
    const bool header_is_stale =
        now - std::min(now, decoded.header.monotonic_ns) >
        HardSafetyLimits::kMaximumTargetAgeNs;

    // A Python supervisor can be descheduled after it has encoded a packet
    // but before the kernel write. Sender-side header age is therefore not a
    // sound heartbeat-liveness clock. HEARTBEAT liveness is enforced from the
    // child's own receipt times in the active loop; its exact sequence, nonce
    // and CRC still prevent replay. A future sender clock remains a protocol
    // fault. STOP is deliberately exempt from timestamp freshness: once its
    // exact authenticated seqpacket arrives, stopping is always safest.
    // TARGET retains this header check and its independent produced-time check.
    if ((kind == MessageKind::kHeartbeat && header_is_future) ||
        (kind == MessageKind::kTarget &&
         (header_is_future || header_is_stale))) {
      const std::uint64_t delta =
          decoded.header.monotonic_ns > now
              ? decoded.header.monotonic_ns - now
              : now - decoded.header.monotonic_ns;
      throw ServoFault(
          FaultCode::kProtocol, false,
          std::string("critical packet timestamp is stale or future: kind=") +
              std::to_string(static_cast<std::uint16_t>(kind)) +
              " relation=" +
              (decoded.header.monotonic_ns > now ? "future" : "stale") +
              " delta_ns=" + std::to_string(delta));
    }
    if (kind == MessageKind::kHeartbeat) {
      HeartbeatPayload heartbeat{};
      if (copy_payload(decoded, &heartbeat) != CodecError::kNone ||
          heartbeat.heartbeat_sequence != state->heartbeat_sequence + 1U) {
        throw ServoFault(FaultCode::kProtocol, false,
                         "heartbeat sequence is not exact");
      }
      state->heartbeat_sequence = heartbeat.heartbeat_sequence;
      state->last_heartbeat_ns = now;
      continue;
    }
    if (kind == MessageKind::kStop) {
      StopPayload stop{};
      if (copy_payload(decoded, &stop) != CodecError::kNone ||
          stop.reserved != 0U ||
          stop.reason_code !=
              static_cast<std::uint32_t>(StopReason::kRequested)) {
        throw ServoFault(FaultCode::kProtocol, false,
                         "STOP payload is invalid");
      }
      result.stop_requested = true;
      result.stop_reason = StopReason::kRequested;
      continue;
    }
    if (kind != MessageKind::kTarget) {
      throw ServoFault(FaultCode::kProtocol, false,
                       "message kind is invalid in active servo");
    }
    if (!state->action_ready) {
      throw ServoFault(FaultCode::kProtocol, false,
                       "TARGET arrived before ACTION_READY");
    }
    if (state->last_target_sequence >= arm.maximum_target_count) {
      throw ServoFault(FaultCode::kTargetValidation, false,
                       "TARGET count exceeds armed maximum_target_count");
    }
    if (state->target_pending_ack) {
      throw ServoFault(FaultCode::kProtocol, false,
                       "new TARGET arrived before prior write ACK");
    }
    TargetPayload target{};
    if (copy_payload(decoded, &target) != CodecError::kNone ||
        target.target_sequence != state->last_target_sequence + 1U ||
        target.observation_sequence != state->last_observation_sequence + 1U) {
      throw ServoFault(FaultCode::kTargetValidation, false,
                       "TARGET/observation sequence is not exact");
    }
    if (target.produced_monotonic_ns >
            now + HardSafetyLimits::kMaximumFutureTargetSkewNs ||
        now - std::min(now, target.produced_monotonic_ns) >
            HardSafetyLimits::kMaximumTargetAgeNs) {
      throw ServoFault(FaultCode::kTargetValidation, false,
                       "TARGET timestamp is stale or future");
    }
    const auto target_q = copy_wire_doubles(target.target_q_rad);
    if (!finite_array(target_q)) {
      throw ServoFault(FaultCode::kTargetValidation, false,
                       "TARGET q is non-finite");
    }
    const auto& previous_target = state->last_target_sequence == 0U
                                      ? state->start_q
                                      : state->held_target;
    if (!previous_target_delta_allowed(
            static_cast<ControllerMode>(arm.controller_mode), target_q,
            previous_target)) {
      throw ServoFault(FaultCode::kTargetValidation, false,
                       "TARGET delta exceeds legacy V94 tick envelope");
    }
    if (!episode_delta_allowed(
            static_cast<ControllerMode>(arm.controller_mode), target_q,
            state->start_q)) {
      throw ServoFault(FaultCode::kTargetValidation, false,
                       "legacy TARGET exceeds home-centered episode envelope");
    }
    for (std::size_t index = 0U; index < 7U; ++index) {
      if (target_q[index] < HardSafetyLimits::kSafeJointLower[index] ||
          target_q[index] > HardSafetyLimits::kSafeJointUpper[index]) {
        throw ServoFault(FaultCode::kTargetValidation, false,
                         "TARGET q is outside safe joint interval");
      }
    }
    state->pending_target = target;
    state->held_target = target_q;
    state->last_target_sequence = target.target_sequence;
    state->last_observation_sequence = target.observation_sequence;
    state->last_target_received_ns = now;
    state->target_pending_ack = true;
  }
  if (clock->monotonic_ns() >= arm.authorization_expires_monotonic_ns) {
    throw ServoFault(FaultCode::kAuthorization, false,
                     "authorization expired in active loop");
  }
  return result;
}

void publish_state_best_effort(
    const ServoProcessConfig& config,
    InheritedChannels* channels,
    ServoClock* clock,
    RuntimeState* state,
    const std::uint64_t read_to_write_ns) noexcept {
  StatePayload payload{};
  payload.control_cycle = state->cycle;
  payload.robot_time_ms = state->latest_sample.robot_time_ms;
  payload.captured_monotonic_ns = clock->monotonic_ns();
  payload.captured_realtime_ns = clock->realtime_ns();
  payload.active_target_sequence = state->last_target_sequence;
  payload.active_observation_sequence = state->last_observation_sequence;
  payload.control_period_ms = state->latest_sample.control_period_ms;
  payload.robot_mode = static_cast<std::uint32_t>(state->latest_sample.mode);
  payload.control_command_success_rate =
      state->latest_sample.control_command_success_rate;
  copy_to_wire(state->latest_sample.q_rad, payload.measured_q_rad);
  copy_to_wire(state->latest_sample.dq_rad_s, payload.measured_dq_rad_s);
  copy_to_wire(state->latest_sample.O_T_EE, payload.O_T_EE);
  copy_to_wire(state->last_command, payload.commanded_q_rad);
  payload.status_flags = state->latest_sample.status_flags;
  payload.cumulative_missed_robot_states = state->missed_robot_states;
  payload.last_read_to_write_ns = read_to_write_ns;
  payload.maximum_read_to_write_ns = state->maximum_read_to_write_ns;
  payload.telemetry_drop_count = state->telemetry_drop_count;
  payload.health_flags = kHealthStaticVerified |
                         (state_is_clear(state->latest_sample)
                              ? kHealthCurrentCycleHealthy
                              : 0U) |
                         (state->action_ready ? kHealthActionReady : 0U);
  payload.consecutive_healthy_cycles = state->consecutive_healthy_cycles;
  payload.active_read_count = state->active_reads;
  payload.active_write_count = state->active_writes;
  payload.target_ack_count = state->target_ack_count;
  std::copy(state->controller_state29.begin(),
            state->controller_state29.end(), payload.controller_state29);
  copy_to_wire(state->shaper_q_d_rad, payload.shaper_q_d_rad);
  copy_to_wire(state->shaper_dq_d_rad_s, payload.shaper_dq_d_rad_s);
  copy_to_wire(state->shaper_ddq_d_rad_s2,
               payload.shaper_ddq_d_rad_s2);
  copy_to_wire(state->held_q_cmd_rad, payload.held_q_cmd_rad);
  EncodedPacket packet{};
  if (encode_payload(MessageKind::kState, state->telemetry_out_sequence,
                     payload.captured_monotonic_ns, config.session_nonce,
                     payload, &packet) != CodecError::kNone) {
    ++state->telemetry_drop_count;
    return;
  }
  int send_errno = 0;
  if (channels->send_telemetry(packet, &send_errno) == IoStatus::kOk) {
    ++state->telemetry_out_sequence;
  } else {
    ++state->telemetry_drop_count;
  }
}

StopProofPayload cleanup_robot(
    std::unique_ptr<RobotBackend>* backend,
    std::unique_ptr<ActiveRobotSession>* active,
    RuntimeState* state,
    const StopReason reason,
    const FaultCode fault,
    const bool finish_allowed,
    const std::string& detail) noexcept {
  StopProofPayload proof{};
  proof.stop_reason = static_cast<std::uint32_t>(reason);
  proof.terminal_fault_code = static_cast<std::uint32_t>(fault);
  proof.control_cycles = state->cycle;
  proof.last_target_sequence = state->last_target_sequence;
  proof.active_read_count = state->active_reads;
  proof.active_write_count = state->active_writes;
  proof.target_ack_count = state->target_ack_count;
  proof.maximum_control_period_ms = state->maximum_control_period_ms;
  proof.maximum_read_to_write_ns = state->maximum_read_to_write_ns;
  proof.pre_stop_robot_time_ms = state->latest_sample.robot_time_ms;
  proof.telemetry_drop_count = state->telemetry_drop_count;
  copy_detail(proof.detail, sizeof(proof.detail), detail);

  // A streaming-policy STOP may arrive while the jerk-limited command is still
  // moving, so it is not a legal MotionFinished boundary.  All requested,
  // signal and fault paths set finish_allowed=false and take the explicit
  // Robot::stop() path below.  Keep the graceful finish branch for a future
  // caller that has independently proved the command generator is at rest.
  //
  // Release the active handle before Robot::stop().  The pinned libfranka
  // destructor cancels an unfinished active motion and releases its control
  // lock; the subsequent explicit stop is the auditable fallback represented
  // in STOP_PROOF.
  bool explicit_stop_required = *active != nullptr && !finish_allowed;
  if (*active != nullptr && finish_allowed) {
    proof.finish_attempted = 1U;
    try {
      (*active)->write_once(state->last_command, true);
      proof.finish_succeeded = 1U;
      ++state->active_writes;
    } catch (...) {
      proof.finish_succeeded = 0U;
      explicit_stop_required = true;
    }
  }
  active->reset();
  proof.active_handle_released = 1U;
  if (explicit_stop_required && *backend != nullptr) {
    proof.robot_stop_attempted = 1U;
    try {
      (*backend)->stop();
      proof.robot_stop_succeeded = 1U;
    } catch (...) {
      // The return status remains diagnostic.  Three fresh Idle/low-dq
      // samples below are the independent physical-stop proof.
      proof.robot_stop_succeeded = 0U;
    }
  }
  proof.active_read_count = state->active_reads;
  proof.active_write_count = state->active_writes;

  std::uint32_t consecutive = 0U;
  std::uint32_t final_stop_status_flags = 0U;
  std::string final_current_errors = "[]";
  std::string final_last_motion_errors = "[]";
  std::uint64_t prior_time = state->latest_sample.robot_time_ms;
  std::array<StopVerificationSample, 3> accepted{};
  const bool clean_explicit_stop_transition =
      explicit_stop_required && proof.robot_stop_succeeded != 0U &&
      fault == FaultCode::kNone &&
      (reason == StopReason::kRequested || reason == StopReason::kSignal);
  // A faulting active session can itself cause FCI to report the exact
  // communication_constraints_violation pair while Robot::stop() is
  // transitioning to Reflex or Idle.  That error is not evidence of residual
  // motion.  After a successful explicit stop, accept only this exact
  // communication-only status (never contacts/collisions/unknown flags), and
  // still require three fresh samples with dq <= the stop limit.
  const bool successful_explicit_stop =
      explicit_stop_required && proof.robot_stop_succeeded != 0U;
  if (*backend != nullptr) {
    for (std::uint32_t attempt = 0U;
         attempt < HardSafetyLimits::kStopMaximumSamples; ++attempt) {
      try {
        const RobotSample sample = (*backend)->read_once();
        ++proof.stop_verification_samples;
        const bool fresh = sample.robot_time_ms > prior_time;
        prior_time = std::max(prior_time, sample.robot_time_ms);
        proof.final_robot_mode = static_cast<std::uint32_t>(sample.mode);
        proof.final_robot_time_ms = sample.robot_time_ms;
        final_stop_status_flags = sample.status_flags;
        final_current_errors = sample.current_errors_text;
        final_last_motion_errors = sample.last_motion_errors_text;
        if (finite_array(sample.dq_rad_s)) {
          proof.maximum_stop_dq_rad_s =
              std::max(proof.maximum_stop_dq_rad_s,
                       max_abs(sample.dq_rad_s));
        }
        const bool stopped_mode =
            sample.mode == RobotModeCode::kIdle ||
            (clean_explicit_stop_transition &&
             (sample.mode == RobotModeCode::kOther ||
              sample.mode == RobotModeCode::kReflex)) ||
            (successful_explicit_stop &&
             sample.mode == RobotModeCode::kReflex);
        constexpr std::uint32_t kRequestedStopCommunicationStatus =
            kStateHasCurrentErrors | kStateHasLastMotionErrors |
            kStateHasOnlyCommunicationConstraintsViolation;
        const bool exact_communication_only_stop_status =
            successful_explicit_stop &&
            (sample.mode == RobotModeCode::kIdle ||
             sample.mode == RobotModeCode::kReflex) &&
            sample.status_flags == kRequestedStopCommunicationStatus;
        const bool stopped_status_clear =
            sample.status_flags == 0U ||
            exact_communication_only_stop_status;
        const bool valid = fresh && stopped_mode &&
                           stopped_status_clear &&
                           finite_array(sample.dq_rad_s) &&
                           max_abs(sample.dq_rad_s) <=
                               HardSafetyLimits::kStopMaximumVelocityRadS;
        if (!valid) {
          consecutive = 0U;
          continue;
        }
        StopVerificationSample summary{};
        summary.robot_time_ms = sample.robot_time_ms;
        summary.robot_mode = static_cast<std::uint32_t>(sample.mode);
        summary.status_flags = sample.status_flags;
        copy_to_wire(sample.dq_rad_s, summary.measured_dq_rad_s);
        if (consecutive < accepted.size()) {
          accepted[consecutive] = summary;
        }
        ++consecutive;
        if (consecutive >= HardSafetyLimits::kStopConsecutiveSamples) {
          break;
        }
      } catch (...) {
        consecutive = 0U;
      }
    }
  }
  proof.stop_consecutive_idle_samples = consecutive;
  if (consecutive >= HardSafetyLimits::kStopConsecutiveSamples) {
    proof.idle_dq_verified = 1U;
    for (std::size_t index = 0U; index < accepted.size(); ++index) {
      proof.verified_samples[index] = accepted[index];
    }
  } else {
    copy_detail(
        proof.detail, sizeof(proof.detail),
        detail + "; final_stop_status_flags=" +
            std::to_string(final_stop_status_flags) +
            "; current=" + final_current_errors +
            "; last=" + final_last_motion_errors);
  }
  backend->reset();
  proof.robot_backend_released = 1U;
  return proof;
}

}  // namespace

bool desired_history_ambiguity_regression_for_test() noexcept {
  try {
    ExactDesiredHistory accepted{};
    accepted.q[0] = 1.0;
    ExactDesiredHistory dropped = accepted;
    dropped.q[0] = static_cast<double>(std::nextafterf(
        1.0F, std::numeric_limits<float>::infinity()));

    RobotSample sample{};
    sample.desired_q_rad = accepted.q;
    sample.desired_dq_rad_s = accepted.dq;
    sample.desired_ddq_rad_s2 = accepted.ddq;
    (void)select_matching_desired_history(sample, accepted, dropped);
  } catch (const ServoFault& error) {
    return error.code == FaultCode::kDynamicState &&
           std::string(error.what()).find(
               "ambiguously matches distinct accepted/dropped") !=
               std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

bool desired_history_closed_form_regression_for_test() noexcept {
  auto promoted_float_from_bits = [](const std::uint32_t bits) {
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<double>(value);
  };
  auto selected_bits_equal =
      [](const ExactDesiredHistory& selected,
         const std::uint32_t q_bits,
         const std::uint32_t dq_bits,
         const std::uint32_t ddq_bits) {
        return float_bits(static_cast<float>(selected.q[0])) == q_bits &&
               float_bits(static_cast<float>(selected.dq[0])) == dq_bits &&
               float_bits(static_cast<float>(selected.ddq[0])) == ddq_bits;
      };

  // Independent binary32 oracles for the real first-motion history:
  // H.q bits=945916694, C.q=H.q+1e-8, C.dq=(C.q-H.q)/1e-3,
  // C.ddq=C.dq/1e-3.  The p=4, p=10, and maximum-returned p=21 values below
  // were derived from the closed-form semi-implicit sums, not from
  // extrapolate_constant_acceleration. p=21 proves the accepted-command-only
  // recovery case after exactly 20 missing packets.
  constexpr std::uint32_t kInitialQBits = 945916694U;
  ExactDesiredHistory confirmed{};
  confirmed.q[0] = promoted_float_from_bits(kInitialQBits);
  ExactDesiredHistory sent = confirmed;
  sent.q[0] += 1.0e-8;
  sent.dq[0] =
      (sent.q[0] - confirmed.q[0]) / kCommandPacketPeriodS;
  sent.ddq[0] =
      (sent.dq[0] - confirmed.dq[0]) / kCommandPacketPeriodS;

  struct Oracle final {
    std::uint32_t period_ms;
    std::uint32_t q_bits;
    std::uint32_t dq_bits;
    std::uint32_t ddq_bits;
  };
  constexpr std::array<Oracle, 3> kAcceptedOracles{{
      {4U, 945944182U, 942130604U, 1008981770U},
      {10U, 946067877U, 953267991U, 1008981770U},
      {21U, 946551662U, 962343794U, 1008981770U},
  }};
  try {
    for (const Oracle& oracle : kAcceptedOracles) {
      RobotSample sample{};
      sample.control_period_ms = oracle.period_ms;
      sample.desired_q_rad[0] =
          promoted_float_from_bits(oracle.q_bits);
      sample.desired_dq_rad_s[0] =
          promoted_float_from_bits(oracle.dq_bits);
      sample.desired_ddq_rad_s2[0] =
          promoted_float_from_bits(oracle.ddq_bits);
      const ExactDesiredHistory selected =
          select_returned_desired_history(sample, confirmed, sent);
      if (!selected_bits_equal(selected, oracle.q_bits, oracle.dq_bits,
                               oracle.ddq_bits)) {
        return false;
      }
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "forward invariant diagnostic: %s\n", error.what());
    return false;
  } catch (...) {
    return false;
  }
  return true;
}

bool desired_history_near_zero_roundoff_regression_for_test() noexcept {
  auto promoted_float_from_bits = [](const std::uint32_t bits) {
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<double>(value);
  };
  constexpr std::uint32_t kObservedBits = 2889260544U;
  constexpr std::uint32_t kAcceptedBits = 2889324544U;
  bool accepted_roundoff = false;
  try {
    ExactDesiredHistory accepted{};
    accepted.ddq[4] = promoted_float_from_bits(kAcceptedBits);
    ExactDesiredHistory dropped = accepted;
    dropped.ddq[4] = -0.01;
    RobotSample sample{};
    sample.desired_ddq_rad_s2[4] =
        promoted_float_from_bits(kObservedBits);
    const ExactDesiredHistory selected =
        select_matching_desired_history(sample, accepted, dropped);
    if (selected.ddq != accepted.ddq) {
      return false;
    }
    accepted_roundoff = true;

    // The tolerance is numerical, not a blanket bypass: a 2e-6 rad/s^2
    // discrepancy matches neither candidate and must remain fail-closed.
    sample.desired_ddq_rad_s2[4] =
        static_cast<double>(static_cast<float>(2.0e-6));
    (void)select_matching_desired_history(sample, accepted, dropped);
  } catch (const ServoFault& error) {
    return accepted_roundoff && error.code == FaultCode::kDynamicState &&
           std::string(error.what()).find(
               "FCI desired history matched neither accepted nor dropped") !=
               std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

bool recoverable_fci_history_regression_for_test() noexcept {
  // Reproduce the false-positive geometry: the last accepted FCI history is
  // still inside the nominal 0.50 rad/s envelope, then one dropped command
  // packet advances it by a=5 rad/s^2 for 1 ms.  This exact returned history
  // is finite and position-safe, and must reach libfranka::limitRate instead
  // of being rejected by a duplicate lower derivative gate.
  ExactDesiredHistory confirmed{};
  confirmed.q = HardSafetyLimits::kQHome;
  confirmed.dq[0] = 0.499;
  confirmed.ddq[0] = HardSafetyLimits::kMaximumCommandAccelerationRadS2;
  const ExactDesiredHistory returned =
      extrapolate_constant_acceleration(confirmed, 1U);
  if (!(max_abs(returned.dq) >
        HardSafetyLimits::kMaximumCommandVelocityRadS)) {
    return false;
  }
  try {
    require_recoverable_fci_history(returned, HardSafetyLimits::kQHome);
  } catch (...) {
    return false;
  }

  // Relaxing only the redundant derivative gate must not weaken position or
  // episode safety.
  ExactDesiredHistory unsafe = returned;
  unsafe.q[0] = HardSafetyLimits::kSafeJointUpper[0] + 0.001;
  try {
    require_recoverable_fci_history(unsafe, HardSafetyLimits::kQHome);
  } catch (const ServoFault& error) {
    return error.code == FaultCode::kDynamicState &&
           std::string(error.what()).find("left safe joint interval") !=
               std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

bool mode_aware_episode_envelope_regression_for_test() noexcept {
  ExactDesiredHistory beyond_legacy{};
  beyond_legacy.q = HardSafetyLimits::kQHome;
  beyond_legacy.q[0] += HardSafetyLimits::kMaximumEpisodeDeltaRad + 0.01;
  if (beyond_legacy.q[0] >= HardSafetyLimits::kSafeJointUpper[0] ||
      episode_delta_allowed(ControllerMode::kLegacy, beyond_legacy.q,
                            HardSafetyLimits::kQHome) ||
      !episode_delta_allowed(ControllerMode::kQdG015, beyond_legacy.q,
                             HardSafetyLimits::kQHome)) {
    return false;
  }
  try {
    require_recoverable_fci_history(
        beyond_legacy, HardSafetyLimits::kQHome, ControllerMode::kQdG015);
  } catch (...) {
    return false;
  }
  try {
    require_recoverable_fci_history(
        beyond_legacy, HardSafetyLimits::kQHome, ControllerMode::kLegacy);
  } catch (const ServoFault& error) {
    return error.code == FaultCode::kDynamicState &&
           std::string(error.what()).find("home-centered episode envelope") !=
               std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

bool jerk_shaper_multi_period_forward_invariant_regression_for_test() noexcept {
  try {
    // Frozen independent oracle for the fixed position reserve.  The maximum
    // occurs on either signed velocity-barrier edge at |v|=V,a=0 and a full
    // 20-packet loss tail.  Recompute all loss lengths and both signs using
    // the exact semi-implicit formulas, then sweep the remaining barrier
    // domain to catch implementation/constant drift.
    constexpr double kExpectedMaximumOutwardExcursionRad = 0.04890004;
    double frozen_oracle_maximum = 0.0;
    for (const double direction : {-1.0, 1.0}) {
      const double velocity =
          direction * HardSafetyLimits::kMaximumCommandVelocityRadS;
      for (std::uint32_t loss = 0U;
           loss <=
               HardSafetyLimits::kMaximumUncontrolledContinuationPackets;
           ++loss) {
        const double packets = static_cast<double>(loss);
        const double oriented_loss_position =
            direction * packets * kCommandPacketPeriodS * velocity;
        const double oriented_loss_velocity = direction * velocity;
        frozen_oracle_maximum = std::max(
            frozen_oracle_maximum,
            oriented_loss_position +
                maximum_jerk_limited_outward_excursion(
                    oriented_loss_velocity, 0.0));
      }
    }
    if (std::abs(frozen_oracle_maximum -
                 kExpectedMaximumOutwardExcursionRad) >
            1.0e-12 ||
        frozen_oracle_maximum >
            kPositionTargetReserveRad -
                kPositionTrackingReserveRad) {
      std::fprintf(stderr, "forward invariant frozen=%0.12f expected=%0.12f reserve=%0.12f\n",
                   frozen_oracle_maximum, kExpectedMaximumOutwardExcursionRad,
                   kPositionTargetReserveRad - kPositionTrackingReserveRad);
      return false;
    }
    double swept_maximum = 0.0;
    constexpr std::uint32_t kVelocityGridIntervals = 400U;
    constexpr std::uint32_t kAccelerationGridIntervals = 8U;
    for (std::uint32_t velocity_index = 0U;
         velocity_index <= kVelocityGridIntervals;
         ++velocity_index) {
      const double velocity =
          -HardSafetyLimits::kMaximumCommandVelocityRadS +
          2.0 * HardSafetyLimits::kMaximumCommandVelocityRadS *
              static_cast<double>(velocity_index) /
              static_cast<double>(kVelocityGridIntervals);
      const double acceleration_lower = std::max(
          -HardSafetyLimits::kMaximumCommandAccelerationRadS2,
          kVelocityBarrierGainPerS *
              (-HardSafetyLimits::kMaximumCommandVelocityRadS - velocity));
      const double acceleration_upper = std::min(
          HardSafetyLimits::kMaximumCommandAccelerationRadS2,
          kVelocityBarrierGainPerS *
              (HardSafetyLimits::kMaximumCommandVelocityRadS - velocity));
      for (std::uint32_t acceleration_index = 0U;
           acceleration_index <= kAccelerationGridIntervals;
           ++acceleration_index) {
        const double acceleration =
            acceleration_lower +
            (acceleration_upper - acceleration_lower) *
                static_cast<double>(acceleration_index) /
                static_cast<double>(kAccelerationGridIntervals);
        for (std::uint32_t loss = 0U;
             loss <=
                 HardSafetyLimits::
                     kMaximumUncontrolledContinuationPackets;
             ++loss) {
          const double packets = static_cast<double>(loss);
          const double loss_position =
              packets * kCommandPacketPeriodS * velocity +
              0.5 * packets * (packets + 1.0) *
                  kCommandPacketPeriodS * kCommandPacketPeriodS *
                  acceleration;
          const double loss_velocity =
              velocity +
              packets * kCommandPacketPeriodS * acceleration;
          swept_maximum = std::max(
              swept_maximum,
              loss_position +
                  maximum_jerk_limited_outward_excursion(
                      loss_velocity, acceleration));
        }
      }
    }
    if (swept_maximum >
        kExpectedMaximumOutwardExcursionRad + 1.0e-12) {
      std::fprintf(stderr, "forward invariant swept=%0.12f expected=%0.12f\n",
                   swept_maximum, kExpectedMaximumOutwardExcursionRad);
      return false;
    }

    auto continuation_is_position_safe =
        [](const ShapedCommand& shaped,
           const std::array<double, 7>& start_q) {
          ExactDesiredHistory continuation{};
          continuation.q = shaped.q;
          continuation.dq = shaped.dq;
          continuation.ddq = shaped.ddq;
          for (std::uint32_t packets = 0U;
               packets <=
                   HardSafetyLimits::
                       kMaximumUncontrolledContinuationPackets;
               ++packets) {
            require_exact_history_envelope(continuation, start_q);
            continuation =
                extrapolate_constant_acceleration(continuation, 1U);
          }
          return true;
        };
    // Deterministic replay of the failure geometry from
    // v94-reselect-20260724-001554: J2 receives +0.003 rad every 17 servo
    // packets for 21 policy ticks.  With the removed one-packet envelope,
    // cycle 371 reached dq=0.17325, ddq=0.5; an accepted p=6 state then
    // extrapolated five packets to dq=0.17575 and produced
    // lower=0.49 > upper=0.485.
    {
      ExactDesiredHistory history{};
      history.q = HardSafetyLimits::kQHome;
      ArchivedJerkLimitedShaper shaper(history.q);
      std::array<double, 7> target = history.q;
      std::uint32_t policy_sequence = 0U;
      for (std::uint32_t cycle = 0U; cycle < 500U; ++cycle) {
        if (cycle % 17U == 0U && policy_sequence < 21U) {
          target[1] += 0.003;
          ++policy_sequence;
        }
        std::uint32_t returned_period_ms = 1U;
        if (cycle == 371U) {
          history = extrapolate_constant_acceleration(history, 5U);
          returned_period_ms = 6U;
        }
        const ShapedCommand shaped =
            shaper.step(target, history, returned_period_ms);
        if (!continuation_is_position_safe(shaped,
                                           HardSafetyLimits::kQHome)) {
          std::fprintf(stderr, "forward invariant stage=replay cycle=%u\n", cycle);
          return false;
        }
        history.q = shaped.q;
        history.dq = shaped.dq;
        history.ddq = shaped.ddq;
      }
      if (policy_sequence != 21U) {
        return false;
      }
    }

    // Sweep both velocity directions, every accepted/dropped extrapolation
    // length through the 20-packet fail-stop horizon, and injection points
    // spanning the acceleration plateau and the approach to the 0.50 rad/s
    // boundary.
    // The old one-packet braking envelope deterministically became infeasible
    // for (among others) a five-packet extrapolation near cycle 390.
    for (const double direction : {-1.0, 1.0}) {
      for (std::uint32_t extrapolated_packets = 1U;
           extrapolated_packets <=
               HardSafetyLimits::kMaximumUncontrolledContinuationPackets;
           ++extrapolated_packets) {
        for (const std::uint32_t injection_cycle :
             {250U, 350U, 450U, 500U}) {
          ExactDesiredHistory history{};
          history.q = HardSafetyLimits::kQHome;
          ArchivedJerkLimitedShaper shaper(history.q);
          std::array<double, 7> target = history.q;
          target[0] += direction * 0.50;
          const std::array<double, 7> start_q = history.q;

          for (std::uint32_t cycle = 0U; cycle < 650U; ++cycle) {
            std::uint32_t returned_period_ms = 1U;
            if (cycle == injection_cycle) {
              history = extrapolate_constant_acceleration(
                  history, extrapolated_packets);
              returned_period_ms = extrapolated_packets;
              require_exact_history_envelope(history, start_q);
            }
            const ExactDesiredHistory previous = history;
            const ShapedCommand shaped =
                shaper.step(target, history, returned_period_ms);
            if (!continuation_is_position_safe(shaped, start_q)) {
              std::fprintf(stderr, "forward invariant stage=loss direction=%g packets=%u injection=%u cycle=%u\n", direction, extrapolated_packets, injection_cycle, cycle);
              return false;
            }
            constexpr double kTolerance = 1.0e-5;
            for (std::size_t index = 0U; index < shaped.q.size(); ++index) {
              const double jerk =
                  (shaped.ddq[index] - previous.ddq[index]) /
                  kCommandPacketPeriodS;
              if (std::abs(shaped.dq[index]) >
                      HardSafetyLimits::kMaximumCommandVelocityRadS +
                          kTolerance ||
                  std::abs(shaped.ddq[index]) >
                      HardSafetyLimits::kMaximumCommandAccelerationRadS2 +
                          kTolerance ||
                  std::abs(jerk) >
                      HardSafetyLimits::kMaximumCommandJerkRadS3 +
                          kTolerance) {
                std::fprintf(stderr, "forward invariant stage=dynamics direction=%g packets=%u injection=%u cycle=%u axis=%zu v=%g a=%g j=%g\n", direction, extrapolated_packets, injection_cycle, cycle, index, shaped.dq[index], shaped.ddq[index], jerk);
                return false;
              }
            }
            history.q = shaped.q;
            history.dq = shaped.dq;
            history.ddq = shaped.ddq;
          }
        }
      }
    }

    // Approach every target-contracted joint/episode edge from q_home for
    // long enough to cover acceleration, cruise, braking, and steady hold.
    // This is the exact geometry missed by the old near-boundary-only test.
    // With a 2/s position barrier the positive one-dimensional trace became
    // infeasible at tick 5487
    // (q=.896351502243,v=.199161797757,a=.000838202243); 1/s must complete
    // 30k nominal packets without an empty jerk interval.
    for (std::size_t axis = 0U; axis < HardSafetyLimits::kQHome.size();
         ++axis) {
      for (const double direction : {-1.0, 1.0}) {
        ExactDesiredHistory history{};
        history.q = HardSafetyLimits::kQHome;
        const std::array<double, 7> start_q = history.q;
        ArchivedJerkLimitedShaper shaper(start_q);
        std::array<double, 7> target = history.q;
        const double raw_boundary =
            direction < 0.0
                ? std::max(HardSafetyLimits::kSafeJointLower[axis],
                           start_q[axis] -
                               HardSafetyLimits::kMaximumEpisodeDeltaRad)
                : std::min(HardSafetyLimits::kSafeJointUpper[axis],
                           start_q[axis] +
                               HardSafetyLimits::kMaximumEpisodeDeltaRad);
        const double contracted_target =
            raw_boundary -
            direction * kPositionTargetReserveRad;
        target[axis] = raw_boundary;
        for (std::uint32_t cycle = 0U; cycle < 30000U; ++cycle) {
          const ShapedCommand shaped = shaper.step(target, history, 1U);
          if (!continuation_is_position_safe(shaped, start_q)) {
            std::fprintf(stderr, "forward invariant stage=edge axis=%zu direction=%g cycle=%u\n", axis, direction, cycle);
            return false;
          }
          history.q = shaped.q;
          history.dq = shaped.dq;
          history.ddq = shaped.ddq;
        }
        if (std::abs(history.q[axis] - contracted_target) > 5.1e-3 ||
            std::abs(history.dq[axis]) > 5.1e-3) {
          std::fprintf(stderr, "forward invariant stage=edge-arrival axis=%zu direction=%g q=%g target=%g dq=%g\n", axis, direction, history.q[axis], contracted_target, history.dq[axis]);
          return false;
        }
      }
    }

    // Exercise the position invariant at both actual boundaries of every
    // joint: the margin-contracted hardware interval and the q_home-relative
    // episode interval.  Begin just inside the tighter boundary and command
    // toward it; every emitted command must remain inside both intervals for
    // the full 20-packet constant-acceleration continuation, including any
    // interior velocity reversal.
    for (std::size_t axis = 0U; axis < HardSafetyLimits::kQHome.size();
         ++axis) {
      for (const double direction : {-1.0, 1.0}) {
        const std::array<double, 7> start_q = HardSafetyLimits::kQHome;
        const double episode_boundary =
            start_q[axis] +
            direction * HardSafetyLimits::kMaximumEpisodeDeltaRad;
        const double boundary =
            direction < 0.0
                ? std::max(HardSafetyLimits::kSafeJointLower[axis],
                           episode_boundary)
                : std::min(HardSafetyLimits::kSafeJointUpper[axis],
                           episode_boundary);
        ExactDesiredHistory history{};
        history.q = start_q;
        history.q[axis] = boundary - direction * 0.02;
        require_exact_history_envelope(history, start_q);
        std::array<double, 7> target = history.q;
        target[axis] = boundary;
        ArchivedJerkLimitedShaper shaper(start_q);
        for (std::uint32_t cycle = 0U; cycle < 800U; ++cycle) {
          const ShapedCommand shaped = shaper.step(target, history, 1U);
          if (!continuation_is_position_safe(shaped, start_q)) {
            std::fprintf(stderr, "forward invariant stage=boundary axis=%zu direction=%g cycle=%u\n", axis, direction, cycle);
            return false;
          }
          history.q = shaped.q;
          history.dq = shaped.dq;
          history.ddq = shaped.ddq;
        }

        // A velocity momentarily pointing away from the boundary is not a
        // zero-excursion case when acceleration still points outward. Verify
        // that every possible 1..20 packet constant-acceleration gap remains
        // position-safe and leaves a feasible jerk-limited recovery command.
        ExactDesiredHistory reversal{};
        reversal.q = start_q;
        reversal.q[axis] = boundary - direction * 0.08;
        reversal.dq[axis] = -direction * 0.001;
        // Stay inside the acceleration-aware position barrier.  The prior
        // arbitrary +/-A fixture was deliberately outside that invariant and
        // therefore described a state the new shaper never emits.
        reversal.ddq[axis] = direction * 0.05;
        require_exact_history_envelope(reversal, start_q);
        const ShapedCommand reversal_command =
            shaper.step(target, reversal, 1U);
        if (!continuation_is_position_safe(reversal_command, start_q)) {
          std::fprintf(stderr, "forward invariant stage=reversal axis=%zu direction=%g\n", axis, direction);
          return false;
        }
        ExactDesiredHistory sent{};
        sent.q = reversal_command.q;
        sent.dq = reversal_command.dq;
        sent.ddq = reversal_command.ddq;
        for (std::uint32_t gap = 1U;
             gap <=
                 HardSafetyLimits::kMaximumUncontrolledContinuationPackets;
             ++gap) {
          const ExactDesiredHistory returned =
              extrapolate_constant_acceleration(sent, gap);
          require_exact_history_envelope(returned, start_q);
          const ShapedCommand recovered =
              shaper.step(target, returned, gap);
          if (!continuation_is_position_safe(recovered, start_q)) {
            std::fprintf(stderr, "forward invariant stage=recovery axis=%zu direction=%g gap=%u\n", axis, direction, gap);
            return false;
          }
        }
      }
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "forward invariant diagnostic: %s\n", error.what());
    return false;
  } catch (...) {
    return false;
  }
  return true;
}

bool jerk_shaper_worst_case_wcet_regression_for_test() noexcept {
  try {
    constexpr std::uint32_t kWarmupTicks = 2000U;
    constexpr std::uint32_t kMeasuredTicks = 20000U;
    constexpr std::uint64_t kMaximumP99Ns = 100000U;
    constexpr std::uint64_t kMaximumSingleTickNs = 750000U;

    const std::array<double, 7> start_q = HardSafetyLimits::kQHome;
    ExactDesiredHistory upper_history{};
    ExactDesiredHistory lower_history{};
    std::array<double, 7> upper_target{};
    std::array<double, 7> lower_target{};
    for (std::size_t axis = 0U; axis < start_q.size(); ++axis) {
      const double raw_lower =
          std::max(HardSafetyLimits::kSafeJointLower[axis],
                   start_q[axis] -
                       HardSafetyLimits::kMaximumEpisodeDeltaRad);
      const double raw_upper =
          std::min(HardSafetyLimits::kSafeJointUpper[axis],
                   start_q[axis] +
                       HardSafetyLimits::kMaximumEpisodeDeltaRad);
      lower_history.q[axis] =
          raw_lower + kPositionTargetReserveRad;
      upper_history.q[axis] =
          raw_upper - kPositionTargetReserveRad;
      lower_target[axis] = raw_lower;
      upper_target[axis] = raw_upper;
    }
    ArchivedJerkLimitedShaper upper_shaper(start_q);
    ArchivedJerkLimitedShaper lower_shaper(start_q);
    auto one_tick =
        [&](const bool upper) {
          ExactDesiredHistory& history =
              upper ? upper_history : lower_history;
          ArchivedJerkLimitedShaper& shaper =
              upper ? upper_shaper : lower_shaper;
          const std::array<double, 7>& target =
              upper ? upper_target : lower_target;
          const ShapedCommand shaped =
              shaper.step(target, history, 1U);
          history.q = shaped.q;
          history.dq = shaped.dq;
          history.ddq = shaped.ddq;
        };
    for (std::uint32_t tick = 0U; tick < kWarmupTicks; ++tick) {
      one_tick((tick & 1U) == 0U);
    }

    std::vector<std::uint64_t> durations_ns;
    durations_ns.reserve(kMeasuredTicks);
    for (std::uint32_t tick = 0U; tick < kMeasuredTicks; ++tick) {
      struct timespec before {};
      struct timespec after {};
      if (::clock_gettime(CLOCK_MONOTONIC_RAW, &before) != 0) {
        return false;
      }
      one_tick((tick & 1U) == 0U);
      if (::clock_gettime(CLOCK_MONOTONIC_RAW, &after) != 0) {
        return false;
      }
      const std::uint64_t before_ns =
          static_cast<std::uint64_t>(before.tv_sec) * 1000000000ULL +
          static_cast<std::uint64_t>(before.tv_nsec);
      const std::uint64_t after_ns =
          static_cast<std::uint64_t>(after.tv_sec) * 1000000000ULL +
          static_cast<std::uint64_t>(after.tv_nsec);
      if (after_ns < before_ns) {
        return false;
      }
      durations_ns.push_back(after_ns - before_ns);
    }
    std::sort(durations_ns.begin(), durations_ns.end());
    const std::size_t p99_index =
        (durations_ns.size() * 99U + 99U) / 100U - 1U;
    const std::uint64_t p99_ns = durations_ns[p99_index];
    const std::uint64_t maximum_ns = durations_ns.back();
    std::fprintf(
        stdout,
        "v94 jerk shaper Release boundary WCET: ticks=%u p99_ns=%llu "
        "max_ns=%llu limits=%llu/%llu\n",
        kMeasuredTicks,
        static_cast<unsigned long long>(p99_ns),
        static_cast<unsigned long long>(maximum_ns),
        static_cast<unsigned long long>(kMaximumP99Ns),
        static_cast<unsigned long long>(kMaximumSingleTickNs));
    return p99_ns <= kMaximumP99Ns &&
           maximum_ns <= kMaximumSingleTickNs;
  } catch (...) {
    return false;
  }
}

std::uint64_t PosixServoClock::monotonic_ns() noexcept {
  struct timespec value {};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0U;
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

std::uint64_t PosixServoClock::realtime_ns() noexcept {
  struct timespec value {};
  if (::clock_gettime(CLOCK_REALTIME, &value) != 0) {
    return 0U;
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

ServoRunResult run_servo_process(const ServoProcessConfig& config,
                                 InheritedChannels* channels,
                                 RobotBackendFactory* backend_factory,
                                 ServoClock* clock) noexcept {
  ServoRunResult result{};
  RuntimeState state{};
  std::unique_ptr<RobotBackend> backend;
  std::unique_ptr<ActiveRobotSession> active;
  FaultCode fault = FaultCode::kNone;
  StopReason stop_reason = StopReason::kRequested;
  bool finish_allowed = true;
  std::string detail = "requested stop";
  try {
    if (channels == nullptr || backend_factory == nullptr || clock == nullptr ||
        config.robot_address.empty()) {
      throw ServoFault(FaultCode::kInternal, false,
                       "servo process dependency/config is invalid");
    }
    const ChannelValidation validation =
        channels->validate_and_configure(config.expected_parent_pid);
    if (!validation.valid) {
      throw ServoFault(FaultCode::kProtocol, false,
                       "inherited channel identity/type validation failed",
                       validation.system_errno);
    }
    send_critical_or_fault(channels, &state, clock, config.session_nonce,
                           MessageKind::kHello, config.hello);
    const ArmPayload arm = wait_for_arm(config, channels, clock, &state);
    const ControllerMode controller_mode =
        static_cast<ControllerMode>(arm.controller_mode);

    backend = backend_factory->create_enforced(config.robot_address);
    if (backend == nullptr) {
      throw ServoFault(FaultCode::kRobotOpen, false,
                       "backend factory returned no enforced Robot");
    }
    const RealtimeSchedulerProofData realtime =
        require_realtime_scheduler_proof(config);
    const RobotSample preflight = backend->read_once();
    require_preflight(preflight, arm);
    IpcReadyPayload ipc_ready{};
    copy_to_wire(preflight.q_rad, ipc_ready.measured_q_rad);
    copy_to_wire(preflight.dq_rad_s, ipc_ready.measured_dq_rad_s);
    ipc_ready.q_home_linf_error_rad =
        max_abs_difference(preflight.q_rad, HardSafetyLimits::kQHome);
    ipc_ready.robot_time_ms = preflight.robot_time_ms;
    ipc_ready.static_provenance_verified = 1U;
    ipc_ready.realtime_scheduler_policy = realtime.policy;
    ipc_ready.realtime_scheduler_priority = realtime.priority;
    ipc_ready.realtime_cpu = realtime.cpu;
    ipc_ready.realtime_affinity_cpu_count = realtime.affinity_cpu_count;
    send_critical_or_fault(channels, &state, clock, config.session_nonce,
                           MessageKind::kIpcReady, ipc_ready);

    active = backend->start_joint_position_control();
    if (active == nullptr) {
      throw ServoFault(FaultCode::kRobotOpen, false,
                       "Robot returned no active joint-position handle");
    }
    const std::uint64_t session_start_ns = clock->monotonic_ns();
    state.last_heartbeat_ns = session_start_ns;
    bool first_period = true;
    std::uint64_t prior_robot_time_ms = preflight.robot_time_ms;
    std::uint64_t healthy_start_robot_time_ms = 0U;
    std::unique_ptr<FrankaV225Interpolator> interpolator;
    ExactDesiredHistory confirmed_history{};
    ExactDesiredHistory last_sent_history{};
    bool exact_history_initialized = false;
    bool sent_history_available = false;

    while (true) {
      if (config.signal_stop_requested != nullptr &&
          *config.signal_stop_requested != 0) {
        stop_reason = StopReason::kSignal;
        finish_allowed = false;
        detail = "signal stop requested";
        break;
      }
      RobotSample sample{};
      try {
        sample = active->read_once();
      } catch (const std::exception& error) {
        throw ServoFault(FaultCode::kActiveRead, false,
                         std::string("active readOnce failed: ") + error.what());
      }
      const std::uint64_t read_complete_ns = clock->monotonic_ns();
      ++state.active_reads;
      ++state.cycle;
      state.latest_sample = sample;
      // Capture the offending period before any rejection so STOP_PROOF and
      // the parent audit do not misleadingly report only the prior maximum.
      state.maximum_control_period_ms =
          std::max(state.maximum_control_period_ms, sample.control_period_ms);
      if (!first_period && sample.control_period_ms > 1U) {
        state.missed_robot_states += sample.control_period_ms - 1U;
      }
      if (sample.robot_time_ms <= prior_robot_time_ms && !first_period) {
        throw ServoFault(FaultCode::kControlPeriod, false,
                         "active RobotState.time is not strictly increasing: "
                         "actual=" +
                             std::to_string(sample.control_period_ms) + "ms");
      }
      prior_robot_time_ms = std::max(prior_robot_time_ms, sample.robot_time_ms);
      if (first_period) {
        if (sample.control_period_ms != 0U) {
          throw ServoFault(FaultCode::kControlPeriod, false,
                           "first active control period is not zero: actual=" +
                               std::to_string(sample.control_period_ms) + "ms");
        }
      } else if (sample.control_period_ms == 0U ||
                 sample.control_period_ms >
                     HardSafetyLimits::kMaximumRecoverableControlPeriodMs) {
        throw ServoFault(FaultCode::kControlPeriod, false,
                         "active control period is outside FCI recoverable "
                         "returned range 1.." +
                             std::to_string(
                                 HardSafetyLimits::
                                     kMaximumRecoverableControlPeriodMs) +
                             " ms: actual=" +
                             std::to_string(sample.control_period_ms) + "ms");
      }
      ExactDesiredHistory reference_history{};
      if (first_period) {
        // Desired derivatives should be zero when the position generator is
        // opened at the commissioned rest pose.  Keep only a tiny numerical
        // allowance; these bounds are respectively 200,000x and 5,000x below
        // the active command envelopes and are used as the bootstrap history,
        // never silently replaced with zero.
        constexpr double kBootstrapMaximumDesiredVelocityRadS = 1.0e-6;
        constexpr double kBootstrapMaximumDesiredAccelerationRadS2 = 1.0e-4;
        if (sample.mode != RobotModeCode::kMove || !state_is_clear(sample) ||
            !sample_vectors_are_finite(sample) ||
            max_abs(sample.dq_rad_s) >
                HardSafetyLimits::kStopMaximumVelocityRadS ||
            max_abs_difference(sample.q_rad, HardSafetyLimits::kQHome) >
                HardSafetyLimits::kMaximumStartErrorRad ||
            max_abs_difference(sample.desired_q_rad,
                               HardSafetyLimits::kQHome) >
                HardSafetyLimits::kMaximumStartErrorRad ||
            max_abs(sample.desired_dq_rad_s) >
                kBootstrapMaximumDesiredVelocityRadS ||
            max_abs(sample.desired_ddq_rad_s2) >
                kBootstrapMaximumDesiredAccelerationRadS2 ||
            max_abs_difference(sample.q_rad, sample.desired_q_rad) >
                HardSafetyLimits::kMaximumTrackingErrorRad) {
          throw ServoFault(FaultCode::kDynamicState, false,
                           "first active state is not a stationary V94 bootstrap");
        }
        for (std::size_t index = 0U; index < 7U; ++index) {
          if (sample.desired_q_rad[index] <
                  HardSafetyLimits::kSafeJointLower[index] ||
              sample.desired_q_rad[index] >
                  HardSafetyLimits::kSafeJointUpper[index]) {
            throw ServoFault(FaultCode::kDynamicState, false,
                             "first active desired q is outside safe joint interval");
          }
        }
        state.start_q = sample.q_rad;
        state.prior_command = sample.desired_q_rad;
        state.last_command = sample.desired_q_rad;
        state.last_command_velocity = sample.desired_dq_rad_s;
        state.last_command_acceleration = sample.desired_ddq_rad_s2;
        // ActiveControl inherits the firmware's exact desired-history tuple.
        // Start the software generator from that same tuple so the first
        // returned packet remains unambiguous.  The policy is admitted only
        // after 100 healthy cycles, and receives the actual coherent 29-D
        // generator snapshot rather than an artificial reset value.
        state.held_target = sample.desired_q_rad;
        state.shaper_q_d_rad = sample.desired_q_rad;
        state.shaper_dq_d_rad_s = sample.desired_dq_rad_s;
        state.shaper_ddq_d_rad_s2 = sample.desired_ddq_rad_s2;
        state.held_q_cmd_rad = state.held_target;
        interpolator = std::make_unique<FrankaV225Interpolator>(
            sample.desired_q_rad, sample.desired_dq_rad_s,
            sample.desired_ddq_rad_s2);
        reference_history.q = state.last_command;
        reference_history.dq = state.last_command_velocity;
        reference_history.ddq = state.last_command_acceleration;
        require_recoverable_fci_history(reference_history, state.start_q,
                                        controller_mode);
        confirmed_history = reference_history;
        exact_history_initialized = true;
      } else {
        require_dynamic_state(sample, state.last_command,
                              state.prior_command, state.start_q,
                              controller_mode,
                              state.maximum_control_period_ms);
        if (!exact_history_initialized || !sent_history_available) {
          throw ServoFault(FaultCode::kInternal, false,
                           "exact desired-history state machine is uninitialized");
        }
        reference_history = select_returned_desired_history(
            sample, confirmed_history, last_sent_history);
        require_recoverable_fci_history(reference_history, state.start_q,
                                        controller_mode);
        confirmed_history = reference_history;
      }

      // Record expiry before consuming the socket. A heartbeat that was
      // queued or sender-delayed beyond the receipt watchdog must not revive
      // an expired supervisor. An authenticated STOP still wins below so a
      // late stop request is never converted into continued motion.
      const bool heartbeat_expired_before_drain =
          read_complete_ns -
                  std::min(read_complete_ns, state.last_heartbeat_ns) >
              arm.heartbeat_timeout_ns;
      const DrainResult drained =
          drain_critical(arm, config, channels, clock, &state);
      if (drained.stop_requested) {
        stop_reason = drained.stop_reason;
        finish_allowed = false;
        detail = "parent requested supervised stop";
        break;
      }
      if (heartbeat_expired_before_drain) {
        throw ServoFault(FaultCode::kHeartbeatTimeout, true,
                         "supervisor heartbeat receipt timed out before IPC drain");
      }
      const std::uint64_t now_ns = clock->monotonic_ns();
      if (now_ns - state.last_heartbeat_ns > arm.heartbeat_timeout_ns) {
        throw ServoFault(FaultCode::kHeartbeatTimeout, true,
                         "supervisor heartbeat timed out");
      }
      if (now_ns - session_start_ns > arm.maximum_session_duration_ns) {
        throw ServoFault(FaultCode::kSessionTimeout, true,
                         "supervised native session timed out");
      }
      if (state.action_ready && state.last_target_sequence == 0U &&
          now_ns - state.action_ready_ns > arm.first_target_timeout_ns) {
        throw ServoFault(FaultCode::kFirstTargetTimeout, true,
                         "first policy target timed out after ACTION_READY");
      }
      if (state.last_target_sequence != 0U &&
          now_ns - state.last_target_received_ns > arm.target_timeout_ns) {
        throw ServoFault(FaultCode::kTargetTimeout, true,
                         "policy inter-target watchdog timed out");
      }

      // The accepted V225/V226 command path keeps a stateful 100 Hz target
      // filter followed by a 6 Hz critically damped interpolator.  Synchronize
      // only its desired command state to the exact accepted/dropped FCI
      // history; never re-anchor it from measured q.  This preserves the
      // temporal filter state while correctly recovering from a lost command
      // packet.
      if (interpolator == nullptr) {
        throw ServoFault(FaultCode::kInternal, false,
                         "V225 interpolator was not initialized");
      }
      interpolator->synchronize_desired_history(
          reference_history.q, reference_history.dq, reference_history.ddq);
      InterpolatedCommand generated = interpolator->state();
      if (!first_period) {
        const bool contact_hold = state_has_contact(sample);
        if (contact_hold) {
          // Use the exact FCI desired history, not measured q, as the
          // deceleration anchor.  This preserves the position-generator
          // derivative contract while stopping further motion into contact.
          // The policy-held target remains untouched and can be resumed after
          // Franka clears its lower contact flags.
          interpolator->hold_filtered_target_at(reference_history.q);
        }
        generated = interpolator->step(
            contact_hold ? reference_history.q : state.held_target,
            static_cast<double>(sample.control_period_ms) *
                kCommandPacketPeriodS);
      }
      std::array<double, 7> command{};
      try {
        // ActiveControl does not automatically run Robot::control's optional
        // limiter/filter wrapper. Apply only libfranka's official limiter
        // explicitly; the built-in low-pass is intentionally absent because
        // the accepted 100 Hz filter is already above.
        command = SendableJointLimits::clamp(
            active->limit_joint_position_command(
                generated.q, reference_history.q, reference_history.dq,
                reference_history.ddq));
      } catch (const std::exception& error) {
        throw ServoFault(FaultCode::kDynamicState, false,
                         std::string("libfranka joint-position limiter failed: ") +
                             error.what());
      }
      if (!episode_delta_allowed(controller_mode, command, state.start_q)) {
        throw ServoFault(FaultCode::kDynamicState, false,
                         "legacy V225 command left home-centered episode envelope");
      }
      // FCI differentiates each newly accepted joint-position command at its
      // fixed 1 ms packet time, including after a returned multi-ms period.
      // Reconstruct the exact post-limiter desired history used by the next
      // accepted/dropped-history proof.
      ShapedCommand limited{};
      limited.q = command;
      for (std::size_t index = 0U; index < command.size(); ++index) {
        limited.dq[index] =
            (limited.q[index] - reference_history.q[index]) /
            kCommandPacketPeriodS;
        limited.ddq[index] =
            (limited.dq[index] - reference_history.dq[index]) /
            kCommandPacketPeriodS;
      }
      ExactDesiredHistory limited_history{};
      limited_history.q = limited.q;
      limited_history.dq = limited.dq;
      limited_history.ddq = limited.ddq;
      require_recoverable_fci_history(limited_history, state.start_q,
                                      controller_mode);
      const std::uint64_t before_write_ns = clock->monotonic_ns();
      const std::uint64_t prewrite_ns = before_write_ns - read_complete_ns;
      state.maximum_read_to_write_ns =
          std::max(state.maximum_read_to_write_ns, prewrite_ns);
      if (prewrite_ns > HardSafetyLimits::kMaximumReadToWriteNs ||
          before_write_ns >= arm.authorization_expires_monotonic_ns) {
        throw ServoFault(FaultCode::kReadToWriteDeadline, false,
                         "pre-write deadline/authorization boundary expired");
      }
      try {
        active->write_once(command, false);
      } catch (const std::exception& error) {
        throw ServoFault(FaultCode::kActiveWrite, false,
                         std::string("active writeOnce failed: ") + error.what());
      }
      const std::uint64_t after_write_ns = clock->monotonic_ns();
      ++state.active_writes;
      state.prior_command = state.last_command;
      state.last_command = command;
      state.last_command_velocity = limited.dq;
      state.last_command_acceleration = limited.ddq;
      last_sent_history = limited_history;
      interpolator->synchronize_desired_history(
          limited.q, limited.dq, limited.ddq);
      const InterpolatedCommand& persistent_shaper_state = interpolator->state();
      state.shaper_q_d_rad = persistent_shaper_state.q;
      state.shaper_dq_d_rad_s = persistent_shaper_state.dq;
      state.shaper_ddq_d_rad_s2 = persistent_shaper_state.ddq;
      state.held_q_cmd_rad = state.held_target;
      state.controller_state29 = build_controller_state29(
          state.held_q_cmd_rad, persistent_shaper_state, sample.q_rad);
      sent_history_available = true;
      const std::uint64_t read_to_write_ns = after_write_ns - read_complete_ns;
      state.maximum_read_to_write_ns =
          std::max(state.maximum_read_to_write_ns, read_to_write_ns);
      if (read_to_write_ns > HardSafetyLimits::kMaximumReadToWriteNs) {
        throw ServoFault(FaultCode::kReadToWriteDeadline, false,
                         "write completion exceeded 0.8 ms");
      }

      if (!first_period) {
        const bool healthy = state_is_dynamic_motion_safe(sample) &&
                             sample.control_command_success_rate >=
                                 HardSafetyLimits::kMinimumHealthySuccessRate;
        if (healthy) {
          if (state.consecutive_healthy_cycles == 0U) {
            healthy_start_robot_time_ms = sample.robot_time_ms;
          }
          ++state.consecutive_healthy_cycles;
          state.minimum_healthy_success_rate =
              std::min(state.minimum_healthy_success_rate,
                       sample.control_command_success_rate);
        } else {
          state.consecutive_healthy_cycles = 0U;
          state.minimum_healthy_success_rate = 1.0;
        }
      }
      if (!state.action_ready &&
          state.consecutive_healthy_cycles >=
              HardSafetyLimits::kHealthyCyclesBeforeAction) {
        state.action_ready = true;
        state.action_ready_ns = after_write_ns;
        ActionReadyPayload ready{};
        ready.consecutive_healthy_cycles = state.consecutive_healthy_cycles;
        ready.maximum_control_period_ms = state.maximum_control_period_ms;
        ready.healthy_hold_robot_time_ms =
            sample.robot_time_ms - healthy_start_robot_time_ms + 1U;
        ready.maximum_read_to_write_ns = state.maximum_read_to_write_ns;
        ready.minimum_control_command_success_rate =
            state.minimum_healthy_success_rate;
        ready.latest_control_command_success_rate =
            sample.control_command_success_rate;
        copy_to_wire(sample.q_rad, ready.measured_q_rad);
        copy_to_wire(sample.dq_rad_s, ready.measured_dq_rad_s);
        ready.robot_time_ms = sample.robot_time_ms;
        ready.active_read_count = state.active_reads;
        ready.active_write_count = state.active_writes;
        ready.status_flags = sample.status_flags;
        ready.cumulative_missed_robot_states = state.missed_robot_states;
        send_critical_or_fault(channels, &state, clock, config.session_nonce,
                               MessageKind::kActionReady, ready);
      }

      if (state.target_pending_ack) {
        AckPayload ack{};
        ack.target_sequence = state.pending_target.target_sequence;
        ack.observation_sequence = state.pending_target.observation_sequence;
        ack.target_produced_monotonic_ns =
            state.pending_target.produced_monotonic_ns;
        ack.target_age_at_write_ns =
            after_write_ns >= state.pending_target.produced_monotonic_ns
                ? after_write_ns - state.pending_target.produced_monotonic_ns
                : 0U;
        ack.control_cycle = state.cycle;
        ack.robot_time_ms = sample.robot_time_ms;
        ack.applied_monotonic_ns = after_write_ns;
        ack.control_period_ms = sample.control_period_ms;
        std::copy_n(state.pending_target.target_q_rad, 7U, ack.target_q_rad);
        copy_to_wire(command, ack.commanded_q_rad);
        copy_to_wire(sample.q_rad, ack.measured_q_rad);
        ack.read_to_write_ns = read_to_write_ns;
        ack.maximum_tracking_error_rad =
            max_abs_difference(sample.q_rad, command);
        send_critical_or_fault(channels, &state, clock, config.session_nonce,
                               MessageKind::kAck, ack);
        state.target_pending_ack = false;
        ++state.target_ack_count;
      }
      if (state.cycle == 1U ||
          state.cycle % HardSafetyLimits::kStateDecimation == 0U) {
        publish_state_best_effort(config, channels, clock, &state,
                                  read_to_write_ns);
      }
      first_period = false;
    }
  } catch (const ServoFault& error) {
    fault = error.code;
    // Fault and watchdog exits are preemptive stops.  A MotionFinished write
    // is legal only after an independently proven at-rest generator state,
    // which no exceptional path can provide.
    finish_allowed = false;
    stop_reason = error.code == FaultCode::kPeerDisconnected
                      ? StopReason::kPeerDisconnected
                      : StopReason::kFault;
    detail = error.what();
    state.fault_reply_delivered = send_fault_best_effort(
        channels, &state, clock, config.session_nonce, fault,
        error.system_errno, detail);
  } catch (const std::exception& error) {
    fault = FaultCode::kInternal;
    finish_allowed = false;
    stop_reason = StopReason::kFault;
    detail = std::string("unexpected native exception: ") + error.what();
    state.fault_reply_delivered = send_fault_best_effort(
        channels, &state, clock, config.session_nonce, fault, 0, detail);
  } catch (...) {
    fault = FaultCode::kInternal;
    finish_allowed = false;
    stop_reason = StopReason::kFault;
    detail = "unknown native exception";
    state.fault_reply_delivered = send_fault_best_effort(
        channels, &state, clock, config.session_nonce, fault, 0, detail);
  }

  StopProofPayload proof = cleanup_robot(&backend, &active, &state, stop_reason,
                                         fault, finish_allowed, detail);
  proof.fault_reply_delivered = state.fault_reply_delivered ? 1U : 0U;
  result.stop_proof = proof;
  if (channels != nullptr && clock != nullptr) {
    EncodedPacket packet{};
    if (encode_payload(MessageKind::kStopProof, state.critical_out_sequence,
                       clock->monotonic_ns(), config.session_nonce, proof,
                       &packet) == CodecError::kNone) {
      int system_errno = 0;
      result.stop_proof_delivered =
          channels->send_critical(packet, &system_errno) == IoStatus::kOk;
    }
  }
  result.terminal_fault = fault;
  const bool healthy_finish = proof.finish_attempted != 0U &&
                              proof.finish_succeeded != 0U &&
                              proof.robot_stop_attempted == 0U;
  const bool explicit_stop_fallback =
      proof.robot_stop_attempted != 0U &&
      !(proof.finish_attempted != 0U && proof.finish_succeeded != 0U);
  result.exit_code =
      (fault == FaultCode::kNone &&
       (healthy_finish || explicit_stop_fallback) &&
       proof.active_handle_released != 0U &&
       proof.robot_backend_released != 0U &&
       proof.idle_dq_verified != 0U)
          ? 0
          : 1;
  return result;
}

}  // namespace anydex::v94_franka_servo
