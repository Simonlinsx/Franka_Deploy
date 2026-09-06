#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace anydex::v94_franka_servo {

constexpr std::uint32_t kProtocolMagic = 0x46343956U;  // "V94F" on little endian.
constexpr std::uint16_t kProtocolVersion = 4U;
constexpr std::size_t kSessionNonceBytes = 16U;
constexpr std::size_t kSha256Bytes = 32U;
constexpr std::size_t kSha1Bytes = 20U;
constexpr std::size_t kFaultDetailBytes = 160U;
constexpr std::size_t kMaximumPacketBytes = 1024U;
constexpr std::uint32_t kStateHasCurrentErrors = 1U << 0U;
constexpr std::uint32_t kStateHasJointContact = 1U << 1U;
constexpr std::uint32_t kStateHasJointCollision = 1U << 2U;
constexpr std::uint32_t kStateHasCartesianContact = 1U << 3U;
constexpr std::uint32_t kStateHasCartesianCollision = 1U << 4U;
constexpr std::uint32_t kStateHasLastMotionErrors = 1U << 5U;
constexpr std::uint32_t kStateHasOnlyCommunicationConstraintsViolation = 1U << 6U;
constexpr std::uint32_t kKnownStateStatusMask =
    kStateHasCurrentErrors | kStateHasJointContact | kStateHasJointCollision |
    kStateHasCartesianContact | kStateHasCartesianCollision |
    kStateHasLastMotionErrors |
    kStateHasOnlyCommunicationConstraintsViolation;

enum class ControllerMode : std::uint32_t {
  // Preserve the commissioned V94/V225 target contract.  Each TARGET is
  // bounded relative to the previously held target by the compiled 0.020 rad
  // high-level guard before entering the physical trajectory generator.
  kLegacy = 0U,
  // q_d g015 targets are generated relative to an earlier coherent software
  // shaper snapshot.  They therefore must not be compared with the target
  // received one policy period earlier.  They are also not constrained by a
  // home-centered episode radius; compiled absolute joint intervals and the
  // final libfranka trajectory limiter remain mandatory.
  kQdG015 = 1U,
};

enum class MessageKind : std::uint16_t {
  kArm = 1U,
  kHeartbeat = 2U,
  kTarget = 3U,
  kStop = 4U,
  kHello = 0x101U,
  kIpcReady = 0x102U,
  kState = 0x103U,
  kAck = 0x104U,
  kFault = 0x105U,
  kStopProof = 0x106U,
  kActionReady = 0x107U,
};

enum class FaultCode : std::uint32_t {
  kNone = 0U,
  kProtocol = 1U,
  kAuthorization = 2U,
  kPeerDisconnected = 3U,
  kHeartbeatTimeout = 4U,
  kFirstTargetTimeout = 5U,
  kTargetTimeout = 6U,
  kSessionTimeout = 7U,
  kRobotOpen = 8U,
  kStaticPreflight = 9U,
  kActiveRead = 10U,
  kControlPeriod = 11U,
  kDynamicState = 12U,
  kTargetValidation = 13U,
  kReadToWriteDeadline = 14U,
  kActiveWrite = 15U,
  kCriticalReplyBlocked = 16U,
  kSignal = 17U,
  kStopCleanup = 18U,
  kInternal = 19U,
};

enum class StopReason : std::uint32_t {
  kRequested = 1U,
  kSignal = 2U,
  kFault = 3U,
  kPeerDisconnected = 4U,
};

enum class RobotModeCode : std::uint32_t {
  kUnknown = 0U,
  kIdle = 1U,
  kMove = 2U,
  kOther = 3U,
  kGuiding = 4U,
  kReflex = 5U,
  kUserStopped = 6U,
  kAutomaticErrorRecovery = 7U,
};

#pragma pack(push, 1)

struct PacketHeader final {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t kind;
  std::uint32_t payload_bytes;
  std::uint32_t flags;
  std::uint64_t packet_sequence;
  std::uint64_t monotonic_ns;
  std::uint8_t session_nonce[kSessionNonceBytes];
  std::uint32_t crc32;
  std::uint32_t reserved;
};

struct HelloPayload final {
  std::uint32_t process_id;
  std::uint32_t state_decimation;
  std::uint32_t protocol_version;
  std::uint32_t safety_limits_schema;
  std::uint8_t libfranka_sha256[kSha256Bytes];
  std::uint8_t libfranka_source_commit[kSha1Bytes];
  std::uint8_t producer_build_sha256[kSha256Bytes];
  double maximum_command_velocity_rad_s;
  double maximum_command_acceleration_rad_s2;
  double maximum_command_jerk_rad_s3;
  double maximum_start_error_rad;
  double maximum_tick_target_delta_rad;
  double maximum_episode_delta_rad;
  double maximum_tracking_error_rad;
  double maximum_read_to_write_s;
};

struct ArmPayload final {
  std::uint8_t profile_sha256[kSha256Bytes];
  std::uint8_t envelope_sha256[kSha256Bytes];
  std::uint8_t permit_sha256[kSha256Bytes];
  std::uint8_t run_id_sha256[kSha256Bytes];
  std::uint8_t authorization_id[16];
  std::uint64_t authorization_issued_monotonic_ns;
  std::uint64_t authorization_expires_monotonic_ns;
  std::uint64_t heartbeat_timeout_ns;
  std::uint64_t first_target_timeout_ns;
  // Receive-to-receive watchdog after TARGET(1).  TARGET packet timestamp age
  // is checked separately against HardSafetyLimits::kMaximumTargetAgeNs.
  std::uint64_t target_timeout_ns;
  std::uint64_t maximum_session_duration_ns;
  double q_home_rad[7];
  double safe_joint_lower_rad[7];
  double safe_joint_upper_rad[7];
  double expected_F_T_EE[16];
  double expected_end_effector_mass_kg;
  double expected_end_effector_com_m[3];
  double expected_end_effector_inertia_kg_m2[9];
  double expected_external_load_mass_kg;
  std::uint32_t maximum_target_count;
  std::uint32_t controller_mode;
  std::uint8_t reserved[8];
};

struct HeartbeatPayload final {
  std::uint64_t heartbeat_sequence;
};

struct TargetPayload final {
  std::uint64_t target_sequence;
  std::uint64_t observation_sequence;
  std::uint64_t produced_monotonic_ns;
  double target_q_rad[7];
};

struct StopPayload final {
  std::uint32_t reason_code;
  std::uint32_t reserved;
  std::uint64_t requested_monotonic_ns;
};

struct IpcReadyPayload final {
  double measured_q_rad[7];
  double measured_dq_rad_s[7];
  double q_home_linf_error_rad;
  std::uint64_t robot_time_ms;
  std::uint32_t static_provenance_verified;
  std::uint32_t realtime_scheduler_policy;
  std::uint32_t realtime_scheduler_priority;
  std::uint32_t realtime_cpu;
  std::uint32_t realtime_affinity_cpu_count;
  std::uint32_t reserved;
};

struct ActionReadyPayload final {
  std::uint32_t consecutive_healthy_cycles;
  std::uint32_t maximum_control_period_ms;
  std::uint64_t healthy_hold_robot_time_ms;
  std::uint64_t maximum_read_to_write_ns;
  double minimum_control_command_success_rate;
  double latest_control_command_success_rate;
  double measured_q_rad[7];
  double measured_dq_rad_s[7];
  std::uint64_t robot_time_ms;
  std::uint64_t active_read_count;
  std::uint64_t active_write_count;
  std::uint32_t status_flags;
  std::uint32_t cumulative_missed_robot_states;
};

struct StatePayload final {
  std::uint64_t control_cycle;
  std::uint64_t robot_time_ms;
  std::uint64_t captured_monotonic_ns;
  std::uint64_t captured_realtime_ns;
  std::uint64_t active_target_sequence;
  std::uint64_t active_observation_sequence;
  std::uint32_t control_period_ms;
  std::uint32_t robot_mode;
  double control_command_success_rate;
  double measured_q_rad[7];
  double measured_dq_rad_s[7];
  double O_T_EE[16];
  double commanded_q_rad[7];
  std::uint32_t status_flags;
  std::uint32_t cumulative_missed_robot_states;
  std::uint64_t last_read_to_write_ns;
  std::uint64_t maximum_read_to_write_ns;
  std::uint64_t telemetry_drop_count;
  std::uint32_t health_flags;
  std::uint32_t consecutive_healthy_cycles;
  std::uint64_t active_read_count;
  std::uint64_t active_write_count;
  std::uint64_t target_ack_count;
  // Exact legacy/V75 deployable controller-state observation, already clipped
  // and converted to float32 by the real-time owner:
  // target lag[7], tracking error[7], desired dq[7], desired ddq[7], alpha.
  float controller_state29[29];
  // Version-4 coherent software-controller snapshot.  These are the exact
  // persistent desired-history values after the final libfranka position
  // limiter accepted this cycle's command, plus the policy target held while
  // deriving controller_state29.  commanded_q_rad and shaper_q_d_rad are
  // intentionally identical; both names remain on the wire so legacy
  // telemetry consumers keep their established commanded-q field while q_d
  // policies receive an unambiguous state contract.
  double shaper_q_d_rad[7];
  double shaper_dq_d_rad_s[7];
  double shaper_ddq_d_rad_s2[7];
  double held_q_cmd_rad[7];
};

struct AckPayload final {
  std::uint64_t target_sequence;
  std::uint64_t observation_sequence;
  std::uint64_t target_produced_monotonic_ns;
  std::uint64_t target_age_at_write_ns;
  std::uint64_t control_cycle;
  std::uint64_t robot_time_ms;
  std::uint64_t applied_monotonic_ns;
  std::uint32_t control_period_ms;
  std::uint32_t reserved;
  double target_q_rad[7];
  double commanded_q_rad[7];
  double measured_q_rad[7];
  std::uint64_t read_to_write_ns;
  double maximum_tracking_error_rad;
};

struct FaultPayload final {
  std::uint32_t fault_code;
  std::uint32_t detail_bytes;
  std::uint64_t control_cycle;
  std::uint64_t target_sequence;
  std::uint64_t robot_time_ms;
  std::int32_t system_errno;
  std::uint32_t reserved;
  char detail[kFaultDetailBytes];
};

struct StopVerificationSample final {
  std::uint64_t robot_time_ms;
  std::uint32_t robot_mode;
  std::uint32_t status_flags;
  double measured_dq_rad_s[7];
};

struct StopProofPayload final {
  std::uint32_t stop_reason;
  std::uint32_t terminal_fault_code;
  std::uint8_t finish_attempted;
  std::uint8_t finish_succeeded;
  std::uint8_t robot_stop_attempted;
  std::uint8_t robot_stop_succeeded;
  std::uint8_t idle_dq_verified;
  std::uint8_t active_handle_released;
  std::uint8_t fault_reply_delivered;
  std::uint8_t robot_backend_released;
  std::uint64_t control_cycles;
  std::uint64_t last_target_sequence;
  std::uint64_t active_read_count;
  std::uint64_t active_write_count;
  std::uint64_t target_ack_count;
  std::uint32_t stop_verification_samples;
  std::uint32_t stop_consecutive_idle_samples;
  std::uint32_t final_robot_mode;
  std::uint32_t maximum_control_period_ms;
  double maximum_stop_dq_rad_s;
  std::uint64_t maximum_read_to_write_ns;
  std::uint64_t pre_stop_robot_time_ms;
  std::uint64_t final_robot_time_ms;
  std::uint64_t telemetry_drop_count;
  StopVerificationSample verified_samples[3];
  char detail[kFaultDetailBytes];
};

#pragma pack(pop)

static_assert(sizeof(PacketHeader) == 56U);
static_assert(sizeof(HelloPayload) == 164U);
static_assert(sizeof(ArmPayload) == 616U);
static_assert(sizeof(HeartbeatPayload) == 8U);
static_assert(sizeof(TargetPayload) == 80U);
static_assert(sizeof(StopPayload) == 16U);
static_assert(sizeof(IpcReadyPayload) == 152U);
static_assert(sizeof(ActionReadyPayload) == 184U);
static_assert(sizeof(StatePayload) == 764U);
static_assert(sizeof(AckPayload) == 248U);
static_assert(sizeof(FaultPayload) == 200U);
static_assert(sizeof(StopVerificationSample) == 72U);
static_assert(sizeof(StopProofPayload) == 488U);

static_assert(static_cast<std::uint32_t>(ControllerMode::kLegacy) == 0U);
static_assert(static_cast<std::uint32_t>(ControllerMode::kQdG015) == 1U);
static_assert(offsetof(ArmPayload, maximum_target_count) == 600U);
static_assert(offsetof(ArmPayload, controller_mode) == 604U);
static_assert(offsetof(ArmPayload, reserved) == 608U);
static_assert(offsetof(StatePayload, controller_state29) == 424U);
static_assert(offsetof(StatePayload, shaper_q_d_rad) == 540U);
static_assert(offsetof(StatePayload, shaper_dq_d_rad_s) == 596U);
static_assert(offsetof(StatePayload, shaper_ddq_d_rad_s2) == 652U);
static_assert(offsetof(StatePayload, held_q_cmd_rad) == 708U);

static_assert(sizeof(double) == 8U);
static_assert(std::numeric_limits<double>::is_iec559);

static_assert(std::is_trivially_copyable_v<PacketHeader>);
static_assert(std::is_trivially_copyable_v<ArmPayload>);
static_assert(std::is_trivially_copyable_v<StatePayload>);

struct EncodedPacket final {
  std::array<std::uint8_t, kMaximumPacketBytes> bytes{};
  std::size_t size{0U};
};

enum class CodecError : std::uint32_t {
  kNone = 0U,
  kHostNotLittleEndian,
  kInvalidKind,
  kPayloadSize,
  kPacketTooLarge,
  kTruncated,
  kMagic,
  kVersion,
  kFlags,
  kReserved,
  kSequence,
  kNonce,
  kCrc,
};

struct DecodedPacket final {
  PacketHeader header{};
  const std::uint8_t* payload{nullptr};
  std::size_t payload_size{0U};
};

bool host_is_little_endian() noexcept;
bool message_kind_is_known(MessageKind kind) noexcept;
std::size_t expected_payload_size(MessageKind kind) noexcept;
std::uint32_t protocol_crc32(const std::uint8_t* data,
                             std::size_t size) noexcept;

CodecError encode_packet(MessageKind kind,
                         std::uint64_t packet_sequence,
                         std::uint64_t monotonic_ns,
                         const std::array<std::uint8_t, kSessionNonceBytes>& nonce,
                         const void* payload,
                         std::size_t payload_size,
                         EncodedPacket* output) noexcept;

CodecError decode_packet(const std::uint8_t* packet,
                         std::size_t packet_size,
                         std::uint64_t expected_packet_sequence,
                         const std::array<std::uint8_t, kSessionNonceBytes>& expected_nonce,
                         DecodedPacket* output) noexcept;

CodecError decode_hello_packet(
    const std::uint8_t* packet,
    std::size_t packet_size,
    DecodedPacket* output,
    std::array<std::uint8_t, kSessionNonceBytes>* adopted_nonce) noexcept;

template <typename Payload>
CodecError encode_payload(
    const MessageKind kind,
    const std::uint64_t packet_sequence,
    const std::uint64_t monotonic_ns,
    const std::array<std::uint8_t, kSessionNonceBytes>& nonce,
    const Payload& payload,
    EncodedPacket* output) noexcept {
  static_assert(std::is_trivially_copyable_v<Payload>);
  if (sizeof(Payload) != expected_payload_size(kind)) {
    return CodecError::kPayloadSize;
  }
  return encode_packet(kind, packet_sequence, monotonic_ns, nonce, &payload,
                       sizeof(Payload), output);
}

template <typename Payload>
CodecError copy_payload(const DecodedPacket& packet,
                        Payload* output) noexcept {
  static_assert(std::is_trivially_copyable_v<Payload>);
  if (output == nullptr || packet.payload == nullptr ||
      packet.payload_size != sizeof(Payload)) {
    return CodecError::kPayloadSize;
  }
  std::uint8_t* destination = reinterpret_cast<std::uint8_t*>(output);
  for (std::size_t index = 0U; index < sizeof(Payload); ++index) {
    destination[index] = packet.payload[index];
  }
  return CodecError::kNone;
}

const char* codec_error_name(CodecError error) noexcept;

}  // namespace anydex::v94_franka_servo
