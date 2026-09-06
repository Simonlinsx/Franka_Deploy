#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace anydex::telemetry {

// Wire ABI v1 is intentionally fixed-width and little-endian.  Changing any
// wire structure, enum value, matrix convention, or hash rule requires a new
// ABI major version and schema digest.
constexpr std::uint16_t kAbiMajor = 1;
constexpr std::uint16_t kAbiMinor = 0;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint64_t kLayoutMagic = 0x314d4c4554584441ULL;  // "ADXTELM1"
constexpr std::uint64_t kLayoutReady = 0x315944524c455441ULL;  // "ATELRDY1"
constexpr std::size_t kDigestBytes = 32;
constexpr std::size_t kUuidBytes = 16;
constexpr std::size_t kStageNameBytes = 32;
constexpr std::size_t kProducerNameBytes = 48;
constexpr std::size_t kRobotIdBytes = 32;
constexpr std::uint32_t kDefaultReadAttempts = 8;

// SHA-256 of docs/ABI.md's canonical v1 schema string.  It is stored in every
// mapping and checked when a reader opens the mapping.
constexpr std::array<std::uint8_t, kDigestBytes> kAbiSchemaSha256{
    0x22, 0x34, 0x77, 0x80, 0xa4, 0xca, 0xac, 0x33,
    0x71, 0x92, 0x48, 0x0a, 0xff, 0x94, 0x8c, 0x65,
    0xb0, 0x1e, 0x9c, 0xe2, 0xab, 0x4f, 0xd1, 0xd1,
    0xae, 0x97, 0xe8, 0xa4, 0xc0, 0x31, 0x7b, 0xb5,
};

enum class Source : std::uint16_t {
  kUnknown = 0,
  kFrankaRobotStateOTEE = 1,
  kInspireRh56AngleAct = 2,
  kSyntheticTest = 32767,
};

enum class MeasurementKind : std::uint16_t {
  kUnknown = 0,
  kMeasured = 1,
  kCommanded = 2,
  kDerived = 3,
  kSyntheticTest = 32767,
};

enum ArmValidity : std::uint32_t {
  kArmTimestampValid = 1U << 0U,
  kArmPoseValid = 1U << 1U,
  kArmQValid = 1U << 2U,
  kArmDqValid = 1U << 3U,
  kArmSuccessRateValid = 1U << 4U,
};

enum HandValidity : std::uint32_t {
  kHandTimestampValid = 1U << 0U,
  kHandAnglesValid = 1U << 1U,
  kHandTargetsValid = 1U << 2U,
  kHandCurrentValid = 1U << 3U,
  kHandForceValid = 1U << 4U,
  kHandTemperatureValid = 1U << 5U,
  kHandStatusValid = 1U << 6U,
  kHandErrorsValid = 1U << 7U,
};

// Immutable per-session provenance supplied before the mapping becomes ready.
// All hashes are raw SHA-256 bytes.  UUID is RFC-4122/network byte order.
struct SessionProvenance final {
  std::uint8_t run_uuid[kUuidBytes];
  std::uint8_t execution_contract_sha256[kDigestBytes];
  std::uint8_t source_snapshot_sha256[kDigestBytes];
  std::uint8_t control_config_sha256[kDigestBytes];
  std::uint8_t calibration_sha256[kDigestBytes];
  std::uint8_t producer_build_sha256[kDigestBytes];
  std::uint64_t created_monotonic_ns;
  std::uint64_t created_unix_ns;
  char producer_name[kProducerNameBytes];
  char robot_id[kRobotIdBytes];
};

// O_T_EE uses libfranka's 16-value column-major convention.  Payload sequence
// is kept in stream storage, not duplicated here.  bundle_sequence==0 means
// asynchronous/unbundled.  A non-zero value only denotes a coherent arm/hand
// pair when both streams publish the same value and the same stage identity.
struct alignas(64) ArmSample final {
  std::uint64_t timestamp_monotonic_ns;
  std::uint64_t timestamp_unix_ns;
  std::uint64_t producer_cycle;
  std::uint64_t stage_epoch;
  std::uint64_t bundle_sequence;
  std::uint64_t stage_name_hash64;
  std::uint32_t validity_flags;
  Source source;
  MeasurementKind measurement_kind;
  std::uint32_t robot_mode;
  std::uint32_t reserved_u32;
  char stage_name[kStageNameBytes];
  double O_T_EE[16];
  double q[7];
  double dq[7];
  double control_command_success_rate;
  std::uint64_t reserved[5];
};

// All RH56 arrays use the fixed order
// pinky, ring, middle, index, thumb_bend, thumb_rotate.
struct alignas(64) HandSample final {
  std::uint64_t timestamp_monotonic_ns;
  std::uint64_t timestamp_unix_ns;
  std::uint64_t producer_poll;
  std::uint64_t stage_epoch;
  std::uint64_t bundle_sequence;
  std::uint64_t stage_name_hash64;
  std::uint32_t validity_flags;
  Source source;
  MeasurementKind measurement_kind;
  std::uint32_t device_state;
  std::uint32_t reserved_u32;
  char stage_name[kStageNameBytes];
  std::int32_t angles[6];
  std::int32_t angle_targets[6];
  std::int32_t current_mA[6];
  std::int32_t force_g[6];
  std::int16_t temperature_c[6];
  std::uint8_t status[6];
  std::uint8_t errors[6];
  std::uint64_t reserved[5];
};

struct LayoutHeader final {
  std::uint64_t magic;
  std::uint32_t endian_marker;
  std::uint16_t abi_major;
  std::uint16_t abi_minor;
  std::uint32_t header_size;
  std::uint32_t total_size;
  std::uint32_t arm_sample_size;
  std::uint32_t hand_sample_size;
  std::uint32_t arm_stream_offset;
  std::uint32_t hand_stream_offset;
  std::uint8_t abi_schema_sha256[kDigestBytes];
  std::uint8_t run_uuid[kUuidBytes];
  std::uint8_t execution_contract_sha256[kDigestBytes];
  std::uint8_t source_snapshot_sha256[kDigestBytes];
  std::uint8_t control_config_sha256[kDigestBytes];
  std::uint8_t calibration_sha256[kDigestBytes];
  std::uint8_t producer_build_sha256[kDigestBytes];
  std::uint64_t created_monotonic_ns;
  std::uint64_t created_unix_ns;
  char producer_name[kProducerNameBytes];
  char robot_id[kRobotIdBytes];
  std::uint8_t reserved[40];
};

enum class InitCode : std::uint32_t {
  kOk = 0,
  kInvalidArgument = 1,
  kAlreadyExists = 2,
  kOpenFailed = 3,
  kTruncateFailed = 4,
  kStatFailed = 5,
  kMapFailed = 6,
  kLayoutNotReady = 7,
  kIncompatibleLayout = 8,
  kAtomicsNotLockFree = 9,
  kReadOnly = 10,
  kWriterBusy = 11,
  kAllocationFailed = 12,
};

struct InitError final {
  InitCode code;
  char message[192];
};

enum class PublishCode : std::uint32_t {
  kOk = 0,
  kNotClaimed = 1,
  kInvalidSample = 2,
  kSequenceExhausted = 3,
};

struct PublishResult final {
  PublishCode code;
  std::uint64_t sequence;
};

enum class ReadCode : std::uint32_t {
  kOk = 0,
  kNoData = 1,
  kContended = 2,
};

struct ReadResult final {
  ReadCode code;
  std::uint32_t attempts;
  std::uint32_t reserved;
  std::uint64_t sequence;
};

class ArmWriter;
class HandWriter;
class TelemetryReader;

// Mapping construction/destruction may allocate and issue syscalls.  No method
// named publish_* or read_* does so.
class TelemetryMapping final {
 public:
  static std::unique_ptr<TelemetryMapping> Create(
      const std::string& path, const SessionProvenance& provenance,
      InitError* error = nullptr);
  static std::unique_ptr<TelemetryMapping> OpenReadOnly(
      const std::string& path, InitError* error = nullptr);
  static std::unique_ptr<TelemetryMapping> OpenReadWrite(
      const std::string& path, InitError* error = nullptr);

  ~TelemetryMapping();
  TelemetryMapping(const TelemetryMapping&) = delete;
  TelemetryMapping& operator=(const TelemetryMapping&) = delete;

  const LayoutHeader& header() const noexcept;
  bool writable() const noexcept;
  static bool platform_is_supported_lock_free() noexcept;

  std::unique_ptr<ArmWriter> ClaimArmWriter(
      std::uint64_t writer_token, InitError* error = nullptr) noexcept;
  std::unique_ptr<HandWriter> ClaimHandWriter(
      std::uint64_t writer_token, InitError* error = nullptr) noexcept;
  TelemetryReader reader() const noexcept;

 private:
  static std::unique_ptr<TelemetryMapping> OpenImpl(
      const std::string& path, bool writable, InitError* error);
  TelemetryMapping(int fd, void* address, std::size_t size,
                   bool writable) noexcept;

  int fd_;
  void* address_;
  std::size_t size_;
  bool writable_;

  friend class ArmWriter;
  friend class HandWriter;
  friend class TelemetryReader;
};

class ArmWriter final {
 public:
  ~ArmWriter();
  ArmWriter(const ArmWriter&) = delete;
  ArmWriter& operator=(const ArmWriter&) = delete;

  PublishResult publish(const ArmSample& sample) noexcept;

 private:
  ArmWriter(void* stream, std::uint64_t writer_token,
            std::uint64_t next_sequence) noexcept;
  void* stream_;
  std::uint64_t writer_token_;
  std::uint64_t next_sequence_;
  friend class TelemetryMapping;
};

class HandWriter final {
 public:
  ~HandWriter();
  HandWriter(const HandWriter&) = delete;
  HandWriter& operator=(const HandWriter&) = delete;

  PublishResult publish(const HandSample& sample) noexcept;

 private:
  HandWriter(void* stream, std::uint64_t writer_token,
             std::uint64_t next_sequence) noexcept;
  void* stream_;
  std::uint64_t writer_token_;
  std::uint64_t next_sequence_;
  friend class TelemetryMapping;
};

class TelemetryReader final {
 public:
  ReadResult read_arm(ArmSample& output,
                      std::uint32_t max_attempts = kDefaultReadAttempts) const
      noexcept;
  ReadResult read_hand(HandSample& output,
                       std::uint32_t max_attempts = kDefaultReadAttempts) const
      noexcept;

 private:
  explicit TelemetryReader(const void* layout) noexcept;
  const void* layout_;
  friend class TelemetryMapping;
};

// FNV-1a over the complete, non-NUL UTF-8 byte sequence.
std::uint64_t stage_name_hash(const char* bytes, std::size_t size) noexcept;

// Fails rather than truncating.  output is always zero-padded on success.
bool set_stage_name(char (&output)[kStageNameBytes], const char* bytes,
                    std::size_t size) noexcept;

const char* source_name(Source source) noexcept;
const char* measurement_kind_name(MeasurementKind kind) noexcept;
const char* read_code_name(ReadCode code) noexcept;
const char* publish_code_name(PublishCode code) noexcept;

static_assert(sizeof(double) == 8, "wire ABI requires 64-bit double");
static_assert(std::is_standard_layout<SessionProvenance>::value &&
                  std::is_trivially_copyable<SessionProvenance>::value,
              "SessionProvenance must be fixed POD");
static_assert(std::is_standard_layout<ArmSample>::value &&
                  std::is_trivially_copyable<ArmSample>::value,
              "ArmSample must be fixed POD");
static_assert(std::is_standard_layout<HandSample>::value &&
                  std::is_trivially_copyable<HandSample>::value,
              "HandSample must be fixed POD");
static_assert(std::is_standard_layout<LayoutHeader>::value &&
                  std::is_trivially_copyable<LayoutHeader>::value,
              "LayoutHeader must be fixed POD");
static_assert(sizeof(SessionProvenance) == 272, "SessionProvenance ABI drift");
static_assert(sizeof(ArmSample) == 384, "ArmSample ABI drift");
static_assert(sizeof(HandSample) == 256, "HandSample ABI drift");
static_assert(sizeof(LayoutHeader) == 384, "LayoutHeader ABI drift");
static_assert(alignof(ArmSample) == 64, "ArmSample alignment drift");
static_assert(alignof(HandSample) == 64, "HandSample alignment drift");

}  // namespace anydex::telemetry
