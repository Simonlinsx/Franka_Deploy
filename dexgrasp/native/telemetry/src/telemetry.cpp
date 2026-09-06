#include "anydex/telemetry/telemetry.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <new>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace anydex::telemetry {
namespace {

constexpr std::size_t kCacheLine = 64;
constexpr std::size_t kArmWords = sizeof(ArmSample) / sizeof(std::uint64_t);
constexpr std::size_t kHandWords = sizeof(HandSample) / sizeof(std::uint64_t);
constexpr std::uint64_t kMaximumSequence =
    (std::numeric_limits<std::uint64_t>::max() - 1U) / 2U;

struct alignas(kCacheLine) Bootstrap final {
  std::atomic<std::uint64_t> state;
  std::atomic<std::uint64_t> generation;
  std::atomic<std::uint64_t> reserved[6];
};

struct alignas(kCacheLine) StreamControl final {
  std::atomic<std::uint64_t> published_sequence;
  std::atomic<std::uint64_t> writer_claim;
  std::atomic<std::uint64_t> publish_count;
  std::atomic<std::uint64_t> reserved[5];
};

template <std::size_t WordCount>
struct alignas(kCacheLine) AtomicSlot final {
  std::atomic<std::uint64_t> guard;
  std::atomic<std::uint64_t> words[WordCount];
};

struct alignas(kCacheLine) ArmStream final {
  StreamControl control;
  AtomicSlot<kArmWords> slots[2];
};

struct alignas(kCacheLine) HandStream final {
  StreamControl control;
  AtomicSlot<kHandWords> slots[2];
};

struct alignas(kCacheLine) TelemetryLayout final {
  Bootstrap bootstrap;
  LayoutHeader header;
  ArmStream arm;
  HandStream hand;
};

static_assert(sizeof(Bootstrap) == 64, "Bootstrap ABI drift");
static_assert(sizeof(StreamControl) == 64, "StreamControl ABI drift");
static_assert(sizeof(AtomicSlot<kArmWords>) == 448,
              "arm atomic slot ABI drift");
static_assert(sizeof(AtomicSlot<kHandWords>) == 320,
              "hand atomic slot ABI drift");
static_assert(sizeof(ArmStream) == 960, "arm stream ABI drift");
static_assert(sizeof(HandStream) == 704, "hand stream ABI drift");
static_assert(offsetof(TelemetryLayout, header) == 64,
              "header offset ABI drift");
static_assert(offsetof(TelemetryLayout, arm) == 448,
              "arm stream offset ABI drift");
static_assert(offsetof(TelemetryLayout, hand) == 1408,
              "hand stream offset ABI drift");
static_assert(sizeof(TelemetryLayout) == 2112, "layout ABI drift");
static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t),
              "uint64 atomic storage ABI is unsupported");
static_assert(alignof(std::atomic<std::uint64_t>) == alignof(std::uint64_t),
              "uint64 atomic alignment ABI is unsupported");

void set_error(InitError* error, const InitCode code,
               const char* message) noexcept {
  if (error == nullptr) {
    return;
  }
  error->code = code;
  std::snprintf(error->message, sizeof(error->message), "%s",
                message == nullptr ? "" : message);
}

void set_errno_error(InitError* error, const InitCode code,
                     const char* operation) noexcept {
  if (error == nullptr) {
    return;
  }
  error->code = code;
  std::snprintf(error->message, sizeof(error->message), "%s failed: errno=%d",
                operation, errno);
}

bool all_zero(const std::uint8_t* bytes, const std::size_t size) noexcept {
  std::uint8_t combined = 0;
  for (std::size_t index = 0; index < size; ++index) {
    combined = static_cast<std::uint8_t>(combined | bytes[index]);
  }
  return combined == 0;
}

bool uuid_valid(const std::uint8_t* bytes) noexcept {
  if (all_zero(bytes, kUuidBytes)) {
    return false;
  }
  const std::uint8_t version = static_cast<std::uint8_t>(bytes[6] >> 4U);
  return version >= 1U && version <= 8U && (bytes[8] & 0xc0U) == 0x80U;
}

bool fixed_string_valid(const char* bytes, const std::size_t capacity,
                        const bool require_nonempty) noexcept {
  std::size_t length = 0;
  while (length < capacity && bytes[length] != '\0') {
    ++length;
  }
  if (length == capacity || (require_nonempty && length == 0)) {
    return false;
  }
  for (std::size_t index = length + 1; index < capacity; ++index) {
    if (bytes[index] != '\0') {
      return false;
    }
  }
  return true;
}

// Strict enough for provenance/stage labels: rejects overlong sequences,
// surrogates, embedded NUL, and values above U+10FFFF without allocation.
bool valid_utf8(const char* bytes, const std::size_t size) noexcept {
  std::size_t index = 0;
  while (index < size) {
    const auto first = static_cast<std::uint8_t>(bytes[index]);
    if (first == 0) {
      return false;
    }
    if (first <= 0x7fU) {
      ++index;
      continue;
    }
    std::size_t continuation_count = 0;
    std::uint32_t value = 0;
    std::uint32_t minimum = 0;
    if ((first & 0xe0U) == 0xc0U) {
      continuation_count = 1;
      value = first & 0x1fU;
      minimum = 0x80U;
    } else if ((first & 0xf0U) == 0xe0U) {
      continuation_count = 2;
      value = first & 0x0fU;
      minimum = 0x800U;
    } else if ((first & 0xf8U) == 0xf0U) {
      continuation_count = 3;
      value = first & 0x07U;
      minimum = 0x10000U;
    } else {
      return false;
    }
    if (index + continuation_count >= size) {
      return false;
    }
    for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
      const auto next = static_cast<std::uint8_t>(bytes[index + offset]);
      if ((next & 0xc0U) != 0x80U) {
        return false;
      }
      value = (value << 6U) | (next & 0x3fU);
    }
    if (value < minimum || value > 0x10ffffU ||
        (value >= 0xd800U && value <= 0xdfffU)) {
      return false;
    }
    index += continuation_count + 1;
  }
  return true;
}

std::size_t fixed_string_length(const char* bytes,
                                const std::size_t capacity) noexcept {
  std::size_t length = 0;
  while (length < capacity && bytes[length] != '\0') {
    ++length;
  }
  return length;
}

bool provenance_valid(const SessionProvenance& provenance) noexcept {
  if (!uuid_valid(provenance.run_uuid) ||
      all_zero(provenance.execution_contract_sha256, kDigestBytes) ||
      all_zero(provenance.source_snapshot_sha256, kDigestBytes) ||
      all_zero(provenance.control_config_sha256, kDigestBytes) ||
      all_zero(provenance.calibration_sha256, kDigestBytes) ||
      all_zero(provenance.producer_build_sha256, kDigestBytes) ||
      provenance.created_monotonic_ns == 0 ||
      provenance.created_unix_ns == 0 ||
      !fixed_string_valid(provenance.producer_name, kProducerNameBytes, true) ||
      !fixed_string_valid(provenance.robot_id, kRobotIdBytes, true)) {
    return false;
  }
  const auto producer_size =
      fixed_string_length(provenance.producer_name, kProducerNameBytes);
  const auto robot_size =
      fixed_string_length(provenance.robot_id, kRobotIdBytes);
  return valid_utf8(provenance.producer_name, producer_size) &&
         valid_utf8(provenance.robot_id, robot_size);
}

template <std::size_t WordCount>
void initialize_slot(AtomicSlot<WordCount>& slot) noexcept {
  std::atomic_init(&slot.guard, std::uint64_t{0});
  for (auto& word : slot.words) {
    std::atomic_init(&word, std::uint64_t{0});
  }
}

void initialize_control(StreamControl& control) noexcept {
  std::atomic_init(&control.published_sequence, std::uint64_t{0});
  std::atomic_init(&control.writer_claim, std::uint64_t{0});
  std::atomic_init(&control.publish_count, std::uint64_t{0});
  for (auto& value : control.reserved) {
    std::atomic_init(&value, std::uint64_t{0});
  }
}

void initialize_layout(TelemetryLayout& layout,
                       const SessionProvenance& provenance) noexcept {
  std::atomic_init(&layout.bootstrap.state, std::uint64_t{0});
  std::atomic_init(&layout.bootstrap.generation, std::uint64_t{1});
  for (auto& value : layout.bootstrap.reserved) {
    std::atomic_init(&value, std::uint64_t{0});
  }
  initialize_control(layout.arm.control);
  initialize_slot(layout.arm.slots[0]);
  initialize_slot(layout.arm.slots[1]);
  initialize_control(layout.hand.control);
  initialize_slot(layout.hand.slots[0]);
  initialize_slot(layout.hand.slots[1]);

  LayoutHeader header{};
  header.magic = kLayoutMagic;
  header.endian_marker = kEndianMarker;
  header.abi_major = kAbiMajor;
  header.abi_minor = kAbiMinor;
  header.header_size = sizeof(LayoutHeader);
  header.total_size = sizeof(TelemetryLayout);
  header.arm_sample_size = sizeof(ArmSample);
  header.hand_sample_size = sizeof(HandSample);
  header.arm_stream_offset = offsetof(TelemetryLayout, arm);
  header.hand_stream_offset = offsetof(TelemetryLayout, hand);
  std::memcpy(header.abi_schema_sha256, kAbiSchemaSha256.data(),
              kDigestBytes);
  std::memcpy(header.run_uuid, provenance.run_uuid, kUuidBytes);
  std::memcpy(header.execution_contract_sha256,
              provenance.execution_contract_sha256, kDigestBytes);
  std::memcpy(header.source_snapshot_sha256,
              provenance.source_snapshot_sha256, kDigestBytes);
  std::memcpy(header.control_config_sha256,
              provenance.control_config_sha256, kDigestBytes);
  std::memcpy(header.calibration_sha256, provenance.calibration_sha256,
              kDigestBytes);
  std::memcpy(header.producer_build_sha256,
              provenance.producer_build_sha256, kDigestBytes);
  header.created_monotonic_ns = provenance.created_monotonic_ns;
  header.created_unix_ns = provenance.created_unix_ns;
  std::memcpy(header.producer_name, provenance.producer_name,
              kProducerNameBytes);
  std::memcpy(header.robot_id, provenance.robot_id, kRobotIdBytes);
  layout.header = header;
}

template <std::size_t WordCount>
bool slot_is_lock_free(const AtomicSlot<WordCount>& slot) noexcept {
  if (!slot.guard.is_lock_free()) {
    return false;
  }
  for (const auto& word : slot.words) {
    if (!word.is_lock_free()) {
      return false;
    }
  }
  return true;
}

bool control_is_lock_free(const StreamControl& control) noexcept {
  if (!control.published_sequence.is_lock_free() ||
      !control.writer_claim.is_lock_free() ||
      !control.publish_count.is_lock_free()) {
    return false;
  }
  for (const auto& value : control.reserved) {
    if (!value.is_lock_free()) {
      return false;
    }
  }
  return true;
}

bool layout_is_lock_free(const TelemetryLayout& layout) noexcept {
  if (!std::atomic<std::uint64_t>::is_always_lock_free ||
      !layout.bootstrap.state.is_lock_free() ||
      !layout.bootstrap.generation.is_lock_free()) {
    return false;
  }
  for (const auto& value : layout.bootstrap.reserved) {
    if (!value.is_lock_free()) {
      return false;
    }
  }
  return control_is_lock_free(layout.arm.control) &&
         slot_is_lock_free(layout.arm.slots[0]) &&
         slot_is_lock_free(layout.arm.slots[1]) &&
         control_is_lock_free(layout.hand.control) &&
         slot_is_lock_free(layout.hand.slots[0]) &&
         slot_is_lock_free(layout.hand.slots[1]);
}

bool host_is_little_endian() noexcept {
  const std::uint32_t value = 1;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

bool header_compatible(const LayoutHeader& header) noexcept {
  const auto producer_size =
      fixed_string_length(header.producer_name, kProducerNameBytes);
  const auto robot_size = fixed_string_length(header.robot_id, kRobotIdBytes);
  return header.magic == kLayoutMagic &&
         header.endian_marker == kEndianMarker &&
         header.abi_major == kAbiMajor && header.abi_minor == kAbiMinor &&
         header.header_size == sizeof(LayoutHeader) &&
         header.total_size == sizeof(TelemetryLayout) &&
         header.arm_sample_size == sizeof(ArmSample) &&
         header.hand_sample_size == sizeof(HandSample) &&
         header.arm_stream_offset == offsetof(TelemetryLayout, arm) &&
         header.hand_stream_offset == offsetof(TelemetryLayout, hand) &&
         std::memcmp(header.abi_schema_sha256, kAbiSchemaSha256.data(),
                     kDigestBytes) == 0 &&
         uuid_valid(header.run_uuid) &&
         !all_zero(header.execution_contract_sha256, kDigestBytes) &&
         !all_zero(header.source_snapshot_sha256, kDigestBytes) &&
         !all_zero(header.control_config_sha256, kDigestBytes) &&
         !all_zero(header.calibration_sha256, kDigestBytes) &&
         !all_zero(header.producer_build_sha256, kDigestBytes) &&
         header.created_monotonic_ns != 0 && header.created_unix_ns != 0 &&
         fixed_string_valid(header.producer_name, kProducerNameBytes, true) &&
         fixed_string_valid(header.robot_id, kRobotIdBytes, true) &&
         valid_utf8(header.producer_name, producer_size) &&
         valid_utf8(header.robot_id, robot_size);
}

template <typename Sample>
bool common_sample_valid(const Sample& sample) noexcept {
  constexpr std::uint32_t kTimestampFlag = 1U;
  if ((sample.validity_flags & kTimestampFlag) == 0 ||
      sample.timestamp_monotonic_ns == 0 || sample.timestamp_unix_ns == 0 ||
      sample.stage_epoch == 0 ||
      !fixed_string_valid(sample.stage_name, kStageNameBytes, true)) {
    return false;
  }
  const auto stage_size = fixed_string_length(sample.stage_name, kStageNameBytes);
  return valid_utf8(sample.stage_name, stage_size) &&
         sample.stage_name_hash64 ==
             stage_name_hash(sample.stage_name, stage_size);
}

bool arm_sample_valid(const ArmSample& sample) noexcept {
  constexpr std::uint32_t kKnown =
      kArmTimestampValid | kArmPoseValid | kArmQValid | kArmDqValid |
      kArmSuccessRateValid;
  if (!common_sample_valid(sample) ||
      (sample.validity_flags & ~kKnown) != 0 ||
      (sample.validity_flags & kArmPoseValid) == 0) {
    return false;
  }
  const bool measured = sample.source == Source::kFrankaRobotStateOTEE &&
                        sample.measurement_kind == MeasurementKind::kMeasured;
  const bool synthetic = sample.source == Source::kSyntheticTest &&
                         sample.measurement_kind ==
                             MeasurementKind::kSyntheticTest;
  if (!measured && !synthetic) {
    return false;
  }
  for (const double value : sample.O_T_EE) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  if ((sample.validity_flags & kArmQValid) != 0) {
    for (const double value : sample.q) {
      if (!std::isfinite(value)) {
        return false;
      }
    }
  }
  if ((sample.validity_flags & kArmDqValid) != 0) {
    for (const double value : sample.dq) {
      if (!std::isfinite(value)) {
        return false;
      }
    }
  }
  return (sample.validity_flags & kArmSuccessRateValid) == 0 ||
         (std::isfinite(sample.control_command_success_rate) &&
          sample.control_command_success_rate >= 0.0 &&
          sample.control_command_success_rate <= 1.0);
}

bool hand_sample_valid(const HandSample& sample) noexcept {
  constexpr std::uint32_t kKnown =
      kHandTimestampValid | kHandAnglesValid | kHandTargetsValid |
      kHandCurrentValid | kHandForceValid | kHandTemperatureValid |
      kHandStatusValid | kHandErrorsValid;
  if (!common_sample_valid(sample) ||
      (sample.validity_flags & ~kKnown) != 0 ||
      (sample.validity_flags & kHandAnglesValid) == 0) {
    return false;
  }
  const bool measured = sample.source == Source::kInspireRh56AngleAct &&
                        sample.measurement_kind == MeasurementKind::kMeasured;
  const bool synthetic = sample.source == Source::kSyntheticTest &&
                         sample.measurement_kind ==
                             MeasurementKind::kSyntheticTest;
  return measured || synthetic;
}

template <typename Sample, std::size_t WordCount, typename Stream>
PublishResult publish_sample(Stream& stream, std::uint64_t& next_sequence,
                             const Sample& sample,
                             bool (*validate)(const Sample&)) noexcept {
  if (!validate(sample)) {
    return {PublishCode::kInvalidSample, 0};
  }
  if (next_sequence == 0 || next_sequence > kMaximumSequence) {
    return {PublishCode::kSequenceExhausted, 0};
  }

  std::array<std::uint64_t, WordCount> encoded{};
  std::memcpy(encoded.data(), &sample, sizeof(sample));
  const std::uint64_t sequence = next_sequence;
  auto& slot = stream.slots[sequence & 1U];
  const std::uint64_t stable_guard = sequence << 1U;

  // exchange is a lock-free RMW on every supported platform checked at init.
  // It prevents any payload store from becoming visible before the odd guard.
  slot.guard.exchange(stable_guard | 1U, std::memory_order_seq_cst);
  for (std::size_t index = 0; index < WordCount; ++index) {
    slot.words[index].store(encoded[index], std::memory_order_relaxed);
  }
  slot.guard.store(stable_guard, std::memory_order_release);
  stream.control.published_sequence.store(sequence, std::memory_order_release);
  stream.control.publish_count.fetch_add(1U, std::memory_order_relaxed);
  next_sequence = sequence + 1U;
  return {PublishCode::kOk, sequence};
}

template <typename Sample, std::size_t WordCount, typename Stream>
ReadResult read_sample(const Stream& stream, Sample& output,
                       const std::uint32_t max_attempts,
                       bool (*validate)(const Sample&)) noexcept {
  if (max_attempts == 0) {
    return {ReadCode::kContended, 0, 0, 0};
  }
  std::array<std::uint64_t, WordCount> encoded{};
  for (std::uint32_t attempt = 1; attempt <= max_attempts; ++attempt) {
    const std::uint64_t sequence =
        stream.control.published_sequence.load(std::memory_order_acquire);
    if (sequence == 0) {
      return {ReadCode::kNoData, attempt, 0, 0};
    }
    const auto& slot = stream.slots[sequence & 1U];
    const std::uint64_t expected_guard = sequence << 1U;
    const std::uint64_t before = slot.guard.load(std::memory_order_acquire);
    if (before != expected_guard) {
      continue;
    }
    for (std::size_t index = 0; index < WordCount; ++index) {
      encoded[index] = slot.words[index].load(std::memory_order_relaxed);
    }
    // Keep all payload loads on the read side of the second guard sample.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const std::uint64_t after = slot.guard.load(std::memory_order_acquire);
    const std::uint64_t latest =
        stream.control.published_sequence.load(std::memory_order_acquire);
    if (before == after && after == expected_guard && latest == sequence) {
      Sample decoded{};
      std::memcpy(&decoded, encoded.data(), sizeof(decoded));
      if (!validate(decoded)) {
        continue;
      }
      output = decoded;
      return {ReadCode::kOk, attempt, 0, sequence};
    }
  }
  return {ReadCode::kContended, max_attempts, 0, 0};
}

template <typename Stream>
bool claim_stream(Stream& stream, const std::uint64_t token,
                  InitError* error) noexcept {
  if (token == 0) {
    set_error(error, InitCode::kInvalidArgument,
              "writer token must be non-zero");
    return false;
  }
  std::uint64_t expected = 0;
  if (!stream.control.writer_claim.compare_exchange_strong(
          expected, token, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    set_error(error, InitCode::kWriterBusy,
              "stream already has a claimed writer");
    return false;
  }
  return true;
}

template <typename Stream>
void release_stream(Stream& stream, const std::uint64_t token) noexcept {
  std::uint64_t expected = token;
  stream.control.writer_claim.compare_exchange_strong(
      expected, 0, std::memory_order_release, std::memory_order_relaxed);
}

}  // namespace

std::unique_ptr<TelemetryMapping> TelemetryMapping::OpenImpl(
    const std::string& path, const bool writable, InitError* error) {
  set_error(error, InitCode::kOk, "");
  if (path.empty() || !host_is_little_endian()) {
    set_error(error, InitCode::kInvalidArgument,
              "path must be non-empty and host must be little-endian");
    return nullptr;
  }
  const int flags = (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW;
  const int fd = ::open(path.c_str(), flags);
  if (fd < 0) {
    set_errno_error(error, InitCode::kOpenFailed, "open");
    return nullptr;
  }
  struct stat stat_buffer {};
  if (::fstat(fd, &stat_buffer) != 0) {
    set_errno_error(error, InitCode::kStatFailed, "fstat");
    ::close(fd);
    return nullptr;
  }
  if (!S_ISREG(stat_buffer.st_mode) ||
      stat_buffer.st_size != static_cast<off_t>(sizeof(TelemetryLayout))) {
    set_error(error, InitCode::kIncompatibleLayout,
              "mapping is not a regular file of the exact ABI size");
    ::close(fd);
    return nullptr;
  }
  const int protection = PROT_READ | (writable ? PROT_WRITE : 0);
  void* address = ::mmap(nullptr, sizeof(TelemetryLayout), protection,
                         MAP_SHARED, fd, 0);
  if (address == MAP_FAILED) {
    set_errno_error(error, InitCode::kMapFailed, "mmap");
    ::close(fd);
    return nullptr;
  }
  const auto* layout = static_cast<const TelemetryLayout*>(address);
  if (layout->bootstrap.state.load(std::memory_order_acquire) != kLayoutReady) {
    set_error(error, InitCode::kLayoutNotReady,
              "telemetry mapping initialization is not committed");
    ::munmap(address, sizeof(TelemetryLayout));
    ::close(fd);
    return nullptr;
  }
  if (!header_compatible(layout->header)) {
    set_error(error, InitCode::kIncompatibleLayout,
              "telemetry ABI header or provenance is incompatible");
    ::munmap(address, sizeof(TelemetryLayout));
    ::close(fd);
    return nullptr;
  }
  if (!layout_is_lock_free(*layout)) {
    set_error(error, InitCode::kAtomicsNotLockFree,
              "every shared uint64 atomic must be lock-free");
    ::munmap(address, sizeof(TelemetryLayout));
    ::close(fd);
    return nullptr;
  }
  auto* raw = new (std::nothrow)
      TelemetryMapping(fd, address, sizeof(TelemetryLayout), writable);
  if (raw == nullptr) {
    set_error(error, InitCode::kAllocationFailed,
              "failed to allocate mapping owner");
    ::munmap(address, sizeof(TelemetryLayout));
    ::close(fd);
    return nullptr;
  }
  return std::unique_ptr<TelemetryMapping>(raw);
}

std::uint64_t stage_name_hash(const char* bytes,
                              const std::size_t size) noexcept {
  constexpr std::uint64_t kOffsetBasis = 14695981039346656037ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  if (bytes == nullptr && size != 0) {
    return 0;
  }
  std::uint64_t hash = kOffsetBasis;
  for (std::size_t index = 0; index < size; ++index) {
    hash ^= static_cast<std::uint8_t>(bytes[index]);
    hash *= kPrime;
  }
  return hash;
}

bool set_stage_name(char (&output)[kStageNameBytes], const char* bytes,
                    const std::size_t size) noexcept {
  if (bytes == nullptr || size == 0 || size >= kStageNameBytes ||
      !valid_utf8(bytes, size)) {
    return false;
  }
  std::memset(output, 0, kStageNameBytes);
  std::memcpy(output, bytes, size);
  return true;
}

const char* source_name(const Source source) noexcept {
  switch (source) {
    case Source::kFrankaRobotStateOTEE:
      return "franka_robot_state.O_T_EE";
    case Source::kInspireRh56AngleAct:
      return "inspire_rh56.ANGLE_ACT";
    case Source::kSyntheticTest:
      return "synthetic_test";
    case Source::kUnknown:
    default:
      return "unknown";
  }
}

const char* measurement_kind_name(const MeasurementKind kind) noexcept {
  switch (kind) {
    case MeasurementKind::kMeasured:
      return "measured";
    case MeasurementKind::kCommanded:
      return "commanded";
    case MeasurementKind::kDerived:
      return "derived";
    case MeasurementKind::kSyntheticTest:
      return "synthetic_test";
    case MeasurementKind::kUnknown:
    default:
      return "unknown";
  }
}

const char* read_code_name(const ReadCode code) noexcept {
  switch (code) {
    case ReadCode::kOk:
      return "ok";
    case ReadCode::kNoData:
      return "no_data";
    case ReadCode::kContended:
      return "contended";
    default:
      return "unknown";
  }
}

const char* publish_code_name(const PublishCode code) noexcept {
  switch (code) {
    case PublishCode::kOk:
      return "ok";
    case PublishCode::kNotClaimed:
      return "not_claimed";
    case PublishCode::kInvalidSample:
      return "invalid_sample";
    case PublishCode::kSequenceExhausted:
      return "sequence_exhausted";
    default:
      return "unknown";
  }
}

TelemetryMapping::TelemetryMapping(const int fd, void* const address,
                                   const std::size_t size,
                                   const bool writable) noexcept
    : fd_(fd), address_(address), size_(size), writable_(writable) {}

TelemetryMapping::~TelemetryMapping() {
  if (address_ != nullptr && address_ != MAP_FAILED) {
    ::munmap(address_, size_);
  }
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

std::unique_ptr<TelemetryMapping> TelemetryMapping::Create(
    const std::string& path, const SessionProvenance& provenance,
    InitError* error) {
  set_error(error, InitCode::kOk, "");
  if (path.empty() || !host_is_little_endian() ||
      !provenance_valid(provenance)) {
    set_error(error, InitCode::kInvalidArgument,
              "invalid path, byte order, UUID, SHA-256 provenance, timestamp, "
              "or fixed UTF-8 identifier");
    return nullptr;
  }
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC |
                                          O_NOFOLLOW,
                        S_IRUSR | S_IWUSR);
  if (fd < 0) {
    set_errno_error(error,
                    errno == EEXIST ? InitCode::kAlreadyExists
                                    : InitCode::kOpenFailed,
                    "create");
    return nullptr;
  }
  if (::ftruncate(fd, sizeof(TelemetryLayout)) != 0) {
    set_errno_error(error, InitCode::kTruncateFailed, "ftruncate");
    ::close(fd);
    ::unlink(path.c_str());
    return nullptr;
  }
  void* address = ::mmap(nullptr, sizeof(TelemetryLayout), PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
  if (address == MAP_FAILED) {
    set_errno_error(error, InitCode::kMapFailed, "mmap");
    ::close(fd);
    ::unlink(path.c_str());
    return nullptr;
  }
  std::memset(address, 0, sizeof(TelemetryLayout));
  auto* layout = ::new (address) TelemetryLayout;
  initialize_layout(*layout, provenance);
  if (!layout_is_lock_free(*layout)) {
    set_error(error, InitCode::kAtomicsNotLockFree,
              "every shared uint64 atomic must be lock-free");
    ::munmap(address, sizeof(TelemetryLayout));
    ::close(fd);
    ::unlink(path.c_str());
    return nullptr;
  }
  layout->bootstrap.state.store(kLayoutReady, std::memory_order_release);
  auto* raw = new (std::nothrow)
      TelemetryMapping(fd, address, sizeof(TelemetryLayout), true);
  if (raw == nullptr) {
    set_error(error, InitCode::kAllocationFailed,
              "failed to allocate mapping owner");
    ::munmap(address, sizeof(TelemetryLayout));
    ::close(fd);
    ::unlink(path.c_str());
    return nullptr;
  }
  return std::unique_ptr<TelemetryMapping>(raw);
}

std::unique_ptr<TelemetryMapping> TelemetryMapping::OpenReadOnly(
    const std::string& path, InitError* error) {
  return OpenImpl(path, false, error);
}

std::unique_ptr<TelemetryMapping> TelemetryMapping::OpenReadWrite(
    const std::string& path, InitError* error) {
  return OpenImpl(path, true, error);
}

const LayoutHeader& TelemetryMapping::header() const noexcept {
  return static_cast<const TelemetryLayout*>(address_)->header;
}

bool TelemetryMapping::writable() const noexcept { return writable_; }

bool TelemetryMapping::platform_is_supported_lock_free() noexcept {
  return host_is_little_endian() &&
         std::atomic<std::uint64_t>::is_always_lock_free &&
         sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t) &&
         alignof(std::atomic<std::uint64_t>) == alignof(std::uint64_t);
}

std::unique_ptr<ArmWriter> TelemetryMapping::ClaimArmWriter(
    const std::uint64_t writer_token, InitError* error) noexcept {
  set_error(error, InitCode::kOk, "");
  if (!writable_) {
    set_error(error, InitCode::kReadOnly,
              "cannot claim writer through a read-only mapping");
    return nullptr;
  }
  auto& stream = static_cast<TelemetryLayout*>(address_)->arm;
  if (!claim_stream(stream, writer_token, error)) {
    return nullptr;
  }
  const auto next =
      stream.control.published_sequence.load(std::memory_order_acquire) + 1U;
  auto* raw = new (std::nothrow) ArmWriter(&stream, writer_token, next);
  if (raw == nullptr) {
    release_stream(stream, writer_token);
    set_error(error, InitCode::kAllocationFailed,
              "failed to allocate arm writer owner");
    return nullptr;
  }
  return std::unique_ptr<ArmWriter>(raw);
}

std::unique_ptr<HandWriter> TelemetryMapping::ClaimHandWriter(
    const std::uint64_t writer_token, InitError* error) noexcept {
  set_error(error, InitCode::kOk, "");
  if (!writable_) {
    set_error(error, InitCode::kReadOnly,
              "cannot claim writer through a read-only mapping");
    return nullptr;
  }
  auto& stream = static_cast<TelemetryLayout*>(address_)->hand;
  if (!claim_stream(stream, writer_token, error)) {
    return nullptr;
  }
  const auto next =
      stream.control.published_sequence.load(std::memory_order_acquire) + 1U;
  auto* raw = new (std::nothrow) HandWriter(&stream, writer_token, next);
  if (raw == nullptr) {
    release_stream(stream, writer_token);
    set_error(error, InitCode::kAllocationFailed,
              "failed to allocate hand writer owner");
    return nullptr;
  }
  return std::unique_ptr<HandWriter>(raw);
}

TelemetryReader TelemetryMapping::reader() const noexcept {
  return TelemetryReader(address_);
}

ArmWriter::ArmWriter(void* const stream, const std::uint64_t writer_token,
                     const std::uint64_t next_sequence) noexcept
    : stream_(stream),
      writer_token_(writer_token),
      next_sequence_(next_sequence) {}

ArmWriter::~ArmWriter() {
  if (stream_ != nullptr && writer_token_ != 0) {
    release_stream(*static_cast<ArmStream*>(stream_), writer_token_);
  }
}

PublishResult ArmWriter::publish(const ArmSample& sample) noexcept {
  if (stream_ == nullptr || writer_token_ == 0) {
    return {PublishCode::kNotClaimed, 0};
  }
  return publish_sample<ArmSample, kArmWords>(
      *static_cast<ArmStream*>(stream_), next_sequence_, sample,
      arm_sample_valid);
}

HandWriter::HandWriter(void* const stream, const std::uint64_t writer_token,
                       const std::uint64_t next_sequence) noexcept
    : stream_(stream),
      writer_token_(writer_token),
      next_sequence_(next_sequence) {}

HandWriter::~HandWriter() {
  if (stream_ != nullptr && writer_token_ != 0) {
    release_stream(*static_cast<HandStream*>(stream_), writer_token_);
  }
}

PublishResult HandWriter::publish(const HandSample& sample) noexcept {
  if (stream_ == nullptr || writer_token_ == 0) {
    return {PublishCode::kNotClaimed, 0};
  }
  return publish_sample<HandSample, kHandWords>(
      *static_cast<HandStream*>(stream_), next_sequence_, sample,
      hand_sample_valid);
}

TelemetryReader::TelemetryReader(const void* const layout) noexcept
    : layout_(layout) {}

ReadResult TelemetryReader::read_arm(ArmSample& output,
                                     const std::uint32_t max_attempts) const
    noexcept {
  return read_sample<ArmSample, kArmWords>(
      static_cast<const TelemetryLayout*>(layout_)->arm, output, max_attempts,
      arm_sample_valid);
}

ReadResult TelemetryReader::read_hand(HandSample& output,
                                      const std::uint32_t max_attempts) const
    noexcept {
  return read_sample<HandSample, kHandWords>(
      static_cast<const TelemetryLayout*>(layout_)->hand, output, max_attempts,
      hand_sample_valid);
}

}  // namespace anydex::telemetry
