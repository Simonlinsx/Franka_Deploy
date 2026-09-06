#include "anydex/telemetry/telemetry.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
std::atomic<std::uint64_t> g_allocations{0};
}

void* operator new(const std::size_t size) {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* pointer = std::malloc(size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](const std::size_t size) {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void* pointer = std::malloc(size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new(const std::size_t size, const std::nothrow_t&) noexcept {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(size);
}

void* operator new[](const std::size_t size, const std::nothrow_t&) noexcept {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(size);
}

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

namespace {

using anydex::telemetry::ArmSample;
using anydex::telemetry::HandSample;
using anydex::telemetry::InitCode;
using anydex::telemetry::InitError;
using anydex::telemetry::MeasurementKind;
using anydex::telemetry::PublishCode;
using anydex::telemetry::ReadCode;
using anydex::telemetry::SessionProvenance;
using anydex::telemetry::Source;
using anydex::telemetry::TelemetryMapping;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                                         \
      return false;                                                            \
    }                                                                          \
  } while (false)

struct TemporaryPath final {
  explicit TemporaryPath(const char* suffix) {
    static std::atomic<std::uint64_t> serial{1};
    path = "/tmp/anydex-telemetry-unit-" + std::to_string(::getpid()) + "-" +
           std::to_string(serial.fetch_add(1)) + "-" + suffix;
    ::unlink(path.c_str());
  }
  ~TemporaryPath() { ::unlink(path.c_str()); }
  std::string path;
};

SessionProvenance provenance() {
  SessionProvenance value{};
  for (std::size_t index = 0; index < anydex::telemetry::kUuidBytes; ++index) {
    value.run_uuid[index] = static_cast<std::uint8_t>(index + 1);
  }
  value.run_uuid[6] = 0x47;
  value.run_uuid[8] = 0x89;
  auto fill_digest = [](std::uint8_t (&digest)[32], std::uint8_t seed) {
    for (std::size_t index = 0; index < 32; ++index) {
      digest[index] = static_cast<std::uint8_t>(seed + index);
    }
  };
  fill_digest(value.execution_contract_sha256, 1);
  fill_digest(value.source_snapshot_sha256, 33);
  fill_digest(value.control_config_sha256, 65);
  fill_digest(value.calibration_sha256, 97);
  fill_digest(value.producer_build_sha256, 129);
  value.created_monotonic_ns = 100;
  value.created_unix_ns = 1'700'000'000'000'000'000ULL;
  std::memcpy(value.producer_name, "native-unit-test", 16);
  std::memcpy(value.robot_id, "offline-fixture", 15);
  return value;
}

ArmSample arm_sample(const std::uint64_t cycle) {
  ArmSample sample{};
  sample.timestamp_monotonic_ns = 1'000 + cycle;
  sample.timestamp_unix_ns = 1'700'000'000'000'000'000ULL + cycle;
  sample.producer_cycle = cycle;
  sample.stage_epoch = 7;
  sample.bundle_sequence = cycle;
  if (!anydex::telemetry::set_stage_name(sample.stage_name, "pregrasp", 8)) {
    std::abort();
  }
  sample.stage_name_hash64 =
      anydex::telemetry::stage_name_hash(sample.stage_name, 8);
  sample.validity_flags = anydex::telemetry::kArmTimestampValid |
                          anydex::telemetry::kArmPoseValid |
                          anydex::telemetry::kArmQValid |
                          anydex::telemetry::kArmDqValid |
                          anydex::telemetry::kArmSuccessRateValid;
  sample.source = Source::kSyntheticTest;
  sample.measurement_kind = MeasurementKind::kSyntheticTest;
  sample.robot_mode = 2;
  for (std::size_t index = 0; index < 16; ++index) {
    sample.O_T_EE[index] = static_cast<double>(cycle * 100 + index);
  }
  for (std::size_t index = 0; index < 7; ++index) {
    sample.q[index] = static_cast<double>(cycle * 10 + index);
    sample.dq[index] = -sample.q[index];
  }
  sample.control_command_success_rate = 0.99;
  return sample;
}

HandSample hand_sample(const std::uint64_t poll) {
  HandSample sample{};
  sample.timestamp_monotonic_ns = 2'000 + poll;
  sample.timestamp_unix_ns = 1'700'000'000'000'100'000ULL + poll;
  sample.producer_poll = poll;
  sample.stage_epoch = 7;
  sample.bundle_sequence = poll;
  if (!anydex::telemetry::set_stage_name(sample.stage_name, "pregrasp", 8)) {
    std::abort();
  }
  sample.stage_name_hash64 =
      anydex::telemetry::stage_name_hash(sample.stage_name, 8);
  sample.validity_flags = anydex::telemetry::kHandTimestampValid |
                          anydex::telemetry::kHandAnglesValid |
                          anydex::telemetry::kHandTargetsValid |
                          anydex::telemetry::kHandCurrentValid |
                          anydex::telemetry::kHandForceValid |
                          anydex::telemetry::kHandTemperatureValid |
                          anydex::telemetry::kHandStatusValid |
                          anydex::telemetry::kHandErrorsValid;
  sample.source = Source::kSyntheticTest;
  sample.measurement_kind = MeasurementKind::kSyntheticTest;
  sample.device_state = 4;
  for (std::size_t index = 0; index < 6; ++index) {
    sample.angles[index] = static_cast<std::int32_t>(poll * 10 + index);
    sample.angle_targets[index] = sample.angles[index] + 1;
    sample.current_mA[index] = static_cast<std::int32_t>(index + 10);
    sample.force_g[index] = static_cast<std::int32_t>(index + 20);
    sample.temperature_c[index] = static_cast<std::int16_t>(index + 30);
    sample.status[index] = static_cast<std::uint8_t>(index);
    sample.errors[index] = 0;
  }
  return sample;
}

bool test_hash_and_stage_name() {
  CHECK(anydex::telemetry::stage_name_hash("", 0) ==
        14695981039346656037ULL);
  CHECK(anydex::telemetry::stage_name_hash("pregrasp", 8) ==
        0x2b65809f1fe32bfdULL);
  char output[anydex::telemetry::kStageNameBytes]{};
  CHECK(anydex::telemetry::set_stage_name(output, "抓取", 6));
  CHECK(std::memcmp(output, "抓取", 6) == 0);
  char too_long[32];
  std::memset(too_long, 'x', sizeof(too_long));
  CHECK(!anydex::telemetry::set_stage_name(output, too_long,
                                           sizeof(too_long)));
  const char invalid_utf8[] = {static_cast<char>(0xc0),
                               static_cast<char>(0x80)};
  CHECK(!anydex::telemetry::set_stage_name(output, invalid_utf8, 2));
  return true;
}

bool test_create_publish_read_and_hot_path() {
  CHECK(TelemetryMapping::platform_is_supported_lock_free());
  TemporaryPath temporary("roundtrip");
  InitError error{};
  auto mapping = TelemetryMapping::Create(temporary.path, provenance(), &error);
  CHECK(mapping != nullptr);
  CHECK(error.code == InitCode::kOk);
  CHECK(mapping->writable());
  CHECK(mapping->header().abi_major == anydex::telemetry::kAbiMajor);
  CHECK(mapping->header().total_size == 2112);

  struct stat stat_buffer {};
  CHECK(::stat(temporary.path.c_str(), &stat_buffer) == 0);
  CHECK((stat_buffer.st_mode & 0777) == 0600);

  auto reader = mapping->reader();
  ArmSample arm_output{};
  HandSample hand_output{};
  CHECK(reader.read_arm(arm_output).code == ReadCode::kNoData);
  CHECK(reader.read_hand(hand_output).code == ReadCode::kNoData);

  auto read_only = TelemetryMapping::OpenReadOnly(temporary.path, &error);
  CHECK(read_only != nullptr);
  CHECK(!read_only->writable());
  CHECK(read_only->ClaimArmWriter(9, &error) == nullptr);
  CHECK(error.code == InitCode::kReadOnly);

  auto arm_writer = mapping->ClaimArmWriter(0xa11U, &error);
  CHECK(arm_writer != nullptr);
  CHECK(mapping->ClaimArmWriter(0xa12U, &error) == nullptr);
  CHECK(error.code == InitCode::kWriterBusy);
  auto hand_writer = mapping->ClaimHandWriter(0xb11U, &error);
  CHECK(hand_writer != nullptr);

  auto invalid = arm_sample(1);
  invalid.stage_name_hash64 ^= 1U;
  CHECK(arm_writer->publish(invalid).code == PublishCode::kInvalidSample);

  const auto first_arm = arm_sample(1);
  const auto arm_result = arm_writer->publish(first_arm);
  CHECK(arm_result.code == PublishCode::kOk);
  CHECK(arm_result.sequence == 1);
  const auto first_hand = hand_sample(1);
  const auto hand_result = hand_writer->publish(first_hand);
  CHECK(hand_result.code == PublishCode::kOk);
  CHECK(hand_result.sequence == 1);

  auto read_result = read_only->reader().read_arm(arm_output);
  CHECK(read_result.code == ReadCode::kOk);
  CHECK(read_result.sequence == 1);
  CHECK(std::memcmp(&arm_output, &first_arm, sizeof(first_arm)) == 0);
  read_result = read_only->reader().read_hand(hand_output);
  CHECK(read_result.code == ReadCode::kOk);
  CHECK(read_result.sequence == 1);
  CHECK(std::memcmp(&hand_output, &first_hand, sizeof(first_hand)) == 0);

  const std::uint64_t allocations_before =
      g_allocations.load(std::memory_order_relaxed);
  for (std::uint64_t cycle = 2; cycle <= 10'001; ++cycle) {
    const auto sample = arm_sample(cycle);
    CHECK(arm_writer->publish(sample).code == PublishCode::kOk);
    CHECK(reader.read_arm(arm_output).code == ReadCode::kOk);
    CHECK(arm_output.producer_cycle == cycle);
  }
  const std::uint64_t allocations_after =
      g_allocations.load(std::memory_order_relaxed);
  CHECK(allocations_after == allocations_before);

  // Stream sequences advance independently.
  CHECK(reader.read_arm(arm_output).sequence == 10'001);
  CHECK(reader.read_hand(hand_output).sequence == 1);

  arm_writer.reset();
  CHECK(mapping->ClaimArmWriter(0xa13U, &error) != nullptr);
  hand_writer.reset();
  read_only.reset();
  return true;
}

bool test_cross_process_reader() {
  TemporaryPath temporary("fork");
  InitError error{};
  auto mapping = TelemetryMapping::Create(temporary.path, provenance(), &error);
  CHECK(mapping != nullptr);
  auto writer = mapping->ClaimArmWriter(0xc11U, &error);
  CHECK(writer != nullptr);
  CHECK(writer->publish(arm_sample(42)).code == PublishCode::kOk);

  const pid_t child = ::fork();
  CHECK(child >= 0);
  if (child == 0) {
    ::execl("/proc/self/exe", "anydex_telemetry_unit", "--reader-child",
            temporary.path.c_str(), static_cast<char*>(nullptr));
    _exit(12);
  }
  int status = 0;
  CHECK(::waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 0);
  writer.reset();
  return true;
}

int reader_child(const char* path) {
  InitError error{};
  auto reader_mapping = TelemetryMapping::OpenReadOnly(path, &error);
  if (reader_mapping == nullptr) {
    return 10;
  }
  ArmSample output{};
  const auto result = reader_mapping->reader().read_arm(output, 64);
  return result.code == ReadCode::kOk && result.sequence == 1 &&
                 output.producer_cycle == 42
             ? 0
             : 11;
}

bool test_fail_closed_inputs() {
  TemporaryPath temporary("invalid");
  InitError error{};
  SessionProvenance invalid{};
  CHECK(TelemetryMapping::Create(temporary.path, invalid, &error) == nullptr);
  CHECK(error.code == InitCode::kInvalidArgument);

  auto mapping = TelemetryMapping::Create(temporary.path, provenance(), &error);
  CHECK(mapping != nullptr);
  CHECK(TelemetryMapping::Create(temporary.path, provenance(), &error) ==
        nullptr);
  CHECK(error.code == InitCode::kAlreadyExists);
  mapping.reset();

  const int fd = ::open(temporary.path.c_str(), O_RDWR | O_CLOEXEC);
  CHECK(fd >= 0);
  const std::uint8_t corrupt = 0xff;
  constexpr off_t kSchemaFirstByte = 64 + 40;
  CHECK(::pwrite(fd, &corrupt, 1, kSchemaFirstByte) == 1);
  CHECK(::close(fd) == 0);
  CHECK(TelemetryMapping::OpenReadOnly(temporary.path, &error) == nullptr);
  CHECK(error.code == InitCode::kIncompatibleLayout);
  return true;
}

}  // namespace

int main(const int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], "--reader-child") == 0) {
    return reader_child(argv[2]);
  }
  if (!test_hash_and_stage_name() ||
      !test_create_publish_read_and_hot_path() ||
      !test_cross_process_reader() || !test_fail_closed_inputs()) {
    return 1;
  }
  std::cout << "native telemetry unit tests passed; lock_free=true "
               "hot_path_allocations=0\n";
  return 0;
}
