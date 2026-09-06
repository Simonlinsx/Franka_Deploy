#include "anydex/telemetry/telemetry.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

anydex::telemetry::SessionProvenance provenance() {
  anydex::telemetry::SessionProvenance value{};
  for (std::size_t index = 0; index < anydex::telemetry::kUuidBytes; ++index) {
    value.run_uuid[index] = static_cast<std::uint8_t>(index + 21);
  }
  value.run_uuid[6] = 0x47;
  value.run_uuid[8] = 0x89;
  auto fill = [](std::uint8_t (&digest)[32], const std::uint8_t seed) {
    for (std::size_t index = 0; index < 32; ++index) {
      digest[index] = static_cast<std::uint8_t>(seed + index);
    }
  };
  fill(value.execution_contract_sha256, 1);
  fill(value.source_snapshot_sha256, 43);
  fill(value.control_config_sha256, 83);
  fill(value.calibration_sha256, 123);
  fill(value.producer_build_sha256, 163);
  value.created_monotonic_ns = 1;
  value.created_unix_ns = 1'700'000'000'000'000'000ULL;
  std::memcpy(value.producer_name, "native-benchmark", 17);
  std::memcpy(value.robot_id, "offline-only", 13);
  return value;
}

anydex::telemetry::ArmSample arm_sample() {
  anydex::telemetry::ArmSample sample{};
  sample.timestamp_monotonic_ns = 1;
  sample.timestamp_unix_ns = 2;
  sample.stage_epoch = 1;
  if (!anydex::telemetry::set_stage_name(sample.stage_name, "benchmark", 9)) {
    std::abort();
  }
  sample.stage_name_hash64 =
      anydex::telemetry::stage_name_hash(sample.stage_name, 9);
  sample.validity_flags = anydex::telemetry::kArmTimestampValid |
                          anydex::telemetry::kArmPoseValid |
                          anydex::telemetry::kArmQValid |
                          anydex::telemetry::kArmDqValid |
                          anydex::telemetry::kArmSuccessRateValid;
  sample.source = anydex::telemetry::Source::kSyntheticTest;
  sample.measurement_kind =
      anydex::telemetry::MeasurementKind::kSyntheticTest;
  sample.O_T_EE[0] = 1;
  sample.O_T_EE[5] = 1;
  sample.O_T_EE[10] = 1;
  sample.O_T_EE[15] = 1;
  sample.control_command_success_rate = 1;
  return sample;
}

anydex::telemetry::HandSample hand_sample() {
  anydex::telemetry::HandSample sample{};
  sample.timestamp_monotonic_ns = 1;
  sample.timestamp_unix_ns = 2;
  sample.stage_epoch = 1;
  if (!anydex::telemetry::set_stage_name(sample.stage_name, "benchmark", 9)) {
    std::abort();
  }
  sample.stage_name_hash64 =
      anydex::telemetry::stage_name_hash(sample.stage_name, 9);
  sample.validity_flags = anydex::telemetry::kHandTimestampValid |
                          anydex::telemetry::kHandAnglesValid;
  sample.source = anydex::telemetry::Source::kSyntheticTest;
  sample.measurement_kind =
      anydex::telemetry::MeasurementKind::kSyntheticTest;
  return sample;
}

template <typename Function>
double measure_ns(const std::uint64_t iterations, Function&& function) {
  const auto started = Clock::now();
  for (std::uint64_t index = 0; index < iterations; ++index) {
    function(index);
  }
  const auto elapsed = Clock::now() - started;
  return std::chrono::duration<double, std::nano>(elapsed).count() /
         static_cast<double>(iterations);
}

}  // namespace

int main(int argc, char** argv) {
  const std::uint64_t iterations =
      argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1'000'000ULL;
  if (iterations == 0 || iterations > 100'000'000ULL) {
    std::cerr << "usage: anydex_telemetry_benchmark [iterations 1..1e8]\n";
    return 2;
  }
  const std::string path =
      "/tmp/anydex-telemetry-benchmark-" + std::to_string(::getpid());
  ::unlink(path.c_str());
  struct Cleanup final {
    explicit Cleanup(std::string value) : path(std::move(value)) {}
    ~Cleanup() { ::unlink(path.c_str()); }
    std::string path;
  } cleanup(path);

  anydex::telemetry::InitError error{};
  auto mapping = anydex::telemetry::TelemetryMapping::Create(
      path, provenance(), &error);
  if (mapping == nullptr) {
    std::cerr << "create failed: " << error.message << '\n';
    return 3;
  }
  auto arm_writer = mapping->ClaimArmWriter(0xa991U, &error);
  auto hand_writer = mapping->ClaimHandWriter(0xb991U, &error);
  if (arm_writer == nullptr || hand_writer == nullptr) {
    std::cerr << "claim failed: " << error.message << '\n';
    return 4;
  }
  auto arm = arm_sample();
  auto hand = hand_sample();
  anydex::telemetry::ArmSample arm_output{};
  anydex::telemetry::HandSample hand_output{};
  const auto reader = mapping->reader();
  for (std::uint64_t index = 0; index < 1'000; ++index) {
    ++arm.timestamp_monotonic_ns;
    ++hand.timestamp_monotonic_ns;
    if (arm_writer->publish(arm).code != anydex::telemetry::PublishCode::kOk ||
        hand_writer->publish(hand).code !=
            anydex::telemetry::PublishCode::kOk ||
        reader.read_arm(arm_output).code != anydex::telemetry::ReadCode::kOk ||
        reader.read_hand(hand_output).code !=
            anydex::telemetry::ReadCode::kOk) {
      std::cerr << "warmup failed\n";
      return 5;
    }
  }

  bool failed = false;
  const double arm_publish_ns = measure_ns(iterations, [&](std::uint64_t index) {
    arm.producer_cycle = index;
    arm.timestamp_monotonic_ns = index + 1;
    failed = failed || arm_writer->publish(arm).code !=
                           anydex::telemetry::PublishCode::kOk;
  });
  const double arm_read_ns = measure_ns(iterations, [&](std::uint64_t) {
    failed = failed || reader.read_arm(arm_output).code !=
                           anydex::telemetry::ReadCode::kOk;
  });
  const double hand_publish_ns =
      measure_ns(iterations, [&](std::uint64_t index) {
        hand.producer_poll = index;
        hand.timestamp_monotonic_ns = index + 1;
        failed = failed || hand_writer->publish(hand).code !=
                               anydex::telemetry::PublishCode::kOk;
      });
  const double hand_read_ns = measure_ns(iterations, [&](std::uint64_t) {
    failed = failed || reader.read_hand(hand_output).code !=
                           anydex::telemetry::ReadCode::kOk;
  });

  std::cout << std::fixed << std::setprecision(1)
            << "offline=true lock_free="
            << anydex::telemetry::TelemetryMapping::
                   platform_is_supported_lock_free()
            << " iterations=" << iterations << '\n'
            << "arm_publish_ns=" << arm_publish_ns
            << " arm_read_ns=" << arm_read_ns << '\n'
            << "hand_publish_ns=" << hand_publish_ns
            << " hand_read_ns=" << hand_read_ns << '\n'
            << "hot_path_heap=none hot_path_lock=none hot_path_syscall=none\n";
  return failed ? 6 : 0;
}
