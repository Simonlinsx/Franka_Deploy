#include "anydex/telemetry/telemetry.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using anydex::telemetry::ArmSample;
using anydex::telemetry::HandSample;
using anydex::telemetry::InitError;
using anydex::telemetry::MeasurementKind;
using anydex::telemetry::PublishCode;
using anydex::telemetry::ReadCode;
using anydex::telemetry::SessionProvenance;
using anydex::telemetry::Source;
using anydex::telemetry::TelemetryMapping;

SessionProvenance provenance() {
  SessionProvenance value{};
  for (std::size_t index = 0; index < anydex::telemetry::kUuidBytes; ++index) {
    value.run_uuid[index] = static_cast<std::uint8_t>(index + 11);
  }
  value.run_uuid[6] = 0x47;
  value.run_uuid[8] = 0x89;
  auto fill = [](std::uint8_t (&digest)[32], const std::uint8_t seed) {
    for (std::size_t index = 0; index < 32; ++index) {
      digest[index] = static_cast<std::uint8_t>(seed + index);
    }
  };
  fill(value.execution_contract_sha256, 1);
  fill(value.source_snapshot_sha256, 41);
  fill(value.control_config_sha256, 81);
  fill(value.calibration_sha256, 121);
  fill(value.producer_build_sha256, 161);
  value.created_monotonic_ns = 1;
  value.created_unix_ns = 1'700'000'000'000'000'000ULL;
  std::memcpy(value.producer_name, "native-stress", 14);
  std::memcpy(value.robot_id, "offline-only", 13);
  return value;
}

ArmSample base_arm() {
  ArmSample sample{};
  sample.stage_epoch = 1;
  if (!anydex::telemetry::set_stage_name(sample.stage_name, "stress", 6)) {
    std::abort();
  }
  sample.stage_name_hash64 =
      anydex::telemetry::stage_name_hash(sample.stage_name, 6);
  sample.validity_flags = anydex::telemetry::kArmTimestampValid |
                          anydex::telemetry::kArmPoseValid |
                          anydex::telemetry::kArmQValid |
                          anydex::telemetry::kArmDqValid |
                          anydex::telemetry::kArmSuccessRateValid;
  sample.source = Source::kSyntheticTest;
  sample.measurement_kind = MeasurementKind::kSyntheticTest;
  sample.control_command_success_rate = 1.0;
  return sample;
}

HandSample base_hand() {
  HandSample sample{};
  sample.stage_epoch = 1;
  if (!anydex::telemetry::set_stage_name(sample.stage_name, "stress", 6)) {
    std::abort();
  }
  sample.stage_name_hash64 =
      anydex::telemetry::stage_name_hash(sample.stage_name, 6);
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
  return sample;
}

bool valid_arm(const ArmSample& sample, const std::uint64_t sequence) {
  if (sample.producer_cycle != sequence ||
      sample.timestamp_monotonic_ns != sequence + 100 ||
      sample.timestamp_unix_ns != sequence + 1'000 ||
      sample.bundle_sequence != sequence) {
    return false;
  }
  for (std::size_t index = 0; index < 16; ++index) {
    if (sample.O_T_EE[index] !=
        static_cast<double>(sequence * 100 + index)) {
      return false;
    }
  }
  for (std::size_t index = 0; index < 7; ++index) {
    if (sample.q[index] != static_cast<double>(sequence * 10 + index) ||
        sample.dq[index] != -sample.q[index]) {
      return false;
    }
  }
  return true;
}

bool valid_hand(const HandSample& sample, const std::uint64_t sequence) {
  if (sample.producer_poll != sequence ||
      sample.timestamp_monotonic_ns != sequence + 200 ||
      sample.timestamp_unix_ns != sequence + 2'000 ||
      sample.bundle_sequence != sequence) {
    return false;
  }
  for (std::size_t index = 0; index < 6; ++index) {
    const auto expected = static_cast<std::int32_t>(sequence * 10 + index);
    if (sample.angles[index] != expected ||
        sample.angle_targets[index] != expected + 1 ||
        sample.current_mA[index] != static_cast<std::int32_t>(sequence + index) ||
        sample.force_g[index] != static_cast<std::int32_t>(index + 20) ||
        sample.temperature_c[index] != static_cast<std::int16_t>(index + 30) ||
        sample.status[index] != static_cast<std::uint8_t>(index) ||
        sample.errors[index] != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::uint64_t iterations =
      argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 500'000ULL;
  const std::uint32_t reader_count =
      argc > 2 ? static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10))
               : 4U;
  if (iterations == 0 || iterations > 100'000'000ULL || reader_count == 0 ||
      reader_count > 64) {
    std::cerr << "usage: anydex_telemetry_stress [iterations 1..1e8] "
                 "[readers 1..64]\n";
    return 2;
  }

  const std::string path =
      "/tmp/anydex-telemetry-stress-" + std::to_string(::getpid());
  ::unlink(path.c_str());
  struct Cleanup final {
    explicit Cleanup(std::string path_value) : path(std::move(path_value)) {}
    ~Cleanup() { ::unlink(path.c_str()); }
    std::string path;
  } cleanup(path);

  InitError error{};
  auto mapping = TelemetryMapping::Create(path, provenance(), &error);
  if (mapping == nullptr) {
    std::cerr << "create failed: " << error.message << '\n';
    return 3;
  }
  auto arm_writer = mapping->ClaimArmWriter(0xa551U, &error);
  auto hand_writer = mapping->ClaimHandWriter(0xb551U, &error);
  if (arm_writer == nullptr || hand_writer == nullptr) {
    std::cerr << "claim failed: " << error.message << '\n';
    return 4;
  }

  std::atomic<bool> arm_done{false};
  std::atomic<bool> hand_done{false};
  std::atomic<bool> start{false};
  std::atomic<std::uint64_t> failures{0};
  std::atomic<std::uint64_t> successful_reads{0};
  std::atomic<std::uint64_t> contended_reads{0};

  std::vector<std::thread> readers;
  readers.reserve(reader_count);
  for (std::uint32_t reader_index = 0; reader_index < reader_count;
       ++reader_index) {
    readers.emplace_back([&]() {
      const auto reader = mapping->reader();
      std::uint64_t last_arm = 0;
      std::uint64_t last_hand = 0;
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      while (!arm_done.load(std::memory_order_acquire) ||
             !hand_done.load(std::memory_order_acquire) ||
             last_arm < iterations || last_hand < iterations) {
        ArmSample arm{};
        const auto arm_result = reader.read_arm(arm, 64);
        if (arm_result.code == ReadCode::kOk) {
          if (arm_result.sequence < last_arm ||
              !valid_arm(arm, arm_result.sequence)) {
            failures.fetch_add(1, std::memory_order_relaxed);
          }
          last_arm = arm_result.sequence;
          successful_reads.fetch_add(1, std::memory_order_relaxed);
        } else if (arm_result.code == ReadCode::kContended) {
          contended_reads.fetch_add(1, std::memory_order_relaxed);
        }

        HandSample hand{};
        const auto hand_result = reader.read_hand(hand, 64);
        if (hand_result.code == ReadCode::kOk) {
          if (hand_result.sequence < last_hand ||
              !valid_hand(hand, hand_result.sequence)) {
            failures.fetch_add(1, std::memory_order_relaxed);
          }
          last_hand = hand_result.sequence;
          successful_reads.fetch_add(1, std::memory_order_relaxed);
        } else if (hand_result.code == ReadCode::kContended) {
          contended_reads.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  std::thread arm_thread([&]() {
    ArmSample sample = base_arm();
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::uint64_t sequence = 1; sequence <= iterations; ++sequence) {
      sample.timestamp_monotonic_ns = sequence + 100;
      sample.timestamp_unix_ns = sequence + 1'000;
      sample.producer_cycle = sequence;
      sample.bundle_sequence = sequence;
      for (std::size_t index = 0; index < 16; ++index) {
        sample.O_T_EE[index] = static_cast<double>(sequence * 100 + index);
      }
      for (std::size_t index = 0; index < 7; ++index) {
        sample.q[index] = static_cast<double>(sequence * 10 + index);
        sample.dq[index] = -sample.q[index];
      }
      if (arm_writer->publish(sample).code != PublishCode::kOk) {
        failures.fetch_add(1, std::memory_order_relaxed);
        break;
      }
    }
    arm_done.store(true, std::memory_order_release);
  });

  std::thread hand_thread([&]() {
    HandSample sample = base_hand();
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::uint64_t sequence = 1; sequence <= iterations; ++sequence) {
      sample.timestamp_monotonic_ns = sequence + 200;
      sample.timestamp_unix_ns = sequence + 2'000;
      sample.producer_poll = sequence;
      sample.bundle_sequence = sequence;
      for (std::size_t index = 0; index < 6; ++index) {
        sample.angles[index] =
            static_cast<std::int32_t>(sequence * 10 + index);
        sample.angle_targets[index] = sample.angles[index] + 1;
        sample.current_mA[index] =
            static_cast<std::int32_t>(sequence + index);
        sample.force_g[index] = static_cast<std::int32_t>(index + 20);
        sample.temperature_c[index] = static_cast<std::int16_t>(index + 30);
        sample.status[index] = static_cast<std::uint8_t>(index);
        sample.errors[index] = 0;
      }
      if (hand_writer->publish(sample).code != PublishCode::kOk) {
        failures.fetch_add(1, std::memory_order_relaxed);
        break;
      }
    }
    hand_done.store(true, std::memory_order_release);
  });

  start.store(true, std::memory_order_release);
  arm_thread.join();
  hand_thread.join();
  for (auto& reader : readers) {
    reader.join();
  }

  std::cout << "offline=true iterations_per_stream=" << iterations
            << " readers=" << reader_count
            << " successful_reads=" << successful_reads.load()
            << " bounded_contention=" << contended_reads.load()
            << " torn_or_invalid=" << failures.load() << '\n';
  return failures.load() == 0 ? 0 : 5;
}
