#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <franka/active_control_base.h>
#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/robot_state.h>

#include "anydex/telemetry/telemetry.hpp"
#include "cartesian_segment.hpp"

#ifndef ANYDEX_PYLIBFRANKA_MODULE_PATH
#error "ANYDEX_PYLIBFRANKA_MODULE_PATH is required"
#endif
#ifndef ANYDEX_PYLIBFRANKA_MODULE_SHA256
#error "ANYDEX_PYLIBFRANKA_MODULE_SHA256 is required"
#endif
#ifndef ANYDEX_LIBFRANKA_LIBRARY_PATH
#error "ANYDEX_LIBFRANKA_LIBRARY_PATH is required"
#endif
#ifndef ANYDEX_LIBFRANKA_LIBRARY_SHA256
#error "ANYDEX_LIBFRANKA_LIBRARY_SHA256 is required"
#endif
#ifndef ANYDEX_LIBFRANKA_VERSION
#error "ANYDEX_LIBFRANKA_VERSION is required"
#endif

namespace py = pybind11;
namespace telemetry = anydex::telemetry;
namespace franka_native = anydex::franka_native;

namespace {

constexpr std::uint64_t kNanosecondsPerMillisecond = 1000000ULL;

thread_local std::optional<franka_native::CartesianSegmentTelemetry>
    last_cartesian_segment_telemetry;

std::uint8_t hex_nibble(const char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  throw std::invalid_argument("digest must be lowercase hexadecimal");
}

template <std::size_t N>
std::array<std::uint8_t, N> parse_lower_hex(const std::string& text,
                                            const char* name) {
  if (text.size() != N * 2U) {
    throw std::invalid_argument(std::string(name) + " has the wrong length");
  }
  std::array<std::uint8_t, N> output{};
  for (std::size_t index = 0; index < N; ++index) {
    try {
      output[index] = static_cast<std::uint8_t>(
          (hex_nibble(text[index * 2U]) << 4U) |
          hex_nibble(text[index * 2U + 1U]));
    } catch (const std::invalid_argument&) {
      throw std::invalid_argument(std::string(name) +
                                  " must be lowercase hexadecimal");
    }
  }
  return output;
}

std::array<std::uint8_t, telemetry::kUuidBytes> parse_uuid(
    const std::string& text) {
  if (text.size() != 36U || text[8] != '-' || text[13] != '-' ||
      text[18] != '-' || text[23] != '-') {
    throw std::invalid_argument(
        "run_uuid must be canonical lowercase hyphenated UUID");
  }
  std::string compact;
  compact.reserve(32U);
  for (const char value : text) {
    if (value != '-') {
      compact.push_back(value);
    }
  }
  return parse_lower_hex<telemetry::kUuidBytes>(compact, "run_uuid");
}

template <std::size_t N>
void copy_fixed_string(char (&output)[N], const std::string& value,
                       const char* name) {
  if (value.empty() || value.size() >= N ||
      value.find('\0') != std::string::npos) {
    throw std::invalid_argument(std::string(name) +
                                " must be nonempty and fit its fixed field");
  }
  std::memset(output, 0, N);
  std::memcpy(output, value.data(), value.size());
}

telemetry::SessionProvenance parse_provenance(const py::dict& mapping) {
  static constexpr std::array<const char*, 10> kExpected{
      "run_uuid",
      "execution_contract_sha256",
      "source_snapshot_sha256",
      "control_config_sha256",
      "calibration_sha256",
      "producer_build_sha256",
      "created_monotonic_ns",
      "created_unix_ns",
      "producer_name",
      "robot_id",
  };
  if (mapping.size() != kExpected.size()) {
    throw std::invalid_argument(
        "native producer provenance has missing or unknown fields");
  }
  for (const char* key : kExpected) {
    if (!mapping.contains(key)) {
      throw std::invalid_argument(std::string("native producer provenance is missing ") +
                                  key);
    }
  }

  telemetry::SessionProvenance result{};
  const auto run_uuid = parse_uuid(py::cast<std::string>(mapping["run_uuid"]));
  const auto execution = parse_lower_hex<telemetry::kDigestBytes>(
      py::cast<std::string>(mapping["execution_contract_sha256"]),
      "execution_contract_sha256");
  const auto snapshot = parse_lower_hex<telemetry::kDigestBytes>(
      py::cast<std::string>(mapping["source_snapshot_sha256"]),
      "source_snapshot_sha256");
  const auto config = parse_lower_hex<telemetry::kDigestBytes>(
      py::cast<std::string>(mapping["control_config_sha256"]),
      "control_config_sha256");
  const auto calibration = parse_lower_hex<telemetry::kDigestBytes>(
      py::cast<std::string>(mapping["calibration_sha256"]),
      "calibration_sha256");
  const auto build = parse_lower_hex<telemetry::kDigestBytes>(
      py::cast<std::string>(mapping["producer_build_sha256"]),
      "producer_build_sha256");
  std::memcpy(result.run_uuid, run_uuid.data(), run_uuid.size());
  std::memcpy(result.execution_contract_sha256, execution.data(), execution.size());
  std::memcpy(result.source_snapshot_sha256, snapshot.data(), snapshot.size());
  std::memcpy(result.control_config_sha256, config.data(), config.size());
  std::memcpy(result.calibration_sha256, calibration.data(), calibration.size());
  std::memcpy(result.producer_build_sha256, build.data(), build.size());
  result.created_monotonic_ns =
      py::cast<std::uint64_t>(mapping["created_monotonic_ns"]);
  result.created_unix_ns = py::cast<std::uint64_t>(mapping["created_unix_ns"]);
  copy_fixed_string(result.producer_name,
                    py::cast<std::string>(mapping["producer_name"]),
                    "producer_name");
  copy_fixed_string(result.robot_id,
                    py::cast<std::string>(mapping["robot_id"]), "robot_id");
  return result;
}

template <std::size_t N>
std::array<double, N> parse_finite_array(const py::handle& value,
                                         const char* name) {
  std::array<double, N> result{};
  py::sequence sequence;
  try {
    sequence = py::reinterpret_borrow<py::sequence>(value);
  } catch (const py::cast_error&) {
    throw std::invalid_argument(std::string(name) + " must be a sequence");
  }
  if (py::len(sequence) != static_cast<py::ssize_t>(N)) {
    throw std::invalid_argument(std::string(name) + " has the wrong length");
  }
  for (std::size_t index = 0; index < N; ++index) {
    try {
      result[index] = py::cast<double>(sequence[index]);
    } catch (const py::cast_error&) {
      throw std::invalid_argument(std::string(name) +
                                  " must contain only numeric values");
    }
    if (!std::isfinite(result[index])) {
      throw std::invalid_argument(std::string(name) +
                                  " must contain only finite values");
    }
  }
  return result;
}

double parse_finite_double(const py::dict& mapping, const char* name) {
  double value{};
  try {
    value = py::cast<double>(mapping[name]);
  } catch (const py::cast_error&) {
    throw std::invalid_argument(std::string(name) + " must be numeric");
  }
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
  return value;
}

std::uint64_t parse_uint64(const py::dict& mapping, const char* name) {
  try {
    return py::cast<std::uint64_t>(mapping[name]);
  } catch (const py::cast_error&) {
    throw std::invalid_argument(std::string(name) +
                                " must be a non-negative integer");
  }
}

franka_native::CartesianSegmentConfig parse_cartesian_segment_config(
    const py::dict& mapping) {
  static constexpr std::array<const char*, 25> kExpected{
      "workspace_min_m",
      "workspace_max_m",
      "joint_lower_rad",
      "joint_upper_rad",
      "joint_limit_margin_rad",
      "duration_s",
      "min_cartesian_duration_s",
      "max_cartesian_speed_m_s",
      "max_angular_speed_rad_s",
      "max_segment_translation_m",
      "max_segment_rotation_rad",
      "endpoint_timeout_s",
      "settle_time_s",
      "translation_arrival_tolerance_m",
      "rotation_arrival_tolerance_rad",
      "settle_max_dq_rad_s",
      "min_control_success_rate",
      "control_success_hard_floor",
      "control_success_evaluation_window_s",
      "startup_deadline_s",
      "startup_min_positive_writes",
      "min_control_period_s",
      "max_control_period_s",
      "read_to_write_budget_ns",
      "wall_deadline_slack_s",
      // wall_deadline_fraction is checked separately below because keeping the
      // schema count explicit makes unknown-key mistakes fail closed.
  };
  constexpr const char* kFinalExpected = "wall_deadline_fraction";
  if (mapping.size() != kExpected.size() + 1U) {
    throw std::invalid_argument(
        "native Cartesian segment config has missing or unknown fields");
  }
  for (const char* key : kExpected) {
    if (!mapping.contains(key)) {
      throw std::invalid_argument(std::string("native Cartesian segment config is missing ") +
                                  key);
    }
  }
  if (!mapping.contains(kFinalExpected)) {
    throw std::invalid_argument(
        "native Cartesian segment config is missing wall_deadline_fraction");
  }

  franka_native::CartesianSegmentConfig result{};
  result.workspace_min_m =
      parse_finite_array<3>(mapping["workspace_min_m"], "workspace_min_m");
  result.workspace_max_m =
      parse_finite_array<3>(mapping["workspace_max_m"], "workspace_max_m");
  result.joint_lower_rad =
      parse_finite_array<7>(mapping["joint_lower_rad"], "joint_lower_rad");
  result.joint_upper_rad =
      parse_finite_array<7>(mapping["joint_upper_rad"], "joint_upper_rad");
  result.joint_limit_margin_rad =
      parse_finite_double(mapping, "joint_limit_margin_rad");
  result.duration_s = parse_finite_double(mapping, "duration_s");
  result.min_cartesian_duration_s =
      parse_finite_double(mapping, "min_cartesian_duration_s");
  result.max_cartesian_speed_m_s =
      parse_finite_double(mapping, "max_cartesian_speed_m_s");
  result.max_angular_speed_rad_s =
      parse_finite_double(mapping, "max_angular_speed_rad_s");
  result.max_segment_translation_m =
      parse_finite_double(mapping, "max_segment_translation_m");
  result.max_segment_rotation_rad =
      parse_finite_double(mapping, "max_segment_rotation_rad");
  result.endpoint_timeout_s =
      parse_finite_double(mapping, "endpoint_timeout_s");
  result.settle_time_s = parse_finite_double(mapping, "settle_time_s");
  result.translation_arrival_tolerance_m =
      parse_finite_double(mapping, "translation_arrival_tolerance_m");
  result.rotation_arrival_tolerance_rad =
      parse_finite_double(mapping, "rotation_arrival_tolerance_rad");
  result.settle_max_dq_rad_s =
      parse_finite_double(mapping, "settle_max_dq_rad_s");
  result.min_control_success_rate =
      parse_finite_double(mapping, "min_control_success_rate");
  result.control_success_hard_floor =
      parse_finite_double(mapping, "control_success_hard_floor");
  result.control_success_evaluation_window_s =
      parse_finite_double(mapping, "control_success_evaluation_window_s");
  result.startup_deadline_s =
      parse_finite_double(mapping, "startup_deadline_s");
  const std::uint64_t startup_writes =
      parse_uint64(mapping, "startup_min_positive_writes");
  if (startup_writes > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("startup_min_positive_writes is too large");
  }
  result.startup_min_positive_writes =
      static_cast<std::uint32_t>(startup_writes);
  result.min_control_period_s =
      parse_finite_double(mapping, "min_control_period_s");
  result.max_control_period_s =
      parse_finite_double(mapping, "max_control_period_s");
  result.read_to_write_budget_ns =
      parse_uint64(mapping, "read_to_write_budget_ns");
  result.wall_deadline_slack_s =
      parse_finite_double(mapping, "wall_deadline_slack_s");
  result.wall_deadline_fraction =
      parse_finite_double(mapping, "wall_deadline_fraction");
  return result;
}

py::dict cartesian_segment_telemetry_to_dict(
    const franka_native::CartesianSegmentTelemetry& value) {
  py::dict result;
  result["schema_version"] = 1;
  result["failure_code"] = static_cast<std::uint32_t>(value.failure_code);
  result["failure_name"] =
      franka_native::CartesianSegmentFailureCodeName(value.failure_code);
  result["reads"] = value.reads;
  result["writes"] = value.writes;
  result["positive_period_writes"] = value.positive_period_writes;
  result["read_to_write_overruns"] = value.read_to_write_overruns;
  result["max_read_to_write_ns"] = value.max_read_to_write_ns;
  result["complete_success_windows"] = value.complete_success_windows;
  result["success_qualified"] = value.success_qualified;
  result["motion_finished_written"] = value.motion_finished_written;
  result["qualification_control_time_s"] =
      value.qualification_control_time_s;
  result["qualification_wall_time_s"] = value.qualification_wall_time_s;
  result["qualification_rate"] = value.qualification_rate;
  result["trajectory_control_time_s"] = value.trajectory_control_time_s;
  result["endpoint_control_time_s"] = value.endpoint_control_time_s;
  result["endpoint_stable_time_s"] = value.endpoint_stable_time_s;
  result["latest_success_rate"] = value.latest_success_rate;
  result["minimum_postqualification_success_rate"] =
      value.minimum_postqualification_success_rate;
  result["latest_complete_window_average"] =
      value.latest_complete_window_average;
  result["final_translation_error_m"] = value.final_translation_error_m;
  result["final_rotation_error_rad"] = value.final_rotation_error_rad;
  result["final_max_abs_dq_rad_s"] = value.final_max_abs_dq_rad_s;
  return result;
}

std::runtime_error init_error(const telemetry::InitError& error) {
  return std::runtime_error(std::string("native telemetry init failed: ") +
                            error.message);
}

std::uint64_t checked_robot_time_ns(const franka::RobotState& state) {
  const std::uint64_t milliseconds = state.time.toMSec();
  if (milliseconds >
      std::numeric_limits<std::uint64_t>::max() /
          kNanosecondsPerMillisecond) {
    throw std::runtime_error("Franka robot time overflowed uint64 nanoseconds");
  }
  return milliseconds * kNanosecondsPerMillisecond;
}

std::uint64_t checked_add(const std::uint64_t base,
                          const std::uint64_t delta,
                          const char* name) {
  if (delta > std::numeric_limits<std::uint64_t>::max() - base) {
    throw std::runtime_error(std::string(name) + " overflowed uint64");
  }
  return base + delta;
}

class NativeTelemetryProducer final {
 public:
  static std::unique_ptr<NativeTelemetryProducer> Create(
      const std::string& path, const py::dict& provenance,
      const std::uint64_t arm_writer_token,
      const std::uint64_t hand_writer_token,
      const std::uint32_t arm_publish_decimation) {
    if (arm_writer_token == 0 || hand_writer_token == 0 ||
        arm_writer_token == hand_writer_token) {
      throw std::invalid_argument(
          "writer tokens must be distinct nonzero uint64 values");
    }
    if (arm_publish_decimation < 1U || arm_publish_decimation > 1000U) {
      throw std::invalid_argument("arm_publish_decimation must be in 1..1000");
    }
    telemetry::InitError error{};
    auto mapping = telemetry::TelemetryMapping::Create(
        path, parse_provenance(provenance), &error);
    if (!mapping) {
      throw init_error(error);
    }
    auto arm = mapping->ClaimArmWriter(arm_writer_token, &error);
    if (!arm) {
      throw init_error(error);
    }
    auto hand = mapping->ClaimHandWriter(hand_writer_token, &error);
    if (!hand) {
      throw init_error(error);
    }
    return std::unique_ptr<NativeTelemetryProducer>(new NativeTelemetryProducer(
        std::move(mapping), std::move(arm), std::move(hand),
        arm_publish_decimation));
  }

  ~NativeTelemetryProducer() { close(); }
  NativeTelemetryProducer(const NativeTelemetryProducer&) = delete;
  NativeTelemetryProducer& operator=(const NativeTelemetryProducer&) = delete;

  void set_stage(const std::string& name, const std::uint64_t epoch,
                 const std::uint64_t bundle_sequence) {
    require_open();
    if (epoch == 0) {
      throw std::invalid_argument("stage epoch must be nonzero");
    }
    char candidate[telemetry::kStageNameBytes]{};
    if (!telemetry::set_stage_name(candidate, name.data(), name.size())) {
      throw std::invalid_argument(
          "stage name must be valid UTF-8 and fit char[32]");
    }
    std::memcpy(stage_name_, candidate, sizeof(stage_name_));
    stage_name_hash64_ = telemetry::stage_name_hash(name.data(), name.size());
    stage_epoch_ = epoch;
    bundle_sequence_ = bundle_sequence;
    stage_is_set_ = true;
  }

  std::uint64_t synchronize_and_publish_arm(
      const franka::RobotState& state, const std::uint64_t timestamp_unix_ns,
      const std::uint64_t timestamp_monotonic_ns) {
    require_ready_for_sample(timestamp_unix_ns, timestamp_monotonic_ns);
    base_robot_time_ns_ = checked_robot_time_ns(state);
    base_unix_ns_ = timestamp_unix_ns;
    base_monotonic_ns_ = timestamp_monotonic_ns;
    clock_is_synchronized_ = true;
    return publish_arm(state, timestamp_unix_ns, timestamp_monotonic_ns);
  }

  py::tuple read_once_tapped(franka::ActiveControlBase& control) {
    require_open();
    if (!stage_is_set_ || !clock_is_synchronized_) {
      throw std::runtime_error(
          "native arm tap requires stage and clock synchronization before control");
    }
    auto result = control.readOnce();
    ++producer_cycle_;
    if (producer_cycle_ == 1U ||
        producer_cycle_ % arm_publish_decimation_ == 0U) {
      const std::uint64_t robot_time_ns = checked_robot_time_ns(result.first);
      if (robot_time_ns < base_robot_time_ns_) {
        throw std::runtime_error("Franka robot time moved backwards");
      }
      const std::uint64_t delta = robot_time_ns - base_robot_time_ns_;
      publish_arm(result.first, checked_add(base_unix_ns_, delta, "unix timestamp"),
                  checked_add(base_monotonic_ns_, delta,
                              "monotonic timestamp"));
    }
    return py::make_tuple(std::move(result.first), result.second);
  }

  std::uint64_t publish_hand(
      const std::array<std::int32_t, 6>& angles,
      const std::array<std::int32_t, 6>& angle_targets,
      const std::array<std::int32_t, 6>& current_mA,
      const std::optional<std::array<std::int32_t, 6>>& force_g,
      const std::array<std::int16_t, 6>& temperature_c,
      const std::array<std::uint8_t, 6>& status,
      const std::array<std::uint8_t, 6>& errors,
      const std::uint64_t timestamp_unix_ns,
      const std::uint64_t timestamp_monotonic_ns) {
    require_ready_for_sample(timestamp_unix_ns, timestamp_monotonic_ns);
    for (std::size_t index = 0; index < angles.size(); ++index) {
      if (angles[index] < 0 || angles[index] > 1000 ||
          angle_targets[index] < -1 || angle_targets[index] > 1000) {
        throw std::invalid_argument(
            "RH56 angles must be 0..1000 and targets -1..1000");
      }
    }
    telemetry::HandSample sample{};
    sample.timestamp_monotonic_ns = timestamp_monotonic_ns;
    sample.timestamp_unix_ns = timestamp_unix_ns;
    sample.producer_poll = ++producer_poll_;
    fill_common(sample);
    sample.validity_flags =
        telemetry::kHandTimestampValid | telemetry::kHandAnglesValid |
        telemetry::kHandTargetsValid | telemetry::kHandCurrentValid |
        telemetry::kHandTemperatureValid | telemetry::kHandStatusValid |
        telemetry::kHandErrorsValid;
    if (force_g.has_value()) {
      sample.validity_flags |= telemetry::kHandForceValid;
    }
    sample.source = telemetry::Source::kInspireRh56AngleAct;
    sample.measurement_kind = telemetry::MeasurementKind::kMeasured;
    std::copy(angles.begin(), angles.end(), sample.angles);
    std::copy(angle_targets.begin(), angle_targets.end(), sample.angle_targets);
    std::copy(current_mA.begin(), current_mA.end(), sample.current_mA);
    if (force_g.has_value()) {
      std::copy(force_g->begin(), force_g->end(), sample.force_g);
    }
    std::copy(temperature_c.begin(), temperature_c.end(), sample.temperature_c);
    std::copy(status.begin(), status.end(), sample.status);
    std::copy(errors.begin(), errors.end(), sample.errors);
    const auto result = hand_->publish(sample);
    if (result.code != telemetry::PublishCode::kOk) {
      throw std::runtime_error(std::string("native hand publish failed: ") +
                               telemetry::publish_code_name(result.code));
    }
    return result.sequence;
  }

  void close() noexcept {
    hand_.reset();
    arm_.reset();
    mapping_.reset();
    closed_ = true;
  }

  bool closed() const noexcept { return closed_; }
  std::uint64_t producer_cycle() const noexcept { return producer_cycle_; }
  std::uint64_t producer_poll() const noexcept { return producer_poll_; }

 private:
  NativeTelemetryProducer(
      std::unique_ptr<telemetry::TelemetryMapping> mapping,
      std::unique_ptr<telemetry::ArmWriter> arm,
      std::unique_ptr<telemetry::HandWriter> hand,
      const std::uint32_t arm_publish_decimation) noexcept
      : mapping_(std::move(mapping)),
        arm_(std::move(arm)),
        hand_(std::move(hand)),
        arm_publish_decimation_(arm_publish_decimation) {}

  void require_open() const {
    if (closed_ || !mapping_ || !arm_ || !hand_) {
      throw std::runtime_error("native telemetry producer is closed");
    }
  }

  void require_ready_for_sample(const std::uint64_t timestamp_unix_ns,
                                const std::uint64_t timestamp_monotonic_ns) const {
    require_open();
    if (!stage_is_set_) {
      throw std::runtime_error("native telemetry stage is not set");
    }
    if (timestamp_unix_ns == 0 || timestamp_monotonic_ns == 0) {
      throw std::invalid_argument("telemetry timestamps must be nonzero");
    }
  }

  template <typename Sample>
  void fill_common(Sample& sample) const noexcept {
    sample.stage_epoch = stage_epoch_;
    sample.bundle_sequence = bundle_sequence_;
    sample.stage_name_hash64 = stage_name_hash64_;
    std::memcpy(sample.stage_name, stage_name_, sizeof(stage_name_));
  }

  std::uint64_t publish_arm(const franka::RobotState& state,
                            const std::uint64_t timestamp_unix_ns,
                            const std::uint64_t timestamp_monotonic_ns) {
    telemetry::ArmSample sample{};
    sample.timestamp_monotonic_ns = timestamp_monotonic_ns;
    sample.timestamp_unix_ns = timestamp_unix_ns;
    sample.producer_cycle = producer_cycle_;
    fill_common(sample);
    sample.validity_flags =
        telemetry::kArmTimestampValid | telemetry::kArmPoseValid |
        telemetry::kArmQValid | telemetry::kArmDqValid |
        telemetry::kArmSuccessRateValid;
    sample.source = telemetry::Source::kFrankaRobotStateOTEE;
    sample.measurement_kind = telemetry::MeasurementKind::kMeasured;
    sample.robot_mode = static_cast<std::uint32_t>(state.robot_mode);
    std::copy(state.O_T_EE.begin(), state.O_T_EE.end(), sample.O_T_EE);
    std::copy(state.q.begin(), state.q.end(), sample.q);
    std::copy(state.dq.begin(), state.dq.end(), sample.dq);
    sample.control_command_success_rate = state.control_command_success_rate;
    const auto result = arm_->publish(sample);
    if (result.code != telemetry::PublishCode::kOk) {
      throw std::runtime_error(std::string("native arm publish failed: ") +
                               telemetry::publish_code_name(result.code));
    }
    return result.sequence;
  }

  std::unique_ptr<telemetry::TelemetryMapping> mapping_;
  std::unique_ptr<telemetry::ArmWriter> arm_;
  std::unique_ptr<telemetry::HandWriter> hand_;
  std::uint32_t arm_publish_decimation_;
  char stage_name_[telemetry::kStageNameBytes]{};
  std::uint64_t stage_name_hash64_{0};
  std::uint64_t stage_epoch_{0};
  std::uint64_t bundle_sequence_{0};
  std::uint64_t producer_cycle_{0};
  std::uint64_t producer_poll_{0};
  std::uint64_t base_robot_time_ns_{0};
  std::uint64_t base_unix_ns_{0};
  std::uint64_t base_monotonic_ns_{0};
  bool stage_is_set_{false};
  bool clock_is_synchronized_{false};
  bool closed_{false};
};

class FakeActiveControl final : public franka::ActiveControlBase {
 public:
  explicit FakeActiveControl(const std::uint64_t initial_time_ms)
      : time_ms_(initial_time_ms) {}

  std::pair<franka::RobotState, franka::Duration> readOnce() override {
    franka::RobotState state{};
    ++time_ms_;
    state.time = franka::Duration(time_ms_);
    state.O_T_EE[0] = 1.0;
    state.O_T_EE[5] = 1.0;
    state.O_T_EE[10] = 1.0;
    state.O_T_EE[15] = 1.0;
    state.O_T_EE[12] = static_cast<double>(time_ms_) * 1.0e-6;
    state.q[0] = static_cast<double>(time_ms_) * 1.0e-7;
    state.control_command_success_rate = 1.0;
    state.robot_mode = franka::RobotMode::kMove;
    return {std::move(state), franka::Duration(1)};
  }

  void writeOnce(const franka::Torques&) override {}
  void writeOnce(const franka::JointPositions&,
                 const std::optional<const franka::Torques>&) override {}
  void writeOnce(const franka::JointVelocities&,
                 const std::optional<const franka::Torques>&) override {}
  void writeOnce(const franka::CartesianPose&,
                 const std::optional<const franka::Torques>&) override {}
  void writeOnce(const franka::CartesianVelocities&,
                 const std::optional<const franka::Torques>&) override {}
  void writeOnce(const franka::JointPositions&) override {}
  void writeOnce(const franka::JointVelocities&) override {}
  void writeOnce(const franka::CartesianPose&) override {}
  void writeOnce(const franka::CartesianVelocities&) override {}

 private:
  std::uint64_t time_ms_;
};

class FakeCartesianControl final : public franka::ActiveControlBase {
 public:
  FakeCartesianControl(std::array<double, 16> start_pose,
                       const double initial_success,
                       const std::uint64_t degrade_after_reads,
                       const double degraded_success,
                       const std::uint64_t period_ms,
                       const bool first_period_zero,
                       const bool follow_commands,
                       std::string fault,
                       const std::uint64_t fault_after_reads)
      : measured_pose_(start_pose),
        commanded_pose_(std::move(start_pose)),
        initial_success_(initial_success),
        degrade_after_reads_(degrade_after_reads),
        degraded_success_(degraded_success),
        period_ms_(period_ms),
        first_period_zero_(first_period_zero),
        follow_commands_(follow_commands),
        fault_(std::move(fault)),
        fault_after_reads_(fault_after_reads) {
    if (period_ms_ == 0U && !first_period_zero_) {
      throw std::invalid_argument("fake period_ms must be positive");
    }
  }

  std::pair<franka::RobotState, franka::Duration> readOnce() override {
    ++reads_;
    time_ms_ += period_ms_;
    franka::RobotState state{};
    state.time = franka::Duration(time_ms_);
    state.O_T_EE = measured_pose_;
    state.O_T_EE_c = commanded_pose_;
    state.O_T_EE_d = commanded_pose_;
    state.q = {0.0, -0.5, 0.0, -1.5, 0.0, 2.0, 0.0};
    state.control_command_success_rate =
        degrade_after_reads_ != 0U && reads_ > degrade_after_reads_
            ? degraded_success_
            : initial_success_;
    state.robot_mode = franka::RobotMode::kMove;
    if (!fault_.empty() && fault_after_reads_ != 0U &&
        reads_ >= fault_after_reads_) {
      if (fault_ == "contact") {
        state.cartesian_contact[0] = 1.0;
      } else if (fault_ == "collision") {
        state.joint_collision[0] = 1.0;
      } else if (fault_ == "error") {
        std::array<bool, 41> errors{};
        errors[0] = true;
        state.current_errors = franka::Errors(errors);
      } else if (fault_ == "joint") {
        state.q[0] = 3.0;
      } else if (fault_ == "workspace") {
        state.O_T_EE[12] = 10.0;
      } else if (fault_ == "mode") {
        state.robot_mode = franka::RobotMode::kReflex;
      } else {
        throw std::runtime_error("unknown fake Cartesian fault");
      }
    }
    const bool first_zero = first_period_zero_ && reads_ == 1U;
    return {std::move(state),
            franka::Duration(first_zero ? 0U : period_ms_)};
  }

  void writeOnce(const franka::CartesianPose& command) override {
    commanded_pose_ = command.O_T_EE;
    if (follow_commands_) {
      measured_pose_ = commanded_pose_;
    }
    commands_.push_back(commanded_pose_);
    finished_.push_back(command.motion_finished);
  }

  void writeOnce(const franka::Torques&) override {
    throw std::runtime_error("fake Cartesian control received torque command");
  }
  void writeOnce(const franka::JointPositions&,
                 const std::optional<const franka::Torques>&) override {
    throw std::runtime_error("fake Cartesian control received joint command");
  }
  void writeOnce(const franka::JointVelocities&,
                 const std::optional<const franka::Torques>&) override {
    throw std::runtime_error("fake Cartesian control received joint command");
  }
  void writeOnce(const franka::CartesianPose& command,
                 const std::optional<const franka::Torques>&) override {
    writeOnce(command);
  }
  void writeOnce(const franka::CartesianVelocities&,
                 const std::optional<const franka::Torques>&) override {
    throw std::runtime_error("fake Cartesian control received velocity command");
  }
  void writeOnce(const franka::JointPositions&) override {
    throw std::runtime_error("fake Cartesian control received joint command");
  }
  void writeOnce(const franka::JointVelocities&) override {
    throw std::runtime_error("fake Cartesian control received joint command");
  }
  void writeOnce(const franka::CartesianVelocities&) override {
    throw std::runtime_error("fake Cartesian control received velocity command");
  }

  std::uint64_t reads() const noexcept { return reads_; }
  const std::vector<std::array<double, 16>>& commands() const noexcept {
    return commands_;
  }
  const std::vector<bool>& finished() const noexcept { return finished_; }

 private:
  std::array<double, 16> measured_pose_{};
  std::array<double, 16> commanded_pose_{};
  double initial_success_{};
  std::uint64_t degrade_after_reads_{};
  double degraded_success_{};
  std::uint64_t period_ms_{};
  bool first_period_zero_{};
  bool follow_commands_{};
  std::string fault_;
  std::uint64_t fault_after_reads_{};
  std::uint64_t reads_{0};
  std::uint64_t time_ms_{1000};
  std::vector<std::array<double, 16>> commands_;
  std::vector<bool> finished_;
};

}  // namespace

PYBIND11_MODULE(_anydex_franka_telemetry, module) {
  // Register the exact wheel's libfranka C++ types and preload its hashed
  // transitive DSOs before this module exposes ActiveControlBase/RobotState
  // signatures.  Importing pylibfranka does not construct a Robot or control
  // handle.  The production executor still delays importing this adapter
  // until its existing offline gates have passed.
  py::module_::import("pylibfranka");
  module.doc() =
      "Single-owner fused libfranka readOnce + fixed-POD telemetry producer";
  py::dict pylibfranka_dependency;
  pylibfranka_dependency["path"] = ANYDEX_PYLIBFRANKA_MODULE_PATH;
  pylibfranka_dependency["sha256"] = ANYDEX_PYLIBFRANKA_MODULE_SHA256;
  py::dict libfranka_dependency;
  libfranka_dependency["path"] = ANYDEX_LIBFRANKA_LIBRARY_PATH;
  libfranka_dependency["sha256"] = ANYDEX_LIBFRANKA_LIBRARY_SHA256;
  py::dict build_dependencies;
  build_dependencies["schema_version"] = 1;
  build_dependencies["pylibfranka"] = pylibfranka_dependency;
  build_dependencies["libfranka"] = libfranka_dependency;
  build_dependencies["libfranka"]["version"] = ANYDEX_LIBFRANKA_VERSION;
  module.attr("BUILD_DEPENDENCIES") = build_dependencies;
  py::register_exception<franka_native::CartesianSegmentError>(
      module, "NativeCartesianSegmentError");
  py::class_<NativeTelemetryProducer>(module, "NativeTelemetryProducer")
      .def_static("create", &NativeTelemetryProducer::Create,
                  py::arg("path"), py::arg("provenance"),
                  py::arg("arm_writer_token"), py::arg("hand_writer_token"),
                  py::arg("arm_publish_decimation") = 20)
      .def("set_stage", &NativeTelemetryProducer::set_stage, py::arg("name"),
           py::arg("epoch"), py::arg("bundle_sequence") = 0)
      .def("synchronize_and_publish_arm",
           &NativeTelemetryProducer::synchronize_and_publish_arm,
           py::arg("state"), py::arg("timestamp_unix_ns"),
           py::arg("timestamp_monotonic_ns"))
      .def("read_once_tapped", &NativeTelemetryProducer::read_once_tapped,
           py::arg("control"))
      .def("publish_hand", &NativeTelemetryProducer::publish_hand,
           py::arg("angles"), py::arg("angle_targets"),
           py::arg("current_mA"), py::arg("force_g"),
           py::arg("temperature_c"), py::arg("status"), py::arg("errors"),
           py::arg("timestamp_unix_ns"), py::arg("timestamp_monotonic_ns"))
      .def("close", &NativeTelemetryProducer::close)
      .def_property_readonly("closed", &NativeTelemetryProducer::closed)
      .def_property_readonly("producer_cycle",
                             &NativeTelemetryProducer::producer_cycle)
      .def_property_readonly("producer_poll",
                             &NativeTelemetryProducer::producer_poll);

  module.def("make_fake_control_for_offline_test",
             [](const std::uint64_t initial_time_ms) {
               return std::unique_ptr<franka::ActiveControlBase>(
                   new FakeActiveControl(initial_time_ms));
             },
             py::arg("initial_time_ms") = 1000);

  module.def(
      "run_bounded_cartesian_segment",
      [](franka::ActiveControlBase& control,
         const py::handle& planned_start_value,
         const py::handle& target_value,
         const py::dict& config_value) {
        const auto planned_start =
            parse_finite_array<16>(planned_start_value, "planned_start");
        const auto target = parse_finite_array<16>(target_value, "target");
        const auto config = parse_cartesian_segment_config(config_value);
        franka_native::ValidateCartesianSegmentContract(planned_start, target,
                                                        config);
        last_cartesian_segment_telemetry.reset();
        try {
          franka_native::CartesianSegmentTelemetry result{};
          {
            // No Python object is accessed until this scope ends.  The entire
            // read/validate/interpolate/write loop therefore runs without one
            // Python or pybind transition per 1 kHz cycle.
            py::gil_scoped_release release;
            result = franka_native::RunBoundedCartesianSegment(
                control, planned_start, target, config);
          }
          last_cartesian_segment_telemetry = result;
          return cartesian_segment_telemetry_to_dict(result);
        } catch (const franka_native::CartesianSegmentError& error) {
          last_cartesian_segment_telemetry = error.telemetry();
          throw;
        }
      },
      py::arg("control"), py::arg("planned_start"), py::arg("target"),
      py::arg("config"),
      "Run one fail-closed Cartesian segment on an existing 0.21.2 handle");

  module.def("last_native_cartesian_segment_telemetry", []() -> py::object {
    if (!last_cartesian_segment_telemetry.has_value()) {
      return py::none();
    }
    return cartesian_segment_telemetry_to_dict(
        last_cartesian_segment_telemetry.value());
  });

  module.def(
      "make_fake_cartesian_control_for_offline_test",
      [](const py::handle& start_pose,
         const double initial_success,
         const std::uint64_t degrade_after_reads,
         const double degraded_success,
         const std::uint64_t period_ms,
         const bool first_period_zero,
         const bool follow_commands,
         const std::string& fault,
         const std::uint64_t fault_after_reads) {
        return std::unique_ptr<franka::ActiveControlBase>(
            new FakeCartesianControl(
                parse_finite_array<16>(start_pose, "start_pose"),
                initial_success, degrade_after_reads, degraded_success,
                period_ms, first_period_zero, follow_commands, fault,
                fault_after_reads));
      },
      py::arg("start_pose"), py::arg("initial_success") = 1.0,
      py::arg("degrade_after_reads") = 0,
      py::arg("degraded_success") = 1.0, py::arg("period_ms") = 1,
      py::arg("first_period_zero") = true,
      py::arg("follow_commands") = true, py::arg("fault") = "",
      py::arg("fault_after_reads") = 0);

  module.def("fake_cartesian_control_snapshot",
             [](franka::ActiveControlBase& control) {
               const auto* fake = dynamic_cast<FakeCartesianControl*>(&control);
               if (fake == nullptr) {
                 throw std::invalid_argument(
                     "control is not the offline fake Cartesian control");
               }
               py::dict result;
               result["reads"] = fake->reads();
               result["commands"] = fake->commands();
               result["motion_finished"] = fake->finished();
               return result;
             });
}
