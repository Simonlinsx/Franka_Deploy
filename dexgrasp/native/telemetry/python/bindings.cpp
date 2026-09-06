#include "anydex/telemetry/telemetry.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using anydex::telemetry::ArmSample;
using anydex::telemetry::ArmWriter;
using anydex::telemetry::HandSample;
using anydex::telemetry::HandWriter;
using anydex::telemetry::InitError;
using anydex::telemetry::MeasurementKind;
using anydex::telemetry::ReadCode;
using anydex::telemetry::ReadResult;
using anydex::telemetry::SessionProvenance;
using anydex::telemetry::Source;
using anydex::telemetry::TelemetryMapping;

std::uint8_t hex_nibble(const char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  if (value >= 'A' && value <= 'F') {
    return static_cast<std::uint8_t>(value - 'A' + 10);
  }
  throw py::value_error("expected hexadecimal input");
}

template <std::size_t Size>
void decode_hex(const std::string& text, std::uint8_t (&output)[Size],
                const char* field) {
  if (text.size() != Size * 2) {
    throw py::value_error(std::string(field) + " must contain exactly " +
                          std::to_string(Size * 2) + " hex characters");
  }
  for (std::size_t index = 0; index < Size; ++index) {
    output[index] = static_cast<std::uint8_t>(
        (hex_nibble(text[index * 2]) << 4U) |
        hex_nibble(text[index * 2 + 1]));
  }
}

void decode_uuid(const std::string& text,
                 std::uint8_t (&output)[anydex::telemetry::kUuidBytes]) {
  if (text.size() != 32 && text.size() != 36) {
    throw py::value_error(
        "run_uuid must be 32 hex characters or canonical 36-character UUID");
  }
  if (text.size() == 36 &&
      (text[8] != '-' || text[13] != '-' || text[18] != '-' ||
       text[23] != '-')) {
    throw py::value_error("run_uuid hyphens are not in canonical positions");
  }
  std::string compact;
  compact.reserve(32);
  for (const char value : text) {
    if (value != '-') {
      compact.push_back(value);
    }
  }
  decode_hex(compact, output, "run_uuid");
  const auto version = static_cast<std::uint8_t>(output[6] >> 4U);
  if (version < 1U || version > 8U || (output[8] & 0xc0U) != 0x80U) {
    throw py::value_error("run_uuid has an unsupported RFC UUID version/variant");
  }
}

std::string encode_hex(const std::uint8_t* bytes, const std::size_t size) {
  constexpr char kDigits[] = "0123456789abcdef";
  std::string output(size * 2, '0');
  for (std::size_t index = 0; index < size; ++index) {
    output[index * 2] = kDigits[bytes[index] >> 4U];
    output[index * 2 + 1] = kDigits[bytes[index] & 0x0fU];
  }
  return output;
}

std::string encode_uuid(const std::uint8_t* bytes) {
  const std::string compact =
      encode_hex(bytes, anydex::telemetry::kUuidBytes);
  return compact.substr(0, 8) + "-" + compact.substr(8, 4) + "-" +
         compact.substr(12, 4) + "-" + compact.substr(16, 4) + "-" +
         compact.substr(20, 12);
}

template <std::size_t Capacity>
void copy_fixed_utf8(const std::string& text, char (&output)[Capacity],
                     const char* field, const bool allow_empty = false) {
  if ((!allow_empty && text.empty()) || text.size() >= Capacity ||
      text.find('\0') != std::string::npos) {
    throw py::value_error(std::string(field) +
                          " must be non-empty UTF-8 with at most " +
                          std::to_string(Capacity - 1) + " bytes");
  }
  std::memset(output, 0, Capacity);
  std::memcpy(output, text.data(), text.size());
}

template <typename Value, std::size_t Size>
void copy_numeric_sequence(const py::handle object, Value (&output)[Size],
                           const char* field) {
  const py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
  if (py::len(sequence) != static_cast<py::ssize_t>(Size)) {
    throw py::value_error(std::string(field) + " must have length " +
                          std::to_string(Size));
  }
  for (std::size_t index = 0; index < Size; ++index) {
    output[index] = py::cast<Value>(sequence[index]);
  }
}

template <std::size_t Size>
void copy_u8_sequence(const py::handle object, std::uint8_t (&output)[Size],
                      const char* field) {
  const py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
  if (py::len(sequence) != static_cast<py::ssize_t>(Size)) {
    throw py::value_error(std::string(field) + " must have length " +
                          std::to_string(Size));
  }
  for (std::size_t index = 0; index < Size; ++index) {
    const auto value = py::cast<std::int64_t>(sequence[index]);
    if (value < 0 || value > 255) {
      throw py::value_error(std::string(field) +
                            " entries must be in 0..255");
    }
    output[index] = static_cast<std::uint8_t>(value);
  }
}

template <typename Value, std::size_t Size>
py::list numeric_list(const Value (&values)[Size]) {
  py::list output(Size);
  for (std::size_t index = 0; index < Size; ++index) {
    output[index] = values[index];
  }
  return output;
}

template <typename Value, std::size_t Size>
py::object optional_list(const Value (&values)[Size], bool valid);

std::string fixed_string(const char* bytes, const std::size_t capacity) {
  std::size_t length = 0;
  while (length < capacity && bytes[length] != '\0') {
    ++length;
  }
  return std::string(bytes, length);
}

SessionProvenance parse_provenance(const py::dict& value) {
  constexpr std::array<const char*, 10> kRequired{
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
  for (const char* key : kRequired) {
    if (!value.contains(key)) {
      throw py::key_error(std::string("missing provenance field: ") + key);
    }
  }
  if (py::len(value) != static_cast<py::ssize_t>(kRequired.size())) {
    throw py::value_error("provenance contains unknown fields");
  }

  SessionProvenance output{};
  decode_uuid(py::cast<std::string>(value["run_uuid"]), output.run_uuid);
  decode_hex(py::cast<std::string>(value["execution_contract_sha256"]),
             output.execution_contract_sha256,
             "execution_contract_sha256");
  decode_hex(py::cast<std::string>(value["source_snapshot_sha256"]),
             output.source_snapshot_sha256, "source_snapshot_sha256");
  decode_hex(py::cast<std::string>(value["control_config_sha256"]),
             output.control_config_sha256, "control_config_sha256");
  decode_hex(py::cast<std::string>(value["calibration_sha256"]),
             output.calibration_sha256, "calibration_sha256");
  decode_hex(py::cast<std::string>(value["producer_build_sha256"]),
             output.producer_build_sha256, "producer_build_sha256");
  output.created_monotonic_ns =
      py::cast<std::uint64_t>(value["created_monotonic_ns"]);
  output.created_unix_ns = py::cast<std::uint64_t>(value["created_unix_ns"]);
  copy_fixed_utf8(py::cast<std::string>(value["producer_name"]),
                  output.producer_name, "producer_name");
  copy_fixed_utf8(py::cast<std::string>(value["robot_id"]), output.robot_id,
                  "robot_id");
  return output;
}

[[noreturn]] void throw_init_error(const char* operation,
                                   const InitError& error) {
  std::ostringstream message;
  message << operation << " failed (code="
          << static_cast<std::uint32_t>(error.code) << "): " << error.message;
  throw std::runtime_error(message.str());
}

py::dict header_mapping(const anydex::telemetry::LayoutHeader& header) {
  py::dict output;
  output["run_uuid"] = encode_uuid(header.run_uuid);
  output["execution_contract_sha256"] =
      encode_hex(header.execution_contract_sha256,
                 anydex::telemetry::kDigestBytes);
  output["source_snapshot_sha256"] =
      encode_hex(header.source_snapshot_sha256,
                 anydex::telemetry::kDigestBytes);
  output["control_config_sha256"] =
      encode_hex(header.control_config_sha256,
                 anydex::telemetry::kDigestBytes);
  output["calibration_sha256"] = encode_hex(
      header.calibration_sha256, anydex::telemetry::kDigestBytes);
  output["producer_build_sha256"] = encode_hex(
      header.producer_build_sha256, anydex::telemetry::kDigestBytes);
  output["created_monotonic_ns"] = header.created_monotonic_ns;
  output["created_unix_ns"] = header.created_unix_ns;
  output["producer_name"] =
      fixed_string(header.producer_name, anydex::telemetry::kProducerNameBytes);
  output["robot_id"] =
      fixed_string(header.robot_id, anydex::telemetry::kRobotIdBytes);
  return output;
}

py::dict stage_mapping(const char* name, const std::uint64_t epoch,
                       const std::uint64_t hash) {
  py::dict output;
  output["name"] = fixed_string(name, anydex::telemetry::kStageNameBytes);
  output["epoch"] = epoch;
  output["name_hash64"] = hash;
  return output;
}

py::dict arm_read_mapping(const ReadResult& result, const ArmSample& sample) {
  py::dict output;
  const bool available = result.code == ReadCode::kOk;
  output["available"] = available;
  output["code"] = static_cast<std::uint32_t>(result.code);
  output["sequence"] = result.sequence;
  output["attempts"] = result.attempts;
  if (!available) {
    output["sample"] = py::none();
    return output;
  }
  py::dict body;
  body["timestamp_monotonic_ns"] = sample.timestamp_monotonic_ns;
  body["timestamp_unix_ns"] = sample.timestamp_unix_ns;
  body["stage"] = stage_mapping(sample.stage_name, sample.stage_epoch,
                                 sample.stage_name_hash64);
  body["bundle_sequence"] = sample.bundle_sequence;
  body["source"] = anydex::telemetry::source_name(sample.source);
  body["source_code"] = static_cast<std::uint16_t>(sample.source);
  body["measurement_kind"] =
      anydex::telemetry::measurement_kind_name(sample.measurement_kind);
  body["measurement_kind_code"] =
      static_cast<std::uint16_t>(sample.measurement_kind);
  body["O_T_EE"] = numeric_list(sample.O_T_EE);
  body["q"] = optional_list(
      sample.q, (sample.validity_flags & anydex::telemetry::kArmQValid) != 0);
  body["dq"] = optional_list(
      sample.dq, (sample.validity_flags & anydex::telemetry::kArmDqValid) != 0);
  body["control_command_success_rate"] =
      (sample.validity_flags & anydex::telemetry::kArmSuccessRateValid) != 0
          ? py::cast(sample.control_command_success_rate)
          : py::none();
  output["sample"] = std::move(body);
  return output;
}

template <typename Value, std::size_t Size>
py::object optional_list(const Value (&values)[Size], const bool valid) {
  if (valid) {
    return numeric_list(values);
  }
  return py::none();
}

py::dict hand_read_mapping(const ReadResult& result, const HandSample& sample) {
  py::dict output;
  const bool available = result.code == ReadCode::kOk;
  output["available"] = available;
  output["code"] = static_cast<std::uint32_t>(result.code);
  output["sequence"] = result.sequence;
  output["attempts"] = result.attempts;
  if (!available) {
    output["sample"] = py::none();
    return output;
  }
  py::dict body;
  body["timestamp_monotonic_ns"] = sample.timestamp_monotonic_ns;
  body["timestamp_unix_ns"] = sample.timestamp_unix_ns;
  body["stage"] = stage_mapping(sample.stage_name, sample.stage_epoch,
                                 sample.stage_name_hash64);
  body["bundle_sequence"] = sample.bundle_sequence;
  body["source"] = anydex::telemetry::source_name(sample.source);
  body["source_code"] = static_cast<std::uint16_t>(sample.source);
  body["measurement_kind"] =
      anydex::telemetry::measurement_kind_name(sample.measurement_kind);
  body["measurement_kind_code"] =
      static_cast<std::uint16_t>(sample.measurement_kind);
  body["angles"] = numeric_list(sample.angles);
  body["angle_targets"] = optional_list(
      sample.angle_targets,
      (sample.validity_flags & anydex::telemetry::kHandTargetsValid) != 0);
  body["current_mA"] = optional_list(
      sample.current_mA,
      (sample.validity_flags & anydex::telemetry::kHandCurrentValid) != 0);
  body["force_g"] = optional_list(
      sample.force_g,
      (sample.validity_flags & anydex::telemetry::kHandForceValid) != 0);
  body["temperature_c"] = optional_list(
      sample.temperature_c,
      (sample.validity_flags & anydex::telemetry::kHandTemperatureValid) != 0);
  body["status"] = optional_list(
      sample.status,
      (sample.validity_flags & anydex::telemetry::kHandStatusValid) != 0);
  body["errors"] = optional_list(
      sample.errors,
      (sample.validity_flags & anydex::telemetry::kHandErrorsValid) != 0);
  output["sample"] = std::move(body);
  return output;
}

py::dict initialize_mapping(const std::string& path,
                            const py::dict& provenance) {
  InitError error{};
  auto mapping =
      TelemetryMapping::Create(path, parse_provenance(provenance), &error);
  if (mapping == nullptr) {
    throw_init_error("initialize_mapping", error);
  }
  return header_mapping(mapping->header());
}

class PyTelemetryReader final {
 public:
  static std::unique_ptr<PyTelemetryReader> OpenReadOnly(
      const std::string& path) {
    InitError error{};
    auto mapping = TelemetryMapping::OpenReadOnly(path, &error);
    if (mapping == nullptr) {
      throw_init_error("open_read_only", error);
    }
    return std::unique_ptr<PyTelemetryReader>(
        new PyTelemetryReader(std::move(mapping)));
  }

  py::dict header() const { return header_mapping(mapping_->header()); }

  py::dict read_arm(const std::uint32_t max_attempts) const {
    ArmSample sample{};
    return arm_read_mapping(mapping_->reader().read_arm(sample, max_attempts),
                            sample);
  }

  py::dict read_hand(const std::uint32_t max_attempts) const {
    HandSample sample{};
    return hand_read_mapping(mapping_->reader().read_hand(sample, max_attempts),
                             sample);
  }

 private:
  explicit PyTelemetryReader(std::unique_ptr<TelemetryMapping> mapping)
      : mapping_(std::move(mapping)) {}
  std::unique_ptr<TelemetryMapping> mapping_;
};

class PyTelemetryTestSession final {
 public:
  static std::unique_ptr<PyTelemetryTestSession> Create(
      const std::string& path, const py::dict& provenance,
      const std::uint64_t arm_writer_token,
      const std::uint64_t hand_writer_token) {
    InitError error{};
    auto mapping =
        TelemetryMapping::Create(path, parse_provenance(provenance), &error);
    if (mapping == nullptr) {
      throw_init_error("create", error);
    }
    auto arm_writer = mapping->ClaimArmWriter(arm_writer_token, &error);
    if (arm_writer == nullptr) {
      throw_init_error("claim arm test writer", error);
    }
    auto hand_writer = mapping->ClaimHandWriter(hand_writer_token, &error);
    if (hand_writer == nullptr) {
      throw_init_error("claim hand test writer", error);
    }
    return std::unique_ptr<PyTelemetryTestSession>(new PyTelemetryTestSession(
        std::move(mapping), std::move(arm_writer), std::move(hand_writer)));
  }

  py::dict header() const { return header_mapping(mapping_->header()); }

  py::dict read_arm(const std::uint32_t max_attempts) const {
    ArmSample sample{};
    return arm_read_mapping(mapping_->reader().read_arm(sample, max_attempts),
                            sample);
  }

  py::dict read_hand(const std::uint32_t max_attempts) const {
    HandSample sample{};
    return hand_read_mapping(mapping_->reader().read_hand(sample, max_attempts),
                             sample);
  }

  py::dict publish_test_arm(const py::dict& value) {
    ArmSample sample{};
    sample.timestamp_monotonic_ns =
        py::cast<std::uint64_t>(value["timestamp_monotonic_ns"]);
    sample.timestamp_unix_ns =
        py::cast<std::uint64_t>(value["timestamp_unix_ns"]);
    sample.stage_epoch = py::cast<std::uint64_t>(value["stage_epoch"]);
    sample.bundle_sequence = value.contains("bundle_sequence")
                                 ? py::cast<std::uint64_t>(
                                       value["bundle_sequence"])
                                 : 0;
    const std::string stage_name = py::cast<std::string>(value["stage_name"]);
    if (!anydex::telemetry::set_stage_name(
            sample.stage_name, stage_name.data(), stage_name.size())) {
      throw py::value_error("stage_name must be valid, non-empty UTF-8 <=31 bytes");
    }
    sample.stage_name_hash64 = anydex::telemetry::stage_name_hash(
        sample.stage_name, stage_name.size());
    sample.validity_flags = anydex::telemetry::kArmTimestampValid |
                            anydex::telemetry::kArmPoseValid;
    sample.source = Source::kSyntheticTest;
    sample.measurement_kind = MeasurementKind::kSyntheticTest;
    copy_numeric_sequence(value["O_T_EE"], sample.O_T_EE, "O_T_EE");
    if (value.contains("q") && !value["q"].is_none()) {
      copy_numeric_sequence(value["q"], sample.q, "q");
      sample.validity_flags |= anydex::telemetry::kArmQValid;
    }
    if (value.contains("dq") && !value["dq"].is_none()) {
      copy_numeric_sequence(value["dq"], sample.dq, "dq");
      sample.validity_flags |= anydex::telemetry::kArmDqValid;
    }
    if (value.contains("control_command_success_rate") &&
        !value["control_command_success_rate"].is_none()) {
      sample.control_command_success_rate =
          py::cast<double>(value["control_command_success_rate"]);
      sample.validity_flags |= anydex::telemetry::kArmSuccessRateValid;
    }
    if (value.contains("producer_cycle")) {
      sample.producer_cycle =
          py::cast<std::uint64_t>(value["producer_cycle"]);
    }
    const auto result = arm_writer_->publish(sample);
    py::dict output;
    output["code"] = static_cast<std::uint32_t>(result.code);
    output["sequence"] = result.sequence;
    return output;
  }

  py::dict publish_test_hand(const py::dict& value) {
    HandSample sample{};
    sample.timestamp_monotonic_ns =
        py::cast<std::uint64_t>(value["timestamp_monotonic_ns"]);
    sample.timestamp_unix_ns =
        py::cast<std::uint64_t>(value["timestamp_unix_ns"]);
    sample.stage_epoch = py::cast<std::uint64_t>(value["stage_epoch"]);
    sample.bundle_sequence = value.contains("bundle_sequence")
                                 ? py::cast<std::uint64_t>(
                                       value["bundle_sequence"])
                                 : 0;
    const std::string stage_name = py::cast<std::string>(value["stage_name"]);
    if (!anydex::telemetry::set_stage_name(
            sample.stage_name, stage_name.data(), stage_name.size())) {
      throw py::value_error("stage_name must be valid, non-empty UTF-8 <=31 bytes");
    }
    sample.stage_name_hash64 = anydex::telemetry::stage_name_hash(
        sample.stage_name, stage_name.size());
    sample.validity_flags = anydex::telemetry::kHandTimestampValid |
                            anydex::telemetry::kHandAnglesValid;
    sample.source = Source::kSyntheticTest;
    sample.measurement_kind = MeasurementKind::kSyntheticTest;
    copy_numeric_sequence(value["angles"], sample.angles, "angles");
    if (value.contains("angle_targets") &&
        !value["angle_targets"].is_none()) {
      copy_numeric_sequence(value["angle_targets"], sample.angle_targets,
                            "angle_targets");
      sample.validity_flags |= anydex::telemetry::kHandTargetsValid;
    }
    if (value.contains("current_mA") && !value["current_mA"].is_none()) {
      copy_numeric_sequence(value["current_mA"], sample.current_mA,
                            "current_mA");
      sample.validity_flags |= anydex::telemetry::kHandCurrentValid;
    }
    if (value.contains("force_g") && !value["force_g"].is_none()) {
      copy_numeric_sequence(value["force_g"], sample.force_g, "force_g");
      sample.validity_flags |= anydex::telemetry::kHandForceValid;
    }
    if (value.contains("temperature_c") &&
        !value["temperature_c"].is_none()) {
      copy_numeric_sequence(value["temperature_c"], sample.temperature_c,
                            "temperature_c");
      sample.validity_flags |= anydex::telemetry::kHandTemperatureValid;
    }
    if (value.contains("status") && !value["status"].is_none()) {
      copy_u8_sequence(value["status"], sample.status, "status");
      sample.validity_flags |= anydex::telemetry::kHandStatusValid;
    }
    if (value.contains("errors") && !value["errors"].is_none()) {
      copy_u8_sequence(value["errors"], sample.errors, "errors");
      sample.validity_flags |= anydex::telemetry::kHandErrorsValid;
    }
    if (value.contains("producer_poll")) {
      sample.producer_poll =
          py::cast<std::uint64_t>(value["producer_poll"]);
    }
    const auto result = hand_writer_->publish(sample);
    py::dict output;
    output["code"] = static_cast<std::uint32_t>(result.code);
    output["sequence"] = result.sequence;
    return output;
  }

 private:
  PyTelemetryTestSession(std::unique_ptr<TelemetryMapping> mapping,
                         std::unique_ptr<ArmWriter> arm_writer,
                         std::unique_ptr<HandWriter> hand_writer)
      : mapping_(std::move(mapping)),
        arm_writer_(std::move(arm_writer)),
        hand_writer_(std::move(hand_writer)) {}

  // Declaration order is intentional: writers are destroyed before mapping.
  std::unique_ptr<TelemetryMapping> mapping_;
  std::unique_ptr<ArmWriter> arm_writer_;
  std::unique_ptr<HandWriter> hand_writer_;
};

}  // namespace

PYBIND11_MODULE(_anydex_telemetry, module) {
  module.doc() =
      "Offline-buildable lock-free telemetry reader and synthetic-test publisher; "
      "contains no Franka, serial, camera, or control API";
  module.attr("ABI_MAJOR") = anydex::telemetry::kAbiMajor;
  module.attr("ABI_MINOR") = anydex::telemetry::kAbiMinor;
  module.attr("ABI_SCHEMA_SHA256") = encode_hex(
      anydex::telemetry::kAbiSchemaSha256.data(),
      anydex::telemetry::kAbiSchemaSha256.size());
  module.attr("READ_OK") = static_cast<std::uint32_t>(ReadCode::kOk);
  module.attr("READ_NO_DATA") = static_cast<std::uint32_t>(ReadCode::kNoData);
  module.attr("READ_CONTENDED") =
      static_cast<std::uint32_t>(ReadCode::kContended);
  module.def("platform_is_supported_lock_free",
             &TelemetryMapping::platform_is_supported_lock_free);
  module.def("initialize_mapping", &initialize_mapping, py::arg("path"),
             py::arg("provenance"),
             "Create one ready mapping without claiming or publishing a stream");

  py::class_<PyTelemetryReader>(module, "TelemetryReader")
      .def_static("open_read_only", &PyTelemetryReader::OpenReadOnly,
                  py::arg("path"))
      .def("header", &PyTelemetryReader::header)
      .def("read_arm", &PyTelemetryReader::read_arm,
           py::arg("max_attempts") = anydex::telemetry::kDefaultReadAttempts)
      .def("read_hand", &PyTelemetryReader::read_hand,
           py::arg("max_attempts") = anydex::telemetry::kDefaultReadAttempts);

  py::class_<PyTelemetryTestSession>(module, "TelemetryTestSession")
      .def_static("create", &PyTelemetryTestSession::Create, py::arg("path"),
                  py::arg("provenance"), py::arg("arm_writer_token"),
                  py::arg("hand_writer_token"))
      .def("header", &PyTelemetryTestSession::header)
      .def("read_arm", &PyTelemetryTestSession::read_arm,
           py::arg("max_attempts") = anydex::telemetry::kDefaultReadAttempts)
      .def("read_hand", &PyTelemetryTestSession::read_hand,
           py::arg("max_attempts") = anydex::telemetry::kDefaultReadAttempts)
      .def("publish_test_arm", &PyTelemetryTestSession::publish_test_arm,
           py::arg("sample"))
      .def("publish_test_hand", &PyTelemetryTestSession::publish_test_hand,
           py::arg("sample"));
}
