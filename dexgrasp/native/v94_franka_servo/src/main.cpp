#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/prctl.h>
#include <sys/random.h>
#include <sys/types.h>
#include <unistd.h>

#include "anydex/v94_franka_servo/channel.hpp"
#include "anydex/v94_franka_servo/dependency_identity.hpp"
#include "anydex/v94_franka_servo/libfranka_backend.hpp"
#include "anydex/v94_franka_servo/safety_limits.hpp"
#include "anydex/v94_franka_servo/servo_core.hpp"

#ifndef ANYDEX_LIBFRANKA_LIBRARY_SHA256
#error "ANYDEX_LIBFRANKA_LIBRARY_SHA256 is required"
#endif
#ifndef ANYDEX_LIBFRANKA_SOURCE_COMMIT
#error "ANYDEX_LIBFRANKA_SOURCE_COMMIT is required"
#endif
#ifndef ANYDEX_PRODUCER_BUILD_SHA256
#error "ANYDEX_PRODUCER_BUILD_SHA256 is required"
#endif

namespace servo = anydex::v94_franka_servo;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

extern "C" void request_stop(const int) { g_stop_requested = 1; }

struct Arguments final {
  std::string robot_ip;
  int critical_fd{-1};
  int telemetry_fd{-1};
  pid_t parent_pid{0};
};

long parse_decimal(const std::string& text, const char* name) {
  if (text.empty()) {
    throw std::invalid_argument(std::string(name) + " is empty");
  }
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0') {
    throw std::invalid_argument(std::string(name) + " is not an exact decimal");
  }
  return value;
}

Arguments parse_arguments(const int argc, char** argv) {
  Arguments output{};
  bool execute = false;
  bool robot = false;
  bool critical = false;
  bool telemetry = false;
  bool parent = false;
  for (int index = 1; index < argc; ++index) {
    const std::string flag(argv[index]);
    if (flag == "--execute-supervised-v94") {
      if (execute) {
        throw std::invalid_argument("duplicate execute sentinel");
      }
      execute = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("option is missing its value: " + flag);
    }
    const std::string value(argv[++index]);
    if (flag == "--robot-ip" && !robot) {
      output.robot_ip = value;
      robot = true;
    } else if (flag == "--critical-fd" && !critical) {
      const long parsed = parse_decimal(value, "critical fd");
      if (parsed < 3 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("critical fd is outside 3..INT_MAX");
      }
      output.critical_fd = static_cast<int>(parsed);
      critical = true;
    } else if (flag == "--telemetry-fd" && !telemetry) {
      const long parsed = parse_decimal(value, "telemetry fd");
      if (parsed < 3 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("telemetry fd is outside 3..INT_MAX");
      }
      output.telemetry_fd = static_cast<int>(parsed);
      telemetry = true;
    } else if (flag == "--parent-pid" && !parent) {
      const long parsed = parse_decimal(value, "parent pid");
      if (parsed <= 1 || parsed > std::numeric_limits<pid_t>::max()) {
        throw std::invalid_argument("parent pid is outside 2..PID_MAX");
      }
      output.parent_pid = static_cast<pid_t>(parsed);
      parent = true;
    } else {
      throw std::invalid_argument("unknown or duplicate option: " + flag);
    }
  }
  if (!execute || !robot || !critical || !telemetry || !parent ||
      output.robot_ip.empty() || output.critical_fd == output.telemetry_fd) {
    throw std::invalid_argument("incomplete supervised V94 invocation");
  }
  return output;
}

std::uint8_t nibble(const char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  throw std::invalid_argument("compiled digest is not lowercase hexadecimal");
}

template <std::size_t N>
std::array<std::uint8_t, N> decode_hex(const char* text) {
  std::array<std::uint8_t, N> output{};
  for (std::size_t index = 0U; index < N; ++index) {
    output[index] = static_cast<std::uint8_t>(
        (nibble(text[index * 2U]) << 4U) | nibble(text[index * 2U + 1U]));
  }
  if (text[N * 2U] != '\0') {
    throw std::invalid_argument("compiled digest has the wrong length");
  }
  return output;
}

std::array<std::uint8_t, servo::kSessionNonceBytes> random_nonce() {
  std::array<std::uint8_t, servo::kSessionNonceBytes> output{};
  std::size_t offset = 0U;
  while (offset < output.size()) {
    const ssize_t count =
        ::getrandom(output.data() + offset, output.size() - offset, 0U);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      throw std::runtime_error("getrandom failed for session nonce");
    }
    offset += static_cast<std::size_t>(count);
  }
  return output;
}

void install_process_guards(const pid_t expected_parent) {
  if (::getppid() != expected_parent) {
    throw std::runtime_error("actual parent PID differs from --parent-pid");
  }
  if (::prctl(PR_SET_PDEATHSIG, SIGTERM) != 0) {
    throw std::runtime_error("PR_SET_PDEATHSIG failed");
  }
  if (::getppid() != expected_parent) {
    throw std::runtime_error("parent exited while installing PDEATHSIG");
  }
  struct sigaction action {};
  action.sa_handler = request_stop;
  ::sigemptyset(&action.sa_mask);
  if (::sigaction(SIGINT, &action, nullptr) != 0 ||
      ::sigaction(SIGTERM, &action, nullptr) != 0) {
    throw std::runtime_error("signal handler installation failed");
  }
  struct sigaction ignore {};
  ignore.sa_handler = SIG_IGN;
  ::sigemptyset(&ignore.sa_mask);
  if (::sigaction(SIGPIPE, &ignore, nullptr) != 0) {
    throw std::runtime_error("SIGPIPE ignore installation failed");
  }
}

servo::HelloPayload make_hello() {
  servo::HelloPayload hello{};
  hello.process_id = static_cast<std::uint32_t>(::getpid());
  hello.state_decimation = servo::HardSafetyLimits::kStateDecimation;
  hello.protocol_version = servo::kProtocolVersion;
  hello.safety_limits_schema = 3U;
  const auto library =
      decode_hex<servo::kSha256Bytes>(ANYDEX_LIBFRANKA_LIBRARY_SHA256);
  const auto source =
      decode_hex<servo::kSha1Bytes>(ANYDEX_LIBFRANKA_SOURCE_COMMIT);
  const auto build =
      decode_hex<servo::kSha256Bytes>(ANYDEX_PRODUCER_BUILD_SHA256);
  std::memcpy(hello.libfranka_sha256, library.data(), library.size());
  std::memcpy(hello.libfranka_source_commit, source.data(), source.size());
  std::memcpy(hello.producer_build_sha256, build.data(), build.size());
  hello.maximum_command_velocity_rad_s =
      servo::HardSafetyLimits::kMaximumCommandVelocityRadS;
  hello.maximum_command_acceleration_rad_s2 =
      servo::HardSafetyLimits::kMaximumCommandAccelerationRadS2;
  hello.maximum_command_jerk_rad_s3 =
      servo::HardSafetyLimits::kMaximumCommandJerkRadS3;
  hello.maximum_start_error_rad =
      servo::HardSafetyLimits::kMaximumStartErrorRad;
  hello.maximum_tick_target_delta_rad =
      servo::HardSafetyLimits::kMaximumTickTargetDeltaRad;
  hello.maximum_episode_delta_rad =
      servo::HardSafetyLimits::kMaximumEpisodeDeltaRad;
  hello.maximum_tracking_error_rad =
      servo::HardSafetyLimits::kMaximumTrackingErrorRad;
  hello.maximum_read_to_write_s =
      static_cast<double>(servo::HardSafetyLimits::kMaximumReadToWriteNs) *
      1.0e-9;
  return hello;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    install_process_guards(arguments.parent_pid);
    std::string dependency_detail;
    if (!servo::verify_linked_libfranka_identity(&dependency_detail)) {
      throw std::runtime_error("libfranka identity rejected: " +
                               dependency_detail);
    }
    servo::ServoProcessConfig config{};
    config.robot_address = arguments.robot_ip;
    config.expected_parent_pid = arguments.parent_pid;
    config.session_nonce = random_nonce();
    config.hello = make_hello();
    config.signal_stop_requested = &g_stop_requested;
    config.require_realtime_scheduler_proof = true;
    servo::InheritedChannels channels(arguments.critical_fd,
                                       arguments.telemetry_fd);
    servo::LibfrankaBackendFactory factory;
    servo::PosixServoClock clock;
    const servo::ServoRunResult result =
        servo::run_servo_process(config, &channels, &factory, &clock);
    ::close(arguments.critical_fd);
    ::close(arguments.telemetry_fd);
    if (result.exit_code != 0) {
      std::cerr << "V94 native servo terminal fault="
                << static_cast<std::uint32_t>(result.terminal_fault) << '\n';
    }
    return result.exit_code;
  } catch (const std::exception& error) {
    std::cerr << "V94 native servo refused to start: " << error.what() << '\n';
    return 2;
  }
}
