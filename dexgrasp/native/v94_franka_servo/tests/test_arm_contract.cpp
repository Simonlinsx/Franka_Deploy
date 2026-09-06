#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <vector>

// This test deliberately compiles the production implementation into this
// translation unit.  require_arm_contract is intentionally private to
// servo_core.cpp; including it here lets the cross-language test exercise that
// exact function without adding a test hook to the production API.
#include "../src/servo_core.cpp"

namespace servo = anydex::v94_franka_servo;

namespace anydex::v94_franka_servo {

void require_python_arm_packet_contract(const std::vector<std::uint8_t>& packet) {
  if (packet.size() != sizeof(PacketHeader) + sizeof(ArmPayload)) {
    throw std::runtime_error("Python ARM packet has the wrong exact size");
  }

  PacketHeader wire_header{};
  std::memcpy(&wire_header, packet.data(), sizeof(wire_header));
  std::array<std::uint8_t, kSessionNonceBytes> nonce{};
  std::copy_n(wire_header.session_nonce, nonce.size(), nonce.begin());

  DecodedPacket decoded{};
  const CodecError error =
      decode_packet(packet.data(), packet.size(), wire_header.packet_sequence,
                    nonce, &decoded);
  if (error != CodecError::kNone) {
    throw std::runtime_error("C++ rejected the Python ARM packet framing/CRC");
  }
  if (decoded.header.kind != static_cast<std::uint16_t>(MessageKind::kArm)) {
    throw std::runtime_error("Python packet is not ARM");
  }

  ArmPayload arm{};
  if (copy_payload(decoded, &arm) != CodecError::kNone) {
    throw std::runtime_error("C++ could not decode the Python ARM payload");
  }
  const std::uint64_t active_time_ns =
      arm.authorization_issued_monotonic_ns +
      (arm.authorization_expires_monotonic_ns -
       arm.authorization_issued_monotonic_ns) /
          2U;
  require_arm_contract(arm, active_time_ns);

  std::array<double, 7> previous_target{};
  std::array<double, 7> qd_g015_target{};
  qd_g015_target[0] = 0.060;
  if (previous_target_delta_allowed(ControllerMode::kLegacy, qd_g015_target,
                                    previous_target)) {
    throw std::runtime_error(
        "legacy controller mode failed to retain the 0.020 target guard");
  }
  if (!previous_target_delta_allowed(ControllerMode::kQdG015,
                                     qd_g015_target, previous_target)) {
    throw std::runtime_error(
        "qd_g015 controller mode incorrectly retained the previous-target guard");
  }
}

}  // namespace anydex::v94_franka_servo

int main() {
  try {
    const std::vector<std::uint8_t> packet{
        std::istreambuf_iterator<char>(std::cin),
        std::istreambuf_iterator<char>()};
    servo::require_python_arm_packet_contract(packet);
    std::cout << "production Python ARM accepted by C++ require_arm_contract\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
