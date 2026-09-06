#include "anydex/v94_franka_servo/protocol.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace servo = anydex::v94_franka_servo;

int main() {
  assert(servo::host_is_little_endian());
  std::array<std::uint8_t, servo::kSessionNonceBytes> nonce{};
  for (std::size_t index = 0; index < nonce.size(); ++index) {
    nonce[index] = static_cast<std::uint8_t>(index + 1U);
  }

  servo::TargetPayload target{};
  target.target_sequence = 7U;
  target.observation_sequence = 19U;
  target.produced_monotonic_ns = 123456789U;
  for (std::size_t index = 0U; index < 7U; ++index) {
    target.target_q_rad[index] = 0.125 * static_cast<double>(index + 1U);
  }
  servo::EncodedPacket encoded{};
  assert(servo::encode_payload(servo::MessageKind::kTarget, 3U, 999U, nonce,
                               target, &encoded) == servo::CodecError::kNone);
  assert(encoded.size == sizeof(servo::PacketHeader) + sizeof(target));

  servo::DecodedPacket decoded{};
  assert(servo::decode_packet(encoded.bytes.data(), encoded.size, 3U, nonce,
                              &decoded) == servo::CodecError::kNone);
  assert(decoded.header.kind ==
         static_cast<std::uint16_t>(servo::MessageKind::kTarget));
  servo::TargetPayload round_trip{};
  assert(servo::copy_payload(decoded, &round_trip) == servo::CodecError::kNone);
  assert(round_trip.target_sequence == target.target_sequence);
  assert(round_trip.observation_sequence == target.observation_sequence);
  assert(std::memcmp(round_trip.target_q_rad, target.target_q_rad,
                     sizeof(target.target_q_rad)) == 0);

  servo::ArmPayload arm{};
  arm.controller_mode =
      static_cast<std::uint32_t>(servo::ControllerMode::kQdG015);
  servo::EncodedPacket encoded_arm{};
  assert(servo::encode_payload(servo::MessageKind::kArm, 4U, 1000U, nonce,
                               arm, &encoded_arm) == servo::CodecError::kNone);
  assert(servo::decode_packet(encoded_arm.bytes.data(), encoded_arm.size, 4U,
                              nonce, &decoded) == servo::CodecError::kNone);
  servo::ArmPayload round_trip_arm{};
  assert(servo::copy_payload(decoded, &round_trip_arm) ==
         servo::CodecError::kNone);
  assert(round_trip_arm.controller_mode ==
         static_cast<std::uint32_t>(servo::ControllerMode::kQdG015));

  servo::StatePayload state{};
  for (std::size_t joint = 0U; joint < 7U; ++joint) {
    state.commanded_q_rad[joint] = 0.01 * static_cast<double>(joint);
    state.shaper_q_d_rad[joint] = state.commanded_q_rad[joint];
    state.shaper_dq_d_rad_s[joint] = 0.02 * static_cast<double>(joint);
    state.shaper_ddq_d_rad_s2[joint] = 0.03 * static_cast<double>(joint);
    state.held_q_cmd_rad[joint] = 0.04 * static_cast<double>(joint);
  }
  state.controller_state29[28] = 1.0F;
  servo::EncodedPacket encoded_state{};
  assert(servo::encode_payload(servo::MessageKind::kState, 5U, 1001U, nonce,
                               state, &encoded_state) ==
         servo::CodecError::kNone);
  assert(encoded_state.size == sizeof(servo::PacketHeader) + 764U);
  assert(servo::decode_packet(encoded_state.bytes.data(), encoded_state.size, 5U,
                              nonce, &decoded) == servo::CodecError::kNone);
  servo::StatePayload round_trip_state{};
  assert(servo::copy_payload(decoded, &round_trip_state) ==
         servo::CodecError::kNone);
  assert(std::memcmp(round_trip_state.shaper_q_d_rad,
                     state.shaper_q_d_rad,
                     sizeof(state.shaper_q_d_rad)) == 0);
  assert(std::memcmp(round_trip_state.held_q_cmd_rad, state.held_q_cmd_rad,
                     sizeof(state.held_q_cmd_rad)) == 0);

  auto corrupted = encoded;
  corrupted.bytes[sizeof(servo::PacketHeader) + 5U] ^= 0x80U;
  assert(servo::decode_packet(corrupted.bytes.data(), corrupted.size, 3U, nonce,
                              &decoded) == servo::CodecError::kCrc);
  assert(servo::decode_packet(encoded.bytes.data(), encoded.size, 4U, nonce,
                              &decoded) == servo::CodecError::kSequence);
  auto wrong_nonce = nonce;
  wrong_nonce[0] ^= 1U;
  assert(servo::decode_packet(encoded.bytes.data(), encoded.size, 3U,
                              wrong_nonce, &decoded) ==
         servo::CodecError::kNonce);
  assert(servo::decode_packet(encoded.bytes.data(), encoded.size - 1U, 3U,
                              nonce, &decoded) == servo::CodecError::kTruncated);

  servo::HelloPayload hello{};
  hello.process_id = 42U;
  servo::EncodedPacket hello_packet{};
  assert(servo::encode_payload(servo::MessageKind::kHello, 1U, 5U, nonce,
                               hello, &hello_packet) == servo::CodecError::kNone);
  std::array<std::uint8_t, servo::kSessionNonceBytes> adopted_nonce{};
  assert(servo::decode_hello_packet(hello_packet.bytes.data(), hello_packet.size,
                                    &decoded, &adopted_nonce) ==
         servo::CodecError::kNone);
  assert(adopted_nonce == nonce);

  servo::EncodedPacket wrong_kind{};
  assert(servo::encode_payload(servo::MessageKind::kAck, 1U, 1U, nonce, target,
                               &wrong_kind) ==
         servo::CodecError::kPayloadSize);

  // Cross-language ABI fixture. The peer must reproduce the complete packet,
  // including the CRC calculated with the header CRC bytes zeroed.
  std::array<std::uint8_t, servo::kSessionNonceBytes> golden_nonce{};
  for (std::size_t index = 0U; index < golden_nonce.size(); ++index) {
    golden_nonce[index] = static_cast<std::uint8_t>(index);
  }
  servo::TargetPayload golden_target{};
  golden_target.target_sequence = 0x2122232425262728ULL;
  golden_target.observation_sequence = 0x3132333435363738ULL;
  golden_target.produced_monotonic_ns = 0x4142434445464748ULL;
  const std::array<double, 7> golden_q{0.0, 1.0, -1.0, 0.5,
                                       -0.5, 3.25, -2.75};
  std::copy(golden_q.begin(), golden_q.end(), golden_target.target_q_rad);
  servo::EncodedPacket golden_packet{};
  assert(servo::encode_payload(
             servo::MessageKind::kTarget, 0x0102030405060708ULL,
             0x1112131415161718ULL, golden_nonce, golden_target,
             &golden_packet) == servo::CodecError::kNone);
  std::ostringstream golden_hex;
  golden_hex << std::hex << std::setfill('0');
  for (std::size_t index = 0U; index < golden_packet.size; ++index) {
    golden_hex << std::setw(2)
               << static_cast<unsigned int>(golden_packet.bytes[index]);
  }
  const std::string expected_golden_hex =
      "5639344604000300500000000000000008070605040302011817161514131211"
      "000102030405060708090a0b0c0d0e0f35311ec4000000002827262524232221"
      "383736353433323148474645444342410000000000000000000000000000f03f"
      "000000000000f0bf000000000000e03f000000000000e0bf0000000000000a40"
      "00000000000006c0";
  assert(golden_hex.str() == expected_golden_hex);
  std::cout << "GOLDEN_TARGET_PACKET_HEX=" << golden_hex.str() << '\n';

  std::cout << "v94 servo protocol fixed-ABI/CRC tests passed\n";
  return 0;
}
