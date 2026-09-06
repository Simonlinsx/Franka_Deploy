#include "anydex/v94_franka_servo/protocol.hpp"

#include <cstring>

namespace anydex::v94_franka_servo {
namespace {

constexpr std::size_t kCrcOffset = offsetof(PacketHeader, crc32);

bool nonce_equal(
    const std::uint8_t* actual,
    const std::array<std::uint8_t, kSessionNonceBytes>& expected) noexcept {
  std::uint8_t difference = 0U;
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    difference = static_cast<std::uint8_t>(difference |
                                           (actual[index] ^ expected[index]));
  }
  return difference == 0U;
}

}  // namespace

bool host_is_little_endian() noexcept {
  const std::uint16_t value = 1U;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1U;
}

bool message_kind_is_known(const MessageKind kind) noexcept {
  return expected_payload_size(kind) != 0U;
}

std::size_t expected_payload_size(const MessageKind kind) noexcept {
  switch (kind) {
    case MessageKind::kArm:
      return sizeof(ArmPayload);
    case MessageKind::kHeartbeat:
      return sizeof(HeartbeatPayload);
    case MessageKind::kTarget:
      return sizeof(TargetPayload);
    case MessageKind::kStop:
      return sizeof(StopPayload);
    case MessageKind::kHello:
      return sizeof(HelloPayload);
    case MessageKind::kIpcReady:
      return sizeof(IpcReadyPayload);
    case MessageKind::kState:
      return sizeof(StatePayload);
    case MessageKind::kAck:
      return sizeof(AckPayload);
    case MessageKind::kFault:
      return sizeof(FaultPayload);
    case MessageKind::kStopProof:
      return sizeof(StopProofPayload);
    case MessageKind::kActionReady:
      return sizeof(ActionReadyPayload);
  }
  return 0U;
}

std::uint32_t protocol_crc32(const std::uint8_t* data,
                             const std::size_t size) noexcept {
  if (data == nullptr && size != 0U) {
    return 0U;
  }
  std::uint32_t crc = 0xFFFFFFFFU;
  for (std::size_t index = 0U; index < size; ++index) {
    crc ^= data[index];
    for (std::uint32_t bit = 0U; bit < 8U; ++bit) {
      const std::uint32_t mask =
          static_cast<std::uint32_t>(-(static_cast<std::int32_t>(crc & 1U)));
      crc = (crc >> 1U) ^ (0xEDB88320U & mask);
    }
  }
  return ~crc;
}

CodecError encode_packet(
    const MessageKind kind,
    const std::uint64_t packet_sequence,
    const std::uint64_t monotonic_ns,
    const std::array<std::uint8_t, kSessionNonceBytes>& nonce,
    const void* payload,
    const std::size_t payload_size,
    EncodedPacket* output) noexcept {
  if (!host_is_little_endian()) {
    return CodecError::kHostNotLittleEndian;
  }
  if (!message_kind_is_known(kind)) {
    return CodecError::kInvalidKind;
  }
  if (payload == nullptr || output == nullptr || packet_sequence == 0U ||
      payload_size != expected_payload_size(kind)) {
    return CodecError::kPayloadSize;
  }
  const std::size_t packet_size = sizeof(PacketHeader) + payload_size;
  if (packet_size > output->bytes.size()) {
    return CodecError::kPacketTooLarge;
  }

  PacketHeader header{};
  header.magic = kProtocolMagic;
  header.version = kProtocolVersion;
  header.kind = static_cast<std::uint16_t>(kind);
  header.payload_bytes = static_cast<std::uint32_t>(payload_size);
  header.flags = 0U;
  header.packet_sequence = packet_sequence;
  header.monotonic_ns = monotonic_ns;
  std::memcpy(header.session_nonce, nonce.data(), nonce.size());
  header.crc32 = 0U;
  header.reserved = 0U;
  std::memcpy(output->bytes.data(), &header, sizeof(header));
  std::memcpy(output->bytes.data() + sizeof(header), payload, payload_size);
  header.crc32 = protocol_crc32(output->bytes.data(), packet_size);
  std::memcpy(output->bytes.data() + kCrcOffset, &header.crc32,
              sizeof(header.crc32));
  output->size = packet_size;
  return CodecError::kNone;
}

CodecError decode_packet(
    const std::uint8_t* packet,
    const std::size_t packet_size,
    const std::uint64_t expected_packet_sequence,
    const std::array<std::uint8_t, kSessionNonceBytes>& expected_nonce,
    DecodedPacket* output) noexcept {
  if (!host_is_little_endian()) {
    return CodecError::kHostNotLittleEndian;
  }
  if (packet == nullptr || output == nullptr || packet_size < sizeof(PacketHeader)) {
    return CodecError::kTruncated;
  }
  PacketHeader header{};
  std::memcpy(&header, packet, sizeof(header));
  if (header.magic != kProtocolMagic) {
    return CodecError::kMagic;
  }
  if (header.version != kProtocolVersion) {
    return CodecError::kVersion;
  }
  const MessageKind kind = static_cast<MessageKind>(header.kind);
  if (!message_kind_is_known(kind)) {
    return CodecError::kInvalidKind;
  }
  if (header.flags != 0U) {
    return CodecError::kFlags;
  }
  if (header.reserved != 0U) {
    return CodecError::kReserved;
  }
  if (header.payload_bytes != expected_payload_size(kind)) {
    return CodecError::kPayloadSize;
  }
  if (packet_size != sizeof(PacketHeader) + header.payload_bytes) {
    return CodecError::kTruncated;
  }
  if (header.packet_sequence != expected_packet_sequence ||
      expected_packet_sequence == 0U) {
    return CodecError::kSequence;
  }
  if (!nonce_equal(header.session_nonce, expected_nonce)) {
    return CodecError::kNonce;
  }

  std::array<std::uint8_t, kMaximumPacketBytes> copy{};
  if (packet_size > copy.size()) {
    return CodecError::kPacketTooLarge;
  }
  std::memcpy(copy.data(), packet, packet_size);
  std::memset(copy.data() + kCrcOffset, 0, sizeof(header.crc32));
  if (protocol_crc32(copy.data(), packet_size) != header.crc32) {
    return CodecError::kCrc;
  }

  output->header = header;
  output->payload = packet + sizeof(PacketHeader);
  output->payload_size = header.payload_bytes;
  return CodecError::kNone;
}

CodecError decode_hello_packet(
    const std::uint8_t* packet,
    const std::size_t packet_size,
    DecodedPacket* output,
    std::array<std::uint8_t, kSessionNonceBytes>* adopted_nonce) noexcept {
  if (packet == nullptr || output == nullptr || adopted_nonce == nullptr ||
      packet_size < sizeof(PacketHeader)) {
    return CodecError::kTruncated;
  }
  PacketHeader header{};
  std::memcpy(&header, packet, sizeof(header));
  std::array<std::uint8_t, kSessionNonceBytes> nonce{};
  std::memcpy(nonce.data(), header.session_nonce, nonce.size());
  const CodecError decoded = decode_packet(packet, packet_size, 1U, nonce, output);
  if (decoded != CodecError::kNone) {
    return decoded;
  }
  if (output->header.kind !=
      static_cast<std::uint16_t>(MessageKind::kHello)) {
    return CodecError::kInvalidKind;
  }
  *adopted_nonce = nonce;
  return CodecError::kNone;
}

const char* codec_error_name(const CodecError error) noexcept {
  switch (error) {
    case CodecError::kNone:
      return "none";
    case CodecError::kHostNotLittleEndian:
      return "host_not_little_endian";
    case CodecError::kInvalidKind:
      return "invalid_kind";
    case CodecError::kPayloadSize:
      return "payload_size";
    case CodecError::kPacketTooLarge:
      return "packet_too_large";
    case CodecError::kTruncated:
      return "truncated";
    case CodecError::kMagic:
      return "magic";
    case CodecError::kVersion:
      return "version";
    case CodecError::kFlags:
      return "flags";
    case CodecError::kReserved:
      return "reserved";
    case CodecError::kSequence:
      return "sequence";
    case CodecError::kNonce:
      return "nonce";
    case CodecError::kCrc:
      return "crc";
  }
  return "unknown";
}

}  // namespace anydex::v94_franka_servo
