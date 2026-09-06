#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>

#include "anydex/v94_franka_servo/protocol.hpp"

namespace anydex::v94_franka_servo {

enum class IoStatus : std::uint32_t {
  kOk = 0U,
  kWouldBlock,
  kPeerClosed,
  kTruncated,
  kError,
};

struct ReceiveResult final {
  IoStatus status{IoStatus::kError};
  std::array<std::uint8_t, kMaximumPacketBytes> bytes{};
  std::size_t size{0U};
  int system_errno{0};
};

struct ChannelValidation final {
  bool valid{false};
  int system_errno{0};
  pid_t critical_peer_pid{0};
  pid_t telemetry_peer_pid{0};
};

// Injection exists so socket/codec lifecycle tests can run in sandboxes that
// deny socket metadata getsockopt calls. Production constructs
// InheritedChannels without a provider and therefore always uses kernel
// SO_TYPE/SO_DOMAIN/SO_PEERCRED for both inherited sockets.
class SocketMetadataProvider {
 public:
  virtual ~SocketMetadataProvider() = default;
  virtual bool read_socket_metadata(int fd,
                                    int* socket_type,
                                    int* socket_domain,
                                    pid_t* peer_pid,
                                    uid_t* peer_uid,
                                    int* system_errno) const noexcept = 0;
};

class InheritedChannels final {
 public:
  InheritedChannels(
      int critical_fd,
      int telemetry_fd,
      const SocketMetadataProvider* socket_metadata_provider = nullptr) noexcept;
  ~InheritedChannels() = default;
  InheritedChannels(const InheritedChannels&) = delete;
  InheritedChannels& operator=(const InheritedChannels&) = delete;

  ChannelValidation validate_and_configure(pid_t expected_parent_pid) noexcept;
  ReceiveResult receive_critical() noexcept;
  IoStatus send_critical(const EncodedPacket& packet,
                         int* system_errno) noexcept;
  IoStatus send_telemetry(const EncodedPacket& packet,
                          int* system_errno) noexcept;

  int critical_fd() const noexcept { return critical_fd_; }
  int telemetry_fd() const noexcept { return telemetry_fd_; }

 private:
  int critical_fd_;
  int telemetry_fd_;
  const SocketMetadataProvider* socket_metadata_provider_;
};

}  // namespace anydex::v94_franka_servo
