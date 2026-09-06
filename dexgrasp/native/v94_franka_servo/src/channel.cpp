#include "anydex/v94_franka_servo/channel.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace anydex::v94_franka_servo {
namespace {

bool set_descriptor_flags(const int fd, int* system_errno) noexcept {
  const int status_flags = ::fcntl(fd, F_GETFL, 0);
  if (status_flags < 0 || ::fcntl(fd, F_SETFL, status_flags | O_NONBLOCK) < 0) {
    *system_errno = errno;
    return false;
  }
  const int descriptor_flags = ::fcntl(fd, F_GETFD, 0);
  if (descriptor_flags < 0 ||
      ::fcntl(fd, F_SETFD, descriptor_flags | FD_CLOEXEC) < 0) {
    *system_errno = errno;
    return false;
  }
  return true;
}

bool validate_socket(const int fd,
                     const int expected_type,
                     const pid_t expected_parent_pid,
                     const SocketMetadataProvider* socket_metadata_provider,
                     pid_t* peer_pid,
                     int* system_errno) noexcept {
  int type = 0;
  int domain = 0;
  pid_t credential_pid = 0;
  uid_t credential_uid = static_cast<uid_t>(-1);
  if (socket_metadata_provider != nullptr) {
    if (!socket_metadata_provider->read_socket_metadata(
            fd, &type, &domain, &credential_pid, &credential_uid,
            system_errno)) {
      return false;
    }
  } else {
    socklen_t type_size = sizeof(type);
    if (::getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &type_size) != 0) {
      *system_errno = errno;
      return false;
    }
#ifdef SO_DOMAIN
    socklen_t domain_size = sizeof(domain);
    if (::getsockopt(fd, SOL_SOCKET, SO_DOMAIN, &domain, &domain_size) != 0) {
      *system_errno = errno;
      return false;
    }
#else
    *system_errno = ENOTSUP;
    return false;
#endif
#ifdef SO_PEERCRED
    struct ucred credentials {};
    socklen_t credentials_size = sizeof(credentials);
    if (::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials,
                     &credentials_size) != 0 ||
        credentials_size != sizeof(credentials)) {
      *system_errno = errno;
      return false;
    }
    credential_pid = credentials.pid;
    credential_uid = credentials.uid;
#else
    *system_errno = ENOTSUP;
    return false;
#endif
  }
  if (type != expected_type) {
    *system_errno = EPROTOTYPE;
    return false;
  }
  if (domain != AF_UNIX) {
    *system_errno = EAFNOSUPPORT;
    return false;
  }
  if (credential_uid != ::geteuid() ||
      credential_pid != expected_parent_pid) {
    *system_errno = EPERM;
    return false;
  }
  *peer_pid = credential_pid;
  return set_descriptor_flags(fd, system_errno);
}

IoStatus send_one(const int fd,
                  const EncodedPacket& packet,
                  int* system_errno) noexcept {
  if (packet.size == 0U || packet.size > packet.bytes.size()) {
    *system_errno = EINVAL;
    return IoStatus::kError;
  }
  const ssize_t sent = ::send(fd, packet.bytes.data(), packet.size,
                              MSG_DONTWAIT | MSG_NOSIGNAL);
  if (sent == static_cast<ssize_t>(packet.size)) {
    return IoStatus::kOk;
  }
  if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    *system_errno = errno;
    return IoStatus::kWouldBlock;
  }
  if (sent < 0 && (errno == EPIPE || errno == ECONNRESET)) {
    *system_errno = errno;
    return IoStatus::kPeerClosed;
  }
  *system_errno = sent < 0 ? errno : EMSGSIZE;
  return IoStatus::kError;
}

}  // namespace

InheritedChannels::InheritedChannels(const int critical_fd,
                                     const int telemetry_fd,
                                     const SocketMetadataProvider*
                                         socket_metadata_provider) noexcept
    : critical_fd_(critical_fd),
      telemetry_fd_(telemetry_fd),
      socket_metadata_provider_(socket_metadata_provider) {}

ChannelValidation InheritedChannels::validate_and_configure(
    const pid_t expected_parent_pid) noexcept {
  ChannelValidation result{};
  if (critical_fd_ < 0 || telemetry_fd_ < 0 ||
      critical_fd_ == telemetry_fd_ || expected_parent_pid <= 1) {
    result.system_errno = EINVAL;
    return result;
  }
  if (!validate_socket(critical_fd_, SOCK_SEQPACKET, expected_parent_pid,
                       socket_metadata_provider_,
                       &result.critical_peer_pid, &result.system_errno)) {
    return result;
  }
  if (!validate_socket(telemetry_fd_, SOCK_DGRAM, expected_parent_pid,
                       socket_metadata_provider_,
                       &result.telemetry_peer_pid, &result.system_errno)) {
    return result;
  }
  result.valid = true;
  return result;
}

ReceiveResult InheritedChannels::receive_critical() noexcept {
  ReceiveResult result{};
  struct iovec vector {};
  vector.iov_base = result.bytes.data();
  vector.iov_len = result.bytes.size();
  struct msghdr message {};
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  const ssize_t received =
      ::recvmsg(critical_fd_, &message, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
  if (received == 0) {
    result.status = IoStatus::kPeerClosed;
    return result;
  }
  if (received < 0) {
    result.system_errno = errno;
    result.status = (errno == EAGAIN || errno == EWOULDBLOCK)
                        ? IoStatus::kWouldBlock
                        : ((errno == ECONNRESET) ? IoStatus::kPeerClosed
                                                : IoStatus::kError);
    return result;
  }
  if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0 ||
      static_cast<std::size_t>(received) > result.bytes.size()) {
    result.status = IoStatus::kTruncated;
    result.size = static_cast<std::size_t>(received);
    return result;
  }
  result.status = IoStatus::kOk;
  result.size = static_cast<std::size_t>(received);
  return result;
}

IoStatus InheritedChannels::send_critical(const EncodedPacket& packet,
                                          int* system_errno) noexcept {
  return send_one(critical_fd_, packet, system_errno);
}

IoStatus InheritedChannels::send_telemetry(const EncodedPacket& packet,
                                           int* system_errno) noexcept {
  return send_one(telemetry_fd_, packet, system_errno);
}

}  // namespace anydex::v94_franka_servo
