#include "anydex/v94_franka_servo/dependency_identity.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

#include <openssl/evp.h>

#ifndef ANYDEX_LIBFRANKA_LIBRARY_PATH
#error "ANYDEX_LIBFRANKA_LIBRARY_PATH is required"
#endif
#ifndef ANYDEX_LIBFRANKA_LIBRARY_SHA256
#error "ANYDEX_LIBFRANKA_LIBRARY_SHA256 is required"
#endif

namespace anydex::v94_franka_servo {
namespace {

std::string canonical_path(const char* path) {
  std::array<char, 4096> resolved{};
  if (::realpath(path, resolved.data()) == nullptr) {
    throw std::runtime_error(std::string("realpath failed: ") +
                             std::strerror(errno));
  }
  return std::string(resolved.data());
}

std::string sha256_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open pinned libfranka for hashing");
  }
  using Context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  Context context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("EVP SHA-256 initialization failed");
  }
  std::array<char, 65536> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 &&
        EVP_DigestUpdate(context.get(), buffer.data(),
                         static_cast<std::size_t>(count)) != 1) {
      throw std::runtime_error("EVP SHA-256 update failed");
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("read failed while hashing pinned libfranka");
  }
  std::array<unsigned char, 32> digest{};
  unsigned int digest_size = 0U;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 ||
      digest_size != digest.size()) {
    throw std::runtime_error("EVP SHA-256 finalization failed");
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const unsigned char byte : digest) {
    output << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return output.str();
}

bool mapped_exactly_once(const std::string& expected) {
  std::ifstream maps("/proc/self/maps");
  if (!maps) {
    return false;
  }
  bool expected_seen = false;
  std::string line;
  while (std::getline(maps, line)) {
    const std::size_t separator = line.find('/');
    if (separator == std::string::npos) {
      continue;
    }
    const std::string path = line.substr(separator);
    const std::size_t basename_offset = path.find_last_of('/');
    const std::string basename =
        basename_offset == std::string::npos
            ? path
            : path.substr(basename_offset + 1U);
    if (basename.rfind("libfranka", 0U) != 0U) {
      continue;
    }
    std::string resolved;
    try {
      resolved = canonical_path(path.c_str());
    } catch (...) {
      return false;
    }
    if (resolved != expected) {
      return false;
    }
    expected_seen = true;
  }
  return expected_seen;
}

}  // namespace

bool verify_linked_libfranka_identity(std::string* detail) noexcept {
  try {
    const std::string expected =
        canonical_path(ANYDEX_LIBFRANKA_LIBRARY_PATH);
    if (!mapped_exactly_once(expected)) {
      if (detail != nullptr) {
        *detail = "loaded libfranka map is absent or not the pinned wheel DSO";
      }
      return false;
    }
    const std::string digest = sha256_file(expected);
    if (digest != ANYDEX_LIBFRANKA_LIBRARY_SHA256) {
      if (detail != nullptr) {
        *detail = "loaded libfranka file SHA-256 differs from pinned digest";
      }
      return false;
    }
    if (detail != nullptr) {
      *detail = expected;
    }
    return true;
  } catch (const std::exception& error) {
    if (detail != nullptr) {
      *detail = error.what();
    }
    return false;
  } catch (...) {
    if (detail != nullptr) {
      *detail = "unknown libfranka identity verification failure";
    }
    return false;
  }
}

}  // namespace anydex::v94_franka_servo
