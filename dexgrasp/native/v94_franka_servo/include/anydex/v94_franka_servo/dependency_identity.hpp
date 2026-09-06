#pragma once

#include <string>

namespace anydex::v94_franka_servo {

// Verifies both the loader-resolved path and the bytes of the already loaded,
// hashed wheel libfranka before a Robot may be constructed.
bool verify_linked_libfranka_identity(std::string* detail) noexcept;

}  // namespace anydex::v94_franka_servo
