#pragma once

#include <memory>
#include <string>

#include "anydex/v94_franka_servo/backend.hpp"

namespace anydex::v94_franka_servo {

class LibfrankaBackendFactory final : public RobotBackendFactory {
 public:
  std::unique_ptr<RobotBackend> create_enforced(
      const std::string& robot_address) override;
};

}  // namespace anydex::v94_franka_servo
