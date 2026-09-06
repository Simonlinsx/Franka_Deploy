#include <cassert>
#include <iostream>
#include <string>

#include "anydex/v94_franka_servo/dependency_identity.hpp"

int main() {
  std::string detail;
  if (!anydex::v94_franka_servo::verify_linked_libfranka_identity(&detail)) {
    std::cerr << "pinned loaded libfranka rejected: " << detail << '\n';
    return 1;
  }
  std::cout << "pinned loaded libfranka verified: " << detail << '\n';
  return 0;
}
