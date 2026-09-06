#include "franka_v225_interpolated_reference.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace v225 = simtoolreal::franka_v225;

int main() {
  const v225::Vector7f previous = {0.0F, 0.0F, 0.0F, -1.0F, 0.0F, 2.0F, 0.0F};
  const v225::Vector7f measured = previous;
  const v225::Vector7f action = {1.0F, -1.0F, 0.5F, 0.0F, -0.5F, 1.0F, -1.0F};
  const v225::Vector7f held_float =
      v225::MapPolicyActionToHeldTarget(previous, measured, action);
  const v225::Vector7d held = v225::ToDouble(held_float);
  const v225::Vector7d start = v225::ToDouble(previous);
  v225::InterpolatedJointPositionGenerator generator(start);

  std::cout << std::setprecision(17);
  std::cout << "held_target";
  for (const double value : held) {
    std::cout << ',' << value;
  }
  std::cout << '\n';

  for (int packet = 1; packet <= 300; ++packet) {
    generator.Step(held);
    if (packet == 1 || packet == 50 || packet == 100 || packet == 300) {
      const auto& state = generator.state();
      std::cout << "packet_" << packet;
      for (const double value : state.q_rad) {
        std::cout << ',' << value;
      }
      std::cout << '\n';
    }
  }
  return 0;
}
