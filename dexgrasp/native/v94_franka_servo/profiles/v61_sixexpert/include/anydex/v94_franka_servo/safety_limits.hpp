#pragma once

// Reuse all commissioned native dynamic/contact/watchdog limits and replace
// only the V61 task-bound reset, absolute joint envelope and immutable
// profile digests.  The V61 task launcher separately enforces a 40-target cap.
#define HardSafetyLimits V94HardSafetyLimits
#define final
#include "../../../../../include/anydex/v94_franka_servo/safety_limits.hpp"
#undef final
#undef HardSafetyLimits

namespace anydex::v94_franka_servo {

struct HardSafetyLimits final : V94HardSafetyLimits {
  static constexpr std::array<double, 7> kQHome{
      0x1.4233580000000p-1, -0x1.b02cf60000000p-1,
      -0x1.0e23b00000000p-7, -0x1.0c52500000000p+1,
      -0x1.9df19e0000000p-2, 0x1.d475700000000p+0,
      -0x1.bf357e0000000p+0};
  static constexpr std::array<double, 7> kSafeJointLower{
      -2.6937, -1.7337, -2.8507, -2.9921, -2.7565, 0.5945, -2.9659};
  static constexpr std::array<double, 7> kSafeJointUpper{
      2.6937, 1.7337, 2.8507, -0.2018, 2.7565, 4.4669, 2.9659};
  static constexpr const char* kProfileSha256 =
      "cb63ff016560a3a052e7773875ffe68888a34482d0f9d9da9869d1ee6272e85b";
  static constexpr const char* kEnvelopeSha256 =
      "531b6deebca026029dc06d403ff7f662bd3483e7a7983878af981cb572f7cd71";
};

}  // namespace anydex::v94_franka_servo
