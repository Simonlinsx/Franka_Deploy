#pragma once

// Reuse the commissioned V94 native dynamic/contact/watchdog limits while
// binding this executable only to the tabletop seq286 reset profile.  Keeping
// a separate build prevents tabletop tests from replacing the legacy thrown
// V57 executable (and vice versa).
#define HardSafetyLimits V94HardSafetyLimits
#define final
#include "../../../../../include/anydex/v94_franka_servo/safety_limits.hpp"
#undef final
#undef HardSafetyLimits

namespace anydex::v94_franka_servo {

struct HardSafetyLimits final : V94HardSafetyLimits {
  static constexpr std::array<double, 7> kQHome{
      // Exact float32 seq286 reset values promoted to double.  These match
      // the production Python ARM payload bit-for-bit.
      0x0.0p+0, -0x1.2353f80000000p-1, 0x0.0p+0,
      -0x1.67ae140000000p+1, 0x0.0p+0, 0x1.84bc6a0000000p+1,
      0x1.7b645a0000000p-1};
  static constexpr const char* kProfileSha256 =
      "b439c5053991029ea7777211499e22df3152c1177012f9f02bbe2df6ef022a6a";
  static constexpr const char* kEnvelopeSha256 =
      "6ae9e2f21c0a30857600b60793ab44fdc2ca2122729e71028064f0246a24d932";
};

}  // namespace anydex::v94_franka_servo
