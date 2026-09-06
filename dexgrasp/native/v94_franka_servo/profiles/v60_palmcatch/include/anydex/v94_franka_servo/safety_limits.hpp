#pragma once

// Reuse every commissioned native safety constant from the V94 servo while
// replacing only the task-bound reset, absolute joint envelope and immutable
// profile digests.  Renaming the base type keeps the default V94 header and
// its already-built binary byte-for-byte untouched.
#define HardSafetyLimits V94HardSafetyLimits
#define final
#include "../../../../../include/anydex/v94_franka_servo/safety_limits.hpp"
#undef final
#undef HardSafetyLimits

namespace anydex::v94_franka_servo {

struct HardSafetyLimits final : V94HardSafetyLimits {
  static constexpr std::array<double, 7> kQHome{
      // Exact float32 V60 task reset values promoted to double.  These match
      // the Python ARM payload bit-for-bit at the native boundary.
      -0x1.45ae780000000p+1, -0x1.6966140000000p+0,
      -0x1.94a58c0000000p-5, -0x1.2b3b680000000p+1,
      0x1.921fb60000000p+0, 0x1.db68c60000000p-2,
      0x1.59999a0000000p+1};
  static constexpr std::array<double, 7> kSafeJointLower{
      -2.8807, -1.8161, -2.8807, -3.0570, -2.8563, 0.4598, -3.0308};
  static constexpr std::array<double, 7> kSafeJointUpper{
      2.8807, 1.8161, 2.8807, -0.1369, 2.8563, 4.6016, 3.0308};
  static constexpr const char* kProfileSha256 =
      "adde3203457453ba67949b93431e4d77c8d6412d86e80575525b80edac2e723a";
  static constexpr const char* kEnvelopeSha256 =
      "5fb2866c16fa101c02d9610aa9c3dda8c64de939a6f26ec176bab205ae0adb38";
};

}  // namespace anydex::v94_franka_servo
