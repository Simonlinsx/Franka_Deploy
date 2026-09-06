#pragma once

#include <array>
#include <cstdint>

namespace anydex::v94_franka_servo {

struct HardSafetyLimits final {
  static constexpr double kMaximumCommandVelocityRadS = 0.50;
  // Deployment/simulator-shared V225 derivative envelope.  The successful
  // simulator replay reached about 3.60 rad/s^2 and 102.76 rad/s^3.  These
  // ceilings retain response margin while preventing a held 20 Hz target
  // step from reaching ~10 rad/s^2 in the first two FCI packets.
  static constexpr double kMaximumCommandAccelerationRadS2 = 5.0;
  static constexpr double kMaximumCommandJerkRadS3 = 250.0;
  static constexpr double kMaximumStartErrorRad = 0.01;
  // Exact-target replay preserves simulator setpoints (observed max adjacent
  // delta 0.018000365 rad).  This limits the high-level target jump only; the
  // 1 kHz command trajectory remains bounded by velocity/acceleration/jerk.
  static constexpr double kMaximumTickTargetDeltaRad = 0.020;
  // Legacy V94 keeps this home-centered episode displacement independently
  // of the longer 720-target session. q_d-g015 intentionally omits that
  // relative radius and instead uses the compiled margin-contracted absolute
  // joint intervals. Tracking, velocity, acceleration, jerk, contact/reflex,
  // and the final libfranka limiter remain common to both modes.
  static constexpr double kMaximumEpisodeDeltaRad = 1.21;
  static constexpr double kMaximumTrackingErrorRad = 0.01;
  // Keep the generated command trajectory at <=0.50 rad/s.  The successful
  // v205 simulator trace reaches about 0.405 rad/s. Measured dq can
  // briefly overshoot that exact command ceiling during physical tracking;
  // reject only a separate 0.70 rad/s measured-state
  // boundary.
  static constexpr double kMaximumMeasuredVelocityRadS = 0.70;
  static constexpr double kStopMaximumVelocityRadS = 0.01;
  static constexpr double kMinimumHealthySuccessRate = 0.99;
  // Franka FCI stops after 20 consecutive missing command packets. A received
  // recovery command resets that counter and becomes the new history; a
  // dropped recovery command does not reset it. Therefore the maximum
  // constant-acceleration continuation of any last received command is 20
  // packets (not 12+20). Reserve that complete fail-stop horizon.
  static constexpr std::uint32_t kFciFailStopDroppedPacketBound = 20U;
  // Do not impose a tighter workstation-side timing policy than FCI.  The
  // returned period includes the recovered current packet: 20 missing packets
  // therefore appear as 21 ms.  Keep the fail-stop/uncontrolled continuation
  // bound at 20 packets while admitting that final recovered state.
  // libfranka/Control remains authoritative for communication failure; zero
  // after bootstrap or a returned value beyond this protocol horizon is
  // internally inconsistent.
  static constexpr std::uint32_t kMaximumRecoverableControlPeriodMs =
      kFciFailStopDroppedPacketBound + 1U;
  static constexpr std::uint32_t
      kMaximumUncontrolledContinuationPackets =
          kFciFailStopDroppedPacketBound;
  static constexpr std::uint64_t kMaximumReadToWriteNs = 800000U;
  // TARGET packet timestamps remain independently freshness-bounded.  During
  // a recoverable observation or USB scheduling gap, converge to and hold the
  // last accepted bounded target for at most 500 ms.  The independent 100 ms
  // parent heartbeat still detects controller-process loss, and any new
  // TARGET remains subject to the unchanged 50 ms timestamp-age ceiling.
  static constexpr std::uint64_t kMaximumTargetAgeNs = 50000000U;
  static constexpr std::uint64_t kMaximumInterTargetTimeoutNs = 500000000U;
  static constexpr std::uint64_t kMaximumFutureTargetSkewNs = 10000000U;
  static constexpr std::uint64_t kMaximumHeartbeatTimeoutNs = 100000000U;
  static constexpr std::uint64_t kMaximumFirstTargetTimeoutNs = 5000000000ULL;
  // 720 targets at 60 Hz span 12 seconds. Reserve three seconds for the fixed
  // healthy-control bootstrap, first-target handoff, and final acknowledged
  // hold/stop request.
  static constexpr std::uint64_t kMaximumSessionDurationNs = 15000000000ULL;
  static constexpr std::uint32_t kMaximumTargetCount = 720U;
  static constexpr std::uint32_t kHealthyCyclesBeforeAction = 100U;
  static constexpr std::uint32_t kMaximumCriticalPacketsPerCycle = 4U;
  static constexpr std::uint64_t kMaximumIpcDrainNs = 100000U;
  static constexpr std::uint32_t kStateDecimation = 16U;
  static constexpr std::uint32_t kStopConsecutiveSamples = 3U;
  // A requested StopMove can remain in a non-Move transition mode for more
  // than one second even after measured dq is already below 0.01 rad/s.  Keep
  // the proof bounded to approximately three seconds while retaining the
  // unchanged requirement for three fresh Idle samples at <=0.01 rad/s.
  static constexpr std::uint32_t kStopMaximumSamples = 3000U;

  static constexpr std::array<double, 7> kQHome{
      // The deploy bundle stores q_home as float32.  These hexadecimal
      // literals are those exact float32 values promoted to double, matching
      // the production ARM wire payload bit-for-bit.
      0x0.0p+0, -0x1.3333340000000p+0, 0x0.0p+0,
      -0x1.19999a0000000p+1, 0x0.0p+0, 0x1.6666660000000p+0,
      0x1.921fb60000000p-1};
  static constexpr std::array<double, 7> kSafeJointLower{
      -2.6937, -1.7337, -2.8507, -2.9921, -2.7565, 0.5945, -2.9659};
  static constexpr std::array<double, 7> kSafeJointUpper{
      2.6937, 1.7337, 2.8507, -0.2018, 2.7565, 4.4669, 2.9659};
  static constexpr std::array<double, 16> kExpectedFTee{
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0};
  static constexpr double kExpectedEndEffectorMassKg = 0.607;
  static constexpr std::array<double, 3> kExpectedEndEffectorComM{
      0.0, 0.0, 0.076};
  static constexpr std::array<double, 9> kExpectedEndEffectorInertiaKgM2{
      0.00151, 0.0, 0.0,
      0.0, 0.00169, 0.0,
      0.0, 0.0, 0.000442};
  static constexpr double kExpectedExternalLoadMassKg = 0.0;

  // Keep libfranka's contact/collision distinction instead of making both
  // thresholds identical.  The lower values are the official external-loop
  // example values and enter the bounded contact-hold path before a reflex.
  // The upper values retain the separately tested installed-RH56 collision
  // stop thresholds.  Crossing an upper threshold still causes Franka's
  // controller-side reflex; it is never converted into a host-side warning.
  static constexpr std::array<double, 7> kContactTorqueThresholdNm{
      20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0};
  static constexpr std::array<double, 6> kContactForceThresholdN{
      20.0, 20.0, 20.0, 25.0, 25.0, 25.0};
  static constexpr std::array<double, 7> kCollisionTorqueThresholdNm{
      40.0, 40.0, 36.0, 36.0, 32.0, 28.0, 24.0};
  static constexpr std::array<double, 6> kCollisionForceThresholdN{
      40.0, 40.0, 40.0, 50.0, 50.0, 50.0};

  static constexpr const char* kProfileSha256 =
      "90a04c8395671905e1e3e1a05430360f5aaa179b247821ebc4a35a0edb330a6a";
  static constexpr const char* kEnvelopeSha256 =
      "8f8842d2fab798bb6541c4a9c3333917f1e555de8ec60b8909bf2eae39899234";
};

}  // namespace anydex::v94_franka_servo
