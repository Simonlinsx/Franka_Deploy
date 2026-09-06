cmake_minimum_required(VERSION 3.16)
project(anydex_v94_franka_servo VERSION 1.0.0 LANGUAGES CXX)

set(ANYDEX_V94_SAFETY_PROFILE "v94" CACHE STRING
    "Compiled Franka safety profile (v94, v94_tabletop, v60_palmcatch or v61_sixexpert)")
set_property(CACHE ANYDEX_V94_SAFETY_PROFILE PROPERTY STRINGS
             v94 v94_tabletop v60_palmcatch v61_sixexpert)
if(ANYDEX_V94_SAFETY_PROFILE STREQUAL "v94_tabletop")
  set(V94_SAFETY_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/profiles/v94_tabletop/include")
  set(V94_SAFETY_PROFILE_SOURCE
      profiles/v94_tabletop/include/anydex/v94_franka_servo/safety_limits.hpp)
elseif(ANYDEX_V94_SAFETY_PROFILE STREQUAL "v60_palmcatch")
  set(V94_SAFETY_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/profiles/v60_palmcatch/include")
  set(V94_SAFETY_PROFILE_SOURCE
      profiles/v60_palmcatch/include/anydex/v94_franka_servo/safety_limits.hpp)
elseif(ANYDEX_V94_SAFETY_PROFILE STREQUAL "v61_sixexpert")
  set(V94_SAFETY_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/profiles/v61_sixexpert/include")
  set(V94_SAFETY_PROFILE_SOURCE
      profiles/v61_sixexpert/include/anydex/v94_franka_servo/safety_limits.hpp)
elseif(NOT ANYDEX_V94_SAFETY_PROFILE STREQUAL "v94")
  message(FATAL_ERROR
          "ANYDEX_V94_SAFETY_PROFILE must be v94, v94_tabletop, v60_palmcatch or v61_sixexpert")
else()
  set(V94_SAFETY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set(V94_SAFETY_PROFILE_SOURCE "")
endif()

include(CTest)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

if(NOT DEFINED LIBFRANKA_INCLUDE_DIR)
  message(FATAL_ERROR "LIBFRANKA_INCLUDE_DIR must name the exact 0.21.2 headers")
endif()
if(NOT DEFINED LIBFRANKA_COMMON_INCLUDE_DIR)
  message(FATAL_ERROR "LIBFRANKA_COMMON_INCLUDE_DIR is required")
endif()
if(NOT DEFINED PYLIBFRANKA_HASHED_LIBRARY OR
   NOT IS_ABSOLUTE "${PYLIBFRANKA_HASHED_LIBRARY}")
  message(FATAL_ERROR "PYLIBFRANKA_HASHED_LIBRARY must be an absolute path")
endif()
if(NOT DEFINED PYLIBFRANKA_LIBS_DIR OR
   NOT IS_ABSOLUTE "${PYLIBFRANKA_LIBS_DIR}")
  message(FATAL_ERROR "PYLIBFRANKA_LIBS_DIR must be an absolute path")
endif()
if(NOT DEFINED PYLIBFRANKA_HASHED_LIBRARY_SHA256)
  message(FATAL_ERROR "PYLIBFRANKA_HASHED_LIBRARY_SHA256 is required")
endif()
if(NOT DEFINED LIBFRANKA_SOURCE_COMMIT)
  message(FATAL_ERROR "LIBFRANKA_SOURCE_COMMIT is required")
endif()

file(SHA256 "${PYLIBFRANKA_HASHED_LIBRARY}" actual_libfranka_sha256)
if(NOT actual_libfranka_sha256 STREQUAL
       PYLIBFRANKA_HASHED_LIBRARY_SHA256)
  message(FATAL_ERROR "pinned wheel libfranka SHA-256 mismatch")
endif()
string(LENGTH "${LIBFRANKA_SOURCE_COMMIT}" libfranka_commit_length)
if(NOT libfranka_commit_length EQUAL 40 OR
   NOT LIBFRANKA_SOURCE_COMMIT MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "LIBFRANKA_SOURCE_COMMIT must be lowercase SHA-1")
endif()

set(v94_public_headers
  include/anydex/v94_franka_servo/backend.hpp
  include/anydex/v94_franka_servo/channel.hpp
  include/anydex/v94_franka_servo/dependency_identity.hpp
  include/anydex/v94_franka_servo/franka_v225_interpolator.hpp
  include/anydex/v94_franka_servo/libfranka_backend.hpp
  include/anydex/v94_franka_servo/protocol.hpp
  include/anydex/v94_franka_servo/safety_limits.hpp
  include/anydex/v94_franka_servo/servo_core.hpp)
set(v94_producer_sources
  ${v94_public_headers}
  ${V94_SAFETY_PROFILE_SOURCE}
  src/channel.cpp
  src/dependency_identity.cpp
  src/libfranka_backend.cpp
  src/main.cpp
  src/protocol.cpp
  src/servo_core.cpp)
set(producer_hash_material "")
foreach(source IN LISTS v94_producer_sources)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_sha256)
  string(APPEND producer_hash_material "${source}:${source_sha256}\n")
endforeach()
string(SHA256 V94_PRODUCER_BUILD_SHA256 "${producer_hash_material}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/producer_build_sha256.txt"
     "${V94_PRODUCER_BUILD_SHA256}\n")
message(STATUS "V94 producer_build_sha256=${V94_PRODUCER_BUILD_SHA256}")

function(v94_strict_target target)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endfunction()

add_library(v94_servo_core STATIC
  src/channel.cpp
  src/protocol.cpp
  src/servo_core.cpp)
target_include_directories(v94_servo_core BEFORE PUBLIC
  "${V94_SAFETY_INCLUDE_DIR}")
target_include_directories(v94_servo_core PUBLIC include)
target_link_libraries(v94_servo_core PUBLIC Threads::Threads)
v94_strict_target(v94_servo_core)

add_executable(v94_franka_servo
  src/dependency_identity.cpp
  src/libfranka_backend.cpp
  src/main.cpp)
target_include_directories(v94_franka_servo PRIVATE
  "${V94_SAFETY_INCLUDE_DIR}"
  include
  "${LIBFRANKA_INCLUDE_DIR}"
  "${LIBFRANKA_COMMON_INCLUDE_DIR}")
target_compile_definitions(v94_franka_servo PRIVATE
  ANYDEX_LIBFRANKA_LIBRARY_PATH="${PYLIBFRANKA_HASHED_LIBRARY}"
  ANYDEX_LIBFRANKA_LIBRARY_SHA256="${PYLIBFRANKA_HASHED_LIBRARY_SHA256}"
  ANYDEX_LIBFRANKA_SOURCE_COMMIT="${LIBFRANKA_SOURCE_COMMIT}"
  ANYDEX_PRODUCER_BUILD_SHA256="${V94_PRODUCER_BUILD_SHA256}")
target_link_libraries(v94_franka_servo PRIVATE
  v94_servo_core
  OpenSSL::Crypto
  "${PYLIBFRANKA_HASHED_LIBRARY}")
target_link_options(v94_franka_servo PRIVATE "-Wl,--disable-new-dtags")
set_target_properties(v94_franka_servo PROPERTIES
  BUILD_RPATH "${PYLIBFRANKA_LIBS_DIR}"
  INSTALL_RPATH "${PYLIBFRANKA_LIBS_DIR}")
v94_strict_target(v94_franka_servo)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_executable(v94_protocol_test tests/test_protocol.cpp)
  target_link_libraries(v94_protocol_test PRIVATE v94_servo_core)
  target_compile_options(v94_protocol_test PRIVATE -UNDEBUG)
  v94_strict_target(v94_protocol_test)
  add_test(NAME v94_protocol_test COMMAND v94_protocol_test)

  add_executable(v225_interpolator_test tests/test_v225_interpolator.cpp)
  target_link_libraries(v225_interpolator_test PRIVATE v94_servo_core)
  target_compile_options(v225_interpolator_test PRIVATE -UNDEBUG)
  v94_strict_target(v225_interpolator_test)
  add_test(NAME v225_interpolator_test COMMAND v225_interpolator_test)

  add_executable(v94_fake_lifecycle_test tests/test_fake_lifecycle.cpp)
  target_link_libraries(v94_fake_lifecycle_test PRIVATE v94_servo_core)
  target_compile_options(v94_fake_lifecycle_test PRIVATE -UNDEBUG)
  v94_strict_target(v94_fake_lifecycle_test)
  add_test(NAME v94_fake_lifecycle_test COMMAND v94_fake_lifecycle_test)
  add_test(NAME v94_fake_lifecycle_direct_stop_settle_test
           COMMAND v94_fake_lifecycle_test --direct-stop-settle)
  add_test(NAME v94_fake_lifecycle_isolated_15ms_period_test
           COMMAND v94_fake_lifecycle_test --isolated-15ms-period)
  add_test(NAME v94_fake_lifecycle_isolated_20ms_period_test
           COMMAND v94_fake_lifecycle_test --isolated-20ms-period)
  add_test(NAME v94_fake_lifecycle_isolated_21ms_period_test
           COMMAND v94_fake_lifecycle_test --isolated-21ms-period)
  add_test(NAME v94_fake_lifecycle_isolated_22ms_period_test
           COMMAND v94_fake_lifecycle_test --isolated-22ms-period)
  add_test(NAME v94_fake_lifecycle_isolated_4ms_extrapolation_test
           COMMAND v94_fake_lifecycle_test --isolated-4ms-extrapolation)
  add_test(NAME v94_fake_lifecycle_consecutive_recoverable_periods_test
           COMMAND v94_fake_lifecycle_test --consecutive-recoverable-periods)
  add_test(NAME v94_fake_lifecycle_dynamic_state_diagnostic_test
           COMMAND v94_fake_lifecycle_test --dynamic-state-diagnostic)
  add_test(NAME v94_fake_lifecycle_float_history_mismatch_test
           COMMAND v94_fake_lifecycle_test --float-history-mismatch)
  add_test(NAME v94_fake_lifecycle_real_first_motion_drop_test
           COMMAND v94_fake_lifecycle_test --real-first-motion-drop)
  add_test(NAME v94_fake_lifecycle_isolated_4ms_drop_test
           COMMAND v94_fake_lifecycle_test --isolated-4ms-drop)
  add_test(NAME v94_fake_lifecycle_isolated_20ms_drop_test
           COMMAND v94_fake_lifecycle_test --isolated-20ms-drop)
  add_test(NAME v94_fake_lifecycle_qd_g015_large_target_test
           COMMAND v94_fake_lifecycle_test --qd-g015-large-target)
  add_test(NAME v94_fake_lifecycle_legacy_large_target_rejected_test
           COMMAND v94_fake_lifecycle_test --legacy-large-target-rejected)
  add_test(NAME v94_realtime_scheduler_proof_test
           COMMAND v94_fake_lifecycle_test --realtime-scheduler-proof)
  add_test(NAME v94_realtime_scheduler_syscall_failure_test
           COMMAND v94_fake_lifecycle_test
                   --realtime-scheduler-syscall-failure)
  add_test(NAME v94_desired_history_ambiguity_test
           COMMAND v94_fake_lifecycle_test --desired-history-ambiguity)
  add_test(NAME v94_desired_history_closed_form_test
           COMMAND v94_fake_lifecycle_test --desired-history-closed-form)
  add_test(NAME v94_desired_history_near_zero_roundoff_test
           COMMAND v94_fake_lifecycle_test
                   --desired-history-near-zero-roundoff)
  add_test(NAME v94_recoverable_fci_history_test
           COMMAND v94_fake_lifecycle_test --recoverable-fci-history)
  add_test(NAME v94_mode_aware_episode_envelope_test
           COMMAND v94_fake_lifecycle_test --mode-aware-episode-envelope)
  # Cross the actual language boundary: production Python encodes the ARM
  # packet and the production C++ require_arm_contract implementation accepts
  # it.  This intentionally does not construct a Robot or access hardware.
  add_executable(v94_arm_contract_test
    tests/test_arm_contract.cpp
    src/channel.cpp
    src/protocol.cpp)
  target_include_directories(v94_arm_contract_test PRIVATE include)
  target_include_directories(v94_arm_contract_test BEFORE PRIVATE
    "${V94_SAFETY_INCLUDE_DIR}")
  target_link_libraries(v94_arm_contract_test PRIVATE Threads::Threads)
  target_compile_options(v94_arm_contract_test PRIVATE -UNDEBUG)
  v94_strict_target(v94_arm_contract_test)
  add_test(
    NAME v94_python_cpp_arm_contract_test
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_cpp_arm_contract.py"
            "$<TARGET_FILE:v94_arm_contract_test>")
  set_tests_properties(v94_python_cpp_arm_contract_test PROPERTIES
    ENVIRONMENT "ANYDEX_V94_SAFETY_PROFILE=${ANYDEX_V94_SAFETY_PROFILE}")
  set_tests_properties(v94_python_cpp_arm_contract_test PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

  add_executable(v94_dependency_identity_test
    tests/test_dependency_identity.cpp
    src/dependency_identity.cpp
    src/libfranka_backend.cpp)
  target_include_directories(v94_dependency_identity_test PRIVATE
    include
    "${LIBFRANKA_INCLUDE_DIR}"
    "${LIBFRANKA_COMMON_INCLUDE_DIR}")
  target_compile_definitions(v94_dependency_identity_test PRIVATE
    ANYDEX_LIBFRANKA_LIBRARY_PATH="${PYLIBFRANKA_HASHED_LIBRARY}"
    ANYDEX_LIBFRANKA_LIBRARY_SHA256="${PYLIBFRANKA_HASHED_LIBRARY_SHA256}")
  target_compile_options(v94_dependency_identity_test PRIVATE -UNDEBUG)
  target_link_libraries(v94_dependency_identity_test PRIVATE
    v94_servo_core
    OpenSSL::Crypto
    "${PYLIBFRANKA_HASHED_LIBRARY}")
  target_link_options(v94_dependency_identity_test PRIVATE
    "-Wl,--disable-new-dtags")
  set_target_properties(v94_dependency_identity_test PROPERTIES
    BUILD_RPATH "${PYLIBFRANKA_LIBS_DIR}")
  v94_strict_target(v94_dependency_identity_test)
  add_test(NAME v94_dependency_identity_test
           COMMAND v94_dependency_identity_test)
endif()
