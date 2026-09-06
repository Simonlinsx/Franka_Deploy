cmake_minimum_required(VERSION 3.16)
project(anydex_franka_telemetry_tap VERSION 1.0.0 LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python3 3.9 EXACT COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 3.0.1 EXACT CONFIG REQUIRED)

if(NOT DEFINED LIBFRANKA_INCLUDE_DIR)
  message(FATAL_ERROR "LIBFRANKA_INCLUDE_DIR must name the exact 0.21.2 headers")
endif()
if(NOT DEFINED PYLIBFRANKA_HASHED_LIBRARY)
  message(FATAL_ERROR "PYLIBFRANKA_HASHED_LIBRARY must name the wheel's hashed libfranka DSO")
endif()
if(NOT DEFINED PYLIBFRANKA_LIBS_DIR)
  message(FATAL_ERROR "PYLIBFRANKA_LIBS_DIR must name the wheel's .libs directory")
endif()
if(NOT DEFINED PYLIBFRANKA_MODULE)
  message(FATAL_ERROR "PYLIBFRANKA_MODULE must name the exact wheel extension")
endif()
if(NOT DEFINED PYLIBFRANKA_MODULE_SHA256)
  message(FATAL_ERROR "PYLIBFRANKA_MODULE_SHA256 is required")
endif()
if(NOT DEFINED PYLIBFRANKA_HASHED_LIBRARY_SHA256)
  message(FATAL_ERROR "PYLIBFRANKA_HASHED_LIBRARY_SHA256 is required")
endif()

set(ANYDEX_TELEMETRY_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(ANYDEX_TELEMETRY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ANYDEX_TELEMETRY_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/../telemetry"
  "${CMAKE_CURRENT_BINARY_DIR}/telemetry_core")

pybind11_add_module(
  _anydex_franka_telemetry
  MODULE
  src/bindings.cpp
  src/cartesian_segment.cpp)
target_compile_features(_anydex_franka_telemetry PRIVATE cxx_std_17)
target_compile_options(
  _anydex_franka_telemetry PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(
  _anydex_franka_telemetry
  PRIVATE
  ANYDEX_PYLIBFRANKA_MODULE_PATH="${PYLIBFRANKA_MODULE}"
  ANYDEX_PYLIBFRANKA_MODULE_SHA256="${PYLIBFRANKA_MODULE_SHA256}"
  ANYDEX_LIBFRANKA_LIBRARY_PATH="${PYLIBFRANKA_HASHED_LIBRARY}"
  ANYDEX_LIBFRANKA_LIBRARY_SHA256="${PYLIBFRANKA_HASHED_LIBRARY_SHA256}"
  ANYDEX_LIBFRANKA_VERSION="0.21.2")
target_include_directories(
  _anydex_franka_telemetry PRIVATE "${LIBFRANKA_INCLUDE_DIR}")
target_link_libraries(
  _anydex_franka_telemetry
  PRIVATE
  anydex::telemetry
  "${PYLIBFRANKA_HASHED_LIBRARY}")
# The audit-pinned wheel libfranka has hashed Poco dependencies in the same
# private directory.  Use DT_RPATH (not non-transitive DT_RUNPATH) so the
# extension can be imported first as well as after pylibfranka, while still
# resolving only that exact wheel directory.
target_link_options(
  _anydex_franka_telemetry PRIVATE "-Wl,--disable-new-dtags")
set_target_properties(
  _anydex_franka_telemetry PROPERTIES
  BUILD_RPATH "${PYLIBFRANKA_LIBS_DIR}"
  INSTALL_RPATH "${PYLIBFRANKA_LIBS_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
