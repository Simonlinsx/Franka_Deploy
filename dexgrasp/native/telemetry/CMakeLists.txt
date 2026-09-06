cmake_minimum_required(VERSION 3.16)
project(anydex_native_telemetry VERSION 1.0.0 LANGUAGES CXX)

option(ANYDEX_TELEMETRY_BUILD_PYTHON "Build the reader/test-only pybind11 module" ON)
option(ANYDEX_TELEMETRY_BUILD_TESTS "Build native unit/stress tests" ON)
option(ANYDEX_TELEMETRY_BUILD_TOOLS "Build offline stress and benchmark tools" ON)

add_library(anydex_telemetry STATIC src/telemetry.cpp)
add_library(anydex::telemetry ALIAS anydex_telemetry)
target_compile_features(anydex_telemetry PUBLIC cxx_std_17)
target_include_directories(
  anydex_telemetry
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
set_target_properties(anydex_telemetry PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(anydex_telemetry PRIVATE -Wall -Wextra -Wpedantic -Werror)

if(ANYDEX_TELEMETRY_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_anydex_telemetry MODULE python/bindings.cpp)
  target_link_libraries(_anydex_telemetry PRIVATE anydex::telemetry)
  target_compile_features(_anydex_telemetry PRIVATE cxx_std_17)
  target_compile_options(_anydex_telemetry PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(
    _anydex_telemetry PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
endif()

if(ANYDEX_TELEMETRY_BUILD_TESTS)
  enable_testing()
  find_package(Threads REQUIRED)
  add_executable(anydex_telemetry_unit tests/test_telemetry.cpp)
  target_link_libraries(anydex_telemetry_unit PRIVATE anydex::telemetry Threads::Threads)
  target_compile_features(anydex_telemetry_unit PRIVATE cxx_std_17)
  target_compile_options(anydex_telemetry_unit PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME anydex_telemetry_unit COMMAND anydex_telemetry_unit)
  if(ANYDEX_TELEMETRY_BUILD_PYTHON)
    add_test(
      NAME anydex_telemetry_python
      COMMAND
        ${CMAKE_COMMAND} -E env
        "PYTHONPATH=$<TARGET_FILE_DIR:_anydex_telemetry>"
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_binding.py")
  endif()
endif()

if(ANYDEX_TELEMETRY_BUILD_TOOLS)
  find_package(Threads REQUIRED)
  add_executable(anydex_telemetry_stress tools/stress.cpp)
  target_link_libraries(anydex_telemetry_stress PRIVATE anydex::telemetry Threads::Threads)
  target_compile_features(anydex_telemetry_stress PRIVATE cxx_std_17)
  target_compile_options(anydex_telemetry_stress PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_executable(anydex_telemetry_benchmark tools/benchmark.cpp)
  target_link_libraries(anydex_telemetry_benchmark PRIVATE anydex::telemetry)
  target_compile_features(anydex_telemetry_benchmark PRIVATE cxx_std_17)
  target_compile_options(anydex_telemetry_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(GNUInstallDirs)
install(TARGETS anydex_telemetry EXPORT anydexTelemetryTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
