cmake_minimum_required(VERSION 3.20)
project(dbs VERSION 1.0.0 LANGUAGES C CXX)

option(DBS_BUILD_SHARED "Build shared library" ON)
option(DBS_BUILD_TESTS "Build tests" ON)
option(DBS_BUILD_BENCHMARKS "Build benchmarks" ON)
option(DBS_BUILD_FUZZER "Build libFuzzer harness" OFF)
option(DBS_ENABLE_SANITIZERS "Enable Address/Undefined sanitizers" OFF)
option(DBS_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(DBS_ENABLE_MSAN "Enable MemorySanitizer with Clang" OFF)
option(DBS_ENABLE_CUDA "Build CUDA backend target when CUDA is available" OFF)
option(DBS_CUDA_USE_FAST_MATH "Compile CUDA kernels with --use_fast_math" OFF)
option(DBS_BUILD_HARDWARE_GATES "Expose hardware validation helper targets" ON)

set(DBS_ABI_VERSION 10 CACHE STRING "Public C ABI version from include/dbs.h")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
  if(NOT DEFINED CMAKE_CXX_FLAGS_RELEASE OR CMAKE_CXX_FLAGS_RELEASE STREQUAL "")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG" CACHE STRING "Release flags")
  endif()
else()
  if(NOT DEFINED CMAKE_CXX_FLAGS_RELEASE OR CMAKE_CXX_FLAGS_RELEASE STREQUAL "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "Release flags")
  endif()
endif()
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

set(DBS_LIBRARY_TYPE STATIC)
if(DBS_BUILD_SHARED)
  set(DBS_LIBRARY_TYPE SHARED)
endif()

add_library(dbs_core_objects OBJECT src/differentiable_beam.cpp)
set_target_properties(dbs_core_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dbs_core_objects PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(DBS_BUILD_SHARED)
  target_compile_definitions(dbs_core_objects PRIVATE DBS_BUILD_SHARED DBS_COMPILING_LIBRARY)
else()
  target_compile_definitions(dbs_core_objects PRIVATE DBS_STATIC)
endif()
if(MSVC)
  target_compile_options(dbs_core_objects PRIVATE /W4 /permissive-)
else()
  target_compile_options(dbs_core_objects PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(DBS_ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(dbs_core_objects PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
endif()
if(DBS_ENABLE_TSAN AND NOT MSVC)
  target_compile_options(dbs_core_objects PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
endif()
if(DBS_ENABLE_MSAN AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(dbs_core_objects PUBLIC -fsanitize=memory -fsanitize-memory-track-origins -fno-omit-frame-pointer)
endif()

add_library(dbs ${DBS_LIBRARY_TYPE} $<TARGET_OBJECTS:dbs_core_objects>)
add_library(dbs::dbs ALIAS dbs)
target_include_directories(dbs PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(DBS_BUILD_SHARED)
  target_compile_definitions(dbs PUBLIC DBS_BUILD_SHARED)
else()
  target_compile_definitions(dbs PUBLIC DBS_STATIC)
endif()
if(DBS_ENABLE_SANITIZERS AND NOT MSVC)
  target_link_options(dbs PUBLIC -fsanitize=address,undefined)
endif()
if(DBS_ENABLE_TSAN AND NOT MSVC)
  target_link_options(dbs PUBLIC -fsanitize=thread)
endif()
if(DBS_ENABLE_MSAN AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_link_options(dbs PUBLIC -fsanitize=memory -fsanitize-memory-track-origins)
endif()
if(UNIX AND NOT APPLE)
  target_link_options(dbs PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/dbs.map")
endif()
set_target_properties(dbs PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${DBS_ABI_VERSION})

if(DBS_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(dbs_cuda ${DBS_LIBRARY_TYPE} cuda/dbs_cuda.cu)
  set_target_properties(dbs_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(dbs_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  if(DBS_CUDA_USE_FAST_MATH)
    target_compile_options(dbs_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()
else()
  # Stable non-CUDA target: downstream consumers can link dbs::dbs_cuda on
  # all platforms. In non-CUDA builds it resolves to a stub that returns
  # DBS_CUDA_STATUS_UNAVAILABLE from every runtime entry point.
  add_library(dbs_cuda ${DBS_LIBRARY_TYPE} cuda/dbs_cuda_stub.cpp)
  target_compile_definitions(dbs_cuda PUBLIC DBS_CUDA_STUB=1)
endif()

add_library(dbs::dbs_cuda ALIAS dbs_cuda)
if(DBS_BUILD_SHARED)
  set_target_properties(dbs_cuda PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${DBS_ABI_VERSION})
endif()

target_include_directories(dbs_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(DBS_BUILD_SHARED)
  target_compile_definitions(dbs_cuda PRIVATE DBS_BUILD_SHARED DBS_COMPILING_LIBRARY)
  target_compile_definitions(dbs_cuda PUBLIC DBS_BUILD_SHARED)
else()
  target_compile_definitions(dbs_cuda PUBLIC DBS_STATIC)
endif()
install(TARGETS dbs_cuda EXPORT dbsTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(TARGETS dbs EXPORT dbsTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT dbsTargets FILE dbsTargets.cmake NAMESPACE dbs:: DESTINATION lib/cmake/dbs)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/dbsConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dbsConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/dbsConfig.cmake"
  INSTALL_DESTINATION lib/cmake/dbs)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/dbsConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/dbsConfigVersion.cmake"
  DESTINATION lib/cmake/dbs)
install(FILES cmake/dbs.map DESTINATION lib/cmake/dbs)

configure_file(pkgconfig/dbs.pc.in dbs.pc @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dbs.pc" DESTINATION lib/pkgconfig)

if(DBS_BUILD_TESTS)
  enable_testing()
  add_executable(dbs_tests tests/dbs_tests.cpp)
  target_link_libraries(dbs_tests PRIVATE dbs)
  add_test(NAME dbs_tests COMMAND dbs_tests)

  add_executable(dbs_simd_internal_tests tests/simd_internal_tests.cpp $<TARGET_OBJECTS:dbs_core_objects>)
  target_include_directories(dbs_simd_internal_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(dbs_simd_internal_tests PRIVATE DBS_STATIC)
  if(DBS_ENABLE_SANITIZERS AND NOT MSVC)
    target_link_options(dbs_simd_internal_tests PRIVATE -fsanitize=address,undefined)
  endif()
  if(DBS_ENABLE_TSAN AND NOT MSVC)
    target_link_options(dbs_simd_internal_tests PRIVATE -fsanitize=thread)
  endif()
  if(DBS_ENABLE_MSAN AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_options(dbs_simd_internal_tests PRIVATE -fsanitize=memory -fsanitize-memory-track-origins)
  endif()
  add_test(NAME dbs_simd_internal_tests COMMAND dbs_simd_internal_tests)

  if(DBS_ENABLE_CUDA)
    add_executable(dbs_cuda_c_api_tests tests/cuda_c_api_tests.cpp)
    target_link_libraries(dbs_cuda_c_api_tests PRIVATE dbs_cuda CUDA::cudart)
    add_test(NAME dbs_cuda_c_api_tests COMMAND dbs_cuda_c_api_tests)
  endif()
endif()

if(DBS_BUILD_BENCHMARKS)
  add_executable(dbs_bench benchmarks/dbs_bench.cpp)
  target_link_libraries(dbs_bench PRIVATE dbs)
endif()

if(DBS_BUILD_FUZZER AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_executable(dbs_fuzz tests/fuzz_dbs.cpp)
  target_link_libraries(dbs_fuzz PRIVATE dbs)
  target_compile_options(dbs_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_options(dbs_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()


if(DBS_BUILD_HARDWARE_GATES)
  add_custom_target(dbs_abi_check
    COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_abi.sh $<TARGET_FILE:dbs>
    DEPENDS dbs
    COMMENT "Checking exported dbs_* ABI symbols")
  add_custom_target(dbs_abi_compat_check
    COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_abi_compat.sh $<TARGET_FILE:dbs>
    DEPENDS dbs
    COMMENT "Checking backwards-compatible dbs_* ABI symbols")
  add_custom_target(dbs_hardware_validation
    COMMAND ${CMAKE_COMMAND} -E env DBS_REQUIRE_CUDA=0 bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_hardware_validation.sh
    COMMENT "Running CPU/CUDA hardware validation gate")
  add_custom_target(dbs_release_gate
    COMMAND ${CMAKE_COMMAND} -E env DBS_REQUIRE_CUDA=1 DBS_REQUIRE_TORCH=1 DBS_REQUIRE_LONG_FUZZ=1 bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_release_gate.sh
    COMMENT "Running fail-closed production release gate")
endif()
