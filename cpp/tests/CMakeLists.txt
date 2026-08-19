# tests/CMakeLists.txt — GoogleTest mirror of python/tests/.
#
# Scope: these are the daemon-independent, x86-runnable logic tests. They cover
# env-var configuration (Config), inference value-type logic (BoundingBox /
# InferenceResult), the packed media/audio wire structs (fd_protocol.hpp), and
# AudioFrame decoding. Anything that needs a live platform daemon (a real
# ai-runtime.sock / camera.sock / DMA-BUF fd) is an on-device integration test
# and is NOT built here — see the plan's Verification section.
#
# Tests link the full ne503::aipc_sdk static lib so they exercise the real
# compiled objects (config.cpp, audio_stream.cpp, …) rather than a parallel
# re-implementation. proto-gen + gRPC + OpenCV are therefore required on the
# host, which matches the native x64 CI image.

# ---- GoogleTest -------------------------------------------------------------
# Prefer the system package (apt libgtest-dev ships GTestConfig.cmake → the
# GTest::* imported targets the executables below link). Fall back to a source
# download only when no system package is present, so this works in both bare
# CI images and vcpkg hosts.
find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        # v1.14.0 has no release assets; the auto-generated source archive is
        # the only downloadable artifact. URL_HASH pins the exact tree.
        URL "https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
    )
    # On Windows / newer toolchains gtest needs this for shared CRT; harmless elsewhere.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# ---- Test helpers ----------------------------------------------------------
# fd_protocol.hpp is a PRIVATE header (lives under src/detail/); the wire-struct
# test needs it on its include path. The public headers (Config, inference,
# AudioFrame) ride the SDK target's PUBLIC include dir.
set(NE503_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")

# ---- test_config -----------------------------------------------------------
add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_config PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_config)

# ---- test_inference --------------------------------------------------------
# BoundingBox + InferenceResult logic is header-inline, but link the SDK anyway
# for symmetry and so a future proto-backed assertion drops in cleanly.
add_executable(test_inference test_inference.cpp)
target_link_libraries(test_inference PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_inference PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_inference)

# ---- test_wire_protocol ----------------------------------------------------
add_executable(test_wire_protocol test_wire_protocol.cpp)
target_include_directories(test_wire_protocol PRIVATE "${NE503_SRC_DIR}")
target_link_libraries(test_wire_protocol PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_wire_protocol PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_wire_protocol)

# ---- test_audio_frame ------------------------------------------------------
add_executable(test_audio_frame test_audio_frame.cpp)
target_link_libraries(test_audio_frame PRIVATE ne503::aipc_sdk GTest::gtest_main)
target_compile_options(test_audio_frame PRIVATE -Wall -Wextra -Wpedantic)
gtest_discover_tests(test_audio_frame)

set_target_properties(test_config test_inference test_wire_protocol test_audio_frame
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
