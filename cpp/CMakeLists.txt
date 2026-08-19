cmake_minimum_required(VERSION 3.20)

# ============================================================================
# NE503 AIPC C++ SDK
#
# Full mirror of python/hailo_ipc_sdk. gRPC clients use sync stubs over UDS;
# media/audio clients use raw UDS + SCM_RIGHTS fd-passing. See cpp/README.md.
#
# Native build (x86 dev host / CI):
#   cmake -S cpp -B build-x64 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-x64 -j && ctest --test-dir build-x64
#
# Cross-compile to aarch64 (NE503 device), chaining vcpkg for deps:
#   cmake -S cpp -B build-arm64 \
#     -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
#     -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=$PWD/cpp/cmake/aarch64-toolchain.cmake \
#     -DVCPKG_TARGET_TRIPLET=arm64-linux
#   cmake --build build-arm64 -j
# ============================================================================

project(ne503_aipc_sdk
    VERSION 0.1.0
    DESCRIPTION "NE503 AIPC C++ SDK"
    LANGUAGES CXX)

# ---- Options ---------------------------------------------------------------
option(NE503_REGEN_PROTOS   "Regenerate C++ protobuf/gRPC stubs at build time" ON)
option(NE503_BUILD_EXAMPLES "Build the example apps" ON)
option(NE503_BUILD_TESTS    "Build the GoogleTest suite" ON)

# ---- Paths -----------------------------------------------------------------
# Repo root (one level up from cpp/): holds the shared proto/ source of truth.
get_filename_component(NE503_REPO_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(NE503_PROTO_ROOT "${NE503_REPO_ROOT}/proto")
set(NE503_PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto_gen")

# ---- Language standard -----------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- Dependencies ----------------------------------------------------------
# Dual-mode dependency resolution: prefer config-mode (vcpkg / modern installs,
# which expose the gRPC::* and protobuf::* imported targets); fall back to what
# a plain system-apt host provides (CMake's built-in FindProtobuf module + pkg
# -config for gRPC). CI uses vcpkg (config-mode); the fallback lets the SDK build
# and run its test suite on a dev box without a vcpkg checkout, so the source is
# verifiable in more than one environment.
#
# gRPC pulls abseil/re2/upb transitively in config-mode — do NOT hand-pin abseil
# (ABI). When cross-compiling, VCPKG_HOST_TRIPLET (x64) supplies protoc + plugin.

# --- Protobuf: config (vcpkg) then CMake's built-in module (apt) -------------
find_package(Protobuf CONFIG QUIET)
set(NE503_PROTOBUF_CONFIG ${Protobuf_FOUND})
if(NOT NE503_PROTOBUF_CONFIG)
    find_package(Protobuf MODULE REQUIRED)
endif()

# --- gRPC: config (vcpkg, gives gRPC::* targets) then pkg-config (apt) ------
find_package(gRPC CONFIG QUIET)
set(NE503_GRPC_CONFIG ${gRPC_FOUND})
if(NOT NE503_GRPC_CONFIG)
    # System-apt gRPC (>=1.30) ships pkg-config .pc files, not cmake config.
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRPCPP IMPORTED_TARGET grpc++ REQUIRED)
endif()

# --- OpenCV / nlohmann_json: config-mode in both vcpkg and apt --------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(nlohmann_json CONFIG REQUIRED)

# Resolve codegen tools, with manual-override escape hatches for unusual
# cross-compile setups (e.g. a Yocto SDK shipping its own protoc/plugin, or a
# system-apt host where the binaries live at /usr/bin).
if(DEFINED NE503_PROTOC)
    set(NE503_PROTOC_EXE "${NE503_PROTOC}")
elseif(NE503_PROTOBUF_CONFIG AND TARGET protobuf::protoc)
    set(NE503_PROTOC_EXE "$<TARGET_FILE:protobuf::protoc>")
elseif(DEFINED Protobuf_PROTOC_EXECUTABLE)
    set(NE503_PROTOC_EXE "${Protobuf_PROTOC_EXECUTABLE}")
else()
    message(FATAL_ERROR "protoc not found. Set -DNE503_PROTOC=/path/to/protoc")
endif()

if(DEFINED NE503_GRPC_CPP_PLUGIN)
    set(NE503_GRPC_PLUGIN_EXE "${NE503_GRPC_CPP_PLUGIN}")
elseif(NE503_GRPC_CONFIG AND TARGET gRPC::grpc_cpp_plugin)
    set(NE503_GRPC_PLUGIN_EXE "$<TARGET_FILE:gRPC::grpc_cpp_plugin>")
else()
    find_program(NE503_GRPC_CPP_PLUGIN_EXE grpc_cpp_plugin)
    if(NE503_GRPC_CPP_PLUGIN_EXE)
        set(NE503_GRPC_PLUGIN_EXE "${NE503_GRPC_CPP_PLUGIN_EXE}")
    else()
        message(FATAL_ERROR "grpc_cpp_plugin not found. Set -DNE503_GRPC_CPP_PLUGIN=...")
    endif()
endif()

# ---- Protobuf/gRPC code generation -----------------------------------------
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateProtos.cmake")

file(GLOB_RECURSE NE503_PROTO_FILES RELATIVE "${NE503_PROTO_ROOT}"
    "${NE503_PROTO_ROOT}/*.proto")

if(NE503_REGEN_PROTOS AND NE503_PROTO_FILES)
    ne503_generate_protos(
        PROTO_ROOT "${NE503_PROTO_ROOT}"
        OUTPUT_DIR "${NE503_PROTO_GEN_DIR}"
        PROTOC     "${NE503_PROTOC_EXE}"
        PLUGIN     "${NE503_GRPC_PLUGIN_EXE}"
        PROTOS     ${NE503_PROTO_FILES})
endif()

# ---- The library -----------------------------------------------------------
add_library(ne503_aipc_sdk STATIC)
add_library(ne503::aipc_sdk ALIAS ne503_aipc_sdk)

target_include_directories(ne503_aipc_sdk
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${NE503_PROTO_GEN_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Hand-written implementation sources. CONFIGURE_DEPENDS picks up newly added
# .cpp files (one per client) as each phase lands.
file(GLOB NE503_SDK_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/media/*.cpp")

# Header-only internal helpers actually on disk (the GLOB above only catches
# .cpp). tensor_io.hpp includes the generated inference.pb.h, and the media
# clients include fd_protocol.hpp / raw_socket.hpp, so all four live here under
# the PRIVATE include path.
target_sources(ne503_aipc_sdk PRIVATE ${NE503_SDK_SOURCES}
    src/detail/endpoint.hpp
    src/detail/grpc_channel.hpp
    src/detail/tensor_io.hpp
    src/detail/fd_protocol.hpp
    src/detail/raw_socket.hpp)

if(NE503_REGEN_PROTOS AND NE503_PROTO_FILES)
    target_sources(ne503_aipc_sdk PRIVATE ${ne503_proto_generated_srcs})
    add_dependencies(ne503_aipc_sdk ne503_proto_gen)
endif()

# Link gRPC. Config-mode exposes gRPC::grpc++ (+reflection); the pkg-config
# fallback exposes PkgConfig::GRPCPP. Reflection is only used by plugin/server
# discovery, so link it when available and skip silently on minimal installs.
if(NE503_GRPC_CONFIG)
    set(NE503_GRPCPP_LINK gRPC::grpc++)
else()
    set(NE503_GRPCPP_LINK PkgConfig::GRPCPP)
endif()
if(NE503_GRPC_CONFIG AND TARGET gRPC::grpc++_reflection)
    set(NE503_GRPC_REFLECTION_LINK gRPC::grpc++_reflection)
else()
    set(NE503_GRPC_REFLECTION_LINK)
endif()

target_link_libraries(ne503_aipc_sdk
    PUBLIC
        ${NE503_GRPCPP_LINK}
        protobuf::libprotobuf
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
    PRIVATE
        ${NE503_GRPC_REFLECTION_LINK})

target_compile_options(ne503_aipc_sdk PRIVATE
    -Wall -Wextra -Wpedantic -Wno-unused-parameter)

# ---- Examples / tests ------------------------------------------------------
if(NE503_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

if(NE503_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Install / export ------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ne503_aipc_sdk
    EXPORT ne503_aipc_sdkTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ne503_aipc_sdkTargets
    FILE ne503_aipc_sdkTargets.cmake
    NAMESPACE ne503::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ne503_aipc_sdk)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ne503-sdk-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ne503_aipc_sdkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ne503_aipc_sdk)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ne503_aipc_sdkConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ne503_aipc_sdk)
