# aarch64-toolchain.cmake
# -----------------------------------------------------------------------------
# Cross-compilation toolchain for the NE503 device (aarch64-linux-gnu).
#
# Intended to be CHAINLOADED by vcpkg's toolchain so vcpkg builds the target
# (arm64-linux) triplet while the host triplet supplies protoc + grpc_cpp_plugin:
#
#   cmake -S cpp -B build-arm64 \
#     -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
#     -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=$PWD/cpp/cmake/aarch64-toolchain.cmake \
#     -DVCPKG_TARGET_TRIPLET=arm64-linux
#
# Or, for a Yocto SDK that already ships gRPC/protobuf/opencv in its sysroot
# (vcpkg skipped), source the SDK environment script first and point CMake at
# the Yocto toolchain file instead — this file then only needs to select the
# aarch64 compiler prefix.
# -----------------------------------------------------------------------------

set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
set(NE503_CROSS_PREFIX aarch64-linux-gnu CACHE STRING "Cross-compiler prefix")

set(CMAKE_C_COMPILER   ${NE503_CROSS_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${NE503_CROSS_PREFIX}-g++)

# Optional explicit sysroot (e.g. a Yocto SDK). When unset, the cross
# compiler's default multiarch sysroot (libc6-dev-arm64-cross) is used.
if(DEFINED ENV{NE503_AARCH64_SYSROOT})
    set(CMAKE_SYSROOT "$ENV{NE503_AARCH64_SYSROOT}")
endif()

# Never probe the build host for target features.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Tell vcpkg which triplets to use for target vs host tools.
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET arm64-linux)
endif()
if(NOT DEFINED VCPK_TARGET_TRIPLET) # guard common typo if set elsewhere
    set(VCPKG_HOST_TRIPLET x64-linux)
endif()
