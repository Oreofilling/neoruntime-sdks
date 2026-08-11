# aarch64-linux.cmake — optional vcpkg overlay triplet for the NE503 device.
#
# vcpkg already ships a battle-tested builtin `arm64-linux` triplet that targets
# aarch64-linux-gnu, so the default cross-compile flow does NOT require this
# file. Install it as an overlay only if you need to pin a specific compiler or
# tweak feature flags:
#
#   cmake ... -DVCPKG_OVERLAY_TRIPLETS=$PWD/cpp/cmake -DVCPKG_TARGET_TRIPLET=aarch64-linux
#
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)

# Match the toolchain prefix used by aarch64-toolchain.cmake.
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE
    "${CMAKE_CURRENT_LIST_DIR}/aarch64-toolchain.cmake"
    CACHE STRING "NE503 aarch64 compiler toolchain")
