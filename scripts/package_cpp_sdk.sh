#!/usr/bin/env bash
#
# Build and package the installed C++ SDK layout as a tarball.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CPP_VERSION="${CPP_VERSION:-}"
if [ -z "$CPP_VERSION" ]; then
    CPP_VERSION="$(
        sed -n 's/^[[:space:]]*VERSION[[:space:]]\+\([0-9][0-9.]*\).*/\1/p' \
            "$REPO_ROOT/cpp/CMakeLists.txt" | head -1
    )"
fi
if [ -z "$CPP_VERSION" ]; then
    echo "ERROR: could not determine C++ SDK version" >&2
    exit 1
fi

host_arch="$(uname -m)"
case "$host_arch" in
    aarch64|arm64) default_arch="arm64" ;;
    x86_64|amd64) default_arch="x64" ;;
    *) default_arch="$host_arch" ;;
esac

PACKAGE_ARCH="${PACKAGE_ARCH:-$default_arch}"
PACKAGE_PLATFORM="${PACKAGE_PLATFORM:-linux-${PACKAGE_ARCH}}"
PACKAGE_NAME="${PACKAGE_NAME:-ne503-aipc-cpp-sdk-${CPP_VERSION}-${PACKAGE_PLATFORM}}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-cpp-package}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/dist}"
INSTALL_ROOT="${INSTALL_ROOT:-$BUILD_DIR/stage/$PACKAGE_NAME}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

# shellcheck disable=SC2206
extra_cmake_args=(${CMAKE_ARGS:-})

rm -rf "$INSTALL_ROOT"
mkdir -p "$OUTPUT_DIR" "$(dirname "$INSTALL_ROOT")"

cmake -S "$REPO_ROOT/cpp" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_ROOT" \
    -DNE503_REGEN_PROTOS="${NE503_REGEN_PROTOS:-ON}" \
    -DNE503_BUILD_EXAMPLES="${NE503_BUILD_EXAMPLES:-ON}" \
    -DNE503_BUILD_TESTS="${NE503_BUILD_TESTS:-OFF}" \
    "${extra_cmake_args[@]}"

cmake --build "$BUILD_DIR" -j "$JOBS"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_ROOT"

doc_dir="$INSTALL_ROOT/share/doc/ne503-aipc-cpp-sdk"
mkdir -p "$doc_dir"
cp "$REPO_ROOT/LICENSE" "$doc_dir/LICENSE"
cp "$REPO_ROOT/README.md" "$doc_dir/README.repository.md"
cp "$REPO_ROOT/cpp/README.md" "$doc_dir/README.cpp.md"
cat > "$doc_dir/VERSION" <<EOF
name=ne503-aipc-cpp-sdk
version=$CPP_VERSION
platform=$PACKAGE_PLATFORM
mirrored_python_sdk=$(sed -n 's/.*kMirroredPythonSdk\[\] = "\([^"]*\)".*/\1/p' "$REPO_ROOT/cpp/include/hailo_ipc_sdk/version.hpp" | head -1)
EOF

tarball="$OUTPUT_DIR/$PACKAGE_NAME.tar.gz"
rm -f "$tarball"
tar -C "$(dirname "$INSTALL_ROOT")" -czf "$tarball" "$PACKAGE_NAME"

echo "$tarball"
