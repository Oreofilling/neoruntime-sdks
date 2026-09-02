# NeoRuntime SDKs

SDKs for building applications on the NeoRuntime edge AI platform.

This repository publishes the Python SDK, the C++ SDK, and shared protocol
definitions for NeoRuntime applications.

## Contents

- `proto/` - source protocol definitions copied from the platform repository
- `python/` - Python SDK package, examples, tests, and Sphinx documentation
- `cpp/` - C++ SDK package, examples, tests, CMake config, and Doxygen docs
- `.github/workflows/pages.yml` - GitHub Pages documentation publishing
- `.github/workflows/sync-platform-protos.yml` - platform proto synchronization
- `.github/workflows/wheel.yml` - SDK release artifact publishing

## Documentation

After GitHub Pages is enabled for this repository, the SDK documentation is
published at:

- `https://camthink-ai.github.io/neoruntime-sdks/`
- `https://camthink-ai.github.io/neoruntime-sdks/python/en/`
- `https://camthink-ai.github.io/neoruntime-sdks/python/zh/`
- `https://camthink-ai.github.io/neoruntime-sdks/cpp/en/`
- `https://camthink-ai.github.io/neoruntime-sdks/cpp/zh/`

## Python SDK

Install from PyPI:

```bash
python -m pip install neoruntime-ipc-sdk
```

Pre-release builds are staged on TestPyPI for validation (dependencies still
come from PyPI):

```bash
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ neoruntime-ipc-sdk
```

Install from source:

```bash
python -m pip install -e ./python
```

Run tests:

```bash
python -m pytest -q python/tests
```

Build a Python wheel:

```bash
cd python
python -m pip install --upgrade build
python -m build --wheel
ls dist/*.whl
```

Generated wheels are written to `python/dist/` and should not be committed.

The repository also builds release artifacts automatically with GitHub Actions:

- Pull requests, pushes to `main`, and manual runs build and upload Python
  distributions (wheel + sdist) as an artifact named `python-sdk-dist`.
- Tags matching `v*` build the Python distributions plus a C++ SDK tarball,
  attach them to a GitHub Release, upload the Python distributions to
  TestPyPI, and then to PyPI once the protected `pypi` environment is
  approved.
- Manual runs can also publish a GitHub Release, TestPyPI, or PyPI when the
  corresponding dispatch inputs are enabled.

Release a version by pushing a tag that matches `python/setup.py` (and
`python/neoruntime_ipc_sdk/__init__.py`, which must stay in sync). The release
tag follows the Python SDK version; the C++ SDK keeps its own package version in
the tarball filename.

```bash
git tag v0.5.0
git push origin v0.5.0
```

The GitHub Release assets are:

- `neoruntime_ipc_sdk-0.5.0-py3-none-any.whl`
- `neoruntime_ipc_sdk-0.5.0.tar.gz`
- `ne503-aipc-cpp-sdk-0.2.0-linux-arm64.tar.gz`

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full release and publishing
runbook.

Build local documentation:

```bash
python -m pip install -r python/docs/requirements.txt
python -m sphinx -b html python/docs /tmp/neoruntime-sdk-docs/python/zh
python -m sphinx -b html python/docs/en /tmp/neoruntime-sdk-docs/python/en
```

## C++ SDK

Build and test locally:

```bash
cmake -S cpp -B build-x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build-x64 -j
ctest --test-dir build-x64
```

Build an installable C++ SDK tarball:

```bash
scripts/package_cpp_sdk.sh
```

Cross-compile the release tarball for the aarch64 device target:

```bash
PACKAGE_ARCH=arm64 \
BUILD_DIR="$PWD/build-arm64-package" \
CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=$PWD/cpp/cmake/aarch64-toolchain.cmake -DVCPKG_TARGET_TRIPLET=arm64-linux" \
  scripts/package_cpp_sdk.sh
```

The tarball contains the installed SDK layout: headers, static library, CMake
package files, documentation metadata, and example binaries when examples are
enabled.

## Protocol Sync

`proto/` mirrors the protobuf interfaces from `camthink-ai/neoruntime`.
When platform proto files change, the platform repository can dispatch
`.github/workflows/sync-platform-protos.yml` in this repository. The sync job
copies the latest proto files, regenerates Python stubs, runs tests, and opens
a pull request when committed SDK files changed.

If the platform repository is private, configure `PLATFORM_REPO_TOKEN` in this
repository's Actions secrets. The token only needs read access to
`camthink-ai/neoruntime` contents.

Run the same check locally against a sibling platform checkout:

```bash
PLATFORM_WORKTREE=../neoruntime scripts/check_interface_drift.sh
```

Regenerate SDK interfaces locally:

```bash
PLATFORM_WORKTREE=../neoruntime scripts/sync_platform_protos.sh
```

## Related Repositories

- `camthink-ai/neoruntime` - NeoRuntime platform core
- `camthink-ai/neoruntime-apps` - sample apps and app templates

## License

This repository is licensed under the MIT License. See [LICENSE](./LICENSE).
