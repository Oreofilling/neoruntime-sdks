---
name: ne503-aipc-sdk-maintainer
description: Maintain and evolve the NE503/NeoRuntime AIPC SDK repository. Use when Codex needs to change, review, debug, document, release, or validate the Python SDK under python/neoruntime_ipc_sdk, the C++ SDK under cpp/, shared protobuf interfaces under proto/, examples, tests, packaging, docs, or platform proto sync workflows.
---

# NE503 AIPC SDK Maintainer

## First Pass

1. Check `git status --short` and preserve user changes.
2. Classify the request as Python SDK, C++ SDK, proto/interface sync, docs, packaging/release, or cross-language.
3. Read the smallest relevant files first. For broad or unfamiliar work, read `references/project-map.md`.
4. Prefer existing module patterns over new abstractions. This repo mirrors concepts between Python and C++; keep that symmetry unless the request is explicitly language-specific.

## Repository Rules

- Treat `proto/` as shared interface source copied from the platform repository.
- Do not hand-edit `python/neoruntime_ipc_sdk/proto/*_pb2*.py`; regenerate those stubs through `scripts/sync_platform_protos.sh`.
- When adding or changing public Python APIs, update implementation, tests, docs/examples when relevant, and `python/neoruntime_ipc_sdk/__init__.py` exports if the symbol is public.
- When adding or changing public C++ APIs, update the public header, source file, examples/tests when relevant, and keep C++17 compatibility.
- Keep daemon-independent tests runnable on a development host. Anything requiring live UDS sockets, camera hardware, DMA-BUF fds, or platform daemons belongs in smoke/integration guidance rather than normal unit tests.
- Keep default endpoints and environment-variable behavior aligned between Python `config.py` and C++ config code.

## Workflows

### Python SDK

- Install editable package when needed: `python -m pip install -e ./python`.
- Run the main test suite: `python -m pytest -q python/tests`.
- For targeted changes, run the relevant `python/tests/test_*.py` first, then broaden if shared behavior changed.
- Build a wheel from `python/`: `python -m pip install --upgrade build && python -m build --wheel`.

### C++ SDK

- Configure and test native builds with:

```sh
cmake -S cpp -B build-x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build-x64 -j
ctest --test-dir build-x64
```

- Reuse the existing build directory when it matches the task. Create a new build directory only when isolation or changed options matter.
- For generated proto build issues, inspect `cpp/cmake/GenerateProtos.cmake` and the `NE503_REGEN_PROTOS`, `NE503_PROTOC`, and `NE503_GRPC_CPP_PLUGIN` options.

### Proto And Interface Sync

- Check drift against a sibling platform checkout with:

```sh
PLATFORM_WORKTREE=../neoruntime scripts/check_interface_drift.sh
```

- Regenerate SDK interfaces from a platform checkout with:

```sh
PLATFORM_WORKTREE=../neoruntime scripts/sync_platform_protos.sh
```

- After proto changes, validate generated Python stubs, Python tests, and C++ compilation if the changed interface is consumed by C++.

### Docs And Releases

- For Python docs, update the relevant Sphinx source under `python/docs/` and mirror English content under `python/docs/en/` when appropriate.
- Build docs with:

```sh
python -m pip install -r python/docs/requirements.txt
python -m sphinx -b html python/docs /tmp/neoruntime-sdk-docs/python/zh
python -m sphinx -b html python/docs/en /tmp/neoruntime-sdk-docs/python/en
```

- Keep release version values aligned between `python/setup.py`, `python/neoruntime_ipc_sdk/__init__.py`, README release examples, and tags such as `v0.4.0`.
- Release tags follow the Python SDK version. GitHub Release assets include the Python wheel and the independently versioned C++ SDK tarball.

## Validation Matrix

- Python-only implementation: run targeted pytest, then `python -m pytest -q python/tests` when behavior is shared.
- C++ implementation: run `cmake --build build-x64 -j` and `ctest --test-dir build-x64`.
- Proto/interface sync: run the sync or drift script, then Python tests and C++ build if consumers changed.
- Docs-only: run the relevant Sphinx build if dependencies are available; otherwise report that docs were not rendered.
- Packaging/release: build the wheel and verify version consistency.
- C++ packaging/release: run `scripts/package_cpp_sdk.sh`; cross-release builds set `PACKAGE_ARCH=arm64` and pass the aarch64 CMake toolchain through `CMAKE_ARGS`.

## Reference

Read `references/project-map.md` for module ownership, command summary, environment variables, and proto mapping details when the task spans more than one file or language.
