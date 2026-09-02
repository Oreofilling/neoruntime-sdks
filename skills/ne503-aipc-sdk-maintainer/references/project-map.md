# Project Map

## Layout

- `proto/`: protobuf definitions copied from the NeoRuntime platform repository.
- `python/neoruntime_ipc_sdk/`: Python SDK package.
- `python/neoruntime_ipc_sdk/proto/`: generated Python protobuf and gRPC stubs committed with the package.
- `python/tests/`: daemon-independent pytest suite.
- `python/docs/` and `python/docs/en/`: Sphinx docs for Chinese and English outputs.
- `cpp/include/neoruntime_ipc_sdk/`: public C++ headers.
- `cpp/src/`: C++ implementation and internal transport helpers.
- `cpp/examples/`: on-device and development examples.
- `cpp/tests/`: GoogleTest suite for daemon-independent behavior.
- `scripts/check_interface_drift.sh`: verifies SDK proto sources and generated Python stubs match the platform repository.
- `scripts/sync_platform_protos.sh`: copies platform proto files and regenerates Python protobuf stubs.
- `scripts/package_cpp_sdk.sh`: builds, installs, and packages the C++ SDK tarball.
- `build-x64/`, `build-arm64/`, and similar directories are generated build output.

## SDK Modules

- `inference`: AI inference, batch inference, streaming subscription, model registration, stats, and result value types.
- `device`: lights, IR, IRCUT, PTZ, zoom/focus, GPIO, RS485, Wiegand, status, and device events.
- `camera`: ISP, encoder, RTSP, OSD, profiles, camera status, and capabilities.
- `app`: app-container lifecycle, logs, and stats.
- `events`: event-bus publish/subscribe, batch publishing, and topic metadata.
- `media`: DMA-BUF raw frames through fd passing, plus encoded H.264/H.265 streams.
- `audio` and `audio_stream`: audio control, PCM/AAC capture, and audio frame decoding.
- `overlay`: AI/video overlay configuration.
- `plugin`: plugin endpoint discovery and plugin server helpers.
- `config`: environment-variable backed endpoint and path configuration.

## Environment Variables

- `APP_ID`: current application id, default `unknown`.
- `AI_RUNTIME_ENDPOINT`: inference service UDS endpoint.
- `EVENT_BUS_ENDPOINT`: event bus UDS endpoint.
- `DEVICE_CONTROL_ENDPOINT`: device control UDS endpoint.
- `CAMERA_CONTROL_ENDPOINT`: camera control UDS endpoint.
- `APP_MANAGER_ENDPOINT`: app manager UDS endpoint.
- `SHM_BASE_PATH`: shared-memory IPC socket base path.
- `ENCODED_SOCKET_DIR`: encoded stream socket directory.
- `AIPC_HOST_PREFIX`: host install prefix used for `/opt/aipc` or `/data/aipc` path translation.
- `DEBUG`: debug flag, enabled only when set to `1`.
- `LOG_LEVEL`: log verbosity, default `INFO`.

## Proto Mapping

`scripts/sync_platform_protos.sh` copies these platform files:

- `platform/ai-runtime/proto/inference.proto` to `proto/ai-runtime/inference.proto`
- `platform/app-manager/proto/app.proto` to `proto/app-manager/app.proto`
- `platform/camera-daemon/proto/camera.proto` to `proto/camera-daemon/camera.proto`
- `platform/camera-daemon/proto/lens_hal.proto` to `proto/camera-daemon/lens_hal.proto`
- `platform/device-control/proto/device.proto` to `proto/device-control/device.proto`
- `platform/device-discovery/proto/discovery.proto` to `proto/device-discovery/discovery.proto`
- `platform/event-bus/proto/event.proto` to `proto/event-bus/event.proto`

The script regenerates Python stubs for inference, app, camera, device, and event protos, then rewrites generated gRPC imports to package-relative imports.

## Common Commands

```sh
python -m pip install -e ./python
python -m pytest -q python/tests
cd python && python -m pip install --upgrade build && python -m build --wheel
cmake -S cpp -B build-x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build-x64 -j
ctest --test-dir build-x64
PLATFORM_WORKTREE=../neoruntime scripts/check_interface_drift.sh
PLATFORM_WORKTREE=../neoruntime scripts/sync_platform_protos.sh
scripts/package_cpp_sdk.sh
python -m sphinx -b html python/docs /tmp/neoruntime-sdk-docs/python/zh
python -m sphinx -b html python/docs/en /tmp/neoruntime-sdk-docs/python/en
```
