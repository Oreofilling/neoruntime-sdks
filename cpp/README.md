# NeoRuntime C++ SDK

C++ client SDK for the NeoRuntime edge-AI camera platform — a full mirror of
the Python SDK (`python/hailo_ipc_sdk`), using OpenCV `cv::Mat` where the
Python SDK uses numpy.

Applications built with this SDK run on-device, inside an app container, and
talk to the platform daemons over local Unix Domain Sockets.

## Client modules

| Header | Client | Purpose |
|---|---|---|
| `inference.hpp` | `InferenceClient` | AI inference: `infer` / batch / streaming `subscribe`, model registration & stats, GenAI (LLM/VLM) streaming |
| `device.hpp` | `DeviceClient` | Lights / IR / IRCUT, PTZ / zoom / focus, GPIO, RS485, Wiegand, status, event subscription |
| `camera.hpp` | `CameraClient` | ISP / encoder / RTSP / OSD / profiles, camera status |
| `app.hpp` | `AppClient` | App-container lifecycle, logs, stats |
| `events.hpp` | `EventClient` | Event-bus publish / subscribe / batch / topics |
| `media.hpp` | `FdMediaClient` | Zero-copy DMA-BUF video frames (fd-passing) |
| `media.hpp` | `EncodedStreamClient` | H.264 / H.265 encoded streams |
| `audio.hpp` | `AudioClient` | Audio control |
| `audio_stream.hpp` | `AudioStreamClient` | PCM / AAC audio capture streams |
| `overlay.hpp` | `OverlayClient` | Video overlay configuration |
| `plugin.hpp` | `PluginDiscovery` / `PluginServer` | Plugin endpoint discovery & serving |

Shared value types live in `types.hpp`; endpoints & environment configuration
in `config.hpp`.

## Transports

Three transports sit under those clients:

- **gRPC over UDS** — `InferenceClient`, `DeviceClient`, `CameraClient`,
  `AppClient`, `EventClient`, `OverlayClient`, plugin discovery. C++ uses
  synchronous stubs; the Python SDK's background-asyncio machinery is
  intentionally not ported (sync stubs already block on a futex).
- **Raw UDS + `SCM_RIGHTS` fd-passing** — `FdMediaClient`. Received DMA-BUF
  plane fds are `mmap`-ed per plane into `cv::Mat` for true zero-copy.
- **Raw UDS, length-prefixed** — `EncodedStreamClient`, `AudioStreamClient`.
  Packed wire structs mirror the Python layouts.

## Build

C++17, CMake ≥ 3.20, vcpkg manifest mode (protobuf, grpc, opencv4 — headless:
core/imgproc/imgcodecs only, nlohmann-json).

```sh
# Native (x86_64 dev host / CI)
cmake -S cpp -B build-x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build-x64 -j && ctest --test-dir build-x64

# Cross-compile for the aarch64 device
cmake -S cpp -B build-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=$PWD/cpp/cmake/aarch64-toolchain.cmake \
  -DVCPKG_TARGET_TRIPLET=arm64-linux
cmake --build build-arm64 -j
```

Build options: `NE503_BUILD_EXAMPLES` (default ON), `NE503_BUILD_TESTS`
(default ON), `NE503_REGEN_PROTOS` (default ON — regenerates protobuf/gRPC
stubs at build time from the shared root `proto/` directory).

## Examples

`examples/` ports the Python examples to C++:

- `person_detection.cpp` — stream inference + event publishing (the canonical demo)
- `perimeter_guard.cpp`, `video_processor.cpp`, `event_subscriber.cpp`
- `connectivity_smoke.cpp`, `media_smoke.cpp` — on-device smoke checks
- `api_tour.cpp` — one core call per client module (all 11), PASS/SKIP/FAIL summary

## Tests

GoogleTest suite mirroring the daemon-independent tests from
`python/tests/`: env-var configuration, inference value types, packed
media/audio wire structs, AudioFrame decoding. Run with `ctest`.

## API reference

Doxygen API reference (English) is published at
`/cpp/en/` on the documentation site, generated from the public headers in
`cpp/include/hailo_ipc_sdk/`. A Chinese quick-start & module overview lives at
`/cpp/zh/`.
