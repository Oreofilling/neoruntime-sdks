# NeoRuntime C++ SDK

C++ client SDK for the NeoRuntime edge-AI camera platform — a full mirror of
the Python SDK (`python/neoruntime_ipc_sdk` **0.8.1**), using OpenCV `cv::Mat`
where the Python SDK uses numpy. This tree is C++ SDK **0.8.0**
(`version.hpp`: `kVersion` / `kMirroredPythonSdk`).

Applications built with this SDK run on-device, inside an app container, and
talk to the platform daemons over local Unix Domain Sockets.

## Client modules

| Header | Client / type | Purpose |
|---|---|---|
| `inference.hpp` | `InferenceClient` | AI inference: `infer` / batch / streaming `subscribe` (with health: latency/skew EMA, drop counter, consecutive-failure limit), model registration & stats, GenAI streaming. `InferInput` takes `cv::Mat` / `Frame` / `FrameHandle` / `BufferRefInput` (daemon-resident buffer id) |
| `inference_types.hpp` | value types | `InferenceResult` (objects, raw outputs, skew), `ModelInfo` (model_type/variant/owner_id/transient/batch_size), `BatchInferItem` |
| `device.hpp` | `DeviceClient` | Lights / IR / IRCUT, PTZ / zoom / focus, GPIO, RS485, Wiegand, status, event subscription, autofocus suite (`AfJob`/`AfStatus`/`AfMeasurement`) |
| `camera.hpp` + `camera_types.hpp` | `CameraClient` | ISP / encoder / RTSP / OSD / profiles, `StreamStatus`, privacy masks, IR presets, pipeline reconfiguration, frame injection (`push_frame`, `InjectionResult` with in-flight lease buffers) |
| `app.hpp` | `AppClient` | App-container lifecycle, logs, stats |
| `events.hpp` | `EventClient` | Event-bus publish / subscribe / batch / topics / stats |
| `frame.hpp` | `Frame` / `FrameHandle` | Zero-copy DMA-BUF frames: per-plane `mmap` into `cv::Mat`, delayed RELEASE, `crop` / `resize` (letterbox), color converts, JPEG |
| `fd_client.hpp` (`media.hpp` facade) | `FdMediaClient` | Zero-copy frame pulls (`get_frame(keep_fd)`), background `subscribe`/`on_frame`, `list_streams`, encoded-stream handles |
| `encoded.hpp` (`media.hpp` facade) | `EncodedStreamClient` | H.264 / H.265 encoded streams (V3 sequence numbers, gap/restart counters) |
| `audio.hpp` | `AudioClient` | Audio control |
| `audio_stream.hpp` | `AudioStreamClient` | PCM / AAC audio capture streams |
| `overlay.hpp` | `OverlayClient` | Hardware overlay: config, `annotate` (detections + zone polygons, ttl / session / frame bind), strict frame lock |
| `dsp.hpp` | `DspClient` | DSP jobs: buffer pools, `resize_hw` / `crop_hw` / `multi_crop_hw` / `convert_hw` / `blend_hw` / `encode_jpeg_hw`, buffer import, hardware-vs-CPU visibility (`last_used_hw`) |
| `injection.hpp` | `FramePublisher` | Replace/overlay a camera stream with generated frames (NV12 pool or RGB staging convert chain, write-lease aware, session-tagged EOS) |
| `preprocess.hpp` / `postprocess.hpp` | `Preprocessor` / `Postprocessor` | Letterbox preprocess from model specs, NMS + YoloV5/V8 decode |
| `pipeline.hpp` | `InferencePipeline` | Client-side pre → infer → post (pixels in your process) |
| `stream_pipeline.hpp` | `StreamPipeline` | Platform-scheduled whole pipeline: subscribe → overlay → app result queue (pixels stay on the device) |
| `runner.hpp` | `PipelineRunner` | Run `InferencePipeline`s over a frame source |
| `color.hpp` / `draw.hpp` | helpers | Color converts (cvtColor parity) and overlay raster drawing |
| `diagnostics.hpp` | `collect_diagnostics` | Environment + connectivity report (`sdk_version` etc.) |
| `accel.hpp` | `AccelRouter` | Route expensive pixel ops to DSP hardware with CPU fallback |
| `plugin.hpp` | — | **Deprecated** (kept as marker only, mirrors Python 0.8) |

Shared value types live in `types.hpp`; endpoints & environment configuration
in `config.hpp`. Public headers are proto-free (all `*.pb.h` stay in `src/`).

**Out of scope** vs Python 0.8.1: web (MJPEG), recording (TS/HLS), plugins —
bring your own HTTP server / muxer for those.

## Python → C++ reading guide

Same architecture, Python-native idioms replaced by C++-native ones:

| Python | C++ | Note |
|---|---|---|
| numpy arrays | `cv::Mat` | layout-checked (NV12 = `(h*3/2, w)` 8UC1) |
| `results()` iterator | `next_result() → std::optional<...>` | SDK-wide pull convention (`next()` on every stream) |
| dict polygons | `OverlayPolygon` / `OverlayPoint` | typed, validated at the boundary |
| `queue.Queue` + sentinel | bounded drop-oldest `ResultQueue` | drops counted in `status()` |
| daemon threads | joinable `std::thread` | `stop()` joins; self-stop from a callback is supported |
| `grpc.aio` threads | sync stubs + `std::future` | sync stubs already park on a futex (measured 17% of a core at 20ms calls) |
| `RuntimeError` / `ValueError` | `std::runtime_error` / `std::invalid_argument` | message texts match verbatim where asserted |
| `DspError` / `HardwareUnavailable` | `DspError` (code −1..−7) / `std::runtime_error` | |

## Transports

- **gRPC over UDS** — inference, device, camera, app, events, overlay, DSP job
  face. 64 MiB max message size, matching Python `_transport`.
- **Raw UDS + `SCM_RIGHTS` fd-passing** — `FdMediaClient` (video planes) and
  the DSP buffer face (`DspClient` keeps one gRPC and one UDS connection).
- **Raw UDS, length-prefixed** — `EncodedStreamClient`, `AudioStreamClient`.

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

## Consuming the packaged tarball

Release tarballs (`ne503-aipc-cpp-sdk-<version>-linux-arm64.tar.gz`) carry
headers, the static library, CMake package files, and prebuilt examples.
Consumer recipe (verified end-to-end against the 0.8.0 tarball on an NE503):

```sh
# 1. Unpack on the build host and point find_package at it.
tar xf ne503-aipc-cpp-sdk-0.8.0-linux-arm64.tar.gz

# 2. Consumer CMakeLists:
#    find_package(ne503_aipc_sdk 0.8.0 REQUIRED CONFIG)
#    target_link_libraries(app PRIVATE ne503::ne503_aipc_sdk)

# 3. The exported target still links protobuf/gRPC/OpenCV/nlohmann_json,
#    so build with a vcpkg manifest declaring those (mirror cpp/vcpkg.json's
#    opencv4 feature set to reuse the binary cache).

# 4. Cross-compile: pass the package dir explicitly — CMAKE_FIND_ROOT_PATH
#    ONLY mode ignores CMAKE_PREFIX_PATH prefixes outside the sysroot.
cmake -S myapp -B build-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=<sdk>/cpp/cmake/aarch64-toolchain.cmake \
  -DVCPKG_TARGET_TRIPLET=arm64-linux \
  -Dne503_aipc_sdk_DIR=<unpacked>/lib/cmake/ne503_aipc_sdk

# 5. Link -static-libstdc++: the device Yocto libstdc++ predates
#    GLIBCXX_3.4.30. The tarball's own examples are linked this way.
```

## Versioning and releases

The C++ SDK is versioned independently of the Python SDK. Release tags use
the `cpp-v*` prefix (`cpp-v0.8.0`); the version tracks the mirrored Python
API surface in `major.minor` while the patch number evolves on its own —
`kMirroredPythonSdk` in `version.hpp` records the exact mirrored Python
version. Python `v*` tags (wheel.yml) and C++ `cpp-v*` tags (cpp.yml) never
trigger each other's release chains.

Pushing a `cpp-v*` tag runs the C++ pipeline: x86_64 build + tests, release
version validation (tag == `CMakeLists.txt` `VERSION` == `kVersion` ==
`vcpkg.json`), aarch64 cross-compile with the installable arm64 tarball, and
a GitHub Release attaching the tarball. Milestones may ship same-day
`v*`/`cpp-v*` pairs, but the lines never lock step numbers. See
[CHANGELOG.md](CHANGELOG.md).

## Examples

`examples/` ports the Python examples and adds SDK-tours:

- `person_detection.cpp` — stream inference + event publishing (the canonical demo)
- `perimeter_guard.cpp`, `video_processor.cpp`, `event_subscriber.cpp`
- `connectivity_smoke.cpp`, `media_smoke.cpp` — on-device smoke checks
- `api_tour.cpp` — one core call per client module, PASS/SKIP/FAIL summary
- `dsp_resize.cpp` — DSP pools, hardware resize, JPEG-by-buffer-id chain
- `frame_injection.cpp` — replace the `sub` stream with generated frames
- `stream_pipeline.cpp` — whole-pipeline zone + overlay + result loop
- `phase3_smoke.cpp` … `phase6_smoke.cpp` — per-phase on-device verification

## Tests

GoogleTest suite mirroring the daemon-independent tests from `python/tests/`
— 375 cases: configuration, wire structs, codecs, frame geometry (parity
truth tables generated by the Python implementation), fake-daemon protocol
tests (fd media / encoded / DSP / injection / event bus / inference
subscribe), and the tool layer (preprocess / postprocess / Yolo decode /
pipeline / runner / stream pipeline / color / draw / diagnostics / accel).
Run with `ctest`.

## API reference

Doxygen API reference (English) is published at
`/cpp/en/` on the documentation site, generated from the public headers in
`cpp/include/neoruntime_ipc_sdk/`. A Chinese quick-start & module overview lives at
`/cpp/zh/`.
