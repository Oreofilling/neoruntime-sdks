# SDK Modular Framework

Status: current `main` working tree, Python SDK `0.8.1`, C++ SDK `1.0.0`
(rebuilt in-tree; architecture and public interfaces mirror the Python SDK).
C++ entries below describe the `cpp/` tree in `main`.

This document organizes the current NeoRuntime SDK repository into module
boundaries, runtime dependencies, and extension rules. It is meant for SDK
users, release owners, and maintainers who need to understand where a feature
belongs before changing APIs.

## 1. Repository Shape

```text
neoruntime-sdks/
  proto/                 Platform protobuf contracts mirrored from platform
  python/                Python wheel source, examples, tests, Sphinx docs
  cpp/                   C++ static SDK source, examples, tests, CMake package
  docs/                  Cross-language architecture/proposals
  scripts/               Proto sync, packaging, docs/image helpers
  .github/workflows/     CI, proto sync, docs publishing, release artifacts
```

The repository has three product surfaces:

| Surface | Release form | Primary users | Integration style |
| --- | --- | --- | --- |
| Python SDK | `neoruntime-ipc-sdk` wheel + sdist | App authors, fast prototyping, container apps | `pip install`, import `neoruntime_ipc_sdk` |
| C++ SDK | `ne503-aipc-cpp-sdk-<version>-linux-arm64.tar.gz` | Native on-device apps | CMake `find_package(ne503_aipc_sdk)` |
| Proto contracts | Source `.proto` files plus generated stubs | Platform and SDK maintainers | Synced from platform, regenerated into SDK clients |

## 2. Runtime Model

SDK code runs inside an app process or app container. It does not own hardware
directly. It talks to platform daemons through local IPC endpoints.

```text
Application
  |
  | imports/links SDK modules
  v
Python wheel or C++ static library
  |
  | gRPC over UDS
  | raw UDS streams
  | raw UDS + SCM_RIGHTS fd passing
  v
Platform daemons
  ai-runtime
  camera-daemon
  device-control
  event-bus
  app-manager
```

Default endpoints live in `Config`:

| Capability | Default endpoint |
| --- | --- |
| AI runtime | `unix:///run/aipc/ai-runtime.sock` |
| Camera control | `unix:///run/aipc/camera-control.sock` |
| Device control | `unix:///run/aipc/device-control.sock` |
| Event bus | `unix:///run/aipc/event-bus.sock` |
| App manager | `unix:///run/aipc/app-manager.sock` |
| Raw frame publisher | `/run/aipc/camera.sock` |
| Encoded stream sockets | `/run/aipc/encoded/<stream>.sock` |

## 3. Layering

The SDK is easiest to reason about in five layers.

### 3.1 Protocol Layer

`proto/` is the SDK-side source of truth for public wire contracts. It mirrors
platform proto files:

| Directory | Service family |
| --- | --- |
| `proto/ai-runtime/` | Inference, batch inference, stream inference, GenAI |
| `proto/camera-daemon/` | Camera control, DSP jobs, media-related control |
| `proto/device-control/` | Lights, PTZ/lens, GPIO, RS485, events |
| `proto/event-bus/` | Publish/subscribe, topics, stats |
| `proto/app-manager/` | App lifecycle, logs, web URL registration |
| `proto/device-discovery/` | Device discovery contracts |

Python commits generated `*_pb2.py` and `*_pb2_grpc.py` under
`python/neoruntime_ipc_sdk/proto/`. C++ regenerates protobuf/gRPC sources at
CMake build time into the build directory.

### 3.2 Transport Layer

Transport code is shared plumbing, not the user-facing API.

| Language | Files | Responsibility |
| --- | --- | --- |
| Python | `python/neoruntime_ipc_sdk/_transport.py` | `GrpcClient`, `UdsStreamClient`, status checks, fd passing helpers |
| Python | `python/neoruntime_ipc_sdk/config.py` | Endpoint environment variables, host path translation |
| C++ | `cpp/src/detail/grpc_channel.hpp` | gRPC channel creation, RPC error handling |
| C++ | `cpp/src/detail/endpoint.hpp` | `unix://` endpoint normalization |
| C++ | `cpp/src/detail/raw_socket.hpp` | Raw AF_UNIX socket helpers |
| C++ | `cpp/src/detail/fd_protocol.hpp` | Frame/audio fd-passing wire structs |
| C++ | `cpp/src/detail/tensor_io.hpp` | `cv::Mat` to/from inference tensor protobufs |

Most Python clients inherit `GrpcClient`. `InferenceClient` intentionally uses
`grpc.aio` on a background event loop to avoid sync completion-queue CPU spin in
tight inference loops.

### 3.3 Daemon Client Layer

These modules are thin, typed wrappers over platform daemon APIs.

| Capability | Python module | C++ header | Daemon | Notes |
| --- | --- | --- | --- | --- |
| AI inference | `inference.py`, `inference_types.py`, `inference_codec.py`, `inference_genai.py` | `inference.hpp`, `inference_types.hpp` | `ai-runtime` | Model registration, `infer`, `infer_async`, `infer_batch`, stream inference with health (latency/skew EMA, drop counters, consecutive-failure limit), stats, GenAI |
| Camera control | `camera.py`, `camera_types.py` | `camera.hpp`, `camera_types.hpp` | `camera-daemon` | ISP, transform, encoder, RTSP, OSD, profiles, hardware status, privacy masks, IR presets, pipeline reconfigure, frame injection |
| Device control | `device.py` | `device.hpp` | `device-control` | Lights, IR/IRCUT, PTZ, zoom/focus, GPIO, RS485, Wiegand, device events, autofocus suite |
| Event bus | `events.py` | `events.hpp` | `event-bus` | Publish, subscribe, topic metadata, stats |
| App management | `app.py` | `app.hpp` | `app-manager` | App install/start/stop/uninstall, stats, logs, web URL registration |
| Audio control | `audio.py` | `audio.hpp` | audio service/control plane | Capture/playback device and status control |
| Audio stream | `audio_stream.py` | `audio_stream.hpp` | raw UDS publisher | PCM/AAC frame stream |
| Overlay | `overlay.py` | `overlay.hpp` | camera/control overlay path | Overlay settings, detection result projection, zone polygons, session/frame binding |
| Plugins | `plugin.py` | `plugin.hpp` | app/plugin discovery | Deprecated in both SDKs (marker only) |

Both surfaces mirror the daemon-client core; the C++ tree additionally ships
the DSP domain, frame injection, and the inference tool layer (see 3.5).

### 3.4 Media And Data Layer

Media modules model frames, packets, pixel formats, and buffer lifetime.

| Module | Role |
| --- | --- |
| `frame.py` / `frame.hpp` | `Frame`, `FrameHandle`, `PixelFormat`, raw pixel materialization, keep-fd lifetime |
| `media.py` / `media.hpp` | Backward-compatible exports for frame and stream clients (C++ facade over `frame.hpp`/`fd_client.hpp`/`encoded.hpp`) |
| `fd_client.py` / `fd_client.hpp` | Raw frame client using UDS + `SCM_RIGHTS` dma-buf fd passing |
| `encoded.py` / `encoded.hpp` | Encoded H.264/H.265 packet stream client |
| `audio_stream.py` / `audio_stream.hpp` | Audio frame wire decoder and stream client |

The important ownership rule is: if an app requests `keep_fd=True`, it must
close or release the retained `FrameHandle`. That handle is also the zero-copy
bridge into DSP hardware jobs.

### 3.5 App Toolkit And Acceleration Layer

These modules are higher-level building blocks for app authors. They are not
all one-to-one daemon wrappers.

| Module | Python | C++ | Hardware path |
| --- | --- | --- | --- |
| `color` | `color.py` | `color.hpp` | CPU by default (cvtColor) |
| `draw` | `draw.py` | `draw.hpp` | CPU/PIL vs CPU/OpenCV |
| `postprocess` | `postprocess.py` | `postprocess.hpp` (NMS + YoloV5/V8 decode) | CPU |
| `preprocess` | `preprocess.py` | `preprocess.hpp` (letterbox from model specs) | CPU |
| `pipeline` / `stream_pipeline` / `runner` | `pipeline.py`, `stream_pipeline.py`, `runner.py` | `pipeline.hpp`, `stream_pipeline.hpp`, `runner.hpp` | client-side vs platform-scheduled |
| `recording` | `recording.py` | not mirrored (app-supplied muxer) | CPU/file output |
| `web` | `web.py` | not mirrored (app-supplied MJPEG server) | CPU/network helper |
| `dsp` (+wire/format) | `dsp.py`, `dsp_wire.py`, `dsp_format.py` | `dsp.hpp`, `src/detail/dsp_wire.hpp`, `src/detail/dsp_format.hpp` | camera-daemon DSP service |
| `injection` | `injection.py` | `injection.hpp` (`FramePublisher`) | camera-daemon frame injection |
| `accel` | `accel.py` | `accel.hpp` (`AccelRouter`) | Delegates to DSP or software legs |
| `diagnostics` | — | `diagnostics.hpp` | n/a |

The SDK ships loose components, not a fixed pipeline. Apps compose frame input,
inference, post-processing, overlay, events, recording, and web serving as
needed.

## 4. Current Public Module Map

```text
neoruntime_ipc_sdk
  InferenceClient        AI inference, async inference, batch, stream, GenAI
  FdMediaClient          raw frame access
  EncodedStreamClient    H.264/H.265 stream access
  Frame, FrameHandle     image buffer and retained dma-buf lifetime
  EventClient            event publish/subscribe
  DeviceClient           hardware controls
  CameraClient           camera pipeline controls
  AppClient              app lifecycle and logs
  OverlayClient          overlay configuration
  AudioClient            audio control
  AudioStreamClient      audio stream access
  DspClient              hardware DSP offload with explicit fallback behavior
  AccelRouter            hardware/software route policy and degradation health
  draw/color/postprocess app-side image utilities
  recording/web          app-side streaming and recording utilities
  PluginDiscovery        local plugin endpoints (deprecated)
```

The C++ tree mirrors the same shape under `neoruntime_ipc_sdk::` — one header
per module in `cpp/include/neoruntime_ipc_sdk/` (see `cpp/README.md` for the
full matrix), minus `recording`/`web` by design.

## 5. Inference Module Notes

`InferenceClient` is the most performance-sensitive client.

| API | Behavior |
| --- | --- |
| `infer` | Blocking single inference, raises if daemon status is failure |
| `infer_async` | Returns `concurrent.futures.Future[InferenceResult]`; useful for depth-N app pipelines |
| `infer_batch` | Sends one `InferBatch` RPC with multiple `InferRequest` items |
| `infer_batch_async` | Async future form of batch RPC |
| `infer_with_tensors` | Direct tensor input/output path |
| `subscribe` | Stream inference result subscription |
| GenAI helpers | LLM/VLM session and streaming helpers via `GenAiMixin` |

`infer_batch` is a real RPC. On current device firmware it is implemented by
`ai-runtime`, which submits items through `model_mgr_->run_async()` and relies on
HAL/HailoRT shared VDevice `ROUND_ROBIN` scheduling. It does not pass through
ai-runtime's `InferenceScheduler::submit()` queue, so scheduler queue depth and
shutdown drain accounting do not cover in-flight batch items.

## 6. Python And C++ Parity

Use this rule of thumb:

| Area | Python | C++ |
| --- | --- | --- |
| Daemon clients | Broad coverage | Broad coverage |
| Raw frame and encoded streams | Yes | Yes |
| Audio stream | Yes | Yes |
| DSP offload, injection, accel router | Yes | Yes (`dsp.hpp`, `injection.hpp`, `accel.hpp`) |
| Inference tool layer (preprocess/postprocess/pipelines/runner) | Yes | Yes |
| draw/color | Yes | Yes (OpenCV legs) |
| recording (TS/HLS), web (MJPEG) | Yes | Not mirrored by design (app-supplied) |
| Plugins | Deprecated | Deprecated marker only |
| Proto generation | Generated files committed | Generated during CMake build |
| Package format | Wheel + sdist | Installable tarball with headers/library/CMake files |

When documenting "SDK support", distinguish "Python SDK support", "C++ SDK
support", and "device firmware support". Some APIs need all three to be useful.

## 7. Extension Rules

When adding a new platform capability:

1. Add or sync the platform `.proto` into `proto/`.
2. Regenerate Python protobuf stubs with `scripts/sync_platform_protos.sh`.
3. Add Python value types and codecs before exposing a high-level client method.
4. Add the C++ public type/header method when the feature should be available to native apps.
5. Keep endpoint defaults in `Config` instead of scattering socket paths.
6. Add a small example or smoke path if the feature touches a daemon.
7. Add tests at the lowest layer that can verify behavior without hardware, then add device validation notes when hardware semantics matter.
8. Update release notes and docs with the exact dependency on platform firmware.

For hardware acceleration APIs, make fallback behavior explicit:

| API style | Failure behavior |
| --- | --- |
| Convenience APIs, such as `Frame.resize` | Fall back to CPU when hardware is unavailable |
| Explicit hardware APIs, such as DSP quota failures | Raise visible errors when CPU fallback would hide a latency cliff |
| `AccelRouter` | Records hardware-to-software degradation in `health()` |

## 8. Release And CI Framework

| Workflow/script | Purpose |
| --- | --- |
| `.github/workflows/wheel.yml` | Build/test Python, publish Python artifacts on `v*` tags |
| `.github/workflows/cpp.yml` | C++ build/test validation and `cpp-v*` tag releases (arm64 tarball + GitHub Release) |
| `.github/workflows/pages.yml` | Publish docs |
| `.github/workflows/sync-platform-protos.yml` | Sync platform proto contracts and open PR |
| `scripts/package_cpp_sdk.sh` | Build installed C++ layout and package tarball |
| `scripts/sync_platform_protos.sh` | Copy platform protos and regenerate Python stubs |
| `scripts/check_interface_drift.sh` | Validate SDK proto/generated stubs match platform |

Current release contract (Python `v*` and C++ `cpp-v*` tags are independent release lines):

| Language | Artifact |
| --- | --- |
| Python | `neoruntime_ipc_sdk-<version>-py3-none-any.whl` and source tarball |
| C++ | `ne503-aipc-cpp-sdk-<cpp-version>-linux-arm64.tar.gz` on `cpp-v*` tags, versioned independently of Python |

## 9. Recommended User-Facing Mental Model

For SDK users, present the SDK as a modular toolbox:

1. Capture frames with `FdMediaClient` or consume encoded packets with `EncodedStreamClient`.
2. Transform or crop frames with `Frame`, `color`, `draw`, or `DspClient` when hardware offload matters.
3. Run AI with `InferenceClient`, choosing `infer_async` for pipeline concurrency and `infer_batch` only when the target firmware and measured workload benefit from batch submission.
4. Publish results with `EventClient`.
5. Render or expose output through `OverlayClient`, `recording`, or `web`.
6. Control device behavior with `DeviceClient` and camera pipeline behavior with `CameraClient`.
7. Package the app in a container, installing the Python wheel or linking the C++ tarball depending on language.
