# Changelog

## 0.8.0 (2026-10-15)

First public release of the NeoRuntime C++ SDK — a native C++17 mirror of the
Python SDK (`neoruntime_ipc_sdk` 0.8.1), using OpenCV `cv::Mat` where Python
uses numpy. The C++ version line is independent of the Python SDK: releases
are tagged `cpp-v*` and the version tracks the mirrored Python API surface in
`major.minor` while the patch number evolves on its own (`kMirroredPythonSdk`
in [include/neoruntime_ipc_sdk/version.hpp](include/neoruntime_ipc_sdk/version.hpp)
records the exact mirrored version).

### Client surface

- Inference: `infer` / batch / streaming `subscribe` with health tracking,
  model registration and stats, GenAI streaming
- Device: lights / IR / IRCUT, PTZ / zoom / focus, GPIO, RS485, Wiegand,
  status, event subscription, autofocus suite
- Camera: ISP / encoder / RTSP / OSD / profiles, privacy masks, pipeline
  reconfiguration, frame injection
- App lifecycle and logs, event-bus publish / subscribe / batch
- Zero-copy media: `FdMediaClient` (`SCM_RIGHTS` fd passing, `keep_fd`),
  `EncodedStreamClient` (H.264/H.265), `AudioStreamClient` (PCM/AAC)
- Hardware overlay (`annotate` with detections and zone polygons)
- DSP jobs: buffer pools, `resize_hw` / `crop_hw` / `multi_crop_hw` /
  `convert_hw` / `blend_hw` / `encode_jpeg_hw`, buffer import
- `FramePublisher` stream replacement, `Preprocessor` / `Postprocessor`
  (letterbox, NMS, YoloV5/V8 decode), `InferencePipeline` (client-side) and
  `StreamPipeline` (platform-scheduled), `PipelineRunner`, `AccelRouter`
  hardware-op routing with CPU fallback, `collect_diagnostics`

### Transports

gRPC over UDS, raw UDS + `SCM_RIGHTS` fd passing, and length-prefixed raw
UDS — message-size and protocol behavior match the Python SDK.

### Tests

375 daemon-independent GoogleTest cases (`ctest`, no device required):
configuration, wire structs, codecs, frame geometry parity tables,
fake-daemon protocol tests (fd media / encoded / DSP / injection / event bus /
inference subscribe), and the tool layer.

### Packaging and CI

- Installable aarch64 (NE503) tarball — headers, static library, CMake
  package config, prebuilt examples — via
  [scripts/package_cpp_sdk.sh](../scripts/package_cpp_sdk.sh); consumer
  recipe in the [README](README.md) (verified end-to-end on an NE503).
- `cpp-v*` tags run the full pipeline in
  [`.github/workflows/cpp.yml`](../.github/workflows/cpp.yml): x86_64 build +
  tests, release version validation (tag == `CMakeLists.txt` == `kVersion` ==
  `vcpkg.json`), aarch64 cross-compile and packaging, GitHub Release with the
  arm64 tarball.

### Dependency contract

Built against the repository-root `proto/` definitions and the platform
firmware contract current at this tag. Each release notes the exact
dependency; building from source requires `proto/` plus `protoc` and
`grpc_cpp_plugin` on the host.
