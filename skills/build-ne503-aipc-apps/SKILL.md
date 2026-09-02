---
name: build-ne503-aipc-apps
description: Help developers use the NE503/NeoRuntime AIPC SDK to build applications. Use when Codex needs to create, adapt, debug, explain, or package app code that consumes the SDK APIs for inference, media streams, encoded video, audio, events, device control, camera control, overlays, plugins, app management, Docker app images, or on-device validation in Python or C++.
---

# Build NE503 AIPC Apps

## Start

1. Identify the target language: Python by default; use C++ when the user asks for C++, low-latency native code, existing CMake integration, or OpenCV `cv::Mat`.
2. Identify the app workflow: inference, video/media, encoded stream, audio, events, device control, camera control, overlay, plugin, or app lifecycle.
3. Ask only for missing deployment facts that affect code shape: stream id, model id, target device paths, app id, Docker/base image, and whether hardware services are available.
4. Read `references/sdk-user-guide.md` before generating a substantial app, Dockerfile, or troubleshooting plan.

## API Selection

- Use `InferenceClient` for one-shot image inference, streaming inference, model registration, model listing, stats, and sessions.
- Use `EventClient` for JSON event publish/subscribe. Prefer topics under `app/<APP_ID>/...` for app-originated events.
- Use `DeviceClient` for lights, IR, IRCUT, PTZ, zoom/focus, GPIO, RS485, Wiegand, status, and device event subscription.
- Use `CameraClient` for ISP, encoder, RTSP, OSD, profiles, camera status, and pipeline configuration.
- Use `FdMediaClient` for raw DMA-BUF video frames. Current Python exports do not include a `MediaClient` class; use `FdMediaClient`.
- Use `EncodedStreamClient` for H.264/H.265 encoded frames from sockets under `ENCODED_SOCKET_DIR`.
- Use `AudioClient` for audio control and `AudioStreamClient` for PCM/AAC audio capture streams.
- Use `OverlayClient` for video overlay configuration.
- Use `PluginDiscovery` and `PluginServer` for plugin endpoint discovery or service registration.
- Use `AppClient` for app-container lifecycle, logs, and stats.

## App Code Rules

- Use `Config` getters for app id, debug mode, endpoints, encoded socket dir, shared-memory path, and path translation.
- Let SDK clients auto-connect unless explicit lifecycle control is useful; always call `close()` in cleanup paths for long-running apps.
- Add `SIGINT` and `SIGTERM` handling for container apps.
- Keep inference result payloads JSON-serializable; convert bbox/object fields to plain dict/list values.
- Make stream id and model id configurable through environment variables when producing reusable examples.
- Avoid writing examples that require daemon services as unit tests. For hardware-dependent behavior, provide an on-device smoke command or manual verification steps.

## Python Pattern

Use this shape for long-running apps:

```python
import os
import signal
from neoruntime_ipc_sdk import Config, EventClient, InferenceClient

class App:
    def __init__(self):
        self.running = True
        self.app_id = Config.get_app_id()
        self.stream = os.getenv("AIPC_STREAM", "cam0_main")
        self.model = os.getenv("AIPC_MODEL", "person_v1")
        self.inference = InferenceClient()
        self.events = EventClient()
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *_):
        self.running = False

    def run(self):
        try:
            for frame_seq, result in self.inference.subscribe(self.stream, self.model, fps=10):
                if not self.running:
                    break
                count = result.count_by_label("person")
                if count:
                    self.events.publish(f"app/{self.app_id}/person_detected", {
                        "frame_sequence": frame_seq,
                        "timestamp_ns": result.timestamp_ns,
                        "person_count": count,
                    })
        finally:
            self.inference.close()
            self.events.close()

if __name__ == "__main__":
    App().run()
```

## C++ Pattern

- Include public headers from `neoruntime_ipc_sdk/*.hpp`.
- Use `neoruntime_ipc_sdk::Config` for endpoints and app id.
- For streaming inference, consume the subscription with `next()` and check for empty optional values.
- Use `nlohmann::json` for event payloads.
- Build examples with the existing `cpp/examples/CMakeLists.txt` pattern and link `ne503::aipc_sdk`.

## Packaging

- For Python app images, base on the SDK image when available, copy app files into `/app`, run as the non-root app user, and set `APP_ID` plus workflow-specific environment variables.
- When adding Python dependencies, install them as root and switch back to `appuser`.
- When app code references host model or data paths, use `Config.translate_path_to_host()` for paths that platform services resolve on the host.

## Validation

- For pure Python logic, run local Python tests or a small dry-run that does not require UDS daemons.
- For SDK service calls, validate on device or in a container with `/run/aipc` sockets mounted.
- For encoded/raw media, verify stream socket names and permissions before blaming decoding logic.
- For device control, test with conservative values first, and include cleanup/stop calls for PTZ or continuous movement.
