# SDK User Guide

## Common Environment

```sh
export APP_ID=my_app
export AI_RUNTIME_ENDPOINT=unix:///run/aipc/ai-runtime.sock
export EVENT_BUS_ENDPOINT=unix:///run/aipc/event-bus.sock
export DEVICE_CONTROL_ENDPOINT=unix:///run/aipc/device-control.sock
export CAMERA_CONTROL_ENDPOINT=unix:///run/aipc/camera-control.sock
export APP_MANAGER_ENDPOINT=unix:///run/aipc/app-manager.sock
export SHM_BASE_PATH=/run/aipc/shm
export ENCODED_SOCKET_DIR=/run/aipc/encoded
export AIPC_HOST_PREFIX=/data/aipc
export DEBUG=0
export LOG_LEVEL=INFO
```

Use additional app-level variables such as `AIPC_STREAM=cam0_main` and `AIPC_MODEL=person_v1` for reusable sample apps.

## Python Imports

```python
from hailo_ipc_sdk import (
    AppClient,
    AudioClient,
    AudioStreamClient,
    CameraClient,
    Config,
    DeviceClient,
    EncodedStreamClient,
    EventClient,
    FdMediaClient,
    InferenceClient,
    OverlayClient,
    PluginDiscovery,
    PluginServer,
)
```

Current Python exports include `FdMediaClient` and `EncodedStreamClient`, not `MediaClient`.

## Client Cheat Sheet

| Need | Python client | Notes |
|---|---|---|
| Stream inference results | `InferenceClient.subscribe(stream, model, fps=10)` | Yields `(frame_seq, InferenceResult)` |
| One-shot image inference | `InferenceClient.infer(image, model_id=...)` | Image is a numpy array |
| Publish JSON event | `EventClient.publish(topic, payload)` | Payload can include numpy scalars/arrays |
| Subscribe events | `EventClient.subscribe(topic)` | Supports wildcard topics |
| Raw frame access | `FdMediaClient.get_frame(stream_id)` / `subscribe_raw(stream_id)` | Needs fd-passing socket access |
| Encoded video | `EncodedStreamClient(socket_path).subscribe()` | Reads H.264/H.265 frames |
| Device hardware | `DeviceClient` | Lights, IRCUT, PTZ, GPIO, lens |
| Camera settings | `CameraClient` | ISP, encoder, RTSP, OSD, profiles |
| Audio capture | `AudioStreamClient(socket_path).subscribe()` | PCM/AAC frames |
| Plugin discovery | `PluginDiscovery().get(...)` / `require(...)` | Connect to plugin endpoints |

## App Skeleton Decisions

- Prefer a class with `running`, clients, signal handlers, `run()`, and `cleanup()`.
- Keep hardware side effects inside small methods such as `_on_person_detected`.
- Set default stream/model ids but make them environment configurable.
- Log startup endpoints from `Config` when troubleshooting.
- Publish compact event payloads with timestamps, counts, labels, scores, and bbox lists.

## Docker App Shape

```dockerfile
FROM registry.local/aipc-sdk:0.4.0

USER root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER appuser
COPY --chown=appuser:appuser app.py app.yaml ./

ENV APP_ID=my_app \
    LOG_LEVEL=INFO \
    AIPC_STREAM=cam0_main \
    AIPC_MODEL=person_v1

CMD ["python3", "app.py"]
```

If no extra dependencies are needed, omit the `USER root`, `requirements.txt`, and `pip install` steps.

## On-Device Smoke Checklist

1. Confirm sockets exist: `ls -la /run/aipc`.
2. Confirm encoded sockets if needed: `ls -la /run/aipc/encoded`.
3. Confirm `APP_ID`, stream id, and model id are set as expected.
4. Run with `DEBUG=1 LOG_LEVEL=DEBUG` for first validation.
5. Check that long-running loops respond to `SIGTERM`.
6. For PTZ or actuator code, verify stop/cleanup behavior before increasing speeds.
