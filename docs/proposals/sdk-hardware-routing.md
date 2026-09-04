# Proposal: SDK Hardware Routing (Loose Components + Capability Router)

Status: SDK skeleton **landed 2026-09-04** (`accel.py` router,
`OverlayClient.annotate`, `color.py` / `postprocess.py` software legs,
338 tests green); this document is the component inventory, the routing
design, and the service-layer asks that turn the remaining software legs
hardware-first.

Relationship to siblings: this is the SDK-side companion to
[dsp-offload.md](dsp-offload.md) (the RPC foundation) and
[hardware-first-roadmap.md](hardware-first-roadmap.md) (the platform/HAL
work breakdown). Nothing here asks hal_v2 for a new op.

## Motivation

The SDK's value proposition is loose components ("散件") an app
assembles into its own pipeline, not a fixed pipeline object. Every
per-frame component the SDK ships should therefore answer one question
uniformly: *is there hardware on this device that already does this?*
Today that answer is scattered — `Frame.resize` knows about DSP,
`draw.py` knows about nothing — and apps hard-code whichever path they
discovered first, so the same app is slow on one device and idle-hardware
on another.

The router makes the preference declarative and per-device:
hardware-first with automatic degradation to numpy/cv2, one `health()`
call to see what actually ran, and one place to attach new hardware legs
as the platform exposes them.

## What landed in the SDK (this branch)

| Piece | Module | Notes |
|---|---|---|
| `AccelRouter` / `RoutePolicy` / `get_default_router()` | `neoruntime_ipc_sdk/accel.py` | register/route/run + degradation counters + `probe()` + `health()`; `HARDWARE_ONLY` policy for strict callers |
| `OverlayClient.annotate(stream_id, objects)` / `annotate_result()` | `overlay.py` | pushes detections over the event bus into camera-daemon's overlay renderer — zero frame copies in the app |
| `rgb_to_nv12` / `nv12_to_rgb` (+ BT.601 limited-range, cv2-free numpy path) | `color.py` | software leg for the color-convert op |
| `nms(boxes, scores, ...)` | `postprocess.py` | software leg for suppression, cross-class aware |
| Registered ops | `get_default_router()` | `resize_nv12` (DSP leg live via `DspClient.resize_hw`), `rgb_to_nv12`, `nms` (software-only today) |

Two production bugs the new tests surfaced and fixed along the way:
`_packed_to_nv12` was missing the `/255` on the BT.601 dot products
(total luma saturation — every input encoded to Y=255), and
`annotate()` crashed unpacking `BoundingBox` (not iterable; now
duck-typed on `.x/.y/.width/.height`).

## Loose-component inventory

"Missing blocks" an app needs between `subscribe` and `publish`, and
where each one runs today:

| Component | SDK entry | Software leg | Hardware leg | Blocking layer |
|---|---|---|---|---|
| NV12 geometry (resize/crop/multi-crop) | `router.run("resize_nv12", ...)` | numpy | ✅ DSP `resize_hw` | — (live) |
| RGB↔NV12 color convert | `rgb_to_nv12` / `nv12_to_rgb` | numpy | ⏸ DSP `CONVERT_FORMAT` | **SDK client only** — see S-1 |
| Box suppression | `nms` | numpy | ⏸ ai-runtime postprocess | ai-runtime params — see S-2 |
| Draw onto outgoing stream | `OverlayClient.annotate` | — | ✅ camera-daemon renderer | — (live; contract below) |
| Raster drawing (local frames) | `draw.py` | CPU raster | ⏸ DSP blend | dsp-offload P1 |
| Snapshot JPEG | `frame.py:_encode_jpeg` | cv2 | ⏸ SoC encoder | platform RPC — see S-3 |
| App frames → main stream | — | — | ⏸ convert + injection | [frame-injection.md](frame-injection.md) |

## Routing design

```python
router = get_default_router()          # policy=prefer_hardware
small = router.run("resize_nv12", nv12, (1920, 1080), (640, 384))
router.health()["ops"]["resize_nv12"]  # backend actually used + counters
```

- `register(op, software=..., hardware=..., note=...)` — both legs
  optional; an op with neither route reports `unavailable`.
- `RoutePolicy` — `PREFER_HARDWARE` (default: try hw, fall back on
  `HardwareUnavailable`, record a degradation), `SOFTWARE_ONLY` (CI,
  reproducibility), `HARDWARE_ONLY` (fail loudly — debugging a
  regression you suspect is the fallback).
- Degradations raise `router.on_degradation` (exceptions swallowed) and
  accumulate in `health()`; apps typically forward them to the event bus
  as an app-health topic.
- `probe()` answers "what exists on this device" (`cv2` import, DSP
  socket reachable) without running an op.
- Routing is a pure decision when it can be: `route(op)` never touches a
  socket, so UIs can render a backend table at startup.

The one deliberate non-goal: the router does not pipeline. Apps compose
`run()` calls themselves; sequencing stays explicit and debuggable.

## Service-layer asks

### S-1 · `DspClient.convert_hw` — SDK work, unblocked now

The wire protocol and daemon already serve format conversion:

- `platform/camera-daemon/proto/camera.proto:775-828` — `DspOp` enum
  includes `DSP_OP_CONVERT_FORMAT = 3`; `DspJobRequest` carries
  `src_buffer_id` / `dst_buffer_ids` / formats.
- `platform/camera-daemon/src/dsp_service.cpp:896-899` — dispatch to
  `dsp_ops_->convert_format`.
- `hal_v2/platforms/hailo15/dsp/hailo15_dsp_impl.cpp:170` — HAL
  implementation (`hailo15_dsp_convert_format_sync`).

The gap is only the Python client: `dsp.py` exposes `resize_hw`
(:598), `crop_hw` (:658), `multi_crop_hw` (:723) and its buffer
allocator already accepts both `"nv12"` and `"rgb24"` (:174-238).
Adding `convert_hw(src_buffer, dst_buffer)` and registering it as the
`rgb_to_nv12` hardware leg is a contained SDK change. Constraint to
carry into the docstring: P0 `CONVERT` requires **equal src/dst
dimensions and differing formats** (`dsp_service.cpp:604-638`) —
resizes compose as `CONVERT → RESIZE`, which is also the cheaper order
(NV12 is half the RGB bytes on the wire).

### S-2 · ai-runtime: expose NMS registration params

The accelerator's postprocess already runs NMS at hardware speed, but
its knobs are compile-time constants per model type:

- `platform/ai-runtime/src/model_manager.cpp:253` and `:309` —
  `nms_threshold = 0.45f`, `confidence_threshold = 0.25f`,
  `max_detections = 64` hard-coded in the detection postprocess config.

Ask: accept these three as model-registration parameters (defaulting to
today's values) so the SDK can register `nms`'s hardware leg as
"the already-configured accelerator postprocess" and stop paying the
CPU round on every frame. Until then the router's `nms` stays
software-only with the current note.

### S-3 · codec: hardware JPEG encode RPC

`frame.py:92` (`_encode_jpeg`) burns CPU cv2 on every snapshot; the
platform's own thumbnails already use the SoC encoder. A
`EncodeImage(buffer_id, format, quality)` camera-daemon RPC (or an
`DSP_OP`-adjacent codec op) closes it. Low urgency: snapshots are
typically 1/few-Hz, unlike the per-frame ops above.

### S-4 · overlay ingestion contract — no platform work, recorded here

`OverlayClient.annotate` rides machinery that already exists; the
contract the SDK now depends on, for the record:

- topic `inference/<stream_id>` — prefix default
  `camera_daemon.h:176`, wildcard subscribe `ai_overlay_subscriber.cpp:379`;
- `event.metadata["stream_id"]` mandatory, events without it are
  skipped (`ai_overlay_subscriber.cpp:412-416`);
- results expire after 500 ms (`RESULT_TTL`, `:90`) → publish at
  inference cadence, empty list clears;
- compact JSON required — the parser string-scans `"bbox":[`
  (`:585`, `:638`); the SDK passes `compact=True` (event-bus
  `compact` kwarg);
- payload kinds: `{"num_detections", "detections":[{bbox, class_id,
  confidence, label}]}` (clamped to `HAL_MAX_DETECTIONS`, `:621`),
  `{"classifications"}`, `{"landmarks"}`, `{"ocr_lines"}` (`:561`).

## Phased rollout

- **P0 (done)** — router skeleton, overlay direct-push, color/NMS
  software legs, tests, zh+en API docs.
- **P1 (SDK-only, no platform dependency)** — S-1 `convert_hw` +
  `rgb_to_nv12` hardware leg; revisit `resize_nv12` to accept a
  dma-buf fast path when `dsp-offload` P1 lands.
- **P2 (blocked on service layers)** — S-2 NMS params, S-3 JPEG RPC;
  `draw.py` CPU raster → DSP blend follows dsp-offload P1; frame
  injection per its own proposal.
