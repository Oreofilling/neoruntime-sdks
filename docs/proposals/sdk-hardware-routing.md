# Proposal: SDK Hardware Routing (Loose Components + Capability Router)

Status: SDK skeleton **landed 2026-09-04** (`accel.py` router,
`OverlayClient.annotate`, `color.py` / `postprocess.py` software legs,
338 tests green); **P1 `DspClient.convert_hw` landed same day** (both
convert directions registered as DSP hardware legs; verified on-device
on 93.72 — RGB↔NV12 run on the DSP, the gray8 pairs are firmware-
refused and degrade to CPU; 362 tests green);
this document is the component inventory, the routing design, and the
service-layer asks that turn the remaining software legs hardware-first.

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
| Registered ops | `get_default_router()` | `resize_nv12`, `rgb_to_nv12`, `nv12_to_rgb` (all with live DSP legs via `DspClient`), `nms` (software-only today) |
| `DspClient.convert_hw(src, dst_fmt, ...)` | `dsp.py` | DSP `CONVERT_FORMAT` leg: equal dims, differing formats, no rects, one dst; `cpu_fallback=False` switch so the router's degradation accounting stays honest (a daemon without the DSP surface raises instead of silently computing on CPU) |

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
| RGB↔NV12 color convert | `router.run("rgb_to_nv12"/"nv12_to_rgb", ...)` | numpy | ✅ DSP `CONVERT_FORMAT` | — (live; S-1 record below) |
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

### S-1 · `DspClient.convert_hw` — ✅ landed 2026-09-04

Landed as `DspClient.convert_hw(src, dst_fmt, fmt=None, ...,
cpu_fallback=True)` with both router directions (`rgb_to_nv12`,
`nv12_to_rgb`) registered on the DSP leg. The pieces it stands on, for
the record:

- `platform/camera-daemon/proto/camera.proto:775-828` — `DspOp` enum
  includes `DSP_OP_CONVERT_FORMAT = 3`; `DspJobRequest` carries
  `src_buffer_id` / `dst_buffer_ids` / formats.
- `platform/camera-daemon/src/dsp_service.cpp:896-899` — dispatch to
  `dsp_ops_->convert_format`.
- `hal_v2/platforms/hailo15/dsp/hailo15_dsp_impl.cpp:170` — HAL
  implementation (`hailo15_dsp_convert_format_sync`).

Design notes carried into the client: the P0 `CONVERT` contract
(**equal src/dst dimensions and differing formats**, zero rects,
exactly one dst buffer — `dsp_service.cpp:604-638`) is validated
client-side; resizes compose as `CONVERT → RESIZE`, the cheaper order
(NV12 is half the RGB bytes on the wire); `rgb24` is RGB byte order on
the wire, so BGR must be pre-swapped (CPU path: `color.bgr_to_nv12`);
CPU fallback covers all six nv12/rgb24/gray8 pairs, RGB↔NV12 delegating
to `color.py`'s BT.601 converters so both legs agree on the colorspace.
The `cpu_fallback=False` kwarg (also retrofitted onto
`resize_hw`/`crop_hw`/`multi_crop_hw`) is what keeps the router's
degradation counters honest.

On-device verification (93.72, parking_lot 1.2.0 container, 1280×720):
`rgb24→nv12` 134 ms and `nv12→rgb24` 74 ms run on the DSP (job charged,
`last_used_hw=True`); **every gray8 pair is refused by the firmware**
— `dsp_convert_format` returns a vendor failure that the HAL collapses
into `HAL_ERR_RESULT` (`-2801`, `hailo15_dsp_impl.cpp:46-56`). The pair
matrix is therefore firmware-dependent, and `convert_hw`'s default
`cpu_fallback=True` covers job rejections the same way it covers an
absent DSP service: warn, compute on CPU, `last_used_hw=False`.
Zero-copy frame sources are still refused a silent copy (use
`frame.to_array()`), and the router legs (`cpu_fallback=False`) keep
raising so a degradation is recorded exactly once.

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
- **P1 (done, SDK-only)** — S-1 `convert_hw` + both convert hardware
  legs; revisit `resize_nv12` to accept a dma-buf fast path when
  `dsp-offload` P1 lands.
- **P2 (blocked on service layers)** — S-2 NMS params, S-3 JPEG RPC;
  `draw.py` CPU raster → DSP blend follows dsp-offload P1; frame
  injection per its own proposal.
