# Proposal: SDK Hardware Routing (Loose Components + Capability Router)

Status: SDK skeleton **landed 2026-09-04** (`accel.py` router,
`OverlayClient.annotate`, `color.py` / `postprocess.py` software legs,
338 tests green); **P1 `DspClient.convert_hw` landed same day** (both
convert directions registered as DSP hardware legs; verified on-device
on 93.72 — RGB↔NV12 run on the DSP, the gray8 pairs are firmware-
refused and degrade to CPU; 362 tests green); **P2 S-2 verified
2026-09-04** — the runtime NMS-tuning chain ships today:
`detection_threshold` is runtime-tunable for family-function
postprocess models, `iou_threshold`/`max_boxes` are HEF compile-time
(verified on 93.72, probes v4-v6); **dsp-offload P2 landed 2026-09-07**
— `wait=False` async jobs (`PendingDspJob`), the keep-fd zero-copy
blend chain, router dma-buf fast paths, and `draw_polygons`/
`render_overlay_rgba(polygons=, tracks=)` (S-6 record below);
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
| Box suppression | `nms` | numpy | ✅ already in the HEF's integrated NMS (compile-time knobs; runtime `detection_threshold` via `update_postprocess_config`) | — (verified; see S-2) |
| Draw onto outgoing stream | `OverlayClient.annotate` | — | ✅ camera-daemon renderer | — (live; contract below) |
| Raster drawing (local frames) | `draw.py` | CPU raster | ✅ DSP `blend_hw` (`render_overlay_rgba` → blend; keep-fd base = zero-copy chain) | — (live; S-5/S-6 records below) |
| Snapshot JPEG | `encode_jpeg` / `encode_jpeg_hw` | cv2/Pillow | ✅ camera-daemon `EncodeImage` (N-threaded libjpeg on the DSP core) | — (live; S-3 record below) |
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

### S-2 · ai-runtime: expose NMS registration params — ✅ verified 2026-09-04 (no platform change needed)

The original ask (accept `nms_threshold` / `confidence_threshold` /
`max_detections` as registration parameters, replacing the hard-codes at
`model_manager.cpp:253`/`:309`) turned out to be already answered by
machinery that ships today. The full chain exists end-to-end:

- SDK `InferenceClient.update_postprocess_config(model_id, json)` →
  gRPC `UpdatePostprocessConfig` → `ModelManager::update_postprocess_config`
  → HAL `apply_config_json` (`hailo15_postprocess_impl.cpp:2165`):
  patches `merged_vendor_json` with the numeric keys
  (`detection_threshold`, `iou_threshold`, `max_boxes`, …), rewrites the
  plugin's temp config file and re-runs the plugin's `init`.
- At registration, the model's variant JSON blob IS the tuning channel
  (`init_post_process`, `model_manager.cpp:313-378`) — a full blob lands
  in the plugin's config verbatim.

On-device ground truth (93.72, hailo15, probes v4-v6 on a fixed test
image with two `vehicle` detections at scores 0.886 / 0.771):

- **`detection_threshold` is runtime-tunable — for family functions
  only.** Sweeping it on `hailo_yolov8n_384_640.hef` (postprocess
  resolves to the `hailo_yolov8n` default) moved the count exactly as
  the scores predict: ≥0.886 → 0 boxes, ≤0.771 → 2 boxes, across ten
  pushed values with no re-registration.
- **Generic plugin exports ignore all JSON tuning.** The identical sweep
  on `yolov5m_vehicles` (variant names `backend_function:
  "yolov5m_vehicles"`) never moved the results; the device's journald
  shows the repo's own guard warning
  (`hailo15_postprocess_impl.cpp:1324`) for that registration. App
  consequence: a custom-named detection model's thresholds are
  compile-time fixed — re-compile the HEF or switch the postprocess to a
  family function to tune at runtime.
- **`iou_threshold` / `max_boxes` are chain-accepted but behaviorally
  inert.** The test pair overlaps at IoU ≈ 0.32, yet `iou_threshold`
  0.99 vs 0.01 and `max_boxes=1` never changed the output: suppression
  and capping happen in the HEF's compile-time integrated NMS
  ("HEF nms: 1 class, 80 boxes/class" at activation), which the plugin
  cannot re-knob after compilation.

SDK side (landed with this update): honest applicability notes in the
`update_postprocess_config` docstring, three new tests (request shape,
failure surfaces the server's `-2801`, CLIP prompts ride the same RPC),
zh+en `inference.rst` examples, and the router's `nms` op note now says
the hardware leg already ran — pre-app, inside the HEF.

Side finding recorded for apps parsing raw NMS tensors: the
family-function NMS tensor layout is `[pad, count, rows of
(ymin, xmin, ymax, xmax, conf)]` — the count is `float[1]`, not
`float[0]` (parking-lot's `parse_nms_raw` reads `float[0]`, which reads
0 for family models; the objects path is unaffected).

### S-3 · codec: hardware JPEG encode RPC — done 2026-09-04 (option a)

`frame.py:_encode_jpeg` burned CPU cv2 on every snapshot. Landed as
option (a): **`EncodeImage(src_buffer_id, quality)` unary RPC on
camera-daemon** — source pinned in DspService (zero-copy for keep-fd
frames, pool-copied for arrays), complete JPEG bytes in the response, no
destination buffer, no read-back. `camera.proto` +
`camera_control_service.cpp` (platform repo) and
`DspClient.encode_jpeg_hw` + router op `encode_jpeg` (SDK) both verified
on 93.72.

**Premise correction**: this doc earlier assumed "the platform's own
thumbnails use the SoC encoder" — hailo15 has **no dedicated JPEG encode
block**. The daemon's encoder is N-threaded libjpeg on the DSP core
behind a GStreamer dispatch (`hailoencodebin` → `hailojpegenc`). The win
is centralized encode + zero-copy dma-buf input (app images can drop
cv2/PIL), not raw speed: ~160 ms warm for 384×216, ~560 ms for a 4K
snapshot (first frame of any key pays pipeline spin-up).

Platform findings the next user of the standalone encoder needs:

- **The pipeline negotiates NV12 only.** Every `hailoencodebin` carries
  an OSD element whose pad templates accept NV12 alone
  (`gsthailoosd.cpp:85`), even though `hailojpegenc` itself lists
  RGB/BGR/GRAY8 among its sink caps — an RGB appsrc fails caps
  negotiation deep in the bin (`Internal data stream error`, no packet,
  2 s shot timeout). The daemon therefore normalizes before feeding:
  NV12 direct; RGB/BGR via the DSP CONVERT job into a per-call scratch
  NV12 buffer; anything else gets a clear error the SDK treats as a
  normal hardware miss.
- **A standalone (non-pipeline) encoder needs a ConfigManager
  interactor.** The HAL clones the compiled-in default profile with
  `sensor_id="SENSOR_1"`, registers it via `ConfigManagerInteractor::
  create`, and discovers the plugin's global slot by scanning
  `/proc/self/maps` for `libgstmedialib.so` + `dlopen(RTLD_NOLOAD|
  RTLD_LOCAL)` + `dlsym` (the plugin exports its symbols RTLD_LOCAL).
  SENSOR_1's real full-json interactor coexists — both registrations
  live in the manager side by side.
- **Vendor destroy-on-error hazard (libmedialib.so.1).** A GStreamer bus
  error quits the encoder's main loop from the bus callback; `stop()`
  then early-returns on `!is_started()` without joining the loop
  thread, and `~Impl` destroys a still-joinable `std::thread` →
  `std::terminate` → daemon SIGABRT (bit us once on device). HAL guard:
  fed-vs-delivered frame accounting; at deinit, an unbalanced encoder is
  parked in a process-lifetime graveyard instead of destroyed (bounded,
  one per failed context), balanced encoders destroy normally. With the
  NV12 normalization in place the only known error trigger is gone and
  the guard is pure defense-in-depth.

SDK contract: `encode_jpeg_hw(src, quality=85)` — arrays copy into one
pool buffer, keep-fd frames import their dma-bufs (and refuse the silent
CPU fallback, as everywhere), gray8 arrays up-convert to rgb24
client-side (R=G=B) and ride the same hardware leg, gray8 keep-fd frames
raise with a `to_array()` hint. Encoder daemon-side is keyed
`(w, h, fmt, quality)` — key changes re-spin the pipeline. The router's
`encode_jpeg` op uses `cpu_fallback=False` so degradations are honest.

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

### S-5 · raster drawing: DSP blend leg — done 2026-09-07 (dsp-offload P1)

`draw.py` was pure CPU raster. The hardware leg is now
`render_overlay_rgba(frame_w, frame_h, boxes, labels, scores)` → a
minimal-canvas straight-alpha RGBA overlay → `DspClient.blend_hw(nv12,
[(rgba, x0, y0)])`, and the router's `draw_detections` op routes NV12
frames through it (RGB input stays on the software raster). Verified on
93.72 (19-check e2e: untouched-region byte-identity, hw==CPU-mirror
luma ≤ 16, alpha=0/1 exact passthrough, sub-16 padding, router legs).

Numbers and platform findings live in
[dsp-offload.md](dsp-offload.md) (P1 record); the short version: DSP
blend numerics are exact where they must be, but the firmware refuses
USERPTR blend overlays (dma-heap staging inside the HAL fixes it), and
at 1280×720 both legs measure ~180 ms wall-clock because the round
trip is transport-bound — the value today is CPU offload, not latency.

### S-6 · async jobs + zero-copy chain + shape polish — done 2026-09-07 (dsp-offload P2)

Three additions on the same routing surface:

- **Async**: every `_hw` op takes `wait=False` and returns a
  `PendingDspJob` (`.wait()` → ndarray, `.wait_result(timeout_s)` →
  completion without read-back, `.done()`, `.buffer_id`, idempotent
  `.release()`). Old daemons without `SubmitDspJobAsync` fall back
  to the sync RPC with a born-done result — honest degradation, no
  error.
- **Zero-copy chain (implemented, gated off)**: `blend_hw` accepts a
  keep-fd `Frame`/`FrameHandle` base via `zero_copy=True` — import →
  1:1 RESIZE onto a fresh pool → in-place BLEND → one `pool.read(0)`;
  with `wait=False` + `encode_jpeg_hw(src_buffer_id=job.buffer_id)`
  the pixels never cross the socket. **Refused by default**: the
  chain is firmware-fatal on current hailo15 (blend never returns,
  DSP wedged device-wide 2/2 — see the dsp-offload P2 record), so
  the default contract is array bases. Router legs pass keep-fd
  sources through verbatim for resize/convert/encode (no
  `ascontiguousarray` pre-copy) — those are safe and verified; the
  `draw_detections` hardware leg is arrays-only.
- **Polish**: `render_overlay_rgba(polygons=[(pts, color)],
  tracks=[(pts, color)])` joins the minimal-canvas union (zones and
  trajectories composite in the same single blend), software
  `draw_polygons(image, shapes, closed=)` for RGB, and gray8
  converts stop paying the submit-refuse-fallback round trip (the
  known firmware refusal is now detected client-side).

Full contract, the `dsp:` daemon config table, verified-on-device
evidence (async jobs bit-exact, quota enforcement, blend correctness
+ 222 ms timing, keep-fd refusal), and the two operational records —
the firmware-fatal zero-copy chain and the media-heap ceiling that
makes blend/encoder availability boot-dependent:
[dsp-offload.md](dsp-offload.md) P2 record.

## Phased rollout

- **P0 (done)** — router skeleton, overlay direct-push, color/NMS
  software legs, tests, zh+en API docs.
- **P1 (done, SDK-only)** — S-1 `convert_hw` + both convert hardware
  legs; revisit `resize_nv12` to accept a dma-buf fast path when
  `dsp-offload` P1 lands.
- **P2 (S-2 done 2026-09-04)** — S-2 verified on-device with no
  platform change needed (`detection_threshold` runtime-tunable on
  family functions; `iou_threshold`/`max_boxes` are HEF compile-time);
  SDK docs/tests landed.
- **P3 (S-3 done 2026-09-04)** — `EncodeImage` unary RPC on
  camera-daemon + `encode_jpeg_hw`/router op landed and e2e-verified on
  93.72 (all legs incl. keep-fd 4K); platform constraints and the
  vendor destroy hazard recorded in the S-3 section. Remaining:
  `draw.py` CPU raster → DSP blend follows dsp-offload P1, frame
  injection per its own proposal.
- **P4 (done 2026-09-07)** — dsp-offload P2: async `PendingDspJob`
  surface, keep-fd zero-copy blend chain (`src_buffer_id` encode
  chaining included), router dma-buf fast paths, polygon/track shapes,
  gray8 warning cleanup. `resize_nv12`'s dma-buf fast path promised in
  P1 is delivered here. Frame injection remains per its own proposal.
