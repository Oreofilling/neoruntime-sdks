# Proposal: App Frame Injection (`PushFrame`)

Status: P0 COMPLETE and device-verified 2026-09-10 (uncommitted on
branch pr-63-dsp-fix). Landed: proto in `proto/camera.proto`;
InjectionService (queue/session/counters, config-gated, ships
disabled); PushFrame/GetInjectionStatus/StopInjection handlers; and
the bake-site hook — REPLACE is a content copy at the encoder bake
site (`handle_video_frame_for_routing` copies the pinned NV12 planes
into the pipeline buffer after the DPM clean-frame offer, before the
DPM/AI-overlay draws; masking wins by construction). Verification on
the deployed validation rig (sub stream 1280x720@30): 21/21 checks —
gate-off reject (-1), unknown buffer (-2), second owner (-4 BUSY),
EOS session close + ISP content restored in the decoded output,
270/270 pushed frames replaced consecutively (pixel-classified on
decode), sub fps 30.1 / main stream 30.0 during injection (9s),
wrong-dims head-of-line never injected, drop-oldest counted.
(Original P1+ list — OVERLAY blend, pts pacing, manifest
permission — completed 2026-09-11; see the P2 paragraph below.)
SDK single-step output interface shipped 2026-09-10 (Python first):
`FramePublisher` (`neoruntime_ipc_sdk/injection.py`) owns a
stream-matched DSP NV12 buffer pool (leak-free constructor, `close()`
idempotent, context manager) and exposes `publish(frame)` /
`publish_eos()`; `CameraClient` gains the thin wrappers
`push_frame` / `injection_status` / `stop_injection`; documented in
`docs/api/injection.rst` (zh/en). Rig E2E closed the full write path —
synthetic color bars → DSP pool → `publish` → PushFrame RPC (buffer_id
by registry) → bake-site REPLACE → H.264 sub stream →
`EncodedStreamClient` capture → host decode: 40 publishes at 10 Hz,
all accepted (`frames_injected=40`, `frames_dropped=0`), 38
bar-pattern frames verified on decode, tail clean after EOS (ISP
restored at the next IDR). The same run pinned the REPLACE cadence
semantics: **one injection replaces exactly one encoded frame** —
10 Hz into a 30 fps encode yields ~10 replaced fps, so apps pace
publishing at their desired replaced-frame rate.
P2 COMPLETE and device-verified 2026-09-11 (uncommitted on
pr-63-dsp-fix). Landed: OVERLAY mode — an NV12 dma-buf pastes
opaquely at `dest_x/dest_y`, an ARGB32 buffer alpha-blends on the
CPU at the bake site (`build_blend`; deliberately CPU because the
DSP queue is one congested worker — see `dsp-offload.md` P1-9);
streaming — `PushFrame` is client-streaming with a shallow
drop-oldest queue and pts pacing (newest-due-wins; `pts_ns` in the
device CLOCK_MONOTONIC domain, far-future pts self-limits via
overflow + EOS reclaim); and the manifest permission gate —
`injection.allowed_apps` resolves the buffer owner's UDS identity
(SO_PEERCRED) and rejects unlisted apps with `-6
INJ_SVC_ERR_PERMISSION` (empty list = allow all, preserving the
P0 behavior). Two rig-found fixes worth remembering: ARGB32 pool
allocation is refused by the deployed HAL (wire OOM), so SDK
alpha overlays ride the shared memfd-import ring instead
(`FramePublisher(fmt="argb")`), and the blend's chroma
byte-addressing for subsampled UV at odd dest offsets was wrong
until corrected (dest coordinates must be even). Verification on
the deployed validation rig: 8/8 e2e checks — ARGB inset blend
pixel-verified on decode (alpha gradient corners), NV12 opaque
paste, streaming multi-frame with drop-oldest, pts pacing,
permission deny (`-6`, session stays inactive) and allow (default
config), EOS restore at next IDR; daemon config restored to
baseline after the deny run. SDK side shipped in 0.7.4:
`FramePublisher(mode="overlay", fmt="nv12"|"argb", inset=, dest=)`.
Remaining (not blocking the contract): RGB888 input, DSP-offloaded
blend (depends on P1-9 queue split), zero-copy buffer swap (handing
the media-library buffer straight to `add_buffer` instead of the
plane copy). Browser-side closure shipped the same day:
`platform_stream_url()` (`web.py`) hands viewers the gateway URL
of the injected stream — see `web-stream-url.md` P0 outcome.
Target service: camera-daemon (`aipc.camera.CameraControl`)
SDK layer affected: Python + C++ (`Frame` producers)

## Motivation (the app developer's problem)

Apps today are read-only consumers of the camera pipeline: they can
subscribe to streams, run inference, and draw overlays — but the
outgoing encoded stream can only show what the ISP produced (plus the
platform's own OSD). Whole classes of camera apps are blocked on the
missing write path:

- **Augmented feeds**: picture-in-picture (thermal inset, zoomed crop of
  a region of interest), app-composited layouts.
- **Redaction-hardened output**: the app scrubs faces/plates (its own
  blur or replacement) and the *scrubbed* frame is what gets encoded and
  streamed, not the raw one.
- **Synthetic sources**: an app that renders a dashboard/heatmap frame
  and wants it on the main stream when the camera is idle.
- **Analytic overlays richer than rectangles**: polygons, tracks, heat
  maps — trivially drawn in app code, impossible via `AiOverlayConfig`
  today (see `ai-overlay-extended.md` for the lighter-weight path).

All of these reduce to one primitive: *let the app hand a composed frame
to the encoder*.

## Existing hardware path (evidence)

**Correction after full probe, 2026-09-10 (read-only, deployed v1.0.2
rig)** — the original P0 assumption below does not hold; see the
revised rollout:

- `/dev/video10` (`hailo-vid-out-mcm-in`, entity 75, fed by
  `hailo-isp:5`, link ENABLED) is the **only OUTPUT node in the whole
  media graph — and it accepts 12-bit Bayer only** (pGCC/pRCC/RG12/
  GB12; `v4l2-ctl --list-formats-out-ext`). It is an **ISP-front
  virtual-sensor input**, not a pre-encoder YUV injection point.
- Every other video node (`video0/1/2/20-23`) is Video **Capture**
  Multiplanar (YVYU/YUYV ISP output taps). **No NV12-accepting OUTPUT
  node exists anywhere**; the encoders are fed internally by the
  pipeline and have no userspace-facing input device.
- Zero processes hold `video10` open; its format is unset (0/0) —
  never configured by anything.

Consequence: "hand the app's composed frame to the encoder" has **no
dedicated hardware door**, but the daemon already owns the exact
software frontier: the bake site (commit 7cfc9526) holds every frame
immediately before the encoder's `add_buffer`. An injected app frame
(NV12 dma-buf received over `camera.sock` SCM_RIGHTS) can be submitted
through the same `add_buffer` path in place of the ISP frame —
REPLACE at the output branch, zero new media-graph nodes.
`/dev/video10` stays documented as a *different* future capability:
synthetic-Bayer virtual-sensor input (sensor playback / test scene),
not the composed-frame path.

Original (superseded) probe note, 2026-08-31: `media-ctl -p` showed
entity 75 with its link `[ENABLED]` and zero holders — read as "HW
injection provisioned and idle"; the format/topology probe above is
what the earlier snapshot lacked.

## Proposed proto

```protobuf
// landed in camera.proto (aipc.camera) 2026-09-10 — this block mirrors
// the checked-in definitions; the file is authoritative.

enum InjectionMode {
  INJECT_REPLACE = 0;   // replace the encoded frame entirely (P0)
  INJECT_OVERLAY = 1;   // blend over the ISP frame (P1, DSP blend)
}

message PushFrameRequest {
  // No fd field on purpose: a bare fd number crossing gRPC is the
  // Tensor.dma_fd anti-pattern (see composable-pipeline-contracts.md
  // design point 3). The buffer travels out-of-band: the app sends
  // its NV12 dma-buf over the camera.sock UDS with SCM_RIGHTS (same
  // framing as DSP_IMPORT) and this RPC references it by registry id.
  uint64 buffer_id = 1;        // DSP-registry id of the NV12 frame;
                               // 0 valid only with end_of_stream=true

  uint32 width = 2;            // must equal the target stream encode width (even)
  uint32 height = 3;           // must equal the target stream encode height (even)
  uint32 stride = 4;           // luma stride of the registered buffer

  InjectionMode mode = 5;      // REPLACE in P0; OVERLAY is P1 (DSP blend)

  uint64 pts_ns = 6;           // app timestamp; daemon maps to the stream clock

  // For OVERLAY (P1): position of the injected frame within the main frame
  uint32 dest_x = 7;           // ignored by REPLACE
  uint32 dest_y = 8;

  bool end_of_stream = 9;      // flush semantics, see below
}

message PushFrameResponse {
  bool success = 1;
  string message = 2;
  int32 error_code = 3;         // negative HAL error or -EINVAL style validation
  uint64 injected_frame_id = 4; // running id; correlates with frames_injected
}

message InjectionStatusResponse {
  bool success = 1;
  string message = 2;
  bool active = 3;              // a session is currently injecting frames
  InjectionMode mode = 4;
  uint64 frames_injected = 5;   // frames handed toward the encoder feed
  uint64 frames_dropped = 6;    // queue-full drops + dimension/format rejects
  uint32 queue_depth = 7;       // live depth of the inject queue (cap 3)
}

service addition:
  rpc PushFrame(PushFrameRequest) returns (PushFrameResponse);
  rpc GetInjectionStatus(Empty) returns (InjectionStatusResponse);
  rpc StopInjection(Empty) returns (InjectionStatusResponse);  // flush + release
```

Design notes:

- **One-shot RPC per frame, no streaming bidir** — keeps the daemon's
  event loop unchanged and matches the per-frame cadence apps already
  have. A gRPC client-streaming variant is a P2 optimization.
- **Control/data plane split**: gRPC carries metadata only (dims,
  format, mode, pts, EOS). The dma-buf rides the camera.sock UDS with
  SCM_RIGHTS and is `dup()`ed into the daemon's fd namespace before
  use — the DSP_IMPORT handshake, not a raw fd number in a protobuf
  field. Vanilla gRPC has no ancillary-data API, so "fd over gRPC" was
  never actually available; the UDS import step is the mechanism.

## Contract constraints

- **Format**: the encoder feed expects planar YUV (`NV12`) with
  width/height matching the target stream's encode resolution and
  even dimensions. Apps holding RGB compose via the DSP
  `CONVERT_FORMAT` op (`dsp-offload.md` P1) rather than CPU.
- **Cadence & drops**: the encoder consumes at stream fps; the daemon
  keeps a shallow queue (2-3 frames). An app pushing faster gets
  `frames_dropped` increments — never backpressure that could stall the
  encoder. `PushFrame` is explicitly *lossy-tolerant*.
- **EOS/flush**: `end_of_stream=true` (or `StopInjection`) drains the
  queue and restores the pure ISP path atomically at the next IDR, so
  the stream never mixes half-replaced GOPs.
- **Ownership**: the app retains ownership of the dma-buf; the daemon
  syncs (`DMA_BUF_IOCTL_SYNC`) before the encoder reads it and never
  holds a reference past the encode.
- **Session tag (P2-13, shipped 2026-09-11)**: `PushFrame` carries an
  optional `session_id`; the injection session opens on the first
  tagged frame (buffer-owner connection recorded, tag observable in
  `InjectionStatus`). The tag is correlation/observability only —
  reclaim is keyed on the buffer owner's UDS connection: on that
  client's disconnect (SIGKILL included) the daemon closes the session
  and frees every buffer the client registered, regardless of the tag.
  One chain does all of it: `FdPublisher::disconnect_client` →
  release_all_outstanding → `release_owner(fd)` →
  `release_client_buffers(fd)`.

## Risks

1. **Frame pacing**: an app injecting at wrong fps produces judder; the
   daemon should pace-release queued frames against the stream clock
   (pts_ns assists), not encode them on arrival.
2. **Security/privacy**: REPLACE mode means the app decides what the
   "camera" shows. This must be gated per-app (manifest permission) and
   reflected in the web console, mirroring how `AiOverlayConfig` is
   app-scoped today.
3. **Resource leak surface**: fds held by dead containers must be
   reaped; tie injected-buffer lifetime to the app's gRPC session
   (release on channel close), which the daemon already tracks for
   overlay subscribers (`ai_overlay_subscriber.cpp:29` pattern).
   **Shipped 2026-09-11** — a SIGKILL-mid-pipeline e2e on the deployed
   rig reclaimed all four layers (injection session closed + tag
   cleared, tagged polygons swept, per-client DSP buffers freed,
   StreamInfer session ended) within the same second, 8/8 checks.
4. **Interaction with DPM/privacy-mask**: if a privacy mask region
   covers the injected area, platform masking must still win. The
   media-graph ordering check this risk originally asked for ran on
   2026-09-10 and ruled out the HW node; with encoder-feed insertion
   the remaining check is daemon-internal ordering — the REPLACE point
   must sit *after* DPM/OSD processing (or re-apply masking to
   injected frames), verified against the dpm_worker → bake →
   add_buffer sequence before REPLACE ships to any real deployment.

## Phased rollout

1. **P0 (encoder-feed insertion)**: daemon implements `PushFrame`
   REPLACE-only at sub-stream resolution, NV12 only, single app
   allow-listed. Injected frames enter at the bake/`add_buffer`
   frontier in place of the ISP frame (no media-graph changes; the
   2026-09-10 probe found no usable hardware injection node — see
   evidence above). Validate: web console shows the injected test
   pattern; encoder bitrate/fps unchanged; `StopInjection` returns
   the ISP path at the next IDR.
2. **P1**: OVERLAY mode via DSP blend; RGB input accepted (daemon-side
   convert); pts-based pacing; drop counters exposed.
3. **P2**: client-streaming variant; per-app manifests; PIP layout
   presets (dest_x/dest_y composition helper in SDK); optionally the
   `/dev/video10` virtual-sensor path for synthetic-Bayer scenes
   (sensor playback), kept separate from composed-frame injection.

## Relationship to other proposals

- `dsp-offload.md` supplies buffer allocation and format conversion —
  this proposal depends on it for the RGB path (P1) and reuses its
  allocation handshake.
- `ai-overlay-extended.md` is the cheap alternative for box/track
  overlays; injection is the escape hatch for anything richer.
