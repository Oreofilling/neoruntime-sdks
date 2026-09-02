# Proposal: App Frame Injection (`PushFrame`)

Status: draft — daemon-side contract proposal, no code yet
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

The SoC media graph already has a memory-consumer/memory-producer
injection node wired for exactly this, and it is idle:

- Device 192.168.93.72, probed 2026-08-31 (read-only):
  - `/dev/video10` exists.
  - `media-ctl -d /dev/media0 -p` lists `entity 75:
    hailo-vid-out-mcm-in (1 pad, 1 link)` with its upstream link
    reported `[ENABLED]` — the node is in the active graph.
  - `/proc/*/fd` scan shows **zero** processes holding `video10` open —
    nothing in the platform currently injects through it.
- The platform's own encoder path demonstrates the dma-buf flow end to
  end (frames are handed to the encoder as dma-buf on this SoC), so the
  import direction app→encoder reuses known-good driver semantics.

The node being enabled-but-unused is the strongest signal in this
proposal set: the hardware capability is provisioned and simply has no
RPC surface.

## Proposed proto

```protobuf
// in camera.proto (aipc.camera)

message PushFrameRequest {
  // dma-buf fd exported by the app (via AllocateDspBuffer from
  // dsp-offload.md, or the app's own ION/dma-buf allocation).
  uint32 fd = 1;

  uint32 width = 2;
  uint32 height = 3;
  uint32 stride = 4;
  string format = 5;              // "NV12" first; RGB via DSP convert

  uint64 pts_ns = 6;              // app timestamp; daemon maps to PCR

  // Injection mode:
  //   OVERLAY  - blend over the ISP frame (needs DSP blend, P1)
  //   REPLACE  - replace the encoded frame entirely
  string mode = 7;

  // For OVERLAY: position of the injected frame within the main frame
  uint32 dest_x = 8;
  uint32 dest_y = 9;

  bool end_of_stream = 10;        // flush semantics, see below
}

message PushFrameResponse {
  bool success = 1;
  string message = 2;
  uint64 injected_frame_id = 3;   // correlates with encoder feedback
}

message InjectionStatusResponse {
  bool success = 1;
  string message = 2;
  bool active = 3;
  string mode = 4;
  uint64 frames_injected = 5;
  uint64 frames_dropped = 6;      // queue-full drops, format rejects
  uint32 queue_depth = 7;
}

service addition:
  rpc PushFrame(PushFrameRequest) returns (PushFrameResponse);
  rpc GetInjectionStatus(Empty) returns (InjectionStatusResponse);
  rpc StopInjection(Empty) returns (PushFrameResponse);  // flush + release
```

Design notes:

- **One-shot RPC per frame, no streaming bidir** — keeps the daemon's
  event loop unchanged and matches the per-frame cadence apps already
  have. A gRPC client-streaming variant is a P2 optimization.
- **fd passing over gRPC**: the daemon receives the fd number as seen
  in its own container namespace via SCM_RIGHTS on the Unix socket
  transport (`/run/aipc/camera.sock` is already Unix-domain), so no
  global-namespace assumption is made.

## Contract constraints

- **Format**: injection node expects planar YUV (`NV12`) with
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
4. **Interaction with DPM/privacy-mask**: if a privacy mask region
   covers the injected area, platform masking must still win — the
   injection point must sit *before* DPM processing in the pipeline
   ordering, verified against the media-graph link order in P0.

## Phased rollout

1. **P0 (probe)**: daemon opens `/dev/video10`, implements
   `PushFrame` REPLACE-only at sub-stream resolution, NV12 only, single
   app allow-listed. Validate: web console shows injected test pattern;
   encoder bitrate/fps unchanged.
2. **P1**: OVERLAY mode via DSP blend; RGB input accepted (daemon-side
   convert); pts-based pacing; drop counters exposed.
3. **P2**: client-streaming variant; per-app manifests; PIP layout
   presets (dest_x/dest_y composition helper in SDK).

## Relationship to other proposals

- `dsp-offload.md` supplies buffer allocation and format conversion —
  this proposal depends on it for the RGB path (P1) and reuses its
  allocation handshake.
- `ai-overlay-extended.md` is the cheap alternative for box/track
  overlays; injection is the escape hatch for anything richer.
