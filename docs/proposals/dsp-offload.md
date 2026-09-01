# Proposal: DSP Offload Service (`SubmitDspJob`)

Status: draft — daemon-side contract proposal, no code yet
Target service: camera-daemon (`aipc.camera.CameraControl`)
SDK layer affected: Python + C++ (Tier 1 helpers would migrate to it)

## Motivation (the app developer's problem)

The two most common per-frame costs in an app loop are geometry work —
crop/resize/letterbox before inference, and scaling the 4K main stream
down for a preview JPEG — and compositing (drawing overlays onto the
encoded stream). Today both happen either:

- in app CPU code (SDK `Frame.resize` / `draw.py`), burning the app
  container's CPU quota at 4K, or
- not at all for the encoded path (apps cannot composite onto the
  outgoing stream; see `ai-overlay-extended.md`).

The SoC has a dedicated DSP that already does exactly this work for the
platform: the encoder's pre-scale, dewarp, and DPM resize all run on it.
Exposing a thin job-submit RPC lets apps do per-frame geometry and blend
at hardware speed with zero CPU cost, and is the foundation for the
extended AI overlay.

## Existing hardware path (evidence)

The HAL surface is complete and production-hardened — this proposal adds
RPC plumbing, not new hardware work:

- `hal_v2/include/dsp/hal_dsp.h` (platform repo) — synchronous ops
  `convert_format`, `resize`, `crop_and_resize`, `multi_crop_and_resize`,
  `blend`, `flip_rotate`, `privacy_mask` (lines 279-285), plus the full
  async job API `submit` / `wait(timeout_ms)` / `cancel` / `job_release`
  (lines 302-337). Context-based (`init`/`deinit`) — but see
  [Context-model decision experiments](#context-model-decision-experiments-2026-09-01):
  measured on hardware, extra in-process contexts buy zero parallelism,
  so the daemon keeps **one** context and multiplexes.
- `platform/camera-daemon/src/dpm_worker.cpp:473` — the daemon already
  initializes and drives a DSP context (`dsp_ops_->init(...)`), proving
  the driver path works inside a daemon.

## Proposed proto

```protobuf
// in camera.proto (aipc.camera)

enum DspOp {
  DSP_OP_RESIZE = 0;
  DSP_OP_CROP_AND_RESIZE = 1;
  DSP_OP_MULTI_CROP_AND_RESIZE = 2;
  DSP_OP_CONVERT_FORMAT = 3;
  DSP_OP_BLEND = 4;
  DSP_OP_FLIP_ROTATE = 5;
}

message DspJobRequest {
  DspOp op = 1;

  // Buffer passing: dma-buf fd number exported into the app container's
  // /dev/dmabuf, or a shared-memory segment id. App allocates via
  // AllocateDspBuffer (below) so width/stride/format are DSP-legal.
  uint32 src_fd = 2;
  uint32 dst_fd = 3;

  uint32 src_width = 4;
  uint32 src_height = 5;
  uint32 src_stride = 6;
  string src_format = 7;          // "NV12", "RGB888", ...
  uint32 dst_width = 8;
  uint32 dst_height = 9;
  uint32 dst_stride = 10;
  // crop rect for CROP_AND_RESIZE; repeated for MULTI_
  repeated DspRect rects = 11;
  // blend params for BLEND (alpha, offset x/y)
  float alpha = 12;

  // Scheduling hint, see Constraints
  DspPriority priority = 13;
}

enum DspPriority {
  DSP_PRIORITY_BACKGROUND = 0;    // may be dropped under load
  DSP_PRIORITY_NORMAL = 1;        // default
  DSP_PRIORITY_REALTIME = 2;      // reserved for platform use
}

message DspJobResponse {
  bool success = 1;
  string message = 2;
  uint64 job_id = 3;              // for the async form
}

message DspBufferRequest {        // allocation handshake
  uint32 width = 1; uint32 height = 2; uint32 stride = 3;
  string format = 4; uint32 count = 5;
}
message DspBufferResponse {
  bool success = 1; string message = 2;
  repeated uint32 fds = 3;
}

service addition:
  rpc AllocateDspBuffer(DspBufferRequest) returns (DspBufferResponse);
  rpc SubmitDspJob(DspJobRequest) returns (DspJobResponse);   // sync form first
```

Deliberately minimal: synchronous single-job form first (daemon runs the
op and replies). The async submit/wait pair from `hal_dsp.h` is a
follow-up once there is a consumer that needs pipelining.

## Scheduling constraints and risks

The DSP is a **shared, single-ordering resource** — this is the core
risk of the proposal, and the reason it needs daemon arbitration rather
than direct app access:

- `dpm_worker.cpp:815-817`: when a DPM resize loses the race with the
  encoder for the shared DSP, the daemon *drops the frame* today
  ("DSP contention with the encoder (shared DSP)"). App-submitted jobs
  must never be able to induce that path for the platform's own frames.
- `camera_daemon.cpp:1458`: streaming-thread work is carefully bounded
  around DSP resizes; arbitrary app job sizes would break that bound.

Mitigations baked into the contract:

1. **Daemon-side priority queue, one serializing lock**: platform jobs
   (encoder pre-scale, DPM) always preempt `BACKGROUND`/`NORMAL` app
   jobs; app jobs run in FIFO batches between platform batches.
2. **Quota**: per-app jobs-per-second and pixels-per-second budget,
   default-deny above it (return `success=false`, not a stall).
3. **Size caps**: reject ops above a max pixel count at validation time.
4. **Timeout policy**: daemon-side `wait(timeout_ms)` already exists in
   the HAL; a job that times out is cancelled and reported, never
   queued indefinitely.

Risk if unaddressed: an app loop submitting 4K resizes at 30 fps would
steal DSP time from the encoder and visibly stutter the main stream —
hence quota-first, open-by-opt-in.

## Phased rollout

1. **P0 (probe)**: daemon exposes `SubmitDspJob` with
   `DSP_OP_CROP_AND_RESIZE` only, quota hard-coded low; SDK gains
   `DspClient.resize_hw(frame, w, h)`. Measure encoder impact on 93.72.
2. **P1**: add `BLEND` + buffer allocation RPC; port SDK
   `Frame.resize` fast path to it opportunistically (keep numpy
   fallback).
3. **P2**: async jobs (submit/wait by `job_id`), priority field wired
   to the real queue, quota configurable per app manifest.

## Relationship to other proposals

- `ai-overlay-extended.md` requires `BLEND` (P1) for on-stream overlay
  compositing.
- `frame-injection.md` requires `CONVERT_FORMAT`/`RESIZE` to turn app
  RGB frames into the NV12 the injection node expects.

## Context-model decision experiments (2026-09-01)

Question that had to be settled before any rollout: should the daemon
hand each app client its own HAL DSP context (per-client `init`), or
keep one context and multiplex every job through it? Executed on device
192.168.93.72 with all daemons running (realistic contention). Probe
source is archived next to this doc (`dsp_p0_probe.cpp`; poky
cross-compile command in its header). Buffers were malloc'd USERPTR
NV12; op under test is `resize` 1920x1080 -> 640x360 bilinear.

### Static evidence (libhailodsp 1.12.0 vendor sources)

- `device.cpp`: `dsp_create_device` = `open("/dev/dsp0", O_RDWR)` — no
  exclusive flag, multiple handles are possible by construction.
- `send_command.cpp:42`: **every** op is enqueued to the process-wide
  `PriorityQueueSingleton` — one queue, one dispatch thread per process.
- `dsp_set_priority` only tags the handle's priority inside that queue;
  a context owns no execution resource of its own.

### Measured results (live load)

| Experiment | Result |
|---|---|
| E1 double-init | 2 contexts in one process: both `init` OK, 2 dsp0 fds held, both functional, identical output checksum — on top of camera-daemon's own cross-process fd |
| E2a serial baseline | 139.9 ops/s, per-op mean 7.1 ms (p95 11.2 ms, p99 13.2 ms) |
| E2a 2 ctx + 2 threads | **speedup 1.00x** — aggregate throughput unchanged; per-op latency doubles to 14.3 ms mean (jobs queue behind the singleton) |
| E2a 1 ctx + 2 threads | also 1.00x with **zero errors** — concurrent sync calls on one context are safe |
| E3 sync vs async (depth 4) | 100.2 vs 106.4 jobs/s — no throughput gain, caller thread freed only; Little's-law check 4 in-flight / 37.2 ms = 107 jobs/s matches the measured 106.4: one serial server |
| E2b cross-process hammer | 10 s, 1093 ops, 0 errors alongside camera-daemon; DSP utilization avg 19 % / peak 21 % (49/49 samples ok); after the run all daemon PIDs unchanged, no new log errors |

### Conclusions

1. **One context, daemon-multiplexed** is the confirmed model. Extra
   in-process contexts buy zero parallelism (they double per-op latency
   instead) while adding shared-state hazards (e.g. the static
   `overlays_storage[50]` in the HAL impl). The earlier
   "per-client sessions" idea is dropped.
2. **The DSP core is not the bottleneck — the ~7-10 ms per-op
   submission path is.** Utilization stayed at 19 % while sustaining
   ~225 MPix/s source plus the daemon's own encoding/scaling work.
   The RPC should therefore encourage batching
   (`DSP_OP_MULTI_CROP_AND_RESIZE`) over per-tile jobs.
3. **Priority and quota belong in the daemon queue**, exactly as the
   mitigations above assume. Coexistence across processes is safe
   (E2b), but the driver/firmware has no app-aware arbitration —
   fairness between apps can only come from the daemon.

Numbers were taken under live system load (load avg ~5-6, encoder and
inference active): treat them as contended-floor figures, not peak
benchmarks.
