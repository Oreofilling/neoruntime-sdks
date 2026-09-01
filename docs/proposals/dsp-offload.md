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
   `DSP_OP_CROP_AND_RESIZE` **and `DSP_OP_MULTI_CROP_AND_RESIZE`** —
   batching is day-one, not a follow-up, per the experiment
   conclusions (the per-op submission cost dominates on USERPTR
   buffers, so per-tile jobs would each pay it in full; on dma-buf it
   drops 10-15x — see [e4](#hal-validation-experiments-e4-2026-09-01)).
   Batch cap **64 rects** at validation (128 verified all-written on
   hardware, 260 silently truncates). Quota hard-coded low (~100
   jobs/s, ~120 MPix/s per app — anchored to the measured dma-buf
   floor; USERPTR buffers must be rejected or throttled, not silently
   accepted); SDK gains `DspClient.resize_hw(frame, w, h)`.
   Acceptance gate: encoder-drop soak under app load (E2b proved
   coexistence safe but did not measure encoder drops). Itemized
   per-layer work: [hardware-first-roadmap.md](hardware-first-roadmap.md).
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

## HAL validation experiments (e4, 2026-09-01)

Follow-up on the same probe (`--mode e4`), same device and load,
validating the three ops the RPC needs and the buffer mode it must
use. 1080p → 640×360 bilinear, 120 iterations per figure.

| Experiment | Result |
|---|---|
| E4-A HAL `multi_crop_and_resize` N=1..7 | USERPTR 6.9→14.0 ms; **dma-buf 0.64→1.06 ms** (N=7: 6 598 rects/s); deterministic (maxd=0) in both modes once `DMA_BUF_IOCTL_SYNC` discipline is applied |
| E4-B vendor-direct N up to 260 | N≤128 all outputs written; **N=260 rc=0 but only outputs 0-3 written** (reproducible ×3 — header's "max 260" does not hold on this firmware). One run after a truncated 260-job: every multi-crop rejected (`DSP_RUN_COMMAND_FAILED`, xrp firmware −6), self-recovered next run |
| E4-B cost split | dmabuf-src/userptr-dst N=1 = 2.1 ms vs 7.0 ms all-userptr → **the P0 "~7-10 ms submission path" is dominated by USERPTR page mapping**, ~5 ms of it on the source side alone |
| E4-C HAL `blend` | exact semantics: outside Δ±0.00, alpha=0 Δ±0.00, alpha=255 Δ−77.52 = pure-red luma (identical both modes); 1-ov/8-ov: USERPTR 9.1/10.4 ms, **dma-buf 0.67/1.57 ms** (~130 µs/ov marginal); NV12 overlay rejected rc −2801 per docs |
| E4-C contract | base NV12 only, overlays A420/ARGB only, base modified in place, alpha from overlay's alpha channel — confirmed on hardware |

Conclusions folded into
[hardware-first-roadmap.md](hardware-first-roadmap.md):

1. **dma-buf everywhere (PLAT-5) is the primary performance lever** —
   10-15x on the same op, before any scheduler sophistication.
2. **Batch cap 64** at RPC validation (128 verified, 260 silently
   truncates and once poisoned the firmware for subsequent commands).
3. **`DMA_BUF_IOCTL_SYNC` is part of the buffer contract**: write-fence
   after any CPU fill, read-fence before any CPU read — without it the
   probe saw stale reads (blend delta −57.78 vs true −77.52) and
   phantom non-determinism.
4. hal_v2's multi-crop wrapper could not take N>7 (stack storage of 7,
   count passed unclamped — OOB read inside the vendor lib). **Fixed
   2026-09-01** (platform branch `fix/hal15-dsp-batch-storage`,
   commit `4c65a595`): per-call dynamic storage, batches above
   `HAL_DSP_MULTI_CROP_MAX_OUTPUTS` (128) rejected with
   `HAL_ERR_INVALID_ARG`, and blend's shared `static
   overlays_storage[50]` replaced with per-call storage — the
   shared-state hazard noted in conclusion 1 above is gone. Verified
   on device: HAL-path N=16/64 bit-exact both mem modes (dma-buf
   N=64: 11.5 ms, 5 561 rects/s), N=129 rejected rc −2814, blend
   semantics unchanged.

## P0 implementation verified on device (2026-09-01)

PLAT-1..5 of the [roadmap](hardware-first-roadmap.md) were implemented
in the platform repo (branch `feat/dsp-service-p0`) and verified on
192.168.93.72 with all daemons live: a purpose-built client probe
(gRPC `SubmitDspJob` on the control UDS + flat `DSP_ALLOC` /
`DSP_BUF_RELEASE` fd-passing on the media UDS, both cross-compiled
against the daemon's own proto) passed **21/21** checks — connects,
allocs (32-buffer dst pool = 64 fds in one SCM_RIGHTS message), hostile
rejections (bad format, 66 fds, batch 66, released/unknown ids),
content correctness (per-ROI means, gradient monotonicity, neutral-gray
convert), sequencing (multi after resize, multi after convert), quota
enforcement (58 ok / 5 quota_rej in an unthrottled loop), and timing.

Measured through the full RPC path (N=16 MULTI_CROP, 16-rect batches of
448×288 from 1080p): **4.7-4.8 ms/job → 213 jobs/s burst, 3410 rects/s**.
Against the 3.26 ms in-process HAL figure for the same batch, the RPC
layer (gRPC serialization + fd passing + scheduler) prices at
~1.4 ms/job — acceptable for P0 and amortized by larger batches.

### Vendor limitation: NEAREST is rejected on MULTI_CROP

`MULTI_CROP_AND_RESIZE` accepts only BILINEAR(1) and BICUBIC(3);
NEAREST(0) returns `DSP_INVALID_ARGUMENT` → `HAL_ERR_RESULT` (−2801).
The vendor perf path gates on `(interp & ~2) == 1`; single-op
`crop_and_resize` still accepts NEAREST. This cost two daemon core
dumps to find, because a gRPC client that omits `set_interpolation()`
silently gets proto-default 0 = NEAREST and every multi-crop fails
with a generic −2801 — the daemon path itself was correct throughout.
Client rules: always set `interpolation` explicitly on multi-crop;
default to BILINEAR. The clean HAL now logs the vendor `dsp_status` by
name before collapsing to `HAL_ERR_RESULT`, making this class of
failure diagnosable from journalctl alone.

### Encoder-contention gate (PLAT-6, 2026-09-01)

Black-box: frame count on `/run/aipc/encoded/main.sock` while generated
`SubmitDspJob` load runs (MULTI_CROP ×16, the inference-preprocess
shape; DPM inactive, no containers — encoder is the only competing DSP
consumer). Baseline 30.05 fps → 30.02 fps under a single quota-sustained
client (19.4 jobs/s) → 30.02 fps under 8 parallel clients (~160 jobs/s
aggregate, 0 errors), worst inter-frame interval 63.9 ms, zero gaps,
zero drop warnings, instant recovery, daemon PID stable. The quota —
not the DSP — binds well-behaved clients (6.27 MPix/job → ~19-20 jobs/s
of the 120 MPix/s budget); saturation cost lands on the load clients'
latency (p50 4.8 → 11.1 ms), not on the encoder. Full table in
[hardware-first-roadmap.md](hardware-first-roadmap.md#plat-6-measurement-2026-09-01-1921689372).
**P0 is cleared to open to apps.**
