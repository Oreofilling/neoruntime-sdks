# Proposal: DSP Offload Service (`SubmitDspJob`)

Status: P0/P1/P2 landed and verified on device — this doc now carries
the implementation records below the original proposal text
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
2. **P1 (done 2026-09-07)**: `BLEND` landed end-to-end — see
   [P1 record](#p1-implementation-verified-on-device-2026-09-07).
   Buffer allocation had already shipped inside P0 (`DSP_ALLOC`); the
   `Frame.resize` fast path routes through `resize_hw` since
   parking-lot 1.1.0.
3. **P2 (done 2026-09-07)**: async jobs (submit/wait by `job_id`),
   configurable quota, and the zero-copy blend chain — see
   [P2 record](#p2-implementation-verified-on-device-2026-09-07). The
   priority field was already wired in P0 (`submit_job` enqueues
   `q_normal_`/`q_background_`); manifest-level per-app identity remains
   future work (quota buckets are per-connection today).

## Relationship to other proposals

- `ai-overlay-extended.md` requires `BLEND` (P1) for on-stream overlay
  compositing.
- `frame-injection.md` requires `CONVERT_FORMAT`/`RESIZE` to turn app
  RGB frames into the NV12 the injection node expects.

## Context-model decision experiments (2026-09-01)

Question that had to be settled before any rollout: should the daemon
hand each app client its own HAL DSP context (per-client `init`), or
keep one context and multiplex every job through it? Executed on the deployed device
with all daemons running (realistic contention). Probe
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
the deployed device with all daemons live: a purpose-built client probe
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

## P1 implementation verified on device (2026-09-07)

`DSP_OP_BLEND` (op=4) is live end-to-end on the deployed device: daemon
`build_blend` validation chain (base NV12 composited **in place** — dst
must be the imported base, BLEND alone is allowed an imported dst;
srcs are the ARGB32 overlays, 1..64, rect w/h == overlay dims, 1:1
paste, no scaling) + UDS import of base and overlays (memfd, read-only
map) + HAL leg. SDK side: `DspClient.blend_hw`, the
`render_overlay_rgba` minimal-canvas renderer, and the router's
`draw_detections` hardware leg. The e2e (19 checks, run inside the
parking-lot container against the live daemon) passes 19/19, with
`journalctl` showing the `SubmitDspJob: op=4 src=… dsts=1 rects=1`
traffic.

Numerics — the DSP is exact where exactness is required:

- outside every overlay footprint the output is **byte-identical** to
  the input (solid and renderer overlays alike);
- alpha=0 and alpha=1 regions inside a blended rect are exact
  passthrough (Δ0 luma and chroma);
- vs the numpy CPU mirror: luma max ≤ 16 across solid, gradient-alpha
  and renderer overlays. Chroma against that mirror is coarse-only by
  construction — the mirror round-trips the whole frame through
  nv12→rgb→nv12 (|Δ| median ~8, p90 ~45 before any blend) and
  re-subsamples thin strokes in RGB domain while the DSP blends in
  YUV; stroke-region chroma p90 measured 67.

### Vendor limitation: USERPTR blend overlays are refused

The firmware that accepts xrp-bounced USERPTR **sources** for resize
rejects them as blend **overlays**: `/dev/dsp_log0` shows
`idma_lookup.c:72 Tried to map buffer (READ) to different
base_address` → `map_planes(overlay) failed with 6` → vendor status 9
(base and overlay idma banks collide when mapped concurrently). The
HAL therefore stages every imported (`HAL_MEM_MALLOC`) overlay plane
through `DmaMemoryAllocator::allocate_dma_buffer` — the same dma-heap
class the vendor's own OSD blends from — syncs, memcpy, passes the fd
as `DSP_MEMORY_TYPE_DMABUF`, and frees on completion. Cost is one
memcpy per overlay per job; dma-buf-backed overlay planes skip
staging.

Two adjacent traps, recorded so nobody re-derives them:
`MediaLibraryBufferPool` **ARGB32 dma-buf acquire fails on this stack**
(`MEDIA_LIBRARY_BUFFER_ALLOCATION_ERROR` → `HAL_ERR_NO_MEM −2809`;
init succeeds — a standalone probe with unbuffered stdout settled
that), which is moot for blend (overlays import per call) but blocks
any future ARGB pool consumer; and `dsp_utils::
release_hailo_dsp_buffer` segfaults on a NULL device — use
`DmaMemoryAllocator::free_dma_buffer` for that memory instead.

### P1 timing: transport-bound, not DSP-bound

1280×720 minimal canvas (706×476): hw **180.2 ms** vs cpu 191.6 ms
per call. Both legs are dominated by the full-frame socket transport
plus NV12 copy-in/copy-back, not DSP time (e4 measured the blend
itself at 0.67-1.57 ms on dma-buf). Today's value is CPU offload and
shared scheduling, not latency; a zero-copy camera-fd base with
dma-buf overlays is the follow-up that would change the math.

## P2 implementation verified on device (2026-09-07)

Landed on branch `feat/dsp-service-p0` (platform) and SDK main:
`SubmitDspJobAsync`/`WaitDspJob` (job_id-keyed registry, monotonic
ids, per-owner pending cap 32, disconnect reaping, `stop()` drain),
the sync `SubmitDspJob` rewritten as submit+wait over the same path,
a `dsp:` YAML section for every tunable, and the SDK's
`wait=False` surface (`PendingDspJob`: `.wait()`, `.wait_result()`,
`.done()`, `.buffer_id`, idempotent `.release()`; old daemons without
the RPC fall back to the sync call with a born-done result). The
zero-copy blend chain (keep-fd base → dma-buf import → 1:1 RESIZE →
in-place BLEND → single read, or with `wait=False` +
`encode_jpeg_hw(src_buffer_id=…)` no read-back at all) is implemented
but **gated off by default** — see the firmware finding below.
Router legs (`resize_nv12`/color converts/encode) pass keep-fd
sources through verbatim instead of `ascontiguousarray`-ing them
first; those paths are safe (the resize leg is the verified one).

### `dsp:` daemon configuration

`/data/aipc/etc/camera-daemon.yaml`, all keys optional under one
`dsp:` section (defaults as logged at startup —
`max_batch=64 quota=100 jobs/s 120 MPix/s timeout=2000ms`):

| key | default | meaning |
|---|---|---|
| `quota_jobs_per_sec` | 100 | per-owner submit rate; over → rc −7 quota rejection |
| `quota_mpix_per_sec` | 120 | per-owner source-megapixel rate |
| `job_timeout_ms` | 2000 | sync wait + async wait_result deadline |
| `max_batch` | 64 | MULTI_CROP rects per job (128 verified, 260 truncates) |
| `max_buffers_per_client` | 128 | live pool buffers per owner |
| `max_client_pixels` | 16 MPix | summed live pool pixels per owner |
| `max_imports_per_client` | 64 | imported (keep-fd) buffers per owner |
| `max_async_jobs_per_client` | 32 | pending async jobs per owner |

Quota ownership is the buffer owner's UDS fd — one bucket per app
connection. There is no app-identity passthrough on `camera.sock`
(no manifest field reaches the DSP service), so "per app" is
per-connection today: an app holding two connections gets two
buckets. Recording that honestly rather than pretending otherwise;
manifest identity is future work alongside a platform-side registry.

### Verified on the live daemon (2026-09-07)

Async contract (checks 1-3 of the e2e, run inside the parking-lot
container; every item below observed across multiple runs and two
reboots — job ids 1339/1341 → 3/4 → 137/138 → 297/298 → 2/3):

- two async resizes in flight → distinct job ids, outputs
  **bit-exact vs the sync call**, luma Δ0 vs the CPU mirror;
- `priority="background"` accepted and completes through the
  background queue;
- **poll semantics, verified from the completed side** — a 0-timeout
  `WaitDspJob` answers immediately, `.done()` is True, stacked jobs
  complete in submit order with exact outputs, double `.release()`
  is idempotent, wait-after-release raises "already released";
- **timeout semantics, exercised for real** — when a blend job
  stalled server-side (heap-pressure boot, below), `wait_result`
  returned the documented "still pending … the daemon keeps it;
  wait again" error cleanly, and the server-side 2000 ms watchdog
  logged `job timed out` and reaped the job. The client never hangs
  on a dead job;
- a poll observing *pending* requires the worker to be slower than
  one UDS round trip, and on hailo15 it never is: five queue-depth
  constructions (two 4K multi-crops → correct 16 MPix client-budget
  refusal; four small multi-crops; 4K upscale + fresh submits;
  pre-staged pools allocated between submits; two fully pre-staged
  single-RPC submits) all completed the 8.3 MPix upscale *and* the
  polled resize within the ~0.6 ms submit+poll path. "Pending" is
  observable only under multi-client contention or a genuinely slow
  op — recorded as a device property, not a contract defect;
- 4K 1:1 RESIZE (the chain's copy leg, run standalone) is
  **lossless** — max delta 0 on a 3840×2160 gradient (first boot of
  the day; later boots ran the media heap at ceiling and the probe
  declined gracefully as pool-limited).

Blend, quota, and encode checks:

- **array-base blend** (the safe contract): byte-identical to the
  CPU mirror outside the overlay footprint and luma Δ0 vs it — twice
  on the final SDK build — with wall-clock median **222 ms** at
  720p (transport-bound, consistent with P1's ~180 ms);
- **keep-fd refusal** (the firmware gate): `blend_hw` raises
  "refuses keep-fd (frame/handle) bases by default" on a live
  keep-fd sub-stream frame — verified standalone and in-run;
- **quota** (check 7): `dsp: quota_jobs_per_sec: 1` in
  camera-daemon.yaml → startup line flips to `quota=1 jobs/s` in the
  journal → a 6-submit burst gets 1 accepted / 5 quota-rejected →
  config restored → defaults line back (`quota=100 jobs/s`);
- **journal evidence** (check 8): `SubmitDspJobAsync: op=… src=…
  dsts=… rects=…` per submit (11 lines across the runs);
  `WaitDspJob` has no log line by design (a pure registry lookup +
  condition-variable wait — its evidence is the client results);
- **`buffer_id` → encode chain**: the async-job half is verified
  (job completes, pools stay live past completion for a later
  read-back), but the composed encode shot is **blocked on this
  boot by an S-3-surface failure**: `EncodeImage` answers
  `standalone jpeg encoder init failed` for plain array sources
  too (discriminator run), so the init failure predates and is
  unrelated to the P2 changes — encoder and `src_buffer_id`
  composition remain covered by SDK unit tests and the 2026-09-04
  S-3 on-device record.

### Operational finding: the zero-copy blend chain has wedged the DSP (state-dependent)

Three e2e attempts, two device wedges, and one overturned theory —
recorded in full because the trap is subtle and the failure is
catastrophic.

**What happens**: `blend_hw` on a keep-fd `Frame` runs the chain
import → 1:1 RESIZE onto a fresh pool → in-place BLEND. The RESIZE
leg completes (journal evidence: the resize submit precedes the blend
submit and the SDK only submits the blend after the resize returns);
the **BLEND command then never completes** — `xrp_wait_for_cmd_
completion: timeout`, error 9 on every subsequent job device-wide,
and the driver latches `DSP encountered fatal error before. Reboot
required`. A daemon restart does not clear it; an xrp driver
unbind/rebind does not clear it (runtime PM is unsupported for the
device; the latch survives re-probe). Only a reboot recovers.
Reproduced 2/2 — once at 4K under CMA pressure, once at 720p on a
fresh boot with ~700 MB general CMA free and the media pipeline live.

**What it is NOT**:

- not an RPC-contract violation — the blend base is a legal daemon
  pool (the resized copy), overlays are legal staged ARGB;
- not the HAL staging path — `hailo15_dsp_impl.cpp:383-405` checks
  every `allocate_dma_buffer` and returns `HAL_ERR_NO_MEM` cleanly on
  failure (verified in source after the second wedge; the first
  incident report blamed an unstaged overlay and was wrong);
- not general CMA exhaustion — `/proc/meminfo` CmaFree measures the
  1.28 GB `linux,cma` window, but the media allocator draws from the
  dedicated `hailo_media_buf,cma` reserved-memory pool; dmesg shows
  `cma_alloc: hailo_media_buf,cma: alloc failed ret:-12` bursts at
  blend time while hundreds of MB of general CMA stand free. Reading
  CmaFree to judge media-buffer headroom is a measurement trap.

**Conclusion (2026-09-07; revised 2026-09-08 — see the re-test
below)**: `dsp_blend` on a pool whose contents were produced by
a RESIZE from an imported (camera dma-buf) source did not return on
the firmware builds running that day, and wedged the device twice.
The trigger is inside the firmware/idma domain. SDK contract since
the finding: `blend_hw` **refuses keep-fd bases by default**
(`zero_copy=True` forces the chain for post-fix experiments; the
router's `draw_detections` hardware leg likewise raises for frames
and stays arrays-only). Array bases — `frame.to_array()` — are the
proven path (P1 19/19). Vendor engagement on the firmware behavior
is the prerequisite for ever flipping the default; identifying the
heap/build state that triggers the wedge (below) is the
prerequisite for the vendor engagement.

What remains true and verified regardless: keep-fd **RESIZE** on
imported camera dma-bufs is safe and lossless (the chain's resize leg
completed in both incidents; a separate 4K 1:1 check measured max
delta 0), so the router's frame passthrough for resize/convert/
encode legs keeps its value.

### 2026-09-08 controlled re-test: wedge NOT reproduced

A controlled reproduction attempt on the same device class — fresh
boot, media heap ~50% free, and a newer deployed HAL build — **failed
to reproduce the wedge**. Baseline array blend OK (4K NV12, 0.30 s);
then the forced zero-copy chain (`zero_copy=True`,
`cpu_fallback=False`) **11/11 passes** at 0.12–0.33 s each, with the
journal confirming the full op=0 RESIZE → op=4 BLEND chain traffic,
zero `xrp_wait_for_cmd_completion` lines, zero fatal latches,
camera-daemon healthy throughout (NRestarts=0), no reboot needed.
Transient `hailo_media_buf,cma alloc failed` (-12) bursts at
2025/1013 pages did occur during the runs and self-recovered on
retry — the chain pressures the media heap but did not wedge it.

What differs from the 2026-09-07 incidents:

- **the deployed HAL build had changed** (the disk library had been
  swapped again between the incidents and the re-test — the 9/7
  wedges ran a different build);
- **the media heap was ~50% free** — both 9/7 incidents were
  ceiling days per the section below, and the media-pool state was
  never measured at incident time; the "~700 MB general CMA free"
  note above is exactly the measurement trap this record documents.

**Revised conclusion**: the wedge is state-dependent — correlated
with media-heap/idma pressure and/or the deployed HAL build — not an
unconditional firmware-fatal property of the chain. The SDK default
(refuse keep-fd bases) stays until the root cause is identified.
Discriminating experiments left: the same chain under artificial
`hailo_media_buf` pressure, and the same chain on the 9/7-era HAL
build. Firmware fingerprint captured for vendor engagement:
`dsp-fw.elf`, Xtensa ELF with symtab, RI-2023.11 dsp_mercury2 build,
runtime banner `DSP-FW [5d2a57a7-release]`, md5
`46782dbb8b4bd4d61be7bd0b41d4f8eb`.

### Operational finding: media-heap ceiling days vs clean days

The 2026-09-07 e2e spanned three boots of the same daemon binary and
config, and the device's DSP service behaved differently on each —
worth recording because it changes what an e2e can prove on a given
day:

- **Boot 1 (clean)**: everything passes — blend correctness, 3×
  blend timing, 4K 1:1 pools, the lot.
- **Later boots (ceiling)**: the pipeline's own frame pools fail to
  grow at startup or within minutes (`hal_v2_frame_request_pool
  pool3840x2160_32_y: Failed to allocate chunk … async_worker_loop:
  Async chunk allocation failed`, `hailo_media_buf,cma alloc failed
  ret:-12` while general CmaFree stands at ~950 MB — the same
  measurement trap as above). Under that ceiling:
  - blend **staging** or the blend job itself stalls probabilistically
    (import succeeded 5×, then a job sat pending until the 2000 ms
    watchdog reaped it; another run completed 4 blends then stalled
    on the 5th). The stall is a blocked dma-heap ioctl, not an error
    return — `DSP_IMPORT` and the job both just stop answering, and
    once the worker thread is inside the blocked call every later
    job queues behind it until a daemon restart;
  - `EncodeImage` answers `standalone jpeg encoder init failed` for
    **plain array sources too** — the standalone encoder's GStreamer
    spin-up is down for the boot, independent of any P2 code path;
  - client-visible allocs keep failing *cleanly* (`out of memory`,
    `client limit exceeded`) — the daemon's own accounting holds; it
    is the vendor allocator/ioctl layer that blocks.
- **Recovery**: a daemon restart restores allocs and small jobs
  immediately, but on ceiling days the pool drains back to the edge
  within minutes (the exhaustion also looked kernel-sticky once —
  restart did *not* restore blend staging that time; only a reboot
  did). Budget one e2e attempt per daemon restart on such days.

App-side implications: run heavy DSP transients sparsely (one 4K
pool at a time, release eagerly — the e2e now models this), treat a
`DSP_IMPORT`/wait timeout as a poisoned connection (close the
client, don't reuse the socket), and don't read media-buffer
headroom from CmaFree. The daemon-side follow-ups this surfaces:
reply with an error (not silence) when staging blocks, and make the
`job timed out` watchdog log line print its dst id (`(dst
undefined)` today — cosmetic but confusing in incident logs).
