# Hardware-First Roadmap: Platform & HAL Work Items

Status: work breakdown — what must change outside the SDK so its
CPU-implemented helpers can become hardware-first. Grounded in the
[context-model decision experiments](dsp-offload.md#context-model-decision-experiments-2026-09-01)
(run 2026-09-01 on 192.168.93.72) and in code references into the
platform repo.

Two decisions from those experiments shape every item below:

- **D1** — one HAL DSP context, daemon-multiplexed. Extra in-process
  contexts buy zero parallelism (measured speedup 1.00x); per-client
  context init is dropped as a design option.
- **D2** — the ~7-10 ms per-op submission path, not the DSP core
  (19 % utilization under load), is the cost center → batching
  (`MULTI_CROP_AND_RESIZE`) is a day-one requirement, not an
  optimization. **Refined by HAL-1..3 (below): that cost is dominated
  by USERPTR page mapping — the same ops on dma-buf buffers run
  10-15x faster (9.1 ms → 0.64 ms).** Buffer management, not batching
  alone, is the primary lever.

## Where each SDK helper lands

| SDK API (today) | Today | Hardware path | Blocking layer |
|---|---|---|---|
| `Frame.resize` / `crop` | numpy/cv2 CPU | DSP resize / crop_and_resize | platform RPC |
| inference preprocess (per-detection tiles) | not offered | DSP multi_crop_and_resize | platform RPC (D2) |
| `draw_detections` / on-stream overlay | CPU raster | DSP blend | platform RPC (P1) |
| synthetic frames → main stream | not offered | DSP convert_format + injection node | platform + HAL validation |

The HAL ops table itself is complete for all rows
(`hal_v2/include/dsp/hal_dsp.h:279-285` sync ops, `:302-337` async
jobs). No item below asks hal_v2 for a new op.

## HAL-layer work items (hal_v2 / vendor lib)

**HAL-1, HAL-2, HAL-3 were executed 2026-09-01** (probe mode `e4`,
192.168.93.72, daemons live; results below the table and in
[dsp-offload.md](dsp-offload.md#hal-validation-experiments-e4-2026-09-01)).
**HAL-4 and HAL-7 were fixed the same day** in the platform repo
(branch `fix/hal15-dsp-batch-storage`) and re-verified on device with
the same probe.

| # | Item | Why | Evidence / starting point |
|---|---|---|---|
| HAL-1 ✅ | **Characterize `multi_crop_and_resize` on hardware** | D2 makes batching the core RPC design bet; the P0 probe had measured single-op `resize` only. **Result: verified all-written envelope N≤128; per-rect marginal cost ~70-150 µs (dmabuf) / ~1.2-1.5 ms (userptr); N=260 silently truncates to 4 outputs with rc=0** | probe mode e4 (E4-A/E4-B) |
| HAL-2 ✅ | **Validate `blend`** — base NV12, overlays A420/ARGB only (alpha from the overlay's alpha channel, base modified in place) | P1 prerequisite for `ai-overlay-extended.md`. **Result: exact semantics verified** — alpha=0 passthrough (Δ±0.00), alpha=255 blends to pure-red luma (Δ−77.52, identical in both mem modes), outside region untouched, NV12 overlay rejected (rc −2801, matches docs) | probe E4-C |
| HAL-3 ✅ | **Validate the `HAL_MEM_DMABUF` path through hal_v2 ops** | app→daemon buffer handoff must be dma-buf. **Result: functional, bit-exact deterministic, and 10-15x faster than USERPTR. Two-side `DMA_BUF_IOCTL_SYNC` discipline required** (CPU-write → SYNC(WRITE) before DSP read; DSP-write → SYNC(READ) before CPU read) — without it, stale reads and phantom non-determinism | probe E4-A/C under `--mem dmabuf` |
| HAL-4 ✅ | Header hygiene: out-of-tree builds of the impl needed `-include` workarounds. **Only `<thread>` was actually missing** from `hailo15_dsp_priv.hpp` (`<mutex>`/`<condition_variable>`/`<atomic>`/`<queue>` were already included; the extra flags were cargo-cult from the probe). **Fixed 2026-09-01**: `<thread>` added to the priv header, `<chrono>`/`<cstdlib>`/`<vector>` now directly included by the impl — the probe compiles with just `-I include` | platform branch `fix/hal15-dsp-batch-storage`, commit `4c65a595` |
| HAL-7 ✅ | **hal_v2 multi-crop wrapper fixed before any platform use**: was fixed stack storage of 7 + `output_count` unclamped (OOB read inside the vendor lib for N>7). **Fixed 2026-09-01**: storage dynamically sized per call; batches above the new `HAL_DSP_MULTI_CROP_MAX_OUTPUTS` (128, `hal_dsp.h`) rejected with `HAL_ERR_INVALID_ARG` instead of truncated; blend's shared `static overlays_storage[50]` (thread-race between sync/async paths + silent 50 clamp) replaced with per-call storage. **Verified on 93.72**: HAL-path N=16/64 bit-exact in both mem modes (dmabuf N=64: 11.5 ms / 5 561 rects/s), N=129 rejected rc −2814, blend semantics unchanged (dY −77.52, NV12 overlay −2801) | platform branch `fix/hal15-dsp-batch-storage`, commit `4c65a595` |
| HAL-5 | *(optional, defense-in-depth)* second daemon handle with `dsp_set_priority(HIGH)` for platform jobs | the vendor singleton queue arbitrates **per handle priority**: a platform job submitted behind an app job still runs first (only the in-flight op is non-preemptible). Cheap because E1 proved multi-handle works. The daemon queue stays the primary arbiter | `send_command.cpp:42`, `dsp_set_priority` in `hailodsp_base.h:103` |
| HAL-6 | *(optional, P2)* expose utilization/stats via the `dsp_get_utilization` redeclaration pattern | quota tuning needs a feedback signal; 19 %/21 % numbers came from this path | probe TU redeclaration |

### HAL-1..3 measured results (2026-09-01, live load, 1080p→640×360 bilinear, 120 iters)

| Op | N | USERPTR mean | dma-buf mean | rects/s (dmabuf) |
|---|---|---|---|---|
| HAL multi_crop | 1 | 6.9 ms | **0.64 ms** | 1 560 ops/s |
| HAL multi_crop | 7 | 14.0 ms | **1.06 ms** | 6 598 rects/s |
| HAL multi_crop (post-HAL-7) | 16 | 30.1 ms | **3.26 ms** | 4 915 rects/s |
| HAL multi_crop (post-HAL-7) | 64 | 79.8 ms | **11.5 ms** | 5 561 rects/s |
| blend | 1 ov | 9.1 ms | **0.67 ms** | — |
| blend | 8 ov | 10.4 ms | **1.57 ms** | ~129 µs/ov marginal |
| vendor multi_crop (dmabuf src, userptr dsts) | 1 | 7.0 ms | 2.1 ms | src-side mapping alone ≈ 5 ms of the ~7 ms userptr cost |

The N=16/64 rows were measured through the HAL ops table after the
HAL-7 fix (2026-09-01, same load and settings); before the fix the HAL
path could not take N>7 at all. Marginal per-rect cost on dma-buf stays
in the ~150-175 µs band from N=2 to N=64 — the batch envelope holds
through the HAL wrapper.

Correctness: deterministic (maxd=0) in both mem modes once the
`DMA_BUF_IOCTL_SYNC` discipline is applied; blend deltas match
analytic values exactly.

**Batch envelope (E4-B):** N=1/7/16/64/128 all outputs written,
rc=0. **N=260: rc=0 but only outputs 0-3 written** (reproducible ×3 —
the vendor header's "max 260" does not hold on this firmware). The
run immediately after a truncated 260-crop job had *every* multi-crop
command rejected (`DSP_RUN_COMMAND_FAILED`, xrp driver logged
firmware error −6); it self-recovered on the next run without a
reboot. Consequences:

- the RPC must cap batch size at **64** (verified comfortably; 128
  also verified but doubles sync-RPC latency to ~150 ms under load),
- rc=0 alone is not proof a batch completed — the daemon must
  validate output counts it constructed itself, and never forward a
  client-supplied count above the cap. The HAL now enforces its own
  ceiling too: batches above `HAL_DSP_MULTI_CROP_MAX_OUTPUTS` (128)
  are rejected with `HAL_ERR_INVALID_ARG` instead of truncated
  (verified on device: N=129 → rc −2814, no DSP work).

## Platform-layer work items (camera-daemon)

**PLAT-1..5 were implemented and verified 2026-09-01** on
192.168.93.72 (platform branch `feat/dsp-service-p0`): the probe
(`/tmp/dsp_plat_probe`, gRPC control + UDS fd-passing client, both
cross-compiled against the daemon's own proto) passed **21/21** against
the live daemons — including two core dumps' worth of bisecting that
ended in a client-side pitfall, not a daemon defect (see
"Vendor limitation" below). End-to-end through the RPC path:
**4.7 ms/job for a 16-rect MULTI_CROP (3410 rects/s burst, 213 jobs/s)**,
quota enforcement observed live (58 accepted / 5 quota-rejected in a
tight client loop), and every cap verified from the hostile side
(33-buffer/66-fd alloc → rejected, batch 66 → rejected, released and
unknown buffer ids → rc −2, bad pixel format → rejected).

| # | Item | Detail |
|---|---|---|
| PLAT-1 ✅ | **`dsp_service` module owning the one HAL context** | implemented as a daemon-owned service; all app jobs ride the same context the encoder/DPM paths use (D1). Buffers come from dedicated CMA dma-heap pools keyed by geometry (`pool_max_buffers = 32` per geometry — a 32-buffer NV12 pool is exactly 64 fds, the SCM_RIGHTS ceiling the UDS path passes in one message, so a full pool can always be handed to one client atomically) | 
| PLAT-2 ✅ | **`SubmitDspJob` RPC, synchronous form first** | live on the UDS control socket `/run/aipc/camera-control.sock`; op/interpolation/scaling enums validated against the proto before HAL conversion (E3: async depth-4 bought +6 % — async moves to P2) |
| PLAT-3 ✅ | **`MULTI_CROP_AND_RESIZE` in P0, not later — batch cap 64** | verified from both sides: N=16 jobs pass with per-ROI content and gradient checks; `rects`/`dst_ids` above 64 rejected at validation with an explicit message; mismatched `dst_ids.size()` also rejected |
| PLAT-4 ✅ | **Scheduler: priority + quota + caps + timeout** | per-client quota **100 jobs/s and 120 MPix/s** (anchors from the dma-buf HAL-3 numbers) enforced under lock — the probe's unthrottled loop saw 58 ok / 5 quota_rej; 2000 ms job timeout via the HAL `wait(timeout_ms)`+cancel path |
| PLAT-5 ✅ | **`AllocateDspBuffer` + fd passing** | UDS `/run/aipc/camera.sock` flat messages `DSP_ALLOC` / `DSP_ALLOC_RESP` / `DSP_BUF_RELEASE` with SCM_RIGHTS (max 64 fds per message, enforced: 66 → rejected); per-plane stride/size returned; released and unknown ids correctly rejected (rc −2); `DMA_BUF_IOCTL_SYNC` write-fence after CPU fill verified as required for correct DSP reads |
| PLAT-6 ✅ | **Encoder-contention regression gate** | measured 2026-09-01 (black-box: frame count on `/run/aipc/encoded/main.sock`, DPM inactive so the encoder is the only other DSP consumer). **P1 (single client at quota-sustained 19.4 jobs/s × 60 s): encoder 30.02 fps vs 30.05 baseline, min-per-sec 30/30, zero gaps, zero drop warnings. P2 (8 parallel clients, ~160 jobs/s aggregate, 0 errors): 30.02 fps, one second at 29, max inter-frame 63.9 ms (< 2× median), zero gaps. Full recovery next window; daemon PID stable throughout.** P0 is cleared to open to apps — details below |

### PLAT-6 measurement (2026-09-01, 192.168.93.72)

The encoder's DSP usage lives inside the vendor medialib, invisible to
daemon instrumentation — so the gate is black-box: count encoded frames on
`/run/aipc/encoded/main.sock` (same 30-byte header protocol as the SDK's
`EncodedStreamClient`) while generated DSP load runs against `SubmitDspJob`.
The load generator replays the realistic inference-preprocess job
(MULTI_CROP_AND_RESIZE ×16, 480×270→512×512, BILINEAR, 1920×1080 NV12
src). DPM is inactive on the test device and no app containers were
running, making the encoder the only competing DSP consumer.

| Phase | DSP load | encoder fps (mean / min-per-sec) | max inter-frame | gaps > 2× median |
|---|---|---|---|---|
| baseline (20 s) | none | 30.05 / 30 | 40.9 ms | 0 |
| P1 (60 s) | 1 client, 19.4 jobs/s ok (4846 quota-rej, 0 err) | 30.02 / 30 | 48.1 ms | 0 |
| recovery | none | 30.00 / 29 | 43.2 ms | 0 |
| P2 (30 s) | 8 clients, ~160 jobs/s aggregate (0 err) | 30.02 / 29 | 63.9 ms | 0 |
| recovery (15 s) | none | 30.07 / 30 | 43.7 ms | 0 |

Acceptance (pre-declared): P1 fps ≥ 98 % of baseline mean, min-per-sec ≥
baseline min, zero "DSP contention / frame dropped" warnings, daemon PID
stable, recovery within 5 s — **all met**. The single 63.9 ms inter-frame
interval under 8-client saturation is one late frame (≈ 2 frame periods),
not a drop: no per-second count fell below 29 and no gap exceeded twice
the median interval. Quota math confirmed in the field: an N=16 job
charges 6.27 MPix, so each client sustained ~19-20 jobs/s of the
120 MPix/s budget regardless of its 100 jobs/s request rate — the quota,
not the DSP, is the binding constraint for well-behaved clients, which is
exactly the arbitration P0 wanted. Per-job latency under 8-client
saturation: p50 11.1 ms / p95 37 ms / max 55 ms (vs 4.8 / 7.0 / 11.8 ms
single-client) — queueing cost, borne entirely by the load clients.

Operational note for reproducing: parallel clients must use distinct dst
geometries (512/510/508/…), because the daemon-wide dst pool cap is 32
buffers per geometry — identical geometries contend at `DSP_ALLOC`, not at
the DSP.

### Vendor limitation discovered during verification

`MULTI_CROP_AND_RESIZE` **rejects NEAREST interpolation**
(`DSP_INTERP_NEAREST` = 0) with `DSP_INVALID_ARGUMENT` →
`HAL_ERR_RESULT` (−2801); only BILINEAR(1) and BICUBIC(3) are accepted
on the batched op (the vendor perf path gates on
`(interp & ~2) == 1`). Single-op `crop_and_resize` accepts NEAREST.
This bit the probe itself, not the daemon: a gRPC client that omits
`set_interpolation()` silently gets 0 = NEAREST and every multi-crop
job fails with −2801 — the exact signature that cost two core dumps to
bisect. Two consequences, both now in place:

- the probe (and any future SDK client) must always set
  `interpolation` explicitly on MULTI_CROP jobs;
- clients of the future `DspClient` should default multi-crop to
  BILINEAR, and the daemon-side validation should consider rejecting
  NEAREST-on-MULTI_CROP with a named error rather than letting the
  vendor code surface as a generic −2801.

The clean HAL also now logs the vendor `dsp_status` by name
(`DSP_INVALID_ARGUMENT`, `DSP_RUN_COMMAND_FAILED`, …) before collapsing
to `HAL_ERR_RESULT`, so this class of failure is diagnosable from
journalctl alone instead of requiring instrumentation.

Deployment note for reproducing on-device runs: the daemon loads
`libaipc_hal.so.2` via the soname symlink — after replacing
`libaipc_hal.so.2.0.0`, verify with
`grep libaipc_hal.so.2.0.0 /proc/$(pidof camera-daemon)/maps` that the
new file (md5) is actually mapped, not a stale inode held open.

## SDK-layer work items (this repo, for completeness)

| # | Item | Detail |
|---|---|---|
| SDK-1 | **Frame fd retention (opt-in)** ✅ done (2026-09-01) | `get_frame/subscribe(..., keep_fd=True)` returns a `FrameHandle` (dma-buf fds, per-plane strides/sizes, frame_id) with idempotent `release()`; `Frame.to_array()` lazily materializes, fenced with `DMA_BUF_IOCTL_SYNC`. Device-verified on 93.72 (4K NV12): 30/30 sustained retains at p50 23 ms; 2-frame × 2 s holds with live content; release after watchdog force-reclaim still unpins the window. **Constraints found:** retention budget ~4 s (FrameWatchdog warns ~4.2 s, force-reclaims buffers at ~5 s); `DMA_BUF_IOCTL_SYNC` returns EINVAL on these exporter fds (content coherent — SDK fences best-effort). **Three fd_publisher defects block fast releasers — ✅ fixed on platform 30a30bee same day** (see note below) |
| SDK-2 | **`DspClient` with CPU fallback** ✅ done (2026-09-01) | `resize_hw` / `crop_hw` / `multi_crop_hw` following the established cv2-optional pattern: hardware-first, numpy fallback when the RPC is absent — mirrors how `Frame.resize` degrades today. `neoruntime_ipc_sdk/dsp.py`: daemon-side buffers via the UDS `DSP_ALLOC`/`DSP_BUF_RELEASE` wire protocol (560-byte RESP parsed with strides/sizes/ids, up to 64 dma-buf fds per response), jobs via `SubmitDspJob` with **always-explicit interpolation** (proto default 0 = NEAREST → HAL −2801 on MULTI_CROP). Fallback triggers only on UNIMPLEMENTED or daemon error −5 (`last_used_hw` records which path served); genuine errors raise `DspError`. Hot loops can pre-allocate `DspBufferPool`s (`src_pool`/`dst_pool`/`dst_pools` params). Client-side validation mirrors daemon caps (dims [16,8192], ≤64 fds per alloc, ≤64 rects, NV12 evenness). 32 new tests. **Device-verified on 93.72 (live 4K NV12 frame, 2026-09-01): full probe ALL PASS — resize vs cv2 mean\|diff\| 1.38, crop exact copy bit-identical, crop+scale 0.37, multi-crop 4 tiles worst 0.37; paced hot loop 8/8 ok with 0 quota rejections.** Latency attribution: daemon-side job time **2.0 ms** (resize) / ~0 ms (multi-crop) vs **111 / 76 ms** SDK-side wall — >95 % is client-side data movement (UDS alloc roundtrips + 12.4 MB mmap src write + readback), and plain cv2 `INTER_AREA` on the Y plane costs ~17 ms, so **with a copy-in source the DSP path loses to CPU; it only wins once the source is zero-copy (SDK-3 / proposal below) or the CPU cores are busy**. Quota semantics (measured): per-client budget (100 jobs/s, 120 MPix/s, 1 s burst on first contact) charges each job `src + Σdst` MPix — a 4K job costs ~8.5 MPix, so back-to-back submissions hit `DspError` code −3 ("quota: MPix/s budget exhausted") within seconds; that error deliberately does **not** silently fall back (a CPU switch is a latency cliff the app should see — documented in the module docstring). ~~Constraint found (P0 contract): job src must be a daemon-allocated buffer, so the input array is copied in~~ **Resolved 2026-09-01: platform `DSP_IMPORT` (UDS msg 10, commit a94ee007) + SDK-side `Frame`/`FrameHandle` sources — the `*_hw` methods now accept a keep-fd frame directly and import its dma-bufs zero-copy (`JobSource = Union[ndarray, Frame, FrameHandle]`); device-verified ALL PASS, handle path 15 ms vs 54 ms copy-in on live 4K (see the gap section below)** |
| SDK-3 | **`Frame.resize` fast-path switch (P1)** ✅ done (2026-09-01) | opportunistic: route through `DspClient` when available, keep the current implementation as the fallback. `Frame._hw_resize` (media.py) fires when the frame carries an open dma-buf handle and the format is DSP-mappable (NV12/RGB/BGR/GRAY8), importing the frame zero-copy via the SDK-2 `JobSource` path; lazy `from .dsp import DspClient` inside the method avoids the media↔dsp import cycle; any `DspError` (socket absent, quota, old daemon) logs at debug and falls back to the existing CPU path — the convenience API must always work, unlike the explicit `*_hw` methods. **Geometry decision (device-found): every mode scales on the DSP into the exact box the CPU formulas compute, then pads/crops on the CPU** — the daemon letterbox pads `HalDspColor{}` = Y=U=V=0 (green-tinted in NV12, not a neutral pad), and the vendor `SCALE_AND_CROP` picks its own cover-scale rounding/centering, which disagreed with the CPU placement by **~21 mean\|diff\|** on a 16:9→4:3 crop of live 4K; with box-then-pad/crop both paths agree by construction (only the bilinear filter differs). `resize_hw`'s `scale_crop`/letterbox scalings remain available for users who want vendor semantics. Per-call `DspClient` setup is the price of the convenience API (docstring points hot loops at a persistent client). 8 new tests. **Device-verified on 93.72 (live 4K NV12): ALL PASS — letterbox 640×480 mean\|diff\| 1.10 vs CPU with correct 114/128 pads, stretch 1.44, crop 1.39; fast path leaves the frame un-materialized and the handle open; end-to-end from a keep-fd frame 29.6 ms vs 68.7 ms CPU (2.3×)** |

### fd_publisher defects found during SDK-1 verification (2026-09-01, daemon v2.0.0 on 93.72) — ✅ all fixed same day

All in `camera-daemon` (platform repo); repro evidence in
`journalctl -u camera-daemon` around 06:59:30 and 07:13:40 on 2026-09-01.
Fixed on platform branch `feat/dsp-service-p0`, commit `30a30bee`
("fix(camera-daemon): close fd_publisher dispatch races"), verified on
93.72 with a raw-socket probe: 4/4 pass — immediate-release 300/300,
churn ×40 with daemon active and NRestarts=0, window pin/resume
semantics intact, slow-client isolation. Journal window clean (no
SIGSEGV, no "unknown frame_id", no outstanding-leak warnings).

1. **Use-after-free → SIGSEGV** (crashed the daemon, restart counter 1):
   `FdPublisher::on_frame` collected `ClientState*` targets under
   `clients_mu_` (fd_publisher.cpp:113-120), then dereferenced them after
   unlocking (:132-141); a concurrent `disconnect_client` freed the object
   in between. Triggered by subscribe→read→release→disconnect churn.
   Stack: `FdPublisher::on_frame` ← `FrameRouter::dispatch_loop`.
   **Fix:** the whole dispatch pass now runs under `clients_mu_` (bounded,
   because sends are non-blocking) — a disconnecting client can no longer
   be freed or have its fd closed-and-recycled mid-iteration. This also
   closes a 4th defect found during the fix: `stream_name`/`subscribed`
   were written by handle_subscribe on the recv thread with no lock (torn
   `std::string` read on the dispatch thread = UB).
2. **RELEASE-before-track race → permanent delivery stall**: `on_frame`
   sent the frame (:147) *before* inserting it into `outstanding` (:150);
   a client RELEASE arriving in that gap was discarded as "unknown"
   (:372) and the slot pinned at `max_outstanding_per_client=3` →
   `frames_dropped` for that client until disconnect. Nondeterministic
   (load-dependent): copy mode's 4K memcpy (~10-30 ms) hides it;
   keep-fd's microsecond release hits it.
   **Fix:** `router_->retain(mf)` + `outstanding[frame_id] = mf` happen
   before sendmsg (an untracked entry is never visible; frame_ids are
   never reused); send failure rolls the entry back and releases the ref.
3. **Non-blocking send never enabled**: `accept_loop` read
   `fcntl(F_GETFL)` (:216) but never called `F_SETFL` with `O_NONBLOCK`,
   and `fd_pub_sendmsg` passed only `MSG_NOSIGNAL` (fd_protocol.h:154) —
   the dispatch thread could block on a slow client.
   **Fix:** frame sends use `MSG_DONTWAIT` per sendmsg (recv loops stay
   blocking — setting O_NONBLOCK on the fd would break their MSG_WAITALL
   framing). EAGAIN maps to "client too slow" (drop frame, keep client);
   a partial send sets errno=EMSGSIZE and drops the client, since with
   SCM_RIGHTS the fds cross with the first byte queued — a partial send
   has already leaked them and desynced the stream. Honest caveat: this
   one was *latent*, not reachable — `max_outstanding_per_client=3` caps
   in-flight data at ~240 B, so the socket buffer cannot fill through the
   frame path (release requires a read). The fix is defense-in-depth that
   makes the "dispatch must not block" invariant explicit.

Post-fix, fast releasers no longer need the ≥10 ms RELEASE delay; the
~4 s retention budget (FrameWatchdog) still applies. Remaining
observation for a future platform pass: `encoded_publisher` still does
blocking `send_all` on its dispatch path (journal shows 8 ms slow-send
warnings under load) — same defect class as #3 in a different
publisher.

### SDK-2 platform gap found during implementation: no zero-copy job source — ✅ RESOLVED 2026-09-01

> **Shipped**, but via a different mechanism than the proto sketch below:
> instead of a gRPC `src_import` field, the daemon takes a new UDS message
> `FD_PUB_MSG_DSP_IMPORT` (type 10, `<12I>` + plane fds via SCM_RIGHTS)
> that dup()s the client's dma-bufs and registers an import id in the
> same namespace as pool buffer ids — valid as `src_buffer_id`, freed with
> the existing `DSP_BUF_RELEASE`. Source-only, ≤64 imports per client,
> quota still charged. Platform commit a94ee007 (`feat/dsp-service-p0`).
> The proto sketch is kept for context; it was not implemented.

SDK side (task #32, same day): `DspClient.resize_hw/crop_hw/multi_crop_hw`
accept a keep-fd `Frame` or `FrameHandle` directly (`JobSource`); numpy
arrays still copy in. With a zero-copy source there is deliberately **no
CPU fallback** — the frame holds fds, not pixels, so DSP-unavailable
raises with a pointer to `frame.to_array()` instead of silently paying
the copy. Frame format metadata (e.g. `NV12`) outranks shape inference,
closing the 2D-array gray8/NV12 ambiguity for pixel frames too.
Device-verified on 93.72 (ALL PASS): resize/crop/multi-crop content vs
cv2 ≤1.6 mean\|diff\|, closed-handle reuse raises, and live-4K latency
**15 ms (handle) vs 54 ms (copy-in) vs ~17 ms plain cv2** — the zero-copy
path is the first configuration where the DSP beats CPU on an idle core.

Original gap, for the record:

`SubmitDspJob` requires `src_buffer_id` to resolve to a buffer the
daemon allocated through `DspService::alloc_buffers` (dsp_service.cpp
buffer table). Camera frames arrive as dma-bufs owned by the HAL/encoder
pool — an app's `FrameHandle` fds are not in that table, so today the
SDK **copies the input array into a daemon DSP buffer** (mmap write).
Measured on 93.72 (SDK-2 probe): a 4K-NV12 `resize_hw` call costs
~111 ms wall of which the DSP job itself is 2.0 ms — the copy-in path
(alloc roundtrips + 12.4 MB mmap write + readback) is >95 % of the
latency budget and makes the whole path slower than cv2 (~17 ms for
the same Y-plane resize). Zero-copy src is therefore not an
optimization nicety but the difference between the DSP path being
useful and being counterproductive for array sources.

Proposed platform extension (small, backward compatible):

```proto
message DspJobRequest {
  ...
  // Import an existing dma-buf as the job source instead of a
  // daemon-allocated src_buffer_id. Daemon takes a reference for the
  // job lifetime only.
  ImportedBuffer src_import = 8;
}
message ImportedBuffer {
  repeated uint32 plane_fds = 1;   // per-plane dma-bufs (app-owned)
  uint32 width = 2;
  uint32 height = 3;
  string format = 4;               // nv12 | rgb24 | gray8
  repeated uint32 strides = 5;
  repeated uint32 sizes = 6;
}
```

Daemon side: `dma_buf_import`/`mmap` the fds for the job duration, run,
drop the mapping. No allocation-table entry, no quota. Combined with
SDK-1 `keep_fd=True` this makes the whole path
`get_frame → multi_crop_hw → release` zero-copy end to end, and removes
the copy that currently dominates the e2e latency.

## Sequencing

```
HAL-1..3 ✅ done (2026-09-01, results above)
HAL-4 + HAL-7 ✅ done (2026-09-01, platform    ──► PLAT-1..5 ✅ done (2026-09-01, 21/21 on device,
      branch fix/hal15-dsp-batch-storage,              branch feat/dsp-service-p0)
      commit 4c65a595)                                 │
                                                         ├──► PLAT-6 ✅ (2026-09-01, encoder 30.02 fps
                                                         │     under 160 jobs/s, 0 drops)
                                                         ▼
                                                 SDK-1 ✅ (2026-09-01, keep-fd verified
                                                         │   on device; fd_publisher defects
                                                         │   fixed on platform 30a30bee)
                                                         ▼
                                                 SDK-2 ✅ (2026-09-01: DspClient
                                                         │   + DSP_IMPORT zero-copy
                                                         │   Frame sources, ALL PASS
                                                         │   on device)
                                                         ▼
                                                 SDK-3 ✅ (2026-09-01:
                                                         │   Frame.resize DSP
                                                         │   fast path, ALL PASS
                                                         │   on device — SDK
                                                         │   hardware chain done)
```

HAL-1's verified batch envelope (≤64 recommended) is folded into
PLAT-3's cap; HAL-3's dma-buf numbers set PLAT-4's quota anchors; the
PLAT-5 measurements (4.7 ms/job end-to-end through gRPC + fd passing,
vs 3.26 ms for the same N=16 batch called in-process through the HAL)
price the RPC layer itself at roughly 1.4 ms/job — acceptable for P0,
and amortized by larger batches. With PLAT-6 measured (encoder loses
nothing even at 8-client saturation), the platform side is done: what
remains is the SDK client work.

## Explicitly deferred

- Per-client HAL contexts — dropped (D1), revisit only if the vendor
  ships multi-queue DSP support.
- Async submit/wait RPC (P2), until a consumer needs caller-thread
  freedom.
- C++ SDK mirror of the toolkit — 0.6.1.
