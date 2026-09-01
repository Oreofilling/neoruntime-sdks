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
| PLAT-6 | **Encoder-contention regression gate** | the existing drop path (`dpm_worker.cpp:815-817`, "DSP contention with the encoder") must be measured under app load before opening the RPC: E2b proved coexistence *safe* (1093 ops, 0 errors) but did not measure encoder frame drops — that is the acceptance metric for P0. **Not yet run — the one remaining platform item.** |

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
| SDK-1 | **Frame fd retention (opt-in)** | `_recv_frame` today mmaps the received dma-buf, copies to numpy, and closes the fd (`media.py:663-669`) — `Frame` holds no fd, so no zero-copy path exists. Add a keep-fd mode carrying fd + per-plane stride/size; with the sync RPC, lifetime is trivial: hold the frame's release until `SubmitDspJob` returns. HAL-3 adds a requirement: any CPU access to a retained dma-buf must be fenced with `DMA_BUF_IOCTL_SYNC` (read-direction before numpy copies — today's mmap+copy path should adopt it too) |
| SDK-2 | **`DspClient` with CPU fallback** | `resize_hw` / `crop_hw` / `multi_crop_hw` following the established cv2-optional pattern: hardware-first, numpy fallback when the RPC is absent — mirrors how `Frame.resize` degrades today |
| SDK-3 | **`Frame.resize` fast-path switch (P1)** | opportunistic: route through `DspClient` when available, keep the current implementation as the fallback |

## Sequencing

```
HAL-1..3 ✅ done (2026-09-01, results above)
HAL-4 + HAL-7 ✅ done (2026-09-01, platform    ──► PLAT-1..5 ✅ done (2026-09-01, 21/21 on device,
      branch fix/hal15-dsp-batch-storage,              branch feat/dsp-service-p0)
      commit 4c65a595)                                 │
                                                         ├──► PLAT-6 encoder soak gate ──► open P0 to apps
                                                         ▼
                                                 SDK-1..2 (fd retention + DspClient)
```

HAL-1's verified batch envelope (≤64 recommended) is folded into
PLAT-3's cap; HAL-3's dma-buf numbers set PLAT-4's quota anchors; the
PLAT-5 measurements (4.7 ms/job end-to-end through gRPC + fd passing,
vs 3.26 ms for the same N=16 batch called in-process through the HAL)
price the RPC layer itself at roughly 1.4 ms/job — acceptable for P0,
and amortized by larger batches. The remaining blockers are PLAT-6
(encoder soak) and the SDK client work; nothing blocks on HAL items.

## Explicitly deferred

- Per-client HAL contexts — dropped (D1), revisit only if the vendor
  ships multi-queue DSP support.
- Async submit/wait RPC (P2), until a consumer needs caller-thread
  freedom.
- C++ SDK mirror of the toolkit — 0.6.1.
