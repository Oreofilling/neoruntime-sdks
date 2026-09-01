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

| # | Item | Why | Evidence / starting point |
|---|---|---|---|
| HAL-1 ✅ | **Characterize `multi_crop_and_resize` on hardware** | D2 makes batching the core RPC design bet; the P0 probe had measured single-op `resize` only. **Result: verified all-written envelope N≤128; per-rect marginal cost ~70-150 µs (dmabuf) / ~1.2-1.5 ms (userptr); N=260 silently truncates to 4 outputs with rc=0** | probe mode e4 (E4-A/E4-B) |
| HAL-2 ✅ | **Validate `blend`** — base NV12, overlays A420/ARGB only (alpha from the overlay's alpha channel, base modified in place) | P1 prerequisite for `ai-overlay-extended.md`. **Result: exact semantics verified** — alpha=0 passthrough (Δ±0.00), alpha=255 blends to pure-red luma (Δ−77.52, identical in both mem modes), outside region untouched, NV12 overlay rejected (rc −2801, matches docs) | probe E4-C |
| HAL-3 ✅ | **Validate the `HAL_MEM_DMABUF` path through hal_v2 ops** | app→daemon buffer handoff must be dma-buf. **Result: functional, bit-exact deterministic, and 10-15x faster than USERPTR. Two-side `DMA_BUF_IOCTL_SYNC` discipline required** (CPU-write → SYNC(WRITE) before DSP read; DSP-write → SYNC(READ) before CPU read) — without it, stale reads and phantom non-determinism | probe E4-A/C under `--mem dmabuf` |
| HAL-4 | Header hygiene: `hailo15_dsp_priv.hpp` lacks `<thread>`/`<mutex>`/`<condition_variable>`/`<atomic>` when compiled outside the CMake build | forced `-include` workarounds in the probe; out-of-tree tools (and any future SDK-side tooling) should compile with just `-I include` | probe build line in its header |
| HAL-7 **new** | **Fix the hal_v2 multi-crop wrapper before any platform use**: `hailo15_dsp_impl.cpp:249-290` stack-allocates `crop_params_storage[DSP_MULTI_RESIZE_OUTPUTS_COUNT]` (7) but passes `output_count` **unclamped** → N>7 through the HAL ops table reads out of bounds inside the vendor lib | batched RPC wants N up to 64+; the HAL path must clamp or dynamically size. (The probe capped E4-A at N=7 for exactly this reason) | `platforms/hailo15/dsp/hailo15_dsp_impl.cpp:249-290` |
| HAL-5 | *(optional, defense-in-depth)* second daemon handle with `dsp_set_priority(HIGH)` for platform jobs | the vendor singleton queue arbitrates **per handle priority**: a platform job submitted behind an app job still runs first (only the in-flight op is non-preemptible). Cheap because E1 proved multi-handle works. The daemon queue stays the primary arbiter | `send_command.cpp:42`, `dsp_set_priority` in `hailodsp_base.h:103` |
| HAL-6 | *(optional, P2)* expose utilization/stats via the `dsp_get_utilization` redeclaration pattern | quota tuning needs a feedback signal; 19 %/21 % numbers came from this path | probe TU redeclaration |

### HAL-1..3 measured results (2026-09-01, live load, 1080p→640×360 bilinear, 120 iters)

| Op | N | USERPTR mean | dma-buf mean | rects/s (dmabuf) |
|---|---|---|---|---|
| HAL multi_crop | 1 | 6.9 ms | **0.64 ms** | 1 560 ops/s |
| HAL multi_crop | 7 | 14.0 ms | **1.06 ms** | 6 598 rects/s |
| blend | 1 ov | 9.1 ms | **0.67 ms** | — |
| blend | 8 ov | 10.4 ms | **1.57 ms** | ~129 µs/ov marginal |
| vendor multi_crop (dmabuf src, userptr dsts) | 1 | 7.0 ms | 2.1 ms | src-side mapping alone ≈ 5 ms of the ~7 ms userptr cost |

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
  client-supplied count above the cap.

## Platform-layer work items (camera-daemon)

| # | Item | Detail |
|---|---|---|
| PLAT-1 | **`dsp_service` module owning the one HAL context** | extend the existing init path (`dpm_worker.cpp:473`) into a service that serializes *all* DSP work — encoder pre-scale, DPM, OSD, app jobs — through one daemon queue (D1). No per-client `init` ever |
| PLAT-2 | **`SubmitDspJob` RPC, synchronous form first** | E3: async depth-4 bought +6 % (100.2→106.4 jobs/s, Little's-law-consistent single server) — async submit/wait moves to P2 with a real pipelining consumer |
| PLAT-3 | **`MULTI_CROP_AND_RESIZE` in P0, not later — batch cap 64** | D2: the RPC contract must make the batched op at least as easy as the single op (repeated `rects`), or every app will submit per-tile and pay the per-op cost each. Cap anchored to measurement: N≤128 verified all-written, N=260 silently truncates (HAL-1) — reject `rects > 64` at validation |
| PLAT-4 | **Scheduler: priority + quota + caps + timeout** | platform jobs preempt app jobs; per-app jobs/s and MPix/s budgets; max-pixel cap at validation; job timeout via the existing HAL `wait(timeout_ms)`+cancel. Quota anchors from measurement — **dma-buf figures now apply** (HAL-3): single-op resize ~1 500 ops/s, multi-crop N=7 ~6 500 rects/s → suggested starting quota **~100 jobs/s and ~120 MPix/s per app** (was ~30/60 on the userptr assumption); keep the encoder-drop soak (PLAT-6) as the real gate before raising |
| PLAT-5 | **`AllocateDspBuffer` + fd passing — now the primary performance item** | dma-buf fds over gRPC, same fd-passing pattern the raw-media UDS path already uses platform-wide; alloc via CMA dma-heap. HAL-3 measured the same ops 10-15x faster on dma-buf (the userptr page-mapping cost *was* the P0 "~7-10 ms submission path") — so fd passing is not just zero-copy hygiene, it is what makes per-frame DSP use viable. Contract must include the `DMA_BUF_IOCTL_SYNC` discipline (write-fence after any CPU fill, read-fence before any CPU read) |
| PLAT-6 | **Encoder-contention regression gate** | the existing drop path (`dpm_worker.cpp:815-817`, "DSP contention with the encoder") must be measured under app load before opening the RPC: E2b proved coexistence *safe* (1093 ops, 0 errors) but did not measure encoder frame drops — that is the acceptance metric for P0 |

## SDK-layer work items (this repo, for completeness)

| # | Item | Detail |
|---|---|---|
| SDK-1 | **Frame fd retention (opt-in)** | `_recv_frame` today mmaps the received dma-buf, copies to numpy, and closes the fd (`media.py:663-669`) — `Frame` holds no fd, so no zero-copy path exists. Add a keep-fd mode carrying fd + per-plane stride/size; with the sync RPC, lifetime is trivial: hold the frame's release until `SubmitDspJob` returns. HAL-3 adds a requirement: any CPU access to a retained dma-buf must be fenced with `DMA_BUF_IOCTL_SYNC` (read-direction before numpy copies — today's mmap+copy path should adopt it too) |
| SDK-2 | **`DspClient` with CPU fallback** | `resize_hw` / `crop_hw` / `multi_crop_hw` following the established cv2-optional pattern: hardware-first, numpy fallback when the RPC is absent — mirrors how `Frame.resize` degrades today |
| SDK-3 | **`Frame.resize` fast-path switch (P1)** | opportunistic: route through `DspClient` when available, keep the current implementation as the fallback |

## Sequencing

```
HAL-1..3 ✅ done (2026-09-01, results above)
HAL-4 (header include fix, trivial)          ──┐
HAL-7 (multi-crop wrapper clamp, required    ──┤
      before the daemon routes batches        │
      through the HAL path)                   ├─► PLAT-1..5 (dsp_service + RPC + scheduler)
                                              │        │
                                              │        ▼
                                              │   SDK-1..2 (fd retention + DspClient)
                                              │        │
                                              └──► PLAT-6 encoder soak gate ──► open P0 to apps
```

HAL-1's verified batch envelope (≤64 recommended) is already folded
into PLAT-3's cap; HAL-3's dma-buf numbers set PLAT-4's quota anchors.
Nothing else blocks on HAL items.

## Explicitly deferred

- Per-client HAL contexts — dropped (D1), revisit only if the vendor
  ships multi-queue DSP support.
- Async submit/wait RPC (P2), until a consumer needs caller-thread
  freedom.
- C++ SDK mirror of the toolkit — 0.6.1.
