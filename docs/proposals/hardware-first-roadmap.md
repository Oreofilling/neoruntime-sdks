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
  optimization.

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

| # | Item | Why | Evidence / starting point |
|---|---|---|---|
| HAL-1 | **Characterize `multi_crop_and_resize` on hardware** — per-rect marginal cost with N=1/4/16 rects, 1080p source | D2 makes batching the core RPC design bet, but the probe measured single-op `resize` only (7.1 ms mean). If per-rect marginal cost is ~µs-scale, one multi-crop job replaces N×7 ms submissions — that number decides the quota defaults | extend archived `dsp_p0_probe.cpp` (mode e4) |
| HAL-2 | **Validate `blend`** (alpha, offset, NV12-on-NV12) | P1 prerequisite for `ai-overlay-extended.md`; op never exercised on this device | same probe extension |
| HAL-3 | **Validate the `HAL_MEM_DMABUF` path through hal_v2 ops** | app→daemon buffer handoff must be dma-buf (USERPTR malloc worked in-process in the probe, but pointers do not cross processes) | probe already contains working CMA dma-heap alloc (`dsp_p0_probe.cpp:130`) |
| HAL-4 | Header hygiene: `hailo15_dsp_priv.hpp` lacks `<thread>`/`<mutex>`/`<condition_variable>`/`<atomic>` when compiled outside the CMake build | forced `-include` workarounds in the probe; out-of-tree tools (and any future SDK-side tooling) should compile with just `-I include` | probe build line in its header |
| HAL-5 | *(optional, defense-in-depth)* second daemon handle with `dsp_set_priority(HIGH)` for platform jobs | the vendor singleton queue arbitrates **per handle priority**: a platform job submitted behind an app job still runs first (only the in-flight op is non-preemptible). Cheap because E1 proved multi-handle works. The daemon queue stays the primary arbiter | `send_command.cpp:42`, `dsp_set_priority` in `hailodsp_base.h:103` |
| HAL-6 | *(optional, P2)* expose utilization/stats via the `dsp_get_utilization` redeclaration pattern | quota tuning needs a feedback signal; 19 %/21 % numbers came from this path | probe TU redeclaration |

HAL-1..HAL-3 are validation work, not development — one probe
extension, a few device hours.

## Platform-layer work items (camera-daemon)

| # | Item | Detail |
|---|---|---|
| PLAT-1 | **`dsp_service` module owning the one HAL context** | extend the existing init path (`dpm_worker.cpp:473`) into a service that serializes *all* DSP work — encoder pre-scale, DPM, OSD, app jobs — through one daemon queue (D1). No per-client `init` ever |
| PLAT-2 | **`SubmitDspJob` RPC, synchronous form first** | E3: async depth-4 bought +6 % (100.2→106.4 jobs/s, Little's-law-consistent single server) — async submit/wait moves to P2 with a real pipelining consumer |
| PLAT-3 | **`MULTI_CROP_AND_RESIZE` in P0, not later** | D2: the RPC contract must make the batched op at least as easy as the single op (repeated `rects`), or every app will submit per-tile and pay 7 ms each |
| PLAT-4 | **Scheduler: priority + quota + caps + timeout** | platform jobs preempt app jobs; per-app jobs/s and MPix/s budgets; max-pixel cap at validation; job timeout via the existing HAL `wait(timeout_ms)`+cancel. Quota anchors from measurement: contended floor ≈ 140 ops/s at 1080p→360p (~225 MPix/s source) with 19 % util → suggested starting quota **~30 jobs/s and ~60 MPix/s per app** (2 apps + daemon within floor with headroom) |
| PLAT-5 | **`AllocateDspBuffer` + fd passing** | dma-buf fds over gRPC, same fd-passing pattern the raw-media UDS path already uses platform-wide; alloc via CMA dma-heap (HAL-3). Buffers must be DSP-legal (stride/format) by construction |
| PLAT-6 | **Encoder-contention regression gate** | the existing drop path (`dpm_worker.cpp:815-817`, "DSP contention with the encoder") must be measured under app load before opening the RPC: E2b proved coexistence *safe* (1093 ops, 0 errors) but did not measure encoder frame drops — that is the acceptance metric for P0 |

## SDK-layer work items (this repo, for completeness)

| # | Item | Detail |
|---|---|---|
| SDK-1 | **Frame fd retention (opt-in)** | `_recv_frame` today mmaps the received dma-buf, copies to numpy, and closes the fd (`media.py:663-669`) — `Frame` holds no fd, so no zero-copy path exists. Add a keep-fd mode carrying fd + per-plane stride/size; with the sync RPC, lifetime is trivial: hold the frame's release until `SubmitDspJob` returns |
| SDK-2 | **`DspClient` with CPU fallback** | `resize_hw` / `crop_hw` / `multi_crop_hw` following the established cv2-optional pattern: hardware-first, numpy fallback when the RPC is absent — mirrors how `Frame.resize` degrades today |
| SDK-3 | **`Frame.resize` fast-path switch (P1)** | opportunistic: route through `DspClient` when available, keep the current implementation as the fallback |

## Sequencing

```
HAL-1..3 (probe extension, device hours)   ──┐
HAL-4 (header include fix, trivial)          ├─► PLAT-1..5 (dsp_service + RPC + scheduler)
                                             │        │
                                             │        ▼
                                             │   SDK-1..2 (fd retention + DspClient)
                                             │        │
                                             └──► PLAT-6 encoder soak gate ──► open P0 to apps
```

HAL-1's multi-crop measurement feeds PLAT-4's quota defaults; nothing
else blocks on HAL items.

## Explicitly deferred

- Per-client HAL contexts — dropped (D1), revisit only if the vendor
  ships multi-queue DSP support.
- Async submit/wait RPC (P2), until a consumer needs caller-thread
  freedom.
- C++ SDK mirror of the toolkit — 0.6.1.
