# Proposal: Composable Video Pipeline — Four-Step Contracts

Status: analysis approved 2026-09-10 — this is the umbrella roadmap the
other proposals hang off; no code yet for the items marked gap.
Scope: platform (camera-daemon, ai-runtime, device-control) + SDK
Position in the set: `dsp-offload.md` shipped the frame/DSP substrate;
`ai-overlay-extended.md` and `frame-injection.md` are the two halves of
the draw/output contracts below. This doc states the shared contracts
all of them must obey, assesses where each stands today, and gives the
priority order.

## Verdict

The direction — a composable pipeline (capture → infer → draw → output)
where the platform owns scheduling/buffering/backpressure, the SDK
offers both a whole-pipeline facade and per-step interfaces, and both
share one set of contracts — is correct, and closer to done than the
self-assessment suggested. Two pieces already exist in essence:

1. **Live-preview mode (design point ①, half) is implemented.** The
   overlay bake moved to the frontend site (`handle_video_frame_for_routing`,
   commit 7cfc9526): pixels drawn before the encoder's `add_buffer`, in
   both auto-feed and manual mode. `apply_overlay` no-ops in O(1) when
   no fresh result matches the stream; `RESULT_TTL = 500ms`
   (ai_overlay_subscriber.cpp:95) is the "freshest result within its
   validity window" semantics; `stream_map` models the clean/dirty split.
   On the SDK side `OverlayClient.annotate_result()` (overlay.py:239)
   closes the loop: results → event-bus → daemon bake → encoded stream.
   **The draw-instruction path is usable end to end today.**
2. **The DSP chain already prototypes the "single-step compose" frame
   contract.** `FrameHandle` (dma-buf fds + frame_id + per-plane
   strides, idempotent release, ~4s retention + watchdog reclaim) +
   SCM_RIGHTS on the camera.sock UDS + DSP_IMPORT registered ids
   (daemon `dup()`s client dma-bufs into its own namespace) is exactly
   the "verifiable buffer handle" model design point ③ asks for.

So the work is not a new pipeline: it is contract closure on four
steps, an output write path, and making the two modes explicit.

## Four-step contract status

| Step | Present (evidence) | Gap | Contract hardening (agreed 2026-09-10) |
|---|---|---|---|
| Capture | `FrameHandle` (frame.py:225); zero-copy `FdMediaClient` keep_fd; `EncodedStreamClient` (keyframe/codec flags); DSP_IMPORT 15ms vs copy-in 54ms; frame association (frame_sequence); layout (per-plane stride/size) | A frame handle cannot enter inference — `InferenceClient.infer()` takes ndarray→bytes only (inference.py:203); proto `Tensor.dma_fd` is a bare int32 | **Quota & lease in the contract**: ≤3 unreturned frames, 200ms lease (the ~4s watchdog is a backstop, not the contract); **processed flag**: frame metadata states whether overlay/DPM already baked this frame (today `stream_map` is config-time discipline, not runtime truth) |
| Infer | Infer (timeout_ms / priority 0-7 / session_id) + infer_async + ordered InferBatch + StreamInfer (frame_sequence + timestamp_ns + fps_limit); SessionConfig (max_qps / max_concurrent); SDK subscribe drop-oldest queue + consecutive-failure breaker | No explicit Cancel (timeout / DestroySession only); **streaming inference has no cancellation**; `Tensor.dma_fd` bare fd across gRPC (red line); InferBatch blocks on the whole batch; hw_infer_time_us = 0 for some model families; GetStats 500ms blocking sample window | **Skew observability**: expose result.timestamp_ns − frame.timestamp_ns so "boxes lag the picture by N frames" becomes a measured metric (the acceptance yardstick for both modes) |
| Draw | Instruction path v1 closed: OverlayClient configure/apply + annotate/annotate_result → AiOverlaySubscriber → bake site; face blur/mosaic; SDK draw.py client-side fallback; "may the source be modified" = today's in-place bake semantics | v2 draft only; RESULT_TTL hardcoded 500ms; **coordinate space undocumented** (normalized bboxes, per-input-resolution scaling, v2 polygon system scattered across code and drafts) | **Target contract states "compose in the output branch; source frames immutable"**: today's bake mutates the shared pipeline buffer in place (zero-copy subscribers see baked pixels) — transitional, mitigated by `stream_map`; the contract doc must declare both states |
| Output | Read side complete: EncodedStreamClient (format/timestamp/keyframe flags), TsWriter/HlsWriter/PrerollBuffer; EOS/IDR atomic restore already drafted for injection | **Write side zero**: PushFrame draft only; encoded_publisher still blocking send_all. 2026-09-10 probe: `/dev/video10` is the only OUTPUT node and takes **12-bit Bayer only** (ISP-front virtual sensor) — no NV12-accepting OUTPUT node exists, so injection rides the daemon's encoder feed, not a HW node | **Encoded packet sequence numbers** (loss/skip currently undetectable); **drop counters at every level** (daemon bake skips, injection queue full, publisher send failures, SDK client drops — four places, no unified view) |

## The three design points

**① Live preview vs strict frame-lock.** Live preview: done (bake site
+ TTL + O(1) no-op), matching the definition "boxes and picture not
strictly same frame". Strict mode: **absent** — StreamInfer is a
latest-frame daemon-fed loop, the bake does not wait for results, the
encoder does not wait for anything. "Frame N waits for its own
inference before output" needs a new path (gated emit + backlog cap +
overflow policy). Verdict: sound; live preview stays the default,
strict mode is opt-in, SDK hides scheduling either way.

**② Instructions vs whole frames.** Correct, and the cost asymmetry is
evidence-backed: the instruction path is cheap and closed; v2 is a
renderer extension. Whole-frame push is the hard gap; the 2026-09-10
probe ruled out a dedicated HW injection node (no NV12 OUTPUT node
exists), which simplifies P0 to encoder-feed substitution at the bake
site, and the phasing in `frame-injection.md` (P0 REPLACE/NV12
sub-stream → P1 OVERLAY/DSP blend → P2 streaming) is
right, including shallow queue 2-3, never backpressure the encoder,
EOS restores the pure ISP path at the next IDR. Verdict: strengthen
the instruction path first; injection is the escape hatch.

**③ One contract across processes.** The exemplar exists
(camera.sock SCM_RIGHTS ≤64 fds + DSP_IMPORT ids). **Red line:
`Tensor.dma_fd` (inference.proto, Tensor field 4) is a bare int32**
and grpc_service.cpp:546 passes it straight through to HAL
(`ht.dma_fd = pb_t.dma_fd()`). Treating one process's fd number as a
valid handle in another process is exactly the named anti-pattern; it
has not bitten only because SDK infer() sends bytes. Also: SDK
`pipeline.py` (client-side pre→infer→post compose) proves the facade
is feasible, but it is a synchronous app-thread convenience layer, not
a platform-scheduled pipeline — the two must be documented as
different things to avoid concept collision.

## Priority order

### P0 — output contract + frame-contract closure (minimum for "pipeline exists")
1. **PushFrame P0** (this doc's sibling `frame-injection.md`):
   REPLACE/NV12 at sub-stream resolution via **encoder-feed insertion
   at the bake site** — the 2026-09-10 probe found no NV12-accepting
   OUTPUT node (`/dev/video10` is Bayer-only, ISP-front virtual
   sensor), so the daemon substitutes the app frame at `add_buffer`;
   buffer arrives over camera.sock SCM_RIGHTS, never as a proto fd
   number. Single allow-listed app. Accept: test pattern visible in
   the web console, encoder bitrate/fps unchanged.
   ✅ shipped 2026-09-10, both sides. Daemon: 21/21 rig checks
   (gate/dims/owner rejections, 270/270 consecutive replaces,
   EOS→IDR restore, fps held) — details in the sibling doc's status
   block. SDK output single-step interface: `FramePublisher` (DSP
   buffer pool + `publish`/`publish_eos`/`close`) +
   `CameraClient.push_frame`/`injection_status`/`stop_injection`,
   docs zh/en; rig E2E 40/40 accepted, 0 dropped, 38 injected frames
   verified on decode, and the REPLACE cadence measured — one
   injection replaces exactly one encoded frame.
2. **Fix the dma_fd red line**: tensor frame input moves to SCM_RIGHTS
   on the camera.sock pattern or reuses DSP_IMPORT ids; SDK gains
   `infer(frame: FrameHandle)`. Zero-copy inference then satisfies ③.
3. **This document + the frame contract written down**: quota/lease
   semantics, coordinate-space spec, the draw target state ("output
   branch composes; sources immutable") with the transitional state
   called out, skew defined.
4. **AiOverlay v2 minimal subset**: ✅ shipped 2026-09-10. Wire:
   `annotate_result(..., polygons=..., track_id=..., result_ttl_ms=...)`
   (SDK) → event bus → daemon sidecar channel (`parse_overlay_polygons`,
   independent of detections; an empty array clears it), track id drawn
   as a " #<id>" label suffix, and the TTL chain
   `resolve_result_ttl_ms`: per-result ttl > per-stream override >
   global > round(2000/fps) (floor 1ms) > 500ms fallback. Verified on
   the rig: detection box + label "person #7 0.93" and the red "ZONE"
   triangle polygon all land at their expected normalized coords
   (box 3868 px/frame, stable); TTL — fps-derived 67ms @30fps shows
   2–3-frame bursts at a 5Hz publish cadence, per-result ttl_ms=800
   yields exactly 24 frames spanning +23→+788ms after the publish,
   and inter-phase gaps render clean (results expire, nothing
   lingers). Fixed en route: per-stream draw configs were
   zero-initialized (`draw_detections=false` → the HAL unified draw
   silently skipped the whole detection branch); `compose_draw_config`
   now seeds every field from `hal_draw_config_init_default` and
   applies only the per-frame overrides (regression-guarded).
5. **subscribe -2814 dead path**: ✅ cleared 2026-09-10. Root cause was
   ai-runtime-side, not HAL: StreamInfer built `data=nullptr` + raw
   dma_fd input tensors, and HAL v2 accepts CPU-pointer inputs only
   (`!in.data` → HAL_ERR_INVALID_ARG -2814; input dma_fds are never
   read). Fix: StreamInfer repacks the daemon-fed frame into a tight
   NV12 CPU buffer and binds one CPU tensor — byte-identical to what
   the Infer buffer_id path builds; the buffer is held by the
   on_complete lambda. Verified on the rig: geometry-matched subscribe
   delivers results, the P0-2 gate still passes (no perf regression),
   no fd leak. Documented boundary: model input geometry must equal
   stream geometry for subscribe (mismatch → per-frame
   HAL_ERR_INVALID_SIZE -2811 → SDK circuit-breaks after 10 frames).

### P1 — strict mode + scheduling quality + contract hardening in code
6. **Strict frame-lock contract**: ✅ shipped 2026-09-10. Strict mode is
   an overlay option, not a new RPC: `overlay.configure(
   strict_frame_lock=True[, strict_wait_cap_ms])` (optional-field
   semantics — omitted args keep the daemon's current setting). The
   bake site reorders dispatch and draw for identity-fed displays
   (`stream_map[D] == D`): the frame is routed to ai-runtime FIRST
   (gating ahead of dispatch would wait for a result that can never
   arrive), then blocks — bounded by the cap — for the frame's OWN
   result, then draws and encodes. Cap chain: configured (clamped
   [1,500]) > round(2000/fps) two frame periods > 66ms fallback.
   Verdicts per frame: LOCK_FRESH (stored seq == frame seq, TTL check
   skipped — the age is the inference latency), LOCK_NEWEST (wait
   overtook), DEGRADE (cap expired or predicted hopeless → preview
   semantics: freshest TTL-valid result; WARN rate-limited 1/s +
   counters), SKIP (no unexpired result → ship clean, O(1)). The wait
   predictor admits a bounded wait only when the next result is
   expected within cap/2 (EMA of result inter-arrival; a ≥2s gap
   resets it) — when results are several caps apart it declines to
   wait at all, so encode cadence wins over lock rate by design.
   Manual encoder feed disables strict (no bridge-ordering guarantee;
   falls back to preview with a one-shot WARN).
   Rig-verified (640x384@15 identity stream, geometry-matched
   yolov8n, results ≈35ms after capture): healthy (fps_limit 30,
   limiter idle) locked ≈97% of gated frames, avg bounded wait
   34.9ms ≈ inference latency, encode fps 15.12 held → output latency
   = inference latency + 1 frame; rate-matched (fps_limit 15 == fps):
   84% locked, ~16% degrade — fps_limit is an at-most cap measured
   against the last dispatch, so boundary jitter skips a few percent
   and those frames degrade by design; broken (fps_limit 5): DEGRADE
   WARNs ~1/s with monotonically rising degraded/skipped counters,
   lock rate 0 (predictor declines: next result ~0.8 cap out), encode
   fps 15.06 held — the wait never stalls the encoder. Two interop
   root causes fixed en route, both deployment gotchas: (a)
   `register_model` must pass `model_type` (e.g. "detection") or
   RegisterModel creates no postprocess session and StreamInfer
   returns hollow responses — no post_result, so the event-bus
   publish (the gate's input) never fires; (b) the daemon's
   primary-result discriminator now also accepts `"frame_sequence":`
   because the publisher omits `"detections":[]` when N=0 — without
   it an object-free scene could never lock (empty results still
   replace the stored one: frame truth is "nothing to draw").
   New RPC / explicit backlog cap were not needed: the shared HAL
   frame counter already identities frame↔result, and the cap +
   predictor + TTL chain bound the backlog.
7. **Frame contract hardening in code**: ✅ shipped 2026-09-10. Two halves.
   (a) Quota/lease enforcement at dispatch: `FdPublisher::on_frame`
   checks each subscribing client under the outstanding-map lock — over
   quota (`held ≥ max_outstanding_per_client`, default 3) or oldest-hold
   older than `lease_ms` (default 200ms) → the frame is NOT lent (the
   router's reference is simply not retained for that client; no
   silent waiting, no mid-flight revocation) and a rate-limited (1/s
   per client) WARN names fd, stream, held count, oldest age and the
   live `quota_rejected` / `lease_rejected` counters. The contract
   truth it encodes: within the lease grace window dispatch is still
   allowed (a get right after the sleep can return an already-buffered
   grace-window frame — that is dispatch-time enforcement, not
   revocation); the negative test drains that residue before asserting
   refusal. (b) Baked flag scoped to the bake set: `Frame.flags` bit 0
   = OVERLAY_BAKED, bit 1 = DPM_BAKED (wire: the FdPubFrame tail
   padding word; old daemons send 0, old SDKs ignore it). OVERLAY is
   set only when the stream is an overlay bake target — a stream_map
   VALUE (identity `D→D` and cross-fed `I→D` both count; a key mapped
   to a foreign display is an inference source, the clean feed) — so
   flags always match the stream_map split. DPM stays global (it draws
   on every stream). Fixed en route: `apply_overlay`'s result matching
   took the direct `results_[stream]` hit for source-only keys too,
   which baked a stream's own inference results into the feed that
   exists to be clean — the bake-target gate now precedes the lookup
   (flag truth and pixel truth share one predicate:
   `AiOverlaySubscriber::is_bake_target`). SDK: `FRAME_FLAG_*`
   constants, `Frame.flags/baked_overlay/baked_dpm`,
   `FrameHandle.flags` (wire round-trip + old-daemon-zero covered by
   tests). Rig-verified (stream_map `third:third,sub:main`): quota —
   4th get blocked the full 2s window while 3 held, resumes after
   release; lease — hold 1 frame 600ms, grace residue (2 buffered
   frames) drained and released, further gets refused, resumes after
   release; journal shows both reject lines with climbing counters
   (`quota_rejected`, `lease_rejected`, per-fd/stream attribution).
   Flags — third/main sampled 0x1, sub sampled 0x0, stable across 10
   samples each, DPM bit clear (privacy mask off). Acceptance script:
   19/19 checks PASS.
8. **Infer observability & cancel**: ✅ shipped 2026-09-10. One metric,
   both modes. Server (ai-runtime): `skew_us` = result-ready − frame
   capture, stamped in the StreamInfer on_complete success tail
   (rc==0 && postproc ok); both ends on the device CLOCK_MONOTONIC
   domain (the same domain as `frame.timestamp_ns`), so the number is
   immune to host/device clock offset — unlike the SDK-side latency EMA,
   whose wall-vs-monotonic mix rejects sane samples on offset clocks
   (pre-existing, intentionally left as-is; existing tests contract it).
   skew is recorded only when positive; 0 means "not measured" (failed
   inference or older server). Wire: `StreamInferResponse.skew_us`
   (int64); GetStats aggregates `InferenceStats.avg_skew_us /
   max_skew_us / skew_samples`. Cancel: three server-side gaps closed —
   post-unsubscribe buffered `latest_frame` is released under
   `frame_mu` (no leak/double-release); the future wait is sliced into
   50ms chunks that check `ctx->IsCancelled()` and break; drain
   deadline drops to 1s once cancelled (5s normally). SDK:
   `InferenceResult.skew_us`; `_SubscribeIterator.last_skew_us` +
   `avg_skew_us` (EMA, server-computed input); `cancel()` is
   cross-thread and idempotent (cancel-hook cancels the pump future and
   offers the sentinel; no-op before first `next()`), propagates into
   the asyncio pump whose except-path cancels the gRPC call, so the
   server actually observes IsCancelled; `get_stats()` rows expose the
   three aggregates. 6 new tests (cross-thread unblock, pre-first-next
   noop, EMA, zero-when-unreported, passthrough, stats); SDK suite
   674 passed / 0 failed. Rig acceptance (deployed rig, acceptance
   script `p18_skew_cancel_accept.py`, 15s per mode, stream
   `third` @15fps): preview p95 17.0ms (bound TTL+1 frame = 567ms);
   strict p95 17.7ms (bound 1 frame = 66.7ms) — same metric both
   modes, PASS; cancel() from a non-consuming thread unblocked the
   consumer in ~0ms and an immediate resubscribe flowed again in
   293ms (no in_flight leak); GetStats aggregates present (n>0,
   max≥avg). VERDICT: PASS.
9. **DSP single-worker queue split + pool retention**: ✅ shipped
   2026-09-11. Root cause convicted by a three-variant probe
   (B persistent client+pools / C persistent client per-call pools /
   A accel router fresh client per call): the 43ms tail was **not**
   queueing contention — it was per-call pool churn. The router's
   fresh client per call allocs temp pools and releases them right
   after; the HAL pool cache is weak (the vendor chunk dies with its
   last buffer), so every call destroys and recreates the 32-buffer
   dma chunk, and the chunk-recreate collides with the prior call's
   background chunk-free. Pre-fix on the rig: variant C 300/300 calls
   >25ms, variant A max 104ms. Fix is two halves in dsp_service: (a)
   lane split — NORMAL contexts submit at cfg priority, BACKGROUND
   at 0 (dsp-lane-split-test); (b) **pool retention** — released
   non-imported pool buffers park by geometry {w,h,fmt} for
   `pool_retention_ms` (default 3000) and alloc_buffers serves from
   the park first (zero HAL rounds, keeps the vendor chunk alive).
   Footprint is chunk-accurate: one parked buffer pins the whole
   vendor chunk, first park of a geometry charges
   kPoolChunkBuffers×bytes, over-cap evicts whole geometries
   oldest-first (cap `pool_retention_max_bytes`, default 192MiB).
   Imports are never parked (their memory is the client's); expiry
   is lazy on the next same-geometry alloc plus a 1s idle worker
   sweep; stop() flushes. Seven-case host harness
   (dsp-pool-retention-test): reuse-hit (same dma_fds back, zero HAL
   rounds), lazy expiry, sweep expiry, imports-never-parked,
   cap-evicts-oldest-geometry, disabled-by-config (either knob at 0
   is the rollback path), stop-flushes. Rig acceptance (deployed,
   paced 25ms/call): variant C 300/300 slow → **0/300**; variant A
   tail eliminated (104 → 36.2ms); official ab_resize_nv12_default
   baseline bimodal 8.0/43.4ms → single-peak p50 8.0 / p90 30.2 /
   p99 35.7 / max 37.4ms, zero samples ≥43ms. Stream co-acceptance
   under the same load: get_frame_main p50 33.34ms (exactly the 30fps
   cadence), p99/max 36-38ms, publisher slow_broadcast=0, journal
   clean of pool/alloc failures across both load windows.
   Diagnostic correction from the probe: the pre-fix "8ms mode" was
   the router's *software fallback*, not a fast DSP mode — under
   jitter the DSP branch raised HardwareUnavailable (`_dsp_call`
   runs `cpu_fallback=False`), the router caught it and ran its own
   numpy resize; per-call latency recorded 8ms because the DSP round
   was never made. **Known gap (quota evasion)**: the per-owner DSP
   quota (120 MPix/s; a 2.59-MPix resize job → 21.6ms/call sustained
   floor) is keyed by client identity, and the router's
   fresh-client-per-call design makes every call a fresh quota owner
   — variant B/C trip the quota at full speed while variant A evades
   it by owner churn. The pacing in the acceptance probe exists
   because of this. Closing it means keying the quota by UDS peer
   identity (SO_PEERCRED) or a per-process admission instead of per
   client id; deferred, and the pacing requirement stands for any
   sustained-load DSP benchmark until then.
10. **Output observability & recovery**: ✅ shipped 2026-09-11. Five
    halves. (a) Wire V3: 38-byte header — V2's 30 bytes + flags bit1
    (SEQ_PRESENT) + uint64 LE packet seq at offset 30; `total_size`
    covers the extension, V2 peers unaffected. Seq is assigned at
    publisher enqueue, first=1, monotonic per stream; a publisher
    restart rebases to 1, which the SDK classifies as restart, not
    loss. SDK `EncodedFrame.seq` plus client-side counters
    `seq_packets / last_seq / seq_gap_events / seq_missing` and a
    one-line hole log ("seq hole A..B missing=N"); counters survive
    auto-reconnect by design so a hole stays visible after recovery.
    (b) Unified drop view on `GetStreamStatus` (wire fields 13-23):
    publisher layer (`packets_published`, `queue_overflow_drops`,
    `client_send_drops`, `client_send_failures`, `client_disconnects`,
    `last_packet_seq`, `publisher_clients`), overlay bake layer
    (`bake_skips`, plus P1-6 strict counters), injection layer via
    `InjectionStatus.frames_dropped`, SDK client layer (drop-oldest +
    seq counters). Reconciliation identity: over a clean drained
    window `received + missing == packets_published delta` exactly.
    `client_disconnects` is the honest counter for a killed peer: the
    publisher's per-client control poll (non-blocking recv before
    every send) sees EOF and reaps the client before any send could
    EPIPE, so `client_send_failures` stays 0 on this path. AF_UNIX
    note: a subscriber-side `shutdown()` is invisible to the publisher
    (sends into a shut-down peer are silently discarded, never EPIPE)
    — only full close (SIGKILL, fds gone) is publisher-visible;
    subscriber-side hole detection covers the shutdown case.
    (c) SubscribeEvents stub implemented in the device-control
    poller; a fresh subscriber receives a baseline
    LIGHT_SENSOR_CHANGE immediately. (d) `get_stats(
    sampling_window_ms=…)` — server clamps 1..5000; omitted keeps the
    legacy Empty request so old servers and old callers are untouched.
    (e) encoded_publisher sends are non-blocking; a would-block or
    failed send counts (`client_send_drops` / `client_send_failures`)
    instead of stalling the broadcast. SDK suite 683 passed / 0
    failed. Rig acceptance (deployed rig, `p110_acceptance.py`):
    SIGKILL subscriber → `client_disconnects 0→1` with
    `client_send_failures` delta 0 (reap preempts EPIPE, by design);
    same-client shutdown+reconnect → hole 195 packets over 6s (~32
    pkt/s), exactly one gap event, missing == hole; clean window →
    301+0 == 301 with every drop layer zero and bake_skips ==
    packets_published; SubscribeEvents baseline event with payload.
    VERDICT: PASS.

### P2 — facade & ecosystem
11. **SDK whole-pipeline facade**: ✅ shipped 2026-09-11.
    `neoruntime_ipc_sdk.StreamPipeline` — one call wires capture →
    infer → overlay: a daemon-fed `InferenceClient.subscribe` worker
    thread, per-result `OverlayClient` annotate (optional static
    `polygons` at start; `min_score`/`labels` draw-filter that still
    publishes an empty list to clear stale boxes; optional `on_result`
    hook to replace or drop), a bounded drop-oldest app queue behind
    `results()`, a `status()` snapshot passing the subscribe
    observability (drops, latency, skew_us) through, and a single-shot
    lifecycle (start-twice and restart-after-stop raise; stop joins
    the worker, clears boxes and zones, closes only the clients it
    created; `draw=False` never touches the overlay). Positioning vs
    the client-side `InferencePipeline` is documented in both module
    docstrings and both API pages (zh+en). Unit 16/16 + full SDK
    regression 699 passed / 0 failed; rig acceptance 21/21 PASS
    (on-rate 95 results/10s and 30/6s, all annotated, zero drops,
    monotonic frame_sequence, skew 24.6/48.7ms well under TTL + 1
    frame, bake-target flags stable 0x1 across start/run/stop, a
    second instance on the same stream runs after the first stopped).
    Client-side latency_ms reads 0.0 on the rig — the subscribe
    layer's same-clock sane-window guard; skew is the accepted metric.
12. **frame-injection P1/P2; web-stream-url**: ✅ shipped
    2026-09-11. Daemon: `PushFrame` is client-streaming with a
    shallow drop-oldest queue and pts pacing (newest-due-wins,
    `pts_ns` in device CLOCK_MONOTONIC); OVERLAY mode accepts an
    NV12 dma-buf (opaque paste at dest_x/dest_y) or an ARGB32
    buffer (CPU alpha blend at the bake site — deliberately CPU
    while the DSP queue is one congested worker, see P1-9); the
    manifest permission gate resolves the buffer owner's UDS
    identity (SO_PEERCRED) against `injection.allowed_apps` and
    rejects with -6 INJ_SVC_ERR_PERMISSION (empty list allows all).
    SDK: `FramePublisher(mode="overlay", fmt="nv12"|"argb",
    inset=, dest=)` — ARGB rides the shared memfd-import ring
    because the deployed HAL refuses ARGB32 pool allocation
    (wire OOM). Rig e2e 8/8: ARGB inset blend pixel-verified on
    decode, NV12 paste, streaming drop-oldest, pts pacing,
    permission deny (session stays inactive) and allow, EOS
    restore at next IDR; daemon config restored to baseline after
    the deny run. web-stream-url P0 shipped SDK-side with **no
    RPC**: `web.py platform_stream_url()` composes the gateway's
    existing `/api/v1/h264/{id}` WS URL (host/scheme/token; the
    server cannot know its own external host, and the path is now
    the test-covered contract) — 20 offline tests, rig gateway
    probes 401/404/301. Remaining tail: RGB888 input, DSP-offloaded
    blend (needs P1-9), zero-copy swap; signed-ttl URL negotiation
    stays future daemon work (`web-stream-url.md` P1/P2).
13. **Lifecycle session**: ✅ shipped 2026-09-11. One `session_id`
    threads all four steps. SDK: `StreamPipeline(..., session_id=)`
    tags the StreamInfer subscription and every result-driven overlay
    write (`annotate_result`, and the filtered `annotate` branch);
    static zones at start and the box/zone clears at stop stay
    **untagged operator writes** so teardown behaves unconditionally.
    `FramePublisher(..., session_id=)` stamps every PushFrame
    request; `OverlayClient.annotate(..., session_id=)` takes the tag
    directly. Daemon: the injection session opens on the first tagged
    frame (owner fd recorded, latest-non-empty tag wins, surfaced in
    InjectionStatus) and closes on the owner's disconnect. **Reclaim
    is fd-anchored, not tag-anchored** — the daemon anchors every
    layer on the buffer owner's UDS connection and reclaims on that
    client's disconnect regardless of the tag; the tag is correlation
    and observability, never the lifetime key. One chain does it all:
    `FdPublisher::disconnect_client` → release_all_outstanding →
    `injection_service->release_owner(fd)` → `dsp_service->
    release_client_buffers(fd)`. On the inference side the SessionGuard
    broadcasts `<result-topic-prefix>session/end` (exact-match topic,
    metadata `{"session_id": …}`) when a StreamInfer session dies, and
    the camera-daemon overlay subscriber sweeps every polygon sidecar
    tagged with that session. Contract note: the server keeps a
    StreamInfer session alive through per-frame inference failures —
    only the client's consecutive-failure breaker (default 10;
    `subscribe(max_consecutive_failures=0)` disables) ends it early.
    Rig e2e 8/8: SIGKILL an app mid-pipeline (inference + tagged
    polygon + tagged injection live under one tag) → within the same
    second InjectionStatus goes inactive with the tag cleared, the
    overlay sweep, the per-client DSP buffer free, and the StreamInfer
    end all land in the journals; pre-kill checks confirm all layers
    were live and tagged first.
14. **Behavior decoupling + frame-sync closure** (output-isolation
    scope cut, 2026-09-11): shipped and rig-verified 2026-09-11.
    The user scope cut defers the D output-buffer chain (separate
    output pool, unified encode entry, OverlayClient→platform buffer
    draw) — the two documented SDK modes instead: `subscribe` returns
    inference results only (zero video side effects by contract), and
    `keep_fd`/copy hands out clean frames the app composes on itself
    (own DSP pool RESIZE→BLEND; keep_fd frames are never a DSP write
    target or blend base — `dsp-import-write-target-test` enforces).
    Daemon: `handle_event` now takes the event-bus `source` —
    platform result events (`ai-runtime` / `auto-infer`) draw only when
    their infer stream is bound via daemon yaml `[ai_overlay] bindings:`
    (infer→display, same direction as `stream_map`) or
    `legacy_auto_bind:` is on (default OFF — a bare `subscribe()` must
    not change the video; the legacy default `stream_map` backfill now
    lives under the same switch, see migration note in the overlay
    docs); every other source is an app event, always admitted as its
    own `(session_id, source_id)` layer — layers stack (app newest
    first, then platform in arrival order), a fresh app layer
    suppresses platform layers on the same stream, `[]` clears only
    the publisher's own layer, `session/end` sweeps app layers and
    session-tagged polygons. Frame sync: per-stream epoch counters
    (lazy-seeded CLOCK_MONOTONIC µs, never 0) bumped by
    ReconfigureEncoder pipeline restarts and full transform reinit
    (both are shared-pipeline restarts: every encoder's stream gets
    the bump) with all held layers purged; app events carrying a stale
    `stream_epoch` or a `frame_sequence` past the bake site's anchor
    are rejected (`overlay_epoch_rejects` / `overlay_late_commands`),
    and a `frame_sequence`-bound layer draws only within the frame-bind
    slack window then expires. SDK: `annotate` / `annotate_result`
    gain `frame_sequence=` / `stream_epoch=` (metadata-only, opt-in —
    binding params never ride the payload the daemon string-scans);
    `StreamStatus` gains wire fields 24-28 (`stream_epoch`,
    `overlay_layer_count`, `overlay_late_commands`,
    `overlay_epoch_rejects`, `overlay_no_binding_drops`) so the app
    reads the live epoch for the restart handshake. Host: new
    `ai-overlay-binding-test` (12 cases: admission, stacking,
    suppression, bind window, epoch reject, anchor-before-enable) and
    `dsp-import-write-target-test` (4 cases), full ctest green; SDK
    +8 tests (frame-binding metadata, StreamStatus mapping), 774P/215S
    offline. Rig 7/7: (a) default config drops platform events
    (`overlay_no_binding_drops` 0→4, zero layers) — bare subscribe
    changes no video; (b) `bindings: third:third` admits third
    (1 layer) while unbound `sub` still drops; (c) `legacy_auto_bind: 1`
    admits every platform event (old behavior, both ai-runtime and
    auto-infer sources); (d) app + platform interleave on one stream
    holds a stable 2 layers, each publish replacing only its own;
    (e) an app's `[]` clears its own boxes, the other layers survive;
    (f) a stale-epoch publish is rejected (`overlay_epoch_rejects` +1),
    a valid one is stored, and an fps-changing ReconfigureEncoder (the
    pipeline-restart branch — bitrate-only HAL override correctly does
    not bump) advances every stream's epoch by one and purges all held
    layers while counters survive; (g) all wire fields 24-28 read back
    on every stream. Daemon config restored to baseline after the runs.

## Verification

- P0-1: inject a test pattern on the deployed rig (encoder-feed
  insertion) → web console visual check + InjectionStatus counters;
  bitrate/fps before-after; `StopInjection` restores the ISP path at
  the next IDR.
- P0-2: SDK `infer(frame)` with a 4K FrameHandle vs bytes path (expect
  copy-in savings, 54ms → ~15ms class); a foreign/invalid fd must be
  rejected (negative test).
- P0-4: polygon/track via annotate → visual check; TTL = 2× frame
  period behavior with fast motion.
- P1-6: strict mode output latency = inference latency + 1 frame;
  backlog overflow logs and degradation.
- P1-7: hold 4 frames / exceed 200ms → daemon refuses new frames and
  the counter is visible (negative test); baked flag vs stream_map
  spot check.
- P1-8: live-preview skew stable ≤ TTL + 1 frame period; strict mode
  skew ≤ 1 frame — one metric accepts both modes.
- P1-9: rerun ab_resize_nv12_default, the 43ms tail disappears.
  Rig PASS (deployed, 2026-09-11): single-peak p50 8.0 / p90 30.2 /
  p99 35.7 / max 37.4ms, zero samples ≥43ms (baseline bimodal
  8.0/43.4); churn probe variant C 300/300 slow → 0/300; stream
  cadence intact under load (get_frame_main p50 = 33.34ms).
- P1-10: kill a subscriber to simulate loss → seq-number gaps
  detectable; four drop levels reconcile in one view.
- P2-13: SIGKILL an app mid-pipeline → all four layers reclaim within
  the same second: InjectionStatus inactive with the session tag
  cleared; `session '<tag>' ended … polygon sidecar cleared`
  (camera-daemon); `DspService: client N disconnected, freed M
  buffer(s)` (camera-daemon); `StreamInfer: ended` (ai-runtime).
  Rig run passed 8/8 (pre-kill liveness + tag, post-kill reclaim ×4).
