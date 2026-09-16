"""Keep-fd -> DSP resize -> infer -> render/blend -> publish.

zero_copy=True keeps every pixel device-side: the resize result is a
DspBufferRef (infer sends its buffer_id), the blend lands directly in a
publisher pool slot and push_slot publishes it — no read-back, no array
over the publish RPC. The default path materializes and re-publishes
the latest composition from a 40Hz pacer with a 40ms lead. Publish rate
is NOT inference/new-picture rate. `e2e` is a compatibility alias for
source-to-compose-complete age, never display latency. Source age
assumes execution on the source device's CLOCK_MONOTONIC domain.
"""
from __future__ import annotations

import logging
import threading
import time

from display import format_b_line, text_chip_rgba

from neoruntime_ipc_sdk.draw import render_overlay_fragments, render_overlay_rgba

logger = logging.getLogger("perf_demo.chain_b")
CHIP_REFRESH_S = 0.333
DEGRADED_COOLDOWN_S = 30.0
# Transient compose failures (e.g. a DSP blend job the firmware rejects
# under load) are absorbed per-frame instead of tearing the chain down:
# short exponential backoff between attempts, escalate to the full
# degraded/rebuild path only after COMPOSE_FAIL_ESCALATE consecutive
# failures. Keeps a momentary DSP hiccup from becoming a 30s+ black
# screen while still bounding the outage when the error persists.
COMPOSE_FAIL_BACKOFF_S = 0.25
COMPOSE_FAIL_BACKOFF_MAX_S = 2.0
COMPOSE_FAIL_ESCALATE = 30
CHIP_MARGIN_PX = 16
PUBLISH_HZ = 40.0
PUBLISH_LEAD_NS = 40_000_000
PUBLISH_FAIL_LIMIT = 10
FRAME_POLL_MS = 500
# daemon BLEND takes one job of at most 64 overlays; each box costs up
# to 4 fragments and the metrics chip takes one slot
_MAX_BLEND_OVERLAYS = 64
_FRAGMENTS_PER_BOX = 4


class ChainB(threading.Thread):
    def __init__(self, *, camera, infer, dsp, media, hub,
                 stream_id: str = "sub", model_id: str = "",
                 min_score: float = 0.3, skip: int = 1,
                 pool_depth: int = 4, zero_copy: bool = False,
                 session_id: str = "perf-demo", fps: float = 0,
                 publish_hz: float = PUBLISH_HZ,
                 metrics_overlay: bool = True, detections_overlay: bool = True) -> None:
        super().__init__(name="chain-b", daemon=True)
        self.camera, self.infer, self.dsp = camera, infer, dsp
        self.media, self.hub = media, hub
        self.stream_id, self.model_id = stream_id, model_id
        self.min_score, self.skip = min_score, skip
        self.pool_depth, self.zero_copy = pool_depth, zero_copy
        self.session_id = session_id
        self.fps, self.publish_hz = fps, publish_hz
        self.metrics_overlay, self.detections_overlay = metrics_overlay, detections_overlay
        self._stop_event = threading.Event()
        self._pacer_stop = threading.Event()
        self._publisher = None
        self._infer_wh = None
        self._chip, self._chip_t = None, 0.0
        self._latest_lock = threading.Lock()
        self._latest = None  # immutable (pixels, source metadata)
        self._pacer = None
        self._publish_failures = 0
        self._compose_failures = 0
        self._last_published = None
        self._next_admit_ns = 0
        self._delivery_count = 0

    def stop(self) -> None:
        self._stop_event.set()
        self._pacer_stop.set()

    def run(self) -> None:
        try:
            while not self._stop_event.is_set() and not self.hub.b.counters.get("degraded_reason"):
                try:
                    self._prepare_infer()
                    self._ensure_publisher()
                    self._frame_loop()
                except Exception as exc:
                    reason = type(exc).__name__
                    logger.warning("chain B degraded: %s", reason)
                    self.hub.b.counters.set("degraded_reason", reason)
                    self.hub.emit("b_error", error=reason, stream_id=self.stream_id)
                    if not self._close_publisher(eos=False):
                        break
                    if self._stop_event.wait(DEGRADED_COOLDOWN_S):
                        break
                    self.hub.b.counters.set("degraded_reason", None)
                    self.hub.b.counters.increment("rebuilds")
        finally:
            self._close_publisher(eos=True)
            try:
                self.media.close()
            except Exception as exc:
                self.hub.b.counters.set("cleanup_error", type(exc).__name__)
            self.hub.emit("chain_exit", chain="b", counters=self.hub.b.counters.snapshot())

    def _prepare_infer(self) -> None:
        info = self.infer.get_model_info(self.model_id)
        inp = (getattr(info, "inputs", None) or [None])[0]
        shape = list(inp.get("shape") or []) if isinstance(inp, dict) else list(getattr(inp, "shape", None) or [])
        if shape and shape[0] == 1:
            shape = shape[1:]
        if len(shape) != 3 or any(not isinstance(v, int) or v <= 0 for v in shape):
            raise ValueError("model input shape must be positive NHWC")
        h, w, _c = shape
        self._infer_wh = (w, h)

    def _ensure_publisher(self) -> None:
        from neoruntime_ipc_sdk import FramePublisher
        self._publisher = FramePublisher(
            self.camera, self.dsp, stream_id=self.stream_id,
            pool_depth=self.pool_depth, mode="replace", fmt="nv12", session_id=self.session_id)
        self._publish_failures = 0
        self._pacer_stop.clear()
        if not self.zero_copy:
            self._pacer = threading.Thread(target=self._pacer_loop, name="chain-b-pub", daemon=True)
            self._pacer.start()
        self.hub.b.counters.set("pool_depth", self.pool_depth)
        self.hub.b.counters.set("lease_mode", "lease" if self._publisher.lease_mode else "legacy")

    def _publish(self, pub, composed, source: dict, due: int, slot: int | None = None) -> None:
        counters = self.hub.b.counters
        counters.increment("publish_calls")
        started = time.monotonic_ns()
        success, error = False, None
        try:
            if slot is None:
                pub.publish(composed, pts_ns=due)
            else:
                pub.push_slot(slot, pts_ns=due)
            success = True
            self._publish_failures = 0
            counters.increment("publishes")
            identity = (source["source_frame_id"], source["source_timestamp_ns"])
            if identity != self._last_published:
                counters.increment("published_unique")
                self._last_published = identity
        except Exception as exc:
            error = type(exc).__name__
            counters.increment("publish_errors")
            self._publish_failures += 1
            if self._publish_failures >= PUBLISH_FAIL_LIMIT:
                counters.set("degraded_reason", "publish dead: " + error)
        finally:
            elapsed = (time.monotonic_ns() - started) / 1e6
            self.hub.b.pub_ms.add(elapsed)
            self.hub.emit("b_publish", **source, publish_started_ns=started,
                          publish_ms=elapsed, due_pts_ns=due, success=success,
                          error=error, lease_wait_ms=None)

    def _pacer_loop(self) -> None:
        interval = 1.0 / self.publish_hz
        while not self._stop_event.is_set() and not self._pacer_stop.is_set():
            started = time.monotonic()
            with self._latest_lock:
                latest = self._latest
            pub = self._publisher
            if latest is not None and pub is not None:
                composed, source = latest
                self._publish(pub, composed, source, time.monotonic_ns() + PUBLISH_LEAD_NS)
                if self.hub.b.counters.get("degraded_reason"):
                    break
            self._pacer_stop.wait(max(0, interval - (time.monotonic() - started)))

    def _source(self, frame) -> dict:
        return dict(stream_id=self.stream_id, source_frame_id=frame.sequence,
                    source_timestamp_ns=frame.timestamp_ns or None,
                    source_timestamp_clock="device_CLOCK_MONOTONIC")

    @staticmethod
    def _age_ms(now: int, source_ns) -> float | None:
        return (now - source_ns) / 1e6 if source_ns and 0 < source_ns <= now else None

    def _frame_loop(self) -> None:
        # subscribe() reconnects forever and exposes no cancellation. The
        # public timed single-frame API preserves keep-fd without that trap.
        while not self._stop_event.is_set() and not self.hub.b.counters.get("degraded_reason"):
            started = time.monotonic_ns()
            frame = self.media.get_frame(self.stream_id, timeout_ms=FRAME_POLL_MS, keep_fd=True)
            if frame is None:
                continue
            backoff = 0.0
            with frame:  # stop and intentional skip also release immediately
                delivered = time.monotonic_ns()
                source = self._source(frame)
                counters = self.hub.b.counters
                counters.increment("frames_delivered")
                self._delivery_count += 1
                reason = self._skip_reason(delivered)
                self.hub.emit("b_delivery", **source, delivered_ns=delivered,
                              delivery_wait_ms=(delivered - started) / 1e6,
                              source_age_ms=self._age_ms(delivered, source["source_timestamp_ns"]),
                              skipped=reason)
                if reason:
                    continue
                # Self-heal transient compose failures in place (DSP reject
                # under load, slot hiccup): absorb, back off briefly, retry
                # the NEXT frame. Only sustained failure escalates to the
                # degraded/rebuild path. The backoff itself waits outside
                # the frame lease so the buffer returns to the daemon.
                try:
                    self._compose_frame(frame, source, delivered)
                    if self._compose_failures:
                        self._compose_failures = 0
                except Exception:
                    self._compose_failures += 1
                    counters.increment("compose_failures_transient")
                    if self._compose_failures == 1:
                        logger.warning("chain B compose failed (transient, "
                                       "absorbing with backoff)")
                    if self._compose_failures >= COMPOSE_FAIL_ESCALATE:
                        raise
                    backoff = min(
                        COMPOSE_FAIL_BACKOFF_S * (2 ** (self._compose_failures - 1)),
                        COMPOSE_FAIL_BACKOFF_MAX_S)
            if backoff:
                self._stop_event.wait(backoff)

    def _skip_reason(self, now: int) -> str | None:
        if self._stop_event.is_set():
            return "stopping"
        if (self._delivery_count - 1) % max(1, self.skip):
            self.hub.b.counters.increment("decimation_skipped")
            return "decimation"
        if self.fps and now < self._next_admit_ns:
            self.hub.b.counters.increment("rate_skipped")
            return "rate_limit"
        if self.fps:
            self._next_admit_ns = now + int(1e9 / self.fps)
        return None

    def _compose_frame(self, frame, source: dict, delivered: int) -> None:
        times = dict(materialize_ms=None, resize_ms=None, rpc_ms=None,
                     render_ms=None, blend_ms=None, hw_infer_time_us=None,
                     queue_time_us=None, daemon_infer_time_us=None)
        success, error = False, None
        out = None
        compose_done = None
        dsp_backend = None
        overlays = None
        try:
            started = time.monotonic_ns()
            base = frame if self.zero_copy else frame.to_array()
            times["materialize_ms"] = (time.monotonic_ns() - started) / 1e6
            if not self.zero_copy:
                self.hub.b.counters.increment("frames_materialized")
            out = self._infer_or_count(frame, times, source)
            started = time.monotonic_ns()
            overlays = self._overlays_for(frame, out)
            rendered = time.monotonic_ns()
            times["render_ms"] = (rendered - started) / 1e6
            if self.zero_copy:
                # P2-5: blend straight into the publisher's slot — the
                # composed frame exists only device-side. An acquired
                # slot that never reaches push_slot is invisible to the
                # lease set, so acquire only when the blend follows.
                slot = self._publisher.acquire_slot()
                self.dsp.blend_hw(base, overlays, zero_copy=True,
                                  out=self._publisher.pool, dst_slot=slot)
                composed = None
            else:
                composed = self.dsp.blend_hw(base, overlays, zero_copy=False)
            compose_done = time.monotonic_ns()
            times["blend_ms"] = (compose_done - rendered) / 1e6
            # Instance-local SDK report, valid for this serial compose path only.
            # False means software path, not necessarily a DSP rejection.
            try:
                reported = getattr(self.dsp, "last_used_hw", None)
                dsp_backend = reported if type(reported) is bool else None
            except Exception:
                pass  # diagnostic access must not fail a completed blend
            success = True
        except Exception as exc:
            error = type(exc).__name__
            self.hub.b.counters.increment("compose_errors")
            raise
        finally:
            # App input layout only, not SDK padding/ARGB packing or HAL allocation.
            # Extract outside stage timings; metadata failures leave the result intact.
            overlay_geometry = None
            try:
                if overlays is not None:
                    overlay_geometry = [dict(x=int(x), y=int(y), w=int(rgba.shape[1]),
                                             h=int(rgba.shape[0]), stride=int(rgba.strides[0]),
                                             bytes=int(rgba.nbytes)) for rgba, x, y in overlays]
            except Exception:
                pass  # unknown geometry, never replace a compose error
            self.hub.emit("b_compose", **source, **times, success=success, error=error,
                          infer_success=out is not None, compose_done_ns=compose_done,
                          compose_ms=(compose_done - delivered) / 1e6 if compose_done else None,
                          source_age_ms=self._age_ms(compose_done, source["source_timestamp_ns"]) if compose_done else None,
                          dsp_backend=dsp_backend if success else None,
                          overlay_geometry_scope="app_blend_input",
                          overlay_geometry=overlay_geometry)
        published_source = {**source, "compose_done_ns": compose_done}
        self._record_composition(out, times, source, compose_done)
        if self.zero_copy:
            self._publish(self._publisher, None, published_source,
                          frame.timestamp_ns, slot=slot)
        else:
            with self._latest_lock:
                self._latest = (composed, published_source)

    def _record_composition(self, out, times: dict, source: dict, done: int) -> None:
        b = self.hub.b
        b.counters.increment("frames_ok")  # compose success, NOT infer success
        b.counters.increment("composed_unique")
        b.counters.set("last_compose_ns", done)
        hw = times["hw_infer_time_us"]
        b.record_frame(pull_ms=times["materialize_ms"],
                       infer_ms=(times["resize_ms"] or 0) + (times["rpc_ms"] or 0),
                       hw_infer_ms=hw / 1000 if hw else None,
                       draw_ms=times["render_ms"] + times["blend_ms"], pub_ms=None,
                       e2e_ms=self._age_ms(done, source["source_timestamp_ns"]),
                       objects=len(out.objects or []) if out is not None else 0)

    def _infer_or_count(self, frame, times=None, source=None):
        times = times if times is not None else {}
        source = source if source is not None else self._source(frame)
        b = self.hub.b.counters
        error, out, stage = None, None, "resize"
        started = time.monotonic_ns()
        try:
            if self.zero_copy:
                # P2-2/P2-3: the model input stays device-side; infer
                # sends its buffer_id and the read-back never happens
                scaled = self.dsp.resize_hw(frame, *self._infer_wh,
                                            scaling="stretch", out="ref")
            else:
                scaled = self.dsp.resize_hw(frame, *self._infer_wh, scaling="stretch")
            times["resize_ms"] = (time.monotonic_ns() - started) / 1e6
            stage = "rpc"
            b.increment("infer_attempts")
            started = time.monotonic_ns()
            try:
                out = self.infer.infer(scaled, self.model_id)
            finally:
                if self.zero_copy:
                    scaled.release()  # RPC settled; buffer back to the daemon
            if out is None:
                error = "empty_result"
            else:
                b.increment("infer_success")
                b.set("last_infer_success_ns", time.monotonic_ns())
                times.update(hw_infer_time_us=getattr(out, "hw_infer_time_us", 0) or None,
                             queue_time_us=getattr(out, "queue_time_us", None),
                             daemon_infer_time_us=getattr(out, "infer_time_us", None))
        except Exception as exc:
            error = type(exc).__name__
        finally:
            times[stage + "_ms"] = (time.monotonic_ns() - started) / 1e6
            if out is None:
                b.increment("frames_err")
                if stage == "rpc":
                    b.increment("infer_failures")
            self.hub.emit("b_infer", **source, **times, success=out is not None,
                          attempted=stage == "rpc", error=error, failure_stage=stage if error else None)
        return out

    def _overlays_for(self, frame, out) -> list[tuple]:
        overlays = []
        objs = [o for o in (out.objects or []) if (getattr(o, "score", 0.0) or 0) >= self.min_score] if out is not None else []
        if objs and self.detections_overlay:
            # Stretch preserves normalized boxes; draw requires source-frame pixels.
            # Create fresh xyxy tuples so repeated rendering never rescales the result.
            # Per box, tight stroke fragments (caption+top/bottom/left/right edges):
            # render, the RGBA->ARGB repack and the DSP blend are all area-bound,
            # and a box's visible pixels are its ~4px stroke perimeter — a
            # full-frame detection used to pay a 0.9MP canvas for them. The daemon
            # blends a job's whole overlay list at once, so extra fragments cost
            # only their (tiny) area; once the fragment budget runs out, the
            # remaining boxes share one union canvas rather than overflowing
            # the 64-overlay batch cap.
            budget = _MAX_BLEND_OVERLAYS - 1  # the metrics chip takes a slot
            boxes = [(o.bbox.x * frame.width, o.bbox.y * frame.height,
                      (o.bbox.x + o.bbox.width) * frame.width,
                      (o.bbox.y + o.bbox.height) * frame.height) for o in objs]
            for i, (box, obj) in enumerate(zip(boxes, objs)):
                kwargs = dict(labels=[getattr(obj, "label", None)],
                              scores=[getattr(obj, "score", None)])
                if budget >= _FRAGMENTS_PER_BOX:
                    overlays.extend(render_overlay_fragments(
                        frame.width, frame.height, boxes=[box], **kwargs))
                    budget -= _FRAGMENTS_PER_BOX
                else:
                    rgba, x0, y0 = render_overlay_rgba(
                        frame.width, frame.height, boxes=boxes[i:],
                        labels=[getattr(o, "label", None) for o in objs[i:]],
                        scores=[getattr(o, "score", None) for o in objs[i:]])
                    overlays.append((rgba, x0, y0))
                    break
        if self.metrics_overlay:
            overlays.append(self._chip_for(frame.width, frame.height))
        return overlays

    def _chip_for(self, frame_w: int, frame_h: int) -> tuple:
        now = time.monotonic()
        if self._chip is None or now - self._chip_t >= CHIP_REFRESH_S:
            chip, _, _ = text_chip_rgba(format_b_line(self.hub.snapshot()["b"]))
            self._chip = (chip, CHIP_MARGIN_PX, max(frame_h - chip.shape[0] - CHIP_MARGIN_PX, 0))
            self._chip_t = now
        return self._chip

    def _close_publisher(self, *, eos: bool) -> bool:
        self._pacer_stop.set()
        if self._pacer is not None:
            self._pacer.join(timeout=2.0)
            if self._pacer.is_alive():
                self.hub.b.counters.set("cleanup_error", "pacer join timeout")
                return False  # never close a lease still used by publish()
        self._pacer = None
        pub, self._publisher = self._publisher, None
        with self._latest_lock:
            self._latest = None
        if pub is None:
            return True
        try:
            if eos:
                pub.publish_eos()
        except Exception as exc:
            self.hub.b.counters.set("cleanup_error", type(exc).__name__)
        finally:
            try:
                pub.close()
            except Exception as exc:
                self.hub.b.counters.set("cleanup_error", type(exc).__name__)
        return True
