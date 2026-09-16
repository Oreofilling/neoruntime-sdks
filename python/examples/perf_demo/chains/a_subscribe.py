"""Chain A: subscribe results, annotate bound boxes and unbound metrics.

Detection and metric layers use distinct session IDs. Raw events record
result arrival and each annotate call, not a claim of encoded visibility.
"""
from __future__ import annotations

import logging
import threading
import time

from display import format_a_line, format_status_line, metric_detections

from neoruntime_ipc_sdk import OverlayClient

logger = logging.getLogger("perf_demo.chain_a")
EPOCH_REFRESH_S = 30.0
REBUILD_BACKOFF_S = 1.0


class ChainA(threading.Thread):
    def __init__(self, *, camera, infer, hub, stream_id: str = "third",
                 display_stream: str = "main", model_id: str = "",
                 fps: int = 10, annotate_hz: float = 3.0,
                 min_score: float = 0.3, session_id: str = "perf-demo",
                 metrics_overlay: bool = True, detections_overlay: bool = True) -> None:
        super().__init__(name="chain-a", daemon=True)
        self.camera, self.infer, self.hub = camera, infer, hub
        self.stream_id, self.display_stream = stream_id, display_stream
        self.model_id, self.fps, self.min_score = model_id, fps, min_score
        self.session_id, self.osd_session = session_id, f"{session_id}-osd"
        self.metrics_overlay, self.detections_overlay = metrics_overlay, detections_overlay
        self.overlay = OverlayClient()
        self._annotate_interval = 1.0 / annotate_hz
        self._stop_event = threading.Event()
        self._iter = None
        self._epoch = None
        self._epoch_next_check, self._last_annotate = 0.0, 0.0

    def stop(self) -> None:
        self._stop_event.set()
        self.cancel_iter()

    def cancel_iter(self) -> None:
        it = self._iter
        if it is not None:
            try:
                it.cancel()
            except Exception as exc:
                logger.warning("subscribe cancel failed: %s", type(exc).__name__)
                self.hub.emit("a_error", error="cancel_failed")

    def degrade(self, reason: str) -> None:
        self.hub.a.counters.set("degraded_reason", reason)
        self.cancel_iter()

    def run(self) -> None:
        enabled = False
        try:
            if not self._stop_event.is_set() and (self.metrics_overlay or self.detections_overlay):
                self.overlay.enable(show_label=True, show_confidence=False, line_thickness=2)
                enabled = True
            while not self._stop_event.is_set() and self.hub.a.counters.get("degraded_reason") is None:
                try:
                    self._subscribe_loop()
                except Exception as exc:
                    self.hub.a.counters.increment("subscribe_errors")
                    self.hub.emit("a_error", error=type(exc).__name__, stream_id=self.stream_id)
                    logger.warning("chain A subscribe ended: %s", type(exc).__name__)
                if self._stop_event.is_set() or self.hub.a.counters.get("degraded_reason") is not None:
                    break
                self.hub.a.counters.increment("reconnects")
                self._stop_event.wait(REBUILD_BACKOFF_S)
        except Exception as exc:
            self.hub.a.counters.set("degraded_reason", type(exc).__name__)
        finally:
            self._clear_drawings()
            if enabled:
                try:
                    self.overlay.disable()
                except Exception as exc:
                    self.hub.a.counters.set("cleanup_error", type(exc).__name__)
            try:
                self.overlay.close()
            except Exception as exc:
                self.hub.a.counters.set("cleanup_error", type(exc).__name__)
            self.hub.emit("chain_exit", chain="a", counters=self.hub.a.counters.snapshot())

    def _subscribe_loop(self) -> None:
        self._refresh_epoch()
        gen = self.infer.subscribe(self.stream_id, self.model_id, fps=self.fps, session_id=self.session_id)
        self._iter = gen
        try:
            if self._stop_event.is_set():
                return
            for seq, result in gen:
                if self._stop_event.is_set():
                    break
                now = time.monotonic()
                self.hub.emit("a_result", source_frame_id=seq, stream_id=self.stream_id,
                              result_timestamp_ns=result.timestamp_ns,
                              result_timestamp_clock="unknown",
                              success=True, infer_failures=None,
                              per_frame_failures_observable=False,
                              latency_ms=None, source_age_ms=None,
                              sdk_latency_ms=getattr(gen, "last_latency_ms", 0) or None,
                              sdk_latency_clock_verified=False,
                              skew_us=getattr(gen, "last_skew_us", 0) or None,
                              hw_infer_time_us=getattr(result, "hw_infer_time_us", 0) or None,
                              queue_time_us=getattr(result, "queue_time_us", None),
                              daemon_infer_time_us=getattr(result, "infer_time_us", None))
                # SDK yields only successful results; it skips per-frame
                # failures and may eventually raise a subscription exception.
                # Its wall-clock latency diagnostic is unverified here, not a
                # source age or a sample for the validated latency summary.
                m = self.hub.a
                m.record_result(latency_ms=None,
                                skew_us=getattr(gen, "last_skew_us", 0) or None, now=now)
                m.counters.set("dropped", getattr(gen, "dropped", 0) or 0)
                m.counters.set("avg_latency_ms", None)
                m.counters.set("avg_skew_us", getattr(gen, "avg_skew_us", 0) or 0)
                m.counters.set("epoch", self._epoch)
                self._maybe_annotate(seq, result, now)
        finally:
            self.cancel_iter()
            self._iter = None
            close = getattr(gen, "close", None)
            if close:
                close()  # owning consumer thread, not concurrently executing next()

    def _annotate(self, seq, layer, detections, **kwargs) -> bool:
        started = time.monotonic_ns()
        success, error = False, None
        try:
            self.overlay.annotate(self.display_stream, detections, **kwargs)
            self.hub.a.counters.increment("annotate_calls")
            success = True
        except Exception as exc:
            error = type(exc).__name__
            self.hub.a.counters.increment("annotate_errors")
        self.hub.emit("a_annotate", source_frame_id=seq, stream_id=self.stream_id,
                      display_stream=self.display_stream, layer=layer,
                      annotate_ms=(time.monotonic_ns() - started) / 1e6,
                      success=success, error=error, stream_epoch=self._epoch)
        return success

    def _maybe_annotate(self, seq: int, result, now: float) -> None:
        if now - self._last_annotate < self._annotate_interval:
            return
        self._last_annotate = now
        if now >= self._epoch_next_check:
            self._refresh_epoch()
            self._epoch_next_check = now + EPOCH_REFRESH_S
        if self.detections_overlay:
            objs = [o for o in (result.objects or []) if (getattr(o, "score", 0) or 0) >= self.min_score]
            if not self._annotate(seq, "detections", objs,
                                  frame_sequence=seq if seq and seq > 0 else None,
                                  stream_epoch=self._epoch, session_id=self.session_id):
                self._refresh_epoch()
                return
        if self.metrics_overlay:
            snap = self.hub.snapshot()
            lines = [format_a_line(snap["a"], snap["sys"].get("stream_delta", {}).get(self.display_stream)),
                     format_status_line(snap["sys"], snap["uptime_s"])]
            self._annotate(seq, "metrics", metric_detections(lines), ttl_ms=2000, session_id=self.osd_session)

    def _refresh_epoch(self) -> None:
        try:
            for s in self.camera.get_stream_status():
                if s.stream_id == self.display_stream:
                    self._epoch = s.stream_epoch
                    return
        except Exception as exc:
            logger.debug("epoch refresh failed: %s", type(exc).__name__)

    def _clear_drawings(self) -> None:
        for enabled, session in ((self.detections_overlay, self.session_id),
                                 (self.metrics_overlay, self.osd_session)):
            if not enabled:
                continue
            try:
                self.overlay.annotate(self.display_stream, [], polygons=[], session_id=session)
            except Exception as exc:
                self.hub.a.counters.set("cleanup_error", type(exc).__name__)
