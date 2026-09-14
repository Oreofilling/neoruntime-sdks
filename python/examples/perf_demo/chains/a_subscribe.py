"""Chain A: platform-scheduled inference via InferenceClient.subscribe.

The app never touches pixels on this chain — it subscribes to results
on the infer stream (default ``third``), then burns the detections AND
its own metric lines through the platform overlay (annotate) onto the
display stream (default ``main``). Two annotate calls per tick keep the
semantics apart: detections bind to their frame (frame_sequence +
stream_epoch), the metric lines stay unbound with a TTL so they are
always visible. The two ride DIFFERENT session_ids because the daemon
layers by (session_id, source) — same identity replaces, different
stacks — so the unbound metric layer must not share the detection
layer's session.

Recovery model: a dead subscribe path (the -2814 deployment condition)
blocks inside next() forever, so the app watchdog cancels the iterator
from another thread (cancel() is documented cross-thread safe) and,
if no result ever arrived, degrades this chain — chain B keeps running.
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
    """Subscribe for results, annotate bound boxes + unbound metric bars."""

    def __init__(self, *, camera, infer, hub, stream_id: str = "third",
                 display_stream: str = "main", model_id: str = "",
                 fps: int = 10, annotate_hz: float = 3.0,
                 min_score: float = 0.3, session_id: str = "perf-demo") -> None:
        super().__init__(name="chain-a", daemon=True)
        self.camera = camera
        self.infer = infer
        self.hub = hub
        self.stream_id = stream_id
        self.display_stream = display_stream
        self.model_id = model_id
        self.fps = fps
        self.min_score = min_score
        self.session_id = session_id
        # separate layer identity: daemon stacks (session, source) layers,
        # and a same-session detections payload would replace the boxes
        self.osd_session = f"{session_id}-osd"
        self.overlay = OverlayClient()
        self._annotate_interval = 1.0 / max(annotate_hz, 0.1)
        self._stop = threading.Event()
        self._iter = None
        self._epoch: int | None = None
        self._epoch_next_check = 0.0
        self._last_annotate = 0.0
        self._t0 = time.monotonic()

    # -- control surface (app watchdog / teardown) --

    def stop(self) -> None:
        self._stop.set()
        self.cancel_iter()

    def cancel_iter(self) -> None:
        """Wake a consumer blocked in next() — cross-thread safe."""
        it = self._iter
        if it is not None:
            try:
                it.cancel()
            except Exception:  # noqa: BLE001 - watchdog path, never raise
                pass

    def degrade(self, reason: str) -> None:
        self.hub.a.counters.set("degraded_reason", reason)
        self.cancel_iter()

    # -- thread body --

    def run(self) -> None:  # noqa: C901 - the rebuild loop is the story
        try:
            while not self._stop.is_set() \
                    and self.hub.a.counters.get("degraded_reason") is None:
                try:
                    self._subscribe_loop()
                except StopIteration:
                    pass  # clean server-side end: fall through to rebuild
                except Exception as exc:  # noqa: BLE001 - count, backoff, retry
                    errs = (self.hub.a.counters.get("subscribe_errors") or 0) + 1
                    self.hub.a.counters.set("subscribe_errors", errs)
                    logger.warning("chain A subscribe loop ended: %s", exc)
                if self._stop.is_set() \
                        or self.hub.a.counters.get("degraded_reason") is not None:
                    break
                n = (self.hub.a.counters.get("reconnects") or 0) + 1
                self.hub.a.counters.set("reconnects", n)
                self._sleep_stop_aware(REBUILD_BACKOFF_S)
        finally:
            self._clear_drawings()

    def _subscribe_loop(self) -> None:
        self._refresh_epoch()
        gen = self.infer.subscribe(
            self.stream_id, self.model_id,
            fps=self.fps, session_id=self.session_id,
        )
        self._iter = gen
        for seq, result in gen:
            if self._stop.is_set():
                break
            now = time.monotonic()
            m = self.hub.a
            m.record_result(
                latency_ms=getattr(gen, "last_latency_ms", 0.0) or None,
                skew_us=getattr(gen, "last_skew_us", 0) or None,
                now=now,
            )
            m.counters.set("dropped", getattr(gen, "dropped", 0) or 0)
            if getattr(gen, "avg_latency_ms", 0.0):
                m.counters.set("avg_latency_ms", gen.avg_latency_ms)
            if getattr(gen, "avg_skew_us", 0.0):
                m.counters.set("avg_skew_us", gen.avg_skew_us)
            m.counters.set("epoch", self._epoch)
            self._maybe_annotate(seq, result, now)
        # generator exhausted: the outer loop reconnects

    # -- annotate side --

    def _maybe_annotate(self, seq: int, result, now: float) -> None:
        if now - self._last_annotate < self._annotate_interval:
            return
        self._last_annotate = now
        if now >= self._epoch_next_check:
            self._refresh_epoch()
            self._epoch_next_check = now + EPOCH_REFRESH_S
        m = self.hub.a
        # 1) bound detections: the frame-sync showcase (P1-2 semantics)
        try:
            objs = [o for o in (result.objects or [])
                    if (getattr(o, "score", 0.0) or 0.0) >= self.min_score]
            self.overlay.annotate(
                self.display_stream, objs,
                frame_sequence=seq if seq and seq > 0 else None,
                stream_epoch=self._epoch,
                session_id=self.session_id,
            )
            m.counters.set("annotate_calls",
                           (m.counters.get("annotate_calls") or 0) + 1)
        except Exception as exc:  # noqa: BLE001 - display must not kill the chain
            m.counters.set("annotate_errors",
                           (m.counters.get("annotate_errors") or 0) + 1)
            logger.debug("chain A detection annotate failed: %s", exc)
            self._refresh_epoch()  # a stale epoch is the usual suspect
            return
        # 2) unbound metric lines: always visible, TTL keeps them alive;
        #    own session so this layer stacks with the bound boxes above
        try:
            snap = self.hub.snapshot()
            lines = [
                format_a_line(snap["a"],
                              snap["sys"].get("stream_delta", {}).get(self.display_stream)),
                format_status_line(snap["sys"], snap["uptime_s"]),
            ]
            self.overlay.annotate(
                self.display_stream,
                detections=metric_detections(lines),
                ttl_ms=2000,
                session_id=self.osd_session,
            )
            m.counters.set("annotate_calls",
                           (m.counters.get("annotate_calls") or 0) + 1)
        except Exception as exc:  # noqa: BLE001
            m.counters.set("annotate_errors",
                           (m.counters.get("annotate_errors") or 0) + 1)
            logger.debug("chain A metric annotate failed: %s", exc)

    def _refresh_epoch(self) -> None:
        try:
            for s in self.camera.get_stream_status():
                if s.stream_id == self.display_stream:
                    self._epoch = s.stream_epoch
                    return
        except Exception as exc:  # noqa: BLE001
            logger.debug("epoch refresh failed: %s", exc)

    def _clear_drawings(self) -> None:
        """Graceful stop: clear BOTH of our layers — the bound boxes and
        the OSD metric layer (SIGKILL relies on the daemon's ttl expiry
        instead, ~2 s)."""
        for session in (self.session_id, self.osd_session):
            try:
                self.overlay.annotate(self.display_stream, [],
                                      polygons=[], session_id=session)
            except Exception:  # noqa: BLE001 - teardown path
                pass

    def _sleep_stop_aware(self, seconds: float) -> None:
        self._stop.wait(seconds)
