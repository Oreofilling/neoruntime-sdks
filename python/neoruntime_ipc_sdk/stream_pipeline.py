"""Whole-pipeline facade: one call wiring capture -> inference -> overlay.

Positioning (read this before reaching for :class:`InferencePipeline` in
``pipeline.py``): this module is the *platform-scheduled* pipeline. The
camera daemon feeds frames to the model (``InferenceClient.subscribe``),
so no pixels ever cross into the app process; results come back over the
stream, get pushed to the hardware overlay via the event bus
(``OverlayClient``), and the annotated video ships inside the encoded
stream — the app only ever sees lightweight ``InferenceResult`` structs.
``InferencePipeline``, by contrast, is the *client-side* convenience: it
pulls frames into the app, runs pre->infer->post on the app thread, and
is the right tool when the app needs the pixels (custom preprocessing,
multi-model chains on the same frame, drawing with its own rasterizer).

Choose by where the pixels should live: keep them on the device and take
results — ``StreamPipeline``; need them in your process —
``InferencePipeline``.

One instance is single-shot: ``start()`` once, ``stop()`` once; build a
new instance for another run. Typical use::

    zones = [{"points": [[0, 0], [1, 0], [1, 1]], "label": "yard"}]
    pipe = StreamPipeline(
        "main", "yolov5n", fps=10,
        min_score=0.5, labels=["person"], polygons=zones,
        on_result=lambda r: r if r.count_by_label("person") else None,
    )
    pipe.start()
    for seq, result in pipe.results():   # optional: results in the app too
        ...
    pipe.stop()                          # clears boxes + zones on the stream
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterator

from .inference import InferenceClient
from .inference_types import InferenceResult
from .overlay import OverlayClient, OverlayConfig

logger = logging.getLogger(__name__)

# Terminal item for the app-facing result queue. Offered exactly once,
# after every result, so backpressure can delay but never drop it: the
# consumer's results() generator always terminates.
_STOP = object()


@dataclass(frozen=True)
class StreamPipelineStatus:
    """Point-in-time health snapshot from :meth:`StreamPipeline.status`.

    ``results_seen`` counts every result the worker accepted from the
    subscription, *before* the ``on_result`` hook runs — results the hook
    dropped or replaced are included. ``last_latency_ms``/``avg_latency_ms``
    likewise time every accepted result, including ones later dropped by
    the hook.
    """

    running: bool
    results_seen: int
    results_annotated: int
    annotate_errors: int
    result_queue_drops: int
    subscribe_dropped: int
    last_error: str
    last_latency_ms: float
    avg_latency_ms: float
    last_skew_us: int
    avg_skew_us: float


class StreamPipeline:
    """One-call pipeline: subscribe, draw on the hardware overlay, observe.

    Args:
        stream: camera stream id the model runs on (e.g. ``"main"``).
        model: registered model id.
        fps: result-rate cap for the subscription.
        session_id: optional inference session id (observability/logging).
        raw_output_only: skip post-processing, hand back raw tensors.
        max_consecutive_failures: consecutive failed frames tolerated
            before the worker gives up (0/None = unlimited); see
            :meth:`InferenceClient.subscribe`.
        queue_size: bound for the subscribe-side result queue.
        draw: push results to the hardware overlay. ``False`` also
            suppresses static ``polygons`` — the encoded stream stays
            clean; results still flow to :meth:`results`.
        overlay_config: optional :class:`OverlayConfig` applied at start
            (labels, colors, strict frame-lock, ...).
        ttl_ms: positive int, per-event validity override forwarded to
            every annotate call; omitted = the daemon derives it from
            the stream fps.
        min_score: draw only detections with ``score >= min_score``.
        labels: draw only these labels (``None`` = all). Filtering
            affects *drawing only* — :meth:`results` yields the full
            result — and a fully-filtered result still publishes an
            empty detection list, clearing stale boxes.
        polygons: static zone polygons pushed once at start and cleared
            at stop (normalized [0,1] points; see
            :meth:`OverlayClient.annotate`).
        on_result: hook called per result on the worker thread. Return a
            (possibly replaced) result to flow onward, or ``None`` to
            drop it (not drawn, not queued) — put tracking/filtering
            logic here. A raised exception drops that result, is
            recorded in ``status().last_error`` and is non-fatal.
        result_queue_size: bound for the app-facing queue behind
            :meth:`results`; a slow app drops its oldest queued results
            (counted in ``status().result_queue_drops``).
        inference: inject an :class:`InferenceClient` (tests / client
            reuse). Injected clients are NOT closed by :meth:`stop`.
        overlay: inject an :class:`OverlayClient`. Not closed on stop.
    """

    def __init__(
        self,
        stream: str,
        model: str,
        *,
        fps: int = 10,
        session_id: str = "",
        raw_output_only: bool = False,
        max_consecutive_failures: int | None = 10,
        queue_size: int | None = 100,
        draw: bool = True,
        overlay_config: OverlayConfig | None = None,
        ttl_ms: int | None = None,
        min_score: float = 0.0,
        labels: list[str] | None = None,
        polygons: list[dict] | None = None,
        on_result: Callable[[InferenceResult], InferenceResult | None] | None = None,
        result_queue_size: int = 100,
        inference: InferenceClient | None = None,
        overlay: OverlayClient | None = None,
    ) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {min_score!r}")
        if result_queue_size < 1:
            raise ValueError(
                f"result_queue_size must be >= 1, got {result_queue_size!r}"
            )
        if ttl_ms is not None and (
            isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0
        ):
            raise ValueError(f"ttl_ms must be a positive int, got {ttl_ms!r}")
        self.stream = stream
        self.model = model
        self.fps = fps
        self.session_id = session_id
        self.raw_output_only = raw_output_only
        self.max_consecutive_failures = max_consecutive_failures
        self.queue_size = queue_size
        self.draw = draw
        self.overlay_config = overlay_config
        self.ttl_ms = ttl_ms
        self.min_score = min_score
        self.labels = list(labels) if labels else None
        self.polygons = polygons
        self.on_result = on_result
        self.result_queue_size = result_queue_size

        self._filter_active = min_score > 0.0 or self.labels is not None

        self._inference = inference
        self._overlay = overlay
        self._owns_inference = inference is None
        self._owns_overlay = overlay is None

        self._thread: threading.Thread | None = None
        self._iterator = None
        self._results: queue.Queue | None = None
        self._lock = threading.Lock()
        self._started = False  # single-shot: set once, never cleared
        self._stopped = False  # stop() idempotency, separate from _started
        self._running = False
        self._results_seen = 0
        self._results_annotated = 0
        self._annotate_errors = 0
        self._result_queue_drops = 0
        self._last_error = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> StreamPipeline:
        """Subscribe, apply overlay config / static zones, start drawing."""
        with self._lock:
            if self._started:
                raise RuntimeError(
                    "this StreamPipeline was already started; "
                    "create a new instance for another run"
                )
            self._started = True
            self._running = True

        if self._inference is None:
            self._inference = InferenceClient()
        if self.draw:
            if self._overlay is None:
                self._overlay = OverlayClient()
            if self.overlay_config is not None:
                self._overlay.apply(self.overlay_config)
            if self.polygons is not None:
                self._overlay.annotate(self.stream, polygons=self.polygons)

        self._results = queue.Queue(maxsize=self.result_queue_size)
        self._thread = threading.Thread(
            target=self._run, name=f"stream-pipeline-{self.stream}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the worker, clear what was drawn, close owned clients.

        Idempotent, and a no-op before ``start()``. The worker is joined
        first so the box-clearing annotate cannot race a final result.
        Safe to call from ``on_result`` (i.e. from the worker thread
        itself): the self-join is skipped and the teardown below runs on
        the caller's thread.
        """
        with self._lock:
            if not self._started or self._stopped:
                return
            self._stopped = True  # a second stop() is a no-op
        iterator = self._iterator
        if iterator is not None:
            try:
                iterator.cancel()
            except Exception:  # noqa: BLE001 — best-effort wake-up
                logger.debug("cancel() during stop failed", exc_info=True)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
            if thread.is_alive():
                logger.error(
                    "StreamPipeline(stream=%r) worker did not exit within 5s",
                    self.stream,
                )

        overlay = self._overlay
        if overlay is not None:
            try:
                if self.draw:
                    overlay.annotate(self.stream, detections=[])  # clear boxes
                if self.draw and self.polygons is not None:
                    overlay.annotate(self.stream, polygons=[])  # clear zones
            except Exception as exc:  # noqa: BLE001 — teardown must not raise
                logger.warning(
                    "StreamPipeline.stop(): clearing overlay on stream %r failed: %s",
                    self.stream,
                    exc,
                )

        if self._owns_inference and self._inference is not None:
            try:
                self._inference.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("closing owned InferenceClient failed: %s", exc)
        if self._owns_overlay and overlay is not None:
            try:
                overlay.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("closing owned OverlayClient failed: %s", exc)

        self._thread = None

    def __enter__(self) -> StreamPipeline:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------

    def results(self) -> Iterator[tuple[int, InferenceResult]]:
        """Yield ``(frame_sequence, InferenceResult)`` post-callback.

        Ends when the worker stops (``stop()`` or a fatal stream error).
        The queue is bounded by ``result_queue_size`` with drop-oldest
        overflow — a slow consumer loses its oldest results, counted in
        ``status().result_queue_drops``. The pre-start check is eager:
        it raises at call time, not on the first ``next()``.
        """
        if self._results is None:
            raise RuntimeError("start() the StreamPipeline before reading results")
        return self._results_gen()

    def _results_gen(self) -> Iterator[tuple[int, InferenceResult]]:
        while True:
            item = self._results.get()
            if item is _STOP:
                return
            yield item

    def status(self) -> StreamPipelineStatus:
        """Snapshot of pipeline health and pass-through subscribe stats."""
        iterator = self._iterator
        with self._lock:
            return StreamPipelineStatus(
                running=self._running,
                results_seen=self._results_seen,
                results_annotated=self._results_annotated,
                annotate_errors=self._annotate_errors,
                result_queue_drops=self._result_queue_drops,
                subscribe_dropped=getattr(iterator, "dropped", 0),
                last_error=self._last_error,
                last_latency_ms=getattr(iterator, "last_latency_ms", 0.0),
                avg_latency_ms=getattr(iterator, "avg_latency_ms", 0.0),
                last_skew_us=getattr(iterator, "last_skew_us", 0),
                avg_skew_us=getattr(iterator, "avg_skew_us", 0.0),
            )

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._iterator = self._inference.subscribe(
                self.stream,
                self.model,
                fps=self.fps,
                session_id=self.session_id,
                raw_output_only=self.raw_output_only,
                max_consecutive_failures=self.max_consecutive_failures,
                queue_size=self.queue_size,
            )
            for seq, result in self._iterator:
                self._handle(seq, result)
        except Exception as exc:  # noqa: BLE001 — surfaced via status/last_error
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "StreamPipeline(stream=%r, model=%r) worker ended: %s",
                self.stream,
                self.model,
                exc,
            )
        finally:
            with self._lock:
                self._running = False
            self._offer(_STOP)

    def _handle(self, seq: int, result: InferenceResult) -> None:
        with self._lock:
            self._results_seen += 1

        if self.on_result is not None:
            try:
                replaced = self.on_result(result)
            except Exception as exc:  # noqa: BLE001 — app hook must not kill us
                with self._lock:
                    self._last_error = f"on_result raised {type(exc).__name__}: {exc}"
                logger.warning(
                    "StreamPipeline(stream=%r): on_result raised for frame %d; "
                    "result dropped",
                    self.stream,
                    seq,
                )
                return
            if replaced is None:
                return  # hook says drop: neither drawn nor queued
            result = replaced

        if self.draw:
            self._annotate(result)

        self._offer((seq, result))

    def _annotate(self, result: InferenceResult) -> None:
        overlay = self._overlay
        assert overlay is not None  # created in start() whenever draw
        try:
            if self._filter_active:
                # Filtered detections go out directly: an empty kept-list
                # must publish (clearing stale boxes) instead of falling
                # through annotate_result's classifications precedence.
                kept = [
                    obj
                    for obj in result.objects
                    if obj.score >= self.min_score
                    and (self.labels is None or obj.label in self.labels)
                ]
                overlay.annotate(
                    self.stream, kept, ttl_ms=self.ttl_ms, session_id=self.session_id
                )
            else:
                overlay.annotate_result(
                    self.stream, result, ttl_ms=self.ttl_ms, session_id=self.session_id
                )
        except Exception as exc:  # noqa: BLE001 — drawing is never fatal
            with self._lock:
                self._annotate_errors += 1
                self._last_error = f"annotate failed: {type(exc).__name__}: {exc}"
                errors = self._annotate_errors
            if errors == 1 or errors % 10 == 0:
                logger.warning(
                    "StreamPipeline(stream=%r): annotate error #%d: %s",
                    self.stream,
                    errors,
                    exc,
                )
            return
        with self._lock:
            self._results_annotated += 1

    def _offer(self, item) -> None:
        # Drop-oldest overflow, same policy as the subscribe queue: the
        # newest result always gets through; _STOP (the one terminal
        # item, offered last) can evict but never be evicted.
        results = self._results
        while True:
            try:
                results.put_nowait(item)
                return
            except queue.Full:
                try:
                    results.get_nowait()
                    with self._lock:
                        self._result_queue_drops += 1
                        drops = self._result_queue_drops
                    if drops == 1 or drops % 10 == 0:
                        logger.warning(
                            "StreamPipeline(stream=%r): results() consumer "
                            "behind; dropped %d queued results",
                            self.stream,
                            drops,
                        )
                except queue.Empty:
                    pass  # consumer drained between put and get; retry
