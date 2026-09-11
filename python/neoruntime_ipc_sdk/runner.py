"""Long-running application skeleton around an :class:`InferencePipeline`.

Productises the B-form loop (see ``examples/local_inference_app.py`` and
``python/PERFORMANCE.md``): a reader thread pulls frames from any
iterable — typically ``FdMediaClient().subscribe("sub", keep_fd=True)``
— into a tiny latest-wins queue, and a worker thread runs the pipeline
and hands each result to a sink. When processing is slower than the
camera, old frames are dropped instead of ballooning memory, keep-fd
handles are released the moment a frame is consumed, and everything is
visible through :attr:`PipelineRunner.stats`.

Frame ownership: frames yielded by ``source`` belong to the runner for
the duration of one pass. The runner releases keep-fd handles after the
sink returns (evicted frames are released immediately), so **a sink must
not retain the frame beyond the call** — copy what you need
(``frame.to_array()`` already copies).

.. code-block:: python

    from neoruntime_ipc_sdk import (
        FdMediaClient, InferencePipeline, PipelineRunner, draw_detections,
    )

    media = FdMediaClient()
    runner = PipelineRunner(
        source=media.subscribe("sub", keep_fd=True),
        pipeline=InferencePipeline.from_model("person_v1"),
        sink=lambda out, frame: print(out.objects),
        queue_size=1,            # latest-wins: always infer the newest frame
    )
    runner.start()
    ...
    runner.stop()               # or: with PipelineRunner(...) as runner: ...
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Iterable

__all__ = ["PipelineRunner"]

logger = logging.getLogger(__name__)

_SENTINEL = object()

# Recent-latency exponential moving average factor (per successful pass).
_EMA_ALPHA = 0.2

# Grace period before shedding: a consumer that is merely *between* frames
# (not genuinely behind) must not cause a drop, so the reader offers each
# frame with this timeout and only evicts the oldest buffered frame when it
# expires — sustained backlog, not momentary races, is what sheds frames.
_PUT_GRACE_S = 0.05

def _release(frame: Any) -> None:
    """Release a frame's resources if it has any (keep-fd dma-bufs)."""
    release = getattr(frame, "release", None)
    if release is None:
        return
    try:
        release()
    except Exception:
        logger.exception("PipelineRunner: frame release failed")


class PipelineRunner:
    """Run ``source → pipeline → sink`` continuously on two threads.

    Args:
        source: iterable of frames (a blocking generator such as
            :meth:`FdMediaClient.subscribe
            <neoruntime_ipc_sdk.FdMediaClient.subscribe>` is the norm).
        pipeline: anything with a ``run(frame)`` method — an
            :class:`~neoruntime_ipc_sdk.InferencePipeline`.
        sink: ``callable(result, frame)`` invoked per pass on the worker
            thread. ``None`` collects results into ``last_results``
            (bounded) for inspection.
        queue_size: frames buffered between reader and worker (default 1).
            Latest-wins: when full, the oldest buffered frame is dropped
            and counted, so the worker always sees the newest frame.
        drop_policy: only ``"latest"`` is supported — freshness beats
            completeness for live video.
        on_error: ``callable(exc)`` per pipeline/sink error, in addition
            to the throttled log and the counters.
        max_consecutive_errors: stop the runner after this many pipeline
            or sink errors in a row (default 10; 0/None disables).
        name: thread-name prefix and stats tag.
    """

    def __init__(
        self,
        source: Iterable[Any],
        pipeline: Any,
        sink: Callable[[Any, Any], None] | None = None,
        *,
        queue_size: int = 1,
        drop_policy: str = "latest",
        on_error: Callable[[Exception], None] | None = None,
        max_consecutive_errors: int | None = 10,
        name: str = "pipeline",
    ):
        if drop_policy != "latest":
            raise ValueError(
                f"unsupported drop_policy {drop_policy!r} — only 'latest' "
                "(drop the oldest buffered frame, keep the newest) is supported"
            )
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")
        self.source = source
        self.pipeline = pipeline
        self.sink = sink
        self.queue_size = queue_size
        self.on_error = on_error
        self.max_consecutive_errors = max_consecutive_errors
        self.name = name

        self._q: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._worker: threading.Thread | None = None

        # Stats. Each counter has a single writer thread (dropped: reader,
        # the rest: worker); reads from other threads are advisory snapshots.
        self.processed = 0
        self.dropped = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.last_error: str | None = None
        self.started_at = 0.0
        self._total_latency_ms = 0.0
        self._latency_ema = 0.0
        self.last_results: list[Any] = []  # bounded sink-less capture

    # -- lifecycle -------------------------------------------------------

    @property
    def running(self) -> bool:
        return any(
            t is not None and t.is_alive() for t in (self._reader, self._worker)
        )

    def start(self) -> PipelineRunner:
        if self._reader is not None:
            raise RuntimeError("PipelineRunner already started")
        self.started_at = time.time()
        self._stop.clear()
        self._reader = threading.Thread(
            target=self._reader_loop, name=f"{self.name}-reader", daemon=True
        )
        self._worker = threading.Thread(
            target=self._worker_loop, name=f"{self.name}-worker", daemon=True
        )
        self._worker.start()
        self._reader.start()
        return self

    def stop(self, timeout_s: float = 5.0) -> bool:
        """Signal stop, unblock the worker and join both threads.

        The reader may sit inside ``next(source)`` until the source
        yields again (its own receive timeout); both threads are daemons
        regardless. Returns True when both threads exited in time.
        """
        self._stop.set()
        self._enqueue_terminal()
        deadline = time.monotonic() + timeout_s
        for thread in (self._reader, self._worker):
            if thread is None:
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not self.running

    def __enter__(self) -> PipelineRunner:
        if self._reader is None:
            self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- stats ------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Snapshot: throughput, drops, latency and error counters."""
        uptime = (time.time() - self.started_at) if self.started_at else 0.0
        return {
            "name": self.name,
            "running": self.running,
            "processed": self.processed,
            "dropped": self.dropped,
            "fps": (self.processed / uptime) if uptime > 0 else 0.0,
            "avg_latency_ms": (
                self._total_latency_ms / self.processed if self.processed else 0.0
            ),
            "recent_latency_ms": round(self._latency_ema, 2),
            "errors": self.errors,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "uptime_s": round(uptime, 1),
        }

    # -- threads ----------------------------------------------------------

    def _enqueue_terminal(self) -> None:
        """Put the stop sentinel through, shedding a buffered frame if needed."""
        while True:
            try:
                self._q.put_nowait(_SENTINEL)
                return
            except queue.Full:
                try:
                    _release(self._q.get_nowait())
                    self.dropped += 1
                except queue.Empty:
                    continue

    def _reader_loop(self) -> None:
        source = iter(self.source)
        exhausted = False
        while not self._stop.is_set():
            try:
                frame = next(source)
            except StopIteration:
                exhausted = True
                break  # source exhausted → normal end
            except Exception as exc:  # subscribe already reconnects internally
                self._record_error(exc, "source")
                break
            if frame is None:
                continue
            while not self._stop.is_set():
                try:
                    self._q.put(frame, timeout=_PUT_GRACE_S)
                    break
                except queue.Full:
                    try:
                        _release(self._q.get_nowait())
                        self.dropped += 1
                    except queue.Empty:
                        continue
        if exhausted and not self._stop.is_set():
            # Natural end: buffered frames are pending work, not garbage —
            # block until the worker drains room for the sentinel.
            self._q.put(_SENTINEL)
        else:
            self._enqueue_terminal()

    def _worker_loop(self) -> None:
        while True:
            frame = self._q.get()
            if frame is _SENTINEL:
                break
            started = time.perf_counter()
            try:
                result = self.pipeline.run(frame)
                if self.sink is not None:
                    self.sink(result, frame)
                else:
                    self.last_results.append(result)
                    del self.last_results[:-16]  # bounded
            except Exception as exc:
                self._record_error(exc, "pipeline")
                if (
                    self.max_consecutive_errors
                    and self.consecutive_errors >= self.max_consecutive_errors
                ):
                    logger.error(
                        "PipelineRunner(%s): %d consecutive errors, stopping "
                        "(last: %s)",
                        self.name,
                        self.consecutive_errors,
                        self.last_error,
                    )
                    self._stop.set()
                    break
            else:
                self._note_success((time.perf_counter() - started) * 1000.0)
            finally:
                _release(frame)
        # Drain whatever the reader left behind, releasing its handles.
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is not _SENTINEL:
                _release(item)

    # -- accounting ---------------------------------------------------------

    def _note_success(self, latency_ms: float) -> None:
        self.processed += 1
        self.consecutive_errors = 0
        self._total_latency_ms += latency_ms
        if self._latency_ema == 0.0:
            self._latency_ema = latency_ms
        else:
            self._latency_ema += _EMA_ALPHA * (latency_ms - self._latency_ema)

    def _record_error(self, exc: Exception, stage: str) -> None:
        self.errors += 1
        self.consecutive_errors += 1
        self.last_error = f"{stage}: {exc}"
        if self.consecutive_errors == 1 or self.consecutive_errors % 10 == 0:
            logger.warning(
                "PipelineRunner(%s): %s error (%d consecutive): %s",
                self.name,
                stage,
                self.consecutive_errors,
                exc,
            )
        if self.on_error is not None:
            try:
                self.on_error(exc)
            except Exception:
                logger.exception("PipelineRunner: on_error callback failed")
