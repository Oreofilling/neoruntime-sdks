"""Thread-safe rolling metrics for the perf-demo app.

One MetricsHub per process; the chain threads add samples, the status
thread snapshots them for the burned-in OSD lines, the console status
line and the operator JSON file. All clocks are time.monotonic-based
except chain B's end-to-end latency, which subtracts the frame's device
CLOCK_MONOTONIC timestamp from a local clock_gettime(CLOCK_MONOTONIC)
reading — valid because the app runs on the device (same clock domain
as the daemon, hal_buffer.h:65-66).
"""

from __future__ import annotations

import threading
import time

from stats import percentile_stats


class Ring:
    """Fixed-capacity (monotonic_time, value) ring, one writer or many.

    Samples older than ``window_s`` are ignored by every reader, so a
    slow status thread never mixes stale numbers into a fresh window.
    """

    def __init__(self, capacity: int = 600, window_s: float = 60.0) -> None:
        self._lock = threading.Lock()
        self._buf: list[tuple[float, float]] = []
        self._capacity = capacity
        self._window_s = window_s

    def add(self, value: float, now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        with self._lock:
            self._buf.append((t, float(value)))
            if len(self._buf) > self._capacity:
                del self._buf[: len(self._buf) - self._capacity]

    def values(self, window_s: float | None = None, now: float | None = None) -> list[float]:
        t1 = time.monotonic() if now is None else now
        win = self._window_s if window_s is None else window_s
        with self._lock:
            return [v for t, v in self._buf if t1 - t <= win]

    def count(self, window_s: float | None = None, now: float | None = None) -> int:
        t1 = time.monotonic() if now is None else now
        win = self._window_s if window_s is None else window_s
        with self._lock:
            return sum(1 for t, _v in self._buf if t1 - t <= win)

    def rate(self, window_s: float = 10.0, now: float | None = None) -> float | None:
        """Events per second over the last ``window_s`` (None when empty)."""
        t1 = time.monotonic() if now is None else now
        with self._lock:
            recent = [t for t, _v in self._buf if t1 - t <= window_s]
        if not recent:
            return None
        span = max(t1 - min(recent), 1e-3)
        return len(recent) / span

    def summary(self, window_s: float = 10.0) -> dict:
        return percentile_stats(self.values(window_s))


class _Counters:
    """Last-writer-wins scalar fields with an atomic read.

    Explicit init values are honored (0 stays 0 — "none so far" — while
    None means "not yet known"); a key nobody ever set keeps its init.
    """

    def __init__(self, **fields) -> None:
        self._lock = threading.Lock()
        self._data = dict(fields)

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = value

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._data[key] = self._data.get(key, 0) + amount

    def get(self, key: str):
        with self._lock:
            return self._data[key]

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)


class ChainAMetrics:
    """subscribe-chain samples: result cadence, latency, skew, annotate."""

    def __init__(self) -> None:
        self.result_t = Ring(capacity=1200)      # result arrival (rate only)
        self.latency_ms = Ring()                 # iterator-reported receive latency
        self.skew_ms = Ring()                    # result.skew_us / 1000 (device clock)
        self.counters = _Counters(
            results=0, subscribe_errors=0, annotate_calls=0, annotate_errors=0,
            dropped=0, avg_latency_ms=None, avg_skew_us=0.0,
            infer_attempts=None, infer_failures=None,
            per_frame_failures_observable=False,
            failure_observability="sdk_skips_failed_frames",
            last_result_age_s=None, epoch=None, degraded_reason=None,
            reconnects=0, cleanup_error=None,
        )
        self._last_result_t: float | None = None

    def record_result(self, *, latency_ms: float | None, skew_us: int | None,
                      now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        self.result_t.add(0.0, t)
        self._last_result_t = t
        # cumulative count — the app watchdog's liveness probe reads this;
        # leaving it at the init 0 falsely trips "no first result in 60s"
        self.counters.set("results", (self.counters.get("results") or 0) + 1)
        if latency_ms is not None:
            self.latency_ms.add(latency_ms, t)
        if skew_us:
            self.skew_ms.add(skew_us / 1000.0, t)
        self.counters.set("last_result_age_s", 0.0)

    def refresh_age(self, now: float | None = None) -> None:
        if self._last_result_t is not None:
            t = time.monotonic() if now is None else now
            self.counters.set("last_result_age_s", t - self._last_result_t)

    def snapshot(self, window_s: float = 10.0) -> dict:
        self.refresh_age()
        return {
            "fps": self.result_t.rate(window_s),
            "latency": self.latency_ms.summary(window_s),
            "skew": self.skew_ms.summary(window_s),
            **self.counters.snapshot(),
        }


class ChainBMetrics:
    """keep-fd-chain samples: per-stage latency, e2e, injection state."""

    def __init__(self) -> None:
        self.frame_t = Ring(capacity=1200)       # processed-frame arrival (rate)
        self.pull_ms = Ring()                    # materialization only (legacy name)
        self.infer_ms = Ring()                   # resize + infer RPC (legacy aggregate)
        self.hw_infer_ms = Ring()                # result.hw_infer_time_us only
        self.draw_ms = Ring()                    # render_overlay_rgba + text chip + blend_hw
        self.pub_ms = Ring()                     # FramePublisher.publish
        self.e2e_ms = Ring()                     # compose done − source; NOT display
        self.counters = _Counters(
            frames_ok=0, frames_err=0, objects_last=0,
            inject_dropped=0, in_flight=0, pool_depth=0,
            retained_frames=None, lease_mode=None,
            degraded_reason=None, rebuilds=0, publishes=0,
            frames_delivered=0, frames_materialized=0, rate_skipped=0,
            decimation_skipped=0, infer_attempts=0, infer_success=0,
            infer_failures=0, composed_unique=0, publish_calls=0,
            publish_errors=0, published_unique=0, compose_errors=0,
            last_infer_success_ns=None, last_compose_ns=None,
            cleanup_error=None,
        )

    def record_frame(self, *, pull_ms: float, infer_ms: float, hw_infer_ms: float | None,
                     draw_ms: float, pub_ms: float | None,
                     e2e_ms: float, objects: int = 0,
                     now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        self.frame_t.add(0.0, t)
        self.pull_ms.add(pull_ms, t)
        self.infer_ms.add(infer_ms, t)
        if hw_infer_ms is not None:
            self.hw_infer_ms.add(hw_infer_ms, t)
        self.draw_ms.add(draw_ms, t)
        if pub_ms is not None:
            self.pub_ms.add(pub_ms, t)
        if e2e_ms is not None:
            self.e2e_ms.add(e2e_ms, t)
        self.counters.set("objects_last", objects)

    def snapshot(self, window_s: float = 10.0) -> dict:
        return {
            "fps": self.frame_t.rate(window_s),
            "pull": self.pull_ms.summary(window_s),
            "infer": self.infer_ms.summary(window_s),
            "hw_infer": self.hw_infer_ms.summary(window_s),
            "draw": self.draw_ms.summary(window_s),
            "pub": self.pub_ms.summary(window_s),
            "e2e": self.e2e_ms.summary(window_s),  # compatibility alias only
            "compose_age": self.e2e_ms.summary(window_s),
            "e2e_semantics": "source_to_compose_done_not_display",
            **self.counters.snapshot(),
        }


class SystemMetrics:
    """Slow-rate platform counters sampled by the status thread."""

    def __init__(self) -> None:
        self.counters = _Counters(
            npu_util=None, dsp_util=None, temp_c=None, cpu_util=None,
            model_avg_latency_us=None, model_hw_fps=None, model_qps=None,
            hw_fallbacks=None,
        )
        # display-stream counter deltas over one status interval
        self.stream_delta: dict[str, dict] = {}
        # encoded-stream inter-packet gap samples per stream (ms)
        self.encode_gap: dict[str, Ring] = {}

    def note_encode_gap(self, stream_id: str, gap_ms: float) -> None:
        ring = self.encode_gap.get(stream_id)
        if ring is None:
            ring = self.encode_gap[stream_id] = Ring(capacity=600)
        ring.add(gap_ms)

    def snapshot(self) -> dict:
        return {
            **self.counters.snapshot(),
            "stream_delta": {k: dict(v) for k, v in self.stream_delta.items()},
            "encode_gap": {
                k: r.summary(60.0) for k, r in self.encode_gap.items()
            },
        }


class MetricsHub:
    """Everything the demo measures, in one lock-guarded object."""

    def __init__(self, recorder=None) -> None:
        self.recorder = recorder
        self.started_monotonic = time.monotonic()
        self.started_wall = time.time()
        self.a = ChainAMetrics()
        self.b = ChainBMetrics()
        self.sys = SystemMetrics()
        self._lock = threading.Lock()
        self._final: dict | None = None

    def emit(self, kind: str, **fields) -> None:
        if self.recorder is not None:
            self.recorder.emit(kind, **fields)

    def uptime_s(self) -> float:
        return time.monotonic() - self.started_monotonic

    def snapshot(self, window_s: float = 10.0) -> dict:
        """One consistent view for OSD lines, console and the JSON file."""
        with self._lock:
            final = self._final
        snap = {
            "window_s": window_s,
            "uptime_s": round(self.uptime_s(), 1),
            "a": self.a.snapshot(window_s),
            "b": self.b.snapshot(window_s),
            "sys": self.sys.snapshot(),
            "recorder": self.recorder.snapshot() if self.recorder else None,
        }
        if final:
            snap["final"] = final
        return snap

    def record_final(self, reason: str) -> None:
        with self._lock:
            self._final = {"reason": reason, "uptime_s": round(self.uptime_s(), 1)}
