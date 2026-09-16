"""Status thread: platform counters, stream deltas, operator JSON.

Every interval this thread samples the slow-rate platform state —
stream-status counter deltas (bake coverage, late commands), injection
counters, get_stats utilization (small sampling window; the call blocks
only inside this thread) — folds them into the MetricsHub, prints the
two burn-in lines to the console, and atomically rewrites the operator
JSON file. Chain A reads the stream deltas back out of the hub for its
burn-in text, so no extra RPC happens on the annotate path.

The encoded-stream watchers are separate daemon threads: one per
watched display stream, consuming EncodedStreamClient packets and
recording inter-packet pts gaps — the daemon-side bake/draw cost shows
up as the gap tail on the A display stream (test_65 T02 method).
"""

from __future__ import annotations

import json
import logging
import os
import threading

from display import format_a_line, format_b_line

logger = logging.getLogger("perf_demo.status")

GAP_SANITY_MS = 10_000.0  # ignore absurd gaps (startup, reconnect)
WATCHER_REBUILD_BACKOFF_S = 0.5
WATCHER_REBUILD_LIMIT = 10


class StatusLine(threading.Thread):
    """Sample platform state into the hub; print and persist it."""

    def __init__(self, *, hub, camera, infer, model_id: str,
                 json_path: str = "/run/aipc/perf-demo.json",
                 interval: float = 5.0,
                 a_display: str = "main", b_stream: str = "sub") -> None:
        super().__init__(name="statusline", daemon=True)
        self.hub = hub
        self.camera = camera
        self.infer = infer
        self.model_id = model_id
        self.json_path = json_path
        self.interval = interval
        self.a_display = a_display
        self.b_stream = b_stream
        self._stop_event = threading.Event()
        self._last_counters: dict[str, dict] = {}  # stream_id -> raw counters

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - status must never kill the demo
                logger.exception("status tick failed")
        # one final tick so the JSON file reflects teardown state
        try:
            self.tick()
        except Exception:  # noqa: BLE001
            pass

    def tick(self) -> dict:
        self._sample_stream_deltas()
        self._sample_injection()
        self._sample_system()
        snap = self.hub.snapshot()
        a_line = format_a_line(
            snap["a"],
            snap["sys"].get("stream_delta", {}).get(self.a_display))
        b_line = format_b_line(snap["b"])
        print(f"[perf-demo] {a_line}", flush=True)
        print(f"[perf-demo] {b_line}", flush=True)
        self._write_json(snap)
        self.hub.emit("status", snapshot=snap)
        return snap

    # -- samplers --

    def _sample_stream_deltas(self) -> None:
        try:
            statuses = {s.stream_id: s for s in self.camera.get_stream_status()}
        except Exception as exc:  # noqa: BLE001 - keep previous deltas
            logger.debug("get_stream_status failed: %s", exc)
            return
        for stream_id, s in statuses.items():
            cur = {
                "packets_published": s.packets_published,
                "bake_skips": s.bake_skips,
                "overlay_late_commands": getattr(s, "overlay_late_commands", 0),
                "stream_epoch": getattr(s, "stream_epoch", None),
            }
            prev = self._last_counters.get(stream_id)
            self._last_counters[stream_id] = cur
            reset = prev is not None and (cur["stream_epoch"] != prev["stream_epoch"] or any(
                cur[k] < prev[k] for k in ("packets_published", "bake_skips", "overlay_late_commands")))
            if prev is None or reset:
                self.hub.sys.stream_delta[stream_id] = dict(
                    packets=None, bake_skips=None, overlay_late_commands=None,
                    epoch=cur["stream_epoch"], bake_pct=None, reset=reset)
                continue
            d_packets = cur["packets_published"] - prev["packets_published"]
            d_skips = cur["bake_skips"] - prev["bake_skips"]
            delta = {
                "packets": d_packets,
                "bake_skips": d_skips,
                "overlay_late_commands":
                    cur["overlay_late_commands"] - prev["overlay_late_commands"],
                "epoch": cur["stream_epoch"],
                "bake_pct": None,
            }
            if d_packets > 0:
                delta["bake_pct"] = round(
                    max(0.0, 100.0 * (1.0 - d_skips / d_packets)), 1)
            self.hub.sys.stream_delta[stream_id] = delta

    def _sample_injection(self) -> None:
        try:
            st = self.camera.injection_status()
        except Exception as exc:  # noqa: BLE001
            logger.debug("injection_status failed: %s", exc)
            return
        b = self.hub.b
        b.counters.set("inject_dropped", st.frames_dropped)
        b.counters.set("in_flight", len(st.in_flight_buffer_ids))

    def _sample_system(self) -> None:
        s = self.hub.sys.counters
        for key in ("npu_util", "dsp_util", "cpu_util", "temp_c", "model_avg_latency_us", "model_hw_fps", "model_qps"):
            s.set(key, None)
        s.set("sample_error", None)
        try:
            stats = self.infer.get_stats(sampling_window_ms=50)
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_stats failed: %s", exc)
            s.set("sample_error", type(exc).__name__)
            return
        s = self.hub.sys.counters
        s.set("npu_util", stats.get("device_utilization"))
        s.set("dsp_util", stats.get("dsp_utilization"))
        s.set("cpu_util", stats.get("cpu_utilization"))
        s.set("temp_c", stats.get("device_temperature"))
        for m in stats.get("model_stats", []):
            if m.get("model_id") == self.model_id:
                s.set("model_avg_latency_us", m.get("avg_latency_us"))
                s.set("model_hw_fps", m.get("hw_fps"))
                s.set("model_qps", m.get("current_qps"))
                break

    def _write_json(self, snap: dict) -> None:
        path = self.json_path
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=1, default=str)
            os.replace(tmp, path)
        except OSError as exc:  # noqa: BLE001 - JSON is best-effort
            logger.debug("json write failed: %s", exc)


class EncodedWatcher(threading.Thread):
    """Bounded polling; the worker owns and closes its socket on every path."""

    def __init__(self, stream_id: str, hub) -> None:
        super().__init__(name=f"enc-watch-{stream_id}", daemon=True)
        self.stream_id, self.hub = stream_id, hub
        self._stop_event = threading.Event()
        self.cleanup_error = None
        self.terminal_error = None
        self._consecutive_rebuilds = 0

    def stop(self) -> None:
        self._stop_event.set()

    def close(self, timeout: float = 2.0) -> bool:
        self.stop()
        self.join(timeout)
        return not self.is_alive()

    def run(self) -> None:
        from neoruntime_ipc_sdk import EncodedStreamClient
        while not self._stop_event.is_set():
            client = None
            reason = "no_packet_timeout_eof_or_invalid"
            try:
                client = EncodedStreamClient(stream_id=self.stream_id)
                self._consume(client)
            except Exception as exc:
                reason = type(exc).__name__
                self.hub.emit("watcher_error", stream_id=self.stream_id, error=reason)
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception as exc:
                        self.cleanup_error = type(exc).__name__
            if self._stop_event.is_set():
                break
            self._consecutive_rebuilds += 1
            self.hub.emit("watcher_reconnect", stream_id=self.stream_id, reason=reason,
                          consecutive_rebuilds=self._consecutive_rebuilds)
            if self._consecutive_rebuilds >= WATCHER_REBUILD_LIMIT:
                self.terminal_error = "reconnect_limit"
                break
            self._stop_event.wait(WATCHER_REBUILD_BACKOFF_S)
        self.hub.emit("watcher_exit", stream_id=self.stream_id,
                      error=self.cleanup_error or self.terminal_error)

    def _consume(self, client) -> None:
        import time
        last_pts, last_arrival = None, None
        while not self._stop_event.is_set():
            pkt = client.get_frame(timeout_ms=500)
            if pkt is None:
                # SDK deliberately conflates EOF, timeout and invalid/partial
                # packets. Rebuild all of them: retrying EOF spins forever,
                # and a partial read may no longer be on a framing boundary.
                return
            self._consecutive_rebuilds = 0
            now = time.monotonic_ns()
            gap = (pkt.pts_ns - last_pts) / 1e6 if last_pts is not None else None
            if gap is not None and 0 < gap < GAP_SANITY_MS:
                self.hub.sys.note_encode_gap(self.stream_id, gap)
            self.hub.emit("encoded_packet", stream_id=self.stream_id,
                          packet_sequence=getattr(pkt, "seq", None),
                          pts_ns=pkt.pts_ns, pts_gap_ms=gap,
                          arrival_gap_ms=(now - last_arrival) / 1e6 if last_arrival else None)
            last_pts, last_arrival = pkt.pts_ns, now


def start_encoded_watcher(stream_id: str, hub) -> EncodedWatcher:
    watcher = EncodedWatcher(stream_id, hub)
    watcher.start()
    return watcher
