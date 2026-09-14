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
        self._stop = threading.Event()
        self._last_counters: dict[str, dict] = {}  # stream_id -> raw counters

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
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
            if prev is None:
                continue  # first reading: establish the baseline only
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
        try:
            stats = self.infer.get_stats(sampling_window_ms=50)
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_stats failed: %s", exc)
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
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=1, default=str)
            os.replace(tmp, path)
        except OSError as exc:  # noqa: BLE001 - JSON is best-effort
            logger.debug("json write failed: %s", exc)


def start_encoded_watcher(stream_id: str, hub) -> threading.Thread:
    """Watch one encoded stream: record inter-packet pts gaps into the hub."""

    _watch_stop = threading.Event()

    def _run() -> None:
        from neoruntime_ipc_sdk import EncodedStreamClient

        last_pts = 0
        while not _watch_stop.is_set():
            try:
                client = EncodedStreamClient(stream_id=stream_id)
                for pkt in client.subscribe():
                    if _watch_stop.is_set():
                        break
                    if last_pts and pkt.pts_ns > last_pts:
                        gap_ms = (pkt.pts_ns - last_pts) / 1e6
                        if 0 < gap_ms < GAP_SANITY_MS:
                            hub.sys.note_encode_gap(stream_id, gap_ms)
                    last_pts = pkt.pts_ns
            except Exception as exc:  # noqa: BLE001 - reconnect and continue
                logger.debug("encoded watcher(%s) ended: %s", stream_id, exc)
                _watch_stop.wait(5.0)

    t = threading.Thread(target=_run, name=f"enc-watch-{stream_id}", daemon=True)
    t.start()
    return t
