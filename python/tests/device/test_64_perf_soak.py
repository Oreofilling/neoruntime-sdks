"""Perf P5 — soak: sustained mixed load with per-minute RSS attribution.

Runs the representative app loop — get_frame → to_rgb → infer →
periodic event publish — for ``PERF_SOAK_S`` seconds (default 1800),
snapshotting /proc every minute for the client process and each SDK
daemon separately. Drift attribution is the point: a client-side RSS
climb is an SDK leak (fd or buffer), a daemon-side climb is a daemon
leak, and the two need different fixes.

If the infer path is dead on this deployment (the -2799 regression)
the loop degrades to the media+events path and says so — a 30-minute
soak of get_frame/to_rgb/publish still measures the media stack.
"""

from __future__ import annotations

import os
import time
import unittest

from neoruntime_ipc_sdk import EventClient, FdMediaClient, InferenceClient
from neoruntime_ipc_sdk import accel

from common import MODEL_DIR  # noqa: F401 — re-exported for clarity
from perf_common import (
    ANOMALY_DECAY_PCT,
    SOAK_S,
    PerfTestCase,
    daemon_pids,
    model_input_geometry,
    pick_model,
    proc_snapshot,
)

# Single-input NV12 model first — same contract reasoning as test_60
# (the daemon rejects byte_size-mismatched inputs with -2799/-2811).
MODEL_PATH = pick_model("hailo_yolov8n_384_640.hef",
                        "yolov5m_vehicles.hef",
                        "yolo_world_v2s_540.hef",
                        "yolo_world_v2s.hef")
TEST_MODEL_ID = "sdk-perf-soak"
TOPIC = "sdk-perf/soak"
SNAPSHOT_EVERY_S = 60
TARGET_ITER_HZ = 2.0


def _infer_input(frame):
    """Geometry-matched NV12 buffer for the model's input contract."""
    native = (frame.width, frame.height)
    geometry = model_input_geometry(MODEL_PATH) or native
    nv12 = frame.to_array()
    return accel.get_default_router().run("resize_nv12", nv12,
                                          native, geometry)


class T01MixedLoad(PerfTestCase):
    area = "perf-soak"
    timeout_s = SOAK_S + 300  # per-test alarm above the soak window

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not MODEL_PATH:
            raise unittest.SkipTest("no HEF model on the device")

    def test_01_mixed_load_soak(self):
        self.mark("mixed-load soak: media+infer+events, RSS attribution")
        infer = InferenceClient()
        events = EventClient()
        media = FdMediaClient()
        infer_ok = True
        smoke_err = None
        buckets = []
        iterations = errs = 0
        try:
            infer.connect()
            events.connect()
            infer.register_model(MODEL_PATH, model_id=TEST_MODEL_ID)

            # Smoke: if infer is dead, degrade the loop, don't abort it.
            frame0 = media.get_frame("main", timeout_ms=5000)
            if frame0 is not None:
                try:
                    infer.infer(_infer_input(frame0), TEST_MODEL_ID)
                except Exception as exc:  # noqa: BLE001 — classify + degrade
                    smoke_err = f"{type(exc).__name__}: {exc}"[:120]
                frame0.release()
            else:
                smoke_err = "no camera frame"
            infer_ok = smoke_err is None

            daemons0 = daemon_pids()
            t_start = time.monotonic()
            t_snap = t_start
            interval = 1.0 / TARGET_ITER_HZ
            t_next = t_start

            def snapshot():
                buckets.append({
                    "t_s": round(time.monotonic() - t_start, 1),
                    "iters": iterations,
                    "client": proc_snapshot(),
                    "daemons": {name: proc_snapshot(pid)
                                for name, pid in daemons0.items()},
                })

            snapshot()
            while time.monotonic() - t_start < SOAK_S:
                try:
                    frame = media.get_frame("main", timeout_ms=5000)
                    if frame is not None:
                        if infer_ok:
                            infer.infer(_infer_input(frame), TEST_MODEL_ID)
                        else:
                            frame.to_rgb()
                        frame.release()
                    if iterations % 10 == 0:
                        events.publish(TOPIC, {"iter": iterations,
                                               "t": time.monotonic()})
                    iterations += 1
                except Exception:  # noqa: BLE001 — soak counts, never stops
                    errs += 1
                if time.monotonic() - t_snap >= SNAPSHOT_EVERY_S:
                    snapshot()
                    t_snap = time.monotonic()
                t_next += interval
                delay = t_next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            snapshot()
        finally:
            for cleanup in (lambda: infer.unregister_model(TEST_MODEL_ID),
                            infer.close, events.close, media.close):
                try:
                    cleanup()
                except Exception:
                    pass

        first, last = buckets[0], buckets[-1]

        def rss(row, daemon=None):
            src = (row["daemons"].get(daemon, {}) if daemon
                   else row["client"])
            return src.get("rss_kb")

        client_delta = (rss(last) or 0) - (rss(first) or 0)
        daemon_deltas = {
            name: (rss(last, name) or 0) - (rss(first, name) or 0)
            for name in last["daemons"]}
        span_s = last["t_s"] - first["t_s"] or 1.0
        # First bucket rate is provisional (iters==0 at t≈0), so decay
        # compares the first full span against the last full span.
        mid = buckets[1] if len(buckets) > 2 else first
        rate_first = ((mid["iters"] - first["iters"])
                      / ((mid["t_s"] - first["t_s"]) or 1.0))
        rate_last = ((last["iters"] - mid["iters"])
                     / ((last["t_s"] - mid["t_s"]) or 1.0))
        decay_pct = (((rate_first - rate_last) / rate_first * 100.0)
                     if rate_first > 0 else None)

        self.evidence(
            model=os.path.basename(MODEL_PATH) if MODEL_PATH else None,
            duration_s=last["t_s"], iterations=iterations, errors=errs,
            mode="full" if infer_ok else "media-only (infer degraded)",
            infer_smoke_error=None if infer_ok else smoke_err,
            client_rss_first_kb=rss(first),
            client_rss_last_kb=rss(last),
            client_rss_delta_kb=client_delta,
            client_fds_first=first["client"].get("fds"),
            client_fds_last=last["client"].get("fds"),
            daemon_rss_delta_kb=daemon_deltas,
            iter_rate_first_per_s=round(rate_first, 2),
            iter_rate_last_per_s=round(rate_last, 2),
            rate_decay_pct=(round(decay_pct, 2)
                            if decay_pct is not None else None),
            buckets=buckets,  # full series for the report
            anomaly_threshold_decay_pct=ANOMALY_DECAY_PCT,
        )
        self.assertGreater(iterations, 0, "soak loop made no iterations")


if __name__ == "__main__":
    unittest.main()
