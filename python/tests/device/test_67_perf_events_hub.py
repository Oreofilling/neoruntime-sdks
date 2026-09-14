"""Perf P8 — event fanout, device event hub, frame-path concurrency
(2026-09-12 platform perf matrix, E3-E5 + H1/H4).

E-group: subscriber fanout scaling (1/4/16), the slow-subscriber
drop_old semantics under a burst, and the new device event hub
(8ee78dc6) consumed passively — arrival lag against the event's own
``timestamp_ns`` (same-machine monotonic clock, since the suite runs
on-device).

H-group: get_frame latency under 1/2/4/8 concurrent subscribers
(frame quota is per-client, so each thread owns its client), and a
mixed-load matrix (frame pull + overlay annotate + event publish +
paced DSP + sub-stream injection) capturing each face's tail while
every other face runs.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest

import numpy as np

from neoruntime_ipc_sdk import (CameraClient, DeviceClient, DspClient,
                                EventClient, FdMediaClient, FramePublisher,
                                OverlayClient)
from neoruntime_ipc_sdk import accel
from neoruntime_ipc_sdk.color import rgb_to_nv12 as _rgb_to_nv12_direct

from perf_common import PerfTestCase, percentile_stats

TOPIC = "sdk-perf/fanout"
SLOW_TOPIC = "sdk-perf/slow"
HUB_WINDOW_S = 45


def _dets(n: int) -> list[dict]:
    out = []
    for k in range(n):
        x = (k % 8) * 0.12 + 0.02
        y = (k // 8) * 0.12 + 0.02
        out.append({"label": f"cls{k % 4}", "score": 0.5 + (k % 5) / 10,
                    "bbox": {"x": x, "y": y, "width": 0.08, "height": 0.06}})
    return out


def _nv12_pattern(w: int, h: int) -> np.ndarray:
    y = np.tile((np.arange(w, dtype=np.uint8) * 3), (h, 1))
    uv = np.full((h // 2, w), 128, dtype=np.uint8)
    return np.vstack([y, uv])


class _FanoutPerfBase(PerfTestCase):
    area = "perf-events-hub"


class T01SubscriberFanout(_FanoutPerfBase):
    """E3: publish→arrival e2e per subscriber at 1/4/16 subscribers."""

    timeout_s = 300

    def test_01_fanout_scan(self):
        self.mark("event delivery e2e vs subscriber count (1/4/16)")
        results = {}
        for n_sub in (1, 4, 16):
            subs = []
            for _ in range(n_sub):
                c = EventClient()
                c.connect()
                q: queue.Queue = queue.Queue(maxsize=1000)
                subs.append((c, q))

                def pump(client=c, out=q):
                    try:
                        for ev in client.subscribe(TOPIC):
                            out.put(ev)
                    except Exception:  # noqa: BLE001 — daemon thread
                        pass

                threading.Thread(target=pump, daemon=True).start()
            time.sleep(1.0)  # subscriptions established

            pub = EventClient()
            pub.connect()
            samples: list[float] = []
            missed = 0
            try:
                for probe in range(30):
                    t0 = time.monotonic()
                    eid = pub.publish(TOPIC, {"probe": probe})
                    deadline_each = t0 + 5.0
                    pending = {id(q) for _, q in subs}
                    while pending and time.monotonic() < deadline_each:
                        for c, q in subs:
                            if id(q) not in pending:
                                continue
                            try:
                                ev = q.get(timeout=0.05)
                            except queue.Empty:
                                continue
                            if ev.event_id == eid:
                                samples.append(
                                    (time.monotonic() - t0) * 1e3)
                                pending.discard(id(q))
                            # non-matching events stay unmatched; the
                            # probe id is unique so this is exact.
                    missed += len(pending)
            finally:
                pub.close()
                for c, _ in subs:
                    try:
                        c.unsubscribe(TOPIC)
                    except Exception:
                        pass
                    c.close()
            stats = percentile_stats(samples)
            stats.update(subscribers=n_sub, missed=missed)
            results[n_sub] = stats
        self.evidence(**{"perf:event_fanout": results})


class T02SlowSubscriber(_FanoutPerfBase):
    """E4: burst vs one fast and one deliberately-slow subscriber."""

    timeout_s = 300

    def test_01_slow_vs_fast(self):
        self.mark("slow-subscriber drop_old: burst 100 @100Hz")
        fast = EventClient()
        fast.connect()
        slow = EventClient()
        slow.connect()
        fast_q: queue.Queue = queue.Queue(maxsize=1000)
        slow_q: queue.Queue = queue.Queue(maxsize=1000)
        stop = threading.Event()

        def pump(client, out):
            try:
                for ev in client.subscribe(SLOW_TOPIC):
                    out.put(ev)
            except Exception:  # noqa: BLE001 — daemon thread
                pass

        def slow_pump():
            try:
                for ev in slow.subscribe(SLOW_TOPIC):
                    if not stop.wait(1.0):  # the "slow" part: 1s per event
                        slow_q.put(ev)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=pump, args=(fast, fast_q), daemon=True).start()
        threading.Thread(target=slow_pump, daemon=True).start()
        time.sleep(1.0)

        pub = EventClient()
        pub.connect()
        burst = 100
        try:
            for k in range(burst):
                pub.publish(SLOW_TOPIC, {"seq": k})
                time.sleep(0.01)  # ~100Hz burst
            time.sleep(3.0)      # let the fast side drain
        finally:
            pub.close()

        fast_got = 0
        while True:
            try:
                fast_q.get_nowait()
                fast_got += 1
            except queue.Empty:
                break
        slow_got = slow_q.qsize()
        topic_stats = None
        try:
            ts = fast.get_topic_stats(SLOW_TOPIC)
            topic_stats = {"subscriber_count": getattr(ts, "subscriber_count",
                                                       None)}
        except Exception:  # noqa: BLE001 — stats surface is optional here
            pass
        stop.set()
        for c in (fast, slow):
            try:
                c.unsubscribe(SLOW_TOPIC)
            except Exception:
                pass
            c.close()
        self.evidence(**{"perf:slow_subscriber": {
            "burst": burst, "fast_received": fast_got,
            "slow_received_so_far": slow_got, "topic_stats": topic_stats,
            "note": "slow drains at 1Hz; its queue is bounded drop-old "
                    "(only the slow subscriber loses events)"}})
        self.assertGreaterEqual(fast_got, burst * 0.95,
                                f"fast subscriber lost {burst - fast_got} "
                                "events — drops are not isolated to the "
                                "slow subscriber")


class T03DeviceEventHub(_FanoutPerfBase):
    """E5: passive device-hub consumption — lag, cadence, event mix."""

    timeout_s = 120

    def test_01_passive_window(self):
        self.mark("DeviceClient.subscribe_events passive 45s window")
        dev = DeviceClient()
        dev.connect()
        events = []
        pump_error: list[str] = []
        stop = threading.Event()

        def pump():
            # The server stream blocks when no event flows, so the
            # deadline cannot be enforced from inside the iteration —
            # a pump thread feeds a list and the main thread owns the
            # clock (a silent hub would otherwise burn the watchdog).
            try:
                for ev in dev.subscribe_events():
                    events.append((time.monotonic_ns(), ev))
                    if stop.is_set():
                        return
            except Exception as exc:  # noqa: BLE001 — hub unavailable: record
                pump_error.append(f"{type(exc).__name__}: {exc}"[:160])

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        time.sleep(HUB_WINDOW_S)
        stop.set()
        # Drain anything already buffered, then hard-stop the stream.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and t.is_alive():
            time.sleep(0.05)
        if t.is_alive():
            dev.close()  # unblocks the parked recv; pump thread records EOF
            t.join(timeout=5)
        else:
            dev.close()
        if pump_error and not events:
            self.evidence(**{"perf:device_hub_events": {
                "error": pump_error[0]}})
            return
        by_type: dict[str, int] = {}
        lag_ms: list[float] = []
        for arrival_ns, ev in events:
            name = ev.type.name if hasattr(ev.type, "name") else str(ev.type)
            by_type[name] = by_type.get(name, 0) + 1
            if ev.timestamp_ns:
                lag_ms.append((arrival_ns - ev.timestamp_ns) / 1e6)
        gaps_ms = [(b[0] - a[0]) / 1e6 for a, b in zip(events, events[1:])]
        rec = {
            "events": len(events),
            "window_s": HUB_WINDOW_S,
            "by_type": by_type,
            "lag_ms": percentile_stats(lag_ms) if lag_ms else None,
            "gap_ms": percentile_stats(gaps_ms) if gaps_ms else None,
            "note": "lag = client arrival − event timestamp_ns; valid "
                    "same-clock because the suite runs on-device; the hub "
                    "has no drop counters (journal-only observability)",
        }
        self.evidence(**{"perf:device_hub_events": rec})


class T04GetFrameFanout(_FanoutPerfBase):
    """H1: get_frame latency under 1/2/4/8 concurrent subscribers."""

    timeout_s = 600

    @staticmethod
    def _puller(out, window_s):
        media = FdMediaClient()
        lat: list[float] = []
        try:
            deadline = time.monotonic() + window_s
            while time.monotonic() < deadline:
                t0 = time.perf_counter_ns()
                f = media.get_frame("main", timeout_ms=5000)
                lat.append((time.perf_counter_ns() - t0) / 1e6)
                if f is not None:
                    f.release()
        finally:
            media.close()
        out.append(lat)

    def test_01_fanout_scan(self):
        self.mark("get_frame(main) latency vs concurrent clients (1/2/4/8)")
        results = {}
        for n in (1, 2, 4, 8):
            out: list[list[float]] = []
            threads = [threading.Thread(target=self._puller,
                                        args=(out, 10.0), daemon=True)
                       for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
            all_lat = [v for lst in out for v in lst]
            stats = percentile_stats(all_lat)
            stats.update(clients=n, per_client_frames=[len(x) for x in out])
            results[n] = stats
            time.sleep(1.0)  # settle between levels
        self.evidence(**{"perf:getframe_fanout": results})


class T05MixedLoad(_FanoutPerfBase):
    """H4: frame pull + overlay + events + DSP + injection, all at once."""

    timeout_s = 300

    def test_01_mixed_matrix(self):
        self.mark("mixed load: get_frame + annotate + publish + resize + inject")
        stop = threading.Event()
        faces: dict[str, list[float]] = {k: [] for k in
                                         ("get_frame", "annotate", "publish",
                                          "dsp_resize", "inject")}

        def face_get():
            media = FdMediaClient()
            try:
                while not stop.is_set():
                    t0 = time.perf_counter_ns()
                    f = media.get_frame("main", timeout_ms=5000)
                    faces["get_frame"].append(
                        (time.perf_counter_ns() - t0) / 1e6)
                    if f is not None:
                        f.release()
            finally:
                media.close()

        def face_annotate():
            ov = OverlayClient()
            ov.connect()
            ov.enable()
            try:
                while not stop.is_set():
                    t0 = time.perf_counter_ns()
                    try:
                        ov.annotate("main", _dets(4), ttl_ms=2000)
                        faces["annotate"].append(
                            (time.perf_counter_ns() - t0) / 1e6)
                    except Exception:  # noqa: BLE001
                        pass
                    stop.wait(1.0 / 3)
            finally:
                try:
                    ov.disable()
                except Exception:
                    pass
                ov.close()

        def face_publish():
            ec = EventClient()
            ec.connect()
            try:
                while not stop.is_set():
                    t0 = time.perf_counter_ns()
                    try:
                        ec.publish(TOPIC, {"mixed": True})
                        faces["publish"].append(
                            (time.perf_counter_ns() - t0) / 1e6)
                    except Exception:  # noqa: BLE001
                        pass
                    stop.wait(0.1)
            finally:
                ec.close()

        def face_dsp():
            media = FdMediaClient()
            f = media.get_frame("sub", timeout_ms=5000)
            media.close()
            if f is None:
                return
            rgb = f.to_rgb()
            f.release()
            h, w = rgb.shape[:2]
            nv12 = _rgb_to_nv12_direct(rgb)
            router = accel.get_default_router()
            while not stop.is_set():
                t0 = time.perf_counter_ns()
                try:
                    router.run("resize_nv12", nv12, (w, h),
                               (w // 2, h // 2))
                    faces["dsp_resize"].append(
                        (time.perf_counter_ns() - t0) / 1e6)
                except Exception:  # noqa: BLE001
                    pass
                stop.wait(0.05)

        def face_inject():
            cam = CameraClient()
            cam.connect()
            dsp = DspClient()
            dsp.connect()
            try:
                pub = FramePublisher(cam, dsp, stream_id="sub")
                frame = _nv12_pattern(pub.width, pub.height)
                while not stop.is_set():
                    t0 = time.perf_counter_ns()
                    try:
                        pub.publish(frame)
                        faces["inject"].append(
                            (time.perf_counter_ns() - t0) / 1e6)
                    except Exception:  # noqa: BLE001
                        pass
                    stop.wait(1.0 / 15)
                pub.publish_eos()
                pub.close()
            finally:
                dsp.close()
                cam.close()

        runners = [threading.Thread(target=f, daemon=True) for f in
                   (face_get, face_annotate, face_publish, face_dsp,
                    face_inject)]
        for t in runners:
            t.start()
        time.sleep(15.0)
        stop.set()
        for t in runners:
            t.join(timeout=15)

        matrix = {name: percentile_stats(lat) for name, lat in faces.items()}
        self.evidence(**{"perf:mixed_load": {
            "window_s": 15,
            "faces": matrix,
            "note": "solo baselines live in their own modules "
                    "(test_61 get_frame, test_65 annotate, test_62 "
                    "publish, test_66 resize/inject) — compare p99s "
                    "cross-module for degradation"}})
        for name, stats in matrix.items():
            self.assertGreater(stats.get("n", 0), 0,
                               f"mixed-load face {name!r} produced no samples")


if __name__ == "__main__":
    unittest.main()
