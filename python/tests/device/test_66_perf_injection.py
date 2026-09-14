"""Perf P7 — frame injection & write-lease (2026-09-12 platform perf matrix, D-group).

Measures the injection surfaces shipped in the current workspace diff:
the PushFrame RPC round-trip, client-streaming vs unary transport,
the contract invariants under paced publishing (encoded fps/bitrate
unchanged, drop-free), the compose cost per mode/format/resolution read
from the encoded gap tail, EOS→IDR recovery, the new write-lease
dimension (capability, in-flight convergence, steady-state depth), the
P1-9 resize pool-residency regression gate, and injection×DSP
contention.

Degradation rules: a config whose daemon-side pool allocation fails
(e.g. 4K REPLACE on a CMA-tight device) is recorded and skipped, never
fails the module.
"""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from neoruntime_ipc_sdk import (CameraClient, DspClient, EncodedStreamClient,
                                FdMediaClient, FramePublisher)
from neoruntime_ipc_sdk import accel
from neoruntime_ipc_sdk.color import rgb_to_nv12 as _rgb_to_nv12_direct

from perf_common import PerfTestCase, arrival_stats, percentile_stats

SUB = "sub"        # 720p: cheap pool, fast windows
MAIN = "main"      # 4K: compose cost upper bound
PUBLISH_HZ = 30    # sub stream rate — the paced-invariant cadence


def _nv12_pattern(w: int, h: int) -> np.ndarray:
    """Deterministic uint8 NV12 plane (luma gradient + neutral chroma)."""
    y = np.tile((np.arange(w, dtype=np.uint8) * 3), (h, 1))
    uv = np.full((h // 2, w), 128, dtype=np.uint8)
    return np.vstack([y, uv])


def _argb_pattern(w: int, h: int) -> np.ndarray:
    """Opaque ARGB test pattern, wire order [A, R, G, B]."""
    px = np.zeros((h, w, 4), dtype=np.uint8)
    px[..., 0] = 255                              # A
    px[..., 1] = (np.arange(w) * 3)[None, :]      # R gradient
    px[..., 2] = (np.arange(h) * 3)[:, None]      # G gradient
    px[..., 3] = 90                               # B
    return px


def _consume(enc: EncodedStreamClient, duration_s: float):
    """Collect (times, seqs, t0) from an encoded stream for ``duration_s``."""
    times: list[float] = []
    seqs: list[int] = []
    t0 = time.monotonic()
    deadline = t0 + duration_s
    for frame in enc.subscribe():
        times.append(time.monotonic())
        seqs.append(frame.seq if frame.seq is not None else -1)
        if time.monotonic() >= deadline:
            break
    return times, seqs, t0


class _InjectionPerfBase(PerfTestCase):
    area = "perf-injection"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cam = CameraClient()
        cls.cam.connect()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.cam.close()
        except Exception:
            pass

    # -- shared: encoded-window A/B with a paced publisher running -------
    def _paced_window(self, label: str, pub: FramePublisher, frame,
                      window_s: float = 10.0, control: dict | None = None):
        """Control window → paced publish window; gap stats for both."""
        enc = EncodedStreamClient(stream_id=pub.stream_id)
        try:
            if control is None:
                c_times, c_seqs, _ = _consume(enc, window_s)
                control = arrival_stats(c_seqs, c_times,
                                        c_times[-1] - c_times[0]
                                        if c_times else window_s)
            stop = threading.Event()

            def pacer():
                interval = 1.0 / PUBLISH_HZ
                t_next = time.monotonic()
                while not stop.is_set():
                    try:
                        pub.publish(frame)
                    except Exception:  # noqa: BLE001 — cadence thread
                        pass
                    t_next += interval
                    delay = t_next - time.monotonic()
                    if delay > 0:
                        stop.wait(delay)

            st_before = self.cam.injection_status()
            thread = threading.Thread(target=pacer, daemon=True)
            thread.start()
            p_times, p_seqs, _ = _consume(enc, window_s)
            stop.set()
            thread.join(timeout=5)
            st_after = self.cam.injection_status()
        finally:
            try:
                enc.close()
            except Exception:
                pass
        stats = arrival_stats(p_seqs, p_times,
                              p_times[-1] - p_times[0] if p_times else window_s)
        stats["control"] = control
        stats["fps_delta_pct"] = (round(
            (stats["fps"] - control["fps"]) * 100.0 / control["fps"], 2)
            if stats.get("fps") and control.get("fps") else None)
        stats["frames_injected"] = (st_after.frames_injected
                                    - st_before.frames_injected)
        stats["frames_dropped"] = (st_after.frames_dropped
                                   - st_before.frames_dropped)
        self.evidence(**{f"perf:{label}": stats})
        return stats


class T01PushFrameRpc(_InjectionPerfBase):
    """D1: unary PushFrame round-trip (pool write + RPC + lease snapshot)."""

    timeout_s = 300

    def test_01_push_frame(self):
        self.mark("FramePublisher.publish RPC latency (sub, REPLACE NV12)")
        dsp = DspClient()
        dsp.connect()
        pub = FramePublisher(self.cam, dsp, stream_id=SUB, pool_depth=6)
        try:
            self.evidence(lease_mode=pub.lease_mode,
                          pool=f"{pub.width}x{pub.height}x6")
            stats = self.perf_sample(pub.publish, _nv12_pattern(
                pub.width, pub.height), label="push_frame_720p_nv12",
                n=200, rounds=1)
            st = self.cam.injection_status()
            self.evidence(status_tail={
                "frames_injected": st.frames_injected,
                "frames_dropped": st.frames_dropped,
                "queue_depth": st.queue_depth})
            self.assertGreater(stats.get("n", 0), 0)
        finally:
            pub.close()


class T02PacedInvariants(_InjectionPerfBase):
    """D3: contract invariants — 38 paced frames, zero drops, fps held."""

    timeout_s = 300

    def test_01_paced_38(self):
        self.mark("paced 30Hz×38: drops/fps/gap invariants (sub)")
        dsp = DspClient()
        dsp.connect()
        with FramePublisher(self.cam, dsp, stream_id=SUB) as pub:
            frame = _nv12_pattern(pub.width, pub.height)
            enc = EncodedStreamClient(stream_id=SUB)
            c_times, c_seqs, _ = _consume(enc, 8.0)   # control window
            control = arrival_stats(c_seqs, c_times,
                                    c_times[-1] - c_times[0]
                                    if c_times else 8.0)
            st0 = self.cam.injection_status()

            stop = threading.Event()

            def pacer():
                interval = 1.0 / PUBLISH_HZ
                t_next = time.monotonic()
                for _ in range(38):
                    if stop.is_set():
                        return
                    try:
                        pub.publish(frame)
                    except Exception:  # noqa: BLE001
                        pass
                    t_next += interval
                    delay = t_next - time.monotonic()
                    if delay > 0:
                        stop.wait(delay)

            p = threading.Thread(target=pacer, daemon=True)
            p.start()
            p_times, p_seqs, _ = _consume(enc, 8.0)  # covers publish + tail
            p.join(timeout=10)
            st1 = self.cam.injection_status()
            enc.close()

        stats = arrival_stats(p_seqs, p_times,
                              p_times[-1] - p_times[0] if p_times else 8.0)
        dropped = st1.frames_dropped - st0.frames_dropped
        stats.update(control=control, frames_dropped=dropped)
        self.evidence(**{"perf:paced_invariants_38": stats})
        # Contract gate: one-for-one frame swap, drop-oldest never engaged.
        self.assertEqual(dropped, 0,
                         f"daemon dropped {dropped} of 38 paced frames")
        if stats.get("fps") and control.get("fps"):
            self.assertGreater(stats["fps"], control["fps"] * 0.9,
                               "encoded fps sagged >10% under injection")


class T03StreamVsUnary(_InjectionPerfBase):
    """D2: one client-streaming RPC vs N unary pushes (transport only)."""

    timeout_s = 300

    def test_01_stream_vs_unary(self):
        self.mark("PushFrameStream (60 frames) vs 60× publish")
        from itertools import repeat
        dsp = DspClient()
        dsp.connect()
        n = 60
        rec = {}
        with FramePublisher(self.cam, dsp, stream_id=SUB, pool_depth=6) as pub:
            frame = _nv12_pattern(pub.width, pub.height)

            t0 = time.perf_counter()
            for _ in range(n):
                pub.publish(frame)
            rec["unary_wall_s"] = round(time.perf_counter() - t0, 3)

            t0 = time.perf_counter()
            res = pub.publish_stream(repeat(frame, n), end_with_eos=True)
            rec["stream_wall_s"] = round(time.perf_counter() - t0, 3)
            rec["stream_accepted"] = getattr(res, "accepted_frame_count", None)
        rec["unary_fps_equiv"] = round(n / rec["unary_wall_s"], 1)
        rec["stream_fps_equiv"] = round(n / rec["stream_wall_s"], 1)
        self.evidence(**{"perf:stream_vs_unary": rec})


class T04ComposeCost(_InjectionPerfBase):
    """D4: compose cost per mode/fmt/resolution, from the encoded gaps."""

    timeout_s = 600

    def test_01_scan(self):
        self.mark("compose A/B: off / REPLACE-NV12 / OVERLAY-ARGB × 720p/4K")
        configs = [
            ("compose_replace_720p_nv12",
             dict(stream_id=SUB, mode="replace", fmt="nv12")),
            ("compose_overlay_720p_argb",
             dict(stream_id=SUB, mode="overlay", fmt="argb")),
            ("compose_overlay_4k_argb",
             dict(stream_id=MAIN, mode="overlay", fmt="argb",
                  inset=(960, 540), dest=(0, 0))),
            ("compose_replace_4k_nv12",
             dict(stream_id=MAIN, mode="replace", fmt="nv12")),
        ]
        controls: dict[str, dict] = {}
        for label, kwargs in configs:
            stream = kwargs["stream_id"]
            dsp = DspClient()
            dsp.connect()
            try:
                pub = FramePublisher(self.cam, dsp, pool_depth=4,
                                     **kwargs)
            except Exception as exc:  # noqa: BLE001 — pool alloc refusal
                self.evidence(**{f"perf:{label}": {
                    "alloc_refused": f"{type(exc).__name__}: {exc}"[:160]}})
                continue
            try:
                if kwargs["fmt"] == "nv12":
                    frame = _nv12_pattern(pub.width, pub.height)
                else:
                    frame = _argb_pattern(pub.width, pub.height)
                if stream not in controls:
                    enc = EncodedStreamClient(stream_id=stream)
                    c_times, c_seqs, _ = _consume(enc, 8.0)
                    enc.close()
                    controls[stream] = arrival_stats(
                        c_seqs, c_times,
                        c_times[-1] - c_times[0] if c_times else 8.0)
                self._paced_window(label, pub, frame,
                                   control=controls[stream])
            finally:
                pub.close()


class T05EosRecovery(_InjectionPerfBase):
    """D6: EOS → next-IDR recovery gap on the encoded stream."""

    timeout_s = 300

    def test_01_eos_gap(self):
        self.mark("publish_eos → encoded stream recovery (sub)")
        dsp = DspClient()
        dsp.connect()
        with FramePublisher(self.cam, dsp, stream_id=SUB) as pub:
            frame = _nv12_pattern(pub.width, pub.height)
            stop = threading.Event()

            def pacer():
                interval = 1.0 / PUBLISH_HZ
                t_next = time.monotonic()
                while not stop.is_set():
                    try:
                        pub.publish(frame)
                    except Exception:  # noqa: BLE001
                        pass
                    t_next += interval
                    delay = t_next - time.monotonic()
                    if delay > 0:
                        stop.wait(delay)

            enc = EncodedStreamClient(stream_id=SUB)
            p = threading.Thread(target=pacer, daemon=True)
            p.start()
            times: list[float] = []
            t0 = time.monotonic()
            deadline = t0 + 12.0
            t_eos = None
            for fr in enc.subscribe():
                times.append(time.monotonic())
                if t_eos is None and time.monotonic() >= t0 + 5.0:
                    stop.set()          # stop injecting first …
                    pub.publish_eos()   # … then flush the session
                    t_eos = times[-1]
                if time.monotonic() >= deadline:
                    break
            p.join(timeout=5)
            enc.close()
        if t_eos is None or len(times) < 10:
            self.evidence(**{"perf:eos_recovery": {"note": "window too short"}})
            return
        after = [t for t in times if t > t_eos]
        recovery = {
            "frames_total": len(times),
            "first_frame_after_eos_ms": round((after[0] - t_eos) * 1e3, 1)
            if after else None,
            "max_gap_after_eos_ms": round(
                max((b - a for a, b in zip(after, after[1:])), default=0)
                * 1e3, 1),
        }
        self.evidence(**{"perf:eos_recovery": recovery})


class T06WriteLease(_InjectionPerfBase):
    """D7: write-lease dimension — capability, convergence, steady depth."""

    timeout_s = 300

    def test_01_capability_and_convergence(self):
        self.mark("in_flight_buffer_ids convergence after single pushes")
        st = self.cam.injection_status()
        self.evidence(reports_in_flight_buffers=st.reports_in_flight_buffers,
                      note="False => legacy daemon; lease tests are vacuous")
        dsp = DspClient()
        dsp.connect()
        with FramePublisher(self.cam, dsp, stream_id=SUB) as pub:
            frame = _nv12_pattern(pub.width, pub.height)
            conv_ms: list[float] = []
            for _ in range(10):
                pub.publish(frame)
                t0 = time.monotonic()
                converged = False
                while time.monotonic() - t0 < 2.0:
                    s = self.cam.injection_status()
                    if (s.reports_in_flight_buffers
                            and not s.in_flight_buffer_ids):
                        conv_ms.append((time.monotonic() - t0) * 1e3)
                        converged = True
                        break
                    time.sleep(0.005)
                if not converged:
                    conv_ms.append(float("nan"))
            stats = percentile_stats([v for v in conv_ms if v == v])
            stats["n_conv"] = sum(1 for v in conv_ms if v == v)
            stats["n_timeout"] = sum(1 for v in conv_ms if v != v)
            self.evidence(**{"perf:lease_convergence": stats})

    def test_02_steady_state_depth(self):
        self.mark("|in_flight| and queue_depth at paced 30Hz")
        dsp = DspClient()
        dsp.connect()
        with FramePublisher(self.cam, dsp, stream_id=SUB) as pub:
            frame = _nv12_pattern(pub.width, pub.height)
            stop = threading.Event()
            depths: list[int] = []
            queues: list[int] = []

            def sampler():
                while not stop.is_set():
                    s = self.cam.injection_status()
                    if s.reports_in_flight_buffers:
                        depths.append(len(s.in_flight_buffer_ids))
                    queues.append(s.queue_depth)
                    stop.wait(0.02)

            def pacer():
                interval = 1.0 / PUBLISH_HZ
                t_next = time.monotonic()
                while not stop.is_set():
                    try:
                        pub.publish(frame)
                    except Exception:  # noqa: BLE001
                        pass
                    t_next += interval
                    delay = t_next - time.monotonic()
                    if delay > 0:
                        stop.wait(delay)

            s1 = threading.Thread(target=sampler, daemon=True)
            s2 = threading.Thread(target=pacer, daemon=True)
            s1.start()
            s2.start()
            time.sleep(10.0)
            stop.set()
            s1.join(timeout=5)
            s2.join(timeout=5)
        self.evidence(**{"perf:lease_steady_state": {
            "samples": len(depths),
            "in_flight_max": max(depths) if depths else None,
            "in_flight_mean": round(sum(depths) / len(depths), 2)
            if depths else None,
            "queue_max": max(queues) if queues else None}})


class T07ResizePoolResidency(_InjectionPerfBase):
    """B3 regression gate: paced single-client resize stays single-peak.

    P1-9 made the router's dsp buffer pool client-resident; the
    acceptance gate (2026-09-11) is p99 ≤ 35.7ms with zero samples
    ≥ 43ms at a paced cadence. Fresh-client-per-call hides regressions
    (known quota gap) — this holds ONE router client for the whole run.
    """

    timeout_s = 300

    def test_01_paced_resize(self):
        self.mark("paced 10Hz resize_nv12 single client (P1-9 gate)")
        media = FdMediaClient()
        frame = media.get_frame("sub", timeout_ms=5000)
        media.close()
        if frame is None:
            raise unittest.SkipTest("camera produced no frame")
        rgb = frame.to_rgb()
        frame.release()
        h, w = rgb.shape[:2]
        nv12 = _rgb_to_nv12_direct(rgb)
        router = accel.get_default_router()
        lat: list[float] = []
        deadline = time.monotonic() + 30.0
        t_next = time.monotonic()
        while time.monotonic() < deadline:
            t0 = time.perf_counter_ns()
            router.run("resize_nv12", nv12, (w, h), (w // 2, h // 2))
            lat.append((time.perf_counter_ns() - t0) / 1e6)
            t_next += 0.1
            delay = t_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        stats = percentile_stats(lat)
        over43 = sum(1 for v in lat if v >= 43.0)
        stats.update(geometry=f"{w}x{h}->{w // 2}x{h // 2}",
                     hz=10, window_s=30, samples_ge_43ms=over43)
        self.evidence(**{"perf:paced_resize_pool": stats})
        self.assertEqual(over43, 0,
                         f"{over43}/{len(lat)} samples ≥43ms — pool "
                         "residency regressed (P1-9 gate)")


class T08InjectionDspConcurrency(_InjectionPerfBase):
    """D8: injection publish paced while DSP jobs run — both p99s."""

    timeout_s = 300

    @staticmethod
    def _dsp_load(stop, lat_out, nv12, hw):
        router = accel.get_default_router()
        while not stop.is_set():
            t0 = time.perf_counter_ns()
            try:
                router.run("resize_nv12", nv12, hw, (hw[0] // 2, hw[1] // 2))
            except Exception:  # noqa: BLE001
                pass
            lat_out.append((time.perf_counter_ns() - t0) / 1e6)
            time.sleep(0.05)  # ~20Hz DSP load

    def test_01_concurrent(self):
        self.mark("paced publish + 20Hz DSP resize: p99 degradation both sides")
        media = FdMediaClient()
        f = media.get_frame("sub", timeout_ms=5000)
        media.close()
        if f is None:
            raise unittest.SkipTest("camera produced no frame")
        rgb = f.to_rgb()
        f.release()
        h, w = rgb.shape[:2]
        nv12 = _rgb_to_nv12_direct(rgb)

        # Solo baselines.
        dsp_lat_solo: list[float] = []
        stop = threading.Event()
        load = threading.Thread(target=self._dsp_load,
                                args=(stop, dsp_lat_solo, nv12, (w, h)),
                                daemon=True)
        load.start()
        time.sleep(8.0)
        stop.set()
        load.join(timeout=5)

        dsp = DspClient()
        dsp.connect()
        with FramePublisher(self.cam, dsp, stream_id=SUB) as pub:
            frame = _nv12_pattern(pub.width, pub.height)
            pub_lat_solo: list[float] = []
            deadline = time.monotonic() + 8.0
            t_next = time.monotonic()
            while time.monotonic() < deadline:   # solo publish baseline
                t0 = time.perf_counter_ns()
                pub.publish(frame)
                pub_lat_solo.append((time.perf_counter_ns() - t0) / 1e6)
                t_next += 1.0 / PUBLISH_HZ
                delay = t_next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

            pub_lat_conc: list[float] = []
            dsp_lat_conc: list[float] = []
            stop2 = threading.Event()
            load2 = threading.Thread(target=self._dsp_load,
                                     args=(stop2, dsp_lat_conc, nv12, (w, h)),
                                     daemon=True)
            load2.start()
            deadline = time.monotonic() + 8.0
            t_next = time.monotonic()
            while time.monotonic() < deadline:   # concurrent window
                t0 = time.perf_counter_ns()
                try:
                    pub.publish(frame)
                except Exception:  # noqa: BLE001
                    pass
                pub_lat_conc.append((time.perf_counter_ns() - t0) / 1e6)
                t_next += 1.0 / PUBLISH_HZ
                delay = t_next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            stop2.set()
            load2.join(timeout=5)

        def deg(solo, conc):
            if solo.get("p99") and conc.get("p99"):
                return round((conc["p99"] - solo["p99"]) * 100.0
                             / solo["p99"], 1)
            return None

        s_solo = percentile_stats(dsp_lat_solo)
        s_conc = percentile_stats(dsp_lat_conc)
        p_solo = percentile_stats(pub_lat_solo)
        p_conc = percentile_stats(pub_lat_conc)
        self.evidence(**{"perf:injection_dsp_concurrency": {
            "dsp_resize_solo": s_solo, "dsp_resize_concurrent": s_conc,
            "publish_solo": p_solo, "publish_concurrent": p_conc,
            "dsp_p99_degradation_pct": deg(s_solo, s_conc),
            "publish_p99_degradation_pct": deg(p_solo, p_conc)}})


if __name__ == "__main__":
    unittest.main()
