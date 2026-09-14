"""Perf P6 — AI overlay bake path (2026-09-12 platform perf matrix, C-group).

Measures the overlay surfaces that have no replayable acceptance
script: the annotate publish cost against payload size, the bake
overhead as seen from the *encoded* stream while layer complexity
scales (the bake runs on the encode thread, so its cost surfaces as
inter-packet gap growth, not as any client-side call), the result TTL
chain and session sweep, the strict-mode counters under the new
"app layer always paints" semantics, and the frame-binding hit rate
after the shared-sequence counter fix.

Degradation rules (mirroring the rest of the 6x suite): missing
prerequisites skip the single test, never the module.
"""

from __future__ import annotations

import threading
import time
import unittest

from neoruntime_ipc_sdk import (CameraClient, EncodedStreamClient,
                                FdMediaClient, OverlayClient)

from perf_common import PerfTestCase, arrival_stats

STREAM = "sub"          # 720p: overlay cost visible without 4K noise
WINDOW_S = 15           # per-level sampling window
ANNOTATE_HZ = 3         # refresh cadence keeping TTL alive during scans


def _dets(n: int) -> list[dict]:
    """``n`` synthetic detections spread over a normalized frame."""
    out = []
    for k in range(n):
        x = (k % 8) * 0.12 + 0.02
        y = (k // 8) * 0.12 + 0.02
        out.append({"label": f"cls{k % 4}", "score": 0.5 + (k % 5) / 10,
                    "bbox": {"x": x, "y": y, "width": 0.08, "height": 0.06}})
    return out


def _polys(n: int) -> list[dict]:
    """``n`` quadrilateral zone polygons (normalized points)."""
    out = []
    for k in range(n):
        ox, oy = (k % 4) * 0.24, (k // 4) * 0.24
        out.append({"points": [[ox, oy], [ox + 0.2, oy], [ox + 0.2, oy + 0.2],
                               [ox, oy + 0.2]],
                    "label": f"zone{k}", "closed": True})
    return out


def _overlay_counters(cam: CameraClient, stream: str) -> dict:
    st = next(s for s in cam.get_stream_status() if s.stream_id == stream)
    return {"packets": st.packets_published, "bake_skips": st.bake_skips,
            "strict_locked": st.strict_locked, "strict_degraded": st.strict_degraded,
            "strict_skips": st.strict_skips, "layer_count": st.overlay_layer_count,
            "late_commands": st.overlay_late_commands,
            "no_binding_drops": st.overlay_no_binding_drops}


class _OverlayPerfBase(PerfTestCase):
    area = "perf-overlay"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cam = CameraClient()
        cls.cam.connect()
        cls.overlay = OverlayClient()
        cls.overlay.connect()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.overlay.disable()
        except Exception:
            pass
        for c in (cls.overlay, cls.cam):
            try:
                c.close()
            except Exception:
                pass

    # -- shared: sample the encoded stream while a background thread
    # -- annotates; the bake cost lands in the packet gap tail.
    def _scan_level(self, label: str, dets: int, polys: int,
                    window_s: float = WINDOW_S) -> dict:
        stop = threading.Event()

        def annotator():
            interval = 1.0 / ANNOTATE_HZ
            t_next = time.monotonic()
            while not stop.is_set():
                try:
                    self.overlay.annotate(
                        STREAM, _dets(dets) if dets else None,
                        polygons=_polys(polys) if polys else None,
                        ttl_ms=2000)
                except Exception:  # noqa: BLE001 — cadence thread, diagnostic only
                    pass
                t_next += interval
                delay = t_next - time.monotonic()
                if delay > 0:
                    stop.wait(delay)

        before = _overlay_counters(self.cam, STREAM)
        enc = EncodedStreamClient(stream_id=STREAM)
        gen = enc.subscribe()
        thread = threading.Thread(target=annotator, daemon=True)
        thread.start()
        times: list[float] = []
        seqs: list[int] = []
        try:
            t0 = time.monotonic()
            deadline = t0 + window_s
            for frame in gen:
                times.append(time.monotonic())
                seqs.append(frame.seq)
                if time.monotonic() >= deadline:
                    break
        finally:
            stop.set()
            thread.join(timeout=5)
            try:
                enc.close()
            except Exception:
                pass
        stats = arrival_stats(seqs, times, (times[-1] - t0) if times else window_s)
        stats["dets"] = dets
        stats["polys"] = polys
        after = _overlay_counters(self.cam, STREAM)
        stats["counter_delta"] = {k: after[k] - before[k] for k in before}
        self.evidence(**{f"perf:{label}": stats})
        return stats


class T01AnnotateCost(_OverlayPerfBase):
    """Publish-side RPC cost vs payload size (event bus on the path)."""

    timeout_s = 300

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.overlay.enable()

    def test_01_payload_scaling(self):
        self.mark("OverlayClient.annotate latency vs detection count")
        for n in (10, 50):
            self.perf_sample(
                self.overlay.annotate, STREAM, _dets(n),
                label=f"annotate_{n}det", n=100, rounds=1)
        self.perf_sample(
            self.overlay.annotate, STREAM, polygons=_polys(16),
            label="annotate_16poly", n=100, rounds=1)


class T02ComplexityScan(_OverlayPerfBase):
    """Bake overhead vs layer complexity, read from the encoded gaps."""

    timeout_s = 300

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.overlay.enable()

    def test_01_scan(self):
        self.mark("bake overhead vs complexity (encoded gap tail, sub)")
        for label, dets, polys in (("level_small", 2, 0),
                                   ("level_mid", 8, 4),
                                   ("level_big", 50, 16)):
            self._scan_level(label, dets, polys)


class T03TtlChain(_OverlayPerfBase):
    """Result TTL expiry and the session/end sweep (layer lifecycle)."""

    timeout_s = 300

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.overlay.enable()

    def _decay_time(self, ttl_ms: int) -> dict:
        self.overlay.annotate(STREAM, _dets(4), ttl_ms=ttl_ms)
        held = _overlay_counters(self.cam, STREAM)["layer_count"]
        t0 = time.monotonic()
        while time.monotonic() - t0 < (ttl_ms / 1000.0) * 4 + 2.0:
            if _overlay_counters(self.cam, STREAM)["layer_count"] < held:
                return {"ttl_ms": ttl_ms, "layers_held": held,
                        "decay_s": round(time.monotonic() - t0, 3)}
            time.sleep(0.05)
        return {"ttl_ms": ttl_ms, "layers_held": held, "decay_s": None}

    def test_01_ttl_expiry(self):
        self.mark("layer decay vs ttl_ms (per-result override)")
        for ttl in (400, 1500):
            self.evidence(**{f"perf:ttl_decay_{ttl}": self._decay_time(ttl)})
        # Annotate with neither clears boxes immediately.
        self.overlay.annotate(STREAM, [])
        time.sleep(0.2)
        self.evidence(clear_layer_count=_overlay_counters(self.cam,
                                                          STREAM)["layer_count"])

    def test_02_session_sweep(self):
        self.mark("session-tagged layers swept by session/end")
        sess = "perf-ttl-sess"
        for _ in range(3):  # a few tagged layers via one session
            self.overlay.annotate(STREAM, _dets(2), session_id=sess, ttl_ms=10000)
        time.sleep(0.3)
        held = _overlay_counters(self.cam, STREAM)["layer_count"]
        # Client close with an active tagged session emits session/end.
        self.overlay.close()
        self.overlay.connect()
        time.sleep(1.0)
        after = _overlay_counters(self.cam, STREAM)["layer_count"]
        self.evidence(**{"perf:session_sweep": {
            "layers_before": held, "layers_after": after}})


class T04NoopOverhead(_OverlayPerfBase):
    """Overlay enabled-but-idle vs disabled: the no-op per-frame cost."""

    timeout_s = 300

    def test_01_ab(self):
        self.mark("overlay on-idle vs off (encoded gap A/B)")
        self.overlay.disable()
        time.sleep(0.5)
        self._scan_level("noop_off", 0, 0, window_s=10)
        self.overlay.enable()
        time.sleep(0.5)
        self._scan_level("noop_on_idle", 0, 0, window_s=10)


class T05StrictAppLayer(_OverlayPerfBase):
    """Strict semantics: gate SKIP must not suppress the app layer.

    With strict on but no platform results flowing, strict_skips grows
    (gate has nothing to lock) while bake_skips must NOT grow in step —
    the app layer painted. bake_skips == strict_skips means frames were
    shipped clean and the old suppress behavior is back (regression).
    """

    timeout_s = 300

    def test_01_strict_with_app_layer(self):
        self.mark("strict gate SKIP + app layer paints (new semantics)")
        self.overlay.enable()
        self.overlay.configure(strict_frame_lock=True, strict_wait_cap_ms=0)
        try:
            before = _overlay_counters(self.cam, STREAM)
            stop = threading.Event()

            def annotator():
                while not stop.is_set():
                    try:
                        self.overlay.annotate(STREAM, _dets(4), ttl_ms=2000)
                    except Exception:  # noqa: BLE001
                        pass
                    stop.wait(1.0 / ANNOTATE_HZ)

            thread = threading.Thread(target=annotator, daemon=True)
            thread.start()
            time.sleep(WINDOW_S)
            stop.set()
            thread.join(timeout=5)
            after = _overlay_counters(self.cam, STREAM)
            delta = {k: after[k] - before[k] for k in before}
            self.evidence(**{"perf:strict_app_layer": delta})
            frames = delta["strict_locked"] + delta["strict_degraded"] \
                + delta["strict_skips"]
            if frames > 50:  # enough traffic for the read to mean anything
                self.assertLess(
                    delta["bake_skips"], frames,
                    "bake_skips grew with every strict frame — app layer "
                    "suppressed again (old strict semantics regression)")
        finally:
            self.overlay.configure(strict_frame_lock=False)


class T06FrameBinding(_OverlayPerfBase):
    """Frame-bound annotate hit rate under the shared-seq counter fix.

    The bind target must live in the HAL shared-counter space (~81
    steps/s across all streams; +2.55 steps per main frame — measured
    2026-09-12), which ``FrameHandle.sequence`` from the fd path
    re-exports verbatim. ``last_packet_seq`` is the per-stream encoded
    packet counter (30/s for main): binding from it lands ~2.5× deep
    in the past and every command late-rejects. Bind to a fresh
    frame's sequence +2..4 steps (inside the 2-display-frame ≈5-step
    bind slack, ahead of the bake frontier).
    """

    timeout_s = 300

    def test_01_bound_hit_rate(self):
        self.mark("frame-bound annotate late_commands rate")
        self.overlay.enable()
        media = FdMediaClient()
        before = _overlay_counters(self.cam, STREAM)
        n = 60
        for k in range(n):
            anchor = media.get_frame(STREAM, timeout_ms=5000)
            base = anchor.sequence if anchor is not None else 0
            if anchor is not None:
                anchor.release()
            try:
                self.overlay.annotate(
                    STREAM, _dets(1),
                    frame_sequence=max(1, base + 2 + (k % 3)),
                    ttl_ms=2000)
            except Exception:  # noqa: BLE001 — counted via late_commands
                pass
            time.sleep(1.0 / ANNOTATE_HZ)
        media.close()
        after = _overlay_counters(self.cam, STREAM)
        late = after["late_commands"] - before["late_commands"]
        self.evidence(**{"perf:frame_binding": {
            "events": n, "late_rejects": late,
            "hit_rate_pct": round((n - late) * 100.0 / n, 1)}})


if __name__ == "__main__":
    unittest.main()
