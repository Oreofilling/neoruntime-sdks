"""Perf P2 — media frame path and the accel routing layer (A/B).

Measures the frame plane (get_frame arrival latency, NV12/RGB
conversions, JPEG encode) and the accel router's decision behavior,
then A/B-samples each routable op under the default policy vs an
explicit SOFTWARE_ONLY router.

Calibration (2026-09-09) settled the two possible readings: on a
device whose dsp probe fails, both sides execute the same software
leg and a ratio ~1.0 documents the routing *decision* only; on the
calibration device the probes passed and the hardware legs really
engaged — see T03AccelAB for what the ratio then means.
"""

from __future__ import annotations

import unittest

from neoruntime_ipc_sdk import FdMediaClient
from neoruntime_ipc_sdk import accel
from neoruntime_ipc_sdk.color import rgb_to_nv12 as _rgb_to_nv12_direct

from perf_common import PerfTestCase, software_only_accel

# Ops sampled in the A/B. encode_jpeg takes an RGB array; the NV12 ops
# take (nv12, geometry). draw_detections needs live detections and is
# covered by the functional suite, not by latency sampling.
AB_OPS = ("resize_nv12", "nv12_to_rgb", "rgb_to_nv12", "encode_jpeg")


def _grab_frame():
    media = FdMediaClient()
    try:
        return media.get_frame("main", timeout_ms=5000)
    finally:
        media.close()


class T01FramePath(PerfTestCase):
    """Frame acquisition + per-frame conversion/encode latency."""

    area = "perf-media"
    timeout_s = 600

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.frame = _grab_frame()
        if cls.frame is None:
            raise unittest.SkipTest("camera produced no frame")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.frame.release()
        except Exception:
            pass

    def test_01_get_frame(self):
        self.mark("FdMediaClient.get_frame latency distribution")
        media = FdMediaClient()
        try:
            meta = {a: getattr(self.frame, a, None) for a in
                    ("width", "height", "pixel_format", "stride")}
            self.evidence(frame_meta=meta)

            def one():
                f = media.get_frame("main", timeout_ms=5000)
                if f is not None:
                    f.release()
                return f

            stats = self.perf_sample(one, label="get_frame_main", n=200)
            # get_frame blocks for the next frame at the stream fps:
            # p50 ≈ frame interval is the *expected* shape, not latency.
            self.evidence(note="p50 tracks the stream frame interval; "
                               "p99/max show queue/daemon stalls")
            self.assertGreater(stats.get("n", 0), 0)
        finally:
            media.close()

    def test_02_conversions(self):
        self.mark("Frame.to_rgb / resize / to_jpeg_bytes latency")
        rgb = self.frame.to_rgb()
        h, w = rgb.shape[:2]
        self.evidence(shape=f"{rgb.shape}")

        self.perf_sample(self.frame.to_rgb, label="frame_to_rgb",
                         n=30, rounds=1)

        def resize_half():
            out = self.frame.resize(w // 2, h // 2)
            out.release()  # fd-backed: release inside the timed window

        self.perf_sample(resize_half, label="frame_resize_half", n=30,
                         rounds=1)

        self.perf_sample(lambda: self.frame.to_jpeg_bytes(quality=85),
                         label="frame_to_jpeg85", n=30, rounds=1)


class T02AccelRouting(PerfTestCase):
    """Router decisions on this device: health, probes, per-op routes."""

    area = "perf-media"
    timeout_s = 120

    def test_01_health_and_routes(self):
        self.mark("accel router health/probe/route decisions")
        router = accel.get_default_router()
        self.evidence(health=router.health(), probes=router.probe())
        routes = {op: vars(router.route(op)) for op in
                  ("resize_nv12", "rgb_to_nv12", "nv12_to_rgb",
                   "encode_jpeg", "nms", "draw_detections")}
        self.evidence(routes=routes)
        self.assertIsInstance(router.health(), dict)


class T03AccelAB(PerfTestCase):
    """A/B: default-policy router vs explicit SOFTWARE_ONLY router.

    Both sides route through ``router.run`` so the policy — not a
    different call path — is the only variable. Two honest outcomes:
    with hardware legs unreachable both sides land on the same software
    leg and the ratio reads ~1.0 (a routing decision, not a speedup);
    with them reachable (secondary test device: cv2+dsp probes pass)
    the ``_*_hw`` legs
    really engage — calibration 2026-09-09 measured them 1.3–4.4×
    SLOWER than the numpy software legs at 540p. The hw legs are DSP
    daemon roundtrips whose zero-copy dma-buf import needs a frame-like
    source; this A/B feeds plain arrays, so the ratio mostly prices
    per-call socket transport of the pixels — a routing-quality
    finding, not a verdict on DSP compute.
    """

    area = "perf-media"
    timeout_s = 600

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.frame = _grab_frame()
        if cls.frame is None:
            raise unittest.SkipTest("camera produced no frame")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.frame.release()
        except Exception:
            pass

    @staticmethod
    def _args_for(op: str, rgb, nv12, src_hw, dst_hw):
        if op == "resize_nv12":
            return (nv12, src_hw, dst_hw)
        if op == "nv12_to_rgb":
            return (nv12, src_hw[0], src_hw[1])
        if op in ("rgb_to_nv12", "encode_jpeg"):
            return (rgb,)
        raise ValueError(op)

    def test_01_ab_sampling(self):
        self.mark("accel A/B: default policy vs SOFTWARE_ONLY")
        rgb = self.frame.to_rgb()
        h, w = rgb.shape[:2]
        # Input NV12 built once, outside every timed window.
        nv12 = _rgb_to_nv12_direct(rgb)
        src_hw, dst_hw = (w, h), (w // 2, h // 2)
        ratios = {}

        for op in AB_OPS:
            args = self._args_for(op, rgb, nv12, src_hw, dst_hw)

            a = self.perf_sample(accel.get_default_router().run, op, *args,
                                 label=f"ab_{op}_default", n=30, rounds=1)
            with software_only_accel() as registered:
                b = self.perf_sample(accel.get_default_router().run, op,
                                     *args, label=f"ab_{op}_swonly", n=30,
                                     rounds=1)
            if registered:
                self.evidence(**{f"ab_{op}_sw_ops": registered})
            if a.get("p50") and b.get("p50"):
                ratios[op] = round(a["p50"] / b["p50"], 2)

        self.evidence(
            p50_ratio_default_over_sw=ratios, input=f"{w}x{h}",
            note="ratio≈1.0: both sides ran the same software leg "
                 "(routing decision only); ratio>1: hardware-routed leg "
                 "engaged and was slower than software (see T02 routes)")


if __name__ == "__main__":
    unittest.main()
