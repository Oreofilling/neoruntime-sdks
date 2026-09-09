"""Phase 5 — pure-toolkit surfaces: color, draw, nms, accel router, Config.

Everything here runs on one *real* camera frame where the math allows
it (color conversions round-trip real pixels); the router section also
exercises a private AccelRouter instance so fallback/degradation
accounting can be forced deterministically without lying about the
device's DSP state.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from neoruntime_ipc_sdk import BoundingBox, DetectedObject, FdMediaClient
from neoruntime_ipc_sdk.accel import (
    AccelRouter,
    HardwareUnavailable,
    RoutePolicy,
    get_default_router,
)
from neoruntime_ipc_sdk.color import (
    bgr_to_nv12,
    nv12_resize,
    nv12_to_bgr,
    nv12_to_rgb,
    rgb_to_nv12,
)
from neoruntime_ipc_sdk.config import Config
from neoruntime_ipc_sdk.draw import (
    draw_boxes,
    draw_detections,
    draw_polygons,
    draw_text,
    render_overlay_rgba,
)
from neoruntime_ipc_sdk.postprocess import nms

from common import DeviceTestCase

MAIN = "main"


def _real_nv12():
    """One real frame as an NV12 array (converting if the stream is RGB)."""
    client = FdMediaClient()
    try:
        frame = client.get_frame(MAIN, timeout_ms=5000)
        if frame is None:
            return None, None, None, "no frame on 'main' in 5s"
        w, h, fmt = frame.width, frame.height, frame.format
        if fmt in ("NV12", "NV21"):
            nv12 = frame.to_array().copy()
            frame.release()
            return nv12, w, h, fmt
        rgb = frame.to_rgb()
        frame.release()
        if w % 2 or h % 2:  # colour ops need even dims
            rgb, w, h = rgb[: h - 1, : w - 1], w - 1, h - 1
        return rgb_to_nv12(rgb), w, h, fmt
    finally:
        client.close()


class T01Color(DeviceTestCase):
    """The five colour conversions, on real frame pixels."""

    area = "toolkit"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.nv12, cls.w, cls.h, cls.src_fmt = _real_nv12()
        if cls.nv12 is None:
            raise unittest.SkipTest(f"no raw frame for colour tests: {cls.src_fmt}")

    def test_01_nv12_to_rgb(self):
        self.mark("color.nv12_to_rgb")
        rgb = self.timed(nv12_to_rgb, self.nv12, self.w, self.h,
                         label="nv12_to_rgb")
        self.evidence(shape=rgb.shape, dtype=str(rgb.dtype),
                      src_format=self.src_fmt)
        self.assertEqual(rgb.shape, (self.h, self.w, 3))

    def test_02_nv12_to_bgr(self):
        self.mark("color.nv12_to_bgr")
        bgr = self.timed(nv12_to_bgr, self.nv12, self.w, self.h,
                         label="nv12_to_bgr")
        rgb = nv12_to_rgb(self.nv12, self.w, self.h)
        self.assertEqual(bgr.shape, (self.h, self.w, 3))
        # nv12_to_bgr goes through cv2.cvtColor while nv12_to_rgb routes
        # through the accel router (DSP path) — the two pipelines round
        # independently (different YUV→RGB matrices / co-siting), so at
        # 4K the per-pixel delta is a few tens of LSB, not 2 (measured:
        # swapped max diff 19, unswapped 129 on a real camera frame).
        # Tolerance 32 covers that spread; the discriminator below is
        # what actually proves the ordering is BGR: the *unswapped* diff
        # must dwarf the swapped one.
        swapped = int(np.abs(
            bgr[:, :, ::-1].astype(int) - rgb.astype(int)).max())
        unswapped = int(np.abs(
            bgr.astype(int) - rgb.astype(int)).max())
        self.evidence(shape=bgr.shape, swapped_max_diff=swapped,
                      unswapped_max_diff=unswapped)
        self.assertLessEqual(swapped, 32,
                             "BGR differs from swapped RGB by more than "
                             "cross-pipeline rounding (cv2 vs DSP)")
        self.assertGreater(unswapped, swapped + 32,
                           "channel order not distinguishable — output "
                           "is not BGR-swapped relative to RGB")

    def test_03_rgb_to_nv12_roundtrip(self):
        self.mark("color.rgb_to_nv12")
        rgb = nv12_to_rgb(self.nv12, self.w, self.h)
        back = self.timed(rgb_to_nv12, rgb, label="rgb_to_nv12")
        again = nv12_to_rgb(back, self.w, self.h)
        err = float(np.abs(again.astype(int) - rgb.astype(int)).mean())
        self.evidence(nv12_shape=back.shape, roundtrip_mean_abs_err=err)
        self.assertEqual(back.shape, (self.h * 3 // 2, self.w))
        self.assertLess(err, 12.0, "colour round-trip lost more than "
                                   "subsampling should")

    def test_04_bgr_to_nv12(self):
        self.mark("color.bgr_to_nv12")
        bgr = nv12_to_bgr(self.nv12, self.w, self.h)
        back = self.timed(bgr_to_nv12, bgr, label="bgr_to_nv12")
        self.evidence(nv12_shape=back.shape)
        self.assertEqual(back.shape, (self.h * 3 // 2, self.w))

    def test_05_nv12_resize(self):
        self.mark("color.nv12_resize")
        dw, dh = self.w // 2, self.h // 2
        out = self.timed(nv12_resize, self.nv12, (self.w, self.h),
                         (dw, dh), label="nv12_resize")
        self.evidence(dst=(dw, dh), out_shape=out.shape)
        self.assertEqual(out.shape, (dh * 3 // 2, dw))


class T02Draw(DeviceTestCase):
    """The draw rasters on a real RGB frame (copy semantics asserted)."""

    area = "toolkit"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        nv12, w, h, note = _real_nv12()
        if nv12 is None:
            raise unittest.SkipTest(f"no raw frame for draw tests: {note}")
        cls.nv12, cls.w, cls.h = nv12, w, h
        cls.rgb = nv12_to_rgb(nv12, w, h)

    def _det(self):
        return DetectedObject(
            label="person", score=0.9,
            bbox=BoundingBox(x=self.w * 0.25, y=self.h * 0.25,
                             width=self.w * 0.4, height=self.h * 0.4),
        )

    def test_01_draw_boxes(self):
        self.mark("draw.draw_boxes")
        out = self.timed(draw_boxes, self.rgb,
                         [self._det().bbox], ["person"], [0.9],
                         label="draw_boxes")
        self.evidence(shape=out.shape,
                      modified=bool(not np.array_equal(out, self.rgb)))
        self.assertEqual(out.shape, self.rgb.shape)
        self.assertFalse(np.array_equal(out, self.rgb),
                         "draw_boxes returned an unmodified copy")

    def test_02_draw_text(self):
        self.mark("draw.draw_text")
        out = self.timed(draw_text, self.rgb, "SDK-TEST", (10, 30),
                         label="draw_text")
        self.evidence(modified=bool(not np.array_equal(out, self.rgb)))
        self.assertFalse(np.array_equal(out, self.rgb))

    def test_03_draw_detections_rgb(self):
        self.mark("draw.draw_detections (RGB input)")
        out = self.timed(draw_detections, self.rgb, [self._det()],
                         label="draw_detections_rgb")
        self.evidence(shape=out.shape,
                      modified=bool(not np.array_equal(out, self.rgb)))
        self.assertFalse(np.array_equal(out, self.rgb))

    def test_04_draw_detections_nv12_routed(self):
        self.mark("draw.draw_detections (NV12, accel-routed)")
        out = self.timed(draw_detections, self.nv12, [self._det()],
                         label="draw_detections_nv12")
        decision = get_default_router().route("draw_detections")
        self.evidence(out_shape=out.shape, backend=decision.backend,
                      provider=decision.provider)
        self.assertEqual(out.shape, self.nv12.shape)

    def test_05_render_overlay_rgba(self):
        self.mark("draw.render_overlay_rgba")
        rgba, x0, y0 = self.timed(
            render_overlay_rgba, self.w, self.h,
            [self._det().bbox], ["person"], [0.9],
            polygons=[(np.array([[0, 0], [200, 0], [200, 200], [0, 200]],
                                np.int32), (255, 0, 0))],
            label="render_overlay_rgba",
        )
        self.evidence(canvas_shape=rgba.shape, x0=x0, y0=y0)
        self.assertEqual(rgba.ndim, 3)
        self.assertEqual(rgba.shape[2], 4)
        self.assertGreaterEqual(rgba.shape[0], 16)

    def test_06_draw_polygons(self):
        self.mark("draw.draw_polygons")
        pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], np.int32)
        out = self.timed(draw_polygons, self.rgb, [(pts, (0, 255, 0))],
                         label="draw_polygons")
        self.evidence(modified=bool(not np.array_equal(out, self.rgb)))
        self.assertFalse(np.array_equal(out, self.rgb))


class T03Nms(DeviceTestCase):
    area = "toolkit"

    def test_01_suppresses_overlaps(self):
        self.mark("postprocess.nms")
        boxes = np.array([[0, 0, 100, 100],
                          [5, 5, 105, 105],    # heavy overlap with box 0
                          [200, 200, 300, 300]], np.float32)
        scores = np.array([0.9, 0.8, 0.7], np.float32)
        kept = self.timed(nms, boxes, scores, 0.5, label="nms")
        self.evidence(kept=list(map(int, kept)))
        self.assertEqual(kept, [0, 2])

    def test_02_class_aware(self):
        self.mark("postprocess.nms (class_ids)")
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 100]], np.float32)
        scores = np.array([0.9, 0.8], np.float32)
        classes = np.array([0, 1])
        kept = nms(boxes, scores, 0.5, class_ids=classes)
        self.evidence(kept=list(map(int, kept)))
        self.assertEqual(kept, [0, 1], "different classes must not "
                                       "suppress each other")

    def test_03_empty(self):
        self.mark("postprocess.nms (empty input)")
        kept = nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
        self.evidence(kept=kept)
        self.assertEqual(kept, [])


class T04AccelRouter(DeviceTestCase):
    """Default router state + a private router with forced fallbacks."""

    area = "toolkit"

    def test_01_default_router_singleton(self):
        self.mark("accel.get_default_router")
        r1 = get_default_router()
        r2 = get_default_router()
        self.evidence(same_object=r1 is r2, policy=r1.policy.value,
                      ops=sorted(r1.health()["ops"]))
        self.assertIs(r1, r2)

    def test_02_route_and_probe_health(self):
        self.mark("AccelRouter.route/probe/health")
        router = get_default_router()
        decision = router.route("nv12_to_rgb")
        probes = router.probe()
        health = router.health()
        self.evidence(decision_backend=decision.backend,
                      decision_reason=decision.reason, probes=probes,
                      health_policy=health["policy"],
                      recent_degradations=len(health["recent_degradations"]))
        self.assertIn(decision.backend, ("hardware", "software"))
        self.assertIsInstance(probes, dict)
        with self.assertRaises(KeyError):
            router.route("no-such-op")

    def test_03_run_resize_via_router(self):
        self.mark("AccelRouter.run")
        nv12, w, h, note = _real_nv12()
        if nv12 is None:
            self.na(f"no raw frame to route: {note}")
        out = get_default_router().run("resize_nv12", nv12, (w, h),
                                       (w // 2, h // 2))
        self.evidence(out_shape=out.shape, src=(w, h))
        self.assertEqual(out.shape, ((h // 2) * 3 // 2, w // 2))

    def test_04_private_router_fallback(self):
        self.mark("AccelRouter.register/use_hardware/note_degradation/run")
        router = AccelRouter(policy=RoutePolicy.PREFER_HARDWARE)
        router.register("sdk-test-op", software=lambda x: x * 2,
                        note="no hardware leg")
        soft = router.route("sdk-test-op")
        self.assertEqual(router.run("sdk-test-op", 21), 42)

        def _broken_hw(x):
            raise HardwareUnavailable("unit-test forced failure")

        router.use_hardware("sdk-test-op", _broken_hw)
        hw = router.route("sdk-test-op")
        result = router.run("sdk-test-op", 21)  # falls back to software
        router.note_degradation("sdk-external", "unit-test external report")
        router.add_probe("always-true", lambda: True)
        health = router.health()
        self.evidence(soft_backend=soft.backend, hw_backend=hw.backend,
                      fallback_result=result,
                      probes=router.probe(),
                      ops=health["ops"].get("sdk-test-op"),
                      external_reports=len(health["recent_degradations"]))
        self.assertEqual(soft.backend, "software")
        self.assertEqual(hw.backend, "hardware")
        self.assertEqual(result, 42, "PREFER_HARDWARE did not fall back")
        self.assertGreaterEqual(health["ops"]["sdk-test-op"]["fallbacks"], 1)
        self.assertGreaterEqual(len(health["recent_degradations"]), 2)


class T05Config(DeviceTestCase):
    area = "toolkit"

    def test_01_endpoint_family(self):
        self.mark("Config endpoint/env getters")
        values = {
            "app_id": Config.get_app_id(),
            "inference": Config.get_inference_endpoint(),
            "event_bus": Config.get_event_bus_endpoint(),
            "device_control": Config.get_device_control_endpoint(),
            "shm_base": Config.get_shm_base_path(),
            "camera_control": Config.get_camera_control_endpoint(),
            "app_manager": Config.get_app_manager_endpoint(),
            "encoded_dir": Config.get_encoded_socket_dir(),
            "host_prefix": Config.get_host_prefix(),
            "log_level": Config.get_log_level(),
            "is_debug": Config.is_debug(),
        }
        self.evidence(config=values)
        # Per-key type contract: endpoints/paths/labels are strings,
        # is_debug() is a bool (it is not an env-string getter).
        for key in ("app_id", "inference", "event_bus", "device_control",
                    "shm_base", "camera_control", "app_manager",
                    "encoded_dir", "host_prefix", "log_level"):
            self.assertIsInstance(values[key], str, f"{key} not a string")
        self.assertIsInstance(values["is_debug"], bool)

    def test_02_env_override(self):
        self.mark("Config env override")
        original = os.environ.get("AI_RUNTIME_ENDPOINT")
        try:
            os.environ["AI_RUNTIME_ENDPOINT"] = "unix:///tmp/sdk-test.sock"
            self.assertEqual(Config.get_inference_endpoint(),
                             "unix:///tmp/sdk-test.sock")
        finally:
            if original is None:
                os.environ.pop("AI_RUNTIME_ENDPOINT", None)
            else:
                os.environ["AI_RUNTIME_ENDPOINT"] = original
        self.evidence(restored=Config.get_inference_endpoint())

    def test_03_translate_path_to_host(self):
        self.mark("Config.translate_path_to_host")
        prefix = Config.get_host_prefix()
        translated = Config.translate_path_to_host("/opt/aipc/models/m.hef")
        untouched = Config.translate_path_to_host("/var/log/x")
        self.evidence(prefix=prefix, translated=translated,
                      untouched=untouched)
        self.assertTrue(translated.startswith(prefix))
        self.assertEqual(untouched, "/var/log/x")


if __name__ == "__main__":
    unittest.main()
