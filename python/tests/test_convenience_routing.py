"""Tests for convenience-layer hardware routing (sdk-hardware-routing).

The public convenience entry points ride the accel router instead of
always running their CPU implementations: ``color.rgb_to_nv12`` /
``color.nv12_to_rgb`` and ``draw.draw_detections`` (NV12 arrays) route
through ``router.run``; ``Frame.to_jpeg_bytes`` gains the zero-copy
daemon encode for keep-fd frames; ``Frame.resize`` keeps its direct DSP
fast path but respects the router policy (SOFTWARE_ONLY skips the
attempt, PREFER_HARDWARE records a degradation on DSP failure,
HARDWARE_ONLY raises) without changing its data path.

Also pins the recursion safety of the wiring: router software legs must
call the private implementations, never the routing public functions.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from neoruntime_ipc_sdk import accel, color, draw
from neoruntime_ipc_sdk import dsp as dsp_module
from neoruntime_ipc_sdk.accel import AccelRouter, HardwareUnavailable, RoutePolicy
from neoruntime_ipc_sdk.dsp import DspError
from neoruntime_ipc_sdk.frame import Frame, FrameHandle
from neoruntime_ipc_sdk.inference_types import BoundingBox, DetectedObject
from tests.test_dsp import make_handle, memfd


# ---------------------------------------------------------------- helpers --
def rgb_array(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def nv12_array(width, height, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 256, (height, width), dtype=np.uint8)
    uv = rng.integers(0, 256, (height // 2, width), dtype=np.uint8)
    return np.vstack([y, uv])


def one_object():
    """car at (8, 6)-(40, 28) — a DetectedObject like a detector returns."""
    return [DetectedObject(label="car", score=0.9, bbox=BoundingBox(8, 6, 32, 22))]


def broken_hw(*args, **kwargs):
    raise HardwareUnavailable("daemon unreachable")


class BoomDsp:
    """DspClient stand-in whose jobs always fail — deterministic DSP outage."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def resize_hw(self, *args, **kwargs):
        raise DspError("DSP unavailable (test outage)")


@pytest.fixture
def router(monkeypatch):
    """Fresh AccelRouter installed as the package default singleton."""
    r = AccelRouter()
    monkeypatch.setattr(accel, "_default_router", r)
    return r


def make_frame(fmt="NV12", width=64, height=32, image=None, handle=True):
    """Frame with an optional memfd-backed keep-fd handle."""
    return Frame(
        sequence=1,
        timestamp_ns=0,
        width=width,
        height=height,
        format=fmt,
        image=image,
        handle=make_handle(width, height, fmt) if handle else None,
    )


# ------------------------------------------------------- color: two-way ----
class TestColorRouting:
    def test_rgb_to_nv12_rides_router_hardware(self, router):
        rgb = rgb_array(64, 32)
        sentinel = np.zeros((48, 64), dtype=np.uint8)
        seen = []

        def hw(src):
            seen.append(src)
            return sentinel

        router.register("rgb_to_nv12", software=color._rgb_to_nv12_impl, hardware=hw)
        assert np.array_equal(color.rgb_to_nv12(rgb), sentinel)
        assert seen == [rgb]
        assert router.health()["ops"]["rgb_to_nv12"]["hardware_calls"] == 1

    def test_rgb_to_nv12_degrades_to_impl_on_unavailable(self, router):
        # also the recursion guard: a software leg that re-entered the
        # public function would recurse forever instead of returning
        rgb = rgb_array(64, 32)
        router.register(
            "rgb_to_nv12", software=color._rgb_to_nv12_impl, hardware=broken_hw
        )
        assert np.array_equal(color.rgb_to_nv12(rgb), color._rgb_to_nv12_impl(rgb))
        ops = router.health()["ops"]["rgb_to_nv12"]
        assert (ops["fallbacks"], ops["software_calls"], ops["hardware_calls"]) == (1, 1, 0)
        assert router.health()["recent_degradations"][0]["op"] == "rgb_to_nv12"

    def test_rgb_to_nv12_software_only_skips_hardware(self, monkeypatch):
        rgb = rgb_array(64, 32)
        router = AccelRouter(policy=RoutePolicy.SOFTWARE_ONLY)
        monkeypatch.setattr(accel, "_default_router", router)
        router.register("rgb_to_nv12", software=color._rgb_to_nv12_impl, hardware=broken_hw)
        assert np.array_equal(color.rgb_to_nv12(rgb), color._rgb_to_nv12_impl(rgb))
        ops = router.health()["ops"]["rgb_to_nv12"]
        assert (ops["hardware_calls"], ops["software_calls"]) == (0, 1)

    def test_rgb_to_nv12_invalid_shape_still_raises(self, router):
        # caller errors are not degradations: the real hardware leg
        # validates the impl's contract BEFORE anything touches the
        # daemon, so ValueError propagates with every counter untouched
        router.register(
            "rgb_to_nv12", software=color._rgb_to_nv12_impl,
            hardware=accel._rgb_to_nv12_hw,
        )
        with pytest.raises(ValueError, match="3D"):
            color.rgb_to_nv12(np.zeros(10, dtype=np.uint8))
        with pytest.raises(ValueError, match="even"):
            color.rgb_to_nv12(np.zeros((33, 65, 3), dtype=np.uint8))
        ops = router.health()["ops"]["rgb_to_nv12"]
        assert ops["fallbacks"] == 0  # misuse never lands in the counters
        assert (ops["software_calls"], ops["hardware_calls"]) == (0, 0)

    def test_nv12_to_rgb_rides_router_hardware(self, router):
        nv12 = nv12_array(64, 32)
        sentinel = rgb_array(64, 32)
        seen = []

        def hw(src, width, height):
            seen.append((src, width, height))
            return sentinel

        router.register("nv12_to_rgb", software=color._nv12_to_rgb_impl, hardware=hw)
        assert np.array_equal(color.nv12_to_rgb(nv12, 64, 32), sentinel)
        assert seen == [(nv12, 64, 32)]
        assert router.health()["ops"]["nv12_to_rgb"]["hardware_calls"] == 1

    def test_nv12_to_rgb_degrades_to_impl_on_unavailable(self, router):
        nv12 = nv12_array(64, 32)
        router.register(
            "nv12_to_rgb", software=color._nv12_to_rgb_impl, hardware=broken_hw
        )
        assert np.array_equal(
            color.nv12_to_rgb(nv12, 64, 32), color._nv12_to_rgb_impl(nv12, 64, 32)
        )
        assert router.health()["ops"]["nv12_to_rgb"]["fallbacks"] == 1

    def test_bgr_variants_stay_off_the_router(self, router):
        # BGR convenience functions have no hardware op — they must not
        # consult (nor crash) the router at all
        bgr = rgb_array(64, 32)
        assert np.array_equal(
            color.bgr_to_nv12(bgr), color.bgr_to_nv12(bgr)
        )
        assert router.health()["ops"] == {}


# ------------------------------------------------------------- draw --------
class TestDrawRouting:
    def test_nv12_input_rides_router_hardware(self, router):
        nv12 = nv12_array(64, 32)
        annotated = nv12_array(64, 32, seed=9)
        seen = []

        def hw(image, objects, color_arg=None):
            seen.append((image, objects, color_arg))
            return annotated

        router.register("draw_detections", software=accel._draw_detections_sw, hardware=hw)
        out = draw.draw_detections(nv12, one_object())
        assert np.array_equal(out, annotated)
        assert seen[0][0] is nv12 and seen[0][1] == one_object()
        assert router.health()["ops"]["draw_detections"]["hardware_calls"] == 1

    def test_nv12_degrades_to_cpu_mirror_without_recursion(self, router):
        nv12 = nv12_array(64, 32)
        router.register(
            "draw_detections", software=accel._draw_detections_sw, hardware=broken_hw
        )
        out = draw.draw_detections(nv12, one_object())
        assert np.array_equal(out, accel._draw_detections_sw(nv12, one_object()))
        ops = router.health()["ops"]["draw_detections"]
        assert ops["fallbacks"] == 1 and ops["hardware_calls"] == 0

    def test_rgb_input_stays_on_raster_no_fake_degradation(self, router):
        # RGB is not a hardware-servable shape: it goes straight to the
        # software raster — no doomed hardware attempt, no fallback row
        rgb = rgb_array(64, 32)
        bomb = mock.Mock(side_effect=AssertionError("RGB must not reach hardware"))
        router.register("draw_detections", software=accel._draw_detections_sw, hardware=bomb)
        out = draw.draw_detections(rgb, one_object())
        assert np.array_equal(out, draw._draw_detections_impl(rgb, one_object()))
        bomb.assert_not_called()
        ops = router.health()["ops"]["draw_detections"]
        assert (ops["hardware_calls"], ops["fallbacks"]) == (0, 0)

    def test_keep_fd_frame_raises_the_documented_contract(self, router):
        # frames are refused by both legs — raise the honest error at the
        # public entry instead of degrading (it is a routing decision,
        # not a hardware failure)
        frame_like = SimpleNamespace(handle=object(), width=64, height=32)
        router.register(
            "draw_detections", software=accel._draw_detections_sw, hardware=broken_hw
        )
        with pytest.raises(HardwareUnavailable, match="keep-fd"):
            draw.draw_detections(frame_like, one_object())
        assert router.health()["ops"]["draw_detections"]["fallbacks"] == 0

    def test_accel_sw_leg_rgb_uses_impl_not_public(self):
        # the router's software leg must not re-enter the routing public
        # function — with RGB input it delegates to the raster impl
        rgb = rgb_array(64, 32)
        out = accel._draw_detections_sw(rgb, one_object())
        assert np.array_equal(out, draw._draw_detections_impl(rgb, one_object()))


# ------------------------------------------------------ Frame.to_jpeg ------
class TestFrameToJpeg:
    def test_keep_fd_frame_encodes_zero_copy_on_hardware(self, router):
        # image=None: any to_rgb() materialization would fail loudly, so
        # a passing call proves the frame rode the daemon encode as-is
        frame = make_frame("NV12", 64, 32, image=None, handle=True)
        jpeg = b"\xff\xd8\xff\xe0 fake jpeg bytes"
        seen = []

        def hw(src, quality=85):
            seen.append((src, quality))
            return jpeg

        router.register("encode_jpeg", software=accel._encode_jpeg_sw, hardware=hw)
        assert frame.to_jpeg_bytes(quality=90) == jpeg
        assert seen == [(frame, 90)]  # the Frame object itself, quality forwarded
        assert frame.image is None  # never materialized
        assert router.health()["ops"]["encode_jpeg"]["hardware_calls"] == 1

    def test_array_frame_stays_on_cpu_encode(self, router):
        # in-memory frames never touch the router: the daemon encoder's
        # win is the zero-copy import, not raw speed (S-3 record), so
        # tight loops (MjpegStream.push_frame) keep the CPU path
        frame = make_frame("RGB", 64, 32, image=rgb_array(64, 32), handle=False)
        assert frame.to_jpeg_bytes()[:2] == b"\xff\xd8"
        assert router.health()["ops"] == {}  # no attempt, no fallback row

    def test_keep_fd_hw_failure_materializes_and_encodes_on_cpu(self, router):
        # frame-aware software leg: to_rgb() materializes the retained
        # handle once, then the cv2/Pillow path encodes it
        frame = make_frame("RGB", 64, 32, image=None, handle=True)
        os.pwrite(frame.handle.fds[0], rgb_array(64, 32).tobytes(), 0)
        router.register("encode_jpeg", software=accel._encode_jpeg_sw, hardware=broken_hw)
        assert frame.to_jpeg_bytes()[:2] == b"\xff\xd8"
        assert frame.image is not None  # materialized by the fallback

    def test_hardware_only_keep_fd_failure_raises(self, monkeypatch):
        # HARDWARE_ONLY must not silently degrade a keep-fd encode
        frame = make_frame("NV12", 64, 32, image=None, handle=True)
        router = AccelRouter(policy=RoutePolicy.HARDWARE_ONLY)
        monkeypatch.setattr(accel, "_default_router", router)
        router.register("encode_jpeg", software=accel._encode_jpeg_sw, hardware=broken_hw)
        with pytest.raises(HardwareUnavailable, match="daemon unreachable"):
            frame.to_jpeg_bytes()
        assert router.health()["ops"]["encode_jpeg"]["fallbacks"] == 0

    def test_software_only_keep_fd_encodes_on_cpu(self, monkeypatch):
        # SOFTWARE_ONLY never attempts the daemon; the frame-aware
        # software leg serves the keep-fd frame directly
        frame = make_frame("RGB", 64, 32, image=None, handle=True)
        os.pwrite(frame.handle.fds[0], rgb_array(64, 32, seed=3).tobytes(), 0)
        router = AccelRouter(policy=RoutePolicy.SOFTWARE_ONLY)
        monkeypatch.setattr(accel, "_default_router", router)
        bomb = mock.Mock(side_effect=AssertionError("must not attempt hardware"))
        router.register("encode_jpeg", software=accel._encode_jpeg_sw, hardware=bomb)
        assert frame.to_jpeg_bytes()[:2] == b"\xff\xd8"
        ops = router.health()["ops"]["encode_jpeg"]
        assert (ops["hardware_calls"], ops["software_calls"]) == (0, 1)

    def test_gray8_keep_fd_degrades_to_cpu(self, monkeypatch):
        # the documented S-3 refusal ("gray8 frames cannot ride") is a
        # HardwareUnavailable under the router — degrade to to_rgb()+CPU,
        # never escape as DspError from the convenience entry
        frame = make_frame("GRAY8", 64, 32, image=None, handle=True)
        os.pwrite(frame.handle.fds[0], np.full((32, 64), 200, np.uint8).tobytes(), 0)
        monkeypatch.setattr(accel, "_default_router", None)  # real legs
        try:
            assert frame.to_jpeg_bytes()[:2] == b"\xff\xd8"
            assert accel.get_default_router().health()["ops"]["encode_jpeg"]["fallbacks"] >= 1
        finally:
            accel._default_router = None  # drop the daemon-miss counters

    def test_closed_handle_frame_keeps_the_direct_error(self, router):
        # a closed handle has no pixels and no daemon path — the CPU
        # entry must raise the same terminal error it always did
        frame = make_frame("RGB", 64, 32, image=None, handle=True)
        frame.handle.close()
        with pytest.raises((OSError, ValueError)):
            frame.to_jpeg_bytes()
        assert router.health()["ops"] == {}  # never routed

    def test_router_encode_leg_passes_fmt_none_for_frames(self, monkeypatch):
        # regression: the leg used to hardcode fmt="rgb24", which
        # _resolve_source rejects for NV12 keep-fd frames ("format
        # mismatch") — frames must ride with fmt=None and let the
        # daemon infer from the handle
        recorded = []
        monkeypatch.setattr(
            accel, "_dsp_call",
            lambda op, src, **kw: recorded.append((op, src, kw)) or b"",
        )
        frame = make_frame("NV12", 64, 32)
        accel._encode_jpeg_hw(frame, 85)
        assert recorded[0][0] == "encode_jpeg_hw"
        assert recorded[0][1] is frame
        assert recorded[0][2]["fmt"] is None

        accel._encode_jpeg_hw(rgb_array(64, 32), 85)
        assert recorded[1][2]["fmt"] == "rgb24"  # plain arrays keep the default


# ------------------------------------------------------ Frame.resize -------
class TestFrameResizePolicyGate:
    def _nv12_frame(self):
        return make_frame("NV12", 64, 32, image=nv12_array(64, 32), handle=True)

    def test_software_only_skips_the_dsp_attempt(self, monkeypatch):
        monkeypatch.setattr(
            dsp_module, "DspClient",
            mock.Mock(side_effect=AssertionError("DSP must not be touched")),
        )
        router = AccelRouter(policy=RoutePolicy.SOFTWARE_ONLY)
        monkeypatch.setattr(accel, "_default_router", router)
        # hardware leg registered like the default router does — the
        # policy (not a missing leg) is what must skip the attempt
        router.register(
            "resize_nv12",
            software=color.nv12_resize,
            hardware=accel._resize_nv12_hw,
        )
        out = self._nv12_frame().resize(32, 16, mode="stretch")
        assert out.image.shape == (24, 32)  # NV12 32x16: h + h/2 rows

    def test_prefer_hardware_records_degradation_on_outage(self, router, monkeypatch):
        monkeypatch.setattr(dsp_module, "DspClient", BoomDsp)
        router.register(
            "resize_nv12",
            software=color.nv12_resize,
            hardware=accel._resize_nv12_hw,
        )
        seen = []
        router.on_degradation = seen.append  # external fallbacks fire the hook too

        out = self._nv12_frame().resize(32, 16, mode="stretch")

        assert out.image.shape == (24, 32)  # CPU fallback still delivers pixels
        ops = router.health()["ops"]["resize_nv12"]
        assert ops["fallbacks"] == 1
        assert router.health()["recent_degradations"][0]["op"] == "resize_nv12"
        assert len(seen) == 1 and seen[0].op == "resize_nv12"

    def test_hardware_only_raises_instead_of_silent_cpu(self, monkeypatch):
        monkeypatch.setattr(dsp_module, "DspClient", BoomDsp)
        hw_only = AccelRouter(policy=RoutePolicy.HARDWARE_ONLY)
        monkeypatch.setattr(accel, "_default_router", hw_only)
        hw_only.register(
            "resize_nv12",
            software=color.nv12_resize,
            hardware=accel._resize_nv12_hw,
        )
        with pytest.raises(HardwareUnavailable, match="resize"):
            self._nv12_frame().resize(32, 16, mode="stretch")

    def test_hardware_only_array_frame_still_resizes_on_cpu(self, monkeypatch):
        # no retained handle -> no DSP attempt was ever possible; the
        # policy must not break the (only) software path
        monkeypatch.setattr(
            dsp_module, "DspClient",
            mock.Mock(side_effect=AssertionError("DSP must not be touched")),
        )
        hw_only = AccelRouter(policy=RoutePolicy.HARDWARE_ONLY)
        monkeypatch.setattr(accel, "_default_router", hw_only)
        hw_only.register(
            "resize_nv12",
            software=color.nv12_resize,
            hardware=accel._resize_nv12_hw,
        )
        frame = make_frame("NV12", 64, 32, image=nv12_array(64, 32), handle=False)
        assert frame.resize(32, 16, mode="stretch").image.shape == (24, 32)

    def test_data_path_unchanged_between_policies(self, monkeypatch):
        # the gate must not alter output: prefer-hardware-under-outage
        # and software-only produce identical pixels
        monkeypatch.setattr(dsp_module, "DspClient", BoomDsp)
        prefer = AccelRouter(policy=RoutePolicy.PREFER_HARDWARE)
        monkeypatch.setattr(accel, "_default_router", prefer)
        prefer.register(
            "resize_nv12",
            software=color.nv12_resize,
            hardware=accel._resize_nv12_hw,
        )
        via_prefer = self._nv12_frame().resize(32, 16, mode="stretch")

        sw_only = AccelRouter(policy=RoutePolicy.SOFTWARE_ONLY)
        monkeypatch.setattr(accel, "_default_router", sw_only)
        sw_only.register(
            "resize_nv12",
            software=color.nv12_resize,
            hardware=accel._resize_nv12_hw,
        )
        via_sw = self._nv12_frame().resize(32, 16, mode="stretch")

        assert np.array_equal(via_prefer.image, via_sw.image)

    def test_note_degradation_on_unregistered_op_is_visible(self, router):
        # Frame._hw_resize supports reporting to a router that never
        # registered the op — the fallback row must still surface in
        # health() and fire the on_degradation hook
        seen = []
        router.on_degradation = seen.append
        router.note_degradation("resize_nv12", "DSP resize fast path unavailable: outage")
        ops = router.health()["ops"]["resize_nv12"]
        assert ops["fallbacks"] == 1
        assert len(seen) == 1 and seen[0].op == "resize_nv12"


# ------------------------------------------- default router stays sane -----
class TestDefaultRouterWiring:
    def test_default_router_software_legs_are_private_impls(self):
        # the singleton must bind the *private* implementations — a
        # public-function leg would recurse the moment its convenience
        # entry routes through the same router
        router = accel.get_default_router()
        legs = {op: route.software for op, route in router._routes.items()}
        assert legs["rgb_to_nv12"].__name__ == "_rgb_to_nv12_impl"
        assert legs["nv12_to_rgb"].__name__ == "_nv12_to_rgb_impl"
        assert legs["encode_jpeg"].__name__ == "_encode_jpeg_sw"

    def test_convenience_calls_survive_default_router_without_daemon(self, monkeypatch):
        # no daemon on the test host: the default router's real hardware
        # legs are unavailable, so the convenience entries must degrade
        # quietly and still return correct pixels (no recursion, no crash)
        monkeypatch.setattr(accel, "_default_router", None)  # rebuild with real legs
        rgb = rgb_array(64, 32)
        nv12 = color.rgb_to_nv12(rgb)
        assert nv12.shape == (48, 64)
        back = color.nv12_to_rgb(nv12, 64, 32)
        assert back.shape == (32, 64, 3)
        frame = make_frame("RGB", 64, 32, image=rgb_array(64, 32), handle=False)
        assert frame.to_jpeg_bytes()[:2] == b"\xff\xd8"
        assert frame.resize(32, 16, mode="stretch").image.shape == (16, 32, 3)
