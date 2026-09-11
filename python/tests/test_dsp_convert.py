"""Tests for DspClient.convert_hw (DSP_OP_CONVERT_FORMAT) and its router legs.

Covers: the wire constant's parity with camera.proto, job request shape
(zero rects, exactly one dst, dst pool in the destination format at the
source geometry), client-side validation of the daemon's CONVERT contract
(equal dims, differing formats), fallback semantics including the
``cpu_fallback=False`` honest-accounting switch, the pure-numpy convert
pairs, and the accel-router registration of both convert directions.
"""

from unittest import mock

import numpy as np
import pytest

from neoruntime_ipc_sdk import accel
from neoruntime_ipc_sdk.color import nv12_to_rgb, rgb_to_nv12
from neoruntime_ipc_sdk.dsp import DspClient, DspError
from neoruntime_ipc_sdk.dsp_format import _cpu_convert
from neoruntime_ipc_sdk.dsp_wire import _HAL_PIXEL_FORMAT, _OP_CONVERT_FORMAT
from neoruntime_ipc_sdk.proto import camera_pb2
from tests.test_dsp import (
    RecordingStub,
    make_pool,
    memfd,
    nv12_array,
    patched_alloc,
    unimplemented_client,
)


# ---------------------------------------------------------------- helpers --
def rgb_array(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def gray_array(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width), dtype=np.uint8)


class FakeDsp:
    """Stand-in for DspClient handed to the router via _lazy_dsp_client."""

    def __init__(self, exc=None, result=None):
        self.calls = []  # (method, args, kwargs)
        self.exc = exc
        self.result = result if result is not None else np.zeros((1, 1))

    def _record(self, method, args, kwargs):
        self.calls.append((method, args, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.result

    def resize_hw(self, *args, **kwargs):
        return self._record("resize_hw", args, kwargs)

    def convert_hw(self, *args, **kwargs):
        return self._record("convert_hw", args, kwargs)


# ------------------------------------------------------------------- wire --
class TestWire:
    def test_op_constant_matches_proto(self):
        # dsp_wire mirrors camera.proto DspOp — drift would garble every job
        assert _OP_CONVERT_FORMAT == camera_pb2.DSP_OP_CONVERT_FORMAT == 3


# --------------------------------------------------------- request shape --
class TestRequest:
    def test_rgb_to_nv12_request_fields(self):
        client = DspClient()
        calls = patched_alloc(client)
        stub = RecordingStub()
        client._stub = stub

        out = client.convert_hw(rgb_array(64, 32), "nv12", fmt="rgb24")

        req = stub.requests[0]
        assert req.op == camera_pb2.DSP_OP_CONVERT_FORMAT
        assert req.src_buffer_id == 1000
        assert list(req.dst_buffer_ids) == [1001]  # exactly one dst
        assert len(req.rects) == 0  # daemon: CONVERT takes no rects
        # CONVERT keeps the dimensions: both pools at 64x32, dst in dst_fmt
        assert calls == [
            (64, 32, _HAL_PIXEL_FORMAT["rgb24"], 1),
            (64, 32, _HAL_PIXEL_FORMAT["nv12"], 1),
        ]
        assert out.shape == (48, 64)  # nv12 = h*3//2 rows
        assert client.last_used_hw is True

    def test_nv12_to_rgb_dst_geometry(self):
        client = DspClient()
        calls = patched_alloc(client)
        client._stub = RecordingStub()

        out = client.convert_hw(nv12_array(64, 32), "rgb24", fmt="nv12")

        assert calls == [
            (64, 32, _HAL_PIXEL_FORMAT["nv12"], 1),
            (64, 32, _HAL_PIXEL_FORMAT["rgb24"], 1),
        ]
        assert out.shape == (32, 64, 3)

    def test_explicit_pools_skip_alloc_and_release(self):
        client = DspClient()
        calls = patched_alloc(client)
        client._send_release = mock.Mock()
        src_pool = make_pool(client, 64, 32, "rgb24", 1)
        dst_pool = make_pool(client, 64, 32, "nv12", 1)
        client._stub = RecordingStub()

        client.convert_hw(rgb_array(64, 32), "nv12", fmt="rgb24",
                          src_pool=src_pool, dst_pool=dst_pool)

        assert calls == []  # caller pools mean zero allocs
        client._send_release.assert_not_called()  # and no release on the wire


# ------------------------------------------------------------- validation --
class TestValidation:
    def test_same_format_rejected(self):
        client = DspClient()
        with pytest.raises(DspError, match="differing formats"):
            client.convert_hw(rgb_array(64, 32), "rgb24", fmt="rgb24")

    def test_nv12_destination_needs_even_dims(self):
        client = DspClient()
        with pytest.raises(DspError, match="even"):
            client.convert_hw(rgb_array(33, 17), "nv12", fmt="rgb24")

    def test_dst_pool_wrong_format_rejected(self):
        client = DspClient()
        wrong = make_pool(client, 64, 32, "rgb24", 1)  # job needs nv12
        with pytest.raises(DspError, match="job needs"):
            client.convert_hw(rgb_array(64, 32), "nv12", fmt="rgb24", dst_pool=wrong)

    def test_src_pool_wrong_geometry_rejected(self):
        client = DspClient()
        wrong = make_pool(client, 64, 64, "rgb24", 1)  # source is 64x32
        with pytest.raises(DspError, match="source is 64x32"):
            client.convert_hw(rgb_array(64, 32), "nv12", fmt="rgb24", src_pool=wrong)

    def test_src_pool_with_handle_rejected(self):
        client = DspClient()
        pool = make_pool(client, 64, 32, "rgb24", 1)
        from neoruntime_ipc_sdk.frame import FrameHandle

        # a real memfd, not a small literal fd: __del__ closes the fds and
        # would otherwise take pytest's saved stdout down with it
        handle = FrameHandle([memfd(64 * 3 * 32)], (64 * 3,), (32 * 64 * 3,), 7,
                             width=64, height=32, format="RGB")
        with pytest.raises(DspError, match="src_pool applies to numpy"):
            client.convert_hw(handle, "nv12", fmt="rgb24", src_pool=pool)


# ---------------------------------------------------------------- fallback --
class TestFallback:
    def test_unimplemented_rpc_warns_and_uses_cpu(self):
        client = unimplemented_client()
        src = rgb_array(64, 32)

        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.convert_hw(src, "nv12", fmt="rgb24")

        assert np.array_equal(out, rgb_to_nv12(src))  # delegates to color.py
        assert client.last_used_hw is False

    def test_cpu_fallback_false_raises_instead(self):
        client = unimplemented_client()

        with pytest.raises(DspError, match="not in daemon"):
            client.convert_hw(rgb_array(64, 32), "nv12", fmt="rgb24", cpu_fallback=False)

    def test_rgb_to_gray_matches_bt601_luma(self):
        src = rgb_array(64, 32)
        gray = _cpu_convert(src, "rgb24", "gray8")
        # same BT.601 weights as the NV12 luma plane — allow rounding slack
        assert gray.shape == (32, 64)
        assert np.max(np.abs(gray.astype(int) - rgb_to_nv12(src)[:32].astype(int)) <= 1)

    def test_gray_nv12_roundtrip_is_lossless(self):
        src = gray_array(64, 32)
        nv12 = _cpu_convert(src, "gray8", "nv12")
        assert nv12.shape == (48, 64)
        assert np.array_equal(_cpu_convert(nv12, "nv12", "gray8"), src)

    def test_gray_to_rgb_replicates_channels(self):
        src = gray_array(64, 32)
        rgb = _cpu_convert(src, "gray8", "rgb24")
        assert rgb.shape == (32, 64, 3)
        assert np.array_equal(rgb[..., 0], src)
        assert np.array_equal(rgb[..., 1], src)
        assert np.array_equal(rgb[..., 2], src)

    def test_nv12_to_gray_is_luma_slice(self):
        src = nv12_array(64, 32)
        assert np.array_equal(_cpu_convert(src, "nv12", "gray8"), src[:32])

    def test_nv12_to_rgb_matches_color_module(self):
        src = nv12_array(64, 32)
        assert np.array_equal(_cpu_convert(src, "nv12", "rgb24"), nv12_to_rgb(src, 64, 32))


# ------------------------------------------------------------ router legs --
class TestRouterLegs:
    def test_both_directions_registered_as_hardware(self):
        ops = accel.get_default_router().health()["ops"]
        for op in ("rgb_to_nv12", "nv12_to_rgb"):
            assert ops[op]["backend"] == "hardware"

    def test_hw_leg_disables_client_fallback(self, monkeypatch):
        # the router must pass cpu_fallback=False or a daemon without the
        # DSP surface would silently compute on CPU under a hardware label
        fake = FakeDsp(exc=DspError("dsp service not running", code=-5))
        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: fake)

        router = accel.AccelRouter()
        router.register("rgb_to_nv12", software=rgb_to_nv12, hardware=accel._rgb_to_nv12_hw)
        src = rgb_array(64, 32)
        out = router.run("rgb_to_nv12", src)

        method, args, kwargs = fake.calls[0]
        assert method == "convert_hw"
        assert args[1] == "nv12"
        assert np.array_equal(args[0], src)
        assert (kwargs["fmt"], kwargs["cpu_fallback"]) == ("rgb24", False)
        assert np.array_equal(out, rgb_to_nv12(src))  # software leg ran once
        op = router.health()["ops"]["rgb_to_nv12"]
        assert (op["fallbacks"], op["software_calls"], op["hardware_calls"]) == (1, 1, 0)
        assert len(router.health()["recent_degradations"]) == 1

    def test_hw_leg_success_counts_as_hardware(self, monkeypatch):
        sentinel = np.full((48, 64), 7, dtype=np.uint8)
        fake = FakeDsp(result=sentinel)
        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: fake)

        router = accel.AccelRouter()
        router.register("rgb_to_nv12", software=rgb_to_nv12, hardware=accel._rgb_to_nv12_hw)
        out = router.run("rgb_to_nv12", rgb_array(64, 32))

        assert out is sentinel
        assert router.health()["ops"]["rgb_to_nv12"]["hardware_calls"] == 1

    def test_nv12_to_rgb_hw_validates_declared_dims(self, monkeypatch):
        fake = FakeDsp()
        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: fake)
        nv12 = nv12_array(64, 32)

        with pytest.raises(ValueError, match="64x32"):
            accel._nv12_to_rgb_hw(nv12, width=32, height=64)

        accel._nv12_to_rgb_hw(nv12, 64, 32)  # correct dims pass through
        method, args, kwargs = fake.calls[0]
        assert method == "convert_hw"
        assert args[1] == "rgb24"
        assert np.array_equal(args[0], nv12)
        assert (kwargs["fmt"], kwargs["cpu_fallback"]) == ("nv12", False)


# ------------------------------------------------- firmware pair rejection --
class TestJobRejection:
    """The daemon accepted the job but the hardware refused the pair.

    On-device ground truth (hailo15 firmware): rgb24<->nv12 run on
    the DSP; every gray8 pair comes back as a failed job with
    ``HAL rc=-2801``. The client must degrade like any other
    unavailability — and never lie about the backend used.
    """

    @staticmethod
    def rejected_client():
        client = DspClient()
        patched_alloc(client)
        client._stub = RecordingStub(
            side_effect=lambda req: camera_pb2.DspJobResponse(
                success=False, error_code=-1, message="convert_format failed (HAL rc=-2801)"
            )
        )
        return client

    def test_rejected_pair_warns_and_falls_back(self):
        client = self.rejected_client()
        src = rgb_array(64, 32)

        with pytest.warns(UserWarning, match="rgb24->gray8.*rc=-2801"):
            out = client.convert_hw(src, "gray8", fmt="rgb24")

        assert np.array_equal(out, _cpu_convert(src, "rgb24", "gray8"))
        assert client.last_used_hw is False

    def test_rejected_pair_strict_raises(self):
        client = self.rejected_client()

        with pytest.raises(DspError, match="rc=-2801"):
            client.convert_hw(rgb_array(64, 32), "gray8", fmt="rgb24", cpu_fallback=False)

    def test_rejected_pair_with_handle_refuses_silent_copy(self, monkeypatch):
        from neoruntime_ipc_sdk.frame import FrameHandle

        client = self.rejected_client()
        client._send_release = mock.Mock()
        monkeypatch.setattr(client, "_import_source", lambda *a: 2000)
        handle = FrameHandle(
            [memfd(64 * 3 * 32)], (64 * 3,), (32 * 64 * 3,), 7,
            width=64, height=32, format="RGB",
        )

        with pytest.raises(DspError, match="zero-copy frame source"):
            client.convert_hw(handle, "gray8", fmt="rgb24")

    def test_router_survives_job_rejection(self, monkeypatch):
        # a firmware-rejected pair raises DspError from the hardware leg —
        # the router must treat it as degradation, not crash
        fake = FakeDsp(exc=DspError("dsp job failed: convert_format failed (HAL rc=-2801)"))
        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: fake)

        router = accel.AccelRouter()
        router.register("rgb_to_nv12", software=rgb_to_nv12, hardware=accel._rgb_to_nv12_hw)
        src = rgb_array(64, 32)

        out = router.run("rgb_to_nv12", src)

        assert np.array_equal(out, rgb_to_nv12(src))  # software leg ran
        op = router.health()["ops"]["rgb_to_nv12"]
        assert (op["fallbacks"], op["software_calls"], op["hardware_calls"]) == (1, 1, 0)


class TestBgrConvertWarning:
    """BGR frames entering a DSP conversion are flagged, not silently swapped."""

    def test_warns_for_bgr_frame(self, caplog):
        import logging

        from neoruntime_ipc_sdk import Frame
        from neoruntime_ipc_sdk.dsp import _warn_bgr_convert

        frame = Frame(sequence=1, timestamp_ns=0, width=8, height=8, format="BGR",
                      image=np.zeros((8, 8, 3), np.uint8))
        with caplog.at_level(logging.WARNING, logger="neoruntime_ipc_sdk.dsp"):
            _warn_bgr_convert(frame)
        assert "BGR" in caplog.text
        assert "rgb24" in caplog.text

    def test_silent_for_rgb_and_raw_arrays(self, caplog):
        import logging

        from neoruntime_ipc_sdk import Frame
        from neoruntime_ipc_sdk.dsp import _warn_bgr_convert

        frame = Frame(sequence=1, timestamp_ns=0, width=8, height=8, format="RGB",
                      image=np.zeros((8, 8, 3), np.uint8))
        with caplog.at_level(logging.WARNING, logger="neoruntime_ipc_sdk.dsp"):
            _warn_bgr_convert(frame)
            _warn_bgr_convert(np.zeros((8, 8, 3), np.uint8))
        assert caplog.text == ""
