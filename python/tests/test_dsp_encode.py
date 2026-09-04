"""Tests for DspClient.encode_jpeg_hw (the EncodeImage RPC, S-3(a)) and its
router legs.

Covers: request shape (one source buffer, quality on the wire, no dst alloc),
zero-copy handle wiring (import id in, release out), client-side validation
(quality bounds, src_pool geometry, src_pool+handle), the fallback semantics
shared with the job methods (UNIMPLEMENTED / service-down → warn + CPU encode,
``cpu_fallback=False`` → raise, handle → always refuse the silent copy), the
daemon's error contract (failed job raises, empty payload raises), and the
accel-router registration of ``encode_jpeg``.
"""

from unittest import mock

import grpc
import numpy as np
import pytest

from neoruntime_ipc_sdk import accel
from neoruntime_ipc_sdk.dsp import DSP_SERVICE_UNAVAILABLE, DspClient, DspError
from neoruntime_ipc_sdk.dsp_wire import _HAL_PIXEL_FORMAT
from neoruntime_ipc_sdk.frame import _encode_jpeg
from neoruntime_ipc_sdk.proto import camera_pb2
from tests.test_dsp import make_pool, memfd, patched_alloc


# ---------------------------------------------------------------- helpers --
def rgb_array(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def ok_resp(jpeg=b"\xff\xd8\xff\xe0 fake jpeg bytes"):
    return camera_pb2.EncodeImageResponse(success=True, jpeg=jpeg, elapsed_ms=12)


class EncodeStub:
    """Fake gRPC stub; records EncodeImage requests."""

    def __init__(self, resp=None, error=None):
        self.requests = []
        self.resp = resp
        self.error = error

    def EncodeImage(self, request, timeout=None):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.resp


def unimplemented_error():
    err = grpc.RpcError("no such method")
    err.code = lambda: grpc.StatusCode.UNIMPLEMENTED
    return err


def encoding_client(resp=None, error=None):
    """Client whose daemon answers EncodeImage with resp (or raises error)."""
    client = DspClient()
    calls = patched_alloc(client)
    client._send_release = mock.Mock()  # no-alloc paths never touch a socket
    stub = EncodeStub(resp=resp, error=error)
    client._stub = stub
    return client, calls, stub


class FakeDsp:
    """Stand-in for DspClient handed to the router via _lazy_dsp_client."""

    def __init__(self, exc=None, result=b"\xff\xd8 hw jpeg"):
        self.calls = []
        self.exc = exc
        self.result = result

    def encode_jpeg_hw(self, *args, **kwargs):
        self.calls.append(("encode_jpeg_hw", args, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.result


# --------------------------------------------------------- request shape --
class TestRequest:
    def test_rgb_source_request_fields(self):
        client, calls, stub = encoding_client(resp=ok_resp())
        src = rgb_array(64, 32)

        out = client.encode_jpeg_hw(src, quality=90)

        req = stub.requests[0]
        assert req.src_buffer_id == 1000  # the one pool buffer
        assert req.quality == 90
        assert isinstance(out, bytes) and out == b"\xff\xd8\xff\xe0 fake jpeg bytes"
        assert calls == [(64, 32, _HAL_PIXEL_FORMAT["rgb24"], 1)]  # src only, no dst
        assert client.last_used_hw is True
        client._send_release.assert_called_once_with(1000)  # own pool returned

    def test_nv12_source_allocates_nv12_pool(self):
        client, calls, stub = encoding_client(resp=ok_resp())
        nv12 = np.zeros((48, 64), dtype=np.uint8)

        client.encode_jpeg_hw(nv12, fmt="nv12")

        assert calls == [(64, 32, _HAL_PIXEL_FORMAT["nv12"], 1)]
        assert stub.requests[0].src_buffer_id == 1000

    def test_gray8_array_upconverts_to_rgb24_pool(self):
        # no hardware gray leg: gray rides as replicated rgb24 (R=G=B=gray)
        client, calls, stub = encoding_client(resp=ok_resp())
        gray = np.full((32, 64), 200, dtype=np.uint8)

        out = client.encode_jpeg_hw(gray, fmt="gray8")

        assert calls == [(64, 32, _HAL_PIXEL_FORMAT["rgb24"], 1)]
        assert stub.requests[0].src_buffer_id == 1000
        assert client.last_used_hw is True

    def test_explicit_src_pool_skips_alloc_and_release(self):
        client, calls, _ = encoding_client(resp=ok_resp())
        client._send_release.reset_mock()
        pool = make_pool(client, 64, 32, "rgb24", 1)

        client.encode_jpeg_hw(rgb_array(64, 32), src_pool=pool)

        assert calls == []  # caller pool means zero allocs
        client._send_release.assert_not_called()  # and no release on the wire

    def test_handle_source_imports_and_releases(self, monkeypatch):
        from neoruntime_ipc_sdk.frame import FrameHandle

        client, _, _ = encoding_client(resp=ok_resp())
        monkeypatch.setattr(client, "_import_source", mock.Mock(return_value=2000))
        handle = FrameHandle(
            [memfd(64 * 3 * 32)], (64 * 3,), (32 * 64 * 3,), 7,
            width=64, height=32, format="RGB",
        )

        out = client.encode_jpeg_hw(handle, fmt="rgb24")

        client._import_source.assert_called_once()
        client._send_release.assert_called_once_with(2000)  # import returned
        assert out == b"\xff\xd8\xff\xe0 fake jpeg bytes"
        assert client.last_used_hw is True


# ------------------------------------------------------------- validation --
class TestValidation:
    def test_quality_zero_rejected(self):
        client = DspClient()
        with pytest.raises(DspError, match="quality must be 1..100"):
            client.encode_jpeg_hw(rgb_array(64, 32), quality=0)

    def test_quality_over_100_rejected(self):
        client = DspClient()
        with pytest.raises(DspError, match="quality must be 1..100"):
            client.encode_jpeg_hw(rgb_array(64, 32), quality=101)

    def test_src_pool_wrong_geometry_rejected(self):
        client = DspClient()
        wrong = make_pool(client, 64, 64, "rgb24", 1)  # source is 64x32
        with pytest.raises(DspError, match="source is 64x32"):
            client.encode_jpeg_hw(rgb_array(64, 32), src_pool=wrong)

    def test_src_pool_with_handle_rejected(self):
        from neoruntime_ipc_sdk.frame import FrameHandle

        client = DspClient()
        pool = make_pool(client, 64, 32, "rgb24", 1)
        handle = FrameHandle(
            [memfd(64 * 3 * 32)], (64 * 3,), (32 * 64 * 3,), 7,
            width=64, height=32, format="RGB",
        )
        with pytest.raises(DspError, match="src_pool applies to numpy"):
            client.encode_jpeg_hw(handle, fmt="rgb24", src_pool=pool)

    def test_unsupported_format_rejected(self):
        client = DspClient()
        with pytest.raises(DspError, match="unsupported format"):
            client.encode_jpeg_hw(np.zeros((32, 64, 4), dtype=np.uint8), fmt="rgba")

    def test_gray8_handle_refuses_the_copy(self):
        # a gray keep-fd frame has no zero-copy path onto an nv12-only
        # encoder — raising beats silently copying the frame
        from neoruntime_ipc_sdk.frame import FrameHandle

        client = DspClient()
        handle = FrameHandle(
            [memfd(64 * 32)], (64,), (64 * 32,), 7,
            width=64, height=32, format="GRAY8",
        )
        with pytest.raises(DspError, match="gray8 frames cannot ride"):
            client.encode_jpeg_hw(handle, fmt="gray8")


# ---------------------------------------------------------------- fallback --
class TestFallback:
    def test_unimplemented_rpc_warns_and_encodes_on_cpu(self):
        client, _, _ = encoding_client(error=unimplemented_error())
        src = rgb_array(64, 32)

        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.encode_jpeg_hw(src, quality=90)

        assert out == _encode_jpeg(src, 90)  # same encoder as the software leg
        assert isinstance(out, bytes) and out[:2] == b"\xff\xd8"
        assert client.last_used_hw is False

    def test_rgb24_fallback_does_not_convert(self):
        # _cpu_convert refuses identical formats — the rgb24 fallback must
        # feed _encode_jpeg directly (regression: the naive call raised)
        client, _, _ = encoding_client(error=unimplemented_error())
        src = rgb_array(64, 32, seed=3)

        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.encode_jpeg_hw(src, fmt="rgb24", quality=70)

        assert out == _encode_jpeg(src, 70)

    def test_gray8_fallback_replicates_channels(self):
        client, _, _ = encoding_client(error=unimplemented_error())
        gray = np.full((32, 64), 200, dtype=np.uint8)

        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.encode_jpeg_hw(gray, fmt="gray8")

        rgb = np.repeat(gray[..., None], 3, axis=2)
        assert out == _encode_jpeg(rgb, 85)

    def test_service_down_response_falls_back(self):
        client, _, _ = encoding_client(
            resp=camera_pb2.EncodeImageResponse(
                success=False, error_code=DSP_SERVICE_UNAVAILABLE,
                message="dsp service not running")
        )
        src = rgb_array(64, 32)

        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.encode_jpeg_hw(src)

        assert out == _encode_jpeg(src, 85)
        assert client.last_used_hw is False

    def test_cpu_fallback_false_raises_instead(self):
        client, _, _ = encoding_client(error=unimplemented_error())

        with pytest.raises(DspError, match="not in daemon"):
            client.encode_jpeg_hw(rgb_array(64, 32), cpu_fallback=False)

    def test_failed_job_raises_even_with_fallback(self):
        # a daemon that accepted the RPC but the encode failed is an error,
        # not an unavailability — never paper over it with a CPU encode
        client, _, _ = encoding_client(
            resp=camera_pb2.EncodeImageResponse(
                success=False, error_code=-4, message="jpeg shot timed out")
        )

        with pytest.raises(DspError, match="jpeg shot timed out"):
            client.encode_jpeg_hw(rgb_array(64, 32))

    def test_success_with_empty_jpeg_raises(self):
        client, _, _ = encoding_client(
            resp=camera_pb2.EncodeImageResponse(success=True, jpeg=b"")
        )

        with pytest.raises(DspError, match="no jpeg payload"):
            client.encode_jpeg_hw(rgb_array(64, 32))

    def test_handle_refuses_silent_cpu_fallback(self, monkeypatch):
        from neoruntime_ipc_sdk.frame import FrameHandle

        client, _, _ = encoding_client(error=unimplemented_error())
        monkeypatch.setattr(client, "_import_source", lambda *a: 2000)
        handle = FrameHandle(
            [memfd(64 * 3 * 32)], (64 * 3,), (32 * 64 * 3,), 7,
            width=64, height=32, format="RGB",
        )

        with pytest.raises(DspError, match="zero-copy frame source"):
            client.encode_jpeg_hw(handle, fmt="rgb24")


# ------------------------------------------------------------ router legs --
class TestRouterLegs:
    def test_encode_jpeg_registered_as_hardware(self):
        ops = accel.get_default_router().health()["ops"]
        assert ops["encode_jpeg"]["backend"] == "hardware"

    def test_hw_leg_success_counts_as_hardware(self, monkeypatch):
        sentinel = b"\xff\xd8 hw jpeg"
        fake = FakeDsp(result=sentinel)
        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: fake)

        router = accel.AccelRouter()
        router.register("encode_jpeg", software=_encode_jpeg, hardware=accel._encode_jpeg_hw)
        out = router.run("encode_jpeg", rgb_array(64, 32), quality=75)

        assert out is sentinel
        op = router.health()["ops"]["encode_jpeg"]
        assert (op["hardware_calls"], op["software_calls"], op["fallbacks"]) == (1, 0, 0)

    def test_hw_leg_disables_client_fallback(self, monkeypatch):
        # the router must pass cpu_fallback=False or a daemon without
        # EncodeImage would silently encode on CPU under a hardware label
        fake = FakeDsp()
        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: fake)

        router = accel.AccelRouter()
        router.register("encode_jpeg", software=_encode_jpeg, hardware=accel._encode_jpeg_hw)
        src = rgb_array(64, 32)
        router.run("encode_jpeg", src, quality=60)

        method, args, kwargs = fake.calls[0]
        assert method == "encode_jpeg_hw"
        assert np.array_equal(args[0], src)
        assert (kwargs["fmt"], kwargs["cpu_fallback"]) == ("rgb24", False)
        assert kwargs["quality"] == 60

    def test_hw_leg_failure_degrades_to_software(self, monkeypatch):
        fake = FakeDsp(exc=DspError("EncodeImage not in daemon"))
        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: fake)

        router = accel.AccelRouter()
        router.register("encode_jpeg", software=_encode_jpeg, hardware=accel._encode_jpeg_hw)
        src = rgb_array(64, 32)
        out = router.run("encode_jpeg", src, quality=80)

        assert out == _encode_jpeg(src, 80)  # software leg ran exactly once
        op = router.health()["ops"]["encode_jpeg"]
        assert (op["fallbacks"], op["software_calls"], op["hardware_calls"]) == (1, 1, 0)
        assert len(router.health()["recent_degradations"]) == 1
