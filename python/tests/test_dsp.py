"""Tests for the DSP offload client (SDK-2).

Covers: UDS alloc/release wire codec, dma-buf pool write/read roundtrips
(memfd-backed), hardware job request construction (op/rects/interpolation),
the NEAREST-default trap, fallback semantics, and client-side validation
mirroring the daemon's caps.
"""

import os
import struct
from unittest import mock

import grpc
import numpy as np
import pytest

from neoruntime_ipc_sdk import dsp
from neoruntime_ipc_sdk.dsp import (
    DSP_SERVICE_UNAVAILABLE,
    DspBufferPool,
    DspClient,
    DspError,
    alloc_request_bytes,
    parse_alloc_resp,
)
from neoruntime_ipc_sdk.proto import camera_pb2

ALLOC_FMT = "<IIIIII"  # type, size, w, h, fmt, count
RESP_SIZE = 560


# ---------------------------------------------------------------- helpers --
def memfd(size):
    fd = os.memfd_create("dsp-test", 0)
    os.ftruncate(fd, size)
    return fd


def make_pool(client, width, height, fmt, count, stride_pad=0, id_base=1000):
    """Build a memfd-backed pool without touching a socket."""
    planes = 2 if fmt == "nv12" else 1
    if fmt == "nv12":
        row_w = (width, width)  # Y row, interleaved-UV row
        rows = (height, height // 2)
    elif fmt == "rgb24":
        row_w, rows = (width * 3,), (height,)
    else:
        row_w, rows = (width,), (height,)

    strides = [r + stride_pad for r in row_w] + [0] * (3 - planes)
    sizes = [strides[p] * rows[p] for p in range(planes)] + [0] * (3 - planes)
    fds, ids = [], []
    for i in range(count):
        ids.append(id_base + i)
        for p in range(planes):
            fds.append(memfd(sizes[p]))
    pool = DspBufferPool(client, width, height, fmt, ids, fds, strides, sizes)
    # keep test ownership: never send release on the wire
    client._send_release = mock.Mock()
    return pool


def nv12_array(width, height, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 256, (height, width), dtype=np.uint8)
    uv = rng.integers(0, 256, (height // 2, width), dtype=np.uint8)
    return np.vstack([y, uv])


def patched_alloc(client):
    """Route alloc_buffers() at memfd pools; record geometry.

    Consecutive allocations get distinct, increasing buffer ids (1000,
    1001, ...) so tests can tell src and dst pools apart.
    """
    calls = []
    state = {"next": 1000}

    def recording(width, height, fmt_wire, count):
        calls.append((width, height, fmt_wire, count))
        fmt = {0: "nv12", 4: "rgb24", 8: "gray8"}[fmt_wire]
        base = state["next"]
        state["next"] += count
        pool = make_pool(client, width, height, fmt, count, id_base=base)
        return (0, pool.count, 2 if fmt == "nv12" else 1,
                list(pool.strides), list(pool.plane_sizes), pool.ids,
                list(pool.plane_fds))

    client._exchange_alloc = recording
    return calls


def ok_resp(elapsed=1):
    return camera_pb2.DspJobResponse(success=True, elapsed_ms=elapsed)


class RecordingStub:
    """Fake gRPC stub; side_effect(request) -> DspJobResponse."""

    def __init__(self, side_effect=None):
        self.requests = []
        self.side_effect = side_effect or (lambda req: ok_resp())

    def SubmitDspJob(self, request, timeout=None):
        self.requests.append(request)
        return self.side_effect(request)


# ------------------------------------------------------------ wire codec --
class TestWireCodec:
    def test_alloc_request_layout(self):
        raw = alloc_request_bytes(1920, 1080, 0, 2)
        assert len(raw) == 24
        mtype, msize, w, h, fmt, count = struct.unpack(ALLOC_FMT, raw)
        assert (mtype, msize, w, h, fmt, count) == (7, 24, 1920, 1080, 0, 2)

    def test_alloc_resp_layout_roundtrip(self):
        payload = bytearray(RESP_SIZE)
        struct.pack_into("<I", payload, 0, 8)          # mtype = ALLOC_RESP
        struct.pack_into("<i", payload, 8, 0)          # code
        struct.pack_into("<I", payload, 12, 2)         # count
        struct.pack_into("<I", payload, 16, 2)         # num_planes
        for p in range(2):
            struct.pack_into("<I", payload, 20 + 4 * p, 1920 + p)
            struct.pack_into("<I", payload, 32 + 4 * p, 1080 * (1920 + p))
        for i in range(2):
            struct.pack_into("<Q", payload, 48 + 8 * i, 0xA000 + i)

        code, count, planes, strides, sizes, ids = parse_alloc_resp(bytes(payload))
        assert code == 0 and count == 2 and planes == 2
        assert strides[:2] == [1920, 1921]
        assert sizes[:2] == [1080 * 1920, 1080 * 1921]
        assert ids == [0xA000, 0xA001]

    def test_alloc_resp_error_payload(self):
        payload = bytearray(RESP_SIZE)
        struct.pack_into("<I", payload, 0, 8)          # mtype = ALLOC_RESP
        struct.pack_into("<i", payload, 8, -7)
        code, count, planes, strides, sizes, ids = parse_alloc_resp(bytes(payload))
        assert code == -7 and count == 0 and ids == []


# ------------------------------------------------------------------ pool --
class TestDspBufferPool:
    def test_nv12_roundtrip_with_stride_padding(self):
        client = DspClient()
        pool = make_pool(client, 64, 32, "nv12", count=1, stride_pad=16)
        src = nv12_array(64, 32)
        pool.write(0, src)
        out = pool.read(0)
        np.testing.assert_array_equal(out, src)

    def test_rgb_roundtrip_with_stride_padding(self):
        client = DspClient()
        pool = make_pool(client, 32, 32, "rgb24", count=2, stride_pad=8)
        src = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        pool.write(1, src)
        np.testing.assert_array_equal(pool.read(1), src)

    def test_gray_roundtrip(self):
        client = DspClient()
        pool = make_pool(client, 16, 16, "gray8", count=1)
        src = np.arange(256, dtype=np.uint8).reshape(16, 16)
        pool.write(0, src)
        np.testing.assert_array_equal(pool.read(0), src)

    def test_write_rejects_wrong_shape(self):
        client = DspClient()
        pool = make_pool(client, 64, 32, "nv12", count=1)
        with pytest.raises(DspError):
            pool.write(0, np.zeros((10, 64), dtype=np.uint8))
        with pytest.raises(DspError):
            pool.write(0, np.zeros((48, 64, 3), dtype=np.uint8))

    def test_release_is_idempotent_and_sends_per_buffer(self):
        client = DspClient()
        pool = make_pool(client, 64, 32, "nv12", count=3)
        ids = list(pool.ids)
        pool.release()
        pool.release()
        sent = [c.args[0] for c in client._send_release.call_args_list]
        assert sent == ids
        # all plane fds actually closed
        for fd in pool.plane_fds:
            with pytest.raises(OSError):
                os.fstat(fd)


# ------------------------------------------------------------- hw resize --
class TestResizeHw:
    def test_request_fields_and_explicit_bilinear(self):
        client = DspClient()
        calls = patched_alloc(client)
        stub = RecordingStub()
        client._stub = stub

        src = nv12_array(64, 32)
        client.resize_hw(src, 32, 16, fmt="nv12")

        req = stub.requests[0]
        assert req.op == camera_pb2.DSP_OP_RESIZE
        assert req.src_buffer_id == 1000
        assert list(req.dst_buffer_ids) == [1001]
        assert len(req.rects) == 0
        # the -2801 trap: proto default 0 (NEAREST) is rejected by vendor
        # MULTI_CROP; SDK must always send an explicit interpolation
        assert req.interpolation == camera_pb2.DSP_INTERP_BILINEAR
        assert req.scaling_mode == camera_pb2.DSP_SCALING_STRETCH
        # pools sized by target geometry
        alloc_geoms = [(c[0], c[1]) for c in calls]
        assert (64, 32) in alloc_geoms and (32, 16) in alloc_geoms
        assert client.last_used_hw is True

    def test_pools_reused_when_passed(self):
        client = DspClient()
        calls = patched_alloc(client)
        src_pool = make_pool(client, 64, 32, "nv12", 1)
        dst_pool = make_pool(client, 32, 16, "nv12", 1)
        client._stub = RecordingStub()
        client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                         src_pool=src_pool, dst_pool=dst_pool)
        n_after_one = len(calls)
        client.resize_hw(nv12_array(64, 32, seed=1), 32, 16, fmt="nv12",
                         src_pool=src_pool, dst_pool=dst_pool)
        assert len(calls) == n_after_one  # no re-alloc on the hot path

    def test_rgb_format_inference_from_3d_shape(self):
        client = DspClient()
        calls = patched_alloc(client)
        client._stub = RecordingStub()
        src = np.zeros((32, 32, 3), dtype=np.uint8)
        client.resize_hw(src, 16, 16)
        assert calls[0][2] == 4  # HalPixelFormat RGB24

    def test_2d_defaults_to_gray8_not_nv12(self):
        client = DspClient()
        calls = patched_alloc(client)
        client._stub = RecordingStub()
        client.resize_hw(np.zeros((32, 32), dtype=np.uint8), 16, 16)
        assert calls[0][2] == 8


# ---------------------------------------------------------------- hw crop --
class TestCropHw:
    def test_request_fields_and_default_dst_equals_crop(self):
        client = DspClient()
        patched_alloc(client)
        stub = RecordingStub()
        client._stub = stub
        client.crop_hw(nv12_array(64, 32), 16, 8, 32, 16, fmt="nv12")
        req = stub.requests[0]
        assert req.op == camera_pb2.DSP_OP_CROP_AND_RESIZE
        rect = req.rects[0]
        assert (rect.x, rect.y, rect.width, rect.height) == (16, 8, 32, 16)
        assert (rect.dst_width, rect.dst_height) == (32, 16)

    def test_explicit_dst_dims(self):
        client = DspClient()
        patched_alloc(client)
        stub = RecordingStub()
        client._stub = stub
        client.crop_hw(nv12_array(64, 32), 16, 8, 32, 16,
                       dst_width=64, dst_height=32, fmt="nv12")
        rect = stub.requests[0].rects[0]
        assert (rect.dst_width, rect.dst_height) == (64, 32)

    def test_out_of_bounds_rect_raises_before_submit(self):
        client = DspClient()
        patched_alloc(client)
        stub = RecordingStub()
        client._stub = stub
        with pytest.raises(DspError):
            client.crop_hw(nv12_array(64, 32), 48, 0, 32, 16, fmt="nv12")
        assert stub.requests == []


# ----------------------------------------------------------- hw multi-crop --
class TestMultiCropHw:
    def test_request_fields(self):
        client = DspClient()
        patched_alloc(client)
        stub = RecordingStub()
        client._stub = stub
        rects = [(0, 0, 32, 32, 16, 16), (32, 0, 32, 32, 32, 32)]
        client.multi_crop_hw(nv12_array(64, 32), rects, fmt="nv12")
        req = stub.requests[0]
        assert req.op == camera_pb2.DSP_OP_MULTI_CROP_AND_RESIZE
        assert len(req.rects) == 2
        assert len(req.dst_buffer_ids) == 2
        got = [(r.x, r.y, r.width, r.height, r.dst_width, r.dst_height)
               for r in req.rects]
        assert got == rects

    def test_empty_rects_rejected(self):
        client = DspClient()
        with pytest.raises(DspError):
            client.multi_crop_hw(nv12_array(64, 32), [], fmt="nv12")

    def test_source_array_is_written_into_the_src_pool(self, monkeypatch):
        """Regression (found on-device): multi_crop_hw once submitted the
        job without src_pool.write(), so the DSP cropped zero-initialized
        memory and every tile came back empty."""
        client = DspClient()
        patched_alloc(client)
        client._stub = RecordingStub()
        written = []
        real_write = DspBufferPool.write

        def spy(pool_self, idx, arr):
            written.append((pool_self.width, pool_self.height))
            return real_write(pool_self, idx, arr)

        monkeypatch.setattr(DspBufferPool, "write", spy)
        src = nv12_array(64, 32, seed=5)
        client.multi_crop_hw(src, [(0, 0, 32, 16, 16, 16),
                                   (32, 0, 32, 16, 16, 16)], fmt="nv12")
        assert (64, 32) in written  # src geometry reached the wire exactly once
        assert written.count((64, 32)) == 1


# -------------------------------------------------------------- fallback --
def unimplemented_client():
    """Client whose daemon lacks SubmitDspJob — must take the CPU path."""
    client = DspClient()
    patched_alloc(client)
    err = grpc.RpcError("no such method")
    err.code = lambda: grpc.StatusCode.UNIMPLEMENTED
    client._stub = mock.Mock()
    client._stub.SubmitDspJob.side_effect = err
    return client


class TestFallback:
    def test_unimplemented_rpc_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(dsp, "_cv2", None)  # deterministic nearest
        client = unimplemented_client()
        src = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
        out = client.resize_hw(src, 16, 16)
        assert out.shape == (16, 16)
        assert client.last_used_hw is False
        np.testing.assert_array_equal(out, src[::2, ::2])  # nearest decimation

    def test_service_unavailable_falls_back(self):
        client = DspClient()
        patched_alloc(client)
        client._stub = RecordingStub(
            lambda req: camera_pb2.DspJobResponse(
                success=False, error_code=DSP_SERVICE_UNAVAILABLE,
                message="dsp service not running"))
        src = np.zeros((32, 32), dtype=np.uint8)
        out = client.resize_hw(src, 16, 16)
        assert out.shape == (16, 16)
        assert client.last_used_hw is False

    def test_genuine_error_raises(self):
        client = DspClient()
        patched_alloc(client)
        client._stub = RecordingStub(
            lambda req: camera_pb2.DspJobResponse(
                success=False, error_code=-1, message="bad rect"))
        with pytest.raises(DspError) as ei:
            client.resize_hw(np.zeros((64, 32), dtype=np.uint8), 32, 16)
        assert ei.value.code == -1

    def test_cpu_letterbox_geometry(self):
        client = unimplemented_client()
        src = np.full((16, 32), 200, dtype=np.uint8)
        out = client.resize_hw(src, 16, 16, scaling="letterbox")
        assert out.shape == (16, 16)
        # 32x16 -> 16x16: scale 0.5 -> content rows 4..11, pad rows 0..3/12..15
        np.testing.assert_array_equal(out[4:12], np.full((8, 16), 200))
        np.testing.assert_array_equal(out[0], np.zeros(16))
        np.testing.assert_array_equal(out[15], np.zeros(16))

    def test_cpu_nv12_crop_requires_even_coords(self):
        client = unimplemented_client()
        with pytest.raises(DspError):
            client.crop_hw(nv12_array(64, 32), 1, 0, 32, 16, fmt="nv12")

    def test_cpu_multi_crop_matches_single_crop(self):
        client = unimplemented_client()
        src = nv12_array(64, 32, seed=3)
        single = client.crop_hw(src, 0, 0, 32, 16, fmt="nv12")
        multi = client.multi_crop_hw(src, [(0, 0, 32, 16, 32, 16)], fmt="nv12")
        np.testing.assert_array_equal(single, multi[0])


# ------------------------------------------------------------ validation --
class TestValidation:
    @pytest.mark.parametrize("w,h", [(8, 32), (32, 8), (8193, 32), (32, 8193)])
    def test_dims_out_of_daemon_range_rejected_locally(self, w, h):
        client = DspClient()
        with pytest.raises(DspError):
            client.resize_hw(np.zeros((32, 64), dtype=np.uint8), w, h)

    def test_nv12_requires_even_width(self):
        client = DspClient()
        with pytest.raises(DspError):
            client.resize_hw(np.zeros((48, 33), dtype=np.uint8), 32, 16,
                             fmt="nv12")

    def test_alloc_count_beyond_fd_cap_rejected(self):
        client = DspClient()
        with pytest.raises(DspError):
            client.alloc_buffers(64, 32, "nv12", count=33)  # 33*2 > 64 fds


# ------------------------------------------------------------ life cycle --
class TestLifecycle:
    def test_close_is_idempotent(self):
        client = DspClient()
        client.close()
        client.close()

    def test_context_manager(self):
        with DspClient() as client:
            pass
        assert client._stub is None
