"""Tests for the DSP offload client (SDK-2).

Covers: UDS alloc/release wire codec, dma-buf pool write/read roundtrips
(memfd-backed), hardware job request construction (op/rects/interpolation),
the NEAREST-default trap, fallback semantics, and client-side validation
mirroring the daemon's caps.
"""

import os
import socket
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
    import_request_bytes,
    parse_alloc_resp,
    parse_import_resp,
)
from neoruntime_ipc_sdk.media import Frame, FrameHandle
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


# ------------------------------------------------- zero-copy handle source --
def make_handle(width, height, fmt="NV12", frame_id=77):
    """memfd-backed FrameHandle mimicking a keep-fd camera frame."""
    planes = 2 if fmt == "NV12" else 1
    if fmt == "NV12":
        row_w, rows = (width, width), (height, height // 2)
    elif fmt in ("RGB", "BGR"):
        row_w, rows = (width * 3,), (height,)
    else:
        row_w, rows = (width,), (height,)
    fds = [memfd(row_w[p] * rows[p]) for p in range(planes)]
    strides = tuple(row_w) + (0,) * (3 - planes)
    sizes = tuple(row_w[p] * rows[p] for p in range(planes)) + (0,) * (3 - planes)
    return FrameHandle(fds, strides, sizes, frame_id,
                       width=width, height=height, format=fmt)


def handle_client():
    """Client wired for handle-source tests: import mocked to id 4242."""
    client = DspClient()
    calls = patched_alloc(client)
    stub = RecordingStub()
    client._stub = stub
    client._import_source = mock.Mock(return_value=4242)
    client._send_release = mock.Mock()
    return client, calls, stub


class TestImportWireCodec:
    def test_import_request_layout(self):
        raw = import_request_bytes(3840, 2160, 0, 2,
                                   [3840, 1920, 0], [3317760, 1658880, 0])
        assert len(raw) == 48
        assert struct.unpack("<12I", raw) == (
            10, 48, 3840, 2160, 0, 2, 3840, 1920, 0, 3317760, 1658880, 0)

    def test_parse_import_resp_roundtrip(self):
        ok = struct.pack("<IIi4xq", 11, 24, 0, 4242)
        assert parse_import_resp(ok) == (0, 4242)
        bad = struct.pack("<IIi4xq", 11, 24, -1, -1)
        assert parse_import_resp(bad) == (-1, -1)

    def test_parse_import_resp_rejects_wrong_type_and_short(self):
        with pytest.raises(DspError):
            parse_import_resp(struct.pack("<IIi4xq", 5, 24, 0, 1))
        with pytest.raises(DspError):
            parse_import_resp(b"\x0b\x00\x00\x00")


class TestHandleSource:
    def test_resize_hw_uses_imported_id_without_src_alloc(self):
        client, calls, stub = handle_client()
        out = client.resize_hw(make_handle(64, 32), 32, 16)
        req = stub.requests[0]
        assert req.src_buffer_id == 4242
        assert list(req.dst_buffer_ids) == [1000]
        # only the destination was allocated — the source never copied in
        assert [(c[0], c[1]) for c in calls] == [(32, 16)]
        client._import_source.assert_called_once()
        # import freed, and the temp dst pool too — in that order
        assert [c.args[0] for c in client._send_release.call_args_list] \
            == [4242, 1000]
        assert out.shape == (16 * 3 // 2, 32)  # nv12 read-back layout
        assert client.last_used_hw is True

    def test_frame_with_handle_and_frame_with_pixels_only(self):
        client, calls, stub = handle_client()
        frame = Frame(sequence=1, timestamp_ns=2, width=64, height=32,
                      format="NV12", image=None, handle=make_handle(64, 32))
        client.resize_hw(frame, 32, 16)
        assert stub.requests[0].src_buffer_id == 4242

        client2, calls2, stub2 = handle_client()
        frame2 = Frame(sequence=1, timestamp_ns=2, width=64, height=32,
                       format="NV12", image=nv12_array(64, 32))
        client2.resize_hw(frame2, 32, 16, fmt="nv12")
        assert stub2.requests[0].src_buffer_id == 1000  # copy-in path
        assert (64, 32) in [(c[0], c[1]) for c in calls2]

    def test_pixel_frame_format_metadata_wins_over_shape_inference(self):
        # a 2D NV12 array is shape-ambiguous (gray8?); a Frame says "NV12"
        # and that metadata must pick nv12 without an explicit fmt=
        client, calls, stub = handle_client()
        frame = Frame(sequence=1, timestamp_ns=2, width=64, height=32,
                      format="NV12", image=nv12_array(64, 32))
        client.resize_hw(frame, 32, 16)
        assert (64, 32) in [(c[0], c[1]) for c in calls]  # nv12 geometry,
        # not the gray8 reading (64x96) a bare array would infer

    def test_multi_crop_hw_on_handle_bounds_checked_against_frame(self):
        client, calls, stub = handle_client()
        client.multi_crop_hw(make_handle(64, 32),
                             [(0, 0, 32, 16, 16, 16), (32, 0, 32, 16, 16, 16)])
        req = stub.requests[0]
        assert req.src_buffer_id == 4242
        assert len(req.dst_buffer_ids) == 2
        with pytest.raises(DspError):
            client.multi_crop_hw(make_handle(64, 32),
                                 [(48, 0, 32, 16, 16, 16)])  # outside 64x32

    def test_handle_error_paths(self):
        client, *_ = handle_client()
        # no handle, no pixels
        with pytest.raises(DspError, match="keep_fd"):
            client.resize_hw(Frame(1, 2, 64, 32, "NV12", None), 32, 16)
        # closed handle
        closed = make_handle(64, 32)
        closed.close()
        with pytest.raises(DspError, match="clos"):
            client.resize_hw(closed, 32, 16)
        # src_pool is meaningless for an imported source
        client2, *_ = handle_client()
        with pytest.raises(DspError, match="src_pool"):
            client2.resize_hw(make_handle(64, 32), 32, 16,
                              src_pool=make_pool(client2, 64, 32, "nv12", 1))
        # frame format the DSP cannot take as-is
        with pytest.raises(DspError, match="NV21"):
            client.resize_hw(make_handle(64, 32, fmt="NV21"), 32, 16)
        # explicit fmt disagreeing with the frame
        with pytest.raises(DspError, match="format"):
            client.resize_hw(make_handle(64, 32), 32, 16, fmt="gray8")

    def test_cpu_fallback_refused_for_handle_source(self):
        client, *_ = handle_client()
        client._submit_job = mock.Mock(
            side_effect=dsp._DspUnavailable("SubmitDspJob not in daemon"))
        with pytest.raises(DspError, match="zero-copy"):
            client.resize_hw(make_handle(64, 32), 32, 16)


# ------------------------------------------------------- import exchange --
class FakeSock:
    """Records sendmsg; replies are fed via the patched _recvmsg_with_fds."""

    def __init__(self):
        self.sent = []
        self.timeout = None

    def sendmsg(self, buffers, ancdata, flags=0, address=None):
        self.sent.append((b"".join(buffers), ancdata))

    def settimeout(self, t):
        self.timeout = t


def chunked_recv(chunks):
    """_recvmsg_with_fds stand-in yielding fixed chunks regardless of the
    requested bufsize (exercises the accumulating loops)."""
    it = iter(chunks)

    def _recv(sock, bufsize, max_fds=0):
        data = next(it)
        assert len(data) <= bufsize
        return data, []

    return _recv


class TestImportExchange:
    def _client(self, monkeypatch, chunks):
        client = DspClient()
        client._sock = FakeSock()
        monkeypatch.setattr(dsp, "_recvmsg_with_fds", chunked_recv(chunks))
        return client

    def test_import_sends_fds_and_skips_ok_ack(self, monkeypatch):
        handle = make_handle(64, 32)
        ok_ack = struct.pack("<II", 5, 12) + b"\x00" * 4  # control ack, 4B body
        resp = struct.pack("<IIi4xq", 11, 24, 0, 555)
        client = self._client(monkeypatch, [ok_ack[:5], ok_ack[5:8],
                                            ok_ack[8:], resp[:8], resp[8:]])
        import_id = client._import_source(handle, 64, 32, "nv12")
        assert import_id == 555

        payload, anc = client._sock.sent[0]
        assert payload == import_request_bytes(
            64, 32, 0, 2, [64, 64, 0], [64 * 32, 64 * 16, 0])
        assert len(anc) == 1
        level, ctype, cdata = anc[0]
        assert level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS
        assert struct.unpack("2i", cdata) == tuple(handle.fds)
        # our fd copies are NOT closed by the send — the handle owns them
        for fd in handle.fds:
            assert os.fstat(fd).st_size > 0
        handle.close()

    def test_import_rejects_error_code(self, monkeypatch):
        resp = struct.pack("<IIi4xq", 11, 24, -1, -1)
        client = self._client(monkeypatch, [resp[:8], resp[8:]])
        with pytest.raises(DspError) as ei:
            client._import_source(make_handle(64, 32), 64, 32, "nv12")
        assert ei.value.code == -1

    def test_import_rejects_unexpected_stream_message(self, monkeypatch):
        # a FRAME on the DSP socket means protocol desync — never skip it
        frame_hdr = struct.pack("<II", 3, 16)
        client = self._client(monkeypatch,
                              [frame_hdr, b"\x00" * 8])
        with pytest.raises(DspError, match="type 3"):
            client._import_source(make_handle(64, 32), 64, 32, "nv12")

    def test_import_timeout_mentions_daemon_version(self, monkeypatch):
        def stall(sock, bufsize, max_fds=0):
            raise socket.timeout("timed out")

        client = DspClient()
        client._sock = FakeSock()
        monkeypatch.setattr(dsp, "_recvmsg_with_fds", stall)
        with pytest.raises(DspError, match="DSP_IMPORT"):
            client._import_source(make_handle(64, 32), 64, 32, "nv12")
