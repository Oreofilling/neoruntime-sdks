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
    DspBufferRef,
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
    elif fmt == "argb":
        row_w, rows = (width * 4,), (height,)
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
        fmt = {0: "nv12", 4: "rgb24", 6: "argb", 8: "gray8"}[fmt_wire]
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
        with pytest.warns(DeprecationWarning, match="fmt="):
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
        from neoruntime_ipc_sdk import dsp_format
        # _resize_plane lives in dsp_format post-split
        monkeypatch.setattr(dsp_format, "_cv2", None)  # deterministic nearest
        client = unimplemented_client()
        src = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.resize_hw(src, 16, 16, fmt="gray8")
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
        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.resize_hw(src, 16, 16, fmt="gray8")
        assert out.shape == (16, 16)
        assert client.last_used_hw is False

    def test_genuine_error_raises(self):
        client = DspClient()
        patched_alloc(client)
        client._stub = RecordingStub(
            lambda req: camera_pb2.DspJobResponse(
                success=False, error_code=-1, message="bad rect"))
        with pytest.raises(DspError) as ei:
            client.resize_hw(np.zeros((64, 32), dtype=np.uint8), 32, 16,
                             fmt="gray8")
        assert ei.value.code == -1

    def test_cpu_letterbox_geometry(self):
        client = unimplemented_client()
        src = np.full((16, 32), 200, dtype=np.uint8)
        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.resize_hw(src, 16, 16, scaling="letterbox",
                                   fmt="gray8")
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
        with pytest.warns(UserWarning, match="CPU fallback"):
            single = client.crop_hw(src, 0, 0, 32, 16, fmt="nv12")
            multi = client.multi_crop_hw(src, [(0, 0, 32, 16, 32, 16)],
                                         fmt="nv12")
        np.testing.assert_array_equal(single, multi[0])


# ------------------------------------------------------------ validation --
class TestValidation:
    @pytest.mark.parametrize("w,h", [(8, 32), (32, 8), (8193, 32), (32, 8193)])
    def test_dims_out_of_daemon_range_rejected_locally(self, w, h):
        client = DspClient()
        with pytest.raises(DspError):
            client.resize_hw(np.zeros((32, 64), dtype=np.uint8), w, h,
                             fmt="gray8")

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

    def test_close_poisons_live_pools_and_refs(self):
        # review regression: any failed shared call drops the resident
        # client; its pools' buffers are reclaimed with the UDS, so a
        # held ref must fail fast instead of submitting stale ids
        client = DspClient()
        patched_alloc(client)
        pool = client.alloc_buffers(64, 48, "rgb24", count=1)
        ref = DspBufferRef(client, pool, 0, owns=())
        assert not ref.released

        client.close()

        assert pool._released
        with pytest.raises(DspError, match="pool is gone"):
            _ = ref.buffer_id
        with pytest.raises(DspError, match="released pool"):
            pool.read(0)


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
        # error responses carry code<0; the daemon's id field is uint64
        # (dsp_service.cpp), so an error body still has an unsigned id.
        bad = struct.pack("<IIi4xQ", 11, 24, -1, 0)
        assert parse_import_resp(bad) == (-1, 0)

    def test_parse_import_resp_high_bit_import_id(self):
        # Regression (device run 2026-09-09): daemon import ids are uint64
        # and routinely have the high bit set; the response format used a
        # signed q, so such ids parsed negative and then failed
        # DspJobRequest field validation with ValueError (blend_hw ERROR).
        big = 16053137586893052802  # observed live on the primary device
        raw = struct.pack("<IIi4xQ", 11, 24, 0, big)
        code, import_id = parse_import_resp(raw)
        assert import_id == big
        # the parsed id must be accepted by the proto field it feeds
        camera_pb2.DspJobRequest(src_buffer_id=import_id)

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


# --------------------------------------------------------- out="ref" (P2-2) --
class TestBufferRef:
    """Device-resident sync results: DspBufferRef out/ref-source wiring."""

    def _ref_client(self):
        client = DspClient()
        patched_alloc(client)
        client._stub = RecordingStub()
        client._send_release = mock.Mock()
        return client

    def test_resize_out_ref_returns_ref_without_readback_release(self):
        client = self._ref_client()
        ref = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                               out="ref")
        assert isinstance(ref, DspBufferRef)
        assert (ref.width, ref.height, ref.fmt) == (32, 16, "nv12")
        # dst pool id (src alloc'd first: 1000 src, 1001 dst)
        assert ref.buffer_id == 1001
        # alive ref owns its buffers — nothing released yet
        client._send_release.assert_not_called()
        assert ref.released is False

    def test_ref_read_matches_sync_shape_and_release_frees_all(self):
        client = self._ref_client()
        ref = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                               out="ref")
        arr = ref.read()
        assert arr.shape == (16 * 3 // 2, 32) and arr.dtype == np.uint8
        ref.release()
        # both the temp src pool and the dst pool go back to the daemon
        assert [c.args[0] for c in client._send_release.call_args_list] \
            == [1000, 1001]
        ref.release()  # idempotent
        assert len(client._send_release.call_args_list) == 2
        with pytest.raises(DspError, match="released"):
            ref.read()
        with pytest.raises(DspError, match="released"):
            ref.buffer_id

    def test_ref_is_context_manager(self):
        client = self._ref_client()
        with client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                              out="ref") as ref:
            assert ref.buffer_id == 1001
        assert ref.released is True
        assert len(client._send_release.call_args_list) == 2

    def test_ref_as_source_uses_id_without_import_or_copy_in(self):
        client = self._ref_client()
        ref = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                               out="ref")
        client._send_release.reset_mock()
        client._import_source = mock.Mock()

        small = client.resize_hw(ref, 16, 16)  # ref consumed as plain src

        req = client._stub.requests[-1]
        assert req.src_buffer_id == 1001  # the ref's daemon buffer, direct
        client._import_source.assert_not_called()
        # only the new dst was allocated; nothing re-imported or copied in
        assert small.shape == (16 * 3 // 2, 16)
        # borrow semantics: the ref was NOT consumed by being a source
        # (only the second call's own temp dst was released)
        assert [c.args[0] for c in client._send_release.call_args_list] \
            == [1002]
        assert ref.released is False
        ref.release()

    def test_released_ref_as_source_raises(self):
        client = self._ref_client()
        ref = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                               out="ref")
        ref.release()
        with pytest.raises(DspError, match="released"):
            client.resize_hw(ref, 16, 8)

    def test_ref_source_format_mismatch_raises(self):
        client = self._ref_client()
        ref = client.resize_hw(np.zeros((32, 32, 3), np.uint8), 16, 16,
                               out="ref")
        with pytest.raises(DspError, match="format mismatch"):
            client.resize_hw(ref, 8, 8, fmt="nv12")

    def test_out_mode_validation(self):
        client = self._ref_client()
        with pytest.raises(DspError, match="must be None or 'ref'"):
            client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                             out="pool")
        with pytest.raises(DspError, match="wait=False"):
            client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                             wait=False, out="ref")

    def test_crop_and_convert_out_ref(self):
        client = self._ref_client()
        rref = client.crop_hw(nv12_array(64, 32), 8, 8, 32, 16, fmt="nv12",
                              out="ref")
        assert (rref.width, rref.height, rref.fmt) == (32, 16, "nv12")
        rref.release()

        cref = client.convert_hw(np.zeros((32, 64, 3), np.uint8), "nv12",
                                 out="ref")
        assert (cref.width, cref.height, cref.fmt) == (64, 32, "nv12")
        arr = cref.read()
        assert arr.shape == (32 * 3 // 2, 64)

    def test_ref_with_caller_dst_pool_is_not_released_by_ref(self):
        client = self._ref_client()
        dst_pool = make_pool(client, 32, 16, "nv12", 1, id_base=7777)
        client._send_release.reset_mock()
        ref = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                               dst_pool=dst_pool, out="ref")
        assert ref.buffer_id == 7777
        ref.release()
        # only the temp src pool (1000) — the caller's pool survives
        assert [c.args[0] for c in client._send_release.call_args_list] \
            == [1000]
        # releasing the pool under a (still-alive) ref is caught on use
        ref2 = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                                dst_pool=dst_pool, out="ref")
        dst_pool.release()
        with pytest.raises(DspError, match="pool is gone"):
            ref2.read()

    def test_encode_jpeg_from_ref_uses_daemon_id(self):
        client = self._ref_client()
        ref = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                               out="ref")
        client._send_release.reset_mock()
        client._encode_rpc = mock.Mock(return_value=b"\xff\xd8jpeg")
        out = client.encode_jpeg_hw(ref, quality=85)
        assert out == b"\xff\xd8jpeg"
        client._encode_rpc.assert_called_once_with(1001, 85, 5.0)
        client._send_release.assert_not_called()  # borrow: ref untouched

    def test_blend_ref_base_requires_zero_copy(self):
        client = self._ref_client()
        ref = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                               out="ref")
        before = len(client._stub.requests)  # the producing resize
        overlay = np.zeros((16, 16, 4), np.uint8)
        with pytest.raises(DspError, match="device-side bases"):
            client.blend_hw(ref, [(overlay, 0, 0)])
        assert len(client._stub.requests) == before  # no new wire work
        ref.release()

    def test_blend_ref_base_zero_copy_wire(self):
        # P2-4: with zero_copy=True the ref rides its daemon id into a
        # 1:1 RESIZE copy, then the blend composites on that copy — no
        # import for the base, no release of the borrowed id.
        client = self._ref_client()
        ref = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                               out="ref")
        client._send_release.reset_mock()
        client._import_source = mock.Mock()  # a ref base must NOT import
        client._import_memfd = mock.Mock(return_value=2000)
        overlay = np.zeros((16, 16, 4), np.uint8)
        overlay[..., 3] = 255

        out = client.blend_hw(ref, [(overlay, 8, 0)], zero_copy=True)

        # the producing resize (1000->1001) is first in the log; the
        # blend itself is the last two requests
        reqs = client._stub.requests[-2:]
        assert [r.op for r in reqs] == [
            camera_pb2.DSP_OP_RESIZE, camera_pb2.DSP_OP_BLEND,
        ]
        # copy leg: the ref's own daemon buffer (1001) -> fresh base
        # pool (1002; 1000/1001 were the producing resize's src/dst)
        assert reqs[0].src_buffer_id == 1001
        assert list(reqs[0].dst_buffer_ids) == [1002]
        client._import_source.assert_not_called()
        # blend leg: composites in place on the base pool copy
        assert reqs[1].src_buffer_id == 1002
        assert list(reqs[1].dst_buffer_ids) == [2000]  # overlay memfd import
        # read-back comes from the base pool; shape is the base geometry
        assert out.shape == (16 * 3 // 2, 32)
        # the borrowed ref id is never released; releasing the ref frees
        # only its own pool ids — the blend's allocations already went
        # back when the sync call unwound
        assert ref.released is False
        client._send_release.reset_mock()
        ref.release()
        released = [c.args[0] for c in client._send_release.call_args_list]
        assert released == [1000, 1001]
    def test_blend_ref_base_failure_names_the_copy_out(self):
        client = self._ref_client()
        ref = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                               out="ref")

        def fail_blend(req):
            if req.op == camera_pb2.DSP_OP_BLEND:
                return camera_pb2.DspJobResponse(
                    success=False, message="firmware says no")
            return camera_pb2.DspJobResponse(success=True)

        client._stub = RecordingStub(side_effect=fail_blend)
        client._import_memfd = mock.Mock(return_value=2000)
        overlay = np.zeros((16, 16, 4), np.uint8)
        try:
            with pytest.raises(DspError, match="ref.read()"):
                client.blend_hw(ref, [(overlay, 0, 0)], zero_copy=True)
        finally:
            ref.release()

    def test_gray8_ref_encode_refused_without_silent_copy(self):
        client = self._ref_client()
        ref = client.resize_hw(np.zeros((32, 32), np.uint8), 16, 16,
                               fmt="gray8", out="ref")
        with pytest.raises(DspError, match="gray8"):
            client.encode_jpeg_hw(ref, quality=85)


# ------------------------------------------------------- blend destination (P2-5) --
class TestBlendDst:
    """out= destinations for blend_hw: a caller pool slot (the zero-copy
    publish tail) and the ref form (annotated pixels stay device-side)."""

    def _client(self):
        client = DspClient()
        patched_alloc(client)
        client._stub = RecordingStub()
        client._send_release = mock.Mock()
        return client

    @staticmethod
    def _opaque_overlay():
        ov = np.zeros((16, 16, 4), np.uint8)
        ov[..., 3] = 255
        return ov

    def test_blend_out_pool_ref_base_wire(self):
        # the publish chain: ref base -> RESIZE into the caller's slot ->
        # BLEND in place there; no base alloc, no read-back, no release
        client = self._client()
        ref = client.resize_hw(nv12_array(64, 32), 32, 16, fmt="nv12",
                               out="ref")
        dst = make_pool(client, 32, 16, "nv12", 2, id_base=7777)
        client._send_release.reset_mock()
        client._import_source = mock.Mock()
        client._import_memfd = mock.Mock(return_value=2000)

        out = client.blend_hw(ref, [(self._opaque_overlay(), 0, 0)],
                              zero_copy=True, out=dst, dst_slot=1)

        assert out is None
        reqs = client._stub.requests[-2:]
        assert [r.op for r in reqs] == [
            camera_pb2.DSP_OP_RESIZE, camera_pb2.DSP_OP_BLEND,
        ]
        # copy leg lands in the caller's slot 1 (7778) — not a fresh alloc
        assert reqs[0].src_buffer_id == 1001
        assert list(reqs[0].dst_buffer_ids) == [7778]
        client._import_source.assert_not_called()
        # blend composites in place on that same slot
        assert reqs[1].src_buffer_id == 7778
        assert list(reqs[1].dst_buffer_ids) == [2000]  # overlay memfd
        # only the overlay import went back; the caller pool is untouched
        assert [c.args[0] for c in client._send_release.call_args_list] == [2000]
        assert dst._released is False
        assert dst.read(1).shape == (16 * 3 // 2, 32)  # slot stays usable
        ref.release()

    def test_blend_out_pool_array_base_writes_slot(self):
        client = self._client()
        dst = make_pool(client, 32, 16, "nv12", 1, id_base=8800)
        client._import_memfd = mock.Mock(return_value=2000)

        out = client.blend_hw(nv12_array(32, 16),
                              [(self._opaque_overlay(), 0, 0)], out=dst)

        assert out is None
        # array base: no RESIZE leg — the pixels were written into the
        # caller's slot, the blend runs on it
        reqs = client._stub.requests[-1:]
        assert reqs[0].op == camera_pb2.DSP_OP_BLEND
        assert reqs[0].src_buffer_id == 8800
        assert [c.args[0] for c in client._send_release.call_args_list] == [2000]
        assert dst.read(0).shape == (16 * 3 // 2, 32)

    def test_blend_out_pool_geometry_and_fmt_mismatch(self):
        client = self._client()
        wrong = make_pool(client, 64, 32, "nv12", 1, id_base=8800)
        with pytest.raises(DspError, match="blend keeps dims"):
            client.blend_hw(nv12_array(32, 16),
                            [(self._opaque_overlay(), 0, 0)], out=wrong)
        rgb = make_pool(client, 32, 16, "rgb24", 1, id_base=8801)
        with pytest.raises(DspError, match="nv12"):
            client.blend_hw(nv12_array(32, 16),
                            [(self._opaque_overlay(), 0, 0)], out=rgb)
        assert client._stub.requests == []  # refused before any wire work

    def test_blend_out_pool_slot_range(self):
        client = self._client()
        dst = make_pool(client, 32, 16, "nv12", 2, id_base=7777)
        with pytest.raises(DspError, match="outside the dst_pool"):
            client.blend_hw(nv12_array(32, 16),
                            [(self._opaque_overlay(), 0, 0)], out=dst,
                            dst_slot=2)

    def test_blend_out_ref_returns_ref(self):
        client = self._client()
        client._import_memfd = mock.Mock(return_value=2000)
        ref = client.blend_hw(nv12_array(32, 16),
                              [(self._opaque_overlay(), 0, 0)], out="ref")
        assert isinstance(ref, DspBufferRef)
        assert (ref.width, ref.height, ref.fmt) == (32, 16, "nv12")
        assert ref.buffer_id == 1000  # the blend's own base pool
        assert ref.read().shape == (16 * 3 // 2, 32)
        client._send_release.assert_not_called()  # owns it until released
        ref.release()
        # base pool + overlay import go back, in own order
        assert [c.args[0] for c in client._send_release.call_args_list] == \
            [1000, 2000]

    def test_blend_out_mode_validation(self):
        client = self._client()
        base, ov = nv12_array(32, 16), [(self._opaque_overlay(), 0, 0)]
        with pytest.raises(DspError, match="None, 'ref' or a DspBufferPool"):
            client.blend_hw(base, ov, out="pool")
        dst = make_pool(client, 32, 16, "nv12", 1, id_base=7777)
        with pytest.raises(DspError, match="wait=False"):
            client.blend_hw(base, ov, out=dst, wait=False)
        with pytest.raises(DspError, match="wait=False"):
            client.blend_hw(base, ov, out="ref", wait=False)
        assert client._stub.requests == []

    def test_blend_out_pool_cpu_fallback_writes_slot(self):
        # an array base with out=pool must land its pixels in the slot
        # even when the blend falls back to CPU — the destination
        # contract holds on every path
        client = self._client()
        dst = make_pool(client, 32, 16, "nv12", 1, id_base=8800)
        client._import_memfd = mock.Mock(return_value=2000)
        client._submit_job = mock.Mock(
            side_effect=dsp._DspUnavailable("dsp service down")
        )
        base = nv12_array(32, 16, seed=3)
        overlay = np.zeros((16, 16, 4), np.uint8)
        overlay[..., 0] = 200  # R
        overlay[..., 3] = 255  # opaque
        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.blend_hw(base, [(overlay, 0, 0)], out=dst)
        assert out is None
        np.testing.assert_array_equal(
            dst.read(0), dsp._cpu_blend(base, "nv12", [(overlay, 0, 0)])
        )


# -------------------------------------------------- blend overlay prep (P2-4) --
class TestBlendOverlayPrep:
    """Transparent padding: 16px daemon floor + even-height DSP contract.

    Both pads grow the rect with fully transparent pixels (no-ops), so
    the in-bounds check must run on the padded geometry.
    """

    def _client(self):
        client = DspClient()
        patched_alloc(client)
        client._stub = RecordingStub()
        client._send_release = mock.Mock()
        client._import_memfd = mock.Mock(return_value=2000)
        return client

    def test_odd_height_overlay_padded_to_even(self):
        # device ruling: odd width/x/y blend fine, odd height is
        # rejected by firmware (-2801) — pad one transparent row.
        client = self._client()
        overlay = np.zeros((17, 33, 4), np.uint8)
        client.blend_hw(nv12_array(64, 32), [(overlay, 1, 1)])

        blend = client._stub.requests[-1]
        rect = blend.rects[0]
        # width/x/y untouched (odd is fine there); height padded 17->18
        assert (rect.x, rect.y, rect.width, rect.height) == (1, 1, 33, 18)
        assert (rect.dst_width, rect.dst_height) == (33, 18)
        # the padded row is fully transparent in the imported plane
        # (wire order is [A, R, G, B])
        wire = client._import_memfd.call_args.args[0]
        argb = np.frombuffer(wire, np.uint8).reshape(18, 33, 4)
        assert (argb[17, :, 0] == 0).all()

    def test_odd_flush_bottom_overlay_refused_with_hint(self):
        # 17-row overlay flush to the base bottom: the transparent pad
        # row would exceed the base — refuse with the padding hint.
        client = self._client()
        overlay = np.zeros((17, 32, 4), np.uint8)
        with pytest.raises(DspError, match="includes transparent padding"):
            client.blend_hw(nv12_array(64, 32), [(overlay, 0, 32 - 17)])
        assert client._stub.requests == []

    def test_small_overlay_pad_now_bounds_checked(self):
        # regression: a <16 overlay used to be padded AFTER the bounds
        # check, so a corner placement could exceed the base unflagged.
        client = self._client()
        overlay = np.zeros((10, 10, 4), np.uint8)
        with pytest.raises(DspError, match="exceeds the"):
            client.blend_hw(nv12_array(32, 16), [(overlay, 22, 6)])
        assert client._stub.requests == []

    def test_small_overlay_corner_fits_after_pad_when_in_bounds(self):
        client = self._client()
        overlay = np.zeros((10, 10, 4), np.uint8)
        out = client.blend_hw(nv12_array(32, 16), [(overlay, 0, 0)])
        rect = client._stub.requests[-1].rects[0]
        assert (rect.width, rect.height) == (16, 16)  # padded to floor
        assert out.shape == (16 * 3 // 2, 32)

    def test_even_overlay_untouched(self):
        client = self._client()
        overlay = np.zeros((16, 16, 4), np.uint8)
        client.blend_hw(nv12_array(64, 32), [(overlay, 3, 5)])
        rect = client._stub.requests[-1].rects[0]
        assert (rect.x, rect.y, rect.width, rect.height) == (3, 5, 16, 16)
