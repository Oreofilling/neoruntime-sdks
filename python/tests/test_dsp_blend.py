"""Tests for DspClient.blend_hw (the BLEND job op, dsp-offload P1),
draw.render_overlay_rgba and the draw_detections router legs.

Covers: request shape (op, in-place base id, one dst id + placement rect
per overlay), the hardware ARGB32 wire byte order ([A, R, G, B] per
pixel), the memfd-import overlay transport (the deployed HAL on 93.72
refuses ARGB32 pool allocs — overlays ride DSP_IMPORT/USERPTR instead),
the daemon floor (sub-16 overlays padded transparent), copy semantics
(blend runs on the pool copy — input untouched, result read back from
the base buffer), the keep-fd base zero-copy chain (import + 1:1 RESIZE
copy + in-place BLEND — P2), client-side validation (nv12-only base,
placement bounds, overlay count/shape), the fallback semantics shared
with the other ``*_hw`` methods, the straight-alpha renderer, and the
accel-router registration of ``draw_detections``.
"""

import os
import socket
import struct
from types import SimpleNamespace
from unittest import mock

import grpc
import numpy as np
import pytest

from neoruntime_ipc_sdk import accel
from neoruntime_ipc_sdk.accel import HardwareUnavailable
from neoruntime_ipc_sdk import dsp as dsp_module
from neoruntime_ipc_sdk.draw import render_overlay_rgba
from neoruntime_ipc_sdk.dsp import DspBufferPool, DspClient, DspError
from neoruntime_ipc_sdk.dsp_format import _cpu_blend
from neoruntime_ipc_sdk.dsp_wire import (
    _HAL_PIXEL_FORMAT,
    _MIN_DIM,
    import_request_bytes,
)
from neoruntime_ipc_sdk.inference_types import BoundingBox, DetectedObject
from neoruntime_ipc_sdk.proto import camera_pb2
from tests.test_dsp import FakeSock, chunked_recv, make_pool, memfd, nv12_array


# ---------------------------------------------------------------- helpers --
def rgba_solid(width, height, rgb=(200, 40, 40), alpha=255):
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return rgba


def argb_wire_pack(rgba):
    """The hardware's ARGB32 byte order: [A, R, G, B] per pixel."""
    return np.ascontiguousarray(rgba[:, :, [3, 0, 1, 2]])


def pools_alloc(client):
    """alloc_buffers() at memfd pools; keep ``(pool, index)`` per buffer id."""
    pools = {}
    state = {"next": 1000}

    def recording(width, height, fmt_wire, count):
        fmt = {0: "nv12", 4: "rgb24", 6: "argb", 8: "gray8"}[fmt_wire]
        base = state["next"]
        state["next"] += count
        pool = make_pool(client, width, height, fmt, count, id_base=base)
        for i, bid in enumerate(pool.ids):
            pools[bid] = (pool, i)
        return (0, pool.count, 2 if fmt == "nv12" else 1,
                list(pool.strides), list(pool.plane_sizes), pool.ids,
                list(pool.plane_fds))

    client._exchange_alloc = recording
    return pools


def importing(client):
    """Patch _import_memfd: record (wire, w, h, fmt, stride) per overlay and
    hand out import ids from 2000 — the overlay side of the transport."""
    imports = []
    state = {"next": 2000}

    def fake(wire, width, height, fmt, stride, timeout_s=5.0):
        imports.append((wire, width, height, fmt, stride))
        bid = state["next"]
        state["next"] += 1
        return bid

    client._import_memfd = fake
    client._blend_imports = imports

    # keep-fd bases import through _import_source (dma-buf import, not a
    # memfd) — ids from 3000 so tests can tell base imports from overlays
    src_imports = []
    state_src = {"next": 3000}

    def fake_source(handle, width, height, fmt, timeout_s=5.0):
        src_imports.append((width, height, fmt))
        bid = state_src["next"]
        state_src["next"] += 1
        return bid

    client._import_source = fake_source
    client._blend_src_imports = src_imports
    return imports


class BlendStub:
    """Fake gRPC stub; side_effect(request, pools) -> DspJobResponse.

    ``pools`` maps buffer id -> (pool, index) so a test can inspect what
    blend_hw wrote into the overlay pools, or play daemon by rewriting
    the base buffer in place before the read-back.
    """

    def __init__(self, side_effect=None, error=None):
        self.requests = []
        self.pools = {}
        self.side_effect = side_effect or (
            lambda req, pools: camera_pb2.DspJobResponse(success=True, elapsed_ms=3)
        )
        self.error = error

    def SubmitDspJob(self, request, timeout=None):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.side_effect(request, self.pools)


def blending_client(side_effect=None, error=None):
    """Client whose daemon answers SubmitDspJob via a BlendStub."""
    client = DspClient()
    pools = pools_alloc(client)
    importing(client)
    client._send_release = mock.Mock()
    stub = BlendStub(side_effect=side_effect, error=error)
    stub.pools = pools
    client._stub = stub
    return client, pools, stub


def unimplemented_error():
    err = grpc.RpcError("no such method")
    err.code = lambda: grpc.StatusCode.UNIMPLEMENTED
    return err


def daemon_blend(request, pools):
    """Play daemon: bump base luma inside overlay 0's placement rect."""
    base_pool, base_i = pools[request.src_buffer_id]
    base = base_pool.read(base_i)
    rect = request.rects[0]
    y0, x0 = int(rect.y), int(rect.x)
    base[y0 : y0 + int(rect.height), x0 : x0 + int(rect.width)] = 250
    base_pool.write(base_i, base)
    return camera_pb2.DspJobResponse(success=True, elapsed_ms=3)


def one_object():
    """car at (8, 6)-(40, 28) — a DetectedObject like a detector returns."""
    return [DetectedObject(label="car", score=0.9, bbox=BoundingBox(8, 6, 32, 22))]


# --------------------------------------------------------- request shape --
class TestRequest:
    def test_blend_request_fields(self):
        client, pools, stub = blending_client(side_effect=daemon_blend)
        nv12 = nv12_array(64, 32, seed=1)
        overlay = rgba_solid(32, 16)

        out = client.blend_hw(nv12, [(overlay, 8, 6)])

        req = stub.requests[0]
        assert req.op == camera_pb2.DSP_OP_BLEND
        assert req.src_buffer_id == 1000  # the nv12 base pool
        assert list(req.dst_buffer_ids) == [2000]  # the one overlay import
        assert len(req.rects) == 1
        r = req.rects[0]
        # placement rect: (x, y) on the base; w/h == overlay dims, pasted 1:1
        assert (r.x, r.y, r.width, r.height, r.dst_width, r.dst_height) == (
            8, 6, 32, 16, 32, 16,
        )
        # blend ran in place on the pool copy — the daemon's write is
        # what comes back, and the caller's array is untouched
        assert out.shape == nv12.shape
        assert out[6, 8] == 250 and nv12[6, 8] != 250
        assert client.last_used_hw is True

    def test_overlay_wire_bytes_are_argb(self):
        # the hardware ARGB32 layout is [A, R, G, B] per pixel in memory —
        # the pack the SDK ships inside the memfd import
        client, _, _ = blending_client()

        client.blend_hw(nv12_array(64, 32),
                        [(rgba_solid(20, 16, rgb=(10, 20, 30), alpha=77), 0, 0)])

        wire, width, height, fmt, stride = client._blend_imports[0]
        arr = np.frombuffer(wire, np.uint8).reshape(height, width, 4)
        assert (width, height, fmt, stride) == (20, 16, "argb", 80)
        assert arr[0, 0].tolist() == [77, 10, 20, 30]  # A R G B
        overlay = rgba_solid(20, 16, rgb=(10, 20, 30), alpha=77)
        assert np.array_equal(arr, argb_wire_pack(overlay))

    def test_multiple_overlays_get_ordered_dst_ids_and_rects(self):
        client, _, stub = blending_client()
        ov_a, ov_b = rgba_solid(16, 16), rgba_solid(20, 24)

        client.blend_hw(nv12_array(64, 32), [(ov_a, 0, 0), (ov_b, 30, 2)])

        req = stub.requests[0]
        assert list(req.dst_buffer_ids) == [2000, 2001]
        assert [(r.x, r.y, r.width, r.height) for r in req.rects] == [
            (0, 0, 16, 16), (30, 2, 20, 24),
        ]

    def test_sub16_overlay_padded_transparent(self):
        # daemon floor is 16x16 — a 6x4 overlay must ride as 16x16 with
        # the original at the top-left and alpha=0 elsewhere
        client, _, stub = blending_client()

        client.blend_hw(nv12_array(64, 32), [(rgba_solid(6, 4, alpha=255), 10, 10)])

        r = stub.requests[0].rects[0]
        assert (r.width, r.height, r.dst_width, r.dst_height) == (16, 16, 16, 16)
        wire, width, height, fmt, stride = client._blend_imports[0]
        assert (width, height, len(wire)) == (16, 16, 16 * 16 * 4)
        arr = np.frombuffer(wire, np.uint8).reshape(16, 16, 4)
        assert np.array_equal(arr[:4, :6], argb_wire_pack(rgba_solid(6, 4)))
        assert arr[4:, :, 0].max() == 0  # padding transparent (A is chan 0)
        assert arr[:, 6:, 0].max() == 0

    def test_own_buffers_released_after_job(self):
        client, _, _ = blending_client()
        client.blend_hw(nv12_array(64, 32), [(rgba_solid(16, 16), 0, 0)])
        # base pool (1000) + overlay import (2000) both returned
        assert [c[0][0] for c in client._send_release.call_args_list] == [1000, 2000]

    def test_base_pool_and_overlay_import_geometries(self):
        client, pools, _ = blending_client()
        client.blend_hw(nv12_array(64, 32), [(rgba_solid(32, 16), 8, 6)])

        base_pool, _ = pools[1000]  # the base is still a HAL pool alloc
        assert (base_pool.width, base_pool.height, base_pool.fmt) == (64, 32, "nv12")
        wire, width, height, fmt, stride = client._blend_imports[0]
        assert (width, height, fmt, stride) == (32, 16, "argb", 128)
        assert len(wire) == stride * height


# ----------------------------------------------------- memfd transport --
class _CapturingSock(FakeSock):
    """FakeSock that snapshots the SCM_RIGHTS fd while we still own it."""

    def sendmsg(self, buffers, ancdata, flags=0, address=None):
        (fd,) = struct.unpack("i", ancdata[0][2])
        self.fd_size = os.fstat(fd).st_size
        self.fd_content = os.pread(fd, self.fd_size, 0)
        super().sendmsg(buffers, ancdata, flags, address)


class TestMemfdTransport:
    def test_import_memfd_sends_geometry_bytes_and_closes(self, monkeypatch):
        # the real _import_memfd: memfd carrying the packed ARGB bytes,
        # ftruncate'd to size, sent over DSP_IMPORT, closed after
        client = DspClient()
        sock = _CapturingSock()
        client._sock = sock
        resp = struct.pack("<IIi4xq", 11, 24, 0, 777)
        monkeypatch.setattr(
            dsp_module, "_recvmsg_with_fds", chunked_recv([resp[:8], resp[8:]])
        )
        wire = bytes(range(256)) * 4  # 1024 B = 16x16 ARGB

        import_id = client._import_memfd(wire, 16, 16, "argb", 64, 1.0)

        assert import_id == 777
        payload, anc = sock.sent[0]
        assert payload == import_request_bytes(
            16, 16, _HAL_PIXEL_FORMAT["argb"], 1, [64, 0, 0], [1024, 0, 0]
        )
        level, ctype, cdata = anc[0]
        assert (level, ctype) == (socket.SOL_SOCKET, socket.SCM_RIGHTS)
        (fd,) = struct.unpack("i", cdata)
        assert sock.fd_size == 1024  # ftruncate'd, not sparse
        assert sock.fd_content == wire  # the exact bytes, written fully
        with pytest.raises(OSError):
            os.fstat(fd)  # our copy is closed — the daemon holds its dup

    def test_import_memfd_propagates_rejection(self, monkeypatch):
        client = DspClient()
        client._sock = FakeSock()
        resp = struct.pack("<IIi4xq", 11, 24, -1, -1)
        monkeypatch.setattr(
            dsp_module, "_recvmsg_with_fds", chunked_recv([resp[:8], resp[8:]])
        )
        with pytest.raises(DspError, match="import rejected"):
            client._import_memfd(b"\x00" * 1024, 16, 16, "argb", 64)


# ------------------------------------------------------------- validation --
class TestValidation:
    def test_explicit_non_nv12_base_rejected(self):
        client = DspClient()
        with pytest.raises(DspError, match="BLEND base must be nv12"):
            client.blend_hw(nv12_array(64, 32), [(rgba_solid(16, 16), 0, 0)], fmt="rgb24")

    def test_rgb_array_base_rejected(self):
        # an (h, w, 3) array is not nv12 geometry — no silent reinterpretation
        client = DspClient()
        with pytest.raises(DspError):
            client.blend_hw(np.zeros((32, 64, 3), np.uint8), [(rgba_solid(16, 16), 0, 0)])

    def test_handle_base_rides_zero_copy_chain(self):
        # P2: a keep-fd base no longer raises — it imports zero-copy, the
        # DSP copies it into the base pool with a 1:1 RESIZE, and the
        # blend composites in place on that copy. The base pixels never
        # cross the client: the pool receives no write before the jobs.
        from neoruntime_ipc_sdk.frame import FrameHandle

        client, pools, stub = blending_client(
            side_effect=lambda req, p: (
                daemon_blend(req, p)
                if req.op == camera_pb2.DSP_OP_BLEND
                else camera_pb2.DspJobResponse(success=True, elapsed_ms=1)
            )
        )
        handle = FrameHandle(
            [memfd(64 * 32 * 3 // 2)], (64,), (64 * 32 * 3 // 2,), 7,
            width=64, height=32, format="NV12",
        )

        out = client.blend_hw(handle, [(rgba_solid(16, 16), 0, 0)],
                              zero_copy=True)

        assert client._blend_src_imports == [(64, 32, "nv12")]
        copy_req, blend_req = stub.requests
        # leg 1: 1:1 RESIZE import -> the fresh base pool (the DSP-side copy)
        assert copy_req.op == camera_pb2.DSP_OP_RESIZE
        assert copy_req.src_buffer_id == 3000
        assert list(copy_req.dst_buffer_ids) == [1000]
        assert list(copy_req.rects) == []
        # leg 2: in-place BLEND on the pool copy
        assert blend_req.op == camera_pb2.DSP_OP_BLEND
        assert blend_req.src_buffer_id == 1000
        assert list(blend_req.dst_buffer_ids) == [2000]
        # the annotated result comes back from the pool, not the frame
        assert out.shape == (48, 64)
        assert out[0, 0] == 250  # daemon_blend bumped the placement rect
        assert client.last_used_hw is True

    def test_handle_base_pool_never_written_client_side(self):
        # zero-copy means it: the base pool's buffer holds no client bytes
        # before the jobs run (the RESIZE leg fills it on the device)
        from neoruntime_ipc_sdk.frame import FrameHandle

        client, pools, stub = blending_client()
        handle = FrameHandle(
            [memfd(64 * 32 * 3 // 2)], (64,), (64 * 32 * 3 // 2,), 8,
            width=64, height=32, format="NV12",
        )

        with mock.patch.object(
            DspBufferPool, "write", side_effect=AssertionError("client write")
        ) as bw:
            client.blend_hw(handle, [(rgba_solid(16, 16), 0, 0)],
                            zero_copy=True)
            # only the overlay memfd import exists; the base had no write
            bw.assert_not_called()

    def test_handle_base_refused_by_default(self):
        # the zero-copy chain is firmware-fatal on current hailo15 (the
        # blend command never returns; DSP wedged device-wide 2/2 on
        # 93.72) — the default contract must refuse before any wire work
        from neoruntime_ipc_sdk.frame import FrameHandle

        client = DspClient()
        handle = FrameHandle(
            [memfd(64 * 32 * 3 // 2)], (64,), (64 * 32 * 3 // 2,), 9,
            width=64, height=32, format="NV12",
        )

        with mock.patch.object(client, "alloc_buffers") as alloc:
            with pytest.raises(DspError, match="refuses keep-fd"):
                client.blend_hw(handle, [(rgba_solid(16, 16), 0, 0)])
            alloc.assert_not_called()  # refused before any pool/import work

    def test_empty_overlays_rejected(self):
        client = DspClient()
        with pytest.raises(DspError, match="at least one overlay"):
            client.blend_hw(nv12_array(64, 32), [])

    def test_too_many_overlays_rejected(self):
        client = DspClient()
        overlays = [(rgba_solid(16, 16), 0, 0)] * 65
        with pytest.raises(DspError, match="too many overlays"):
            client.blend_hw(nv12_array(1024, 1024), overlays)

    def test_out_of_bounds_placement_rejected(self):
        client = DspClient()
        with pytest.raises(DspError, match="exceeds the 64x32 base"):
            client.blend_hw(nv12_array(64, 32), [(rgba_solid(16, 16), 60, 0)])
        with pytest.raises(DspError, match="exceeds the 64x32 base"):
            client.blend_hw(nv12_array(64, 32), [(rgba_solid(16, 16), 0, -1)])

    def test_non_rgba_overlay_rejected(self):
        client = DspClient()
        with pytest.raises(DspError, match=r"\(h, w, 4\) uint8 rgba"):
            client.blend_hw(nv12_array(64, 32), [(np.zeros((16, 16, 3), np.uint8), 0, 0)])
        with pytest.raises(DspError, match=r"\(h, w, 4\) uint8 rgba"):
            client.blend_hw(
                nv12_array(64, 32), [(np.zeros((16, 16, 4), np.float32), 0, 0)]
            )

    def test_odd_nv12_base_rejected_by_geometry(self):
        client = DspClient()
        # 36 rows -> h=24 even, but width 65 is odd
        with pytest.raises(DspError, match="even width/height"):
            client.blend_hw(np.zeros((36, 65), np.uint8), [(rgba_solid(16, 16), 0, 0)])


# ---------------------------------------------------------------- fallback --
class TestFallback:
    def test_unimplemented_rpc_warns_and_blends_on_cpu(self):
        client, _, _ = blending_client(error=unimplemented_error())
        nv12 = nv12_array(64, 32, seed=5)
        overlay = rgba_solid(32, 16, alpha=128)

        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.blend_hw(nv12, [(overlay, 8, 6)])

        assert np.array_equal(out, _cpu_blend(nv12, "nv12", [(overlay, 8, 6)]))
        assert client.last_used_hw is False

    def test_service_down_response_falls_back(self):
        client, _, _ = blending_client(side_effect=lambda req, pools: camera_pb2.DspJobResponse(
            success=False, error_code=-5, message="dsp service not running"))
        nv12 = nv12_array(64, 32, seed=6)
        overlay = rgba_solid(16, 16, alpha=200)

        with pytest.warns(UserWarning, match="CPU fallback"):
            out = client.blend_hw(nv12, [(overlay, 0, 0)])

        assert np.array_equal(out, _cpu_blend(nv12, "nv12", [(overlay, 0, 0)]))
        assert client.last_used_hw is False

    def test_cpu_fallback_false_raises_on_unavailable(self):
        client, _, _ = blending_client(error=unimplemented_error())
        with pytest.raises(DspError, match="not in daemon"):
            client.blend_hw(nv12_array(64, 32), [(rgba_solid(16, 16), 0, 0)],
                            cpu_fallback=False)

    def test_rejected_job_warns_and_blends_on_cpu(self):
        # daemon took the RPC, hardware refused (e.g. firmware without
        # blend) — same honest tail as convert_hw
        client, _, _ = blending_client(side_effect=lambda req, pools: camera_pb2.DspJobResponse(
            success=False, error_code=-1, message="blend refused by firmware"))
        nv12 = nv12_array(64, 32, seed=7)
        overlay = rgba_solid(16, 16, alpha=200)

        with pytest.warns(UserWarning, match="DSP rejected the blend"):
            out = client.blend_hw(nv12, [(overlay, 4, 4)])

        assert np.array_equal(out, _cpu_blend(nv12, "nv12", [(overlay, 4, 4)]))
        assert client.last_used_hw is False

    def test_rejected_job_with_cpu_fallback_false_raises(self):
        client, _, _ = blending_client(side_effect=lambda req, pools: camera_pb2.DspJobResponse(
            success=False, error_code=-1, message="blend refused by firmware"))

        with pytest.raises(DspError, match="blend refused by firmware"):
            client.blend_hw(nv12_array(64, 32), [(rgba_solid(16, 16), 0, 0)],
                            cpu_fallback=False)


# ---------------------------------------------------------------- renderer --
class TestRenderOverlay:
    def test_canvas_is_union_bbox_plus_stroke_and_label(self):
        rgba, x0, y0 = render_overlay_rgba(
            640, 480, [(100, 100, 300, 220)], labels=["car"], scores=[0.87],
            colors=[(0, 255, 0)],
        )
        # label strip (20) + stroke margin (3 per side) around the box
        assert (x0, y0) == (100 - 3, 100 - 3 - 20)
        assert rgba.shape == ((220 - 100) + 6 + 20, (300 - 100) + 6, 4)
        assert rgba[..., 3].max() == 255  # something was drawn

    def test_multiple_boxes_union(self):
        rgba, x0, y0 = render_overlay_rgba(
            640, 480, [(10, 40, 60, 90), (300, 200, 500, 400)],
            labels=[None, None],
        )
        assert (x0, y0) == (10 - 3, 40 - 3)
        assert rgba.shape == ((400 + 3) - y0, (500 + 3) - x0, 4)

    def test_straight_alpha_colorization(self):
        # coverage mask drives alpha; color is full-strength wherever hit
        rgba, x0, y0 = render_overlay_rgba(
            640, 480, [(100, 100, 300, 220)], labels=[None],
            colors=[(10, 20, 30)],
        )
        hit = rgba[..., 3] > 0
        assert hit.any()
        assert (rgba[..., :3][hit] == (10, 20, 30)).all()
        assert (rgba[..., 3][~hit] == 0).all()

    def test_canvas_floored_at_daemon_minimum(self):
        rgba, x0, y0 = render_overlay_rgba(
            640, 480, [(300, 200, 306, 208)], labels=[None], thickness=1,
        )
        assert rgba.shape[:2] == (_MIN_DIM, _MIN_DIM)  # 6x8 box -> 16x16

    def test_offset_clamps_to_frame_edges(self):
        rgba, x0, y0 = render_overlay_rgba(
            640, 480, [(0, 0, 50, 60)], labels=["x"],
        )
        assert (x0, y0) == (0, 0)  # label strip clamped, not negative

    def test_out_of_frame_boxes_raise(self):
        with pytest.raises(ValueError, match="outside the frame"):
            render_overlay_rgba(640, 480, [(700, 500, 900, 700)])

    def test_no_boxes_raise(self):
        with pytest.raises(ValueError, match="at least one shape"):
            render_overlay_rgba(640, 480, [])

    def test_polygon_canvas_from_point_extents(self):
        pts = [(50, 60), (120, 60), (120, 140)]
        rgba, x0, y0 = render_overlay_rgba(
            640, 480, polygons=[(pts, (255, 0, 0))], thickness=2,
        )
        # same stroke margin (t + 1 per side) as boxes, no label strip
        assert (x0, y0) == (50 - 3, 60 - 3)
        assert rgba.shape == ((140 + 3) - y0, (120 + 3) - x0, 4)
        assert rgba[..., 3].max() == 255

    def test_polygon_closed_track_open(self):
        # the closing segment (last -> first point) exists for polygons,
        # never for tracks
        pts = [(10, 10), (90, 10), (90, 90)]
        mid = (50, 50)  # midpoint of the closing diagonal
        rgba_c, xc, yc = render_overlay_rgba(
            200, 200, polygons=[(pts, None)], thickness=1,
        )
        assert rgba_c[mid[1] - yc, mid[0] - xc, 3] > 0
        rgba_o, xo, yo = render_overlay_rgba(
            200, 200, tracks=[(pts, None)], thickness=1,
        )
        assert rgba_o[mid[1] - yo, mid[0] - xo, 3] == 0

    def test_polygon_color_and_green_default(self):
        pts = [(10, 10), (60, 10), (60, 60)]
        rgba, *_ = render_overlay_rgba(
            100, 100, polygons=[(pts, (1, 2, 3))], thickness=2,
        )
        hit = rgba[..., 3] > 0
        assert hit.any()
        assert (rgba[..., :3][hit] == (1, 2, 3)).all()

        rgba2, *_ = render_overlay_rgba(100, 100, polygons=[(pts, None)])
        hit2 = rgba2[..., 3] > 0
        assert (rgba2[..., :3][hit2] == (0, 255, 0)).all()

    def test_boxes_and_tracks_union_canvas(self):
        rgba, x0, y0 = render_overlay_rgba(
            640, 480, [(300, 200, 500, 400)], labels=[None],
            tracks=[([(10, 40), (80, 120)], None)],
        )
        assert (x0, y0) == (10 - 3, 40 - 3)
        assert rgba.shape == ((400 + 3) - y0, (500 + 3) - x0, 4)

    def test_bad_points_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            render_overlay_rgba(
                100, 100, polygons=[([(10, 10, 1), (60, 10, 1)], None)],
            )

    def test_renderer_composite_matches_software_raster(self):
        # the whole point: renderer + straight-alpha composite == the
        # draw_boxes cv2 raster, up to text-AA rounding
        from neoruntime_ipc_sdk.draw import draw_boxes

        boxes = [(40, 30, 200, 160)]
        base = np.full((240, 320, 3), 100, np.uint8)
        sw = draw_boxes(base.copy(), boxes, labels=["obj"], scores=[0.9],
                        color=(0, 255, 0), thickness=2)
        rgba, x0, y0 = render_overlay_rgba(
            320, 240, boxes, labels=["obj"], scores=[0.9],
            colors=[(0, 255, 0)], thickness=2,
        )
        composited = _cpu_blend(base, "rgb24", [(rgba, x0, y0)])

        # outside the canvas the composite is the base, untouched
        mask = np.zeros(base.shape[:2], bool)
        mask[y0 : y0 + rgba.shape[0], x0 : x0 + rgba.shape[1]] = True
        assert np.array_equal(composited[~mask], base[~mask])

        region = sw[y0 : y0 + rgba.shape[0], x0 : x0 + rgba.shape[1]]
        got = composited[y0 : y0 + rgba.shape[0], x0 : x0 + rgba.shape[1]]
        delta = np.abs(got.astype(int) - region.astype(int))
        assert delta.max() <= 3
        assert (delta > 0).mean() < 0.5  # overwhelmingly identical

    def test_polygon_composite_matches_software_raster(self):
        # same contract as the boxes test above, on the polygons leg
        from neoruntime_ipc_sdk.draw import draw_polygons

        pts = [(40, 30), (200, 30), (200, 150), (90, 170)]
        base = np.full((240, 320, 3), 100, np.uint8)
        sw = draw_polygons(base.copy(), [(pts, (0, 255, 0))], thickness=2)
        rgba, x0, y0 = render_overlay_rgba(
            320, 240, polygons=[(pts, (0, 255, 0))], thickness=2,
        )
        composited = _cpu_blend(base, "rgb24", [(rgba, x0, y0)])

        mask = np.zeros(base.shape[:2], bool)
        mask[y0 : y0 + rgba.shape[0], x0 : x0 + rgba.shape[1]] = True
        assert np.array_equal(composited[~mask], base[~mask])

        region = sw[y0 : y0 + rgba.shape[0], x0 : x0 + rgba.shape[1]]
        got = composited[y0 : y0 + rgba.shape[0], x0 : x0 + rgba.shape[1]]
        delta = np.abs(got.astype(int) - region.astype(int))
        assert delta.max() <= 3
        assert (delta > 0).mean() < 0.5


# ------------------------------------------------------------ router legs --
class TestRouterLegs:
    def test_draw_detections_registered_as_hardware(self):
        ops = accel.get_default_router().health()["ops"]
        assert ops["draw_detections"]["backend"] == "hardware"

    def test_hw_leg_blend_call_and_hardware_count(self, monkeypatch):
        calls = []

        class FakeDsp:
            def blend_hw(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "annotated nv12"

        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: FakeDsp())
        router = accel.AccelRouter()
        router.register(
            "draw_detections", software=accel._draw_detections_sw,
            hardware=accel._draw_detections_hw,
        )
        nv12 = nv12_array(64, 32, seed=9)

        out = router.run("draw_detections", nv12, one_object())

        assert out == "annotated nv12"
        args, kwargs = calls[0]
        assert np.array_equal(args[0], nv12)
        assert len(args[1]) == 1  # one overlay
        rgba, x0, y0 = args[1][0]
        assert rgba.shape[2] == 4 and x0 >= 0 and y0 >= 0
        assert kwargs == {"fmt": "nv12", "cpu_fallback": False}
        op = router.health()["ops"]["draw_detections"]
        assert (op["hardware_calls"], op["software_calls"], op["fallbacks"]) == (1, 0, 0)

    def test_hw_leg_refuses_rgb_and_degrades(self, monkeypatch):
        router = accel.AccelRouter()
        router.register(
            "draw_detections", software=accel._draw_detections_sw,
            hardware=accel._draw_detections_hw,
        )
        rgb = np.full((32, 64, 3), 90, np.uint8)

        out = router.run("draw_detections", rgb, one_object())

        assert out.shape == rgb.shape
        assert not np.array_equal(out, rgb)  # sw raster drew the box
        op = router.health()["ops"]["draw_detections"]
        assert (op["fallbacks"], op["software_calls"]) == (1, 1)
        assert len(router.health()["recent_degradations"]) == 1

    def test_hw_leg_failure_degrades_to_software_nv12(self, monkeypatch):
        # nv12 must stay nv12 after a degradation — the sw leg mirrors
        # the hardware path (renderer + cpu composite) on that format
        class FakeDsp:
            def blend_hw(self, *args, **kwargs):
                raise DspError("SubmitDspJob not in daemon")

        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: FakeDsp())
        router = accel.AccelRouter()
        router.register(
            "draw_detections", software=accel._draw_detections_sw,
            hardware=accel._draw_detections_hw,
        )
        nv12 = nv12_array(64, 32, seed=10)

        out = router.run("draw_detections", nv12, one_object())

        assert out.shape == nv12.shape
        assert not np.array_equal(out, nv12)  # annotation drawn on CPU
        op = router.health()["ops"]["draw_detections"]
        assert (op["fallbacks"], op["software_calls"], op["hardware_calls"]) == (1, 1, 0)

    def test_empty_objects_copy_without_dsp(self, monkeypatch):
        def boom():
            raise AssertionError("DSP must not be reached for an empty result")

        monkeypatch.setattr(accel, "_lazy_dsp_client", boom)
        nv12 = nv12_array(64, 32, seed=11)

        out = accel._draw_detections_hw(nv12, [])

        assert np.array_equal(out, nv12) and out is not nv12

    def test_sw_leg_rgb_matches_draw_detections(self):
        rgb = np.zeros((32, 64, 3), np.uint8)
        expected = accel._draw_detections_sw(rgb.copy(), one_object())
        from neoruntime_ipc_sdk.draw import draw_detections

        assert np.array_equal(expected, draw_detections(rgb.copy(), one_object()))


# ------------------------------------------------------ dma-buf fast paths --
class TestFastPaths:
    """Keep-fd sources (Frame/FrameHandle-like) must reach the *_hw
    client methods verbatim — no ascontiguousarray copy — while plain
    arrays keep arriving contiguous."""

    def _frame(self, width=64, height=32):
        return SimpleNamespace(handle=object(), width=width, height=height)

    def test_frames_pass_through_uncopied(self, monkeypatch):
        seen = {}

        class FakeDsp:
            def resize_hw(self, src, w, h, **kw):
                seen["resize"] = (src, w, h)
                return src

            def convert_hw(self, src, dst_fmt, **kw):
                seen[f"convert->{dst_fmt}"] = src
                return src

            def encode_jpeg_hw(self, src, **kw):
                seen["encode"] = src
                return b"J"

            def blend_hw(self, base, overlays, **kw):
                seen["blend"] = base
                return base

        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: FakeDsp())
        frame = self._frame()

        accel._resize_nv12_hw(frame, (64, 32), (32, 16))
        accel._rgb_to_nv12_hw(frame)
        accel._nv12_to_rgb_hw(frame)
        accel._encode_jpeg_hw(frame)
        # frames no longer take the blend leg at all (firmware-fatal
        # zero-copy chain — DspClient.blend_hw refuses them)
        with pytest.raises(HardwareUnavailable, match="keep-fd"):
            accel._draw_detections_hw(frame, one_object())

        # every surviving leg handed the client the frame object itself —
        # the zero-copy import happens inside DspClient, not in the router
        assert seen["resize"][0] is frame
        assert (seen["resize"][1], seen["resize"][2]) == (32, 16)
        assert seen["convert->nv12"] is frame
        assert seen["convert->rgb24"] is frame
        assert seen["encode"] is frame

    def test_frame_geometry_mismatch_raises(self):
        with pytest.raises(ValueError, match="frame is 64x32"):
            accel._nv12_to_rgb_hw(self._frame(), width=32, height=32)

    def test_noncontiguous_arrays_arrive_contiguous(self, monkeypatch):
        seen = {}

        class FakeDsp:
            def resize_hw(self, src, w, h, **kw):
                seen["resize"] = src
                return src

            def convert_hw(self, src, dst_fmt, **kw):
                seen["convert"] = src
                return src

        monkeypatch.setattr(accel, "_lazy_dsp_client", lambda: FakeDsp())
        # sliced arrays are not C-contiguous on the way in
        strided = np.zeros((48, 128), np.uint8)[::2]

        accel._resize_nv12_hw(strided, (128, 32), (64, 32))
        accel._rgb_to_nv12_hw(np.zeros((32, 64, 6), np.uint8)[:, :, ::2])

        assert seen["resize"].flags["C_CONTIGUOUS"]
        assert seen["convert"].flags["C_CONTIGUOUS"]
