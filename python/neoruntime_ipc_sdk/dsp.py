"""DSP offload client (SDK-2).

Thin wrapper over the camera-daemon DSP service (platform PLAT-1..6):
buffers are allocated on the FD-publisher UDS (``/run/aipc/camera.sock``,
FD_PUB_MSG_DSP_ALLOC/RESP/BUF_RELEASE, dma-buf fds via SCM_RIGHTS) and
jobs are submitted through the ``SubmitDspJob`` gRPC on camera-control.

Hardware-first with a numpy CPU fallback: on a daemon without the DSP
RPC (grpc UNIMPLEMENTED) or with the service not running (error -5) the
``*_hw`` helpers compute the result on CPU instead of raising, and
``client.last_used_hw`` records which path served the last call.

Caveat (P0 platform contract): a job source must be a daemon-allocated
DSP buffer, so the input array is copied into one — camera dma-buf fds
cannot yet be a job source directly (zero-copy input needs a platform
extension; see docs/proposals/hardware-first-roadmap.md, SDK-2 notes).

Rate limiting: the daemon enforces a per-client MPix/s budget (a new
client gets a 1-second burst; it then replenishes continuously). Each
job is charged ``src + sum(dst)`` megapixels, so hot-looping 4K sources
(≈8.3 MPix/frame) exhausts the budget within a few jobs and further
submissions raise ``DspError`` (``code == -3``, message like "quota:
MPix/s budget exhausted"). That error is deliberately NOT silently
fallen back to CPU — a switch to CPU is a large latency cliff the app
should see. Pace submissions, or crop to a smaller source first.

Usage::

    client = DspClient()
    small = client.resize_hw(frame.image, 640, 640, fmt="nv12")
    tiles = client.multi_crop_hw(frame.image, rects, fmt="nv12")
"""

import logging
import mmap
import os
import socket
import struct
from typing import List, Optional, Sequence, Tuple

import grpc
import numpy as np

try:  # cv2 accelerates the CPU fallback only; never required
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None

from .config import Config
from .media import (
    _DMA_BUF_SYNC_END,
    _DMA_BUF_SYNC_READ,
    _DMA_BUF_SYNC_START,
    _DMA_BUF_SYNC_WRITE,
    _dma_buf_sync,
    _recvmsg_with_fds,
)
from .proto import camera_pb2, camera_pb2_grpc

logger = logging.getLogger(__name__)

# ---- UDS wire constants (platform camera-daemon include/fd_protocol.h) ----
_FD_PUB_MSG_DSP_ALLOC = 7
_FD_PUB_MSG_DSP_ALLOC_RESP = 8
_FD_PUB_MSG_DSP_BUF_RELEASE = 9

_ALLOC_REQ_FMT = "<IIIIII"                         # hdr + w, h, fmt, count
_ALLOC_REQ_SIZE = struct.calcsize(_ALLOC_REQ_FMT)  # 24
_ALLOC_RESP_FMT = "<II i I I 3I 3I 4x 64Q"         # C layout incl. u64 align
_ALLOC_RESP_SIZE = struct.calcsize(_ALLOC_RESP_FMT)
_RELEASE_FMT = "<IIQ"
_DSP_MAX_FDS = 64                                  # FD_PUB_DSP_MAX_FDS

# HalPixelFormat wire values (hal_v2 hal_buffer.h) — deliberately separate
# from the SDK PixelFormat enum, whose numbering does not match the wire.
_HAL_PIXEL_FORMAT = {"nv12": 0, "rgb24": 4, "gray8": 8}
_DSP_FORMATS = ("nv12", "rgb24", "gray8")

# ---- daemon caps (dsp_service.cpp), mirrored for fail-fast validation ----
_MIN_DIM = 16
_MAX_DIM = 8192
_MAX_BATCH = 64

# ---- DspService error codes ----
DSP_SERVICE_UNAVAILABLE = -5

_ERROR_TEXT = {
    -1: "invalid request",
    -2: "no such buffer id",
    -3: "quota exceeded",
    -4: "job timeout",
    -5: "dsp service unavailable",
    -6: "out of memory",
    -7: "client limit exceeded",
}

_OP_RESIZE = 0
_OP_CROP_AND_RESIZE = 1
_OP_MULTI_CROP = 2

_INTERP_WIRE = {
    "nearest": camera_pb2.DSP_INTERP_NEAREST,
    "bilinear": camera_pb2.DSP_INTERP_BILINEAR,
    "area": camera_pb2.DSP_INTERP_AREA,
    "bicubic": camera_pb2.DSP_INTERP_BICUBIC,
}
_CV_INTERP = {
    "nearest": "INTER_NEAREST",
    "bilinear": "INTER_LINEAR",
    "area": "INTER_AREA",
    "bicubic": "INTER_CUBIC",
}
_SCALING_WIRE = {
    "stretch": camera_pb2.DSP_SCALING_STRETCH,
    "letterbox": camera_pb2.DSP_SCALING_LETTERBOX_MIDDLE,
    "letterbox_middle": camera_pb2.DSP_SCALING_LETTERBOX_MIDDLE,
    "letterbox_up_left": camera_pb2.DSP_SCALING_LETTERBOX_UP_LEFT,
    "scale_crop": camera_pb2.DSP_SCALING_SCALE_AND_CROP,
}
_PRIORITY_WIRE = {
    "background": camera_pb2.DSP_PRIORITY_BACKGROUND,
    "normal": camera_pb2.DSP_PRIORITY_NORMAL,
}


class DspError(Exception):
    """DSP job or buffer error. ``code`` mirrors the daemon error codes."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class _DspUnavailable(DspError):
    """Internal: daemon lacks the DSP surface — take the CPU fallback."""


def alloc_request_bytes(width: int, height: int, fmt_wire: int,
                        count: int) -> bytes:
    """Encode FD_PUB_MSG_DSP_ALLOC (24 bytes)."""
    return struct.pack(_ALLOC_REQ_FMT, _FD_PUB_MSG_DSP_ALLOC,
                       _ALLOC_REQ_SIZE, width, height, fmt_wire, count)


def parse_alloc_resp(payload: bytes) -> Tuple[int, int, int, List[int],
                                              List[int], List[int]]:
    """Decode FD_PUB_MSG_DSP_ALLOC_RESP (560 bytes; fds arrive separately).

    Returns ``(code, count, num_planes, strides[3], sizes[3], ids[count])``.
    """
    if len(payload) < _ALLOC_RESP_SIZE:
        raise DspError(f"short DSP alloc response: {len(payload)} bytes")
    mtype, _size, code, count, num_planes, s0, s1, s2, z0, z1, z2 = \
        struct.unpack_from("<II i I I 3I 3I", payload, 0)
    if mtype != _FD_PUB_MSG_DSP_ALLOC_RESP:
        raise DspError(f"unexpected DSP alloc response type {mtype}")
    ids = list(struct.unpack_from("<64Q", payload, 48))[:count]
    return code, count, num_planes, [s0, s1, s2], [z0, z1, z2], ids


def _plane_rows(fmt: str, width: int, height: int) -> List[Tuple[int, int]]:
    """(row_bytes, rows) per plane for a tightly-packed geometry."""
    if fmt == "nv12":
        return [(width, height), (width, height // 2)]
    if fmt == "rgb24":
        return [(width * 3, height)]
    return [(width, height)]


def _plane_count(fmt: str) -> int:
    return 2 if fmt == "nv12" else 1


def _validate_geometry(width: int, height: int, fmt: str, what: str) -> None:
    if fmt not in _DSP_FORMATS:
        raise DspError(f"unsupported format {fmt!r} (nv12/rgb24/gray8)")
    if fmt == "nv12" and (width % 2 or height % 2):
        raise DspError(f"{what}: nv12 needs even width/height "
                       f"(got {width}x{height})")
    if not (_MIN_DIM <= width <= _MAX_DIM and _MIN_DIM <= height <= _MAX_DIM):
        raise DspError(f"{what}: dims {width}x{height} outside daemon range "
                       f"[{_MIN_DIM}, {_MAX_DIM}]")


class DspBufferPool:
    """Daemon-allocated dma-buf buffers sharing one geometry.

    One wire allocation returns ``count`` buffers; every buffer exposes
    ``_plane_count(fmt)`` dma-buf fds (NV12: Y + interleaved-UV). Plane
    rows may be padded (``strides`` > row bytes); write/read copy
    row-by-row so padding is preserved. ``release()`` returns the buffers
    to the daemon and closes every fd; closing the client's UDS releases
    them too (daemon-side cleanup on disconnect).
    """

    def __init__(self, client: "DspClient", width: int, height: int,
                 fmt: str, ids: Sequence[int], fds: Sequence[int],
                 strides: Sequence[int], sizes: Sequence[int]):
        self._client = client
        self.width = width
        self.height = height
        self.fmt = fmt
        self.ids = list(ids)
        self.strides = tuple(strides[:3])
        self.plane_sizes = tuple(sizes[:3])
        self.plane_fds = list(fds)
        self._released = False
        if len(self.plane_fds) != len(self.ids) * _plane_count(fmt):
            raise DspError(f"alloc returned {len(self.plane_fds)} fds for "
                           f"{len(self.ids)} {fmt} buffers (need "
                           f"{len(self.ids) * _plane_count(fmt)})")

    @property
    def count(self) -> int:
        return len(self.ids)

    def buffer_id(self, index: int) -> int:
        return self.ids[index]

    # -- CPU -> device -------------------------------------------------------
    def write(self, index: int, arr: np.ndarray) -> None:
        """Copy a numpy array into buffer ``index`` (uint8, SDK layout).

        nv12: ``(h*3//2, w)`` (Y then interleaved UV); rgb24: ``(h, w, 3)``;
        gray8: ``(h, w)``.
        """
        if self._released:
            raise DspError("write on released pool")
        expected = self._expected_shape()
        if arr.dtype != np.uint8 or arr.shape != expected:
            raise DspError(f"write expects uint8 {expected}, got "
                           f"{arr.dtype} {arr.shape}")

        h, w = self.height, self.width
        if self.fmt == "nv12":
            planes = [arr[:h], arr[h:]]
        elif self.fmt == "rgb24":
            planes = [np.ascontiguousarray(arr).reshape(h, w * 3)]
        else:
            planes = [arr]

        base = index * _plane_count(self.fmt)
        for p, plane in enumerate(planes):
            fd = self.plane_fds[base + p]
            _dma_buf_sync(fd, _DMA_BUF_SYNC_WRITE | _DMA_BUF_SYNC_START)
            with mmap.mmap(fd, self.plane_sizes[p],
                           prot=mmap.PROT_READ | mmap.PROT_WRITE) as mm:
                self._copy_rows(mm, self.strides[p], plane, to_mem=True)
            _dma_buf_sync(fd, _DMA_BUF_SYNC_WRITE | _DMA_BUF_SYNC_END)

    # -- device -> CPU -------------------------------------------------------
    def read(self, index: int) -> np.ndarray:
        """Read buffer ``index`` back as a numpy array (SDK layout)."""
        if self._released:
            raise DspError("read on released pool")
        h, w = self.height, self.width
        base = index * _plane_count(self.fmt)
        planes = []
        for p, (row_bytes, rows) in enumerate(_plane_rows(self.fmt, w, h)):
            fd = self.plane_fds[base + p]
            _dma_buf_sync(fd, _DMA_BUF_SYNC_READ | _DMA_BUF_SYNC_START)
            with mmap.mmap(fd, self.plane_sizes[p],
                           prot=mmap.PROT_READ | mmap.PROT_WRITE) as mm:
                flat = self._copy_rows(mm, self.strides[p],
                                       np.empty((rows, row_bytes), np.uint8),
                                       to_mem=False)
            _dma_buf_sync(fd, _DMA_BUF_SYNC_READ | _DMA_BUF_SYNC_END)
            planes.append(flat)
        if self.fmt == "nv12":
            return np.vstack(planes)
        if self.fmt == "rgb24":
            return planes[0].reshape(h, w, 3)
        return planes[0]

    @staticmethod
    def _copy_rows(mm, stride: int, plane: np.ndarray,
                   to_mem: bool) -> np.ndarray:
        """Stride-respecting row copy between an mmap and a plane array."""
        rows, row_bytes = plane.shape
        if stride == row_bytes:  # fast path: tightly packed
            if to_mem:
                mm[0:rows * row_bytes] = plane.tobytes()
            else:
                return np.frombuffer(mm[0:rows * row_bytes],
                                     dtype=np.uint8).reshape(rows, row_bytes)
        elif to_mem:
            for r in range(rows):
                off = r * stride
                mm[off:off + row_bytes] = plane[r].tobytes()
        else:
            buf = bytearray(rows * row_bytes)
            for r in range(rows):
                off = r * stride
                buf[r * row_bytes:(r + 1) * row_bytes] = mm[off:off + row_bytes]
            return np.frombuffer(bytes(buf), dtype=np.uint8).reshape(
                rows, row_bytes)
        return plane

    def _expected_shape(self) -> Tuple[int, ...]:
        if self.fmt == "nv12":
            return (self.height * 3 // 2, self.width)
        if self.fmt == "rgb24":
            return (self.height, self.width, 3)
        return (self.height, self.width)

    def release(self) -> None:
        """Return all buffers to the daemon (idempotent)."""
        if self._released:
            return
        self._released = True
        for bid in self.ids:
            self._client._send_release(bid)
        for fd in self.plane_fds:
            try:
                os.close(fd)
            except OSError:
                pass


class DspClient:
    """Hardware resize/crop on the camera-daemon DSP service.

    Usage::

        dsp = DspClient()
        out = dsp.resize_hw(frame.image, 640, 640, fmt="nv12")

    The ``*_hw`` methods allocate a source and destination buffer, run
    one job and return the decoded result. For hot loops, pre-allocate
    pools with :meth:`alloc_buffers` and pass ``src_pool``/``dst_pool``
    (``dst_pools`` for multi-crop) so each call only writes, submits and
    reads.
    """

    def __init__(self, sock_path: Optional[str] = None,
                 endpoint: Optional[str] = None):
        if sock_path is None:
            sock_path = os.getenv("CAMERA_SOCK_PATH", "/run/aipc/camera.sock")
        self.sock_path = sock_path
        self.endpoint = endpoint or Config.get_camera_control_endpoint()
        self._sock: Optional[socket.socket] = None
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[camera_pb2_grpc.CameraControlStub] = None
        self.last_used_hw: Optional[bool] = None

    # -- life cycle ----------------------------------------------------------
    def _ensure_sock(self) -> socket.socket:
        if self._sock is None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.sock_path)
            except OSError as e:
                sock.close()
                raise DspError(f"cannot connect to camera socket "
                               f"{self.sock_path}: {e}") from e
            self._sock = sock
        return self._sock

    def _connect(self) -> camera_pb2_grpc.CameraControlStub:
        if self._stub is None:
            self._channel = grpc.insecure_channel(self.endpoint)
            self._stub = camera_pb2_grpc.CameraControlStub(self._channel)
        return self._stub

    def close(self) -> None:
        """Close both transports. The daemon releases our DSP buffers."""
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self) -> "DspClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- UDS buffer management -----------------------------------------------
    def _exchange_alloc(self, width: int, height: int, fmt_wire: int,
                        count: int):
        """Send DSP_ALLOC, await RESP (with fds). Returns pool ingredients."""
        sock = self._ensure_sock()
        sock.sendall(alloc_request_bytes(width, height, fmt_wire, count))

        buf = b""
        fds: List[int] = []
        while len(buf) < _ALLOC_RESP_SIZE:
            data, got = _recvmsg_with_fds(sock, _ALLOC_RESP_SIZE - len(buf),
                                          max_fds=_DSP_MAX_FDS)
            if not data and not got:
                raise DspError("camera socket closed during DSP alloc")
            buf += data
            fds.extend(got)

        code, n, num_planes, strides, sizes, ids = parse_alloc_resp(buf)
        if code != 0:
            for fd in fds:
                os.close(fd)
            raise DspError(_ERROR_TEXT.get(code, "alloc failed"), code=code)
        if n != count or len(fds) != n * num_planes:
            for fd in fds:
                os.close(fd)
            raise DspError(f"alloc returned {n} buffers / {len(fds)} fds, "
                           f"requested {count}")
        return code, n, num_planes, strides, sizes, ids, fds

    def alloc_buffers(self, width: int, height: int, fmt: str = "nv12",
                      count: int = 1) -> DspBufferPool:
        """Allocate ``count`` daemon-side DSP buffers of one geometry."""
        _validate_geometry(width, height, fmt, "alloc")
        if count < 1:
            raise DspError("count must be >= 1")
        if count * _plane_count(fmt) > _DSP_MAX_FDS:
            raise DspError(f"count*num_planes exceeds the {_DSP_MAX_FDS}-fd "
                           f"UDS response cap")
        _code, _n, _planes, strides, sizes, ids, fds = \
            self._exchange_alloc(width, height, _HAL_PIXEL_FORMAT[fmt], count)
        return DspBufferPool(self, width, height, fmt, ids, fds,
                             strides, sizes)

    def _send_release(self, buffer_id: int) -> None:
        """Fire-and-forget DSP_BUF_RELEASE (the daemon never answers)."""
        if self._sock is None:
            return
        try:
            self._sock.sendall(struct.pack(
                _RELEASE_FMT, _FD_PUB_MSG_DSP_BUF_RELEASE,
                struct.calcsize(_RELEASE_FMT), buffer_id))
        except OSError:
            logger.debug("DSP release send failed", exc_info=True)

    # -- job submission --------------------------------------------------------
    def _submit_job(self, op: int, src_id: int, dst_ids: Sequence[int],
                    rects: Sequence[Tuple[int, ...]], interpolation: str,
                    scaling: str, priority: str, timeout_s: float) -> int:
        try:
            interp = _INTERP_WIRE[interpolation]
            scale = _SCALING_WIRE[scaling]
            prio = _PRIORITY_WIRE[priority]
        except KeyError as e:
            raise DspError(f"unknown job parameter {e}") from e

        req = camera_pb2.DspJobRequest(
            op=op,
            src_buffer_id=src_id,
            dst_buffer_ids=list(dst_ids),
            rects=[camera_pb2.DspRect(x=r[0], y=r[1], width=r[2], height=r[3],
                                      dst_width=r[4], dst_height=r[5])
                   for r in rects],
            # ALWAYS explicit: proto default 0 = NEAREST, which the vendor
            # MULTI_CROP path rejects (HAL -2801)
            interpolation=interp,
            scaling_mode=scale,
            priority=prio,
        )
        try:
            resp = self._connect().SubmitDspJob(req, timeout=timeout_s)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                raise _DspUnavailable("SubmitDspJob not in daemon") from e
            raise DspError(f"SubmitDspJob rpc failed: {e}") from e
        if not resp.success:
            if resp.error_code == DSP_SERVICE_UNAVAILABLE:
                raise _DspUnavailable("dsp service not running")
            raise DspError("dsp job failed: "
                           f"{resp.message or _ERROR_TEXT.get(resp.error_code)}",
                           code=resp.error_code)
        return resp.elapsed_ms

    # -- public hw API ----------------------------------------------------------
    def resize_hw(self, src: np.ndarray, width: int, height: int,
                  fmt: Optional[str] = None, interpolation: str = "bilinear",
                  scaling: str = "stretch", priority: str = "normal",
                  timeout_s: float = 5.0,
                  src_pool: Optional[DspBufferPool] = None,
                  dst_pool: Optional[DspBufferPool] = None) -> np.ndarray:
        """Scale ``src`` to ``(width, height)`` on the DSP."""
        fmt = _infer_fmt(src, fmt)
        _validate_geometry(width, height, fmt, "destination")
        try:
            src_pool, pools, own = self._prep(src, fmt,
                                              [(width, height, 1)],
                                              src_pool,
                                              [dst_pool] if dst_pool else None)
            try:
                src_pool.write(0, src)
                self._submit_job(_OP_RESIZE, src_pool.buffer_id(0),
                                 [pools[0].buffer_id(0)], [],
                                 interpolation, scaling, priority, timeout_s)
                self.last_used_hw = True
                return pools[0].read(0)
            finally:
                self._release_owned(own)
        except _DspUnavailable as e:
            logger.warning("DSP unavailable (%s); CPU fallback", e)
            self.last_used_hw = False
            return _cpu_resize(src, fmt, width, height, scaling, interpolation)

    def crop_hw(self, src: np.ndarray, x: int, y: int, width: int,
                height: int, dst_width: Optional[int] = None,
                dst_height: Optional[int] = None, fmt: Optional[str] = None,
                interpolation: str = "bilinear", scaling: str = "stretch",
                priority: str = "normal", timeout_s: float = 5.0,
                src_pool: Optional[DspBufferPool] = None,
                dst_pool: Optional[DspBufferPool] = None) -> np.ndarray:
        """Crop ``(x, y, w, h)`` and scale to the destination size."""
        fmt = _infer_fmt(src, fmt)
        dst_width = width if dst_width is None else dst_width
        dst_height = height if dst_height is None else dst_height
        rect = _validated_rect(src, fmt, x, y, width, height,
                               dst_width, dst_height)
        try:
            src_pool, pools, own = self._prep(
                src, fmt, [(dst_width, dst_height, 1)], src_pool,
                [dst_pool] if dst_pool else None)
            try:
                src_pool.write(0, src)
                self._submit_job(_OP_CROP_AND_RESIZE, src_pool.buffer_id(0),
                                 [pools[0].buffer_id(0)], [rect],
                                 interpolation, scaling, priority, timeout_s)
                self.last_used_hw = True
                return pools[0].read(0)
            finally:
                self._release_owned(own)
        except _DspUnavailable as e:
            logger.warning("DSP unavailable (%s); CPU fallback", e)
            self.last_used_hw = False
            out = _cpu_crop(src, fmt, x, y, width, height)
            if (dst_width, dst_height) != (width, height):
                out = _cpu_resize(out, fmt, dst_width, dst_height,
                                  "stretch", interpolation)
            return out

    def multi_crop_hw(self, src: np.ndarray,
                      rects: Sequence[Tuple[int, int, int, int, int, int]],
                      fmt: Optional[str] = None,
                      interpolation: str = "bilinear", scaling: str = "stretch",
                      priority: str = "normal", timeout_s: float = 5.0,
                      src_pool: Optional[DspBufferPool] = None,
                      dst_pools: Optional[List[DspBufferPool]] = None
                      ) -> List[np.ndarray]:
        """Crop/resize many windows in one job.

        ``rects`` are ``(x, y, w, h, dst_width, dst_height)``. Destination
        buffers are grouped by geometry (one pool per distinct output
        size); results come back in rect order.
        """
        fmt = _infer_fmt(src, fmt)
        if not rects:
            raise DspError("multi_crop needs at least one rect")
        if len(rects) > _MAX_BATCH:
            raise DspError(f"{len(rects)} rects exceed daemon batch cap "
                           f"{_MAX_BATCH}")
        rects = [_validated_rect(src, fmt, *r) for r in rects]

        # one pool per distinct destination geometry, sized by multiplicity
        order: List[Tuple[int, int]] = []
        per_geom: dict = {}
        for r in rects:
            geom = (r[4], r[5])
            if geom not in per_geom:
                per_geom[geom] = 0
                order.append(geom)
            per_geom[geom] += 1
        specs = [(dw, dh, per_geom[(dw, dh)]) for dw, dh in order]

        try:
            src_pool, pools, own = self._prep(src, fmt, specs, src_pool,
                                              dst_pools)
            try:
                src_pool.write(0, src)
                slots = {g: 0 for g in order}
                dst_ids = []
                for r in rects:
                    geom = (r[4], r[5])
                    dst_ids.append(pools[order.index(geom)]
                                   .buffer_id(slots[geom]))
                    slots[geom] += 1
                self._submit_job(_OP_MULTI_CROP, src_pool.buffer_id(0),
                                 dst_ids, rects, interpolation, scaling,
                                 priority, timeout_s)
                self.last_used_hw = True
                out, done = [], {g: 0 for g in order}
                for r in rects:
                    geom = (r[4], r[5])
                    pool = pools[order.index(geom)]
                    out.append(pool.read(done[geom]))
                    done[geom] += 1
                return out
            finally:
                self._release_owned(own)
        except _DspUnavailable as e:
            logger.warning("DSP unavailable (%s); CPU fallback", e)
            self.last_used_hw = False
            return [_cpu_crop_resize(src, fmt, r) for r in rects]

    # -- internal plumbing ------------------------------------------------------
    def _prep(self, src: np.ndarray, fmt: str,
              dst_specs: Sequence[Tuple[int, int, int]],
              src_pool: Optional[DspBufferPool],
              dst_pools: Optional[List[DspBufferPool]]
              ) -> Tuple[DspBufferPool, List[DspBufferPool],
                         List[DspBufferPool]]:
        """Assemble source/destination pools for one call; owned = temp."""
        sh, sw = _src_dims(src, fmt)
        _validate_geometry(sw, sh, fmt, "source")
        for dw, dh, _c in dst_specs:
            _validate_geometry(dw, dh, fmt, "destination")
        own: List[DspBufferPool] = []
        if src_pool is None:
            src_pool = self.alloc_buffers(sw, sh, fmt, 1)
            own.append(src_pool)
        if dst_pools is None:
            dst_pools = [self.alloc_buffers(dw, dh, fmt, c)
                         for dw, dh, c in dst_specs]
            own.extend(dst_pools)
        return src_pool, dst_pools, own

    def _release_owned(self, own: List[DspBufferPool]) -> None:
        for pool in own:
            pool.release()


# ---- format / geometry helpers ---------------------------------------------
def _infer_fmt(src: np.ndarray, fmt: Optional[str]) -> str:
    if fmt is not None:
        if fmt not in _DSP_FORMATS:
            raise DspError(f"unsupported format {fmt!r} (nv12/rgb24/gray8)")
        return fmt
    if src.ndim == 3 and src.shape[2] == 3:
        return "rgb24"
    if src.ndim == 2:
        return "gray8"  # ambiguous with nv12 — pass fmt explicitly for YUV
    raise DspError(f"cannot infer format from shape {src.shape}")


def _src_dims(src: np.ndarray, fmt: str) -> Tuple[int, int]:
    if fmt == "nv12":
        if src.ndim != 2:
            raise DspError("nv12 src must be (h*3//2, w)")
        return src.shape[0] * 2 // 3, src.shape[1]
    if fmt == "rgb24":
        if src.ndim != 3 or src.shape[2] != 3:
            raise DspError("rgb24 src must be (h, w, 3)")
        return src.shape[0], src.shape[1]
    if src.ndim != 2:
        raise DspError("gray8 src must be (h, w)")
    return src.shape


def _validated_rect(src: np.ndarray, fmt: str, x: int, y: int, w: int, h: int,
                    dw: int, dh: int) -> Tuple[int, int, int, int, int, int]:
    sh, sw = _src_dims(src, fmt)
    if w < 1 or h < 1:
        raise DspError(f"crop size must be positive (got {w}x{h})")
    if x < 0 or y < 0 or x + w > sw or y + h > sh:
        raise DspError(f"crop ({x},{y},{w}x{h}) outside source {sw}x{sh}")
    if fmt == "nv12" and any(v % 2 for v in (x, y, w, h, dw, dh)):
        raise DspError("nv12 crop/dest coords and sizes must be even")
    if not (_MIN_DIM <= dw <= _MAX_DIM and _MIN_DIM <= dh <= _MAX_DIM):
        raise DspError(f"destination dims {dw}x{dh} outside daemon range "
                       f"[{_MIN_DIM}, {_MAX_DIM}]")
    return x, y, w, h, dw, dh


# ---- CPU fallback ------------------------------------------------------------
def _resize_plane(plane: np.ndarray, out_h: int, out_w: int,
                  interpolation: str) -> np.ndarray:
    if _cv2 is not None:
        return _cv2.resize(plane, (out_w, out_h),
                           interpolation=getattr(_cv2, _CV_INTERP[interpolation]))
    rows = np.arange(out_h) * plane.shape[0] // out_h
    cols = np.arange(out_w) * plane.shape[1] // out_w
    return plane[np.ix_(rows, cols)]


def _cpu_resize(src: np.ndarray, fmt: str, dw: int, dh: int, scaling: str,
                interpolation: str) -> np.ndarray:
    sh, sw = _src_dims(src, fmt)
    if scaling in ("letterbox", "letterbox_middle", "letterbox_up_left"):
        scale = min(dw / sw, dh / sh)
        cw, ch = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
        if fmt == "nv12":
            cw &= ~1
            ch &= ~1
        content = _cpu_resize(src, fmt, cw, ch, "stretch", interpolation)
        ox = (dw - cw) // 2 if scaling != "letterbox_up_left" else 0
        oy = (dh - ch) // 2 if scaling != "letterbox_up_left" else 0
        if fmt == "nv12":  # keep UV half-sample alignment
            ox &= ~1
            oy &= ~1
        return _place(fmt, content, dw, dh, ox, oy)
    if scaling == "scale_crop":
        scale = max(dw / sw, dh / sh)
        cw, ch = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
        if fmt == "nv12":
            cw &= ~1
            ch &= ~1
        big = _cpu_resize(src, fmt, cw, ch, "stretch", interpolation)
        return _cpu_crop(big, fmt, (cw - dw) // 2 & ~1, (ch - dh) // 2 & ~1,
                         dw, dh)
    # stretch
    if fmt == "nv12":
        y = _resize_plane(src[:sh], dh, dw, interpolation)
        # UV rows are interleaved (U,V) pairs: dw/2 samples = dw bytes wide
        uv = _resize_plane(src[sh:], dh // 2, dw, "nearest")
        return np.vstack([y, uv])
    return _resize_plane(src, dh, dw, interpolation)


def _place(fmt: str, content: np.ndarray, dw: int, dh: int, ox: int, oy: int
           ) -> np.ndarray:
    """Paste ``content`` onto a black canvas at (ox, oy); nv12 pads UV 128."""
    if fmt == "nv12":
        ch = content.shape[0] * 2 // 3
        cw = content.shape[1]
        canvas = np.zeros((dh * 3 // 2, dw), dtype=np.uint8)
        canvas[dh:, :] = 128                      # neutral chroma
        canvas[oy:oy + ch, ox:ox + cw] = content[:ch]
        canvas[dh + oy // 2:dh + (oy + ch) // 2,
               ox // 2:(ox + cw) // 2] = content[ch:]
        return canvas
    canvas = np.zeros((dh, dw) if fmt == "gray8" else (dh, dw, 3),
                      dtype=np.uint8)
    canvas[oy:oy + content.shape[0], ox:ox + content.shape[1]] = content
    return canvas


def _cpu_crop(src: np.ndarray, fmt: str, x: int, y: int, w: int,
              h: int) -> np.ndarray:
    if fmt == "nv12":
        sh = src.shape[0] * 2 // 3
        # interleaved UV: a w-pixel crop spans w bytes of chroma rows
        return np.vstack([
            src[y:y + h, x:x + w],
            src[sh + y // 2:sh + (y + h) // 2, x:x + w],
        ])
    return src[y:y + h, x:x + w].copy()


def _cpu_crop_resize(src: np.ndarray, fmt: str,
                     rect: Tuple[int, ...]) -> np.ndarray:
    x, y, w, h, dw, dh = rect
    out = _cpu_crop(src, fmt, x, y, w, h)
    if (dw, dh) != (w, h):
        out = _cpu_resize(out, fmt, dw, dh, "stretch", "bilinear")
    return out
