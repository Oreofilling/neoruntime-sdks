"""DSP offload client (SDK-2).

Thin wrapper over the camera-daemon DSP service (platform PLAT-1..6):
buffers are allocated on the FD-publisher UDS (``/run/aipc/camera.sock``,
FD_PUB_MSG_DSP_ALLOC/RESP/BUF_RELEASE, dma-buf fds via SCM_RIGHTS) and
jobs are submitted through the ``SubmitDspJob`` gRPC on camera-control.

Hardware-first with a numpy CPU fallback: on a daemon without the DSP
RPC (grpc UNIMPLEMENTED) or with the service not running (error -5) the
``*_hw`` helpers compute the result on CPU instead of raising, and
``client.last_used_hw`` records which path served the last call.

Caveat (P0 platform contract): a job source must be a daemon-registered
dma-buf, so a plain numpy array is copied into one. Zero-copy input IS
available for camera frames: pass a :class:`Frame` received with
``keep_fd=True` (or its ``.handle``) and the dma-buf fds are imported
straight into the DSP service (DSP_IMPORT) — no pixel copy, ~15x faster
than the copy-in path on 4K frames.

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

    # zero-copy: keep the frame's dma-bufs and hand them over directly
    frame = media.get_frame("main", keep_fd=True)
    small = client.resize_hw(frame, 640, 640)
"""

from __future__ import annotations

import logging
import mmap
import os
import socket
import struct
import warnings
from typing import Sequence, Union

import grpc
import numpy as np

from ._transport import GrpcClient
from ._transport import recvmsg_with_fds as _recvmsg_with_fds
from .dsp_format import (  # noqa: F401 — re-exported for API compat
    _CV_INTERP,
    _FRAME_FMT_TO_DSP,
    _as_pixels,
    _cpu_crop,
    _cpu_crop_resize,
    _cpu_resize,
    _infer_fmt,
    _src_dims,
    _validated_rect,
)
from .dsp_wire import (  # noqa: F401 — re-exported for API compat
    _ALLOC_RESP_SIZE,
    _DSP_MAX_FDS,
    _ERROR_TEXT,
    _FD_PUB_MSG_DSP_ALLOC,
    _FD_PUB_MSG_DSP_ALLOC_RESP,
    _FD_PUB_MSG_DSP_BUF_RELEASE,
    _FD_PUB_MSG_DSP_IMPORT,
    _FD_PUB_MSG_DSP_IMPORT_RESP,
    _FD_PUB_MSG_ERROR,
    _FD_PUB_MSG_OK,
    _HAL_PIXEL_FORMAT,
    _INTERP_WIRE,
    _MAX_BATCH,
    _MAX_DIM,
    _MIN_DIM,
    _OP_CROP_AND_RESIZE,
    _OP_MULTI_CROP,
    _OP_RESIZE,
    _PRIORITY_WIRE,
    _RELEASE_FMT,
    _SCALING_WIRE,
    DSP_SERVICE_UNAVAILABLE,
    DspError,
    _DspUnavailable,
    _plane_count,
    _plane_rows,
    _validate_geometry,
    alloc_request_bytes,
    import_request_bytes,
    parse_alloc_resp,
    parse_import_resp,
)
from .frame import (
    _DMA_BUF_SYNC_END,
    _DMA_BUF_SYNC_READ,
    _DMA_BUF_SYNC_START,
    _DMA_BUF_SYNC_WRITE,
    Frame,
    FrameHandle,
    _dma_buf_sync,
)
from .proto import camera_pb2, camera_pb2_grpc

logger = logging.getLogger(__name__)

JobSource = Union[np.ndarray, Frame, FrameHandle]


class DspBufferPool:
    """Daemon-allocated dma-buf buffers sharing one geometry.

    One wire allocation returns ``count`` buffers; every buffer exposes
    ``_plane_count(fmt)`` dma-buf fds (NV12: Y + interleaved-UV). Plane
    rows may be padded (``strides`` > row bytes); write/read copy
    row-by-row so padding is preserved. ``release()`` returns the buffers
    to the daemon and closes every fd; closing the client's UDS releases
    them too (daemon-side cleanup on disconnect).
    """

    def __init__(
        self,
        client: DspClient,
        width: int,
        height: int,
        fmt: str,
        ids: Sequence[int],
        fds: Sequence[int],
        strides: Sequence[int],
        sizes: Sequence[int],
    ):
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
            raise DspError(
                f"alloc returned {len(self.plane_fds)} fds for "
                f"{len(self.ids)} {fmt} buffers (need "
                f"{len(self.ids) * _plane_count(fmt)})"
            )

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
            raise DspError(f"write expects uint8 {expected}, got {arr.dtype} {arr.shape}")

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
            with mmap.mmap(fd, self.plane_sizes[p], prot=mmap.PROT_READ | mmap.PROT_WRITE) as mm:
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
            with mmap.mmap(fd, self.plane_sizes[p], prot=mmap.PROT_READ | mmap.PROT_WRITE) as mm:
                flat = self._copy_rows(
                    mm, self.strides[p], np.empty((rows, row_bytes), np.uint8), to_mem=False
                )
            _dma_buf_sync(fd, _DMA_BUF_SYNC_READ | _DMA_BUF_SYNC_END)
            planes.append(flat)
        if self.fmt == "nv12":
            return np.vstack(planes)
        if self.fmt == "rgb24":
            return planes[0].reshape(h, w, 3)
        return planes[0]

    @staticmethod
    def _copy_rows(mm, stride: int, plane: np.ndarray, to_mem: bool) -> np.ndarray:
        """Stride-respecting row copy between an mmap and a plane array."""
        rows, row_bytes = plane.shape
        if stride == row_bytes:  # fast path: tightly packed
            if to_mem:
                mm[0 : rows * row_bytes] = plane.tobytes()
            else:
                return np.frombuffer(mm[0 : rows * row_bytes], dtype=np.uint8).reshape(
                    rows, row_bytes
                )
        elif to_mem:
            for r in range(rows):
                off = r * stride
                mm[off : off + row_bytes] = plane[r].tobytes()
        else:
            buf = bytearray(rows * row_bytes)
            for r in range(rows):
                off = r * stride
                buf[r * row_bytes : (r + 1) * row_bytes] = mm[off : off + row_bytes]
            return np.frombuffer(bytes(buf), dtype=np.uint8).reshape(rows, row_bytes)
        return plane

    def _expected_shape(self) -> tuple[int, ...]:
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


class _ImportedSource:
    """A zero-copy job source: dma-buf fds imported via DSP_IMPORT.

    Quacks like a one-buffer DspBufferPool for the ``*_hw`` call sites
    (``buffer_id``/``release``); there is no ``write`` — the pixels
    already live in the frame's dma-bufs.
    """

    def __init__(self, client: DspClient, import_id: int):
        self._client = client
        self.import_id = import_id

    def buffer_id(self, index: int) -> int:
        return self.import_id  # single buffer; index kept for symmetry

    def release(self) -> None:
        """Return the import to the daemon (idempotent)."""
        if self.import_id < 0:
            return
        self._client._send_release(self.import_id)
        self.import_id = -1


def _recv_one_msg(sock: socket.socket) -> tuple[int, bytes, list[int]]:
    """One complete UDS message: ``(type, payload-with-header, fds)``.

    Every byte — the header included — must come from recvmsg: on a
    stream socket SCM_RIGHTS rides with the first byte of the sender's
    sendmsg, and a plain recv consuming that byte silently drops the
    ancillary record (a bug that cost two debugging rounds on-device).
    Any fds that do arrive are the caller's to close.
    """
    hdr = b""
    fds: list[int] = []
    while len(hdr) < 8:
        data, got = _recvmsg_with_fds(sock, 8 - len(hdr))
        if not data and not got:
            raise DspError("camera socket closed waiting for a message")
        hdr += data
        fds.extend(got)
    mtype, msize = struct.unpack_from("<II", hdr)
    if msize < 8 or msize > 1 << 20:
        raise DspError(f"corrupt camera-sock header: type={mtype} size={msize}")
    body = b""
    while len(body) < msize - 8:
        data, got = _recvmsg_with_fds(sock, msize - 8 - len(body))
        if not data and not got:
            raise DspError("camera socket closed mid-message")
        body += data
        fds.extend(got)
    return mtype, hdr + body, fds


def _resolve_source(src, fmt: str | None) -> tuple[int, int, FrameHandle | None, str]:
    """Normalize a ``*_hw`` source into ``(width, height, handle, fmt)``.

    ``handle`` is None for array sources (ndarray, or a Frame that only
    carries pixels); for a Frame/FrameHandle it is the retained dma-buf
    handle and the geometry comes with it.
    """
    if isinstance(src, FrameHandle):
        frame = None
        handle = src
        if handle.closed:
            raise DspError(
                "frame handle is closed — its dma-bufs are gone; "
                "keep the Frame/FrameHandle alive across the call"
            )
        src_fmt = _FRAME_FMT_TO_DSP.get(handle.format)
        if src_fmt is None:
            raise DspError(
                f"{handle.format or 'unknown-format'} frames "
                "cannot be imported as a DSP source "
                "(supported: NV12/RGB/BGR/GRAY8)"
            )
        if handle.width <= 0 or handle.height <= 0:
            raise DspError(
                "frame handle carries no geometry — it predates SDK 0.6.0; re-fetch the frame"
            )
    elif isinstance(src, Frame):
        frame = src
        handle = src.handle
        if handle is None:
            if src.image is None:
                raise DspError(
                    "frame has neither pixels nor a dma-buf handle "
                    "— subscribe/receive with keep_fd=True to use "
                    "it as a zero-copy source"
                )
            # the frame's format metadata outranks shape inference — a 2D
            # NV12 array is indistinguishable from gray8 by shape alone
            frame_fmt = _FRAME_FMT_TO_DSP.get(src.format)
            if fmt is not None and frame_fmt is not None and fmt != frame_fmt:
                raise DspError(f"format mismatch: source is {frame_fmt!r}, fmt={fmt!r}")
            resolved = frame_fmt if fmt is None else _infer_fmt(src.image, fmt)
            sh, sw = _src_dims(src.image, resolved)
            return sw, sh, None, resolved
        if handle.closed:
            raise DspError(
                "frame handle is closed — its dma-bufs are gone; "
                "keep the Frame/FrameHandle alive across the call"
            )
        src_fmt = _FRAME_FMT_TO_DSP.get(src.format)
        if src_fmt is None:
            raise DspError(
                f"{src.format or 'unknown-format'} frames cannot "
                "be imported as a DSP source "
                "(supported: NV12/RGB/BGR/GRAY8)"
            )
    else:
        resolved = _infer_fmt(src, fmt)
        sh, sw = _src_dims(src, resolved)
        return sw, sh, None, resolved

    if fmt is not None and fmt != src_fmt:
        raise DspError(f"format mismatch: source is {src_fmt!r}, fmt={fmt!r}")
    width = frame.width if frame is not None else handle.width
    height = frame.height if frame is not None else handle.height
    return width, height, handle, src_fmt


class DspClient(GrpcClient):
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

    _stub_factory = camera_pb2_grpc.CameraControlStub

    def __init__(self, sock_path: str | None = None, endpoint: str | None = None):
        super().__init__(endpoint)
        if sock_path is None:
            sock_path = os.getenv("CAMERA_SOCK_PATH", "/run/aipc/camera.sock")
        self.sock_path = sock_path
        self._sock: socket.socket | None = None
        self.last_used_hw: bool | None = None

    # -- life cycle ----------------------------------------------------------
    def _ensure_sock(self) -> socket.socket:
        if self._sock is None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.sock_path)
            except OSError as e:
                sock.close()
                raise DspError(f"cannot connect to camera socket {self.sock_path}: {e}") from e
            self._sock = sock
        return self._sock

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

    def __enter__(self) -> DspClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- UDS buffer management -----------------------------------------------
    def _exchange_alloc(self, width: int, height: int, fmt_wire: int, count: int):
        """Send DSP_ALLOC, await RESP (with fds). Returns pool ingredients."""
        sock = self._ensure_sock()
        sock.sendall(alloc_request_bytes(width, height, fmt_wire, count))

        buf = b""
        fds: list[int] = []
        while len(buf) < _ALLOC_RESP_SIZE:
            data, got = _recvmsg_with_fds(sock, _ALLOC_RESP_SIZE - len(buf), max_fds=_DSP_MAX_FDS)
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
            raise DspError(f"alloc returned {n} buffers / {len(fds)} fds, requested {count}")
        return code, n, num_planes, strides, sizes, ids, fds

    def alloc_buffers(
        self, width: int, height: int, fmt: str = "nv12", count: int = 1
    ) -> DspBufferPool:
        """Allocate ``count`` daemon-side DSP buffers of one geometry."""
        _validate_geometry(width, height, fmt, "alloc")
        if count < 1:
            raise DspError("count must be >= 1")
        if count * _plane_count(fmt) > _DSP_MAX_FDS:
            raise DspError(f"count*num_planes exceeds the {_DSP_MAX_FDS}-fd UDS response cap")
        _code, _n, _planes, strides, sizes, ids, fds = self._exchange_alloc(
            width, height, _HAL_PIXEL_FORMAT[fmt], count
        )
        return DspBufferPool(self, width, height, fmt, ids, fds, strides, sizes)

    def _send_release(self, buffer_id: int) -> None:
        """Fire-and-forget DSP_BUF_RELEASE (the daemon never answers)."""
        if self._sock is None:
            return
        try:
            self._sock.sendall(
                struct.pack(
                    _RELEASE_FMT,
                    _FD_PUB_MSG_DSP_BUF_RELEASE,
                    struct.calcsize(_RELEASE_FMT),
                    buffer_id,
                )
            )
        except OSError:
            logger.debug("DSP release send failed", exc_info=True)

    def _import_source(
        self, handle: FrameHandle, width: int, height: int, fmt: str, timeout_s: float = 5.0
    ) -> int:
        """Import a frame's dma-bufs as a job source (DSP_IMPORT).

        The daemon dups the fds, so the import outlives the FrameHandle;
        our fd copies stay owned (and open) by the handle. Returns the
        import id — same registry namespace as pool buffer ids, valid as
        a ``src_buffer_id`` until freed with DSP_BUF_RELEASE.
        """
        num_planes = len(handle.fds)
        if num_planes != _plane_count(fmt):
            raise DspError(
                f"{fmt} source carries {num_planes} dma-buf fd(s), expected {_plane_count(fmt)}"
            )
        strides = list(handle.strides[:3]) + [0] * (3 - len(handle.strides[:3]))
        sizes = list(handle.plane_sizes[:3]) + [0] * (3 - len(handle.plane_sizes[:3]))
        payload = import_request_bytes(
            width, height, _HAL_PIXEL_FORMAT[fmt], num_planes, strides, sizes
        )
        anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack(f"{num_planes}i", *handle.fds))]
        sock = self._ensure_sock()
        # scatter/gather form: some device python builds reject
        # sendmsg(bytes, ancdata) with a TypeError but accept a buffer list
        sock.sendmsg([payload], anc)

        sock.settimeout(timeout_s)
        try:
            for _drain in range(64):
                try:
                    mtype, msg, fds = _recv_one_msg(sock)
                except socket.timeout:
                    raise DspError(
                        f"no DSP_IMPORT response in {timeout_s}s — the daemon "
                        "may predate DSP_IMPORT (needs platform a94ee007+); "
                        "close this client, the socket may hold a partial "
                        "message"
                    ) from None
                for fd in fds:  # the reply itself never carries fds
                    os.close(fd)
                if mtype == _FD_PUB_MSG_DSP_IMPORT_RESP:
                    code, import_id = parse_import_resp(msg)
                    if code != 0:
                        raise DspError(
                            f"frame import rejected: {_ERROR_TEXT.get(code, 'error')}", code=code
                        )
                    return import_id
                if mtype in (_FD_PUB_MSG_OK, _FD_PUB_MSG_ERROR):
                    continue  # control acks from an earlier request
                # a FRAME here means this socket is subscribed somewhere —
                # a DspClient socket never is, so treat it as protocol desync
                raise DspError(
                    f"unexpected camera-sock message type {mtype} "
                    "while awaiting DSP import response"
                )
        finally:
            sock.settimeout(None)
        raise DspError("too many control messages before import response")

    # -- job submission --------------------------------------------------------
    def _submit_job(
        self,
        op: int,
        src_id: int,
        dst_ids: Sequence[int],
        rects: Sequence[tuple[int, ...]],
        interpolation: str,
        scaling: str,
        priority: str,
        timeout_s: float,
    ) -> int:
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
            rects=[
                camera_pb2.DspRect(
                    x=r[0], y=r[1], width=r[2], height=r[3], dst_width=r[4], dst_height=r[5]
                )
                for r in rects
            ],
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
            raise DspError(
                f"dsp job failed: {resp.message or _ERROR_TEXT.get(resp.error_code)}",
                code=resp.error_code,
            )
        return resp.elapsed_ms

    # -- public hw API ----------------------------------------------------------
    def resize_hw(
        self,
        src: JobSource,
        width: int,
        height: int,
        fmt: str | None = None,
        interpolation: str = "bilinear",
        scaling: str = "stretch",
        priority: str = "normal",
        timeout_s: float = 5.0,
        src_pool: DspBufferPool | None = None,
        dst_pool: DspBufferPool | None = None,
    ) -> np.ndarray:
        """Scale ``src`` to ``(width, height)`` on the DSP.

        ``src`` is a numpy array (copied in) or a keep-fd Frame/FrameHandle
        (imported zero-copy — see the module docstring).
        """
        _sw, _sh, handle, fmt = _resolve_source(src, fmt)
        _validate_geometry(width, height, fmt, "destination")
        try:
            source, pools, own = self._prep(
                src,
                fmt,
                [(width, height, 1)],
                src_pool,
                [dst_pool] if dst_pool else None,
                timeout_s,
            )
            try:
                self._submit_job(
                    _OP_RESIZE,
                    source.buffer_id(0),
                    [pools[0].buffer_id(0)],
                    [],
                    interpolation,
                    scaling,
                    priority,
                    timeout_s,
                )
                self.last_used_hw = True
                return pools[0].read(0)
            finally:
                self._release_owned(own)
        except _DspUnavailable as e:
            if handle is not None:
                raise DspError(
                    "DSP unavailable with a zero-copy frame source — refusing "
                    "the silent CPU fallback (the frame holds fds, not "
                    "pixels; use frame.to_array() to accept the copy)"
                ) from e
            warnings.warn(
                f"DSP unavailable ({e}); CPU fallback engaged "
                "(client.last_used_hw records the path used)",
                UserWarning,
                stacklevel=3,
            )
            self.last_used_hw = False
            return _cpu_resize(_as_pixels(src), fmt, width, height, scaling, interpolation)

    def crop_hw(
        self,
        src: JobSource,
        x: int,
        y: int,
        width: int,
        height: int,
        dst_width: int | None = None,
        dst_height: int | None = None,
        fmt: str | None = None,
        interpolation: str = "bilinear",
        scaling: str = "stretch",
        priority: str = "normal",
        timeout_s: float = 5.0,
        src_pool: DspBufferPool | None = None,
        dst_pool: DspBufferPool | None = None,
    ) -> np.ndarray:
        """Crop ``(x, y, w, h)`` and scale to the destination size."""
        sw, sh, handle, fmt = _resolve_source(src, fmt)
        dst_width = width if dst_width is None else dst_width
        dst_height = height if dst_height is None else dst_height
        rect = _validated_rect(sw, sh, fmt, x, y, width, height, dst_width, dst_height)
        try:
            source, pools, own = self._prep(
                src,
                fmt,
                [(dst_width, dst_height, 1)],
                src_pool,
                [dst_pool] if dst_pool else None,
                timeout_s,
            )
            try:
                self._submit_job(
                    _OP_CROP_AND_RESIZE,
                    source.buffer_id(0),
                    [pools[0].buffer_id(0)],
                    [rect],
                    interpolation,
                    scaling,
                    priority,
                    timeout_s,
                )
                self.last_used_hw = True
                return pools[0].read(0)
            finally:
                self._release_owned(own)
        except _DspUnavailable as e:
            if handle is not None:
                raise DspError(
                    "DSP unavailable with a zero-copy frame source — refusing "
                    "the silent CPU fallback (the frame holds fds, not "
                    "pixels; use frame.to_array() to accept the copy)"
                ) from e
            warnings.warn(
                f"DSP unavailable ({e}); CPU fallback engaged "
                "(client.last_used_hw records the path used)",
                UserWarning,
                stacklevel=3,
            )
            self.last_used_hw = False
            out = _cpu_crop(_as_pixels(src), fmt, x, y, width, height)
            if (dst_width, dst_height) != (width, height):
                out = _cpu_resize(out, fmt, dst_width, dst_height, "stretch", interpolation)
            return out

    def multi_crop_hw(
        self,
        src: JobSource,
        rects: Sequence[tuple[int, int, int, int, int, int]],
        fmt: str | None = None,
        interpolation: str = "bilinear",
        scaling: str = "stretch",
        priority: str = "normal",
        timeout_s: float = 5.0,
        src_pool: DspBufferPool | None = None,
        dst_pools: list[DspBufferPool] | None = None,
    ) -> list[np.ndarray]:
        """Crop/resize many windows in one job.

        ``rects`` are ``(x, y, w, h, dst_width, dst_height)``. Destination
        buffers are grouped by geometry (one pool per distinct output
        size); results come back in rect order.
        """
        sw, sh, handle, fmt = _resolve_source(src, fmt)
        if not rects:
            raise DspError("multi_crop needs at least one rect")
        if len(rects) > _MAX_BATCH:
            raise DspError(f"{len(rects)} rects exceed daemon batch cap {_MAX_BATCH}")
        rects = [_validated_rect(sw, sh, fmt, *r) for r in rects]

        # one pool per distinct destination geometry, sized by multiplicity
        order: list[tuple[int, int]] = []
        per_geom: dict = {}
        for r in rects:
            geom = (r[4], r[5])
            if geom not in per_geom:
                per_geom[geom] = 0
                order.append(geom)
            per_geom[geom] += 1
        specs = [(dw, dh, per_geom[(dw, dh)]) for dw, dh in order]

        try:
            source, pools, own = self._prep(src, fmt, specs, src_pool, dst_pools, timeout_s)
            try:
                slots = {g: 0 for g in order}
                dst_ids = []
                for r in rects:
                    geom = (r[4], r[5])
                    dst_ids.append(pools[order.index(geom)].buffer_id(slots[geom]))
                    slots[geom] += 1
                self._submit_job(
                    _OP_MULTI_CROP,
                    source.buffer_id(0),
                    dst_ids,
                    rects,
                    interpolation,
                    scaling,
                    priority,
                    timeout_s,
                )
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
            if handle is not None:
                raise DspError(
                    "DSP unavailable with a zero-copy frame source — refusing "
                    "the silent CPU fallback (the frame holds fds, not "
                    "pixels; use frame.to_array() to accept the copy)"
                ) from e
            warnings.warn(
                f"DSP unavailable ({e}); CPU fallback engaged "
                "(client.last_used_hw records the path used)",
                UserWarning,
                stacklevel=3,
            )
            self.last_used_hw = False
            return [_cpu_crop_resize(_as_pixels(src), fmt, r) for r in rects]

    # -- internal plumbing ------------------------------------------------------
    def _prep(
        self,
        src: JobSource,
        fmt: str,
        dst_specs: Sequence[tuple[int, int, int]],
        src_pool: DspBufferPool | None,
        dst_pools: list[DspBufferPool] | None,
        timeout_s: float = 5.0,
    ) -> tuple[object, list[DspBufferPool], list[object]]:
        """Prepare one job's buffers: ``(source, dst_pools, owned)``.

        ``src`` is either a numpy array — copied into a daemon-allocated
        pool (or the caller's ``src_pool``) — or a Frame/FrameHandle whose
        dma-buf fds are imported zero-copy via DSP_IMPORT; the pixels are
        never touched on that path. ``owned`` entries are released by the
        caller's ``finally`` (temp pools and the import alike).
        """
        sw, sh, handle, fmt = _resolve_source(src, fmt)
        _validate_geometry(sw, sh, fmt, "source")
        for dw, dh, _c in dst_specs:
            _validate_geometry(dw, dh, fmt, "destination")
        own: list[object] = []
        if handle is not None:
            if src_pool is not None:
                raise DspError(
                    "src_pool applies to numpy sources; a frame handle imports its own dma-bufs"
                )
            source = _ImportedSource(self, self._import_source(handle, sw, sh, fmt, timeout_s))
            own.append(source)
        else:
            if src_pool is None:
                src_pool = self.alloc_buffers(sw, sh, fmt, 1)
                own.append(src_pool)
            src_pool.write(0, _as_pixels(src))
            source = src_pool
        if dst_pools is None:
            dst_pools = [self.alloc_buffers(dw, dh, fmt, c) for dw, dh, c in dst_specs]
            own.extend(dst_pools)
        return source, dst_pools, own

    def _release_owned(self, own: list[object]) -> None:
        for pool in own:
            pool.release()
