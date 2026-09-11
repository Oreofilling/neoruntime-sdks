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
    nv12 = client.convert_hw(frame.image, "nv12", fmt="rgb24")
    jpeg = client.encode_jpeg_hw(frame.image, quality=85, fmt="rgb24")
    annotated = client.blend_hw(nv12, [(overlay_rgba, 64, 48)])

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
    _cpu_blend,
    _cpu_convert,
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
    _OP_BLEND,
    _OP_CONVERT_FORMAT,
    _OP_CROP_AND_RESIZE,
    _OP_MULTI_CROP,
    _OP_RESIZE,
    _PRIORITY_WIRE,
    _RELEASE_FMT,
    _SCALING_WIRE,
    DSP_ERR_NO_BUFFER,
    DSP_ERR_TIMEOUT,
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
    _encode_jpeg,
)
from .proto import camera_pb2, camera_pb2_grpc

logger = logging.getLogger(__name__)

JobSource = Union[np.ndarray, Frame, FrameHandle]


def _warn_bgr_convert(src: JobSource) -> None:
    """Warn (unconditionally) when a BGR frame enters a DSP conversion.

    The DSP wire has no BGR variant — ``rgb24`` is RGB order — so BGR
    pixels come back with R/B-swapped channels. Only frames are reliably
    detectable (their ``format`` metadata says BGR); a raw BGR ndarray
    without a ``fmt`` hint is indistinguishable from RGB by shape.
    """
    if isinstance(src, Frame) and src.format == "BGR":
        logger.warning(
            "convert_hw: source frame is BGR — the DSP wire only carries "
            "rgb24, so R and B come back swapped; use frame.to_rgb() first "
            "or stay on the CPU leg (color.bgr_to_nv12)"
        )


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
        argb: ``(h, w, 4)`` (wire byte order [A, R, G, B]); gray8: ``(h, w)``.
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
        elif self.fmt == "argb":
            planes = [np.ascontiguousarray(arr).reshape(h, w * 4)]
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
        if self.fmt == "argb":
            return planes[0].reshape(h, w, 4)
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
        if self.fmt == "argb":
            return (self.height, self.width, 4)
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


class SharedImportPool:
    """A round-robin ring of shared ARGB32 imports the daemon can bake.

    The deployed HAL refuses ARGB32 dma-buf pool allocation (wire OOM),
    but its bake path blends ARGB imports fine: the daemon maps the
    memfd we pass over SCM_RIGHTS as USERPTR planes, so both processes
    share the same pages. :meth:`write` therefore lands directly in the
    memory the daemon's blend reads — no dma-buf, no copy through the
    wire.

    Lease rules mirror :class:`DspBufferPool`: slots recycle
    round-robin, the caller paces writes, and a slot must not be
    rewritten while its frame may still sit in a daemon-side queue
    (queue cap 3 ⇒ ring depth >= 4 for faster-than-bake publishing).

    Not per-publish import/release: client-streaming has no per-frame
    ack, so a release could race the daemon's pin. Not hold-until-RPC-end
    either: the daemon caps imports at 64 per client. A small persistent
    ring sidesteps both.
    """

    def __init__(
        self,
        client: DspClient,
        width: int,
        height: int,
        ids: Sequence[int],
        maps: Sequence[mmap.mmap],
        fds: Sequence[int],
    ):
        self._client = client
        self.width = width
        self.height = height
        self.fmt = "argb"
        self.ids = tuple(ids)
        self._maps = list(maps)
        self._fds = list(fds)
        self._released = False

    @property
    def count(self) -> int:
        """Number of ring slots."""
        return len(self.ids)

    @property
    def strides(self) -> tuple[int, ...]:
        """Per-plane byte strides — ARGB32 is one ``width * 4`` plane."""
        return (self.width * 4,)

    def buffer_id(self, index: int) -> int:
        """The daemon registry id of ring slot ``index``."""
        return self.ids[index]

    def write(self, index: int, arr: np.ndarray) -> None:
        """Copy one ARGB32 frame into ring slot ``index``'s shared pages."""
        if self._released:
            raise DspError("write on released pool")
        expected = (self.height, self.width, 4)
        if arr.dtype != np.uint8 or arr.shape != expected:
            raise DspError(
                f"write expects uint8 {expected}, got {arr.dtype} {arr.shape}"
            )
        self._maps[index][:] = arr.tobytes()

    def release(self) -> None:
        """Return every import and close the shared pages (idempotent)."""
        if self._released:
            return
        self._released = True
        for bid in self.ids:
            self._client._send_release(bid)
        for mm in self._maps:
            try:
                mm.close()
            except ValueError:
                pass
        for fd in self._fds:
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


class PendingDspJob:
    """The ``wait=False`` return of the ``*_hw`` job methods (P2 async).

    Wraps one submitted job and owns every buffer it needs — destination
    pool, imported sources — until the job is consumed:

    * :meth:`wait` — block, read the destination array(s), release
      everything. The sync-call equivalent, just later.
    * :meth:`wait_result` — block but keep the result device-side and the
      buffers owned: chain ``buffer_id`` into
      :meth:`DspClient.encode_jpeg_hw` (``src_buffer_id=``) for a zero
      read-back encode, then :meth:`release`.
    * :meth:`done` — poll; ``timeout_s=0`` is a pure non-blocking check.
    * :meth:`release` — drop the result and free the buffers (idempotent).

    A job the daemon *refused* or a daemon without the async rpcs never
    produces one of these — refused jobs fall back to CPU and return
    pixels (the ``wait=False`` contract only covers accepted jobs), and
    the sync fallback inside ``_submit_job`` yields ``job_id=None``,
    meaning the job already ran by construction.
    """

    _PENDING, _DONE, _RELEASED = "pending", "done", "released"

    def __init__(
        self,
        client: DspClient,
        reads: Sequence[tuple[DspBufferPool, int]],
        job_id: int | None,
        owns: Sequence[object],
        timeout_s: float,
        multi: bool = False,
    ):
        self._client = client
        self._reads = list(reads)
        self._owns = list(owns)
        self._timeout_s = timeout_s
        self._multi = multi
        self._error: DspError | None = None
        self.job_id = job_id
        # job_id None = the sync fallback already executed the job
        self._state = self._DONE if job_id is None else self._PENDING

    @property
    def buffer_id(self) -> int:
        """Daemon id of the (first) destination buffer."""
        pool, index = self._reads[0]
        return pool.buffer_id(index)

    def done(self, timeout_s: float = 0.0) -> bool:
        """Poll for completion without consuming the result.

        ``timeout_s=0`` maps to the daemon's non-blocking wait. A job
        that *failed* still counts as done — the error surfaces from
        :meth:`wait`/:meth:`wait_result`.
        """
        if self._state != self._PENDING:
            return self._state == self._DONE
        try:
            resp = self._wait_rpc(int(max(timeout_s, 0.0) * 1000))
        except _DspUnavailable:
            raise  # transport-level: the job's state is simply unknown
        except DspError as e:
            self._state, self._error = self._DONE, e
            return True
        if resp is None:
            return False
        self._state = self._DONE
        return True

    def wait_result(self, timeout_s: float | None = None) -> PendingDspJob:
        """Block until the job completes, the result staying device-side.

        A timed-out job raises but stays pending in the daemon — re-wait
        with a longer timeout. A failed job raises its error; release the
        buffers afterwards either way.
        """
        if self._state == self._RELEASED:
            raise DspError("pending job already released")
        if self._state == self._PENDING:
            limit = self._timeout_s if timeout_s is None else timeout_s
            resp = self._wait_rpc(int(max(limit, 0.0) * 1000))
            if resp is None:
                raise DspError(
                    f"dsp job {self.job_id} still pending after {limit}s — "
                    "the daemon keeps it; wait again with a longer timeout"
                )
            self._state = self._DONE
        if self._error is not None:
            raise self._error
        return self

    def wait(self, timeout_s: float | None = None) -> np.ndarray | list[np.ndarray]:
        """Block, read the destination, release the buffers."""
        self.wait_result(timeout_s)
        out = [pool.read(index) for pool, index in self._reads]
        self.release()
        return out if self._multi else out[0]

    def release(self) -> None:
        """Drop the result and release the owned buffers (idempotent).

        The daemon-side job is *not* cancelled: a still-pending job is
        first reaped with one bounded wait (it executes regardless — the
        daemon's single worker runs it either way); a job that outlives
        that wait lingers in the daemon registry until client disconnect.
        """
        if self._state == self._RELEASED:
            return
        if self._state == self._PENDING:
            try:
                self._wait_rpc(int(max(self._timeout_s, 0.0) * 1000))
            except DspError:
                pass  # callers that care surface errors from wait_result
        self._state = self._RELEASED
        for owned in self._owns:
            owned.release()
        self._reads = []

    def _wait_rpc(self, timeout_ms: int):
        """One WaitDspJob round-trip.

        Returns the response when the job completed (the daemon reaped
        its registry entry), ``None`` while it is still pending
        (``DSP_ERR_TIMEOUT``), and raises for rpc failures, unknown ids
        and failed jobs.
        """
        req = camera_pb2.DspWaitRequest(
            job_id=self.job_id if self.job_id is not None else 0,
            timeout_ms=timeout_ms,
        )
        try:
            # the server holds its reply for up to timeout_ms — keep the
            # rpc deadline comfortably past it
            resp = self._client._connect().WaitDspJob(
                req, timeout=timeout_ms / 1000.0 + 2.0
            )
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                raise _DspUnavailable("WaitDspJob not in daemon") from e
            raise DspError(f"WaitDspJob rpc failed: {e}") from e
        if not resp.success:
            if resp.error_code == DSP_ERR_TIMEOUT:
                return None  # still pending — entry kept for a re-wait
            raise DspError(
                f"dsp job failed: {resp.message or _ERROR_TEXT.get(resp.error_code)}",
                code=resp.error_code,
            )
        return resp


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

    Every ``*_hw`` method also takes ``cpu_fallback`` (default ``True``):
    when the daemon lacks the DSP surface, array-source calls warn and
    compute on CPU. Pass ``cpu_fallback=False`` to make unavailability
    raise instead — :mod:`neoruntime_ipc_sdk.accel` does this so its
    degradation accounting sees the real backend rather than CPU work
    labeled as hardware. :meth:`convert_hw` additionally falls back on a
    job the firmware refused: the pair matrix is device-dependent
    (hailo15 ``dsp_convert_format`` takes RGB<->NV12 and rejects every
    gray8 pair with ``HAL_ERR_RESULT``), so "the hardware doesn't do
    this conversion" is a runtime outcome, not a caller bug.
    """

    _stub_factory = camera_pb2_grpc.CameraControlStub

    def __init__(self, sock_path: str | None = None, endpoint: str | None = None):
        super().__init__(endpoint)
        if sock_path is None:
            sock_path = os.getenv("CAMERA_SOCK_PATH", "/run/aipc/camera.sock")
        self.sock_path = sock_path
        self._sock: socket.socket | None = None
        self.last_used_hw: bool | None = None
        # set once the daemon refuses a gray8 CONVERT (hailo15 firmware
        # gap) — later gray8 pairs skip the doomed submit quietly
        self._gray8_refused = False

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

    def import_shared_buffers(
        self,
        width: int,
        height: int,
        fmt: str = "argb",
        count: int = 2,
        timeout_s: float = 5.0,
    ) -> SharedImportPool:
        """Import ``count`` shared ARGB32 buffers as a writable ring.

        The deployed HAL refuses ARGB32 dma-buf pool allocation, so
        alpha-blend overlays ride memfd imports instead: each slot is a
        memfd the daemon maps (SCM_RIGHTS dup, MAP_SHARED both sides) and
        bakes straight from. Writing the returned pool puts pixels in the
        exact pages the daemon's blend reads. See :class:`SharedImportPool`
        for the lease contract and why the ring is persistent.
        """
        if fmt != "argb":
            raise DspError("shared import supports ARGB32 only")
        _validate_geometry(width, height, fmt, "shared import")
        if count < 1:
            raise DspError("count must be >= 1")
        stride = width * 4
        size = stride * height
        ids: list[int] = []
        maps: list[mmap.mmap] = []
        fds: list[int] = []
        fd = -1
        try:
            for _slot in range(count):
                fd = os.memfd_create("dsp-shared-import")
                os.ftruncate(fd, size)
                ids.append(
                    self._import_planes(
                        width, height, fmt, 1, [stride, 0, 0], [size, 0, 0], [fd], timeout_s
                    )
                )
                maps.append(mmap.mmap(fd, size))
                fds.append(fd)
                fd = -1
        except Exception:
            for mm in maps:
                try:
                    mm.close()
                except ValueError:
                    pass
            if fd >= 0:  # the failing slot's own memfd, never in fds
                os.close(fd)
            for opened in fds:
                try:
                    os.close(opened)
                except OSError:
                    pass
            for bid in ids:  # earlier imports back to the daemon
                self._send_release(bid)
            raise
        return SharedImportPool(self, width, height, ids, maps, fds)

    def import_frame(self, handle: FrameHandle) -> int:
        """Import a frame's dma-bufs into the daemon buffer registry (DSP_IMPORT).

        The daemon dups the fds, so the returned id outlives the
        FrameHandle; it lives in the same registry namespace as pool
        buffer ids and stays valid until :meth:`release_buffer` (or the
        daemon's lease watchdog). This is the zero-copy entry point for
        ``InferenceClient.infer(frame, ...)``.
        """
        width, height, handle, fmt = _resolve_source(handle, None)
        return self._import_source(handle, width, height, fmt)

    def release_buffer(self, buffer_id: int) -> None:
        """Free a registry id (import id or pool buffer) with DSP_BUF_RELEASE."""
        if not isinstance(buffer_id, int) or buffer_id <= 0:
            raise DspError(f"invalid buffer id: {buffer_id!r}")
        self._send_release(buffer_id)

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
        return self._import_planes(
            width, height, fmt, num_planes, strides, sizes, handle.fds, timeout_s
        )

    def _import_planes(
        self,
        width: int,
        height: int,
        fmt: str,
        num_planes: int,
        strides: Sequence[int],
        sizes: Sequence[int],
        fds: Sequence[int],
        timeout_s: float = 5.0,
    ) -> int:
        """DSP_IMPORT wire core: geometry + fds out, import id back.

        The daemon classifies the fds itself: real dma-bufs ride the
        zero-copy fd plane path, anything mmap-able (memfds) is mapped
        and rides USERPTR — see ``_import_memfd`` for the client side.
        """
        payload = import_request_bytes(
            width, height, _HAL_PIXEL_FORMAT[fmt], num_planes, strides, sizes
        )
        anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack(f"{num_planes}i", *fds))]
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
                            f"buffer import rejected: {_ERROR_TEXT.get(code, 'error')}",
                            code=code,
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

    def _import_memfd(
        self, wire: bytes, width: int, height: int, fmt: str, stride: int, timeout_s: float = 5.0
    ) -> int:
        """Import client-owned plane bytes as a single-plane buffer.

        The bytes are written to a memfd and imported; the daemon maps it
        (USERPTR planes). This bypasses HAL buffer allocation entirely —
        the transport for formats the device HAL refuses to pool-allocate
        (ARGB32 on some deployed HALs) while the DSP itself accepts
        them. The caller's pixels are copied exactly once into the memfd.
        """
        fd = os.memfd_create("dsp-import")
        try:
            os.ftruncate(fd, len(wire))
            view = memoryview(wire)
            while view:  # os.write may be partial on large buffers
                view = view[os.write(fd, view) :]
            return self._import_planes(
                width, height, fmt, 1, [stride, 0, 0], [len(wire), 0, 0], [fd], timeout_s
            )
        finally:
            os.close(fd)  # the daemon holds its own dup from SCM_RIGHTS

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
        wait: bool = True,
    ) -> int | None:
        """Submit one job; the sync form returns ``elapsed_ms``.

        With ``wait=False`` the job rides ``SubmitDspJobAsync`` and the
        return value is the daemon job id for :class:`PendingDspJob` —
        or ``None`` when the daemon lacks the rpc and transparently fell
        back to the sync submit (the job already ran; a pending object
        built on that is born done).
        """
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
        shadowed = False
        try:
            if wait:
                resp = self._connect().SubmitDspJob(req, timeout=timeout_s)
            else:
                try:
                    resp = self._connect().SubmitDspJobAsync(req, timeout=timeout_s)
                except grpc.RpcError as e:
                    if e.code() != grpc.StatusCode.UNIMPLEMENTED:
                        raise
                    # old daemon: run the job synchronously instead — the
                    # pending handle is born already done
                    shadowed = True
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
        if shadowed:
            return None
        return resp.elapsed_ms if wait else resp.job_id

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
        cpu_fallback: bool = True,
        wait: bool = True,
    ) -> np.ndarray | PendingDspJob:
        """Scale ``src`` to ``(width, height)`` on the DSP.

        ``src`` is a numpy array (copied in) or a keep-fd Frame/FrameHandle
        (imported zero-copy — see the module docstring). ``wait=False``
        returns a :class:`PendingDspJob` instead of the array: the job is
        enqueued without blocking and the buffers stay owned until the
        pending job consumes them.
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
            handed = False
            try:
                job_id = self._submit_job(
                    _OP_RESIZE,
                    source.buffer_id(0),
                    [pools[0].buffer_id(0)],
                    [],
                    interpolation,
                    scaling,
                    priority,
                    timeout_s,
                    wait,
                )
                self.last_used_hw = True
                if not wait:
                    handed = True
                    return PendingDspJob(
                        self, [(pools[0], 0)], job_id, own, timeout_s
                    )
                return pools[0].read(0)
            finally:
                if not handed:
                    self._release_owned(own)
        except _DspUnavailable as e:
            if handle is not None:
                raise DspError(
                    "DSP unavailable with a zero-copy frame source — refusing "
                    "the silent CPU fallback (the frame holds fds, not "
                    "pixels; use frame.to_array() to accept the copy)"
                ) from e
            if not cpu_fallback:
                raise
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
        cpu_fallback: bool = True,
        wait: bool = True,
    ) -> np.ndarray | PendingDspJob:
        """Crop ``(x, y, w, h)`` and scale to the destination size.

        ``wait=False`` returns a :class:`PendingDspJob` (async submit).
        """
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
            handed = False
            try:
                job_id = self._submit_job(
                    _OP_CROP_AND_RESIZE,
                    source.buffer_id(0),
                    [pools[0].buffer_id(0)],
                    [rect],
                    interpolation,
                    scaling,
                    priority,
                    timeout_s,
                    wait,
                )
                self.last_used_hw = True
                if not wait:
                    handed = True
                    return PendingDspJob(
                        self, [(pools[0], 0)], job_id, own, timeout_s
                    )
                return pools[0].read(0)
            finally:
                if not handed:
                    self._release_owned(own)
        except _DspUnavailable as e:
            if handle is not None:
                raise DspError(
                    "DSP unavailable with a zero-copy frame source — refusing "
                    "the silent CPU fallback (the frame holds fds, not "
                    "pixels; use frame.to_array() to accept the copy)"
                ) from e
            if not cpu_fallback:
                raise
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
        cpu_fallback: bool = True,
        wait: bool = True,
    ) -> list[np.ndarray] | PendingDspJob:
        """Crop/resize many windows in one job.

        ``rects`` are ``(x, y, w, h, dst_width, dst_height)``. Destination
        buffers are grouped by geometry (one pool per distinct output
        size); results come back in rect order. ``wait=False`` returns a
        :class:`PendingDspJob` whose ``wait()`` then yields the list.
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
            handed = False
            try:
                slots = {g: 0 for g in order}
                dst_ids = []
                for r in rects:
                    geom = (r[4], r[5])
                    dst_ids.append(pools[order.index(geom)].buffer_id(slots[geom]))
                    slots[geom] += 1
                job_id = self._submit_job(
                    _OP_MULTI_CROP,
                    source.buffer_id(0),
                    dst_ids,
                    rects,
                    interpolation,
                    scaling,
                    priority,
                    timeout_s,
                    wait,
                )
                self.last_used_hw = True
                if not wait:
                    handed = True
                    # slot per rect = its position within its geometry
                    # group (mirrors the dst_ids loop above)
                    slot_of = {g: 0 for g in order}
                    reads = []
                    for r in rects:
                        geom = (r[4], r[5])
                        reads.append((pools[order.index(geom)], slot_of[geom]))
                        slot_of[geom] += 1
                    return PendingDspJob(
                        self, reads, job_id, own, timeout_s, multi=True
                    )
                out, done = [], {g: 0 for g in order}
                for r in rects:
                    geom = (r[4], r[5])
                    pool = pools[order.index(geom)]
                    out.append(pool.read(done[geom]))
                    done[geom] += 1
                return out
            finally:
                if not handed:
                    self._release_owned(own)
        except _DspUnavailable as e:
            if handle is not None:
                raise DspError(
                    "DSP unavailable with a zero-copy frame source — refusing "
                    "the silent CPU fallback (the frame holds fds, not "
                    "pixels; use frame.to_array() to accept the copy)"
                ) from e
            if not cpu_fallback:
                raise
            warnings.warn(
                f"DSP unavailable ({e}); CPU fallback engaged "
                "(client.last_used_hw records the path used)",
                UserWarning,
                stacklevel=3,
            )
            self.last_used_hw = False
            return [_cpu_crop_resize(_as_pixels(src), fmt, r) for r in rects]

    def convert_hw(
        self,
        src: JobSource,
        dst_fmt: str,
        fmt: str | None = None,
        priority: str = "normal",
        timeout_s: float = 5.0,
        src_pool: DspBufferPool | None = None,
        dst_pool: DspBufferPool | None = None,
        cpu_fallback: bool = True,
        wait: bool = True,
    ) -> np.ndarray | PendingDspJob:
        """Convert ``src`` to ``dst_fmt`` (``nv12``/``rgb24``/``gray8``) on
        the DSP, keeping the dimensions.

        The daemon's CONVERT contract (P0): source and destination share
        geometry and differ in format — no rects, exactly one destination
        buffer. Compose with :meth:`resize_hw` when you also need
        scaling, and convert first: NV12 is half the rgb24 bytes, so
        ``CONVERT → RESIZE`` moves less data than the reverse.

        Byte order: ``rgb24`` means RGB order on the wire — BGR pixels
        must be swapped beforehand (or kept on the CPU path via
        ``color.bgr_to_nv12``); the DSP wire has no BGR variant, so
        unswapped BGR comes back with R/B-swapped chroma.

        Supported pairs are firmware-dependent: on hailo15 only
        ``rgb24 <-> nv12`` run on the DSP. Every gray8 pair is refused
        (``HAL_ERR_RESULT``) — the first refusal warns and falls back to
        CPU, and this client remembers: later gray8 pairs go straight to
        the CPU leg, quietly (no repeat warning, ``last_used_hw=False``).

        ``wait=False`` returns a :class:`PendingDspJob` (async submit) —
        only for accepted jobs; a refused pair still returns CPU pixels.
        """
        sw, sh, handle, fmt = _resolve_source(src, fmt)
        _warn_bgr_convert(src)
        _validate_geometry(sw, sh, fmt, "source")
        _validate_geometry(sw, sh, dst_fmt, "destination")
        if dst_fmt == fmt:
            raise DspError(
                f"CONVERT needs differing formats (src is {fmt!r}); dimensions "
                "stay equal — resize_hw scales"
            )
        if src_pool is not None and (src_pool.width, src_pool.height, src_pool.fmt) != (sw, sh, fmt):
            raise DspError(
                f"src_pool is {src_pool.width}x{src_pool.height} {src_pool.fmt}, "
                f"source is {sw}x{sh} {fmt}"
            )
        if dst_pool is not None and (dst_pool.width, dst_pool.height, dst_pool.fmt) != (sw, sh, dst_fmt):
            raise DspError(
                f"dst_pool is {dst_pool.width}x{dst_pool.height} {dst_pool.fmt}, "
                f"job needs {sw}x{sh} {dst_fmt} (CONVERT keeps dims)"
            )
        if "gray8" in (fmt, dst_fmt) and self._gray8_refused:
            # the daemon already refused a gray8 pair this session (the
            # firmware has no gray8 leg) — skip the doomed submit and the
            # repeat warning; zero-copy frames can't take this path (no
            # pixels to convert), see _job_rejected_fallback
            if handle is not None:
                raise DspError(
                    "gray8 conversions are firmware-refused on this device — "
                    "refusing the silent CPU fallback for a zero-copy frame "
                    "source; use frame.to_array() to accept the copy"
                )
            self.last_used_hw = False
            return _cpu_convert(_as_pixels(src), fmt, dst_fmt)
        try:
            # bespoke prep: _prep assumes one fmt for src AND dst pools,
            # but CONVERT needs the dst pool in dst_fmt at the source geometry
            own: list[object] = []
            if handle is not None:
                if src_pool is not None:
                    raise DspError(
                        "src_pool applies to numpy sources; a frame handle imports its own dma-bufs"
                    )
                source = _ImportedSource(self, self._import_source(handle, sw, sh, fmt, timeout_s))
                own.append(source)
            else:
                pool = src_pool if src_pool is not None else self.alloc_buffers(sw, sh, fmt, 1)
                if src_pool is None:
                    own.append(pool)
                pool.write(0, _as_pixels(src))
                source = pool
            out_pool = dst_pool if dst_pool is not None else self.alloc_buffers(sw, sh, dst_fmt, 1)
            if dst_pool is None:
                own.append(out_pool)
            handed = False
            try:
                try:
                    job_id = self._submit_job(
                        _OP_CONVERT_FORMAT,
                        source.buffer_id(0),
                        [out_pool.buffer_id(0)],
                        [],  # CONVERT takes no rects (daemon: wants_rects is False)
                        "bilinear",
                        "stretch",
                        priority,
                        timeout_s,
                        wait,
                    )
                except DspError as e:
                    # the daemon took the job but the hardware refused it —
                    # on hailo15 firmware every gray8 pair lands here
                    return self._job_rejected_fallback(
                        src, fmt, dst_fmt, handle, cpu_fallback, str(e), e
                    )
                self.last_used_hw = True
                if not wait:
                    handed = True
                    return PendingDspJob(
                        self, [(out_pool, 0)], job_id, own, timeout_s
                    )
                return out_pool.read(0)
            finally:
                if not handed:
                    self._release_owned(own)
        except _DspUnavailable as e:
            if handle is not None:
                raise DspError(
                    "DSP unavailable with a zero-copy frame source — refusing "
                    "the silent CPU fallback (the frame holds fds, not "
                    "pixels; use frame.to_array() to accept the copy)"
                ) from e
            if not cpu_fallback:
                raise
            warnings.warn(
                f"DSP unavailable ({e}); CPU fallback engaged "
                "(client.last_used_hw records the path used)",
                UserWarning,
                stacklevel=3,
            )
            self.last_used_hw = False
            return _cpu_convert(_as_pixels(src), fmt, dst_fmt)

    def _job_rejected_fallback(
        self,
        src: JobSource,
        fmt: str,
        dst_fmt: str,
        handle: object,
        cpu_fallback: bool,
        reason: str,
        exc: Exception,
    ) -> np.ndarray:
        """Tail for a submitted-but-refused CONVERT job.

        Mirrors the ``_DspUnavailable`` tail: zero-copy frame sources are
        refused (the fallback would need pixels the frame doesn't hold),
        ``cpu_fallback=False`` re-raises, and the default warns and
        computes on CPU with ``last_used_hw=False`` so health reporting
        stays truthful. A refused gray8 pair also sets the client's
        firmware-gap flag — later gray8 pairs skip the doomed submit.
        """
        if "gray8" in (fmt, dst_fmt):
            self._gray8_refused = True
        if handle is not None:
            raise DspError(
                "DSP rejected the conversion with a zero-copy frame source — "
                "refusing the silent CPU fallback (the frame holds fds, not "
                "pixels; use frame.to_array() to accept the copy)"
            ) from exc
        if not cpu_fallback:
            raise exc
        warnings.warn(
            f"DSP rejected {fmt}->{dst_fmt} ({reason}); CPU fallback engaged "
            "(client.last_used_hw records the path used)",
            UserWarning,
            stacklevel=3,
        )
        self.last_used_hw = False
        return _cpu_convert(_as_pixels(src), fmt, dst_fmt)

    def blend_hw(
        self,
        base: JobSource,
        overlays: Sequence[tuple[np.ndarray, int, int]],
        fmt: str | None = None,
        priority: str = "normal",
        timeout_s: float = 5.0,
        cpu_fallback: bool = True,
        wait: bool = True,
        zero_copy: bool = False,
    ) -> np.ndarray | PendingDspJob:
        """Composite ARGB32 ``overlays`` onto an NV12 ``base`` on the DSP (P1).

        ``overlays`` is a sequence of ``(rgba, x, y)`` — an ``(h, w, 4)``
        uint8 array plus its position on the base; overlays paste 1:1 in
        order (no scaling, later overlays draw over earlier ones). The
        blend runs IN PLACE on a daemon pool copy and the annotated NV12
        array is returned — the input is never modified.

        The base must be NV12 (the vendor op writes NV12 only). Arrays
        are copied in. Keep-fd Frame/FrameHandle bases are **refused
        by default** (``zero_copy=False``): the import->1:1
        RESIZE->BLEND chain has wedged the DSP device-wide until a
        reboot in the field — twice, under media-heap pressure; a
        controlled re-test on a healthy heap passed 11/11, so the
        wedge is state-dependent and the root cause is still open
        (see docs/proposals/dsp-offload.md P2 record). Pass
        ``frame.to_array()`` — the array path is the proven one.
        ``zero_copy=True`` forces the chain for experiments on
        future firmware; nothing about it is guaranteed today.
        Use
        :func:`draw.render_overlay_rgba` to turn detection boxes into a
        minimal overlay canvas, then blend it here; that keeps the
        overlay small and the DSP footprint (quota charges ``base +
        overlays`` megapixels) tight.

        Overlays smaller than 16x16 (the daemon floor) are padded with
        fully transparent pixels to 16 — a semantic no-op. Wire byte
        order is ARGB32 ([A, R, G, B] per pixel); the RGBA->ARGB pack is
        internal. ``wait=False`` submits the blend (and the keep-fd copy
        leg) without blocking and returns a :class:`PendingDspJob`.
        """
        if fmt is not None and fmt != "nv12":
            raise DspError(f"BLEND base must be nv12 (daemon contract), got {fmt!r}")
        # fmt is nv12 by definition — never leave it to shape inference
        # (a 2D nv12 array would ambiguously infer gray8)
        bw, bh, handle, fmt = _resolve_source(base, "nv12")
        if handle is not None and not zero_copy:
            raise DspError(
                "blend_hw refuses keep-fd (frame/handle) bases by default: "
                "the import->resize->blend chain has wedged the DSP "
                "device-wide until reboot in the field (state-dependent, "
                "root cause open). Pass frame.to_array() — the array path "
                "is the proven one — or zero_copy=True to force the chain "
                "at your own risk."
            )
        _validate_geometry(bw, bh, fmt, "base")
        if not overlays:
            raise DspError("BLEND needs at least one overlay")
        if len(overlays) > _MAX_BATCH:
            raise DspError(f"too many overlays ({len(overlays)}); max is {_MAX_BATCH}")

        # validate + pad overlays to the daemon floor before any wire work
        prepared: list[tuple[np.ndarray, int, int]] = []
        for i, (rgba, x, y) in enumerate(overlays):
            if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
                raise DspError(
                    f"overlay {i} must be an (h, w, 4) uint8 rgba array, "
                    f"got shape {getattr(rgba, 'shape', None)} dtype "
                    f"{getattr(rgba, 'dtype', None)}"
                )
            oh, ow = rgba.shape[:2]
            if x < 0 or y < 0 or x + ow > bw or y + oh > bh:
                raise DspError(
                    f"overlay {i} ({ow}x{oh} at ({x},{y})) exceeds the "
                    f"{bw}x{bh} base — clamp or clip before blending"
                )
            if ow < _MIN_DIM or oh < _MIN_DIM:
                # transparent padding composites as a no-op; the rect
                # references the padded dims
                canvas = np.zeros((max(oh, _MIN_DIM), max(ow, _MIN_DIM), 4), np.uint8)
                canvas[:oh, :ow] = rgba
                rgba = canvas
                oh, ow = rgba.shape[:2]
            _validate_geometry(ow, oh, "argb", f"overlay {i}")
            prepared.append((rgba, x, y))

        try:
            own: list[object] = []
            handed = False
            try:
                base_pool = self.alloc_buffers(bw, bh, "nv12", 1)
                own.append(base_pool)
                if handle is not None:
                    # keep-fd base (P2): import the frame zero-copy and let
                    # the DSP copy it into the pool with a 1:1 RESIZE — the
                    # blend then composites in place on that copy. The
                    # daemon rejects imported BLEND *destinations* and the
                    # camera's dma-bufs must never be written, so the copy
                    # is the point; it just never crosses the client. The
                    # RESIZE leg always runs synchronously: execution order
                    # equals submit order (single daemon worker, one
                    # priority queue), but a queued-and-never-waited job
                    # would leak its registry entry — the sync path reaps
                    # its own.
                    imported = _ImportedSource(
                        self, self._import_source(handle, bw, bh, fmt, timeout_s)
                    )
                    own.append(imported)
                    self._submit_job(
                        _OP_RESIZE,
                        imported.buffer_id(0),
                        [base_pool.buffer_id(0)],
                        [],
                        "bilinear",
                        "stretch",
                        priority,
                        timeout_s,
                    )
                else:
                    base_pool.write(0, _as_pixels(base))
                # Overlays travel as memfd imports, not pool allocs:
                # some deployed HALs reject ARGB32 pool allocation
                # (rc=-2809) while the DSP itself blends ARGB
                # fine — the daemon maps the memfd as USERPTR planes.
                ov_srcs: list[_ImportedSource] = []
                for rgba, _x, _y in prepared:
                    ow, oh = rgba.shape[1], rgba.shape[0]
                    # hardware ARGB32 is [A, R, G, B] per pixel in memory
                    wire = np.ascontiguousarray(rgba[:, :, [3, 0, 1, 2]]).tobytes()
                    src = _ImportedSource(
                        self, self._import_memfd(wire, ow, oh, "argb", ow * 4, timeout_s)
                    )
                    own.append(src)
                    ov_srcs.append(src)
                try:
                    job_id = self._submit_job(
                        _OP_BLEND,
                        base_pool.buffer_id(0),
                        [s.buffer_id(0) for s in ov_srcs],
                        [  # placement rect: (x, y, w, h, dst repeats w, h)
                            (x, y, rgba.shape[1], rgba.shape[0],
                             rgba.shape[1], rgba.shape[0])
                            for rgba, x, y in prepared
                        ],
                        "bilinear",
                        "stretch",
                        priority,
                        timeout_s,
                        wait,
                    )
                except DspError as e:
                    if handle is not None:
                        raise DspError(
                            "DSP rejected the blend with a zero-copy frame "
                            "base — refusing the silent CPU fallback (the "
                            "frame holds fds, not pixels; use "
                            "frame.to_array() to accept the copy)"
                        ) from e
                    if not cpu_fallback:
                        raise
                    warnings.warn(
                        f"DSP rejected the blend ({e}); CPU fallback engaged "
                        "(client.last_used_hw records the path used)",
                        UserWarning,
                        stacklevel=3,
                    )
                    self.last_used_hw = False
                    return _cpu_blend(_as_pixels(base), fmt, prepared)
                self.last_used_hw = True
                if not wait:
                    handed = True
                    return PendingDspJob(
                        self, [(base_pool, 0)], job_id, own, timeout_s
                    )
                return base_pool.read(0)  # blend ran in place on the copy
            finally:
                if not handed:
                    self._release_owned(own)
        except _DspUnavailable as e:
            if handle is not None:
                raise DspError(
                    "DSP unavailable with a zero-copy frame base — refusing "
                    "the silent CPU fallback (the frame holds fds, not "
                    "pixels; use frame.to_array() to accept the copy)"
                ) from e
            if not cpu_fallback:
                raise
            warnings.warn(
                f"DSP unavailable ({e}); CPU fallback engaged "
                "(client.last_used_hw records the path used)",
                UserWarning,
                stacklevel=3,
            )
            self.last_used_hw = False
            return _cpu_blend(_as_pixels(base), fmt, prepared)

    def encode_jpeg_hw(
        self,
        src: JobSource | None,
        quality: int = 85,
        fmt: str | None = None,
        timeout_s: float = 5.0,
        src_pool: DspBufferPool | None = None,
        cpu_fallback: bool = True,
        src_buffer_id: int | None = None,
    ) -> bytes:
        """Encode ``src`` as one JPEG frame on the camera-daemon (S-3(a)).

        Unlike the ``*_hw`` job methods this is the daemon's one-shot
        ``EncodeImage`` RPC: the source is pinned in the DSP registry
        (imported zero-copy for keep-fd frames, copied into a pool buffer
        for arrays) and the complete JPEG bytes come back in the response
        — no destination buffer, no read-back. The daemon owns one
        standalone encoder keyed by ``(width, height, format, quality)``
        and recreates it when that key changes, so alternating qualities
        or geometries re-spins the encoder (first frame after a change
        pays the pipeline start-up).

        Despite the name, the encoder is N-threaded libjpeg on the DSP
        core behind a GStreamer dispatch — hailo15 has no dedicated JPEG
        block. The win is central encode + zero-copy input, not raw speed;
        keep it out of tight per-frame loops that a CPU encode already
        serves (see docs/proposals/sdk-hardware-routing.md S-3).

        ``quality`` is 1..100. Inputs are ``rgb24``/``nv12`` arrays and
        keep-fd frames (the daemon normalizes RGB through its DSP convert
        — the encoder pipeline negotiates NV12 only). gray8 arrays
        up-convert to rgb24 client-side (R=G=B=gray) and ride the same
        hardware leg; gray8 keep-fd frames raise instead of silently
        copying — accept the copy yourself with ``frame.to_array()``.

        ``src_buffer_id`` (with ``src=None``) encodes straight from a
        daemon-side buffer — the zero-copy chain tail:
        ``blend_hw(..., wait=False).wait_result().buffer_id`` lands here
        and the annotated frame becomes JPEG without a single read-back.
        There is no CPU fallback on that leg (the client holds no
        pixels); unavailability raises.
        """
        if src_buffer_id is not None:
            if src is not None:
                raise DspError("pass either src or src_buffer_id, not both")
            if not 1 <= int(quality) <= 100:
                raise DspError(f"quality must be 1..100, got {quality}")
            try:
                jpeg = self._encode_rpc(int(src_buffer_id), int(quality), timeout_s)
            except _DspUnavailable as e:
                raise DspError(
                    "EncodeImage unavailable for a daemon-side buffer — no "
                    "client pixels to fall back on; pass the array instead"
                ) from e
            self.last_used_hw = True
            return jpeg
        sw, sh, handle, fmt = _resolve_source(src, fmt)
        _validate_geometry(sw, sh, fmt, "source")
        if not 1 <= int(quality) <= 100:
            raise DspError(f"quality must be 1..100, got {quality}")
        if fmt == "gray8":
            if handle is not None:
                raise DspError(
                    "gray8 frames cannot ride the hardware jpeg encoder "
                    "without a copy (the daemon feeds nv12/rgb24); "
                    "use frame.to_array() and pass the array"
                )
            # no hardware gray leg: replicate to rgb24 (R=G=B=gray) and
            # let the daemon's DSP convert take it from there
            src = _cpu_convert(_as_pixels(src), "gray8", "rgb24")
            fmt = "rgb24"
        if src_pool is not None and (src_pool.width, src_pool.height, src_pool.fmt) != (sw, sh, fmt):
            raise DspError(
                f"src_pool is {src_pool.width}x{src_pool.height} {src_pool.fmt}, "
                f"source is {sw}x{sh} {fmt}"
            )
        try:
            # bespoke prep (no dst pool — the JPEG rides the response)
            own: list[object] = []
            if handle is not None:
                if src_pool is not None:
                    raise DspError(
                        "src_pool applies to numpy sources; a frame handle imports its own dma-bufs"
                    )
                source = _ImportedSource(self, self._import_source(handle, sw, sh, fmt, timeout_s))
                own.append(source)
            else:
                pool = src_pool if src_pool is not None else self.alloc_buffers(sw, sh, fmt, 1)
                if src_pool is None:
                    own.append(pool)
                pool.write(0, _as_pixels(src))
                source = pool
            try:
                jpeg = self._encode_rpc(source.buffer_id(0), int(quality), timeout_s)
                self.last_used_hw = True
                return jpeg
            finally:
                self._release_owned(own)
        except _DspUnavailable as e:
            if handle is not None:
                raise DspError(
                    "EncodeImage unavailable with a zero-copy frame source — refusing "
                    "the silent CPU fallback (the frame holds fds, not "
                    "pixels; use frame.to_array() to accept the copy)"
                ) from e
            if not cpu_fallback:
                raise
            warnings.warn(
                f"EncodeImage unavailable ({e}); CPU fallback engaged "
                "(client.last_used_hw records the path used)",
                UserWarning,
                stacklevel=3,
            )
            self.last_used_hw = False
            pixels = _as_pixels(src)
            if fmt != "rgb24":  # _cpu_convert refuses identical formats
                pixels = _cpu_convert(pixels, fmt, "rgb24")
            return _encode_jpeg(pixels, int(quality))

    def _encode_rpc(self, src_id: int, quality: int, timeout_s: float) -> bytes:
        """One EncodeImage round-trip; raises _DspUnavailable when absent."""
        req = camera_pb2.EncodeImageRequest(src_buffer_id=src_id, quality=quality)
        try:
            resp = self._connect().EncodeImage(req, timeout=timeout_s)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                raise _DspUnavailable("EncodeImage not in daemon") from e
            raise DspError(f"EncodeImage rpc failed: {e}") from e
        if not resp.success:
            if resp.error_code == DSP_SERVICE_UNAVAILABLE:
                raise _DspUnavailable("dsp service not running")
            raise DspError(
                f"jpeg encode failed: {resp.message or _ERROR_TEXT.get(resp.error_code)}",
                code=resp.error_code,
            )
        if not resp.jpeg:
            raise DspError("EncodeImage returned success with no jpeg payload")
        return bytes(resp.jpeg)

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
