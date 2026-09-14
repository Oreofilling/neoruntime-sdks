"""
App frame injection front end (PushFrame P0-P2).

FramePublisher composes the two halves of the output write path that the
daemon exposes: a daemon-side DSP buffer pool (geometry-pinned dma-bufs,
NV12 or ARGB32) and the CameraControl PushFrame RPCs (metadata only —
the buffer crosses as a DSP-registry id, never as a raw fd number).
"""

from __future__ import annotations

import logging
import time
from itertools import repeat

import numpy as np

from .camera import CameraClient
from .camera_types import InjectionResult
from .dsp import DspClient

__all__ = ["FramePublisher"]

logger = logging.getLogger(__name__)

_MODES = ("replace", "overlay")


class FramePublisher:
    """Publish frames into a stream's encoder feed (REPLACE / OVERLAY).

    The publisher resolves the target stream's geometry from
    ``GetStreamStatus`` at construction and rejects wrong-shaped input
    client-side, so the daemon never sees a doomed request.

    ``mode="replace"`` (P0): the pool matches the stream's encode
    resolution and the daemon composites each pushed frame over the
    whole encode output. NV12 only — the RGB escape hatch is
    :meth:`publish_rgb`.

    ``mode="overlay"`` (P2): the pool matches ``inset`` (default: one
    quarter of the stream, even-rounded) and each frame is composed at
    ``dest`` inside the live picture. ``fmt="nv12"`` insets paste
    opaquely; ``fmt="argb"`` insets alpha-blend on the CPU (wire byte
    order [A, R, G, B]). Use :func:`neoruntime_ipc_sdk.camera.pip_dest`
    to place a corner-anchored picture-in-picture window.

    Frames round-robin across a small daemon-side pool: each
    :meth:`publish` writes one slot and pushes that slot's registry id.
    On daemons that report the write-lease set (Fix-1) a slot is
    rewritten only after the daemon acknowledges it is no longer
    reading it — every PushFrame response carries the ids still queued
    or mid-bake, and with all slots busy :meth:`publish` waits for the
    bake frontier (up to ``lease_timeout_s``) and raises instead of
    silently tearing pixels. On older daemons the publisher falls back
    to the legacy blind rotation (a warning is logged once); give such
    deployments ``pool_depth >= 4`` and publish at stream rate. The
    daemon-side queue is cap-3 drop-oldest, so publishing never
    backpressures the encoder; overflow shows up in
    :meth:`CameraClient.injection_status` as ``frames_dropped``.

    To show the injected stream in a browser, no server is needed:
    :func:`neoruntime_ipc_sdk.web.platform_stream_url` composes the
    gateway URL of the (now app-affected) platform stream.

    Usage::

        with FramePublisher(cam, dsp, stream_id="sub") as pub:
            pub.publish(frame)          # uint8 (h*3//2, w) ndarray or tight NV12 bytes
            ...
            pub.publish_eos()           # flush; ISP path restores at the next IDR

        x, y = pip_dest(1920, 1080, 480, 270, "bottom-right", 32)
        with FramePublisher(cam, dsp, stream_id="main", mode="overlay",
                            fmt="argb", inset=(480, 270), dest=(x, y)) as pub:
            pub.publish(badge_rgba)     # (270, 480, 4) uint8, [A, R, G, B]
    """

    def __init__(
        self,
        camera: CameraClient,
        dsp: DspClient,
        stream_id: str = "sub",
        pool_depth: int = 4,
        *,
        mode: str = "replace",
        fmt: str = "nv12",
        inset: tuple[int, int] | None = None,
        dest: tuple[int, int] = (0, 0),
        session_id: str = "",
        lease_timeout_s: float = 5.0,
    ):
        if pool_depth < 1:
            raise ValueError("pool_depth must be >= 1")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        streams = camera.get_stream_status()
        match = [s for s in streams if s.stream_id == stream_id]
        if not match:
            raise RuntimeError(
                f"no live stream named {stream_id!r}; known: "
                f"{[s.stream_id for s in streams]}"
            )
        st = match[0]
        if not st.has_encoder:
            raise RuntimeError(f"stream {stream_id!r} has no encoder")
        if st.width <= 0 or st.height <= 0 or (st.width & 1) or (st.height & 1):
            raise ValueError(
                f"stream {stream_id!r} geometry {st.width}x{st.height} "
                "must be nonzero and even"
            )

        if mode == "replace":
            if fmt != "nv12":
                raise ValueError(
                    'REPLACE consumes NV12 only (fmt="nv12"); '
                    "RGB sources go through publish_rgb"
                )
            pool_w, pool_h = st.width, st.height
            dest_x = dest_y = 0
        else:
            if fmt not in ("nv12", "argb"):
                raise ValueError(
                    f"overlay fmt must be 'nv12' (opaque paste) or 'argb' "
                    f"(alpha blend), got {fmt!r}"
                )
            if inset is None:
                # PIP default: one quarter of the stream, even-aligned.
                pool_w = (st.width // 4) & ~1
                pool_h = (st.height // 4) & ~1
            else:
                pool_w, pool_h = inset
            if pool_w <= 0 or pool_h <= 0 or (pool_w & 1) or (pool_h & 1):
                raise ValueError(
                    f"inset {pool_w}x{pool_h} must be nonzero and even "
                    "(the bake walks 2x2 chroma blocks)"
                )
            dest_x, dest_y = dest
            if (dest_x & 1) or (dest_y & 1):
                raise ValueError(
                    f"dest ({dest_x}, {dest_y}) must be even — NV12 chroma "
                    "is 2x2 subsampled"
                )
            if dest_x + pool_w > st.width or dest_y + pool_h > st.height:
                raise ValueError(
                    f"inset {pool_w}x{pool_h} at ({dest_x}, {dest_y}) "
                    f"exceeds stream {st.width}x{st.height}"
                )

        # Allocating last: every validation above runs before the pools
        # exist, so a failed constructor leaks nothing by construction.
        self._camera = camera
        self._dsp = dsp
        self._stream_id = stream_id
        self._session_id = session_id  # lifecycle tag on every request (P2-13)
        self._mode = mode
        self._fmt = fmt
        self._width = pool_w
        self._height = pool_h
        self._dest_x = dest_x
        self._dest_y = dest_y
        if fmt == "argb":
            # The deployed HAL refuses ARGB32 pool allocation (wire OOM):
            # alpha overlays ride the shared memfd-import ring instead —
            # the daemon maps the same pages the bake reads.
            self._pool = dsp.import_shared_buffers(pool_w, pool_h, fmt, pool_depth)
        else:
            self._pool = dsp.alloc_buffers(pool_w, pool_h, fmt, pool_depth)
        self._pool_depth = pool_depth
        self._slot = 0
        # Write-lease state (Fix-1). A daemon that reports the in-flight
        # buffer set gets slot-level backpressure: publish waits for the
        # bake frontier to release a slot instead of blindly rotating
        # (the 11/22/33 -> 33/22/33 tear of P1-1). The capability probe
        # is best-effort — any failure (daemon restarting, ancient build)
        # degrades to the legacy depth-paced rotation with one warning.
        self._lease_timeout_s = lease_timeout_s
        self._in_flight: set[int] = set()
        self._pool_ids = [self._pool.buffer_id(i) for i in range(pool_depth)]
        self._lease = False
        try:
            self._lease = camera.injection_status().reports_in_flight_buffers
        except Exception:
            self._lease = False
        if not self._lease:
            logger.warning(
                "FramePublisher(%s): daemon does not report the write-lease "
                "set; falling back to blind slot rotation — publish at "
                "stream rate with pool_depth >= 4 or a queued frame may "
                "bake with newer pixels",
                stream_id,
            )
        # Lazily created by publish_rgb (RGB->NV12 staging, REPLACE only).
        self._rgb_pool = None
        self._nv12_stage = None
        self._closed = False
        # Lifecycle bookkeeping for __exit__'s best-effort EOS: whether
        # any frame went out, and whether an EOS has been sent already.
        self._published_any = False
        self._eos_sent = False

    # -- geometry -----------------------------------------------------------

    @property
    def width(self) -> int:
        """Pool width (stream encode width for REPLACE, inset width for OVERLAY)."""
        return self._width

    @property
    def height(self) -> int:
        """Pool height (stream encode height for REPLACE, inset height for OVERLAY)."""
        return self._height

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def session_id(self) -> str:
        """Lifecycle tag stamped on every push ("" = untagged, P2-13)."""
        return self._session_id

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def dest(self) -> tuple[int, int]:
        """Paste origin for OVERLAY (always (0, 0) for REPLACE)."""
        return (self._dest_x, self._dest_y)

    # -- write-lease slots (Fix-1) -------------------------------------------

    @property
    def lease_mode(self) -> bool:
        """True when the daemon reports the write-lease set (slot-aware
        publishing); False on legacy daemons (blind rotation)."""
        return self._lease

    def _refresh_in_flight(self) -> None:
        """Re-read the daemon's write-lease snapshot (GetInjectionStatus).

        Used only where no fresher source exists: pacing the
        client-streaming generator (its single response arrives after the
        last frame) and the initial wait inside :meth:`_acquire_slot`.
        Unary :meth:`publish` updates from each PushFrame response.
        """
        st = self._camera.injection_status()
        if st.reports_in_flight_buffers:
            self._in_flight = set(st.in_flight_buffer_ids)

    def _acquire_slot(self, ids: list[int] | None = None) -> int:
        """Return a pool slot index whose buffer the daemon has released.

        A slot is writable only when its registry id is absent from the
        daemon's queued+mid-bake set. With every slot busy, poll
        :meth:`injection_status` — the bake frontier frees slots at
        stream fps — until one appears or ``lease_timeout_s`` elapses;
        then raise rather than silently rewriting pixels the daemon may
        still bake. ``ids`` defaults to the round-robin pool's ids;
        :meth:`publish_rgb` passes its one-deep NV12 staging id.
        """
        if ids is None:
            ids = self._pool_ids
        depth = len(ids)
        deadline = time.monotonic() + self._lease_timeout_s
        while True:
            for off in range(depth):
                slot = (self._slot + off) % depth
                if ids[slot] not in self._in_flight:
                    self._slot = (slot + 1) % depth
                    return slot
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"FramePublisher({self._stream_id!r}): no free pool "
                    f"slot after {self._lease_timeout_s}s — the daemon "
                    f"still holds {sorted(self._in_flight) or 'all slots'} "
                    "in flight; is the encoder feed consuming frames?"
                )
            self._refresh_in_flight()
            time.sleep(0.01)

    # -- publish ------------------------------------------------------------

    def publish(self, frame, pts_ns: int = 0) -> InjectionResult:
        """Write one frame to the next pool slot and push its id.

        NV12 pools take a uint8 ndarray shaped ``(height*3//2, width)``
        or tight NV12 bytes; ARGB32 pools take a uint8 ndarray shaped
        ``(height, width, 4)`` (wire byte order [A, R, G, B]) or tight
        bytes of ``width * height * 4``. Returns the daemon's
        :class:`InjectionResult`; raises on RPC failure (the pool slot
        was still written — the daemon-side drop-oldest queue absorbs
        it, see ``frames_dropped``).

        ``pts_ns`` paces the bake (device CLOCK_MONOTONIC, the same
        clock as frame timestamps): 0 = due immediately, a future value
        holds the frame until the encoder frontier passes it.

        Lease mode (daemon reporting ``in_flight_buffer_ids``): the slot
        is chosen among ids the daemon has released and the response's
        lease snapshot replaces the local in-flight set, so a slot is
        never rewritten while its pixels may still be queued or mid-bake.
        """
        arr = self._to_pool_array(frame)
        if self._closed:
            raise RuntimeError("FramePublisher is closed")
        if self._lease:
            slot = self._acquire_slot()
        else:
            slot = self._slot
            self._slot = (slot + 1) % self._pool_depth
        self._pool.write(slot, arr)
        self._published_any = True
        res = self._camera.push_frame(
            buffer_id=self._pool.buffer_id(slot),
            width=self._width,
            height=self._height,
            stride=self._pool.strides[0],
            mode=self._mode,
            pts_ns=pts_ns,
            dest_x=self._dest_x,
            dest_y=self._dest_y,
            stream_id=self._stream_id,
            end_of_stream=False,
            session_id=self._session_id,
        )
        if self._lease:
            self._in_flight = set(res.in_flight_buffer_ids)
        return res

    def publish_stream(
        self,
        frames,
        pts_ns=None,
        end_with_eos: bool = False,
        timeout_s: float | None = None,
    ) -> InjectionResult:
        """Push a run of frames over one client-streaming RPC.

        ``frames`` is an iterable of the same payloads :meth:`publish`
        accepts; ``pts_ns`` is an optional parallel iterable of due
        timestamps (missing values default to 0). All requests ride a
        single PushFrameStream call: the daemon applies per-request
        validation and the first rejection ends the stream — raised
        here as ``RuntimeError`` with the accepted-frame count in the
        text. With ``end_with_eos=True`` a final EOS request closes the
        session before the RPC returns; otherwise the stream ends in a
        clean half-close that leaves the session open (close it later
        via :meth:`publish_eos`).

        Slots recycle through the same pool as :meth:`publish`: the
        generator writes each frame into the next slot as gRPC pulls
        it. Client-streaming has no per-frame response, so in lease
        mode the generator blocks on :meth:`_acquire_slot`, whose
        freshness comes from :meth:`injection_status` polling — the
        single-RPC efficiency is kept, only the write pace adapts. On
        legacy daemons pass ``pool_depth >= 4`` and push at stream
        rate: a shallower pool rewrites a slot whose frame may still
        be queued (the queue then bakes the newer pixels).
        """
        if self._closed:
            raise RuntimeError("FramePublisher is closed")

        def _requests():
            pts_iter = iter(pts_ns) if pts_ns is not None else repeat(0)
            for frame in frames:
                if self._closed:
                    raise RuntimeError("FramePublisher closed mid-stream")
                try:
                    due = next(pts_iter)
                except StopIteration:
                    due = 0
                arr = self._to_pool_array(frame)
                if self._lease:
                    slot = self._acquire_slot()
                    # No per-frame response exists on this transport, so
                    # the lease is marked locally at yield time (the
                    # request is on its way to the daemon's queue) and a
                    # later status poll clears it once the daemon
                    # reports the id released.
                    self._in_flight.add(self._pool_ids[slot])
                else:
                    slot = self._slot
                    self._slot = (slot + 1) % self._pool_depth
                self._pool.write(slot, arr)
                self._published_any = True
                yield {
                    "buffer_id": self._pool.buffer_id(slot),
                    "width": self._width,
                    "height": self._height,
                    "stride": self._pool.strides[0],
                    "mode": self._mode,
                    "pts_ns": due,
                    "dest_x": self._dest_x,
                    "dest_y": self._dest_y,
                    "stream_id": self._stream_id,
                    "session_id": self._session_id,
                }
            if end_with_eos:
                yield {
                    "buffer_id": 0,
                    "width": self._width,
                    "height": self._height,
                    "stride": self._pool.strides[0],
                    "end_of_stream": True,
                    "session_id": self._session_id,
                }
                self._eos_sent = True

        return self._camera.push_frame_stream(_requests(), timeout_s=timeout_s)

    def publish_rgb(self, rgb, pts_ns: int = 0) -> InjectionResult:
        """Convert an RGB frame to NV12 and REPLACE the stream with it.

        The daemon's REPLACE contract is NV12-only, so the RGB path is
        composed here on the SDK side: one DMA-side ``rgb24`` staging
        buffer, a hardware ``CONVERT`` (:meth:`DspClient.convert_hw`)
        into a one-deep NV12 staging pool, then a push of that pool's
        buffer id. ``rgb`` is a uint8 ``(height, width, 3)`` ndarray
        (RGB order — swap BGR beforehand) or tight bytes. If the DSP
        refuses or is unavailable the convert falls back to CPU and the
        result is copied into the staging pool, so the push is always
        well-defined (``dsp.last_used_hw`` records the path taken).

        The staging slot is reused every call: lease mode blocks until
        the daemon has released the staging NV12 (the pushed id; the rgb
        staging is only ever read by the convert itself), legacy mode
        relies on the caller pacing at stream rate — faster publishing
        lets a still-queued frame bake with the next frame's pixels.
        """
        if self._mode != "replace":
            raise RuntimeError("publish_rgb targets REPLACE publishers only")
        if self._closed:
            raise RuntimeError("FramePublisher is closed")
        w, h = self._width, self._height
        if isinstance(rgb, np.ndarray):
            arr = rgb
        elif isinstance(rgb, (bytes, bytearray, memoryview)):
            arr = np.frombuffer(bytes(rgb), dtype=np.uint8).reshape(h, w, 3)
        else:
            raise TypeError(
                f"rgb must be a uint8 ndarray or tight RGB bytes, got {type(rgb)!r}"
            )
        if self._rgb_pool is None:
            self._rgb_pool = self._dsp.alloc_buffers(w, h, "rgb24", 1)
            self._nv12_stage = self._dsp.alloc_buffers(w, h, "nv12", 1)
        elif self._lease:
            self._acquire_slot([self._nv12_stage.buffer_id(0)])
        out = self._dsp.convert_hw(
            src=arr,
            dst_fmt="nv12",
            src_pool=self._rgb_pool,
            dst_pool=self._nv12_stage,
        )
        if not self._dsp.last_used_hw:
            # CPU fallback computed pixels without touching the pool.
            self._nv12_stage.write(0, out)
        self._published_any = True
        res = self._camera.push_frame(
            buffer_id=self._nv12_stage.buffer_id(0),
            width=w,
            height=h,
            stride=self._nv12_stage.strides[0],
            mode="replace",
            pts_ns=pts_ns,
            dest_x=0,
            dest_y=0,
            stream_id=self._stream_id,
            end_of_stream=False,
            session_id=self._session_id,
        )
        if self._lease:
            self._in_flight = set(res.in_flight_buffer_ids)
        return res

    def publish_eos(self) -> None:
        """Flush the injection session (``buffer_id`` 0 + end_of_stream).

        The daemon drops the session and its queue; the pure ISP path
        restores at the next IDR. Sent automatically — best effort — by
        the context-manager exit when frames were published and no EOS
        went out; failure there is a warning, not an error.
        """
        if self._closed:
            raise RuntimeError("FramePublisher is closed")
        self._camera.push_frame(
            buffer_id=0,
            width=self._width,
            height=self._height,
            stride=self._pool.strides[0],
            end_of_stream=True,
            session_id=self._session_id,
        )
        self._eos_sent = True

    # -- conversion ---------------------------------------------------------

    def _to_pool_array(self, frame) -> np.ndarray:
        if self._fmt == "nv12":
            rows = self._height * 3 // 2
            expected_shape = (rows, self._width)
        else:  # argb
            expected_shape = (self._height, self._width, 4)
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                raise ValueError(
                    f"{self._fmt} ndarray dtype must be uint8, got {frame.dtype}"
                )
            if frame.shape != expected_shape:
                raise ValueError(
                    f"{self._fmt} ndarray shape must be {expected_shape}, got {frame.shape}"
                )
            return frame
        if isinstance(frame, (bytes, bytearray, memoryview)):
            expected_len = int(np.prod(expected_shape))
            data = bytes(frame)
            if len(data) != expected_len:
                raise ValueError(
                    f"tight {self._fmt} bytes must be {expected_len} bytes "
                    f"({self._width}x{self._height} {self._fmt}), got {len(data)}"
                )
            return np.frombuffer(data, dtype=np.uint8).reshape(expected_shape)
        raise TypeError(
            f"frame must be a uint8 ndarray or tight {self._fmt} bytes, got {type(frame)!r}"
        )

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the buffer pools (idempotent; sends no EOS itself).

        The context-manager exit sends a best-effort EOS first when frames
        were published without one; a bare ``close()`` never does —
        callers needing explicit control call :meth:`publish_eos`.
        """
        if not self._closed:
            self._closed = True
            self._pool.release()
            if self._rgb_pool is not None:
                self._rgb_pool.release()
            if self._nv12_stage is not None:
                self._nv12_stage.release()

    def __enter__(self) -> FramePublisher:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Best-effort EOS: the daemon keeps a clean half-closed session
        # open, so a with-block that published frames but never sent EOS
        # would hold the session (and its queue) until the connection
        # drops. A failure here is a warning — the pools release anyway.
        if not self._closed and self._published_any and not self._eos_sent:
            try:
                self.publish_eos()
            except Exception:
                logger.warning(
                    "FramePublisher.__exit__: best-effort EOS failed; the "
                    "injection session stays open until the connection drops",
                    exc_info=True,
                )
        self.close()
        return False
