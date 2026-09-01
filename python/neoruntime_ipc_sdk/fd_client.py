"""Zero-copy media client over DMA-BUF fd passing (camera.sock fd publisher)."""

from __future__ import annotations

import logging
import mmap
import os
import socket as _socket
import struct
import threading
import time
import weakref
from typing import Callable, Iterator

import numpy as np

from ._transport import recvmsg_with_fds as _recvmsg_with_fds
from ._transport import sendmsg_plain as _sendmsg_plain
from .encoded import EncodedStreamClient
from .frame import (
    _DMA_BUF_SYNC_END,
    _DMA_BUF_SYNC_READ,
    _DMA_BUF_SYNC_START,
    PIXEL_FORMAT_NAMES,
    Frame,
    FrameHandle,
    _decode_raw,
    _dma_buf_sync,
)

logger = logging.getLogger("neoruntime_ipc_sdk.fd_client")

__all__ = ["FdMediaClient"]


# ---------------------------------------------------------------------------
# FD Protocol constants (must match fd_protocol.h)
# ---------------------------------------------------------------------------

_FD_PUB_MSG_SUBSCRIBE = 1
_FD_PUB_MSG_UNSUBSCRIBE = 2
_FD_PUB_MSG_FRAME = 3
_FD_PUB_MSG_RELEASE = 4
_FD_PUB_MSG_OK = 5
_FD_PUB_MSG_ERROR = 6

_FD_PUB_MAX_STREAM_NAME = 64
_FD_PUB_MAX_FDS = 3
_FD_PUB_PROTOCOL_VERSION = 1

# struct FdPubMsgHeader { uint32 type; uint32 size; }
_HDR_FMT = "<II"
_HDR_SIZE = struct.calcsize(_HDR_FMT)

# struct FdPubSubscribeMsg { header(8) + uint32 version + char[64] stream_name }
_SUB_FMT = "<II I 64s"
_SUB_SIZE = struct.calcsize(_SUB_FMT)

# struct FdPubFrameMsg (aarch64 pads to 8-byte alignment: 76 data + 4 padding = 80)
_FRAME_FMT = "<II QQQ IIII 3I 3I I 4x"
_FRAME_SIZE = struct.calcsize(_FRAME_FMT)

# struct FdPubReleaseMsg { header(8) + uint64 frame_id }
_REL_FMT = "<II Q"
_REL_SIZE = struct.calcsize(_REL_FMT)

# struct FdPubResponseMsg { header(8) + int32 code }
_RESP_FMT = "<II i"
_RESP_SIZE = struct.calcsize(_RESP_FMT)


class FdMediaClient:
    """Zero-copy media client using DMA-BUF FD passing over Unix Domain Socket."""

    def __init__(self, socket_path: str | None = None):
        if socket_path is None:
            socket_path = os.getenv("CAMERA_SOCK_PATH", "/run/aipc/camera.sock")
        self.socket_path = socket_path
        self._streams: dict[str, _socket.socket] = {}
        self._lock = threading.Lock()
        # Retained keep-fd handles. WeakSet: tracking without extending
        # lifetime — a dropped Frame is GC-released back to the daemon.
        self._retained: weakref.WeakSet[FrameHandle] = weakref.WeakSet()

    def _connect_stream(self, stream_id: str) -> _socket.socket:
        logger.info("FdMediaClient: connecting to %s for stream '%s'", self.socket_path, stream_id)

        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        logger.info("FdMediaClient: socket fd=%d connected", sock.fileno())

        name_bytes = stream_id.encode("utf-8")[: _FD_PUB_MAX_STREAM_NAME - 1]
        name_padded = name_bytes.ljust(_FD_PUB_MAX_STREAM_NAME, b"\x00")
        sub_msg = struct.pack(
            _SUB_FMT, _FD_PUB_MSG_SUBSCRIBE, _SUB_SIZE, _FD_PUB_PROTOCOL_VERSION, name_padded
        )
        _sendmsg_plain(sock, sub_msg)

        resp_data = sock.recv(_RESP_SIZE)
        if len(resp_data) < _RESP_SIZE:
            sock.close()
            raise ConnectionError(f"FdMediaClient: no response for stream '{stream_id}'")

        msg_type, msg_size, code = struct.unpack(_RESP_FMT, resp_data[:_RESP_SIZE])
        if msg_type != _FD_PUB_MSG_OK:
            sock.close()
            raise ConnectionError(
                f"FdMediaClient: subscribe rejected for '{stream_id}' (code={code})"
            )

        logger.info("FdMediaClient: subscribed to '%s' successfully", stream_id)
        return sock

    def _get_sock(self, stream_id: str) -> _socket.socket:
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = self._connect_stream(stream_id)
            return self._streams[stream_id]

    def _release_frame(self, sock: _socket.socket, frame_id: int) -> None:
        rel = struct.pack(_REL_FMT, _FD_PUB_MSG_RELEASE, _REL_SIZE, frame_id)
        try:
            _sendmsg_plain(sock, rel)
        except OSError:
            pass

    def _recv_frame(self, sock: _socket.socket, keep_fd: bool = False) -> Frame | None:
        skipped = 0
        eof_count = 0
        for _attempt in range(32):
            data, fds = _recvmsg_with_fds(sock, _FRAME_SIZE)

            # Detect EOF (server closed connection)
            if len(data) == 0:
                eof_count += 1
                if eof_count >= 3:
                    raise ConnectionError("FdMediaClient: socket EOF (server closed connection)")
                continue

            if len(data) < _FRAME_SIZE:
                for fd in fds:
                    os.close(fd)
                skipped += 1
                continue

            values = struct.unpack(_FRAME_FMT, data[:_FRAME_SIZE])
            msg_type = values[0]
            if msg_type != _FD_PUB_MSG_FRAME:
                for fd in fds:
                    os.close(fd)
                skipped += 1
                continue

            break
        else:
            if skipped > 0:
                logger.warning("FdMediaClient: skipped %d non-frame messages, giving up", skipped)
            return None

        if skipped > 0:
            logger.debug("FdMediaClient: skipped %d non-frame messages before frame", skipped)

        frame_id = values[2]
        timestamp_ns = values[3]
        sequence = values[4]
        width = values[5]
        height = values[6]
        fmt_code = values[7]
        num_planes = values[8]
        strides = values[9:12]
        sizes = values[12:15]
        _num_fds_expected = values[15]

        fmt_name = PIXEL_FORMAT_NAMES.get(fmt_code, f"UNKNOWN({fmt_code})")

        if not fds:
            self._release_frame(sock, frame_id)
            return None

        if keep_fd:

            def _on_release(h: FrameHandle) -> None:
                self._retained.discard(h)
                self._release_frame(sock, h.frame_id)

            handle = FrameHandle(
                fds=fds,
                strides=strides,
                plane_sizes=sizes,
                frame_id=frame_id,
                on_release=_on_release,
                width=width,
                height=height,
                format=fmt_name,
            )
            self._retained.add(handle)
            logger.debug(
                "FdMediaClient: retained frame seq=%d %dx%d %s (frame_id=%d)",
                sequence,
                width,
                height,
                fmt_name,
                frame_id,
            )
            return Frame(
                sequence=sequence,
                timestamp_ns=timestamp_ns,
                width=width,
                height=height,
                format=fmt_name,
                image=None,
                handle=handle,
            )

        # Copy path: mmap each dma-buf plane (fenced per HAL-3), copy to
        # numpy, close the fds, then hand the buffer back to the daemon.
        try:
            # DMA-BUF fds must be mmapped per-plane using the fd's actual size,
            # not the protocol-reported plane size (which excludes alignment padding).
            planes = []
            for i in range(min(num_planes, len(fds))):
                fd = fds[i]
                _dma_buf_sync(fd, _DMA_BUF_SYNC_READ | _DMA_BUF_SYNC_START)
                actual_size = os.fstat(fd).st_size
                buf = mmap.mmap(fd, actual_size, access=mmap.ACCESS_READ)
                plane_data = np.frombuffer(buf, dtype=np.uint8)[: sizes[i]].copy()
                buf.close()
                _dma_buf_sync(fd, _DMA_BUF_SYNC_READ | _DMA_BUF_SYNC_END)
                planes.append(plane_data)
            raw = np.concatenate(planes) if len(planes) > 1 else planes[0]
        finally:
            for fd in fds:
                os.close(fd)

        self._release_frame(sock, frame_id)

        logger.debug(
            "FdMediaClient: frame seq=%d %dx%d %s released (frame_id=%d)",
            sequence,
            width,
            height,
            fmt_name,
            frame_id,
        )

        image = _decode_raw(raw, width, height, fmt_name)
        return Frame(
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            width=width,
            height=height,
            format=fmt_name,
            image=image,
        )

    def get_frame(
        self, stream_id: str, timeout_ms: int = 5000, *, keep_fd: bool = False
    ) -> Frame | None:
        """Receive one frame.

        With ``keep_fd=True`` the frame's dma-buf fds are retained
        (zero-copy handoff; see :class:`FrameHandle`) instead of copied,
        and the daemon-side buffer release is deferred until
        ``frame.release()`` / GC / client close.
        """
        sock = self._get_sock(stream_id)
        sock.settimeout(timeout_ms / 1000.0)
        try:
            return self._recv_frame(sock, keep_fd=keep_fd)
        except _socket.timeout:
            return None
        except (ConnectionError, OSError):
            # Stale socket — clear cache so next call reconnects
            with self._lock:
                old = self._streams.pop(stream_id, None)
                if old:
                    try:
                        old.close()
                    except OSError:
                        pass
            raise

    def subscribe_raw(
        self, stream_id: str, skip_frames: bool = True, keep_fd: bool = False
    ) -> Iterator[Frame]:
        sock = self._get_sock(stream_id)
        sock.settimeout(5.0)
        while True:
            try:
                frame = self._recv_frame(sock, keep_fd=keep_fd)
                if frame is not None:
                    yield frame
            except _socket.timeout:
                continue
            except (ConnectionError, OSError):
                with self._lock:
                    self._streams.pop(stream_id, None)
                try:
                    sock.close()
                except OSError:
                    pass
                time.sleep(0.5)
                sock = self._get_sock(stream_id)
                sock.settimeout(5.0)

    def subscribe(
        self, stream_id: str, skip_frames: bool = True, keep_fd: bool = False
    ) -> Iterator[Frame]:
        return self.subscribe_raw(stream_id, skip_frames, keep_fd)

    def on_frame(self, stream_id: str, callback: Callable[[Frame], None]) -> threading.Thread:
        def _run():
            for frame in self.subscribe_raw(stream_id):
                try:
                    callback(frame)
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def close(self) -> None:
        logger.info(
            "FdMediaClient: closing %d stream connections",
            len(self._streams),
        )
        # Release retained frames first so the daemon recycles their
        # buffers before the subscriptions and sockets go away.
        for handle in list(self._retained):
            try:
                handle.close()
            except Exception:
                pass
        with self._lock:
            for sock in self._streams.values():
                try:
                    unsub = struct.pack(_HDR_FMT, _FD_PUB_MSG_UNSUBSCRIBE, _HDR_SIZE)
                    sock.sendall(unsub)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
            self._streams.clear()

    # -- Encoded stream convenience methods --

    def get_encoded_stream(
        self, stream_id: str = "main", socket_dir: str | None = None
    ) -> EncodedStreamClient:
        """Return an :class:`EncodedStreamClient` for the given encoded stream.

        Args:
            stream_id: Stream name (e.g. ``"main"``, ``"sub"``).
            socket_dir: Directory containing EncodedPublisher UDS sockets
                (default ``/run/aipc/encoded``, or ``ENCODED_SOCK_DIR``).

        Returns:
            A connected :class:`EncodedStreamClient` reading from
            ``{socket_dir}/{stream_id}.sock``.
        """
        return EncodedStreamClient(stream_id=stream_id, socket_dir=socket_dir)

    def list_streams(self) -> list[str]:
        """List available raw stream IDs by scanning the camera socket.

        Returns common stream IDs. For detailed status use
        :class:`CameraClient.get_stream_status`.
        """
        return ["main", "sub"]

    def get_rtsp_url(
        self, stream_id: str = "main", host: str = "192.0.2.72", port: int = 8554
    ) -> str:
        """Return an RTSP URL for the given stream.

        Note: RTSP must be enabled on the device first (via CameraClient
        or REST API).
        """
        return f"rtsp://{host}:{port}/{stream_id}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
