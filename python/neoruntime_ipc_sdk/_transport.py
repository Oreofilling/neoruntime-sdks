"""Shared transport plumbing for the SDK clients.

One place for the boilerplate every client used to copy-paste:

* :class:`GrpcClient` — lazy gRPC channel lifecycle, stub caching, the
  ``connect/close/connected`` context-manager surface, and the public
  ``channel``/``stub`` aliases older client bodies (and app code) read.
* :class:`UdsStreamClient` — single-socket AF_UNIX framing: connect with
  timeout, ``recv``-exact, reconnect, ``subscribe``/``get_frame`` loops.
* :func:`recvmsg_with_fds` / :func:`sendmsg_plain` — SCM_RIGHTS fd
  passing helpers shared by the fd-based media/DSP protocols.

InferenceClient deliberately does NOT use GrpcClient: it runs grpc.aio on
a dedicated event loop to avoid the sync completion-queue busy-poll that
saturates a core in tight infer loops (see inference.py). Forcing it onto
the synchronous base would reintroduce exactly that cost.
"""

import socket
import struct
import threading
from typing import Any, Callable, List, Optional, Tuple

import grpc

from .config import Config

__all__ = [
    "GrpcClient",
    "UdsStreamClient",
    "check_status",
    "recvmsg_with_fds",
    "sendmsg_plain",
]


def check_status(resp, label: str) -> None:
    """Raise RuntimeError if a Status-bearing response reports failure.

    Accepts either the response itself or one wrapping ``resp.status``.
    """
    s = resp.status if hasattr(resp, "status") and hasattr(resp.status, "success") else resp
    if hasattr(s, "success") and not s.success:
        msg = s.message if hasattr(s, "message") else "unknown error"
        raise RuntimeError(f"{label} failed: {msg}")


def recvmsg_with_fds(sock: socket.socket, bufsize: int,
                     max_fds: int = 3) -> Tuple[bytes, List[int]]:
    """Receive data + SCM_RIGHTS file descriptors via recvmsg."""
    fds_space = socket.CMSG_SPACE(max_fds * struct.calcsize("i"))
    data, ancdata, _flags, _addr = sock.recvmsg(bufsize, fds_space)
    fds: List[int] = []
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            n = len(cmsg_data) // struct.calcsize("i")
            fds.extend(struct.unpack(f"{n}i", cmsg_data[:n * struct.calcsize("i")]))
    return data, fds


def sendmsg_plain(sock: socket.socket, data: bytes) -> None:
    """Send bytes with no ancillary data."""
    sock.sendall(data)


class GrpcClient:
    """Base class for lazy-connecting synchronous gRPC clients.

    Subclasses set ``_stub_factory`` (channel -> stub callable) and either
    ``_endpoint_env``/``_endpoint_default`` or override
    ``_get_default_endpoint()``. Override ``channel_options`` to pass
    channel options (e.g. epoll1) at creation time.

    The ``_stub`` attribute stays a plain attribute on purpose: tests poke
    mock stubs into it, and ``stub`` is a read/write property onto the same
    slot for the older public ``client.stub`` access style.
    """

    _stub_factory: Callable[[Any], Any]
    _endpoint_env: Optional[str] = None
    _endpoint_default: Optional[str] = None

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or self._get_default_endpoint()
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[Any] = None

    def _get_default_endpoint(self) -> str:
        if self._endpoint_env is not None:
            import os
            return os.getenv(self._endpoint_env, self._endpoint_default or "")
        return Config.get_camera_control_endpoint()

    @property
    def channel_options(self) -> List[Tuple[str, Any]]:
        """gRPC channel options applied at connect time."""
        return []

    def _connect(self):
        """Return the stub, creating the channel on first use."""
        if self._stub is not None:
            return self._stub
        self._channel = grpc.insecure_channel(
            self.endpoint, options=self.channel_options)
        self._stub = self._stub_factory(self._channel)
        return self._stub

    def _ensure_connected(self):
        return self._connect()

    # -- legacy public surface (app code and tests use these) ---------------
    def connect(self) -> None:
        self._connect()

    @property
    def connected(self) -> bool:
        return self._channel is not None

    @property
    def channel(self) -> Optional[grpc.Channel]:
        return self._channel

    @property
    def stub(self):
        return self._stub

    @stub.setter
    def stub(self, value) -> None:
        self._stub = value

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *exc):
        self.close()


class UdsStreamClient:
    """Base class for single-socket framed AF_UNIX stream clients.

    The base owns the socket lifecycle: connect with timeout, recv-exact,
    reconnect with backoff, ``get_frame``/``subscribe``/``on_frame`` and
    close, all behind a per-instance lock. The subclass implements
    ``_recv_frame(sock) -> Optional[frame]`` (return None on a skipped or
    unreadable frame) and defines the frame type it yields.
    """

    connect_timeout = 5.0

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        sock.settimeout(self.connect_timeout)
        return sock

    def _get_sock(self) -> socket.socket:
        with self._lock:
            if self._sock is None:
                self._sock = self._connect()
            return self._sock

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(
                    f"{type(self).__name__}: socket closed")
            buf.extend(chunk)
        return bytes(buf)

    def _recv_frame(self, sock: socket.socket):
        """Read one frame or None; subclass protocol lives here."""
        raise NotImplementedError

    def _reconnect(self) -> socket.socket:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
            self._sock = self._connect()
            return self._sock

    def get_frame(self, timeout_ms: int = 5000):
        """Get a single frame. Returns None on timeout."""
        sock = self._get_sock()
        sock.settimeout(timeout_ms / 1000.0)
        try:
            return self._recv_frame(sock)
        except socket.timeout:
            return None

    def subscribe(self, reconnect: bool = True):
        """Yield frames continuously. Auto-reconnects if enabled."""
        import time

        sock = self._get_sock()
        name = type(self).__name__
        while True:
            frame = self._recv_frame(sock)
            if frame is not None:
                yield frame
                continue
            if not reconnect:
                break
            time.sleep(0.5)
            try:
                sock = self._reconnect()
            except OSError:
                from logging import getLogger
                getLogger(__name__).warning(
                    "%s: reconnect failed, retrying in 2s", name)
                time.sleep(2.0)

    def on_frame(self, callback: Callable[[Any], None]) -> threading.Thread:
        """Start a background thread that calls callback for each frame."""
        from logging import getLogger

        def _run():
            name = type(self).__name__
            log = getLogger(__name__)
            for frame in self.subscribe():
                try:
                    callback(frame)
                except Exception:
                    log.exception("%s: callback error", name)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
