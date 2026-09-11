"""
MJPEG streaming helpers - serve live JPEG frames over HTTP.

MjpegStream is a thread-safe latest-frame holder (slow clients simply
miss frames instead of building latency); mjpeg_wsgi_app mounts it into
any WSGI server (flask/waitress/gunicorn); MjpegServer is a zero-dependency
standalone HTTP server for apps that do not run a web framework.

Pair with AppClient.register_web_url() so the platform web console can
reach the app container's MJPEG page.

platform_stream_url() is the zero-server alternative: it composes the
WebSocket URL of a PLATFORM camera stream (the same /api/v1/h264/{id}
endpoint the console plays), so an app that pushes frames through
FramePublisher can hand viewers a URL without opening any port of its
own - the platform gateway already serves the stream.

Example (flask):
    source = MjpegStream()
    app = Flask(__name__)
    app.route("/video")(mjpeg_wsgi_app(source, fps=15))
    # push frames from a worker: source.push_frame(frame)

Example (standalone):
    server = MjpegServer(port=8080, source=source)
    server.start()
    ...
    server.stop()

Example (no own server at all):
    # inject frames into the platform's sub stream, then:
    url = platform_stream_url("sub", host="192.168.1.10", token=jwt)
"""

from __future__ import annotations

import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

from .media import Frame

BOUNDARY = b"frame"  # multipart boundary token

#: Environment variable consulted when ``host`` is not passed explicitly to
#: platform_stream_url(). Nothing injects it yet - set it in the app
#: container or shell when the SDK runs somewhere that needs a non-default
#: host (the SDK itself talks to the daemon over a local socket and never
#: learns the device's external address).
WEB_HOST_ENV = "AIPC_WEB_HOST"

#: Path the platform gateway serves encoded camera streams on (WebSocket,
#: proxied to platform-api; the web console's player consumes the same URL).
H264_WS_PATH = "/api/v1/h264"


def platform_stream_url(
    stream_id: str = "sub",
    host: str | None = None,
    scheme: str | None = None,
    token: str | None = None,
) -> str:
    """Compose the WebSocket URL of a platform camera stream.

    The platform gateway (HTTPS on 443) already reverse-proxies encoded
    camera streams at ``/api/v1/h264/{stream_id}``; the web console plays
    this URL. Apps that publish frames via
    :class:`~neoruntime_ipc_sdk.injection.FramePublisher` (REPLACE or
    OVERLAY) can hand this URL to any viewer instead of running their own
    MJPEG server.

    Args:
        stream_id: Platform stream name (``main`` / ``sub`` / ``third``).
        host: Device host, optionally with a port (``"192.168.1.10"``,
            ``"localhost:8080"``). Full origins (``"https://..."``) are
            accepted - the scheme is mapped to ws/wss. Resolved from the
            ``AIPC_WEB_HOST`` environment variable when omitted, else
            ``localhost``.
        scheme: ``"wss"`` (default - the TLS gateway) or ``"ws"`` for a
            plain-HTTP face (e.g. platform-api directly on
            ``localhost:8080``). An explicit value overrides any scheme
            inferred from a full-origin ``host``.
        token: Auth token appended as ``?token=...`` - network access to
            ``/api/v1`` requires one (the console passes its JWT here).

    Returns:
        The stream URL, e.g. ``wss://192.168.1.10/api/v1/h264/sub``.

    Raises:
        ValueError: If ``stream_id`` is empty.
    """
    stream = stream_id.strip("/")
    if not stream:
        raise ValueError("stream_id must be a non-empty stream name")

    if host is None:
        host = os.getenv(WEB_HOST_ENV) or "localhost"
    host = host.strip().rstrip("/")

    # a full-origin host carries its own scheme prefix: always strip it,
    # and let it pick the scheme when the caller did not pass one
    # (http -> ws, https/wss -> wss)
    lower = host.lower()
    if lower.startswith(("https://", "wss://")):
        host = host.partition("://")[2]
        inferred = "wss"
    elif lower.startswith(("http://", "ws://")):
        host = host.partition("://")[2]
        inferred = "ws"
    else:
        inferred = None
    if scheme is None:
        scheme = inferred or "wss"

    url = f"{scheme}://{host}{H264_WS_PATH}/{stream}"
    if token:
        # quote_via=quote -> %20 for spaces (matches the console's
        # encodeURIComponent) and keeps a literal '+' in a JWT unambiguous
        url += "?" + urllib.parse.urlencode(
            {"token": token}, quote_via=urllib.parse.quote
        )
    return url


class MjpegStream:
    """Thread-safe holder of the most recent JPEG frame."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0

    def push_jpeg(self, data: bytes) -> None:
        """Publish an already-encoded JPEG frame (bytes)."""
        with self._cond:
            self._jpeg = data
            self._seq += 1
            self._cond.notify_all()

    def push_frame(self, frame: Frame, quality: int = 85) -> None:
        """Encode a Frame to JPEG and publish it."""
        self.push_jpeg(frame.to_jpeg_bytes(quality=quality))

    def latest(self) -> bytes | None:
        """Most recent JPEG bytes, or None before the first push."""
        with self._cond:
            return self._jpeg

    def latest_seq(self) -> int:
        """Monotonic push counter (used with wait_new)."""
        with self._cond:
            return self._seq

    def wait_new(self, after_seq: int, timeout: float = 1.0) -> bool:
        """Block until a frame newer than after_seq arrives (or timeout)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= after_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._cond.wait(remaining):
                    if self._seq <= after_seq:
                        return False
            return True


def _frame_chunks(source: MjpegStream, fps: float) -> Iterator[bytes]:
    """Yield multipart chunks at most fps times per second."""
    interval = 1.0 / max(fps, 0.1)
    last_seq = source.latest_seq()
    deadline = time.monotonic()
    while True:
        last_seq = source.latest_seq()
        if source.wait_new(last_seq - 1, timeout=interval):
            jpeg = source.latest()
            if jpeg is not None:
                yield (
                    b"--" + BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )
        # pace output; slow consumers naturally drop frames on next read
        deadline += interval
        pause = deadline - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        else:
            deadline = time.monotonic()


def mjpeg_wsgi_app(source: MjpegStream, fps: float = 15):
    """Build a WSGI callable serving the stream as multipart/x-mixed-replace."""

    def app(environ, start_response):
        headers = [
            ("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}"),
            ("Cache-Control", "no-store, private"),
            ("Pragma", "no-cache"),
            ("Access-Control-Allow-Origin", "*"),
        ]
        start_response("200 OK", headers)
        return _frame_chunks(source, fps)

    return app


class _MjpegHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close connection after each stream

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        server: MjpegServer = self.server
        if self.path != server.path:
            self.send_error(404)
            return
        try:
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}"
            )
            self.send_header("Cache-Control", "no-store, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            for chunk in _frame_chunks(server.source, server.fps):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionError, OSError):
            return  # client disconnected - stop stream

    def log_message(self, format: str, *args) -> None:
        pass  # keep stdout clean inside apps


class MjpegServer:
    """Standalone threaded HTTP server exposing one MJPEG stream."""

    def __init__(
        self,
        port: int = 8080,
        host: str = "0.0.0.0",
        source: MjpegStream | None = None,
        path: str = "/",
        fps: float = 15,
    ):
        if source is None:
            raise ValueError("source MjpegStream is required")
        self.source = source
        self.path = path
        self.fps = fps
        self._host = host
        self._requested_port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """Actual bound port (useful when constructed with port=0)."""
        if self._server is None:
            return self._requested_port
        return self._server.server_address[1]

    def start(self) -> None:
        """Bind and start serving in a daemon thread."""
        if self._server is not None:
            return
        server = ThreadingHTTPServer((self._host, self._requested_port), _MjpegHandler)
        server.daemon_threads = True
        server.source = self.source  # type: ignore[attr-defined]
        server.path = self.path  # type: ignore[attr-defined]
        server.fps = self.fps  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, daemon=True, name="mjpeg-server"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and close the listening socket (idempotent)."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def __enter__(self) -> MjpegServer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
