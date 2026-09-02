"""
Tests for MJPEG streaming helpers: MjpegStream / mjpeg_wsgi_app / MjpegServer
"""

import threading
import time
import urllib.request

import numpy as np
import pytest

from neoruntime_ipc_sdk.media import Frame
from neoruntime_ipc_sdk.web import MjpegServer, MjpegStream, mjpeg_wsgi_app

JPEG_MAGIC = b"\xff\xd8\xff"


def make_frame(seq=0):
    yy, xx = np.mgrid[0:48, 0:64]
    img = np.stack([yy, xx, (xx + yy) % 256], axis=-1).astype(np.uint8)
    return Frame(sequence=seq, timestamp_ns=seq * 33_000_000, width=64,
                 height=48, format="RGB", image=img, metadata={})


class TestMjpegStream:
    def test_push_jpeg_latest(self):
        s = MjpegStream()
        assert s.latest() is None
        s.push_jpeg(b"\xff\xd8\xffabc")
        assert s.latest() == b"\xff\xd8\xffabc"
        s.push_jpeg(b"\xff\xd8\xffdef")
        assert s.latest() == b"\xff\xd8\xffdef"

    def test_push_frame_encodes_jpeg(self):
        s = MjpegStream()
        s.push_frame(make_frame())
        assert s.latest()[:3] == JPEG_MAGIC

    def test_wait_new_frame(self):
        s = MjpegStream()
        seq = s.latest_seq()
        assert s.wait_new(seq, timeout=0.05) is False
        s.push_jpeg(b"\xff\xd8\xffx")
        assert s.wait_new(seq, timeout=1.0) is True


class TestWsgiApp:
    def _call(self, app):
        captured = {}

        def start_response(status, headers, exc_info=None):
            captured["status"] = status
            captured["headers"] = dict(headers)

        body = app({"REQUEST_METHOD": "GET", "PATH_INFO": "/"}, start_response)
        return captured, body

    def test_response_envelope(self):
        app = mjpeg_wsgi_app(MjpegStream())
        captured, body = self._call(app)
        assert captured["status"] == "200 OK"
        assert captured["headers"]["Content-Type"].startswith(
            "multipart/x-mixed-replace")
        assert hasattr(body, "__iter__")
        close = getattr(body, "close", None)
        if close:
            close()

    def test_body_yields_multipart_chunks(self):
        source = MjpegStream()
        app = mjpeg_wsgi_app(source, fps=30)
        captured, body = self._call(app)
        source.push_jpeg(JPEG_MAGIC + b"payload")
        chunk = next(body)
        assert b"--frame" in chunk
        assert b"Content-Type: image/jpeg" in chunk
        assert JPEG_MAGIC in chunk
        assert str(len(JPEG_MAGIC + b"payload")).encode() in chunk
        if hasattr(body, "close"):
            body.close()

    def test_body_skips_empty_stream_without_hanging(self):
        source = MjpegStream()
        app = mjpeg_wsgi_app(source, fps=100)
        _, body = self._call(app)
        source.push_jpeg(JPEG_MAGIC + b"x")
        next(body)                          # first frame arrives
        if hasattr(body, "close"):
            body.close()


class TestMjpegServer:
    def test_serves_mjpeg_over_http(self):
        source = MjpegStream()
        server = MjpegServer(port=0, host="127.0.0.1", source=source)
        server.start()
        try:
            stop = threading.Event()

            def produce():
                seq = 0
                while not stop.is_set():
                    source.push_frame(make_frame(seq))
                    seq += 1
                    time.sleep(0.02)

            t = threading.Thread(target=produce, daemon=True)
            t.start()
            url = f"http://127.0.0.1:{server.port}/"
            with urllib.request.urlopen(url, timeout=5) as resp:
                assert resp.headers["Content-Type"].startswith(
                    "multipart/x-mixed-replace")
                data = resp.read(1024)
                assert b"--frame" in data
                assert JPEG_MAGIC in data
            stop.set()
            t.join(timeout=2)
        finally:
            server.stop()

    def test_unknown_path_404(self):
        source = MjpegStream()
        server = MjpegServer(port=0, host="127.0.0.1", source=source,
                             path="/stream")
        server.start()
        try:
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{server.port}/other", timeout=5)
            assert ei.value.code == 404
        finally:
            server.stop()

    def test_stop_is_idempotent(self):
        server = MjpegServer(port=0, host="127.0.0.1", source=MjpegStream())
        server.start()
        server.stop()
        server.stop()

    def test_connection_refused_after_stop(self):
        source = MjpegStream()
        server = MjpegServer(port=0, host="127.0.0.1", source=source)
        server.start()
        port = server.port
        server.stop()
        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5)
