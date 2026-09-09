"""Phase 5 — recording + MJPEG web surfaces (real encoded frames, loopback HTTP).

TsWriter/HlsWriter/PrerollBuffer are fed *real* H.264 packets from the
encoded 'main' stream and verified at the byte level (188-byte TS grid,
0x47 sync bytes, m3u8 structure). The web side runs a real loopback HTTP
round-trip against MjpegServer plus a direct WSGI-callable probe, so the
multipart/x-mixed-replace contract is checked end to end.
"""

from __future__ import annotations

import http.client
import os
import shutil
import threading
import time
import unittest
from itertools import islice

from neoruntime_ipc_sdk import FdMediaClient
from neoruntime_ipc_sdk.recording import HlsWriter, PrerollBuffer, TsWriter
from neoruntime_ipc_sdk.web import (
    BOUNDARY,
    MjpegServer,
    MjpegStream,
    mjpeg_wsgi_app,
)

from common import DEVICE_TMP_DIR, DeviceTestCase

MAIN = "main"

_CACHE: dict = {"frames": None}


def _cached_frames():
    """Collect ~8s of real encoded frames once; reused by every class."""
    if _CACHE["frames"] is not None:
        return _CACHE["frames"]
    client = FdMediaClient()
    frames = []
    try:
        enc = client.get_encoded_stream(MAIN)
        if enc.get_frame(timeout_ms=10000) is None:
            return []  # encoder idle — callers skip honestly
        # setUpClass runs with no per-test alarm armed, so this loop
        # carries its own wall-clock deadline (the frame-count and pts
        # caps alone would spin forever on a stalled encoder).
        deadline = time.monotonic() + 60.0
        for frame in enc.subscribe():
            frames.append(frame)
            if len(frames) >= 240:
                break
            if len(frames) > 1 and frame.pts_ns - frames[0].pts_ns >= 8_000_000_000:
                break
            if time.monotonic() > deadline:
                break
    finally:
        client.close()
    _CACHE["frames"] = frames
    return frames


def _ts_grid_ok(path):
    """(ok, size): file is whole 188B packets with 0x47 at each boundary."""
    size = os.path.getsize(path)
    if size == 0 or size % 188:
        return False, size
    with open(path, "rb") as fh:
        data = fh.read()
    return all(data[i] == 0x47 for i in range(0, len(data), 188)), size


def _one_real_jpeg():
    """A real JPEG from the camera (synthetic fallback for the HTTP plumbing)."""
    client = FdMediaClient()
    try:
        frame = client.get_frame(MAIN, timeout_ms=5000)
        if frame is None:
            return b"\xff\xd8" + b"sdk-test-frame" * 64 + b"\xff\xd9"
        jpeg = frame.to_jpeg_bytes(quality=80)
        frame.release()
        return jpeg
    finally:
        client.close()


class _RecordingArea(DeviceTestCase):
    area = "recording_web"
    timeout_s = 180

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.frames = _cached_frames()
        if not cls.frames:
            raise unittest.SkipTest(
                "encoded stream 'main' produced no frames in 10s — "
                "recording writers unexercised"
            )
        cls.codec = (cls.frames[0].codec_name
                     if cls.frames[0].codec_name in ("h264", "h265") else "h264")

    @classmethod
    def _tmp(cls, name):
        return os.path.join(DEVICE_TMP_DIR, name)


class T01TsWriter(_RecordingArea):
    def test_01_write_real_frames(self):
        self.mark("recording.TsWriter.write")
        path = self._tmp("sdk_test_event.ts")
        writer = TsWriter(path, codec=self.codec)
        t0 = time.monotonic()
        for frame in self.frames:
            writer.write(frame)
        writer.close()
        write_ms = round((time.monotonic() - t0) * 1000.0, 1)
        grid_ok, size = _ts_grid_ok(path)
        self.evidence(path=path, frames=len(self.frames), bytes=size,
                      packets=size // 188, ts_grid_ok=grid_ok,
                      write_ms=write_ms, codec=self.codec)
        self.assertTrue(grid_ok,
                        "output not on the 188-byte TS grid with 0x47 sync")

    def test_02_context_manager_and_closed_write(self):
        self.mark("recording.TsWriter context manager/close contract")
        path = self._tmp("sdk_test_ctx.ts")
        with TsWriter(path, codec=self.codec) as writer:
            writer.write(self.frames[0])
        closed_rejected = True
        try:
            writer.write(self.frames[0])
            closed_rejected = False
        except RuntimeError:
            pass
        writer.close()  # idempotent — must not raise
        grid_ok, size = _ts_grid_ok(path)
        self.evidence(runtime_error_on_write=closed_rejected, bytes=size,
                      ts_grid_ok=grid_ok)
        self.assertTrue(closed_rejected,
                        "write() after close() must raise RuntimeError")
        self.assertTrue(grid_ok)

    def test_03_rejects_unknown_codec(self):
        self.mark("recording.TsWriter codec validation")
        with self.assertRaises(ValueError):
            TsWriter(self._tmp("never_created.ts"), codec="av1")
        self.evidence(raised="ValueError", codec_tried="av1")


class T02HlsWriter(_RecordingArea):
    def test_01_segments_and_playlist(self):
        self.mark("recording.HlsWriter.write/close")
        out_dir = os.path.join(DEVICE_TMP_DIR, "sdk_test_hls")
        shutil.rmtree(out_dir, ignore_errors=True)
        # Batch health first: HlsWriter starts segments only on
        # keyframe-flagged frames — a degenerate encoder batch (observed
        # once right after a daemon wedge + reboot: ~190 B/frame, zero
        # keyframe flags vs ~33 KB/frame baseline) must fail with a
        # self-explaining message, not a bare FileNotFoundError.
        keyframes = sum(1 for f in self.frames if f.is_keyframe)
        avg_bytes = int(sum(len(f.data) for f in self.frames)
                        / max(len(self.frames), 1))
        self.evidence(frames=len(self.frames), keyframes=keyframes,
                      avg_frame_bytes=avg_bytes)
        hls = HlsWriter(out_dir, segment_seconds=1.0, window=3, codec=self.codec)
        for frame in self.frames:
            hls.write(frame)
        hls.close()
        playlist_path = os.path.join(out_dir, "index.m3u8")
        if not os.path.exists(playlist_path):
            self.fail(
                f"HlsWriter wrote no playlist: {keyframes} keyframe-flagged "
                f"frames in a batch of {len(self.frames)} (avg {avg_bytes} "
                "B/frame) — segments only start on keyframes; a zero here "
                "means the encoded stream was degenerate, not that the "
                "writer dropped healthy data")
        with open(playlist_path) as fh:
            playlist = fh.read().splitlines()
        seg_files = sorted(f for f in os.listdir(out_dir) if f.endswith(".ts"))
        segs_ok = all(_ts_grid_ok(os.path.join(out_dir, f))[0]
                      for f in seg_files)
        playlist_segs = [ln for ln in playlist if ln.endswith(".ts")]
        span_s = (self.frames[-1].pts_ns - self.frames[0].pts_ns) / 1e9
        self.evidence(frames=len(self.frames), pts_span_s=round(span_s, 2),
                      segments_on_disk=seg_files,
                      segments_in_playlist=len(playlist_segs),
                      playlist_head=playlist[:10])
        self.assertIn("#EXTM3U", playlist)
        self.assertIn("#EXT-X-ENDLIST", playlist)
        self.assertGreaterEqual(len(seg_files), 1, "no HLS segment was cut")
        self.assertLessEqual(len(playlist_segs), 3,
                             "window=3 exceeded in the live playlist")
        self.assertTrue(segs_ok, "a segment file broke the TS packet grid")

    def test_02_rejects_bad_args(self):
        self.mark("recording.HlsWriter argument validation")
        never = os.path.join(DEVICE_TMP_DIR, "never_hls")
        with self.assertRaises(ValueError):
            HlsWriter(never, segment_seconds=0)
        with self.assertRaises(ValueError):
            HlsWriter(never, codec="mp4v")
        self.evidence(raised="ValueError x2")


class T03Preroll(_RecordingArea):
    def test_01_ring_window(self):
        self.mark("recording.PrerollBuffer.push/frames/__len__")
        buffer = PrerollBuffer(seconds=0.5)
        for frame in self.frames:
            buffer.push(frame)
        retained = buffer.frames
        span_ns = (retained[-1].pts_ns - retained[0].pts_ns
                   if retained else 0)
        self.evidence(pushed=len(self.frames), retained=len(retained),
                      window_s=0.5, retained_span_s=round(span_ns / 1e9, 3))
        self.assertEqual(len(buffer), len(retained))
        self.assertGreaterEqual(len(retained), 1)
        self.assertLessEqual(span_ns, 900_000_000,
                             "retained span exceeds the 0.5s window "
                             "+ one frame")

    def test_02_dump_returns_live_writer(self):
        self.mark("recording.PrerollBuffer.dump")
        buffer = PrerollBuffer(seconds=10.0)
        for frame in self.frames:
            buffer.push(frame)
        path = self._tmp("sdk_test_preroll.ts")
        writer = buffer.dump(path)
        writer.write(self.frames[-1])  # the live tail keeps working
        writer.close()
        grid_ok, size = _ts_grid_ok(path)
        self.evidence(path=path, buffered=len(buffer), bytes=size,
                      ts_grid_ok=grid_ok)
        self.assertIsInstance(writer, TsWriter)
        self.assertTrue(grid_ok)

    def test_03_dump_empty_raises(self):
        self.mark("recording.PrerollBuffer.dump (empty)")
        with self.assertRaises(ValueError):
            PrerollBuffer().dump(self._tmp("never_preroll.ts"))
        self.evidence(raised="ValueError")

    def test_04_rejects_nonpositive_window(self):
        self.mark("recording.PrerollBuffer validation")
        with self.assertRaises(ValueError):
            PrerollBuffer(seconds=0)
        self.evidence(raised="ValueError")


class T04MjpegStream(DeviceTestCase):
    area = "recording_web"
    timeout_s = 60

    def test_01_push_jpeg_latest_wait_new(self):
        self.mark("web.MjpegStream.push_jpeg/latest/latest_seq/wait_new")
        stream = MjpegStream()
        initial = stream.latest()
        seq0 = stream.latest_seq()
        payload = b"\xff\xd8" + b"sdk-test" * 8 + b"\xff\xd9"
        stream.push_jpeg(payload)
        seq1 = stream.latest_seq()
        got_new = stream.wait_new(seq0, timeout=1.0)
        timed_out = stream.wait_new(seq1, timeout=0.2)
        self.evidence(initial_latest=initial, seq0=seq0, seq1=seq1,
                      latest_is_payload=stream.latest() == payload,
                      wait_new_signaled=got_new,
                      wait_new_timeout_returned=timed_out)
        self.assertIsNone(initial, "latest() must be None before first push")
        self.assertEqual(seq1, seq0 + 1)
        self.assertEqual(stream.latest(), payload)
        self.assertTrue(got_new, "wait_new missed an already-published frame")
        self.assertFalse(timed_out,
                         "wait_new returned True with no new frame")

    def test_02_push_frame_real(self):
        self.mark("web.MjpegStream.push_frame")
        client = FdMediaClient()
        try:
            frame = client.get_frame(MAIN, timeout_ms=5000)
        finally:
            client.close()
        if frame is None:
            self.na("camera produced no raw frame for push_frame")
        stream = MjpegStream()
        stream.push_frame(frame, quality=75)
        jpeg = stream.latest()
        frame.release()
        self.evidence(jpeg_bytes=len(jpeg), magic=jpeg[:2].hex())
        self.assertEqual(jpeg[:2], b"\xff\xd8", "push_frame did not "
                                                "publish a JPEG")


class T05MjpegServing(DeviceTestCase):
    """Real loopback HTTP round-trip + the WSGI callable, side by side."""

    area = "recording_web"
    timeout_s = 90

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source = MjpegStream()
        cls.jpeg = _one_real_jpeg()
        cls._stop = threading.Event()

        def _pump():
            while not cls._stop.is_set():
                cls.source.push_jpeg(cls.jpeg)
                time.sleep(0.05)

        cls.pumper = threading.Thread(target=_pump, daemon=True,
                                      name="mjpeg-pump")
        cls.pumper.start()
        cls.server = MjpegServer(port=0, host="127.0.0.1",
                                  source=cls.source, path="/video", fps=20)
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls._stop.set()
        cls.pumper.join(timeout=2.0)

    def test_01_http_stream_round_trip(self):
        self.mark("web.MjpegServer HTTP round-trip")
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port,
                                          timeout=5)
        conn.request("GET", "/video")
        resp = conn.getresponse()
        ctype = resp.getheader("Content-Type")
        head = resp.read(64)   # blocks until the first streamed chunk
        more = resp.read(256)
        conn.close()           # client disconnect is an expected server path
        self.evidence(status=resp.status, content_type=ctype,
                      port=self.server.port, head=head[:32],
                      boundary=BOUNDARY.decode())
        self.assertEqual(resp.status, 200)
        self.assertIn("multipart/x-mixed-replace", ctype or "")
        self.assertIn("boundary=frame", ctype or "")
        self.assertTrue(head.startswith(b"--" + BOUNDARY),
                        "stream body does not start with the multipart "
                        "boundary")
        self.assertIn(b"Content-Type: image/jpeg", head + more)

    def test_02_http_404_other_path(self):
        self.mark("web.MjpegServer path routing (404)")
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port,
                                          timeout=5)
        conn.request("GET", "/nope")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.evidence(status=resp.status, path="/nope")
        self.assertEqual(resp.status, 404)

    def test_03_wsgi_app_callable(self):
        self.mark("web.mjpeg_wsgi_app")
        source = MjpegStream()
        source.push_jpeg(self.jpeg)
        app = mjpeg_wsgi_app(source, fps=30)
        captured = {}

        def start_response(status, headers, exc_info=None):
            captured["status"] = status
            captured["headers"] = dict(headers)

        iterable = app({"REQUEST_METHOD": "GET"}, start_response)
        try:
            chunks = list(islice(iterable, 2))
        finally:
            closer = getattr(iterable, "close", None)
            if closer is not None:
                closer()
        blob = b"".join(chunks)
        self.evidence(status=captured.get("status"),
                      headers=captured.get("headers"), chunks=len(chunks),
                      blob_head=blob[:40])
        self.assertEqual(captured.get("status"), "200 OK")
        self.assertIn("multipart/x-mixed-replace",
                      captured["headers"].get("Content-Type", ""))
        self.assertEqual(len(chunks), 2)
        self.assertTrue(blob.startswith(b"--" + BOUNDARY))
        self.assertIn(b"Content-Type: image/jpeg", blob)
        self.assertIn(self.jpeg, blob, "WSGI chunk did not carry the "
                                       "pushed JPEG payload")


if __name__ == "__main__":
    unittest.main()
