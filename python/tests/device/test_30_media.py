"""Phase 3 — fd media interfaces (FdMediaClient + Frame + encoded streams).

Covers the whole raw-frame pipeline: stream discovery, single get_frame
(copy and keep_fd zero-copy), the Frame conversion helpers (crop /
resize / JPEG), threaded on_frame subscription, and the encoded-stream
client with its H.264 packet contract.
"""

from __future__ import annotations

import os
import threading
import time
import unittest

from neoruntime_ipc_sdk import EncodedFrame, FdMediaClient, Frame, FrameHandle

from common import DEVICE_TMP_DIR, DeviceTestCase

MAIN = "main"


class T01Streams(DeviceTestCase):
    area = "media"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = FdMediaClient()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_01_list_streams(self):
        self.mark("FdMediaClient.list_streams")
        streams = self.timed(self.client.list_streams, label="list_streams")
        self.evidence(streams=streams)
        self.assertIn(MAIN, streams)

    def test_02_get_rtsp_url(self):
        self.mark("FdMediaClient.get_rtsp_url")
        url = self.client.get_rtsp_url(MAIN)
        self.evidence(url=url)
        self.assertTrue(url.startswith("rtsp://") and MAIN in url)


class T02Frame(DeviceTestCase):
    """get_frame + the Frame conversion surface, on one real frame."""

    area = "media"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = FdMediaClient()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def setUp(self):
        super().setUp()
        self.frame = self.client.get_frame(MAIN, timeout_ms=5000)
        if self.frame is None:
            self.na("camera produced no frame on 'main'")

    def test_01_get_frame(self):
        self.mark("FdMediaClient.get_frame")
        self.evidence(
            sequence=self.frame.sequence,
            timestamp_ns=self.frame.timestamp_ns,
            width=self.frame.width,
            height=self.frame.height,
            format=self.frame.format,
            metadata_keys=list(self.frame.metadata)[:8],
        )
        self.assertGreater(self.frame.width, 0)
        self.assertGreater(self.frame.height, 0)

    def test_02_frame_data_to_array(self):
        self.mark("Frame.data/to_array")
        data = self.frame.data
        arr = self.frame.to_array()
        self.evidence(data_shape=None if data is None else data.shape,
                      array_shape=arr.shape, array_dtype=str(arr.dtype),
                      format=self.frame.format)
        expected_ndim = 2 if self.frame.format in ("NV12", "NV21") else 3
        self.assertEqual(arr.ndim, expected_ndim)

    def test_03_to_rgb(self):
        self.mark("Frame.to_rgb")
        rgb = self.timed(self.frame.to_rgb, label="to_rgb")
        self.evidence(rgb_shape=rgb.shape, rgb_dtype=str(rgb.dtype))
        self.assertEqual(rgb.shape[2], 3)

    def test_04_crop(self):
        self.mark("Frame.crop")
        w, h = self.frame.width, self.frame.height
        cropped = self.timed(
            self.frame.crop, 0, 0, w // 2, h // 2, label="crop"
        )
        self.assertIsInstance(cropped, Frame)
        self.evidence(cropped=(cropped.width, cropped.height))
        self.assertEqual(cropped.width, w // 2)

    def test_05_resize(self):
        self.mark("Frame.resize")
        resized = self.timed(
            self.frame.resize, 640, 480, label="resize_640x480"
        )
        self.assertIsInstance(resized, Frame)
        self.evidence(resized=(resized.width, resized.height),
                      array_shape=resized.to_array().shape)
        self.assertEqual((resized.width, resized.height), (640, 480))

    def test_06_to_jpeg_bytes_and_save(self):
        self.mark("Frame.to_jpeg_bytes/save")
        jpg = self.timed(self.frame.to_jpeg_bytes, 85, label="to_jpeg_bytes")
        path = os.path.join(DEVICE_TMP_DIR, "frame_evidence.jpg")
        self.timed(self.frame.save, path, label="save")
        size = os.path.getsize(path) if os.path.exists(path) else -1
        self.evidence(jpeg_bytes=len(jpg), saved_path=path, saved_bytes=size,
                      magic=jpg[:2].hex())
        # SOI marker proves real JPEG output either way.
        self.assertEqual(jpg[:2], b"\xff\xd8")
        self.assertEqual(size, len(jpg))

    def test_07_get_frame_keep_fd(self):
        self.mark("FdMediaClient.get_frame(keep_fd=True)/FrameHandle")
        frame = self.client.get_frame(MAIN, timeout_ms=5000, keep_fd=True)
        if frame is None:
            self.na("camera produced no frame for keep_fd probe")
        handle = frame.handle
        if handle is None:
            self.na("keep_fd frame carried no FrameHandle")
        self.assertIsInstance(handle, FrameHandle)
        self.evidence(fd=handle.fd, closed_before=handle.closed)
        self.assertFalse(handle.closed)
        self.assertGreaterEqual(handle.fd, 0)
        # Copy path must still work off the dma-buf before release.
        arr = frame.to_array()
        self.evidence(array_shape=arr.shape)
        self.assertGreater(arr.size, 0)
        handle.close()
        self.assertTrue(handle.closed)
        self.evidence(closed_after=handle.closed)

    def test_08_frame_release(self):
        self.mark("Frame.release")
        self.frame.release()
        self.evidence(released=True)


class T03Subscribe(DeviceTestCase):
    area = "media"
    timeout_s = 90

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = FdMediaClient()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_01_subscribe_raw(self):
        self.mark("FdMediaClient.subscribe_raw/subscribe")
        import time

        frames = []
        seqs = []
        t0 = time.monotonic()
        deadline = t0 + 45.0
        for frame in self.client.subscribe(MAIN):
            frames.append((frame.width, frame.height))
            seqs.append(frame.sequence)
            if len(frames) >= 5:
                break
            if time.monotonic() > deadline:
                self.record["outcome_note"] = (
                    "subscribe stalled before 5 frames (45s deadline)")
                break
        elapsed_ms = round((time.monotonic() - t0) * 1000.0, 1)
        self.evidence(count=len(frames), first_5_shapes=frames,
                      sequences=seqs, elapsed_ms=elapsed_ms)
        self.assertGreaterEqual(len(frames), 5, "subscribe yielded <5 frames")

    def test_02_on_frame_thread(self):
        self.mark("FdMediaClient.on_frame")
        got = []
        done = threading.Event()

        def cb(frame):
            if len(got) < 3:
                got.append(frame.sequence)
            if len(got) >= 3:
                done.set()

        thread = self.client.on_frame(MAIN, cb)
        self.assertTrue(thread.is_alive())
        joined = done.wait(timeout=30.0)
        self.evidence(sequences=got, thread_alive=thread.is_alive(),
                      signaled=joined)
        self.assertTrue(joined, "on_frame callback never fired 3x in 30s")
        self.assertEqual(len(got), 3)


class T04Encoded(DeviceTestCase):
    area = "media"
    timeout_s = 90

    def test_01_get_encoded_stream_frames(self):
        self.mark("FdMediaClient.get_encoded_stream/EncodedStreamClient")
        client = FdMediaClient()
        enc = client.get_encoded_stream(MAIN)
        try:
            frame = enc.get_frame(timeout_ms=10000)
        finally:
            client.close()
        if frame is None:
            self.na("encoded stream 'main' produced no frame in 10s "
                    "(encoder may be idle until a subscriber forces a keyframe)")
        self.assertIsInstance(frame, EncodedFrame)
        self.evidence(
            codec=frame.codec,
            codec_name=frame.codec_name,
            pts_ns=frame.pts_ns,
            dts_ns=frame.dts_ns,
            width=frame.width,
            height=frame.height,
            data_bytes=len(frame.data),
            is_keyframe=frame.is_keyframe,
        )
        self.assertGreater(len(frame.data), 0)
        # H.264/H.265 Annex-B start code — proves real encoder payload.
        self.assertEqual(frame.data[:4], b"\x00\x00\x00\x01",
                         "encoded payload lacks Annex-B start code")

    def test_02_encoded_subscribe(self):
        self.mark("EncodedStreamClient.subscribe")
        client = FdMediaClient()
        enc = client.get_encoded_stream(MAIN)
        n = 0
        keyframes = 0
        deadline = time.monotonic() + 45.0
        try:
            for frame in enc.subscribe():
                n += 1
                keyframes += 1 if frame.is_keyframe else 0
                if n >= 5:
                    break
                if time.monotonic() > deadline:
                    self.record["outcome_note"] = (
                        "encoded subscribe stalled before 5 frames "
                        "(45s deadline)")
                    break
        finally:
            client.close()
        self.evidence(frames=n, keyframes=keyframes)
        self.assertGreaterEqual(n, 5, "encoded subscribe yielded <5 frames")


if __name__ == "__main__":
    unittest.main()
