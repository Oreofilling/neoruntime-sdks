"""
Tests for Frame, PixelFormat
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from neoruntime_ipc_sdk import Frame, PixelFormat


class TestFrame:
    def test_creation(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = Frame(
            sequence=1,
            timestamp_ns=1000000,
            width=100,
            height=100,
            format="RGB",
            image=image
        )
        assert frame.sequence == 1
        assert frame.width == 100
        assert frame.height == 100
        assert frame.format == "RGB"
    
    def test_to_rgb_from_rgb(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        frame = Frame(
            sequence=1,
            timestamp_ns=1000000,
            width=100,
            height=100,
            format="RGB",
            image=image
        )
        rgb = frame.to_rgb()
        assert rgb.shape == (100, 100, 3)
        np.testing.assert_array_equal(rgb, image)
    
    def test_to_rgb_from_gray8(self):
        image = np.ones((100, 100), dtype=np.uint8) * 128
        frame = Frame(
            sequence=1,
            timestamp_ns=1000000,
            width=100,
            height=100,
            format="GRAY8",
            image=image
        )
        rgb = frame.to_rgb()
        assert rgb.shape == (100, 100, 3)


class TestPixelFormat:
    def test_values(self):
        assert PixelFormat.NV12.value == 0
        assert PixelFormat.RGB.value == 2
        assert PixelFormat.BGR.value == 3




class TestEncodedStreamClientDefaults:
    """Socket-path derivation: stream_id × socket_dir × env × explicit."""

    def test_default_is_main_stream(self):
        from neoruntime_ipc_sdk import EncodedStreamClient
        assert EncodedStreamClient().socket_path == \
            "/run/aipc/encoded/main.sock"

    def test_stream_id_derives_path(self):
        from neoruntime_ipc_sdk import EncodedStreamClient
        assert EncodedStreamClient(stream_id="sub").socket_path == \
            "/run/aipc/encoded/sub.sock"

    def test_explicit_path_wins(self):
        from neoruntime_ipc_sdk import EncodedStreamClient
        assert EncodedStreamClient("/tmp/custom.sock").socket_path == \
            "/tmp/custom.sock"

    def test_socket_dir_overrides_base(self):
        from neoruntime_ipc_sdk import EncodedStreamClient
        got = EncodedStreamClient(stream_id="sub", socket_dir="/tmp/enc").socket_path
        assert got == "/tmp/enc/sub.sock"

    def test_env_var_overrides_default_dir(self, monkeypatch):
        from neoruntime_ipc_sdk import EncodedStreamClient
        monkeypatch.setenv("ENCODED_SOCK_DIR", "/tmp/envdir")
        assert EncodedStreamClient().socket_path == "/tmp/envdir/main.sock"

    def test_explicit_socket_dir_beats_env(self, monkeypatch):
        from neoruntime_ipc_sdk import EncodedStreamClient
        monkeypatch.setenv("ENCODED_SOCK_DIR", "/tmp/envdir")
        got = EncodedStreamClient(socket_dir="/tmp/explicit").socket_path
        assert got == "/tmp/explicit/main.sock"

    def test_get_encoded_stream_delegates(self, monkeypatch):
        from neoruntime_ipc_sdk import FdMediaClient
        monkeypatch.setenv("ENCODED_SOCK_DIR", "/tmp/envdir")
        client = FdMediaClient.__new__(FdMediaClient)  # no socket connect
        got = client.get_encoded_stream("sub")
        assert got.socket_path == "/tmp/envdir/sub.sock"


class TestListStreams:
    """list_streams() queries the camera daemon and degrades to main/sub."""

    @staticmethod
    def _fake_camera(statuses):
        from types import SimpleNamespace

        class _FakeCam:
            last_timeout = None

            def __init__(self):
                self._statuses = statuses

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_stream_status(self, timeout_s=None):
                _FakeCam.last_timeout = timeout_s
                if isinstance(statuses, Exception):
                    raise statuses
                return [SimpleNamespace(stream_id=sid, status=st) for sid, st in statuses]

        return _FakeCam

    def test_queries_active_streams(self, monkeypatch):

        from neoruntime_ipc_sdk import FdMediaClient
        fake = self._fake_camera([("main", "active"), ("sub", "active"), ("third", "stopped")])
        with patch("neoruntime_ipc_sdk.camera.CameraClient", fake):
            assert FdMediaClient().list_streams() == ["main", "sub"]

    def test_probe_is_time_bounded(self):
        # Review P2: a hung camera daemon must not stall list_streams —
        # the status probe carries a short RPC deadline.
        from neoruntime_ipc_sdk import FdMediaClient
        fake = self._fake_camera([("main", "active")])
        with patch("neoruntime_ipc_sdk.camera.CameraClient", fake):
            FdMediaClient().list_streams()
        assert fake.last_timeout == 1.0

    def test_falls_back_when_camera_service_fails(self):

        from neoruntime_ipc_sdk import FdMediaClient
        fake = self._fake_camera(RuntimeError("GetStreamStatus failed: no HAL"))
        with patch("neoruntime_ipc_sdk.camera.CameraClient", fake):
            assert FdMediaClient().list_streams() == ["main", "sub"]

    def test_falls_back_when_nothing_active(self):

        from neoruntime_ipc_sdk import FdMediaClient
        fake = self._fake_camera([("main", "starting")])
        with patch("neoruntime_ipc_sdk.camera.CameraClient", fake):
            assert FdMediaClient().list_streams() == ["main", "sub"]


class TestFrameTransformMetadata:
    """Affine geometry recorded by Frame.crop/resize — letterbox maths lives here."""

    @staticmethod
    def _rgb(w, h):
        return Frame(sequence=1, timestamp_ns=0, width=w, height=h, format="RGB",
                     image=np.zeros((h, w, 3), np.uint8))

    def test_letterbox_transform_roundtrip(self):
        out = self._rgb(1920, 1080).resize(640, 640, mode="letterbox")
        t = out.metadata["transform"]
        assert t["op"] == "resize" and t["mode"] == "letterbox"
        assert t["src_size"] == (1920, 1080) and t["dst_size"] == (640, 640)
        assert t["origin"] == (0, 140)  # 360-high content centred in a 640 canvas
        # dst centre maps back to the src centre
        sx = (320 - t["origin"][0]) / t["scale"][0]
        sy = (320 - t["origin"][1]) / t["scale"][1]
        assert sx == pytest.approx(960, abs=1)
        assert sy == pytest.approx(540, abs=1)

    def test_stretch_transform(self):
        out = self._rgb(64, 48).resize(32, 24, mode="stretch")
        t = out.metadata["transform"]
        assert t["mode"] == "stretch"
        assert t["scale"] == (0.5, 0.5) and t["origin"] == (0, 0)

    def test_crop_transform(self):
        out = self._rgb(64, 48).crop(10, 6, 32, 24)
        t = out.metadata["transform"]
        assert t["op"] == "crop"
        assert t["scale"] == (1.0, 1.0) and t["origin"] == (-10, -6)
        # dst pixel (0, 0) shows src pixel (10, 6)
        assert (0 - t["origin"][0]) / t["scale"][0] == 10

    def test_crop_then_resize_composes(self):
        crop = self._rgb(1920, 1080).crop(960, 540, 960, 540)
        t1 = crop.metadata["transform"]
        out = crop.resize(320, 320, mode="stretch")
        t2 = out.metadata["transform"]
        assert t2["src_size"] == (1920, 1080)  # composed src→dst directly
        assert t2["scale"][0] == pytest.approx(t1["scale"][0] * (320 / 960))
        # dst (0, 0) maps back to the crop origin in the original frame
        px = (0 - t2["origin"][0]) / t2["scale"][0]
        py = (0 - t2["origin"][1]) / t2["scale"][1]
        assert (px, py) == (960, 540)

    def test_parent_custom_metadata_survives(self):
        f = self._rgb(32, 32)
        f.metadata["stream"] = "sub"
        out = f.resize(16, 16)
        assert out.metadata["stream"] == "sub"
        assert "transform" in out.metadata

    def test_nv12_letterbox_even_geometry(self):
        frame = Frame(sequence=1, timestamp_ns=0, width=64, height=48, format="NV12",
                      image=np.zeros((72, 64), np.uint8))
        out = frame.resize(32, 32, mode="letterbox")
        t = out.metadata["transform"]
        assert t["origin"] == (0, 4)  # 32x24 content, chroma-aligned offset


class TestListStreamsCache:
    def test_results_cached_within_ttl(self):
        from unittest.mock import patch

        from neoruntime_ipc_sdk import FdMediaClient

        calls = []

        class _CountingCam:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_stream_status(self, timeout_s=None):
                from types import SimpleNamespace

                calls.append(timeout_s)
                return [SimpleNamespace(stream_id="main", status="active")]

        client = FdMediaClient(socket_path="/nonexistent/cache-test")
        with patch("neoruntime_ipc_sdk.camera.CameraClient", _CountingCam):
            assert client.list_streams() == ["main"]
            assert client.list_streams() == ["main"]   # served from cache

        assert calls == [1.0]  # second call never hit the daemon

    def test_cache_expiry_queries_again(self):
        from unittest.mock import patch

        from neoruntime_ipc_sdk import FdMediaClient

        calls = []

        class _CountingCam:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_stream_status(self, timeout_s=None):
                from types import SimpleNamespace

                calls.append(1)
                return [SimpleNamespace(stream_id="main", status="active")]

        client = FdMediaClient(socket_path="/nonexistent/cache-test")
        with patch("neoruntime_ipc_sdk.camera.CameraClient", _CountingCam):
            client.list_streams()
            client._list_streams_cache = None  # simulate TTL expiry
            client.list_streams()

        assert len(calls) == 2
        client.close()


class TestFrameContextManager:
    def test_with_block_releases_retained_frame(self):
        released = []

        class _FakeHandle:
            closed = False

            def close(self):
                self.closed = True
                released.append(1)

        frame = Frame(sequence=1, timestamp_ns=0, width=4, height=4,
                      format="GRAY8", image=np.zeros((4, 4), np.uint8))
        frame.handle = _FakeHandle()
        with frame as f:
            assert f is frame
        assert released == [1]

    def test_with_block_safe_for_plain_frames(self):
        frame = Frame(sequence=1, timestamp_ns=0, width=4, height=4,
                      format="GRAY8", image=np.zeros((4, 4), np.uint8))
        with frame:
            pass  # release() is a no-op without a handle — must not raise
