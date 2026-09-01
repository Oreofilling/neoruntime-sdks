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
