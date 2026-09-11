"""One-shot cv2-fallback warnings (tier-2 observability)."""

import logging
import sys

import numpy as np
import pytest

from neoruntime_ipc_sdk import Frame
from neoruntime_ipc_sdk._fallbacks import reset_fallback_warnings, warn_numpy_fallback


@pytest.fixture(autouse=True)
def _clean_warnings():
    reset_fallback_warnings()
    yield
    reset_fallback_warnings()


class TestWarnNumpyFallback:
    def test_warns_once_per_operation(self, caplog):
        with caplog.at_level(logging.WARNING, logger="neoruntime_ipc_sdk.fallbacks"):
            warn_numpy_fallback("resize")
            warn_numpy_fallback("resize")
            warn_numpy_fallback("resize")
        assert len(caplog.records) == 1
        assert "resize" in caplog.text
        assert "cv2 not installed" in caplog.text

    def test_different_operations_warn_separately(self, caplog):
        with caplog.at_level(logging.WARNING, logger="neoruntime_ipc_sdk.fallbacks"):
            warn_numpy_fallback("resize")
            warn_numpy_fallback("JPEG encode")
        assert len(caplog.records) == 2


class TestIntegrationWithoutCv2:
    """sys.modules['cv2'] = None makes ``import cv2`` raise ImportError."""

    def test_frame_resize_warns_without_cv2(self, caplog, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        frame = Frame(
            sequence=1, timestamp_ns=0, width=32, height=16, format="RGB",
            image=np.zeros((16, 32, 3), np.uint8),
        )
        with caplog.at_level(logging.WARNING, logger="neoruntime_ipc_sdk.fallbacks"):
            frame.resize(16, 8)
        assert "cv2 not installed" in caplog.text

    def test_nv12_to_rgb_warns_without_cv2(self, caplog, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        frame = Frame(
            sequence=1, timestamp_ns=0, width=32, height=16, format="NV12",
            image=np.zeros((24, 32), np.uint8),
        )
        with caplog.at_level(logging.WARNING, logger="neoruntime_ipc_sdk.fallbacks"):
            frame.to_rgb()
        assert "NV12->RGB" in caplog.text
