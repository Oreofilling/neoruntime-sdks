"""Tests for neoruntime_ipc_sdk.color — NV12 <-> RGB/BGR conversions."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from neoruntime_ipc_sdk.color import (
    bgr_to_nv12,
    nv12_resize,
    nv12_to_bgr,
    nv12_to_rgb,
    rgb_to_nv12,
)

H, W = 32, 64


@pytest.fixture
def no_cv2(monkeypatch):
    """Force the numpy fallback paths even when cv2 is installed."""
    monkeypatch.setitem(sys.modules, "cv2", None)


def _uniform_image(rgb_tuple):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :] = rgb_tuple
    return img


class TestNv12ToPacked:
    def test_output_shape(self, no_cv2):
        nv12 = np.zeros((H * 3 // 2, W), dtype=np.uint8)
        nv12[:H] = 128  # mid gray luma
        nv12[H:] = 128  # neutral chroma
        rgb = nv12_to_rgb(nv12, W, H)
        assert rgb.shape == (H, W, 3)
        assert rgb.dtype == np.uint8

    def test_gray_roundtrip_numpy(self, no_cv2):
        nv12 = np.zeros((H * 3 // 2, W), dtype=np.uint8)
        nv12[:H] = 128
        nv12[H:] = 128
        rgb = nv12_to_rgb(nv12, W, H)
        # neutral chroma -> perfectly gray; Y=128 decodes to 130 in
        # limited range (1.1644 * (128-16) = 130.4 -> 130)
        assert rgb.reshape(-1, 3).std(axis=0).max() == 0.0
        assert np.all(rgb == 130)

    def test_red_channel_dominates_for_red(self, no_cv2):
        nv12 = rgb_to_nv12(_uniform_image((255, 0, 0)))
        rgb = nv12_to_rgb(nv12, W, H)
        assert rgb[:, :, 0].astype(int).mean() > 200
        assert rgb[:, :, 2].astype(int).mean() < 60

    def test_bgr_order_swapped(self, no_cv2):
        nv12 = rgb_to_nv12(_uniform_image((255, 0, 0)))
        bgr = nv12_to_bgr(nv12, W, H)
        assert bgr[:, :, 2].astype(int).mean() > 200
        assert bgr[:, :, 0].astype(int).mean() < 60

    def test_wrong_dims_raise(self):
        nv12 = np.zeros((H, W), dtype=np.uint8)
        with pytest.raises(ValueError, match="NV12 buffer"):
            nv12_to_rgb(nv12, W, H)


class TestPackedToNv12:
    def test_output_shape(self, no_cv2):
        nv12 = rgb_to_nv12(_uniform_image((30, 200, 90)))
        assert nv12.shape == (H * 3 // 2, W)
        assert nv12.dtype == np.uint8

    def test_odd_dimensions_raise(self, no_cv2):
        with pytest.raises(ValueError, match="even"):
            rgb_to_nv12(np.zeros((33, 64, 3), dtype=np.uint8))

    def test_smooth_roundtrip_tolerance(self, no_cv2):
        # realistic content keeps structure through 4:2:0 subsampling
        yy, xx = np.mgrid[0:H, 0:W]
        rgb = np.stack(
            [60 + xx * 2, 30 + yy * 3, np.full((H, W), 90)], axis=-1
        ).clip(0, 255).astype(np.uint8)
        back = nv12_to_rgb(rgb_to_nv12(rgb), W, H)
        assert np.abs(back.astype(int) - rgb.astype(int)).mean() < 4

    def test_noise_roundtrip_stays_bounded(self, no_cv2):
        # pure noise is the 4:2:0 worst case — chroma error per 2x2 block
        # is amplified ~2x on decode, so allow a coarse bound only
        rng = np.random.default_rng(7)
        rgb = rng.integers(0, 256, size=(H, W, 3)).astype(np.uint8)
        back = nv12_to_rgb(rgb_to_nv12(rgb), W, H)
        assert np.abs(back.astype(int) - rgb.astype(int)).mean() < 60

    def test_gray_survives_roundtrip(self, no_cv2):
        for level in (40, 120, 210):
            rgb = _uniform_image((level, level, level))
            back = nv12_to_rgb(rgb_to_nv12(rgb), W, H)
            assert np.abs(back.astype(int) - level).mean() < 2

    def test_bgr_entry_point_matches_rgb_for_gray(self, no_cv2):
        gray = _uniform_image((100, 100, 100))
        assert np.array_equal(rgb_to_nv12(gray), bgr_to_nv12(gray.copy()))


class TestNv12Resize:
    def _sample(self):
        nv12 = np.zeros((H * 3 // 2, W), dtype=np.uint8)
        nv12[:H] = 200
        nv12[H:] = 128
        return nv12

    def test_output_shape(self, no_cv2):
        out = nv12_resize(self._sample(), (W, H), (32, 16))
        assert out.shape == (16 * 3 // 2, 32)

    def test_values_preserved_for_uniform_planes(self, no_cv2):
        out = nv12_resize(self._sample(), (W, H), (32, 16))
        assert out[:16].mean() == pytest.approx(200, abs=1)
        assert out[16:].mean() == pytest.approx(128, abs=1)

    def test_upscale_rejected(self, no_cv2):
        with pytest.raises(ValueError, match="upscaling"):
            nv12_resize(self._sample(), (W, H), (W * 2, H * 2))

    def test_odd_destination_rejected(self, no_cv2):
        with pytest.raises(ValueError, match="even"):
            nv12_resize(self._sample(), (W, H), (33, 17))

    def test_wrong_source_dims_raise(self):
        with pytest.raises(ValueError, match="NV12 buffer"):
            nv12_resize(np.zeros((H, W), dtype=np.uint8), (W, H), (32, 16))
