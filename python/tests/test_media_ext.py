"""
Tests for Frame extensions: crop / resize / to_jpeg_bytes
"""

import sys

import numpy as np
import pytest

from neoruntime_ipc_sdk.media import Frame, PixelFormat


def make_rgb_frame(w=64, h=48, seq=7, ts=12345):
    """RGB frame where pixel (y, x) has value (y, x, (x+y) % 256)."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([yy, xx, (xx + yy) % 256], axis=-1).astype(np.uint8)
    return Frame(sequence=seq, timestamp_ns=ts, width=w, height=h,
                 format="RGB", image=img, metadata={"stream": "main"})


def make_nv12_frame(w=64, h=48, seq=7, ts=12345):
    """NV12 frame: Y gradient, neutral chroma (128)."""
    yy, xx = np.mgrid[0:h, 0:w]
    y = ((xx + yy) % 256).astype(np.uint8)
    uv = np.full((h // 2, w), 128, dtype=np.uint8)
    img = np.vstack([y, uv])
    return Frame(sequence=seq, timestamp_ns=ts, width=w, height=h,
                 format="NV12", image=img, metadata={"stream": "main"})


class TestFrameCropRgb:
    def test_crop_shape_and_content(self):
        f = make_rgb_frame(w=64, h=48)
        c = f.crop(8, 12, 32, 24)
        assert c.width == 32 and c.height == 24
        assert c.format == "RGB"
        assert c.image.shape == (24, 32, 3)
        np.testing.assert_array_equal(c.image, f.image[12:36, 8:40])

    def test_crop_returns_new_frame_preserving_metadata(self):
        f = make_rgb_frame(seq=7, ts=12345)
        c = f.crop(0, 0, 16, 16)
        assert c is not f
        assert c.sequence == 7
        assert c.timestamp_ns == 12345
        assert c.metadata == {"stream": "main"}

    def test_crop_does_not_mutate_original(self):
        f = make_rgb_frame()
        before = f.image.copy()
        f.crop(4, 4, 8, 8)
        np.testing.assert_array_equal(f.image, before)

    def test_crop_out_of_bounds_raises(self):
        f = make_rgb_frame(w=64, h=48)
        with pytest.raises(ValueError):
            f.crop(60, 0, 8, 8)     # x + w > width
        with pytest.raises(ValueError):
            f.crop(0, 40, 8, 16)    # y + h > height
        with pytest.raises(ValueError):
            f.crop(0, 0, 0, 8)      # zero size

    def test_crop_negative_offset_raises(self):
        f = make_rgb_frame()
        with pytest.raises(ValueError):
            f.crop(-1, 0, 8, 8)


class TestFrameCropNv12:
    def test_crop_planes(self):
        f = make_nv12_frame(w=64, h=48)
        c = f.crop(8, 12, 32, 24)
        assert c.width == 32 and c.height == 24
        assert c.image.shape == (24 + 12, 32)  # (h + h/2, w)
        # Y plane
        np.testing.assert_array_equal(
            c.image[:24], f.image[12:36, 8:40])
        # UV plane: rows h/2, cols w (interleaved, chroma sample pairs)
        np.testing.assert_array_equal(
            c.image[24:], f.image[48 + 6:48 + 18, 4:4 + 32])

    def test_crop_requires_even_geometry(self):
        f = make_nv12_frame(w=64, h=48)
        with pytest.raises(ValueError):
            f.crop(3, 0, 32, 24)    # odd x
        with pytest.raises(ValueError):
            f.crop(0, 0, 31, 24)    # odd w
        with pytest.raises(ValueError):
            f.crop(0, 0, 32, 25)    # odd h

    def test_crop_unsupported_format_raises(self):
        f = make_rgb_frame(w=64, h=48)
        f.format = "YUYV"
        with pytest.raises(ValueError):
            f.crop(0, 0, 8, 8)


class TestFrameResizeLetterbox:
    def test_letterbox_rgb_geometry(self):
        # 64x48 -> 64x64: scale = min(64/64, 64/48) = 1.0 -> content 64x48, bars top/bottom
        f = make_rgb_frame(w=64, h=48)
        r = f.resize(64, 64, mode="letterbox", pad_value=114)
        assert r.width == 64 and r.height == 64
        assert r.format == "RGB"
        img = r.image
        # content occupies rows 8..56 (48 rows centered)
        np.testing.assert_array_equal(img[8:56], f.image)
        # pad bars
        assert (img[:8] == 114).all()
        assert (img[56:] == 114).all()

    def test_letterbox_rgb_side_bars(self):
        # 64x48 -> 48x48: scale = min(48/64, 48/48) = 0.75 -> content 48x36, bars top/bottom
        f = make_rgb_frame(w=64, h=48)
        r = f.resize(48, 48, mode="letterbox", pad_value=0)
        img = r.image
        assert (img[:6] == 0).all()     # top bar
        assert (img[42:] == 0).all()    # bottom bar
        # content rows 6..42 are a resized version, non-pad
        assert not (img[6:42] == 0).all()

    def test_stretch_rgb(self):
        f = make_rgb_frame(w=64, h=48)
        r = f.resize(32, 24, mode="stretch")
        assert r.image.shape == (24, 32, 3)

    def test_scale_crop_rgb(self):
        # cover mode: 64x48 -> 48x48 crops horizontally after scaling up to cover
        f = make_rgb_frame(w=64, h=48)
        r = f.resize(48, 48, mode="crop")
        assert r.image.shape == (48, 48, 3)

    def test_resize_preserves_metadata(self):
        f = make_rgb_frame(seq=9, ts=999)
        r = f.resize(32, 32)
        assert r.sequence == 9 and r.timestamp_ns == 999

    def test_resize_does_not_mutate_original(self):
        f = make_rgb_frame()
        before = f.image.copy()
        f.resize(32, 32)
        np.testing.assert_array_equal(f.image, before)

    def test_invalid_mode_raises(self):
        f = make_rgb_frame()
        with pytest.raises(ValueError):
            f.resize(32, 32, mode="bogus")


class TestFrameResizeNv12:
    def test_stretch_nv12_shape(self):
        f = make_nv12_frame(w=64, h=48)
        r = f.resize(32, 24, mode="stretch")
        assert r.format == "NV12"
        assert r.image.shape == (24 + 12, 32)

    def test_letterbox_nv12_shape_and_neutral_chroma_pad(self):
        f = make_nv12_frame(w=64, h=48)
        r = f.resize(64, 64, mode="letterbox", pad_value=16)
        assert r.image.shape == (64 + 32, 64)
        y = r.image[:64]
        uv = r.image[64:]
        # content rows 8..56 in Y
        assert (y[:8] == 16).all()
        assert (y[56:] == 16).all()
        # chroma pad is neutral 128 everywhere outside content rows
        assert (uv[:4] == 128).all()
        assert (uv[28:] == 128).all()

    def test_nv12_requires_even_target(self):
        f = make_nv12_frame(w=64, h=48)
        with pytest.raises(ValueError):
            f.resize(33, 24, mode="stretch")
        with pytest.raises(ValueError):
            f.resize(32, 25, mode="stretch")

    def test_letterbox_nv12_neutral_chroma_keeps_gray(self):
        # neutral chroma + luma gradient -> to_rgb() is grayscale-ish (r==b)
        f = make_nv12_frame(w=64, h=48)
        r = f.resize(32, 32, mode="letterbox")
        rgb = r.to_rgb()
        assert rgb.shape == (32, 32, 3)
        # inside content area chroma stays neutral -> r == b
        assert np.allclose(rgb[4:28, :, 0].astype(int), rgb[4:28, :, 2].astype(int), atol=2)


class TestFrameResizePureNumpyFallback:
    def test_stretch_without_cv2(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        f = make_rgb_frame(w=64, h=48)
        r = f.resize(32, 24, mode="stretch")
        assert r.image.shape == (24, 32, 3)
        # nearest-neighbour of scale 0.5 must sample even source coords
        np.testing.assert_array_equal(r.image[:, :, 1], f.image[::2, ::2, 1])

    def test_letterbox_without_cv2(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        f = make_rgb_frame(w=64, h=48)
        r = f.resize(64, 64, mode="letterbox", pad_value=7)
        assert (r.image[:8] == 7).all()
        assert r.image.shape == (64, 64, 3)

    def test_nv12_stretch_without_cv2(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        f = make_nv12_frame(w=64, h=48)
        r = f.resize(32, 24, mode="stretch")
        assert r.image.shape == (36, 32)


class TestFrameToJpegBytes:
    def test_jpeg_magic_bytes(self):
        f = make_rgb_frame()
        data = f.to_jpeg_bytes()
        assert data[:3] == b"\xff\xd8\xff"
        assert data[-2:] == b"\xff\xd9"

    def test_jpeg_from_nv12(self):
        f = make_nv12_frame()
        data = f.to_jpeg_bytes()
        assert data[:3] == b"\xff\xd8\xff"

    def test_jpeg_quality_accepted(self):
        f = make_rgb_frame()
        hi = f.to_jpeg_bytes(quality=95)
        lo = f.to_jpeg_bytes(quality=20)
        assert hi[:3] == b"\xff\xd8\xff"
        assert lo[:3] == b"\xff\xd8\xff"
        # higher quality on a gradient image should not be smaller
        assert len(hi) >= len(lo)

    def test_jpeg_without_cv2(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        f = make_rgb_frame()
        data = f.to_jpeg_bytes()
        assert data[:3] == b"\xff\xd8\xff"

    def test_save_jpg_uses_jpeg_encoder(self, tmp_path):
        import io

        from PIL import Image
        f = make_rgb_frame()
        p = tmp_path / "out.jpg"
        f.save(str(p))
        data = p.read_bytes()
        assert data[:3] == b"\xff\xd8\xff"
        img = Image.open(io.BytesIO(data))
        assert img.size == (64, 48)


class _FakeDspFactory:
    """Stand-in for dsp.DspClient; records resize_hw calls per client."""

    def __init__(self, result=None, error=None):
        self.result = result      # ndarray, or callable(src, dw, dh) -> ndarray
        self.error = error
        self.clients = []

    def __call__(self, *args, **kwargs):
        parent = self

        class _Client:
            def __init__(self):
                self.calls = []
                parent.clients.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def resize_hw(self, src, dw, dh, **kw):
                self.calls.append((src, dw, dh, kw))
                if parent.error is not None:
                    raise parent.error
                r = parent.result
                return r(src, dw, dh) if callable(r) else r

        return _Client()


def make_keepfd_frame(w=64, h=48, fmt="NV12", image=None):
    """Frame backed by an (unopened) FrameHandle, as keep-fd receive yields."""
    from neoruntime_ipc_sdk.media import FrameHandle
    handle = FrameHandle(fds=[-1], strides=[w], plane_sizes=[w * h],
                         frame_id=9, width=w, height=h, format=fmt)
    return Frame(sequence=7, timestamp_ns=12345, width=w, height=h,
                 format=fmt, image=image, metadata={"stream": "main"},
                 handle=handle)


class TestResizeDspFastPath:
    """Frame.resize routes keep-fd frames through the DSP (SDK-3)."""

    def _patch(self, monkeypatch, factory):
        import neoruntime_ipc_sdk.dsp as dsp_mod
        monkeypatch.setattr(dsp_mod, "DspClient", factory)

    def test_stretch_uses_dsp_and_preserves_metadata(self, monkeypatch):
        out = np.zeros((72, 96), dtype=np.uint8)
        f = make_keepfd_frame(64, 48, "GRAY8")
        fac = _FakeDspFactory(result=out)
        self._patch(monkeypatch, fac)

        r = f.resize(96, 72, mode="stretch")

        assert len(fac.clients) == 1
        (src, dw, dh, kw), = fac.clients[0].calls
        assert src is f and (dw, dh) == (96, 72)
        assert kw.get("scaling") == "stretch"
        assert r.image is out                       # DSP output passed through
        assert (r.width, r.height, r.format) == (96, 72, "GRAY8")
        assert r.sequence == 7 and r.timestamp_ns == 12345
        assert r.metadata == {"stream": "main"}
        assert not f.handle.closed                  # input frame not consumed
        assert f.image is None                      # never materialized

    def test_letterbox_scales_content_on_dsp_pads_on_cpu(self, monkeypatch):
        # 16x32 -> 64x64 letterbox: scale 2 -> content 32x64 at offset (16, 0)
        f = make_keepfd_frame(16, 32, "NV12")

        def content(src, dw, dh):
            assert (dw, dh) == (32, 64)             # content box, not target
            y = np.full((dh, dw), 7, dtype=np.uint8)
            uv = np.full((dh // 2, dw), 200, dtype=np.uint8)
            return np.vstack([y, uv])

        fac = _FakeDspFactory(result=content)
        self._patch(monkeypatch, fac)

        r = f.resize(64, 64)                        # default: letterbox

        img = r.image
        assert img.shape == (96, 64)                # NV12 2D layout
        assert img[0, 0] == 114 and img[63, 63] == 114      # luma pad
        assert img[0, 16] == 7 and img[63, 47] == 7         # content box
        assert img[64, 0] == 128 and img[64, 63] == 128     # chroma pad
        assert img[64, 16] == 200                            # content chroma

    def test_crop_scales_cover_box_and_center_crops(self, monkeypatch):
        # 16x32 -> 64x48 crop: cover scale 4 -> box 64x128, crop rows 40..88
        f = make_keepfd_frame(16, 32, "GRAY8")
        content = np.arange(64 * 128, dtype=np.uint8).reshape(128, 64)
        fac = _FakeDspFactory(result=content)
        self._patch(monkeypatch, fac)

        r = f.resize(64, 48, mode="crop")

        (src, dw, dh, kw), = fac.clients[0].calls
        assert (dw, dh) == (64, 128)                # cover box, not target
        assert kw.get("scaling") == "stretch"
        np.testing.assert_array_equal(r.image, content[40:88])

    def test_crop_nv12_crops_both_planes(self, monkeypatch):
        # 16x32 NV12 -> 64x48 crop: box 64x128; Y rows 40..88, UV rows 20..44
        f = make_keepfd_frame(16, 32, "NV12")
        y = np.full((128, 64), 30, dtype=np.uint8)
        uv = np.full((64, 64), 77, dtype=np.uint8)
        fac = _FakeDspFactory(result=np.vstack([y, uv]))
        self._patch(monkeypatch, fac)

        r = f.resize(64, 48, mode="crop")

        assert r.image.shape == (72, 64)            # NV12 2D layout
        np.testing.assert_array_equal(r.image[:48], np.full((48, 64), 30))
        np.testing.assert_array_equal(r.image[48:], np.full((24, 64), 77))

    def test_dsp_error_falls_back_to_cpu(self, monkeypatch):
        from neoruntime_ipc_sdk.dsp import DspError
        twin = make_nv12_frame(64, 48)
        f = make_keepfd_frame(64, 48, "NV12", image=twin.image.copy())
        fac = _FakeDspFactory(error=DspError("service down"))
        self._patch(monkeypatch, fac)

        r = f.resize(32, 32)

        assert len(fac.clients) == 1                # fast path was attempted
        np.testing.assert_array_equal(r.image, twin.resize(32, 32).image)

    def test_pixel_frame_never_touches_dsp(self, monkeypatch):
        fac = _FakeDspFactory(result=None)
        self._patch(monkeypatch, fac)
        f = make_rgb_frame(64, 48)

        r = f.resize(32, 24)

        assert fac.clients == []
        assert r.image.shape == (24, 32, 3)

    def test_closed_handle_skips_dsp(self, monkeypatch):
        fac = _FakeDspFactory(result=None)
        self._patch(monkeypatch, fac)
        f = make_keepfd_frame(64, 48, "GRAY8",
                              image=np.full((48, 64), 9, np.uint8))
        f.handle.close()

        r = f.resize(32, 24)

        assert fac.clients == []
        assert r.image.shape == (24, 32)

    def test_unsupported_format_skips_dsp(self, monkeypatch):
        fac = _FakeDspFactory(result=None)
        self._patch(monkeypatch, fac)
        f = make_keepfd_frame(64, 48, "RGBA",
                              image=np.zeros((48, 64, 4), np.uint8))

        r = f.resize(32, 24)

        assert fac.clients == []
        assert r.image.shape == (24, 32, 4)
