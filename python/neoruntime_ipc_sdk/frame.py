"""Frame types and pixel-format helpers.

:class:`Frame` (a decoded or fd-backed video frame), :class:`FrameHandle`
(retained dma-buf planes) and the pixel-format tables shared by the media
clients and the DSP path. Split out of media.py; media.py re-exports the
public surface for backwards compatibility.
"""

from __future__ import annotations

import fcntl
import logging
import mmap
import os
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("neoruntime_ipc_sdk.frame")

__all__ = [
    "Frame",
    "FrameHandle",
    "PixelFormat",
    "PIXEL_FORMAT_NAMES",
    "StreamInfo",
]


class PixelFormat(IntEnum):
    NV12 = 0
    NV21 = 1
    RGB = 2
    BGR = 3
    RGBA = 4
    BGRA = 5
    GRAY8 = 6
    YUYV = 7


PIXEL_FORMAT_NAMES = {
    PixelFormat.NV12: "NV12",
    PixelFormat.NV21: "NV21",
    PixelFormat.RGB: "RGB",
    PixelFormat.BGR: "BGR",
    PixelFormat.RGBA: "RGBA",
    PixelFormat.BGRA: "BGRA",
    PixelFormat.GRAY8: "GRAY8",
    PixelFormat.YUYV: "YUYV",
}

# Formats stored as plain pixel arrays (crop/resize via numpy slicing)
_PACKED_FORMATS = ("RGB", "BGR", "RGBA", "BGRA", "GRAY8")
# Planar/semi-planar YUV formats (crop/resize with per-plane handling)
_YUV_FORMATS = ("NV12", "NV21")
# Formats Frame.resize can hand to the DSP as a zero-copy source
# (must stay in sync with dsp._FRAME_FMT_TO_DSP)
_DSP_RESIZE_FORMATS = ("NV12", "RGB", "BGR", "GRAY8")


def _resize_array(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a uint8 2D/3D array to (height, width).

    Uses cv2 when available (INTER_AREA down / INTER_LINEAR up); falls back
    to pure-numpy nearest-neighbour indexing so the SDK works without cv2.
    """
    if img.shape[0] == height and img.shape[1] == width:
        return img
    try:
        import cv2

        interp = (
            cv2.INTER_AREA if (height < img.shape[0] and width < img.shape[1]) else cv2.INTER_LINEAR
        )
        return cv2.resize(img, (width, height), interpolation=interp)
    except ImportError:
        rows = np.arange(height) * img.shape[0] // height
        cols = np.arange(width) * img.shape[1] // width
        if img.ndim == 2:
            return img[rows[:, None], cols[None, :]]
        return img[rows[:, None], cols[None, :], :]


def _even(value: int) -> int:
    """Round down to the nearest even number, minimum 2 (YUV plane safety)."""
    return max(2, value - (value % 2))


def _encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    """Encode an RGB uint8 array as JPEG bytes. cv2 first, Pillow fallback."""
    try:
        import cv2

        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise OSError("cv2.imencode failed to encode JPEG")
        return buf.tobytes()
    except ImportError:
        import io

        from PIL import Image

        out = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(out, format="JPEG", quality=int(quality))
        return out.getvalue()


class FrameHandle:
    """Retained dma-buf backing store for one received frame (SDK-1).

    In keep-fd mode the per-plane dma-buf fds the daemon passed with the
    FRAME message are kept open here and the RELEASE message — the
    daemon's buffer-recycling ticket — is deferred until :meth:`close`.
    CPU access to the pixels must go through :meth:`Frame.to_array`,
    which applies the required DMA_BUF_IOCTL_SYNC read fences. The fds
    can also be handed to a :class:`~neoruntime_ipc_sdk.dsp.DspClient`
    job as-is for a zero-copy hardware path (``resize_hw(frame, ...)``);
    the geometry carried here is what the daemon needs to import them.
    """

    def __init__(
        self,
        fds: list[int],
        strides,
        plane_sizes,
        frame_id: int,
        on_release: Callable[[FrameHandle], None] | None = None,
        width: int = 0,
        height: int = 0,
        format: str = "",
    ):
        self.fds = list(fds)
        self.strides = tuple(strides)
        self.plane_sizes = tuple(plane_sizes)
        self.frame_id = frame_id
        self.width = width
        self.height = height
        self.format = format
        self._on_release = on_release
        self._closed = False

    @property
    def closed(self) -> bool:
        """True once :meth:`close` ran (fds gone, frame released)."""
        return self._closed

    @property
    def fd(self) -> int:
        """Convenience accessor for the first plane's fd."""
        return self.fds[0]

    def close(self) -> None:
        """Close the fds and send the deferred RELEASE (idempotent)."""
        if self._closed:
            return
        self._closed = True
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        if self._on_release is not None:
            try:
                self._on_release(self)
            except Exception:
                logger.exception("FrameHandle: release callback failed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"FrameHandle(frame_id={self.frame_id}, "
            f"{self.width}x{self.height} {self.format}, "
            f"fds={len(self.fds)}, released={self._closed})"
        )


@dataclass
class Frame:
    sequence: int
    timestamp_ns: int
    width: int
    height: int
    format: str
    image: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)
    handle: FrameHandle | None = None

    @property
    def data(self) -> np.ndarray | None:
        """Alias for image, returns raw frame data as flat numpy array."""
        if self.image is None:
            if self.handle is None:
                return None
            return self.to_array().flatten()
        return self.image.flatten()

    def to_array(self) -> np.ndarray:
        """Return the raw pixel buffer, materializing a retained fd.

        For keep-fd frames this maps the dma-buf planes once (with the
        DMA_BUF_IOCTL_SYNC read fences) and caches the copy in
        ``image``; later calls return the cache without re-mapping.
        """
        if self.image is None:
            if self.handle is None:
                raise ValueError("Frame has no image data and no retained fd")
            self.image = _decode_raw(
                _materialize_handle(self.handle), self.width, self.height, self.format
            )
        return self.image

    def release(self) -> None:
        """Release a retained fd frame back to the daemon (idempotent).

        No-op for frames that were copied on receive.
        """
        if self.handle is not None:
            self.handle.close()

    def to_rgb(self) -> np.ndarray:
        arr = self.to_array()
        if self.format == "RGB":
            return arr
        elif self.format == "BGR":
            return arr[:, :, ::-1]
        elif self.format == "NV12":
            return self._nv12_to_rgb()
        elif self.format == "GRAY8":
            return np.stack([arr] * 3, axis=-1)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _nv12_to_rgb(self) -> np.ndarray:
        try:
            import cv2

            return cv2.cvtColor(self.image, cv2.COLOR_YUV2RGB_NV12)
        except ImportError:
            return self._nv12_to_rgb_pure()

    def _nv12_to_rgb_pure(self) -> np.ndarray:
        h, w = self.height, self.width
        y = self.image[:h, :].astype(np.float32)
        uv = self.image[h:, :].reshape(h // 2, w // 2, 2)
        u, v = uv[:, :, 0], uv[:, :, 1]

        u = u.repeat(2, axis=0).repeat(2, axis=1)
        v = v.repeat(2, axis=0).repeat(2, axis=1)

        y = y - 16
        u = u - 128
        v = v - 128

        r = np.clip(1.164 * y + 1.596 * v, 0, 255).astype(np.uint8)
        g = np.clip(1.164 * y - 0.813 * v - 0.391 * u, 0, 255).astype(np.uint8)
        b = np.clip(1.164 * y + 2.018 * u, 0, 255).astype(np.uint8)

        return np.stack([r, g, b], axis=-1)

    def crop(self, x: int, y: int, width: int, height: int) -> Frame:
        """Return a new Frame cropped to the given pixel rectangle.

        NV12/NV21 require even x, y, width, height (chroma subsampling).
        The original Frame is left untouched.
        """
        if width <= 0 or height <= 0:
            raise ValueError("crop width/height must be positive")
        if x < 0 or y < 0 or x + width > self.width or y + height > self.height:
            raise ValueError(
                f"crop ({x},{y},{width}x{height}) out of bounds for "
                f"{self.width}x{self.height} frame"
            )
        fmt = self.format
        arr = self.to_array()
        if fmt in _PACKED_FORMATS:
            sub = np.ascontiguousarray(arr[y : y + height, x : x + width])
        elif fmt in _YUV_FORMATS:
            if x % 2 or y % 2 or width % 2 or height % 2:
                raise ValueError(f"{fmt} crop requires even x, y, width, height")
            y_plane = arr[: self.height]
            uv_plane = arr[self.height :]
            new_y = y_plane[y : y + height, x : x + width]
            new_uv = uv_plane[y // 2 : (y + height) // 2, (x // 2) * 2 : (x // 2 + width // 2) * 2]
            sub = np.ascontiguousarray(np.vstack([new_y, new_uv]))
        else:
            raise ValueError(f"crop not supported for format: {fmt}")
        return Frame(
            sequence=self.sequence,
            timestamp_ns=self.timestamp_ns,
            width=width,
            height=height,
            format=fmt,
            image=sub,
            metadata=dict(self.metadata),
        )

    def resize(
        self, width: int, height: int, mode: str = "letterbox", pad_value: int = 114
    ) -> Frame:
        """Return a new Frame resized to width x height.

        Modes:
            - "letterbox": fit inside, preserve aspect ratio, pad with
              pad_value (NV12 pads luma with pad_value and chroma with
              neutral 128). Default.
            - "stretch": fill exactly, aspect ratio not preserved.
            - "crop": scale to cover, center-crop the overflow.

        NV12/NV21 require even target dimensions. Frames received with
        keep_fd=True are scaled on the DSP without materializing their
        dma-bufs first (falling back to the CPU path when the DSP
        service is unavailable). cv2 accelerates the CPU path when
        available; a pure-numpy nearest-neighbour path is the fallback.
        """
        if width <= 0 or height <= 0:
            raise ValueError("resize width/height must be positive")
        if mode not in ("letterbox", "stretch", "crop"):
            raise ValueError(f"unsupported resize mode: {mode}")
        fmt = self.format
        if fmt in _YUV_FORMATS and (width % 2 or height % 2):
            raise ValueError(f"{fmt} resize requires even width/height")
        image = self._hw_resize(width, height, mode, pad_value)
        if image is None:
            self.to_array()  # materialize a retained fd before slicing planes
            if fmt in _YUV_FORMATS:
                image = self._resize_yuv(width, height, mode, pad_value)
            elif fmt in _PACKED_FORMATS:
                image = self._resize_packed(width, height, mode, pad_value)
            else:
                raise ValueError(f"resize not supported for format: {fmt}")
        return Frame(
            sequence=self.sequence,
            timestamp_ns=self.timestamp_ns,
            width=width,
            height=height,
            format=fmt,
            image=image,
            metadata=dict(self.metadata),
        )

    def _hw_resize(self, dw: int, dh: int, mode: str, pad: int) -> np.ndarray | None:
        """Zero-copy DSP scale for keep-fd frames; None means "use CPU".

        Returns the resized pixels when the frame carries a dma-buf
        handle the daemon can import, or None for every other case (no
        handle, closed handle, format the DSP cannot take, service
        down) so :meth:`resize` can fall back to its CPU path.

        Every mode scales on the DSP into the exact box the CPU path
        computes, then pads/crops on the CPU, so both paths agree on
        content placement:

        * "stretch"  — one DSP stretch to the target, used as-is.
        * "letterbox" — DSP stretch to the fitted box; CPU pads with
          ``pad``. (The daemon's own letterbox pads Y=U=V=0 — green in
          YUV, not a neutral pad — so it is never used here.)
        * "crop" — DSP stretch to the cover box; CPU center-crops.
          (The vendor SCALE_AND_CROP picks its own rounding; on device
          it disagreed with the CPU placement by ~21 luma levels.)

        Hot loops should hold a :class:`~neoruntime_ipc_sdk.dsp.DspClient`
        open and call ``resize_hw`` directly instead of paying this
        method's per-call client setup.
        """
        handle = self.handle
        if handle is None or handle.closed:
            return None
        if self.format not in _DSP_RESIZE_FORMATS:
            return None
        yuv = self.format in _YUV_FORMATS
        # (rw, rh, ox, oy) must match _resize_yuv/_resize_packed placement,
        # or the two paths would disagree on the same frame
        sw, sh = self.width, self.height
        if mode == "stretch":
            rw, rh, ox, oy = dw, dh, 0, 0
        elif mode == "letterbox":
            scale = min(dw / sw, dh / sh)
            rw = max(1, int(round(sw * scale)))
            rh = max(1, int(round(sh * scale)))
            if yuv:
                rw, rh = _even(rw), _even(rh)
            ox, oy = (dw - rw) // 2, (dh - rh) // 2
            if yuv:
                ox, oy = ox & ~1, oy & ~1  # chroma-aligned pad offsets
        else:  # "crop": scale to cover, center-crop the overflow
            scale = max(dw / sw, dh / sh)
            rw = max(dw, int(round(sw * scale)))
            rh = max(dh, int(round(sh * scale)))
            if yuv:
                rw, rh = _even(rw), _even(rh)
            ox, oy = (rw - dw) // 2, (rh - dh) // 2
        try:
            from .dsp import DspClient, DspError  # lazy: dsp imports media

            with DspClient() as dsp:
                content = dsp.resize_hw(self, rw, rh, scaling="stretch")
        except DspError as exc:
            logger.debug("DSP resize fast path unavailable (%s); CPU path", exc)
            return None
        if mode == "stretch":
            return content
        if not yuv:
            if mode == "crop":
                return np.ascontiguousarray(content[oy : oy + dh, ox : ox + dw])
            canvas = (
                np.full((dh, dw), pad, dtype=np.uint8)
                if content.ndim == 2
                else np.full((dh, dw, content.shape[2]), [pad] * content.shape[2], dtype=np.uint8)
            )
            canvas[oy : oy + rh, ox : ox + rw] = content
            return canvas
        if mode == "crop":
            return np.ascontiguousarray(
                np.vstack(
                    [
                        content[:rh][oy : oy + dh, ox : ox + dw],
                        content[rh:][oy // 2 : oy // 2 + dh // 2, ox : ox + dw],
                    ]
                )
            )
        canvas_y = np.full((dh, dw), pad, dtype=np.uint8)
        canvas_uv = np.full((dh // 2, dw), 128, dtype=np.uint8)
        canvas_y[oy : oy + rh, ox : ox + rw] = content[:rh]
        canvas_uv[oy // 2 : oy // 2 + rh // 2, ox : ox + rw] = content[rh:]
        return np.vstack([canvas_y, canvas_uv])

    def _resize_packed(self, dw: int, dh: int, mode: str, pad: int) -> np.ndarray:
        src = self.image
        sw, sh = self.width, self.height
        if mode == "stretch":
            return _resize_array(src, dw, dh)
        if mode == "letterbox":
            scale = min(dw / sw, dh / sh)
            rw = max(1, int(round(sw * scale)))
            rh = max(1, int(round(sh * scale)))
            content = _resize_array(src, rw, rh)
            ox, oy = (dw - rw) // 2, (dh - rh) // 2
            if src.ndim == 2:
                canvas = np.full((dh, dw), pad, dtype=np.uint8)
            else:
                fill = [pad] * src.shape[2]
                if self.format in ("RGBA", "BGRA"):
                    fill[-1] = 255
                canvas = np.full((dh, dw, src.shape[2]), fill, dtype=np.uint8)
            canvas[oy : oy + rh, ox : ox + rw] = content
            return canvas
        # mode == "crop": scale to cover, then center-crop
        scale = max(dw / sw, dh / sh)
        rw = max(dw, int(round(sw * scale)))
        rh = max(dh, int(round(sh * scale)))
        tmp = _resize_array(src, rw, rh)
        ox, oy = (rw - dw) // 2, (rh - dh) // 2
        return np.ascontiguousarray(tmp[oy : oy + dh, ox : ox + dw])

    def _resize_yuv(self, dw: int, dh: int, mode: str, pad: int) -> np.ndarray:
        sw, sh = self.width, self.height
        y_plane = self.image[:sh]
        uv_plane = np.ascontiguousarray(self.image[sh:])
        if uv_plane.shape[0] * 2 != sh or uv_plane.shape[1] != sw:
            raise ValueError(
                f"{self.format} buffer shape {self.image.shape} does not match {sw}x{sh} frame"
            )

        def uv_resize(uv: np.ndarray, w: int, h: int) -> np.ndarray:
            # Packed interleaved chroma: resize as (h/2, w/2, 2) image so
            # U and V stay on separate channels, then flatten back.
            src_h, src_w = uv.shape
            paired = uv.reshape(src_h, src_w // 2, 2)
            out = _resize_array(paired, w // 2, h // 2)
            return out.reshape(h // 2, w)

        if mode == "stretch":
            return np.vstack(
                [
                    _resize_array(y_plane, dw, dh),
                    uv_resize(uv_plane, dw, dh),
                ]
            )
        if mode == "letterbox":
            scale = min(dw / sw, dh / sh)
            rw = _even(int(round(sw * scale)))
            rh = _even(int(round(sh * scale)))
            content_y = _resize_array(y_plane, rw, rh)
            content_uv = uv_resize(uv_plane, rw, rh)
            ox, oy = (dw - rw) // 2 & ~1, (dh - rh) // 2 & ~1
            canvas_y = np.full((dh, dw), pad, dtype=np.uint8)
            canvas_uv = np.full((dh // 2, dw), 128, dtype=np.uint8)
            canvas_y[oy : oy + rh, ox : ox + rw] = content_y
            canvas_uv[oy // 2 : oy // 2 + rh // 2, ox : ox + rw] = content_uv
            return np.vstack([canvas_y, canvas_uv])
        # mode == "crop": scale to cover, then center-crop both planes
        scale = max(dw / sw, dh / sh)
        rw = _even(max(dw, int(round(sw * scale))))
        rh = _even(max(dh, int(round(sh * scale))))
        tmp_y = _resize_array(y_plane, rw, rh)
        tmp_uv = uv_resize(uv_plane, rw, rh)
        ox, oy = (rw - dw) // 2, (rh - dh) // 2
        return np.vstack(
            [
                tmp_y[oy : oy + dh, ox : ox + dw],
                tmp_uv[oy // 2 : oy // 2 + dh // 2, ox : ox + dw],
            ]
        )

    def to_jpeg_bytes(self, quality: int = 85) -> bytes:
        """Encode the frame as JPEG bytes (RGB conversion first)."""
        return _encode_jpeg(self.to_rgb(), quality)

    def save(self, path: str) -> None:
        if path.lower().endswith((".jpg", ".jpeg")):
            with open(path, "wb") as fh:
                fh.write(_encode_jpeg(self.to_rgb()))
            return
        try:
            import cv2

            rgb = self.to_rgb()
            bgr = rgb[:, :, ::-1]
            cv2.imwrite(path, bgr)
        except ImportError:
            from PIL import Image

            rgb = self.to_rgb()
            Image.fromarray(rgb).save(path)


@dataclass
class StreamInfo:
    stream_id: str
    width: int
    height: int
    format: str
    fps: float
    buffer_count: int


# DMA_BUF_IOCTL_SYNC (linux/dma-buf.h): _IOW('b', 0, u64) on 64-bit.
_DMA_BUF_IOCTL_SYNC = 0x40086200
_DMA_BUF_SYNC_READ = 1 << 0
_DMA_BUF_SYNC_WRITE = 2 << 0
_DMA_BUF_SYNC_START = 1 << 2
_DMA_BUF_SYNC_END = 2 << 2


def _dma_buf_sync(fd: int, flags: int) -> None:
    """Fence CPU access to a dma-buf (HAL-3 discipline).

    Device-written dma-bufs must be fenced READ|START before a CPU read
    and READ|END after it, or the read can serve stale cache lines. On
    non-dma-buf fds (memfd in tests, plain anon memory) the ioctl raises
    ENOTTY — there is nothing to fence, so OSError is swallowed.
    """
    try:
        fcntl.ioctl(fd, _DMA_BUF_IOCTL_SYNC, struct.pack("=Q", flags))
    except OSError:
        pass


def _decode_raw(raw: np.ndarray, w: int, h: int, fmt: str) -> np.ndarray:
    """Reshape a flat frame buffer into its per-format image layout."""
    if fmt in ("NV12", "NV21"):
        return raw.reshape(h * 3 // 2, w)
    elif fmt in ("RGB", "BGR"):
        return raw.reshape(h, w, 3)
    elif fmt in ("RGBA", "BGRA"):
        return raw.reshape(h, w, 4)
    elif fmt == "GRAY8":
        return raw.reshape(h, w)
    elif fmt == "YUYV":
        return raw.reshape(h, w, 2)
    return raw.reshape(h, w, 3)


def _materialize_handle(handle: FrameHandle) -> np.ndarray:
    """Map a retained frame's planes and return them as one numpy copy."""
    planes = []
    for i, fd in enumerate(handle.fds):
        size = handle.plane_sizes[i] if i < len(handle.plane_sizes) else 0
        _dma_buf_sync(fd, _DMA_BUF_SYNC_READ | _DMA_BUF_SYNC_START)
        actual_size = os.fstat(fd).st_size
        buf = mmap.mmap(fd, actual_size, access=mmap.ACCESS_READ)
        try:
            plane = np.frombuffer(buf, dtype=np.uint8)[:size].copy()
        finally:
            buf.close()
        _dma_buf_sync(fd, _DMA_BUF_SYNC_READ | _DMA_BUF_SYNC_END)
        planes.append(plane)
    return np.concatenate(planes) if len(planes) > 1 else planes[0]
