"""
Media Client - Zero-copy video stream access via DMA-BUF FD passing
and encoded stream access via EncodedPublisher UDS sockets.
"""

import logging
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

logger = logging.getLogger("neoruntime_ipc_sdk.media")


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


def _resize_array(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a uint8 2D/3D array to (height, width).

    Uses cv2 when available (INTER_AREA down / INTER_LINEAR up); falls back
    to pure-numpy nearest-neighbour indexing so the SDK works without cv2.
    """
    if img.shape[0] == height and img.shape[1] == width:
        return img
    try:
        import cv2
        interp = cv2.INTER_AREA if (height < img.shape[0] and width < img.shape[1]) \
            else cv2.INTER_LINEAR
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
        ok, buf = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise IOError("cv2.imencode failed to encode JPEG")
        return buf.tobytes()
    except ImportError:
        import io
        from PIL import Image
        out = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(
            out, format="JPEG", quality=int(quality))
        return out.getvalue()


class FrameHandle:
    """Retained dma-buf backing store for one received frame (SDK-1).

    In keep-fd mode the per-plane dma-buf fds the daemon passed with the
    FRAME message are kept open here and the RELEASE message — the
    daemon's buffer-recycling ticket — is deferred until :meth:`close`.
    CPU access to the pixels must go through :meth:`Frame.to_array`,
    which applies the required DMA_BUF_IOCTL_SYNC read fences. The fds
    can also be handed to a future DspClient job as-is for a zero-copy
    hardware path.
    """

    def __init__(self, fds: List[int], strides, plane_sizes, frame_id: int,
                 on_release: Optional[Callable[["FrameHandle"], None]] = None):
        self.fds = list(fds)
        self.strides = tuple(strides)
        self.plane_sizes = tuple(plane_sizes)
        self.frame_id = frame_id
        self._on_release = on_release
        self._closed = False

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
        return (f"FrameHandle(frame_id={self.frame_id}, "
                f"fds={len(self.fds)}, released={self._closed})")


@dataclass
class Frame:
    sequence: int
    timestamp_ns: int
    width: int
    height: int
    format: str
    image: Optional[np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    handle: Optional[FrameHandle] = None

    @property
    def data(self) -> Optional[np.ndarray]:
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
                _materialize_handle(self.handle),
                self.width, self.height, self.format)
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
    
    def crop(self, x: int, y: int, width: int, height: int) -> "Frame":
        """Return a new Frame cropped to the given pixel rectangle.

        NV12/NV21 require even x, y, width, height (chroma subsampling).
        The original Frame is left untouched.
        """
        if width <= 0 or height <= 0:
            raise ValueError("crop width/height must be positive")
        if x < 0 or y < 0 or x + width > self.width or y + height > self.height:
            raise ValueError(
                f"crop ({x},{y},{width}x{height}) out of bounds for "
                f"{self.width}x{self.height} frame")
        fmt = self.format
        arr = self.to_array()
        if fmt in _PACKED_FORMATS:
            sub = np.ascontiguousarray(
                arr[y:y + height, x:x + width])
        elif fmt in _YUV_FORMATS:
            if x % 2 or y % 2 or width % 2 or height % 2:
                raise ValueError(
                    f"{fmt} crop requires even x, y, width, height")
            y_plane = arr[:self.height]
            uv_plane = arr[self.height:]
            new_y = y_plane[y:y + height, x:x + width]
            new_uv = uv_plane[y // 2:(y + height) // 2,
                              (x // 2) * 2:(x // 2 + width // 2) * 2]
            sub = np.ascontiguousarray(np.vstack([new_y, new_uv]))
        else:
            raise ValueError(f"crop not supported for format: {fmt}")
        return Frame(sequence=self.sequence, timestamp_ns=self.timestamp_ns,
                     width=width, height=height, format=fmt, image=sub,
                     metadata=dict(self.metadata))

    def resize(self, width: int, height: int, mode: str = "letterbox",
               pad_value: int = 114) -> "Frame":
        """Return a new Frame resized to width x height.

        Modes:
            "letterbox" — fit inside, preserve aspect ratio, pad with
                          pad_value (NV12 pads luma with pad_value and
                          chroma with neutral 128). Default.
            "stretch"   — fill exactly, aspect ratio not preserved.
            "crop"      — scale to cover, center-crop the overflow.

        NV12/NV21 require even target dimensions. cv2 accelerates when
        available; a pure-numpy nearest-neighbour path is the fallback.
        """
        if width <= 0 or height <= 0:
            raise ValueError("resize width/height must be positive")
        if mode not in ("letterbox", "stretch", "crop"):
            raise ValueError(f"unsupported resize mode: {mode}")
        self.to_array()  # materialize a retained fd before slicing planes
        fmt = self.format
        if fmt in _YUV_FORMATS:
            if width % 2 or height % 2:
                raise ValueError(f"{fmt} resize requires even width/height")
            image = self._resize_yuv(width, height, mode, pad_value)
        elif fmt in _PACKED_FORMATS:
            image = self._resize_packed(width, height, mode, pad_value)
        else:
            raise ValueError(f"resize not supported for format: {fmt}")
        return Frame(sequence=self.sequence, timestamp_ns=self.timestamp_ns,
                     width=width, height=height, format=fmt, image=image,
                     metadata=dict(self.metadata))

    def _resize_packed(self, dw: int, dh: int, mode: str,
                       pad: int) -> np.ndarray:
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
                canvas = np.full((dh, dw, src.shape[2]), fill,
                                 dtype=np.uint8)
            canvas[oy:oy + rh, ox:ox + rw] = content
            return canvas
        # mode == "crop": scale to cover, then center-crop
        scale = max(dw / sw, dh / sh)
        rw = max(dw, int(round(sw * scale)))
        rh = max(dh, int(round(sh * scale)))
        tmp = _resize_array(src, rw, rh)
        ox, oy = (rw - dw) // 2, (rh - dh) // 2
        return np.ascontiguousarray(
            tmp[oy:oy + dh, ox:ox + dw])

    def _resize_yuv(self, dw: int, dh: int, mode: str,
                    pad: int) -> np.ndarray:
        sw, sh = self.width, self.height
        y_plane = self.image[:sh]
        uv_plane = np.ascontiguousarray(self.image[sh:])
        if uv_plane.shape[0] * 2 != sh or uv_plane.shape[1] != sw:
            raise ValueError(
                f"{self.format} buffer shape {self.image.shape} does not "
                f"match {sw}x{sh} frame")

        def uv_resize(uv: np.ndarray, w: int, h: int) -> np.ndarray:
            # Packed interleaved chroma: resize as (h/2, w/2, 2) image so
            # U and V stay on separate channels, then flatten back.
            src_h, src_w = uv.shape
            paired = uv.reshape(src_h, src_w // 2, 2)
            out = _resize_array(paired, w // 2, h // 2)
            return out.reshape(h // 2, w)

        if mode == "stretch":
            return np.vstack([
                _resize_array(y_plane, dw, dh),
                uv_resize(uv_plane, dw, dh),
            ])
        if mode == "letterbox":
            scale = min(dw / sw, dh / sh)
            rw = _even(int(round(sw * scale)))
            rh = _even(int(round(sh * scale)))
            content_y = _resize_array(y_plane, rw, rh)
            content_uv = uv_resize(uv_plane, rw, rh)
            ox, oy = (dw - rw) // 2 & ~1, (dh - rh) // 2 & ~1
            canvas_y = np.full((dh, dw), pad, dtype=np.uint8)
            canvas_uv = np.full((dh // 2, dw), 128, dtype=np.uint8)
            canvas_y[oy:oy + rh, ox:ox + rw] = content_y
            canvas_uv[oy // 2:oy // 2 + rh // 2, ox:ox + rw] = content_uv
            return np.vstack([canvas_y, canvas_uv])
        # mode == "crop": scale to cover, then center-crop both planes
        scale = max(dw / sw, dh / sh)
        rw = _even(max(dw, int(round(sw * scale))))
        rh = _even(max(dh, int(round(sh * scale))))
        tmp_y = _resize_array(y_plane, rw, rh)
        tmp_uv = uv_resize(uv_plane, rw, rh)
        ox, oy = (rw - dw) // 2, (rh - dh) // 2
        return np.vstack([
            tmp_y[oy:oy + dh, ox:ox + dw],
            tmp_uv[oy // 2:oy // 2 + dh // 2, ox:ox + dw],
        ])

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


# ---------------------------------------------------------------------------
# FD Protocol constants (must match fd_protocol.h)
# ---------------------------------------------------------------------------

_FD_PUB_MSG_SUBSCRIBE   = 1
_FD_PUB_MSG_UNSUBSCRIBE = 2
_FD_PUB_MSG_FRAME       = 3
_FD_PUB_MSG_RELEASE     = 4
_FD_PUB_MSG_OK          = 5
_FD_PUB_MSG_ERROR       = 6

_FD_PUB_MAX_STREAM_NAME = 64
_FD_PUB_MAX_FDS         = 3
_FD_PUB_PROTOCOL_VERSION = 1

# struct FdPubMsgHeader { uint32 type; uint32 size; }
_HDR_FMT = '<II'
_HDR_SIZE = struct.calcsize(_HDR_FMT)

# struct FdPubSubscribeMsg { header(8) + uint32 version + char[64] stream_name }
_SUB_FMT = '<II I 64s'
_SUB_SIZE = struct.calcsize(_SUB_FMT)

# struct FdPubFrameMsg (aarch64 pads to 8-byte alignment: 76 data + 4 padding = 80)
_FRAME_FMT = '<II QQQ IIII 3I 3I I 4x'
_FRAME_SIZE = struct.calcsize(_FRAME_FMT)

# struct FdPubReleaseMsg { header(8) + uint64 frame_id }
_REL_FMT = '<II Q'
_REL_SIZE = struct.calcsize(_REL_FMT)

# struct FdPubResponseMsg { header(8) + int32 code }
_RESP_FMT = '<II i'
_RESP_SIZE = struct.calcsize(_RESP_FMT)

@dataclass
class EncodedFrame:
    """Encoded video frame (H.264/H.265) from the EncodedPublisher."""
    codec: int          # 0=h264, 1=h265
    flags: int          # bit0 = keyframe
    pts_ns: int         # Presentation timestamp (nanoseconds)
    width: int
    height: int
    dts_ns: int         # Decode timestamp (nanoseconds)
    data: bytes         # Encoded NALU payload

    @property
    def is_keyframe(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def codec_name(self) -> str:
        return {0: "h264", 1: "h265"}.get(self.codec, f"unknown({self.codec})")


# Encoded video header: 30 bytes, little-endian
# [0:4]   uint32  total_size (header + payload)
# [4]     uint8   codec (0=h264, 1=h265)
# [5]     uint8   flags (bit0 = keyframe)
# [6:14]  uint64  pts_ns
# [14:18] uint32  width
# [18:22] uint32  height
# [22:30] uint64  dts_ns
_ENC_HEADER_SIZE = 30
_ENC_HEADER_FMT = "<I BB Q II Q"


class EncodedStreamClient:
    """Read encoded video frames from an EncodedPublisher UDS socket.

    Connects to sockets like ``/run/aipc/encoded/main.sock`` and yields
    :class:`EncodedFrame` objects containing H.264/H.265 NAL units.

    Usage::

        client = EncodedStreamClient("/run/aipc/encoded/main.sock")
        for frame in client.subscribe():
            print(f"{frame.codec_name} {frame.width}x{frame.height} "
                  f"keyframe={frame.is_keyframe} {len(frame.data)}B")
    """

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        sock.settimeout(5.0)
        logger.info("EncodedStreamClient: connected to %s", self.socket_path)
        return sock

    def _get_sock(self) -> socket.socket:
        with self._lock:
            if self._sock is None:
                self._sock = self._connect()
            return self._sock

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("EncodedStreamClient: socket closed")
            buf.extend(chunk)
        return bytes(buf)

    def _recv_frame(self, sock: socket.socket) -> Optional[EncodedFrame]:
        try:
            header_data = self._recv_exact(sock, _ENC_HEADER_SIZE)
        except (ConnectionError, OSError):
            return None

        if len(header_data) < _ENC_HEADER_SIZE:
            return None

        values = struct.unpack(_ENC_HEADER_FMT, header_data)
        total_size = values[0]
        codec = values[1]
        flags = values[2]
        pts_ns = values[3]
        width = values[4]
        height = values[5]
        dts_ns = values[6]

        payload_size = total_size - _ENC_HEADER_SIZE
        if payload_size < 0 or payload_size > 50 * 1024 * 1024:
            logger.warning("EncodedStreamClient: bogus payload_size=%d", payload_size)
            return None

        try:
            payload = self._recv_exact(sock, payload_size) if payload_size > 0 else b""
        except (ConnectionError, OSError):
            return None

        return EncodedFrame(
            codec=codec, flags=flags, pts_ns=pts_ns,
            width=width, height=height, dts_ns=dts_ns,
            data=payload,
        )

    def _reconnect(self) -> socket.socket:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
            self._sock = self._connect()
            return self._sock

    def get_frame(self, timeout_ms: int = 5000) -> Optional[EncodedFrame]:
        """Get a single encoded frame. Returns None on timeout."""
        sock = self._get_sock()
        sock.settimeout(timeout_ms / 1000.0)
        try:
            return self._recv_frame(sock)
        except socket.timeout:
            return None

    def subscribe(self, reconnect: bool = True) -> Iterator[EncodedFrame]:
        """Yield encoded frames continuously. Auto-reconnects if enabled."""
        sock = self._get_sock()
        while True:
            frame = self._recv_frame(sock)
            if frame is not None:
                yield frame
                continue
            if not reconnect:
                break
            logger.info("EncodedStreamClient: reconnecting...")
            time.sleep(0.5)
            try:
                sock = self._reconnect()
            except OSError:
                logger.warning("EncodedStreamClient: reconnect failed, retrying in 2s")
                time.sleep(2.0)

    def on_frame(self, callback: Callable[[EncodedFrame], None]) -> threading.Thread:
        """Start a background thread that calls callback for each frame."""
        def _run():
            for frame in self.subscribe():
                try:
                    callback(frame)
                except Exception:
                    logger.exception("EncodedStreamClient: callback error")
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
        logger.info("EncodedStreamClient: closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


import fcntl  # noqa: E402
import mmap  # noqa: E402
import socket as _socket  # noqa: E402
import weakref  # noqa: E402


def _recvmsg_with_fds(sock: _socket.socket, bufsize: int, max_fds: int = _FD_PUB_MAX_FDS):
    """Receive data + SCM_RIGHTS file descriptors via recvmsg."""
    fds_space = _socket.CMSG_SPACE(max_fds * struct.calcsize('i'))
    data, ancdata, _flags, _addr = sock.recvmsg(bufsize, fds_space)
    fds: list[int] = []
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == _socket.SOL_SOCKET and cmsg_type == _socket.SCM_RIGHTS:
            n = len(cmsg_data) // struct.calcsize('i')
            fds.extend(struct.unpack(f'{n}i', cmsg_data[:n * struct.calcsize('i')]))
    return data, fds


def _sendmsg_plain(sock: _socket.socket, data: bytes) -> None:
    sock.sendall(data)


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


class FdMediaClient:
    """Zero-copy media client using DMA-BUF FD passing over Unix Domain Socket."""

    def __init__(self, socket_path: str | None = None):
        if socket_path is None:
            socket_path = os.getenv("CAMERA_SOCK_PATH", "/run/aipc/camera.sock")
        self.socket_path = socket_path
        self._streams: dict[str, _socket.socket] = {}
        self._lock = threading.Lock()
        # Retained keep-fd handles. WeakSet: tracking without extending
        # lifetime — a dropped Frame is GC-released back to the daemon.
        self._retained: "weakref.WeakSet[FrameHandle]" = weakref.WeakSet()

    # PLACEHOLDER_FDMEDIACLIENT_METHODS

    def _connect_stream(self, stream_id: str) -> _socket.socket:
        logger.info("FdMediaClient: connecting to %s for stream '%s'", self.socket_path, stream_id)

        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        logger.info("FdMediaClient: socket fd=%d connected", sock.fileno())

        name_bytes = stream_id.encode('utf-8')[:_FD_PUB_MAX_STREAM_NAME - 1]
        name_padded = name_bytes.ljust(_FD_PUB_MAX_STREAM_NAME, b'\x00')
        sub_msg = struct.pack(_SUB_FMT, _FD_PUB_MSG_SUBSCRIBE, _SUB_SIZE,
                              _FD_PUB_PROTOCOL_VERSION, name_padded)
        _sendmsg_plain(sock, sub_msg)

        resp_data = sock.recv(_RESP_SIZE)
        if len(resp_data) < _RESP_SIZE:
            sock.close()
            raise ConnectionError(f"FdMediaClient: no response for stream '{stream_id}'")

        msg_type, msg_size, code = struct.unpack(_RESP_FMT, resp_data[:_RESP_SIZE])
        if msg_type != _FD_PUB_MSG_OK:
            sock.close()
            raise ConnectionError(f"FdMediaClient: subscribe rejected for '{stream_id}' (code={code})")

        logger.info("FdMediaClient: subscribed to '%s' successfully", stream_id)
        return sock

    def _get_sock(self, stream_id: str) -> _socket.socket:
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = self._connect_stream(stream_id)
            return self._streams[stream_id]

    def _release_frame(self, sock: _socket.socket, frame_id: int) -> None:
        rel = struct.pack(_REL_FMT, _FD_PUB_MSG_RELEASE, _REL_SIZE, frame_id)
        try:
            _sendmsg_plain(sock, rel)
        except OSError:
            pass

    def _recv_frame(self, sock: _socket.socket,
                    keep_fd: bool = False) -> Frame | None:
        skipped = 0
        eof_count = 0
        for _attempt in range(32):
            data, fds = _recvmsg_with_fds(sock, _FRAME_SIZE)

            # Detect EOF (server closed connection)
            if len(data) == 0:
                eof_count += 1
                if eof_count >= 3:
                    raise ConnectionError("FdMediaClient: socket EOF (server closed connection)")
                continue

            if len(data) < _FRAME_SIZE:
                for fd in fds:
                    os.close(fd)
                skipped += 1
                continue

            values = struct.unpack(_FRAME_FMT, data[:_FRAME_SIZE])
            msg_type = values[0]
            if msg_type != _FD_PUB_MSG_FRAME:
                for fd in fds:
                    os.close(fd)
                skipped += 1
                continue

            break
        else:
            if skipped > 0:
                logger.warning("FdMediaClient: skipped %d non-frame messages, giving up", skipped)
            return None

        if skipped > 0:
            logger.debug("FdMediaClient: skipped %d non-frame messages before frame", skipped)

        frame_id = values[2]
        timestamp_ns = values[3]
        sequence = values[4]
        width = values[5]
        height = values[6]
        fmt_code = values[7]
        num_planes = values[8]
        strides = values[9:12]
        sizes = values[12:15]
        _num_fds_expected = values[15]

        # PLACEHOLDER_FDMEDIACLIENT_RECV_CONT

        fmt_name = PIXEL_FORMAT_NAMES.get(fmt_code, f"UNKNOWN({fmt_code})")

        if not fds:
            self._release_frame(sock, frame_id)
            return None

        if keep_fd:
            def _on_release(h: FrameHandle) -> None:
                self._retained.discard(h)
                self._release_frame(sock, h.frame_id)

            handle = FrameHandle(
                fds=fds, strides=strides, plane_sizes=sizes,
                frame_id=frame_id, on_release=_on_release,
            )
            self._retained.add(handle)
            logger.debug(
                "FdMediaClient: retained frame seq=%d %dx%d %s (frame_id=%d)",
                sequence, width, height, fmt_name, frame_id,
            )
            return Frame(
                sequence=sequence,
                timestamp_ns=timestamp_ns,
                width=width,
                height=height,
                format=fmt_name,
                image=None,
                handle=handle,
            )

        # Copy path: mmap each dma-buf plane (fenced per HAL-3), copy to
        # numpy, close the fds, then hand the buffer back to the daemon.
        try:
            # DMA-BUF fds must be mmapped per-plane using the fd's actual size,
            # not the protocol-reported plane size (which excludes alignment padding).
            planes = []
            for i in range(min(num_planes, len(fds))):
                fd = fds[i]
                _dma_buf_sync(fd, _DMA_BUF_SYNC_READ | _DMA_BUF_SYNC_START)
                actual_size = os.fstat(fd).st_size
                buf = mmap.mmap(fd, actual_size, access=mmap.ACCESS_READ)
                plane_data = np.frombuffer(buf, dtype=np.uint8)[:sizes[i]].copy()
                buf.close()
                _dma_buf_sync(fd, _DMA_BUF_SYNC_READ | _DMA_BUF_SYNC_END)
                planes.append(plane_data)
            raw = np.concatenate(planes) if len(planes) > 1 else planes[0]
        finally:
            for fd in fds:
                os.close(fd)

        self._release_frame(sock, frame_id)

        logger.debug(
            "FdMediaClient: frame seq=%d %dx%d %s released (frame_id=%d)",
            sequence, width, height, fmt_name, frame_id,
        )

        image = _decode_raw(raw, width, height, fmt_name)
        return Frame(
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            width=width,
            height=height,
            format=fmt_name,
            image=image,
        )

    def get_frame(self, stream_id: str, timeout_ms: int = 5000, *,
                  keep_fd: bool = False) -> Frame | None:
        """Receive one frame.

        With ``keep_fd=True`` the frame's dma-buf fds are retained
        (zero-copy handoff; see :class:`FrameHandle`) instead of copied,
        and the daemon-side buffer release is deferred until
        ``frame.release()`` / GC / client close.
        """
        sock = self._get_sock(stream_id)
        sock.settimeout(timeout_ms / 1000.0)
        try:
            return self._recv_frame(sock, keep_fd=keep_fd)
        except _socket.timeout:
            return None
        except (ConnectionError, OSError):
            # Stale socket — clear cache so next call reconnects
            with self._lock:
                old = self._streams.pop(stream_id, None)
                if old:
                    try:
                        old.close()
                    except OSError:
                        pass
            raise

    def subscribe_raw(self, stream_id: str, skip_frames: bool = True,
                      keep_fd: bool = False) -> Iterator[Frame]:
        sock = self._get_sock(stream_id)
        sock.settimeout(5.0)
        while True:
            try:
                frame = self._recv_frame(sock, keep_fd=keep_fd)
                if frame is not None:
                    yield frame
            except _socket.timeout:
                continue
            except (ConnectionError, OSError):
                with self._lock:
                    self._streams.pop(stream_id, None)
                try:
                    sock.close()
                except OSError:
                    pass
                time.sleep(0.5)
                sock = self._get_sock(stream_id)
                sock.settimeout(5.0)

    def subscribe(self, stream_id: str, skip_frames: bool = True,
                  keep_fd: bool = False) -> Iterator[Frame]:
        return self.subscribe_raw(stream_id, skip_frames, keep_fd)

    def on_frame(self, stream_id: str, callback: Callable[[Frame], None]) -> threading.Thread:
        def _run():
            for frame in self.subscribe_raw(stream_id):
                try:
                    callback(frame)
                except Exception:
                    pass
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def close(self) -> None:
        logger.info(
            "FdMediaClient: closing %d stream connections", len(self._streams),
        )
        # Release retained frames first so the daemon recycles their
        # buffers before the subscriptions and sockets go away.
        for handle in list(self._retained):
            try:
                handle.close()
            except Exception:
                pass
        with self._lock:
            for sock in self._streams.values():
                try:
                    unsub = struct.pack(_HDR_FMT, _FD_PUB_MSG_UNSUBSCRIBE, _HDR_SIZE)
                    sock.sendall(unsub)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
            self._streams.clear()

    # -- Encoded stream convenience methods --

    def get_encoded_stream(self, stream_id: str = "main",
                           socket_dir: str = "/run/aipc/encoded") -> EncodedStreamClient:
        """Return an :class:`EncodedStreamClient` for the given encoded stream.

        Args:
            stream_id: Stream name (e.g. ``"main"``, ``"sub"``).
            socket_dir: Directory containing EncodedPublisher UDS sockets.

        Returns:
            A connected :class:`EncodedStreamClient` reading from
            ``{socket_dir}/{stream_id}.sock``.
        """
        path = os.path.join(socket_dir, f"{stream_id}.sock")
        return EncodedStreamClient(path)

    def list_streams(self) -> List[str]:
        """List available raw stream IDs by scanning the camera socket.

        Returns common stream IDs. For detailed status use
        :class:`CameraClient.get_stream_status`.
        """
        return ["main", "sub"]

    def get_rtsp_url(self, stream_id: str = "main",
                     host: str = "192.0.2.72", port: int = 8554) -> str:
        """Return an RTSP URL for the given stream.

        Note: RTSP must be enabled on the device first (via CameraClient
        or REST API).
        """
        return f"rtsp://{host}:{port}/{stream_id}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()