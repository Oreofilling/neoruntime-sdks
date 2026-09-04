"""
Color-space conversions between NV12 and packed RGB/BGR.

Camera streams arrive as NV12 (semi-planar YUV 4:2:0, limited range);
models and drawing want packed RGB/BGR. ``Frame.to_rgb()`` already covers
NV12→RGB in numpy — the reverse direction and the plane-level utilities
were missing, so apps kept re-implementing them (parking-lot carries its
own trio of converters, parking_lot/app.py:312-358).

cv2 accelerates every conversion when installed; otherwise a vectorised
numpy path is used (BT.601 limited range, matching cv2's
``COLOR_YUV2BGR_NV12`` semantics). The numpy chroma path is
nearest-neighbour on decode / 2×2 block mean on encode, so round-trips
lose a little chroma detail — fine for model input, not for pixel-exact
comparisons.

.. code-block:: python

    from neoruntime_ipc_sdk.color import rgb_to_nv12, nv12_resize

    nv12 = rgb_to_nv12(rgb)                      # (h*3/2, w) uint8
    small = nv12_resize(nv12, (1920, 1080), (640, 384))
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bgr_to_nv12",
    "nv12_resize",
    "nv12_to_bgr",
    "nv12_to_rgb",
    "rgb_to_nv12",
]

# BT.601 limited-range (studio swing) coefficients, same family cv2 uses
# for its NV12 conversions.
_Y_FROM_RGB = np.array([65.481, 128.553, 24.966], dtype=np.float32)
_U_FROM_RGB = np.array([-37.797, -74.203, 112.0], dtype=np.float32)
_V_FROM_RGB = np.array([112.0, -93.786, -18.214], dtype=np.float32)


def _check_dims(width: int, height: int, nv12: np.ndarray) -> None:
    expected = height * 3 // 2
    if nv12.shape[0] != expected or nv12.shape[1] != width:
        raise ValueError(
            f"NV12 buffer is {nv12.shape}, expected ({expected}, {width}) "
            f"for {width}x{height}"
        )


def _check_even(height: int, width: int, image: np.ndarray) -> None:
    if height % 2 or width % 2:
        raise ValueError(
            f"NV12 needs even dimensions, got {width}x{height} (shape {image.shape})"
        )


def nv12_to_rgb(nv12: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert an NV12 buffer to an RGB ``(height, width, 3)`` uint8 array."""
    _check_dims(width, height, nv12)
    try:
        import cv2  # noqa: PLC0415 — optional accelerator, like draw.py

        return cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
    except ImportError:
        pass
    return _nv12_to_packed(nv12, width, height, "rgb")


def nv12_to_bgr(nv12: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert an NV12 buffer to a BGR ``(height, width, 3)`` uint8 array."""
    _check_dims(width, height, nv12)
    try:
        import cv2  # noqa: PLC0415

        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    except ImportError:
        pass
    return _nv12_to_packed(nv12, width, height, "bgr")


def _nv12_to_packed(nv12: np.ndarray, width: int, height: int, order: str) -> np.ndarray:
    y = nv12[:height].astype(np.float32)
    uv = nv12[height:].reshape(height // 2, width // 2, 2)
    u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1)

    yd = 1.1644 * (y - 16.0)
    r = yd + 1.5960 * (v - 128.0)
    g = yd - 0.3918 * (u - 128.0) - 0.8130 * (v - 128.0)
    b = yd + 2.0172 * (u - 128.0)

    channels = [r, g, b] if order == "rgb" else [b, g, r]
    packed = np.stack(channels, axis=-1)
    return np.clip(np.round(packed), 0, 255).astype(np.uint8)


def rgb_to_nv12(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB ``(h, w, 3)`` array to an NV12 ``(h*3/2, w)`` buffer."""
    return _packed_to_nv12(rgb, "rgb")


def bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR ``(h, w, 3)`` array to an NV12 ``(h*3/2, w)`` buffer."""
    return _packed_to_nv12(bgr, "bgr")


def _packed_to_nv12(image: np.ndarray, order: str) -> np.ndarray:
    height, width = image.shape[:2]
    _check_even(height, width, image)
    try:
        import cv2  # noqa: PLC0415

        code = cv2.COLOR_RGB2YUV_I420 if order == "rgb" else cv2.COLOR_BGR2YUV_I420
        i420 = cv2.cvtColor(image, code)
        y_plane = i420[:height, :]
        u_plane = i420[height : height + height // 4, :].reshape(height // 2, width // 2)
        v_plane = i420[height + height // 4 :, :].reshape(height // 2, width // 2)
        uv_plane = np.empty((height // 2, width), dtype=np.uint8)
        uv_plane[:, 0::2] = u_plane
        uv_plane[:, 1::2] = v_plane
        return np.vstack([y_plane, uv_plane])
    except ImportError:
        pass

    planar = image.astype(np.float32)
    if order == "bgr":
        planar = planar[:, :, ::-1]
    # coefficients are per-255 (65.481*255/255 == 65.481 at full swing) —
    # the /255 keeps 16/235 studio swing instead of saturating
    y = 16.0 + (planar @ _Y_FROM_RGB) / 255.0
    # chroma comes from the mean of each 2x2 luma block
    blocks = planar.reshape(height // 2, 2, width // 2, 2, 3).mean(axis=(1, 3))
    u = 128.0 + (blocks @ _U_FROM_RGB) / 255.0
    v = 128.0 + (blocks @ _V_FROM_RGB) / 255.0

    y_plane = np.clip(np.round(y), 0, 255).astype(np.uint8)
    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = np.clip(np.round(u), 0, 255).astype(np.uint8)
    uv_plane[:, 1::2] = np.clip(np.round(v), 0, 255).astype(np.uint8)
    return np.vstack([y_plane, uv_plane])


def nv12_resize(
    nv12: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    """Resize an NV12 buffer from ``src_size`` to ``dst_size`` (width, height).

    Resizes the Y and UV planes separately — avoids the
    NV12→BGR→resize→BGR→NV12 round-trip (the approach proven in
    parking-lot). With cv2 the planes are resized with bilinear
    interpolation; the numpy fallback uses nearest-neighbour.
    """
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    _check_dims(src_w, src_h, nv12)
    if dst_w % 2 or dst_h % 2:
        raise ValueError(f"destination must be even-sized, got {dst_w}x{dst_h}")
    if dst_w > src_w or dst_h > src_h:
        raise ValueError(
            f"upscaling {src_w}x{src_h} -> {dst_w}x{dst_h} not supported "
            "(model inputs are smaller than the source frame)"
        )

    y_plane = nv12[:src_h, :]
    uv_plane = nv12[src_h : src_h + src_h // 2, :]
    try:
        import cv2  # noqa: PLC0415

        y_out = cv2.resize(y_plane, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        uv_out = cv2.resize(uv_plane, (dst_w, dst_h // 2), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        y_out = _resize_nearest(y_plane, dst_w, dst_h)
        uv_out = _resize_nearest(uv_plane, dst_w, dst_h // 2)
    return np.vstack([y_out, uv_out])


def _resize_nearest(plane: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    src_h, src_w = plane.shape[:2]
    rows = np.minimum((np.arange(dst_h) * src_h // dst_h), src_h - 1)
    cols = np.minimum((np.arange(dst_w) * src_w // dst_w), src_w - 1)
    return plane[np.ix_(rows, cols)]
