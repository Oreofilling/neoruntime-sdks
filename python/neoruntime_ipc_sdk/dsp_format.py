"""DSP job format inference, geometry validation and CPU fallback.

The pure-numpy side of the DSP module: mapping SDK frame formats onto
DSP formats, validating crop/dest geometry against daemon caps, and the
software path used when the daemon lacks the DSP surface (or for tests).
"""

from typing import Optional, Tuple

import numpy as np

try:  # cv2 accelerates the CPU fallback only; never required
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None

from .dsp_wire import _DSP_FORMATS, _MAX_DIM, _MIN_DIM, DspError
from .frame import Frame

# Frame.format names (media.py PIXEL_FORMAT_NAMES) a handle may carry into
# an import. RGB and BGR both map to rgb24: these ops are byte-order
# agnostic per-pixel geometry transforms, so the plane imports verbatim.
_FRAME_FMT_TO_DSP = {"NV12": "nv12", "RGB": "rgb24", "BGR": "rgb24",
                     "GRAY8": "gray8"}

_CV_INTERP = {
    "nearest": "INTER_NEAREST",
    "bilinear": "INTER_LINEAR",
    "area": "INTER_AREA",
    "bicubic": "INTER_CUBIC",
}
# ---- format / geometry helpers -------------------------------------
def _infer_fmt(src: np.ndarray, fmt: Optional[str]) -> str:
    if fmt is not None:
        if fmt not in _DSP_FORMATS:
            raise DspError(f"unsupported format {fmt!r} (nv12/rgb24/gray8)")
        return fmt
    if src.ndim == 3 and src.shape[2] == 3:
        return "rgb24"
    if src.ndim == 2:
        return "gray8"  # ambiguous with nv12 — pass fmt explicitly for YUV
    raise DspError(f"cannot infer format from shape {src.shape}")


def _as_pixels(src) -> np.ndarray:
    """The numpy pixels behind a non-handle source (ndarray or Frame)."""
    return src.image if isinstance(src, Frame) else src


def _src_dims(src: np.ndarray, fmt: str) -> Tuple[int, int]:
    if fmt == "nv12":
        if src.ndim != 2:
            raise DspError("nv12 src must be (h*3//2, w)")
        return src.shape[0] * 2 // 3, src.shape[1]
    if fmt == "rgb24":
        if src.ndim != 3 or src.shape[2] != 3:
            raise DspError("rgb24 src must be (h, w, 3)")
        return src.shape[0], src.shape[1]
    if src.ndim != 2:
        raise DspError("gray8 src must be (h, w)")
    return src.shape


def _validated_rect(sw: int, sh: int, fmt: str, x: int, y: int, w: int, h: int,
                    dw: int, dh: int) -> Tuple[int, int, int, int, int, int]:
    """Validate one crop rect against source dims ``sw x sh`` (frame handles
    arrive as geometry, not pixels — dims are resolved by the caller)."""
    if w < 1 or h < 1:
        raise DspError(f"crop size must be positive (got {w}x{h})")
    if x < 0 or y < 0 or x + w > sw or y + h > sh:
        raise DspError(f"crop ({x},{y},{w}x{h}) outside source {sw}x{sh}")
    if fmt == "nv12" and any(v % 2 for v in (x, y, w, h, dw, dh)):
        raise DspError("nv12 crop/dest coords and sizes must be even")
    if not (_MIN_DIM <= dw <= _MAX_DIM and _MIN_DIM <= dh <= _MAX_DIM):
        raise DspError(f"destination dims {dw}x{dh} outside daemon range "
                       f"[{_MIN_DIM}, {_MAX_DIM}]")
    return x, y, w, h, dw, dh


# ---- CPU fallback ------------------------------------------------------------
def _resize_plane(plane: np.ndarray, out_h: int, out_w: int,
                  interpolation: str) -> np.ndarray:
    if _cv2 is not None:
        return _cv2.resize(plane, (out_w, out_h),
                           interpolation=getattr(_cv2, _CV_INTERP[interpolation]))
    rows = np.arange(out_h) * plane.shape[0] // out_h
    cols = np.arange(out_w) * plane.shape[1] // out_w
    return plane[np.ix_(rows, cols)]


def _cpu_resize(src: np.ndarray, fmt: str, dw: int, dh: int, scaling: str,
                interpolation: str) -> np.ndarray:
    sh, sw = _src_dims(src, fmt)
    if scaling in ("letterbox", "letterbox_middle", "letterbox_up_left"):
        scale = min(dw / sw, dh / sh)
        cw, ch = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
        if fmt == "nv12":
            cw &= ~1
            ch &= ~1
        content = _cpu_resize(src, fmt, cw, ch, "stretch", interpolation)
        ox = (dw - cw) // 2 if scaling != "letterbox_up_left" else 0
        oy = (dh - ch) // 2 if scaling != "letterbox_up_left" else 0
        if fmt == "nv12":  # keep UV half-sample alignment
            ox &= ~1
            oy &= ~1
        return _place(fmt, content, dw, dh, ox, oy)
    if scaling == "scale_crop":
        scale = max(dw / sw, dh / sh)
        cw, ch = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
        if fmt == "nv12":
            cw &= ~1
            ch &= ~1
        big = _cpu_resize(src, fmt, cw, ch, "stretch", interpolation)
        return _cpu_crop(big, fmt, (cw - dw) // 2 & ~1, (ch - dh) // 2 & ~1,
                         dw, dh)
    # stretch
    if fmt == "nv12":
        y = _resize_plane(src[:sh], dh, dw, interpolation)
        # UV rows are interleaved (U,V) pairs: dw/2 samples = dw bytes wide
        uv = _resize_plane(src[sh:], dh // 2, dw, "nearest")
        return np.vstack([y, uv])
    return _resize_plane(src, dh, dw, interpolation)


def _place(fmt: str, content: np.ndarray, dw: int, dh: int, ox: int, oy: int
           ) -> np.ndarray:
    """Paste ``content`` onto a black canvas at (ox, oy); nv12 pads UV 128."""
    if fmt == "nv12":
        ch = content.shape[0] * 2 // 3
        cw = content.shape[1]
        canvas = np.zeros((dh * 3 // 2, dw), dtype=np.uint8)
        canvas[dh:, :] = 128                      # neutral chroma
        canvas[oy:oy + ch, ox:ox + cw] = content[:ch]
        canvas[dh + oy // 2:dh + (oy + ch) // 2,
               ox // 2:(ox + cw) // 2] = content[ch:]
        return canvas
    canvas = np.zeros((dh, dw) if fmt == "gray8" else (dh, dw, 3),
                      dtype=np.uint8)
    canvas[oy:oy + content.shape[0], ox:ox + content.shape[1]] = content
    return canvas


def _cpu_crop(src: np.ndarray, fmt: str, x: int, y: int, w: int,
              h: int) -> np.ndarray:
    if fmt == "nv12":
        sh = src.shape[0] * 2 // 3
        # interleaved UV: a w-pixel crop spans w bytes of chroma rows
        return np.vstack([
            src[y:y + h, x:x + w],
            src[sh + y // 2:sh + (y + h) // 2, x:x + w],
        ])
    return src[y:y + h, x:x + w].copy()


def _cpu_crop_resize(src: np.ndarray, fmt: str,
                     rect: Tuple[int, ...]) -> np.ndarray:
    x, y, w, h, dw, dh = rect
    out = _cpu_crop(src, fmt, x, y, w, h)
    if (dw, dh) != (w, h):
        out = _cpu_resize(out, fmt, dw, dh, "stretch", "bilinear")
    return out
