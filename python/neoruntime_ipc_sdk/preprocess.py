"""Client-side preprocessing for model inference.

Composes the SDK's existing pieces into one call and returns the affine
:class:`PreprocessMeta` alongside the model-ready array, so
postprocessing can map model-input coordinates back to the source frame
without hand-derived letterbox maths:

* geometry rides :meth:`Frame.resize` — keep-fd frames scale on the DSP
  (zero-copy import) and the metadata comes straight from the shared
  geometry helper, so the pixels and the meta cannot drift;
* colour conversion runs on the *resized* image (convert-after-shrink:
  a 640x640 convert is ~9x cheaper than a 1080p one);
* ``normalize`` is opt-in and off by default — HEF models on this
  platform take uint8 inputs with the quantisation scales compiled into
  the HEF; pre-normalising would corrupt them. ``from_model`` enables
  it automatically for float-dtype inputs only.

.. code-block:: python

    from neoruntime_ipc_sdk import InferenceClient, Preprocessor

    pre = Preprocessor.from_model(InferenceClient(), "person_v1")
    tensor, meta = pre(frame)                     # frame or ndarray
    result = inf.infer(tensor, model_id="person_v1")
    x_src, y_src = meta.to_source(x_model, y_model)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .frame import Frame

__all__ = ["PreprocessMeta", "Preprocessor"]

# Formats _to_color actually handles; validated at construction so an
# unsupported hint fails immediately instead of on the first frame.
_SOURCE_FORMATS = frozenset({"NV12", "RGB", "BGR", "GRAY8"})

# DataType enum values from proto/ai-runtime/inference.proto (the parsed
# ModelInfo dicts carry the raw enum int; accept strings too).
_DTYPE_ENUM_TO_STR = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "float16",
    5: "float32",
    6: "int32",
    7: "uint32",
}


@dataclass(frozen=True)
class PreprocessMeta:
    """Affine geometry of a preprocessing pass (see :class:`Preprocessor`).

    Semantics per axis: ``dst = src * scale + origin``, where dst is the
    model-input image and src the original frame — so the inverse (what
    postprocessing needs) is ``src = (dst - origin) / scale``.
    """

    original_size: tuple[int, int]
    input_size: tuple[int, int]
    scale: tuple[float, float]
    origin: tuple[int, int]
    color: str = "RGB"
    layout: str = "NHWC"

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        """Map a model-input point back to source-frame coordinates."""
        return (
            (x - self.origin[0]) / self.scale[0],
            (y - self.origin[1]) / self.scale[1],
        )

    def to_source_box(self, box: Any) -> tuple[float, float, float, float]:
        """Map an (x1, y1, x2, y2) box back to source-frame coordinates."""
        x1, y1, x2, y2 = (float(v) for v in box)
        a = self.to_source(x1, y1)
        b = self.to_source(x2, y2)
        return (a[0], a[1], b[0], b[1])


def _dtype_str(dtype: Any) -> str:
    if isinstance(dtype, str):
        return dtype.lower()
    return _DTYPE_ENUM_TO_STR.get(dtype, "")


def _infer_source_format(arr: np.ndarray, hint: str | None) -> str:
    if hint:
        return hint
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 3:
            return "RGB"  # honest default for packed 3-channel input
        if channels == 1:
            return "GRAY8"
        if channels == 4:
            raise ValueError(
                "4-channel (RGBA) arrays are not supported — strip alpha "
                "first (e.g. arr[:, :, :3]) and pass the RGB array"
            )
    if arr.ndim == 2:
        raise ValueError(
            "2D arrays are ambiguous (NV12 vs GRAY8) — pass "
            "source_format='NV12' or 'GRAY8'"
        )
    raise ValueError(f"unsupported source array shape {arr.shape}")


def _derive_layout_size(
    shape: list[int], layout_hint: str
) -> tuple[str, tuple[int, int]]:
    """Derive (layout, (width, height)) from a model input shape.

    A leading batch dimension of 1 is expected to be stripped by the
    caller. The TensorSpec layout string wins when present; otherwise
    channels are located by size (1/3/4) with HWC preferred.
    """
    if len(shape) != 3:
        raise ValueError(
            f"cannot derive spatial size from input shape {shape} — "
            "expected [N,]H,W,C or [N,]C,H,W"
        )
    layout_hint = (layout_hint or "").upper()
    a, b, c = shape
    if layout_hint in ("HWC", "NHWC"):
        (h, w), ch, layout = (a, b), c, "NHWC"
    elif layout_hint in ("CHW", "NCHW"):
        ch, (h, w), layout = a, (b, c), "NCHW"
    elif c in (1, 3, 4):
        (h, w), ch, layout = (a, b), c, "NHWC"
    elif a in (1, 3, 4):
        ch, (h, w), layout = a, (b, c), "NCHW"
    else:
        raise ValueError(f"ambiguous input shape {shape} — no channels dimension")
    if ch not in (1, 3, 4):
        raise ValueError(f"unsupported channel count {ch} in input shape {shape}")
    return layout, (w, h)


class Preprocessor:
    """Resize + colour-convert (+ optional normalise/layout) in one call.

    Args:
        size: target (width, height); ``None`` until :meth:`from_model`
            derives it (calling before then raises).
        color: target channel order, ``"RGB"`` or ``"BGR"`` (case-insensitive).
        resize_mode: ``"letterbox"`` (default) / ``"stretch"`` / ``"crop"``,
            forwarded to :meth:`Frame.resize`.
        pad_value: letterbox pad value (114 is the YOLO convention).
        normalize: divide by 255 into float32. Off by default — HEF
            models quantise internally and expect uint8.
        layout: ``"NHWC"`` (default) or ``"NCHW"`` output arrangement
            (case-insensitive).
        source_format: format hint for ndarray inputs (``"NV12"`` /
            ``"GRAY8"`` / ``"RGB"`` / ``"BGR"``); 2D arrays require it.
            Case-insensitive. 4-channel (RGBA) arrays are rejected —
            strip alpha (``arr[:, :, :3]``) first.
    """

    def __init__(
        self,
        size: tuple[int, int] | None = None,
        color: str = "RGB",
        resize_mode: str = "letterbox",
        pad_value: int = 114,
        normalize: bool = False,
        layout: str = "NHWC",
        source_format: str | None = None,
    ):
        # Normalise first, validate after: "rgb"/"Nchw"/"nv12" are all fine.
        color = color.upper()
        layout = layout.upper()
        resize_mode = resize_mode.lower()
        if source_format is not None:
            source_format = source_format.upper()
        if color not in ("RGB", "BGR"):
            raise ValueError(f"color must be 'RGB' or 'BGR', got {color!r}")
        if resize_mode not in ("letterbox", "stretch", "crop"):
            raise ValueError(f"unsupported resize_mode: {resize_mode!r}")
        if layout not in ("NHWC", "NCHW"):
            raise ValueError(f"layout must be 'NHWC' or 'NCHW', got {layout!r}")
        if source_format is not None and source_format not in _SOURCE_FORMATS:
            raise ValueError(
                f"source_format must be one of {sorted(_SOURCE_FORMATS)}, "
                f"got {source_format!r}"
            )
        self.size = size
        self.color = color
        self.resize_mode = resize_mode
        self.pad_value = pad_value
        self.normalize = normalize
        self.layout = layout
        self.source_format = source_format

    @classmethod
    def from_model(cls, client: Any, model_id: str, **overrides: Any) -> Preprocessor:
        """Derive size/layout/normalisation from the model's input spec.

        ``client`` is anything with a ``get_model_info`` RPC surface
        (an :class:`~neoruntime_ipc_sdk.InferenceClient`). Keyword
        overrides replace any derived value.
        """
        info = client.get_model_info(model_id)
        if info is None:
            raise ValueError(f"model {model_id!r} is not registered with the service")
        if not info.inputs:
            raise ValueError(f"model {model_id!r} exposes no input specs")

        inp = info.inputs[0]
        shape = [int(d) for d in (inp.get("shape") or [])]
        if shape and shape[0] == 1 and len(shape) >= 3:
            shape = shape[1:]  # strip the batch dimension
        layout, size = _derive_layout_size(shape, inp.get("layout") or "")
        normalize = _dtype_str(inp.get("dtype")).startswith("float")

        kwargs: dict[str, Any] = dict(size=size, layout=layout, normalize=normalize)
        kwargs.update(overrides)
        return cls(**kwargs)

    def _as_frame(self, source: Any) -> Frame:
        if isinstance(source, Frame):
            return source
        arr = np.asarray(source)
        fmt = _infer_source_format(arr, self.source_format)
        if fmt in ("NV12", "NV21"):
            height = arr.shape[0] * 2 // 3
            width = arr.shape[1]
        else:
            height, width = arr.shape[0], arr.shape[1]
        return Frame(
            sequence=0, timestamp_ns=0, width=width, height=height,
            format=fmt, image=arr,
        )

    def __call__(self, source: Any) -> tuple[np.ndarray, PreprocessMeta]:
        if self.size is None:
            raise ValueError("no target size — construct via from_model() or pass size=")
        frame = self._as_frame(source)

        if (frame.width, frame.height) == self.size:
            resized = frame
            # No further geometry — but a frame that was already cropped or
            # resized carries its transform; inherit it so postprocessing
            # still maps back to the ORIGINAL frame, not this intermediate.
            prev = frame.metadata.get("transform")
            if isinstance(prev, dict) and "scale" in prev and "origin" in prev:
                original_size = prev["src_size"]
                scale = prev["scale"]
                origin = prev["origin"]
            else:
                original_size = self.size
                scale = (1.0, 1.0)
                origin = (0, 0)
        else:
            resized = frame.resize(
                self.size[0], self.size[1],
                mode=self.resize_mode, pad_value=self.pad_value,
            )
            transform = resized.metadata["transform"]
            original_size = transform["src_size"]
            scale = transform["scale"]
            origin = transform["origin"]

        arr = self._to_color(resized)
        if self.normalize:
            arr = arr.astype(np.float32) / 255.0
        if self.layout == "NCHW":
            arr = np.ascontiguousarray(arr.transpose(2, 0, 1))

        meta = PreprocessMeta(
            original_size=original_size,
            input_size=self.size,
            scale=scale,
            origin=origin,
            color=self.color,
            layout=self.layout,
        )
        return arr, meta

    def _to_color(self, frame: Frame) -> np.ndarray:
        fmt = frame.format
        if fmt == "NV12":
            arr = frame.to_rgb()  # convert-after-shrink: the small image
            want_swap = self.color == "BGR"
        elif fmt in ("RGB", "BGR", "GRAY8"):
            arr = frame.to_array()
            if fmt == "GRAY8":
                arr = np.stack([arr] * 3, axis=-1)  # expand to 3 channels
            want_swap = (fmt == "RGB" and self.color == "BGR") or (
                fmt == "BGR" and self.color == "RGB"
            )
        else:
            raise ValueError(f"unsupported source format for preprocessing: {fmt}")
        if want_swap:
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
