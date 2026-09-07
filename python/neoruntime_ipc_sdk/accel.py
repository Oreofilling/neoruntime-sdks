"""
Hardware-first capability router.

The SDK ships loose components (散件), not a pipeline; apps compose them.
Most components have (or will have) two implementations: a hardware leg
executed by a daemon (DSP color convert / resize, ai-runtime NMS,
codec JPEG, camera-daemon overlay) and a numpy/cv2 software leg that
always works. Today several hardware legs are not reachable because the
service layer does not expose them yet — see
docs/proposals/sdk-hardware-routing.md for the platform ask list.

This router makes that situation explicit and uniform:

* one **policy** per app — prefer hardware, software-only, or
  hardware-only;
* one **route table** — which leg serves each operation, and why the
  other leg is not used ("pending platform exposure", "probe failed",
  "runtime error");
* **degradation tracking** — every hardware→software fallback is
  recorded and surfaced through ``health()`` (and an optional
  ``on_degradation`` hook an app can forward to the event bus as a
  health event).

.. code-block:: python

    from neoruntime_ipc_sdk.accel import get_default_router

    router = get_default_router()
    small = router.run("resize_nv12", nv12, (1920, 1080), (640, 384))
    if router.health()["ops"]["resize_nv12"]["backend"] == "software":
        ...  # hardware path not available on this platform build
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .color import nv12_resize as _nv12_resize_sw
from .color import nv12_to_rgb as _nv12_to_rgb_sw
from .color import rgb_to_nv12 as _rgb_to_nv12_sw
from .frame import _encode_jpeg as _encode_jpeg_sw
from .postprocess import nms as _nms_sw

__all__ = [
    "AccelRouter",
    "DegradationRecord",
    "HardwareUnavailable",
    "RouteDecision",
    "RoutePolicy",
    "get_default_router",
]


class HardwareUnavailable(RuntimeError):
    """A hardware provider could not run (service down, op not exposed)."""


class RoutePolicy(Enum):
    """How the router picks a leg for each operation.

    ``PREFER_HARDWARE`` uses the hardware leg when it is registered and
    healthy, falling back to software otherwise. ``SOFTWARE_ONLY`` never
    touches hardware — for benchmarks and reproducible tests.
    ``HARDWARE_ONLY`` raises instead of degrading — when the zero-CPU
    guarantee matters more than uptime.
    """

    PREFER_HARDWARE = "prefer_hardware"
    SOFTWARE_ONLY = "software_only"
    HARDWARE_ONLY = "hardware_only"


@dataclass(frozen=True)
class RouteDecision:
    """Why a given backend serves an operation right now."""

    op: str
    backend: str  # "hardware" | "software" | "unavailable"
    provider: str
    reason: str


@dataclass(frozen=True)
class DegradationRecord:
    """One hardware→software fallback, kept for health reporting."""

    op: str
    reason: str
    timestamp: float


@dataclass
class _Route:
    software: Callable[..., Any] | None
    hardware: Callable[..., Any] | None = None
    note: str = ""


def _lazy_dsp_client() -> Any:
    """Connect to the DSP service on first use (raises HardwareUnavailable)."""
    from .dsp import DspClient  # noqa: PLC0415 — deferred: heavy proto import

    try:
        return DspClient()
    except Exception as exc:  # connect failures must degrade, not crash
        raise HardwareUnavailable(f"DSP service unreachable: {exc}") from exc


def _dsp_call(method: str, *args: Any, **kwargs: Any) -> Any:
    """Run one DspClient method with its built-in CPU fallback disabled.

    The client's internal fallback would silently compute on CPU under a
    "hardware" label — the router would count the op as hardware and
    never record the degradation. With ``cpu_fallback=False`` the client
    raises instead; the router records the fallback and runs its own
    software leg exactly once.
    """
    client = _lazy_dsp_client()
    try:
        return getattr(client, method)(*args, **kwargs)
    except Exception as exc:
        raise HardwareUnavailable(f"DSP {method} failed: {exc}") from exc


def _resize_nv12_hw(nv12: Any, src_size: tuple[int, int], dst_size: tuple[int, int]) -> Any:
    """DSP resize with the same signature as :func:`color.nv12_resize`."""
    import numpy as np  # noqa: PLC0415 — keep module import light

    dst_w, dst_h = dst_size
    return _dsp_call(
        "resize_hw", np.ascontiguousarray(nv12), dst_w, dst_h, fmt="nv12", cpu_fallback=False
    )


def _rgb_to_nv12_hw(rgb: Any) -> Any:
    """DSP color convert with the same signature as :func:`color.rgb_to_nv12`."""
    import numpy as np  # noqa: PLC0415 — keep module import light

    return _dsp_call(
        "convert_hw", np.ascontiguousarray(rgb), "nv12", fmt="rgb24", cpu_fallback=False
    )


def _nv12_to_rgb_hw(nv12: Any, width: int | None = None, height: int | None = None) -> Any:
    """DSP color convert with the same signature as :func:`color.nv12_to_rgb`."""
    import numpy as np  # noqa: PLC0415 — keep module import light

    if nv12.ndim != 2:
        raise ValueError(f"nv12 source must be 2D, got shape {nv12.shape}")
    src_w, src_h = nv12.shape[1], nv12.shape[0] * 2 // 3
    if (width, height) != (None, None) and (width, height) != (src_w, src_h):
        raise ValueError(f"nv12 is {src_w}x{src_h}, got width/height {width}/{height}")
    return _dsp_call(
        "convert_hw", np.ascontiguousarray(nv12), "rgb24", fmt="nv12", cpu_fallback=False
    )


def _encode_jpeg_hw(rgb: Any, quality: int = 85) -> bytes:
    """camera-daemon EncodeImage with the same signature as the software leg
    (:func:`frame._encode_jpeg` — RGB uint8 array, quality 1..100 → bytes)."""
    import numpy as np  # noqa: PLC0415 — keep module import light

    return _dsp_call(
        "encode_jpeg_hw", np.ascontiguousarray(rgb), quality=quality, fmt="rgb24",
        cpu_fallback=False,
    )


def _extract_annotations(result_or_objects: Any, color: tuple[int, int, int] | None):
    """Split an InferenceResult / object list into draw.py conventions.

    Returns ``(objects, boxes, labels, scores, colors)`` with the same
    PALETTE-by-class_id default :func:`draw.draw_detections` uses.
    """
    from .draw import PALETTE as _palette
    from .draw import _to_xyxy

    objects = (
        list(result_or_objects.objects)
        if hasattr(result_or_objects, "objects")
        else list(result_or_objects)
    )
    boxes, labels, scores, colors = [], [], [], []
    for obj in objects:
        boxes.append(_to_xyxy(obj.bbox if hasattr(obj, "bbox") else obj))
        labels.append(getattr(obj, "label", None))
        scores.append(getattr(obj, "score", None))
        if color is not None:
            colors.append(color)
        else:
            class_id = getattr(obj, "class_id", 0) or 0
            colors.append(_palette[int(class_id) % len(_palette)])
    return objects, boxes, labels, scores, colors


def _draw_detections_hw(
    nv12: Any, result_or_objects: Any, color: tuple[int, int, int] | None = None
) -> Any:
    """DSP blend with the same signature as the software leg (draw.py).

    Renders the annotation once as a minimal RGBA overlay
    (:func:`draw.render_overlay_rgba`) and composites it on the DSP in a
    single blend job. NV12 input only — the vendor op writes NV12 in
    place, so RGB arrays raise and the router serves them on the
    software leg (round-tripping RGB through two color converts would
    cost more than the raster it offloads).
    """
    import numpy as np  # noqa: PLC0415 — keep module import light

    from .draw import render_overlay_rgba as _render_overlay_rgba

    if getattr(nv12, "ndim", 0) != 2:
        raise HardwareUnavailable(
            f"draw_detections hardware leg is nv12-only (DSP blends onto NV12), "
            f"got shape {getattr(nv12, 'shape', None)}"
        )
    width, height = nv12.shape[1], nv12.shape[0] * 2 // 3

    objects, boxes, labels, scores, colors = _extract_annotations(result_or_objects, color)
    if not objects:
        return np.ascontiguousarray(nv12).copy()  # nothing to draw, like the sw leg

    rgba, x0, y0 = _render_overlay_rgba(width, height, boxes, labels, scores, colors)
    return _dsp_call(
        "blend_hw", np.ascontiguousarray(nv12), [(rgba, x0, y0)],
        fmt="nv12", cpu_fallback=False,
    )


def _draw_detections_sw(
    image: Any, result_or_objects: Any, color: tuple[int, int, int] | None = None
) -> Any:
    """Software leg: draw.py raster on RGB arrays; on NV12 the numpy
    mirror of the hardware path (render_overlay_rgba + straight-alpha
    composite), so a degradation never changes the output format."""
    import numpy as np  # noqa: PLC0415 — keep module import light

    from .draw import draw_detections
    from .draw import render_overlay_rgba as _render_overlay_rgba
    from .dsp_format import _cpu_blend

    if getattr(image, "ndim", 0) != 2:
        return draw_detections(image, result_or_objects, color)

    width, height = image.shape[1], image.shape[0] * 2 // 3
    objects, boxes, labels, scores, colors = _extract_annotations(result_or_objects, color)
    if not objects:
        return np.ascontiguousarray(image).copy()
    rgba, x0, y0 = _render_overlay_rgba(width, height, boxes, labels, scores, colors)
    return _cpu_blend(np.ascontiguousarray(image), "nv12", [(rgba, x0, y0)])


class AccelRouter:
    """Route-table dispatcher between hardware and software providers.

    Operations are registered with ``register()``; ``run()`` executes
    through the leg chosen by the policy, recording any fallback.
    :func:`get_default_router` returns a singleton pre-registered with
    the operations the SDK can serve today.
    """

    def __init__(self, policy: RoutePolicy = RoutePolicy.PREFER_HARDWARE):
        self._policy = policy
        self._routes: dict[str, _Route] = {}
        self._probes: dict[str, Callable[[], bool]] = {}
        self._lock = threading.Lock()
        self._counters: dict[str, dict[str, int]] = {}
        self._degradations: deque[DegradationRecord] = deque(maxlen=64)
        self.on_degradation: Callable[[DegradationRecord], None] | None = None

    # -- registration --------------------------------------------------

    def register(
        self,
        op: str,
        software: Callable[..., Any] | None = None,
        hardware: Callable[..., Any] | None = None,
        note: str = "",
    ) -> None:
        """Map an operation name to its software/hardware providers.

        Either leg may be ``None``: a missing hardware leg means the
        capability awaits platform exposure (state it in ``note``), a
        missing software leg means there is no fallback.
        """
        with self._lock:
            self._routes[op] = _Route(software=software, hardware=hardware, note=note)
            self._counters[op] = {"hardware_calls": 0, "software_calls": 0, "fallbacks": 0}

    def use_hardware(self, op: str, hardware: Callable[..., Any]) -> None:
        """Attach (or replace) the hardware leg of an operation."""
        with self._lock:
            route = self._routes.get(op)
            if route is None:
                raise KeyError(f"operation {op!r} is not registered")
            route.hardware = hardware

    def add_probe(self, name: str, probe: Callable[[], bool]) -> None:
        """Register a cheap availability probe (e.g. service reachable)."""
        self._probes[name] = probe

    # -- routing -------------------------------------------------------

    def route(self, op: str) -> RouteDecision:
        """Decide which backend serves ``op`` under the current policy."""
        with self._lock:
            entry = self._routes.get(op)
            if entry is None:
                raise KeyError(
                    f"operation {op!r} is not registered (known: {sorted(self._routes)})"
                )
            route = _Route(software=entry.software, hardware=entry.hardware, note=entry.note)

        if self._policy is RoutePolicy.SOFTWARE_ONLY:
            if route.software is None:
                return RouteDecision(op, "unavailable", "none", "software-only policy and no software provider")
            return RouteDecision(op, "software", _name(route.software), "policy is software-only")

        if route.hardware is not None:
            return RouteDecision(op, "hardware", _name(route.hardware), "hardware leg registered")
        if route.software is None:
            return RouteDecision(op, "unavailable", "none", "no provider registered")
        return RouteDecision(op, "software", _name(route.software), route.note or "no hardware provider registered")

    def run(self, op: str, *args: Any, **kwargs: Any) -> Any:
        """Execute ``op`` through the leg chosen by :meth:`route`."""
        decision = self.route(op)
        if decision.backend == "unavailable":
            raise HardwareUnavailable(f"{op}: {decision.reason}")

        entry = self._routes[op]
        if decision.backend == "software":
            with self._lock:
                self._counters[op]["software_calls"] += 1
            return entry.software(*args, **kwargs)

        try:
            result = entry.hardware(*args, **kwargs)
        except HardwareUnavailable as exc:
            reason = str(exc)
            if self._policy is RoutePolicy.HARDWARE_ONLY:
                raise
            self._record_degradation(op, reason)
            if entry.software is None:
                raise HardwareUnavailable(
                    f"{op}: hardware failed ({reason}) and no software fallback exists"
                ) from exc
            with self._lock:
                self._counters[op]["software_calls"] += 1
            return entry.software(*args, **kwargs)
        with self._lock:
            self._counters[op]["hardware_calls"] += 1
        return result

    # -- health ----------------------------------------------------------

    def _record_degradation(self, op: str, reason: str) -> None:
        record = DegradationRecord(op=op, reason=reason, timestamp=time.time())
        with self._lock:
            self._degradations.append(record)
            self._counters[op]["fallbacks"] += 1
        if self.on_degradation is not None:
            try:
                self.on_degradation(record)
            except Exception:  # a health-sink failure must not break the app
                pass

    def probe(self) -> dict[str, bool]:
        """Run all registered probes → name → available."""
        return {name: _safe_bool(probe) for name, probe in self._probes.items()}

    def health(self) -> dict[str, Any]:
        """Snapshot of routing state, per-op counters and recent fallbacks."""
        ops = {}
        for op in sorted(self._routes):
            decision = self.route(op)
            counters = dict(self._counters.get(op, {}))
            counters["backend"] = decision.backend
            ops[op] = counters
        return {
            "policy": self._policy.value,
            "ops": ops,
            "recent_degradations": [
                {"op": r.op, "reason": r.reason, "timestamp": r.timestamp}
                for r in self._degradations
            ],
        }


def _name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", repr(fn))


def _safe_bool(probe: Callable[[], bool]) -> bool:
    try:
        return bool(probe())
    except Exception:
        return False


def _probe_cv2() -> bool:
    try:
        import cv2  # noqa: PLC0415, F401

        return True
    except ImportError:
        return False


def _probe_dsp() -> bool:
    try:
        _lazy_dsp_client()
        return True
    except HardwareUnavailable:
        return False


_default_router: AccelRouter | None = None
_default_router_lock = threading.Lock()


def get_default_router() -> AccelRouter:
    """Return the pre-registered router singleton.

    Operations served today: ``resize_nv12``, ``rgb_to_nv12`` and
    ``nv12_to_rgb`` (DSP when reachable, numpy otherwise), ``encode_jpeg``
    (camera-daemon EncodeImage when reachable, cv2/Pillow otherwise),
    ``draw_detections`` (DSP blend when reachable and the frame is NV12,
    draw.py raster otherwise), ``nms`` (software only — suppression
    already runs in the HEF's
    integrated hardware NMS before the app sees boxes, and its
    ``iou_threshold`` / ``max_boxes`` are compile-time there; the
    runtime-tunable ``detection_threshold`` lives in
    :meth:`InferenceClient.update_postprocess_config`, honored for
    family-function postprocess models. See
    docs/proposals/sdk-hardware-routing.md S-2).
    ``OverlayClient.annotate`` is hardware-first already and needs no
    routing.
    """
    global _default_router  # noqa: PLW0603 — singleton cache
    with _default_router_lock:
        if _default_router is None:
            router = AccelRouter()
            router.register(
                "resize_nv12",
                software=_nv12_resize_sw,
                hardware=_resize_nv12_hw,
                note="DSP resize via DspClient.resize_hw",
            )
            router.register(
                "rgb_to_nv12",
                software=_rgb_to_nv12_sw,
                hardware=_rgb_to_nv12_hw,
                note="DSP convert via DspClient.convert_hw",
            )
            router.register(
                "nv12_to_rgb",
                software=_nv12_to_rgb_sw,
                hardware=_nv12_to_rgb_hw,
                note="DSP convert via DspClient.convert_hw",
            )
            router.register(
                "encode_jpeg",
                software=_encode_jpeg_sw,
                hardware=_encode_jpeg_hw,
                note="camera-daemon EncodeImage via DspClient.encode_jpeg_hw "
                "(libjpeg on the DSP core — no dedicated JPEG block on hailo15)",
            )
            router.register(
                "nms",
                software=_nms_sw,
                note="HEF-integrated hardware NMS already suppressed pre-app; "
                "runtime detection_threshold via InferenceClient."
                "update_postprocess_config (family functions only)",
            )
            router.register(
                "draw_detections",
                software=_draw_detections_sw,
                hardware=_draw_detections_hw,
                note="DSP blend via DspClient.blend_hw on a minimal RGBA canvas "
                "(nv12 frames; RGB arrays stay on the software raster)",
            )
            router.add_probe("cv2", _probe_cv2)
            router.add_probe("dsp", _probe_dsp)
            _default_router = router
        return _default_router
