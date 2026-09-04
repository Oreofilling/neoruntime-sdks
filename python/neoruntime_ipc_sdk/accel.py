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
from .color import rgb_to_nv12 as _rgb_to_nv12_sw
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


def _resize_nv12_hw(nv12: Any, src_size: tuple[int, int], dst_size: tuple[int, int]) -> Any:
    """DSP resize with the same signature as :func:`color.nv12_resize`."""
    import numpy as np  # noqa: PLC0415 — keep module import light

    dst_w, dst_h = dst_size
    client = _lazy_dsp_client()
    try:
        return client.resize_hw(np.ascontiguousarray(nv12), dst_w, dst_h, fmt="nv12")
    except Exception as exc:
        raise HardwareUnavailable(f"DSP resize failed: {exc}") from exc


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

    Operations served today: ``resize_nv12`` (DSP when reachable, numpy
    otherwise), ``rgb_to_nv12`` and ``nms`` (software only — hardware
    legs pending, see docs/proposals/sdk-hardware-routing.md).
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
                note="hardware leg pending DspClient.convert_hw — the daemon "
                "already serves DSP_OP_CONVERT_FORMAT (equal dims, differing "
                "formats) but the SDK client exposes resize/crop/multi_crop only",
            )
            router.register(
                "nms",
                software=_nms_sw,
                note="hardware leg pending ai-runtime NMS registration params",
            )
            router.add_probe("cv2", _probe_cv2)
            router.add_probe("dsp", _probe_dsp)
            _default_router = router
        return _default_router
