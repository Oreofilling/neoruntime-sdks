"""One-shot environment diagnostics for SDK apps ("why is my app slow?").

Collects, without ever raising: optional-dependency availability (cv2 is
the performance-critical one), platform socket reachability, and the
accel router's view of which backend is actually serving each operation.

.. code-block:: python

    from neoruntime_ipc_sdk import diagnostics

    report = diagnostics()
    if not report["optional_deps"]["cv2"]["available"]:
        ...  # add opencv to the app image
    if report["accel_router"]["ops"]["nv12_to_rgb"]["backend"] == "software":
        ...  # DSP unreachable or refused — check camera-daemon
"""

from __future__ import annotations

import os
import socket
import sys
from typing import Any

from .config import Config

__all__ = ["diagnostics", "check", "CheckReport"]

_SOCKET_TIMEOUT = 1.0


def _uds_path(endpoint: str) -> str:
    if endpoint.startswith("unix://"):
        return endpoint[len("unix://"):]
    return endpoint


def _probe_uds(path: str) -> dict[str, Any]:
    """Existence + connectability of a Unix domain socket (never raises)."""
    exists = os.path.exists(path)
    connectable = False
    if exists:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(_SOCKET_TIMEOUT)
            try:
                sock.connect(path)
                connectable = True
            finally:
                sock.close()
        except OSError:
            connectable = False
    return {"path": path, "exists": exists, "connectable": connectable}


def _probe_deps() -> dict[str, Any]:
    deps: dict[str, Any] = {}
    try:
        import cv2  # noqa: PLC0415 — optional accelerator

        deps["cv2"] = {"available": True, "version": getattr(cv2, "__version__", None)}
    except ImportError:
        deps["cv2"] = {"available": False, "version": None}
    try:
        import PIL  # noqa: PLC0415 — hard dependency, reported for completeness

        deps["pillow"] = {"available": True, "version": getattr(PIL, "__version__", None)}
    except ImportError:
        deps["pillow"] = {"available": False, "version": None}
    import numpy

    deps["numpy"] = {"available": True, "version": numpy.__version__}
    return deps


def diagnostics() -> dict[str, Any]:
    """Snapshot the SDK's execution environment. Never raises."""
    from . import __version__
    from .accel import get_default_router  # deferred: heavy proto imports

    services = {
        "ai_runtime": _probe_uds(_uds_path(Config.get_inference_endpoint())),
        "event_bus": _probe_uds(_uds_path(Config.get_event_bus_endpoint())),
        "device_control": _probe_uds(_uds_path(Config.get_device_control_endpoint())),
        "camera_control": _probe_uds(_uds_path(Config.get_camera_control_endpoint())),
        "app_manager": _probe_uds(_uds_path(Config.get_app_manager_endpoint())),
    }

    encoded_dir = Config.get_encoded_socket_dir()
    camera_sock = os.getenv("CAMERA_SOCK_PATH", "/run/aipc/camera.sock")

    router = get_default_router()
    return {
        "sdk_version": __version__,
        "python": {
            "version": sys.version.split()[0],
            "implementation": sys.implementation.name,
        },
        "optional_deps": _probe_deps(),
        "services": services,
        "media": {
            "camera_sock": _probe_uds(camera_sock),
            "encoded_dir": {
                "path": encoded_dir,
                "streams": (
                    sorted(
                        name[: -len(".sock")]
                        for name in os.listdir(encoded_dir)
                        if name.endswith(".sock")
                    )
                    if os.path.isdir(encoded_dir)
                    else []
                ),
            },
        },
        "accel_router": router.health(),
        "accel_probes": router.probe(),
    }


# Named pre-flight checks over the diagnostics() snapshot. The lambda keys
# avoid extra RPCs; model/stream probes are opt-in via live clients below.
_CHECKS = {
    "inference": lambda r: r["services"]["ai_runtime"]["connectable"],
    "events": lambda r: r["services"]["event_bus"]["connectable"],
    "device": lambda r: r["services"]["device_control"]["connectable"],
    "camera": lambda r: r["services"]["camera_control"]["connectable"],
    "app_manager": lambda r: r["services"]["app_manager"]["connectable"],
    "camera_sock": lambda r: r["media"]["camera_sock"]["connectable"],
    "cv2": lambda r: r["optional_deps"]["cv2"]["available"],
    "dsp": lambda r: r["accel_probes"].get("dsp", False),
}

_PROBLEM_HINTS = {
    "inference": "ai-runtime socket not reachable — is the inference daemon running?",
    "events": "event-bus socket not reachable — is the event bus running?",
    "device": "device-control socket not reachable — is the device daemon running?",
    "camera": "camera-control socket not reachable — is camera-daemon running?",
    "app_manager": "app-manager socket not reachable — app management unavailable",
    "camera_sock": "camera.sock fd publisher not reachable — video frames unavailable",
    "cv2": "cv2 not installed — conversions/resize/JPEG run 10-50x slower (pure numpy)",
    "dsp": "DSP service not reachable — pixel operations fall back to CPU",
}


class CheckReport(dict):
    """A :func:`diagnostics` snapshot plus a health verdict.

    ``report["checks"]`` maps each available check name to a bool,
    ``report["healthy"]`` is True when no required check failed, and
    ``report["problems"]`` lists actionable messages for the failures.
    """

    def raise_if_unhealthy(self) -> None:
        """Raise RuntimeError naming every failed requirement, if any."""
        if self["problems"]:
            raise RuntimeError(
                "SDK health check failed:\n- " + "\n- ".join(self["problems"])
            )


def check(
    required: list[str] | None = None,
    *,
    inference: Any | None = None,
    model_id: str | None = None,
    camera: Any | None = None,
    stream_id: str | None = None,
) -> CheckReport:
    """Pre-flight an app: run :func:`diagnostics` and grade it.

    ``required`` names checks from: ``inference``, ``events``, ``device``,
    ``camera``, ``app_manager``, ``camera_sock``, ``cv2``, ``dsp`` — plus
    ``model_registered`` / ``stream_active`` when the corresponding live
    client + id are passed. Each failed requirement lands in
    ``problems`` with an actionable hint.

    >>> from neoruntime_ipc_sdk.diagnostics import check
    >>> check(["inference", "camera_sock"]).raise_if_unhealthy()

    The live probes are opt-in so the base check stays offline-safe.
    """
    report = dict(diagnostics())
    checks = {key: bool(fn(report)) for key, fn in _CHECKS.items()}

    if inference is not None and model_id is not None:
        try:
            checks["model_registered"] = inference.get_model_info(model_id) is not None
        except Exception as exc:  # service down / RPC error → not registered
            checks["model_registered"] = False
            report["model_lookup_error"] = str(exc)

    if camera is not None and stream_id is not None:
        try:
            statuses = {s.stream_id: s.status for s in camera.get_stream_status()}
            checks["stream_active"] = statuses.get(stream_id) == "active"
            report["stream_status"] = statuses.get(stream_id, "absent")
        except Exception as exc:
            checks["stream_active"] = False
            report["stream_lookup_error"] = str(exc)

    problems = []
    for key in required or []:
        if key not in checks:
            raise ValueError(
                f"unknown check {key!r} (available: {', '.join(sorted(checks))})"
            )
        if checks[key]:
            continue
        if key == "model_registered":
            problems.append(
                f"model {model_id!r} is not registered with the inference service"
            )
        elif key == "stream_active":
            status = report.get("stream_status", "unknown")
            problems.append(f"stream {stream_id!r} is not active (status: {status})")
        else:
            hint = _PROBLEM_HINTS.get(key)
            problems.append(hint if hint is not None else f"{key}: check failed")

    report["checks"] = checks
    report["healthy"] = not problems
    report["problems"] = problems
    return CheckReport(report)
