"""
AI Overlay Control Client

Control the platform's AI overlay system that draws detection boxes,
labels, and confidence scores directly on NV12 frames before encoding.
Zero CPU cost — drawing happens in camera-daemon before encoding.

Two halves:

* configuration — ``enable`` / ``disable`` / ``configure`` / ``apply``
  via the ``UpdateAiOverlay`` RPC;
* content — ``annotate`` / ``annotate_result`` push detection boxes (or
  classifications / landmarks / OCR lines) through the event bus;
  camera-daemon's overlay subscriber renders them with its HAL draw ops.

Uses gRPC over Unix domain socket, consistent with other SDK clients.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import grpc  # noqa: F401 — keeps a module-level grpc anchor like the other clients

from ._transport import GrpcClient
from .events import EventClient
from .inference_types import (
    Classification,
    DetectedObject,
    InferenceResult,
    LandmarkSet,
    OcrLine,
)
from .proto import camera_pb2, camera_pb2_grpc

logger = logging.getLogger("neoruntime_ipc_sdk.overlay")

# camera-daemon subscribes to "inference/**" (ai_overlay_subscriber.h:29
# topic_prefix default) and requires event.metadata["stream_id"];
# ai-runtime's own auto-inference publisher uses "inference/<stream_id>"
# (auto_infer.cpp:523), so annotate() matches that exact topic shape.
_OVERLAY_TOPIC_PREFIX = "inference/"

# Wire-side caps mirrored from the daemon/HAL: HAL_MAX_POLYGON_POINTS
# (hal_draw.h) bounds one polygon, the daemon's sidecar keeps at most
# _POLYGON_CAP per stream — loud validation here beats silent clipping.
_POLYGON_CAP = 16
_POLYGON_POINT_CAP = 128

# Strict wait cap bounds: the daemon derives 2 frame periods (clamped to
# [1, 500] ms) when the cap is 0. An explicit cap must stay in the same
# order — strict mode may add bounded latency, never stall the encoder.
_STRICT_WAIT_CAP_MS_MAX = 5000


@dataclass
class OverlayConfig:
    """AI overlay configuration.

    ``strict_frame_lock`` / ``strict_wait_cap_ms`` mirror the optional wire
    fields: ``None`` (the default) leaves them unset, so the daemon keeps its
    current (yaml) strict settings.
    """

    enabled: bool = True
    show_label: bool = True
    show_confidence: bool = True
    line_thickness: int = 2
    box_color: int = 0
    label_color: int = 0
    font_size: int = 0
    strict_frame_lock: bool | None = None
    strict_wait_cap_ms: int | None = None

    def to_proto(self) -> camera_pb2.AiOverlayConfig:
        cfg = camera_pb2.AiOverlayConfig(
            enabled=self.enabled,
            show_label=self.show_label,
            show_confidence=self.show_confidence,
            line_thickness=self.line_thickness,
        )
        if self.box_color:
            cfg.box_color = self.box_color
        if self.label_color:
            cfg.label_color = self.label_color
        if self.font_size:
            cfg.font_size = self.font_size
        if self.strict_frame_lock is not None:
            cfg.strict_frame_lock = self.strict_frame_lock
        if self.strict_wait_cap_ms is not None:
            cfg.strict_wait_cap_ms = self.strict_wait_cap_ms
        return cfg


class OverlayClient(GrpcClient):
    """
    AI Overlay Control Client

    Uses gRPC to communicate with camera-daemon's CameraControl service.

    Usage::

        from neoruntime_ipc_sdk import OverlayClient

        oc = OverlayClient()

        # Enable overlay with default settings
        oc.enable()

        # Customize appearance
        oc.configure(
            show_label=True,
            show_confidence=True,
            line_thickness=3,
        )

        # Disable overlay
        oc.disable()

    Environment variables:
        CAMERA_CONTROL_ENDPOINT: Camera control gRPC endpoint
                                  (default: unix:///run/aipc/camera-control.sock)
    """

    _stub_factory = camera_pb2_grpc.CameraControlStub
    # Default endpoint comes from Config.get_camera_control_endpoint() (same
    # env var and default this class used inline before).

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._event_client: EventClient | None = None

    def _events(self) -> EventClient:
        """Lazily connect the event-bus client used for annotate()."""
        if self._event_client is None:
            self._event_client = EventClient()
        return self._event_client

    def close(self) -> None:
        if self._event_client is not None:
            self._event_client.close()
            self._event_client = None
        super().close()

    @property
    def channel_options(self):
        # epoll1; avoid the sched_yield busy-poll of the default poll strategy
        # (plus the base 64 MiB receive limit — see MAX_GRPC_MESSAGE_LENGTH)
        return super().channel_options + [("grpc.poll_strategy", 1)]

    def _update(self, config: camera_pb2.AiOverlayConfig) -> None:
        if self.stub is None:
            self.connect()
        resp = self.stub.UpdateAiOverlay(config)
        if not resp.success:
            raise RuntimeError(f"UpdateAiOverlay failed: {resp.message}")

    def enable(
        self,
        show_label: bool = True,
        show_confidence: bool = True,
        line_thickness: int = 2,
    ) -> None:
        """Enable AI overlay with specified settings."""
        self._update(
            camera_pb2.AiOverlayConfig(
                enabled=True,
                show_label=show_label,
                show_confidence=show_confidence,
                line_thickness=line_thickness,
            )
        )
        logger.info("AI overlay enabled")

    def disable(self) -> None:
        """Disable AI overlay."""
        self._update(camera_pb2.AiOverlayConfig(enabled=False))
        logger.info("AI overlay disabled")

    def configure(
        self,
        enabled: bool = True,
        show_label: bool = True,
        show_confidence: bool = True,
        line_thickness: int = 2,
        box_color: int = 0,
        label_color: int = 0,
        font_size: int = 0,
        strict_frame_lock: bool | None = None,
        strict_wait_cap_ms: int | None = None,
    ) -> None:
        """
        Configure AI overlay with full control.

        Args:
            enabled: Enable or disable overlay
            show_label: Show class label on detections
            show_confidence: Show confidence score on detections
            line_thickness: Box line thickness (1-10)
            box_color: Box color in ARGB format (e.g. 0xFFFF0000 for red)
            label_color: Label color in ARGB format
            font_size: Font size (8-72)
            strict_frame_lock: opt in to / opt out of strict frame-lock on
                identity-fed display streams (streams whose ``stream_map``
                entry feeds them their own inference results). Strict mode
                waits — bounded by ``strict_wait_cap_ms`` — for each
                frame's own inference result at the bake site, so a box
                lands on the frame it was computed from instead of the
                newest frame. ``None`` keeps the daemon's current setting.
            strict_wait_cap_ms: max ms the bake site waits per frame before
                degrading to the freshest result (warned + counted). 0
                derives the cap from the stream fps (2 frame periods,
                clamp [1, 500]). ``None`` keeps the current setting; only
                meaningful together with ``strict_frame_lock``.
        """
        if strict_wait_cap_ms is not None and (
            isinstance(strict_wait_cap_ms, bool)
            or not isinstance(strict_wait_cap_ms, int)
            or not 0 <= strict_wait_cap_ms <= _STRICT_WAIT_CAP_MS_MAX
        ):
            raise ValueError(
                "strict_wait_cap_ms must be an int in "
                f"[0, {_STRICT_WAIT_CAP_MS_MAX}] (0 = derive from fps), "
                f"got {strict_wait_cap_ms!r}"
            )
        cfg = camera_pb2.AiOverlayConfig(
            enabled=enabled,
            show_label=show_label,
            show_confidence=show_confidence,
            line_thickness=line_thickness,
        )
        if box_color:
            cfg.box_color = box_color
        if label_color:
            cfg.label_color = label_color
        if font_size:
            cfg.font_size = font_size
        if strict_frame_lock is not None:
            cfg.strict_frame_lock = strict_frame_lock
        if strict_wait_cap_ms is not None:
            cfg.strict_wait_cap_ms = strict_wait_cap_ms
        self._update(cfg)
        logger.info("AI overlay configured")

    def apply(self, config: OverlayConfig) -> None:
        """Apply an OverlayConfig object."""
        self._update(config.to_proto())
        logger.info("AI overlay config applied")

    # ------------------------------------------------------------------
    # Content: push results to the hardware overlay via the event bus.
    # camera-daemon's AiOverlaySubscriber picks them off "inference/**"
    # and renders with HAL draw ops — no CPU raster in the app process.
    # ------------------------------------------------------------------

    def annotate(
        self,
        stream_id: str,
        detections: list | None = None,
        *,
        polygons: list | None = None,
        ttl_ms: int | None = None,
        session_id: str | None = None,
        frame_sequence: int | None = None,
        stream_epoch: int | None = None,
    ) -> str:
        """Draw detection boxes / zone polygons on ``stream_id``'s video.

        Args:
            stream_id: the camera stream the boxes belong to (e.g.
                ``"main"``) — camera-daemon keys overlay results by it.
            detections: :class:`~neoruntime_ipc_sdk.DetectedObject` items
                or dicts with ``label`` / ``score`` (or ``confidence``) and
                ``bbox`` (``{"x","y","width","height"}`` or ``[x, y, w, h]``).
                A detection's ``track_id`` (>= 0) rides along and renders
                as a ``#id`` label suffix. ``None`` leaves the stream's
                previous boxes untouched, so polygons can push alone.
            polygons: dicts ``{"points": [[x, y], ...], "label": str,
                "closed": bool, "color": ARGB int, "thickness": px int
                (-1 = filled)}`` with only ``points`` required — points
                are normalized [0, 1] floats, 2..128 per polygon, at most
                16 polygons per call. Any list (``[]`` included) replaces
                the stream's previous polygons; ``None`` keeps them.
            ttl_ms: positive int — per-event validity override for the
                stream's results. Omitted: the daemon derives a default
                from the stream's fps (about two frame periods).
            session_id: tag this event as belonging to a lifecycle
                session (P2-13). The daemon marks the stream's results
                with the tag and a ``"session/end"`` event sweeps matching
                tagged polygons on disconnect; an untagged write resets
                the mark. ``None`` / ``""`` leave the event untagged.
            frame_sequence: bind this event to that frame of the
                stream (positive int — ``InferenceResult.frame_sequence``
                from a ``subscribe()`` iteration is the natural source).
                The daemon draws the layer only while the bake site is
                within its frame-bind slack of it, then expires it, and
                rejects events whose frame the pipeline already passed.
                ``None`` leaves the event unbound (legacy semantics).
            stream_epoch: the stream generation counter (positive int,
                from ``get_stream_status()``'s ``stream_epoch``) this
                event was made against. A ReconfigureEncoder restart
                bumps the epoch — the daemon rejects events carrying a
                stale one instead of drawing ghosts over the new
                generation. ``None`` skips the check.

        Returns:
            The published event id.

        Results expire ``ttl_ms`` after the last event (fps-derived
        default), so call this at a few Hz while detections are fresh —
        publish an empty ``detections`` list to clear boxes, an empty
        ``polygons`` list to clear zones. The overlay itself must be on:
        ``enable()`` first.

        .. code-block:: python

            oc = OverlayClient()
            oc.enable()
            for result in inf.subscribe(stream, model):
                oc.annotate(stream, result.objects)

            # zone polygons ride independently of the boxes channel
            oc.annotate(stream, polygons=[{"points": ZONE, "label": "yard"}])
        """
        if ttl_ms is not None and (
            isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0
        ):
            raise ValueError(f"ttl_ms must be a positive int, got {ttl_ms!r}")
        payload: dict = {}
        if detections is not None:
            items = [_detection_dict(d) for d in detections]
            payload["num_detections"] = len(items)
            payload["detections"] = items
        if polygons is not None:
            if len(polygons) > _POLYGON_CAP:
                raise ValueError(
                    f"at most {_POLYGON_CAP} polygons per call, got {len(polygons)}"
                )
            payload["polygons"] = [_polygon_dict(p) for p in polygons]
        if not payload:  # annotate(stream) with neither — v1 clear-boxes call
            payload = {"num_detections": 0, "detections": []}
        return self._publish_overlay_event(
            stream_id,
            payload,
            ttl_ms=ttl_ms,
            session_id=session_id,
            frame_sequence=frame_sequence,
            stream_epoch=stream_epoch,
        )

    def annotate_result(
        self,
        stream_id: str,
        result: InferenceResult,
        *,
        ttl_ms: int | None = None,
        session_id: str | None = None,
        frame_sequence: int | None = None,
        stream_epoch: int | None = None,
    ) -> str:
        """Push whichever section of an :class:`InferenceResult` is populated.

        Detections take precedence, then classifications, landmarks and
        OCR lines — a payload carries one kind (that is how
        camera-daemon's parser discriminates them). An empty result
        publishes zero detections, clearing stale boxes. ``ttl_ms``
        overrides the stream's result validity window for this event.
        ``session_id`` tags the event with a lifecycle session (P2-13).
        ``frame_sequence`` / ``stream_epoch`` bind the event to the
        stream's frame generation — see :meth:`annotate`.
        """
        if result.objects:
            return self.annotate(
                stream_id,
                result.objects,
                ttl_ms=ttl_ms,
                session_id=session_id,
                frame_sequence=frame_sequence,
                stream_epoch=stream_epoch,
            )
        if result.classifications:
            payload = {
                "classifications": [_classification_dict(c) for c in result.classifications]
            }
        elif result.landmarks:
            payload = {"landmarks": [_landmark_dict(lm) for lm in result.landmarks]}
        elif result.ocr_lines:
            payload = {"ocr_lines": [_ocr_dict(line) for line in result.ocr_lines]}
        else:
            payload = {"num_detections": 0, "detections": []}
        return self._publish_overlay_event(
            stream_id,
            payload,
            ttl_ms=ttl_ms,
            session_id=session_id,
            frame_sequence=frame_sequence,
            stream_epoch=stream_epoch,
        )

    def _publish_overlay_event(
        self,
        stream_id: str,
        payload: dict,
        *,
        ttl_ms: int | None = None,
        session_id: str | None = None,
        frame_sequence: int | None = None,
        stream_epoch: int | None = None,
    ) -> str:
        metadata = {"stream_id": stream_id}
        if ttl_ms is not None:
            metadata["result_ttl_ms"] = str(int(ttl_ms))  # metadata is dict[str, str]
        if session_id:  # truthy only — None and "" are untagged, never a session named ""
            metadata["session_id"] = session_id
        # Frame-binding params ride metadata, never payload — the daemon's
        # parser string-scans the payload for its seven section keywords and
        # must not grow new ones. Validated here (the single sink both
        # annotate and annotate_result funnel through) so every path fails
        # fast client-side instead of being silently dropped daemon-side.
        if frame_sequence is not None:
            _validate_positive_int("frame_sequence", frame_sequence)
            metadata["frame_sequence"] = str(int(frame_sequence))
        if stream_epoch is not None:
            _validate_positive_int("stream_epoch", stream_epoch)
            metadata["stream_epoch"] = str(int(stream_epoch))
        return self._events().publish(
            _OVERLAY_TOPIC_PREFIX + stream_id,
            payload,
            metadata=metadata,
            compact=True,  # the daemon's parser string-scans for "bbox":[ / "polygons":[
        )


def _validate_positive_int(name: str, value) -> None:
    """int >= 1 guard for frame-binding metadata (bool excluded)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def _bbox_list(bbox) -> list[float]:
    if isinstance(bbox, dict):
        return [float(bbox["x"]), float(bbox["y"]), float(bbox["width"]), float(bbox["height"])]
    if hasattr(bbox, "width"):  # BoundingBox and duck-typed .x/.y/.width/.height
        return [float(bbox.x), float(bbox.y), float(bbox.width), float(bbox.height)]
    x, y, w, h = bbox
    return [float(x), float(y), float(w), float(h)]


def _detection_dict(item) -> dict:
    """Normalise a DetectedObject (or equivalent dict) to the wire schema."""
    if isinstance(item, DetectedObject):
        out = {
            "bbox": _bbox_list(item.bbox),
            "class_id": int(item.class_id),
            "confidence": float(item.score),
            "label": item.label,
        }
        track_id = item.track_id
    elif isinstance(item, dict):
        score = item.get("score", item.get("confidence", 0.0))
        out = {
            "bbox": _bbox_list(item["bbox"]),
            "class_id": int(item.get("class_id", 0)),
            "confidence": float(score),
            "label": str(item.get("label", "")),
        }
        track_id = item.get("track_id")
    else:
        # duck-typed fallback for app-local dataclasses
        out = {
            "bbox": _bbox_list(item.bbox),
            "class_id": int(getattr(item, "class_id", 0)),
            "confidence": float(item.score),
            "label": item.label,
        }
        track_id = getattr(item, "track_id", None)
    if track_id is not None and track_id >= 0:
        out["track_id"] = int(track_id)  # daemon renders it as a "#id" label suffix
    return out


def _polygon_dict(poly) -> dict:
    """Normalise one polygon spec to the daemon's wire schema.

    ``points`` are normalized [0, 1] floats — the daemon scales them to
    the frame at draw time. Validation is loud (ValueError/TypeError): a
    silently-clipped polygon is a wrong zone, not a cosmetic defect.
    """
    if not isinstance(poly, dict):
        raise TypeError(f"polygon must be a dict, got {type(poly).__name__}")
    raw_points = poly.get("points")
    if raw_points is None:
        raise ValueError("polygon requires 'points'")
    if not 2 <= len(raw_points) <= _POLYGON_POINT_CAP:
        raise ValueError(
            f"polygon needs 2..{_POLYGON_POINT_CAP} points, got {len(raw_points)}"
        )
    points = []
    for pt in raw_points:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"polygon point must be a 2-number pair, got {pt!r}")
        x, y = pt
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
        ):
            raise ValueError(f"polygon point must be numbers, got {pt!r}")
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"polygon point must be finite, got {pt!r}")
        points.append([x, y])
    out = {
        "points": points,
        "label": str(poly.get("label", "")),
        "closed": bool(poly.get("closed", True)),
    }
    if "color" in poly:
        out["color"] = int(poly["color"])
    if "thickness" in poly:
        out["thickness"] = int(poly["thickness"])
    return out


def _classification_dict(item: Classification) -> dict:
    if isinstance(item, dict):
        return {
            "class_id": int(item["class_id"]),
            "label": str(item.get("label", "")),
            "confidence": float(item.get("confidence", 0.0)),
        }
    return {
        "class_id": int(item.class_id),
        "label": item.label,
        "confidence": float(item.confidence),
    }


def _landmark_dict(item: LandmarkSet) -> dict:
    if isinstance(item, dict):
        points = item.get("points", [])
        lm_type = str(item.get("type", ""))
    else:
        points = item.points
        lm_type = str(item.type)
    return {
        "type": lm_type,
        "points": [
            {
                "x": float(p["x"] if isinstance(p, dict) else p.x),
                "y": float(p["y"] if isinstance(p, dict) else p.y),
                "confidence": float(
                    p.get("confidence", 1.0) if isinstance(p, dict) else getattr(p, "confidence", 1.0)
                ),
            }
            for p in points
        ],
    }


def _ocr_dict(item: OcrLine) -> dict:
    if isinstance(item, dict):
        return {
            "text": str(item.get("text", "")),
            "confidence": float(item.get("confidence", 0.0)),
            "bbox": _bbox_list(item["bbox"]),
        }
    return {
        "text": item.text,
        "confidence": float(item.confidence),
        "bbox": _bbox_list(item.bbox),
    }
