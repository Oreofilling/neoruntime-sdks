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


@dataclass
class OverlayConfig:
    """AI overlay configuration."""

    enabled: bool = True
    show_label: bool = True
    show_confidence: bool = True
    line_thickness: int = 2
    box_color: int = 0
    label_color: int = 0
    font_size: int = 0

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
        """
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

    def annotate(self, stream_id: str, detections: list) -> str:
        """Draw detection boxes on ``stream_id``'s encoded video.

        Args:
            stream_id: the camera stream the boxes belong to (e.g.
                ``"main"``) — camera-daemon keys overlay results by it.
            detections: :class:`~neoruntime_ipc_sdk.DetectedObject` items
                or dicts with ``label`` / ``score`` (or ``confidence``) and
                ``bbox`` (``{"x","y","width","height"}`` or ``[x, y, w, h]``).

        Returns:
            The published event id.

        The overlay expires results 500 ms after the last event, so call
        this at a few Hz while detections are fresh — and publish an
        empty ``detections`` list to clear the boxes. The overlay itself
        must be on: ``enable()`` first.

        .. code-block:: python

            oc = OverlayClient()
            oc.enable()
            for result in inf.subscribe(stream, model):
                oc.annotate(stream, result.objects)
        """
        items = [_detection_dict(d) for d in detections]
        payload = {
            "num_detections": len(items),
            "detections": items,
        }
        return self._publish_overlay_event(stream_id, payload)

    def annotate_result(self, stream_id: str, result: InferenceResult) -> str:
        """Push whichever section of an :class:`InferenceResult` is populated.

        Detections take precedence, then classifications, landmarks and
        OCR lines — a payload carries one kind (that is how
        camera-daemon's parser discriminates them). An empty result
        publishes zero detections, clearing stale boxes.
        """
        if result.objects:
            return self.annotate(stream_id, result.objects)
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
        return self._publish_overlay_event(stream_id, payload)

    def _publish_overlay_event(self, stream_id: str, payload: dict) -> str:
        return self._events().publish(
            _OVERLAY_TOPIC_PREFIX + stream_id,
            payload,
            metadata={"stream_id": stream_id},
            compact=True,  # the daemon's parser string-scans for "bbox":[
        )


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
        return {
            "bbox": _bbox_list(item.bbox),
            "class_id": int(item.class_id),
            "confidence": float(item.score),
            "label": item.label,
        }
    if isinstance(item, dict):
        score = item.get("score", item.get("confidence", 0.0))
        return {
            "bbox": _bbox_list(item["bbox"]),
            "class_id": int(item.get("class_id", 0)),
            "confidence": float(score),
            "label": str(item.get("label", "")),
        }
    # duck-typed fallback for app-local dataclasses
    return {
        "bbox": _bbox_list(item.bbox),
        "class_id": int(getattr(item, "class_id", 0)),
        "confidence": float(item.score),
        "label": item.label,
    }


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
