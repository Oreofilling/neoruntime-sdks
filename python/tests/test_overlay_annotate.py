"""Tests for OverlayClient.annotate / annotate_result — event-bus wire format.

camera-daemon's AiOverlaySubscriber requires topic "inference/<stream_id>",
metadata["stream_id"], and compact JSON (it string-scans for "bbox":[), so
these tests pin all three. No real socket is touched: EventClient is faked.
"""

from __future__ import annotations

import pytest

import neoruntime_ipc_sdk.overlay as overlay_module
from neoruntime_ipc_sdk import (
    BoundingBox,
    Classification,
    DetectedObject,
    InferenceResult,
    LandmarkPoint,
    LandmarkSet,
    OcrLine,
    OverlayClient,
)


class FakeEventClient:
    instances: list[FakeEventClient] = []

    def __init__(self, *args, **kwargs):
        self.published: list[dict] = []
        self.closed = False
        FakeEventClient.instances.append(self)

    def publish(self, topic, payload, persistent=False, ttl_ms=None, metadata=None, compact=False):
        self.published.append(
            {
                "topic": topic,
                "payload": payload,
                "persistent": persistent,
                "ttl_ms": ttl_ms,
                "metadata": metadata,
                "compact": compact,
            }
        )
        return f"evt-{len(self.published)}"

    def close(self):
        self.closed = True


@pytest.fixture
def fake_bus(monkeypatch):
    FakeEventClient.instances = []
    monkeypatch.setattr(overlay_module, "EventClient", FakeEventClient)
    return FakeEventClient


@pytest.fixture
def client(fake_bus):
    oc = OverlayClient(endpoint="unix:///tmp/test-overlay.sock")
    return oc


def _result(**sections) -> InferenceResult:
    return InferenceResult(frame_sequence=1, timestamp_ns=1, **sections)


class TestAnnotate:
    def test_wire_format_for_detected_objects(self, client, fake_bus):
        det = DetectedObject(
            label="person",
            score=0.87,
            bbox=BoundingBox(10.0, 20.0, 30.0, 40.0),
            class_id=3,
        )
        client.annotate("main", [det])
        bus = fake_bus.instances[0]
        assert len(bus.published) == 1
        call = bus.published[0]
        # topic + metadata + compact are the daemon's hard requirements
        assert call["topic"] == "inference/main"
        assert call["metadata"] == {"stream_id": "main"}
        assert call["compact"] is True
        assert call["payload"] == {
            "num_detections": 1,
            "detections": [
                {
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "class_id": 3,
                    "confidence": 0.87,
                    "label": "person",
                }
            ],
        }

    def test_dict_detections_with_score_and_list_bbox(self, client, fake_bus):
        client.annotate("sub", [{"label": "car", "score": 0.5, "bbox": [1, 2, 3, 4]}])
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload["detections"][0]["confidence"] == 0.5
        assert payload["detections"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]

    def test_dict_detections_with_confidence_and_bbox_dict(self, client, fake_bus):
        client.annotate(
            "sub",
            [
                {
                    "label": "car",
                    "confidence": 0.9,
                    "class_id": 2,
                    "bbox": {"x": 5, "y": 6, "width": 7, "height": 8},
                }
            ],
        )
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload["detections"][0] == {
            "bbox": [5.0, 6.0, 7.0, 8.0],
            "class_id": 2,
            "confidence": 0.9,
            "label": "car",
        }

    def test_empty_list_clears_boxes(self, client, fake_bus):
        client.annotate("main", [])
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload == {"num_detections": 0, "detections": []}

    def test_returns_event_id(self, client, fake_bus):
        assert client.annotate("main", []) == "evt-1"


class TestAnnotateResult:
    def test_objects_take_precedence(self, client, fake_bus):
        result = _result(
            objects=[DetectedObject("cat", 0.9, BoundingBox(0, 0, 5, 5))],
            classifications=[Classification("kind", 0, "cat", 0.9)],
        )
        client.annotate_result("main", result)
        payload = fake_bus.instances[0].published[0]["payload"]
        assert "detections" in payload and "classifications" not in payload

    def test_classifications_payload(self, client, fake_bus):
        result = _result(classifications=[Classification("kind", 7, "dog", 0.66)])
        client.annotate_result("main", result)
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload == {"classifications": [{"class_id": 7, "label": "dog", "confidence": 0.66}]}

    def test_landmarks_payload(self, client, fake_bus):
        lm = LandmarkSet(
            type="face",
            points=[LandmarkPoint(1.5, 2.5, 0.9), LandmarkPoint(3.5, 4.5)],
        )
        client.annotate_result("main", _result(landmarks=[lm]))
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload == {
            "landmarks": [
                {
                    "type": "face",
                    "points": [
                        {"x": 1.5, "y": 2.5, "confidence": 0.9},
                        {"x": 3.5, "y": 4.5, "confidence": 1.0},
                    ],
                }
            ]
        }

    def test_ocr_payload(self, client, fake_bus):
        line = OcrLine("ABC-123", 0.8, BoundingBox(1, 2, 3, 4))
        client.annotate_result("main", _result(ocr_lines=[line]))
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload == {"ocr_lines": [{"text": "ABC-123", "confidence": 0.8, "bbox": [1.0, 2.0, 3.0, 4.0]}]}

    def test_empty_result_clears(self, client, fake_bus):
        client.annotate_result("main", _result())
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload == {"num_detections": 0, "detections": []}


class TestLifecycle:
    def test_event_client_is_lazy(self, client, fake_bus):
        assert fake_bus.instances == []

    def test_close_closes_child_event_client(self, client, fake_bus):
        client.annotate("main", [])
        bus = fake_bus.instances[0]
        assert not bus.closed
        client.close()
        assert bus.closed

    def test_close_without_annotate_is_safe(self, client):
        client.close()
