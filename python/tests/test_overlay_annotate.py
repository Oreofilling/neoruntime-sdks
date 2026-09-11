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


class TestAnnotatePolygons:
    """annotate(polygons=...) wire format — the daemon's polygon sidecar.

    Payload key "polygons" replaces the per-stream polygon sidecar on the
    daemon (empty list clears it); a payload without detection keys leaves
    stale boxes untouched, so zones and boxes can be pushed independently.
    """

    def test_polygon_payload_defaults(self, client, fake_bus):
        client.annotate(
            "main",
            polygons=[{"points": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]}],
        )
        call = fake_bus.instances[0].published[0]
        assert call["topic"] == "inference/main"
        assert call["metadata"] == {"stream_id": "main"}
        assert call["payload"] == {
            "polygons": [
                {"points": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], "label": "", "closed": True}
            ]
        }

    def test_polygon_options_passthrough(self, client, fake_bus):
        client.annotate(
            "sub",
            polygons=[
                {
                    "points": [(0, 0), (1, 0), (1, 1)],
                    "label": "zone-A",
                    "closed": False,
                    "color": 0xFFFFFF00,  # ARGB opaque yellow
                    "thickness": -1,  # filled
                }
            ],
        )
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload["polygons"][0] == {
            "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            "label": "zone-A",
            "closed": False,
            "color": 0xFFFFFF00,
            "thickness": -1,
        }

    def test_detections_and_polygons_combined(self, client, fake_bus):
        det = DetectedObject("person", 0.9, BoundingBox(1, 2, 3, 4))
        client.annotate(
            "main",
            [det],
            polygons=[{"points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "label": "zone"}],
        )
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload["num_detections"] == 1
        assert payload["detections"][0]["label"] == "person"
        assert payload["polygons"][0]["label"] == "zone"

    def test_empty_polygons_clears_only_polygons(self, client, fake_bus):
        client.annotate("main", polygons=[])
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload == {"polygons": []}

    def test_polygons_without_detections_keeps_boxes_channel(self, client, fake_bus):
        # a polygons-only payload must not carry detection keys, so the
        # daemon keeps the boxes from the previous detections event
        client.annotate("main", polygons=[{"points": [[0, 0], [1, 1]]}])
        payload = fake_bus.instances[0].published[0]["payload"]
        assert "detections" not in payload and "num_detections" not in payload

    def test_rejects_too_many_polygons(self, client):
        with pytest.raises(ValueError):
            client.annotate("main", polygons=[{"points": [[0, 0], [1, 1]]}] * 17)

    def test_rejects_bad_point_counts(self, client):
        with pytest.raises(ValueError):
            client.annotate("main", polygons=[{"points": [[0, 0]]}])  # 1 point
        with pytest.raises(ValueError):
            client.annotate("main", polygons=[{"points": [[0, 0]] * 129}])  # > HAL cap 128

    def test_rejects_non_finite_or_malformed_points(self, client):
        with pytest.raises(ValueError):
            client.annotate("main", polygons=[{"points": [[float("nan"), 0], [1, 1]]}])
        with pytest.raises(ValueError):
            client.annotate("main", polygons=[{"points": [[0, 0, 0], [1, 1]]}])
        with pytest.raises(ValueError):
            client.annotate("main", polygons=[{"points": ["a", "b"]}])

    def test_rejects_non_dict_polygon(self, client):
        with pytest.raises(TypeError):
            client.annotate("main", polygons=[(0.1, 0.1, 0.9, 0.9)])


class TestAnnotateTrackId:
    """track_id rides the detection dict; daemon renders it as a '#id' label suffix."""

    def test_detected_object_with_track_id(self, client, fake_bus):
        det = DetectedObject(
            label="person",
            score=0.8,
            bbox=BoundingBox(1, 2, 3, 4),
            track_id=7,
        )
        client.annotate("main", [det])
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload["detections"][0]["track_id"] == 7

    def test_track_id_omitted_when_absent_or_negative(self, client, fake_bus):
        det = DetectedObject("person", 0.8, BoundingBox(1, 2, 3, 4))
        client.annotate("main", [det])
        payload = fake_bus.instances[0].published[0]["payload"]
        assert "track_id" not in payload["detections"][0]
        client.annotate("main", [{"label": "x", "score": 0.5, "bbox": [1, 2, 3, 4], "track_id": -1}])
        payload = fake_bus.instances[0].published[1]["payload"]
        assert "track_id" not in payload["detections"][0]

    def test_dict_detection_with_track_id(self, client, fake_bus):
        client.annotate(
            "main", [{"label": "car", "score": 0.5, "bbox": [1, 2, 3, 4], "track_id": 3}]
        )
        payload = fake_bus.instances[0].published[0]["payload"]
        assert payload["detections"][0]["track_id"] == 3


class TestAnnotateTtl:
    """ttl_ms lands in event metadata — the daemon's per-result TTL override."""

    def test_ttl_ms_in_metadata(self, client, fake_bus):
        client.annotate("main", [], ttl_ms=120)
        call = fake_bus.instances[0].published[0]
        assert call["metadata"] == {"stream_id": "main", "result_ttl_ms": "120"}

    def test_annotate_result_accepts_ttl_ms(self, client, fake_bus):
        client.annotate_result("main", _result(), ttl_ms=66)
        call = fake_bus.instances[0].published[0]
        assert call["metadata"]["result_ttl_ms"] == "66"

    def test_invalid_ttl_rejected(self, client):
        with pytest.raises(ValueError):
            client.annotate("main", [], ttl_ms=0)
        with pytest.raises(ValueError):
            client.annotate("main", [], ttl_ms=-5)
        with pytest.raises(ValueError):
            client.annotate("main", [], ttl_ms="soon")


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


class TestAnnotateSessionTag:
    """P2-13: annotate/annotate_result tag events with the app session id.

    camera-daemon keeps session-scoped polygons per stream: a tagged
    write marks the stream's results as belonging to that session (an
    untagged write resets the mark), and a "session/end" event sweeps
    the matching tagged polygons. The tag rides METADATA only — the
    payload schema the daemon string-scans is untouched.
    """

    def test_annotate_session_id_rides_metadata(self, client, fake_bus):
        client.annotate("main", [], session_id="s1")
        call = fake_bus.instances[0].published[0]
        assert call["metadata"] == {"stream_id": "main", "session_id": "s1"}

    def test_annotate_without_session_leaves_metadata_untagged(self, client, fake_bus):
        client.annotate("main", [])
        call = fake_bus.instances[0].published[0]
        assert call["metadata"] == {"stream_id": "main"}

    def test_annotate_empty_session_id_is_untagged(self, client, fake_bus):
        # "" is the "no session" default, not a session named ""
        client.annotate("main", [], session_id="")
        call = fake_bus.instances[0].published[0]
        assert call["metadata"] == {"stream_id": "main"}

    def test_annotate_result_forwards_session_id(self, client, fake_bus):
        det = DetectedObject(
            label="person", score=0.9, bbox=BoundingBox(1.0, 2.0, 3.0, 4.0), class_id=0
        )
        client.annotate_result("main", _result(objects=[det]), session_id="s2")
        call = fake_bus.instances[0].published[0]
        assert call["metadata"]["session_id"] == "s2"

    def test_annotate_result_default_is_untagged(self, client, fake_bus):
        det = DetectedObject(
            label="person", score=0.9, bbox=BoundingBox(1.0, 2.0, 3.0, 4.0), class_id=0
        )
        client.annotate_result("main", _result(objects=[det]))
        call = fake_bus.instances[0].published[0]
        assert "session_id" not in call["metadata"]


class TestFrameBinding:
    """Frame-sync binding kwargs — annotate_result(frame_sequence=, stream_epoch=).

    camera-daemon's behavior-decoupling contract: an app event may pin
    itself to the stream's frame generation. frame_sequence (from
    InferenceResult.frame_sequence of a subscribe() iteration) binds the
    layer to that frame — it draws only while the bake site is within
    kFrameBindSlack of it; stream_epoch (from get_stream_status()) is
    the generation counter that ReconfigureEncoder bumps — an event
    carrying a stale epoch is rejected instead of haunting the stream.
    Both ride METADATA only; the payload schema is untouched. Both are
    opt-in: absent kwargs leave the event unbound (legacy semantics).
    """

    def test_annotate_result_frame_binding_rides_metadata(self, client, fake_bus):
        client.annotate_result("main", _result(), frame_sequence=102, stream_epoch=7)
        call = fake_bus.instances[0].published[0]
        assert call["metadata"] == {
            "stream_id": "main",
            "frame_sequence": "102",
            "stream_epoch": "7",
        }

    def test_annotate_result_detections_branch_forwards_binding(self, client, fake_bus):
        det = DetectedObject(
            label="person", score=0.9, bbox=BoundingBox(1.0, 2.0, 3.0, 4.0), class_id=0
        )
        client.annotate_result("main", _result(objects=[det]), frame_sequence=5, stream_epoch=9)
        call = fake_bus.instances[0].published[0]
        assert call["metadata"]["frame_sequence"] == "5"
        assert call["metadata"]["stream_epoch"] == "9"
        # the binding rides metadata, never the string-scanned payload
        assert "frame_sequence" not in call["payload"]

    def test_annotate_accepts_frame_binding_too(self, client, fake_bus):
        client.annotate("main", [], frame_sequence=11, stream_epoch=3)
        call = fake_bus.instances[0].published[0]
        assert call["metadata"]["frame_sequence"] == "11"
        assert call["metadata"]["stream_epoch"] == "3"

    def test_default_leaves_event_unbound(self, client, fake_bus):
        client.annotate_result("main", _result())
        call = fake_bus.instances[0].published[0]
        assert call["metadata"] == {"stream_id": "main"}

    def test_invalid_frame_sequence_rejected(self, client):
        for bad in (0, -1, "102", 1.5, True):
            with pytest.raises(ValueError):
                client.annotate_result("main", _result(), frame_sequence=bad)

    def test_invalid_stream_epoch_rejected(self, client):
        for bad in (0, -1, "7", 2.5, True):
            with pytest.raises(ValueError):
                client.annotate_result("main", _result(), stream_epoch=bad)
