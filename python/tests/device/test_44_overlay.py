"""Phase 4 — AI overlay interfaces (OverlayClient).

``annotate`` pushes over the event bus (topic ``inference/<stream>``),
so its verification is a subscription round-trip: the payload we publish
must come back with the wire schema the daemon's parser expects
(``bbox`` as a 4-list). The overlay config itself ends disabled —
the pre-test state — after being exercised.
"""

from __future__ import annotations

import threading
import time
import unittest

from neoruntime_ipc_sdk import (
    BoundingBox,
    DetectedObject,
    EventClient,
    InferenceResult,
    OverlayClient,
    OverlayConfig,
)

from common import DeviceTestCase

STREAM = "main"


class T01Config(DeviceTestCase):
    area = "overlay"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = OverlayClient()

    @classmethod
    def tearDownClass(cls):
        cls.client.disable()  # pre-test state: overlay off
        cls.client.close()

    def test_01_enable_disable(self):
        self.mark("OverlayClient.enable/disable")
        self.timed(self.client.enable, label="enable")
        self.timed(self.client.disable, label="disable")
        self.evidence(note="enabled then disabled; left disabled")

    def test_02_configure(self):
        self.mark("OverlayClient.configure")
        self.timed(
            self.client.configure, True, True, True, 3,
            0xFF00FF00, 0xFFFFFF00, 24,  # box green, label yellow, 24pt
            label="configure",
        )
        self.client.disable()
        self.evidence(note="full-parameter configure accepted, then disabled")

    def test_03_apply(self):
        self.mark("OverlayClient.apply/OverlayConfig")
        cfg = OverlayConfig(enabled=True, show_label=True,
                            show_confidence=True, line_thickness=2,
                            box_color=0xFFFF0000, label_color=0xFFFFFFFF,
                            font_size=16)
        self.timed(self.client.apply, cfg, label="apply")
        self.client.disable()
        self.evidence(config=vars(cfg))


class T02Annotate(DeviceTestCase):
    area = "overlay"
    timeout_s = 90

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = OverlayClient()
        cls.events = EventClient()
        cls.client.enable()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.annotate(STREAM, [])  # clear any drawn boxes
        except Exception:
            pass
        cls.client.disable()
        cls.client.close()
        cls.events.close()

    def _subscribe(self, n: int):
        got: list = []
        done = threading.Event()

        def cb(event):
            if len(got) < n:
                got.append(event)
            if len(got) >= n:
                done.set()

        self.events.on_event(f"inference/{STREAM}", cb)
        time.sleep(0.5)  # registration lag on the server side
        return got, done

    def test_01_annotate_detected_objects(self):
        self.mark("OverlayClient.annotate")
        got, done = self._subscribe(1)
        det = DetectedObject(
            label="sdk-test", score=0.99,
            bbox=BoundingBox(x=100, y=100, width=200, height=150),
        )
        event_id = self.timed(self.client.annotate, STREAM, [det],
                              label="annotate")
        joined = done.wait(timeout=15.0)
        if not joined:
            self.client.annotate(STREAM, [])
            self.fail("annotate event never appeared on inference/main")
        payload = got[0].payload
        wire = payload["detections"][0]
        self.evidence(event_id=event_id, num=payload.get("num_detections"),
                      first=wire)
        self.assertEqual(payload.get("num_detections"), 1)
        self.assertEqual([float(v) for v in wire["bbox"]],
                         [100.0, 100.0, 200.0, 150.0],
                         "bbox not serialised to the daemon's 4-list schema")
        self.client.annotate(STREAM, [])  # clear

    def test_02_annotate_result(self):
        self.mark("OverlayClient.annotate_result")
        got, done = self._subscribe(1)
        result = InferenceResult(frame_sequence=0, timestamp_ns=0)
        result.objects = [DetectedObject(
            label="probe", score=0.5,
            bbox=BoundingBox(x=0, y=0, width=64, height=64),
        )]
        event_id = self.timed(self.client.annotate_result, STREAM, result,
                              label="annotate_result")
        joined = done.wait(timeout=15.0)
        self.evidence(event_id=event_id, signaled=joined)
        if not joined:
            self.client.annotate(STREAM, [])
            self.fail("annotate_result event never appeared on the bus")
        self.client.annotate(STREAM, [])  # clear

    def test_03_annotate_clears(self):
        self.mark("OverlayClient.annotate (clear)")
        got, done = self._subscribe(1)
        event_id = self.timed(self.client.annotate, STREAM, [],
                              label="annotate_clear")
        joined = done.wait(timeout=15.0)
        self.evidence(event_id=event_id, signaled=joined,
                      payload=got[0].payload if got else None)
        if not joined:
            self.fail("clear event never appeared on the bus")
        self.assertEqual(got[0].payload.get("num_detections"), 0)


if __name__ == "__main__":
    unittest.main()
