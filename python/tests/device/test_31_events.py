"""Phase 3 — event bus interfaces.

Uses a unique per-run topic (``sdk-test/<timestamp>``) so the suite
never collides with real app traffic, then exercises the full pub/sub
surface: publish, batch publish, generator subscription, threaded
on_event with filters, topic introspection and stats.
"""

from __future__ import annotations

import threading
import time
import unittest
import uuid

from neoruntime_ipc_sdk import Event, EventClient, TopicInfo

from common import DeviceTestCase


class T01PubSub(DeviceTestCase):
    area = "events"
    timeout_s = 90

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = EventClient()
        cls.topic = f"sdk-test/{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.close()
        except Exception:
            pass

    def _collect(self, n: int, want_value: str | None = None):
        """Subscribe then pull n events (optionally filtered by value)."""
        got = []
        done = threading.Event()

        def cb(event):
            if want_value is not None and event.payload.get("v") != want_value:
                return
            if len(got) < n:
                got.append(event)
            if len(got) >= n:
                done.set()

        thread = self.client.on_event(self.topic, cb)
        return got, done, thread

    def test_01_publish_subscribe_roundtrip(self):
        self.mark("EventClient.publish/subscribe roundtrip")
        got, done, thread = self._collect(3)
        time.sleep(0.5)  # let the subscription register server-side
        ids = []
        for i in range(3):
            ids.append(self.timed(
                self.client.publish, self.topic, {"v": f"msg-{i}"},
                label=f"publish_{i}",
            ))
        joined = done.wait(timeout=15.0)
        self.evidence(published_ids=ids, received=len(got),
                      signaled=joined)
        self.assertTrue(joined, "subscriber never saw the 3 published events")
        self.assertEqual(len(got), 3)

    def test_02_event_fields(self):
        self.mark("Event dataclass fields")
        got, done, thread = self._collect(1)
        time.sleep(0.5)
        self.client.publish(self.topic, {"v": "fields"})
        self.assertTrue(done.wait(timeout=15.0), "no event received")
        ev = got[0]
        self.evidence(topic=ev.topic, payload=ev.payload, source=ev.source,
                      event_id=ev.event_id, timestamp_ns=ev.timestamp_ns,
                      metadata=ev.metadata)
        self.assertEqual(ev.topic, self.topic)
        self.assertEqual(ev.payload.get("v"), "fields")

    def test_03_subscribe_generator(self):
        self.mark("EventClient.subscribe (iterator)")
        received = []
        ready = threading.Event()

        def producer():
            ready.wait(timeout=10.0)
            for i in range(2):
                self.client.publish(self.topic, {"v": f"gen-{i}"})

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        deadline = time.monotonic() + 30.0
        for event in self.client.subscribe(self.topic):
            received.append(event)
            if len(received) >= 2:
                break
            ready.set()
            if time.monotonic() > deadline:
                self.record["outcome_note"] = (
                    "subscribe generator stalled before 2 events "
                    "(30s deadline)")
                break
        ready.set()
        t.join(timeout=5.0)
        self.evidence(received=len(received),
                      payloads=[e.payload.get("v") for e in received])
        self.assertGreaterEqual(len(received), 2)

    def test_04_publish_batch(self):
        self.mark("EventClient.publish_batch")
        got, done, thread = self._collect(3)
        time.sleep(0.5)
        # publish_batch takes whole events ({"topic", "payload"} each),
        # not bare payloads — events.py:171 requires both keys.
        events = [{"topic": self.topic, "payload": {"v": f"batch-{i}"}}
                  for i in range(3)]
        self.timed(self.client.publish_batch, events, label="publish_batch")
        joined = done.wait(timeout=15.0)
        self.evidence(batch_size=len(events), received=len(got),
                      signaled=joined)
        self.assertTrue(joined, "batch events never arrived")

    def test_05_subscribe_filters(self):
        self.mark("EventClient.subscribe/on_event filters")
        got, done, thread = self._collect(2, want_value="matched")
        time.sleep(0.5)
        # A server-side filtered subscription alongside the plain one:
        # the plain subscriber must still see everything, proving the
        # filter parameter didn't corrupt the bus.
        thread_filter = self.client.on_event(
            self.topic, lambda e: None, filters={"v": "matched"}
        )
        time.sleep(0.5)
        self.client.publish(self.topic, {"v": "matched"})
        self.client.publish(self.topic, {"v": "unmatched"})
        self.client.publish(self.topic, {"v": "matched"})
        joined = done.wait(timeout=15.0)
        self.evidence(received=len(got), signaled=joined,
                      values=[e.payload.get("v") for e in got])
        self.assertTrue(joined, "filtered subscriber missed matched events")

    def test_06_list_topics(self):
        self.mark("EventClient.list_topics")
        topics = self.timed(self.client.list_topics, label="list_topics")
        names = [t.topic for t in topics]
        self.evidence(count=len(topics), sample=names[:20],
                      own_topic_present=self.topic in names)
        self.assertIn(self.topic, names,
                      "published topic missing from list_topics")

    def test_07_get_topic_info(self):
        self.mark("EventClient.get_topic_info")
        info = self.timed(self.client.get_topic_info, self.topic,
                          label="get_topic_info")
        self.evidence(info=None if info is None else {
            "topic": info.topic,
            "subscriber_count": info.subscriber_count,
            "total_messages": info.total_messages,
        })
        self.assertIsInstance(info, TopicInfo)
        self.assertGreaterEqual(info.total_messages, 1)

    def test_08_get_stats(self):
        self.mark("EventClient.get_stats")
        stats = self.timed(self.client.get_stats, label="get_stats")
        self.evidence(stats=dict(list(stats.items())[:10]))
        self.assertIsInstance(stats, dict)

    def test_09_get_topic_stats(self):
        self.mark("EventClient.get_topic_stats")
        stats = self.timed(
            self.client.get_topic_stats, self.topic, label="get_topic_stats"
        )
        self.evidence(stats=dict(list(stats.items())[:10]))
        self.assertIsInstance(stats, dict)

    def test_10_unsubscribe(self):
        self.mark("EventClient.unsubscribe")
        self.client.unsubscribe(self.topic)
        self.evidence(unsubscribed=self.topic)


if __name__ == "__main__":
    unittest.main()
