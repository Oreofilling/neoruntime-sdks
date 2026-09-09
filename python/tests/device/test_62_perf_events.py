"""Perf P3 — event bus: publish RPC, batch throughput, delivery E2E.

Delivery latency is measured by matching ``event_id`` from a live
subscriber (publish → arrival on the client side), and arrival jitter
is measured by publishing at a fixed 10 Hz cadence and consuming the
subscriber stream — the payload carries a monotonically increasing
``seq`` so drop accounting works even though the bus assigns no
sequence numbers of its own.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest

from neoruntime_ipc_sdk import EventClient

from perf_common import PerfTestCase

TOPIC = "sdk-perf/events"
STREAM_S = 30
PUBLISH_HZ = 10


class _EventPerfBase(PerfTestCase):
    area = "perf-events"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = EventClient()
        cls.client.connect()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.unsubscribe(TOPIC)
        except Exception:
            pass
        cls.client.close()


class T01Publish(_EventPerfBase):
    """Fire-and-forget publish RPC latency (no subscriber attached)."""

    timeout_s = 300

    def test_01_publish(self):
        self.mark("EventClient.publish latency")
        i = [0]

        def one():
            i[0] += 1
            return self.client.publish(TOPIC, {"seq": i[0]})

        self.perf_sample(one, label="publish", n=200)

    def test_02_publish_batch(self):
        self.mark("EventClient.publish_batch throughput (100/batch)")
        base = [0]

        def one():
            base[0] += 100
            events = [{"topic": TOPIC, "payload": {"seq": base[0] + k}}
                      for k in range(100)]
            return self.client.publish_batch(events)

        stats = self.perf_sample(one, label="publish_batch_100", n=10,
                                 rounds=1)
        if stats.get("p50"):
            self.evidence(
                events_per_sec=round(100 / (stats["p50"] / 1000.0), 1))


class T02Delivery(_EventPerfBase):
    """Publish → subscriber arrival, matched by event_id."""

    timeout_s = 300

    def test_01_end_to_end(self):
        self.mark("EventClient publish→subscribe delivery latency")
        q: queue.Queue = queue.Queue(maxsize=500)

        def pump():
            try:
                for ev in self.client.subscribe(TOPIC):
                    q.put(ev)
            except Exception:  # noqa: BLE001 — daemon thread, diagnostic only
                pass

        threading.Thread(target=pump, daemon=True).start()
        time.sleep(1.0)  # subscription established before first publish

        def one():
            eid = self.client.publish(TOPIC, {"probe": True})
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    ev = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if ev.event_id == eid:
                    return ev
            raise TimeoutError("event not delivered within 5s")

        stats = self.perf_sample(one, label="delivery_e2e", n=100,
                                 rounds=1)
        self.assertLess((stats.get("err_pct") or 0), 50.0,
                        "over half the probes never arrived")

    def test_02_arrival_jitter(self):
        self.mark("EventClient 10 Hz stream arrival statistics")
        seq_box = [0]
        stop = threading.Event()

        def publish_cadence():
            interval = 1.0 / PUBLISH_HZ
            t_next = time.monotonic()
            while not stop.is_set():
                seq_box[0] += 1
                try:
                    self.client.publish(TOPIC, {"seq": seq_box[0]})
                except Exception:  # noqa: BLE001 — publisher must keep cadence
                    pass
                t_next += interval
                delay = t_next - time.monotonic()
                if delay > 0:
                    stop.wait(delay)

        gen = self.client.subscribe(TOPIC)
        pub = threading.Thread(target=publish_cadence, daemon=True)
        pub.start()
        try:
            stats = self.perf_stream(
                gen, label="arrival_10hz", duration_s=STREAM_S,
                seq_of=lambda ev: ev.payload.get("seq"),
            )
            self.assertGreater(stats.get("frames", 0), 0,
                               "no events arrived in the window")
        finally:
            stop.set()
            pub.join(timeout=5)
            try:
                self.client.unsubscribe(TOPIC)
            except Exception:
                pass

    def test_03_bus_stats(self):
        self.mark("EventClient.get_topic_stats / list_topics")
        self.perf_sample(self.client.get_topic_stats, TOPIC,
                         label="get_topic_stats", n=50)
        topics = self.client.list_topics()
        self.evidence(topic_info={
            t.topic: {"subscribers": getattr(t, "subscriber_count", None),
                      "published": getattr(t, "published_count", None)}
            for t in topics if t.topic == TOPIC})


if __name__ == "__main__":
    unittest.main()
