"""
P2-11 whole-pipeline facade: StreamPipeline ties the per-step interfaces
into one platform-scheduled call.

Contract under test:
  - start() subscribes via InferenceClient.subscribe (daemon-fed, no app
    pixels) on a worker thread, pushes each result through
    OverlayClient.annotate_result / annotate, and mirrors results into a
    bounded drop-oldest app queue consumed via results().
  - min_score/labels filter only what is DRAW; the app queue still sees
    the unfiltered result. A fully-filtered result still publishes an
    empty detection list so stale boxes clear.
  - on_result may replace the result or drop it (None) — a dropped
    result is neither drawn nor queued.
  - Annotate errors are counted and non-fatal; the loop keeps going.
  - stop() is idempotent, joins the worker, clears boxes (and static
    polygons when any were set), and closes only the clients it created.
    Safe to call from ``on_result``: the self-join is skipped.
"""

import queue
import threading
import time

import pytest

from neoruntime_ipc_sdk import stream_pipeline
from neoruntime_ipc_sdk.inference_types import (
    BoundingBox,
    DetectedObject,
    InferenceResult,
)
from neoruntime_ipc_sdk.overlay import OverlayConfig
from neoruntime_ipc_sdk.stream_pipeline import StreamPipeline, StreamPipelineStatus

_STOP = object()
_EXC = object()


def _obj(label="person", score=0.9):
    return DetectedObject(
        label=label,
        score=score,
        bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
        class_id=1,
    )


def _result(seq, objects=()):
    return InferenceResult(frame_sequence=seq, timestamp_ns=1000, objects=list(objects))


def _wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class _FakeSubIterator:
    """Stand-in for the _SubscribeIterator subscribe() returns."""

    dropped = 0
    last_latency_ms = 0.0
    avg_latency_ms = 0.0
    last_skew_us = 0
    avg_skew_us = 0.0

    def __init__(self, client):
        self._client = client

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self._client._items.get(timeout=0.05)
            except queue.Empty:
                if self._client.cancelled.is_set():
                    raise StopIteration from None
                continue
            if item is _STOP:
                raise StopIteration from None
            if item is _EXC:
                raise RuntimeError("stream broken") from None
            return item

    def cancel(self):
        self._client.cancelled.set()


class _FakeInferenceClient:
    def __init__(self):
        self._items = queue.Queue()
        self.cancelled = threading.Event()
        self.subscriptions = []
        self.iterator = None
        self.closed = False

    def subscribe(self, stream, model, fps=10, session_id="", raw_output_only=False,
                  max_consecutive_failures=10, queue_size=100):
        self.subscriptions.append(
            {
                "stream": stream,
                "model": model,
                "fps": fps,
                "session_id": session_id,
                "raw_output_only": raw_output_only,
                "queue_size": queue_size,
            }
        )
        self.iterator = _FakeSubIterator(self)
        return self.iterator

    def close(self):
        self.closed = True

    # test helpers -----------------------------------------------------
    def push(self, seq, result):
        self._items.put((seq, result))

    def end(self):
        self._items.put(_STOP)

    def fail(self):
        self._items.put(_EXC)


class _FakeOverlayClient:
    def __init__(self):
        self.calls = []
        self.session_ids = []  # parallel to calls: the session_id kwarg (P2-13)
        self.closed = False
        self.fail_next = 0

    def apply(self, config):
        self.calls.append(("apply", config))
        self.session_ids.append(None)

    def annotate(self, stream_id, detections=None, *, polygons=None, ttl_ms=None,
                 session_id=None):
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("overlay down")
        self.calls.append(("annotate", stream_id, detections, polygons, ttl_ms))
        self.session_ids.append(session_id)
        return "evt"

    def annotate_result(self, stream_id, result, *, ttl_ms=None, session_id=None):
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("overlay down")
        self.calls.append(("annotate_result", stream_id, result, ttl_ms))
        self.session_ids.append(session_id)
        return "evt"

    def close(self):
        self.closed = True


ZONES = [{"points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], "label": "yard"}]


class TestStartAndLifecycle:
    def test_start_forwards_subscribe_params_applies_config_pushes_zones(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        cfg = OverlayConfig(enabled=True, show_label=False)
        p = StreamPipeline(
            "third", "yolov5n", fps=7, session_id="s1", queue_size=11,
            polygons=ZONES, overlay_config=cfg,
            inference=inf, overlay=ov,
        )
        p.start()
        try:
            assert inf.subscriptions == [{
                "stream": "third", "model": "yolov5n", "fps": 7,
                "session_id": "s1", "raw_output_only": False, "queue_size": 11,
            }]
            kinds = [c[0] for c in ov.calls]
            assert kinds[0] == "apply"
            assert ov.calls[0][1] is cfg
            assert ("annotate", "third", None, ZONES, None) in ov.calls

            inf.push(1, _result(1, [_obj()]))
            gen = p.results()
            seq, got = next(gen)
            assert seq == 1 and got.frame_sequence == 1
            assert _wait_until(lambda: p.status().results_annotated == 1)
        finally:
            p.stop()

    def test_ttl_forwarded_to_annotate_result(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", ttl_ms=800, inference=inf, overlay=ov)
        p.start()
        try:
            r = _result(2, [_obj()])
            inf.push(2, r)
            _wait_until(lambda: ov.calls)
            assert ov.calls == [("annotate_result", "third", r, 800)]
        finally:
            p.stop()

    def test_draw_false_never_touches_overlay(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", draw=False, polygons=ZONES,
                           inference=inf, overlay=ov)
        p.start()
        try:
            inf.push(1, _result(1, [_obj()]))
            seq, _ = next(p.results())
            assert seq == 1
            _wait_until(lambda: p.status().results_seen == 1)
            st = p.status()
            assert st.results_annotated == 0 and ov.calls == []
        finally:
            p.stop()
            assert ov.calls == []  # nothing to clear: nothing was drawn

    def test_start_twice_and_restart_after_stop_both_raise(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", inference=inf, overlay=ov)
        p.start()
        with pytest.raises(RuntimeError):
            p.start()
        p.stop()
        with pytest.raises(RuntimeError):
            p.start()  # single-shot: create a new StreamPipeline instead

    def test_stop_before_start_is_a_noop(self):
        StreamPipeline("third", "m").stop()  # must not raise

    def test_stop_closes_clients_it_created(self, monkeypatch):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        monkeypatch.setattr(stream_pipeline, "InferenceClient", lambda: inf)
        monkeypatch.setattr(stream_pipeline, "OverlayClient", lambda: ov)
        p = StreamPipeline("third", "m")
        p.start()
        p.stop()
        assert inf.closed and ov.closed

    def test_stop_leaves_injected_clients_open_and_is_idempotent(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", inference=inf, overlay=ov)
        p.start()
        p.stop()
        p.stop()  # second stop: no error, no double-clear
        assert not inf.closed and not ov.closed
        clears = [c for c in ov.calls if c[0] == "annotate"]
        assert clears == [("annotate", "third", [], None, None)]  # boxes only

    def test_stop_clears_static_polygons_too(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", polygons=ZONES, inference=inf, overlay=ov)
        p.start()
        p.stop()
        assert ("annotate", "third", [], None, None) in ov.calls
        assert ("annotate", "third", None, [], None) in ov.calls  # zones cleared

    def test_context_manager_stops_on_exit(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        with StreamPipeline("third", "m", inference=inf, overlay=ov) as p:
            p.start()
            inf.push(1, _result(1))
            assert next(p.results())[0] == 1
        assert not p.status().running
        assert ("annotate", "third", [], None, None) in ov.calls

    def test_results_before_start_raises(self):
        with pytest.raises(RuntimeError):
            StreamPipeline("third", "m").results()

    def test_stop_from_on_result_clears_boxes_without_self_join(self):
        # stop() from the hook runs on the worker thread: the self-join
        # must be skipped (pre-fix: "cannot join current thread" was
        # recorded in last_error and the box-clearing never ran) and the
        # clear annotate still happens.
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", inference=inf, overlay=ov)

        def stop_and_drop(result):
            p.stop()
            return None  # dropped: nothing drawn after the clear

        p.on_result = stop_and_drop
        p.start()
        inf.push(1, _result(1, [_obj()]))
        assert _wait_until(lambda: not p.status().running), "worker did not stop"
        assert p.status().last_error == ""
        assert ("annotate", "third", [], None, None) in ov.calls
        assert list(p.results()) == []  # dropped result; _STOP terminates


class TestProcessing:
    def test_min_score_and_labels_filter_only_the_drawing(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", min_score=0.5, labels=["person"],
                           inference=inf, overlay=ov)
        p.start()
        try:
            r1 = _result(1, [_obj("person", 0.9), _obj("car", 0.8), _obj("person", 0.3)])
            r2 = _result(2, [_obj("car", 0.9)])  # fully filtered
            inf.push(1, r1)
            inf.push(2, r2)
            gen = p.results()
            assert next(gen)[1] is r1  # queue sees the UNFILTERED results
            assert next(gen)[1] is r2
            assert _wait_until(lambda: len(
                [c for c in ov.calls if c[0] == "annotate"]) == 2)
            draw_calls = [c for c in ov.calls if c[0] == "annotate"]
            kept = draw_calls[0][2]
            assert [o.label for o in kept] == ["person"]
            assert kept[0].score == 0.9
            # fully-filtered still publishes an empty list -> clears stale boxes
            assert draw_calls[1][2] == []
        finally:
            p.stop()

    def test_on_result_replaces_or_drops(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        seen = []
        replacement = _result(99, [_obj()])

        def cb(result):
            seen.append(result.frame_sequence)
            return replacement if result.frame_sequence == 1 else None

        p = StreamPipeline("third", "m", on_result=cb, inference=inf, overlay=ov)
        p.start()
        try:
            inf.push(1, _result(1))
            inf.push(2, _result(2))
            gen = p.results()
            assert next(gen)[1] is replacement  # replaced result flows onward
            assert _wait_until(lambda: p.status().results_seen == 2)
            assert seen == [1, 2]
            assert _wait_until(lambda: p.status().results_annotated == 1)
        finally:
            p.stop()

    def test_annotate_error_counted_and_non_fatal(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", inference=inf, overlay=ov)
        p.start()
        try:
            ov.fail_next = 1
            inf.push(1, _result(1, [_obj()]))
            inf.push(2, _result(2, [_obj()]))
            gen = p.results()
            assert next(gen)[0] == 1  # both results still reach the app
            assert next(gen)[0] == 2
            assert _wait_until(lambda: p.status().annotate_errors == 1)
            st = p.status()
            assert st.results_seen == 2
            assert st.results_annotated == 1
            assert "overlay down" in st.last_error
        finally:
            p.stop()

    def test_stream_error_recorded_and_results_end(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", inference=inf, overlay=ov)
        p.start()
        inf.push(1, _result(1))
        inf.fail()
        gen = p.results()
        assert next(gen)[0] == 1
        with pytest.raises(StopIteration):
            next(gen)  # the worker's failure ends the app-side generator too
        assert _wait_until(lambda: not p.status().running)
        assert "stream broken" in p.status().last_error

    def test_result_queue_drop_oldest(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", result_queue_size=2,
                           draw=False, inference=inf, overlay=ov)
        p.start()
        try:
            for seq in range(1, 5):
                inf.push(seq, _result(seq))
            assert _wait_until(lambda: p.status().result_queue_drops == 2)
            gen = p.results()
            assert [next(gen)[0], next(gen)[0]] == [3, 4]  # newest kept
        finally:
            p.stop()

    def test_status_carries_subscribe_observability(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", inference=inf, overlay=ov)
        p.start()
        try:
            inf.iterator.dropped = 5
            inf.iterator.last_skew_us = 1234
            inf.iterator.avg_skew_us = 2000.0
            inf.iterator.last_latency_ms = 3.5
            inf.iterator.avg_latency_ms = 4.0
            st = p.status()
            assert isinstance(st, StreamPipelineStatus)
            assert st.running and st.results_seen == 0
            assert st.subscribe_dropped == 5
            assert st.last_skew_us == 1234 and st.avg_skew_us == 2000.0
            assert st.last_latency_ms == 3.5 and st.avg_latency_ms == 4.0
        finally:
            p.stop()


class TestSessionTagForwarding:
    """P2-13: result annotations carry the pipeline's session_id; operator
    writes (static zones at start, box/zone clears at stop) stay untagged
    so they behave unconditionally regardless of any session's marks."""

    def test_result_annotations_carry_session_id(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", session_id="s1",
                           inference=inf, overlay=ov)
        p.start()
        try:
            inf.push(1, _result(1, [_obj()]))
            _wait_until(lambda: any(c[0] == "annotate_result" for c in ov.calls))
            i = next(i for i, c in enumerate(ov.calls)
                     if c[0] == "annotate_result")
            assert ov.session_ids[i] == "s1"
        finally:
            p.stop()

    def test_filter_branch_annotate_carries_session_id(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", session_id="s1", min_score=0.5,
                           inference=inf, overlay=ov)
        p.start()
        try:
            inf.push(1, _result(1, [_obj("person", 0.9)]))
            _wait_until(lambda: any(
                c[0] == "annotate" and c[2] for c in ov.calls))
            i = next(i for i, c in enumerate(ov.calls)
                     if c[0] == "annotate" and c[2])
            assert ov.session_ids[i] == "s1"
        finally:
            p.stop()

    def test_static_zones_and_teardown_clears_stay_untagged(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", session_id="s1", polygons=ZONES,
                           inference=inf, overlay=ov)
        p.start()
        inf.push(1, _result(1, [_obj()]))
        _wait_until(lambda: any(c[0] == "annotate_result" for c in ov.calls))
        p.stop()
        # the zone push at start() and both clears at stop() are operator
        # writes: None/"" (untagged), never the session's tag
        tagged = [sid for sid in ov.session_ids if sid]
        assert tagged == ["s1"]  # exactly one tagged write: the result

    def test_pipeline_without_session_tags_nothing(self):
        inf, ov = _FakeInferenceClient(), _FakeOverlayClient()
        p = StreamPipeline("third", "m", inference=inf, overlay=ov)
        p.start()
        try:
            inf.push(1, _result(1, [_obj()]))
            _wait_until(lambda: any(c[0] == "annotate_result" for c in ov.calls))
            assert all(not sid for sid in ov.session_ids)
        finally:
            p.stop()
