"""
Tests for InferenceClient
"""

import asyncio
import os
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from neoruntime_ipc_sdk import BoundingBox, DetectedObject, InferenceClient, InferenceResult


class TestBoundingBox:
    def test_to_xyxy(self):
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        xyxy = bbox.to_xyxy()
        assert abs(xyxy[0] - 0.1) < 1e-6
        assert abs(xyxy[1] - 0.2) < 1e-6
        assert abs(xyxy[2] - 0.4) < 1e-6
        assert abs(xyxy[3] - 0.6) < 1e-6
    
    def test_to_xywh(self):
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        xywh = bbox.to_xywh()
        assert xywh == (0.1, 0.2, 0.3, 0.4)


class TestDetectedObject:
    def test_creation(self):
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        obj = DetectedObject(
            label="person",
            score=0.95,
            bbox=bbox,
            class_id=1,
            track_id=100
        )
        assert obj.label == "person"
        assert obj.score == 0.95
        assert obj.bbox == bbox
        assert obj.class_id == 1
        assert obj.track_id == 100


class TestInferenceResult:
    def test_has_person_true(self):
        bbox = BoundingBox(x=0, y=0, width=1, height=1)
        objects = [
            DetectedObject(label="person", score=0.9, bbox=bbox),
            DetectedObject(label="car", score=0.8, bbox=bbox),
        ]
        result = InferenceResult(frame_sequence=1, timestamp_ns=1000, objects=objects)
        assert result.has_person() is True
    
    def test_has_person_false(self):
        bbox = BoundingBox(x=0, y=0, width=1, height=1)
        objects = [
            DetectedObject(label="car", score=0.8, bbox=bbox),
        ]
        result = InferenceResult(frame_sequence=1, timestamp_ns=1000, objects=objects)
        assert result.has_person() is False
    
    def test_count_by_label(self):
        bbox = BoundingBox(x=0, y=0, width=1, height=1)
        objects = [
            DetectedObject(label="person", score=0.9, bbox=bbox),
            DetectedObject(label="person", score=0.8, bbox=bbox),
            DetectedObject(label="car", score=0.7, bbox=bbox),
        ]
        result = InferenceResult(frame_sequence=1, timestamp_ns=1000, objects=objects)
        assert result.count_by_label("person") == 2
        assert result.count_by_label("car") == 1
        assert result.count_by_label("dog") == 0
    
    def test_get_objects_by_label(self):
        bbox = BoundingBox(x=0, y=0, width=1, height=1)
        obj1 = DetectedObject(label="person", score=0.9, bbox=bbox)
        obj2 = DetectedObject(label="person", score=0.8, bbox=bbox)
        obj3 = DetectedObject(label="car", score=0.7, bbox=bbox)
        objects = [obj1, obj2, obj3]
        
        result = InferenceResult(frame_sequence=1, timestamp_ns=1000, objects=objects)
        persons = result.get_objects_by_label("person")
        
        assert len(persons) == 2
        assert persons[0].score == 0.9
        assert persons[1].score == 0.8


class TestInferenceClient:
    def test_default_endpoint(self):
        client = InferenceClient()
        assert "ai-runtime.sock" in client.endpoint
    
    def test_custom_endpoint(self):
        client = InferenceClient(endpoint="unix:///custom/path.sock")
        assert client.endpoint == "unix:///custom/path.sock"
    
    def test_context_manager(self):
        with InferenceClient() as client:
            assert client.channel is not None
    
    @patch('neoruntime_ipc_sdk.inference.inference_pb2_grpc.InferenceServiceStub')
    @patch('neoruntime_ipc_sdk.inference.grpc.aio.insecure_channel')
    def test_connect(self, mock_channel, mock_stub):
        class FakeAioChannel:
            async def close(self):
                return None

        mock_channel.return_value = FakeAioChannel()
        client = InferenceClient()
        client.connect()

        assert client.channel is not None
        mock_channel.assert_called_once()
        mock_stub.assert_called_once_with(client.channel)
        client.close()
    
    def test_numpy_to_tensor(self):
        from neoruntime_ipc_sdk.proto import inference_pb2
        client = InferenceClient()
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        
        tensor = client._numpy_to_tensor(arr, "test")
        
        assert list(tensor.shape) == [100, 100, 3]
        assert tensor.dtype == inference_pb2.UINT8

    def test_parse_infer_response(self):
        # Regression: the codec function briefly kept a stray `self` after the
        # 0.7.0 refactor, so every infer()/infer_batch() call failed with
        # "_parse_infer_response() missing 1 required positional argument".
        from neoruntime_ipc_sdk.proto import inference_pb2

        client = InferenceClient()
        resp = inference_pb2.InferResponse()
        resp.status.success = True
        resp.infer_time_us = 1234

        result = client._parse_infer_response(resp)

        assert isinstance(result, InferenceResult)
        assert result.infer_time_us == 1234

    def test_parse_infer_response_outputs(self):
        # Regression: _tensor_to_numpy kept a stray `self` after the 0.7.0
        # refactor, so infer() on models returning raw output tensors failed
        # (seen live on a deployed device with SDK 0.7.2).
        from neoruntime_ipc_sdk.proto import inference_pb2

        client = InferenceClient()
        resp = inference_pb2.InferResponse()
        resp.status.success = True
        tensor = resp.outputs.add()
        tensor.shape.extend([2, 2])
        tensor.dtype = inference_pb2.FLOAT32
        tensor.data = np.arange(4, dtype=np.float32).tobytes()

        result = client._parse_infer_response(resp)

        assert result.raw_outputs is not None
        assert len(result.raw_outputs) == 1
        assert result.raw_outputs[0].shape == (2, 2)
        assert result.raw_outputs[0][1, 1] == 3.0

    def test_parse_infer_response_post_result(self):
        # Regression: _parse_post_result kept a stray `self` after the 0.7.0
        # refactor, so infer() on detection models failed.
        from neoruntime_ipc_sdk.proto import inference_pb2

        client = InferenceClient()
        resp = inference_pb2.InferResponse()
        resp.status.success = True
        det = resp.post_result.detections.add()
        det.label = "person"
        det.confidence = 0.9
        det.bbox.x = 0.1
        det.bbox.y = 0.2
        det.bbox.w = 0.3
        det.bbox.h = 0.4

        result = client._parse_infer_response(resp)

        assert len(result.objects) == 1
        assert result.objects[0].label == "person"
        assert result.objects[0].score == pytest.approx(0.9)
    
    def test_dtype_conversion(self):
        client = InferenceClient()
        
        from neoruntime_ipc_sdk.proto import inference_pb2
        
        assert client._dtype_str_to_enum("uint8") == inference_pb2.UINT8
        assert client._dtype_str_to_enum("float32") == inference_pb2.FLOAT32
        assert client._dtype_str_to_enum("int32") == inference_pb2.INT32
        assert client._dtype_str_to_enum("unknown") == inference_pb2.FLOAT32

    def test_update_postprocess_config_success(self):
        # The runtime-tuning surface for detection postprocess (NMS
        # thresholds): request must carry model_id + verbatim config_json.
        from neoruntime_ipc_sdk.proto import inference_pb2

        client, thread = _client_with_fake_loop()
        captured = {}

        class _FakeUpdateStub:
            async def UpdatePostprocessConfig(self, request):
                captured["request"] = request
                resp = inference_pb2.UpdatePostprocessConfigResponse()
                resp.status.success = True
                return resp

        client.stub = _FakeUpdateStub()

        cfg = '{"detection_threshold": 0.38, "iou_threshold": 0.45, "max_boxes": 80}'
        assert client.update_postprocess_config("yolov8n", cfg) is True
        _stop_fake_loop(client, thread)

        assert captured["request"].model_id == "yolov8n"
        assert captured["request"].config_json == cfg

    def test_update_postprocess_config_failure_raises(self):
        # HAL rejects unknown keys with -2801; the SDK must surface the
        # server's message instead of returning False.
        from neoruntime_ipc_sdk.proto import inference_pb2

        client, thread = _client_with_fake_loop()

        class _FakeUpdateStub:
            async def UpdatePostprocessConfig(self, request):
                resp = inference_pb2.UpdatePostprocessConfigResponse()
                resp.status.success = False
                resp.status.message = "apply_config_json failed: -2801"
                return resp

        client.stub = _FakeUpdateStub()

        with pytest.raises(RuntimeError, match="-2801"):
            client.update_postprocess_config("yolov8n", '{"bogus_key": 1}')
        _stop_fake_loop(client, thread)

    def test_update_postprocess_config_clip_prompts(self):
        # CLIP prompt updates ride the same RPC; the config_json is opaque
        # to the SDK (schema lives server-side).
        from neoruntime_ipc_sdk.proto import inference_pb2

        client, thread = _client_with_fake_loop()
        captured = {}

        class _FakeUpdateStub:
            async def UpdatePostprocessConfig(self, request):
                captured["request"] = request
                resp = inference_pb2.UpdatePostprocessConfigResponse()
                resp.status.success = True
                return resp

        client.stub = _FakeUpdateStub()

        cfg = '{"prompts": ["a person", "a car"], "score_threshold": 0.3}'
        assert client.update_postprocess_config("clip_vit_b_32", cfg) is True
        _stop_fake_loop(client, thread)

        assert captured["request"].config_json == cfg

    def test_subscribe_close_cancels_background_stream(self):
        client, thread = _client_with_fake_loop()
        response = _FakeStreamInferResponse()
        stub = _FakeStreamingStub(response)
        client.stub = stub

        gen = client.subscribe("main", "person_v1")
        frame_sequence, result = next(gen)
        assert frame_sequence == response.frame_sequence
        assert result.frame_sequence == response.frame_sequence

        gen.close()
        assert _wait_until(lambda: stub.last_call.cancelled)
        _stop_fake_loop(client, thread)

    def test_subscribe_skips_failed_frames_and_warns(self, caplog):
        import logging
        client, thread = _client_with_fake_loop()
        responses = [
            _failed_response(frame_sequence=1, message="Inference failed: -2814"),
            _failed_response(frame_sequence=2, message="Inference failed: -2814"),
            _ok_response(frame_sequence=42),
        ]
        stub = _FakeMultiStreamingStub(responses)
        client.stub = stub

        with caplog.at_level(logging.WARNING, logger="neoruntime_ipc_sdk.inference"):
            gen = client.subscribe("main", "person_v1")
            frame_sequence, result = next(gen)
            gen.close()

        assert frame_sequence == 42
        assert result.frame_sequence == 42
        assert "-2814" in caplog.text
        assert _wait_until(lambda: stub.last_call.cancelled)
        _stop_fake_loop(client, thread)

    def test_subscribe_raises_after_consecutive_failures(self):
        client, thread = _client_with_fake_loop()
        responses = [
            _failed_response(frame_sequence=i, message="Inference failed: -2814")
            for i in range(3)
        ]
        stub = _FakeMultiStreamingStub(responses)
        client.stub = stub

        gen = client.subscribe("main", "person_v1", max_consecutive_failures=3)
        with pytest.raises(RuntimeError, match="3 consecutive times"):
            next(gen)

        assert _wait_until(lambda: stub.last_call.cancelled)
        _stop_fake_loop(client, thread)

    def test_subscribe_failure_counter_resets_on_success(self):
        client, thread = _client_with_fake_loop()
        responses = [
            _failed_response(frame_sequence=1),
            _failed_response(frame_sequence=2),
            _ok_response(frame_sequence=10),
            _failed_response(frame_sequence=3),
            _failed_response(frame_sequence=4),
            _ok_response(frame_sequence=20),
        ]
        stub = _FakeMultiStreamingStub(responses)
        client.stub = stub

        gen = client.subscribe("main", "person_v1", max_consecutive_failures=3)
        frames = [next(gen)[0], next(gen)[0]]
        gen.close()

        assert frames == [10, 20]
        assert _wait_until(lambda: stub.last_call.cancelled)
        _stop_fake_loop(client, thread)

    def test_subscribe_failure_limit_disabled(self):
        client, thread = _client_with_fake_loop()
        responses = [_failed_response(frame_sequence=i) for i in range(5)]
        responses.append(_ok_response(frame_sequence=99))
        stub = _FakeMultiStreamingStub(responses)
        client.stub = stub

        gen = client.subscribe("main", "person_v1", max_consecutive_failures=0)
        assert next(gen)[0] == 99
        gen.close()
        assert _wait_until(lambda: stub.last_call.cancelled)
        _stop_fake_loop(client, thread)

    def test_subscribe_bounded_queue_drops_oldest(self, caplog):
        import logging

        client, thread = _client_with_fake_loop()
        responses = [_ok_response(frame_sequence=i) for i in (1, 2, 3)]
        stub = _FakeMultiStreamingStub(responses)
        client.stub = stub

        consumed = []

        def slow_consumer():
            time.sleep(0.3)  # let the pump fill and overflow the queue first
            for _ in range(2):
                consumed.append(next(gen)[0])
            gen.close()

        with caplog.at_level(logging.WARNING, logger="neoruntime_ipc_sdk.inference"):
            gen = client.subscribe("main", "person_v1", queue_size=2)
            assert gen.dropped == 0
            consumer = threading.Thread(target=slow_consumer)
            consumer.start()
            consumer.join(timeout=2)

        assert consumed == [2, 3]  # the oldest queued result (frame 1) was shed
        assert gen.dropped == 1
        assert "dropped 1 queued results" in caplog.text
        assert _wait_until(lambda: stub.last_call.cancelled)
        _stop_fake_loop(client, thread)

    def test_subscribe_unbounded_queue_keeps_all(self):
        client, thread = _client_with_fake_loop()
        responses = [_ok_response(frame_sequence=i) for i in (1, 2, 3)]
        stub = _FakeMultiStreamingStub(responses)
        client.stub = stub

        gen = client.subscribe("main", "person_v1", queue_size=0)
        frames = [next(gen)[0]]  # starts the pump
        assert _wait_until(lambda: stub.last_call.index >= 3)
        frames += [next(gen)[0] for _ in range(2)]
        gen.close()

        assert frames == [1, 2, 3]
        assert gen.dropped == 0
        _stop_fake_loop(client, thread)

    def test_subscribe_stream_error_survives_backpressure(self):
        # Regression (review P1): with a bounded queue and a consumer behind
        # the stream, the terminal error used to be evicted by the SENTINEL
        # offered after it — the consumer saw a clean StopIteration instead
        # of the real failure.
        client, thread = _client_with_fake_loop()
        responses = [_ok_response(frame_sequence=i) for i in (1, 2, 3, 4)]
        stub = _FailingStreamingStub(responses, RuntimeError("stream broke"))
        client.stub = stub
        outcome = {}

        gen = client.subscribe("main", "person_v1", queue_size=2)

        def slow_consumer():
            time.sleep(0.3)  # queue is full and shedding by now
            try:
                for _ in gen:
                    pass
                outcome["end"] = "stopped"
            except RuntimeError as exc:
                outcome["end"] = exc

        consumer = threading.Thread(target=slow_consumer)
        consumer.start()
        consumer.join(timeout=2)

        assert isinstance(outcome.get("end"), RuntimeError)
        assert "stream broke" in str(outcome["end"])
        _stop_fake_loop(client, thread)

    def test_genai_generate_close_cancels_background_stream(self):
        client, thread = _client_with_fake_loop()
        token = _FakeGenaiToken("hello")
        stub = _FakeStreamingStub(token, genai=True)
        client.stub = stub

        gen = client.genai_generate("session-1", ["{}"])
        assert next(gen) == "hello"

        gen.close()
        assert _wait_until(lambda: stub.last_call.cancelled)
        _stop_fake_loop(client, thread)


def _client_with_fake_loop():
    client = InferenceClient()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    client._loop = loop
    return client, thread


def _stop_fake_loop(client, thread):
    client._loop.call_soon_threadsafe(client._loop.stop)
    thread.join(timeout=2)
    client._loop = None


def _wait_until(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _FakeStatus:
    success = True
    message = ""


class _FakeStreamInferResponse:
    status = _FakeStatus()
    frame_sequence = 42
    timestamp_ns = 123
    outputs = []

    def HasField(self, name):
        return False


class _FakeGenaiToken:
    def __init__(self, text):
        self.token = text

    def HasField(self, name):
        return name == "token"


class _FakeStreamingCall:
    def __init__(self, first_item):
        self.first_item = first_item
        self.sent_first = False
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.sent_first:
            self.sent_first = True
            return self.first_item
        await asyncio.sleep(60)
        raise StopAsyncIteration


class _FakeStreamingStub:
    def __init__(self, first_item, genai=False):
        self.first_item = first_item
        self.genai = genai
        self.last_call = None

    def StreamInfer(self, request):
        self.last_call = _FakeStreamingCall(self.first_item)
        return self.last_call

    def GenaiGenerate(self, request):
        self.last_call = _FakeStreamingCall(self.first_item)
        return self.last_call


def _ok_response(frame_sequence=42, message=""):
    status = _FakeStatus()
    status.success = True
    status.message = message
    response = _FakeStreamInferResponse()
    response.status = status
    response.frame_sequence = frame_sequence
    return response


def _failed_response(frame_sequence=1, message="Inference failed: -2814"):
    status = _FakeStatus()
    status.success = False
    status.message = message
    response = _FakeStreamInferResponse()
    response.status = status
    response.frame_sequence = frame_sequence
    return response


class _FakeMultiItemCall:
    def __init__(self, items):
        self.items = list(items)
        self.index = 0
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            return item
        await asyncio.sleep(60)
        raise StopAsyncIteration


class _FakeMultiStreamingStub:
    def __init__(self, items):
        self.items = items
        self.last_call = None

    def StreamInfer(self, request):
        self.last_call = _FakeMultiItemCall(self.items)
        return self.last_call


class _FailingAfterItemsCall:
    """Yields its items, then raises — a stream that dies mid-flight."""

    def __init__(self, items, error):
        self.items = list(items)
        self.index = 0
        self.error = error
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            return item
        raise self.error


class _FailingStreamingStub:
    def __init__(self, items, error):
        self.items = items
        self.error = error
        self.last_call = None

    def StreamInfer(self, request):
        self.last_call = _FailingAfterItemsCall(self.items, self.error)
        return self.last_call


class TestSubscribeLatency:
    def test_latency_stats_recorded_from_timestamps(self):
        import time as _time

        client, thread = _client_with_fake_loop()
        response = _ok_response(frame_sequence=1)
        response.timestamp_ns = _time.time_ns() - 50_000_000  # 50 ms ago
        stub = _FakeMultiStreamingStub([response])
        client.stub = stub

        gen = client.subscribe("main", "person_v1")
        assert next(gen)[0] == 1
        gen.close()

        assert 30.0 < gen.last_latency_ms < 500.0  # ~50 ms, generous bounds
        assert gen.avg_latency_ms > 0.0
        _stop_fake_loop(client, thread)

    def test_latency_ignored_for_bogus_timestamps(self):
        client, thread = _client_with_fake_loop()
        response = _ok_response(frame_sequence=1)
        response.timestamp_ns = 123  # _FakeStreamInferResponse's ancient stamp
        stub = _FakeMultiStreamingStub([response])
        client.stub = stub

        gen = client.subscribe("main", "person_v1")
        assert next(gen)[0] == 1
        gen.close()

        assert gen.last_latency_ms == 0.0 and gen.avg_latency_ms == 0.0
        _stop_fake_loop(client, thread)


class TestSubscribeSkewAndCancel:
    """P1-8: server-computed skew observability + cross-thread cancel()."""

    def test_skew_ema_recorded_from_response(self):
        # Skew arrives precomputed on the wire; the SDK only tracks an EMA.
        client, thread = _client_with_fake_loop()
        r1 = _ok_response(frame_sequence=1)
        r1.skew_us = 30_000
        r2 = _ok_response(frame_sequence=2)
        r2.skew_us = 60_000
        stub = _FakeMultiStreamingStub([r1, r2])
        client.stub = stub

        gen = client.subscribe("main", "person_v1")
        assert next(gen)[0] == 1
        assert next(gen)[0] == 2
        gen.close()

        assert gen.last_skew_us == 60_000
        # EMA (alpha 0.1): 30000 + 0.1 * (60000 - 30000)
        assert gen.avg_skew_us == pytest.approx(33_000.0)
        _stop_fake_loop(client, thread)

    def test_skew_stays_zero_when_not_reported(self):
        # Failed frames carry no skew, and an older server sends no skew_us
        # field at all (_FakeStreamInferResponse has no such attribute).
        client, thread = _client_with_fake_loop()
        responses = [
            _failed_response(frame_sequence=1),
            _ok_response(frame_sequence=2),  # no skew_us set
        ]
        stub = _FakeMultiStreamingStub(responses)
        client.stub = stub

        gen = client.subscribe("main", "person_v1")
        assert next(gen)[0] == 2
        gen.close()

        assert gen.last_skew_us == 0
        assert gen.avg_skew_us == 0.0
        _stop_fake_loop(client, thread)

    def test_result_carries_skew_us(self):
        client, thread = _client_with_fake_loop()
        response = _ok_response(frame_sequence=7)
        response.skew_us = 41_200
        stub = _FakeMultiStreamingStub([response])
        client.stub = stub

        gen = client.subscribe("main", "person_v1")
        _, result = next(gen)
        gen.close()

        assert isinstance(result, InferenceResult)
        assert result.skew_us == 41_200
        _stop_fake_loop(client, thread)

    def test_cancel_from_another_thread_unblocks_consumer(self):
        # The fake stream sleeps 60s after its last item, so the second
        # next() blocks in q.get(); cancel() from this (other) thread must
        # wake it with a clean StopIteration and stop the pump.
        client, thread = _client_with_fake_loop()
        stub = _FakeMultiStreamingStub([_ok_response(frame_sequence=1)])
        client.stub = stub

        gen = client.subscribe("main", "person_v1")
        outcome = {}

        def blocked_consumer():
            try:
                next(gen)  # starts the pump, consumes item 1
                next(gen)  # blocks — nothing left in the stream
                outcome["end"] = "item"
            except StopIteration:
                outcome["end"] = "stopped"

        consumer = threading.Thread(target=blocked_consumer)
        consumer.start()
        time.sleep(0.2)  # let the consumer reach the blocking q.get()
        gen.cancel()
        consumer.join(timeout=2)

        assert not consumer.is_alive()
        assert outcome.get("end") == "stopped"
        assert _wait_until(lambda: stub.last_call.cancelled)
        _stop_fake_loop(client, thread)

    def test_cancel_before_first_next_is_noop(self):
        # No pump exists yet; cancel() must neither raise nor break the
        # subscription for a later consumer.
        client, thread = _client_with_fake_loop()
        stub = _FakeMultiStreamingStub([_ok_response(frame_sequence=1)])
        client.stub = stub

        gen = client.subscribe("main", "person_v1")
        gen.cancel()

        assert next(gen)[0] == 1
        gen.close()
        _stop_fake_loop(client, thread)

    def test_get_stats_includes_skew_aggregates(self):
        # The regenerated proto carries the new InferenceStats fields; the
        # stats dict must surface them (0 via getattr on older servers).
        from neoruntime_ipc_sdk.proto import inference_pb2

        client, thread = _client_with_fake_loop()
        resp = inference_pb2.SystemStats()
        s = resp.model_stats.add()
        s.model_id = "person_v1"
        s.avg_skew_us = 33_000
        s.max_skew_us = 61_000
        s.skew_samples = 57

        class _FakeStatsStub:
            async def GetStats(self, request):
                return resp

        client.stub = _FakeStatsStub()

        stats = client.get_stats()
        (m,) = stats["model_stats"]
        assert m["avg_skew_us"] == 33_000
        assert m["max_skew_us"] == 61_000
        assert m["skew_samples"] == 57
        _stop_fake_loop(client, thread)


def _frame_handle(width=3840, height=2160, fmt="NV12", closed=False):
    """A FrameHandle backed by real (devnull) fds so close() is genuine.

    Geometry mirrors a 4K NV12 frame: 2 planes, Y stride == width, UV
    plane half the Y size. ``closed=True`` pre-releases the handle.
    """
    from neoruntime_ipc_sdk.frame import FrameHandle

    fds = [os.open(os.devnull, os.O_RDONLY) for _ in range(2)]
    handle = FrameHandle(
        fds=fds,
        strides=[width, width],
        plane_sizes=[width * height, width * height // 2],
        frame_id=7,
        width=width,
        height=height,
        format=fmt,
    )
    if closed:
        handle.close()
    return handle


class _FakeInferStub:
    """Records Infer requests; answers with a canned InferResponse."""

    def __init__(self, success=True, message="", infer_time_us=15000):
        from neoruntime_ipc_sdk.proto import inference_pb2

        self.requests = []
        resp = inference_pb2.InferResponse()
        resp.status.success = success
        resp.status.message = message
        resp.infer_time_us = infer_time_us
        self.response = resp

    async def Infer(self, request, timeout=None):
        self.requests.append(request)
        return self.response


class TestInferWithFrame:
    """infer()/infer_async() with keep-fd Frame/FrameHandle inputs.

    The frame path imports the dma-bufs via DspClient and references
    them by Tensor.buffer_id (no pixel copy crosses the wire), and the
    imported buffer must be released exactly once whether the RPC
    succeeds or fails.
    """

    def _client(self, stub):
        client, thread = _client_with_fake_loop()
        client.stub = stub
        dsp = Mock()
        dsp.import_frame.return_value = 777
        client._dsp = dsp
        return client, thread, dsp

    def test_frame_handle_rides_buffer_id(self):
        from neoruntime_ipc_sdk.proto import inference_pb2

        client, thread, dsp = self._client(_FakeInferStub())
        handle = _frame_handle()
        try:
            result = client.infer(handle, "yolov8n")

            assert isinstance(result, InferenceResult)
            assert result.infer_time_us == 15000
            dsp.import_frame.assert_called_once_with(handle)

            (request,) = client.stub.requests
            tensor = request.inputs[0]
            assert tensor.buffer_id == 777
            assert tensor.dtype == inference_pb2.UINT8
            assert list(tensor.shape) == [2160 * 3 // 2, 3840]
            assert tensor.data == b""  # no pixels on the wire

            dsp.release_buffer.assert_called_once_with(777)
        finally:
            handle.close()
            _stop_fake_loop(client, thread)

    def test_frame_wraps_handle(self):
        # A keep-fd Frame delegates to its handle; same wire contract.
        from neoruntime_ipc_sdk.frame import Frame

        client, thread, dsp = self._client(_FakeInferStub())
        handle = _frame_handle()
        try:
            frame = Frame(sequence=3, timestamp_ns=1, width=3840, height=2160,
                          format="NV12", image=None, handle=handle)
            result = client.infer(frame, "yolov8n")

            assert isinstance(result, InferenceResult)
            dsp.import_frame.assert_called_once_with(handle)
            dsp.release_buffer.assert_called_once_with(777)
        finally:
            handle.close()
            _stop_fake_loop(client, thread)

    def test_frame_without_handle_raises(self):
        from neoruntime_ipc_sdk.frame import Frame

        client, thread, dsp = self._client(_FakeInferStub())
        try:
            frame = Frame(sequence=3, timestamp_ns=1, width=3840, height=2160,
                          format="NV12", image=None)  # handle=None
            with pytest.raises(ValueError, match="keep_fd=True"):
                client.infer(frame, "yolov8n")

            assert client.stub.requests == []  # no RPC went out
            dsp.import_frame.assert_not_called()
            dsp.release_buffer.assert_not_called()
        finally:
            _stop_fake_loop(client, thread)

    def test_non_nv12_handle_rejected(self):
        # Runtime repack is NV12-only; other formats must fail before
        # any import or RPC.
        client, thread, dsp = self._client(_FakeInferStub())
        handle = _frame_handle(fmt="RGB")
        try:
            with pytest.raises(ValueError, match="NV12"):
                client.infer(handle, "yolov8n")

            assert client.stub.requests == []
            dsp.import_frame.assert_not_called()
            dsp.release_buffer.assert_not_called()
        finally:
            handle.close()
            _stop_fake_loop(client, thread)

    def test_closed_handle_rejected(self):
        client, thread, dsp = self._client(_FakeInferStub())
        try:
            handle = _frame_handle(closed=True)  # fds already gone
            with pytest.raises(ValueError, match="closed"):
                client.infer(handle, "yolov8n")

            assert client.stub.requests == []
            dsp.import_frame.assert_not_called()
            dsp.release_buffer.assert_not_called()
        finally:
            _stop_fake_loop(client, thread)

    def test_release_still_fires_when_rpc_fails(self):
        client, thread, dsp = self._client(
            _FakeInferStub(success=False, message="Inference failed: -2811"))
        handle = _frame_handle()
        try:
            with pytest.raises(RuntimeError, match="-2811"):
                client.infer(handle, "yolov8n")

            # The import did happen, so the release must happen too.
            dsp.release_buffer.assert_called_once_with(777)
        finally:
            handle.close()
            _stop_fake_loop(client, thread)

    def test_infer_async_frame_releases_on_completion(self):
        client, thread, dsp = self._client(_FakeInferStub())
        handle = _frame_handle()
        try:
            fut = client.infer_async(handle, "yolov8n")
            result = fut.result(timeout=2)

            assert isinstance(result, InferenceResult)
            (request,) = client.stub.requests
            assert request.inputs[0].buffer_id == 777
            # The coroutine's finally ran before the future resolved.
            dsp.release_buffer.assert_called_once_with(777)
        finally:
            handle.close()
            _stop_fake_loop(client, thread)

    def test_infer_async_frame_releases_on_failure(self):
        client, thread, dsp = self._client(
            _FakeInferStub(success=False, message="Inference failed: -2814"))
        handle = _frame_handle()
        try:
            fut = client.infer_async(handle, "yolov8n")
            with pytest.raises(RuntimeError, match="-2814"):
                fut.result(timeout=2)

            dsp.release_buffer.assert_called_once_with(777)
        finally:
            handle.close()
            _stop_fake_loop(client, thread)

    def test_ndarray_path_never_touches_dsp(self):
        # Regression guard: the bytes path must not lazily create a
        # DspClient nor send buffer_id.
        client, thread = _client_with_fake_loop()
        try:
            client.stub = _FakeInferStub()
            arr = np.zeros((100, 100, 3), dtype=np.uint8)
            client.infer(arr, "yolov8n")

            assert client._dsp is None
            (request,) = client.stub.requests
            assert request.inputs[0].buffer_id == 0
            assert len(request.inputs[0].data) > 0
        finally:
            _stop_fake_loop(client, thread)
