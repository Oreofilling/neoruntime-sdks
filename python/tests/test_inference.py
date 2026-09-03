"""
Tests for InferenceClient
"""

import asyncio
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
        # (seen live on device 93.72 with SDK 0.7.2).
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
