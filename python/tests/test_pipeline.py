"""InferencePipeline composition (pre → infer → post)."""

import time

import numpy as np
import pytest

from neoruntime_ipc_sdk import (
    BoundingBox,
    DetectedObject,
    InferencePipeline,
    InferenceResult,
    ModelInfo,
    Preprocessor,
    YoloV8Postprocessor,
)


def make_v8_head(entries, nc=2, n=200):
    arr = np.zeros((4 + nc, n), np.float32)
    for col, (cx, cy, w, h), score, cls in entries:
        arr[0, col], arr[1, col], arr[2, col], arr[3, col] = cx, cy, w, h
        arr[4 + cls, col] = score
    return arr


HEAD = make_v8_head([(0, (320, 300, 100, 80), 0.9, 0)])


class FakeClient:
    """Stand-in with the surface the pipeline touches."""

    def __init__(self, result=None, info=None):
        self.infer_calls = []
        self.async_calls = []
        self._result = result or InferenceResult(
            frame_sequence=1, timestamp_ns=0, objects=[], raw_outputs=[HEAD],
        )
        self._info = info or ModelInfo(
            model_id="m", model_path="x.hef", version="1",
            inputs=[{"shape": [1, 640, 640, 3], "dtype": 0, "name": "in",
                     "layout": "NHWC"}],
        )

    def infer(self, image, model_id, timeout_ms=5000):
        self.infer_calls.append((model_id, getattr(image, "shape", None), timeout_ms))
        return self._result

    def infer_async(self, image, model_id, timeout_ms=5000):
        self.async_calls.append((model_id, getattr(image, "shape", None)))
        result = self._result

        class _Fut:
            """Already-resolved future mirroring infer_async's contract."""

            def __init__(self, value):
                self._value = value

            def add_done_callback(self, cb):
                cb(self)

            def result(self):
                return self._value

        return _Fut(result)

    def get_model_info(self, model_id):
        return self._info


class TestRun:
    def test_pre_infer_post_composition(self):
        client = FakeClient()
        pre = Preprocessor(size=(640, 640), source_format="RGB")
        post = YoloV8Postprocessor(labels=["person"], score_threshold=0.3)
        pipe = InferencePipeline(client=client, model_id="m",
                                 preprocessor=pre, postprocessor=post)

        out = pipe.run(np.zeros((1080, 1920, 3), np.uint8))

        # preprocessor shrank the image before the RPC
        assert client.infer_calls == [("m", (640, 640, 3), 5000)]
        # client-side decode ran (raw_outputs present) with tunable threshold
        assert [o.label for o in out.objects] == ["person"]
        assert out.objects[0].score == pytest.approx(0.9)
        assert out.result.raw_outputs == [HEAD]
        assert out.tensor.shape == (640, 640, 3)
        assert out.meta.input_size == (640, 640)
        assert out.latency_ms >= 0

    def test_server_decoded_passes_through_without_raw_outputs(self):
        server_obj = DetectedObject(
            label="person", score=0.8, bbox=BoundingBox(1, 2, 3, 4)
        )
        client = FakeClient(result=InferenceResult(
            frame_sequence=1, timestamp_ns=0, objects=[server_obj]))
        pipe = InferencePipeline(client=client, model_id="m",
                                 postprocessor=YoloV8Postprocessor())
        out = pipe.run(np.zeros((8, 8, 3), np.uint8))
        assert out.objects == [server_obj]  # no double decode

    def test_no_model_id_raises(self):
        pipe = InferencePipeline(client=FakeClient())
        with pytest.raises(ValueError, match="model_id"):
            pipe.run(np.zeros((8, 8, 3), np.uint8))

    def test_passthrough_without_preprocessor(self):
        client = FakeClient()
        pipe = InferencePipeline(client=client, model_id="m")
        pipe.run(np.zeros((8, 8, 3), np.uint8))
        assert client.infer_calls[0][1] == (8, 8, 3)  # untouched


class TestRunAsync:
    def test_future_resolves_to_pipeline_result(self):
        client = FakeClient()
        pre = Preprocessor(size=(640, 640), source_format="RGB")
        pipe = InferencePipeline(client=client, model_id="m", preprocessor=pre,
                                 postprocessor=YoloV8Postprocessor(labels=["person"],
                                                                   score_threshold=0.3))
        fut = pipe.run_async(np.zeros((1080, 1920, 3), np.uint8))
        out = fut.result(timeout=5)
        assert client.async_calls == [("m", (640, 640, 3))]
        assert [o.label for o in out.objects] == ["person"]


class TestFromModel:
    def test_wires_derived_preprocessor(self):
        client = FakeClient()
        pipe = InferencePipeline.from_model(
            "m", client=client,
            postprocessor=YoloV8Postprocessor(labels=["person"], score_threshold=0.3),
        )
        assert isinstance(pipe.preprocessor, Preprocessor)
        assert pipe.preprocessor.size == (640, 640)  # derived from the fake spec
        out = pipe.run(np.zeros((1080, 1920, 3), np.uint8))
        assert [o.label for o in out.objects] == ["person"]
