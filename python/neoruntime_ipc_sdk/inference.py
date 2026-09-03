"""
AI Inference Client
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from concurrent.futures import Future
from typing import Any, Iterator

import grpc
import numpy as np

from .config import Config
from ._transport import MAX_GRPC_MESSAGE_LENGTH
from .inference_codec import (  # noqa: F401 — re-exported for API compat
    _numpy_to_tensor,
    _parse_infer_response,
    _parse_post_result,
    _tensor_to_numpy,
)
from .inference_genai import GenAiMixin  # noqa: F401 — mixed into client
from .inference_types import (  # noqa: F401 — re-exported for API compat
    BatchInferItem,
    BoundingBox,
    Classification,
    DepthMap,
    DetectedObject,
    Embedding,
    InferenceResult,
    LandmarkPoint,
    LandmarkSet,
    ModelInfo,
    OcrLine,
    SegmentationMask,
)
from .proto import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)


class InferenceClient(GenAiMixin):
    """
    AI Inference Client

    Usage::

        inf = InferenceClient()

        # Single inference
        result = inf.infer(image, model_id="person_v1")

        # Stream inference
        for frame, res in inf.subscribe(stream="cam0_main", model="person_v1", fps=10):
            print(f"Detected {len(res.objects)} objects")
    """

    def __init__(self, endpoint: str | None = None):
        if endpoint is None:
            endpoint = self._get_default_endpoint()

        self.endpoint = endpoint
        self.channel: grpc.aio.Channel | None = None
        self.stub: inference_pb2_grpc.InferenceServiceStub | None = None
        # Background asyncio loop running grpc.aio. Sync callers bridge via
        # run_coroutine_threadsafe(...).result() — the caller thread blocks on
        # a futex, NOT a sched_yield busy-poll, eliminating the sync-CQ spin.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    def _get_default_endpoint(self) -> str:
        import os

        return os.getenv("AI_RUNTIME_ENDPOINT", "unix:///run/aipc/ai-runtime.sock")

    def connect(self) -> None:
        if self.channel is not None:
            return
        # grpc.aio uses an async completion queue backed by epoll (true blocking,
        # no sched_yield busy-poll), which is the only way to eliminate the
        # sync-CQ spin that saturates a core in tight infer loops. It needs a
        # running event loop, so spin one on a dedicated daemon thread.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="grpc-aio-loop"
        )
        self._loop_thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect_async(), self._loop)
        fut.result(timeout=10)  # channel + stub creation

    async def _connect_async(self) -> None:
        # epoll1 (belt-and-suspenders); the async CQ already uses epoll.
        # max_receive_message_length lifts grpc's 4 MiB default so inference
        # responses can use the server's full 64 MiB limit.
        self.channel = grpc.aio.insecure_channel(
            self.endpoint,
            options=[
                ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_LENGTH),
                ("grpc.poll_strategy", 1),  # 1 = epoll1
            ],
        )
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)

    def _invoke(self, fn, *args, result_timeout: float = 30, **kwargs):
        """Call an async gRPC stub method on the background loop and block the
        caller until it completes (or result_timeout elapses).

        The stub call object is created AND awaited on the loop thread (via the
        native-coroutine wrapper), so grpc.aio's loop affinity is respected.
        The caller thread blocks on fut.result() — a futex, not a spin.
        """
        if self._loop is None:
            raise RuntimeError("InferenceClient not connected")

        async def _wrap():
            return await fn(*args, **kwargs)

        fut = asyncio.run_coroutine_threadsafe(_wrap(), self._loop)
        return fut.result(timeout=result_timeout)

    @property
    def connected(self) -> bool:
        return self.channel is not None

    def close(self) -> None:
        if self.channel:
            asyncio.run_coroutine_threadsafe(self.channel.close(), self._loop).result(timeout=5)
            self.channel = None
            self.stub = None
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread:
                self._loop_thread.join(timeout=5)
            self._loop = None
            self._loop_thread = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _numpy_to_tensor(self, arr: np.ndarray, name: str = "") -> inference_pb2.Tensor:
        dtype_map = {
            np.uint8: inference_pb2.UINT8,
            np.int8: inference_pb2.INT8,
            np.uint16: inference_pb2.UINT16,
            np.int16: inference_pb2.INT16,
            np.float16: inference_pb2.FLOAT16,
            np.float32: inference_pb2.FLOAT32,
            np.int32: inference_pb2.INT32,
            np.uint32: inference_pb2.UINT32,
        }

        dtype = dtype_map.get(arr.dtype.type, inference_pb2.FLOAT32)

        return inference_pb2.Tensor(shape=list(arr.shape), dtype=dtype, data=arr.tobytes())

    def _tensor_to_numpy(self, tensor: inference_pb2.Tensor) -> np.ndarray:
        dtype_map = {
            inference_pb2.UINT8: np.uint8,
            inference_pb2.INT8: np.int8,
            inference_pb2.UINT16: np.uint16,
            inference_pb2.INT16: np.int16,
            inference_pb2.FLOAT16: np.float16,
            inference_pb2.FLOAT32: np.float32,
            inference_pb2.INT32: np.int32,
            inference_pb2.UINT32: np.uint32,
        }

        dtype = dtype_map.get(tensor.dtype, np.float32)
        arr = np.frombuffer(tensor.data, dtype=dtype)

        # Handle empty or invalid shape - keep as 1D array
        if tensor.shape and len(tensor.shape) > 0:
            try:
                return arr.reshape(tensor.shape)
            except ValueError:
                # Shape doesn't match data size, return flat array
                return arr
        return arr

    def _parse_post_result(
        self, post_result: inference_pb2.PostResult
    ) -> tuple[
        list[DetectedObject],
        list[Classification],
        list[LandmarkSet],
        list[SegmentationMask],
        list[OcrLine],
        list[Embedding],
        list[DepthMap],
    ]:
        objects = []
        for det in post_result.detections:
            obj = DetectedObject(
                label=det.label,
                score=det.confidence,
                bbox=BoundingBox(x=det.bbox.x, y=det.bbox.y, width=det.bbox.w, height=det.bbox.h),
                class_id=det.class_id,
            )
            objects.append(obj)

        classifications = []
        for cls in post_result.classifications:
            classifications.append(
                Classification(
                    type=cls.type, class_id=cls.class_id, label=cls.label, confidence=cls.confidence
                )
            )

        landmarks = []
        for lm_set in post_result.landmarks:
            points = [LandmarkPoint(x=p.x, y=p.y, confidence=p.confidence) for p in lm_set.points]
            landmarks.append(LandmarkSet(type=lm_set.type, points=points))

        masks = []
        for m in post_result.masks:
            masks.append(
                SegmentationMask(
                    class_id=m.class_id,
                    label=m.label,
                    confidence=m.confidence,
                    bbox=BoundingBox(x=m.bbox.x, y=m.bbox.y, width=m.bbox.w, height=m.bbox.h),
                    mask_rle=m.mask_rle,
                    mask_width=m.mask_width,
                    mask_height=m.mask_height,
                )
            )

        ocr_lines = []
        for line in post_result.ocr_lines:
            ocr_lines.append(
                OcrLine(
                    text=line.text,
                    confidence=line.confidence,
                    bbox=BoundingBox(
                        x=line.bbox.x, y=line.bbox.y, width=line.bbox.w, height=line.bbox.h
                    ),
                )
            )

        embeddings = []
        for emb in post_result.embeddings:
            embeddings.append(Embedding(dim=emb.dim, data=list(emb.data)))

        depth_maps = []
        for dm in post_result.depth_maps:
            arr = np.frombuffer(dm.depth_data, dtype=np.float32).reshape(dm.height, dm.width)
            depth_maps.append(DepthMap(width=dm.width, height=dm.height, data=arr.copy()))

        return objects, classifications, landmarks, masks, ocr_lines, embeddings, depth_maps

    def _parse_infer_response(self, response: inference_pb2.InferResponse) -> InferenceResult:
        """Parse an InferResponse proto into an InferenceResult dataclass.

        Shared by infer() and infer_batch() to avoid duplication.
        """
        objects: list[DetectedObject] = []
        classifications: list[Classification] = []
        landmarks: list[LandmarkSet] = []
        masks: list[SegmentationMask] = []
        ocr_lines: list[OcrLine] = []
        embeddings: list[Embedding] = []
        depth_maps: list[DepthMap] = []

        try:
            if response.HasField("post_result"):
                objects, classifications, landmarks, masks, ocr_lines, embeddings, depth_maps = (
                    self._parse_post_result(response.post_result)
                )
        except (ValueError, AttributeError):
            pass

        raw_outputs = None
        if response.outputs:
            raw_outputs = [self._tensor_to_numpy(t) for t in response.outputs]

        return InferenceResult(
            frame_sequence=0,
            timestamp_ns=0,
            objects=objects,
            classifications=classifications,
            landmarks=landmarks,
            masks=masks,
            ocr_lines=ocr_lines,
            embeddings=embeddings,
            depth_maps=depth_maps,
            raw_outputs=raw_outputs,
            infer_time_us=response.infer_time_us,
            queue_time_us=response.queue_time_us,
            hw_infer_time_us=getattr(response, "hw_infer_time_us", 0),
            status_message=response.status.message if not response.status.success else "",
        )

    # -- codec (implementation in inference_codec; kept as methods for
    # backward compat with existing callers/tests) --------------------------
    def _numpy_to_tensor(self, arr: np.ndarray, name: str = "") -> inference_pb2.Tensor:
        return _numpy_to_tensor(arr, name)

    def _tensor_to_numpy(self, tensor: inference_pb2.Tensor) -> np.ndarray:
        return _tensor_to_numpy(tensor)

    def _parse_post_result(self, post_result: inference_pb2.PostResult) -> tuple:
        return _parse_post_result(post_result)

    def _parse_infer_response(self, response: inference_pb2.InferResponse) -> InferenceResult:
        return _parse_infer_response(response)

    def infer(
        self,
        image: np.ndarray,
        model_id: str,
        timeout_ms: int = 5000,
        priority: int = 4,
        session_id: str = "",
    ) -> InferenceResult:
        if self.stub is None:
            self.connect()

        tensor = self._numpy_to_tensor(image, "input")

        request = inference_pb2.InferRequest(
            model_id=model_id,
            inputs=[tensor],
            timeout_ms=timeout_ms,
            priority=priority,
            session_id=session_id,
        )

        fut = asyncio.run_coroutine_threadsafe(self._infer_async(request, timeout_ms), self._loop)
        # +5s slack covers NPU cold start / HEF context init; the gRPC deadline
        # itself is timeout_ms/1000.
        response = fut.result(timeout=timeout_ms / 1000 + 5)

        if not response.status.success:
            raise RuntimeError(f"Inference failed: {response.status.message}")

        return self._parse_infer_response(response)

    async def _infer_async(self, request, timeout_ms):
        return await self.stub.Infer(request, timeout=timeout_ms / 1000)

    async def _infer_full_async(self, request, timeout_ms):
        """Await Infer, check status, return a PARSED InferenceResult.

        Used by infer_async() so the returned future resolves directly to an
        InferenceResult, letting callers build depth-N pipelines without a
        separate parse step.
        """
        response = await self.stub.Infer(request, timeout=timeout_ms / 1000)
        if not response.status.success:
            raise RuntimeError(f"Inference failed: {response.status.message}")
        return self._parse_infer_response(response)

    def infer_async(
        self,
        image: np.ndarray,
        model_id: str,
        timeout_ms: int = 5000,
        priority: int = 4,
        session_id: str = "",
    ) -> Future:
        """Non-blocking infer: returns a concurrent.futures.Future that resolves
        to a parsed InferenceResult. The caller MUST call fut.result(timeout=...)
        to obtain the result (or propagate the error).

        Enables depth-N pipelines: submit frame N+1 while still awaiting frame N
        so the NPU stays busy across the host-side gap between jobs. Schedules
        onto the same background asyncio loop infer() already uses. The existing
        blocking infer() is unchanged.
        """
        if self.stub is None:
            self.connect()

        tensor = self._numpy_to_tensor(image, "input")
        request = inference_pb2.InferRequest(
            model_id=model_id,
            inputs=[tensor],
            timeout_ms=timeout_ms,
            priority=priority,
            session_id=session_id,
        )
        return asyncio.run_coroutine_threadsafe(
            self._infer_full_async(request, timeout_ms), self._loop
        )

    def infer_batch(
        self, items: list[BatchInferItem], timeout_ms: int = 10000
    ) -> list[InferenceResult]:
        """Submit multiple model inferences in a single batch RPC.

        ai-runtime runs them in parallel on the NPU via shared VDevice
        ROUND_ROBIN scheduling, returning all results together.

        Args:
            items: List of (image, model_id, ...) tuples.
            timeout_ms: Overall wall-clock timeout for the entire batch.

        Returns:
            List of InferenceResult, one per item, in the same order.
        """
        if self.stub is None:
            self.connect()

        requests = []
        for item in items:
            tensor = self._numpy_to_tensor(item.image, "input")
            requests.append(
                inference_pb2.InferRequest(
                    model_id=item.model_id,
                    inputs=[tensor],
                    timeout_ms=item.timeout_ms,
                    priority=item.priority,
                )
            )

        batch_request = inference_pb2.InferBatchRequest(
            requests=requests,
            timeout_ms=timeout_ms,
        )
        fut = asyncio.run_coroutine_threadsafe(
            self._infer_batch_async(batch_request, timeout_ms), self._loop
        )
        response = fut.result(timeout=timeout_ms / 1000 + 5)

        if not response.status.success:
            # Partial failure: still return per-item results
            pass

        results = []
        for resp in response.responses:
            results.append(self._parse_infer_response(resp))
        return results

    async def _infer_batch_async(self, batch_request, timeout_ms):
        return await self.stub.InferBatch(batch_request, timeout=timeout_ms / 1000)

    async def _infer_batch_full_async(self, batch_request, timeout_ms):
        """Await InferBatch, return parsed InferenceResult per item (in order).

        Mirrors infer_batch()'s partial-failure handling (still returns per-item
        results). Used by infer_batch_async().
        """
        response = await self.stub.InferBatch(batch_request, timeout=timeout_ms / 1000)
        results = []
        for resp in response.responses:
            results.append(self._parse_infer_response(resp))
        return results

    def infer_batch_async(self, items: list[BatchInferItem], timeout_ms: int = 10000) -> Future:
        """Non-blocking infer_batch: returns a concurrent.futures.Future that
        resolves to List[InferenceResult] (one per item, in submission order).
        The caller MUST call fut.result(timeout=...). Symmetric to infer_batch();
        enables depth-N pipelines on the dual-model (pose+detect) path. The
        existing blocking infer_batch() is unchanged.
        """
        if self.stub is None:
            self.connect()

        requests = []
        for item in items:
            tensor = self._numpy_to_tensor(item.image, "input")
            requests.append(
                inference_pb2.InferRequest(
                    model_id=item.model_id,
                    inputs=[tensor],
                    timeout_ms=item.timeout_ms,
                    priority=item.priority,
                )
            )

        batch_request = inference_pb2.InferBatchRequest(
            requests=requests,
            timeout_ms=timeout_ms,
        )
        return asyncio.run_coroutine_threadsafe(
            self._infer_batch_full_async(batch_request, timeout_ms), self._loop
        )

    def infer_with_tensors(
        self,
        model_id: str,
        inputs: list[np.ndarray],
        input_names: list[str] | None = None,
        timeout_ms: int = 5000,
    ) -> list[np.ndarray]:
        if self.stub is None:
            self.connect()

        if input_names is None:
            input_names = [f"input_{i}" for i in range(len(inputs))]

        tensors = [self._numpy_to_tensor(arr, name) for arr, name in zip(inputs, input_names)]

        request = inference_pb2.InferRequest(
            model_id=model_id, inputs=tensors, timeout_ms=timeout_ms
        )

        fut = asyncio.run_coroutine_threadsafe(
            self._infer_tensors_async(request, timeout_ms), self._loop
        )
        response = fut.result(timeout=timeout_ms / 1000 + 5)

        if not response.status.success:
            raise RuntimeError(f"Inference failed: {response.status.message}")

        return [self._tensor_to_numpy(t) for t in response.outputs]

    async def _infer_tensors_async(self, request, timeout_ms):
        return await self.stub.Infer(request, timeout=timeout_ms / 1000)

    def subscribe(
        self,
        stream: str,
        model: str,
        fps: int = 10,
        session_id: str = "",
        raw_output_only: bool = False,
        max_consecutive_failures: int | None = 10,
    ) -> Iterator[tuple[int, InferenceResult]]:
        """Yield (frame_sequence, InferenceResult) for a camera stream subscription.

        Failed frames are skipped with a warning. If ``max_consecutive_failures``
        frames fail in a row (default 10), a RuntimeError is raised instead of
        yielding nothing forever. Pass 0 or None to disable the limit.
        """
        if self.stub is None:
            self.connect()

        request = inference_pb2.StreamInferRequest(
            model_id=model,
            stream_id=stream,
            fps_limit=fps,
            session_id=session_id,
            raw_output_only=raw_output_only,
        )

        # Bridge the async server-stream to a sync generator via a queue. The
        # caller blocks on q.get() (a futex), not a sync-CQ spin.
        q: queue.Queue[Any] = queue.Queue()
        SENTINEL = object()

        async def _pump():
            call = self.stub.StreamInfer(request)
            try:
                async for response in call:
                    q.put(response)
            except asyncio.CancelledError:
                cancel = getattr(call, "cancel", None)
                if cancel:
                    cancel()
                raise
            except Exception as e:
                q.put(e)
            finally:
                q.put(SENTINEL)

        pump_future = asyncio.run_coroutine_threadsafe(_pump(), self._loop)

        consecutive_failures = 0
        try:
            while True:
                item = q.get()
                if item is SENTINEL:
                    return
                if isinstance(item, Exception):
                    raise item
                response = item

                if not response.status.success:
                    consecutive_failures += 1
                    if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                        logger.warning(
                            "subscribe(stream=%r, model=%r): inference failed for frame %d "
                            "(%d consecutive): %s",
                            stream,
                            model,
                            response.frame_sequence,
                            consecutive_failures,
                            response.status.message,
                        )
                    if (
                        max_consecutive_failures
                        and consecutive_failures >= max_consecutive_failures
                    ):
                        raise RuntimeError(
                            f"Stream inference failed {consecutive_failures} consecutive times "
                            f"(stream={stream!r}, model={model!r}, "
                            f"last frame={response.frame_sequence}): "
                            f"{response.status.message!r}"
                        )
                    continue
                consecutive_failures = 0

                objects = []
                classifications = []
                landmarks = []
                masks = []
                ocr_lines = []
                embeddings = []
                depth_maps = []

                if response.HasField("post_result"):
                    (
                        objects,
                        classifications,
                        landmarks,
                        masks,
                        ocr_lines,
                        embeddings,
                        depth_maps,
                    ) = self._parse_post_result(response.post_result)

                raw_outputs = None
                if response.outputs:
                    raw_outputs = [self._tensor_to_numpy(t) for t in response.outputs]

                result = InferenceResult(
                    frame_sequence=response.frame_sequence,
                    timestamp_ns=response.timestamp_ns,
                    objects=objects,
                    classifications=classifications,
                    landmarks=landmarks,
                    masks=masks,
                    ocr_lines=ocr_lines,
                    embeddings=embeddings,
                    depth_maps=depth_maps,
                    raw_outputs=raw_outputs,
                    status_message=response.status.message,
                )

                yield response.frame_sequence, result
        finally:
            if not pump_future.done():
                pump_future.cancel()

    def register_model(
        self,
        model_path: str,
        model_id: str | None = None,
        owner_id: str | None = None,
        model_type: str | None = None,
        model_variant: str | None = None,
        inputs: list[dict] | None = None,
        outputs: list[dict] | None = None,
    ) -> str:
        if self.stub is None:
            self.connect()

        # Translate container path to host path for ai-runtime
        host_path = Config.translate_path_to_host(model_path)

        request = inference_pb2.ModelRegisterRequest(model_path=host_path, model_id=model_id or "")
        if owner_id:
            request.owner_id = owner_id
        if model_type:
            request.model_type = model_type
        if model_variant:
            request.model_variant = model_variant

        if inputs:
            for inp in inputs:
                spec = inference_pb2.TensorSpec(
                    shape=inp.get("shape", []),
                    dtype=self._dtype_str_to_enum(inp.get("dtype", "float32")),
                    name=inp.get("name", ""),
                )
                request.inputs.append(spec)

        if outputs:
            for out in outputs:
                spec = inference_pb2.TensorSpec(
                    shape=out.get("shape", []),
                    dtype=self._dtype_str_to_enum(out.get("dtype", "float32")),
                    name=out.get("name", ""),
                )
                request.outputs.append(spec)

        response = self._invoke(self.stub.RegisterModel, request, result_timeout=120)

        if not response.status.success:
            raise RuntimeError(f"Model registration failed: {response.status.message}")

        return response.model_id

    def _dtype_str_to_enum(self, dtype_str: str) -> int:
        dtype_map = {
            "uint8": inference_pb2.UINT8,
            "int8": inference_pb2.INT8,
            "uint16": inference_pb2.UINT16,
            "int16": inference_pb2.INT16,
            "float16": inference_pb2.FLOAT16,
            "float32": inference_pb2.FLOAT32,
            "int32": inference_pb2.INT32,
            "uint32": inference_pb2.UINT32,
        }
        return dtype_map.get(dtype_str.lower(), inference_pb2.FLOAT32)

    def unregister_model(self, model_id: str) -> None:
        if self.stub is None:
            self.connect()

        request = inference_pb2.ModelInfo(model_id=model_id)
        response = self._invoke(self.stub.UnregisterModel, request, result_timeout=30)

        if not response.success:
            raise RuntimeError(f"Model unregistration failed: {response.message}")

    def list_models(self) -> list[ModelInfo]:
        if self.stub is None:
            self.connect()

        response = self._invoke(self.stub.ListModels, inference_pb2.Empty(), result_timeout=30)

        models = []
        for m in response.models:
            models.append(
                ModelInfo(
                    model_id=m.model_id,
                    model_path=m.model_path,
                    version=m.version,
                    inputs=[
                        {"shape": list(i.shape), "dtype": i.dtype, "name": i.name} for i in m.inputs
                    ],
                    outputs=[
                        {"shape": list(o.shape), "dtype": o.dtype, "name": o.name}
                        for o in m.outputs
                    ],
                    estimated_tops=m.estimated_tops,
                    estimated_memory=m.estimated_memory,
                    load_timestamp=m.load_timestamp,
                )
            )

        return models

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        if self.stub is None:
            self.connect()

        request = inference_pb2.ModelInfo(model_id=model_id)
        response = self._invoke(self.stub.GetModelInfo, request, result_timeout=30)

        if not response.model_id:
            return None

        return ModelInfo(
            model_id=response.model_id, model_path=response.model_path, version=response.version
        )

    def get_stats(self) -> dict[str, Any]:
        if self.stub is None:
            self.connect()

        response = self._invoke(self.stub.GetStats, inference_pb2.Empty(), result_timeout=30)

        return {
            "device_utilization": response.device_utilization,
            "device_temperature": response.device_temperature,
            "total_memory_bytes": response.total_memory_bytes,
            "used_memory_bytes": response.used_memory_bytes,
            "cpu_utilization": response.cpu_utilization,
            "dsp_utilization": response.dsp_utilization,
            "ram_total_kib": response.ram_total_kib,
            "ram_used_kib": response.ram_used_kib,
            "model_stats": [
                {
                    "model_id": s.model_id,
                    "total_inferences": s.total_inferences,
                    "total_errors": s.total_errors,
                    "avg_latency_us": s.avg_latency_us,
                    "current_qps": s.current_qps,
                    "queue_depth": s.queue_depth,
                    "hw_fps": getattr(s, "hw_fps", 0),
                }
                for s in response.model_stats
            ],
        }

    def create_session(
        self,
        session_id: str,
        app_id: str = "",
        allowed_models: list[str] | None = None,
        max_qps: int = 0,
        max_concurrent: int = 0,
        priority: int = 4,
    ) -> str:
        if self.stub is None:
            self.connect()

        request = inference_pb2.SessionConfig(
            session_id=session_id,
            app_id=app_id,
            max_qps=max_qps,
            max_concurrent=max_concurrent,
            priority=priority,
        )

        if allowed_models:
            request.allowed_models.extend(allowed_models)

        response = self._invoke(self.stub.CreateSession, request, result_timeout=30)

        if not response.status.success:
            raise RuntimeError(f"Session creation failed: {response.status.message}")

        return response.session_id

    def destroy_session(self, session_id: str) -> None:
        if self.stub is None:
            self.connect()

        request = inference_pb2.SessionConfig(session_id=session_id)
        response = self._invoke(self.stub.DestroySession, request, result_timeout=30)

        if not response.success:
            raise RuntimeError(f"Session destruction failed: {response.message}")

    def update_postprocess_config(self, model_id: str, config_json: str) -> bool:
        """Update postprocess configuration for a model at runtime.

        For CLIP models, config_json can contain:
            {"prompts": ["a person", "a car"], "score_threshold": 0.3}

        Returns True on success.
        """
        if self.stub is None:
            self.connect()

        request = inference_pb2.UpdatePostprocessConfigRequest(
            model_id=model_id, config_json=config_json
        )
        response = self._invoke(self.stub.UpdatePostprocessConfig, request, result_timeout=30)

        if not response.status.success:
            raise RuntimeError(f"UpdatePostprocessConfig failed: {response.status.message}")

        return True
