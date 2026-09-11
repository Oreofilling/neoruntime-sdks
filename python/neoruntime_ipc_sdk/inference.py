"""
AI Inference Client
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Iterator, TYPE_CHECKING

import grpc
import numpy as np

from ._transport import MAX_GRPC_MESSAGE_LENGTH
from .config import Config
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

if TYPE_CHECKING:
    from .dsp import DspClient
    from .frame import Frame, FrameHandle

logger = logging.getLogger(__name__)


class _SubscribeIterator:
    """Consumer-side iterator over a stream-infer subscription.

    Wraps the generator that bridges the async server stream so the pump
    thread can update the ``dropped`` tally (generators reject attribute
    assignment) while the consumer reads it as a plain int.
    """

    def __init__(self, gen: Iterator[tuple[int, InferenceResult]]) -> None:
        self._gen = gen
        self.dropped = 0
        # End-to-end result latency (result timestamp → local receive), ms.
        # ``avg`` is an EMA (alpha 0.1); both stay 0.0 when the server does
        # not stamp result timestamps.
        self.last_latency_ms = 0.0
        self.avg_latency_ms = 0.0
        # Result skew (result-ready − frame capture, µs), server-computed
        # on the device clock so it is immune to host/device clock skew.
        # ``avg`` is an EMA (alpha 0.1); both stay 0 when the server does
        # not report skew (older server, or failed inferences).
        self.last_skew_us = 0
        self.avg_skew_us = 0.0
        # Cross-thread cancel hooks, installed by the wrapped generator once
        # the pump task exists (empty before the first next()).
        self._cancel_hooks: list[Callable[[], None]] = []

    def __iter__(self) -> _SubscribeIterator:
        return self

    def __next__(self) -> tuple[int, InferenceResult]:
        return next(self._gen)

    def close(self) -> None:
        self._gen.close()

    def cancel(self) -> None:
        """Stop the subscription from any thread.

        Unlike ``close()`` (only valid on the consuming thread), this is
        safe to call from any thread: it cancels the pump task and offers
        the terminal sentinel so a consumer blocked in ``next()`` wakes
        immediately and the generator exits. Idempotent; a no-op before
        the first ``next()`` (nothing has started yet).
        """
        for hook in list(self._cancel_hooks):
            hook()

    def throw(self, typ, val=None, tb=None):
        return self._gen.throw(typ, val, tb)


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
        # Lazily-created DspClient for zero-copy frame inference
        # (``infer(frame, ...)``); closed by close().
        self._dsp: DspClient | None = None

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
        if self._dsp is not None:
            self._dsp.close()
            self._dsp = None
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

    def _frame_input_tensor(self, image: Any) -> inference_pb2.Tensor | None:
        """Zero-copy input path: import a Frame/FrameHandle as Tensor.buffer_id.

        Returns None for array inputs (the caller falls back to the bytes
        path). The frame's dma-bufs are imported via DSP_IMPORT; ai-runtime
        resolves the id against the daemon buffer registry, so the pixels
        never cross a socket. The tensor carries the import id — the caller
        MUST release it via :meth:`_release_input` when the RPC settles.
        """
        from .frame import Frame, FrameHandle

        if isinstance(image, FrameHandle):
            handle = image
        elif isinstance(image, Frame):
            handle = image.handle
            if handle is None:
                raise ValueError(
                    "frame carries no dma-buf handle — subscribe/receive with "
                    "keep_fd=True, or pass the pixel array directly"
                )
        else:
            return None

        if handle.closed:
            raise ValueError(
                "frame handle is closed — its dma-bufs are gone; keep the "
                "Frame/FrameHandle alive across the call"
            )
        if handle.format != "NV12":
            # ai-runtime's registry-side repack only knows tight NV12
            raise ValueError(
                f"zero-copy inference supports NV12 frames only "
                f"(got {handle.format or 'unknown format'})"
            )

        if self._dsp is None:
            from .dsp import DspClient

            self._dsp = DspClient()
        buffer_id = self._dsp.import_frame(handle)
        return inference_pb2.Tensor(
            buffer_id=buffer_id,
            dtype=inference_pb2.UINT8,
            shape=[handle.height * 3 // 2, handle.width],
        )

    def _release_input(self, buffer_id: int) -> None:
        """Free an imported frame buffer after its Infer RPC settled."""
        if self._dsp is None:
            return
        try:
            self._dsp.release_buffer(buffer_id)
        except Exception:
            # Best-effort: the daemon's lease watchdog reaps unreleased ids
            logger.debug("buffer release after infer failed", exc_info=True)

    def infer(
        self,
        image: Any,
        model_id: str,
        timeout_ms: int = 5000,
        priority: int = 4,
        session_id: str = "",
    ) -> InferenceResult:
        """Run one inference.

        ``image`` is an ndarray (pixels shipped as bytes) or a
        keep-fd ``Frame``/``FrameHandle`` (NV12 only): the frame's
        dma-bufs are imported and referenced by buffer id, so no pixel
        copy crosses the transport.
        """
        if self.stub is None:
            self.connect()

        tensor = self._frame_input_tensor(image)
        if tensor is None:
            tensor = self._numpy_to_tensor(image, "input")

        request = inference_pb2.InferRequest(
            model_id=model_id,
            inputs=[tensor],
            timeout_ms=timeout_ms,
            priority=priority,
            session_id=session_id,
        )

        try:
            fut = asyncio.run_coroutine_threadsafe(self._infer_async(request, timeout_ms), self._loop)
            # +5s slack covers NPU cold start / HEF context init; the gRPC deadline
            # itself is timeout_ms/1000.
            response = fut.result(timeout=timeout_ms / 1000 + 5)
        finally:
            if tensor.buffer_id:
                self._release_input(tensor.buffer_id)

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
        image: Any,
        model_id: str,
        timeout_ms: int = 5000,
        priority: int = 4,
        session_id: str = "",
    ) -> Future:
        """Non-blocking infer: returns a concurrent.futures.Future that resolves
        to a parsed InferenceResult. The caller MUST call fut.result(timeout=...)
        to obtain the result (or propagate the error).

        ``image`` may be an ndarray or a keep-fd NV12 Frame/FrameHandle
        (zero-copy buffer-id input, same as :meth:`infer`); the import is
        released when the RPC settles.

        Enables depth-N pipelines: submit frame N+1 while still awaiting frame N
        so the NPU stays busy across the host-side gap between jobs. Schedules
        onto the same background asyncio loop infer() already uses. The existing
        blocking infer() is unchanged.
        """
        if self.stub is None:
            self.connect()

        tensor = self._frame_input_tensor(image)
        if tensor is None:
            tensor = self._numpy_to_tensor(image, "input")

        request = inference_pb2.InferRequest(
            model_id=model_id,
            inputs=[tensor],
            timeout_ms=timeout_ms,
            priority=priority,
            session_id=session_id,
        )

        if not tensor.buffer_id:
            return asyncio.run_coroutine_threadsafe(
                self._infer_full_async(request, timeout_ms), self._loop
            )

        async def _infer_and_release():
            try:
                return await self._infer_full_async(request, timeout_ms)
            finally:
                self._release_input(tensor.buffer_id)

        return asyncio.run_coroutine_threadsafe(_infer_and_release(), self._loop)

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
        queue_size: int | None = 100,
    ) -> Iterator[tuple[int, InferenceResult]]:
        """Yield (frame_sequence, InferenceResult) for a camera stream subscription.

        Failed frames are skipped with a warning. If ``max_consecutive_failures``
        frames fail in a row (default 10), a RuntimeError is raised instead of
        yielding nothing forever. Pass 0 or None to disable the limit.

        ``queue_size`` bounds the client-side result queue. A consumer slower
        than the stream drops its *oldest* queued results (the newest always
        gets through); each drop is counted on the returned iterator's
        ``dropped`` attribute and reported in a throttled warning — instead
        of the queue growing memory without bound. Pass 0 or None for an
        unbounded queue. A stream error is the queue's one terminal item
        (offered exactly once, after every result) so backpressure can
        delay but never drop it: the consumer always sees the exception.
        """
        gen_ref: list[_SubscribeIterator] = []

        def _gen() -> Iterator[tuple[int, InferenceResult]]:
            gen = gen_ref[0]
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
            q: queue.Queue[Any] = (
                queue.Queue(maxsize=queue_size) if queue_size else queue.Queue()
            )
            SENTINEL = object()

            def _offer(item: Any) -> None:
                # put_nowait with drop-oldest overflow. The one terminal
                # item (a stream error, or the SENTINEL) is offered exactly
                # once, last, and nothing follows it — so it can evict a
                # queued result but can never be evicted itself.
                while True:
                    try:
                        q.put_nowait(item)
                        return
                    except queue.Full:
                        try:
                            q.get_nowait()
                            gen.dropped += 1
                            if gen.dropped == 1 or gen.dropped % 10 == 0:
                                logger.warning(
                                    "subscribe(stream=%r, model=%r): consumer behind; "
                                    "dropped %d queued results",
                                    stream,
                                    model,
                                    gen.dropped,
                                )
                        except queue.Empty:
                            pass  # consumer drained between put and get; retry

            async def _pump():
                call = self.stub.StreamInfer(request)
                terminal: Any = SENTINEL
                try:
                    async for response in call:
                        _offer(response)
                except asyncio.CancelledError:
                    cancel = getattr(call, "cancel", None)
                    if cancel:
                        cancel()
                    raise  # terminal stays SENTINEL: the consumer ends normally
                except Exception as e:
                    terminal = e  # the error becomes the one terminal item
                finally:
                    _offer(terminal)

            pump_future = asyncio.run_coroutine_threadsafe(_pump(), self._loop)

            # Cross-thread cancel(): stop the pump and deliver the terminal
            # sentinel so a consumer blocked in q.get() wakes immediately.
            # queue.Queue is thread-safe; concurrent.futures cancel() is
            # thread-safe; a second SENTINEL after the generator returned is
            # just an unread queue item.
            def _cancel_hook() -> None:
                if not pump_future.done():
                    pump_future.cancel()
                _offer(SENTINEL)

            gen._cancel_hooks.append(_cancel_hook)

            consecutive_failures = 0
            try:
                while True:
                    item = q.get()
                    if item is SENTINEL:
                        return
                    if isinstance(item, Exception):
                        raise item
                    response = item

                    recv_ns = time.time_ns()
                    latency_ms = (recv_ns - response.timestamp_ns) / 1e6
                    if 0.0 < latency_ms < 60_000.0:  # sane window (same-host clock)
                        gen.last_latency_ms = latency_ms
                        if gen.avg_latency_ms == 0.0:
                            gen.avg_latency_ms = latency_ms
                        else:
                            gen.avg_latency_ms += 0.1 * (latency_ms - gen.avg_latency_ms)

                    # Skew comes precomputed from the server (device clock on
                    # both stamps), so it is valid regardless of any
                    # host/device clock offset that limits latency_ms above.
                    skew_us = getattr(response, "skew_us", 0)
                    if skew_us > 0:
                        gen.last_skew_us = skew_us
                        if gen.avg_skew_us == 0.0:
                            gen.avg_skew_us = float(skew_us)
                        else:
                            gen.avg_skew_us += 0.1 * (skew_us - gen.avg_skew_us)

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
                        skew_us=getattr(response, "skew_us", 0),
                    )

                    yield response.frame_sequence, result
            finally:
                if not pump_future.done():
                    pump_future.cancel()

        it = _SubscribeIterator(_gen())
        gen_ref.append(it)
        return it

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
                        {
                            "shape": list(i.shape),
                            "dtype": i.dtype,
                            "name": i.name,
                            "layout": i.layout,
                        }
                        for i in m.inputs
                    ],
                    outputs=[
                        {
                            "shape": list(o.shape),
                            "dtype": o.dtype,
                            "name": o.name,
                            "layout": o.layout,
                        }
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
            model_id=response.model_id,
            model_path=response.model_path,
            version=response.version,
            inputs=[
                {
                    "shape": list(i.shape),
                    "dtype": i.dtype,
                    "name": i.name,
                    "layout": i.layout,
                }
                for i in response.inputs
            ],
            outputs=[
                {
                    "shape": list(o.shape),
                    "dtype": o.dtype,
                    "name": o.name,
                    "layout": o.layout,
                }
                for o in response.outputs
            ],
            estimated_tops=response.estimated_tops,
            estimated_memory=response.estimated_memory,
            load_timestamp=response.load_timestamp,
        )

    def get_stats(self, sampling_window_ms: int | None = None) -> dict[str, Any]:
        """System and per-model stats.

        ``sampling_window_ms`` bounds the blocking HAL sampling window
        behind device/CPU/DSP utilization (they are measured, not read —
        the call blocks roughly this long). ``None`` sends the
        server-default request (500 ms); the server clamps explicit
        values to [1, 5000]. Cheap periodic snapshots should ask for a
        small window (1-50 ms).
        """
        if self.stub is None:
            self.connect()

        if sampling_window_ms is None:
            request: Any = inference_pb2.Empty()
        else:
            request = inference_pb2.GetStatsRequest(
                sampling_window_ms=max(1, min(5000, int(sampling_window_ms)))
            )

        response = self._invoke(self.stub.GetStats, request, result_timeout=30)

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
                    # Stream-infer skew aggregates (µs); 0 when the model has
                    # never run on a stream or the server predates the fields.
                    "avg_skew_us": getattr(s, "avg_skew_us", 0),
                    "max_skew_us": getattr(s, "max_skew_us", 0),
                    "skew_samples": getattr(s, "skew_samples", 0),
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

        For detection models, the numeric postprocess keys are accepted:
            {"detection_threshold": 0.38, "iou_threshold": 0.45,
             "max_boxes": 80}

        Applicability, verified on-device (hailo15, 2026-09):

        - ``detection_threshold`` is honored at runtime **only when the
          model's postprocess resolves to a family function**
          (``hailo_yolov8n``/``hailo_yolov8s``/``hailo_yolov8m`` — the
          default for detection models registered without a
          ``backend_function`` in their variant JSON). Generic plugin
          exports (e.g. ``yolov5m_vehicles``) hardcode their thresholds
          and ignore JSON tuning entirely.
        - ``iou_threshold`` / ``max_boxes`` are accepted by the chain but
          have no behavioral effect: suppression and box capping happen
          in the HEF's compile-time integrated NMS on the accelerator,
          so they cannot be moved after compilation.

        Unknown keys are rejected server-side (the RPC raises) — push
        only keys the model's postprocess schema knows.

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
