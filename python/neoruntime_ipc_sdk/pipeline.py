"""Composed pre → infer → post pipeline over :class:`InferenceClient`.

The pipeline is for the **client-side processing path**: apps that need
their own preprocessing (raw-tensor models, bandwidth-sensitive loops on
the sub stream) and/or their own postprocessing (generic plugin exports
whose thresholds cannot be tuned server-side). When a family model with
server-side postprocess fits, ``InferenceClient.subscribe`` remains the
cheaper path — no pixel ever reaches the app (see python/PERFORMANCE.md
for the decision table).

For the whole capture → infer → overlay loop as one platform-scheduled
call — daemon-fed frames, results drawn on the hardware overlay and
baked into the encoded stream, the app only seeing lightweight results —
use :class:`~neoruntime_ipc_sdk.StreamPipeline` instead; this module
pulls pixels into the app, that one never does. Choose by where the
pixels should live.

Result priority: when a postprocessor is set and the model returned
``raw_outputs``, the client-side decode wins (that is the point — its
thresholds are yours); otherwise the server-decoded
``InferenceResult.objects`` pass through untouched.

.. code-block:: python

    from neoruntime_ipc_sdk import InferenceClient, InferencePipeline, YoloV8Postprocessor

    inf = InferenceClient()
    pipe = InferencePipeline.from_model(
        "custom_yolov8n",
        client=inf,
        postprocessor=YoloV8Postprocessor(labels=["person", "car"], score_threshold=0.3),
    )
    for frame in FdMediaClient().subscribe("sub", keep_fd=True):
        out = pipe.run(frame)          # .objects in source-frame coordinates
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any

from .inference import InferenceClient
from .inference_types import DetectedObject, InferenceResult
from .preprocess import Preprocessor

__all__ = ["InferencePipeline", "PipelineResult"]


def _release_ref_like(tensor: Any) -> bool:
    """Release ``tensor`` if it exposes the device-resident ref protocol.

    That protocol is ``buffer_id`` plus a callable ``release()`` — what
    :class:`~neoruntime_ipc_sdk.dsp.DspBufferRef` implements. Anything
    else (ndarrays, opaque test sentinels) passes through untouched.
    Returns whether a ref was released; a release that itself fails
    with :class:`~neoruntime_ipc_sdk.dsp.DspError` still counts — the
    ref is dead either way, and a dropped transport reclaims its
    buffers at disconnect.
    """
    release = getattr(tensor, "release", None)
    if not callable(release) or not hasattr(tensor, "buffer_id"):
        return False
    from .dsp import DspError  # noqa: PLC0415 — deferred: heavy proto import

    try:
        release()
    except DspError:
        pass
    return True


@dataclass
class PipelineResult:
    """One pipeline pass: decoded objects plus everything underneath."""

    objects: list[DetectedObject]
    result: InferenceResult
    meta: Any | None = None
    latency_ms: float = 0.0
    tensor: Any | None = None
    """The preprocessed image actually sent to the model (when a
    preprocessor ran) — handy for drawing previews at input size. On
    the NV12 passthrough path this is a device-resident buffer ref,
    which the pipeline releases once inference settles unless it was
    built with ``retain_input=True`` (then :meth:`release` hands it
    back when the chain using it is done)."""

    def release(self) -> None:
        """Release a retained passthrough input ref, if any.

        Idempotent, and a no-op for plain-array tensors — only a
        ``retain_input=True`` pipeline leaves a live ref in ``tensor``.
        """
        if _release_ref_like(self.tensor):
            self.tensor = None


class InferencePipeline:
    """Bind a preprocessor, an :class:`InferenceClient` and a postprocessor.

    Args:
        client: inference client (a default one is created when omitted).
        model_id: default model for :meth:`run`.
        preprocessor: :class:`~neoruntime_ipc_sdk.Preprocessor`, or
            ``None`` to pass sources through untouched.
        postprocessor: a :class:`~neoruntime_ipc_sdk.Postprocessor`, or
            ``None`` to keep server-decoded results as-is.
        infer_timeout_ms: per-call inference timeout.
        retain_input: keep a device-resident passthrough ref in
            ``PipelineResult.tensor`` instead of releasing it once
            inference settles (the default — refs carry no garbage
            collection, so one leaked per frame would hold daemon-side
            DSP buffers for the process lifetime). Retain it to chain
            the ref further (e.g. as a zero-copy blend base) and hand
            it back with :meth:`PipelineResult.release`.
    """

    def __init__(
        self,
        client: InferenceClient | None = None,
        model_id: str | None = None,
        preprocessor: Preprocessor | None = None,
        postprocessor: Any | None = None,
        infer_timeout_ms: int = 5000,
        retain_input: bool = False,
    ):
        self.client = client if client is not None else InferenceClient()
        self.model_id = model_id
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.infer_timeout_ms = infer_timeout_ms
        self.retain_input = bool(retain_input)

    @classmethod
    def from_model(
        cls,
        model_id: str,
        client: InferenceClient | None = None,
        postprocessor: Any | None = None,
        **preprocess_kwargs: Any,
    ) -> InferencePipeline:
        """Derive the preprocessor from the model's input spec and bind it.

        Accepts the same keyword overrides as
        :meth:`Preprocessor.from_model
        <neoruntime_ipc_sdk.Preprocessor.from_model>`.
        """
        client = client if client is not None else InferenceClient()
        pre = Preprocessor.from_model(client, model_id, **preprocess_kwargs)
        return cls(
            client=client, model_id=model_id, preprocessor=pre,
            postprocessor=postprocessor,
        )

    def _prepare(self, source: Any) -> tuple[Any, Any | None]:
        if self.preprocessor is None:
            return source, None
        return self.preprocessor(source)

    def _decode(self, result: InferenceResult, meta: Any | None) -> list[DetectedObject]:
        if self.postprocessor is not None and result.raw_outputs:
            return self.postprocessor(result.raw_outputs, meta)
        return result.objects

    def _model(self, model_id: str | None) -> str:
        model_id = model_id or self.model_id
        if not model_id:
            raise ValueError("no model_id — pass one to run() or the constructor")
        return model_id

    def run(self, source: Any, model_id: str | None = None) -> PipelineResult:
        """Preprocess → infer → postprocess one frame or array, blocking."""
        model_id = self._model(model_id)
        started = time.perf_counter()
        tensor, meta = self._prepare(source)
        try:
            result = self.client.infer(tensor, model_id, timeout_ms=self.infer_timeout_ms)
        except BaseException:
            self._settle_input(tensor)  # the pass is dead — don't leak its ref
            raise
        objects = self._decode(result, meta)
        return PipelineResult(
            objects=objects,
            result=result,
            meta=meta,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            tensor=self._settle_input(tensor),
        )

    def _settle_input(self, tensor: Any) -> Any:
        """Release a passthrough input ref once inference has settled.

        Returns what ``PipelineResult.tensor`` should carry: the tensor
        when there is nothing to release — plain arrays, or a
        ``retain_input=True`` pipeline keeping its ref for a zero-copy
        chain — and ``None`` behind a released ref.
        """
        if self.retain_input or not _release_ref_like(tensor):
            return tensor
        return None

    def run_async(self, source: Any, model_id: str | None = None) -> concurrent.futures.Future:
        """Non-blocking :meth:`run`; the Future resolves to PipelineResult.

        Preprocessing runs inline (cheap — it is the resize/convert on
        the small image); the NPU call rides
        :meth:`InferenceClient.infer_async
        <neoruntime_ipc_sdk.InferenceClient.infer_async>` and the
        postprocessor runs on the completing thread. ``latency_ms`` is
        not measured on this path. A passthrough input ref is released
        when the inner future settles — success or failure — unless the
        pipeline was built with ``retain_input=True``. As with
        ``infer_async``, the returned
        Future must be resolved (``.result()``) or exceptions stay
        silently swallowed.
        """
        model_id = self._model(model_id)
        tensor, meta = self._prepare(source)
        inner = self.client.infer_async(tensor, model_id, timeout_ms=self.infer_timeout_ms)
        outer: concurrent.futures.Future = concurrent.futures.Future()
        decode = self._decode

        def _finish(fut: concurrent.futures.Future) -> None:
            try:
                result = fut.result()
                objects = decode(result, meta)
                outer.set_result(
                    PipelineResult(
                        objects=objects, result=result, meta=meta,
                        tensor=self._settle_input(tensor),
                    )
                )
            except Exception as exc:  # propagate through the outer Future
                self._settle_input(tensor)  # the pass is dead — don't leak its ref
                outer.set_exception(exc)

        inner.add_done_callback(_finish)
        return outer
