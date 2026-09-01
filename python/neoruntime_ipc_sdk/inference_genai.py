"""GenAI extension methods for InferenceClient (mixin).

encode_text + the genai_* session/generate/abort surface. Lives apart
from the core client so the hot infer path stays readable; mixed in as
``class InferenceClient(GenAiMixin)`` so ``client.genai_*`` is
unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from typing import Any, Iterator

from .config import Config
from .proto import inference_pb2

logger = logging.getLogger(__name__)


class GenAiMixin:
    """Requires the host class to provide ``_invoke``, ``connect``,
    ``stub`` and ``_loop`` (see InferenceClient)."""

    def encode_text(self, text: str, timeout_ms: int = 5000) -> list[float]:
        """Encode a text string to a CLIP embedding via NPU.

        Returns a list of floats (512-dim for ViT-B/32).
        """
        if self.stub is None:
            self.connect()

        request = inference_pb2.EncodeTextRequest(text=text)
        response = self._invoke(
            self.stub.EncodeText,
            request,
            timeout=timeout_ms / 1000,
            result_timeout=timeout_ms / 1000 + 5,
        )

        if response.status.code != 0:
            raise RuntimeError(f"EncodeText failed: {response.status.message}")

        return list(response.embedding.data)

    # ── GenAI (LLM/VLM) ──────────────────────────────────────────────────────

    def genai_create_session(
        self, hef_path: str, kind: str = "llm", lora_name: str = "", optimize_memory: bool = False
    ) -> str:
        """Create a GenAI (LLM/VLM) session.

        Args:
            hef_path: Path to the HEF model file on the device.
            kind: "llm" or "vlm".
            lora_name: Optional LoRA adapter name.
            optimize_memory: Enable memory-optimized tokenization.

        Returns:
            session_id string.
        """
        if self.stub is None:
            self.connect()

        host_path = Config.translate_path_to_host(hef_path)
        kind_map = {"llm": 0, "vlm": 1}
        request = inference_pb2.GenaiCreateSessionRequest(
            hef_path=host_path,
            kind=kind_map.get(kind, 0),
            lora_name=lora_name,
            optimize_memory=optimize_memory,
        )
        response = self._invoke(
            self.stub.GenaiCreateSession,
            request,
            timeout=300,
            result_timeout=305,
        )

        if response.status.code != 0:
            raise RuntimeError(f"GenAI create session failed: {response.status.message}")

        return response.session_id

    def genai_destroy_session(self, session_id: str) -> None:
        """Destroy a GenAI session and free resources."""
        if self.stub is None:
            self.connect()

        # hef_path is reused as session_id carrier for destroy
        request = inference_pb2.GenaiCreateSessionRequest(hef_path=session_id)
        response = self._invoke(
            self.stub.GenaiDestroySession,
            request,
            timeout=10,
            result_timeout=15,
        )

        if response.code != 0:
            raise RuntimeError(f"GenAI destroy session failed: {response.message}")

    def genai_generate(
        self,
        session_id: str,
        messages: list[str],
        images: list[bytes] | None = None,
        stop_tokens: list[str] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        max_tokens: int = 512,
        do_sample: bool = False,
    ) -> Iterator[str]:
        """Stream generated tokens from a GenAI session.

        Args:
            session_id: Session from genai_create_session().
            messages: List of JSON-encoded chat messages.
            images: Optional RGB image frames for VLM.
            stop_tokens: Optional stop sequences.
            temperature, top_p, top_k, max_tokens, do_sample: Generation params.

        Yields:
            Token strings as they are generated.
        """
        if self.stub is None:
            self.connect()

        request = inference_pb2.GenaiGenerateRequest(
            session_id=session_id,
            messages_json=messages,
            image_frames=images or [],
            stop_tokens=stop_tokens or [],
        )
        if do_sample or temperature > 0 or max_tokens != 512:
            request.params.temperature = temperature
            request.params.top_p = top_p
            request.params.top_k = top_k
            request.params.max_generated_tokens = max_tokens
            request.params.do_sample = do_sample

        # Bridge the async server-stream to a sync generator via a queue.
        q: queue.Queue[Any] = queue.Queue()
        SENTINEL = object()

        async def _pump():
            call = self.stub.GenaiGenerate(request)
            try:
                async for resp in call:
                    q.put(resp)
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

        try:
            while True:
                item = q.get()
                if item is SENTINEL:
                    return
                if isinstance(item, Exception):
                    raise item
                resp = item
                if resp.HasField("token"):
                    yield resp.token
                elif resp.HasField("finish"):
                    break
        finally:
            if not pump_future.done():
                pump_future.cancel()

    def genai_abort(self, session_id: str) -> None:
        """Abort an ongoing generation."""
        if self.stub is None:
            self.connect()

        request = inference_pb2.GenaiAbortRequest(session_id=session_id)
        self._invoke(
            self.stub.GenaiAbort,
            request,
            timeout=5,
            result_timeout=10,
        )
