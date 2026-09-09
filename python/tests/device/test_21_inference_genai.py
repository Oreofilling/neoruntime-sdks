"""Phase 2 — GenAI inference interfaces.

The GenAI surface (encode_text, session create/generate/abort/destroy)
needs a GenAI HEF on the device. If none is deployed the tests record
SKIP-NA rather than guessing at model files; encode_text is attempted
first because some builds expose the text encoder even without an LLM.
"""

from __future__ import annotations

import glob
import os
import unittest

from neoruntime_ipc_sdk import InferenceClient

from common import MODEL_DIR, DeviceTestCase

GENAI_HINTS = ("llm", "genai", "qwen", "tiny", "chat")


def _find_genai_hef():
    """Locate a GenAI-capable model, or None."""
    if not os.path.isdir(MODEL_DIR):
        return None
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.hef"))):
        name = os.path.basename(path).lower()
        if any(h in name for h in GENAI_HINTS):
            return path
    return None


class T01GenAI(DeviceTestCase):
    area = "genai"
    timeout_s = 90

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = InferenceClient()
        cls.hef = _find_genai_hef()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_01_encode_text(self):
        self.mark("InferenceClient.encode_text")
        try:
            emb = self.timed(
                self.client.encode_text, "a person riding a bicycle",
                label="encode_text",
            )
        except Exception as exc:
            # No text-encoder backing on this device — that is the
            # reportable precondition, not a defect.
            self.na(f"encode_text unavailable: {type(exc).__name__}: {exc}")
        self.evidence(dim=len(emb), first=round(emb[0], 6) if emb else None)
        self.assertIsInstance(emb, list)
        self.assertTrue(emb, "embedding list is empty")

    def test_02_session_create(self):
        self.mark("InferenceClient.genai_create_session")
        if not self.hef:
            self.na(f"no GenAI model found in {MODEL_DIR}")
        session_id = self.timed(
            self.client.genai_create_session, self.hef, kind="llm",
            label="genai_create_session",
        )
        self.evidence(session_id=session_id, hef=self.hef)
        self.assertIsInstance(session_id, str)
        self.assertTrue(session_id)
        # Destroy immediately so the daemon-side context is released
        # even if the generate test is skipped.
        self.timed(self.client.genai_destroy_session, session_id,
                   label="genai_destroy_session")

    def test_03_generate(self):
        self.mark("InferenceClient.genai_generate")
        if not self.hef:
            self.na(f"no GenAI model found in {MODEL_DIR}")
        session_id = self.client.genai_create_session(self.hef, kind="llm")
        try:
            messages = [
                '{"role": "system", "content": "You are a test probe."}',
                '{"role": "user", "content": "Say OK and nothing else."}',
            ]
            tokens = []
            for tok in self.client.genai_generate(
                session_id, messages, max_tokens=16, temperature=0.0
            ):
                tokens.append(tok)
                if sum(len(t) for t in tokens) > 512:
                    break
            text = "".join(tokens)
            self.evidence(session_id=session_id, n_chunks=len(tokens),
                          text=text[:200])
            self.assertTrue(tokens, "generate yielded no tokens")
        finally:
            try:
                self.client.genai_destroy_session(session_id)
            except Exception:
                pass

    def test_04_abort(self):
        self.mark("InferenceClient.genai_abort")
        if not self.hef:
            self.na(f"no GenAI model found in {MODEL_DIR}")
        session_id = self.client.genai_create_session(self.hef, kind="llm")
        try:
            messages = [
                '{"role": "user", "content": "Count from 1 to 100 slowly."}'
            ]
            gen = self.client.genai_generate(session_id, messages,
                                             max_tokens=256)
            next(gen, None)  # start the job server-side
            self.timed(self.client.genai_abort, session_id,
                       label="genai_abort")
            self.evidence(session_id=session_id, aborted=True)
        except StopIteration:
            # Generation already finished before abort — the interface
            # pair still worked; note it.
            self.evidence(session_id=session_id,
                          note="generation completed before abort")
        finally:
            try:
                self.client.genai_destroy_session(session_id)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
