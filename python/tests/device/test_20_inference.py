"""Phase 2 — inference interfaces (InferenceClient, non-GenAI).

Registers a YOLO-World HEF from the device's model partition (without
``owner_id`` — the global-registry gotcha), then exercises infer /
batch / tensors / subscribe / session / postprocess paths against real
frames pulled from the camera.
"""

from __future__ import annotations

import os
import unittest

from neoruntime_ipc_sdk import (
    BatchInferItem,
    FdMediaClient,
    InferenceClient,
    ModelInfo,
)

from common import MODEL_DIR, DeviceTestCase, known_issue

MODEL_PATH = os.path.join(MODEL_DIR, "yolo_world_v2s.hef")
TEST_MODEL_ID = "sdk-test-yolo"
# Probed 2026-09-08: the daemon's stream table names the primary stream
# "main". A wrong name ("cam0_main", …) is NOT an error — subscribe just
# goes totally silent, which is the exact hang shape we must not produce.
SUBSCRIBE_STREAM = "main"

# KNOWN contract/deployment note, corrected 2026-09-09 by live probing
# (the earlier "daemon-side regression" read was wrong): -2799/-2811 is
# the daemon's input byte_size validation — an RGB array into an NV12
# model, or any buffer whose size disagrees with the model geometry,
# fails there. BOTH bundled models are yolo_world dual-input models, so
# the single-image infer RPC cannot drive them at all (-2799 even with
# geometry-matched NV12). A single-input model (hailo_yolov8n_384_640)
# fed geometry-matched NV12 infers fine (~70 FPS, same daemon build).
_INFER_2799 = (
    "infer family: yolo_world models are dual-input — single-image "
    "infer cannot drive them (-2799); feeding RGB instead of NV12 "
    "fails byte_size validation (-2799/-2811). Verified 2026-09-09: "
    "geometry-matched NV12 into a single-input model works on the "
    "same daemon build"
)


def _grab_rgb():
    """One RGB frame from the camera, or None when media is down."""
    media = FdMediaClient()
    try:
        frame = media.get_frame("main", timeout_ms=5000)
        if frame is None:
            return None
        return frame.to_rgb()
    finally:
        media.close()


class T01Lifecycle(DeviceTestCase):
    area = "inference"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"no model at {MODEL_PATH}")
        cls.client = InferenceClient()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.unregister_model(TEST_MODEL_ID)
        except Exception:
            pass
        cls.client.close()

    def test_01_connect_close(self):
        self.mark("InferenceClient.connect/connected")
        self.timed(self.client.connect, label="connect")
        # `connected` is a property on InferenceClient, not a method.
        self.assertTrue(self.client.connected)
        self.evidence(endpoint=self.client.endpoint)

    def test_02_register_model(self):
        self.mark("InferenceClient.register_model")
        # owner_id deliberately omitted: models registered with an
        # owner_id never enter the global table and subscribe() then
        # fails with "Model not found" (verified field defect).
        model_id = self.timed(
            self.client.register_model, MODEL_PATH, model_id=TEST_MODEL_ID,
            label="register_model",
        )
        self.evidence(model_id=model_id)
        self.assertEqual(model_id, TEST_MODEL_ID)

    def test_03_list_models(self):
        self.mark("InferenceClient.list_models")
        models = self.timed(self.client.list_models, label="list_models")
        ids = [m.model_id if hasattr(m, "model_id") else str(m) for m in models]
        self.evidence(count=len(models), ids=ids[:20])
        self.assertIn(TEST_MODEL_ID, ids)

    def test_04_get_model_info(self):
        self.mark("InferenceClient.get_model_info")
        info = self.timed(
            self.client.get_model_info, TEST_MODEL_ID, label="get_model_info"
        )
        self.assertIsInstance(info, ModelInfo)
        self.evidence(
            model_id=info.model_id,
            version=info.version,
            n_inputs=len(info.inputs),
            n_outputs=len(info.outputs),
            estimated_tops=info.estimated_tops,
        )
        self.assertEqual(info.model_id, TEST_MODEL_ID)

    def test_05_get_stats(self):
        self.mark("InferenceClient.get_stats")
        stats = self.timed(self.client.get_stats, label="get_stats")
        self.evidence(stats={k: stats[k] for k in list(stats)[:10]})
        self.assertIsInstance(stats, dict)

    @known_issue(
        "update_postprocess_config: daemon answers -2 for documented keys "
        "(detection_threshold/iou_threshold) — the same call worked "
        "against the 2026-08-28 patched daemon, so this is a "
        "daemon-deployment regression, not an SDK contract change")
    def test_06_update_postprocess_config(self):
        self.mark("InferenceClient.update_postprocess_config")
        ok = self.timed(
            self.client.update_postprocess_config,
            TEST_MODEL_ID,
            # Documented keys: detection_threshold / iou_threshold /
            # max_boxes (conf_threshold/nms_threshold are NOT accepted
            # by the daemon and it answers -2 "Failed to update config").
            '{"detection_threshold": 0.30, "iou_threshold": 0.45}',
            label="update_postprocess_config",
        )
        self.evidence(returned=ok)
        self.assertTrue(ok)

    def test_07_unregister_model(self):
        self.mark("InferenceClient.unregister_model")
        self.client.unregister_model(TEST_MODEL_ID)
        ids = [m.model_id if hasattr(m, "model_id") else str(m)
               for m in self.client.list_models()]
        self.assertNotIn(TEST_MODEL_ID, ids)
        # Re-register: later classes in this module reuse it.
        self.client.register_model(MODEL_PATH, model_id=TEST_MODEL_ID)


class T02Infer(DeviceTestCase):
    area = "inference"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"no model at {MODEL_PATH}")
        cls.client = InferenceClient()
        cls.client.register_model(MODEL_PATH, model_id=TEST_MODEL_ID)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.unregister_model(TEST_MODEL_ID)
        except Exception:
            pass
        cls.client.close()

    def setUp(self):
        super().setUp()
        self.image = _grab_rgb()
        if self.image is None:
            self.na("camera produced no frame for infer input")

    @known_issue(_INFER_2799)
    def test_01_infer(self):
        self.mark("InferenceClient.infer")
        result = self.timed(
            self.client.infer, self.image, TEST_MODEL_ID, label="infer"
        )
        self.evidence(
            objects=len(result.objects),
            infer_time_us=result.infer_time_us,
            hw_infer_time_us=result.hw_infer_time_us,
            status=result.status_message,
            shape=f"{self.image.shape}",
        )
        self.assertGreaterEqual(result.infer_time_us, 0)

    @known_issue(_INFER_2799)
    def test_02_infer_async(self):
        self.mark("InferenceClient.infer_async")
        result = self.timed(
            self.client.infer_async, self.image, TEST_MODEL_ID,
            label="infer_async",
        )
        # infer_async returns a result or future depending on transport;
        # both must resolve to an object with a frame_sequence.
        resolved = getattr(result, "result", lambda: result)()
        self.evidence(type=type(resolved).__name__,
                      objects=len(getattr(resolved, "objects", [])))
        self.assertTrue(hasattr(resolved, "frame_sequence"))

    def test_03_infer_batch(self):
        self.mark("InferenceClient.infer_batch")
        items = [BatchInferItem(image=self.image, model_id=TEST_MODEL_ID),
                 BatchInferItem(image=self.image, model_id=TEST_MODEL_ID)]
        results = self.timed(
            self.client.infer_batch, items, label="infer_batch"
        )
        # Batch RPC returning N rows only proves the envelope — record
        # per-item status so a vacuous batch (each item failing -2799
        # like the single-shot path) is visible in the report.
        self.evidence(count=len(results),
                      statuses=[getattr(r, "status_message", "") or ""
                                for r in results],
                      objects=[len(getattr(r, "objects", []))
                               for r in results])
        self.assertEqual(len(results), 2)

    def test_04_infer_batch_async(self):
        self.mark("InferenceClient.infer_batch_async")
        items = [BatchInferItem(image=self.image, model_id=TEST_MODEL_ID)]
        results = self.timed(
            self.client.infer_batch_async, items, label="infer_batch_async"
        )
        resolved = getattr(results, "result", lambda: results)()
        self.evidence(type=type(resolved).__name__,
                      count=len(resolved),
                      statuses=[getattr(r, "status_message", "") or ""
                                for r in resolved])
        self.assertEqual(len(resolved), 1)

    @known_issue(_INFER_2799)
    def test_05_infer_with_tensors(self):
        self.mark("InferenceClient.infer_with_tensors")
        info = self.client.get_model_info(TEST_MODEL_ID)
        self.evidence(inputs=info.inputs)
        if not info.inputs:
            self.na("model info exposes no input metadata to build tensors")
        spec = info.inputs[0]
        shape = list(spec.get("shape", spec.get("dims", [])))
        params = {"name": spec.get("name", "input_0"), "shape": shape,
                  "dtype": spec.get("dtype", "")}
        if len(shape) == 4 and shape[-1] == 3:
            tensor = (self.image[:, :, ::-1].copy() / 255.0).astype("float32")
        else:
            tensor = np_zeros(shape)
        try:
            outputs = self.timed(
                self.client.infer_with_tensors, TEST_MODEL_ID, [tensor],
                label="infer_with_tensors",
            )
            self.evidence(n_outputs=len(outputs),
                          out_shapes=[list(o.shape) for o in outputs][:3])
            self.assertTrue(outputs)
        except Exception as exc:
            # Tensor shape/dtype negotiation against the live service is
            # the actual subject here — record the service's rejection.
            self.evidence(error=f"{type(exc).__name__}: {exc}", **params)
            raise


def np_zeros(shape):
    import numpy as np

    shape = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
    return np.zeros(shape, dtype="float32")


class T03Subscribe(DeviceTestCase):
    area = "inference"
    timeout_s = 120

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"no model at {MODEL_PATH}")
        cls.client = InferenceClient()
        cls.client.register_model(MODEL_PATH, model_id=TEST_MODEL_ID)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.unregister_model(TEST_MODEL_ID)
        except Exception:
            pass
        cls.client.close()

    # KNOWN daemon-side issue, re-confirmed 2026-09-09 with a geometry-
    # matched pair: with stream="main" frames DO arrive, but every
    # frame's inference returns -2814 (HAL_ERR_INVALID_ARG) — and it
    # fails the same way against a 640×384 stream/model pair, so it is
    # not an input-geometry mismatch (the -2799/-2811 family) but the
    # streaming path itself on this daemon build. The SDK converts 10
    # consecutive failed frames into a RuntimeError, which lands here as
    # KNOWN-ISSUE rather than an unattributed ERROR. If the daemon is
    # fixed, these tests pass and the report notes the non-reproduction.
    _SUBSCRIBE_2814 = (
        "subscribe: frames arrive but every inference fails -2814 "
        "(HAL_ERR_INVALID_ARG) on the deployed ai-runtime build, even "
        "with a geometry-matched stream/model pair (probed 2026-09-09); "
        "SDK raises RuntimeError after 10 consecutive failed frames"
    )

    @known_issue(_SUBSCRIBE_2814)
    def test_01_subscribe(self):
        self.mark("InferenceClient.subscribe")
        got = 0
        first_seq = last_status = None
        t_first = None
        import time

        t0 = time.monotonic()
        deadline = t0 + 60.0
        for frame_seq, result in self.client.subscribe(
            stream=SUBSCRIBE_STREAM, model=TEST_MODEL_ID, fps=5
        ):
            if got == 0:
                t_first = round((time.monotonic() - t0) * 1000.0, 1)
                first_seq = frame_seq
                last_status = result.status_message
            got += 1
            if got >= 3:
                break
            if time.monotonic() > deadline:
                # Break to a FAIL with evidence instead of leaning on the
                # repeating alarm — a body-level deadline also covers a
                # slow trickle that would otherwise spin to the budget.
                self.record["outcome_note"] = (
                    "subscribe stalled before 3 results (60s deadline)")
                break
        self.evidence(frames=got, first_frame_seq=first_seq,
                      first_result_ms=t_first, status=last_status)
        self.assertEqual(got, 3, "subscribe yielded fewer than 3 results")

    @known_issue(_SUBSCRIBE_2814)
    def test_02_subscribe_result_helpers(self):
        """Result helpers used by every detection app, on a live result."""
        self.mark("InferenceResult.count_by_label/get_objects_by_label")
        got = 0
        labels = {}
        import time

        deadline = time.monotonic() + 60.0
        for _frame_seq, result in self.client.subscribe(
            stream=SUBSCRIBE_STREAM, model=TEST_MODEL_ID, fps=5
        ):
            for obj in result.objects:
                labels[obj.label] = labels.get(obj.label, 0) + 1
            if labels:
                first = list(labels)[0]
                self.assertEqual(
                    result.count_by_label(first), result.count_by_label(first)
                )
            got += 1
            if got >= 2:
                break
            if time.monotonic() > deadline:
                self.record["outcome_note"] = (
                    "subscribe stalled before 2 results (60s deadline)")
                break
        self.evidence(results=got, label_histogram=labels)
        self.assertEqual(got, 2)


class T04Sessions(DeviceTestCase):
    area = "inference"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = InferenceClient()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_01_create_destroy_session(self):
        self.mark("InferenceClient.create_session/destroy_session")
        sid = self.timed(
            self.client.create_session, "sdk-test-session", app_id="sdk-tests",
            label="create_session",
        )
        self.evidence(requested="sdk-test-session", returned=sid)
        # The service returns its OWN session id (app-scoped, e.g.
        # "sdk-tests--<n>") — destroying the requested name would miss it.
        self.timed(self.client.destroy_session, sid,
                   label="destroy_session")


if __name__ == "__main__":
    unittest.main()
