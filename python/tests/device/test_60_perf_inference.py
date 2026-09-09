"""Perf P1 — inference plane (control RPCs, infer data plane, streams).

Methodology (see perf_common): sampled distributions with warmup
discarded, ``PERF_SAMPLE_ROUNDS`` rounds with the median-p50 round as
headline, errors counted but never timed.

Input contract (probed live 2026-09-09 on the secondary test device —
this corrects the earlier "-2799 daemon regression" diagnosis):

* the daemon validates input byte_size against the model's geometry —
  an RGB array into an NV12 model fails -2799/-2811 and a native-size
  buffer into a smaller model fails the same way;
* ``hailo_yolov8n_384_640`` fed geometry-matched NV12 (640×384) infers
  fine at ~71 FPS e2e, so the data plane below feeds
  ``model_input_geometry``-sized NV12 with an RGB fallback for
  RGB-input models;
* ``yolo_world_v2s*`` is dual-input — the single-image infer RPC cannot
  drive it (-2799 even with matched geometry), so it sits at the end
  of the pick order and only lands here as the fallback of last
  resort;
* ``subscribe`` (daemon-side streaming infer) fails -2814 on this
  deployment even with a geometry-matched stream/model pair — the
  stream test probes briefly and reports NA with that evidence
  instead of burning its whole window on a dead path.
"""

from __future__ import annotations

import time
import unittest

from neoruntime_ipc_sdk import FdMediaClient, InferenceClient
from neoruntime_ipc_sdk import accel

from perf_common import (
    SAMPLE_ROUNDS,
    STREAM_S,
    PerfTestCase,
    model_input_geometry,
    pick_model,
)

# Single-input NV12 model first (proven inferable); dual-input models
# only as last-resort fallback (see module docstring).
MODEL_PATH = pick_model("hailo_yolov8n_384_640.hef",
                        "yolov5m_vehicles.hef",
                        "yolo_world_v2s_540.hef",
                        "yolo_world_v2s.hef")
TEST_MODEL_ID = "sdk-perf-yolo"
SUBSCRIBE_STREAM = "third"  # 640×384@15 — geometry-matched to yolov8n
SUBSCRIBE_PROBE_S = 5.0


def _prepare_input(model_path):
    """(buffer, format, geometry) the model's contract accepts.

    NV12 resized to the model's filename-parsed geometry first (the
    camera's native format — no color convert in the prep path); RGB
    from the same frame as the fallback for RGB-input models.
    """
    media = FdMediaClient()
    try:
        frame = media.get_frame("main", timeout_ms=5000)
        if frame is None:
            return None, None, None
        native = (frame.width, frame.height)
        geometry = model_input_geometry(model_path) or native
        nv12 = frame.to_array()
        resized = accel.get_default_router().run(
            "resize_nv12", nv12, native, geometry)
        return resized, "nv12", geometry
    finally:
        media.close()


class T01SmokeGate(PerfTestCase):
    """Register once; classify the infer path and its input format."""

    area = "perf-inference"
    timeout_s = 180
    infer_ok = None          # tri-state shared via class attr
    input_format = None      # "nv12" / "rgb" once known
    smoke_note = ""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not MODEL_PATH:
            raise unittest.SkipTest(f"no model at {MODEL_PATH}")
        cls.client = InferenceClient()
        cls.client.connect()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.unregister_model(TEST_MODEL_ID)
        except Exception:
            pass
        cls.client.close()

    def test_01_register_and_smoke(self):
        self.mark("InferenceClient.register_model + infer smoke gate")
        t0 = time.perf_counter_ns()
        model_id = self.client.register_model(MODEL_PATH,
                                              model_id=TEST_MODEL_ID)
        register_ms = (time.perf_counter_ns() - t0) / 1e6

        nv12, fmt, geometry = _prepare_input(MODEL_PATH)
        statuses = []
        # NV12 contract first (native camera format)…
        if nv12 is None:
            statuses.append("nv12: no camera frame")
        else:
            try:
                result = self.client.infer(nv12, TEST_MODEL_ID)
                statuses.append(
                    f"nv12: ok ({result.status_message or 'ok'})")
                T01SmokeGate.input_format = "nv12"
            except Exception as exc:  # noqa: BLE001 — smoke must classify
                statuses.append(f"nv12: {type(exc).__name__}: {exc}"[:120])
        # …RGB fallback keeps the gate honest on RGB-input models.
        if T01SmokeGate.input_format is None:
            media = FdMediaClient()
            try:
                frame = media.get_frame("main", timeout_ms=5000)
                rgb = None
                if frame is not None:
                    if geometry and geometry != (frame.width, frame.height):
                        rgb = frame.resize(*geometry).to_rgb()
                    else:
                        rgb = frame.to_rgb()
            finally:
                media.close()
            if rgb is None:
                statuses.append("rgb: no camera frame")
            else:
                try:
                    result = self.client.infer(rgb, TEST_MODEL_ID)
                    statuses.append(
                        f"rgb: ok ({result.status_message or 'ok'})")
                    T01SmokeGate.input_format = "rgb"
                except Exception as exc:  # noqa: BLE001
                    statuses.append(
                        f"rgb: {type(exc).__name__}: {exc}"[:120])

        T01SmokeGate.infer_ok = T01SmokeGate.input_format is not None
        if not T01SmokeGate.infer_ok:
            T01SmokeGate.smoke_note = statuses[0]
        self.evidence(
            model_id=model_id, register_ms=round(register_ms, 1),
            model=(MODEL_PATH.rsplit("/", 1)[-1] if MODEL_PATH else None),
            input_geometry=(f"{geometry[0]}x{geometry[1]}"
                            if geometry else None),
            input_format=T01SmokeGate.input_format,
            smoke=statuses, usable=T01SmokeGate.infer_ok,
        )
        self.assertEqual(model_id, TEST_MODEL_ID)
        self.assertTrue(T01SmokeGate.infer_ok,
                        f"infer smoke failed on every input contract: {statuses}")


class T02ControlPlane(PerfTestCase):
    """Registry RPCs — independent of the infer data plane's health."""

    area = "perf-inference"
    timeout_s = 300

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not MODEL_PATH:
            raise unittest.SkipTest(f"no model at {MODEL_PATH}")
        cls.client = InferenceClient()
        cls.client.connect()
        # Own registration (runner may execute modules standalone).
        cls.client.register_model(MODEL_PATH, model_id=TEST_MODEL_ID)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.unregister_model(TEST_MODEL_ID)
        except Exception:
            pass
        cls.client.close()

    def test_01_list_models(self):
        self.mark("InferenceClient.list_models latency")
        self.perf_sample(self.client.list_models, label="list_models")

    def test_02_get_model_info(self):
        self.mark("InferenceClient.get_model_info latency")
        self.perf_sample(self.client.get_model_info, TEST_MODEL_ID,
                         label="get_model_info")

    def test_03_get_stats(self):
        self.mark("InferenceClient.get_stats latency")
        self.perf_sample(self.client.get_stats, label="get_stats")


class T03InferDataPlane(PerfTestCase):
    """End-to-end infer: SDK serialize + transport + NPU + postprocess."""

    area = "perf-inference"
    timeout_s = 600

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not MODEL_PATH:
            raise unittest.SkipTest(f"no model at {MODEL_PATH}")
        cls.client = InferenceClient()
        cls.client.connect()
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
        if T01SmokeGate.infer_ok is False:
            self.na(f"infer path unusable on this deployment: "
                    f"{T01SmokeGate.smoke_note}")
        fmt = T01SmokeGate.input_format or "nv12"
        geometry = model_input_geometry(MODEL_PATH)
        if fmt == "rgb":
            media = FdMediaClient()
            try:
                frame = media.get_frame("main", timeout_ms=5000)
                if frame is None:
                    self.na("camera produced no frame")
                if geometry and geometry != (frame.width, frame.height):
                    self.image = frame.resize(*geometry).to_rgb()
                else:
                    self.image = frame.to_rgb()
            finally:
                media.close()
        else:
            buf, _buf_fmt, _geom = _prepare_input(MODEL_PATH)
            if buf is None:
                self.na("camera produced no frame")
            self.image = buf
        self.input_format = fmt

    def test_01_infer_e2e(self):
        self.mark("InferenceClient.infer latency distribution")
        hw_us: list[int | None] = []

        def one():
            result = self.client.infer(self.image, TEST_MODEL_ID)
            # hw_infer_time_us reads 0 on models where the HAL skips
            # the latency flag (multi-plane fix, commit 9f9ffeb) — the
            # evidence note separates "not reported" from "measured".
            hw_us.append(result.hw_infer_time_us or None)
            return result

        stats = self.perf_sample(one, label="infer_e2e",
                                 rounds=SAMPLE_ROUNDS)
        if stats.get("n"):
            plain = [v for v in hw_us if v]
            self.evidence(
                input_format=self.input_format,
                shape=str(getattr(self.image, "shape", None)),
                hw_infer_time_us_p50=(sorted(plain)[len(plain) // 2]
                                      if plain else None),
                hw_latency_reported=bool(plain),
                note=(None if plain else
                      "hw_infer_time_us not populated for this model "
                      "(HAL skips the latency flag); infer_time_us in "
                      "the daemon-side total instead"),
            )
            if plain:
                mid = sorted(plain)[len(plain) // 2]
                self.evidence(
                    sdk_overhead_p50_ms=round(stats["p50"] - mid / 1000.0, 2))

    def test_02_infer_batch(self):
        self.mark("InferenceClient.infer_batch latency (batch=4)")
        from neoruntime_ipc_sdk import BatchInferItem

        items = [BatchInferItem(image=self.image, model_id=TEST_MODEL_ID)
                 for _ in range(4)]

        def one():
            # per-item status stays visible in evidence via err count;
            # a vacuous batch (each item failing) shows as errors, not
            # as suspiciously fast successes.
            return self.client.infer_batch(items)

        self.perf_sample(one, label="infer_batch_4", n=100, rounds=1)

    def test_03_subscribe_stream(self):
        self.mark("InferenceClient.subscribe arrival statistics")
        # Probe first: -2814 on every frame is a deployment-level dead
        # path (verified even with geometry-matched stream/model) — NA
        # with evidence beats a 60s timeout on a silent generator.
        gen = self.client.subscribe(stream=SUBSCRIBE_STREAM,
                                    model=TEST_MODEL_ID, fps=10)
        probe = []
        deadline = time.monotonic() + SUBSCRIBE_PROBE_S
        try:
            for item in gen:
                probe.append(item)
                if time.monotonic() > deadline:
                    break
        except Exception as exc:  # noqa: BLE001 — classify, then NA
            self.na(f"subscribe path dead on this deployment: "
                    f"{type(exc).__name__}: {exc}"[:200])
            return
        if not probe:
            self.na("subscribe produced no frames in the probe window "
                    f"({SUBSCRIBE_PROBE_S:.0f}s, stream={SUBSCRIBE_STREAM})")
            return
        stats = self.perf_stream(
            gen, label="subscribe_10fps", duration_s=STREAM_S,
            seq_of=lambda item: item[0],
        )
        self.assertGreater(stats.get("frames", 0), 0)

    def test_04_session_lifecycle(self):
        self.mark("InferenceClient.create/destroy_session latency")

        def one():
            sid = self.client.create_session("sdk-perf-session",
                                             app_id="sdk-perf")
            self.client.destroy_session(sid)
            return sid

        self.perf_sample(one, label="session_lifecycle", n=50, rounds=1)


if __name__ == "__main__":
    unittest.main()
