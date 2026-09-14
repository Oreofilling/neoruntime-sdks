"""One-shot probe: daemon-side per-infer latency fields vs client wall time.

Mirrors test_60's input path (NV12 at the model's native geometry, model id
file-derived). Prints percentiles for: client wall ms, response.infer_time_us,
queue_time_us, hw_infer_time_us.
"""
import time

from neoruntime_ipc_sdk import FdMediaClient, InferenceClient
from perf_common import model_input_geometry, perf_model_id, pick_model

MODEL_PATH = pick_model("hailo_yolov8n_384_640.hef",
                        "hailo_yolov8n_384_640.hef")
TEST_MODEL_ID = perf_model_id(MODEL_PATH)

media = FdMediaClient()
frame = media.get_frame("third", timeout_ms=5000)  # 640x384 native == geometry
nv12 = frame.to_array()
frame.release()
media.close()
geom = model_input_geometry(MODEL_PATH)
print(f"model={TEST_MODEL_ID} geom={geom} input={nv12.shape} dtype={nv12.dtype}")

cli = InferenceClient()
cli.connect()
cli.register_model(MODEL_PATH, model_id=TEST_MODEL_ID)

wall, infer_us, queue_us, hw_us = [], [], [], []
r = cli.infer(nv12, TEST_MODEL_ID)  # warmup
print(f"warmup ok: {r.status_message or 'ok'} objects={len(r.objects)} "
      f"infer_us={r.infer_time_us} queue_us={r.queue_time_us} hw_us={r.hw_infer_time_us}")
for _ in range(30):
    t0 = time.perf_counter_ns()
    r = cli.infer(nv12, TEST_MODEL_ID)
    wall.append((time.perf_counter_ns() - t0) / 1e6)
    infer_us.append(r.infer_time_us)
    queue_us.append(r.queue_time_us)
    hw_us.append(r.hw_infer_time_us)


def pct(v, p):
    s = sorted(v)
    return round(s[min(len(s) - 1, int(len(s) * p))], 2)


for name, v in (("client_wall_ms", wall), ("daemon_infer_us", infer_us),
                ("daemon_queue_us", queue_us), ("hw_npu_us", hw_us)):
    print(f"{name:16s} p50={pct(v, 0.5):>10} p99={pct(v, 0.99):>10} "
          f"min={min(v):>10} max={max(v):>10}")

# batch=4 same fields if the API exposes per-batch stats
from neoruntime_ipc_sdk import BatchInferItem  # noqa: E402

items = [BatchInferItem(image=nv12, model_id=TEST_MODEL_ID) for _ in range(4)]
bw, bi = [], []
for _ in range(15):
    t0 = time.perf_counter_ns()
    rs = cli.infer_batch(items)
    wall_ms = (time.perf_counter_ns() - t0) / 1e6
    bw.append(wall_ms)
    bi.append([x.infer_time_us for x in rs if x is not None])
print(f"batch4 client_wall_ms p50={pct(bw, 0.5)}  "
      f"daemon_infer_us(per-item p50 of item0)="
      f"{pct([x[0] for x in bi], 0.5)}")
cli.close()
