"""A/B probe: infer bytes-input vs keep-fd buffer_id input (same model).

Quantifies the pixel-transport bundle inside the client e2e: the bytes path
ships 360KB through protobuf/gRPC; the fd path imports the dma-buf and ships
only a buffer id. Prints p50/p99 for both paths' client wall ms and the
daemon-side infer_time_us each reports.
"""
import time

from neoruntime_ipc_sdk import FdMediaClient, InferenceClient
from perf_common import perf_model_id, pick_model

MODEL_PATH = pick_model("hailo_yolov8n_384_640.hef")
TEST_MODEL_ID = perf_model_id(MODEL_PATH)

media = FdMediaClient()
fh = media.get_frame("third", timeout_ms=5000, keep_fd=True)  # 640x384 handle
arr = fh.to_array()  # same pixels as ndarray (bytes-path input)
print(f"handle_type={type(fh).__name__} arr={arr.shape} "
      f"payload_bytes={arr.nbytes}")

cli = InferenceClient()
cli.connect()
cli.register_model(MODEL_PATH, model_id=TEST_MODEL_ID)


def pct(v, p):
    s = sorted(v)
    return round(s[min(len(s) - 1, int(len(s) * p))], 2)


def run(label, image):
    wall, daemon = [], []
    r = cli.infer(image, TEST_MODEL_ID)  # warmup
    assert r.status_message == "" or r.objects is not None, r.status_message
    for _ in range(40):
        t0 = time.perf_counter_ns()
        r = cli.infer(image, TEST_MODEL_ID)
        wall.append((time.perf_counter_ns() - t0) / 1e6)
        daemon.append(r.infer_time_us / 1000.0)
    print(f"{label:10s} client_wall_ms p50={pct(wall, .5):>7} "
          f"p99={pct(wall, .99):>7} | daemon_infer_ms p50={pct(daemon, .5):>6} "
          f"p99={pct(daemon, .99):>6}")
    return wall


w_bytes = run("bytes", arr)
w_fd = run("fd", fh)
fh.release()
media.close()

d = [a - b for a, b in zip(sorted(w_bytes), sorted(w_fd))]
print(f"delta(bytes-fd) p50={pct(d, .5)}ms p99={pct(d, .99)}ms "
      f"(sorted-pair diff; same-n anchors)")
cli.close()
