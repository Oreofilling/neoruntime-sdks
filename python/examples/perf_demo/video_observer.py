"""External RTSP decode observer; run on the host, never on the camera.

Set PERF_RTSP_URL in the environment. Samples contain no URL or pixels.
Delivery timestamps are host-side decode completions, not camera capture time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import urlsplit

DEFAULT_MAX_BYTES = 256 * 1024 * 1024
MIN_FREE_BYTES = 64 * 1024 * 1024
# OpenCV's stable VideoCapture property id; cv2 is optional until live capture.
POSITION_MSEC = 0


def positive_duration(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 86400:
        raise argparse.ArgumentTypeError("duration must be finite and in (0, 86400]")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=positive_duration, required=True)
    parser.add_argument("--stream", choices=("main", "sub", "third"), default="sub")
    parser.add_argument("--run-id", default="run")
    parser.add_argument("--url-env", default="PERF_RTSP_URL")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)
    if args.max_bytes < 1024:
        parser.error("max-bytes must be at least 1024")
    return args


class SampleWriter:
    def __init__(self, stream, label, run_id, max_bytes=DEFAULT_MAX_BYTES):
        self.stream = stream
        self.label = label
        self.run_id = run_id
        self.max_bytes = max_bytes

    def write(self, row):
        data = {**row, "stream": self.label, "run_id": self.run_id,
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "write_monotonic_ns": time.monotonic_ns()}
        line = json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
        if self.stream.tell() + len(line.encode("utf-8")) > self.max_bytes:
            raise RuntimeError("video observation size limit reached")
        stats = os.fstatvfs(self.stream.fileno())
        pending = self.stream.tell() - os.fstat(self.stream.fileno()).st_size
        available = stats.f_bavail * stats.f_frsize
        if available - max(0, pending) - len(line.encode("utf-8")) < MIN_FREE_BYTES:
            raise RuntimeError("video observation disk reserve reached")
        self.stream.write(line)


def frame_samples(capture, duration, clock=time.monotonic, stopped=lambda: False):
    start = clock()
    previous = None
    frame_index = 0
    while not stopped():
        before = clock()
        if before - start >= duration:
            break
        ok, frame = capture.read()
        after = clock()
        if not ok or frame is None:
            raise RuntimeError("RTSP decode failed")
        frame_index += 1
        pts = capture.get(POSITION_MSEC)
        yield {
            "type": "frame", "frame_index": frame_index,
            "monotonic_ns": int(after * 1e9),
            "elapsed_s": after - start,
            "decode_wall_ms": (after - before) * 1000.0,
            "delivery_gap_ms": None if previous is None else (after - previous) * 1000.0,
            "source_pts_ms": pts if math.isfinite(pts) else None,
            "width": frame.shape[1], "height": frame.shape[0],
        }
        previous = after


class StopFlag:
    """Signal-safe stop request: setting it never acquires a thread lock."""

    def __init__(self):
        self.requested = False

    def set(self):
        self.requested = True

    def is_set(self):
        return self.requested


@contextmanager
def stop_signals(event):
    def request_stop(_signum, _frame):
        event.set()

    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, request_stop)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def observe(capture, writer, duration, stop):
    last_flush = time.monotonic()
    cpu_start = time.process_time()
    count = 0
    for row in frame_samples(capture, duration, stopped=stop.is_set):
        writer.write(row)
        count += 1
        now = time.monotonic()
        if now - last_flush >= 1.0:
            writer.write({"type": "host_resource", "monotonic_ns": time.monotonic_ns(),
                          "pid": os.getpid(), "cpu_seconds": time.process_time() - cpu_start})
            writer.stream.flush()
            last_flush = now
    writer.write({"type": "final", "monotonic_ns": time.monotonic_ns(),
                  "frames": count, "cpu_seconds": time.process_time() - cpu_start,
                  "reason": "signal" if stop.is_set() else "duration", "errors": 0})
    writer.stream.flush()
    return count


def run_capture(url, args, output):
    import cv2

    cv2.setNumThreads(2)
    writer = SampleWriter(output, args.stream, args.run_id, args.max_bytes)
    stop = StopFlag()
    capture = None
    try:
        with stop_signals(stop):
            capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_N_THREADS, 2,
            ])
            if not capture.isOpened():
                raise RuntimeError("RTSP open failed")
            writer.write({"type": "start", "monotonic_ns": time.monotonic_ns(),
                          "pid": os.getpid(), "backend": capture.getBackendName(),
                          "opencv_version": cv2.__version__,
                          "decoder_threads": capture.get(cv2.CAP_PROP_N_THREADS)})
            count = observe(capture, writer, args.duration, stop)
            print(f"VIDEO_DONE stream={args.stream} frames={count}", flush=True)
            return 0
    except Exception as exc:
        try:
            writer.write({"type": "error", "monotonic_ns": time.monotonic_ns(),
                          "error_type": type(exc).__name__})
            output.flush()
        except (OSError, RuntimeError):
            pass  # Primary error is still reported to the supervisor below.
        print(f"VIDEO_FAILED stream={args.stream} error={type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()


def main(argv=None):
    args = parse_args(argv)
    url = os.environ.get(args.url_env, "")
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("rtsp", "rtsps") or not parsed.hostname:
            raise ValueError("RTSP environment variable missing or invalid")
        def private_opener(path, flags):
            return os.open(path, flags, 0o600)

        with open(args.output, "x", encoding="utf-8", opener=private_opener) as output:
            return run_capture(url, args, output)
    except (OSError, ValueError, ImportError) as exc:
        print(f"VIDEO_SETUP_FAILED error={type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
