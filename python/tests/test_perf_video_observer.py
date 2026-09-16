"""Offline tests for the external RTSP observation process."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "perf_demo"
sys.path.insert(0, str(DEMO))
import video_observer as observer  # noqa: E402 - standalone example module


class FakeCapture:
    def __init__(self, results):
        self.results = iter(results)
        self.released = False

    def read(self):
        return next(self.results)

    def get(self, _prop):
        return 100.0

    def release(self):
        self.released = True


class FakeFrame:
    shape = (720, 1280, 3)


def test_samples_separate_read_duration_and_delivery_gap():
    cap = FakeCapture([(True, FakeFrame()), (True, FakeFrame())])
    times = iter([0.0, 0.01, 0.04, 0.05, 0.08, 1.1])
    rows = list(observer.frame_samples(cap, 1.0, lambda: next(times), lambda: False))
    assert len(rows) == 2
    assert rows[0]["decode_wall_ms"] == pytest.approx(30.0)
    assert rows[0]["delivery_gap_ms"] is None
    assert rows[1]["delivery_gap_ms"] == pytest.approx(40.0)
    assert rows[0]["source_pts_ms"] == 100.0
    assert rows[1]["frame_index"] == 2


def test_decode_failure_is_not_silently_retried():
    cap = FakeCapture([(False, None)])
    times = iter([0.0, 0.01, 0.04])
    with pytest.raises(RuntimeError, match="decode failed"):
        list(observer.frame_samples(cap, 1.0, lambda: next(times), lambda: False))


def test_stop_without_frames_is_supported():
    cap = FakeCapture([])
    assert list(observer.frame_samples(cap, 1.0, lambda: 0.0, lambda: True)) == []


def test_json_writer_enforces_limit_and_does_not_log_url(tmp_path):
    path = tmp_path / "samples.jsonl"
    with path.open("w+") as stream:
        writer = observer.SampleWriter(stream, "main", "sample-run", max_bytes=1000)
        writer.write({"type": "frame", "frame_index": 1})
        stream.flush()
        row = json.loads(path.read_text())
        assert row["stream"] == "main"
        assert row["run_id"] == "sample-run"
        assert "url" not in row
        with pytest.raises(RuntimeError, match="size limit"):
            writer.write({"payload": "x" * 2000})


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_nonpositive_or_nonfinite_durations_rejected(value):
    with pytest.raises(SystemExit):
        observer.parse_args(["--output", "unused", "--duration", value])


def test_missing_url_does_not_create_output(tmp_path, monkeypatch):
    path = tmp_path / "samples.jsonl"
    monkeypatch.delenv("PERF_RTSP_URL", raising=False)
    assert observer.main(["--output", str(path), "--duration", "1"]) == 2
    assert not path.exists()


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "samples.jsonl"
    path.write_text("preserve")
    monkeypatch.setenv("PERF_RTSP_URL", "rtsp://example.invalid/sub")
    assert observer.main(["--output", str(path), "--duration", "1"]) == 2
    assert path.read_text() == "preserve"


@pytest.mark.parametrize("opened", [True, False])
def test_capture_released_on_success_and_open_failure(tmp_path, monkeypatch, opened):
    from types import SimpleNamespace

    cap = FakeCapture([])
    cap.isOpened = lambda: opened
    cap.getBackendName = lambda: "FAKE"
    calls = []

    def open_capture(*args):
        calls.append(args)
        return cap

    fake_cv = SimpleNamespace(
        setNumThreads=lambda _: None, CAP_FFMPEG=1,
        CAP_PROP_OPEN_TIMEOUT_MSEC=2, CAP_PROP_READ_TIMEOUT_MSEC=3,
        CAP_PROP_N_THREADS=4, __version__="test",
        VideoCapture=open_capture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv)
    monkeypatch.setattr(observer, "frame_samples", lambda *_a, **_k: iter([
        {"type": "frame", "frame_index": 1, "monotonic_ns": 123},
    ]))
    monkeypatch.setenv("PERF_RTSP_URL", "rtsp://example.invalid/sub")
    path = tmp_path / "capture.jsonl"
    assert observer.main(["--output", str(path), "--duration", "1"]) == (0 if opened else 1)
    assert cap.released
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]["type"] == ("final" if opened else "error")
    assert "example.invalid" not in path.read_text()
    assert calls[0][2][-2:] == [fake_cv.CAP_PROP_N_THREADS, 2]
    assert all(row["utc_time"].endswith("+00:00") for row in rows)
    if opened:
        assert rows[0]["decoder_threads"] == 100.0
        assert rows[0]["opencv_version"] == "test"


def test_observe_flushes_host_cost_and_final_record(tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr(observer, "frame_samples", lambda *_a, **_k: iter([
        {"type": "frame", "frame_index": 1},
    ]))
    times = iter([0.0, 2.0])
    monkeypatch.setattr(observer.time, "monotonic", lambda: next(times))
    path = tmp_path / "observe.jsonl"
    with path.open("w") as stream:
        count = observer.observe(None, observer.SampleWriter(stream, "main", "run"),
                                 1, threading.Event())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert count == 1
    assert [row["type"] for row in rows] == ["frame", "host_resource", "final"]
    assert rows[-1]["reason"] == "duration"


def test_observe_stops_when_disk_reserve_exhausted(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    monkeypatch.setattr(observer, "frame_samples", lambda *_a, **_k: iter([
        {"type": "frame", "frame_index": 1},
    ]))
    times = iter([0.0, 2.0])
    monkeypatch.setattr(observer.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(observer.os, "fstatvfs", lambda _: SimpleNamespace(f_bavail=0, f_frsize=4096))
    with (tmp_path / "full.jsonl").open("w") as stream:
        with pytest.raises(RuntimeError, match="disk reserve"):
            observer.observe(None, observer.SampleWriter(stream, "sub", "run"),
                             1, threading.Event())


def test_signal_handler_sets_event_and_restores_previous_handler():
    import signal

    stop = observer.StopFlag()
    previous = signal.getsignal(signal.SIGTERM)
    with observer.stop_signals(stop):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert stop.is_set()
    assert signal.getsignal(signal.SIGTERM) is previous


def test_writer_preserves_disk_reserve_before_first_write(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(observer.os, "fstatvfs", lambda _: SimpleNamespace(
        f_bavail=observer.MIN_FREE_BYTES, f_frsize=1))
    with (tmp_path / "reserve.jsonl").open("w") as stream:
        writer = observer.SampleWriter(stream, "sub", "run")
        with pytest.raises(RuntimeError, match="disk reserve"):
            writer.write({"type": "start"})
        assert stream.tell() == 0


def test_stop_flag_can_be_set_repeatedly_without_locking():
    stop = observer.StopFlag()
    assert not stop.is_set()
    with observer.stop_signals(stop):
        handler = observer.signal.getsignal(observer.signal.SIGTERM)
        handler(observer.signal.SIGTERM, None)
        handler(observer.signal.SIGTERM, None)
    assert stop.is_set()


def test_new_output_is_private(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_RTSP_URL", "rtsp://example.invalid/sub")
    monkeypatch.setattr(observer, "run_capture", lambda *_args: 0)
    path = tmp_path / "private.jsonl"
    old_umask = observer.os.umask(0o022)
    try:
        assert observer.main(["--output", str(path), "--duration", "1"]) == 0
    finally:
        observer.os.umask(old_umask)
    assert path.stat().st_mode & 0o777 == 0o600


def test_small_output_budget_rejected():
    with pytest.raises(SystemExit):
        observer.parse_args(["--output", "unused", "--duration", "1", "--max-bytes", "1"])
