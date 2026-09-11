"""PipelineRunner: the long-running app skeleton (priority-1 framework)."""

import threading
import time

import pytest

from neoruntime_ipc_sdk import PipelineRunner


class FakeFrame:
    def __init__(self, seq):
        self.seq = seq
        self.released = False

    def release(self):
        self.released = True

    def __repr__(self):
        return f"FakeFrame({self.seq})"


class FakePipeline:
    def __init__(self, fail=False):
        self.fail = fail
        self.seen = []

    def run(self, frame):
        if self.fail:
            raise RuntimeError(f"infer exploded on {frame.seq}")
        self.seen.append(frame.seq)
        return {"frame": frame.seq}


def wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestLifecycle:
    def test_rejects_bad_config(self):
        with pytest.raises(ValueError, match="drop_policy"):
            PipelineRunner(source=[], pipeline=FakePipeline(), drop_policy="oldest")
        with pytest.raises(ValueError, match="queue_size"):
            PipelineRunner(source=[], pipeline=FakePipeline(), queue_size=0)

    def test_double_start_rejected(self):
        runner = PipelineRunner(source=iter([]), pipeline=FakePipeline())
        runner.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                runner.start()
        finally:
            runner.stop()

    def test_source_exhaustion_stops_runner(self):
        frames = [FakeFrame(i) for i in range(5)]
        got = []

        runner = PipelineRunner(
            source=iter(frames), pipeline=FakePipeline(),
            sink=lambda out, frame: got.append(out["frame"]),
        )
        runner.start()
        assert wait_until(lambda: not runner.running)
        assert runner.stats["processed"] == 5
        assert got == [0, 1, 2, 3, 4]
        assert all(f.released for f in frames)  # consumed frames released

    def test_stop_is_graceful(self):
        stop_source = threading.Event()

        def endless():
            while not stop_source.is_set():
                yield FakeFrame(0)
                time.sleep(0.01)

        runner = PipelineRunner(source=endless(), pipeline=FakePipeline())
        with runner:  # context manager starts and stops
            assert runner.running
            time.sleep(0.1)
        assert not runner.running
        stop_source.set()


class TestLatestWins:
    def test_slow_consumer_drops_old_frames(self):
        frames = [FakeFrame(i) for i in range(3)]
        allow_sink = threading.Event()
        got = []

        def sink(out, frame):
            allow_sink.wait(timeout=2)  # worker parks here → backlog builds
            got.append(out["frame"])

        runner = PipelineRunner(
            source=iter(frames), pipeline=FakePipeline(), sink=sink, queue_size=1,
        )
        runner.start()
        try:
            # Frame 0 reached the worker before the backlog; frame 1 is shed
            # after the grace period; frame 2 (the newest) is queued.
            assert wait_until(lambda: runner.dropped == 1)
            allow_sink.set()
            assert wait_until(lambda: not runner.running)
        finally:
            allow_sink.set()
            runner.stop()

        assert got == [0, 2]                       # newest frame after the shed
        assert runner.stats["dropped"] == 1
        assert all(f.released for f in frames)     # shed + consumed all released

    def test_fast_consumer_never_drops(self):
        # Grace period: a worker that keeps up must not lose frames to the
        # reader/worker race on a size-1 queue.
        frames = [FakeFrame(i) for i in range(50)]
        runner = PipelineRunner(source=iter(frames), pipeline=FakePipeline())
        runner.start()
        assert wait_until(lambda: not runner.running)

        assert runner.stats["dropped"] == 0
        assert runner.stats["processed"] == 50

    def test_sinkless_mode_collects_bounded_results(self):
        runner = PipelineRunner(
            source=iter([FakeFrame(i) for i in range(20)]),
            pipeline=FakePipeline(),
            queue_size=4,
        )
        runner.start()
        assert wait_until(lambda: not runner.running)
        assert len(runner.last_results) <= 16
        assert runner.stats["processed"] == 20


class TestErrorPolicy:
    def test_consecutive_errors_stop_the_runner(self):
        pipeline = FakePipeline(fail=True)
        runner = PipelineRunner(
            source=iter([FakeFrame(i) for i in range(50)]),
            pipeline=pipeline, max_consecutive_errors=3,
        )
        runner.start()
        assert wait_until(lambda: not runner.running)

        stats = runner.stats
        assert stats["consecutive_errors"] == 3
        assert "infer exploded" in stats["last_error"]

    def test_success_resets_error_counter(self):
        class Flaky:
            def __init__(self):
                self.calls = 0

            def run(self, frame):
                self.calls += 1
                if self.calls % 2:  # fail every other call
                    raise RuntimeError("transient")
                return "ok"

        errors = []
        runner = PipelineRunner(
            source=iter([FakeFrame(i) for i in range(10)]),
            pipeline=Flaky(), max_consecutive_errors=3,
            on_error=errors.append,
        )
        runner.start()
        assert wait_until(lambda: not runner.running)

        assert runner.stats["processed"] == 5
        assert runner.stats["consecutive_errors"] == 0  # reset by successes
        assert len(errors) == 5
