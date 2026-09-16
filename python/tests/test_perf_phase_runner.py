"""Offline phase-supervisor capability and regression tests; no SDK/device."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "examples/perf_demo"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("tested_phase_runner", DIRECTORY / "phase_runner.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def stat_line(pid, name, start=100):
    fields = ["S", "1"] + ["0"] * 48
    fields[19] = str(start)
    return f"{pid} ({name}) " + " ".join(fields)


@pytest.fixture
def proc(tmp_path):
    root = tmp_path / "proc"
    for pid, name in [(10, "camera-daemon"), (11, "ai-runtime"),
                      (12, "hailort_server"), (13, "isp_media_serve")]:
        put(root / str(pid) / "stat", stat_line(pid, name))
    put(root / "meminfo", "MemTotal: 4194304 kB\nMemAvailable: 1048576 kB\n")
    return root


def event(kind, seconds=1, **fields):
    return {"type": kind, "run_id": "test-run", "phase": "p1",
            "monotonic_ns": int(seconds * 1e9), **fields}


def good_events(seconds=1, frame=1):
    return [event("a_result", seconds, success=True, source_frame_id=frame),
            event("b_infer", seconds, success=True, source_frame_id=frame),
            event("b_compose", seconds, success=True, source_frame_id=frame),
            event("encoded_packet", seconds, stream_id="main", pts_ns=frame),
            event("encoded_packet", seconds, stream_id="sub", pts_ns=frame)]


def terminal_events(seconds=2):
    return [event("run_exit", seconds, exit_code=0, cleanup_errors=[], alive_threads=[]),
            event("recorder_final", seconds, dropped=0, error=None, writer_alive=False, exit_code=0)]


@pytest.mark.parametrize("kept", [True, False, None])
def test_final_model_ownership_is_preserved_without_coercing_unknown(kept):
    monitor = mod.EventMonitor("test-run", "p1", "none", 0)
    evidence = {"model_kept": kept, "model_kept_reason": "reported_reason",
                "model_ownership": "unknown"}
    monitor.consume(good_events() + [event("run_exit", 2, exit_code=0, **evidence),
                                     terminal_events()[1]], 2_000_000_000)
    monitor.finish()
    assert monitor.model_evidence == evidence


def test_watcher_terminal_reconnect_failure_stops_without_waiting_for_stall():
    monitor = mod.EventMonitor("test-run", "p1", "none", 0)
    monitor.consume([event("watcher_reconnect", stream_id="main", reason="no_packet", consecutive_rebuilds=1)], 2_000_000_000)
    with pytest.raises(mod.PhaseFailure, match="watcher_failed:main"):
        monitor.consume([event("watcher_exit", stream_id="main", error="reconnect_limit")], 2_000_000_000)


def test_a_unknown_failure_counts_and_unknown_source_clock_do_not_invent_age():
    monitor = mod.EventMonitor("test-run", "p1", "a", 0)
    row = event("a_result", success=True, source_frame_id=1,
                source_timestamp_ns=9000000000000000000, result_timestamp_clock="unknown")
    status = event("status", snapshot={"a": {"infer_attempts": None, "infer_failures": None,
                                            "per_frame_failures_observable": False},
                                      "recorder": {"dropped": 0, "error": None, "writer_alive": True}})
    monitor.consume([row, status], 2_000_000_000)
    assert monitor.last_success["a_result"] == 1_000_000_000


def test_raw_event_health_all_selected_channels_and_final_records():
    monitor = mod.EventMonitor("test-run", "p1", "ab", 0)
    monitor.consume(good_events() + terminal_events(), 2_000_000_000)
    monitor.check(39_000_000_000)
    monitor.finish()
    assert set(monitor.last_success) == {"a_result", "b_infer", "b_compose", "encoded:main", "encoded:sub"}


@pytest.mark.parametrize("chains,missing", [("a", "a_result"), ("b", "b_infer"),
                                          ("b", "b_compose"), ("none", "encoded:main"),
                                          ("none", "encoded:sub")])
def test_grace_then_ten_seconds_without_success_fails(chains, missing):
    monitor = mod.EventMonitor("test-run", "p1", chains, 0)
    rows = [row for row in good_events(39) if
            ("encoded:" + row["stream_id"] if row["type"] == "encoded_packet" else row["type"]) != missing]
    monitor.consume(rows, 39_000_000_000)
    monitor.check(39_999_999_999)
    with pytest.raises(mod.PhaseFailure, match=missing):
        monitor.check(40_000_000_000)


def test_failed_infer_and_old_frame_replays_do_not_refresh_success():
    monitor = mod.EventMonitor("test-run", "p1", "b", 0)
    monitor.consume(good_events(31), 31_000_000_000)
    monitor.consume(good_events(40), 40_000_000_000)  # same source IDs
    monitor.consume([event("b_infer", 40, success=False, source_frame_id=2)], 40_000_000_000)
    with pytest.raises(mod.PhaseFailure, match="stale:"):
        monitor.check(41_000_000_000)


@pytest.mark.parametrize("record", [None, [], {}, event("a_result", run_id="other"),
                                   event("a_result", phase="other"),
                                   event("a_result", monotonic_ns=-1),
                                   event("a_result", monotonic_ns=True),
                                   event("a_result", 1000),
                                   event("a_result", success="true", source_frame_id=1)])
def test_invalid_event_envelopes_fail_explicitly(record):
    monitor = mod.EventMonitor("test-run", "p1", "a", 0)
    with pytest.raises(mod.PhaseFailure):
        monitor.consume([record], 2_000_000_000)


@pytest.mark.parametrize("record", [
    event("status", snapshot={"recorder": {"dropped": 1, "error": None}}),
    event("status", snapshot={"recorder": {"dropped": 0, "error": "secret-token"}}),
    event("status", snapshot={"recorder": {"dropped": 0, "error": None, "writer_alive": False}}),
    event("status", snapshot={}),
    event("recorder_final", dropped=2, error=None, exit_code=0),
    event("run_exit", exit_code=1),
    event("run_exit", exit_code=0, snapshot="bad"),
    event("run_exit", exit_code=0, alive_threads=["worker"]),
])
def test_recorder_loss_error_and_bad_cleanup_fail_without_leaking_details(record):
    monitor = mod.EventMonitor("test-run", "p1", "none", 0)
    with pytest.raises(mod.PhaseFailure) as exc:
        monitor.consume([record], 2_000_000_000)
    assert "secret-token" not in str(exc.value)


def test_short_phase_cannot_pass_without_real_success_or_final_evidence():
    monitor = mod.EventMonitor("test-run", "p1", "a", 0)
    with pytest.raises(mod.PhaseFailure):
        monitor.finish()
    monitor.consume(good_events(), 2_000_000_000)
    with pytest.raises(mod.PhaseFailure):
        monitor.finish()


def test_event_reader_partial_lines_identity_and_truncation(tmp_path):
    path = tmp_path / "samples.jsonl"
    reader = mod.EventReader(path)
    assert reader.read() == []
    put(path, json.dumps(event("unused")) + "\n" + '{"type":')
    assert len(reader.read()) == 1
    with path.open("a") as handle:
        handle.write('"unused"}\n')
    assert reader.read() == [{"type": "unused"}]
    put(path, "")
    with pytest.raises(mod.PhaseFailure, match="samples_replaced_or_truncated"):
        reader.read()


@pytest.mark.parametrize("payload", [b"not-json\n", b"\xff\n", b"x" * 300000])
def test_reader_invalid_or_unbounded_record_is_failure(tmp_path, payload):
    path = tmp_path / "samples.jsonl"
    path.write_bytes(payload)
    with pytest.raises(mod.PhaseFailure):
        mod.EventReader(path).read()


def test_final_measurement_boundary_cannot_hide_ten_second_stall():
    monitor = mod.EventMonitor("test-run", "p1", "ab", 0, end_ns=45_000_000_000)
    monitor.consume(good_events(35) + good_events(46, frame=2) + terminal_events(47), 47_000_000_000)
    with pytest.raises(mod.PhaseFailure, match="stale:"):
        monitor.finish(steady_end_ns=45_000_000_000)


def test_reader_supports_ten_thousand_records_without_reprocessing(tmp_path):
    path = tmp_path / "samples.jsonl"
    put(path, (json.dumps(event("unused")) + "\n") * 10000)
    reader = mod.EventReader(path)
    assert len(reader.read()) == 10000
    assert reader.read() == []


def test_platform_fingerprint_requires_all_roles_and_detects_reuse(proc):
    baseline = mod.platform_fingerprint(proc)
    assert set(baseline) == {10, 11, 12, 13}
    mod.check_platform(proc, baseline, {10: 100})
    put(proc / "10/stat", stat_line(10, "camera-daemon", 101))
    with pytest.raises(mod.PhaseFailure, match="platform_changed"):
        mod.check_platform(proc, baseline, {})
    (proc / "13/stat").unlink()
    with pytest.raises(mod.PhaseFailure, match="platform_missing"):
        mod.platform_fingerprint(proc)


def test_expected_platform_identity_cannot_silently_rebaseline(proc):
    baseline = mod.platform_fingerprint(proc)
    with pytest.raises(mod.PhaseFailure, match="expected_platform_mismatch"):
        mod.check_platform(proc, baseline, {10: 200})


def test_memory_threshold_consecutive_low_reset_and_large_machine(proc):
    guard = mod.MemoryGuard()
    assert guard.check(proc)["threshold_bytes"] == 512 * 1024 * 1024
    put(proc / "meminfo", "MemTotal: 4194304 kB\nMemAvailable: 1 kB\n")
    guard.check(proc)
    guard.check(proc)
    put(proc / "meminfo", "MemTotal: 4194304 kB\nMemAvailable: 524288 kB\n")
    guard.check(proc)  # equality is not below threshold
    put(proc / "meminfo", "MemTotal: 20000000 kB\nMemAvailable: 600000 kB\n")
    assert guard.check(proc)["threshold_bytes"] == 1024000000
    guard.check(proc)
    with pytest.raises(mod.PhaseFailure, match="low_memory"):
        guard.check(proc)


@pytest.mark.parametrize("text", ["", "MemTotal: nope kB", "MemTotal: 10 kB\nMemAvailable: -1 kB"])
def test_missing_or_bad_memory_is_monitor_failure(proc, text):
    put(proc / "meminfo", text)
    with pytest.raises(mod.PhaseFailure, match="memory_unavailable"):
        mod.MemoryGuard().check(proc)


def arguments(tmp_path, extra=()):
    return mod.parse_args(["--output-dir", str(tmp_path / "run"), "--phase", "p1",
                           "--chains", "none", "--duration", "0.1", "--warmup", "0",
                           "--model-path", "model.hef", "--model-id", "test-model",
                           "--run-id", "test-run", *extra])


def test_demo_command_uses_current_interpreter_fixed_paths_and_model_retention(tmp_path):
    args = arguments(tmp_path, ["--warmup", "2", "--duration", "3", "--reuse-model",
                               "--a-fps", "5", "--b-fps", "6", "--publish-hz", "20",
                               "--no-metrics-overlay"])
    command = mod.demo_command(args)
    assert command[:2] == [sys.executable, str(DIRECTORY / "app.py")]
    for option, value in [("--duration", "38.0"), ("--chains", "none"),
                          ("--json-path", str(args.output_dir / "status.json")),
                          ("--samples-path", str(args.output_dir / "samples.jsonl")),
                          ("--run-id", "test-run"), ("--phase", "p1")]:
        assert command[command.index(option) + 1] == value
    assert all(flag in command for flag in ("--keep-model", "--reuse-model", "--no-metrics-overlay"))


@pytest.mark.parametrize("extra", [["--duration", "0"], ["--duration", "nan"],
                                   ["--duration", "86001"], ["--warmup", "-1"],
                                   ["--warmup", "86000"], ["--phase", ""],
                                   ["--model-id", "bad/id"], ["--expect-platform", "1"],
                                   ["--expect-platform", "0:2"], ["--b-fps", "inf"],
                                   ["--publish-hz", "0"]])
def test_cli_invalid_inputs_fail_before_creating_output(tmp_path, extra):
    with pytest.raises(SystemExit):
        arguments(tmp_path, extra)
    assert not (tmp_path / "run").exists()


class FakeSampler:
    def sample(self, run_id, phase):
        return {"type": "resources", "monotonic_ns": time.monotonic_ns(),
                "run_id": run_id, "phase": phase, "system_cpu_pct": 25, "errors": []}


@pytest.fixture
def fake_app(tmp_path):
    path = tmp_path / "app.py"
    put(path, '''import argparse,json,time,signal
stopped=False
def stop(*_):
    global stopped
    stopped=True
signal.signal(signal.SIGTERM,stop)
p=argparse.ArgumentParser()
p.add_argument('--samples-path');p.add_argument('--run-id');p.add_argument('--phase')
a,_=p.parse_known_args()
def emit(kind, **fields):
    with open(a.samples_path,'a') as f:
        f.write(json.dumps(dict(type=kind,run_id=a.run_id,phase=a.phase,monotonic_ns=time.monotonic_ns(),**fields))+'\\n')
for stream in ('main','sub'):emit('encoded_packet',stream_id=stream,pts_ns=1)
while not stopped:time.sleep(.01)
emit('run_exit',exit_code=0,cleanup_errors=[],alive_threads=[])
emit('recorder_final',dropped=0,error=None,writer_alive=False,exit_code=0)
''')
    return path


def test_phase_integration_success_final_resources_and_manifest(tmp_path, proc, fake_app):
    args = arguments(tmp_path)
    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler):
        assert mod.run_phase(args, proc_root=proc) == 0
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "pass"
    assert manifest["returncodes"]["app"] == 0
    assert manifest["phase_start_monotonic_ns"] <= manifest["steady_start_monotonic_ns"]
    assert manifest["steady_end_monotonic_ns"] <= manifest["end_monotonic_ns"]
    assert manifest["platform_initial"] == manifest["platform_final"]
    assert manifest["keep_model"] is True  # Requested policy, not observed registration state.
    assert manifest["model_observed"]["model_kept"] is None
    rows = [json.loads(line) for line in (args.output_dir / "resources.jsonl").read_text().splitlines()]
    assert rows[-1]["monotonic_ns"] >= manifest["app_exit_observed_monotonic_ns"]
    assert (args.output_dir / "app.stdout.log").exists()


def test_existing_directory_is_never_reused(tmp_path, proc):
    args = arguments(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        mod.run_phase(args, proc_root=proc)


def test_preflight_missing_services_creates_failed_manifest_without_spawning(tmp_path, proc):
    args = arguments(tmp_path)
    (proc / "10/stat").unlink()
    with patch.object(mod.subprocess, "Popen") as popen:
        assert mod.run_phase(args, proc_root=proc) == 1
    popen.assert_not_called()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "fail"
    assert "platform_missing" in manifest["reason"]


@pytest.mark.parametrize("emit_final", [False, True])
def test_early_demo_exit_is_failure_not_success(tmp_path, proc, fake_app, emit_final):
    put(fake_app, fake_app.read_text().replace("while not stopped:time.sleep(.01)", "time.sleep(.01)")
        if emit_final else "raise SystemExit(0)\n")
    args = arguments(tmp_path, ["--duration", "0.5"])
    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler):
        assert mod.run_phase(args, proc_root=proc) == 1
    assert "app_early_exit" in json.loads((args.output_dir / "manifest.json").read_text())["reason"]


def test_cancel_terminates_only_spawned_process_group_and_records_interruption(tmp_path, proc, fake_app):
    put(fake_app, "import time\ntime.sleep(60)\n")
    args = arguments(tmp_path, ["--duration", "60"])
    start = time.monotonic()
    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler):
        assert mod.run_phase(args, proc_root=proc, cancelled=lambda: time.monotonic() - start > .2) == 1
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["reason"] == "cancelled"
    assert manifest["cleanup"]["term_sent"] is True
    assert manifest["cleanup"]["kill_sent"] is False
    assert mod.platform_fingerprint(proc)


def test_final_events_written_between_read_and_child_exit_are_consumed(tmp_path, proc, fake_app):
    args = arguments(tmp_path)
    real_read = mod.EventReader.read
    delayed = []

    def raced_read(reader):
        rows = real_read(reader)
        if delayed:
            return delayed.pop() + rows
        final = [row for row in rows if row["type"] in ("run_exit", "recorder_final")]
        if final:
            delayed.append(final)
            return [row for row in rows if row not in final]
        return rows

    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler), \
            patch.object(mod.EventReader, "read", raced_read):
        assert mod.run_phase(args, proc_root=proc) == 0


def test_unexpected_monitor_exception_cleans_up_and_never_marks_pass(tmp_path, proc, fake_app):
    args = arguments(tmp_path)
    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler), \
            patch.object(mod.EventReader, "read", side_effect=RuntimeError("secret-detail")):
        assert mod.run_phase(args, proc_root=proc) == 1
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "fail"
    assert manifest["returncodes"]["app"] is not None
    assert "secret-detail" not in json.dumps(manifest)


def test_status_file_reports_recorder_failure_when_raw_writer_is_broken(tmp_path, proc, fake_app):
    put(fake_app, "import argparse,json,time\np=argparse.ArgumentParser();p.add_argument('--json-path');a,_=p.parse_known_args()\nwith open(a.json_path,'w') as f:json.dump({'recorder':{'dropped':1,'error':'write failed','writer_alive':True}},f)\ntime.sleep(60)\n")
    args = arguments(tmp_path, ["--duration", "60"])
    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler):
        assert mod.run_phase(args, proc_root=proc, cancelled=lambda: (args.output_dir / "cancel").exists()) == 1
    assert "recorder_loss_or_error" in json.loads((args.output_dir / "manifest.json").read_text())["reason"]


def test_status_final_stopped_writer_is_valid_but_live_stopped_writer_is_not():
    monitor = mod.EventMonitor("test-run", "p1", "none", 0)
    snapshot = {"recorder": {"dropped": 0, "error": None, "writer_alive": False}}
    with pytest.raises(mod.PhaseFailure):
        monitor.inspect_status(snapshot)
    monitor.inspect_status({**snapshot, "exit_status": {"exit_code": 0}})


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_runner_cli_signal_cancels_and_reaps_only_its_demo(tmp_path, proc, fake_app, sig):
    put(fake_app, "import time\ntime.sleep(60)\n")
    wrapper = tmp_path / "runner_wrapper.py"
    put(wrapper, f"import sys\nfrom pathlib import Path\nsys.path.insert(0,{str(DIRECTORY)!r})\nimport phase_runner as m\nm.APP_PATH=Path({str(fake_app)!r})\nreal=m.run_phase\nm.run_phase=lambda a,**kw:real(a,proc_root=Path({str(proc)!r}),**kw)\nraise SystemExit(m.main())\n")
    output = tmp_path / "run"
    child = subprocess.Popen([sys.executable, str(wrapper), "--output-dir", str(output),
                              "--phase", "p1", "--chains", "none", "--duration", "60",
                              "--model-path", "model.hef", "--model-id", "test-model"],
                             stderr=subprocess.PIPE)
    demo_pid = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            path = output / "manifest.json"
            if path.exists():
                data = json.loads(path.read_text())
                demo_pid = data.get("app_pid")
                if demo_pid:
                    break
            assert child.poll() is None
            time.sleep(.01)
        assert demo_pid is not None
        child.send_signal(sig)
        _, stderr = child.communicate(timeout=5)
        assert child.returncode == 1, stderr
        result = json.loads((output / "manifest.json").read_text())
        assert result["reason"] == "cancelled"
        assert result["cleanup"]["kill_sent"] is False
        with pytest.raises(ProcessLookupError):
            os.kill(demo_pid, 0)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        if demo_pid:
            try:
                os.killpg(demo_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.stderr.close()


def test_main_failure_restores_signal_handlers(tmp_path):
    args = arguments(tmp_path)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with patch.object(mod, "parse_args", return_value=args), \
            patch.object(mod, "run_phase", side_effect=OSError("sensitive-path")):
        assert mod.main([]) == 1
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers


def test_resource_write_failure_cleans_child_and_writes_failed_manifest(tmp_path, proc, fake_app):
    args = arguments(tmp_path)
    original_write = mod.JsonlWriter.write
    calls = 0

    def fail_after_spawn(writer, record):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("secret-path")
        return original_write(writer, record)

    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler), \
            patch.object(mod.JsonlWriter, "write", fail_after_spawn):
        assert mod.run_phase(args, proc_root=proc) == 1
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "fail"
    assert manifest["returncodes"]["app"] is not None
    assert "secret-path" not in json.dumps(manifest)


def test_stdout_hard_limit_terminates_child_without_pipe_deadlock(tmp_path, proc, fake_app):
    put(fake_app, "import os,time\nwhile True:os.write(1,b'x'*65536)\n")
    args = arguments(tmp_path, ["--duration", "60", "--max-log-bytes", "1024"])
    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler):
        assert mod.run_phase(args, proc_root=proc) == 1
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["reason"] == "stdout_limit"
    assert (args.output_dir / "app.stdout.log").stat().st_size <= 1024
    assert manifest["returncodes"]["app"] is not None
    assert manifest["cleanup"]["kill_sent"] is False


def test_log_sink_checks_reserve_before_each_short_write_and_exact_limit(tmp_path):
    path = tmp_path / "stdout.log"
    real_write = os.write
    with mod.BoundedLog(path, max_bytes=6, min_free_bytes=10) as sink:
        with patch.object(mod.os, "write", side_effect=lambda fd, data: real_write(fd, data[:2])):
            sink.write(b"abcdef")
        with pytest.raises(mod.PhaseFailure, match="stdout_limit"):
            sink.write(b"x")
    assert path.read_bytes() == b"abcdef"
    with mod.BoundedLog(tmp_path / "reserve.log", max_bytes=100, min_free_bytes=10) as sink:
        space = type("Space", (), {"f_bavail": 10, "f_frsize": 1})()
        with patch.object(mod.os, "fstatvfs", return_value=space), patch.object(mod.os, "write") as write:
            with pytest.raises(mod.PhaseFailure, match="stdout_disk_reserve"):
                sink.write(b"x")
        write.assert_not_called()


def test_stdout_is_drained_while_term_handler_flushes_more_than_pipe_capacity(tmp_path, proc, fake_app):
    put(fake_app, "import os,signal,time\ndef stop(*_):\n for i in range(32):os.write(1,b'z'*65536)\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\nprint('ready',flush=True)\ntime.sleep(60)\n")
    args = arguments(tmp_path, ["--duration", "60", "--max-log-bytes", "4194304"])
    ready = args.output_dir / "app.stdout.log"
    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler):
        assert mod.run_phase(args, proc_root=proc, cancelled=lambda: ready.exists() and ready.stat().st_size > 0) == 1
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["reason"] == "cancelled"
    assert manifest["cleanup"]["kill_sent"] is False
    assert manifest["returncodes"]["app"] == 0
    assert ready.stat().st_size == len(b"ready\n") + 32 * 65536


def test_waitid_peek_preserves_leader_until_all_group_signals_are_finished():
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while mod.child_exit_code(child) is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert mod.child_exit_code(child) == 0
        assert child.returncode is None
        assert os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT).si_pid == child.pid
        with patch.object(mod.os, "killpg") as killpg:
            assert mod.stop_child(child, grace=.1)["returncode"] == 0
        killpg.assert_not_called()
    finally:
        child.wait(timeout=5)


def test_lost_leader_identity_never_signals_reused_group_number():
    child = type("Reaped", (), {"pid": 123, "returncode": 0,
                                "poll": lambda self: 0, "wait": lambda self, **kw: 0})()
    with patch.object(mod.os, "waitid", side_effect=ChildProcessError), \
            patch.object(mod.os, "killpg") as killpg:
        with pytest.raises(mod.PhaseFailure, match="child_identity_lost"):
            mod.stop_child(child, grace=0)
    killpg.assert_not_called()


def test_exited_leader_live_descendant_is_cleaned_but_independent_sentinel_survives(tmp_path):
    # Isolate PR_SET_CHILD_SUBREAPER in a helper so the test also reaps its orphan.
    script = '''import ctypes,json,os,signal,subprocess,sys,time
sys.path.insert(0,sys.argv[1])
import phase_runner as m
assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
code="import os,signal,time;pid=os.fork();\\nif pid==0:\\n signal.signal(signal.SIGTERM,signal.SIG_IGN);print(os.getpid(),flush=True);time.sleep(60)\\nelse:\\n time.sleep(.1);os._exit(0)"
leader=subprocess.Popen([sys.executable,'-c',code],start_new_session=True,stdout=subprocess.PIPE)
try:
 descendant=int(leader.stdout.readline())
 deadline=time.monotonic()+5
 while m.child_exit_code(leader) is None and time.monotonic()<deadline:time.sleep(.01)
 result=m.stop_child(leader,grace=.1)
 assert result['kill_sent'] and result['returncode']==0
 assert sentinel.poll() is None
 os.waitpid(descendant,0)
 print(json.dumps(result))
finally:
 sentinel.terminate();sentinel.wait(timeout=5)
 if leader.returncode is None:os.killpg(leader.pid,signal.SIGKILL);leader.wait(timeout=5)
 leader.stdout.close()
'''
    completed = subprocess.run([sys.executable, "-c", script, str(DIRECTORY)], capture_output=True,
                               text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["kill_sent"] is True


@pytest.mark.parametrize("window", ["cleanup", "final_commit"])
def test_cancellation_after_successful_loop_never_commits_pass(tmp_path, proc, fake_app, window):
    args = arguments(tmp_path)
    cancelled = False
    original_stop, original_write = mod.stop_child, mod._manifest_write

    def stop(*a, **kw):
        nonlocal cancelled
        result = original_stop(*a, **kw)
        if window == "cleanup":
            cancelled = True
        return result

    def persist(path, data):
        nonlocal cancelled
        original_write(path, data)
        if window == "final_commit" and data["status"] == "pass":
            cancelled = True

    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler), \
            patch.object(mod, "stop_child", stop), patch.object(mod, "_manifest_write", persist):
        assert mod.run_phase(args, proc_root=proc, cancelled=lambda: cancelled) == 1
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "fail"
    assert manifest["reason"] == "cancelled"


def test_late_cancellation_preserves_first_failure(tmp_path, proc, fake_app):
    put(fake_app, "raise SystemExit(1)\n")
    args = arguments(tmp_path)
    cancelled = False
    original_stop = mod.stop_child

    def stop(*a, **kw):
        nonlocal cancelled
        result = original_stop(*a, **kw)
        cancelled = True
        return result

    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler), \
            patch.object(mod, "stop_child", stop):
        assert mod.run_phase(args, proc_root=proc, cancelled=lambda: cancelled) == 1
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["reason"] == "app_failed_exit"
    assert "cancelled" in manifest["errors"]


def test_replayed_alternating_ids_cannot_keep_freshness_alive():
    monitor = mod.EventMonitor("test-run", "p1", "ab", 0)
    monitor.consume(good_events(31, 1) + good_events(32, 2), 32_000_000_000)
    monitor.consume(good_events(39, 1) + good_events(40, 2), 40_000_000_000)
    with pytest.raises(mod.PhaseFailure, match="stale:"):
        monitor.check(42_000_000_000)


def test_recovered_batch_latches_prior_producer_time_gap():
    monitor = mod.EventMonitor("test-run", "p1", "ab", 0)
    with pytest.raises(mod.PhaseFailure, match="stale:"):
        monitor.consume(good_events(31, 1) + good_events(46, 2), 46_000_000_000)
    with pytest.raises(mod.PhaseFailure, match="stale:"):
        monitor.consume(good_events(47, 3), 47_000_000_000)


def cpu_sample(second, value=25, errors=None):
    return {"monotonic_ns": int(second * 1e9), "system_cpu_pct": value,
            "errors": errors or []}


def test_cpu_coverage_excludes_warmup_crossing_interval_and_optional_sensor_errors():
    coverage = mod.CpuCoverage()
    coverage.begin(cpu_sample(10))
    coverage.add(cpu_sample(11, errors=[{"source": "cma", "code": "PermissionError"}]))
    result = coverage.finish()
    assert result["covered_ns"] == 1_000_000_000
    assert result["observed_ns"] == 1_000_000_000
    assert result["coverage_pct"] == 100
    with pytest.raises(mod.PhaseFailure, match="cpu_measurement_invalid"):
        coverage.add(cpu_sample(14, errors=[{"source": "stat", "code": "PermissionError"}]))


@pytest.mark.parametrize("bad", [None, float("nan"), -1, 101])
def test_cpu_missing_invalid_or_no_interval_never_passes(bad):
    coverage = mod.CpuCoverage()
    coverage.begin(cpu_sample(1))
    with pytest.raises(mod.PhaseFailure, match="cpu_measurement_invalid"):
        coverage.finish()
    coverage.add(cpu_sample(2, bad))
    with pytest.raises(mod.PhaseFailure, match="cpu_measurement_invalid"):
        coverage.finish()


def test_warmup_starts_only_after_all_channels_are_ready(tmp_path, proc, fake_app):
    put(fake_app, fake_app.read_text().replace("for stream in", "time.sleep(.3)\nfor stream in"))
    args = arguments(tmp_path, ["--warmup", ".2"])
    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler):
        assert mod.run_phase(args, proc_root=proc) == 0
    state = json.loads((args.output_dir / "manifest.json").read_text())
    assert state["ready_monotonic_ns"] - state["phase_start_monotonic_ns"] >= 300_000_000
    assert state["steady_start_monotonic_ns"] - state["ready_monotonic_ns"] >= 200_000_000
    assert state["steady_end_monotonic_ns"] - state["steady_start_monotonic_ns"] >= 100_000_000
    assert state["cpu_measurement"]["coverage_pct"] == 100


@pytest.mark.parametrize("bad", [None, True, -1])
def test_cpu_baseline_rejects_invalid_clocks(bad):
    with pytest.raises(mod.PhaseFailure, match="cpu_measurement_invalid"):
        mod.CpuCoverage().begin({"monotonic_ns": bad})


@pytest.mark.parametrize("stat_error", [False, True])
def test_runner_rejects_invalid_cpu_measurement_not_optional_sensors(tmp_path, proc, fake_app, stat_error):
    class MissingCpu(FakeSampler):
        def sample(self, run_id, phase):
            result = super().sample(run_id, phase)
            return {**result, "system_cpu_pct": 25 if stat_error else None,
                    "errors": [{"source": "stat", "code": "PermissionError"}] if stat_error else []}
    args = arguments(tmp_path)
    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", MissingCpu):
        assert mod.run_phase(args, proc_root=proc) == 1
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["reason"] == "cpu_measurement_invalid"
    assert manifest["cpu_measurement"]["coverage_pct"] == 0


@pytest.mark.parametrize("pump_error", [False, True])
def test_cleanup_pump_error_cannot_abort_termination_of_stubborn_child(pump_error):
    child = subprocess.Popen([sys.executable, "-c", "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(60)"], start_new_session=True, stdout=subprocess.PIPE)
    try:
        assert child.stdout.readline() == b"ready\n"
        def broken_pipe():
            if pump_error:
                raise OSError("closed pipe")
        result = mod.stop_child(child, grace=.1, pump=broken_pipe)
        assert result["kill_sent"] is True
        assert result["returncode"] == -signal.SIGKILL
        assert result.get("pump_error") == ("OSError" if pump_error else None)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        child.stdout.close()


@pytest.mark.parametrize("now,first,steady,error", [
    (29, 1, False, "stale:encoded:main"), (29, 19, False, "stale:encoded:main"),
    (29, 20, False, None), (30, 29, False, "ready_timeout"),
    (47, 36, True, None), (47, 35, True, "stale:"),
    (47, 36, True, "demo_cleanup_failed"),
])
def test_tick_readiness_and_sample_boundary(tmp_path, proc, now, first, steady, error):
    phase = mod._Phase(arguments(tmp_path, ["--duration", "5", "--warmup", "10"]), proc, lambda: False)
    phase.baseline = mod.platform_fingerprint(proc)
    phase.monitor = mod.EventMonitor("test-run", "p1", "none", 0)
    phase.state["phase_start_monotonic_ns"] = 0
    phase.monitor.consume(good_events(first), int(first * 1e9))
    rows = good_events(now, 2)[-1:] if not steady else good_events(now, 2)
    if steady:
        phase.state.update(ready_monotonic_ns=30_000_000_000,
                           planned_steady_start_monotonic_ns=40_000_000_000)
        phase.monitor.health_start_ns = 30_000_000_000
        with patch.object(phase, "_persist"):
            phase._measurement_tick(cpu_sample(40))
    if error == "demo_cleanup_failed":
        rows.append(event("run_exit", now, exit_code=1))
    with patch.object(phase, "_pump_output"), patch.object(mod, "child_exit_code", return_value=None), \
            patch.object(mod.time, "monotonic_ns", return_value=int(now * 1e9)), \
            patch.object(mod.ProcSampler, "sample", return_value=cpu_sample(45 if steady else now)), \
            patch.object(phase.reader, "read", return_value=rows), patch.object(mod.JsonlWriter, "write") as write:
        phase.collector = mod.ProcSampler()
        if error:
            with pytest.raises(mod.PhaseFailure, match=error):
                phase._tick(type("Writer", (), {"write": write})())
        else:
            assert phase._tick(type("Writer", (), {"write": write})()) is steady
            assert phase.state["ready_monotonic_ns"] is not None
            if steady:
                assert phase.monitor.end_ns == phase.state["steady_end_monotonic_ns"] == 45_000_000_000
                assert phase.monitor.last_success["encoded:main"] == int(first * 1e9)


def test_app_timeout_force_kill_is_recorded_as_failure(tmp_path, proc, fake_app):
    put(fake_app, "import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(60)\n")
    args = arguments(tmp_path)
    original_stop = mod.stop_child
    with patch.object(mod, "APP_PATH", fake_app), patch.object(mod, "ProcSampler", FakeSampler), \
            patch.object(mod, "INIT_GRACE_NS", 250_000_000), \
            patch.object(mod, "stop_child", side_effect=lambda child, **kw: original_stop(child, grace=.1, **kw)):
        assert mod.run_phase(args, proc_root=proc) == 1
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert manifest["reason"] == "ready_timeout"
    assert "forced_kill" in manifest["errors"]
    assert manifest["cleanup"]["kill_sent"] is True
    assert manifest["returncodes"]["app"] == -signal.SIGKILL


def test_nonblocking_setup_failure_cannot_hang_cleanup(tmp_path, proc, fake_app):
    put(fake_app, "import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nprint('ready',flush=True)\ntime.sleep(60)\n")
    script = '''import runpy,sys,signal,os
from pathlib import Path
ns=runpy.run_path(sys.argv[1]); m=ns['mod']
m.APP_PATH=Path(sys.argv[2]); m.ProcSampler=ns['FakeSampler']
a=ns['arguments'](Path(sys.argv[3])); a.output_dir.mkdir()
p=m._Phase(a,Path(sys.argv[4]),lambda:False)
stop=m.stop_child; m.stop_child=lambda child,**kw:stop(child,grace=.05,**kw)
def watchdog(*_):raise SystemExit('cleanup watchdog expired')
def fail(fd,blocking):
 assert os.read(fd,6)==b'ready\\n'
 signal.setitimer(signal.ITIMER_REAL,.8)
 raise OSError('injected nonblocking setup failure')
signal.signal(signal.SIGALRM,watchdog); signal.setitimer(signal.ITIMER_REAL,3); m.os.set_blocking=fail
try:
 assert p.run()==1
 assert p.state['reason']=='OSError' and p.state['status']=='fail'
 assert p.state['cleanup']['term_sent'] and p.state['cleanup']['kill_sent']
 assert 'forced_kill' in p.state['errors']
 assert p.child.returncode==-signal.SIGKILL and p.child.stdout.closed
 try:os.waitpid(p.child.pid,os.WNOHANG)
 except ChildProcessError:pass
 else:raise AssertionError('child not reaped')
finally:
 signal.setitimer(signal.ITIMER_REAL,0)
 if p.child is not None and p.child.returncode is None:
  os.killpg(p.child.pid,signal.SIGKILL); p.child.wait(timeout=2)
'''
    result = subprocess.run([sys.executable, "-c", script, __file__, str(fake_app), str(tmp_path), str(proc)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
