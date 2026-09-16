"""Offline capability/regression evals for the standalone resource collector."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples/perf_demo/proc_sampler.py"
spec = importlib.util.spec_from_file_location("perf_proc_sampler", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def stat_text(pid=42, name="camera-daemon", user=10, system=5, start=100, ppid=1):
    fields = ["S", str(ppid)] + ["0"] * 48
    for index, value in {11: user, 12: system, 17: 3, 19: start, 20: 10000, 21: 4}.items():
        fields[index] = str(value)
    return f"{pid} ({name}) " + " ".join(fields) + "\n"


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def process(root, pid=42, name="camera-daemon", **kwargs):
    put(root / str(pid) / "stat", stat_text(pid, name, **kwargs))
    put(root / str(pid) / "status", "VmRSS:\t20 kB\nThreads:\t3\n"
        "voluntary_ctxt_switches:\t11\nnonvoluntary_ctxt_switches:\t7\n")
    put(root / str(pid) / "cmdline", f"/usr/bin/{name}\0")
    (root / str(pid) / "fd").mkdir(exist_ok=True)
    put(root / str(pid) / "fd/0", "")
    put(root / str(pid) / "smaps_rollup", "Pss: 12 kB\n")


def system(root, busy=100, idle=100, guest=40, iowait=0):
    put(root / "stat", f"cpu {busy} 0 0 {idle} {iowait} 0 0 0 {guest} 0\n"
        f"cpu0 {busy} 0 0 0 0 0 0 0 {guest} 0\n"
        f"cpu1 0 0 0 {idle} {iowait} 0 0 0 0 0\n")
    put(root / "meminfo", "MemAvailable: 1000 kB\nCmaTotal: 400 kB\nCmaFree: 300 kB\n")


@pytest.fixture
def tree(tmp_path):
    root, sysroot = tmp_path / "proc", tmp_path / "sys"
    root.mkdir()
    sysroot.mkdir()
    system(root)
    process(root)
    return root, sysroot


def sampler(tree, **kwargs):
    root, sysroot = tree
    return mod.ProcSampler(proc_root=root, sys_root=sysroot, hz=100,
                           page_size=4096, collector_pid=999, **kwargs)


def sample(collector, seconds):
    return collector.sample("run 'unicode-测试", "baseline", now_ns=int(seconds * 1e9))


@pytest.mark.parametrize("name", ["comm with spaces", "a(b) c))", "测试 ');--", "x\ny"])
def test_stat_parser_preserves_comm_and_correct_field_offsets(name):
    result = mod.parse_process_stat(stat_text(name=name))
    assert result == {"pid": 42, "name": name, "ppid": 1, "utime": 10,
                      "stime": 5, "start_ticks": 100, "rss_pages": 4, "num_threads": 3}


@pytest.mark.parametrize("text", [None, 12, b"bad", "", "42 no parentheses", "42 (x) S 1", stat_text(user=-1)])
def test_bad_stat_input_is_explicit(text):
    with pytest.raises(ValueError):
        mod.parse_process_stat(text)


def test_cpu_guest_not_counted_twice_and_whole_machine_differs_from_one_core(tree):
    collector = sampler(tree)
    first = sample(collector, 1)
    assert first["system_cpu_pct"] is None
    assert first["per_core_cpu_pct"] == {"cpu0": None, "cpu1": None}
    assert first["processes"][0]["cpu_pct_one_core"] is None
    system(tree[0], busy=200, idle=200, guest=140)
    process(tree[0], user=210)
    second = sample(collector, 2)
    assert second["system_cpu_pct"] == 50.0
    assert second["per_core_cpu_pct"] == {"cpu0": 100.0, "cpu1": 0.0}
    assert second["cpu_count"] == 2
    assert second["processes"][0]["cpu_pct_one_core"] == 200.0
    assert second["mem_available_bytes"] == 1024000
    assert second["utc_time"].endswith("Z")
    assert second["type"] == "resources"
    assert second["monotonic_ns"] == 2000000000
    assert second["run_id"] == "run 'unicode-测试"
    assert second["phase"] == "baseline"


def test_cpu_uses_actual_elapsed_time_and_excludes_iowait(tree):
    collector = sampler(tree)
    sample(collector, 1)
    system(tree[0], busy=200, idle=200, iowait=200)
    process(tree[0], user=110)
    result = sample(collector, 3)
    assert result["system_cpu_pct"] == 25.0
    assert result["system_iowait_pct"] == 50.0
    assert result["system_irq_pct"] == 0.0
    assert result["system_softirq_pct"] == 0.0
    assert result["processes"][0]["cpu_pct_one_core"] == 50.0


@pytest.mark.parametrize("change", ["reuse", "utime_reset", "stime_reset", "zero_time", "reverse_time", "cpu_reset"])
def test_counter_reset_or_invalid_clock_produces_null_not_negative(tree, change):
    collector = sampler(tree)
    sample(collector, 2)
    next_time = 3
    if change == "reuse":
        process(tree[0], start=200, user=200)
    elif change == "utime_reset":
        process(tree[0], user=0, system=100)
    elif change == "stime_reset":
        process(tree[0], user=100, system=0)
    elif change == "zero_time":
        next_time = 2
    elif change == "reverse_time":
        next_time = 1
    else:
        system(tree[0], busy=50, idle=1000)
    result = sample(collector, next_time)
    if change == "cpu_reset":
        assert result["system_cpu_pct"] is None
    else:
        assert result["processes"][0]["cpu_pct_one_core"] is None
    if change != "reuse":
        assert result["errors"]


def test_missing_pid_and_missing_system_sample_reset_baselines(tree):
    collector = sampler(tree)
    sample(collector, 1)
    (tree[0] / "42/stat").unlink()
    (tree[0] / "stat").unlink()
    missing = sample(collector, 2)
    assert missing["processes"] == []
    assert missing["system_cpu_pct"] is None
    assert missing["errors"]
    process(tree[0], user=110)
    system(tree[0], busy=200)
    recovered = sample(collector, 3)
    assert recovered["system_cpu_pct"] is None
    assert recovered["processes"][0]["cpu_pct_one_core"] is None


def test_discovery_roles_children_and_no_cmdline_credentials(tree):
    root = tree[0]
    for pid, name in [(50, "python3"), (51, "worker"), (52, "ai-runtime"),
                      (53, "hailort_server"), (54, "isp_media_serve"),
                      (55, "ffmpeg"), (56, "unrelated"), (999, "python3"),
                      (57, "python3"), (58, "gst-launch-1.0")]:
        process(root, pid, name, ppid=50 if pid == 51 else 1)
    put(root / "50/cmdline", "/bin/python3\0-m\0perf_demo.app\0--password\0do-not-log-me\0")
    put(root / "57/cmdline", "python3\0/opt/perf-demo/app.py\0--token\0also-secret\0")
    result = sample(sampler(tree), 1)
    roles = {p["pid"]: p["role"] for p in result["processes"]}
    assert roles == {42: "camera-daemon", 50: "sdk", 51: "sdk-child", 52: "ai-runtime",
                     53: "hailort", 54: "video-service", 55: "video-service", 57: "sdk",
                     58: "video-service", 999: "collector"}
    assert "do-not-log-me" not in json.dumps(result)
    assert "also-secret" not in json.dumps(result)
    assert "cmdline" not in result["processes"][0]


def test_late_discovery_and_pid_reuse_do_not_inherit_sdk_role(tree):
    collector = sampler(tree)
    sample(collector, 1)
    process(tree[0], 50, "python3")
    put(tree[0] / "50/cmdline", "python3\0perf_demo/app.py\0")
    assert 50 in {p["pid"] for p in sample(collector, 2)["processes"]}
    process(tree[0], 50, "python3", start=200)
    assert 50 not in {p["pid"] for p in sample(collector, 3)["processes"]}


def test_pid_reused_during_multi_file_read_is_not_mixed(tree):
    collector = sampler(tree)
    real_read = Path.read_text
    calls = 0

    def raced_read(path, *args, **kwargs):
        nonlocal calls
        if path == tree[0] / "42/stat":
            calls += 1
            return stat_text(start=100 if calls == 1 else 200)
        return real_read(path, *args, **kwargs)

    with patch.object(Path, "read_text", raced_read):
        result = sample(collector, 1)
    assert result["processes"] == []
    assert any(e["code"] == "pid_changed" for e in result["errors"])


def test_status_missing_bad_fields_and_permission_denied_are_null(tree):
    put(tree[0] / "42/status", "VmRSS: bad kB\nThreads: -1\nvoluntary_ctxt_switches: 4\n")
    original_iterdir = Path.iterdir

    def denied(path):
        if path.name == "fd":
            raise PermissionError("secret-token-must-not-leak")
        return original_iterdir(path)

    with patch.object(Path, "iterdir", denied):
        result = sample(sampler(tree), 1)
    proc = result["processes"][0]
    assert proc["rss_bytes"] is None
    assert proc["num_threads"] is None
    assert proc["fd_count"] is None
    assert proc["ctxswitch"] == {"voluntary": 4, "nonvoluntary": None}
    assert result["errors"]
    assert "secret-token-must-not-leak" not in json.dumps(result)


def test_slow_sensors_fd_and_optional_pss_have_explicit_cadence(tree):
    root, sysroot = tree
    put(sysroot / "class/thermal/thermal_zone0/temp", "42000\n")
    put(sysroot / "devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "1200000\n")
    put(sysroot / "kernel/debug/cma/pool/count", "100\n")
    put(sysroot / "kernel/debug/cma/pool/order_per_bit", "0\n")
    collector = sampler(tree, include_pss=True)
    first = sample(collector, 1)
    assert first["temperature_c"] == {"thermal_zone0": 42.0}
    assert first["cpu_frequency_khz"] == {"cpu0": 1200000}
    assert first["cma_pools"]["pool"]["total_bytes"] == 409600
    assert first["cma_pools"]["pool"]["used_bytes"] is None
    assert first["cma_total_bytes"] == 409600
    assert first["cma_free_bytes"] == 307200
    assert first["processes"][0]["fd_count"] == 1
    assert first["processes"][0]["pss_bytes"] == 12288
    assert first["slow_sample_monotonic_ns"] == 1000000000
    second = sample(collector, 2)
    assert second["temperature_c"] is None
    assert second["processes"][0]["fd_count"] is None
    assert second["processes"][0]["pss_bytes"] is None
    assert second["slow_sample_monotonic_ns"] is None
    fifth = sample(collector, 6)
    assert fifth["processes"][0]["fd_count"] == 1
    assert fifth["processes"][0]["pss_bytes"] is None
    assert sample(collector, 31)["processes"][0]["pss_bytes"] == 12288


def test_missing_sensors_status_and_pss_are_explicit(tree):
    (tree[0] / "42/status").unlink()
    (tree[0] / "42/smaps_rollup").unlink()
    result = sample(sampler(tree, include_pss=True), 1)
    assert result["temperature_c"] is None
    assert result["cpu_frequency_khz"] is None
    assert result["cma_pools"] is None
    assert result["processes"][0]["rss_bytes"] is None
    assert result["processes"][0]["pss_bytes"] is None
    assert result["errors"]


def test_bad_system_data_and_empty_tree_do_not_crash(tree):
    put(tree[0] / "stat", "cpu garbage\ncpu0 1 2\n")
    put(tree[0] / "meminfo", "MemAvailable: -1 kB\n")
    result = sample(sampler(tree), 1)
    assert result["system_cpu_pct"] is None
    assert result["mem_available_bytes"] is None
    assert result["errors"]


def test_process_discovery_handles_ten_thousand_entries_with_bounded_errors(tree):
    original_iterdir = Path.iterdir

    def many_entries(path):
        if path == tree[0]:
            return iter([path / str(pid) for pid in range(10000, 20000)] + [path / "42"])
        return original_iterdir(path)

    with patch.object(Path, "iterdir", many_entries):
        result = sample(sampler(tree), 1)
    assert len(result["processes"]) == 1
    assert len(result["errors"]) <= 100
    assert result["errors_omitted"] >= 9900


def test_writer_appends_valid_json_and_enforces_total_size(tmp_path):
    output = tmp_path / "resources.jsonl"
    put(output, '{"existing":true}\n')
    with mod.JsonlWriter(output, max_bytes=100, min_free_bytes=0) as writer:
        writer.write({"hello": "测试"})
        with pytest.raises(OSError, match="output_limit"):
            writer.write({"oversized": "x" * 100})
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [{"existing": True}, {"hello": "测试"}]
    assert output.stat().st_size <= 100


def test_writer_free_space_guard_and_write_failure(tmp_path):
    output = tmp_path / "resources.jsonl"
    with mod.JsonlWriter(output, max_bytes=1000, min_free_bytes=100) as writer:
        with patch.object(mod.os, "fstatvfs", return_value=Mock(f_bavail=1, f_frsize=100)):
            with pytest.raises(OSError, match="disk_reserve"):
                writer.write({"message": "x"})
        with patch.object(mod.os, "write", side_effect=OSError("write failed")):
            with pytest.raises(OSError):
                writer.write({"message": "x"})
    assert output.read_bytes() == b""


def test_writer_handles_short_writes_and_rejects_nonregular_output(tmp_path):
    output = tmp_path / "resources.jsonl"
    real_write = os.write
    with mod.JsonlWriter(output, max_bytes=1000, min_free_bytes=0) as writer:
        with patch.object(mod.os, "write", side_effect=lambda fd, data: real_write(fd, data[:3])):
            writer.write({"ok": True})
    assert json.loads(output.read_text()) == {"ok": True}
    link = tmp_path / "link"
    link.symlink_to(output)
    with pytest.raises(OSError):
        with mod.JsonlWriter(link, max_bytes=1000, min_free_bytes=0):
            pass


def test_cma_used_and_maxchunk_are_pages_not_count_or_bitmap_bytes(tree):
    pool = tree[1] / "kernel/debug/cma/cma-hailo_media"
    put(pool / "count", "100")
    put(pool / "used", "25")
    put(pool / "maxchunk", "50")
    result = sample(sampler(tree), 1)
    assert result["cma_pools"]["cma-hailo_media"] == {
        "total_bytes": 409600, "used_bytes": 102400,
        "free_bytes": 307200, "maxchunk_bytes": 204800,
    }
    assert (pool / "count").read_text() == "100"
    assert (pool / "used").read_text() == "25"


def test_cma_inconsistent_usage_and_malformed_sensors_are_null(tree):
    pool = tree[1] / "kernel/debug/cma/pool"
    put(pool / "count", "100")
    put(pool / "used", "101")
    put(pool / "maxchunk", "-1")
    put(tree[1] / "class/thermal/thermal_zone0/temp", "bad")
    put(tree[1] / "devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "-1")
    result = sample(sampler(tree), 1)
    assert result["cma_pools"]["pool"]["used_bytes"] is None
    assert result["cma_pools"]["pool"]["free_bytes"] is None
    assert result["temperature_c"] == {"thermal_zone0": None}
    assert result["cpu_frequency_khz"] == {"cpu0": None}
    assert any(e["code"] == "inconsistent_cma" for e in result["errors"])


def test_corrupt_core_remains_present_and_null_not_undercounted(tree):
    put(tree[0] / "stat", "cpu 1 0 1 10\ncpu0 bad\ncpu1 1 0 1 10\n")
    result = sample(sampler(tree), 1)
    assert result["cpu_count"] == 2
    assert result["per_core_cpu_pct"] == {"cpu0": None, "cpu1": None}


def test_writer_rejects_existing_unterminated_json_without_modifying_it(tmp_path):
    output = tmp_path / "resources.jsonl"
    put(output, '{"partial":')
    with pytest.raises(OSError, match="incomplete_output"):
        with mod.JsonlWriter(output, max_bytes=1000, min_free_bytes=0) as writer:
            writer.write({"next": True})
    assert output.read_text() == '{"partial":'


def test_writer_excludes_concurrent_collectors_and_rolls_back_partial_record(tmp_path):
    output = tmp_path / "resources.jsonl"
    real_write = os.write
    calls = 0

    def interrupted_write(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, data[:4])
        raise OSError("disk failure")

    with mod.JsonlWriter(output, max_bytes=1000, min_free_bytes=0) as writer:
        writer.write({"before": True})
        with pytest.raises(OSError):
            with mod.JsonlWriter(output, max_bytes=1000, min_free_bytes=0):
                pass
        with patch.object(mod.os, "write", side_effect=interrupted_write):
            with pytest.raises(OSError):
                writer.write({"partial": True})
        writer.write({"after": True})
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"before": True}, {"after": True}]


def test_writer_zero_write_exact_size_boundary_and_nonregular_path(tmp_path):
    output = tmp_path / "resources.jsonl"
    with mod.JsonlWriter(output, max_bytes=3, min_free_bytes=0) as writer:
        with patch.object(mod.os, "write", return_value=0):
            with pytest.raises(OSError, match="short_write"):
                writer.write({})
        writer.write({})
        with pytest.raises(OSError, match="output_limit"):
            writer.write({})
    assert output.read_bytes() == b"{}\n"
    with pytest.raises(OSError, match="output_not_regular"):
        with mod.JsonlWriter("/dev/null", max_bytes=1000, min_free_bytes=0):
            pass


def test_missing_cmdline_pid_mismatch_and_cycle_do_not_break_discovery(tree):
    root = tree[0]
    process(root, 50, "python3")
    (root / "50/cmdline").unlink()
    process(root, 51, "python3", ppid=52)
    put(root / "51/cmdline", "python3\0perf_demo.app\0")
    process(root, 52, "worker", ppid=51)
    put(root / "53/stat", stat_text(pid=54))
    result = sample(sampler(tree), 1)
    assert {p["pid"] for p in result["processes"]} == {42, 51, 52}
    assert any(e["code"] == "invalid_stat" for e in result["errors"])
    assert any(e["source"] == "50/cmdline" for e in result["errors"])


def test_disappearing_before_second_stat_read_is_an_explicit_gap(tree):
    real_read = Path.read_text
    calls = 0

    def disappears(path, *args, **kwargs):
        nonlocal calls
        if path == tree[0] / "42/stat":
            calls += 1
            if calls > 1:
                raise FileNotFoundError()
        return real_read(path, *args, **kwargs)

    with patch.object(Path, "read_text", disappears):
        result = sample(sampler(tree), 1)
    assert result["processes"] == []
    assert any(e["code"] == "pid_changed" for e in result["errors"])


@pytest.mark.parametrize("kwargs", [{"hz": 0}, {"page_size": -1}])
def test_sampler_rejects_invalid_sysconf_values(kwargs):
    with pytest.raises(ValueError):
        mod.ProcSampler(**kwargs)


@pytest.mark.parametrize("kwargs", [{"max_bytes": 0}, {"min_free_bytes": -1}])
def test_writer_rejects_invalid_limits(tmp_path, kwargs):
    with pytest.raises(ValueError):
        mod.JsonlWriter(tmp_path / "output", **kwargs)


def cli_args(output, duration="0.1"):
    return ["--output", str(output), "--interval", "0.03", "--duration", duration,
            "--run-id", "test-run", "--phase", "baseline", "--min-free-bytes", "0"]


def test_cli_local_end_to_end_duration_and_schema(tmp_path):
    output = tmp_path / "resources.jsonl"
    completed = subprocess.run([sys.executable, str(SCRIPT), *cli_args(output)],
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows
    assert rows[0]["system_cpu_pct"] is None
    assert all(row["type"] == "resources" and row["run_id"] == "test-run" for row in rows)
    assert any(proc["role"] == "collector" for proc in rows[0]["processes"])


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_cli_signal_interrupts_long_wait_and_flushes_complete_json(tmp_path, sig):
    output = tmp_path / "resources.jsonl"
    args = cli_args(output, "60") + ["--interval", "30"]
    child = subprocess.Popen([sys.executable, str(SCRIPT), *args], stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if output.exists() and output.stat().st_size:
                break
            assert child.poll() is None
            threading.Event().wait(0.01)
        else:
            pytest.fail("collector did not produce its first record")
        child.send_signal(sig)
        _, stderr = child.communicate(timeout=3)
        assert child.returncode == 0, stderr
        assert all(json.loads(line)["type"] == "resources" for line in output.read_text().splitlines())
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.parametrize("extra", [["--interval", "0"], ["--interval", "nan"],
                                   ["--duration", "-1"], ["--duration", "inf"],
                                   ["--run-id", ""], ["--phase", ""],
                                   ["--max-bytes", "0"], ["--min-free-bytes", "-1"]])
def test_cli_rejects_invalid_arguments_before_writing(tmp_path, extra):
    output = tmp_path / "resources.jsonl"
    with pytest.raises(SystemExit) as exc:
        mod.main(cli_args(output) + extra)
    assert exc.value.code == 2
    assert not output.exists()


def test_main_success_and_fsync_failure_restore_handlers(tmp_path, capsys):
    output = tmp_path / "resources.jsonl"
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    assert mod.main(cli_args(output)) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["phase"] == "baseline"
    with patch.object(mod.os, "fsync", side_effect=OSError("sensitive-path")):
        assert mod.main(cli_args(output)) == 1
    assert "sensitive-path" not in capsys.readouterr().err
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers


def test_cli_output_failure_returns_nonzero_and_restores_signal_handlers(tmp_path, capsys):
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    result = mod.main(cli_args(tmp_path / "resources.jsonl") + ["--max-bytes", "1"])
    assert result == 1
    assert "output_limit" in capsys.readouterr().err
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
