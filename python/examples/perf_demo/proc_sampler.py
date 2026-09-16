#!/usr/bin/env python3
"""Standalone Linux resource JSONL collector; Python stdlib only.

Run directly (does not import the demo or SDK)::

    python3 proc_sampler.py --output resources.jsonl --interval 1 \
        --duration 3600 --run-id run-01 --phase baseline

Each resources record contains monotonic_ns, ISO8601 UTC utc_time, run_id,
phase, cpu_count, system_cpu_pct (0..100), per_core_cpu_pct (CPU-name map),
mem_available_bytes and processes. A process is identified by pid+start_ticks;
cpu_pct_one_core=100 means one full logical core and may exceed 100. CPU
baselines are discarded after missing samples, counter decreases or PID reuse.
Guest ticks are already included in user/nice: sum only the first 8 CPU fields.

FDs, temperature_c (zone map), cpu_frequency_khz (CPU map), global CMA bytes
and readable debugfs CMA pools are sampled every 5 seconds. CMA count is TOTAL
pages; used and maxchunk are separate page counters. Missing values are null.
Optional --pss reads smaps_rollup every 30 seconds, with no costly smaps fallback.
Off-cadence or unavailable values are null, not stale values or zero. Cadence
is marked by slow_sample_monotonic_ns and pss_sample_monotonic_ns. ctxswitch
contains absolute voluntary/nonvoluntary counts from the process leader's
status, NOT sums of all threads. Errors have source/code only (no exception
messages or cmdline); errors_omitted bounds error output on large process trees.

Only output is written; proc/sys/debugfs are read-only. Append output is locked,
size/free-space guarded, flushed each record and fsynced on orderly exit.
SIGTERM/SIGINT interrupt the wait; failures return nonzero with a safe error.
Defaults: max output 256 MiB including existing data, reserve 64 MiB disk space.
The disk reserve is best-effort; other programs can consume disk concurrently.

Parsing follows proc(5)/proc_pid_stat(5) and kernel filesystems/proc.html.
The existing tests/device/perf_common.py proc_snapshot status/FD approach is
reused without importing its device-test dependencies. No psutil install needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
import signal
import stat
import sys
import time
from collections import deque
from pathlib import Path

SLOW_INTERVAL_NS = 5_000_000_000
PSS_INTERVAL_NS = 30_000_000_000
MAX_ERRORS = 100
MAX_CMDLINE_BYTES = 65536
STOP_POLL_SECONDS = 0.1
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_MIN_FREE_BYTES = 64 * 1024 * 1024


def parse_process_stat(text):
    """Parse stat using the final closing parenthesis, not whitespace in comm."""
    if not isinstance(text, str):
        raise ValueError("invalid_stat")
    left, right = text.find("("), text.rfind(")")
    if left < 1 or right <= left:
        raise ValueError("invalid_stat")
    fields = text[right + 1:].split()
    if len(fields) < 22:
        raise ValueError("short_stat")
    result = {"pid": int(text[:left]), "name": text[left + 1:right]}
    indexes = {"ppid": 1, "utime": 11, "stime": 12, "num_threads": 17,
               "start_ticks": 19, "rss_pages": 21}
    values = {key: int(fields[index]) for key, index in indexes.items()}
    if result["pid"] <= 0 or any(value < 0 for value in values.values()):
        raise ValueError("invalid_stat_counter")
    return {**result, **values}


def _error(errors, source, code):
    # A bounded record plus a counter, not an unbounded per-PID error backlog.
    if len(errors["items"]) < MAX_ERRORS:
        errors["items"].append({"source": source, "code": code})
    else:
        errors["omitted"] += 1


def _read(path, errors, source):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _error(errors, source, type(exc).__name__)
        return None


def _entries(path, errors, source):
    try:
        return list(path.iterdir())
    except OSError as exc:
        _error(errors, source, type(exc).__name__)
        return []


def _field(text, key, errors, source, unit=None):
    if text is None:
        return None
    for line in text.splitlines():
        label, sep, raw = line.partition(":")
        if not sep or label != key:
            continue
        parts = raw.split()
        try:
            if len(parts) != (2 if unit else 1) or (unit and parts[-1] != unit):
                raise ValueError("invalid_unit")
            value = int(parts[0])
            if value < 0:
                raise ValueError("negative_counter")
            return value * (1024 if unit == "kB" else 1)
        except ValueError:
            _error(errors, source + "/" + key, "invalid_field")
            return None
    _error(errors, source + "/" + key, "missing_field")
    return None


def _cpu_counters(text, errors):
    counters = {}
    for line in (text or "").splitlines():
        fields = line.split()
        if not fields or not (fields[0] == "cpu" or
                              (fields[0].startswith("cpu") and fields[0][3:].isdigit())):
            continue
        try:
            values = tuple(int(value) for value in fields[1:9])
            if len(values) < 4 or any(value < 0 for value in values):
                raise ValueError("invalid_cpu")
            counters[fields[0]] = values + (0,) * (8 - len(values))
        except ValueError:
            counters[fields[0]] = None
            _error(errors, "stat/" + fields[0], "invalid_cpu")
    if "cpu" not in counters:
        _error(errors, "stat/cpu", "missing_cpu")
    return counters


def _cpu_percentages(current, previous, errors, source):
    if current is None or previous is None:
        return (None,) * 4
    deltas = tuple(a - b for a, b in zip(current, previous))
    total = sum(deltas)
    if any(value < 0 for value in deltas) or total <= 0:
        _error(errors, source, "counter_reset_or_no_delta")
        return (None,) * 4
    busy = total - deltas[3] - deltas[4]
    return tuple(100.0 * value / total for value in (busy, deltas[4], deltas[5], deltas[6]))


class ProcSampler:
    """Single-threaded collector with injectable filesystem roots for offline tests."""

    def __init__(self, proc_root="/proc", sys_root="/sys", *, hz=None,
                 page_size=None, collector_pid=None, include_pss=False):
        self.proc_root, self.sys_root = Path(proc_root), Path(sys_root)
        self.hz = os.sysconf("SC_CLK_TCK") if hz is None else hz
        self.page_size = os.sysconf("SC_PAGE_SIZE") if page_size is None else page_size
        if self.hz <= 0 or self.page_size <= 0:
            raise ValueError("invalid_sysconf")
        self.collector_pid = os.getpid() if collector_pid is None else collector_pid
        self.include_pss = include_pss
        self._previous_ns = None
        self._previous_cpu = {}
        self._previous_processes = {}
        self._last_slow_ns = None
        self._last_pss_ns = None

    def _stat(self, entry, errors):
        text = _read(entry / "stat", errors, entry.name + "/stat")
        if text is None:
            return None
        try:
            result = parse_process_stat(text)
            if result["pid"] != int(entry.name):
                raise ValueError("pid_mismatch")
            return result
        except ValueError:
            _error(errors, entry.name + "/stat", "invalid_stat")
            return None

    def _role(self, entry, proc, errors):
        if proc["pid"] == self.collector_pid:
            return "collector"
        name = proc["name"].lower()
        for prefix, role in (("camera-daemon", "camera-daemon"), ("ai-runtime", "ai-runtime"),
                             ("hailort", "hailort"), ("isp_media", "video-service"),
                             ("ffmpeg", "video-service"), ("gst-launch", "video-service"),
                             ("mediamtx", "video-service"), ("rtsp", "video-service"),
                             ("video", "video-service"), ("v4l2", "video-service")):
            if name.startswith(prefix):
                return role
        if name.startswith("python"):
            try:
                with (entry / "cmdline").open("rb") as handle:
                    args = handle.read(MAX_CMDLINE_BYTES).decode("utf-8", "replace").split("\0")
                if any("perf_demo" in arg or "perf-demo" in arg for arg in args):
                    return "sdk"
            except OSError as exc:
                _error(errors, entry.name + "/cmdline", type(exc).__name__)
        return None

    def _discover(self, errors):
        processes, roles, children = {}, {}, {}
        for entry in _entries(self.proc_root, errors, "proc"):
            if not entry.name.isdigit():
                continue
            proc = self._stat(entry, errors)
            if proc is None:
                continue
            pid = proc["pid"]
            processes[pid] = proc
            roles[pid] = self._role(entry, proc, errors)
            children.setdefault(proc["ppid"], []).append(pid)
        pending = deque(pid for pid, role in roles.items() if role == "sdk")
        visited = set(pending)
        while pending:
            for child in children.get(pending.popleft(), []):
                if child not in visited:
                    visited.add(child)
                    roles[child] = roles[child] or "sdk-child"
                    pending.append(child)
        return [(proc, roles[pid]) for pid, proc in sorted(processes.items()) if roles[pid]]

    def _process_cpu(self, proc, elapsed, errors):
        key = (proc["pid"], proc["start_ticks"])
        previous = self._previous_processes.get(key)
        if previous is None:
            return None
        delta = (proc["utime"] - previous["utime"], proc["stime"] - previous["stime"])
        if elapsed is None or elapsed <= 0 or min(delta) < 0:
            _error(errors, str(proc["pid"]) + "/cpu", "counter_reset_or_clock")
            return None
        return 100.0 * sum(delta) / self.hz / elapsed

    def _process(self, proc, role, elapsed, slow, pss, errors):
        source = str(proc["pid"])
        entry = self.proc_root / source
        status = _read(entry / "status", errors, source + "/status")
        rss = _field(status, "VmRSS", errors, source, "kB")
        threads = _field(status, "Threads", errors, source)
        ctx = {key: _field(status, field, errors, source) for key, field in
               (("voluntary", "voluntary_ctxt_switches"),
                ("nonvoluntary", "nonvoluntary_ctxt_switches"))}
        fd_count = None
        if slow:
            try:
                fd_count = sum(1 for _ in (entry / "fd").iterdir())
            except OSError as exc:
                _error(errors, source + "/fd", type(exc).__name__)
        pss_bytes = None
        if pss:
            text = _read(entry / "smaps_rollup", errors, source + "/smaps_rollup")
            pss_bytes = _field(text, "Pss", errors, source, "kB")
        final = self._stat(entry, errors)
        if final is None or final["start_ticks"] != proc["start_ticks"]:
            _error(errors, source, "pid_changed")
            return None
        return {"pid": proc["pid"], "start_ticks": proc["start_ticks"],
                "name": proc["name"], "role": role,
                "cpu_pct_one_core": self._process_cpu(proc, elapsed, errors),
                "rss_bytes": rss, "num_threads": threads, "fd_count": fd_count,
                "ctxswitch": ctx, "pss_bytes": pss_bytes}

    def _sensor_value(self, path, errors, source, scale=1, signed=False):
        text = _read(path, errors, source)
        if text is None:
            return None
        try:
            value = int(text.strip())
            if not signed and value < 0:
                raise ValueError("negative_sensor")
            return value * scale
        except ValueError:
            _error(errors, source, "invalid_sensor")
            return None

    def _cma_pool(self, path, errors):
        # Linux mm/cma_debug.c exposes count, used, maxchunk in pages.
        values = {key: self._sensor_value(path / field, errors,
                  "cma/" + path.name + "/" + field, scale=self.page_size)
                  for key, field in (("total_bytes", "count"), ("used_bytes", "used"),
                                     ("maxchunk_bytes", "maxchunk"))}
        total, used = values["total_bytes"], values["used_bytes"]
        if total is not None and used is not None and used > total:
            _error(errors, "cma/" + path.name, "inconsistent_cma")
            used = None
        free = total - used if total is not None and used is not None else None
        return {**values, "used_bytes": used, "free_bytes": free}

    def _sensors(self, errors):
        thermal = self.sys_root / "class/thermal"
        zones = [path for path in _entries(thermal, errors, "thermal")
                 if path.name.startswith("thermal_zone")]
        temperatures = {path.name: self._sensor_value(path / "temp", errors,
                        "thermal/" + path.name, scale=0.001, signed=True) for path in zones}
        cpu_root = self.sys_root / "devices/system/cpu"
        cpus = [path for path in _entries(cpu_root, errors, "cpufreq")
                if path.name.startswith("cpu") and path.name[3:].isdigit()]
        frequencies = {path.name: self._sensor_value(path / "cpufreq/scaling_cur_freq",
                       errors, "cpufreq/" + path.name) for path in cpus}
        cma_root = self.sys_root / "kernel/debug/cma"
        pools = {path.name: self._cma_pool(path, errors)
                 for path in _entries(cma_root, errors, "cma")}
        return {"temperature_c": temperatures or None, "cpu_frequency_khz": frequencies or None,
                "cma_pools": pools or None}

    def _system(self, errors, elapsed):
        current = _cpu_counters(_read(self.proc_root / "stat", errors, "stat"), errors)
        previous = self._previous_cpu if elapsed is None or elapsed > 0 else {}
        if elapsed is not None and elapsed <= 0:
            _error(errors, "monotonic_ns", "nonpositive_elapsed")
        percentages = {name: _cpu_percentages(value, previous.get(name), errors, "stat/" + name)
                       for name, value in current.items()}
        busy, iowait, irq, softirq = percentages.get("cpu", (None,) * 4)
        cores = {name: values[0] for name, values in percentages.items() if name != "cpu"}
        return current, {"cpu_count": len(cores) or os.cpu_count(), "system_cpu_pct": busy,
                         "per_core_cpu_pct": cores or None, "system_iowait_pct": iowait,
                         "system_irq_pct": irq, "system_softirq_pct": softirq}

    def sample(self, run_id, phase, *, now_ns=None):
        """Read one snapshot; never sleeps or makes SDK/network requests."""
        now = time.monotonic_ns() if now_ns is None else now_ns
        utc_time = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        errors = {"items": [], "omitted": 0}
        elapsed = None if self._previous_ns is None else (now - self._previous_ns) / 1e9
        slow = self._last_slow_ns is None or now - self._last_slow_ns >= SLOW_INTERVAL_NS
        pss = self.include_pss and (self._last_pss_ns is None or
                                   now - self._last_pss_ns >= PSS_INTERVAL_NS)
        current_cpu, system = self._system(errors, elapsed)
        mem = _read(self.proc_root / "meminfo", errors, "meminfo")
        available = _field(mem, "MemAvailable", errors, "meminfo", "kB")
        processes, current_processes = [], {}
        for proc, role in self._discover(errors):
            result = self._process(proc, role, elapsed, slow, pss, errors)
            if result is not None:
                processes.append(result)
                current_processes[(proc["pid"], proc["start_ticks"])] = proc
        sensors = self._sensors(errors) if slow else {
            "temperature_c": None, "cpu_frequency_khz": None, "cma_pools": None}
        cma = {key: _field(mem, field, errors, "meminfo", "kB") if slow else None
               for key, field in (("cma_total_bytes", "CmaTotal"), ("cma_free_bytes", "CmaFree"))}
        self._previous_ns, self._previous_cpu = now, current_cpu
        self._previous_processes = current_processes
        self._last_slow_ns = now if slow else self._last_slow_ns
        self._last_pss_ns = now if pss else self._last_pss_ns
        return {"type": "resources", "monotonic_ns": now, "utc_time": utc_time,
                "run_id": run_id, "phase": phase, **system, "mem_available_bytes": available,
                "processes": processes, **sensors, **cma,
                "slow_sample_monotonic_ns": now if slow else None,
                "pss_sample_monotonic_ns": now if pss else None,
                "errors": errors["items"], "errors_omitted": errors["omitted"]}


class JsonlWriter:
    """Exclusive append writer; protect regular-file size and filesystem reserve."""

    def __init__(self, path, *, max_bytes=DEFAULT_MAX_BYTES,
                 min_free_bytes=DEFAULT_MIN_FREE_BYTES):
        if max_bytes <= 0 or min_free_bytes < 0:
            raise ValueError("invalid_output_limits")
        self.path = Path(path)
        self.max_bytes, self.min_free_bytes = max_bytes, min_free_bytes
        self.fd = None

    def __enter__(self):
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND |
                     os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("output_not_regular")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                raise OSError("incomplete_output")
        except OSError:
            os.close(fd)
            raise
        self.fd = fd
        return self

    def write(self, record):
        """Write one complete UTF-8 JSON line or raise; no unbounded buffering."""
        data = (json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
                + "\n").encode("utf-8")
        start_size = os.fstat(self.fd).st_size
        if start_size + len(data) > self.max_bytes:
            raise OSError("output_limit")
        space = os.fstatvfs(self.fd)
        if space.f_bavail * space.f_frsize - len(data) < self.min_free_bytes:
            raise OSError("disk_reserve")
        remaining = memoryview(data)
        try:
            while remaining:
                written = os.write(self.fd, remaining)
                if written <= 0:
                    raise OSError("short_write")
                remaining = remaining[written:]
        except OSError:
            # Exclusive flock makes rolling back this writer's incomplete line safe.
            os.ftruncate(self.fd, start_size)
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            os.fsync(self.fd)
        finally:
            os.close(self.fd)
            self.fd = None


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _label(value):
    if not value.strip() or len(value) > 200 or any(ord(char) < 32 for char in value):
        raise argparse.ArgumentTypeError("must be 1..200 characters without control characters")
    return value


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=_positive_float, default=1.0)
    parser.add_argument("--duration", type=_positive_float, required=True)
    parser.add_argument("--run-id", required=True, type=_label)
    parser.add_argument("--phase", required=True, type=_label)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--pss", action="store_true", help="sample smaps_rollup PSS every 30 seconds")
    args = parser.parse_args(argv)
    if args.max_bytes <= 0 or args.min_free_bytes < 0:
        parser.error("max-bytes must be positive and min-free-bytes nonnegative")
    return args


def _wait_until(deadline, should_stop):
    # A signal handler must not acquire Event/Condition locks: it can interrupt
    # the same thread while that lock is held. Poll only the signal-owned flag.
    while not should_stop():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, STOP_POLL_SECONDS))


def _run(args, should_stop):
    collector = ProcSampler(include_pss=args.pss)
    with JsonlWriter(args.output, max_bytes=args.max_bytes, min_free_bytes=args.min_free_bytes) as writer:
        start = time.monotonic()
        deadline, next_sample = start + args.duration, start
        while not should_stop() and time.monotonic() < deadline:
            writer.write(collector.sample(args.run_id, args.phase))
            now = time.monotonic()
            # Skip missed slots instead of a catch-up burst or accumulating drift.
            next_sample += args.interval
            if next_sample <= now:
                next_sample = start + (math.floor((now - start) / args.interval) + 1) * args.interval
            _wait_until(min(next_sample, deadline), should_stop)
    return 0


def main(argv=None):
    """CLI entry point; SIGINT/SIGTERM are clean exits, I/O errors return 1."""
    args = _arguments(argv)
    stopped = False

    def stop_handler(*_):
        nonlocal stopped
        stopped = True

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in handlers:
            signal.signal(sig, stop_handler)
        return _run(args, lambda: stopped)
    except (OSError, ValueError) as exc:
        safe_codes = {"output_limit", "disk_reserve", "short_write", "output_not_regular", "incomplete_output"}
        code = str(exc) if str(exc) in safe_codes else type(exc).__name__
        print(f"proc_sampler: {code}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
