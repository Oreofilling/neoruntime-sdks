#!/usr/bin/env python3
"""Run ONE guarded perf-demo phase; never operate platform services.

Creates a NEW private output directory. Only app.py is a child (new process
session); ProcSampler runs inside this controller and labels it collector.
Raw contracts are a_result/b_infer/b_compose success, encoded_packet,
status.snapshot.recorder, run_exit and recorder_final. Progress must increase;
replays/regressions cannot refresh health. Producer-time gaps are latched.
All required channels must be ready within 30s, before the FULL warmup starts.
After readiness, 10s without progress is fatal, even if a later batch recovers.

A fresh CPU baseline starts measurement after warmup. Actual sample-aligned
boundaries provide at least duration seconds (up to one poll of rounding at
both boundaries). At least 90% valid CPU interval coverage is required; a 3s
missing run fails. Optional CMA/temperature errors do not invalidate CPU.
App duration includes startup/scheduling allowance; the controller sends TERM
at measurement completion. Only its own pinned process group is signalled,
with 15s TERM grace; forced KILL is failure. stdout uses a bounded nonblocking
PIPE sink (64 MiB default) with a disk reserve before EVERY write (64 MiB).
SIGTERM/SIGINT cancel the phase, clean up, and return nonzero. No next phase,
service restart, model unregister, sysfs writes or device installation occurs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from proc_sampler import JsonlWriter, ProcSampler, parse_process_stat

APP_PATH = Path(__file__).resolve().with_name("app.py")
MAX_DURATION = 86000
INIT_GRACE_NS = 30_000_000_000
STALE_NS = 10_000_000_000
TERM_GRACE = 30  # > app teardown worst case (~15s joins + 5s recorder close): a shorter
                 # grace SIGKILLs the child before its terminal records are written
READ_LIMIT = 8 * 1024 * 1024
LINE_LIMIT = 256 * 1024
POLL_SECONDS = 1.0
SERVICE_PREFIXES = {"camera-daemon": "camera", "ai-runtime": "ai",
                    "hailort": "hailort", "isp_media": "isp"}


class PhaseFailure(RuntimeError):
    """Safe machine-readable failure reason, never an external error message."""


def platform_fingerprint(proc_root):
    """Discover all required platform service identities without cmdline output."""
    found = {}
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            item = parse_process_stat((entry / "stat").read_text())
        except (OSError, ValueError):
            continue  # Unrelated /proc tasks commonly vanish while enumerating.
        role = next((role for prefix, role in SERVICE_PREFIXES.items()
                     if item["name"].startswith(prefix)), None)
        if role:
            if item["pid"] != int(entry.name):
                raise PhaseFailure("platform_pid_mismatch")
            found[item["pid"]] = {"name": item["name"], "role": role,
                                  "start_ticks": item["start_ticks"]}
    if {item["role"] for item in found.values()} != set(SERVICE_PREFIXES.values()):
        raise PhaseFailure("platform_missing")
    return found


def check_platform(proc_root, baseline, expected):
    """Reject missing/new/reused PIDs; supplied identities cannot be rebased."""
    current = platform_fingerprint(proc_root)
    for pid, ticks in expected.items():
        if pid not in current or current[pid]["start_ticks"] != ticks:
            raise PhaseFailure("expected_platform_mismatch")
    if current != baseline:
        raise PhaseFailure("platform_changed")
    return current


class MemoryGuard:
    def __init__(self):
        self.low_count = 0

    def check(self, proc_root):
        """Three consecutive samples strictly below max(512 MiB, 5%) fail."""
        try:
            fields = {}
            for line in (Path(proc_root) / "meminfo").read_text().splitlines():
                parts = line.split()
                if parts and parts[0] in ("MemTotal:", "MemAvailable:"):
                    if len(parts) != 3 or parts[2] != "kB":
                        raise ValueError()
                    fields[parts[0][:-1]] = int(parts[1]) * 1024
            total, available = fields["MemTotal"], fields["MemAvailable"]
            if total <= 0 or available < 0 or available > total:
                raise ValueError()
        except (OSError, KeyError, ValueError):
            raise PhaseFailure("memory_unavailable") from None
        threshold = max(512 * 1024 * 1024, math.ceil(total / 20))
        self.low_count = self.low_count + 1 if available < threshold else 0
        if self.low_count >= 3:
            raise PhaseFailure("low_memory")
        return {"available_bytes": available, "total_bytes": total,
                "threshold_bytes": threshold, "low_consecutive": self.low_count}


class EventReader:
    """Bounded incremental JSONL read; tolerate only an unfinished tail line."""

    def __init__(self, path):
        self.path, self.offset, self.identity, self.pending = Path(path), 0, None, b""

    def read(self):
        try:
            handle = self.path.open("rb")
        except FileNotFoundError:
            if self.identity is not None:
                raise PhaseFailure("samples_disappeared") from None
            return []
        with handle:
            st = os.fstat(handle.fileno())
            identity = (st.st_dev, st.st_ino)
            if self.identity is not None and (self.identity != identity or st.st_size < self.offset):
                raise PhaseFailure("samples_replaced_or_truncated")
            handle.seek(self.offset)
            data = handle.read(READ_LIMIT + 1)
        if len(data) > READ_LIMIT:
            raise PhaseFailure("samples_reader_backlog")
        lines = (self.pending + data).split(b"\n")
        if any(len(line) > LINE_LIMIT for line in lines):
            raise PhaseFailure("samples_line_limit")
        self.identity, self.offset, self.pending = identity, self.offset + len(data), lines[-1]
        try:
            return [json.loads(line) for line in lines[:-1]]
        except (ValueError, UnicodeError):
            raise PhaseFailure("invalid_samples_json") from None


class EventMonitor:
    def __init__(self, run_id, phase, chains, start_ns, end_ns=None):
        self.run_id, self.phase, self.start_ns = run_id, phase, start_ns
        self.end_ns, self.failure = end_ns, None
        self.health_start_ns = start_ns + INIT_GRACE_NS
        self.required = {"encoded:main", "encoded:sub"}
        if chains in ("a", "ab"):
            self.required |= {"a_result"}
        if chains in ("b", "ab"):
            self.required |= {"b_infer", "b_compose"}
        self.last_success, self.identities = {}, {}
        self.saw_run_exit, self.saw_recorder_final = False, False
        self.run_exit_ns = None
        self.model_evidence = {"model_kept": None, "model_kept_reason": "not_reported",
                               "model_ownership": "unknown"}

    def _recorder(self, info, final=False):
        if not isinstance(info, dict) or type(info.get("dropped")) is not int:
            raise PhaseFailure("recorder_metadata_missing")
        if info["dropped"] != 0 or info.get("error") is not None:
            raise PhaseFailure("recorder_loss_or_error")
        if not final and info.get("writer_alive") is False:
            raise PhaseFailure("recorder_writer_stopped")
        if final and info.get("exit_code") != 0:
            raise PhaseFailure("recorder_failed_exit")

    def inspect_status(self, snapshot):
        """Also inspect the atomic status file if a broken recorder cannot emit."""
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("recorder"), dict):
            raise PhaseFailure("recorder_metadata_missing")
        info = snapshot["recorder"]
        final = snapshot.get("exit_status")
        if final is not None:
            if not isinstance(final, dict):
                raise PhaseFailure("invalid_exit_status")
            self._recorder({**info, "exit_code": final.get("exit_code")}, final=True)
        else:
            self._recorder(info)

    def _terminal_or_status(self, row):
        kind = row["type"]
        if kind == "status":
            self.inspect_status(row.get("snapshot"))
        elif kind == "recorder_final":
            self._recorder(row, final=True)
            self.saw_recorder_final = True
        elif kind == "watcher_exit":
            stream = row.get("stream_id")
            if stream in ("main", "sub") and row.get("error"):
                raise PhaseFailure("watcher_failed:" + stream)
        elif kind == "run_exit":
            self.model_evidence = {key: row.get(key, default)
                                   for key, default in self.model_evidence.items()}
            if row.get("exit_code") != 0 or row.get("cleanup_errors") or row.get("alive_threads"):
                raise PhaseFailure("demo_cleanup_failed")
            if "snapshot" in row:
                snapshot = row["snapshot"]
                self._recorder(snapshot.get("recorder") if isinstance(snapshot, dict) else None)
            self.saw_run_exit = True
            self.run_exit_ns = row["monotonic_ns"]

    def consume(self, records, now_ns):
        """Use producer monotonic time; stale backlog cannot refresh success."""
        if self.failure:
            raise PhaseFailure(self.failure)
        for row in records:
            if not isinstance(row, dict) or row.get("run_id") != self.run_id or row.get("phase") != self.phase:
                raise PhaseFailure("samples_identity_mismatch")
            stamp = row.get("monotonic_ns")
            if type(stamp) is not int or not self.start_ns <= stamp <= now_ns + 1_000_000_000:
                raise PhaseFailure("invalid_event_clock")
            if not isinstance(row.get("type"), str):
                raise PhaseFailure("invalid_event_type")
            self._terminal_or_status(row)
            kind = row["type"]
            if kind in ("a_result", "b_infer", "b_compose"):
                if type(row.get("success")) is not bool:
                    raise PhaseFailure("invalid_success_flag")
                if not row["success"]:
                    continue
                identity = (row.get("source_frame_id"), row.get("source_timestamp_ns"))
            elif kind == "encoded_packet":
                kind = "encoded:" + str(row.get("stream_id"))
                identity = (row.get("packet_sequence"), row.get("pts_ns"))
            else:
                continue
            if kind not in self.required:
                continue
            self._note_progress(kind, stamp, identity)

    def _note_progress(self, kind, stamp, identity):
        progress = next((value for value in identity if type(value) is int and value >= 0), None)
        if progress is None:
            raise PhaseFailure("missing_source_identity")
        if self.end_ns is not None and stamp > self.end_ns:
            return
        if progress <= self.identities.get(kind, -1):
            return  # Strict progress, O(channels) storage: 1,2,1,2 cannot heal a stall.
        previous = self.last_success.get(kind, 0)
        if stamp < previous:
            raise PhaseFailure("event_time_regression")
        if stamp - max(previous, self.health_start_ns) >= STALE_NS:
            self.failure = "stale:" + kind
            raise PhaseFailure(self.failure)  # Latch historical gaps before accepting recovery.
        self.last_success[kind], self.identities[kind] = stamp, progress

    def check(self, now_ns):
        if self.failure:
            raise PhaseFailure(self.failure)
        for kind in sorted(self.required):
            baseline = max(self.health_start_ns, self.last_success.get(kind, 0))
            if now_ns - baseline >= STALE_NS:
                self.failure = "stale:" + kind
                raise PhaseFailure(self.failure)

    def finish(self, steady_end_ns=None):
        """Require real events, clean closure and health at the measurement boundary."""
        if steady_end_ns is not None:
            self.check(steady_end_ns)
        if self.required - self.last_success.keys():
            raise PhaseFailure("missing_success_evidence")
        if not self.saw_run_exit or not self.saw_recorder_final:
            raise PhaseFailure("missing_terminal_evidence")


class CpuCoverage:
    """Only intervals after an explicit steady baseline count; >=90% is required."""
    def __init__(self):
        self.previous_ns = None
        self.covered_ns = self.observed_ns = self.gap_ns = 0

    def begin(self, sample):
        now = sample.get("monotonic_ns")
        if type(now) is not int or now < 0:
            raise PhaseFailure("cpu_measurement_invalid")
        self.previous_ns = now  # Discard the warmup-crossing delta.

    def add(self, sample):
        now = sample.get("monotonic_ns")
        if type(now) is not int or self.previous_ns is None or now <= self.previous_ns:
            raise PhaseFailure("cpu_measurement_invalid")
        delta = now - self.previous_ns
        value = sample.get("system_cpu_pct")
        valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100
        valid = valid and not any(e.get("source", "").startswith("stat") for e in sample.get("errors", []))
        self.previous_ns, self.observed_ns = now, self.observed_ns + delta
        self.covered_ns += delta if valid else 0
        self.gap_ns = 0 if valid else self.gap_ns + delta
        if self.gap_ns >= 3_000_000_000:
            raise PhaseFailure("cpu_measurement_invalid")

    def report(self):
        return {"covered_ns": self.covered_ns, "observed_ns": self.observed_ns,
                "missing_ns": self.observed_ns - self.covered_ns,
                "coverage_pct": 100 * self.covered_ns / self.observed_ns if self.observed_ns else 0,
                "minimum_coverage_pct": 90, "maximum_missing_run_ns": 3_000_000_000,
                "boundary_policy": "new_baseline_after_warmup"}

    def finish(self):
        if not self.covered_ns or self.covered_ns < .9 * self.observed_ns:
            raise PhaseFailure("cpu_measurement_invalid")
        return self.report()


class BoundedLog:
    """Exclusive raw-byte output with a hard limit and per-write disk reserve."""
    def __init__(self, path, *, max_bytes, min_free_bytes):
        if max_bytes <= 0 or min_free_bytes < 0:
            raise ValueError("invalid_log_budget")
        self.path, self.max_bytes, self.min_free_bytes = path, max_bytes, min_free_bytes
        self.fd, self.written = None, 0

    def __enter__(self):
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        return self

    def write(self, data):
        remaining = memoryview(data)
        while remaining:
            if self.written + len(remaining) > self.max_bytes:
                raise PhaseFailure("stdout_limit")
            space = os.fstatvfs(self.fd)
            if space.f_bavail * space.f_frsize - len(remaining) < self.min_free_bytes:
                raise PhaseFailure("stdout_disk_reserve")
            count = os.write(self.fd, remaining)
            if count <= 0:
                raise OSError("stdout_short_write")
            self.written += count
            remaining = remaining[count:]

    def __exit__(self, *_):
        try:
            os.fsync(self.fd)
        finally:
            os.close(self.fd)


def child_exit_code(child):
    """Observe, never reap: the retained leader pins its PID/PGID until cleanup."""
    if child.returncode is not None:
        raise PhaseFailure("child_identity_lost")
    try:
        info = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        raise PhaseFailure("child_identity_lost") from None
    if info is None:
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def _live_group(pgid):
    # A pinned zombie leader keeps the PGID safe, but is not a live descendant.
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                text = (path / "stat").read_text()
                fields = text[text.rfind(")") + 1:].split()
                if int(fields[2]) == pgid and fields[0] not in ("Z", "X", "x"):
                    return True
            except (OSError, ValueError, IndexError):
                continue
    return False


def _cleanup_pump(pump, result):
    try:
        pump()
    except Exception as exc:
        result["pump_error"] = type(exc).__name__  # Still finish TERM/KILL/reap.


def stop_child(child, grace=TERM_GRACE, pump=lambda: None):
    """Keep WNOWAIT ownership through the last group signal; reap only at end."""
    child_exit_code(child)  # Refuse ALL group operations if someone already reaped it.
    result = {"term_sent": False, "kill_sent": False, "returncode": None}
    for sig, allowance, key in ((signal.SIGTERM, grace, "term_sent"),
                                (signal.SIGKILL, 5, "kill_sent")):
        if child_exit_code(child) is not None and not _live_group(child.pid):
            break
        try:
            os.killpg(child.pid, sig)
            result[key] = True
        except ProcessLookupError:
            break
        deadline = time.monotonic() + allowance
        while time.monotonic() < deadline:
            _cleanup_pump(pump, result)  # Never let a broken PIPE abort cleanup.
            if child_exit_code(child) is not None and not _live_group(child.pid):
                break
            time.sleep(0.01)
    _cleanup_pump(pump, result)
    result["returncode"] = child.wait(timeout=5)  # No group signals after this point.
    return result


def demo_command(args):
    """No shell or arbitrary extra flags; model retention is unconditional."""
    command = [sys.executable, str(APP_PATH), "--model-path", args.model_path,
               "--model-id", args.model_id, "--chains", args.chains,
               "--duration", str(args.warmup + args.duration + INIT_GRACE_NS / 1e9 + 3 * POLL_SECONDS), "--keep-model",
               "--json-path", str(args.output_dir / "status.json"),
               "--samples-path", str(args.output_dir / "samples.jsonl"),
               "--run-id", args.run_id, "--phase", args.phase,
               "--session-id", args.run_id, "--watch-encoded", "main,sub",
               "--a-fps", str(args.a_fps), "--b-fps", str(args.b_fps),
               "--publish-hz", str(args.publish_hz)]
    for flag in ("reuse_model", "no_metrics_overlay"):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    if getattr(args, "variant", None):
        # Without the passthrough a variant-dependent HEF registers as
        # model_variant=None: inference "succeeds" with zero detections
        # and the phase passes with misleading numbers.
        command += ["--variant", args.variant]
    return command


def _manifest_write(path, data):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


class _Phase:
    def __init__(self, args, proc_root, cancelled):
        self.args, self.proc_root, self.cancelled = args, proc_root, cancelled
        self.child, self.monitor, self.collector = None, None, None
        self.reader = EventReader(args.output_dir / "samples.jsonl")
        self.memory, self.cpu = MemoryGuard(), CpuCoverage()
        self.baseline = None
        self.log, self.log_failed = None, False
        self.state = {"run_id": args.run_id, "phase": args.phase, "chains": args.chains,
                      "status": "running", "reason": None, "errors": [],
                      "utc_start": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "phase_start_monotonic_ns": None, "ready_monotonic_ns": None,
                      "planned_steady_start_monotonic_ns": None, "planned_steady_end_monotonic_ns": None,
                      "steady_start_monotonic_ns": None,
                      "steady_end_monotonic_ns": None, "end_monotonic_ns": None,
                      "returncodes": {"app": None}, "cleanup": {},
                      "platform_initial": None, "platform_final": None,
                      "expected_platform": args.expect_platform,
                      "warmup_seconds": args.warmup, "duration_seconds": args.duration,
                      "model_path": args.model_path, "model_id": args.model_id,
                      "keep_model": True, "reuse_model": args.reuse_model,
                      "a_fps": args.a_fps, "b_fps": args.b_fps, "publish_hz": args.publish_hz,
                      "no_metrics_overlay": args.no_metrics_overlay,
                      "stdout_limit_bytes": args.max_log_bytes, "min_free_bytes": args.min_free_bytes}

    def _fail(self, exc):
        reason = str(exc) if isinstance(exc, PhaseFailure) else type(exc).__name__
        self.state["reason"] = self.state["reason"] or reason
        self.state["errors"].append(reason)

    def _persist(self):
        _manifest_write(self.args.output_dir / "manifest.json", self.state)

    def _start(self, writer):
        self.baseline = platform_fingerprint(self.proc_root)
        check_platform(self.proc_root, self.baseline, self.args.expect_platform)
        self.state["platform_initial"] = self.baseline
        self.state["memory_initial"] = MemoryGuard().check(self.proc_root)
        self.collector = ProcSampler()
        writer.write(self.collector.sample(self.args.run_id, self.args.phase))
        if self.cancelled():
            raise PhaseFailure("cancelled")
        start = time.monotonic_ns()
        self.state["phase_start_monotonic_ns"] = start
        self.monitor = EventMonitor(self.args.run_id, self.args.phase, self.args.chains, start)
        self.child = subprocess.Popen(demo_command(self.args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True, bufsize=0)
        try:
            os.set_blocking(self.child.stdout.fileno(), False)
        except OSError:
            self.child.stdout.close()  # Cleanup must never pump an uninitialized blocking pipe.
            raise
        self.state["app_pid"] = self.child.pid
        self.state["app_pgid"] = self.child.pid
        self._persist()

    def _pump_output(self, cleaning=False):
        if self.child is None or self.child.stdout.closed:
            return
        for _ in range(64):  # Bounded work even if a producer floods continuously.
            try:
                chunk = os.read(self.child.stdout.fileno(), 65536)
            except BlockingIOError:
                return
            if not chunk:
                self.child.stdout.close()
                return
            if not self.log_failed:
                try:
                    self.log.write(chunk)
                except Exception as exc:
                    self.log_failed = True
                    if not cleaning:
                        raise
                    self._fail(exc)

    def _status_file(self):
        try:
            with (self.args.output_dir / "status.json").open("rb") as handle:
                data = handle.read(LINE_LIMIT + 1)
        except FileNotFoundError:
            return  # First status tick is later than app startup.
        if len(data) > LINE_LIMIT:
            raise PhaseFailure("status_size_limit")
        try:
            snapshot = json.loads(data)
        except (ValueError, UnicodeError):
            raise PhaseFailure("invalid_status_json") from None
        self.monitor.inspect_status(snapshot)

    def _tick(self, writer):
        self._pump_output()
        sample = self.collector.sample(self.args.run_id, self.args.phase)
        target_end = self.state["planned_steady_end_monotonic_ns"]
        if target_end is not None and sample["monotonic_ns"] >= target_end:
            self.monitor.end_ns = sample["monotonic_ns"]  # Freeze before consuming newer events.
        writer.write(sample)
        self.state["platform_final"] = check_platform(self.proc_root, self.baseline, self.args.expect_platform)
        self.state["memory_last"] = self.memory.check(self.proc_root)
        rows = self.reader.read()
        now = time.monotonic_ns()
        self.monitor.consume(rows, now)
        self._status_file()
        code = child_exit_code(self.child)
        if code is not None:
            self.state["app_exit_observed_monotonic_ns"] = now
            raise PhaseFailure("app_failed_exit" if code != 0 else "app_early_exit")
        if self.state["ready_monotonic_ns"] is None:
            if now - self.state["phase_start_monotonic_ns"] >= INIT_GRACE_NS:
                raise PhaseFailure("ready_timeout")
            self.monitor.check(now)  # A stale backlog cannot satisfy the readiness barrier.
            if self.monitor.required <= self.monitor.last_success.keys():
                for kind in sorted(self.monitor.required):
                    if now - self.monitor.last_success[kind] >= STALE_NS:
                        raise PhaseFailure("stale:" + kind)
                self.state["ready_monotonic_ns"] = now
                self.monitor.health_start_ns = now
                self.state["planned_steady_start_monotonic_ns"] = now + int(self.args.warmup * 1e9)
        self.monitor.check(self.monitor.end_ns if self.monitor.end_ns is not None else now)
        limit_ns = INIT_GRACE_NS + int((self.args.warmup + self.args.duration + 3 * POLL_SECONDS) * 1e9)
        if now - self.state["phase_start_monotonic_ns"] > limit_ns:
            raise PhaseFailure("app_timeout")
        return self._measurement_tick(sample)

    def _measurement_tick(self, sample):
        target = self.state["planned_steady_start_monotonic_ns"]
        stamp = sample["monotonic_ns"]
        if target is None or stamp < target:
            return False
        if self.cpu.previous_ns is None:
            self.cpu.begin(sample)
            self.state["steady_start_monotonic_ns"] = stamp
            self.state["planned_steady_end_monotonic_ns"] = stamp + int(self.args.duration * 1e9)
            self._persist()
            return False
        self.cpu.add(sample)
        if stamp < self.state["planned_steady_end_monotonic_ns"]:
            return False
        self.state["steady_end_monotonic_ns"] = stamp
        self.monitor.end_ns = stamp
        self.state["cpu_measurement"] = self.cpu.finish()
        self.state["normal_term_requested"] = True
        return True

    def _loop(self, writer):
        self._start(writer)
        next_tick = time.monotonic()
        while True:
            if self.cancelled():
                raise PhaseFailure("cancelled")
            if self._tick(writer):
                return
            next_tick = max(next_tick + POLL_SECONDS, time.monotonic())
            while time.monotonic() < next_tick and not self.cancelled():
                self._pump_output()
                time.sleep(min(0.1, max(0, next_tick - time.monotonic())))

    def _cleanup(self, writer):
        if self.child is not None:
            try:
                cleanup = stop_child(self.child, pump=lambda: self._pump_output(cleaning=True))
                self.state["cleanup"] = cleanup
                self.state["returncodes"]["app"] = cleanup["returncode"]
                self.state.setdefault("app_exit_observed_monotonic_ns", time.monotonic_ns())
                if cleanup["returncode"] != 0 and not self.state["reason"]:
                    self._fail(PhaseFailure("app_failed_exit"))
                if cleanup.get("pump_error"):
                    self._fail(PhaseFailure("stdout_cleanup_error"))
                if cleanup["kill_sent"]:
                    self._fail(PhaseFailure("forced_kill"))
            except Exception as exc:
                self._fail(exc)
            finally:
                self.child.stdout.close()
                self.state["stdout_bytes"] = self.log.written
                self.state["stdout_truncated"] = self.log_failed
        if self.collector is not None:
            try:
                writer.write(self.collector.sample(self.args.run_id, self.args.phase))
            except Exception as exc:
                self._fail(exc)
        if self.monitor is not None:
            try:
                self.monitor.consume(self.reader.read(), time.monotonic_ns())
                self.monitor.consume(self.reader.read(), time.monotonic_ns())
                self._status_file()
                if self.reader.pending:
                    raise PhaseFailure("incomplete_samples_tail")
                self.monitor.finish(steady_end_ns=self.state["steady_end_monotonic_ns"])
                end = self.state["steady_end_monotonic_ns"]
                if end is None or self.monitor.run_exit_ns < end:
                    raise PhaseFailure("app_early_exit")
            except Exception as exc:  # Cleanup must continue even for a monitor programming error.
                self._fail(exc)
            self.state["last_success_monotonic_ns"] = self.monitor.last_success
            self.state["model_observed"] = self.monitor.model_evidence
        try:
            if self.baseline is not None:
                self.state["platform_final"] = check_platform(self.proc_root, self.baseline, self.args.expect_platform)
        except (OSError, PhaseFailure) as exc:
            self._fail(exc)

    def run(self):
        self._persist()
        try:
            with BoundedLog(self.args.output_dir / "app.stdout.log", max_bytes=self.args.max_log_bytes,
                            min_free_bytes=self.args.min_free_bytes) as self.log, \
                    JsonlWriter(self.args.output_dir / "resources.jsonl", min_free_bytes=self.args.min_free_bytes) as writer:
                try:
                    self._loop(writer)
                except Exception as exc:  # Never let an unexpected monitor failure mark PASS.
                    self._fail(exc)
                finally:
                    self._cleanup(writer)
        except Exception as exc:
            self._fail(exc)
        finally:
            self.state["end_monotonic_ns"] = time.monotonic_ns()
            self.state["cpu_measurement"] = self.cpu.report()
            if self.state["steady_start_monotonic_ns"] is not None and self.state["steady_end_monotonic_ns"] is None:
                self.state["steady_end_monotonic_ns"] = self.cpu.previous_ns
            self._check_cancel()
            self.state["status"] = "fail" if self.state["reason"] else "pass"
            self._persist()
            if self._check_cancel():  # Includes cancellation during fsync/atomic replace.
                self.state["status"] = "fail"
                self._persist()
        return int(self.state["status"] != "pass")

    def _check_cancel(self):
        if self.cancelled() and "cancelled" not in self.state["errors"]:
            self._fail(PhaseFailure("cancelled"))
            return True
        return False


def run_phase(args, *, proc_root="/proc", cancelled=lambda: False):
    """Run exactly one phase, writing evidence even when preflight fails."""
    args.output_dir.mkdir(mode=0o700)  # Existing dirs/symlinks fail; never overwrite a prior run.
    return _Phase(args, Path(proc_root), cancelled).run()


def _label(value):
    if not value.strip() or len(value) > 128 or any(ord(c) < 32 for c in value):
        raise argparse.ArgumentTypeError("must be 1..128 printable characters")
    return value


def _variant(value):
    # Same charset as the demo app's --variant: a dlsym postprocess
    # symbol name, passed through to the demo's register_model.
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value or ""):
        raise argparse.ArgumentTypeError(
            "must use 1..128 letters/digits/underscore/dot/colon/hyphen")
    return value


def _expected(value):
    if not re.fullmatch(r"[1-9][0-9]*:[0-9]+", value):
        raise argparse.ArgumentTypeError("expected PID:START_TICKS")
    return tuple(int(part) for part in value.split(":"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=lambda value: Path(value).absolute())
    parser.add_argument("--phase", required=True, type=_label)
    parser.add_argument("--chains", choices=("none", "a", "b", "ab"), required=True)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--warmup", type=float, default=0)
    parser.add_argument("--model-path", required=True, type=_label)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-id", type=_label, default=uuid.uuid4().hex)
    parser.add_argument("--reuse-model", action="store_true")
    parser.add_argument("--expect-platform", action="append", type=_expected, default=[])
    parser.add_argument("--a-fps", type=int, default=10)
    parser.add_argument("--b-fps", type=float, default=0)
    parser.add_argument("--publish-hz", type=float, default=40)
    parser.add_argument("--no-metrics-overlay", action="store_true")
    parser.add_argument("--variant", default=None, type=_variant)
    parser.add_argument("--max-log-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--min-free-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args(argv)
    if args.max_log_bytes <= 0 or args.min_free_bytes < 0:
        parser.error("invalid output budgets")
    for key, low, high in (("duration", 0, MAX_DURATION), ("warmup", 0, MAX_DURATION),
                           ("a_fps", 1, 120), ("b_fps", 0, 120), ("publish_hz", 1, 120)):
        value = getattr(args, key)
        if not math.isfinite(value) or not low <= value <= high or (key == "duration" and value == 0):
            parser.error("invalid " + key)
    if args.warmup + args.duration + INIT_GRACE_NS / 1e9 + TERM_GRACE + 10 + 3 * POLL_SECONDS > MAX_DURATION:
        parser.error("warmup + duration + cleanup allowance must not exceed 86000 seconds")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", args.model_id):
        parser.error("invalid model-id")
    if len(dict(args.expect_platform)) != len(args.expect_platform):
        parser.error("duplicate expected PID")
    args.expect_platform = dict(args.expect_platform)
    return args


def main(argv=None):
    args = parse_args(argv)
    cancelled = False

    def cancel(*_):
        nonlocal cancelled
        cancelled = True

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in handlers:
            signal.signal(sig, cancel)
        return run_phase(args, cancelled=lambda: cancelled)
    except (OSError, ValueError) as exc:
        print("phase_runner: " + type(exc).__name__, file=sys.stderr)
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
