"""Bounded, append-only JSONL recorder; producers never do file IO.

Only small JSON-compatible metadata belongs here, never pixels/results.
Loss is visible in status snapshots and a terminal recorder_final record.
Durations use monotonic_ns; source timestamps require the same device clock.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time

logger = logging.getLogger("perf_demo.recorder")


class SampleRecorder:
    def __init__(self, path: str, *, run_id: str, phase: str,
                 capacity: int = 4096, batch_size: int = 64,
                 max_bytes: int = 1024 * 1024 * 1024,
                 free_reserve_bytes: int = 64 * 1024 * 1024) -> None:
        if not path or not run_id or not phase:
            raise ValueError("path, run_id and phase must not be empty")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (capacity, batch_size, max_bytes)):
            raise ValueError("capacity, batch_size and max_bytes must be positive integers")
        if isinstance(free_reserve_bytes, bool) or not isinstance(free_reserve_bytes, int) \
                or free_reserve_bytes < 0:
            raise ValueError("free_reserve_bytes must be a non-negative integer (0 disables)")
        self.run_id, self.phase = run_id, phase
        self._queue = queue.Queue(maxsize=capacity)
        self._batch_size = batch_size
        self._max_bytes = max_bytes
        self._free_reserve = free_reserve_bytes
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._counts = dict(accepted=0, written=0, dropped=0, invalid=0, error=None)
        self._exit_code = 0
        self._file = open(path, "a", encoding="utf-8")
        self._bytes = os.fstat(self._file.fileno()).st_size
        self._dir = os.path.dirname(os.path.abspath(path))
        self._thread = threading.Thread(target=self._run, name="sample-writer", daemon=True)
        self._thread.start()

    def _envelope(self, kind: str, fields: dict) -> dict:
        return {**fields, "type": kind, "run_id": self.run_id,
                "phase": self.phase, "monotonic_ns": time.monotonic_ns(),
                "source_frame_id": fields.get("source_frame_id"),
                "stream_id": fields.get("stream_id")}

    def emit(self, kind: str, **fields) -> bool:
        if not isinstance(kind, str) or not kind or len(kind) > 80:
            raise ValueError("record type must be a nonempty short string")
        record = self._envelope(kind, fields)
        with self._lock:
            if self._stop_event.is_set() or self._counts["error"]:
                self._counts["dropped"] += 1
                return False
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._counts["dropped"] += 1
                return False
            self._counts["accepted"] += 1
        return True

    def snapshot(self) -> dict:
        with self._lock:
            return {**self._counts, "pending": self._queue.qsize(),
                    "writer_alive": self._thread.is_alive()}

    def close(self, timeout: float = 5.0, *, exit_code: int = 0) -> bool:
        with self._lock:
            self._exit_code = exit_code
            self._stop_event.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _drop(self, count: int, error: str | None = None) -> None:
        with self._lock:
            self._counts["dropped"] += count
            if error:
                self._counts["error"] = error
        if error:
            logger.error("sample recorder: %s", error)

    def _write_batch(self, batch: list) -> None:
        lines = []
        for record in batch:
            try:
                line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            except (TypeError, ValueError):
                # One unserializable record must not end evidence for the
                # whole run (writer-level failures below stay terminal):
                # count it, drop it, keep recording.
                with self._lock:
                    self._counts["dropped"] += 1
                    self._counts["invalid"] += 1
                continue
            size = len(line.encode("utf-8"))
            if self._bytes + size > self._max_bytes:
                self._drop(1, "sample size limit reached")
                continue
            self._bytes += size
            lines.append(line)
        if not lines:
            return
        if self._free_reserve:
            # The byte cap knows the file, not the filesystem: stop writing
            # while the phase runner's evidence writers still have headroom
            # instead of failing the whole partition with ENOSPC.
            try:
                stat = os.statvfs(self._dir)
                free = stat.f_bavail * stat.f_frsize
            except OSError:
                free = None
            if free is not None and free < self._free_reserve:
                self._drop(len(lines), "free space reserve reached")
                return
        try:
            self._file.write("".join(lines))
            self._file.flush()  # batch only, never producer/per-inference IO
        except OSError as exc:
            self._drop(len(lines), type(exc).__name__)
            return
        with self._lock:
            self._counts["written"] += len(lines)

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                batch = []
                deadline = time.monotonic() + 0.5
                while len(batch) < self._batch_size:
                    try:
                        batch.append(self._queue.get(timeout=max(0.001, deadline - time.monotonic())))
                    except queue.Empty:
                        break
                    if self._stop_event.is_set() and self._queue.empty():
                        break
                    if time.monotonic() >= deadline:
                        break
                if batch:
                    self._write_batch(batch)
            # Terminal metadata is allowed beyond the sample byte budget.
            final = self._envelope("recorder_final", self.snapshot())
            final["writer_alive"] = False
            final["exit_code"] = int(bool(self._exit_code or final["error"]))
            self._file.write(json.dumps(final, ensure_ascii=False) + "\n")
            self._file.flush()
        except OSError as exc:
            self._drop(0, type(exc).__name__)
        finally:
            self._file.close()
