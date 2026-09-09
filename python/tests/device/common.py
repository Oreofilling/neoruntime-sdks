"""Shared scaffolding for on-device interface tests.

These tests run **on the target device** against the live daemons
(ai-runtime, camera-daemon, isp_media_server, event bus, app manager).
They are excluded from normal CI runs: everything is skipped unless the
environment variable ``NEORUNTIME_DEVICE=1`` is set.

The suite is deliberately stdlib-only (unittest + signal + json): the
device venv carries the SDK's runtime dependencies but not pytest, and
the device may have no network to install it.

Result model (mirrored by ``run_device_tests.py`` into device-report.json):

* PASS           — interface called, response sane
* FAIL / ERROR   — assertion or unexpected exception (evidence recorded)
* SKIP-NA        — precondition missing on this device (no model, no
                   hardware peripheral, ...), recorded via ``self.na()``
* KNOWN-ISSUE    — failed in a way we already track as a defect; use the
                   ``@known_issue`` decorator so it does not masquerade
                   as a fresh failure in the report
"""

from __future__ import annotations

import functools
import os
import signal
import time
import traceback
import unittest

# Guard: default-off so `python -m unittest discover` on a dev machine
# (or in CI) is a no-op rather than a wall of connection errors.
ON_DEVICE = os.getenv("NEORUNTIME_DEVICE", "") == "1"

# Scratch space for tests that write files (JPEG evidence, TS segments…).
# Keep everything under one directory so cleanup is a single rm -rf.
DEVICE_TMP_DIR = "/data/sdk-test/tmp"

# Models discovered on the device's data partition (populated by the env
# survey test, reused by the inference/media tests).
MODEL_DIR = "/data/aipc-data/models"


def _default_timeout(test_cls) -> int:
    """Per-test wall-clock budget (seconds).

    Every gRPC / UDS call gets an alarm so a wedged daemon fails one
    test instead of hanging the whole suite (app-manager has wedged
    before). Override via the ``timeout_s`` class attribute.
    """
    return int(getattr(test_cls, "timeout_s", 60))


class _NotApplicable(Exception):
    """Raised by ``DeviceTestCase.na()`` — surfaces as SKIP-NA."""


class _TestTimeout(BaseException):
    """Raised by the per-test alarm when the wall-clock budget is spent.

    Deliberately **not** an ``Exception``: a subscribe-style SDK generator
    that does ``except Exception: continue`` inside its recv loop must not
    be able to swallow the timeout and block forever (this exact wedge
    hung a full device run for 40 minutes with a one-shot alarm).
    unittest's ``testPartExecutor`` catches it with a bare ``except:``
    and records a normal ERROR, traceback included.
    """


def known_issue(reason: str):
    """Mark a test as exercising a tracked defect.

    If the decorated test fails, the failure is recorded with verdict
    KNOWN-ISSUE instead of FAIL. If it unexpectedly passes, the verdict
    is PASS and the report notes the issue was not reproduced — both
    outcomes are informative.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except unittest.SkipTest:
                raise
            except _NotApplicable:
                raise
            # _TestTimeout is a BaseException (so SDK generators cannot
            # swallow it) — a wall-clock hang IS the reproduction shape of
            # the daemon-wedge known issues, so convert it too.
            except (Exception, _TestTimeout) as exc:  # noqa: BLE001 — evidence, not control flow
                self.record["known_issue"] = reason
                self.record["known_issue_error"] = f"{type(exc).__name__}: {exc}"
                self.record["outcome_note"] = f"known issue reproduced: {reason}"
                raise _KnownIssueReproduced(reason) from exc

        return wrapper

    return deco


class _KnownIssueReproduced(Exception):
    """Internal: lets the result reporter tell a known failure apart."""


# Class-level gate: unlike a setUpClass guard, the inherited
# ``__unittest_skip__`` attribute survives every subclass override of
# setUpClass, so `pytest python/tests` on a dev machine (or in CI) skips
# the whole area even when a subclass forgets super().
_GATE_REASON = "on-device suite: set NEORUNTIME_DEVICE=1 (only on the target)"


@unittest.skipUnless(ON_DEVICE, _GATE_REASON)
class DeviceTestCase(unittest.TestCase):
    """Base class with evidence capture, call timing and alarm timeouts."""

    #: Report area, e.g. "inference", "camera". Set per test module.
    area = "generic"
    #: Wall-clock budget for one test method (seconds).
    timeout_s = 60

    @classmethod
    def setUpClass(cls):
        if not ON_DEVICE:
            raise unittest.SkipTest("set NEORUNTIME_DEVICE=1 to run on-device tests")
        os.makedirs(DEVICE_TMP_DIR, exist_ok=True)

    def setUp(self):
        # Per-test record: filled by mark()/evidence(), harvested by the
        # runner's TestResult after the test finishes.
        self.record = {"area": self.area}
        budget = _default_timeout(type(self))
        signal.signal(signal.SIGALRM, self._on_alarm)
        # Repeating interval timer, not a one-shot alarm: a generator that
        # swallows the first timeout gets a fresh one every `budget`
        # seconds until the test dies.
        signal.setitimer(signal.ITIMER_REAL, budget, budget)

    def tearDown(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)

    def _on_alarm(self, signum, frame):
        self.record["outcome_note"] = f"test exceeded {self.timeout_s}s wall clock"
        raise _TestTimeout(f"test exceeded {self.timeout_s}s wall clock")

    # -- evidence helpers -------------------------------------------------

    def mark(self, interface: str) -> None:
        """Name the SDK interface under test (report matrix key)."""
        self.record["interface"] = interface

    def evidence(self, **kw) -> None:
        """Attach proof of what the call returned (kept small + serialisable).

        bytes land as a hex preview — a raw bytes value once poisoned every
        later report flush of its module (json.dump raised, so the on-disk
        report froze at the last good test and lost the tail).
        """
        self.record.setdefault("evidence", {}).update(
            {k: self._jsonable(v) for k, v in kw.items()})

    @staticmethod
    def _jsonable(value):
        if isinstance(value, bytes):
            return value[:32].hex() + ("…" if len(value) > 32 else "")
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def na(self, reason: str) -> None:
        """Abort the test as not-applicable on this device."""
        raise _NotApplicable(reason)

    # -- timed invocation ---------------------------------------------------

    def timed(self, fn, *args, label: str | None = None, **kwargs):
        """Call ``fn`` and record wall-clock latency under ``label``.

        Returns fn's result; the latency lands in the evidence dict as
        ``{label or fn.__name__}_ms``.
        """
        name = label or getattr(fn, "__name__", "call")
        t0 = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            self.record.setdefault("evidence", {})[f"{name}_ms"] = round(
                (time.monotonic() - t0) * 1000.0, 2
            )

    # -- sanity helpers -----------------------------------------------------

    def assert_between(self, value, low, high, what: str) -> None:
        self.assertGreaterEqual(value, low, f"{what}={value} below {low}")
        self.assertLessEqual(value, high, f"{what}={value} above {high}")


class DeviceTestResult(unittest.TextTestResult):
    """unittest result that harvests DeviceTestCase records into JSON rows."""

    #: Optional callable set by the runner (``run_device_tests.py``):
    #: called with the rows collected so far after every finished test so
    #: the JSON report on disk stays at most one test stale — a wedged or
    #: killed run still yields its report.
    flush_hook = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows: list[dict] = []

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _record_of(test) -> dict:
        return getattr(test, "record", None) or {}

    def _row(self, test, verdict: str) -> dict:
        record = self._record_of(test)
        return {
            "id": test.id(),
            "area": record.get("area", getattr(test, "area", "generic")),
            "interface": record.get("interface") or test.id().rsplit(".", 1)[-1],
            "verdict": verdict,
            "duration_ms": None,  # filled by stopTest
            "evidence": record.get("evidence", {}),
            "note": record.get("outcome_note"),
        }

    # -- unittest hooks ------------------------------------------------------

    def addSuccess(self, test):
        super().addSuccess(test)
        self.rows.append(self._row(test, "PASS"))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        row = self._row(test, "FAIL")
        row["error"] = self._format(err)
        self.rows.append(row)

    def addError(self, test, err):
        exc = err[1]
        if isinstance(exc, _NotApplicable):
            row = self._row(test, "SKIP-NA")
            row["note"] = str(exc)
            self.rows.append(row)
            if self.showAll:
                self.stream.writeln(f"NA       {self.getDescription(test)} ({exc})")
            return
        if isinstance(exc, _KnownIssueReproduced):
            record = self._record_of(test)
            row = self._row(test, "KNOWN-ISSUE")
            row["note"] = record.get("known_issue", str(exc))
            row["error"] = record.get("known_issue_error", "")
            self.rows.append(row)
            if self.showAll:
                self.stream.writeln(
                    f"KNOWN    {self.getDescription(test)} ({record.get('known_issue', '')})"
                )
            return
        row = self._row(test, "ERROR")
        row["error"] = self._format(err)
        self.rows.append(row)
        super().addError(test, err)

    def addSkip(self, test, reason):
        row = self._row(test, "SKIP")
        row["note"] = str(reason)
        self.rows.append(row)
        super().addSkip(test, reason)

    def startTest(self, test):
        self._t0 = time.monotonic()
        super().startTest(test)

    def stopTest(self, test, *a):
        for row in reversed(self.rows):
            if row["id"] == test.id() and row["duration_ms"] is None:
                row["duration_ms"] = round((time.monotonic() - self._t0) * 1000.0, 2)
                break
        super().stopTest(test, *a)
        hook = type(self).flush_hook
        if hook is not None:
            try:
                hook(self.rows)
            except Exception:  # noqa: BLE001 — reporting must not kill tests
                pass

    @staticmethod
    def _format(err) -> str:
        exc_type, exc, tb = err
        return "".join(traceback.format_exception(exc_type, exc, tb)[-6:])
