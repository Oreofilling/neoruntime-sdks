#!/usr/bin/env python3
"""On-device test runner: executes the interface suite and emits a JSON report.

Usage (on the target device, inside its venv)::

    NEORUNTIME_DEVICE=1 python3 run_device_tests.py [output.json] [module ...]

Optional ``module`` arguments (``test_20_inference``, ``test_20_inference.py``
or ``20_inference`` — all normalize the same way) restrict the run to those
modules; the orchestrator drives one module per process so a wedged area
only loses its own budget, never the rest of the suite.

Design notes:

* stdlib only — the device venv has no pytest and may have no network;
* discovery is filename-ordered (``test_10_*`` before ``test_50_*``) so
  read-only probes run before state-changing and physical side-effect
  areas, and the lens-AF block always runs late with its own reset;
* per-test repeating interval timer inside DeviceTestCase bounds each
  case, so a wedged daemon fails one area instead of hanging the whole
  run; on top of that the report JSON is rewritten after every test
  (atomic replace), so even a killed run leaves a readable report with
  everything that completed;
* environment metadata (kernel, daemons, sockets, models, SDK version)
  is gathered opportunistically — a missing /proc entry must not abort
  the report;
* exit code: 0 when nothing failed outright (PASS/NA/SKIP/KNOWN-ISSUE
  are all reportable outcomes), 1 when any FAIL/ERROR was recorded,
  2 when a module filter names no existing module.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DeviceTestResult  # noqa: E402

DEFAULT_REPORT = "/data/sdk-test/device-report.json"
SOCKET_DIR = "/run/aipc"


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    """Best-effort shell probe: never raises, output trimmed."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr).strip()[:2000]
    except Exception as exc:  # noqa: BLE001 — metadata must not abort the run
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def collect_env() -> dict:
    """Snapshot everything the report needs to be reproducible."""
    env = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "node": {
            "uname": " ".join(platform.uname()),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "os_release": _run(
                ["sh", "-c", ". /etc/os-release 2>/dev/null; echo $PRETTY_NAME"]
            ),
        },
        "sockets": {},
        "daemons": {},
        "models": [],
    }

    # SDK version under test (from the installed wheel, not the repo).
    try:
        import neoruntime_ipc_sdk

        env["sdk"] = {
            "version": neoruntime_ipc_sdk.__version__,
            "module_path": os.path.dirname(neoruntime_ipc_sdk.__file__),
        }
    except Exception as exc:  # noqa: BLE001
        env["sdk"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Every UDS the SDK talks to, with existence + size evidence.
    for sock in sorted(glob.glob(f"{SOCKET_DIR}/**/*.sock", recursive=True)):
        try:
            st = os.stat(sock)
            env["sockets"][sock] = {"size": st.st_size}
        except OSError as exc:
            env["sockets"][sock] = {"error": str(exc)}

    # Daemon processes with their config paths (args are more stable
    # than pgrep -x across builds).
    ps = _run(["ps", "-eo", "pid,comm,args"])
    watched = ("ai-runtime", "camera-daemon", "isp_media_server",
               "app-manager", "dsp_service", "event-bus")
    for line in ps.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, comm, args = parts
        for daemon in watched:
            if daemon in comm or f"/{daemon}" in args:
                env["daemons"].setdefault(daemon, []).append(
                    {"pid": pid, "args": args[:300]}
                )

    # Model inventory (path + size only; feeds the inference area).
    for model_dir in ("/data/aipc-data/models", "/app/models"):
        if os.path.isdir(model_dir):
            for name in sorted(os.listdir(model_dir)):
                path = os.path.join(model_dir, name)
                try:
                    env["models"].append(
                        {"path": path, "size_bytes": os.path.getsize(path)}
                    )
                except OSError:
                    env["models"].append({"path": path, "size_bytes": -1})
    return env


def _verdicts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    return counts


def _normalize_module(name: str) -> str:
    """Accept 'test_20_inference.py', 'test_20_inference' or '20_inference'."""
    name = name[:-3] if name.endswith(".py") else name
    return name if name.startswith("test_") else f"test_{name}"


def main() -> int:
    report_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPORT
    only = {_normalize_module(name) for name in sys.argv[2:]}

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    here = os.path.dirname(os.path.abspath(__file__))
    # Sort by filename: numeric prefixes encode execution phase
    # (read-only probes first, physical lens effects last).
    for path in sorted(glob.glob(os.path.join(here, "test_*.py"))):
        module = os.path.splitext(os.path.basename(path))[0]
        if only and module not in only:
            continue
        suite.addTests(loader.loadTestsFromName(module))
    if only:
        unknown = sorted(only - {
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(here, "test_*.py"))
        })
        if unknown:
            print(f"unknown module(s): {', '.join(unknown)}", file=sys.stderr)
            return 2

    # Environment is captured *before* the run and the report is flushed
    # after every finished test: a wedged or SIGKILLed run still leaves
    # a parseable report covering everything that completed.
    env = collect_env()
    t0 = time.monotonic()

    def _flush(rows: list[dict]) -> None:
        report = {
            "schema": "neoruntime-device-test-report/1",
            "env": env,
            "summary": {
                "total": len(rows),
                "verdicts": _verdicts(rows),
                "wall_time_s": round(time.monotonic() - t0, 1),
            },
            "cases": rows,
        }
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        tmp_path = report_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            # default= is the last line of defence: an exotic evidence
            # value must degrade to text, never kill the flush (a bytes
            # value once froze a module's report at its last good test).
            json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp_path, report_path)

    DeviceTestResult.flush_hook = _flush

    result = unittest.TextTestRunner(
        verbosity=2, resultclass=DeviceTestResult
    ).run(suite)

    verdicts = _verdicts(result.rows)
    _flush(result.rows)

    print(f"\nreport: {report_path}")
    print(f"summary: {json.dumps(verdicts)} total={len(result.rows)}")
    hard_failures = verdicts.get("FAIL", 0) + verdicts.get("ERROR", 0)
    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
