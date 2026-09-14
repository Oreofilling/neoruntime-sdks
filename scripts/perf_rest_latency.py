#!/usr/bin/env python3
"""perf_rest_latency.py — REST control-plane latency over the local
unix-socket face (G1/G2, 2026-09-12 platform perf matrix).

The unix face (/run/aipc/platform-api.sock) is auth-free by design
(socket permissions ARE the auth), so the numbers exclude bearer-token
overhead — the plan's requirement for the control-plane baseline.
Endpoint classes sampled by default:

* fast reads: health / monitor/* / ai models / events topics
* cross-service chains: media/status (camera-daemon), device/status
  and lens/status (device-control), system/info
* known-slow 500ms-window stats: system/stats, ai/stats

GPIO endpoints are REFUSED outright: a single GET hard-resets the
device (issue #46) and must never appear in any sweep.

Usage (on the device):
    python3 perf_rest_latency.py [--socket /run/aipc/platform-api.sock] \
        [-n 100] [--request-delay 0.05] [--endpoints /system/health,...]

Output: one JSON object on stdout, {endpoint: stats-dict}.
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import statistics
import sys
import time
from collections import Counter

SOCKET_DEFAULT = "/run/aipc/platform-api.sock"
BASE_DEFAULT = "/api/v1"

FAST = ["/system/health", "/monitor/summary", "/monitor/cpu",
        "/monitor/memory", "/monitor/disk", "/monitor/network",
        "/ai/models", "/ai/capabilities", "/events/topics"]
CHAIN = ["/media/status", "/media/profiles", "/device/status",
         "/device/infrared/status", "/device/lens/status", "/system/info"]
SLOW = ["/system/stats", "/ai/stats"]


class UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection dialing an AF_UNIX socket instead of TCP."""

    def __init__(self, path: str, timeout: float = 10.0):
        super().__init__("localhost", timeout=timeout)
        self._unix_path = path

    def connect(self):  # noqa: D102 — http.client override
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._unix_path)
        self.sock = s


def percentile_stats(samples_ms: list[float]) -> dict:
    data = sorted(samples_ms)
    stats = {"n": len(data), "min": data[0] if data else None,
             "mean": round(statistics.fmean(data), 2) if data else None,
             "p50": None, "p90": None, "p95": None, "p99": None,
             "max": data[-1] if data else None, "unit": "ms"}
    if len(data) >= 2:
        qs = statistics.quantiles(data, n=100, method="inclusive")
        stats.update(p50=round(qs[49], 2), p90=round(qs[89], 2),
                     p95=round(qs[94], 2), p99=round(qs[98], 2))
    return stats


def sample_endpoint(conn: UnixHTTPConnection, path: str, n: int,
                    delay_s: float, warmup: int) -> dict:
    lat: list[float] = []
    statuses: Counter = Counter()
    err = 0
    err_text = None
    for i in range(warmup + n):
        t0 = time.perf_counter_ns()
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            resp.read()  # drain inside the timed window
            statuses[resp.status] += 1
            if i >= warmup:
                lat.append((time.perf_counter_ns() - t0) / 1e6)
        except Exception as exc:  # noqa: BLE001 — counted, not timed
            err += 1
            if err_text is None:
                err_text = f"{type(exc).__name__}: {exc}"[:120]
            try:
                conn.close()  # force a fresh connection on next attempt
            except Exception:
                pass
        if delay_s:
            time.sleep(delay_s)
    stats = percentile_stats(lat)
    stats["ok"] = stats["n"]
    stats["err"] = err
    if err_text:
        stats["err_text"] = err_text
    stats["status_hist"] = dict(statuses)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--socket", default=SOCKET_DEFAULT)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("-n", type=int, default=100,
                    help="samples per endpoint (default 100)")
    ap.add_argument("--request-delay", type=float, default=0.05,
                    help="sleep between requests, s (default 0.05; "
                         "raise to 0.2 to be extra gentle)")
    ap.add_argument("--endpoints", default=None,
                    help="comma-separated path list overriding the "
                         "default fast+chain+slow set")
    args = ap.parse_args()

    classes = {}
    for p in FAST:
        classes[p] = "fast"
    for p in CHAIN:
        classes[p] = "chain"
    for p in SLOW:
        classes[p] = "slow"
    if args.endpoints:
        endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    else:
        endpoints = FAST + CHAIN + SLOW

    # issue #46: GPIO read/write hard-resets the device. Refuse even if
    # explicitly requested — no sweep may ever include it.
    for p in endpoints:
        if "gpio" in p.lower():
            print(f"REFUSED: {p!r} touches GPIO (issue #46: single call "
                  "hard-resets the device)", file=sys.stderr)
            return 2

    conn = UnixHTTPConnection(args.socket)
    out: dict = {"_meta": {
        "socket": args.socket, "base": args.base, "n": args.n,
        "request_delay_s": args.request_delay,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }}
    try:
        for path in endpoints:
            full = args.base + path
            warmup = max(5, args.n // 10)
            stats = sample_endpoint(conn, full, args.n,
                                    args.request_delay, warmup)
            stats["class"] = classes.get(path, "custom")
            out[path] = stats
            p50 = stats.get("p50")
            print(f"{path:35s} p50={p50 if p50 is not None else '—'}ms "
                  f"p99={stats.get('p99')}ms err={stats['err']}",
                  file=sys.stderr)
    finally:
        conn.close()
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
