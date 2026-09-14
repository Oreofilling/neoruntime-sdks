"""Rolling-window statistics for the perf-demo app.

The two distribution helpers are semantic copies of the device perf
suite's pure functions (tests/device/perf_common.py:59-132) so the demo
numbers and the suite numbers are comparable; their exact behavior is
pinned by tests/test_perf_common.py in this repository.
"""

from __future__ import annotations

import statistics


def percentile_stats(samples_ms, ok: int | None = None, err: int = 0) -> dict:
    """Distribution summary for a list of latencies in milliseconds.

    ``ok`` defaults to ``len(samples_ms)``; callers that keep counting
    errors separately pass both. Percentiles use the inclusive method so
    values are exact members of small samples. ``None`` for n < 2.
    """
    data = sorted(float(x) for x in samples_ms)
    n = len(data)
    ok_n = n if ok is None else ok
    stats = {
        "n": n,
        "ok": ok_n,
        "err": err,
        "err_pct": None,
        "min": data[0] if data else None,
        "mean": round(statistics.fmean(data), 3) if data else None,
        "p50": None, "p90": None, "p95": None, "p99": None,
        "max": data[-1] if data else None,
    }
    total = ok_n + err
    if total > 0 and err > 0:
        stats["err_pct"] = round(err * 100.0 / total, 2)
    if n >= 2:
        qs = statistics.quantiles(data, n=100, method="inclusive")
        stats["p50"] = round(qs[49], 3)
        stats["p90"] = round(qs[89], 3)
        stats["p95"] = round(qs[94], 3)
        stats["p99"] = round(qs[98], 3)
    return stats


def arrival_stats(seqs, times, duration_s: float) -> dict:
    """Stream-side summary from per-arrival monotonic timestamps.

    ``seqs`` may contain non-int entries (None per frame) or be shorter
    than ``times`` — both degrade drop accounting to None rather than
    misattribute gaps; inter-arrival gap stats stay computed from the
    full arrival timeline. Drops are counted from sequence continuity:
    a jump of k missing frames is k drops (duplicates / negative jumps
    are ignored).
    """
    frames = len(times)
    if len(seqs) != frames:  # misaligned seq data is worse than none
        seqs = [None] * frames
    stats = {
        "frames": frames,
        "duration_s": round(duration_s, 2),
        "fps": round(frames / duration_s, 2) if duration_s > 0 else None,
        "drops": None,
        "drop_pct": None,
        "gap_p50": None, "gap_p95": None, "gap_max": None,
    }
    ordered = sorted(zip(seqs, times), key=lambda p: p[1])
    clean = [s for s, _ in ordered if isinstance(s, int)]
    drops = 0
    for prev, cur in zip(clean, clean[1:]):
        delta = cur - prev
        if delta > 1:
            drops += delta - 1
    if clean:
        stats["drops"] = drops
        if clean[-1] > clean[0]:  # span known -> drops/expected is meaningful
            expected = clean[-1] - clean[0] + 1
            stats["drop_pct"] = round(drops * 100.0 / expected, 2)
    if frames >= 2:
        gaps_ms = [(b - a) * 1000.0 for (_, a), (_, b) in zip(ordered, ordered[1:])]
        stats["gap_p50"] = round(statistics.median(gaps_ms), 2)
        if len(gaps_ms) >= 20:
            qs = statistics.quantiles(gaps_ms, n=100, method="inclusive")
            stats["gap_p95"] = round(qs[94], 2)
        stats["gap_max"] = round(max(gaps_ms), 2)
    return stats


def fmt_ms(value: float | None, digits: int = 0) -> str:
    """Compact milliseconds formatting for burned-in OSD lines."""
    return "--" if value is None else f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 0) -> str:
    """Compact percentage formatting ('--' when the ratio is unknown)."""
    return "--" if value is None else f"{value:.{digits}f}%"


def fmt_duration(seconds: float) -> str:
    """Human uptime '3h07m' / '12m05s' for OSD lines."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s"
