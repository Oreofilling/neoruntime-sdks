"""Shared scaffolding for the on-device performance suite (test_6x_perf_*).

The functional suite (``common.py``) records one latency per interface
call. A performance baseline needs more: sampled distributions with the
warmup discarded, stream arrival statistics derived from ``frame_seq``
continuity, and /proc sampling that attributes drift to the *client*
process separately from the daemons it talks to.

Layout:

* Pure, offline-testable helpers — ``percentile_stats``, ``arrival_stats``,
  ``median_round``, ``proc_snapshot`` (unit-tested in
  ``python/tests/test_perf_common.py``, importable without a device).
* ``PerfTestCase`` — device-gated base class adding ``perf_sample`` /
  ``perf_stream`` on top of ``DeviceTestCase`` evidence records.
* ``software_only_accel`` — context manager swapping the accel router
  singleton to ``SOFTWARE_ONLY`` for the A/B routing measurement; it
  touches accel privates (``_default_router``, ``_*_sw`` legs) on
  purpose: the router policy is fixed at construction and there is no
  public reconfiguration, and duplicating the registration list would
  drift. The original singleton is restored untouched on exit.

Evidence convention (what gen_perf_report.py keys on): every
measurement lands under ``evidence["perf:<label>"]`` as the stats dict
from ``percentile_stats`` / ``arrival_stats`` plus a ``unit`` field
("ms" for call latencies, "" for streams).
"""

from __future__ import annotations

import contextlib
import os
import re
import statistics
import time

from common import MODEL_DIR, DeviceTestCase

# Perf defaults; overridable per environment so a calibration run can
# shrink sample counts without editing code (device has no easy edit loop).
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


SAMPLE_N = _env_int("PERF_SAMPLE_N", 300)
SAMPLE_ROUNDS = _env_int("PERF_SAMPLE_ROUNDS", 3)
STREAM_S = _env_int("PERF_STREAM_S", 60)
SOAK_S = _env_int("PERF_SOAK_S", 1800)

# anomaly thresholds (report-time conventions, mirrored by the generator)
ANOMALY_TAIL_RATIO = 8.0   # p99/p50 above this is flagged
ANOMALY_ERR_PCT = 0.5      # error rate above this is flagged
ANOMALY_DECAY_PCT = 10.0   # soak first-vs-last bucket drop above this


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
    """Stream-side summary from per-arrival monotonic-ish timestamps.

    ``seqs`` may contain non-int entries (None per frame when the
    interface exposes no sequence numbers) or be shorter than
    ``times`` — both degrade drop accounting to None rather than
    misattribute gaps, but inter-arrival gap stats stay computed from
    the full arrival timeline. Gaps are counted from ``frame_seq``
    continuity: a jump of k missing frames is k drops (duplicates /
    negative jumps, which seq wraparounds produce, are ignored).
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


def median_round(rounds: list[dict], key: str = "p50") -> tuple[dict, dict]:
    """Pick the round whose ``key`` is the median; report round spread."""
    vals = [r[key] for r in rounds if r.get(key) is not None]
    best = min(rounds, key=lambda r: abs((r.get(key) or 0) - statistics.median(vals))) if vals else {}
    spread = {
        "rounds": len(rounds),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "spread_pct": None,
    }
    if vals and spread["min"]:
        spread["spread_pct"] = round(
            (spread["max"] - spread["min"]) * 100.0 / spread["min"], 2)
    return best, spread


def proc_snapshot(pid: int | None = None) -> dict:
    """RSS / fd count / thread count for ``pid`` (default: this process)."""
    pid = pid or os.getpid()
    snap = {"pid": pid, "rss_kb": None, "fds": None, "threads": None}
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    snap["rss_kb"] = int(line.split()[1])
                elif line.startswith("Threads:"):
                    snap["threads"] = int(line.split()[1])
    except OSError:
        pass
    try:
        snap["fds"] = len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        pass
    return snap


def daemon_pids() -> dict[str, int]:
    """{name: pid} for the daemons the SDK talks to (soak attribution)."""
    wanted = ("ai-runtime", "camera-daemon", "event-bus", "app-manager",
              "isp_media_server")
    found: dict[str, int] = {}
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/comm") as fh:
                    comm = fh.read().strip()
            except OSError:
                continue
            if comm in wanted and comm not in found:
                found[comm] = int(entry)
    except OSError:
        pass
    return found


def pick_model(*preferred: str) -> str | None:
    """First existing HEF under MODEL_DIR from ``preferred``, else any.

    The secondary test device's model partition was wiped and
    repopulated from a different set than the primary's — the perf
    suite must degrade to whatever the
    device has (the chosen file is recorded in the smoke evidence)
    instead of hard-requiring one filename.
    """
    for name in preferred:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            return path
    try:
        hefs = sorted(f for f in os.listdir(MODEL_DIR)
                      if f.endswith(".hef"))
        return os.path.join(MODEL_DIR, hefs[0]) if hefs else None
    except OSError:
        return None


def model_input_geometry(model_path: str) -> tuple[int, int] | None:
    """Best-effort (w, h) input geometry parsed from the HEF filename.

    The daemon validates input byte_size against the model's expected
    geometry: an RGB array into an NV12 model fails -2799/-2811, and a
    1920×1080 buffer into a 640×384 model fails the same way (verified
    2026-09-09 on the secondary test device — the earlier "-2799 daemon
    regression" read
    was wrong; the infer RPC works with geometry-matched NV12 at
    ~71 FPS). Device filenames follow two conventions:
    ``..._384_640.hef`` (H then W) and ``..._540.hef`` (height only →
    16:9). Unknown → None (caller falls back to native stream size).
    """
    name = os.path.basename(model_path)
    pair = re.search(r"_(\d{2,4})[_x](\d{2,4})(?=\.\w+$|$)", name)
    if pair:
        height, width = int(pair[1]), int(pair[2])
        return (width, height)
    single = re.search(r"_(\d{3,4})(?=\.\w+$|$)", name)
    if single:
        height = int(single[1])
        return (height * 16 // 9, height)
    return None


def perf_model_id(model_path: str | None, prefix: str = "sdk-perf") -> str:
    """Stable per-file test id: same HEF → same id across runs.

    Two device behaviors make a FIXED test id a trap (formal-run
    incident, 2026-09-09): the daemon keeps an existing id's binding
    when a re-registration names a different file — register-with-
    existing-id adds an owner and returns success — and the platform's
    self-heal resurrects owner-less registrations from its database
    about a minute after unregister. A calibration run's model then
    shadows the formal run's model under the same id and every infer
    lands on the stale geometry (-2799). Keying the id to the file
    makes any healed entry match the file under test.
    """
    stem = os.path.basename(model_path or "model").rsplit(".", 1)[0]
    return f"{prefix}-{stem}"


class PerfTestCase(DeviceTestCase):
    """Base class for perf modules: sampling with evidence conventions."""

    area = "perf"

    # -- sampled call latency -------------------------------------------

    def perf_sample(self, fn, *args, label: str, n: int | None = None,
                    warmup: int | None = None, rounds: int = 1,
                    **kwargs) -> dict:
        """Sample ``fn(*args, **kwargs)`` latency (ms); stats to evidence.

        Positional ``args`` follow ``timed()``'s convention so bound
        client methods sample naturally:
        ``perf_sample(client.get_model_info, MODEL_ID, label=...)``.

        ``rounds > 1`` repeats the whole sampling and keeps the
        median-p50 round as the headline number plus a spread record —
        the plan's 3-round methodology. Errors are counted, never timed;
        an error rate above 50% returns early so a broken path does not
        burn the whole budget.
        """
        n = n or SAMPLE_N
        warmup = max(5, n // 10) if warmup is None else warmup
        all_rounds = []
        for _ in range(rounds):
            for _ in range(warmup):
                try:
                    fn(*args, **kwargs)
                except Exception:  # noqa: BLE001 — warmup must never abort
                    pass
            samples: list[float] = []
            err = 0
            err_text: str | None = None
            t_end = time.monotonic() + max(120, n * 30)  # hard floor vs n×30ms
            while len(samples) + err < n:
                if time.monotonic() > t_end:  # path much slower than assumed
                    break
                t0 = time.perf_counter_ns()
                try:
                    fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 — counted, not raised
                    err += 1
                    if err_text is None:  # first error names the broken path
                        err_text = f"{type(exc).__name__}: {exc}"[:120]
                    continue
                samples.append((time.perf_counter_ns() - t0) / 1e6)
            stats = percentile_stats(samples, err=err)
            stats["warmup"] = warmup
            if err_text:
                stats["err_text"] = err_text
            all_rounds.append(stats)
            if (stats["err_pct"] or 0) > 50.0:
                break  # path is broken — no point re-sampling it
        headline, spread = (median_round(all_rounds) if len(all_rounds) > 1
                            else (all_rounds[0] if all_rounds else {}, {}))
        rec = dict(headline)
        rec["unit"] = "ms"
        if spread:
            rec["round_spread"] = spread
        return_vals = rec
        self.evidence(**{f"perf:{label}": rec})
        return return_vals

    # -- stream consumption ----------------------------------------------

    def perf_stream(self, gen, *, label: str, duration_s: float | None = None,
                    seq_of=None):
        """Consume ``gen`` for ``duration_s``; arrival stats to evidence.

        ``seq_of(item)`` extracts the per-item sequence number when the
        interface provides one (InferenceClient.subscribe yields
        ``(frame_seq, result)``) — drop accounting needs it.
        """
        duration_s = duration_s or STREAM_S
        times: list[float] = []
        seqs: list[int] = []
        t0 = time.monotonic()
        deadline = t0 + duration_s
        for item in gen:
            times.append(time.monotonic())
            seqs.append(seq_of(item) if seq_of else None)
            if time.monotonic() >= deadline:
                break
        elapsed = (times[-1] - t0) if times else duration_s
        # seqs stays index-aligned with times (None entries included):
        # arrival_stats tolerates the Nones, and pre-filtering here once
        # silently truncated the zip and misattributed drop gaps.
        stats = arrival_stats(seqs, times, elapsed)
        self.evidence(**{f"perf:{label}": stats})
        return stats


@contextlib.contextmanager
def software_only_accel():
    """Swap the accel router singleton to SOFTWARE_ONLY (A/B baseline).

    Re-registers the software legs from accel's own module privates so
    the measurement exercises the same implementations the AUTO policy
    falls back to; restores the original singleton afterwards. When the
    DSP legs are unreachable anyway (the secondary test device has no
    dsp socket) both
    sides measure software and the health evidence says so — the
    report treats that as "hw path unavailable", not as a speedup of 1.
    """
    import neoruntime_ipc_sdk.accel as accel

    original = accel.get_default_router()
    sw_router = accel.AccelRouter(policy=accel.RoutePolicy.SOFTWARE_ONLY)
    # Same op set the default router registers (accel.py get_default_router):
    # resize_nv12 / rgb_to_nv12 / nv12_to_rgb / encode_jpeg / nms /
    # draw_detections, each bound to accel's own software leg.
    legs = {
        "resize_nv12": getattr(accel, "_nv12_resize_sw", None),
        "rgb_to_nv12": getattr(accel, "_rgb_to_nv12_sw", None),
        "nv12_to_rgb": getattr(accel, "_nv12_to_rgb_sw", None),
        "encode_jpeg": getattr(accel, "_encode_jpeg_sw", None),
        "nms": getattr(accel, "_nms_sw", None),
        "draw_detections": getattr(accel, "_draw_detections_sw", None),
    }
    registered = []
    for op, leg in legs.items():
        if leg is not None:
            sw_router.register(op, software=leg)
            registered.append(op)
    for probe_name in ("cv2", "dsp"):
        probe_fn = getattr(accel, f"_probe_{probe_name}", None)
        if probe_fn is not None:
            sw_router.add_probe(probe_name, probe_fn)
    accel._default_router = sw_router  # noqa: SLF001 — see module docstring
    try:
        yield registered
    finally:
        accel._default_router = original  # noqa: SLF001
