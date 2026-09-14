"""Offline unit tests for the perf-demo app's pure pieces.

Host-only, no device: pins the semantics of the rolling stats helpers
(copied from tests/device/perf_common.py, whose own behavior is pinned
by tests/test_perf_common.py), the burn-in formatters, the chain A
polygon carrier and the chain B text chip. The chain modules themselves
are exercised on-device by the deploy/verification runs.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "examples", "perf_demo",
    ),
)

from display import (  # noqa: E402
    MAX_BURN_LINES,
    MAX_LABEL_CHARS,
    format_a_line,
    format_b_line,
    format_status_line,
    metric_detections,
    text_chip_rgba,
)
from metrics import MetricsHub, Ring  # noqa: E402
from stats import arrival_stats, fmt_duration, fmt_ms, percentile_stats  # noqa: E402


class PercentileStatsTest(unittest.TestCase):
    def test_known_distribution(self):
        # 1..100 ms — mirrors tests/test_perf_common.py exactly: the
        # inclusive method interpolates, so these are the pinned values.
        s = percentile_stats([float(x) for x in range(1, 101)])
        self.assertEqual(s["n"], 100)
        self.assertEqual(s["min"], 1.0)
        self.assertEqual(s["max"], 100.0)
        self.assertEqual(s["p50"], 50.5)
        self.assertEqual(s["p90"], 90.1)
        self.assertEqual(s["p95"], 95.05)
        self.assertEqual(s["p99"], 99.01)

    def test_small_samples(self):
        empty = percentile_stats([])
        self.assertIsNone(empty["p50"])
        self.assertEqual(empty["n"], 0)
        single = percentile_stats([7.5])
        self.assertIsNone(single["p99"])  # percentiles need n >= 2
        self.assertEqual(single["mean"], 7.5)

    def test_err_accounting(self):
        s = percentile_stats([1.0, 2.0], ok=8, err=2)
        self.assertEqual(s["err_pct"], 20.0)


class ArrivalStatsTest(unittest.TestCase):
    def test_drops_from_seq_continuity(self):
        seqs = [1, 2, 3, 6, 7]          # 4,5 missing -> 2 drops (delta-1)
        times = [0.0, 0.033, 0.066, 0.1, 0.133]
        s = arrival_stats(seqs, times, 0.2)
        self.assertEqual(s["drops"], 2)
        # loss vs expected span 1..7 (7 frames) -> 2/7
        self.assertAlmostEqual(s["drop_pct"], 28.57, places=2)

    def test_misaligned_seq_degrades_to_none(self):
        s = arrival_stats([1, 2], [0.0, 0.033, 0.066], 0.1)
        self.assertIsNone(s["drops"])
        self.assertEqual(s["frames"], 3)  # gap stats still computed


class RingTest(unittest.TestCase):
    def test_rate_over_window(self):
        r = Ring()
        t0 = 1000.0
        for i in range(10):
            r.add(0.0, t0 + i * 0.1)      # 10 events in ~1 s
        rate = r.rate(window_s=5.0, now=t0 + 1.0)
        self.assertAlmostEqual(rate, 10.0, delta=0.5)

    def test_window_excludes_old_samples(self):
        r = Ring()
        r.add(5.0, 1000.0)
        r.add(6.0, 1010.0)
        self.assertEqual(r.values(window_s=5.0, now=1010.0), [6.0])
        self.assertEqual(r.count(window_s=5.0, now=1010.0), 1)

    def test_capacity_trim(self):
        r = Ring(capacity=4)
        for i in range(10):
            r.add(float(i), 2000.0 + i)
        self.assertEqual(len(r.values(window_s=100.0, now=2010.0)), 4)


class MetricsHubSnapshotTest(unittest.TestCase):
    def test_snapshot_shape_and_consistency(self):
        hub = MetricsHub()
        now = __import__("time").monotonic()  # rings window on the real clock
        hub.a.record_result(latency_ms=12.0, skew_us=800, now=now)
        hub.b.record_frame(pull_ms=3.0, infer_ms=22.0, hw_infer_ms=18.0,
                           draw_ms=9.0, pub_ms=4.0, e2e_ms=58.0, now=now)
        snap = hub.snapshot()
        for key in ("window_s", "uptime_s", "a", "b", "sys"):
            self.assertIn(key, snap)
        self.assertEqual(snap["a"]["latency"]["n"], 1)
        self.assertEqual(snap["b"]["e2e"]["max"], 58.0)
        self.assertIn("stream_delta", snap["sys"])

    def test_final_section(self):
        hub = MetricsHub()
        hub.record_final("stop")
        self.assertEqual(hub.snapshot()["final"]["reason"], "stop")


class MetricDetectionsTest(unittest.TestCase):
    def test_schema(self):
        dets = metric_detections(["line one", "line two"])
        self.assertLessEqual(len(dets), 16)
        self.assertEqual(len(dets), 2)
        for d in dets:
            self.assertIn(d["label"], ("line one", "line two"))
            self.assertEqual(d["score"], 1.0)
            bbox = d["bbox"]
            for key in ("x", "y", "width", "height"):
                self.assertIn(key, bbox)
                self.assertGreaterEqual(bbox[key], 0.0)
                self.assertLessEqual(bbox[key], 1.0)

    def test_line_cap(self):
        lines = [f"l{i}" for i in range(10)]
        self.assertEqual(len(metric_detections(lines)), MAX_BURN_LINES)

    def test_label_clamp(self):
        dets = metric_detections(["x" * 500])
        self.assertEqual(len(dets[0]["label"]), MAX_LABEL_CHARS)

    def test_stacking_order(self):
        (a, b) = metric_detections(["first", "second"])
        self.assertLess(a["bbox"]["y"], b["bbox"]["y"])


class TextChipTest(unittest.TestCase):
    def test_shape_and_alpha(self):
        chip, _x0, _y0 = text_chip_rgba("B fd | fps 14.6 | e2e 58ms")
        self.assertEqual(chip.ndim, 3)
        self.assertEqual(chip.shape[2], 4)
        self.assertEqual(chip.dtype.name, "uint8")
        self.assertGreaterEqual(chip.shape[0], 16)
        self.assertGreaterEqual(chip.shape[1], 16)
        # translucent black background, straight alpha
        self.assertEqual(int(chip[0, 0, 3]), 150)
        self.assertEqual(int(chip[0, 0, 0]), 0)
        # some text pixels are opaque white
        opaque = chip[:, :, 3] == 255
        self.assertTrue(opaque.any())
        ys, xs = opaque.nonzero()
        self.assertEqual(int(chip[ys[0], xs[0], 0]), 255)

    def test_min_floor_for_tiny_text(self):
        chip, _, _ = text_chip_rgba(".")
        self.assertGreaterEqual(chip.shape[0], 16)
        self.assertGreaterEqual(chip.shape[1], 16)


class FormatLinesTest(unittest.TestCase):
    A_OK = {
        "fps": 9.8, "dropped": 0, "epoch": 4, "degraded_reason": None,
        "latency": {"p50": 41.0, "p99": 63.0},
        "skew": {"mean": 18.0},
    }
    B_OK = {
        "fps": 14.6, "objects_last": 3, "in_flight": 1, "pool_depth": 4,
        "inject_dropped": 0, "degraded_reason": None,
        "pull": {"p50": 3.0}, "infer": {"p50": 22.0},
        "hw_infer": {"p50": 18.0}, "draw": {"p50": 9.0},
        "pub": {"p50": 4.0}, "e2e": {"p99": 58.0},
    }

    def test_a_line_fields(self):
        delta = {"bake_pct": 96.0, "overlay_late_commands": 3}
        line = format_a_line(self.A_OK, delta)
        self.assertIn("A sub", line)
        self.assertIn("fps 9.8", line)
        self.assertIn("lat 41/63ms", line)
        self.assertIn("bake 96%", line)
        self.assertIn("late 3", line)
        self.assertIn("ep 4", line)

    def test_a_line_degraded(self):
        line = format_a_line({"degraded_reason": "subscribe path dead",
                              "last_result_age_s": 12.0}, None)
        self.assertIn("DEGRADED", line)
        self.assertIn("subscribe path dead", line)

    def test_a_line_missing_delta(self):
        line = format_a_line(self.A_OK, None)
        self.assertNotIn("bake", line)

    def test_b_line_fields(self):
        line = format_b_line(self.B_OK)
        self.assertIn("B fd", line)
        self.assertIn("fps 14.6", line)
        self.assertIn("infer 22(hw 18)", line)
        self.assertIn("e2e 58ms", line)
        self.assertIn("lease 1/4", line)
        self.assertIn("obj 3", line)

    def test_b_line_degraded(self):
        line = format_b_line({"degraded_reason": "lease timeout",
                              "frames_ok": 120})
        self.assertIn("DEGRADED", line)

    def test_status_line(self):
        line = format_status_line(
            {"npu_util": 71.0, "dsp_util": 12.0, "temp_c": 58.0}, 22340.0)
        self.assertIn("NPU 71%", line)
        self.assertIn("58C", line)
        self.assertIn("up 6h12m", line)  # 22340 s = 6h12m20s


class FmtHelpersTest(unittest.TestCase):
    def test_fmt_ms_none(self):
        self.assertEqual(fmt_ms(None), "--")
        self.assertEqual(fmt_ms(41.234), "41")

    def test_fmt_duration(self):
        self.assertEqual(fmt_duration(3720), "1h02m")
        self.assertEqual(fmt_duration(125), "2m05s")


if __name__ == "__main__":
    unittest.main()
