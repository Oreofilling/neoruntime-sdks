"""Offline unit tests for the device perf-suite statistics helpers.

These run in normal CI (no device needed): the pure functions in
``tests/device/perf_common.py`` must be importable and correct without
``NEORUNTIME_DEVICE`` — only the SDK-touching test classes are gated.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "device")
)

from perf_common import (  # noqa: E402
    arrival_stats,
    median_round,
    model_input_geometry,
    perf_model_id,
    percentile_stats,
    pick_model,
)


class PercentileStatsTest(unittest.TestCase):
    def test_known_distribution(self):
        # 1..100 inclusive: inclusive-method percentiles are exact members.
        samples = [float(i) for i in range(1, 101)]
        stats = percentile_stats(samples)
        self.assertEqual(stats["n"], 100)
        self.assertEqual(stats["ok"], 100)
        self.assertEqual(stats["err"], 0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 100.0)
        self.assertEqual(stats["p50"], 50.5)
        self.assertEqual(stats["p90"], 90.1)
        self.assertEqual(stats["p95"], 95.05)
        self.assertEqual(stats["p99"], 99.01)

    def test_error_count_and_rate(self):
        stats = percentile_stats([1.0, 2.0, 3.0], ok=2, err=1)
        self.assertEqual((stats["ok"], stats["err"]), (2, 1))
        self.assertAlmostEqual(stats["err_pct"], 33.33, places=2)

    def test_single_sample_has_no_percentiles(self):
        stats = percentile_stats([7.0])
        self.assertEqual(stats["n"], 1)
        self.assertIsNone(stats["p50"])
        self.assertEqual(stats["max"], 7.0)

    def test_empty_samples(self):
        stats = percentile_stats([])
        self.assertEqual(stats["n"], 0)
        self.assertIsNone(stats["mean"])


class ArrivalStatsTest(unittest.TestCase):
    def test_continuous_stream(self):
        # 10 arrivals, no gaps, 2 seconds -> 5 fps, zero drops.
        seqs = list(range(100, 110))
        times = [i * 0.2 for i in range(10)]
        stats = arrival_stats(seqs, times, duration_s=2.0)
        self.assertEqual(stats["frames"], 10)
        self.assertAlmostEqual(stats["fps"], 5.0)
        self.assertEqual(stats["drops"], 0)
        # Span is known (100..109) so 0% drops is a measurement, not a
        # unknown — None is reserved for "no sequence numbers at all".
        self.assertEqual(stats["drop_pct"], 0.0)

    def test_gaps_count_as_drops(self):
        # seq jumps 0,1,2,5,6 -> one gap of 2 missing frames; loss is
        # measured against the expected span 0..6 (7 frames), the
        # standard lost/expected packet-loss ratio -> 2/7.
        seqs = [0, 1, 2, 5, 6]
        times = [float(i) for i in range(5)]
        stats = arrival_stats(seqs, times, duration_s=1.0)
        self.assertEqual(stats["frames"], 5)
        self.assertEqual(stats["drops"], 2)
        self.assertAlmostEqual(stats["drop_pct"], 28.57, places=2)

    def test_no_seq_yields_null_drop_metrics(self):
        # Shorter/absent seq list degrades to no-drop accounting but
        # still reports arrival gaps from the timestamps alone.
        stats = arrival_stats([], [0.0, 0.1, 0.2], duration_s=1.0)
        self.assertEqual(stats["frames"], 3)
        self.assertIsNone(stats["drops"])
        self.assertAlmostEqual(stats["gap_p50"], 100.0)

    def test_gap_percentiles_from_intervals(self):
        seqs = list(range(4))
        times = [0.0, 0.1, 0.3, 0.7]
        stats = arrival_stats(seqs, times, duration_s=1.0)
        self.assertAlmostEqual(stats["gap_p50"], 200.0)  # ms
        self.assertAlmostEqual(stats["gap_max"], 400.0)


class MedianRoundTest(unittest.TestCase):
    def test_picks_median_round_and_reports_spread(self):
        rounds = [
            {"p50": 10.0, "p95": 20.0},
            {"p50": 12.0, "p95": 22.0},
            {"p50": 11.0, "p95": 21.0},
        ]
        best, spread = median_round(rounds, key="p50")
        self.assertEqual(best["p50"], 11.0)
        self.assertEqual(spread["rounds"], 3)
        self.assertEqual(spread["min"], 10.0)
        self.assertEqual(spread["max"], 12.0)
        # spread relative to the fastest round (min), the pessimistic
        # "how much slower than best case" reading.
        self.assertAlmostEqual(spread["spread_pct"], 20.0, places=2)

    def test_single_round(self):
        best, spread = median_round([{"p50": 5.0}], key="p50")
        self.assertEqual(best["p50"], 5.0)
        self.assertEqual(spread["spread_pct"], 0.0)


class PickModelTest(unittest.TestCase):
    def test_preference_order_then_fallback(self):
        import tempfile

        import perf_common

        with tempfile.TemporaryDirectory() as td:
            old = perf_common.MODEL_DIR
            perf_common.MODEL_DIR = td
            try:
                self.assertIsNone(pick_model("yolo_world_v2s.hef"))
                open(os.path.join(td, "hailo_yolov8n_384_640.hef"),
                     "wb").close()
                open(os.path.join(td, "yolo_world_v2s_540.hef"),
                     "wb").close()
                # preferred name wins regardless of alphabetical order
                self.assertEqual(
                    os.path.basename(pick_model("yolo_world_v2s.hef",
                                                 "yolo_world_v2s_540.hef")),
                    "yolo_world_v2s_540.hef")
                # no preference match → first .hef alphabetically
                self.assertEqual(
                    os.path.basename(pick_model("absent.hef")),
                    "hailo_yolov8n_384_640.hef")
            finally:
                perf_common.MODEL_DIR = old


class ModelInputGeometryTest(unittest.TestCase):
    def test_height_width_pair(self):
        # hailo_yolov8n_384_640.hef → 640×384 (H_W convention)
        self.assertEqual(
            model_input_geometry("/m/hailo_yolov8n_384_640.hef"), (640, 384))

    def test_single_height_assumes_16_9(self):
        # yolo_world_v2s_540.hef → 960×540
        self.assertEqual(
            model_input_geometry("/m/yolo_world_v2s_540.hef"), (960, 540))

    def test_version_digits_are_not_geometry(self):
        # v2s must not be read as a height; only the trailing number is
        self.assertEqual(
            model_input_geometry("/m/yolo_world_v2s.hef"), None)

    def test_no_numbers(self):
        self.assertIsNone(model_input_geometry("/m/yolov5m_vehicles.hef"))


class PerfModelIdTest(unittest.TestCase):
    def test_id_derives_from_filename(self):
        # Same file → same id across runs and devices: the daemon keeps an
        # existing id's binding when a re-registration names a different
        # file, so ids must never outlive their file choice.
        self.assertEqual(
            perf_model_id("/m/hailo_yolov8n_384_640.hef"),
            "sdk-perf-hailo_yolov8n_384_640")

    def test_different_files_never_share_an_id(self):
        self.assertNotEqual(
            perf_model_id("/m/hailo_yolov8n_384_640.hef"),
            perf_model_id("/m/yolo_world_v2s_540.hef"))

    def test_pathless_or_none_degrades_not_crashes(self):
        self.assertEqual(perf_model_id("model.hef"), "sdk-perf-model")
        self.assertTrue(perf_model_id(None).startswith("sdk-perf-"))


if __name__ == "__main__":
    unittest.main()
