"""Perf P4 — device control plane: read-only status RPC latency.

Deliberately **non-invasive**: only status/config reads are sampled.
Physical-effect calls (focus/zoom/pan/iris/preset/ircut/led/rs485)
are excluded by policy, and GPIO is excluded outright — a single
``GET /device/gpio`` call hard-resets this device generation
(tracked as issue #46), so gpio_get/gpio_set must never run here.
"""

from __future__ import annotations

import threading
import time
import unittest

from neoruntime_ipc_sdk import CameraClient, DeviceClient

from common import known_issue
from perf_common import PerfTestCase, arrival_stats

# Issue #46: one GET on the gpio face hard-resets the device. Guarded
# by exclusion, not by try/except — the call must not happen at all.
GPIO_EXCLUDED = ("gpio_get/gpio_set excluded: issue #46 "
                 "(single read hard-resets the device)")

EVENT_WINDOW_S = 30


class T01DeviceStatus(PerfTestCase):
    """DeviceControl UDS reads."""

    area = "perf-device"
    timeout_s = 300

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = DeviceClient()
        cls.client.connect()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    @known_issue(
        "SDK 0.7.4 defect (same as test_40): device.py reads "
        "response.ir_led_on unconditionally but no bundled proto message "
        "carries the field — every call raises AttributeError regardless "
        "of daemon health. The probe below raises through; once the SDK "
        "guards the field, sampling starts on its own.")
    def test_01_get_device_status(self):
        self.mark("DeviceClient.get_device_status latency")
        self.client.get_device_status()  # probe: classifies the known defect
        self.perf_sample(self.client.get_device_status,
                         label="get_device_status", n=200)

    def test_02_get_lens_status(self):
        self.mark("DeviceClient.get_lens_status latency")
        self.perf_sample(self.client.get_lens_status,
                         label="get_lens_status", n=200)

    def test_03_af_readouts(self):
        self.mark("DeviceClient.get_af_measurement/get_autofocus_status")
        # Capability gate first (test_40 semantics): some lens HAL bridges
        # answer "not yet supported" — a device-capability outcome, not a
        # latency path. Sampling a deterministic error 100× measures
        # nothing, so probe once and record the reason instead.
        try:
            self.client.get_af_measurement()
            af_supported = True
        except Exception as exc:  # noqa: BLE001 — capability gate
            af_supported = False
            self.evidence(
                af_measurement_unsupported=f"{type(exc).__name__}: {exc}"[:120])
        if af_supported:
            self.perf_sample(self.client.get_af_measurement,
                             label="get_af_measurement", n=100)
        self.perf_sample(self.client.get_autofocus_status,
                         label="get_autofocus_status", n=100)
        self.evidence(gpio_excluded=GPIO_EXCLUDED)


class T02CameraStatus(PerfTestCase):
    """Camera-control UDS reads (config/status only)."""

    area = "perf-device"
    timeout_s = 300

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = CameraClient()
        cls.client.connect()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_01_status_reads(self):
        self.mark("CameraClient status read latency")
        self.perf_sample(self.client.get_capabilities,
                         label="get_capabilities", n=100)
        self.perf_sample(self.client.get_sensor_info,
                         label="get_sensor_info", n=100)
        self.perf_sample(self.client.get_stream_status,
                         label="get_stream_status", n=200)
        self.perf_sample(self.client.get_hardware_status,
                         label="get_hardware_status", n=100)
        self.perf_sample(self.client.get_infrared_status,
                         label="get_infrared_status", n=100)


class T03DeviceEventStream(PerfTestCase):
    """subscribe_events arrival gaps — read-only, drained off-thread.

    Device events are sparse and spontaneous (no publisher to drive);
    the drain thread collects arrival timestamps for a fixed window
    while the main thread sleeps, so a quiet bus yields "0 events",
    which is a measurement, not a failure. The blocking iterator lives
    on a daemon thread — the per-test alarm would otherwise fire while
    the main thread waits on a blocked iterator.
    """

    area = "perf-device"
    timeout_s = EVENT_WINDOW_S + 120

    def test_01_event_arrivals(self):
        self.mark("DeviceClient.subscribe_events arrival statistics")
        arrivals: list[float] = []
        types: list[str] = []

        def drain(client):
            try:
                for ev in client.subscribe_events():
                    arrivals.append(time.monotonic())
                    types.append(getattr(ev, "type", "?"))
                    if len(arrivals) >= 2000:
                        break
            except Exception:  # noqa: BLE001 — daemon thread
                pass

        client = DeviceClient()
        try:
            worker = threading.Thread(target=drain, args=(client,),
                                      daemon=True)
            worker.start()
            time.sleep(EVENT_WINDOW_S)
        finally:
            client.close()  # iterator ends; daemon thread exits anyway

        stats = arrival_stats([], arrivals, EVENT_WINDOW_S)
        self.evidence(**{"perf:device_event_arrivals": stats},
                      event_types=sorted(set(types))[:10])
        if not arrivals:
            self.record["outcome_note"] = (
                "no spontaneous device events in the window (quiet bus)")


if __name__ == "__main__":
    unittest.main()
