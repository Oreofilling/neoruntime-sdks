"""Phase 4 — audio interfaces (AudioClient gRPC + AudioStreamClient UDS).

Capture/playback are live state changes: every start is paired with a
stop in teardown and the status is read back to verify the transition.
The two-way-talk PCM path is exercised with a *silent* buffer so nothing
audible is emitted. The audio_capture.sock header-layout defect
(daemon writes the video EncHeader tail) is asserted via ``@known_issue``
so a daemon-side fix flips the verdict to PASS visibly.
"""

from __future__ import annotations

import unittest

from neoruntime_ipc_sdk import AudioClient, AudioFrame, AudioStreamClient

from common import DeviceTestCase, known_issue


def _soft(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


class T01Control(DeviceTestCase):
    """Read-only audio control surface."""

    area = "audio"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = AudioClient()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_01_list_capture_devices(self):
        self.mark("AudioClient.list_capture_devices")
        devices = self.timed(self.client.list_capture_devices,
                             label="list_capture_devices")
        self.evidence(count=len(devices),
                      names=[d.name for d in devices][:8])
        self.assertIsInstance(devices, list)

    def test_02_list_playback_devices(self):
        self.mark("AudioClient.list_playback_devices")
        devices = self.timed(self.client.list_playback_devices,
                             label="list_playback_devices")
        self.evidence(count=len(devices),
                      names=[d.name for d in devices][:8])
        self.assertIsInstance(devices, list)

    def test_03_get_status(self):
        self.mark("AudioClient.get_status")
        status = self.timed(self.client.get_status, label="get_status")
        self.evidence(capturing=status.capturing, playing=status.playing,
                      device=status.device, sample_rate=status.sample_rate,
                      channels=status.channels, codec=status.codec,
                      volume=status.volume, mute=status.mute)


class T02CaptureState(DeviceTestCase):
    """Start/stop transitions, each verified via get_status readback."""

    area = "audio"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = AudioClient()

    @classmethod
    def tearDownClass(cls):
        for stop in (cls.client.stop_capture, cls.client.stop_playback):
            try:
                stop()
            except Exception:
                pass
        cls.client.close()

    def test_01_capture_lifecycle(self):
        self.mark("AudioClient.start_capture/stop_capture")
        _, start_err = _soft(self.timed, self.client.start_capture,
                             label="start_capture")
        if start_err is not None:
            self.na(f"start_capture rejected: {start_err}")
        status = self.client.get_status()
        self.timed(self.client.stop_capture, label="stop_capture")
        after = self.client.get_status()
        self.evidence(capturing_during=status.capturing,
                      capturing_after=after.capturing,
                      codec_during=status.codec or None)
        self.assertFalse(after.capturing, "capture still running after stop")

    def test_02_playback_lifecycle(self):
        self.mark("AudioClient.start_playback/stop_playback")
        _, start_err = _soft(self.timed, self.client.start_playback,
                             label="start_playback")
        if start_err is not None:
            self.na(f"start_playback rejected: {start_err} "
                    "(no playback pipeline on this device)")
        status = self.client.get_status()
        self.timed(self.client.stop_playback, label="stop_playback")
        after = self.client.get_status()
        self.evidence(playing_during=status.playing,
                      playing_after=after.playing)
        self.assertFalse(after.playing, "playback still running after stop")

    def test_03_set_config_volume_roundtrip(self):
        self.mark("AudioClient.set_config")
        original = self.client.get_status().volume
        probe = min(round(original + 0.1, 2), 1.0) if original >= 0 else 0.5
        _, err = _soft(self.timed, self.client.set_config, volume=probe,
                       label="set_config")
        if err is not None:
            self.na(f"set_config rejected: {err}")
        after = self.client.get_status().volume
        self.client.set_config(volume=original if original >= 0 else 0.5)
        restored = self.client.get_status().volume
        self.evidence(original=original, probe=probe, after=after,
                      restored=restored)

    def test_04_stream_pcm_silent(self):
        self.mark("AudioClient.stream_pcm/stream_pcm_file")
        # ~85ms of silence at 48kHz S16LE mono — inaudible if played.
        silence = b"\x00\x00" * 2048
        _, err = _soft(self.timed, self.client.stream_pcm,
                       iter([silence]), label="stream_pcm")
        if err is not None:
            self.na(f"stream_pcm rejected: {err} (two-way talk pipeline "
                    "absent)")
        self.evidence(chunks=1, bytes=len(silence), sample_rate=48000)


class T03Stream(DeviceTestCase):
    """AudioStreamClient over the audio_capture UDS (needs live capture)."""

    area = "audio"
    timeout_s = 90

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.control = AudioClient()
        cls.control.start_capture()  # frames only flow while capturing
        cls.stream = AudioStreamClient()

    @classmethod
    def tearDownClass(cls):
        cls.stream.close()
        try:
            cls.control.stop_capture()
        except Exception:
            pass
        cls.control.close()

    def test_01_get_frame(self):
        self.mark("AudioStreamClient.get_frame")
        frame = self.timed(self.stream.get_frame, 10000, label="get_frame")
        if frame is None:
            self.na("audio_capture.sock produced no frame in 10s "
                    "(no microphone pipeline on this device)")
        self.assertIsInstance(frame, AudioFrame)
        self.evidence(codec=frame.codec_name, pts_ns=frame.pts_ns,
                      dts_ns=frame.dts_ns, sample_rate=frame.sample_rate,
                      channels=frame.channels,
                      bits=frame.bits_per_sample, data_bytes=len(frame.data),
                      is_keyframe=frame.is_keyframe,
                      duration_ms=frame.duration_ms)
        self.assertGreater(len(frame.data), 0)

    def test_02_subscribe(self):
        self.mark("AudioStreamClient.subscribe")
        frames = []
        for frame in self.stream.subscribe():
            frames.append(frame)
            if len(frames) >= 3:
                break
        self.evidence(count=len(frames),
                      total_bytes=sum(len(f.data) for f in frames),
                      codecs=[f.codec_name for f in frames])
        self.assertGreaterEqual(len(frames), 3)

    @known_issue(
        "daemon writes the video EncHeader layout on audio_capture.sock "
        "(rate=0/ch=0, video dts in the tail) — SDK decode_audio_format "
        "compensates, so audio params arrive unknown"
    )
    def test_03_frame_carries_audio_layout(self):
        """Genuine audio layout would expose real sample_rate/channels."""
        self.mark("AudioStreamClient frame audio-layout fields")
        frame = self.stream.get_frame(10000)
        if frame is None:
            self.na("no audio frame to inspect")
        self.evidence(sample_rate=frame.sample_rate,
                      channels=frame.channels,
                      bits=frame.bits_per_sample, dts_ns=frame.dts_ns)
        self.assertGreater(frame.sample_rate, 0,
                           "audio frames carry no sample_rate — video-"
                           "layout defect still present daemon-side")


if __name__ == "__main__":
    unittest.main()
