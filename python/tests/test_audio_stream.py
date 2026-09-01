"""
Tests for decode_audio_format (audio_capture dual-layout auto-detection) and
AudioFrame.
"""

import struct

from neoruntime_ipc_sdk.audio_stream import AudioFrame, decode_audio_format

# Raw 30-byte header of frame 0 captured on a live NE503 (a test device,
# 2026-08-19): the platform daemon writes the VIDEO EncHeader layout on
# audio_capture.sock — total=2078, codec=pcm, pts=0x00004b9214fa5ce6,
# width=0, height=0, dts==pts. Audio-view values would be rate=0, ch=0,
# bits=low32(pts)=0x14fa5ce6, fsize=high32(pts)=0x4b92 != payload 2048.
GOLDEN_DEVICE_HEADER = bytes.fromhex(
    "1e0800000000e65cfa14924b00000000000000000000e65cfa14924b0000"
)
GOLDEN_PTS = 0x00004B9214FA5CE6
GOLDEN_PTS_LOW = 0x14FA5CE6
GOLDEN_PTS_HIGH = 0x4B92


class TestDecodeAudioFormat:
    def test_golden_device_video_layout(self):
        # Video-layout tail: must report unknown format, not garbage values,
        # and surface the 64-bit dts (== pts on the current device).
        rate, channels, bits, dts = decode_audio_format(GOLDEN_DEVICE_HEADER,
                                                        2078 - 30)
        assert rate == 0
        assert channels == 0
        assert bits == 0
        assert dts == GOLDEN_PTS

    def test_plausible_audio_layout(self):
        # A future daemon that fills the audio layout is picked up unchanged.
        hdr = struct.pack("<IBBQIIII", 30 + 9600, 0, 0, 500000,
                          48000, 1, 16, 9600)
        rate, channels, bits, dts = decode_audio_format(hdr, 9600)
        assert (rate, channels, bits, dts) == (48000, 1, 16, 0)

    def test_frame_size_mismatch_falls_back(self):
        # Each field is individually plausible, but the wire frame_size does
        # not match the real payload -> not a genuine audio header.
        hdr = struct.pack("<IBBQIIII", 30 + 2078, 0, 0, 0,
                          48000, 1, GOLDEN_PTS_LOW, GOLDEN_PTS_HIGH)
        rate, channels, bits, dts = decode_audio_format(hdr, 2048)
        assert (rate, channels, bits) == (0, 0, 0)
        assert dts == GOLDEN_PTS

    def test_invalid_tails_fall_back(self):
        # 12-bit samples are invalid on the wire (% 8 != 0).
        hdr = struct.pack("<IBBQIIII", 30 + GOLDEN_PTS_HIGH, 0, 0, 0,
                          44100, 2, 12, GOLDEN_PTS_HIGH)
        rate, _, _, dts = decode_audio_format(hdr, GOLDEN_PTS_HIGH)
        assert rate == 0
        assert dts == (12 | (GOLDEN_PTS_HIGH << 32))

        # Video widths are arbitrary (< 8 kHz here) -> never a plausible rate.
        hdr = struct.pack("<IBBQIIII", 30 + GOLDEN_PTS_HIGH, 0, 0, 0,
                          100, 4, GOLDEN_PTS_LOW, GOLDEN_PTS_HIGH)
        rate, channels, bits, dts = decode_audio_format(hdr, GOLDEN_PTS_HIGH)
        assert (rate, channels, bits) == (0, 0, 0)
        assert dts == GOLDEN_PTS


class TestAudioFrame:
    def test_dts_ns_defaults_to_zero(self):
        frame = AudioFrame(codec=0, flags=0, pts_ns=1,
                           sample_rate=0, channels=0, bits_per_sample=0,
                           data=b"x")
        assert frame.dts_ns == 0

    def test_duration_ms_guards_unknown_format(self):
        # Video-layout fallback -> rate/ch/bits all 0 -> duration_ms stays 0.0.
        frame = AudioFrame(codec=0, flags=0, pts_ns=1,
                           sample_rate=0, channels=0, bits_per_sample=0,
                           data=b"\x00" * 2048, dts_ns=GOLDEN_PTS)
        assert frame.dts_ns == GOLDEN_PTS
        assert frame.duration_ms == 0.0  # property, not a method (Python SDK)