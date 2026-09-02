"""
Audio Stream Client

Reads encoded audio frames from the camera-daemon EncodedPublisher UDS socket.
Uses the same 30-byte header protocol as the video encoded publisher.

NOTE: the current platform daemon writes the VIDEO header layout on this
socket (see decode_audio_format below); the audio-specific format fields are
auto-detected and reported as 0/unknown until the daemon fills them.

Socket path: /run/aipc/encoded/audio_capture.sock
"""

from __future__ import annotations

import logging
import os
import socket
import struct
from dataclasses import dataclass

from ._transport import UdsStreamClient

logger = logging.getLogger("neoruntime_ipc_sdk.audio_stream")

# EncodedPublisher frame header: 30 bytes, all little-endian
# [0:4]   uint32  total_size (header + payload)
# [4]     uint8   codec (0=raw/pcm, 1=aac, 2=g711a, 3=g711u)
# [5]     uint8   flags (bit0 = keyframe)
# [6:14]  uint64  pts_ns
# [14:18] uint32  sample_rate
# [18:22] uint32  channels
# [22:26] uint32  bits_per_sample
# [26:30] uint32  frame_size (payload bytes)
#
# The daemon observed on-device (2026-08) actually sends the VIDEO layout
# "<I BB Q II Q" (width[14:18]=0, height[18:22]=0, dts_ns[22:30]==pts) and
# never fills the audio fields above. decode_audio_format() therefore
# validates the tail for audio plausibility and falls back to the video view.
_HEADER_SIZE = 30
_HEADER_FMT = "<I BB Q III I"

# Plausibility bounds for the audio-layout tail (a frame passing all of these
# is treated as a genuine audio header; anything else is a video-layout tail).
_AUDIO_RATE_MIN = 8000
_AUDIO_RATE_MAX = 192000


def _audio_layout_plausible(
    rate: int, channels: int, bits: int, frame_size: int, payload_size: int
) -> bool:
    return (
        frame_size == payload_size
        and _AUDIO_RATE_MIN <= rate <= _AUDIO_RATE_MAX
        and 1 <= channels <= 8
        and 8 <= bits <= 32
        and bits % 8 == 0
    )


def decode_audio_format(header: bytes, payload_size: int) -> tuple:
    """Decode the format fields of a 30-byte audio-capture header.

    The platform daemon currently writes the VIDEO EncHeader layout on
    audio_capture.sock (width/height at [14:22] = 0, dts at [22:30]) instead
    of the documented audio layout, so the tail [14:30] is first validated for
    audio plausibility; when that fails it is re-read as the video tail
    (width, height, dts_ns) and the format parameters are reported as unknown
    (0). A future daemon that fills the audio layout is picked up
    automatically, with no client change.

    Returns (sample_rate, channels, bits_per_sample, dts_ns); dts_ns is 0
    unless the video-layout fallback was taken.
    """
    rate, channels, bits, frame_size = struct.unpack("<IIII", header[14:30])
    if _audio_layout_plausible(rate, channels, bits, frame_size, payload_size):
        return rate, channels, bits, 0
    # Video-layout tail: [22:30] is a full 64-bit dts timestamp.
    dts = bits | (frame_size << 32)
    return 0, 0, 0, dts


@dataclass
class AudioFrame:
    """Encoded or raw audio frame from the audio capture pipeline."""

    codec: int  # 0=pcm, 1=aac, 2=g711a, 3=g711u
    flags: int  # bit0 = keyframe
    pts_ns: int  # Presentation timestamp (nanoseconds)
    sample_rate: int
    channels: int
    bits_per_sample: int
    data: bytes  # Raw audio payload
    dts_ns: int = 0  # Decode timestamp; set only when the daemon sends the
    # video-style header tail (0 otherwise)

    @property
    def is_keyframe(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def codec_name(self) -> str:
        return {0: "pcm", 1: "aac", 2: "g711a", 3: "g711u"}.get(
            self.codec, f"unknown({self.codec})"
        )

    @property
    def duration_ms(self) -> float:
        """Estimated frame duration in ms based on PCM parameters."""
        if (
            self.codec == 0
            and self.sample_rate > 0
            and self.channels > 0
            and self.bits_per_sample > 0
        ):
            bytes_per_sample = self.bits_per_sample // 8
            total_samples = len(self.data) // (bytes_per_sample * self.channels)
            return total_samples / self.sample_rate * 1000.0
        return 0.0


class AudioStreamClient(UdsStreamClient):
    """
    Audio frame subscriber via Unix Domain Socket.

    Connects to the EncodedPublisher audio_capture socket and yields
    AudioFrame objects containing captured audio data.

    Usage::

        client = AudioStreamClient()

        # Iterator pattern
        for frame in client.subscribe():
            print(f"Audio: {frame.codec_name} {frame.sample_rate}Hz "
                  f"{frame.channels}ch {len(frame.data)} bytes")

        # Callback pattern
        client.on_frame(lambda f: process(f))
        # ... later ...
        client.close()
    """

    def __init__(self, socket_path: str | None = None):
        if socket_path is None:
            socket_path = os.getenv(
                "AUDIO_CAPTURE_SOCK_PATH",
                "/run/aipc/encoded/audio_capture.sock",
            )
        super().__init__(socket_path)

    # Socket lifecycle, reconnect, get_frame/subscribe/on_frame and close live
    # in UdsStreamClient; only the audio wire framing stays here.

    def _recv_frame(self, sock: socket.socket) -> AudioFrame | None:
        try:
            header_data = self._recv_exact(sock, _HEADER_SIZE)
        except (ConnectionError, OSError):
            return None

        if len(header_data) < _HEADER_SIZE:
            return None

        values = struct.unpack(_HEADER_FMT, header_data)
        total_size = values[0]
        codec = values[1]
        flags = values[2]
        pts_ns = values[3]

        payload_size = total_size - _HEADER_SIZE
        if payload_size < 0 or payload_size > 10 * 1024 * 1024:
            logger.warning("AudioStreamClient: bogus payload_size=%d, skipping", payload_size)
            return None

        sample_rate, channels, bits_per_sample, dts_ns = decode_audio_format(
            header_data, payload_size
        )

        try:
            payload = self._recv_exact(sock, payload_size) if payload_size > 0 else b""
        except (ConnectionError, OSError):
            return None

        return AudioFrame(
            codec=codec,
            flags=flags,
            pts_ns=pts_ns,
            sample_rate=sample_rate,
            channels=channels,
            bits_per_sample=bits_per_sample,
            data=payload,
            dts_ns=dts_ns,
        )
