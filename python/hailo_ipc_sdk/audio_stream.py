"""
Audio Stream Client

Reads encoded audio frames from the camera-daemon EncodedPublisher UDS socket.
Uses the same 30-byte header protocol as the video encoded publisher.

NOTE: the current platform daemon writes the VIDEO header layout on this
socket (see decode_audio_format below); the audio-specific format fields are
auto-detected and reported as 0/unknown until the daemon fills them.

Socket path: /run/aipc/encoded/audio_capture.sock
"""

import logging
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

logger = logging.getLogger("hailo_ipc_sdk.audio_stream")

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


def _audio_layout_plausible(rate: int, channels: int, bits: int,
                            frame_size: int, payload_size: int) -> bool:
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
    codec: int          # 0=pcm, 1=aac, 2=g711a, 3=g711u
    flags: int          # bit0 = keyframe
    pts_ns: int         # Presentation timestamp (nanoseconds)
    sample_rate: int
    channels: int
    bits_per_sample: int
    data: bytes         # Raw audio payload
    dts_ns: int = 0     # Decode timestamp; set only when the daemon sends the
                        # video-style header tail (0 otherwise)

    @property
    def is_keyframe(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def codec_name(self) -> str:
        return {0: "pcm", 1: "aac", 2: "g711a", 3: "g711u"}.get(self.codec, f"unknown({self.codec})")

    @property
    def duration_ms(self) -> float:
        """Estimated frame duration in ms based on PCM parameters."""
        if self.codec == 0 and self.sample_rate > 0 and self.channels > 0 and self.bits_per_sample > 0:
            bytes_per_sample = self.bits_per_sample // 8
            total_samples = len(self.data) // (bytes_per_sample * self.channels)
            return total_samples / self.sample_rate * 1000.0
        return 0.0


class AudioStreamClient:
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

    def __init__(self, socket_path: Optional[str] = None):
        if socket_path is None:
            socket_path = os.getenv(
                "AUDIO_CAPTURE_SOCK_PATH",
                "/run/aipc/encoded/audio_capture.sock",
            )
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        sock.settimeout(5.0)
        logger.info("AudioStreamClient: connected to %s", self.socket_path)
        return sock

    def _get_sock(self) -> socket.socket:
        with self._lock:
            if self._sock is None:
                self._sock = self._connect()
            return self._sock

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("AudioStreamClient: socket closed")
            buf.extend(chunk)
        return bytes(buf)

    def _recv_frame(self, sock: socket.socket) -> Optional[AudioFrame]:
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
            header_data, payload_size)

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

    def _reconnect(self) -> socket.socket:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
            self._sock = self._connect()
            return self._sock

    def get_frame(self, timeout_ms: int = 5000) -> Optional[AudioFrame]:
        """Get a single audio frame. Returns None on timeout."""
        sock = self._get_sock()
        sock.settimeout(timeout_ms / 1000.0)
        try:
            return self._recv_frame(sock)
        except socket.timeout:
            return None

    def subscribe(self, reconnect: bool = True) -> Iterator[AudioFrame]:
        """Yield audio frames continuously. Auto-reconnects if enabled."""
        sock = self._get_sock()
        while True:
            frame = self._recv_frame(sock)
            if frame is not None:
                yield frame
                continue

            if not reconnect:
                break

            logger.info("AudioStreamClient: reconnecting...")
            time.sleep(0.5)
            try:
                sock = self._reconnect()
            except OSError:
                logger.warning("AudioStreamClient: reconnect failed, retrying in 2s")
                time.sleep(2.0)

    def on_frame(self, callback: Callable[[AudioFrame], None]) -> threading.Thread:
        """Start a background thread that calls callback for each frame."""
        def _run():
            for frame in self.subscribe():
                try:
                    callback(frame)
                except Exception:
                    logger.exception("AudioStreamClient: callback error")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
        logger.info("AudioStreamClient: closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
