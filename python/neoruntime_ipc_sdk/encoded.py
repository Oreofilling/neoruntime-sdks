"""Encoded video stream client (EncodedPublisher UDS socket)."""

from __future__ import annotations

import logging
import os
import socket
import struct
from dataclasses import dataclass

from ._transport import UdsStreamClient

logger = logging.getLogger("neoruntime_ipc_sdk.encoded")

__all__ = ["EncodedFrame", "EncodedStreamClient"]


@dataclass
class EncodedFrame:
    """Encoded video frame (H.264/H.265) from the EncodedPublisher."""

    codec: int  # 0=h264, 1=h265
    flags: int  # bit0 = keyframe
    pts_ns: int  # Presentation timestamp (nanoseconds)
    width: int
    height: int
    dts_ns: int  # Decode timestamp (nanoseconds)
    data: bytes  # Encoded NALU payload

    @property
    def is_keyframe(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def codec_name(self) -> str:
        return {0: "h264", 1: "h265"}.get(self.codec, f"unknown({self.codec})")


# Encoded video header: 30 bytes, little-endian
# [0:4]   uint32  total_size (header + payload)
# [4]     uint8   codec (0=h264, 1=h265)
# [5]     uint8   flags (bit0 = keyframe)
# [6:14]  uint64  pts_ns
# [14:18] uint32  width
# [18:22] uint32  height
# [22:30] uint64  dts_ns
_ENC_HEADER_SIZE = 30
_ENC_HEADER_FMT = "<I BB Q II Q"


class EncodedStreamClient(UdsStreamClient):
    """Read encoded video frames from an EncodedPublisher UDS socket.

    Connects to sockets like ``/run/aipc/encoded/main.sock`` and yields
    :class:`EncodedFrame` objects containing H.264/H.265 NAL units.

    Usage::

        client = EncodedStreamClient()                    # main stream
        client = EncodedStreamClient(stream_id="sub")     # sub stream
        client = EncodedStreamClient("/run/aipc/encoded/main.sock")  # explicit
        for frame in client.subscribe():
            print(f"{frame.codec_name} {frame.width}x{frame.height} "
                  f"keyframe={frame.is_keyframe} {len(frame.data)}B")
    """

    # Socket lifecycle, reconnect, get_frame/subscribe/on_frame and close live
    # in UdsStreamClient; only the wire framing stays here.

    def __init__(
        self,
        socket_path: str | None = None,
        *,
        stream_id: str = "main",
        socket_dir: str | None = None,
    ):
        """Resolve the socket path.

        ``socket_path`` (explicit) wins; otherwise the path is derived as
        ``{socket_dir}/{stream_id}.sock`` with ``socket_dir`` defaulting to
        ``/run/aipc/encoded`` (overridable via ``ENCODED_SOCK_DIR``).
        """
        if socket_path is None:
            base = socket_dir or os.getenv("ENCODED_SOCK_DIR", "/run/aipc/encoded")
            socket_path = os.path.join(base, f"{stream_id}.sock")
        super().__init__(socket_path)

    def _recv_frame(self, sock: socket.socket) -> EncodedFrame | None:
        try:
            header_data = self._recv_exact(sock, _ENC_HEADER_SIZE)
        except (ConnectionError, OSError):
            return None

        if len(header_data) < _ENC_HEADER_SIZE:
            return None

        values = struct.unpack(_ENC_HEADER_FMT, header_data)
        total_size = values[0]
        codec = values[1]
        flags = values[2]
        pts_ns = values[3]
        width = values[4]
        height = values[5]
        dts_ns = values[6]

        payload_size = total_size - _ENC_HEADER_SIZE
        if payload_size < 0 or payload_size > 50 * 1024 * 1024:
            logger.warning("EncodedStreamClient: bogus payload_size=%d", payload_size)
            return None

        try:
            payload = self._recv_exact(sock, payload_size) if payload_size > 0 else b""
        except (ConnectionError, OSError):
            return None

        return EncodedFrame(
            codec=codec,
            flags=flags,
            pts_ns=pts_ns,
            width=width,
            height=height,
            dts_ns=dts_ns,
            data=payload,
        )
