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
    flags: int  # bit0 = keyframe, bit1 = seq present
    pts_ns: int  # Presentation timestamp (nanoseconds)
    width: int
    height: int
    dts_ns: int  # Decode timestamp (nanoseconds)
    data: bytes  # Encoded NALU payload
    # Publisher packet sequence number (wire V3, flags bit1). Assigned at
    # publisher enqueue, first packet = 1; monotonically increasing per
    # stream. None when the server predates V3 (30-byte header).
    seq: int | None = None

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

# V3 extension (flags bit1 = SEQ_PRESENT): 8 more bytes, little-endian —
# [30:38] uint64 packet seq (first packet = 1, assigned at publisher
# enqueue). total_size covers the extended header. A hole in the seq
# stream is a packet the publisher accepted but this client never
# received: queue-overflow eviction, a dropped client send, or time
# spent disconnected. Clients reconcile the hole count against the
# publisher's drop counters via GetStreamStatus.
_ENC_FLAG_SEQ_PRESENT = 0x02
_ENC_SEQ_SIZE = 8


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
        # Per-client packet-seq observability. Deliberately NOT reset on
        # reconnect: packets enqueued while this client was away were
        # publisher-accepted but never received here, and the hole is the
        # thing worth seeing. A seq that goes backwards is a publisher
        # restart, not a loss — handled by rebasing in _recv_frame.
        self.seq_packets = 0    # frames that carried a seq
        self.seq_gap_events = 0 # discontinuities seen (count, not size)
        self.seq_missing = 0    # total packets inside those holes
        self.last_seq: int | None = None

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

        seq = None
        header_size = _ENC_HEADER_SIZE
        if flags & _ENC_FLAG_SEQ_PRESENT:
            try:
                seq_bytes = self._recv_exact(sock, _ENC_SEQ_SIZE)
            except (ConnectionError, OSError):
                return None
            seq = struct.unpack("<Q", seq_bytes)[0]
            header_size += _ENC_SEQ_SIZE

        payload_size = total_size - header_size
        if payload_size < 0 or payload_size > 50 * 1024 * 1024:
            logger.warning("EncodedStreamClient: bogus payload_size=%d", payload_size)
            return None

        try:
            payload = self._recv_exact(sock, payload_size) if payload_size > 0 else b""
        except (ConnectionError, OSError):
            return None

        if seq is not None:
            if self.last_seq is not None:
                if seq > self.last_seq + 1:
                    missing = seq - self.last_seq - 1
                    self.seq_gap_events += 1
                    self.seq_missing += missing
                    logger.warning(
                        "EncodedStreamClient[%s]: seq hole %d..%d missing=%d "
                        "(overflow eviction / dropped send / disconnect)",
                        self.socket_path, self.last_seq + 1, seq - 1, missing,
                    )
                # seq <= last_seq: publisher restarted (seq rebased to 1).
                # Not a loss — the next forward jump from the new baseline
                # is what counts.
            self.last_seq = seq
            self.seq_packets += 1

        return EncodedFrame(
            codec=codec,
            flags=flags,
            pts_ns=pts_ns,
            width=width,
            height=height,
            dts_ns=dts_ns,
            data=payload,
            seq=seq,
        )
