"""
Recording utilities - pure-python MPEG-TS muxing for EncodedFrame streams.

TsWriter writes a single .ts event clip; HlsWriter cuts keyframe-aligned
segments plus an m3u8 live playlist; PrerollBuffer keeps a ring of encoded
frames so an app can dump "seconds before the event" when something happens.

No ffmpeg is required: Annex-B payloads from the EncodedPublisher are
packetised into PES/TS directly, without re-encoding.

Example:
    with HlsWriter("/var/tmp/hls", segment_seconds=6.0) as hls:
        for frame in client.subscribe_encoded("main"):
            hls.write(frame)
    # serve /var/tmp/hls/index.m3u8 over HTTP for hls.js
"""

import math
import os
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from .media import EncodedFrame

TS_PACKET_SIZE = 188
PAT_PID = 0x0000
PMT_PID = 0x1000
VIDEO_PID = 0x0100

STREAM_TYPES = {"h264": 0x1B, "h265": 0x24}

_PTS_DTS_MASK = (1 << 33) - 1


# --------------------------------------------------------------------------
# low-level bit packing
# --------------------------------------------------------------------------

def _crc32_mpeg(data: bytes) -> bytes:
    """MPEG-2 CRC-32 (poly 0x04C11DB7, init 0xFFFFFFFF, MSB-first)."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else crc << 1
            crc &= 0xFFFFFFFF
    return crc.to_bytes(4, "big")


def _ts_timestamp(value: int, marker: int) -> bytes:
    """Encode a 33-bit PTS/DTS into the 5-byte marker format (2^33 wrap)."""
    v = value & _PTS_DTS_MASK
    return bytes([
        marker | ((v >> 29) & 0x0E),
        (v >> 22) & 0xFF,
        0x01 | ((v >> 14) & 0xFE),
        (v >> 7) & 0xFF,
        0x01 | ((v << 1) & 0xFE),
    ])


def _pcr_bytes(base: int) -> bytes:
    """6-byte PCR: 33-bit base (90kHz) + 6 reserved bits + 9-bit extension."""
    return (((base & _PTS_DTS_MASK) << 15) | 0x7E00).to_bytes(6, "big")


def _ticks_90k(ns: int) -> int:
    return (ns * 9) // 100_000


def _psi_table(table_id: int, body: bytes) -> bytes:
    """Wrap a PSI section body with header + CRC32."""
    length = len(body) + 4
    head = bytes([table_id, 0xB0 | ((length >> 8) & 0x0F), length & 0xFF])
    data = head + body
    return data + _crc32_mpeg(data)


def _pat_table() -> bytes:
    body = (
        b"\x00\x01"                                # transport_stream_id
        b"\xc1\x00\x00"                            # version/section numbers
        b"\x00\x01"                                # program_number 1
        + bytes([0xE0 | (PMT_PID >> 8), PMT_PID & 0xFF])
    )
    return _psi_table(0x00, body)


def _pmt_table(stream_type: int) -> bytes:
    body = (
        b"\x00\x01"                                # program_number
        b"\xc1\x00\x00"                            # version/section numbers
        + bytes([0xE0 | (VIDEO_PID >> 8), VIDEO_PID & 0xFF])   # PCR_PID
        + b"\xf0\x00"                              # program_info_length
        + bytes([stream_type,
                 0xE0 | (VIDEO_PID >> 8), VIDEO_PID & 0xFF,
                 0xF0, 0x00])                      # ES loop
    )
    return _psi_table(0x02, body)


def _ts_packet(pid: int, payload: bytes, cc: int, unit_start: bool = False,
               pcr_base: Optional[int] = None) -> bytes:
    """Build one 188-byte TS packet; adds an adaptation field when needed."""
    afc = 0x01                                     # payload only
    body = bytearray()
    if pcr_base is not None or len(payload) < TS_PACKET_SIZE - 4:
        if pcr_base is None and len(payload) == TS_PACKET_SIZE - 5:
            # 183B tail: exactly one byte left, a length-only adaptation
            # field (ISO 13818-1: the length byte itself is the stuffing)
            afc = 0x03
            body.append(0)
        else:
            af = bytearray([0x10 if pcr_base is not None else 0x00])
            if pcr_base is not None:
                af += _pcr_bytes(pcr_base)
            pad = (TS_PACKET_SIZE - 4) - 1 - len(af) - len(payload)
            if pad < 0:
                raise ValueError("TS payload overflow with adaptation field")
            af += b"\xff" * pad
            afc = 0x03
            body.append(len(af))
            body += af
    body += payload
    header = bytes([
        0x47,
        (0x40 if unit_start else 0x00) | ((pid >> 8) & 0x1F),
        pid & 0xFF,
        (afc << 4) | (cc & 0x0F),
    ])
    return header + bytes(body)


def _build_pes(frame: EncodedFrame) -> bytes:
    """One Annex-B frame -> one PES packet (stream_id 0xE0, unbounded)."""
    pts = _ticks_90k(frame.pts_ns)
    if frame.dts_ns != frame.pts_ns:
        flags = 0xC0
        ext = _ts_timestamp(pts, 0x20) + _ts_timestamp(_ticks_90k(frame.dts_ns), 0x10)
    else:
        flags = 0x80
        ext = _ts_timestamp(pts, 0x20)
    header = bytearray(b"\x00\x00\x01\xe0")
    header += b"\x00\x00"                          # PES_packet_length (video)
    header += bytes([0x80, flags, len(ext)]) + ext
    return bytes(header) + frame.data


# --------------------------------------------------------------------------
# TsWriter
# --------------------------------------------------------------------------

class TsWriter:
    """Single-file MPEG-TS recorder for EncodedFrame Annex-B payloads.

    Usage:
        w = TsWriter("event.ts", codec="h264")
        for frame in client.subscribe_encoded("main"):
            w.write(frame)
        w.close()
    """

    def __init__(self, path: str, codec: str = "h264"):
        if codec not in STREAM_TYPES:
            raise ValueError(f"codec must be one of {sorted(STREAM_TYPES)}")
        self._stream_type = STREAM_TYPES[codec]
        self._f = open(path, "wb")
        self._cc: Dict[int, int] = {}
        self._pcr_written = False
        self._write_psi()

    def _next_cc(self, pid: int) -> int:
        cc = self._cc.get(pid, 0)
        self._cc[pid] = (cc + 1) % 16
        return cc

    def _write_psi(self) -> None:
        self._f.write(_ts_packet(PAT_PID, b"\x00" + _pat_table(),
                                 self._next_cc(PAT_PID), unit_start=True))
        self._f.write(_ts_packet(PMT_PID, b"\x00" + _pmt_table(self._stream_type),
                                 self._next_cc(PMT_PID), unit_start=True))

    def write(self, frame: EncodedFrame) -> None:
        """Packetise one encoded frame into the TS stream."""
        if self._f is None:
            raise RuntimeError("TsWriter is closed")
        pes = _build_pes(frame)
        pcr = None
        if not self._pcr_written:
            pcr = _ticks_90k(frame.pts_ns)
            self._pcr_written = True
        pos = 0
        first = True
        while pos < len(pes):
            # adaptation field carrying the PCR costs 8 bytes (length +
            # flags + PCR), so the first chunk of a PES is capped lower
            cap = (TS_PACKET_SIZE - 12) if (first and pcr is not None) \
                else (TS_PACKET_SIZE - 4)
            chunk = pes[pos:pos + cap]
            pos += len(chunk)
            self._f.write(_ts_packet(VIDEO_PID, chunk, self._next_cc(VIDEO_PID),
                                     unit_start=first, pcr_base=pcr if first else None))
            first = False

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> "TsWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# HlsWriter
# --------------------------------------------------------------------------

class HlsWriter:
    """Keyframe-aligned HLS segmenter with a live-window m3u8 playlist.

    Every segment is self-contained (PAT+PMT first) so hls.js can join
    the stream at any segment boundary. The playlist is rewritten
    atomically (tmp + os.replace) after every segment close.

    Usage:
        hls = HlsWriter("/srv/hls", segment_seconds=6.0, window=5)
        for frame in client.subscribe_encoded("main"):
            hls.write(frame)      # blocks of frames -> seg000001.ts ...
        hls.close()               # appends #EXT-X-ENDLIST
    """

    def __init__(self, out_dir: str, segment_seconds: float = 6.0,
                 window: int = 5, codec: str = "h264"):
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if codec not in STREAM_TYPES:
            raise ValueError(f"codec must be one of {sorted(STREAM_TYPES)}")
        os.makedirs(out_dir, exist_ok=True)
        self._dir = out_dir
        self._seg_ns = int(segment_seconds * 1_000_000_000)
        self._window = max(1, int(window))
        self._codec = codec
        self._segments: List[Dict] = []            # {name, duration, discont}
        self._started = 0
        self._writer: Optional[TsWriter] = None
        self._cur_name = ""
        self._cur_discont = False
        self._first_pts = 0
        self._prev_pts: Optional[int] = None
        self._last_pts = 0
        self._closed = False

    # -- segment lifecycle --------------------------------------------------

    def write(self, frame: EncodedFrame) -> None:
        """Append a frame; cuts a segment on keyframes once it is long enough."""
        if self._closed:
            raise RuntimeError("HlsWriter is closed")
        if self._writer is None:
            if not frame.is_keyframe:
                return                              # segments start on keyframes
            self._start_segment(frame, discont=False)
            return
        if frame.is_keyframe:
            backwards = frame.pts_ns < self._last_pts
            if (frame.pts_ns - self._first_pts) >= self._seg_ns or backwards:
                self._close_segment(next_pts=frame.pts_ns, final=False)
                self._start_segment(frame, discont=backwards)
                return
        self._append(frame)

    def _start_segment(self, frame: EncodedFrame, discont: bool) -> None:
        self._started += 1
        self._cur_name = f"seg{self._started:06d}.ts"
        self._cur_discont = discont
        self._writer = TsWriter(os.path.join(self._dir, self._cur_name),
                                codec=self._codec)
        self._first_pts = frame.pts_ns
        self._prev_pts = None
        self._append(frame)

    def _append(self, frame: EncodedFrame) -> None:
        self._writer.write(frame)
        self._prev_pts = self._last_pts
        self._last_pts = frame.pts_ns

    def _close_segment(self, next_pts: Optional[int], final: bool) -> None:
        if next_pts is not None:
            duration = max((next_pts - self._first_pts) / 1e9, 0.1)
        elif self._prev_pts is not None:
            duration = (self._last_pts - self._first_pts) / 1e9 + \
                       max(self._last_pts - self._prev_pts, 0) / 1e9
        else:
            duration = 0.1
        self._writer.close()
        self._writer = None
        self._segments.append({"name": self._cur_name,
                               "duration": duration,
                               "discont": self._cur_discont})
        self._trim_window()
        self._write_playlist(final=final)

    def _trim_window(self) -> None:
        while len(self._segments) > self._window:
            removed = self._segments.pop(0)
            try:
                os.remove(os.path.join(self._dir, removed["name"]))
            except OSError:
                pass

    # -- playlist -----------------------------------------------------------

    def _write_playlist(self, final: bool) -> None:
        target = max(1, math.ceil(max(s["duration"] for s in self._segments)
                                  if self._segments else 1))
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target}",
            f"#EXT-X-MEDIA-SEQUENCE:{self._started - len(self._segments)}",
        ]
        for seg in self._segments:
            if seg["discont"]:
                lines.append("#EXT-X-DISCONTINUITY")
            lines.append(f"#EXTINF:{seg['duration']:.3f},")
            lines.append(seg["name"])
        if final:
            lines.append("#EXT-X-ENDLIST")
        tmp = os.path.join(self._dir, "index.m3u8.tmp")
        with open(tmp, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, os.path.join(self._dir, "index.m3u8"))

    def close(self) -> None:
        """Finalise the last segment and end the playlist."""
        if self._closed:
            return
        if self._writer is not None:
            self._close_segment(next_pts=None, final=True)
        elif self._segments:
            self._write_playlist(final=True)
        self._closed = True

    def __enter__(self) -> "HlsWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# PrerollBuffer
# --------------------------------------------------------------------------

class PrerollBuffer:
    """Ring buffer of encoded frames for "seconds before the event" clips.

    Usage:
        preroll = PrerollBuffer(seconds=10.0)
        for frame in client.subscribe_encoded("main"):
            preroll.push(frame)
            if event_detected:
                writer = preroll.dump("event.ts")
                # keep writing live frames to `writer`, then writer.close()
    """

    def __init__(self, seconds: float = 10.0):
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        self._seconds_ns = int(seconds * 1_000_000_000)
        self._frames: Deque[EncodedFrame] = deque()

    def push(self, frame: EncodedFrame) -> None:
        """Add a frame and evict frames older than the window."""
        self._frames.append(frame)
        while len(self._frames) > 1 and \
                frame.pts_ns - self._frames[0].pts_ns > self._seconds_ns:
            self._frames.popleft()

    @property
    def frames(self) -> List[EncodedFrame]:
        return list(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def dump(self, path: str,
             on_frame: Optional[Callable[[TsWriter], None]] = None) -> TsWriter:
        """Flush buffered preroll into a new .ts file and return the writer.

        on_frame(writer) is called right after the preroll flush - the
        typical use is starting a live-tail thread that keeps calling
        writer.write() until the event window ends.
        """
        if not self._frames:
            raise ValueError("no frames buffered")
        writer = TsWriter(path, codec=self._frames[0].codec_name)
        for frame in self._frames:
            writer.write(frame)
        if on_frame is not None:
            on_frame(writer)
        return writer
