"""
P1-10 output observability: encoded-packet sequence numbers, unified
drop counters on StreamStatus, and the GetStats sampling window.

Wire contract under test (camera-daemon EncodedPublisher):
  V2: 30-byte header "<I BB Q II Q" — total, codec, flags, pts, w, h, dts.
  V3: flags bit1 set -> 8 more bytes, uint64 LE packet seq at offset 30;
      total_size covers the 38-byte header. seq=1 for the first packet,
      assigned at publisher enqueue; monotonic per stream. A hole is a
      packet the publisher accepted but this client never received.
"""

import asyncio
import struct
import threading

from neoruntime_ipc_sdk import CameraClient
from neoruntime_ipc_sdk.camera_types import StreamStatus
from neoruntime_ipc_sdk.encoded import (
    _ENC_FLAG_SEQ_PRESENT,
    _ENC_HEADER_FMT,
    _ENC_HEADER_SIZE,
    EncodedStreamClient,
)
from neoruntime_ipc_sdk.inference import InferenceClient

V2_FMT = _ENC_HEADER_FMT
V3_FMT = V2_FMT + "Q"


def _packet(seq=None, payload=b"PAY", keyframe=True):
    """Build one wire packet. seq=None -> V2 header (no seq extension)."""
    flags = (0x01 if keyframe else 0x00) | (
        _ENC_FLAG_SEQ_PRESENT if seq is not None else 0x00
    )
    header_size = _ENC_HEADER_SIZE + (8 if seq is not None else 0)
    total = header_size + len(payload)
    fields = (total, 0, flags, 111, 640, 384, 222)
    if seq is not None:
        return struct.pack(V3_FMT, *fields, seq) + payload
    return struct.pack(V2_FMT, *fields) + payload


class _FakeSock:
    """socket.socket stand-in: recv() drains a fixed byte buffer."""

    def __init__(self, data: bytes):
        self._buf = data
        self._pos = 0

    def recv(self, n):
        if self._pos >= len(self._buf):
            return b""
        chunk = self._buf[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


def _feed(client, packets):
    """Run _recv_frame over concatenated packets; return parsed frames."""
    frames = []
    for p in packets:
        frame = client._recv_frame(_FakeSock(p))
        assert frame is not None
        frames.append(frame)
    return frames


class TestEncodedSeqParsing:
    def test_v2_header_yields_seq_none(self):
        client = EncodedStreamClient(stream_id="main")
        (frame,) = _feed(client, [_packet(payload=b"ABC")])
        assert frame.seq is None
        assert frame.data == b"ABC"
        assert frame.is_keyframe
        # V2 frames never touch the seq counters.
        assert client.seq_packets == 0
        assert client.last_seq is None

    def test_v3_header_parses_seq_and_payload_over_38_bytes(self):
        client = EncodedStreamClient(stream_id="main")
        (frame,) = _feed(client, [_packet(seq=7, payload=b"XYZ123")])
        assert frame.seq == 7
        assert frame.data == b"XYZ123"
        # total_size covers the 38-byte header; payload must not eat the
        # seq bytes nor leave one behind.
        assert client.seq_packets == 1
        assert client.last_seq == 7
        assert client.seq_gap_events == 0

    def test_gap_detected_and_counted(self):
        client = EncodedStreamClient(stream_id="main")
        frames = _feed(client, [_packet(seq=1), _packet(seq=2), _packet(seq=6)])
        assert [f.seq for f in frames] == [1, 2, 6]
        # One discontinuity (3,4,5 missing) of size 3.
        assert client.seq_gap_events == 1
        assert client.seq_missing == 3
        assert client.seq_packets == 3

    def test_regression_is_restart_not_loss(self):
        # A seq that goes backwards means the publisher restarted and
        # rebased to 1 — no gap may be counted.
        client = EncodedStreamClient(stream_id="main")
        _feed(client, [_packet(seq=10), _packet(seq=1), _packet(seq=2)])
        assert client.seq_gap_events == 0
        assert client.seq_missing == 0
        assert client.last_seq == 2

    def test_counters_survive_reconnect_by_design(self):
        # The acceptance scenario: subscriber killed mid-stream, then a
        # new subscribe on the SAME client object. The hole spanned by
        # the reconnect must stay visible.
        client = EncodedStreamClient(stream_id="main")
        _feed(client, [_packet(seq=1), _packet(seq=2)])
        # "reconnect": base class would swap the socket; counters are
        # instance state and nothing resets them.
        _feed(client, [_packet(seq=9)])
        assert client.seq_gap_events == 1
        assert client.seq_missing == 6  # seq 3..8


class TestGetStatsWindow:
    def _client(self):
        client = InferenceClient()
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        client._loop = loop
        return client, thread

    def _stop(self, client, thread):
        client._loop.call_soon_threadsafe(client._loop.stop)
        thread.join(timeout=2)
        client._loop = None

    def test_default_sends_empty(self):
        # None = server-default window (500 ms); must serialize as the
        # legacy Empty request, not GetStatsRequest.
        from neoruntime_ipc_sdk.proto import inference_pb2

        client, thread = self._client()
        seen = []

        class _Stub:
            async def GetStats(self, request):
                seen.append(request)
                return inference_pb2.SystemStats()

        client.stub = _Stub()
        client.get_stats()
        self._stop(client, thread)
        assert isinstance(seen[0], inference_pb2.Empty)

    def test_window_sent_and_clamped(self):
        from neoruntime_ipc_sdk.proto import inference_pb2

        client, thread = self._client()
        seen = []

        class _Stub:
            async def GetStats(self, request):
                seen.append(request)
                return inference_pb2.SystemStats()

        client.stub = _Stub()
        client.get_stats(sampling_window_ms=50)
        client.get_stats(sampling_window_ms=99_999)  # clamp -> 5000
        client.get_stats(sampling_window_ms=0)  # clamp -> 1
        self._stop(client, thread)
        assert [r.sampling_window_ms for r in seen] == [50, 5000, 1]


class TestStreamStatusDropCounters:
    def test_mapping_carries_publisher_and_overlay_counters(self):
        from neoruntime_ipc_sdk.proto import camera_pb2

        resp = camera_pb2.GetStreamStatusResponse()
        s = resp.streams.add()
        s.stream_id = "third"
        s.status = "active"
        s.has_encoder = True
        s.codec = "h264"
        s.width = 640
        s.height = 384
        s.fps = 15
        s.bitrate_bps = 2_000_000
        s.gop = 30
        # Publisher side
        s.packets_published = 1000
        s.queue_overflow_drops = 4
        s.client_send_drops = 7
        s.client_send_failures = 1
        s.client_disconnects = 3
        s.last_packet_seq = 1000
        s.publisher_clients = 2
        # Overlay bake side
        s.bake_skips = 120
        s.strict_locked = 200
        s.strict_degraded = 3
        s.strict_skips = 12

        class _Stub:
            def GetStreamStatus(self, request, timeout=None):
                return resp

        client = CameraClient()
        client._stub = _Stub()
        (st,) = client.get_stream_status()

        assert isinstance(st, StreamStatus)
        assert st.packets_published == 1000
        assert st.queue_overflow_drops == 4
        assert st.client_send_drops == 7
        assert st.client_send_failures == 1
        assert st.client_disconnects == 3
        assert st.last_packet_seq == 1000
        assert st.publisher_clients == 2
        assert st.bake_skips == 120
        assert st.strict_locked == 200
        assert st.strict_degraded == 3
        assert st.strict_skips == 12
        # Invariant the acceptance script reconciles against.
        assert st.packets_published == st.last_packet_seq

    def test_defaults_zero_when_server_predates_fields(self):
        from neoruntime_ipc_sdk.proto import camera_pb2

        resp = camera_pb2.GetStreamStatusResponse()
        s = resp.streams.add()
        s.stream_id = "main"
        s.status = "active"
        s.has_encoder = True

        class _Stub:
            def GetStreamStatus(self, request, timeout=None):
                return resp

        client = CameraClient()
        client._stub = _Stub()
        (st,) = client.get_stream_status()
        assert st.packets_published == 0
        assert st.bake_skips == 0
        assert st.strict_skips == 0
        assert st.client_disconnects == 0


class TestStreamStatusOverlayDecoupling:
    """Wire fields 24-28: stream generation + overlay ingest/drop observability.

    stream_epoch is the restart handshake — an app holding an old epoch
    (from an earlier get_stream_status()) learns the stream restarted by
    comparing against the fresh one, instead of publishing overlays that
    the daemon silently rejects. overlay_late_commands / overlay_epoch_rejects
    count those rejections; overlay_no_binding_drops counts platform
    result events dropped because no binding admitted them (behavior
    decoupling: a bare subscribe() must not change the video).
    """

    def test_mapping_carries_epoch_and_overlay_counters(self):
        from neoruntime_ipc_sdk.proto import camera_pb2

        resp = camera_pb2.GetStreamStatusResponse()
        s = resp.streams.add()
        s.stream_id = "main"
        s.status = "active"
        s.has_encoder = True
        s.stream_epoch = 140_923_555_812_345
        s.overlay_layer_count = 2
        s.overlay_late_commands = 4
        s.overlay_epoch_rejects = 1
        s.overlay_no_binding_drops = 991

        class _Stub:
            def GetStreamStatus(self, request, timeout=None):
                return resp

        client = CameraClient()
        client._stub = _Stub()
        (st,) = client.get_stream_status()
        assert st.stream_epoch == 140_923_555_812_345
        assert st.overlay_layer_count == 2
        assert st.overlay_late_commands == 4
        assert st.overlay_epoch_rejects == 1
        assert st.overlay_no_binding_drops == 991

    def test_overlay_fields_default_zero(self):
        from neoruntime_ipc_sdk.proto import camera_pb2

        resp = camera_pb2.GetStreamStatusResponse()
        s = resp.streams.add()
        s.stream_id = "main"
        s.status = "active"
        s.has_encoder = True

        class _Stub:
            def GetStreamStatus(self, request, timeout=None):
                return resp

        client = CameraClient()
        client._stub = _Stub()
        (st,) = client.get_stream_status()
        assert st.stream_epoch == 0
        assert st.overlay_layer_count == 0
        assert st.overlay_late_commands == 0
        assert st.overlay_epoch_rejects == 0
        assert st.overlay_no_binding_drops == 0
