"""
Tests for recording utilities: TsWriter / HlsWriter / PrerollBuffer
(pure-python MPEG-TS muxing of EncodedFrame Annex-B payloads).
"""

import glob
import os

import pytest

from neoruntime_ipc_sdk.media import EncodedFrame
from neoruntime_ipc_sdk.recording import HlsWriter, PrerollBuffer, TsWriter

# ---------- synthetic encoded frames ----------

def enc(seq, pts_ns, key=False, codec=0, size=400):
    payload = b"\x00\x00\x01\x09\xf0" + (f"FRAME{seq:04d}".encode() * 40)
    return EncodedFrame(codec=codec, flags=1 if key else 0, pts_ns=pts_ns,
                        dts_ns=pts_ns, width=64, height=48,
                        data=payload[:size].ljust(size, b"\x7f"))


FRAME_MS = 100_000_000          # 10 fps
KEY_EVERY = 10                  # keyframe each second


def stream(n, fps_ns=FRAME_MS, key_every=KEY_EVERY, codec=0, start_pts=0):
    return [enc(i, start_pts + i * fps_ns, key=(i % key_every == 0), codec=codec)
            for i in range(n)]


# ---------- TS packet parsing helpers ----------

def packets(data):
    assert len(data) % 188 == 0
    return [data[i:i + 188] for i in range(0, len(data), 188)]


def pid_of(p):
    return ((p[1] & 0x1F) << 8) | p[2]


def unit_start(p):
    return bool(p[1] & 0x40)


def has_payload(p):
    return bool(p[3] & 0x10)


def cc_of(p):
    return p[3] & 0x0F


def psi_payload(p):
    off = 4
    if (p[3] >> 4) & 0x3 in (2, 3):
        off += 1 + p[4]
    off += 1 + p[off]           # pointer_field
    return p[off:]


def pes_payload(p):
    off = 4
    if (p[3] >> 4) & 0x3 in (2, 3):
        off += 1 + p[4]
    return p[off:]


def decode_ts_marker(b):
    """Decode a 5-byte PTS/DTS field into a 33-bit value."""
    return (((b[0] >> 1) & 0x07) << 30) | (b[1] << 22) | \
           ((b[2] >> 1) << 15) | (b[3] << 7) | (b[4] >> 1)


def reassemble_pes(data, pid=0x100):
    """Concatenate PES payloads for each unit-start group on pid."""
    groups, cur = [], b""
    for p in packets(data):
        if pid_of(p) != pid or not has_payload(p):
            continue
        if unit_start(p) and cur:
            groups.append(cur)
            cur = b""
        cur += pes_payload(p)
    if cur:
        groups.append(cur)
    return groups


# ---------- TsWriter ----------

class TestTsWriter:
    def _write(self, tmp_path, frames, codec="h264"):
        path = str(tmp_path / "out.ts")
        w = TsWriter(path, codec=codec)
        for f in frames:
            w.write(f)
        w.close()
        return path

    def test_packet_structure(self, tmp_path):
        path = self._write(tmp_path, stream(5))
        data = open(path, "rb").read()
        assert len(data) % 188 == 0
        for p in packets(data):
            assert p[0] == 0x47

    def test_pat_pmt_present(self, tmp_path):
        path = self._write(tmp_path, stream(5))
        pkts = packets(open(path, "rb").read())
        assert pid_of(pkts[0]) == 0x0000          # PAT first
        assert pid_of(pkts[1]) == 0x1000          # PMT second
        # PAT declares PMT pid 0x1000 (table: id,len,tsid,ver,sec,last,prog,pid)
        pat = psi_payload(pkts[0])
        assert pat[0] == 0x00                      # table_id
        assert ((pat[10] & 0x1F) << 8) | pat[11] == 0x1000
        # PMT declares video pid 0x100, stream_type 0x1B (H.264)
        pmt = psi_payload(pkts[1])
        assert pmt[0] == 0x02                      # table_id
        pcr_pid = ((pmt[8] & 0x1F) << 8) | pmt[9]
        assert pcr_pid == 0x0100
        assert pmt[12] == 0x1B                     # H.264 stream_type
        assert ((pmt[13] & 0x1F) << 8) | pmt[14] == 0x0100

    def test_h265_stream_type(self, tmp_path):
        path = self._write(tmp_path, stream(3, codec=1), codec="h265")
        pmt = psi_payload(packets(open(path, "rb").read())[1])
        assert pmt[12] == 0x24                     # HEVC stream_type

    def test_continuity_counters(self, tmp_path):
        path = self._write(tmp_path, stream(20))   # multi-packet PES
        per_pid = {}
        for p in packets(open(path, "rb").read()):
            if not has_payload(p):
                continue
            per_pid.setdefault(pid_of(p), []).append(cc_of(p))
        for _pid, ccs in per_pid.items():
            for a, b in zip(ccs, ccs[1:]):
                assert b == (a + 1) % 16

    def test_pes_carries_frame_payload(self, tmp_path):
        frames = stream(3)
        path = self._write(tmp_path, frames)
        data = open(path, "rb").read()
        groups = reassemble_pes(data)
        assert len(groups) == 3
        for g, f in zip(groups, frames):
            assert g[:4] == b"\x00\x00\x01\xe0"    # PES start + video stream_id
            assert g.endswith(f.data[-32:])         # payload intact

    def test_pts_conversion_90khz(self, tmp_path):
        path = self._write(tmp_path, [enc(0, 1_000_000_000)])  # 1s
        g = reassemble_pes(open(path, "rb").read())[0]
        # 00 00 01 E0 len len flags hdrdatalen PTS(5)
        assert g[7] & 0x80                          # PTS flag
        pts = decode_ts_marker(g[9:14])
        assert pts == 90_000

    def test_pts_wraparound(self, tmp_path):
        ticks = (1 << 33) + 12349                   # one full wrap + 12349
        # offset chosen so ticks*1e9/90000 is an exact integer
        pts_ns = ticks * 1_000_000_000 // 90_000
        path = self._write(tmp_path, [enc(0, pts_ns)])
        g = reassemble_pes(open(path, "rb").read())[0]
        assert decode_ts_marker(g[9:14]) == 12349

    def test_dts_written_when_differs(self, tmp_path):
        f = enc(0, 2_000_000_000)
        f.dts_ns = 1_000_000_000
        path = self._write(tmp_path, [f])
        g = reassemble_pes(open(path, "rb").read())[0]
        assert g[7] & 0x40                          # DTS flag
        assert g[8] == 10                           # header data length
        assert decode_ts_marker(g[9:14]) == 180_000
        assert decode_ts_marker(g[14:19]) == 90_000

    def test_first_pes_has_pcr(self, tmp_path):
        path = self._write(tmp_path, stream(3))
        for p in packets(open(path, "rb").read()):
            if pid_of(p) == 0x100 and unit_start(p):
                assert (p[3] >> 4) & 0x3 == 3       # adaptation+payload
                assert p[4] >= 7                    # AF with PCR
                assert p[5] & 0x10                  # PCR flag
                break

    def test_device_regression_tail_chunk_of_183_bytes(self, tmp_path):
        # PES total = 14B header + 353B data = 367 = 184 + 183. The 183B
        # tail must be packed, not rejected: it leaves exactly one byte
        # for a length-only adaptation field. Live 4K frames on
        # a test device hit this ("TS payload overflow with adaptation
        # field") whenever pes_len % 184 == 183.
        first = enc(0, 0, key=True, size=100)
        tail_frame = enc(1, FRAME_MS, size=353)
        path = self._write(tmp_path, [first, tail_frame])
        data = open(path, "rb").read()
        assert len(data) % 188 == 0
        for p in packets(data):
            assert p[0] == 0x47
            assert has_payload(p)
        groups = reassemble_pes(data)
        assert groups[0].endswith(first.data)
        assert groups[1].endswith(tail_frame.data)

    def test_device_regression_short_pes_with_pcr(self, tmp_path):
        # First frame carries the PCR. A PES of 183B total (14B header +
        # 169B data) must be split so the PCR adaptation field fits:
        # cap the first chunk at 176B instead of overflowing.
        frame = enc(0, 0, key=True, size=169)
        path = self._write(tmp_path, [frame])
        data = open(path, "rb").read()
        assert len(data) % 188 == 0
        for p in packets(data):
            assert p[0] == 0x47
        groups = reassemble_pes(data)
        assert len(groups) == 1
        assert groups[0].endswith(frame.data)

    def test_write_after_close_raises(self, tmp_path):
        w = TsWriter(str(tmp_path / "x.ts"), codec="h264")
        w.close()
        with pytest.raises(RuntimeError):
            w.write(enc(0, 0))


# ---------- HlsWriter ----------

class TestHlsWriter:
    def test_segments_and_playlist(self, tmp_path):
        out = str(tmp_path / "hls")
        w = HlsWriter(out, segment_seconds=2.0)
        for f in stream(45):                        # 4.5s of frames
            w.write(f)
        w.close()
        segs = sorted(glob.glob(os.path.join(out, "seg*.ts")))
        assert len(segs) >= 2
        m3u8 = open(os.path.join(out, "index.m3u8")).read()
        assert m3u8.startswith("#EXTM3U")
        assert "#EXT-X-VERSION:3" in m3u8
        assert "#EXT-X-TARGETDURATION:" in m3u8
        assert m3u8.count("#EXTINF:") == len(segs)
        assert "#EXT-X-ENDLIST" in m3u8

    def test_no_endlist_until_close(self, tmp_path):
        out = str(tmp_path / "hls")
        w = HlsWriter(out, segment_seconds=2.0)
        for f in stream(45):
            w.write(f)
        m3u8 = open(os.path.join(out, "index.m3u8")).read()
        assert "#EXT-X-ENDLIST" not in m3u8
        w.close()

    def test_segments_self_contained(self, tmp_path):
        out = str(tmp_path / "hls")
        w = HlsWriter(out, segment_seconds=2.0)
        for f in stream(45):
            w.write(f)
        w.close()
        segs = sorted(glob.glob(os.path.join(out, "seg*.ts")))
        for s in segs:
            pkts = packets(open(s, "rb").read())
            assert pid_of(pkts[0]) == 0x0000        # PAT leads every segment
            assert pid_of(pkts[1]) == 0x1000        # PMT follows

    def test_segments_start_on_keyframe(self, tmp_path):
        out = str(tmp_path / "hls")
        w = HlsWriter(out, segment_seconds=2.0)
        for f in stream(45):
            w.write(f)
        w.close()
        segs = sorted(glob.glob(os.path.join(out, "seg*.ts")))
        for s in segs:
            data = open(s, "rb").read()
            g = reassemble_pes(data)[0]             # first PES in segment
            idx = int(g[-400:].split(b"FRAME")[1][:4])
            assert idx % KEY_EVERY == 0             # segment starts at keyframe

    def test_window_trims_old_segments(self, tmp_path):
        out = str(tmp_path / "hls")
        w = HlsWriter(out, segment_seconds=1.0, window=2)
        for f in stream(100):                       # ~10 segments
            w.write(f)
        w.close()
        segs = glob.glob(os.path.join(out, "seg*.ts"))
        assert len(segs) <= 2
        m3u8 = open(os.path.join(out, "index.m3u8")).read()
        assert m3u8.count("#EXTINF:") == len(segs)
        assert "#EXT-X-MEDIA-SEQUENCE:" in m3u8

    def test_discontinuity_on_pts_jump_back(self, tmp_path):
        out = str(tmp_path / "hls")
        w = HlsWriter(out, segment_seconds=1.0)
        for f in stream(30, start_pts=10_000_000_000):
            w.write(f)
        for f in stream(30, start_pts=0):           # clock jumps backwards
            w.write(f)
        w.close()
        m3u8 = open(os.path.join(out, "index.m3u8")).read()
        assert "#EXT-X-DISCONTINUITY" in m3u8


# ---------- PrerollBuffer ----------

class TestPrerollBuffer:
    def test_keeps_only_recent_frames(self, tmp_path):
        buf = PrerollBuffer(seconds=5.0)
        frames = stream(100)                        # 10s of frames
        for f in frames:
            buf.push(f)
        assert len(buf) <= 51                       # ~5s + newest
        oldest = buf.frames[0]
        assert frames[-1].pts_ns - oldest.pts_ns <= 5_000_000_000

    def test_dump_writes_playable_ts(self, tmp_path):
        buf = PrerollBuffer(seconds=2.0)
        frames = stream(50)
        for f in frames:
            buf.push(f)
        path = str(tmp_path / "event.ts")
        writer = buf.dump(path)
        writer.close()
        data = open(path, "rb").read()
        pkts = packets(data)
        assert pid_of(pkts[0]) == 0x0000            # PAT
        groups = reassemble_pes(data)
        assert groups, "preroll frames present"
        assert groups[-1].endswith(frames[-1].data[-32:])

    def test_dump_derives_codec(self, tmp_path):
        buf = PrerollBuffer(seconds=1.0)
        for f in stream(5, codec=1):                # h265 frames
            buf.push(f)
        w = buf.dump(str(tmp_path / "e.ts"))
        w.close()
        pmt = psi_payload(packets(open(str(tmp_path / "e.ts"), "rb").read())[1])
        assert pmt[12] == 0x24

    def test_dump_on_frame_callback(self, tmp_path):
        buf = PrerollBuffer(seconds=1.0)
        for f in stream(5):
            buf.push(f)
        seen = []

        def on_frame(writer):
            seen.append(1)
            writer.write(enc(999, 50 * FRAME_MS))   # live tail after preroll

        writer = buf.dump(str(tmp_path / "e.ts"), on_frame=on_frame)
        writer.close()
        assert seen == [1]
        groups = reassemble_pes(open(str(tmp_path / "e.ts"), "rb").read())
        assert len(groups) == 6                     # preroll + live tail
