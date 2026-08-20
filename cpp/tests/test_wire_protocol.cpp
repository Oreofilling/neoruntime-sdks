// test_wire_protocol.cpp — wire-layout tests for the packed UDS protocol structs.
//
// The FdPublisher / EncodedPublisher / Audio protocols exchange packed,
// little-endian C structs over a UDS. Their byte layout MUST match the
// device-side C header (fd_protocol.h) exactly — any drift is silent on-host
// corruption. These tests pin:
//   1. struct sizes (the static_asserts in the header, re-checked at runtime),
//   2. field offsets (via offsetof — catches accidental reordering), and
//   3. end-to-end decode (hand-build a byte buffer at the documented offsets,
//      memcpy into the struct, and assert every field reads back correctly).
//
// This is the C++ analogue of media.py's struct.format("<…") round-trips and
// is the highest-value daemon-free check in the suite: it guards the ABI
// contract the whole media/audio path depends on.
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <gtest/gtest.h>

#include "detail/fd_protocol.hpp"

using namespace neoruntime_ipc_sdk::detail;

// ---- sizes -----------------------------------------------------------------
TEST(WireProtocol, StructSizesMatchDeviceHeader) {
    EXPECT_EQ(sizeof(FdPubMsgHeader), 8u);
    EXPECT_EQ(sizeof(FdPubSubscribeMsg), 76u);
    EXPECT_EQ(sizeof(FdPubFrameMsg), 80u);
    EXPECT_EQ(sizeof(FdPubReleaseMsg), 16u);
    EXPECT_EQ(sizeof(FdPubResponseMsg), 12u);
    EXPECT_EQ(sizeof(EncHeader), 30u);
    EXPECT_EQ(sizeof(AudioEncHeader), 30u);  // same size, different layout
}

// ---- field offsets: FdPubFrameMsg -----------------------------------------
TEST(WireProtocol, FdPubFrameMsgOffsets) {
    EXPECT_EQ(offsetof(FdPubFrameMsg, header), 0u);
    EXPECT_EQ(offsetof(FdPubFrameMsg, frame_id), 8u);
    EXPECT_EQ(offsetof(FdPubFrameMsg, timestamp_ns), 16u);
    EXPECT_EQ(offsetof(FdPubFrameMsg, sequence), 24u);
    EXPECT_EQ(offsetof(FdPubFrameMsg, width), 32u);
    EXPECT_EQ(offsetof(FdPubFrameMsg, height), 36u);
    EXPECT_EQ(offsetof(FdPubFrameMsg, format), 40u);
    EXPECT_EQ(offsetof(FdPubFrameMsg, num_planes), 44u);
    EXPECT_EQ(offsetof(FdPubFrameMsg, strides), 48u);
    EXPECT_EQ(offsetof(FdPubFrameMsg, sizes), 60u);
    EXPECT_EQ(offsetof(FdPubFrameMsg, num_fds), 72u);
}

TEST(WireProtocol, FdPubFrameMsgRoundTrip) {
    // Build the message byte-by-byte at the documented offsets, then read it
    // back through the struct. If offsets drift, fields mismatch.
    std::uint8_t buf[sizeof(FdPubFrameMsg)] = {};
    auto put32 = [&](std::size_t off, std::uint32_t v) {
        std::memcpy(buf + off, &v, sizeof(v));
    };
    auto put64 = [&](std::size_t off, std::uint64_t v) {
        std::memcpy(buf + off, &v, sizeof(v));
    };

    put32(0,  FD_PUB_MSG_FRAME);
    put32(4,  80);
    put64(8,  0xCAFEBABEull);            // frame_id
    put64(16, 1'700'000'000'000ull);     // timestamp_ns
    put64(24, 42);                       // sequence
    put32(32, 1920);                     // width
    put32(36, 1080);                     // height
    put32(40, 0);                        // format (NV12)
    put32(44, 2);                        // num_planes
    put32(48, 1920);                     // strides[0]
    put32(60, 1920 * 1080 * 3 / 2);      // sizes[0]
    put32(72, 2);                        // num_fds

    FdPubFrameMsg m;
    std::memcpy(&m, buf, sizeof(m));
    EXPECT_EQ(m.header.type, FD_PUB_MSG_FRAME);
    EXPECT_EQ(m.header.size, 80u);
    EXPECT_EQ(m.frame_id, 0xCAFEBABEull);
    EXPECT_EQ(m.sequence, 42ull);
    EXPECT_EQ(m.width, 1920u);
    EXPECT_EQ(m.height, 1080u);
    EXPECT_EQ(m.format, 0u);
    EXPECT_EQ(m.num_planes, 2u);
    EXPECT_EQ(m.strides[0], 1920u);
    EXPECT_EQ(m.sizes[0], 1920u * 1080u * 3 / 2);
    EXPECT_EQ(m.num_fds, 2u);
}

// ---- EncHeader (video) -----------------------------------------------------
TEST(WireProtocol, EncHeaderOffsets) {
    EXPECT_EQ(offsetof(EncHeader, total_size), 0u);
    EXPECT_EQ(offsetof(EncHeader, codec), 4u);
    EXPECT_EQ(offsetof(EncHeader, flags), 5u);
    EXPECT_EQ(offsetof(EncHeader, pts_ns), 6u);
    EXPECT_EQ(offsetof(EncHeader, width), 14u);
    EXPECT_EQ(offsetof(EncHeader, height), 18u);
    EXPECT_EQ(offsetof(EncHeader, dts_ns), 22u);
}

TEST(WireProtocol, EncHeaderRoundTrip) {
    std::uint8_t buf[30] = {};
    std::uint32_t total = 30 + 1000;
    std::uint64_t pts = 1'234'567'890ull;
    std::uint64_t dts = 1'234'500'000ull;
    std::memcpy(buf + 0, &total, 4);
    buf[4] = 1;                          // codec h265
    buf[5] = 0x01;                       // keyframe flag
    std::memcpy(buf + 6, &pts, 8);
    std::uint32_t w = 1280, h = 720;
    std::memcpy(buf + 14, &w, 4);
    std::memcpy(buf + 18, &h, 4);
    std::memcpy(buf + 22, &dts, 8);

    EncHeader hdr;
    std::memcpy(&hdr, buf, sizeof(hdr));
    EXPECT_EQ(hdr.total_size, 30u + 1000u);
    EXPECT_EQ(hdr.codec, 1);
    EXPECT_EQ(hdr.flags, 0x01);
    EXPECT_EQ(hdr.pts_ns, pts);
    EXPECT_EQ(hdr.width, 1280u);
    EXPECT_EQ(hdr.height, 720u);
    EXPECT_EQ(hdr.dts_ns, dts);
}

// ---- AudioEncHeader (audio — different layout, same 30 B) -----------------
TEST(WireProtocol, AudioEncHeaderOffsets) {
    EXPECT_EQ(offsetof(AudioEncHeader, total_size), 0u);
    EXPECT_EQ(offsetof(AudioEncHeader, codec), 4u);
    EXPECT_EQ(offsetof(AudioEncHeader, flags), 5u);
    EXPECT_EQ(offsetof(AudioEncHeader, pts_ns), 6u);
    EXPECT_EQ(offsetof(AudioEncHeader, sample_rate), 14u);
    EXPECT_EQ(offsetof(AudioEncHeader, channels), 18u);
    EXPECT_EQ(offsetof(AudioEncHeader, bits_per_sample), 22u);
    EXPECT_EQ(offsetof(AudioEncHeader, frame_size), 26u);
}

TEST(WireProtocol, AudioEncHeaderRoundTrip) {
    std::uint8_t buf[30] = {};
    std::uint32_t total = 30 + 9600;
    std::uint64_t pts = 500'000ull;
    std::memcpy(buf + 0, &total, 4);
    buf[4] = 0;                          // pcm
    buf[5] = 0x01;                       // keyframe
    std::memcpy(buf + 6, &pts, 8);
    std::uint32_t sr = 48000, ch = 1, bps = 16, fsz = 9600;
    std::memcpy(buf + 14, &sr, 4);
    std::memcpy(buf + 18, &ch, 4);
    std::memcpy(buf + 22, &bps, 4);
    std::memcpy(buf + 26, &fsz, 4);

    AudioEncHeader hdr;
    std::memcpy(&hdr, buf, sizeof(hdr));
    EXPECT_EQ(hdr.total_size, 30u + 9600u);
    EXPECT_EQ(hdr.codec, 0);
    EXPECT_EQ(hdr.flags, 0x01);
    EXPECT_EQ(hdr.pts_ns, pts);
    EXPECT_EQ(hdr.sample_rate, 48000u);
    EXPECT_EQ(hdr.channels, 1u);
    EXPECT_EQ(hdr.bits_per_sample, 16u);
    EXPECT_EQ(hdr.frame_size, 9600u);
}

// ---- decode_audio_format: dual-layout auto-detection ------------------------
// The platform daemon currently writes the VIDEO EncHeader layout on
// audio_capture.sock (width=0, height=0, dts[22:30]==pts) and never fills the
// audio fields. decode_audio_format() must therefore: (1) pass through genuine
// audio tails, and (2) instead of emitting garbage sample_rate/channels/bits,
// report unknown (0) and surface the video-tail 64-bit dts as dts_ns.
TEST(WireProtocol, AudioDecodeFormatGoldenDeviceVideoLayout) {
    // Raw 30-byte head of frame 0 captured on a live NE503 (192.168.93.72,
    // 2026-08-19): total=2078, codec=pcm, pts=0x00004b9214fa5ce6=83090789260518,
    // width=0, height=0, dts==pts. Audio-view values would be rate=0, ch=0,
    // bits=low32(pts)=0x14fa5ce6, fsize=high32(pts)=0x4b92 != payload 2048.
    std::uint8_t buf[30] = {
        0x1e, 0x08, 0x00, 0x00,                          // total_size = 2078
        0x00, 0x00,                                      // codec=pcm, flags=0
        0xe6, 0x5c, 0xfa, 0x14, 0x92, 0x4b, 0x00, 0x00,  // pts_ns
        0x00, 0x00, 0x00, 0x00,                          // width  (video view)
        0x00, 0x00, 0x00, 0x00,                          // height (video view)
        0xe6, 0x5c, 0xfa, 0x14, 0x92, 0x4b, 0x00, 0x00,  // dts_ns (video view)
    };
    AudioEncHeader hdr;
    std::memcpy(&hdr, buf, sizeof(hdr));

    const AudioFormatFields fmt = decode_audio_format(hdr, 2078 - 30);
    EXPECT_EQ(fmt.sample_rate, 0u);        // unknown, not garbage
    EXPECT_EQ(fmt.channels, 0u);
    EXPECT_EQ(fmt.bits_per_sample, 0u);
    EXPECT_EQ(fmt.dts_ns, 0x00004b9214fa5ce6ull);  // == pts
}

TEST(WireProtocol, AudioDecodeFormatPlausibleAudioLayout) {
    std::uint8_t buf[30] = {};
    std::uint32_t total = 30 + 9600;
    std::uint64_t pts = 500'000ull;
    std::memcpy(buf + 0, &total, 4);
    buf[4] = 0;                            // pcm
    std::memcpy(buf + 6, &pts, 8);
    std::uint32_t sr = 48000, ch = 1, bps = 16, fsz = 9600;
    std::memcpy(buf + 14, &sr, 4);
    std::memcpy(buf + 18, &ch, 4);
    std::memcpy(buf + 22, &bps, 4);
    std::memcpy(buf + 26, &fsz, 4);

    AudioEncHeader hdr;
    std::memcpy(&hdr, buf, sizeof(hdr));
    const AudioFormatFields fmt = decode_audio_format(hdr, 9600);
    EXPECT_EQ(fmt.sample_rate, 48000u);
    EXPECT_EQ(fmt.channels, 1u);
    EXPECT_EQ(fmt.bits_per_sample, 16u);
    EXPECT_EQ(fmt.dts_ns, 0u);
}

// A frame carries total_size in [0:4]; when the wire frame_size disagrees with
// the actual payload the tail is not a genuine audio header -> fallback.
TEST(WireProtocol, AudioDecodeFormatFrameSizeMismatchFallsBack) {
    AudioEncHeader hdr{};
    hdr.sample_rate = 48000;               // each field is individually plausible
    hdr.channels = 1;
    hdr.bits_per_sample = 0x14fa5ce6;      // video dts low32
    hdr.frame_size = 0x4b92;               // video dts high32 (19346)
    const AudioFormatFields fmt = decode_audio_format(hdr, 2048);  // payload 2048
    EXPECT_EQ(fmt.sample_rate, 0u);
    EXPECT_EQ(fmt.channels, 0u);
    EXPECT_EQ(fmt.bits_per_sample, 0u);
    EXPECT_EQ(fmt.dts_ns, 0x00004b9214fa5ce6ull);
}

// Boundary guards: non-multiple-of-8 bits, out-of-range rate, and a negative
// payload each fail the plausibility gate and fall back.
TEST(WireProtocol, AudioDecodeFormatInvalidTailsFallBack) {
    constexpr std::uint64_t kDts = 0x00004b9214fa5ce6ull;
    // 12-bit samples are invalid on the wire (% 8 != 0).
    {
        AudioEncHeader hdr{};
        hdr.sample_rate = 44100;
        hdr.channels = 2;
        hdr.bits_per_sample = 12;
        hdr.frame_size = 0x4b92;
        const AudioFormatFields fmt = decode_audio_format(hdr, 0x4b92);
        EXPECT_EQ(fmt.sample_rate, 0u);
        EXPECT_EQ(fmt.dts_ns, 12ull | (0x4b92ull << 32));
    }
    // Video widths are arbitrary (< 8 kHz here) -> never a plausible rate.
    {
        AudioEncHeader hdr{};
        hdr.sample_rate = 100;             // below 8 kHz
        hdr.channels = 4;
        hdr.bits_per_sample = 0x14fa5ce6;
        hdr.frame_size = 0x4b92;
        const AudioFormatFields fmt = decode_audio_format(hdr, 0x4b92);
        EXPECT_EQ(fmt.sample_rate, 0u);
        EXPECT_EQ(fmt.channels, 0u);
        EXPECT_EQ(fmt.bits_per_sample, 0u);
        EXPECT_EQ(fmt.dts_ns, kDts);
    }
    // Negative payload_size (bogus total_size) never passes the gate.
    {
        AudioEncHeader hdr{};
        hdr.sample_rate = 48000;
        hdr.channels = 1;
        hdr.bits_per_sample = 0x14fa5ce6;
        hdr.frame_size = 0x4b92;
        const AudioFormatFields fmt = decode_audio_format(hdr, -1);
        EXPECT_EQ(fmt.sample_rate, 0u);
        EXPECT_EQ(fmt.dts_ns, kDts);
    }
}

// ---- protocol constants ----------------------------------------------------
TEST(WireProtocol, MessageTypeConstants) {
    EXPECT_EQ(FD_PUB_MSG_SUBSCRIBE, 1u);
    EXPECT_EQ(FD_PUB_MSG_UNSUBSCRIBE, 2u);
    EXPECT_EQ(FD_PUB_MSG_FRAME, 3u);
    EXPECT_EQ(FD_PUB_MSG_RELEASE, 4u);
    EXPECT_EQ(FD_PUB_MSG_OK, 5u);
    EXPECT_EQ(FD_PUB_MSG_ERROR, 6u);
    EXPECT_EQ(FD_PUB_PROTOCOL_VERSION, 1u);
    EXPECT_EQ(FD_PUB_MAX_STREAM_NAME, 64);
    EXPECT_EQ(FD_PUB_MAX_FDS, 3);
}
