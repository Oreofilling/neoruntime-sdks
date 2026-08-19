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

using namespace hailo_ipc_sdk::detail;

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
