// fd_protocol.hpp — packed wire structs for the FdPublisher / EncodedPublisher
// UDS protocols. MUST match the device-side C header fd_protocol.h byte-for-byte.
//
// INTERNAL (PRIVATE include path). These are little-endian host-local structures;
// the client and the platform daemon run on the same aarch64 host (both LE), so a
// plain memcpy is correct — there is no cross-endian exchange over a UDS.
//
// Faithful to python/hailo_ipc_sdk/media.py struct format strings:
//   _HDR_FMT   "<II"                 -> FdPubMsgHeader       (8 B)
//   _SUB_FMT   "<II I 64s"           -> FdPubSubscribeMsg    (76 B)
//   _FRAME_FMT "<II QQQ IIII 3I 3I I 4x" -> FdPubFrameMsg    (80 B, incl. 4 pad)
//   _REL_FMT   "<II Q"               -> FdPubReleaseMsg      (16 B)
//   _RESP_FMT  "<II i"               -> FdPubResponseMsg     (12 B)
//   _ENC_HEADER_FMT "<I BB Q II Q"   -> EncHeader            (30 B)
#pragma once

#include <cstdint>

namespace hailo_ipc_sdk::detail {

// ---- FdPublisher protocol message types (must match fd_protocol.h) ----------
inline constexpr std::uint32_t FD_PUB_MSG_SUBSCRIBE   = 1;
inline constexpr std::uint32_t FD_PUB_MSG_UNSUBSCRIBE = 2;
inline constexpr std::uint32_t FD_PUB_MSG_FRAME       = 3;
inline constexpr std::uint32_t FD_PUB_MSG_RELEASE     = 4;
inline constexpr std::uint32_t FD_PUB_MSG_OK          = 5;
inline constexpr std::uint32_t FD_PUB_MSG_ERROR       = 6;

inline constexpr int FD_PUB_MAX_STREAM_NAME  = 64;
inline constexpr int FD_PUB_MAX_FDS          = 3;
inline constexpr std::uint32_t FD_PUB_PROTOCOL_VERSION = 1;

#pragma pack(push, 1)

// Common header: { uint32 type; uint32 size; }
struct FdPubMsgHeader {
    std::uint32_t type;
    std::uint32_t size;
};
static_assert(sizeof(FdPubMsgHeader) == 8, "wire layout: FdPubMsgHeader");

// struct FdPubSubscribeMsg { header(8) + uint32 version + char[64] stream_name }
struct FdPubSubscribeMsg {
    FdPubMsgHeader header;
    std::uint32_t  version;
    char stream_name[64];
};
static_assert(sizeof(FdPubSubscribeMsg) == 76, "wire layout: FdPubSubscribeMsg");

// struct FdPubFrameMsg — aarch64 pads to 8-byte alignment; the trailing 4 bytes
// are explicit padding (Python's "4x"), giving an 80-byte message.
struct FdPubFrameMsg {
    FdPubMsgHeader header;       // 8
    std::uint64_t  frame_id;     // 8
    std::uint64_t  timestamp_ns; // 8
    std::uint64_t  sequence;     // 8
    std::uint32_t  width;        // 4
    std::uint32_t  height;       // 4
    std::uint32_t  format;       // 4  (PixelFormat code)
    std::uint32_t  num_planes;   // 4
    std::uint32_t  strides[3];   // 12
    std::uint32_t  sizes[3];     // 12
    std::uint32_t  num_fds;      // 4
    std::uint8_t   pad[4];       // 4  (explicit, matches Python "4x")
};
static_assert(sizeof(FdPubFrameMsg) == 80, "wire layout: FdPubFrameMsg");

// struct FdPubReleaseMsg { header(8) + uint64 frame_id }
struct FdPubReleaseMsg {
    FdPubMsgHeader header;
    std::uint64_t  frame_id;
};
static_assert(sizeof(FdPubReleaseMsg) == 16, "wire layout: FdPubReleaseMsg");

// struct FdPubResponseMsg { header(8) + int32 code }
struct FdPubResponseMsg {
    FdPubMsgHeader header;
    std::int32_t   code;
};
static_assert(sizeof(FdPubResponseMsg) == 12, "wire layout: FdPubResponseMsg");

// EncodedPublisher header — 30 bytes, little-endian.
struct EncHeader {
    std::uint32_t total_size;  // [0:4]   header + payload
    std::uint8_t  codec;       // [4]     0=h264, 1=h265
    std::uint8_t  flags;       // [5]     bit0 = keyframe
    std::uint64_t pts_ns;      // [6:14]
    std::uint32_t width;       // [14:18]
    std::uint32_t height;      // [18:22]
    std::uint64_t dts_ns;      // [22:30]
};
static_assert(sizeof(EncHeader) == 30, "wire layout: EncHeader");

// Audio EncodedPublisher header — 30 bytes, little-endian. Same 30-byte size as
// the video EncHeader but a DIFFERENT field layout (audio has sample_rate /
// channels / bits_per_sample / frame_size). Faithful to audio_stream.py's
// _HEADER_FMT "<I BB Q III I".
struct AudioEncHeader {
    std::uint32_t total_size;       // [0:4]   header + payload
    std::uint8_t  codec;            // [4]     0=pcm, 1=aac, 2=g711a, 3=g711u
    std::uint8_t  flags;            // [5]     bit0 = keyframe
    std::uint64_t pts_ns;           // [6:14]
    std::uint32_t sample_rate;      // [14:18]
    std::uint32_t channels;         // [18:22]
    std::uint32_t bits_per_sample;  // [22:26]
    std::uint32_t frame_size;       // [26:30]
};
static_assert(sizeof(AudioEncHeader) == 30, "wire layout: AudioEncHeader");

#pragma pack(pop)

}  // namespace hailo_ipc_sdk::detail
