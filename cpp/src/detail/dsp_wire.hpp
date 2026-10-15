// dsp_wire.hpp — DSP service wire protocol: UDS messages, codec, error
// codes (port of python dsp_wire.py, 0.8.1).
//
// Everything here mirrors the daemon side (camera-daemon dsp_service.cpp
// and include/fd_protocol.h) so client modules can encode/decode without
// knowing socket plumbing. Pure data + pure functions; no I/O.
//
// Byte layouts are the daemon's C structs, little-endian on the wire —
// offsets are written out explicitly (Python's struct format strings name
// the same offsets, incl. the 4-byte align pad before the u64 id array).
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "camera-daemon/camera.pb.h"

#include "neoruntime_ipc_sdk/dsp.hpp"

namespace neoruntime_ipc_sdk::detail::dsp_wire {

// ---- UDS wire constants (platform camera-daemon include/fd_protocol.h) ----
inline constexpr std::uint32_t kMsgOk = 5;            // control ack, no payload
inline constexpr std::uint32_t kMsgError = 6;         // control error ack
inline constexpr std::uint32_t kMsgDspAlloc = 7;
inline constexpr std::uint32_t kMsgDspAllocResp = 8;
inline constexpr std::uint32_t kMsgDspBufRelease = 9;
inline constexpr std::uint32_t kMsgDspImport = 10;
inline constexpr std::uint32_t kMsgDspImportResp = 11;

inline constexpr std::size_t kAllocReqSize = 24;    // hdr(8) + w,h,fmt,count
inline constexpr std::size_t kAllocRespSize = 560;  // hdr(8) + meta + pad + 64*u64
inline constexpr std::size_t kReleaseSize = 16;     // hdr(8) + buffer_id
inline constexpr std::size_t kImportReqSize = 48;   // hdr(8) + 10*u32
inline constexpr std::size_t kImportRespSize = 24;  // hdr(8) + code + pad + u64
inline constexpr unsigned kDspMaxFds = 64;          // FD_PUB_DSP_MAX_FDS

// HalPixelFormat wire values (hal_v2 hal_buffer.h) — deliberately separate
// from the SDK PixelFormat enum, whose numbering does not match the wire.
// argb: blend overlays only — no job op reads an argb *source* today.
inline constexpr std::uint32_t kHalFormatNv12 = 0;
inline constexpr std::uint32_t kHalFormatArgb = 6;
inline constexpr std::uint32_t kHalFormatRgb24 = 4;
inline constexpr std::uint32_t kHalFormatGray8 = 8;

inline bool is_dsp_format(const std::string& fmt) {
    return fmt == "nv12" || fmt == "rgb24" || fmt == "gray8" || fmt == "argb";
}

inline std::uint32_t hal_pixel_format(const std::string& fmt) {
    if (fmt == "nv12") return kHalFormatNv12;
    if (fmt == "rgb24") return kHalFormatRgb24;
    if (fmt == "gray8") return kHalFormatGray8;
    if (fmt == "argb") return kHalFormatArgb;
    throw DspError("unsupported format '" + fmt + "' (nv12/rgb24/gray8)");
}

// ---- daemon caps (dsp_service.cpp), mirrored for fail-fast validation ----
inline constexpr int kMinDim = 16;
inline constexpr int kMaxDim = 8192;
inline constexpr int kMaxBatch = 64;

// ---- DspService error codes ----
inline constexpr std::int32_t kSvcUnavailable = -5;
// wait_job-specific readings of the shared codes (dsp_service.cpp):
inline constexpr std::int32_t kErrNoBuffer = -2;  // unknown or already-reaped job id
inline constexpr std::int32_t kErrTimeout = -4;   // still pending after timeout_ms (re-wait works)

inline std::string error_text(std::int32_t code) {
    switch (code) {
        case -1: return "invalid request";
        case -2: return "no such buffer id";
        case -3: return "quota exceeded";
        case -4: return "job timeout";
        case -5: return "dsp service unavailable";
        case -6: return "out of memory";
        case -7: return "client limit exceeded";
        default: return "dsp error " + std::to_string(code);
    }
}

// ---- camera.proto DspOp values ----
inline constexpr std::int32_t kOpResize = 0;
inline constexpr std::int32_t kOpCropAndResize = 1;
inline constexpr std::int32_t kOpMultiCrop = 2;
inline constexpr std::int32_t kOpConvertFormat = 3;  // equal dims, differing formats
inline constexpr std::int32_t kOpBlend = 4;          // ARGB32 overlays onto NV12, in place

namespace pb = aipc::camera;

inline pb::DspInterpolation interp_wire(const std::string& s) {
    if (s == "nearest") return pb::DSP_INTERP_NEAREST;
    if (s == "bilinear") return pb::DSP_INTERP_BILINEAR;
    if (s == "area") return pb::DSP_INTERP_AREA;
    if (s == "bicubic") return pb::DSP_INTERP_BICUBIC;
    throw DspError("unsupported interpolation '" + s + "'");
}

inline pb::DspScalingMode scaling_wire(const std::string& s) {
    if (s == "stretch") return pb::DSP_SCALING_STRETCH;
    if (s == "letterbox" || s == "letterbox_middle")
        return pb::DSP_SCALING_LETTERBOX_MIDDLE;
    if (s == "letterbox_up_left") return pb::DSP_SCALING_LETTERBOX_UP_LEFT;
    if (s == "scale_crop") return pb::DSP_SCALING_SCALE_AND_CROP;
    throw DspError("unsupported scaling '" + s + "'");
}

inline pb::DspPriority priority_wire(const std::string& s) {
    if (s == "background") return pb::DSP_PRIORITY_BACKGROUND;
    if (s == "normal") return pb::DSP_PRIORITY_NORMAL;
    throw DspError("unsupported priority '" + s + "'");
}

// ---- little-endian load/store (the daemon structs are LE on the wire) ----
inline void put_u32(std::uint8_t* p, std::uint32_t v) {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
    p[2] = static_cast<std::uint8_t>(v >> 16);
    p[3] = static_cast<std::uint8_t>(v >> 24);
}
inline void put_u64(std::uint8_t* p, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
inline std::uint32_t get_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
inline std::int32_t get_i32(const std::uint8_t* p) {
    return static_cast<std::int32_t>(get_u32(p));
}
inline std::uint64_t get_u64(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

// ---- FD_PUB_MSG_DSP_ALLOC (24 bytes) ----------------------------------------
inline std::array<std::uint8_t, kAllocReqSize> alloc_request(
    std::uint32_t width, std::uint32_t height, std::uint32_t fmt_wire,
    std::uint32_t count) {
    std::array<std::uint8_t, kAllocReqSize> b{};
    put_u32(&b[0], kMsgDspAlloc);
    put_u32(&b[4], static_cast<std::uint32_t>(kAllocReqSize));
    put_u32(&b[8], width);
    put_u32(&b[12], height);
    put_u32(&b[16], fmt_wire);
    put_u32(&b[20], count);
    return b;
}

// FD_PUB_MSG_DSP_ALLOC_RESP (560 bytes; fds arrive separately via SCM_RIGHTS).
struct AllocResp {
    std::int32_t code = 0;
    std::uint32_t count = 0;
    std::uint32_t num_planes = 0;
    std::uint32_t strides[3] = {0, 0, 0};
    std::uint32_t sizes[3] = {0, 0, 0};
    std::vector<std::uint64_t> ids;  // `count` entries
};

inline AllocResp parse_alloc_resp(const std::uint8_t* data, std::size_t len) {
    if (len < kAllocRespSize)
        throw DspError("short DSP alloc response: " + std::to_string(len) + " bytes");
    const std::uint32_t mtype = get_u32(data);
    if (mtype != kMsgDspAllocResp)
        throw DspError("unexpected DSP alloc response type " + std::to_string(mtype));
    AllocResp r;
    r.code = get_i32(data + 8);
    r.count = get_u32(data + 12);
    r.num_planes = get_u32(data + 16);
    for (int i = 0; i < 3; ++i) r.strides[i] = get_u32(data + 20 + 4 * i);
    for (int i = 0; i < 3; ++i) r.sizes[i] = get_u32(data + 32 + 4 * i);
    // ids[64] starts at byte 48 (4-byte align pad at 44..48 before the u64s)
    for (std::uint32_t i = 0; i < r.count && i < 64; ++i)
        r.ids.push_back(get_u64(data + 48 + 8 * i));
    return r;
}

// ---- FD_PUB_MSG_DSP_IMPORT (48 bytes; fds travel via SCM_RIGHTS) ------------
inline std::array<std::uint8_t, kImportReqSize> import_request(
    std::uint32_t width, std::uint32_t height, std::uint32_t fmt_wire,
    std::uint32_t num_planes, const std::uint32_t strides[3],
    const std::uint32_t sizes[3]) {
    std::array<std::uint8_t, kImportReqSize> b{};
    put_u32(&b[0], kMsgDspImport);
    put_u32(&b[4], static_cast<std::uint32_t>(kImportReqSize));
    put_u32(&b[8], width);
    put_u32(&b[12], height);
    put_u32(&b[16], fmt_wire);
    put_u32(&b[20], num_planes);
    for (int i = 0; i < 3; ++i) put_u32(&b[24 + 4 * i], strides[i]);
    for (int i = 0; i < 3; ++i) put_u32(&b[36 + 4 * i], sizes[i]);
    return b;
}

// FD_PUB_MSG_DSP_IMPORT_RESP (24 bytes). code mirrors the daemon error
// codes (0 on success, -1 validation failure, -7 client import cap).
struct ImportResp {
    std::int32_t code = 0;
    std::uint64_t import_id = 0;  // daemon-side
};

inline ImportResp parse_import_resp(const std::uint8_t* data, std::size_t len) {
    if (len < kImportRespSize)
        throw DspError("short DSP import response: " + std::to_string(len) + " bytes");
    const std::uint32_t mtype = get_u32(data);
    if (mtype != kMsgDspImportResp)
        throw DspError("unexpected DSP import response type " + std::to_string(mtype));
    ImportResp r;
    r.code = get_i32(data + 8);
    r.import_id = get_u64(data + 16);
    return r;
}

// ---- FD_PUB_MSG_DSP_BUF_RELEASE (16 bytes) -----------------------------------
inline std::array<std::uint8_t, kReleaseSize> release_request(std::uint64_t buffer_id) {
    std::array<std::uint8_t, kReleaseSize> b{};
    put_u32(&b[0], kMsgDspBufRelease);
    put_u32(&b[4], static_cast<std::uint32_t>(kReleaseSize));
    put_u64(&b[8], buffer_id);
    return b;
}

// ---- tightly-packed plane geometry -------------------------------------------
struct PlaneRows {
    std::uint32_t row_bytes;
    std::uint32_t rows;
};

// (row_bytes, rows) per plane for a tightly-packed geometry.
inline std::vector<PlaneRows> plane_rows(const std::string& fmt, int width, int height) {
    if (fmt == "nv12")
        return {{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)},
                {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height / 2)}};
    if (fmt == "rgb24")
        return {{static_cast<std::uint32_t>(width * 3), static_cast<std::uint32_t>(height)}};
    if (fmt == "argb")
        return {{static_cast<std::uint32_t>(width * 4), static_cast<std::uint32_t>(height)}};
    return {{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)}};
}

inline int plane_count(const std::string& fmt) { return fmt == "nv12" ? 2 : 1; }

inline void validate_geometry(int width, int height, const std::string& fmt,
                              const std::string& what) {
    if (!is_dsp_format(fmt))
        throw DspError("unsupported format '" + fmt + "' (nv12/rgb24/gray8)");
    if (fmt == "nv12" && (width % 2 || height % 2))
        throw DspError(what + ": nv12 needs even width/height (got " +
                       std::to_string(width) + "x" + std::to_string(height) + ")");
    if (!(kMinDim <= width && width <= kMaxDim && kMinDim <= height && height <= kMaxDim))
        throw DspError(what + ": dims " + std::to_string(width) + "x" + std::to_string(height) +
                       " outside daemon range [" + std::to_string(kMinDim) + ", " +
                       std::to_string(kMaxDim) + "]");
}

}  // namespace neoruntime_ipc_sdk::detail::dsp_wire
