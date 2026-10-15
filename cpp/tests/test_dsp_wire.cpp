// test_dsp_wire.cpp — DSP UDS wire codec parity with python dsp_wire.py
// (0.8.1), mirroring tests/test_dsp.py TestWireCodec + TestImportWireCodec:
// request layouts byte-for-byte, response round-trips (incl. the u64
// high-bit import-id regression), error payloads, geometry validation and
// the enum wire maps.
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/dsp.hpp"

#include "detail/dsp_wire.hpp"

namespace w = neoruntime_ipc_sdk::detail::dsp_wire;
using neoruntime_ipc_sdk::DspError;

// ---- ALLOC request layout (python: mtype,msize,w,h,fmt,count == 7,24,1920,1080,0,2)
TEST(DspWire, AllocRequestLayout) {
    const auto raw = w::alloc_request(1920, 1080, 0, 2);
    ASSERT_EQ(raw.size(), 24u);
    EXPECT_EQ(w::get_u32(&raw[0]), 7u);     // FD_PUB_MSG_DSP_ALLOC
    EXPECT_EQ(w::get_u32(&raw[4]), 24u);    // size
    EXPECT_EQ(w::get_u32(&raw[8]), 1920u);
    EXPECT_EQ(w::get_u32(&raw[12]), 1080u);
    EXPECT_EQ(w::get_u32(&raw[16]), 0u);    // nv12 wire value
    EXPECT_EQ(w::get_u32(&raw[20]), 2u);
}

// ---- ALLOC response round-trip with per-plane stride padding -----------------
TEST(DspWire, AllocRespRoundtrip) {
    std::vector<std::uint8_t> payload(w::kAllocRespSize, 0);
    w::put_u32(payload.data(), w::kMsgDspAllocResp);
    w::put_u32(payload.data() + 4, static_cast<std::uint32_t>(w::kAllocRespSize));
    w::put_u32(payload.data() + 8, 0);    // code (i32)
    w::put_u32(payload.data() + 12, 2);   // count
    w::put_u32(payload.data() + 16, 2);   // num_planes
    for (int p = 0; p < 2; ++p) {
        w::put_u32(payload.data() + 20 + 4 * p, 1920 + p);             // strides
        w::put_u32(payload.data() + 32 + 4 * p, 1080u * (1920u + p));  // sizes
    }
    for (int i = 0; i < 2; ++i) w::put_u64(payload.data() + 48 + 8 * i, 0xA000 + i);

    const w::AllocResp r = w::parse_alloc_resp(payload.data(), payload.size());
    EXPECT_EQ(r.code, 0);
    EXPECT_EQ(r.count, 2u);
    EXPECT_EQ(r.num_planes, 2u);
    EXPECT_EQ(r.strides[0], 1920u);
    EXPECT_EQ(r.strides[1], 1921u);
    EXPECT_EQ(r.sizes[0], 1080u * 1920u);
    EXPECT_EQ(r.sizes[1], 1080u * 1921u);
    ASSERT_EQ(r.ids.size(), 2u);
    EXPECT_EQ(r.ids[0], 0xA000ull);
    EXPECT_EQ(r.ids[1], 0xA001ull);
}

TEST(DspWire, AllocRespErrorPayload) {
    std::vector<std::uint8_t> payload(w::kAllocRespSize, 0);
    w::put_u32(payload.data(), w::kMsgDspAllocResp);
    w::put_u32(payload.data() + 8, static_cast<std::uint32_t>(-7));  // code i32
    const w::AllocResp r = w::parse_alloc_resp(payload.data(), payload.size());
    EXPECT_EQ(r.code, -7);
    EXPECT_EQ(r.count, 0u);
    EXPECT_TRUE(r.ids.empty());
}

// ---- IMPORT request layout: (10,48,3840,2160,0,2,3840,1920,0,3317760,1658880,0)
TEST(DspWire, ImportRequestLayout) {
    const std::uint32_t strides[3] = {3840, 1920, 0};
    const std::uint32_t sizes[3] = {3317760, 1658880, 0};
    const auto raw = w::import_request(3840, 2160, 0, 2, strides, sizes);
    ASSERT_EQ(raw.size(), 48u);
    const std::uint32_t want[12] = {10, 48, 3840, 2160, 0, 2, 3840, 1920, 0, 3317760, 1658880, 0};
    for (int i = 0; i < 12; ++i) EXPECT_EQ(w::get_u32(&raw[4 * i]), want[i]) << "field " << i;
}

// ---- IMPORT response: round-trip, error, and the u64 high-bit regression -----
TEST(DspWire, ImportRespRoundtrip) {
    std::vector<std::uint8_t> ok(w::kImportRespSize, 0);
    w::put_u32(ok.data(), w::kMsgDspImportResp);
    w::put_u32(ok.data() + 4, 24);
    w::put_u32(ok.data() + 8, 0);
    w::put_u64(ok.data() + 16, 4242);
    const w::ImportResp r = w::parse_import_resp(ok.data(), ok.size());
    EXPECT_EQ(r.code, 0);
    EXPECT_EQ(r.import_id, 4242ull);

    std::vector<std::uint8_t> bad(w::kImportRespSize, 0);
    w::put_u32(bad.data(), w::kMsgDspImportResp);
    w::put_u32(bad.data() + 8, static_cast<std::uint32_t>(-1));
    const w::ImportResp e = w::parse_import_resp(bad.data(), bad.size());
    EXPECT_EQ(e.code, -1);
    EXPECT_EQ(e.import_id, 0ull);
}

// Regression (device run 2026-09-09): daemon import ids are uint64 and
// routinely have the high bit set; a signed parse made blend_hw fail proto
// field validation.
TEST(DspWire, ImportRespHighBitId) {
    const std::uint64_t big = 16053137586893052802ull;  // observed live
    std::vector<std::uint8_t> raw(w::kImportRespSize, 0);
    w::put_u32(raw.data(), w::kMsgDspImportResp);
    w::put_u64(raw.data() + 16, big);
    EXPECT_EQ(w::parse_import_resp(raw.data(), raw.size()).import_id, big);
}

// ---- malformed responses raise DspError --------------------------------------
TEST(DspWire, RejectsWrongTypeAndShort) {
    std::vector<std::uint8_t> wrong(w::kImportRespSize, 0);
    w::put_u32(wrong.data(), 5);  // OK ack, not IMPORT_RESP
    EXPECT_THROW(w::parse_import_resp(wrong.data(), wrong.size()), DspError);
    const std::uint8_t one[1] = {0x0b};
    EXPECT_THROW(w::parse_import_resp(one, sizeof(one)), DspError);

    std::vector<std::uint8_t> short_alloc(100, 0);
    EXPECT_THROW(w::parse_alloc_resp(short_alloc.data(), short_alloc.size()), DspError);
    std::vector<std::uint8_t> wrong_alloc(w::kAllocRespSize, 0);
    w::put_u32(wrong_alloc.data(), 6);  // ERROR ack, not ALLOC_RESP
    EXPECT_THROW(w::parse_alloc_resp(wrong_alloc.data(), wrong_alloc.size()), DspError);
}

// ---- RELEASE encoding ----------------------------------------------------------
TEST(DspWire, ReleaseRequest) {
    const auto raw = w::release_request(0xDEADBEEFull);
    ASSERT_EQ(raw.size(), 16u);
    EXPECT_EQ(w::get_u32(&raw[0]), 9u);  // FD_PUB_MSG_DSP_BUF_RELEASE
    EXPECT_EQ(w::get_u32(&raw[4]), 16u);
    EXPECT_EQ(w::get_u64(&raw[8]), 0xDEADBEEFull);
}

// ---- error text table -----------------------------------------------------------
TEST(DspWire, ErrorText) {
    EXPECT_EQ(w::error_text(-1), "invalid request");
    EXPECT_EQ(w::error_text(-2), "no such buffer id");
    EXPECT_EQ(w::error_text(-3), "quota exceeded");
    EXPECT_EQ(w::error_text(-4), "job timeout");
    EXPECT_EQ(w::error_text(-5), "dsp service unavailable");
    EXPECT_EQ(w::error_text(-6), "out of memory");
    EXPECT_EQ(w::error_text(-7), "client limit exceeded");
}

// ---- HAL pixel format map --------------------------------------------------------
TEST(DspWire, HalPixelFormatWireValues) {
    EXPECT_EQ(w::hal_pixel_format("nv12"), 0u);
    EXPECT_EQ(w::hal_pixel_format("rgb24"), 4u);
    EXPECT_EQ(w::hal_pixel_format("gray8"), 8u);
    EXPECT_EQ(w::hal_pixel_format("argb"), 6u);
    EXPECT_THROW(w::hal_pixel_format("yuyv"), DspError);
}

// ---- enum wire maps (camera.proto) -------------------------------------------------
TEST(DspWire, EnumWireMaps) {
    EXPECT_EQ(w::interp_wire("nearest"), w::pb::DSP_INTERP_NEAREST);
    EXPECT_EQ(w::interp_wire("bicubic"), w::pb::DSP_INTERP_BICUBIC);
    EXPECT_THROW(w::interp_wire("lanczos"), DspError);

    EXPECT_EQ(w::scaling_wire("stretch"), w::pb::DSP_SCALING_STRETCH);
    EXPECT_EQ(w::scaling_wire("letterbox"), w::pb::DSP_SCALING_LETTERBOX_MIDDLE);
    EXPECT_EQ(w::scaling_wire("letterbox_middle"), w::pb::DSP_SCALING_LETTERBOX_MIDDLE);
    EXPECT_EQ(w::scaling_wire("scale_crop"), w::pb::DSP_SCALING_SCALE_AND_CROP);
    EXPECT_THROW(w::scaling_wire("fit"), DspError);

    EXPECT_EQ(w::priority_wire("background"), w::pb::DSP_PRIORITY_BACKGROUND);
    EXPECT_EQ(w::priority_wire("normal"), w::pb::DSP_PRIORITY_NORMAL);
}

// ---- plane geometry + validation -----------------------------------------------------
TEST(DspWire, PlaneGeometry) {
    auto nv12 = w::plane_rows("nv12", 64, 32);
    ASSERT_EQ(nv12.size(), 2u);
    EXPECT_EQ(nv12[0].row_bytes, 64u);
    EXPECT_EQ(nv12[0].rows, 32u);
    EXPECT_EQ(nv12[1].row_bytes, 64u);
    EXPECT_EQ(nv12[1].rows, 16u);
    EXPECT_EQ(w::plane_count("nv12"), 2);

    auto rgb = w::plane_rows("rgb24", 32, 16);
    ASSERT_EQ(rgb.size(), 1u);
    EXPECT_EQ(rgb[0].row_bytes, 96u);
    EXPECT_EQ(rgb[0].rows, 16u);
    EXPECT_EQ(w::plane_count("rgb24"), 1);

    auto gray = w::plane_rows("gray8", 16, 16);
    EXPECT_EQ(gray[0].row_bytes, 16u);
    EXPECT_EQ(w::plane_count("gray8"), 1);
}

TEST(DspWire, GeometryValidation) {
    EXPECT_NO_THROW(w::validate_geometry(64, 32, "nv12", "alloc"));
    EXPECT_THROW(w::validate_geometry(63, 32, "nv12", "alloc"), DspError);  // odd width
    EXPECT_THROW(w::validate_geometry(64, 33, "nv12", "alloc"), DspError);  // odd height
    EXPECT_THROW(w::validate_geometry(8, 8, "rgb24", "alloc"), DspError);   // below kMinDim
    EXPECT_THROW(w::validate_geometry(8193, 16, "rgb24", "alloc"), DspError);
    EXPECT_THROW(w::validate_geometry(64, 32, "yuv420", "alloc"), DspError);  // unknown fmt
    // odd dims are fine for packed formats
    EXPECT_NO_THROW(w::validate_geometry(33, 17, "rgb24", "alloc"));
}
