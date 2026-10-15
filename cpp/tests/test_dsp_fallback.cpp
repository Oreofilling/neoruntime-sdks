// test_dsp_fallback.cpp — CPU-fallback parity for detail::dsp_format
// (python dsp_format.py): stretch identity, letterbox/scale_crop geometry,
// nv12 plane-split semantics (UV nearest), crop byte-copy semantics,
// BT.601 convert vectors, straight-alpha blend math, and the validation
// error surface. The memfd-backed DspBufferPool lifecycle tests join this
// file when the client lands (Phase 5 continues).
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/dsp.hpp"

#include "detail/dsp_format.hpp"

namespace f = neoruntime_ipc_sdk::detail::dsp_format;
using neoruntime_ipc_sdk::DspError;

namespace {

cv::Mat gray8(int h, int w, std::uint8_t v) { return cv::Mat(h, w, CV_8UC1, cv::Scalar(v)); }

// Mat-vs-constant / Mat-vs-Mat equality (cv::norm against a Scalar
// type-mismatches on OpenCV 4.5 — Scalar promotes to CV_64F).
bool all_eq(const cv::Mat& m, std::uint8_t v) {
    cv::Mat diff;
    cv::compare(m, v, diff, cv::CMP_NE);
    return cv::countNonZero(diff) == 0;
}

bool mat_eq(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type()) return false;
    cv::Mat diff;
    cv::absdiff(a, b, diff);  // countNonZero needs cn==1: flatten
    return cv::countNonZero(diff.reshape(1, static_cast<int>(diff.total()))) == 0;
}

// NV12 plane with distinct luma / chroma constants.
cv::Mat nv12(int h, int w, std::uint8_t y, std::uint8_t u, std::uint8_t v) {
    cv::Mat m(h * 3 / 2, w, CV_8UC1, cv::Scalar(y));
    for (int r = 0; r < h / 2; ++r) {
        std::uint8_t* row = m.ptr<std::uint8_t>(h + r);
        for (int c = 0; c < w / 2; ++c) {
            row[2 * c] = u;
            row[2 * c + 1] = v;
        }
    }
    return m;
}

}  // namespace

// ---- stretch to the same size is an identity copy -----------------------------
TEST(DspFallback, StretchIdentity) {
    const cv::Mat g = gray8(32, 64, 77);
    const cv::Mat g2 = f::cpu_resize(g, "gray8", 64, 32, "stretch", "bilinear");
    EXPECT_TRUE(mat_eq(g, g2));

    cv::Mat rng(16, 32, CV_8UC3);
    cv::randu(rng, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    const cv::Mat rgb = f::cpu_resize(rng, "rgb24", 32, 16, "stretch", "bicubic");
    EXPECT_TRUE(mat_eq(rng, rgb));
}

// ---- nv12 stretch: Y rides the chosen interp, UV rides nearest -----------------
// Constant planes make any interpolation exact: output must stay constant.
TEST(DspFallback, Nv12StretchConstant) {
    const cv::Mat src = nv12(32, 64, 100, 80, 160);
    const cv::Mat out = f::cpu_resize(src, "nv12", 32, 16, "stretch", "bilinear");
    ASSERT_EQ(out.rows, 24);
    ASSERT_EQ(out.cols, 32);
    EXPECT_TRUE(all_eq(out.rowRange(0, 16), 100));
    for (int r = 16; r < 24; ++r) {
        const std::uint8_t* row = out.ptr<std::uint8_t>(r);
        for (int c = 0; c < 16; ++c) {
            EXPECT_EQ(row[2 * c], 80);
            EXPECT_EQ(row[2 * c + 1], 160);
        }
    }
}

// ---- letterbox: black bars, centered content, neutral chroma padding -----------
TEST(DspFallback, LetterboxNv12) {
    const cv::Mat src = nv12(32, 64, 100, 80, 160);
    // 64x32 into 64x64: scale 1.0, content at oy=16; scale==1 keeps bytes
    const cv::Mat out = f::cpu_resize(src, "nv12", 64, 64, "letterbox", "bilinear");
    ASSERT_EQ(out.rows, 96);
    ASSERT_EQ(out.cols, 64);
    // luma: top/bottom bars black, center is a byte copy of the source
    EXPECT_TRUE(all_eq(out.rowRange(0, 16), 0));
    EXPECT_TRUE(all_eq(out.rowRange(48, 64), 0));
    EXPECT_TRUE(mat_eq(out.rowRange(16, 48), src.rowRange(0, 32)));
    // chroma: pad rows neutral 128, content rows carry the source chroma
    EXPECT_TRUE(all_eq(out.rowRange(64, 72), 128));
    EXPECT_TRUE(all_eq(out.rowRange(88, 96), 128));
    EXPECT_TRUE(mat_eq(out.rowRange(72, 88), src.rowRange(32, 48)));
}

TEST(DspFallback, LetterboxUpLeftPinsOrigin) {
    const cv::Mat src = gray8(16, 16, 200);
    const cv::Mat out =
        f::cpu_resize(src, "gray8", 32, 32, "letterbox_up_left", "bilinear");
    // scale = min(2, 2) = 2 -> 32x32 content fills everything (no bars)
    EXPECT_TRUE(all_eq(out, 200));
}

// ---- scale_crop: center crop of the upscaled canvas ----------------------------
TEST(DspFallback, ScaleCropCenterCrop) {
    // 32x16 -> 16x16: scale = max(0.5, 1.0) = 1.0, crop x=8..24
    cv::Mat src(16, 32, CV_8UC1);
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 32; ++c)
            src.at<std::uint8_t>(r, c) = static_cast<std::uint8_t>(c);
    const cv::Mat out = f::cpu_resize(src, "gray8", 16, 16, "scale_crop", "bilinear");
    ASSERT_EQ(out.cols, 16);
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            EXPECT_EQ(out.at<std::uint8_t>(r, c), static_cast<std::uint8_t>(8 + c));
}

// ---- crop: byte-copy semantics per plane layout --------------------------------
TEST(DspFallback, Nv12CropBytes) {
    cv::Mat src = nv12(32, 64, 0, 0, 0);
    // make luma position-dependent so the crop is observable
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 64; ++c)
            src.at<std::uint8_t>(r, c) = static_cast<std::uint8_t>(r * 64 + c);
    const cv::Mat out = f::cpu_crop(src, "nv12", 8, 8, 16, 8);
    ASSERT_EQ(out.rows, 12);  // 8 luma + 4 chroma
    ASSERT_EQ(out.cols, 16);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 16; ++c)
            EXPECT_EQ(out.at<std::uint8_t>(r, c),
                      static_cast<std::uint8_t>((8 + r) * 64 + 8 + c));
    // chroma rows come from src rows 32+4..32+8, cols 8..24 (byte-wide)
    for (int r = 8; r < 12; ++r)
        for (int c = 0; c < 16; ++c)
            EXPECT_EQ(out.at<std::uint8_t>(r, c), src.at<std::uint8_t>(32 + (r - 8) + 4, 8 + c));
}

TEST(DspFallback, CropResizeSameDimsEqualsCrop) {
    cv::Mat src = gray8(64, 64, 42);
    src.at<std::uint8_t>(10, 10) = 255;
    const f::CropRect rect{8, 8, 16, 16, 16, 16};
    const cv::Mat out = f::cpu_crop_resize(src, "gray8", rect);
    EXPECT_EQ(out.at<std::uint8_t>(2, 2), 255);  // (10,10) - (8,8)
    EXPECT_EQ(out.rows, 16);
}

// ---- convert: BT.601 vectors -----------------------------------------------------
TEST(DspFallback, RgbToGrayVector) {
    // y = round(16 + (65.481R + 128.553G + 24.966B)/255), studio swing
    cv::Mat rgb(1, 3, CV_8UC3);
    rgb.at<cv::Vec3b>(0, 0) = {255, 0, 0};  // -> 81.481 -> 81
    rgb.at<cv::Vec3b>(0, 1) = {0, 255, 0};  // -> 144.553 -> 145
    rgb.at<cv::Vec3b>(0, 2) = {0, 0, 255};  // -> 40.966 -> 41
    const cv::Mat out = f::cpu_convert(rgb, "rgb24", "gray8");
    EXPECT_EQ(out.at<std::uint8_t>(0, 0), 81);
    EXPECT_EQ(out.at<std::uint8_t>(0, 1), 145);
    EXPECT_EQ(out.at<std::uint8_t>(0, 2), 41);
}

TEST(DspFallback, GrayPairs) {
    const cv::Mat g = gray8(16, 16, 99);
    const cv::Mat rgb = f::cpu_convert(g, "gray8", "rgb24");
    ASSERT_EQ(rgb.channels(), 3);
    EXPECT_EQ(rgb.at<cv::Vec3b>(5, 5), cv::Vec3b(99, 99, 99));

    const cv::Mat back = f::cpu_convert(rgb, "rgb24", "gray8");
    // full-swing 99 gray through studio-swing luma: 16 + 219*99/255 = 101.06
    // -> 101 (the same value python's coefficients produce)
    EXPECT_EQ(back.at<std::uint8_t>(5, 5), 101);

    const cv::Mat n = f::cpu_convert(g, "gray8", "nv12");
    ASSERT_EQ(n.rows, 24);
    EXPECT_TRUE(all_eq(n.rowRange(0, 16), 99));
    EXPECT_TRUE(all_eq(n.rowRange(16, 24), 128));

    const cv::Mat luma = f::cpu_convert(n, "nv12", "gray8");
    EXPECT_TRUE(mat_eq(luma, g));

    EXPECT_THROW(f::cpu_convert(g, "gray8", "gray8"), DspError);
}

TEST(DspFallback, Nv12RgbRoundTrip) {
    const cv::Mat src = nv12(32, 64, 100, 128, 128);  // neutral chroma
    const cv::Mat rgb = f::cpu_convert(src, "nv12", "rgb24");
    ASSERT_EQ(rgb.rows, 32);
    ASSERT_EQ(rgb.cols, 64);
    const cv::Vec3b px = rgb.at<cv::Vec3b>(10, 10);
    EXPECT_EQ(px[0], px[1]);  // neutral chroma -> gray
    EXPECT_EQ(px[1], px[2]);
    EXPECT_NEAR(px[0], 100, 2);  // limited-range luma 100 ~ 98..100

    const cv::Mat back = f::cpu_convert(rgb, "rgb24", "nv12");
    EXPECT_TRUE(cv::norm(back, src, cv::NORM_INF) <= 2);
}

// ---- blend: straight alpha math ----------------------------------------------------
TEST(DspFallback, BlendAlphaCorners) {
    cv::Mat base(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat ov(2, 2, CV_8UC4);
    ov.at<cv::Vec4b>(0, 0) = {255, 0, 0, 255};  // opaque red
    ov.at<cv::Vec4b>(0, 1) = {255, 0, 0, 128};  // half red on black
    ov.at<cv::Vec4b>(1, 0) = {255, 0, 0, 0};    // fully transparent
    ov.at<cv::Vec4b>(1, 1) = {0, 0, 0, 128};    // half black on black
    const f::RgbaOverlay overlay{ov, 1, 1};
    const cv::Mat out = f::cpu_blend(base, "rgb24", {overlay});
    EXPECT_EQ(out.at<cv::Vec3b>(1, 1), cv::Vec3b(255, 0, 0));
    // (128*255 + 127*0)/255 = 128
    EXPECT_EQ(out.at<cv::Vec3b>(1, 2), cv::Vec3b(128, 0, 0));
    EXPECT_EQ(out.at<cv::Vec3b>(2, 1), cv::Vec3b(0, 0, 0));
    EXPECT_EQ(out.at<cv::Vec3b>(2, 2), cv::Vec3b(0, 0, 0));
    // untouched corners
    EXPECT_EQ(out.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 0));
    EXPECT_EQ(out.at<cv::Vec3b>(3, 3), cv::Vec3b(0, 0, 0));
}

TEST(DspFallback, BlendOverhangClipped) {
    cv::Mat base(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat ov(4, 4, CV_8UC4, cv::Scalar(200, 200, 200, 255));
    const f::RgbaOverlay overlay{ov, 2, 2};  // 2x2 lands, 2x2 overhangs
    const cv::Mat out = f::cpu_blend(base, "rgb24", {overlay});
    EXPECT_EQ(out.at<cv::Vec3b>(2, 2), cv::Vec3b(200, 200, 200));
    EXPECT_EQ(out.at<cv::Vec3b>(3, 3), cv::Vec3b(200, 200, 200));
    EXPECT_EQ(out.at<cv::Vec3b>(0, 0), cv::Vec3b(10, 20, 30));
}

// ---- validation surface --------------------------------------------------------------
TEST(DspFallback, InferFmt) {
    cv::Mat rgb(16, 16, CV_8UC3);
    EXPECT_EQ(f::infer_fmt(rgb, ""), "rgb24");
    EXPECT_EQ(f::infer_fmt(rgb, "nv12"), "nv12");
    EXPECT_THROW(f::infer_fmt(rgb, "yuyv"), DspError);
    cv::Mat gray(16, 16, CV_8UC1);
    EXPECT_EQ(f::infer_fmt(gray, "gray8"), "gray8");  // explicit: no warning path
    // frame-format mapping
    EXPECT_EQ(f::frame_fmt_to_dsp("NV12"), "nv12");
    EXPECT_EQ(f::frame_fmt_to_dsp("BGR"), "rgb24");
    EXPECT_EQ(f::frame_fmt_to_dsp("GRAY8"), "gray8");
    EXPECT_EQ(f::frame_fmt_to_dsp("RGB565"), "");
}

TEST(DspFallback, SrcDims) {
    const cv::Mat n = nv12(32, 64, 0, 0, 0);
    EXPECT_EQ(f::src_dims(n, "nv12"), cv::Size(64, 32));
    cv::Mat rgb(16, 32, CV_8UC3);
    EXPECT_EQ(f::src_dims(rgb, "rgb24"), cv::Size(32, 16));
    EXPECT_THROW(f::src_dims(n, "rgb24"), DspError);  // 1ch where 3 needed
    EXPECT_THROW(f::src_dims(rgb, "nv12"), DspError);
}

TEST(DspFallback, ValidatedRect) {
    EXPECT_NO_THROW(f::validated_rect(64, 32, "nv12", 8, 8, 16, 8, 32, 16));
    EXPECT_THROW(f::validated_rect(64, 32, "nv12", 8, 8, 0, 8, 32, 16), DspError);
    EXPECT_THROW(f::validated_rect(64, 32, "nv12", 60, 0, 8, 8, 32, 16), DspError);
    EXPECT_THROW(f::validated_rect(64, 32, "nv12", 7, 8, 16, 8, 32, 16), DspError);  // odd x
    EXPECT_THROW(f::validated_rect(64, 32, "nv12", 8, 8, 16, 8, 8, 16), DspError);   // dw < min
    // odd coords fine for packed formats
    EXPECT_NO_THROW(f::validated_rect(63, 31, "rgb24", 7, 7, 17, 17, 33, 17));
}
