// test_color.cpp — NV12 <-> RGB/BGR conversion parity (port of python
// tests/test_color.py). The python suite forces the numpy fallback legs
// with a no_cv2 fixture; the C++ SDK's OpenCV is a hard dependency, so
// these pin the cvtColor legs against the same numeric expectations
// (BT.601 limited range — Y=128 with neutral chroma decodes to 130).
#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "neoruntime_ipc_sdk/color.hpp"

using namespace neoruntime_ipc_sdk;

namespace {

constexpr int kH = 32;
constexpr int kW = 64;

cv::Mat uniform_image(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    return cv::Mat(kH, kW, CV_8UC3, cv::Scalar(r, g, b));
}

cv::Mat uniform_nv12(std::uint8_t y, std::uint8_t uv) {
    cv::Mat nv12(kH * 3 / 2, kW, CV_8UC1, cv::Scalar(y));
    nv12.rowRange(kH, kH * 3 / 2).setTo(uv);
    return nv12;
}

double channel_mean(const cv::Mat& img, int channel) {
    cv::Mat plane;
    cv::extractChannel(img, plane, channel);
    return cv::mean(plane)[0];
}

double mean_abs_diff(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::mean(diff)[0];
}

// python's pytest.raises(match=...) — returns the message of the thrown
// exception, or "" when nothing fired.
template <typename Fn>
std::string thrown_message(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& exc) {
        return exc.what();
    }
    return "";
}

}  // namespace

// ---- TestNv12ToPacked ----------------------------------------------------------------

TEST(Nv12ToPacked, OutputShape) {
    cv::Mat rgb = nv12_to_rgb(uniform_nv12(128, 128), kW, kH);
    EXPECT_EQ(rgb.rows, kH);
    EXPECT_EQ(rgb.cols, kW);
    EXPECT_EQ(rgb.type(), CV_8UC3);
}

TEST(Nv12ToPacked, GrayDecodesToLimitedRange130) {
    // neutral chroma -> perfectly gray; Y=128 decodes to 130 in limited
    // range (1.1644 * (128-16) = 130.4 -> 130)
    cv::Mat rgb = nv12_to_rgb(uniform_nv12(128, 128), kW, kH);
    cv::Mat ch0, ch1, ch2;
    cv::extractChannel(rgb, ch0, 0);
    cv::extractChannel(rgb, ch1, 1);
    cv::extractChannel(rgb, ch2, 2);
    EXPECT_EQ(cv::countNonZero(ch0 != ch1), 0);
    EXPECT_EQ(cv::countNonZero(ch1 != ch2), 0);
    EXPECT_NEAR(cv::mean(ch0)[0], 130.0, 1.0);
}

TEST(Nv12ToPacked, RedChannelDominatesForRed) {
    cv::Mat nv12 = rgb_to_nv12(uniform_image(255, 0, 0));
    cv::Mat rgb = nv12_to_rgb(nv12, kW, kH);
    EXPECT_GT(channel_mean(rgb, 0), 200.0);
    EXPECT_LT(channel_mean(rgb, 2), 60.0);
}

TEST(Nv12ToPacked, BgrOrderSwapped) {
    cv::Mat nv12 = rgb_to_nv12(uniform_image(255, 0, 0));
    cv::Mat bgr = nv12_to_bgr(nv12, kW, kH);
    EXPECT_GT(channel_mean(bgr, 2), 200.0);
    EXPECT_LT(channel_mean(bgr, 0), 60.0);
}

TEST(Nv12ToPacked, WrongDimsRaise) {
    cv::Mat nv12(kH, kW, CV_8UC1, cv::Scalar(0));  // missing UV plane
    const std::string msg = thrown_message(
        [&] { nv12_to_rgb(nv12, kW, kH); });
    EXPECT_NE(msg.find("NV12 buffer"), std::string::npos);
}

// ---- TestPackedToNv12 ----------------------------------------------------------------

TEST(PackedToNv12, OutputShape) {
    cv::Mat nv12 = rgb_to_nv12(uniform_image(30, 200, 90));
    EXPECT_EQ(nv12.rows, kH * 3 / 2);
    EXPECT_EQ(nv12.cols, kW);
    EXPECT_EQ(nv12.type(), CV_8UC1);
}

TEST(PackedToNv12, OddDimensionsRaise) {
    cv::Mat odd(33, 64, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::string msg = thrown_message([&] { rgb_to_nv12(odd); });
    EXPECT_NE(msg.find("even"), std::string::npos);
}

TEST(PackedToNv12, SmoothRoundtripTolerance) {
    // realistic content keeps structure through 4:2:0 subsampling
    cv::Mat rgb(kH, kW, CV_8UC3);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            cv::Vec3b px;
            px[0] = static_cast<std::uint8_t>(std::min(255, 60 + x * 2));
            px[1] = static_cast<std::uint8_t>(std::min(255, 30 + y * 3));
            px[2] = 90;
            rgb.at<cv::Vec3b>(y, x) = px;
        }
    cv::Mat back = nv12_to_rgb(rgb_to_nv12(rgb), kW, kH);
    EXPECT_LT(mean_abs_diff(back, rgb), 4.0);
}

TEST(PackedToNv12, NoiseRoundtripStaysBounded) {
    // pure noise is the 4:2:0 worst case — chroma error per 2x2 block is
    // amplified ~2x on decode, so allow a coarse bound only
    cv::theRNG().state = 7;
    cv::Mat rgb(kH, kW, CV_8UC3);
    cv::randu(rgb, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Mat back = nv12_to_rgb(rgb_to_nv12(rgb), kW, kH);
    EXPECT_LT(mean_abs_diff(back, rgb), 60.0);
}

TEST(PackedToNv12, GraySurvivesRoundtrip) {
    for (int level : {40, 120, 210}) {
        cv::Mat gray = uniform_image(static_cast<std::uint8_t>(level),
                                     static_cast<std::uint8_t>(level),
                                     static_cast<std::uint8_t>(level));
        cv::Mat back = nv12_to_rgb(rgb_to_nv12(gray), kW, kH);
        EXPECT_LT(mean_abs_diff(back, gray), 2.0) << "level " << level;
    }
}

TEST(PackedToNv12, BgrEntryPointMatchesRgbForGray) {
    cv::Mat gray = uniform_image(100, 100, 100);
    cv::Mat a = rgb_to_nv12(gray.clone());
    cv::Mat b = bgr_to_nv12(gray.clone());
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

// ---- TestNv12Resize ------------------------------------------------------------------

namespace {
cv::Mat resize_sample() {
    return uniform_nv12(200, 128);
}
}  // namespace

TEST(Nv12Resize, OutputShape) {
    cv::Mat out = nv12_resize(resize_sample(), {kW, kH}, {32, 16});
    EXPECT_EQ(out.rows, 16 * 3 / 2);
    EXPECT_EQ(out.cols, 32);
}

TEST(Nv12Resize, ValuesPreservedForUniformPlanes) {
    cv::Mat out = nv12_resize(resize_sample(), {kW, kH}, {32, 16});
    EXPECT_NEAR(cv::mean(out.rowRange(0, 16))[0], 200.0, 1.0);
    EXPECT_NEAR(cv::mean(out.rowRange(16, 24))[0], 128.0, 1.0);
}

TEST(Nv12Resize, UpscaleRejected) {
    const std::string msg = thrown_message(
        [&] { nv12_resize(resize_sample(), {kW, kH}, {kW * 2, kH * 2}); });
    EXPECT_NE(msg.find("upscaling"), std::string::npos);
}

TEST(Nv12Resize, OddDestinationRejected) {
    const std::string msg = thrown_message(
        [&] { nv12_resize(resize_sample(), {kW, kH}, {33, 17}); });
    EXPECT_NE(msg.find("even"), std::string::npos);
}

TEST(Nv12Resize, WrongSourceDimsRaise) {
    cv::Mat nv12(kH, kW, CV_8UC1, cv::Scalar(0));
    const std::string msg = thrown_message(
        [&] { nv12_resize(nv12, {kW, kH}, {32, 16}); });
    EXPECT_NE(msg.find("NV12 buffer"), std::string::npos);
}
