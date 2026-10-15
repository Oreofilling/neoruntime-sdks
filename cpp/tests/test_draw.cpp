// test_draw.cpp — draw_boxes / draw_text / draw_detections /
// draw_polygons parity (port of python tests/test_draw.py). The python
// suite's without_cv2 legs pin the Pillow fallbacks, which do not port
// (OpenCV is the C++ SDK's hard dependency); the cv2 expectations are
// the same pixels either way.
#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/draw.hpp"
#include "neoruntime_ipc_sdk/inference_types.hpp"

using namespace neoruntime_ipc_sdk;

namespace {

cv::Mat black_canvas(int h = 100, int w = 160) {
    return cv::Mat::zeros(h, w, CV_8UC3);
}

bool mats_equal(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::countNonZero(diff.reshape(1)) == 0;
}

// python's out[y0:y1, x0:x1].any()
bool region_any(const cv::Mat& img, int y0, int y1, int x0, int x1) {
    const cv::Scalar s = cv::sum(img(cv::Range(y0, y1), cv::Range(x0, x1)));
    return s[0] > 0 || s[1] > 0 || s[2] > 0;
}

// python's (out[y, x0:x1, ch] == v).all()
bool row_channel_all(const cv::Mat& img, int y, int x0, int x1, int ch, std::uint8_t v) {
    for (int x = x0; x < x1; ++x)
        if (img.at<cv::Vec3b>(y, x)[ch] != v) return false;
    return true;
}

std::vector<DetectedObject> two_objects() {
    DetectedObject a;
    a.label = "person";
    a.score = 0.91f;
    a.bbox = BoundingBox{10, 10, 50, 40};
    a.class_id = 0;
    DetectedObject b;
    b.label = "vehicle";
    b.score = 0.75f;
    b.bbox = BoundingBox{80, 50, 40, 30};
    b.class_id = 1;
    return {a, b};
}

}  // namespace

// ---- TestDrawBoxes -------------------------------------------------------------------

TEST(DrawBoxes, BoxPixelsDrawn) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_boxes(img, {{10, 10, 60, 50}}, {}, {}, {0, 255, 0}, 2);
    EXPECT_TRUE(row_channel_all(out, 10, 12, 58, 1, 255));  // top edge green
    EXPECT_EQ(out.rows, img.rows);
    EXPECT_EQ(out.cols, img.cols);
}

TEST(DrawBoxes, InputNotMutated) {
    cv::Mat img = black_canvas();
    cv::Mat before = img.clone();
    draw_boxes(img, {{5, 5, 40, 40}});
    EXPECT_TRUE(mats_equal(img, before));
}

TEST(DrawBoxes, EmptyBoxesReturnsCopy) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_boxes(img, {});
    EXPECT_TRUE(mats_equal(out, img));
    EXPECT_NE(out.data, img.data);  // a copy, not the same buffer
}

TEST(DrawBoxes, MultipleBoxes) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_boxes(img, {{0, 0, 20, 20}, {40, 40, 80, 80}}, {}, {},
                             {255, 0, 0});
    EXPECT_TRUE(row_channel_all(out, 0, 2, 18, 0, 255));
    EXPECT_TRUE(row_channel_all(out, 40, 42, 78, 0, 255));
}

TEST(DrawBoxes, LabelDrawnNearBox) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_boxes(img, {{10, 30, 60, 80}},
                             {std::optional<std::string>("person")}, {}, {0, 255, 0});
    // label region above the box top edge has non-zero pixels
    EXPECT_TRUE(region_any(out, 16, 28, 10, 60));
}

TEST(DrawBoxes, ScoresAccepted) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_boxes(img, {{10, 30, 60, 80}},
                             {std::optional<std::string>("person")},
                             {std::optional<double>(0.87)}, {0, 255, 0});
    EXPECT_TRUE(region_any(out, 10, 28, 10, 60));
}

// ---- TestDrawText --------------------------------------------------------------------

TEST(DrawText, TextPixelsDrawn) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_text(img, "hello", {5, 5}, {255, 255, 255});
    EXPECT_TRUE(region_any(out, 5, 20, 5, 60));
}

TEST(DrawText, InputNotMutated) {
    cv::Mat img = black_canvas();
    cv::Mat before = img.clone();
    draw_text(img, "x", {5, 5});
    EXPECT_TRUE(mats_equal(img, before));
}

// ---- TestDrawDetections ----------------------------------------------------------------

TEST(DrawDetections, DetectionsDrawnWithLabels) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_detections(img, two_objects());
    EXPECT_TRUE(region_any(out, 10, 11, 12, 58));  // first box edge
    EXPECT_TRUE(region_any(out, 50, 51, 82, 118)); // second box edge
}

TEST(DrawDetections, AcceptsResultObject) {
    cv::Mat img = black_canvas();
    InferenceResult result;
    result.objects = two_objects();
    cv::Mat out = draw_detections(img, result);
    EXPECT_TRUE(region_any(out, 10, 11, 12, 58));
}

TEST(DrawDetections, InputNotMutated) {
    cv::Mat img = black_canvas();
    cv::Mat before = img.clone();
    draw_detections(img, two_objects());
    EXPECT_TRUE(mats_equal(img, before));
}

TEST(DrawDetections, DistinctClassColors) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_detections(img, two_objects());
    // two class ids -> two different edge colors; both edges must be drawn
    const cv::Vec3b edge1 = out.at<cv::Vec3b>(10, 30);
    const cv::Vec3b edge2 = out.at<cv::Vec3b>(50, 100);
    EXPECT_NE(edge1, cv::Vec3b(0, 0, 0));
    EXPECT_NE(edge2, cv::Vec3b(0, 0, 0));
}

TEST(DrawDetections, EmptyDetections) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_detections(img, std::vector<DetectedObject>{});
    EXPECT_TRUE(mats_equal(out, img));
}

// ---- TestDrawPolygons -----------------------------------------------------------------

namespace {

LineShape square_shape() {
    LineShape shape;
    shape.points = {{10, 10}, {90, 10}, {90, 90}, {10, 90}};
    shape.color = Rgb{0, 255, 0};
    shape.closed = true;
    return shape;
}

}  // namespace

TEST(DrawPolygons, PolygonPixelsDrawn) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_polygons(img, {square_shape()});
    EXPECT_TRUE(row_channel_all(out, 10, 12, 88, 1, 255));  // top edge
    EXPECT_TRUE(row_channel_all(out, 90, 12, 88, 1, 255));  // closing side
    EXPECT_EQ(out.rows, img.rows);
    EXPECT_EQ(out.cols, img.cols);
}

TEST(DrawPolygons, InputNotMutated) {
    cv::Mat img = black_canvas();
    cv::Mat before = img.clone();
    LineShape shape = square_shape();
    shape.color = std::nullopt;
    draw_polygons(img, {shape});
    EXPECT_TRUE(mats_equal(img, before));
}

TEST(DrawPolygons, EmptyShapesReturnsCopy) {
    cv::Mat img = black_canvas();
    cv::Mat out = draw_polygons(img, {});
    EXPECT_TRUE(mats_equal(out, img));
    EXPECT_NE(out.data, img.data);
}

TEST(DrawPolygons, OpenTrackSkipsClosingSegment) {
    cv::Mat img = black_canvas();
    LineShape shape;
    shape.points = {{10, 10}, {90, 10}, {90, 90}};
    shape.color = Rgb{0, 255, 0};
    cv::Mat out = draw_polygons(img, {shape}, 2, /*closed=*/false);
    // the closing diagonal through (50, 50) is not drawn
    EXPECT_FALSE(region_any(out, 45, 56, 45, 56));
}

TEST(DrawPolygons, NoneColorDefaultsGreen) {
    cv::Mat img = black_canvas();
    LineShape shape = square_shape();
    shape.color = std::nullopt;
    cv::Mat out = draw_polygons(img, {shape});
    EXPECT_TRUE(row_channel_all(out, 10, 12, 88, 1, 255));
}
