// test_yolo_decode.cpp — YOLO decode adapter parity (port of python
// tests/test_yolo_decode.py): ONNX-standard export shapes, synthetic
// tensors. Public header only (postprocess.hpp + preprocess.hpp meta).
#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/postprocess.hpp"

using namespace neoruntime_ipc_sdk;

namespace {

constexpr int NC = 2;  // person, car

// python make_v8_head: (4 + NC, n) float32; entries (col, cx,cy,w,h, score, cls)
struct V8Col {
    int col;
    float cx, cy, w, h;
    float score;
    int cls;
};
cv::Mat make_v8_head(const std::vector<V8Col>& entries, int n = 200) {
    cv::Mat arr(4 + NC, n, CV_32F, cv::Scalar(0));
    for (const V8Col& e : entries) {
        arr.at<float>(0, e.col) = e.cx;
        arr.at<float>(1, e.col) = e.cy;
        arr.at<float>(2, e.col) = e.w;
        arr.at<float>(3, e.col) = e.h;
        arr.at<float>(4 + e.cls, e.col) = e.score;
    }
    return arr;
}

// python make_v5_head: (n, 5 + NC) float32; entries (row, cx,cy,w,h, obj, cls_score, cls)
struct V5Row {
    int row;
    float cx, cy, w, h;
    float obj, cls_score;
    int cls;
};
cv::Mat make_v5_head(const std::vector<V5Row>& entries, int n = 200) {
    cv::Mat arr(n, 5 + NC, CV_32F, cv::Scalar(0));
    for (const V5Row& e : entries) {
        const float xywh[] = {e.cx, e.cy, e.w, e.h};
        for (int k = 0; k < 4; ++k) arr.at<float>(e.row, k) = xywh[k];
        arr.at<float>(e.row, 4) = e.obj;
        arr.at<float>(e.row, 5 + e.cls) = e.cls_score;
    }
    return arr;
}

// python META: source 1920x1080, input 640x640, letterbox scale 1/3, pad y=140
PreprocessMeta make_meta() {
    PreprocessMeta meta;
    meta.original_size = {1920, 1080};
    meta.input_size = {640, 640};
    meta.scale = {640.0 / 1920.0, 360.0 / 1080.0};
    meta.origin = {0, 140};
    return meta;
}

std::vector<DetectedObject> run8(const YoloV8Postprocessor& post, const cv::Mat& head,
                                 const PreprocessMeta* meta = nullptr) {
    return post(std::vector<cv::Mat>{head}, meta);
}

}  // namespace

// ---- TestYoloV8 ----------------------------------------------------------------------

TEST(YoloV8, DecodeThresholdAndNms) {
    const cv::Mat head = make_v8_head({
        {0, 320, 300, 100, 80, 0.90f, 0},  // keep
        {1, 322, 301, 100, 80, 0.80f, 0},  // near-duplicate -> suppressed
        {2, 100, 100, 50, 50, 0.05f, 1},   // below threshold
    });
    YoloV8Postprocessor post({"person", "car"}, 0.3, 0.45);
    const std::vector<DetectedObject> objects = run8(post, head);
    ASSERT_EQ(objects.size(), 1u);
    const DetectedObject& obj = objects[0];
    EXPECT_EQ(obj.label, "person");
    EXPECT_NEAR(obj.score, 0.9, 1e-6);
    EXPECT_EQ(obj.class_id, 0);
    // input-coordinate xyxy: (270, 260) - (370, 340)
    EXPECT_NEAR(obj.bbox.x, 270, 1e-4);
    EXPECT_NEAR(obj.bbox.y, 260, 1e-4);
    EXPECT_NEAR(obj.bbox.width, 100, 1e-4);
    EXPECT_NEAR(obj.bbox.height, 80, 1e-4);
}

TEST(YoloV8, BatchedAndTransposedShapes) {
    const cv::Mat head = make_v8_head({{0, 320, 300, 100, 80, 0.9f, 0}});
    YoloV8Postprocessor post({}, 0.3);
    // (1, C, N): reshape the (C, N) head over the same pixels
    const int sz[] = {1, 4 + NC, 200};
    const cv::Mat batched = head.reshape(1, 3, sz);
    EXPECT_EQ(run8(post, batched).size(), 1u);
    // (N, C): transposed variant
    const cv::Mat transposed = head.t();
    EXPECT_EQ(run8(post, transposed).size(), 1u);
}

TEST(YoloV8, MetaMapsBackToSource) {
    const cv::Mat head = make_v8_head({{0, 320, 300, 100, 80, 0.9f, 0}});
    YoloV8Postprocessor post({}, 0.3);
    const PreprocessMeta meta = make_meta();
    const std::vector<DetectedObject> objects = run8(post, head, &meta);
    ASSERT_EQ(objects.size(), 1u);
    // x_src = (x_in - origin) / scale = (270 - 0) / (1/3) = 810
    EXPECT_NEAR(objects[0].bbox.x, 810, 1.0);
    // y_src = (260 - 140) / (1/3) = 360
    EXPECT_NEAR(objects[0].bbox.y, 360, 1.0);
}

TEST(YoloV8, DifferentClassesDoNotSuppress) {
    const cv::Mat head = make_v8_head({
        {0, 320, 300, 100, 80, 0.9f, 0},
        {1, 321, 301, 100, 80, 0.8f, 1},  // same place, other class -> kept
    });
    YoloV8Postprocessor post({"person", "car"}, 0.3);
    const std::vector<DetectedObject> objects = run8(post, head);
    ASSERT_EQ(objects.size(), 2u);
    EXPECT_NE(objects[0].label, objects[1].label);
    EXPECT_TRUE((objects[0].label == "person" && objects[1].label == "car") ||
                (objects[0].label == "car" && objects[1].label == "person"));
}

TEST(YoloV8, EmptyAndMalformed) {
    YoloV8Postprocessor post;
    EXPECT_TRUE(post(std::nullopt).empty());
    EXPECT_TRUE(post(std::vector<cv::Mat>{cv::Mat(0, 0, CV_32F)}).empty());
    try {
        post(std::vector<cv::Mat>{cv::Mat::zeros(3, 10, CV_32F)});  // 3 channels < 4+1
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& exc) {
        EXPECT_NE(std::string(exc.what()).find("channels"), std::string::npos);
    }
}

// ---- TestYoloV5 ------------------------------------------------------------------------

namespace {

std::vector<DetectedObject> run5(const YoloV5Postprocessor& post, const cv::Mat& head,
                                 const PreprocessMeta* meta = nullptr) {
    return post(std::vector<cv::Mat>{head}, meta);
}

}  // namespace

TEST(YoloV5, ObjectnessMultiplies) {
    const cv::Mat head = make_v5_head({{0, 320, 300, 100, 80, 0.9f, 0.95f, 0}});
    YoloV5Postprocessor post({"person", "car"}, 0.5);
    const std::vector<DetectedObject> objects = run5(post, head);
    ASSERT_EQ(objects.size(), 1u);
    EXPECT_NEAR(objects[0].score, 0.9 * 0.95, 1e-5);
    EXPECT_NEAR(objects[0].bbox.width, 100, 1e-4);
}

TEST(YoloV5, TransposedShape) {
    const cv::Mat head = make_v5_head({{0, 320, 300, 100, 80, 0.9f, 0.95f, 0}});
    YoloV5Postprocessor post({}, 0.5);
    // (C, N) -> transposed to (N, C) internally
    EXPECT_EQ(run5(post, head.t()).size(), 1u);
}

TEST(YoloV5, MetaMapping) {
    const cv::Mat head = make_v5_head({{0, 320, 300, 100, 80, 0.9f, 0.95f, 0}});
    YoloV5Postprocessor post({}, 0.5);
    const PreprocessMeta meta = make_meta();
    const std::vector<DetectedObject> objects = run5(post, head, &meta);
    ASSERT_EQ(objects.size(), 1u);
    EXPECT_NEAR(objects[0].bbox.x, 810, 1.0);
    EXPECT_NEAR(objects[0].bbox.y, 360, 1.0);
}

// ---- TestSharedBehaviour -----------------------------------------------------------------

TEST(SharedBehaviour, ScoresSortedDescending) {
    const cv::Mat head = make_v8_head({
        {0, 320, 300, 100, 80, 0.6f, 0},
        {1, 100, 100, 50, 50, 0.9f, 0},  // far apart, no suppression
    });
    YoloV8Postprocessor post({}, 0.3);
    const std::vector<DetectedObject> objects = run8(post, head);
    ASSERT_EQ(objects.size(), 2u);
    EXPECT_GE(objects[0].score, objects[1].score);
    EXPECT_NEAR(objects[0].score, 0.9, 1e-6);
    EXPECT_NEAR(objects[1].score, 0.6, 1e-6);
}

TEST(SharedBehaviour, FallbackLabelWithoutList) {
    const cv::Mat head = make_v8_head({{0, 320, 300, 100, 80, 0.9f, 1}});
    YoloV8Postprocessor post({}, 0.3);
    const std::vector<DetectedObject> objects = run8(post, head);
    ASSERT_EQ(objects.size(), 1u);
    EXPECT_EQ(objects[0].label, "1");
}

// ---- TestSourceClipping --------------------------------------------------------------------
// Review P2: boxes landing in the letterbox pad clip to the source frame.

TEST(SourceClipping, PadBoxesClippedOrDropped) {
    // meta: origin (0, 140), scale (1/3, 1/3), source 1920x1080
    const cv::Mat head = make_v8_head({
        {0, 320, 40, 100, 40, 0.90f, 0},    // entirely in the top pad
        {1, 320, 200, 100, 200, 0.80f, 0},  // straddles the content edge
    });
    YoloV8Postprocessor post({"person"}, 0.3);
    const PreprocessMeta meta = make_meta();
    const std::vector<DetectedObject> objects = run8(post, head, &meta);

    ASSERT_EQ(objects.size(), 1u);          // pad-only box dropped
    const DetectedObject& obj = objects[0];
    EXPECT_FLOAT_EQ(obj.bbox.y, 0.0f);      // clipped at the top edge
    EXPECT_GE(obj.bbox.x, 0.0f);
    EXPECT_GE(obj.bbox.y, 0.0f);
    EXPECT_LE(obj.bbox.x + obj.bbox.width, 1920.0f + 1e-6f);
}

TEST(SourceClipping, ClipOptOutKeepsNegativeCoordinates) {
    const cv::Mat head = make_v8_head({{0, 320, 40, 100, 40, 0.9f, 0}});
    YoloV8Postprocessor post({"person"}, 0.3, 0.45, /*clip=*/false);
    const PreprocessMeta meta = make_meta();
    const std::vector<DetectedObject> objects = run8(post, head, &meta);
    ASSERT_EQ(objects.size(), 1u);
    EXPECT_LT(objects[0].bbox.y, 0.0f);  // unclipped, as requested
}
