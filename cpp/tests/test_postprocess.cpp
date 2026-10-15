// test_postprocess.cpp — greedy NMS parity (port of python
// tests/test_postprocess.py). Public header only (postprocess.hpp).
#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/postprocess.hpp"

using namespace neoruntime_ipc_sdk;

namespace {

std::vector<std::array<float, 4>> fboxes(std::initializer_list<std::array<float, 4>> xs) {
    return std::vector<std::array<float, 4>>(xs);
}

}  // namespace

TEST(Nms, OverlappingBoxesKeepHigherScore) {
    const auto boxes = fboxes({{0, 0, 10, 10}, {1, 1, 11, 11}, {50, 50, 60, 60}});
    const std::vector<float> scores = {0.9f, 0.8f, 0.7f};
    // box 1 (~68% overlap with box 0) suppressed; box 2 disjoint
    EXPECT_EQ(nms(boxes, scores, 0.5), (std::vector<int>{0, 2}));
}

TEST(Nms, DisjointBoxesAllKept) {
    const auto boxes = fboxes({{0, 0, 10, 10}, {20, 20, 30, 30}, {40, 0, 50, 10}});
    const std::vector<float> scores = {0.5f, 0.9f, 0.7f};
    const std::vector<int> keep = nms(boxes, scores, 0.5);
    // all three survive; output order follows descending score
    ASSERT_EQ(keep.size(), 3u);
    EXPECT_EQ(keep[0], 1);
    EXPECT_EQ((std::vector<int>{keep[1], keep[2]}), (std::vector<int>{2, 0}));
}

TEST(Nms, DifferentClassesDoNotSuppressEachOther) {
    const auto boxes = fboxes({{0, 0, 10, 10}, {1, 1, 11, 11}});
    const std::vector<float> scores = {0.9f, 0.8f};
    const std::vector<int> classes = {0, 1};
    const std::vector<int> keep = nms(boxes, scores, 0.5, &classes);
    ASSERT_EQ(keep.size(), 2u);
    EXPECT_EQ((std::vector<int>{keep[0], keep[1]}), (std::vector<int>{0, 1}));
}

TEST(Nms, SameClassStillSuppresses) {
    const auto boxes = fboxes({{0, 0, 10, 10}, {1, 1, 11, 11}});
    const std::vector<float> scores = {0.9f, 0.8f};
    const std::vector<int> classes = {3, 3};
    EXPECT_EQ(nms(boxes, scores, 0.5, &classes), (std::vector<int>{0}));
}

TEST(Nms, EmptyInput) {
    EXPECT_TRUE(nms({}, {}).empty());
}

TEST(Nms, LengthMismatchRaises) {
    const auto boxes2 = fboxes({{0, 0, 1, 1}, {2, 2, 3, 3}});
    try {
        nms(boxes2, {0.1f, 0.2f, 0.3f});
        FAIL() << "expected std::invalid_argument (scores length)";
    } catch (const std::invalid_argument& exc) {
        EXPECT_NE(std::string(exc.what()).find("mismatch"), std::string::npos);
    }
    try {
        const std::vector<int> classes5 = {0, 0, 0, 0, 0};
        nms(boxes2, {0.1f, 0.2f}, 0.5, &classes5);
        FAIL() << "expected std::invalid_argument (class_ids length)";
    } catch (const std::invalid_argument& exc) {
        EXPECT_NE(std::string(exc.what()).find("mismatch"), std::string::npos);
    }
}

TEST(Nms, BadThresholdRaises) {
    const auto boxes = fboxes({{0, 0, 1, 1}});
    try {
        nms(boxes, {0.1f}, 1.5);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& exc) {
        EXPECT_NE(std::string(exc.what()).find("iou_threshold"), std::string::npos);
    }
}

TEST(Nms, TouchingBoxesHaveZeroIou) {
    // side-by-side, sharing an edge: inter == 0 -> both survive even at 0.0
    const auto boxes = fboxes({{0, 0, 10, 10}, {10, 0, 20, 10}});
    const std::vector<float> scores = {0.9f, 0.8f};
    const std::vector<int> keep = nms(boxes, scores, 0.0);
    ASSERT_EQ(keep.size(), 2u);
    EXPECT_EQ((std::vector<int>{keep[0], keep[1]}), (std::vector<int>{0, 1}));
}
