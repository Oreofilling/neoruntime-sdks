// test_inference.cpp — logic tests for the inference value types.
//
// Covers BoundingBox coordinate accessors and the InferenceResult query helpers
// (has_person / count_by_label / get_objects_by_label). These mirror the
// behavior the examples rely on (person_detection.cpp, perimeter_guard.cpp).
// The helpers are header-inline, so this is a pure compile+run check — no
// daemon or gRPC channel is involved.
#include <gtest/gtest.h>

#include "neoruntime_ipc_sdk/inference.hpp"

using neoruntime_ipc_sdk::BoundingBox;
using neoruntime_ipc_sdk::DetectedObject;
using neoruntime_ipc_sdk::InferenceResult;

// ---- BoundingBox -----------------------------------------------------------
TEST(BoundingBox, ToXyxyFromTopLeftWh) {
    BoundingBox b{0.1f, 0.2f, 0.3f, 0.4f};
    auto xyxy = b.to_xyxy();
    EXPECT_FLOAT_EQ(xyxy[0], 0.1f);
    EXPECT_FLOAT_EQ(xyxy[1], 0.2f);
    EXPECT_FLOAT_EQ(xyxy[2], 0.4f);  // x + width
    EXPECT_FLOAT_EQ(xyxy[3], 0.6f);  // y + height
}

TEST(BoundingBox, ToXywhIsIdentity) {
    BoundingBox b{0.25f, 0.75f, 0.5f, 0.125f};
    auto xywh = b.to_xywh();
    EXPECT_FLOAT_EQ(xywh[0], b.x);
    EXPECT_FLOAT_EQ(xywh[1], b.y);
    EXPECT_FLOAT_EQ(xywh[2], b.width);
    EXPECT_FLOAT_EQ(xywh[3], b.height);
}

TEST(BoundingBox, DefaultConstructedIsZero) {
    BoundingBox b;
    auto xyxy = b.to_xyxy();
    for (float v : xyxy) EXPECT_FLOAT_EQ(v, 0.0f);
}

// ---- InferenceResult helpers ----------------------------------------------
TEST(InferenceResult, HasPersonFalseWhenEmpty) {
    InferenceResult r;
    EXPECT_FALSE(r.has_person());
}

TEST(InferenceResult, HasPersonTrue) {
    InferenceResult r;
    r.objects.push_back({"person", 0.9f, {}, 0, std::nullopt});
    EXPECT_TRUE(r.has_person());
}

TEST(InferenceResult, HasPersonFalseForOtherLabel) {
    InferenceResult r;
    r.objects.push_back({"car", 0.9f, {}, 1, std::nullopt});
    EXPECT_FALSE(r.has_person());
}

TEST(InferenceResult, HasPersonIsLabelExact) {
    InferenceResult r;
    r.objects.push_back({"person_v1", 0.9f, {}, 0, std::nullopt});  // not "person"
    EXPECT_FALSE(r.has_person());
}

TEST(InferenceResult, CountByLabel) {
    InferenceResult r;
    r.objects.push_back({"person", 0.9f, {}, 0, std::nullopt});
    r.objects.push_back({"car", 0.8f, {}, 1, std::nullopt});
    r.objects.push_back({"person", 0.7f, {}, 0, std::nullopt});
    EXPECT_EQ(r.count_by_label("person"), 2u);
    EXPECT_EQ(r.count_by_label("car"), 1u);
    EXPECT_EQ(r.count_by_label("dog"), 0u);
}

TEST(InferenceResult, GetObjectsByLabelFiltersAndPreserves) {
    InferenceResult r;
    DetectedObject p1{"person", 0.9f, {0.1f, 0.1f, 0.2f, 0.2f}, 0, std::nullopt};
    DetectedObject p2{"person", 0.8f, {0.3f, 0.3f, 0.2f, 0.2f}, 0, std::nullopt};
    DetectedObject c1{"car", 0.7f, {0.5f, 0.5f, 0.1f, 0.1f}, 1, std::nullopt};
    r.objects = {p1, c1, p2};

    auto persons = r.get_objects_by_label("person");
    ASSERT_EQ(persons.size(), 2u);
    EXPECT_FLOAT_EQ(persons[0].score, 0.9f);
    EXPECT_FLOAT_EQ(persons[1].score, 0.8f);
    EXPECT_FLOAT_EQ(persons[0].bbox.x, 0.1f);

    auto cars = r.get_objects_by_label("car");
    ASSERT_EQ(cars.size(), 1u);
    EXPECT_EQ(cars[0].label, "car");
}

TEST(InferenceResult, GetObjectsByLabelEmptyWhenNoMatch) {
    InferenceResult r;
    r.objects.push_back({"person", 0.9f, {}, 0, std::nullopt});
    EXPECT_TRUE(r.get_objects_by_label("truck").empty());
}
