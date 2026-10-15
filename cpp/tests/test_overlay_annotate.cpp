// test_overlay_annotate.cpp — annotate content parity, ported from
// python/tests/test_overlay_annotate.py. Python asserts on what its
// FakeEventClient captured (topic/metadata/payload); the same equalities
// run against detail::overlay_codec's builders — the payload/metadata that
// OverlayClient::annotate would publish. Plus one client-level shape:
// close() without annotate() must be safe (no event client ever created).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/overlay.hpp"

#include "detail/overlay_codec.hpp"

using namespace neoruntime_ipc_sdk;
namespace codec = detail::overlay_codec;

namespace {

DetectedObject det(float x, float y, float w, float h, const std::string& label,
                   std::int32_t class_id, float score) {
    DetectedObject d;
    d.label = label;
    d.score = score;
    d.class_id = class_id;
    d.bbox = BoundingBox{x, y, w, h};
    return d;
}

OverlayPolygon poly(const std::vector<OverlayPoint>& points) {
    OverlayPolygon p;
    p.points = points;
    return p;
}

}  // namespace

// ---- topic -----------------------------------------------------------------------
TEST(OverlayTopic, IsInferencePrefix) {
    EXPECT_EQ(codec::overlay_topic("main"), "inference/main");
    EXPECT_EQ(codec::overlay_topic("sub2"), "inference/sub2");
}

// ---- payloads: detections -----------------------------------------------------------
TEST(AnnotatePayload, DetectionShapesMatchPython) {
    std::vector<DetectedObject> dets{det(10.0f, 20.0f, 30.0f, 40.0f, "person", 3, 0.87f)};
    nlohmann::json payload = codec::annotate_payload(dets, std::nullopt);
    nlohmann::json expected = nlohmann::json{
        {"num_detections", 1},
        {"detections", nlohmann::json::array({nlohmann::json{
                            {"bbox", nlohmann::json::array({10.0f, 20.0f, 30.0f, 40.0f})},
                            {"class_id", 3},
                            {"confidence", 0.87f},
                            {"label", "person"},
                        }})}};
    EXPECT_EQ(payload, expected);
}

TEST(AnnotatePayload, EmptyDetectionsClearsBoxes) {
    nlohmann::json payload = codec::annotate_payload(std::vector<DetectedObject>{}, std::nullopt);
    EXPECT_EQ(payload, (nlohmann::json{{"num_detections", 0}, {"detections", nlohmann::json::array()}}));
    // Both sections nullopt (annotate(stream) bare call) is the same
    // clear-boxes event.
    nlohmann::json bare = codec::annotate_payload(std::nullopt, std::nullopt);
    EXPECT_EQ(bare, payload);
}

TEST(AnnotatePayload, TrackIdRidesWhenValid) {
    DetectedObject with_track = det(0, 0, 10, 10, "car", 1, 0.9f);
    with_track.track_id = 7;
    nlohmann::json j = codec::detection_json(with_track);
    ASSERT_TRUE(j.contains("track_id"));
    EXPECT_EQ(j["track_id"], 7);

    DetectedObject negative = det(0, 0, 10, 10, "car", 1, 0.9f);
    negative.track_id = -1;  // -1 = "no track" sentinel: omitted
    EXPECT_FALSE(codec::detection_json(negative).contains("track_id"));
    DetectedObject none = det(0, 0, 10, 10, "car", 1, 0.9f);
    EXPECT_FALSE(codec::detection_json(none).contains("track_id"));
}

// ---- payloads: polygons --------------------------------------------------------------
TEST(AnnotatePayload, PolygonDefaultsAndOptions) {
    OverlayPolygon p = poly({{0.1f, 0.2f}, {0.3f, 0.4f}});
    nlohmann::json j = codec::polygon_json(p);
    EXPECT_EQ(j, (nlohmann::json{{"points", nlohmann::json::array(
                                             {nlohmann::json::array({0.1f, 0.2f}),
                                              nlohmann::json::array({0.3f, 0.4f})})},
                                 {"label", ""},
                                 {"closed", true}}));

    p.label = "gate";
    p.closed = false;
    p.color = 0xFFFFFF00u;
    p.thickness = -1;
    nlohmann::json full = codec::polygon_json(p);
    EXPECT_EQ(full["label"], "gate");
    EXPECT_EQ(full["closed"], false);
    EXPECT_EQ(full["color"], 0xFFFFFF00u);
    EXPECT_EQ(full["thickness"], -1);
}

TEST(AnnotatePayload, PolygonsSectionIsIndependent) {
    // Polygons only: no detection keys at all (daemon discriminates by key).
    std::vector<OverlayPolygon> polys{poly({{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}})};
    nlohmann::json payload = codec::annotate_payload(std::nullopt, polys);
    EXPECT_TRUE(payload.contains("polygons"));
    EXPECT_FALSE(payload.contains("num_detections"));
    EXPECT_FALSE(payload.contains("detections"));

    // Both sections ride one event together.
    nlohmann::json both =
        codec::annotate_payload(std::vector<DetectedObject>{det(1, 2, 3, 4, "dog", 0, 0.5f)}, polys);
    EXPECT_EQ(both["num_detections"], 1);
    EXPECT_TRUE(both.contains("polygons"));

    // Empty polygons vector replaces zones without touching boxes.
    nlohmann::json cleared = codec::annotate_payload(std::nullopt, std::vector<OverlayPolygon>{});
    EXPECT_EQ(cleared, (nlohmann::json{{"polygons", nlohmann::json::array()}}));
}

TEST(AnnotatePayload, PolygonValidation) {
    std::vector<OverlayPolygon> seventeen;
    for (int i = 0; i < 17; ++i) seventeen.push_back(poly({{0.0f, 0.0f}, {1.0f, 1.0f}}));
    EXPECT_THROW(codec::annotate_payload(std::nullopt, seventeen), std::invalid_argument);

    EXPECT_THROW(codec::polygon_json(poly({{0.5f, 0.5f}})), std::invalid_argument);  // 1 point

    std::vector<OverlayPoint> many;
    for (int i = 0; i < 129; ++i) many.push_back({0.5f, 0.5f});
    EXPECT_THROW(codec::polygon_json(poly(many)), std::invalid_argument);  // 129 points

    EXPECT_NO_THROW(codec::polygon_json(poly({{0.5f, 0.5f}, {0.6f, 0.6f}})));  // 2 ok
    std::vector<OverlayPoint> capped;
    for (int i = 0; i < 128; ++i) capped.push_back({0.5f, 0.5f});
    EXPECT_NO_THROW(codec::polygon_json(poly(capped)));  // 128 ok

    EXPECT_THROW(codec::polygon_json(poly({{std::nanf(""), 0.5f}, {0.6f, 0.6f}})),
                 std::invalid_argument);
    EXPECT_THROW(codec::polygon_json(poly({{0.5f, std::nanf("")}, {0.6f, 0.6f}})),
                 std::invalid_argument);
}

// ---- section builders (annotate_result payload kinds) ---------------------------------
TEST(SectionBuilders, ClassificationLandmarkOcr) {
    Classification c;
    c.class_id = 7;
    c.label = "dog";
    c.confidence = 0.66f;
    EXPECT_EQ(codec::classification_json(c),
              (nlohmann::json{{"class_id", 7}, {"label", "dog"}, {"confidence", 0.66f}}));

    LandmarkSet lm;
    lm.type = "pose";
    lm.points = {LandmarkPoint{0.1f, 0.2f}, LandmarkPoint{0.3f, 0.4f, 0.5f}};
    nlohmann::json lmj = codec::landmark_json(lm);
    EXPECT_EQ(lmj["type"], "pose");
    ASSERT_EQ(lmj["points"].size(), 2u);
    EXPECT_EQ(lmj["points"][0], (nlohmann::json{{"x", 0.1f}, {"y", 0.2f}, {"confidence", 1.0f}}));
    EXPECT_EQ(lmj["points"][1]["confidence"], 0.5f);

    OcrLine line;
    line.text = "HELLO";
    line.confidence = 0.9f;
    line.bbox = BoundingBox{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(codec::ocr_json(line),
              (nlohmann::json{{"text", "HELLO"},
                              {"confidence", 0.9f},
                              {"bbox", nlohmann::json::array({1.0f, 2.0f, 3.0f, 4.0f})}}));
}

// ---- metadata -------------------------------------------------------------------------
TEST(AnnotateMetadata, StreamIdOnlyByDefault) {
    std::map<std::string, std::string> md =
        codec::annotate_metadata("main", std::nullopt, "", std::nullopt, std::nullopt);
    EXPECT_EQ(md, (std::map<std::string, std::string>{{"stream_id", "main"}}));
}

TEST(AnnotateMetadata, TtlRidesAsString) {
    std::map<std::string, std::string> md =
        codec::annotate_metadata("main", 120u, "", std::nullopt, std::nullopt);
    EXPECT_EQ(md, (std::map<std::string, std::string>{
                      {"stream_id", "main"}, {"result_ttl_ms", "120"}}));
    EXPECT_THROW(codec::annotate_metadata("main", 0u, "", std::nullopt, std::nullopt),
                 std::invalid_argument);
}

TEST(AnnotateMetadata, SessionTaggedOnlyWhenNonEmpty) {
    std::map<std::string, std::string> tagged =
        codec::annotate_metadata("main", std::nullopt, "s1", std::nullopt, std::nullopt);
    EXPECT_EQ(tagged.count("session_id"), 1u);
    std::map<std::string, std::string> untagged =
        codec::annotate_metadata("main", std::nullopt, "", std::nullopt, std::nullopt);
    EXPECT_EQ(untagged.count("session_id"), 0u);
}

TEST(AnnotateMetadata, FrameBindingRidesMetadataNotPayload) {
    std::map<std::string, std::string> md =
        codec::annotate_metadata("main", std::nullopt, "", 102ull, 7ull);
    EXPECT_EQ(md, (std::map<std::string, std::string>{{"stream_id", "main"},
                                                      {"frame_sequence", "102"},
                                                      {"stream_epoch", "7"}}));
    EXPECT_THROW(codec::annotate_metadata("main", std::nullopt, "", 0ull, std::nullopt),
                 std::invalid_argument);
    EXPECT_THROW(codec::annotate_metadata("main", std::nullopt, "", std::nullopt, 0ull),
                 std::invalid_argument);
}

// ---- client-level shape ----------------------------------------------------------------
TEST(OverlayClient, CloseWithoutAnnotateIsSafe) {
    // Constructing must not connect anything; close() with no lazily-created
    // event client (annotate never called) is a no-op, not a crash.
    OverlayClient client("");
    EXPECT_FALSE(client.connected());
    client.close();
    EXPECT_FALSE(client.connected());
    client.close();  // idempotent
}
