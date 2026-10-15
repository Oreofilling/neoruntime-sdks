// detail/overlay_codec.hpp — pure builders for the overlay content/config
// path (C++ analogue of overlay.py's module-level helpers: OverlayConfig
// .to_proto, _detection_dict/_polygon_dict/_classification_dict/
// _landmark_dict/_ocr_dict, and the _publish_overlay_event validation sink).
//
// Header-only and channel-free so the payload/config shapes are unit-testable
// without a daemon — the payload equality assertions of
// python tests/test_overlay_annotate.py port onto these directly.
#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "neoruntime_ipc_sdk/overlay.hpp"

#include "camera-daemon/camera.pb.h"

namespace neoruntime_ipc_sdk::detail::overlay_codec {

namespace pb = aipc::camera;

// Wire-side caps mirrored from the daemon/HAL (hal_draw.h
// HAL_MAX_POLYGON_POINTS; daemon sidecar keeps at most 16 per stream) —
// loud validation here beats silent clipping.
inline constexpr std::size_t kPolygonCap = 16;
inline constexpr std::size_t kPolygonPointCap = 128;
inline constexpr std::uint32_t kStrictWaitCapMsMax = 5000;

// camera-daemon subscribes to "inference/**" (ai_overlay_subscriber.h:29)
// and requires event.metadata["stream_id"]; ai-runtime's publisher uses
// "inference/<stream_id>", so annotate() matches that exact topic shape.
inline std::string overlay_topic(const std::string& stream_id) {
    return "inference/" + stream_id;
}

// OverlayConfig.to_proto parity. box_color/label_color/font_size are plain
// proto3 scalars — setting them to 0 is wire-identical to leaving them
// unset. The strict fields are proto3 optional: set ONLY when the caller
// passed a value, so nullopt keeps the daemon's yaml setting and an
// explicit false/0 still goes on the wire.
inline pb::AiOverlayConfig ai_overlay_config(const OverlayConfig& c) {
    pb::AiOverlayConfig cfg;
    cfg.set_enabled(c.enabled);
    cfg.set_show_label(c.show_label);
    cfg.set_show_confidence(c.show_confidence);
    cfg.set_line_thickness(c.line_thickness);
    cfg.set_box_color(c.box_color);
    cfg.set_label_color(c.label_color);
    cfg.set_font_size(c.font_size);
    if (c.strict_frame_lock.has_value()) cfg.set_strict_frame_lock(*c.strict_frame_lock);
    if (c.strict_wait_cap_ms.has_value()) cfg.set_strict_wait_cap_ms(*c.strict_wait_cap_ms);
    return cfg;
}

// configure()-path validation (overlay.py configure raises before building).
inline void validate_strict_wait_cap(std::optional<std::uint32_t> cap_ms) {
    if (cap_ms.has_value() && *cap_ms > kStrictWaitCapMsMax) {
        throw std::invalid_argument(
            "strict_wait_cap_ms must be an int in [0, " +
            std::to_string(kStrictWaitCapMsMax) + "] (0 = derive from fps), got " +
            std::to_string(*cap_ms));
    }
}

// _detection_dict: wire schema is bbox [x,y,w,h] / class_id / confidence /
// label, plus track_id only when present and >= 0 (rendered "#id" suffix).
inline nlohmann::json detection_json(const DetectedObject& d) {
    nlohmann::json out;
    out["bbox"] = {d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height};
    out["class_id"] = d.class_id;
    out["confidence"] = d.score;
    out["label"] = d.label;
    if (d.track_id.has_value() && *d.track_id >= 0) {
        out["track_id"] = *d.track_id;
    }
    return out;
}

// _polygon_dict: points normalized [0,1], 2..kPolygonPointCap per polygon,
// finite floats only. Loud invalid_argument — a silently-clipped polygon is
// a wrong zone, not a cosmetic defect.
inline nlohmann::json polygon_json(const OverlayPolygon& p) {
    if (p.points.size() < 2 || p.points.size() > kPolygonPointCap) {
        throw std::invalid_argument("polygon needs 2.." + std::to_string(kPolygonPointCap) +
                                    " points, got " + std::to_string(p.points.size()));
    }
    nlohmann::json points = nlohmann::json::array();
    for (const auto& pt : p.points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) {
            throw std::invalid_argument("polygon point must be finite");
        }
        points.push_back(nlohmann::json::array({pt.x, pt.y}));
    }
    nlohmann::json out;
    out["points"] = std::move(points);
    out["label"] = p.label;
    out["closed"] = p.closed;
    if (p.color.has_value()) out["color"] = *p.color;
    if (p.thickness.has_value()) out["thickness"] = *p.thickness;
    return out;
}

inline nlohmann::json classification_json(const Classification& c) {
    return nlohmann::json{{"class_id", c.class_id},
                          {"label", c.label},
                          {"confidence", c.confidence}};
}

inline nlohmann::json landmark_json(const LandmarkSet& lm) {
    nlohmann::json points = nlohmann::json::array();
    for (const auto& p : lm.points) {
        points.push_back(nlohmann::json{{"x", p.x}, {"y", p.y}, {"confidence", p.confidence}});
    }
    return nlohmann::json{{"type", lm.type}, {"points", std::move(points)}};
}

inline nlohmann::json ocr_json(const OcrLine& line) {
    return nlohmann::json{{"text", line.text},
                          {"confidence", line.confidence},
                          {"bbox", nlohmann::json::array({line.bbox.x, line.bbox.y,
                                                          line.bbox.width, line.bbox.height})}};
}

// annotate() payload assembly: detections and/or polygons — each optional;
// nullopt leaves that section absent (previous values kept daemon-side).
// Neither given (Python annotate(stream) with both None) is the v1
// clear-boxes call. >kPolygonCap polygons per call throws client-side.
inline nlohmann::json annotate_payload(
    const std::optional<std::vector<DetectedObject>>& detections,
    const std::optional<std::vector<OverlayPolygon>>& polygons) {
    nlohmann::json payload = nlohmann::json::object();
    if (detections.has_value()) {
        nlohmann::json items = nlohmann::json::array();
        for (const auto& d : *detections) items.push_back(detection_json(d));
        payload["num_detections"] = items.size();
        payload["detections"] = std::move(items);
    }
    if (polygons.has_value()) {
        if (polygons->size() > kPolygonCap) {
            throw std::invalid_argument("at most " + std::to_string(kPolygonCap) +
                                        " polygons per call, got " +
                                        std::to_string(polygons->size()));
        }
        nlohmann::json items = nlohmann::json::array();
        for (const auto& p : *polygons) items.push_back(polygon_json(p));
        payload["polygons"] = std::move(items);
    }
    if (payload.empty()) {
        payload = {{"num_detections", 0}, {"detections", nlohmann::json::array()}};
    }
    return payload;
}

// The _publish_overlay_event validation sink: the daemon parses
// result_ttl_ms as an unsigned int, so ttl must be a positive int HERE (a
// negative/non-int value would wrap to a never-expiring layer on the wire);
// frame-binding params ride metadata, never payload, and must be positive.
inline std::map<std::string, std::string> annotate_metadata(
    const std::string& stream_id, std::optional<std::uint32_t> ttl_ms,
    const std::string& session_id, std::optional<std::uint64_t> frame_sequence,
    std::optional<std::uint64_t> stream_epoch) {
    if (ttl_ms.has_value() && *ttl_ms == 0) {
        throw std::invalid_argument("ttl_ms must be a positive int, got 0");
    }
    auto validate_positive = [](const char* name, std::uint64_t value) {
        if (value < 1) {
            throw std::invalid_argument(std::string(name) +
                                        " must be a positive int, got " +
                                        std::to_string(value));
        }
    };
    std::map<std::string, std::string> metadata{{"stream_id", stream_id}};
    if (ttl_ms.has_value()) metadata["result_ttl_ms"] = std::to_string(*ttl_ms);
    if (!session_id.empty()) metadata["session_id"] = session_id;
    if (frame_sequence.has_value()) {
        validate_positive("frame_sequence", *frame_sequence);
        metadata["frame_sequence"] = std::to_string(*frame_sequence);
    }
    if (stream_epoch.has_value()) {
        validate_positive("stream_epoch", *stream_epoch);
        metadata["stream_epoch"] = std::to_string(*stream_epoch);
    }
    return metadata;
}

}  // namespace neoruntime_ipc_sdk::detail::overlay_codec
