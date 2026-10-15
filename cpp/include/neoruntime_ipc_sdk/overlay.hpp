// overlay.hpp — AI overlay control client. 1:1 port of overlay.py (0.8.1).
//
// Two halves, like the Python module:
//   * configuration — enable/disable/configure/apply via the camera-daemon
//     UpdateAiOverlay(AiOverlayConfig) RPC (same CameraControl service as
//     camera.hpp);
//   * content — annotate()/annotate_result() push detection boxes (or
//     classifications / landmarks / OCR lines / zone polygons) through the
//     event bus; camera-daemon's overlay subscriber draws them onto NV12
//     frames with HAL draw ops before encoding — zero CPU cost in the app.
//
// Transport: sync gRPC stubs over UDS (Config::get_camera_control_endpoint())
// plus a lazily-connected EventClient for the content half; close() closes
// both.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "neoruntime_ipc_sdk/inference.hpp"

namespace neoruntime_ipc_sdk {

// AI overlay appearance. Field defaults mirror overlay.py OverlayConfig.
// strict_frame_lock / strict_wait_cap_ms mirror the optional wire fields:
// nullopt (the default) leaves them unset, so the daemon keeps its current
// (yaml) strict settings — an explicit false/0 still goes on the wire.
struct OverlayConfig {
    bool enabled = true;
    bool show_label = true;
    bool show_confidence = true;
    int line_thickness = 2;             // 1-10
    std::uint32_t box_color = 0;        // ARGB, e.g. 0xFFFF0000 = red; 0 = default
    std::uint32_t label_color = 0;      // ARGB; 0 = default
    std::uint32_t font_size = 0;        // 8-72; 0 = daemon default
    std::optional<bool> strict_frame_lock;
    std::optional<std::uint32_t> strict_wait_cap_ms;
};

// One polygon vertex, normalized [0,1] frame coordinates.
struct OverlayPoint {
    float x = 0.0f;
    float y = 0.0f;
};

// Zone polygon spec for annotate(polygons=...). Only points is required;
// color is ARGB and thickness is pixels (-1 = filled).
struct OverlayPolygon {
    std::vector<OverlayPoint> points;   // 2..128 vertices
    std::string label;
    bool closed = true;
    std::optional<std::uint32_t> color;
    std::optional<std::int32_t> thickness;
};

class OverlayClient {
public:
    // Empty endpoint => Config::get_camera_control_endpoint().
    explicit OverlayClient(std::string endpoint = "");
    ~OverlayClient();
    OverlayClient(const OverlayClient&) = delete;
    OverlayClient& operator=(const OverlayClient&) = delete;
    OverlayClient(OverlayClient&&) noexcept;
    OverlayClient& operator=(OverlayClient&&) noexcept;

    void connect();
    // Closes the gRPC channel AND the lazily-created event client.
    void close();
    bool connected() const noexcept;

    // Enable the overlay with display options.
    void enable(bool show_label = true, bool show_confidence = true, int line_thickness = 2);
    // Disable the overlay.
    void disable();
    // Full configuration. Colors are ARGB; 0 = daemon default.
    // strict_frame_lock opts in/out of strict frame-lock on identity-fed
    // display streams (each frame waits — bounded by strict_wait_cap_ms —
    // for its OWN inference result at the bake site, so a box lands on the
    // frame it was computed from). nullopt keeps the daemon's setting.
    // strict_wait_cap_ms must be [0, 5000] (0 = derive from stream fps);
    // anything else throws std::invalid_argument client-side.
    void configure(bool enabled = true, bool show_label = true, bool show_confidence = true,
                   int line_thickness = 2, std::uint32_t box_color = 0,
                   std::uint32_t label_color = 0, std::uint32_t font_size = 0,
                   std::optional<bool> strict_frame_lock = std::nullopt,
                   std::optional<std::uint32_t> strict_wait_cap_ms = std::nullopt);
    // Apply a pre-built OverlayConfig (no strict-cap validation — mirrors
    // Python apply()/to_proto()).
    void apply(const OverlayConfig& config);

    // -- Content: push results to the hardware overlay via the event bus ----
    //
    // Draw detection boxes / zone polygons on stream_id's video. Results
    // expire ttl_ms after the last event (fps-derived default when omitted),
    // so call at a few Hz while detections are fresh. The overlay itself
    // must be on: enable() first.
    //
    // detections: nullopt leaves the stream's previous boxes untouched (so
    // polygons can push alone); any vector — {} included — replaces them.
    // A DetectedObject's track_id (>= 0) rides along and renders as a "#id"
    // label suffix.
    // polygons: nullopt keeps the previous zones; any vector replaces them.
    // Points are normalized [0,1] floats, 2..128 per polygon, at most 16
    // polygons per call — violations throw std::invalid_argument.
    // ttl_ms: positive per-event validity override.
    // session_id: lifecycle tag (P2-13); empty/omitted leaves the event
    // untagged. frame_sequence / stream_epoch bind the event to a frame /
    // stream generation (positive; from InferenceResult.frame_sequence /
    // get_stream_status().stream_epoch) — see overlay.py for the daemon-side
    // semantics.
    //
    // Returns the published event id.
    std::string annotate(
        const std::string& stream_id,
        const std::optional<std::vector<DetectedObject>>& detections = std::nullopt,
        const std::optional<std::vector<OverlayPolygon>>& polygons = std::nullopt,
        std::optional<std::uint32_t> ttl_ms = std::nullopt,
        const std::string& session_id = "",
        std::optional<std::uint64_t> frame_sequence = std::nullopt,
        std::optional<std::uint64_t> stream_epoch = std::nullopt);

    // Push whichever section of an InferenceResult is populated. Detections
    // take precedence, then classifications, landmarks and OCR lines — a
    // payload carries one kind (that is how the daemon's parser
    // discriminates them). An empty result publishes zero detections,
    // clearing stale boxes.
    std::string annotate_result(const std::string& stream_id, const InferenceResult& result,
                                std::optional<std::uint32_t> ttl_ms = std::nullopt,
                                const std::string& session_id = "",
                                std::optional<std::uint64_t> frame_sequence = std::nullopt,
                                std::optional<std::uint64_t> stream_epoch = std::nullopt);

private:
    // Single choke point for every publish path (annotate, annotate_result):
    // validates ttl/frame binding and publishes to "inference/<stream_id>".
    std::string publish_overlay_event_(
        const std::string& stream_id, const nlohmann::json& payload,
        std::optional<std::uint32_t> ttl_ms, const std::string& session_id,
        std::optional<std::uint64_t> frame_sequence,
        std::optional<std::uint64_t> stream_epoch);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace neoruntime_ipc_sdk
