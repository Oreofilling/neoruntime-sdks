// overlay.cpp — OverlayClient implementation. See overlay.hpp.
//
// Shares the camera-daemon CameraControl service (pb = aipc::camera) for the
// configuration half (UpdateAiOverlay) and a lazily-connected EventClient
// for the content half (annotate / annotate_result), like overlay.py.
#include "neoruntime_ipc_sdk/overlay.hpp"

#include <grpcpp/grpcpp.h>

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "neoruntime_ipc_sdk/config.hpp"
#include "neoruntime_ipc_sdk/events.hpp"

#include "detail/grpc_channel.hpp"
#include "detail/overlay_codec.hpp"
#include "detail/transport.hpp"
#include "camera-daemon/camera.grpc.pb.h"
#include "camera-daemon/camera.pb.h"

namespace neoruntime_ipc_sdk {

namespace pb = aipc::camera;

struct OverlayClient::Impl : detail::GrpcClientBase<pb::CameraControl> {
    explicit Impl(std::string endpoint)
        : GrpcClientBase(std::move(endpoint)) {}

    // Lazily connect the event-bus client used for annotate() (Python
    // OverlayClient._events). Close order in close(): events first.
    EventClient& events() {
        if (!event_client) event_client = std::make_unique<EventClient>();
        return *event_client;
    }

    std::unique_ptr<EventClient> event_client;
};

OverlayClient::OverlayClient(std::string endpoint)
    : impl_(std::make_unique<Impl>(
          endpoint.empty() ? Config::get_camera_control_endpoint() : std::move(endpoint))) {}

OverlayClient::~OverlayClient() = default;
OverlayClient::OverlayClient(OverlayClient&&) noexcept = default;
OverlayClient& OverlayClient::operator=(OverlayClient&&) noexcept = default;

void OverlayClient::connect() { impl_->connect(); }

void OverlayClient::close() {
    if (impl_->event_client) {
        impl_->event_client->close();
        impl_->event_client.reset();
    }
    impl_->close();
}

bool OverlayClient::connected() const noexcept { return impl_->connected(); }

// _update: every config path funnels through UpdateAiOverlay.
namespace {

void update_ai_overlay(detail::GrpcClientBase<pb::CameraControl>& base,
                       const pb::AiOverlayConfig& cfg) {
    pb::Status resp;
    grpc::ClientContext ctx;
    detail::check_grpc(base.stub().UpdateAiOverlay(&ctx, cfg, &resp), "UpdateAiOverlay");
    detail::require_success(resp.success(), resp.message(), "UpdateAiOverlay");
}

}  // namespace

void OverlayClient::enable(bool show_label, bool show_confidence, int line_thickness) {
    // Routed through the same codec as configure()/apply() — one wire-shape
    // choke point (strict fields absent, exactly like overlay.py's inline
    // construction).
    OverlayConfig c;
    c.enabled = true;
    c.show_label = show_label;
    c.show_confidence = show_confidence;
    c.line_thickness = line_thickness;
    update_ai_overlay(*impl_, detail::overlay_codec::ai_overlay_config(c));
}

void OverlayClient::disable() {
    OverlayConfig c;
    c.enabled = false;
    update_ai_overlay(*impl_, detail::overlay_codec::ai_overlay_config(c));
}

void OverlayClient::configure(bool enabled, bool show_label, bool show_confidence,
                              int line_thickness, std::uint32_t box_color,
                              std::uint32_t label_color, std::uint32_t font_size,
                              std::optional<bool> strict_frame_lock,
                              std::optional<std::uint32_t> strict_wait_cap_ms) {
    detail::overlay_codec::validate_strict_wait_cap(strict_wait_cap_ms);
    OverlayConfig c;
    c.enabled = enabled;
    c.show_label = show_label;
    c.show_confidence = show_confidence;
    c.line_thickness = line_thickness;
    c.box_color = box_color;
    c.label_color = label_color;
    c.font_size = font_size;
    c.strict_frame_lock = strict_frame_lock;
    c.strict_wait_cap_ms = strict_wait_cap_ms;
    update_ai_overlay(*impl_, detail::overlay_codec::ai_overlay_config(c));
}

void OverlayClient::apply(const OverlayConfig& config) {
    update_ai_overlay(*impl_, detail::overlay_codec::ai_overlay_config(config));
}

// ---- content -------------------------------------------------------------------

std::string OverlayClient::annotate(const std::string& stream_id,
                                    const std::optional<std::vector<DetectedObject>>& detections,
                                    const std::optional<std::vector<OverlayPolygon>>& polygons,
                                    std::optional<std::uint32_t> ttl_ms,
                                    const std::string& session_id,
                                    std::optional<std::uint64_t> frame_sequence,
                                    std::optional<std::uint64_t> stream_epoch) {
    // Payload build (incl. polygon caps) before any publish side effect —
    // an invalid polygon list throws without touching the event bus.
    nlohmann::json payload = detail::overlay_codec::annotate_payload(detections, polygons);
    return publish_overlay_event_(stream_id, payload, ttl_ms, session_id, frame_sequence,
                                  stream_epoch);
}

std::string OverlayClient::annotate_result(const std::string& stream_id,
                                           const InferenceResult& result,
                                           std::optional<std::uint32_t> ttl_ms,
                                           const std::string& session_id,
                                           std::optional<std::uint64_t> frame_sequence,
                                           std::optional<std::uint64_t> stream_epoch) {
    if (!result.objects.empty()) {
        return annotate(stream_id, result.objects, std::nullopt, ttl_ms, session_id,
                        frame_sequence, stream_epoch);
    }
    // One payload carries one section kind — that is how camera-daemon's
    // parser discriminates them.
    nlohmann::json payload;
    if (!result.classifications.empty()) {
        nlohmann::json items = nlohmann::json::array();
        for (const auto& c : result.classifications) {
            items.push_back(detail::overlay_codec::classification_json(c));
        }
        payload = {{"classifications", std::move(items)}};
    } else if (!result.landmarks.empty()) {
        nlohmann::json items = nlohmann::json::array();
        for (const auto& lm : result.landmarks) {
            items.push_back(detail::overlay_codec::landmark_json(lm));
        }
        payload = {{"landmarks", std::move(items)}};
    } else if (!result.ocr_lines.empty()) {
        nlohmann::json items = nlohmann::json::array();
        for (const auto& line : result.ocr_lines) {
            items.push_back(detail::overlay_codec::ocr_json(line));
        }
        payload = {{"ocr_lines", std::move(items)}};
    } else {
        // Empty result: publish zero detections, clearing stale boxes.
        payload = {{"num_detections", 0}, {"detections", nlohmann::json::array()}};
    }
    return publish_overlay_event_(stream_id, payload, ttl_ms, session_id, frame_sequence,
                                  stream_epoch);
}

std::string OverlayClient::publish_overlay_event_(
    const std::string& stream_id, const nlohmann::json& payload,
    std::optional<std::uint32_t> ttl_ms, const std::string& session_id,
    std::optional<std::uint64_t> frame_sequence,
    std::optional<std::uint64_t> stream_epoch) {
    std::map<std::string, std::string> metadata = detail::overlay_codec::annotate_metadata(
        stream_id, ttl_ms, session_id, frame_sequence, stream_epoch);
    // EventClient::publish serializes with payload.dump() — nlohmann's
    // default dump is compact (Python's compact=True separators), which the
    // daemon's "bbox":[ / "polygons":[ string scan relies on.
    return impl_->events().publish(detail::overlay_codec::overlay_topic(stream_id), payload,
                                   /*persistent=*/false, /*ttl_ms=*/std::nullopt, metadata);
}

}  // namespace neoruntime_ipc_sdk
