// overlay.cpp — OverlayClient implementation. See overlay.hpp.
//
// Shares the camera-daemon CameraControl service (pb = aipc::camera). overlay.py
// is a thin wrapper over the single UpdateAiOverlay(AiOverlayConfig) RPC.
#include "neoruntime_ipc_sdk/overlay.hpp"

#include <grpcpp/grpcpp.h>

#include <memory>
#include <stdexcept>
#include <utility>

#include "detail/grpc_channel.hpp"
#include "camera-daemon/camera.grpc.pb.h"
#include "camera-daemon/camera.pb.h"

namespace neoruntime_ipc_sdk {

namespace pb = aipc::camera;

struct OverlayClient::Impl {
    std::string endpoint;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<pb::CameraControl::Stub> stub;

    void ensure_connected() {
        if (!stub) {
            channel = detail::make_channel(endpoint);
            stub = pb::CameraControl::NewStub(channel);
        }
    }
};

OverlayClient::OverlayClient(std::string endpoint)
    : impl_(std::make_unique<Impl>()) {
    impl_->endpoint = endpoint.empty()
                          ? Config::get_camera_control_endpoint()
                          : std::move(endpoint);
}

OverlayClient::~OverlayClient() = default;
OverlayClient::OverlayClient(OverlayClient&&) noexcept = default;
OverlayClient& OverlayClient::operator=(OverlayClient&&) noexcept = default;

void OverlayClient::connect() { impl_->ensure_connected(); }

void OverlayClient::close() {
    impl_->stub.reset();
    impl_->channel.reset();
}

bool OverlayClient::connected() const noexcept { return impl_->stub != nullptr; }

void OverlayClient::enable(bool show_label, bool show_confidence, int line_thickness) {
    pb::AiOverlayConfig cfg;
    cfg.set_enabled(true);
    cfg.set_show_label(show_label);
    cfg.set_show_confidence(show_confidence);
    cfg.set_line_thickness(line_thickness);
    impl_->ensure_connected();
    grpc::ClientContext ctx;
    pb::Status resp;
    detail::check_grpc(impl_->stub->UpdateAiOverlay(&ctx, cfg, &resp), "UpdateAiOverlay");
    detail::require_success(resp.success(), resp.message(), "UpdateAiOverlay");
}

void OverlayClient::disable() {
    pb::AiOverlayConfig cfg;
    cfg.set_enabled(false);
    impl_->ensure_connected();
    grpc::ClientContext ctx;
    pb::Status resp;
    detail::check_grpc(impl_->stub->UpdateAiOverlay(&ctx, cfg, &resp), "UpdateAiOverlay");
    detail::require_success(resp.success(), resp.message(), "UpdateAiOverlay");
}

void OverlayClient::configure(bool enabled, bool show_label, bool show_confidence,
                              int line_thickness, std::uint32_t box_color,
                              std::uint32_t label_color, std::uint32_t font_size) {
    pb::AiOverlayConfig cfg;
    cfg.set_enabled(enabled);
    cfg.set_show_label(show_label);
    cfg.set_show_confidence(show_confidence);
    cfg.set_line_thickness(line_thickness);
    // overlay.py only sets these when non-zero (proto3: 0 == default). Setting
    // unconditionally is wire-equivalent and simpler.
    cfg.set_box_color(box_color);
    cfg.set_label_color(label_color);
    cfg.set_font_size(font_size);
    impl_->ensure_connected();
    grpc::ClientContext ctx;
    pb::Status resp;
    detail::check_grpc(impl_->stub->UpdateAiOverlay(&ctx, cfg, &resp), "UpdateAiOverlay");
    detail::require_success(resp.success(), resp.message(), "UpdateAiOverlay");
}

void OverlayClient::apply(const OverlayConfig& config) {
    pb::AiOverlayConfig cfg;
    cfg.set_enabled(config.enabled);
    cfg.set_show_label(config.show_label);
    cfg.set_show_confidence(config.show_confidence);
    cfg.set_line_thickness(config.line_thickness);
    cfg.set_box_color(config.box_color);
    cfg.set_label_color(config.label_color);
    cfg.set_font_size(config.font_size);
    impl_->ensure_connected();
    grpc::ClientContext ctx;
    pb::Status resp;
    detail::check_grpc(impl_->stub->UpdateAiOverlay(&ctx, cfg, &resp), "UpdateAiOverlay");
    detail::require_success(resp.success(), resp.message(), "UpdateAiOverlay");
}

}  // namespace neoruntime_ipc_sdk
