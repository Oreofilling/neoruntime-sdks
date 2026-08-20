// overlay.hpp — AI overlay control client. 1:1 port of overlay.py.
//
// Drives the camera-daemon AI overlay: detection boxes/labels/confidence drawn
// onto NV12 frames before encoding (zero CPU cost). Uses the SAME CameraControl
// gRPC service as camera.py — specifically the UpdateAiOverlay(AiOverlayConfig)
// RPC. Transport: sync gRPC stubs over UDS (Config::get_camera_control_endpoint()).
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace neoruntime_ipc_sdk {

// AI overlay appearance. Field defaults mirror overlay.py OverlayConfig.
struct OverlayConfig {
    bool enabled = true;
    bool show_label = true;
    bool show_confidence = true;
    int line_thickness = 2;     // 1-10
    std::uint32_t box_color = 0;    // ARGB, e.g. 0xFFFF0000 = red
    std::uint32_t label_color = 0;  // ARGB
    std::uint32_t font_size = 0;    // 8-72 (0 = daemon default)
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
    void close();
    bool connected() const noexcept;

    // Enable the overlay with display options.
    void enable(bool show_label = true, bool show_confidence = true, int line_thickness = 2);
    // Disable the overlay.
    void disable();
    // Full configuration (all fields). Colors are ARGB; 0 = daemon default.
    void configure(bool enabled = true, bool show_label = true, bool show_confidence = true,
                   int line_thickness = 2, std::uint32_t box_color = 0,
                   std::uint32_t label_color = 0, std::uint32_t font_size = 0);
    // Apply a pre-built OverlayConfig.
    void apply(const OverlayConfig& config);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace neoruntime_ipc_sdk
