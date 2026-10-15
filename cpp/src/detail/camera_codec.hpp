// detail/camera_codec.hpp — pb ↔ public-type pure functions for the camera
// client (the C++ analogue of the conversion code inline in camera.py).
//
// Kept free of any socket/channel state so tests can pin the mapping without
// a daemon: fill a pb response, call the converter, compare fields. Lives
// under src/ (private); included by camera.cpp and test_camera_codec.cpp.
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "neoruntime_ipc_sdk/camera_types.hpp"

#include "camera-daemon/camera.pb.h"

namespace neoruntime_ipc_sdk::detail::camera_codec {

namespace pb = aipc::camera;

// Apply an optional per-RPC timeout (seconds) to a ClientContext, mirroring
// Python's `stub.X(req, timeout=timeout_s)`. nullopt = gRPC default.
inline void apply_timeout(grpc::ClientContext& ctx, std::optional<double> timeout_s) {
    if (timeout_s.has_value()) {
        ctx.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::duration<double>(*timeout_s)));
    }
}

// "replace"/"overlay" → pb enum; anything else throws invalid_argument
// (camera.py: mode must be one of ['overlay', 'replace']).
inline pb::InjectionMode inject_mode_from_string(const std::string& mode) {
    if (mode == "replace") return pb::INJECT_REPLACE;
    if (mode == "overlay") return pb::INJECT_OVERLAY;
    throw std::invalid_argument(
        "mode must be one of ['overlay', 'replace'], got '" + mode + "'");
}

inline pb::PushFrameRequest push_frame_request(const PushFrameSpec& spec) {
    pb::PushFrameRequest req;
    req.set_buffer_id(spec.buffer_id);
    req.set_width(spec.width);
    req.set_height(spec.height);
    req.set_stride(spec.stride);
    req.set_mode(inject_mode_from_string(spec.mode));
    req.set_pts_ns(spec.pts_ns);
    req.set_dest_x(spec.dest_x);
    req.set_dest_y(spec.dest_y);
    req.set_stream_id(spec.stream_id);
    req.set_end_of_stream(spec.end_of_stream);
    req.set_session_id(spec.session_id);
    return req;
}

inline InjectionResult injection_result_from(const pb::PushFrameResponse& resp) {
    InjectionResult out;
    out.success = resp.success();
    out.message = resp.message();
    out.error_code = resp.error_code();
    out.injected_frame_id = resp.injected_frame_id();
    out.accepted_frame_count = resp.accepted_frame_count();
    out.session_id = resp.session_id();
    out.in_flight_buffer_ids.assign(resp.in_flight_buffer_ids().begin(),
                                    resp.in_flight_buffer_ids().end());
    return out;
}

inline InjectionStatus injection_status_from(const pb::InjectionStatusResponse& resp) {
    InjectionStatus out;
    out.success = resp.success();
    out.message = resp.message();
    out.active = resp.active();
    out.mode = resp.mode() == pb::INJECT_OVERLAY ? "overlay" : "replace";
    out.frames_injected = resp.frames_injected();
    out.frames_dropped = resp.frames_dropped();
    out.queue_depth = resp.queue_depth();
    out.session_id = resp.session_id();
    out.in_flight_buffer_ids.assign(resp.in_flight_buffer_ids().begin(),
                                    resp.in_flight_buffer_ids().end());
    out.reports_in_flight_buffers = resp.reports_in_flight_buffers();
    return out;
}

inline InfraredStatus infrared_status_from(const pb::InfraredStatusResponse& resp) {
    InfraredStatus out;
    out.mode = resp.mode();
    out.transition = resp.transition();
    out.output_source = resp.output_source();
    out.auto_follow = resp.auto_follow();
    out.follow_active = resp.follow_active();
    out.manual_override = resp.manual_override();
    out.degraded = resp.degraded();
    out.requested_near_pwm = resp.requested_near_pwm();
    out.requested_far_pwm = resp.requested_far_pwm();
    out.applied_near_pwm = resp.applied_near_pwm();
    out.applied_far_pwm = resp.applied_far_pwm();
    out.zoom_ratio = resp.zoom_ratio();
    out.active_profile = resp.active_profile();
    out.selected_mode = resp.selected_mode();
    out.light_percent = resp.light_percent();
    out.light_mv = resp.light_mv();
    out.light_milli = resp.light_milli();
    out.light_valid = resp.light_valid();
    out.night_enter = resp.night_enter();
    out.day_enter = resp.day_enter();
    return out;
}

inline std::vector<IrPreset> ir_presets_from(const pb::IrPresetListResponse& resp) {
    std::vector<IrPreset> out;
    out.reserve(resp.presets_size());
    for (const auto& p : resp.presets()) {
        out.push_back(IrPreset{p.name(), p.zoom_ratio(), p.near_pwm(), p.far_pwm()});
    }
    return out;
}

// StreamStatusInfo (wire fields 1-28) → StreamStatus. The Python dataclass
// maps the same subset: identity/encode (1-9), publisher drops (13-18, 23),
// bake/strict (19-22), decoupling/frame-sync (24-28). Wire-only diagnostic
// fields 10-12 (ms_since_last_frame / measured_fps / status_detail) are not
// part of the public struct — parity with camera_types.py.
inline StreamStatus stream_status_from(const pb::StreamStatusInfo& s) {
    StreamStatus st;
    st.stream_id = s.stream_id();
    st.status = s.status();
    st.has_encoder = s.has_encoder();
    st.codec = s.codec();
    st.width = s.width();
    st.height = s.height();
    st.fps = s.fps();
    st.bitrate_bps = s.bitrate_bps();
    st.gop = s.gop();
    st.packets_published = s.packets_published();
    st.queue_overflow_drops = s.queue_overflow_drops();
    st.client_send_drops = s.client_send_drops();
    st.client_send_failures = s.client_send_failures();
    st.client_disconnects = s.client_disconnects();
    st.last_packet_seq = s.last_packet_seq();
    st.publisher_clients = s.publisher_clients();
    st.bake_skips = s.bake_skips();
    st.strict_locked = s.strict_locked();
    st.strict_degraded = s.strict_degraded();
    st.strict_skips = s.strict_skips();
    st.stream_epoch = s.stream_epoch();
    st.overlay_layer_count = s.overlay_layer_count();
    st.overlay_late_commands = s.overlay_late_commands();
    st.overlay_epoch_rejects = s.overlay_epoch_rejects();
    st.overlay_no_binding_drops = s.overlay_no_binding_drops();
    return st;
}

inline void add_mask_region(pb::PrivacyMaskConfig& cfg, const PrivacyMaskRegion& region) {
    auto* r = cfg.add_regions();
    r->set_id(region.id);
    r->set_name(region.name);
    r->set_enabled(region.enabled);
    for (float x : region.points_x) r->add_points_x(x);
    for (float y : region.points_y) r->add_points_y(y);
}

inline PrivacyMaskSettings privacy_mask_from(const pb::PrivacyMaskConfig& cfg) {
    PrivacyMaskSettings out;
    out.color = cfg.color();
    out.blur_radius = cfg.blur_radius();
    out.enabled = cfg.enabled();
    out.regions.reserve(cfg.regions_size());
    for (const auto& r : cfg.regions()) {
        PrivacyMaskRegion region;
        region.id = r.id();
        region.name = r.name();
        region.enabled = r.enabled();
        region.points_x.assign(r.points_x().begin(), r.points_x().end());
        region.points_y.assign(r.points_y().begin(), r.points_y().end());
        out.regions.push_back(std::move(region));
    }
    out.dpm_enabled = cfg.dpm_enabled();
    out.dpm_labels = cfg.dpm_labels();
    out.dpm_mode = cfg.dpm_mode();
    out.dpm_color = cfg.dpm_color();
    return out;
}

inline pb::PrivacyMaskConfig privacy_mask_config_from(const PrivacyMaskSettings& settings) {
    pb::PrivacyMaskConfig cfg;
    cfg.set_color(settings.color);
    cfg.set_blur_radius(settings.blur_radius);
    cfg.set_enabled(settings.enabled);
    cfg.set_dpm_enabled(settings.dpm_enabled);
    cfg.set_dpm_labels(settings.dpm_labels);
    cfg.set_dpm_mode(settings.dpm_mode);
    cfg.set_dpm_color(settings.dpm_color);
    for (const auto& region : settings.regions) add_mask_region(cfg, region);
    return cfg;
}

// Merge-style partial update (camera.py set_privacy_mask keyword path):
// start from `current` and overwrite only the fields the caller passed.
// regions.has_value() replaces the whole region list.
inline pb::PrivacyMaskConfig merged_privacy_mask(
    const pb::PrivacyMaskConfig& current, std::optional<std::uint32_t> color,
    std::optional<std::int32_t> blur_radius, std::optional<bool> enabled,
    const std::optional<std::vector<PrivacyMaskRegion>>& regions,
    std::optional<bool> dpm_enabled, std::optional<std::string> dpm_labels,
    std::optional<std::string> dpm_mode, std::optional<std::uint32_t> dpm_color) {
    pb::PrivacyMaskConfig cfg;
    cfg.CopyFrom(current);
    if (color.has_value()) cfg.set_color(*color);
    if (blur_radius.has_value()) cfg.set_blur_radius(*blur_radius);
    if (enabled.has_value()) cfg.set_enabled(*enabled);
    if (regions.has_value()) {
        cfg.clear_regions();
        for (const auto& region : *regions) add_mask_region(cfg, region);
    }
    if (dpm_enabled.has_value()) cfg.set_dpm_enabled(*dpm_enabled);
    if (dpm_labels.has_value()) cfg.set_dpm_labels(*dpm_labels);
    if (dpm_mode.has_value()) cfg.set_dpm_mode(*dpm_mode);
    if (dpm_color.has_value()) cfg.set_dpm_color(*dpm_color);
    return cfg;
}

}  // namespace neoruntime_ipc_sdk::detail::camera_codec
