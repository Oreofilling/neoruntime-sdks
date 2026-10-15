// detail/device_codec.hpp — pb ↔ public-type pure functions for the device
// client's native-autofocus family (mirrors the inline conversions in
// device.py's start_oneshot_af / get_autofocus_status / get_af_measurement
// / set_af_windows). Header-only, no channel state — unit-testable directly.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "neoruntime_ipc_sdk/device.hpp"

#include "device-control/device.pb.h"

namespace neoruntime_ipc_sdk::detail::device_codec {

namespace pb = aipc::device;

inline AfJob af_job_from(const pb::AfJobResponse& resp) {
    return AfJob{resp.accepted(), resp.job_id(), resp.message()};
}

inline AfStatus af_status_from(const pb::AfStatusResponse& r) {
    AfStatus st;
    st.job_id = r.job_id();
    st.operation = r.operation();
    st.state = r.state();
    st.progress = r.progress();
    st.busy = r.busy();
    st.anchor_valid = r.anchor_valid();
    st.requested_ratio = r.requested_ratio();
    st.effective_ratio = r.effective_ratio();
    st.zoom_pos = r.zoom_pos();
    st.focus_pos = r.focus_pos();
    st.best_focus = r.best_focus();
    st.metric = r.metric();
    st.confidence = r.confidence();
    st.reproducibility = r.reproducibility();
    st.estimated_distance_m = r.estimated_distance_m();
    st.elapsed_ms = r.elapsed_ms();
    st.error_code = r.error_code();
    st.message = r.message();
    return st;
}

inline AfMeasurement af_measurement_from(const pb::AfMeasurementResponse& r) {
    AfMeasurement m;
    m.focus_energy.assign(r.focus_energy().begin(), r.focus_energy().end());
    m.mean_luma.assign(r.mean_luma().begin(), r.mean_luma().end());
    m.frame_id = r.frame_id();
    return m;
}

// Build SetAfWindowsRequest; more than 3 windows is a client-side
// invalid_argument (device.py: "at most 3 AF windows are supported").
inline pb::SetAfWindowsRequest af_windows_request(
    bool enabled, const std::vector<AfWindow>& windows, const std::string& stream_id) {
    if (windows.size() > 3) {
        throw std::invalid_argument("at most 3 AF windows are supported");
    }
    pb::SetAfWindowsRequest req;
    req.set_enabled(enabled);
    req.set_stream_id(stream_id);
    for (const auto& w : windows) {
        auto* win = req.add_windows();
        win->set_x(w.x);
        win->set_y(w.y);
        win->set_w(w.w);
        win->set_h(w.h);
    }
    return req;
}

}  // namespace neoruntime_ipc_sdk::detail::device_codec
