// camera_types.hpp — data types returned by the camera control client.
//
// 1:1 port of camera_types.py (0.8.1). Kept as a separate header so the type
// layer mirrors the Python module split (camera.py imports from
// camera_types.py); camera.hpp includes this file, so existing
// `#include <neoruntime_ipc_sdk/camera.hpp>` code sees everything unchanged.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace neoruntime_ipc_sdk {

// ISP image-pipeline settings. Int fields use -1 = "no change" (mirrors app.py).
// manual_mode / auto_exposure are truly optional (proto3 optional bool).
struct ISPConfig {
    int brightness = -1;        // [0..100]
    int contrast = -1;
    int saturation = -1;
    int sharpness = -1;
    std::optional<bool> manual_mode;
    std::optional<bool> auto_exposure;
    int backlight = -1;         // [0..100]
    int exposure_time_us = -1;
    int gain = -1;
    int noise_reduction = -1;   // [0..100]
    int wdr_value = -1;         // [0..100]
    int powerline_freq = -1;    // 0=off, 1=50Hz, 2=60Hz
    int awb_index = -1;
};

struct TransformConfig {
    int rotation = 0;   // 0/1/2/3 => 0/90/180/270
    int flip = 0;       // 0=none, 1=H, 2=V, 3=both
    bool dewarp = false;
    bool grayscale = false;
};

struct EncoderReconfigResult {
    bool success = false;
    std::string message;
    std::uint32_t interrupt_ms = 0;
};

// Result of one CameraClient::push_frame call (PushFrameResponse).
struct InjectionResult {
    bool success = false;
    std::string message;
    std::int32_t error_code = 0;
    std::uint64_t injected_frame_id = 0;
    // PushFrameStream only: frames accepted before the stream returned
    // (0 for unary push_frame).
    std::uint64_t accepted_frame_count = 0;
    // Lifecycle session tag echoed by the daemon (P2-13): correlation
    // and observability only — ownership is fd-anchored daemon-side.
    std::string session_id;
    // Write-lease snapshot (Fix-1): registry ids the daemon may still
    // read at response time (queued + mid-bake). A pool slot is safe to
    // rewrite only when its id is absent. Empty on daemons predating the
    // lease protocol — check InjectionStatus::reports_in_flight_buffers
    // before trusting it as exhaustive.
    std::vector<std::uint64_t> in_flight_buffer_ids;
};

// Snapshot of the daemon-side injection session (GetInjectionStatus).
struct InjectionStatus {
    bool success = false;
    std::string message;
    bool active = false;
    std::string mode = "replace";  // "replace" (P0) | "overlay" (P1, DSP blend)
    std::uint64_t frames_injected = 0;
    std::uint64_t frames_dropped = 0;  // drop-oldest overflow, never backpressure
    std::uint32_t queue_depth = 0;
    // Latest non-empty lifecycle tag of the live session (P2-13); empty
    // when the session is untagged or closed.
    std::string session_id;
    // Write-lease snapshot (Fix-1), same semantics as
    // InjectionResult::in_flight_buffer_ids.
    std::vector<std::uint64_t> in_flight_buffer_ids;
    // Capability flag: true only on daemons that report the write-lease
    // set. False => old daemon; the in_flight lists are absent and the
    // FramePublisher lease logic stays in its legacy depth-paced mode.
    bool reports_in_flight_buffers = false;
};

// One frame spec for push_frame / push_frame_stream (proto-free mirror of
// PushFrameRequest). buffer_id must be a DSP-registry buffer id — a raw fd
// number never crosses gRPC.
struct PushFrameSpec {
    std::uint64_t buffer_id = 0;
    std::uint32_t width = 0;       // injected content width (even)
    std::uint32_t height = 0;      // injected content height (even)
    std::uint32_t stride = 0;      // luma stride (NV12) or byte stride (ARGB32)
    std::string mode = "replace";  // "replace" | "overlay"
    std::uint64_t pts_ns = 0;      // device CLOCK_MONOTONIC; 0 = due immediately
    std::uint32_t dest_x = 0;      // OVERLAY position within the target frame
    std::uint32_t dest_y = 0;      // ignored by REPLACE (full-frame)
    std::string stream_id;         // REQUIRED for OVERLAY; "": legacy matching
    bool end_of_stream = false;    // true = flush + restore ISP path at next IDR
    std::string session_id;        // app session tag (P2-13)
};

// One display stream from CameraClient::get_stream_status.
//
// Fields come in four groups, in declaration order — identity & encode
// geometry (stream_id..gop); encoded-stream fan-out observability
// (packets_published..publisher_clients); overlay bake / strict frame-lock
// counters (bake_skips..strict_skips); behavior decoupling & frame-sync
// state (stream_epoch..overlay_no_binding_drops). The unified drop/throughput
// fields default to 0 and stay 0 when the layer is absent (publisher
// disabled, stream never through the overlay bake site) or the server
// predates the fields — the response shape is stable.
struct StreamStatus {
    std::string stream_id;
    std::string status;     // active, starting, stalled, stopped, error, unknown
    bool has_encoder = false;
    std::string codec;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t fps = 0;
    std::uint32_t bitrate_bps = 0;
    std::uint32_t gop = 0;
    // Publisher side (packets keyed by seq, assigned at enqueue):
    std::uint64_t packets_published = 0;    // packets assigned a seq (== last_packet_seq)
    std::uint64_t queue_overflow_drops = 0; // shallow-queue evictions before send
    std::uint64_t client_send_drops = 0;    // per-client non-blocking send skips
    std::uint64_t client_send_failures = 0; // per-client hard send failures (client dropped)
    std::uint64_t client_disconnects = 0;   // subscribers lost via control-poll EOF/ERR
    std::uint64_t last_packet_seq = 0;      // newest seq assigned (0 = none yet)
    std::uint32_t publisher_clients = 0;    // connected encoded-stream subscribers
    // Overlay bake side (frames shipped clean when no fresh result):
    std::uint64_t bake_skips = 0;       // frames shipped clean: no result / TTL expired
    std::uint64_t strict_locked = 0;    // strict-gate draws that waited and matched
    std::uint64_t strict_degraded = 0;  // strict-gate frames degraded (cap/hopeless)
    std::uint64_t strict_skips = 0;     // strict-gate frames with no result at all
    // Behavior decoupling + frame sync. stream_epoch is the stream
    // generation counter — reconfigure_encoder / full transform reinit
    // bumps it and purges held layers; an app pinning overlay events to an
    // epoch taken from an earlier snapshot learns of the restart by
    // comparing, instead of publishing silently-rejected overlays.
    // 0 = the server predates the fields (epoch is never 0 on a current
    // daemon).
    std::uint64_t stream_epoch = 0;             // stream generation (restart bumps)
    std::uint64_t overlay_layer_count = 0;      // overlay layers currently held
    std::uint64_t overlay_late_commands = 0;    // app events rejected: frame passed
    std::uint64_t overlay_epoch_rejects = 0;    // app events rejected: stale epoch
    std::uint64_t overlay_no_binding_drops = 0; // platform events dropped: no binding
};

struct Capabilities {
    bool has_video = false;
    bool has_codec = false;
    bool has_led = false;
    bool has_sensor = false;
    bool has_mcu = false;
    bool has_env_ctrl = false;
    bool has_alarm = false;
    bool has_rs485 = false;
    bool has_osd = false;
    bool has_draw = false;
    bool has_audio = false;
};

struct SensorInfo {
    bool available = false;
    std::string sensor_model;
    int i2c_bus = -1;
    std::string i2c_address;
    int pixel_format = -1;
};

struct HardwareStatus {
    std::uint32_t light_sensor_mv = 0;
    int light_sensor_lux = 0;
    int mcu_temp_millic = 0;
    int ain_mv = 0;
    std::string mcu_version;
    std::uint32_t white_light_duty = 0;
    std::uint32_t ir_led_duty = 0;
    std::uint32_t ircut_mode = 0;  // 0=day, 1=night
};

struct PipelineStreamConfig {
    std::string stream_id;
    std::uint32_t input_width = 0;
    std::uint32_t input_height = 0;
    std::uint32_t input_framerate = 0;
    std::string codec = "h264";
    std::uint32_t encoder_width = 0;
    std::uint32_t encoder_height = 0;
    std::uint32_t encoder_framerate = 0;
    std::uint32_t encoder_bitrate = 0;
    std::uint32_t encoder_gop = 0;
};

struct EnvStatus {
    bool enabled = false;
};

// Day/night imaging state reported by the camera pipeline.
struct InfraredStatus {
    std::string mode;
    std::string transition;
    std::string output_source;
    bool auto_follow = false;
    bool follow_active = false;
    bool manual_override = false;
    bool degraded = false;
    std::uint32_t requested_near_pwm = 0;
    std::uint32_t requested_far_pwm = 0;
    std::uint32_t applied_near_pwm = 0;
    std::uint32_t applied_far_pwm = 0;
    float zoom_ratio = 0.0f;
    std::string active_profile;
    std::string selected_mode;
    std::int32_t light_percent = 0;
    std::uint32_t light_mv = 0;
    std::int32_t light_milli = 0;
    bool light_valid = false;
    std::int32_t night_enter = 0;
    std::int32_t day_enter = 0;
};

// Saved IR-light profile bound to a zoom ratio.
struct IrPreset {
    std::string name;
    float zoom_ratio = 0.0f;  // 1.0 ~ 2.88 (optical zoom ratio)
    std::uint32_t near_pwm = 0;  // 0 ~ 100
    std::uint32_t far_pwm = 0;   // 0 ~ 100
};

// One privacy-mask region (polygon with up to 8 vertices, normalized coords).
struct PrivacyMaskRegion {
    std::string id;
    std::string name;
    bool enabled = true;
    std::vector<float> points_x;  // normalized X coords [0.0-1.0]
    std::vector<float> points_y;  // normalized Y coords [0.0-1.0]
};

// Static and dynamic (AI) privacy-mask configuration.
struct PrivacyMaskSettings {
    std::uint32_t color = 0;        // 0x00RRGGBB fill color (when blur_radius=0)
    std::int32_t blur_radius = 0;   // pixelation block size [2-64]; 0=solid color
    bool enabled = false;
    std::vector<PrivacyMaskRegion> regions;
    bool dpm_enabled = false;
    std::string dpm_labels;         // e.g. "person,vehicle"
    std::string dpm_mode;           // "mosaic", "blur", or "overlay"
    std::uint32_t dpm_color = 0;
};

// Result of GetAlarmOutputs: the four digital output lines.
struct AlarmOutputs {
    bool alarm_out0 = false;
    bool alarm_out1 = false;
    bool wiegand0 = false;
    bool wiegand1 = false;
};

// OSD overlay descriptors (typed replacement for camera.py's dict-based OSD API).
struct OsdTextOverlay {
    std::string id;
    std::string text;
    float x = 0.0f;
    float y = 0.0f;
    float font_size = 32.0f;
    std::uint32_t text_color = 0;  // RGBA
    bool enabled = true;
    int h_align = 0;  // 0=LEFT, 1=CENTER, 2=RIGHT
    int v_align = 0;  // 0=TOP, 1=CENTER, 2=BOTTOM
};

struct OsdDateTimeOverlay {
    std::string id;
    float x = 0.0f;
    float y = 0.0f;
    std::string format = "%Y-%m-%d %H:%M:%S";
    float font_size = 32.0f;
    std::uint32_t text_color = 0;
    bool enabled = true;
    int h_align = 0;
    int v_align = 0;
};

struct OsdImageOverlay {
    std::string id;
    std::string image_path;
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;   // normalized (0.0-1.0), like x/y — proto parity
    float height = 0.0f;  // normalized (0.0-1.0)
    bool enabled = true;
    int h_align = 0;
    int v_align = 0;
};

struct OsdStreamConfig {
    std::string stream_name = "main";
    std::vector<OsdTextOverlay> text_overlays;
    std::vector<OsdDateTimeOverlay> datetime_overlays;
    std::vector<OsdImageOverlay> image_overlays;
};

}  // namespace neoruntime_ipc_sdk
